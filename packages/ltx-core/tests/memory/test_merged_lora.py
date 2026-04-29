"""Tests for LoRA merge via ``BlockOffloader.set_loras()``.

Covers set_loras validation, lifecycle (activate/deactivate), LoRA
switching, and forward-output correctness against a manually-merged
baseline.

Most lifecycle tests run on CPU (the merge math is device-agnostic);
CUDA-only tests gate on availability.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from ltx_core.memory import (
    BlockOffloader,
    LoRA,
)

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bf16_model(num_blocks: int = 4, dim: int = 16) -> nn.Module:
    """Tiny block-streaming-shaped model with bf16 frozen params."""

    class Block(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.attn = nn.Linear(dim, dim, bias=False)
            self.ff = nn.Linear(dim, dim, bias=False)

        def forward(self, x):
            return self.ff(self.attn(x))

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(dim, dim, bias=False)
            self.transformer_blocks = nn.ModuleList(
                [Block(dim) for _ in range(num_blocks)]
            )
            self.head = nn.Linear(dim, dim, bias=False)

        def forward(self, x):
            x = self.embed(x)
            for blk in self.transformer_blocks:
                x = blk(x)
            return self.head(x)

    m = M()
    m = m.to(torch.bfloat16)
    for p in m.parameters():
        p.requires_grad = False
    return m


def _make_lora(
    num_blocks: int, dim: int, rank: int = 4, strength: float = 1.0,
    seed: int = 0, prefix: str = "",
) -> LoRA:
    """Build a LoRA with flat safetensors-style keys targeting attn.weight
    across all blocks."""
    g = torch.Generator().manual_seed(seed)
    sd: dict[str, torch.Tensor] = {}
    for b in range(num_blocks):
        base = f"{prefix}transformer_blocks.{b}.attn"
        sd[f"{base}.lora_A.weight"] = torch.randn(
            rank, dim, generator=g, dtype=torch.float32,
        )
        sd[f"{base}.lora_B.weight"] = torch.randn(
            dim, rank, generator=g, dtype=torch.float32,
        )
    return LoRA(state_dict=sd, strength=strength)


def _expected_merged_weight(
    base: torch.Tensor, loras: list[LoRA], block_idx: int, qual: str,
    key_transform=None,
) -> torch.Tensor:
    """Compute the target weight by summing all LoRA deltas onto the base."""
    out = base.clone()
    for lora in loras:
        stem = qual.replace(".weight", "")
        for key in lora.state_dict:
            if not key.endswith(".lora_A.weight"):
                continue
            base_key = key[: -len(".lora_A.weight")]
            target = f"{base_key}.weight"
            if key_transform is not None:
                target = key_transform(target)
            if target != f"transformer_blocks.{block_idx}.{qual}":
                continue
            a = lora.state_dict[f"{base_key}.lora_A.weight"].to(base.dtype)
            b = lora.state_dict[f"{base_key}.lora_B.weight"].to(base.dtype)
            out = out + lora.strength * (b @ a)
    return out


def _make_strategy(model, device="cpu", blocks_to_swap=1, **kwargs):
    """Shorthand for constructing the strategy with sensible defaults."""
    return BlockOffloader(
        model, torch.device(device),
        layers_attr="transformer_blocks",
        blocks_to_swap=blocks_to_swap,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


class TestConstructionValidation:
    def test_rejects_fp32_lora_target(self) -> None:
        m = _make_bf16_model().to(torch.float32)
        s = _make_strategy(m)
        with pytest.raises(ValueError, match="bf16/fp16"):
            s.set_loras([_make_lora(4, 16)])

    def test_accepts_fp16_base(self) -> None:
        m = _make_bf16_model().to(torch.float16)
        s = _make_strategy(m)
        assert s.cache_bytes > 0

    def test_rejects_empty_layers_attr(self) -> None:
        m = _make_bf16_model(num_blocks=0)
        with pytest.raises(ValueError, match="resolved to empty list"):
            _make_strategy(m, blocks_to_swap=0)


# ---------------------------------------------------------------------------
# set_loras validation
# ---------------------------------------------------------------------------


class TestSetLorasValidation:
    def test_unpaired_a_factor(self) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m)
        sd = {"transformer_blocks.0.attn.lora_A.weight": torch.randn(4, 16)}
        with pytest.raises(ValueError, match="Unpaired"):
            s.set_loras([LoRA(state_dict=sd)])

    def test_unpaired_b_factor(self) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m)
        sd = {"transformer_blocks.0.attn.lora_B.weight": torch.randn(16, 4)}
        with pytest.raises(ValueError, match="Unpaired"):
            s.set_loras([LoRA(state_dict=sd)])

    def test_rejects_non_floating_factor_dtype(self) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m)
        sd = {
            "transformer_blocks.0.attn.lora_A.weight": torch.zeros(4, 16, dtype=torch.int32),
            "transformer_blocks.0.attn.lora_B.weight": torch.zeros(16, 4, dtype=torch.int32),
        }
        with pytest.raises(ValueError, match="floating-point"):
            s.set_loras([LoRA(state_dict=sd)])

    def test_rejects_rank_mismatch(self) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m)
        sd = {
            "transformer_blocks.0.attn.lora_A.weight": torch.randn(4, 16),
            "transformer_blocks.0.attn.lora_B.weight": torch.randn(16, 8),
        }
        with pytest.raises(ValueError, match="shape mismatch"):
            s.set_loras([LoRA(state_dict=sd)])

    def test_rejects_target_shape_mismatch(self) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m)
        sd = {
            "transformer_blocks.0.attn.lora_A.weight": torch.randn(4, 16),
            "transformer_blocks.0.attn.lora_B.weight": torch.randn(8, 4),
        }
        with pytest.raises(ValueError, match="shape mismatch"):
            s.set_loras([LoRA(state_dict=sd)])

    def test_rejects_non_2d_factor(self) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m)
        sd = {
            "transformer_blocks.0.attn.lora_A.weight": torch.randn(4),
            "transformer_blocks.0.attn.lora_B.weight": torch.randn(16, 4),
        }
        with pytest.raises(ValueError, match="shape mismatch"):
            s.set_loras([LoRA(state_dict=sd)])

    def test_non_block_targets_counted_in_bytes(self) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m)
        sd = {
            "embed.lora_A.weight": torch.randn(4, 16),
            "embed.lora_B.weight": torch.randn(16, 4),
        }
        s.set_loras([LoRA(state_dict=sd)])
        assert s._lora_factor_bytes > 0

    def test_key_transform_strips_prefix(self) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m)
        lora = _make_lora(4, 16, prefix="diffusion_model.")
        s.set_loras([lora])
        assert s._lora_factor_bytes > 0

    def test_key_transform_none_requires_exact_keys(self) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m, key_transform=None)
        lora = _make_lora(4, 16)
        s.set_loras([lora])
        assert s._lora_factor_bytes > 0

    def test_key_transform_none_skips_prefixed_keys(self) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m, key_transform=None)
        lora = _make_lora(4, 16, prefix="diffusion_model.")
        s.set_loras([lora])
        assert s._lora_factor_bytes == 0

    @CUDA
    def test_set_loras_raises_while_active(self) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m, device="cuda")
        s.set_loras([_make_lora(4, 16)])
        s.activate()
        try:
            with pytest.raises(RuntimeError, match="inactive"):
                s.set_loras([])
        finally:
            s.deactivate()

    def test_set_loras_clears_previous(self) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m)
        s.set_loras([_make_lora(4, 16, rank=4)])
        bytes_first = s._lora_factor_bytes
        assert bytes_first > 0
        s.set_loras([_make_lora(4, 16, rank=8)])
        bytes_second = s._lora_factor_bytes
        assert bytes_second > bytes_first
        s.set_loras([])
        assert s._lora_factor_bytes == 0


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    @CUDA
    def test_activate_runs_components(self) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m, device="cuda")
        s.set_loras([_make_lora(4, 16)])
        try:
            s.activate()
            assert m.embed.weight.is_cuda
            assert m.head.weight.is_cuda
        finally:
            s.deactivate()

    @CUDA
    def test_deactivate_returns_to_pinned(self) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m, device="cuda")
        s.set_loras([_make_lora(4, 16)])
        s.activate()
        s.deactivate()
        assert m.embed.weight.is_pinned()
        assert m.head.weight.is_pinned()

    @CUDA
    def test_reactivation_with_different_loras(self) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m, device="cuda")
        s.set_loras([_make_lora(4, 16, seed=1)])
        s.activate()
        s.deactivate()
        s.set_loras([_make_lora(4, 16, seed=2)])
        s.activate()
        s.deactivate()
        assert m.embed.weight.is_pinned()

    @CUDA
    def test_activate_with_no_loras_runs_base_only(self) -> None:
        m = _make_bf16_model()
        captured = m.transformer_blocks[0].attn.weight.detach().clone()
        s = _make_strategy(m, device="cuda")
        s.activate()
        try:
            x = torch.randn(2, 16, dtype=torch.bfloat16, device="cuda")
            for blk in m.transformer_blocks:
                x = blk(x)
            torch.cuda.synchronize()
            actual = m.transformer_blocks[0].attn.weight.detach()
            assert torch.allclose(
                actual, captured.to("cuda"), rtol=0.0, atol=0.0,
            ), "no LoRAs must leave base weights unmodified"
        finally:
            s.deactivate()


# ---------------------------------------------------------------------------
# Forward correctness
# ---------------------------------------------------------------------------


class TestMergeCorrectness:
    @CUDA
    def test_merged_weights_match_manual_baseline(self) -> None:
        m = _make_bf16_model(num_blocks=4, dim=16)
        captured_base = {
            i: m.transformer_blocks[i].attn.weight.detach().clone()
            for i in range(4)
        }

        loras = [
            _make_lora(num_blocks=4, dim=16, strength=0.5, seed=10),
            _make_lora(num_blocks=4, dim=16, strength=0.25, seed=20),
        ]
        s = _make_strategy(m, device="cuda")
        s.set_loras(loras)
        s.activate()
        try:
            x = torch.randn(2, 16, dtype=torch.bfloat16, device="cuda")
            for blk in m.transformer_blocks:
                x = blk(x)
            torch.cuda.synchronize()
            for i in range(4):
                expected = _expected_merged_weight(
                    captured_base[i], loras, i, "attn.weight",
                ).to("cuda")
                actual = m.transformer_blocks[i].attn.weight.detach()
                assert torch.allclose(actual, expected, rtol=0.01, atol=0.01), (
                    f"block {i} merged weight mismatch:\n"
                    f"  expected: {expected.flatten()[:4]}\n"
                    f"  actual:   {actual.flatten()[:4]}"
                )
        finally:
            s.deactivate()

    @CUDA
    def test_prefixed_lora_keys_merge_correctly(self) -> None:
        """LoRAs with ``diffusion_model.`` prefix (ComfyUI format) should
        merge identically to unprefixed keys via the default key_transform."""
        m = _make_bf16_model(num_blocks=4, dim=16)
        captured_base = {
            i: m.transformer_blocks[i].attn.weight.detach().clone()
            for i in range(4)
        }

        lora = _make_lora(4, 16, strength=0.7, seed=42, prefix="diffusion_model.")
        s = _make_strategy(m, device="cuda")
        s.set_loras([lora])
        s.activate()
        try:
            x = torch.randn(2, 16, dtype=torch.bfloat16, device="cuda")
            for blk in m.transformer_blocks:
                x = blk(x)
            torch.cuda.synchronize()
            from ltx_core.memory.lora import default_key_transform
            for i in range(4):
                expected = _expected_merged_weight(
                    captured_base[i], [lora], i, "attn.weight",
                    key_transform=default_key_transform,
                ).to("cuda")
                actual = m.transformer_blocks[i].attn.weight.detach()
                assert torch.allclose(actual, expected, rtol=0.01, atol=0.01)
        finally:
            s.deactivate()

    @CUDA
    def test_non_block_lora_merges_correctly(self) -> None:
        """LoRA targeting embed (non-block) should be merged at activate."""
        m = _make_bf16_model(num_blocks=4, dim=16)
        captured_embed = m.embed.weight.detach().clone()

        g = torch.Generator().manual_seed(99)
        sd = {
            "embed.lora_A.weight": torch.randn(4, 16, generator=g, dtype=torch.float32),
            "embed.lora_B.weight": torch.randn(16, 4, generator=g, dtype=torch.float32),
        }
        lora = LoRA(state_dict=sd, strength=0.5)
        s = _make_strategy(m, device="cuda")
        s.set_loras([lora])
        s.activate()
        try:
            a = sd["embed.lora_A.weight"].to(torch.bfloat16)
            b = sd["embed.lora_B.weight"].to(torch.bfloat16)
            expected = (captured_embed + 0.5 * (b @ a)).to("cuda")
            actual = m.embed.weight.detach()
            assert torch.allclose(actual, expected, rtol=0.01, atol=0.01), (
                f"non-block merge mismatch:\n"
                f"  expected: {expected.flatten()[:4]}\n"
                f"  actual:   {actual.flatten()[:4]}"
            )
        finally:
            s.deactivate()


# ---------------------------------------------------------------------------
# Cleanup invariants
# ---------------------------------------------------------------------------


class TestDeactivateCleanupInvariants:
    @CUDA
    def test_cleanup_runs_even_when_streamer_deactivate_raises(
        self, monkeypatch,
    ) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m, device="cuda")
        s.set_loras([_make_lora(4, 16)])

        def streamer_boom() -> None:
            raise RuntimeError("streamer cleanup failed")

        monkeypatch.setattr(s._streamers[0], "deactivate", streamer_boom)
        s.activate()

        with pytest.raises(RuntimeError):
            s.deactivate()


# ---------------------------------------------------------------------------
# Cache budget
# ---------------------------------------------------------------------------


class TestCacheBytes:
    def test_factors_count_toward_cache_bytes(self) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m)
        baseline = s.cache_bytes

        s.set_loras([_make_lora(num_blocks=4, dim=16, rank=4)])
        assert s.cache_bytes > baseline

    def test_cache_bytes_resets_on_clear(self) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m)
        baseline = s.cache_bytes
        s.set_loras([_make_lora(4, 16)])
        assert s.cache_bytes > baseline
        s.set_loras([])
        assert s.cache_bytes == baseline
