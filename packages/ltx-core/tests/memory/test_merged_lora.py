"""Tests for ``ltx_core.memory.merged_lora.MergedLoRAStrategy``.

Covers construction validation, lifecycle (activate/deactivate),
active-set switching, factor lifetime ordering, and forward-output
correctness against a manually-merged baseline.

Most lifecycle tests run on CPU (the merge math is device-agnostic);
CUDA-only tests gate on availability.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from ltx_core.memory import (
    LoRABundle,
    LoRALayerFactors,
    MergedLoRAStrategy,
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
    name: str, num_blocks: int, dim: int, rank: int = 4, scaling: float = 1.0,
    seed: int = 0,
) -> LoRABundle:
    """Build a LoRABundle with random factors for the given attention
    q-projection target across all blocks."""
    g = torch.Generator().manual_seed(seed)
    blocks: dict[int, dict[str, LoRALayerFactors]] = {}
    for b in range(num_blocks):
        # Target the attn.weight in each block — single target for
        # simplicity. Real LoRAs target multiple layers per block.
        A = torch.randn(rank, dim, generator=g, dtype=torch.float32)
        B = torch.randn(dim, rank, generator=g, dtype=torch.float32)
        blocks[b] = {
            "attn.weight": LoRALayerFactors(A=A, B=B, scaling=scaling),
        }
    return LoRABundle(name=name, blocks=blocks)


def _expected_merged_weight(
    base: torch.Tensor, loras: list[LoRABundle], block_idx: int, qual: str,
) -> torch.Tensor:
    """Compute the target weight for a layer by summing all LoRA deltas
    onto the base, in the same order MergedLoRAStrategy will."""
    out = base.clone()
    for lora in loras:
        f = lora.blocks.get(block_idx, {}).get(qual)
        if f is None:
            continue
        delta = f.scaling * (f.B.to(base.dtype) @ f.A.to(base.dtype))
        out = out + delta
    return out


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


class TestConstructionValidation:
    def test_rejects_fp32_base(self) -> None:
        m = _make_bf16_model().to(torch.float32)
        with pytest.raises(ValueError, match="bf16/fp16 base"):
            MergedLoRAStrategy(
                m, torch.device("cpu"),
                loras=[_make_lora("a", 4, 16)],
                layers_attr="transformer_blocks",
                blocks_to_swap=1,
            )

    def test_accepts_fp16_base(self) -> None:
        m = _make_bf16_model().to(torch.float16)
        # Construction should succeed without raising.
        s = MergedLoRAStrategy(
            m, torch.device("cpu"),
            loras=[_make_lora("a", 4, 16)],
            layers_attr="transformer_blocks",
            blocks_to_swap=1,
        )
        assert s.cache_bytes > 0

    def test_rejects_duplicate_lora_names(self) -> None:
        m = _make_bf16_model()
        loras = [_make_lora("dup", 4, 16, seed=0), _make_lora("dup", 4, 16, seed=1)]
        with pytest.raises(ValueError, match="unique"):
            MergedLoRAStrategy(
                m, torch.device("cpu"),
                loras=loras,
                layers_attr="transformer_blocks",
                blocks_to_swap=1,
            )

    def test_rejects_empty_layers_attr(self) -> None:
        # transformer_blocks=[] would resolve to an empty list.
        m = _make_bf16_model(num_blocks=0)
        with pytest.raises(ValueError, match="empty ModuleList"):
            MergedLoRAStrategy(
                m, torch.device("cpu"),
                loras=[],
                layers_attr="transformer_blocks",
                blocks_to_swap=0,
            )

    def test_rejects_out_of_range_block_idx(self) -> None:
        m = _make_bf16_model(num_blocks=4, dim=16)
        # LoRA targets a block index that doesn't exist.
        bad = LoRABundle(
            name="oob",
            blocks={
                99: {"attn.weight": LoRALayerFactors(
                    A=torch.randn(4, 16), B=torch.randn(16, 4), scaling=1.0,
                )},
            },
        )
        with pytest.raises(ValueError, match="block_idx=99 out of range"):
            MergedLoRAStrategy(
                m, torch.device("cpu"),
                loras=[bad],
                layers_attr="transformer_blocks", blocks_to_swap=1,
            )

    def test_rejects_non_floating_factor_dtype(self) -> None:
        m = _make_bf16_model(num_blocks=4, dim=16)
        bad = LoRABundle(
            name="int_factors",
            blocks={
                0: {"attn.weight": LoRALayerFactors(
                    A=torch.zeros(4, 16, dtype=torch.int32),
                    B=torch.zeros(16, 4, dtype=torch.int32),
                    scaling=1.0,
                )},
            },
        )
        with pytest.raises(ValueError, match="floating-point"):
            MergedLoRAStrategy(
                m, torch.device("cpu"),
                loras=[bad],
                layers_attr="transformer_blocks", blocks_to_swap=1,
            )

    def test_rejects_non_2d_factor_shape(self) -> None:
        m = _make_bf16_model(num_blocks=4, dim=16)
        bad = LoRABundle(
            name="bad_shape",
            blocks={
                0: {"attn.weight": LoRALayerFactors(
                    A=torch.randn(4),                # 1D, not 2D
                    B=torch.randn(16, 4),
                    scaling=1.0,
                )},
            },
        )
        with pytest.raises(ValueError, match="must be 2D"):
            MergedLoRAStrategy(
                m, torch.device("cpu"),
                loras=[bad],
                layers_attr="transformer_blocks", blocks_to_swap=1,
            )

    def test_rejects_rank_mismatch(self) -> None:
        m = _make_bf16_model(num_blocks=4, dim=16)
        bad = LoRABundle(
            name="rank_mismatch",
            blocks={
                0: {"attn.weight": LoRALayerFactors(
                    A=torch.randn(4, 16),    # rank=4
                    B=torch.randn(16, 8),    # rank=8 — mismatch
                    scaling=1.0,
                )},
            },
        )
        with pytest.raises(ValueError, match="rank mismatch"):
            MergedLoRAStrategy(
                m, torch.device("cpu"),
                loras=[bad],
                layers_attr="transformer_blocks", blocks_to_swap=1,
            )

    def test_accepts_factors_already_on_gpu(self) -> None:
        # If the user constructs factors on GPU (e.g., from a model on
        # GPU), the strategy should .cpu() them before pinning rather
        # than failing.
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
        m = _make_bf16_model(num_blocks=4, dim=16)
        gpu_factors = LoRABundle(
            name="from_gpu",
            blocks={
                0: {"attn.weight": LoRALayerFactors(
                    A=torch.randn(4, 16, device="cuda"),
                    B=torch.randn(16, 4, device="cuda"),
                    scaling=1.0,
                )},
            },
        )
        s = MergedLoRAStrategy(
            m, torch.device("cuda"),
            loras=[gpu_factors],
            layers_attr="transformer_blocks", blocks_to_swap=1,
        )
        assert "from_gpu" in s.register_lora_names()


# ---------------------------------------------------------------------------
# Active set management
# ---------------------------------------------------------------------------


class TestActiveSet:
    def test_set_active_rejects_unknown_name(self) -> None:
        m = _make_bf16_model()
        s = MergedLoRAStrategy(
            m, torch.device("cpu"),
            loras=[_make_lora("a", 4, 16)],
            layers_attr="transformer_blocks", blocks_to_swap=1,
        )
        with pytest.raises(ValueError, match="Unknown LoRA names"):
            s.set_active(["unknown"])

    def test_set_active_rejects_duplicates(self) -> None:
        m = _make_bf16_model()
        s = MergedLoRAStrategy(
            m, torch.device("cpu"),
            loras=[_make_lora("a", 4, 16), _make_lora("b", 4, 16)],
            layers_attr="transformer_blocks", blocks_to_swap=1,
        )
        with pytest.raises(ValueError, match="Duplicate LoRA names"):
            s.set_active(["a", "b", "a"])

    def test_set_active_preserves_order(self) -> None:
        m = _make_bf16_model()
        s = MergedLoRAStrategy(
            m, torch.device("cpu"),
            loras=[_make_lora("a", 4, 16), _make_lora("b", 4, 16),
                   _make_lora("c", 4, 16)],
            layers_attr="transformer_blocks", blocks_to_swap=1,
        )
        s.set_active(["c", "a", "b"])
        assert s.active == ("c", "a", "b")

    @CUDA
    def test_set_active_raises_while_active(self) -> None:
        m = _make_bf16_model()
        s = MergedLoRAStrategy(
            m, torch.device("cuda"),
            loras=[_make_lora("a", 4, 16), _make_lora("b", 4, 16)],
            layers_attr="transformer_blocks", blocks_to_swap=1,
        )
        s.set_active(["a"])
        s.activate()
        try:
            with pytest.raises(RuntimeError, match="inactive"):
                s.set_active(["b"])
        finally:
            s.deactivate()


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    @CUDA
    def test_activate_runs_components(self) -> None:
        m = _make_bf16_model()
        s = MergedLoRAStrategy(
            m, torch.device("cuda"),
            loras=[_make_lora("a", 4, 16)],
            layers_attr="transformer_blocks", blocks_to_swap=1,
        )
        s.set_active(["a"])
        try:
            s.activate()
            # Non-block params (embed, head) on GPU via PinnedWeights component.
            assert m.embed.weight.is_cuda
            assert m.head.weight.is_cuda
        finally:
            s.deactivate()

    @CUDA
    def test_deactivate_returns_to_pinned(self) -> None:
        m = _make_bf16_model()
        s = MergedLoRAStrategy(
            m, torch.device("cuda"),
            loras=[_make_lora("a", 4, 16)],
            layers_attr="transformer_blocks", blocks_to_swap=1,
        )
        s.set_active(["a"])
        s.activate()
        s.deactivate()
        # Non-block params back on pinned CPU.
        assert m.embed.weight.is_pinned()
        assert m.head.weight.is_pinned()

    @CUDA
    def test_reactivation_with_different_active_set(self) -> None:
        # Switching LoRA combos by deactivate → set_active → activate.
        m = _make_bf16_model()
        s = MergedLoRAStrategy(
            m, torch.device("cuda"),
            loras=[_make_lora("a", 4, 16, seed=1),
                   _make_lora("b", 4, 16, seed=2)],
            layers_attr="transformer_blocks", blocks_to_swap=1,
        )
        s.set_active(["a"])
        s.activate()
        s.deactivate()
        s.set_active(["b"])
        s.activate()
        s.deactivate()
        # Final state: pinned, no GPU residency.
        assert m.embed.weight.is_pinned()


# ---------------------------------------------------------------------------
# Forward correctness — does the merge actually produce the right weights?
# ---------------------------------------------------------------------------


class TestMergeCorrectness:
    @CUDA
    def test_merged_weights_match_manual_baseline(self) -> None:
        # Capture base weights before construction (PinnedWeights and
        # BlockStreamer both clone-and-pin in __init__, so the model's
        # original tensor objects get replaced — we need a snapshot).
        m = _make_bf16_model(num_blocks=4, dim=16)
        captured_base = {
            i: m.transformer_blocks[i].attn.weight.detach().clone()
            for i in range(4)
        }

        loras = [
            _make_lora("a", num_blocks=4, dim=16, scaling=0.5, seed=10),
            _make_lora("b", num_blocks=4, dim=16, scaling=0.25, seed=20),
        ]
        s = MergedLoRAStrategy(
            m, torch.device("cuda"),
            loras=loras,
            layers_attr="transformer_blocks", blocks_to_swap=1,
        )
        s.set_active(["a", "b"])
        s.activate()
        try:
            # Drive a forward to trigger prefetches across all blocks.
            x = torch.randn(2, 16, dtype=torch.bfloat16, device="cuda")
            for blk in m.transformer_blocks:
                x = blk(x)
            torch.cuda.synchronize()
            # Each block's attn.weight should now equal base + sum(LoRA deltas).
            for i in range(4):
                expected = _expected_merged_weight(
                    captured_base[i], loras, i, "attn.weight",
                ).to("cuda")
                actual = m.transformer_blocks[i].attn.weight.detach()
                # bf16 has limited precision; addmm rounds. Tolerate
                # a few units in the last place.
                assert torch.allclose(actual, expected, rtol=0.01, atol=0.01), (
                    f"block {i} merged weight mismatch:\n"
                    f"  expected: {expected.flatten()[:4]}\n"
                    f"  actual:   {actual.flatten()[:4]}"
                )
        finally:
            s.deactivate()

    @CUDA
    def test_empty_active_set_runs_base_only(self) -> None:
        # set_active([]) means no LoRAs merged — forward sees pure base.
        m = _make_bf16_model(num_blocks=4, dim=16)
        captured = m.transformer_blocks[0].attn.weight.detach().clone()
        s = MergedLoRAStrategy(
            m, torch.device("cuda"),
            loras=[_make_lora("a", 4, 16)],
            layers_attr="transformer_blocks", blocks_to_swap=1,
        )
        s.set_active([])
        s.activate()
        try:
            x = torch.randn(2, 16, dtype=torch.bfloat16, device="cuda")
            for blk in m.transformer_blocks:
                x = blk(x)
            torch.cuda.synchronize()
            actual = m.transformer_blocks[0].attn.weight.detach()
            assert torch.allclose(
                actual, captured.to("cuda"), rtol=0.0, atol=0.0,
            ), "active=[] must leave base weights unmodified"
        finally:
            s.deactivate()


# ---------------------------------------------------------------------------
# Cache budget reporting
# ---------------------------------------------------------------------------


class TestCacheBytes:
    def test_factors_count_toward_cache_bytes(self) -> None:
        m = _make_bf16_model(num_blocks=4, dim=16)
        s_no_lora = MergedLoRAStrategy(
            m, torch.device("cpu"),
            loras=[],
            layers_attr="transformer_blocks", blocks_to_swap=1,
        )
        baseline = s_no_lora.cache_bytes

        m2 = _make_bf16_model(num_blocks=4, dim=16)
        s_with_lora = MergedLoRAStrategy(
            m2, torch.device("cpu"),
            loras=[_make_lora("a", num_blocks=4, dim=16, rank=4)],
            layers_attr="transformer_blocks", blocks_to_swap=1,
        )
        # Pinned host now also stores the LoRA factors.
        assert s_with_lora.cache_bytes > baseline
