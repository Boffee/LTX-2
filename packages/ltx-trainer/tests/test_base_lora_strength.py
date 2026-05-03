"""Regression tests for ``LtxvTrainer._set_base_lora_strength``.

Setting ``base_lora.strength: 0`` must fully detach the LoRA from the
offloader so that no ``LoRATransform`` is attached and no per-DMA
``addmm_`` runs. The helper is invoked unbound on a stub holding only
``_base_lora`` and ``_model_offloader`` to avoid spinning up a full
trainer (which requires GPU + heavy model files).
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from ltx_trainer.trainer import LtxvTrainer
from torch_offload import LoRA, ModelOffloader


def _make_block_model(num_blocks: int = 2, dim: int = 16) -> nn.Module:
    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = nn.Linear(dim, dim, bias=False)

    class M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.transformer_blocks = nn.ModuleList(
                [Block() for _ in range(num_blocks)]
            )

    m = M().to(torch.bfloat16)
    for p in m.parameters():
        p.requires_grad = False
    return m


def _make_lora(num_blocks: int = 2, dim: int = 16, rank: int = 4) -> LoRA:
    sd: dict[str, torch.Tensor] = {}
    for b in range(num_blocks):
        base = f"transformer_blocks.{b}.attn"
        sd[f"{base}.lora_A.weight"] = torch.randn(rank, dim, dtype=torch.float32)
        sd[f"{base}.lora_B.weight"] = torch.randn(dim, rank, dtype=torch.float32)
    return LoRA(state_dict=sd)


class _Stub:
    """Minimal duck-typed self for the unbound helper."""

    def __init__(self, base_lora: LoRA | None, offloader: ModelOffloader | None) -> None:
        self._base_lora = base_lora
        self._model_offloader = offloader


def _has_transform(offloader: ModelOffloader, target_key: str) -> bool:
    buf = offloader._reverse_index.get(target_key)
    return buf is not None and buf.transform is not None


@pytest.fixture
def offloader_with_lora() -> tuple[ModelOffloader, LoRA, list[str]]:
    model = _make_block_model()
    offloader = ModelOffloader(
        model,
        torch.device("cpu"),
        layers_attr="transformer_blocks",
        blocks_to_swap=1,
    )
    lora = _make_lora()
    target_keys = [f"transformer_blocks.{b}.attn.weight" for b in range(2)]
    yield offloader, lora, target_keys
    offloader.deactivate()


def test_nonzero_strength_attaches_transform(
    offloader_with_lora: tuple[ModelOffloader, LoRA, list[str]],
) -> None:
    offloader, lora, keys = offloader_with_lora
    LtxvTrainer._set_base_lora_strength(_Stub(lora, offloader), 0.5)

    for k in keys:
        assert _has_transform(offloader, k), f"Expected transform attached for {k}"


def test_zero_strength_attaches_no_transform(
    offloader_with_lora: tuple[ModelOffloader, LoRA, list[str]],
) -> None:
    offloader, lora, keys = offloader_with_lora
    LtxvTrainer._set_base_lora_strength(_Stub(lora, offloader), 0.0)

    for k in keys:
        assert not _has_transform(offloader, k), (
            f"Transform should be detached when strength=0, but {k} has one"
        )


def test_zero_strength_clears_previously_attached_transform(
    offloader_with_lora: tuple[ModelOffloader, LoRA, list[str]],
) -> None:
    """Mirrors the validation→training transition where val_strength != 0
    but training strength is 0: the helper must clear the transform that
    validation just attached, not re-attach with strength=0.
    """
    offloader, lora, keys = offloader_with_lora
    stub = _Stub(lora, offloader)

    LtxvTrainer._set_base_lora_strength(stub, 0.6)
    assert all(_has_transform(offloader, k) for k in keys)

    LtxvTrainer._set_base_lora_strength(stub, 0.0)
    for k in keys:
        assert not _has_transform(offloader, k)


def test_no_base_lora_is_noop() -> None:
    """If no base LoRA is configured, the helper must not touch the offloader."""
    model = _make_block_model()
    offloader = ModelOffloader(
        model,
        torch.device("cpu"),
        layers_attr="transformer_blocks",
        blocks_to_swap=1,
    )
    try:
        LtxvTrainer._set_base_lora_strength(_Stub(None, offloader), 0.5)
        for k in (f"transformer_blocks.{b}.attn.weight" for b in range(2)):
            assert not _has_transform(offloader, k)
    finally:
        offloader.deactivate()


def test_no_offloader_is_noop() -> None:
    """When offloading is disabled (e.g. blocks_to_swap=0), the helper short-circuits."""
    lora = _make_lora()
    LtxvTrainer._set_base_lora_strength(_Stub(lora, None), 0.5)
