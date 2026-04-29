"""Utilities for composing block-streaming strategies.

:class:`TrainableWeights` handles the lifecycle of trainable params
(LoRA adapters, PEFT layers) alongside frozen block streaming.

:func:`detect_streaming_region_ties` validates that no frozen storage
is shared across streaming regions before pinning begins.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from types import TracebackType

import torch
from torch import nn

from .pinned_buffer import storage_key
from .slot_graph import iter_buffer_slots, iter_param_slots

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TrainableWeights — lifecycle handler for trainable params
# ---------------------------------------------------------------------------


class TrainableWeights:
    """Strategy component for the model's trainable parameters.

    The trainable counterpart to :class:`PinnedWeights`. Both components
    bring their managed params to the target device on
    :meth:`activate` and return them to CPU on :meth:`deactivate`,
    but the mechanisms are mirror images:

    - :class:`PinnedWeights` owns pinned-CPU clones, slot-replaces the
      Parameter wrapper at every transition. Frozen-only — slot
      replacement orphans optimizer state.
    - :class:`TrainableWeights` owns nothing (``cache_bytes=0``); the
      user's Parameter objects stay alive in their slots, and only
      ``p.data`` storage moves via ``p.data = p.data.to(device)``.
      Identity-preserving — optimizer state survives.

    Walks ``model.parameters()`` each transition (deduped by Parameter
    identity), so the standard ``tie_weights()`` pattern (one Parameter
    aliased at multiple slots) is handled correctly. Distinct-Parameter
    tied storage is rejected upstream by
    :func:`detect_streaming_region_ties` because moving each Parameter
    independently would untie the alias on GPU.
    """

    def __init__(self, model: nn.Module, target_device: torch.device) -> None:
        self._model = model
        self._target_device = target_device

    @property
    def cache_bytes(self) -> int:
        return 0

    @property
    def name(self) -> str:
        return "TrainableWeights"

    def activate(self) -> None:
        _move_trainable(self._model, self._target_device)

    def deactivate(self) -> None:
        """Idempotent — safe to call before activate or multiple
        times (.to(cpu) on cpu tensor is a no-op)."""
        _move_trainable(self._model, torch.device("cpu"))

    def __enter__(self) -> None:
        self.activate()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.deactivate()


def _move_trainable(model: nn.Module, device: torch.device) -> None:
    for p in model.parameters():
        if p.requires_grad:
            if p.data.device != device:
                p.data = p.data.to(device)
            if p.grad is not None and p.grad.device != device:
                p.grad = p.grad.to(device)


# ---------------------------------------------------------------------------
# Cross-region tied-weight detection
# ---------------------------------------------------------------------------


def detect_streaming_region_ties(  # noqa: PLR0912
    model: nn.Module, block_groups: Sequence[Sequence[nn.Module]]
) -> None:
    """Raise if any frozen storage is shared across streaming regions.

    Each entry in ``block_groups`` is one block list — one "region"
    per block. Everything else in ``model`` is the "non_block"
    region.

    Three configurations are unsupported:

    - **Cross-region ties** (block<->block, block<->non-block): the per-
      region pinning regimes can't coordinate to share storage.
    - **Mixed frozen/trainable ties** anywhere — across regions OR
      within a single region. The frozen side gets pinned and slot-
      replaced while the trainable side is moved via storage swap on
      activate; the two mechanisms cannot share a tied storage
      without breaking aliasing invariants silently.
    - **Intra-block ties** (two slots in the same block sharing
      storage): per-block stores walk ``named_parameters()`` with
      default duplicate removal and only swap one alias slot,
      leaving the other pointing at non-pinned data.

    Non-block-internal all-frozen ties (the standard ``tie_weights()``
    embed<->head pattern) are handled correctly by
    :class:`PinnedWeights`'s storage-key dedup and are NOT rejected
    here.
    """
    param_id_to_region: dict[int, str] = {}
    for group_idx, blocks in enumerate(block_groups):
        for block_idx, layer in enumerate(blocks):
            for p in layer.parameters():
                param_id_to_region.setdefault(
                    id(p), f"block:{group_idx}:{block_idx}"
                )

    groups: dict[tuple, list[tuple[str, str, bool, int, str, int]]] = {}
    for s in iter_param_slots(model):
        if s.param.numel() == 0:
            continue
        region = param_id_to_region.get(id(s.param), "non_block")
        skey = storage_key(s.param.data)
        groups.setdefault(skey, []).append(
            (region, s.name, s.param.requires_grad, id(s.parent), s.leaf, id(s.param))
        )

    for members in groups.values():
        regions = {region for region, _, _, _, _, _ in members}
        names = sorted(name for _, name, _, _, _, _ in members)
        grads = {grad for _, _, grad, _, _, _ in members}
        if len(grads) > 1:
            raise ValueError(
                f"Tied storage spans both trainable and frozen parameters: "
                f"{names}. Slot-replace (frozen) and storage-swap "
                "(trainable) mechanisms cannot share a tied storage. "
                "Untie the parameters or freeze/unfreeze them consistently."
            )
        if all(grads):
            param_ids = {pid for _, _, _, _, _, pid in members}
            if len(param_ids) > 1:
                raise ValueError(
                    f"All-trainable tied storage with distinct Parameter "
                    f"objects: {names}. TrainableWeights moves each Parameter "
                    "independently via p.data = ... and would break the "
                    "storage alias on GPU. Untie the parameters or use "
                    "tie_weights() to share a single Parameter object."
                )
        if len(regions) > 1:
            raise ValueError(
                f"Block streaming does not support tied parameters across "
                f"streamed regions: storage shared by {names}. Slot-local "
                "block streaming cannot preserve cross-region tying. Use "
                "whole-model PinnedWeights, disable block streaming, or "
                "untie the parameters."
            )
        sole_region = next(iter(regions))
        if sole_region.startswith("block:"):
            slot_locs = {(pid, leaf) for _, _, _, pid, leaf, _ in members}
            if len(slot_locs) > 1:
                raise ValueError(
                    f"Block streaming does not support intra-block tied "
                    f"parameters: storage shared by {names} within "
                    f"{sole_region}. _BlockPinnedStore cannot preserve "
                    "the tying invariant — one alias would stay pointing "
                    "at non-pinned data. Untie the parameters or use "
                    "whole-model PinnedWeights instead."
                )

    block_buffer_slot_regions: dict[tuple[int, str], set[str]] = {}
    for group_idx, blocks in enumerate(block_groups):
        for block_idx, layer in enumerate(blocks):
            for s in iter_buffer_slots(layer):
                block_buffer_slot_regions.setdefault(
                    (id(s.parent), s.leaf), set()
                ).add(f"block:{group_idx}:{block_idx}")

    buf_groups: dict[tuple, list[tuple[str, str, int]]] = {}
    for s in iter_buffer_slots(model):
        if s.buffer.numel() == 0:
            continue
        regions = block_buffer_slot_regions.get((id(s.parent), s.leaf), {"non_block"})
        for region in regions:
            buf_groups.setdefault(storage_key(s.buffer), []).append(
                (region, s.name, id(s.buffer))
            )

    for members in buf_groups.values():
        regions = {region for region, _, _ in members}
        names = sorted({name for _, name, _ in members})
        if len(regions) > 1:
            raise ValueError(
                f"Block streaming does not support tied buffers across "
                f"streamed regions: storage shared by {names}. The two "
                "pinning regimes (per-block clone vs composed "
                "PinnedWeights) can't coordinate to preserve the alias. "
                "Untie the buffers or use whole-model PinnedWeights "
                "instead."
            )
        sole_region = next(iter(regions))
        if sole_region.startswith("block:"):
            distinct_ids = {bid for _, _, bid in members}
            if len(distinct_ids) > 1:
                raise ValueError(
                    f"Block streaming does not support intra-block tied "
                    f"buffers: storage shared by {names} within "
                    f"{sole_region}. _BlockPinnedStore clones each "
                    "buffer independently — the alias would break. "
                    "Untie the buffers or use whole-model PinnedWeights."
                )


__all__ = [
    "TrainableWeights",
    "detect_streaming_region_ties",
]
