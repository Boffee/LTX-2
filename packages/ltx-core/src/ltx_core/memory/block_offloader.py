"""Unified block-streaming strategy with optional LoRA merge.

Composes block streaming, non-block pinning, trainable parameter
movement, and optional per-weight LoRA transforms into a single
:class:`BlockOffloader` class.

Also provides :class:`TrainableWeights` (identity-preserving mover
for trainable params) and :func:`detect_streaming_region_ties`
(construction-time validation), both used internally by
:class:`BlockOffloader` and exported for direct use / testing.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Sequence
from types import TracebackType
from typing import Any

import torch
from torch import nn

from .block_streamer import BlockStreamer
from .lora import (
    KeyTransformT,
    LoRA,
    LoRATransform,
    concat_lora_factors,
    default_key_transform,
    pair_and_validate,
)
from .pinned_buffer import PinnedParamBuffer, storage_key
from .pinned_weights import PinnedWeights
from .protocols import ModelStrategyComponent, SlotOwnership
from .slots import iter_buffer_slots, iter_param_slots

logger = logging.getLogger(__name__)

__all__ = [
    "BlockOffloader",
    "TrainableWeights",
    "detect_streaming_region_ties",
]


class BlockOffloader:
    """Stream transformer blocks between pinned CPU and GPU with
    optional LoRA merge and trainable-parameter support.

    Composes :class:`PinnedWeights` (non-block frozen params),
    :class:`TrainableWeights` (LoRA / adapter params), and one or more
    :class:`BlockStreamer`\\ s internally. LoRA transforms are set via
    :meth:`set_loras` and attach to individual
    :class:`PinnedParamBuffer` objects so the merge fires automatically
    during DMA — no separate merge strategy needed.

    Parameters
    ----------
    model:
        The model containing the block list(s). Must be on CPU.
    target_device:
        GPU device for inference / training.
    layers_attr:
        Dotted attribute path(s) to ``nn.ModuleList`` block list(s).
        Single string or sequence. For PEFT-wrapped models, include
        the PEFT prefix (e.g. ``"base_model.model.transformer_blocks"``).
    blocks_to_swap:
        Per-group count of blocks to keep on CPU. Single int (broadcast
        to all groups) or one int per group.
    prefetch_count:
        Per-group prefetch depth. Same broadcasting as *blocks_to_swap*.
    strict_homogeneous:
        Forwarded to each :class:`BlockStreamer`. When True (default),
        non-homogeneous groups raise at construction. Pass False for
        the per-load-allocation fallback.
    key_transform:
        Applied to LoRA state-dict base keys before matching against
        model parameter names. Defaults to stripping the
        ``diffusion_model.`` prefix common in ComfyUI LoRA files.
        Pass ``None`` to disable.
    """

    def __init__(
        self,
        model: nn.Module,
        target_device: torch.device,
        *,
        layers_attr: str | Sequence[str],
        blocks_to_swap: int | Sequence[int],
        prefetch_count: int | Sequence[int] = 2,
        strict_homogeneous: bool = True,
        key_transform: KeyTransformT = default_key_transform,
    ) -> None:
        layer_paths: list[str] = (
            [layers_attr] if isinstance(layers_attr, str) else list(layers_attr)
        )
        if not layer_paths:
            raise ValueError("layers_attr must contain at least one path")

        n = len(layer_paths)
        swap_list = _broadcast(blocks_to_swap, n, "blocks_to_swap")
        pf_list = _broadcast(prefetch_count, n, "prefetch_count")

        block_groups: list[list[nn.Module]] = [
            list(_resolve_layers_attr(model, p)) for p in layer_paths
        ]
        for i, blocks in enumerate(block_groups):
            if not blocks:
                raise ValueError(
                    f"layers_attr[{i}] = {layer_paths[i]!r} resolved to empty list"
                )

        detect_streaming_region_ties(model, block_groups)
        model.to("cpu")

        trainable_slots: set[SlotOwnership] = {
            s.slot for s in iter_param_slots(model) if s.param.requires_grad
        }

        streamers: list[BlockStreamer] = []
        for i, blocks in enumerate(block_groups):
            streamers.append(
                BlockStreamer(
                    blocks=blocks,
                    target_device=target_device,
                    blocks_to_swap=swap_list[i],
                    prefetch_count=pf_list[i],
                    name=f"BlockStreamer[{layer_paths[i]}]",
                    strict_homogeneous=strict_homogeneous,
                    skip_slots=trainable_slots,
                )
            )

        skip_slots: set[SlotOwnership] = set(trainable_slots)
        for s in streamers:
            skip_slots |= s.slot_filter

        non_block: PinnedWeights | None = None
        if _has_pinnable_content(model, skip_slots):
            non_block = PinnedWeights(model, target_device, skip_slots=skip_slots)

        components: list[ModelStrategyComponent] = []
        if non_block is not None:
            components.append(non_block)
        components.append(TrainableWeights(model, target_device))
        components.extend(streamers)

        self._model = model
        self._target_device = target_device
        self._key_transform = key_transform
        self._layer_paths = layer_paths
        self._components = components
        self._streamers = streamers
        self._non_block = non_block
        self._teardown_stack: contextlib.ExitStack | None = None

        self._reverse_index = self._build_reverse_index(
            streamers, layer_paths, non_block,
        )
        self._lora_factor_bytes: int = 0

    # ------------------------------------------------------------------ API

    def set_loras(self, loras: Sequence[LoRA]) -> None:
        """Replace all LoRAs. Must be called while deactivated.

        Processes flat safetensors state dicts: applies
        ``key_transform``, pairs A/B factors, matches to model
        parameters via the reverse index, concatenates per target,
        and attaches a :class:`LoRATransform` to each matched
        :class:`PinnedParamBuffer`.

        Pass an empty sequence to clear all LoRAs (base-only forward).
        """
        if self._teardown_stack is not None:
            raise RuntimeError(
                "BlockOffloader.set_loras() requires the offloader "
                "to be inactive. Call deactivate() first."
            )
        for buf in self._reverse_index.values():
            buf.transform = None
        self._lora_factor_bytes = 0

        if not loras:
            return

        shape_index: dict[str, tuple[int, ...]] = {
            name: tuple(buf.cpu_param.shape)
            for name, buf in self._reverse_index.items()
        }
        raw = pair_and_validate(loras, shape_index, self._key_transform)

        pending: dict[str, LoRATransform] = {}
        for target_key, factors in raw.items():
            buf = self._reverse_index[target_key]
            if buf.cpu_param.dtype not in (torch.bfloat16, torch.float16):
                raise ValueError(
                    f"LoRA target {target_key!r} has dtype "
                    f"{buf.cpu_param.dtype}; addmm_ merge requires "
                    f"bf16/fp16. Use PEFT routed mode for quantized params."
                )
            pair = concat_lora_factors(
                factors, buf.cpu_param.dtype, torch.device("cpu"),
            )
            if pair is None:
                continue
            a_cat, b_cat = pair
            pending[target_key] = LoRATransform(a_cat, b_cat)

        for target_key, transform in pending.items():
            self._reverse_index[target_key].transform = transform
            self._lora_factor_bytes += transform.nbytes

    # ------------------------------------------------- ModelStrategy interface

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def cache_bytes(self) -> int:
        return (
            sum(c.cache_bytes for c in self._components)
            + self._lora_factor_bytes
        )

    def activate(self) -> None:
        with contextlib.ExitStack() as stack:
            for component in self._components:
                stack.callback(component.deactivate)
                component.activate()
            self._teardown_stack = stack.pop_all()

    def deactivate(self) -> None:
        stack = self._teardown_stack
        self._teardown_stack = None
        if stack is not None:
            stack.close()

    def __enter__(self) -> nn.Module:
        self.activate()
        return self.model

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.deactivate()

    # ----------------------------------------------------------- Internals

    @staticmethod
    def _build_reverse_index(
        streamers: list[BlockStreamer],
        layer_paths: list[str],
        non_block: PinnedWeights | None,
    ) -> dict[str, PinnedParamBuffer]:
        """Map model param qualified names to their PinnedParamBuffer.

        Block params are reconstructed as
        ``"{layer_path}.{block_idx}.{buf.name}"``.
        Non-block params use ``buf.name`` directly (already model-relative).
        """
        index: dict[str, PinnedParamBuffer] = {}

        for streamer, layer_path in zip(streamers, layer_paths, strict=True):
            for block_idx, block_bufs in enumerate(streamer.param_bufs_per_block):
                for buf in block_bufs:
                    full_name = f"{layer_path}.{block_idx}.{buf.name}"
                    index[full_name] = buf

        if non_block is not None:
            for buf, _locs in non_block.slots:
                index[buf.name] = buf

        return index


# ---------------------------------------------------------------------------
# Module-private helpers (used only by BlockOffloader constructor)
# ---------------------------------------------------------------------------


def _resolve_layers_attr(module: nn.Module, dotted_path: str) -> nn.ModuleList:
    obj: Any = module
    for part in dotted_path.split("."):
        obj = getattr(obj, part)
    if not isinstance(obj, nn.ModuleList):
        raise TypeError(
            f"Expected nn.ModuleList at '{dotted_path}', got {type(obj).__name__}"
        )
    return obj


def _broadcast(value: int | Sequence[int], n: int, name: str) -> list[int]:
    if isinstance(value, int):
        return [value] * n
    out = list(value)
    if len(out) != n:
        raise ValueError(f"{name} length {len(out)} != layers_attr length {n}")
    return out


def _has_pinnable_content(
    model: nn.Module, skip_slots: set[SlotOwnership]
) -> bool:
    return any(
        s.slot not in skip_slots for s in iter_param_slots(model)
    ) or any(
        s.slot not in skip_slots for s in iter_buffer_slots(model)
    )


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
