"""Unified block-streaming strategy with optional LoRA merge.

Replaces the former ``BlockStreamingStrategy`` (streaming without LoRA)
and ``MergedLoRAStrategy`` (streaming with LoRA but no trainable support)
with a single class that composes block streaming, non-block pinning,
trainable parameter movement, and optional per-weight LoRA transforms.

LoRA transforms attach to :class:`~ltx_core.memory.PinnedParamBuffer`
objects, so they fire automatically when any consumer (pooled or
fallback, block or non-block) copies a weight to GPU. See
:class:`~ltx_core.memory.LoRATransform` for the per-weight primitive.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Sequence
from types import TracebackType
from typing import Any

import torch
from torch import nn

from .block_compose import (
    TrainableWeights,
    detect_streaming_region_ties,
)
from .block_streamer import BlockStreamer
from .merged_lora import (
    KeyTransformT,
    LoRA,
    LoRATransform,
    concat_lora_factors,
    default_key_transform,
    pair_and_validate,
)
from .pinned_buffer import PinnedParamBuffer
from .pinned_weights import PinnedWeights
from .slot_graph import iter_buffer_slots, iter_param_slots
from .strategy import ModelStrategyComponent, SlotOwnership

logger = logging.getLogger(__name__)

__all__ = ["BlockOffloader"]


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
