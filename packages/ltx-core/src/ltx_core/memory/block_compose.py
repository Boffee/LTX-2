"""Compose a model with one or more :class:`BlockStreamer`s.

A :class:`BlockStreamingStrategy` glues together a model plus an
ordered list of components — typically:

1. A non-block :class:`PinnedWeights` (sibling modules, parent-module
   direct state) constructed with the streamers' :class:`SlotOwnership`
   filter so it ignores block-owned slots.
2. A :class:`TrainableMover` that moves LoRA / adapter weights to GPU
   on activate and back to CPU on deactivate.
3. One :class:`BlockStreamer` per homogeneous block list (single-list
   models use one; heterogeneous ones like Flux use two:
   ``transformer_blocks`` + ``single_transformer_blocks``).

Activate iterates the components in order; deactivate reverses
automatically via :class:`contextlib.ExitStack`. Cross-region tied
weights — block↔block, block↔non-block, or block↔trainable — are
detected at construction and raise; slot-local block streaming
cannot preserve such ties.

:func:`make_block_offloader` is the blessed factory for the common
"resolve a dotted attribute path to a block list and stream it"
case; for bespoke configurations (multiple homogeneous groups with
different ``blocks_to_swap``) construct components directly and
hand them to :class:`BlockStreamingStrategy`.
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
from .pinned_buffer import storage_key
from .pinned_weights import PinnedWeights
from .slot_graph import iter_buffer_slots, iter_param_slots
from .strategy import SlotOwnership

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TrainableMover — a component that moves trainable params on activate
# ---------------------------------------------------------------------------


class TrainableMover:
    """Moves all ``requires_grad=True`` params (and their grads) to a
    target device on activate; back to CPU on deactivate.

    Component shape (no model returned from activate). Reports
    ``cache_bytes=0`` because it owns no pinned storage — it just
    relocates parameters that already exist somewhere. Useful as a
    component inside :class:`BlockStreamingStrategy` so the activate /
    deactivate pipeline doesn't need to special-case trainable
    movement.
    """

    def __init__(self, model: nn.Module, target_device: torch.device) -> None:
        self._model = model
        self._target_device = target_device

    @property
    def cache_bytes(self) -> int:
        return 0

    @property
    def name(self) -> str:
        return "TrainableMover"

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


def detect_streaming_region_ties(  # noqa: PLR0912, PLR0915 (3-category check is naturally branchy)
    model: nn.Module, block_groups: Sequence[Sequence[nn.Module]]
) -> None:
    """Raise if any frozen storage is shared across streaming regions.

    Each entry in ``block_groups`` is one block list — one "region"
    per block. Everything else in ``model`` is the "non_block"
    region.

    Three configurations are unsupported:

    - **Cross-region ties** (block↔block, block↔non-block): the per-
      region pinning regimes can't coordinate to share storage.
    - **Mixed frozen/trainable ties** across any region boundary:
      the frozen side gets pinned and slot-swapped while the
      trainable side is moved separately on activate, breaking the
      sharing invariant silently.
    - **Intra-block ties** (two slots in the same block sharing
      storage): per-block stores walk ``named_parameters()`` with
      default duplicate removal and only swap one alias slot,
      leaving the other pointing at non-pinned data.

    Non-block-internal ties (the standard ``tie_weights()`` embed↔head
    pattern) are handled correctly by :class:`PinnedWeights`'s
    storage-key dedup and are NOT rejected here.

    Buffer detection classifies by SLOT location ``(parent_id, leaf)``,
    not by buffer object id. This catches the case where the same
    Python buffer object is registered at both a block-internal path
    and a non-block path: the composer's two pinning regimes would
    split the alias by replacing the non-block slot with a fresh
    pinned clone while the streamer pins the block side separately.
    """
    # Map each block param's id → "block:<group_idx>:<block_idx>" so any
    # param in the model can be classified into its region in O(1).
    param_id_to_region: dict[int, str] = {}
    for group_idx, blocks in enumerate(block_groups):
        for block_idx, layer in enumerate(blocks):
            for p in layer.parameters():
                param_id_to_region.setdefault(
                    id(p), f"block:{group_idx}:{block_idx}"
                )

    groups: dict[tuple, list[tuple[str, str, bool, int, str]]] = {}
    for s in iter_param_slots(model):
        if s.param.numel() == 0:
            continue
        region = param_id_to_region.get(id(s.param), "non_block")
        skey = storage_key(s.param.data)
        groups.setdefault(skey, []).append(
            (region, s.name, s.param.requires_grad, id(s.parent), s.leaf)
        )

    for members in groups.values():
        regions = {region for region, _, _, _, _ in members}
        names = sorted(name for _, name, _, _, _ in members)
        if len(regions) > 1:
            raise ValueError(
                f"Block streaming does not support tied parameters across "
                f"streamed regions: storage shared by {names}. Slot-local "
                "block streaming cannot preserve cross-region tying "
                "(neither frozen↔frozen nor frozen↔trainable). Use "
                "whole-model PinnedWeights, disable block streaming, or "
                "untie the parameters."
            )
        sole_region = next(iter(regions))
        if sole_region.startswith("block:"):
            slot_locs = {(pid, leaf) for _, _, _, pid, leaf in members}
            if len(slot_locs) > 1:
                raise ValueError(
                    f"Block streaming does not support intra-block tied "
                    f"parameters: storage shared by {names} within "
                    f"{sole_region}. _BlockPinnedStore cannot preserve "
                    "the tying invariant — one alias would stay pointing "
                    "at non-pinned data. Untie the parameters or use "
                    "whole-model PinnedWeights instead."
                )

    # Classify by SLOT location (parent_id, leaf), not buffer object id.
    # The same buffer object at both block-internal and non-block paths
    # would otherwise be classified only as block — we'd miss that the
    # composer's two pinning regimes (per-block clone vs composed
    # PinnedWeights) would split the alias by replacing the non-block
    # slot.
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


# ---------------------------------------------------------------------------
# BlockStreamingStrategy — public composite ModelStrategy
# ---------------------------------------------------------------------------


class BlockStreamingStrategy:
    """A :class:`~block_offload.strategy.ModelStrategy` that streams
    one or more block lists, plus pins everything else, plus moves
    trainable params, by composing an ordered list of components.

    Activation order = component order. Deactivation reverses
    automatically. Use :func:`make_block_offloader` for the common
    ``layers_attr=...`` case; construct components manually only for
    bespoke configurations (e.g., per-group ``blocks_to_swap``).

    Components must implement ``cache_bytes`` / ``activate()`` /
    ``deactivate()`` (the same structural shape as
    :class:`ModelStrategy` minus the model return). Standard
    components in this package:

    - :class:`PinnedWeights` (with a ``skip_slots`` filter for the
      streamers' slots)
    - :class:`TrainableMover`
    - one or more :class:`BlockStreamer`s

    Parameters
    ----------
    model:
        The full model. ``activate()`` returns this. After
        :meth:`deactivate`, drop both the strategy reference and the
        model reference to release pinned memory — strategies don't
        have a destructive ``close()``; resource cleanup happens via
        reference dropping + GC.
    components:
        Ordered list of components. Activated in order;
        deactivated in reverse via :class:`contextlib.ExitStack`.
    """

    def __init__(
        self, model: nn.Module, components: Sequence[Any],
    ) -> None:
        self._model: nn.Module | None = model
        self._components: list[Any] = list(components)
        # ExitStack of registered deactivate callbacks, set by
        # activate() and consumed by deactivate(). Presence is the
        # de-facto "active" indicator.
        self._teardown_stack: contextlib.ExitStack | None = None

    @property
    def cache_bytes(self) -> int:
        return sum(c.cache_bytes for c in self._components)

    # ------------------------------------------------------------------
    # ModelStrategy lifecycle
    # ------------------------------------------------------------------

    def activate(self) -> nn.Module:
        """Activate components in order. ``with ExitStack`` auto-rolls
        back partial activation on exception (its ``__exit__`` chains
        cleanup failures via ``__context__`` so all are visible in the
        traceback). On full success, ``stack.pop_all()`` detaches the
        callbacks so they survive until our :meth:`deactivate`.

        **Lifecycle is caller's responsibility.** Calling activate()
        twice without an intervening deactivate() will double-activate
        components — undefined behavior."""
        assert self._model is not None

        with contextlib.ExitStack() as stack:
            for component in self._components:
                component.activate()
                stack.callback(component.deactivate)
            self._teardown_stack = stack.pop_all()
        return self._model

    def deactivate(self) -> None:
        """Run registered deactivate callbacks in reverse order.
        Idempotent — safe to call before activate or multiple times.
        If multiple components raise, only the latest propagates
        (``ExitStack.close()`` without an incoming exception does NOT
        chain via ``__context__``); if multi-failure visibility
        matters, components should log internally.

        After deactivate, drop the strategy reference (and the model
        reference if you don't need it anymore) to release pinned
        memory."""
        stack = self._teardown_stack
        self._teardown_stack = None
        if stack is not None:
            stack.close()

    def __enter__(self) -> nn.Module:
        return self.activate()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.deactivate()


# ---------------------------------------------------------------------------
# make_block_offloader — blessed factory
# ---------------------------------------------------------------------------


def make_block_offloader(
    model: nn.Module,
    target_device: torch.device,
    *,
    layers_attr: str | Sequence[str],
    blocks_to_swap: int | Sequence[int],
    prefetch_count: int | Sequence[int] = 2,
    strict_homogeneous: bool = True,
) -> BlockStreamingStrategy:
    """Build a :class:`BlockStreamingStrategy` for a model whose
    block lists live at the given dotted attribute path(s).

    Single-list models pass ``layers_attr="transformer_blocks"``;
    heterogeneous models pass a list, e.g.
    ``layers_attr=["transformer_blocks", "single_transformer_blocks"]``.
    Each path becomes its own homogeneous streaming group with its
    own GPU slot pool — heterogeneous models keep the slot-pool
    benefit per-group instead of degrading to per-load
    ``cudaMalloc``.

    For PEFT-wrapped models, include the wrapper prefix in the
    path (e.g. ``"base_model.model.transformer_blocks"``).

    Parameters
    ----------
    model:
        The model containing the block list(s).
    target_device:
        GPU device.
    layers_attr:
        Dotted attribute path(s) to the ``nn.ModuleList`` block
        list(s). Single string or list.
    blocks_to_swap:
        Per-group blocks to keep on CPU. Either a single int (applied
        to all groups) or one int per group.
    prefetch_count:
        Per-group prefetch depth. Same broadcasting as
        ``blocks_to_swap``.
    strict_homogeneous:
        Forwarded to each :class:`BlockStreamer`. When True (default),
        any non-homogeneous group raises at construction. Pass False
        to opt into the per-load-allocation fallback.

    Pre-conditions
    --------------
    Cross-region tied weights are rejected here BEFORE any pinning
    runs, so tied-weight failures don't leave the model half-pinned.

    Lifecycle
    ---------
    The returned strategy is in the "constructed" state — pinning
    is done, ``cache_bytes`` is final, no GPU resources allocated.
    Call ``activate()`` (or use as context manager) to bring
    everything to GPU.

    Failure semantics
    -----------------
    No factory-level rollback. If construction raises mid-way (e.g.,
    the K-th BlockStreamer's ``pin_memory()`` fails after streamers
    0..K-1 succeeded), the partial state stays in the user's model:
    the model's slots for blocks 0..K-1 are mutated to point at
    pinned cpu_param Parameters and stay that way until the caller
    drops the model reference. Acceptable for a low-level library —
    the caller's recovery is to drop the model and rebuild.
    """
    layer_paths: list[str] = (
        [layers_attr] if isinstance(layers_attr, str) else list(layers_attr)
    )
    if not layer_paths:
        raise ValueError("layers_attr must contain at least one path")

    n = len(layer_paths)
    swap_list = _broadcast(blocks_to_swap, n, "blocks_to_swap")
    pf_list = _broadcast(prefetch_count, n, "prefetch_count")

    block_groups: list[list[nn.Module]] = [
        list(_resolve_attr(model, p)) for p in layer_paths
    ]
    for i, blocks in enumerate(block_groups):
        if not blocks:
            raise ValueError(f"layers_attr[{i}] = {layer_paths[i]!r} resolved to empty list")

    # Validate ties on the live model BEFORE pinning anything.
    detect_streaming_region_ties(model, block_groups)
    # Whole-model CPU move so subsequent .pin_memory() calls succeed.
    # (BlockStreamer + PinnedWeights also each auto-move; this just
    # consolidates the move once for clarity.)
    model.to("cpu")

    # Build streamers + PinnedWeights. If construction fails partway,
    # the partial state goes out of scope — GC frees the
    # PinnedParamBuffer objects. Any model slots already mutated stay
    # mutated (caller must drop the model to release pinned memory).
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
            )
        )

    # Union of every streamer's slot filter. SlotOwnership tuples
    # are (id(parent), leaf, kind) which survive slot mutation,
    # so the PinnedWeights walk below correctly skips block-owned
    # slots.
    skip_slots: set[SlotOwnership] = set()
    for s in streamers:
        skip_slots |= s.slot_filter

    non_block: PinnedWeights | None = None
    if _has_non_block_pinnable_content(model, skip_slots):
        non_block = PinnedWeights(model, target_device, skip_slots=skip_slots)

    components: list[Any] = []
    if non_block is not None:
        components.append(non_block)
    components.append(TrainableMover(model, target_device))
    components.extend(streamers)

    return BlockStreamingStrategy(model=model, components=components)


def _resolve_attr(module: nn.Module, dotted_path: str) -> nn.ModuleList:
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


def _has_non_block_pinnable_content(
    model: nn.Module, skip_slots: set[SlotOwnership]
) -> bool:
    """True if the outer model has any frozen param or buffer at a
    slot the streamers DON'T own. Decides whether to construct the
    composed PinnedWeights at all (it raises on empty input)."""
    if any(
        not s.param.requires_grad and s.slot not in skip_slots
        for s in iter_param_slots(model)
    ):
        return True
    return any(s.slot not in skip_slots for s in iter_buffer_slots(model))


__all__ = [
    "BlockStreamingStrategy",
    "TrainableMover",
    "detect_streaming_region_ties",
    "make_block_offloader",
]
