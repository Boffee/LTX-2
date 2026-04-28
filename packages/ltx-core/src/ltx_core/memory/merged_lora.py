"""Merged-LoRA strategy for stacked-LoRA inference.

For workloads that need 3+ stacked LoRAs on a bf16/fp16 base, the
PEFT-style routed forward pays a per-LoRA cost on every step. This
strategy takes the alternative path used by ComfyUI's ModelPatcher
and most production fp16 inference servers: merge LoRA deltas into
the base weights at activation time, run forward as a single matmul
per layer regardless of how many LoRAs are stacked.

Architecture
------------
Built on the existing :class:`BlockStreamer` + :class:`PinnedWeights`
infrastructure. The novel piece is a ``post_load`` callback registered
with the streamer that runs on the prefetch CUDA stream after each
block's bytes are DMA'd in:

    pinned bf16 base        ─DMA→  pool slot
    GPU-resident LoRA factors ──→  slot.weights += scale * (B @ A)   (addmm_)
                                    via post_load hook on prefetch stream

CUDA orders the merge after the DMA automatically because both run
on the same (prefetch) stream. The pool's existing readiness event
fires AFTER both, so the main compute stream sees fully-merged
weights when it consumes the slot.

The "free unmerge" property comes from the streamer's slot recycling:
when a block is evicted and re-loaded, ``slot.copy_from`` overwrites
GPU bytes with pristine pinned base bytes. The next merge starts from
clean base — no subtract-and-restore pattern, no drift.

Constraints
-----------
- **Base must be bf16 or fp16**. ``addmm_`` requires arithmetic-
  capable target dtype; fp8 wrappers and quanto :class:`WeightQBytesTensor`
  do not support in-place add. For fp8 base, use PEFT routed mode
  (no merge) — at K≤4 LoRAs, the fp8 forward speedup outweighs the
  routing overhead anyway. fp8 + merge with prefetch-time upcast is
  a planned v2 extension.
- **LoRA set is fixed during the active window**. Switching combos
  requires :meth:`deactivate` → :meth:`set_active` → :meth:`activate`.
  In stream-offload mode this is cheap because activation doesn't
  bulk-load anything — first forward triggers per-block merges with
  the new active set lazily.
- **All LoRAs registered at construction**. ``cache_bytes`` is final
  at admission per the package contract. Adding a LoRA after admission
  would invalidate :class:`ModelCache` budget accounting.
- **Active list is ordered**. ``set_active`` takes a sequence, not a
  set. bf16 addition is non-associative, so deterministic output
  requires a stable iteration order.

Cost
----
At LTX-2 shape (d≈2048, r=128, ~30 blocks, ~6 LoRA-target layers per
block), the per-block merge cost is roughly 60μs per active LoRA
(low-rank matmul + element-wise add). For K=3 active, that's ~180μs
per block × ~30 blocks ≈ 5ms total per forward — fully hidden by
prefetch overlap when ``prefetch_count`` is appropriate. Steady-state
forward overhead vs. unmerged base: ~0%.

Compare PEFT routed at K=3 on the same hardware: ~10-15% forward
overhead on every step, every layer.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Any

import torch
from torch import nn

from .block_compose import _has_non_block_pinnable_content, _resolve_attr
from .block_streamer import BlockStreamer
from .pinned_weights import PinnedWeights
from .slot_graph import iter_param_slots
from .strategy import SlotOwnership

logger = logging.getLogger(__name__)


__all__ = [
    "LoRABundle",
    "LoRALayerFactors",
    "MergedLoRAStrategy",
]


# ---------------------------------------------------------------------------
# LoRA data types — user-facing input format
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class LoRALayerFactors:
    """LoRA factors for one target layer.

    ``A`` is shape ``(rank, in_dim)``; ``B`` is shape ``(out_dim, rank)``.
    The merged delta is ``scaling * (B @ A)`` — a ``(out_dim, in_dim)``
    matrix added to the corresponding base weight.

    ``scaling`` follows the standard PEFT convention (``alpha / rank``)
    when constructing from a PEFT-trained adapter.

    The strategy clones, dtype-casts (to match base dtype), and pins
    these tensors at construction. The user-supplied tensors are not
    retained after :class:`MergedLoRAStrategy.__init__` returns.
    """

    A: torch.Tensor
    B: torch.Tensor
    scaling: float


@dataclass(slots=True)
class LoRABundle:
    """All factors for one named adapter.

    ``blocks`` maps ``block_idx`` to a dict from in-block parameter
    qualname (relative to the block module) to :class:`LoRALayerFactors`.
    For example, with target attention's q-projection in block 0:

        LoRABundle(
            name="character_a",
            blocks={
                0: {"attn.q_proj.weight": LoRALayerFactors(A, B, scaling)},
                1: {"attn.q_proj.weight": LoRALayerFactors(A, B, scaling)},
                ...
            }
        )

    Block indices align with positions in the ``layers_attr`` ModuleList
    given to :class:`MergedLoRAStrategy`. In-block qualnames are the
    paths used by the underlying :class:`BlockStreamer` slots
    (e.g., ``"attn.q_proj.weight"``, not the full
    ``"transformer_blocks.0.attn.q_proj.weight"``).

    Conversion from PEFT state-dict format is left to a future utility;
    for v1 the caller is responsible for supplying parsed factors.
    """

    name: str
    blocks: dict[int, dict[str, LoRALayerFactors]]


# ---------------------------------------------------------------------------
# Internal pinned/GPU representations
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _PinnedFactors:
    """Host-pinned factors for one target layer of one LoRA."""

    A: torch.Tensor   # pinned, base dtype
    B: torch.Tensor   # pinned, base dtype
    scaling: float


@dataclass(slots=True)
class _GpuFactors:
    """GPU-resident factors for one target layer of one LoRA, valid
    only during the active window."""

    A: torch.Tensor
    B: torch.Tensor
    scaling: float


# pinned/GPU container types: lora_name → block_idx → in_block_qualname → factors
_PinnedMap = dict[str, dict[int, dict[str, _PinnedFactors]]]
_GpuMap = dict[str, dict[int, dict[str, _GpuFactors]]]


# ---------------------------------------------------------------------------
# MergedLoRAStrategy
# ---------------------------------------------------------------------------


class MergedLoRAStrategy:
    """Top-level :class:`ModelStrategy` that merges stacked LoRAs into
    block weights at prefetch time.

    See module docstring for architecture, constraints, and cost.

    Parameters
    ----------
    model:
        The full model. Must be bf16 or fp16 throughout — fp8 and
        quanto wrappers are unsupported (raises at construction).
    target_device:
        GPU device.
    loras:
        Sequence of all available :class:`LoRABundle` instances. The
        strategy clones, dtype-casts, and pins their factors. After
        construction, use :meth:`set_active` to choose which subset
        merges during the next active window.
    layers_attr:
        Dotted attribute path to the ``nn.ModuleList`` of streamable
        blocks (e.g., ``"transformer_blocks"``). Single list only in
        v1 — heterogeneous block lists (Flux-style) are a planned
        extension.
    blocks_to_swap:
        Number of blocks kept on CPU at any time, forwarded to the
        underlying :class:`BlockStreamer`.
    prefetch_count:
        Background prefetch depth, forwarded to :class:`BlockStreamer`.

    Lifecycle
    ---------
    1. Construct: pins LoRA factors, builds underlying streamer +
       non-block :class:`PinnedWeights` (``cache_bytes`` final).
    2. :meth:`set_active`: configure which LoRAs to merge. Must be
       called before :meth:`activate` (raises if active).
    3. :meth:`activate`: copy active factors to GPU (synchronous),
       activate underlying components. Forward thereafter sees merged
       weights.
    4. :meth:`deactivate`: deactivate underlying first (drains any
       in-flight prefetch — no merges referencing factors after this),
       then free GPU factor storage.
    """

    def __init__(
        self,
        model: nn.Module,
        target_device: torch.device,
        loras: Sequence[LoRABundle],
        *,
        layers_attr: str,
        blocks_to_swap: int,
        prefetch_count: int = 2,
    ) -> None:
        self._validate_base_dtype(model)
        base_dtype = next(iter(model.parameters())).dtype

        self._model: nn.Module | None = model
        self._device = target_device

        # Resolve blocks first so factor validation can range-check
        # block indices against the actual ModuleList length.
        blocks = list(_resolve_attr(model, layers_attr))
        if not blocks:
            raise ValueError(
                f"layers_attr={layers_attr!r} resolved to an empty ModuleList"
            )

        self._available: set[str] = {b.name for b in loras}
        if len(self._available) != len(loras):
            raise ValueError("LoRA names must be unique")

        # Pin factors up front; clone-and-cast to base dtype, validate.
        self._pinned_factors: _PinnedMap = self._pin_factors(
            loras, base_dtype, num_blocks=len(blocks),
        )
        self._active: tuple[str, ...] = ()
        self._gpu_factors: _GpuMap | None = None
        self._streamer = BlockStreamer(
            blocks=blocks,
            target_device=target_device,
            blocks_to_swap=blocks_to_swap,
            prefetch_count=prefetch_count,
            name=f"BlockStreamer[{layers_attr}]",
            post_load=self._apply_active_loras,
        )

        # Non-block content via PinnedWeights, scoped via skip_slots.
        skip: set[SlotOwnership] = set(self._streamer.slot_filter)
        self._pinned: PinnedWeights | None = None
        if _has_non_block_pinnable_content(model, skip):
            self._pinned = PinnedWeights(model, target_device, skip_slots=skip)

    # ---------------------------------------------------------------- API

    def register_lora_names(self) -> set[str]:
        """The names of all LoRAs registered at construction."""
        return set(self._available)

    def set_active(self, names: Sequence[str]) -> None:
        """Configure which LoRAs to merge during the next active window.

        ``names`` is an ordered sequence — the merge order is preserved
        because bf16 addition is non-associative and deterministic
        output requires a stable order. Pass ``()`` to merge no LoRAs
        (base-only forward).

        Raises if called while active. To switch active sets:
        :meth:`deactivate` → ``set_active(...)`` → :meth:`activate`.
        """
        if self._gpu_factors is not None:
            raise RuntimeError(
                "MergedLoRAStrategy.set_active() requires the strategy "
                "to be inactive. Call deactivate() first."
            )
        unknown = set(names) - self._available
        if unknown:
            raise ValueError(
                f"Unknown LoRA names: {sorted(unknown)}. "
                f"Registered: {sorted(self._available)}"
            )
        active = tuple(names)
        if len(set(active)) != len(active):
            raise ValueError(
                f"Duplicate LoRA names in active list: {active}. Each "
                f"LoRA can be active at most once per window — repeated "
                f"merges would compound the delta."
            )
        self._active = active

    @property
    def active(self) -> tuple[str, ...]:
        """The current active LoRA list (snapshot at last set_active)."""
        return self._active

    # --------------------------------------------------- ModelStrategy

    @property
    def model(self) -> nn.Module:
        assert self._model is not None
        return self._model

    @property
    def cache_bytes(self) -> int:
        total = self._streamer.cache_bytes
        if self._pinned is not None:
            total += self._pinned.cache_bytes
        # Pinned LoRA factors are part of the host budget — count all
        # registered factors regardless of active state, since they
        # stay pinned for the strategy's lifetime.
        for blocks in self._pinned_factors.values():
            for layers in blocks.values():
                for f in layers.values():
                    total += f.A.numel() * f.A.element_size()
                    total += f.B.numel() * f.B.element_size()
        return total

    def activate(self) -> None:
        # Order: copy factors to GPU FIRST so they exist before any
        # streamer prefetch can fire. Then activate components, tracking
        # which succeeded so a mid-activation failure can roll them back.
        self._gpu_factors = self._copy_factors_to_gpu(self._active)
        activated: list[Any] = []
        try:
            if self._pinned is not None:
                self._pinned.activate()
                activated.append(self._pinned)
            self._streamer.activate()
            activated.append(self._streamer)
        except BaseException:
            # Best-effort rollback in reverse activation order. Swallow
            # any rollback exceptions — the original activate failure is
            # what surfaces.
            for c in reversed(activated):
                try:
                    c.deactivate()
                except BaseException:
                    logger.exception(
                        "MergedLoRAStrategy: rollback deactivate raised "
                        "during activation cleanup; original error follows"
                    )
            self._gpu_factors = None
            raise
        # Sync to close the race between the streamer's resident-block
        # initial merges (addmm_ enqueued on the default stream during
        # streamer.activate(), kernels async even though copy_from is
        # blocking) and any forward the caller may run on a non-default
        # compute stream. The cost is microseconds; the alternative is a
        # subtle cross-stream visibility race.
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)

    def deactivate(self) -> None:
        # Order: deactivate streamers FIRST to drain prefetch (no
        # in-flight merge ops reference factors after this), then
        # non-block PinnedWeights, then free factor storage. Each step
        # is in its own try/finally so a raise in one stage doesn't
        # leak the rest — particularly the GPU factor storage, which
        # would otherwise survive a poisoned strategy reference.
        try:
            self._streamer.deactivate()
        finally:
            try:
                if self._pinned is not None:
                    self._pinned.deactivate()
            finally:
                self._gpu_factors = None

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

    # ------------------------------------------------------- Internals

    @staticmethod
    def _validate_base_dtype(model: nn.Module) -> None:
        """Raise if any parameter is not bf16/fp16. fp8 and quanto are
        unsupported because :func:`torch.Tensor.addmm_` requires an
        arithmetic-capable target dtype."""
        bad: list[tuple[str, torch.dtype]] = []
        for name, p in model.named_parameters():
            if p.dtype not in (torch.bfloat16, torch.float16):
                bad.append((name, p.dtype))
                if len(bad) > 3:
                    break
        if bad:
            raise ValueError(
                f"MergedLoRAStrategy requires bf16/fp16 base; found "
                f"{bad}. fp8 and quanto are unsupported in v1 because "
                f"the in-place merge (Tensor.addmm_) requires an "
                f"arithmetic-capable target dtype. For fp8 base, use "
                f"PEFT routed mode."
            )

    @staticmethod
    def _pin_factors(
        loras: Sequence[LoRABundle],
        base_dtype: torch.dtype,
        num_blocks: int,
    ) -> _PinnedMap:
        """Validate, clone-cast-and-pin user-supplied factors.

        Validation surfaces bad bundles at construction rather than as
        cryptic failures inside prefetch futures. Trainable factors are
        silently detached (we own a frozen snapshot) — that's
        intentional, not a bug, since the strategy is inference-only.
        """
        pinned: _PinnedMap = {}
        for lora in loras:
            per_block: dict[int, dict[str, _PinnedFactors]] = {}
            for block_idx, layer_factors in lora.blocks.items():
                if not 0 <= block_idx < num_blocks:
                    raise ValueError(
                        f"LoRA {lora.name!r}: block_idx={block_idx} out of "
                        f"range [0, {num_blocks})"
                    )
                per_layer: dict[str, _PinnedFactors] = {}
                for qual_name, f in layer_factors.items():
                    if not f.A.is_floating_point() or not f.B.is_floating_point():
                        raise ValueError(
                            f"LoRA {lora.name!r} block {block_idx} "
                            f"layer {qual_name!r}: factors must be floating-"
                            f"point; got A.dtype={f.A.dtype}, B.dtype={f.B.dtype}"
                        )
                    if f.A.dim() != 2 or f.B.dim() != 2:
                        raise ValueError(
                            f"LoRA {lora.name!r} block {block_idx} "
                            f"layer {qual_name!r}: A and B must be 2D; got "
                            f"A.shape={tuple(f.A.shape)}, B.shape={tuple(f.B.shape)}"
                        )
                    if f.A.shape[0] != f.B.shape[1]:
                        raise ValueError(
                            f"LoRA {lora.name!r} block {block_idx} "
                            f"layer {qual_name!r}: rank mismatch — A.shape[0]"
                            f"={f.A.shape[0]}, B.shape[1]={f.B.shape[1]} "
                            f"(expected A=(rank, in_dim), B=(out_dim, rank))"
                        )
                    # .cpu() handles the case of factors stored on GPU
                    # (e.g., loaded from a model that's on GPU). Without
                    # it, .pin_memory() would raise.
                    per_layer[qual_name] = _PinnedFactors(
                        A=f.A.detach().cpu().to(base_dtype).clone().pin_memory(),
                        B=f.B.detach().cpu().to(base_dtype).clone().pin_memory(),
                        scaling=float(f.scaling),
                    )
                per_block[block_idx] = per_layer
            pinned[lora.name] = per_block
        return pinned

    def _copy_factors_to_gpu(self, names: tuple[str, ...]) -> _GpuMap:
        """Copy the active subset of pinned factors to the target device.
        Synchronous — all factors are GPU-resident and ready before
        the streamer can start prefetching."""
        gpu: _GpuMap = {}
        for name in names:
            blocks_pinned = self._pinned_factors[name]
            per_block: dict[int, dict[str, _GpuFactors]] = {}
            for block_idx, layer_factors in blocks_pinned.items():
                per_layer: dict[str, _GpuFactors] = {}
                for qual_name, p in layer_factors.items():
                    per_layer[qual_name] = _GpuFactors(
                        A=p.A.to(self._device, non_blocking=True),
                        B=p.B.to(self._device, non_blocking=True),
                        scaling=p.scaling,
                    )
                per_block[block_idx] = per_layer
            gpu[name] = per_block
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        return gpu

    def _apply_active_loras(self, slot: Any, block_idx: int) -> None:
        """post_load callback invoked on the prefetch stream after each
        block's DMA. Merges every active LoRA's factors for this block
        into the slot's weights via in-place ``addmm_``.

        CUDA orders these ops after the preceding ``slot.copy_from``
        because both run on the current (prefetch) stream. The pool's
        readiness event records after this returns, so the main stream
        sees fully-merged weights."""
        if self._gpu_factors is None or not self._active:
            return
        for name in self._active:
            block_factors = self._gpu_factors[name].get(block_idx)
            if not block_factors:
                continue
            for qual_name, f in block_factors.items():
                # slot.get_param returns the GPU nn.Parameter wrapping
                # the slot's storage; .data is the underlying Tensor.
                # addmm_ computes target += alpha * (B @ A) in place,
                # without allocating a (d, d) intermediate.
                target = slot.get_param(qual_name).data
                target.addmm_(f.B, f.A, beta=1.0, alpha=f.scaling)
