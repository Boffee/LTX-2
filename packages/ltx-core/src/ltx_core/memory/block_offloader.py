"""Block-level CPU offloading for memory-efficient training and inference.

Keeps most frozen transformer block weights on CPU in persistent pinned
memory buffers. Uses a background thread and dedicated CUDA stream to
prefetch upcoming blocks, overlapping DMA with compute.

Quanto ``WeightQBytesTensor`` weights are decomposed into their inner
``_data`` (int8) and ``_scale`` (float) components for pinned-buffer DMA,
then reconstructed on GPU via ``WeightQBytesTensor.create()``. A naive
``param.data.clone()`` on a quanto tensor silently dequantizes it; the
explicit decomposition is required for quantized configurations.

Uses LRU eviction so the pre_hook works regardless of traversal direction
(forward 0→47 or backward recomputation 47→0 with gradient checkpointing).

Trainable parameters (e.g. LoRA adapters with ``requires_grad=True``) stay
on GPU permanently while the offloader is active so the offload doesn't
disrupt backward. For inference with frozen LoRA adapters, merge the LoRA
into the base weights first.

Implements :class:`~ltx_core.memory.strategy.ModelStrategy` via the
``prepare`` / ``activate`` / ``deactivate`` / ``close`` lifecycle so it
plugs into :class:`~ltx_core.memory.model_cache.ModelCache`.
``auto_setup=True`` (the default) runs ``prepare(); activate()``
immediately in the constructor so long-lived training callers don't
need to phase the lifecycle by hand.
"""

from __future__ import annotations

import functools
import logging
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from types import TracebackType
from typing import Any

import torch
from torch import nn

from ltx_core.memory.pinned_buffer import PinnedParamBuffer, storage_key
from ltx_core.memory.pinned_weights import PinnedWeights

logger = logging.getLogger(__name__)


def _resolve_attr(module: nn.Module, dotted_path: str) -> nn.ModuleList:
    obj: Any = module
    for part in dotted_path.split("."):
        obj = getattr(obj, part)
    if not isinstance(obj, nn.ModuleList):
        raise TypeError(f"Expected nn.ModuleList at '{dotted_path}', got {type(obj).__name__}")
    return obj


def _resolve_dotted(module: nn.Module, dotted_path: str) -> nn.Module:
    obj: Any = module
    for part in dotted_path.split("."):
        obj = getattr(obj, part)
    return obj


# ---------------------------------------------------------------------------
# Pre-allocated GPU buffer pool
# ---------------------------------------------------------------------------


class _GpuSlot:
    """Pre-allocated GPU storage for one block's frozen params.

    For each ``PinnedParamBuffer`` template, allocates matching GPU
    tensors (data + optional scale for quanto) once at construction
    and builds a stable ``nn.Parameter`` wrapping each. Subsequent
    ``copy_from`` calls write the pinned bytes into the pre-allocated
    GPU tensors in place — no malloc on the hot path, and the
    Parameter wrappers stay identity-stable across loads (required
    for PEFT compatibility and to avoid Python ref churn).
    """

    __slots__ = ("_gpu_data", "_gpu_scale", "_gpu_params")

    def __init__(self, template: list[PinnedParamBuffer], device: torch.device) -> None:
        self._gpu_data: dict[str, torch.Tensor] = {}
        self._gpu_scale: dict[str, torch.Tensor | None] = {}
        self._gpu_params: dict[str, nn.Parameter] = {}
        for buf in template:
            gpu_data, gpu_scale = buf.allocate_gpu_storage(device)
            self._gpu_data[buf.name] = gpu_data
            self._gpu_scale[buf.name] = gpu_scale
            self._gpu_params[buf.name] = buf.make_gpu_param(gpu_data, gpu_scale)

    def copy_from(self, bufs: list[PinnedParamBuffer], non_blocking: bool = False) -> None:
        for buf in bufs:
            buf.copy_to_gpu(
                self._gpu_data[buf.name],
                self._gpu_scale[buf.name],
                non_blocking=non_blocking,
            )

    def get_param(self, name: str) -> nn.Parameter:
        return self._gpu_params[name]


class _GpuSlotPool:
    """Pool of pre-allocated :class:`_GpuSlot` instances.

    All blocks share the same parameter structure (in homogeneous
    mode), so one template list of ``PinnedParamBuffer`` is used to
    construct ``num_slots`` identical GPU slots. Slots are acquired
    on load and released on eviction. Per-slot CUDA events enforce
    multi-stream safety: the prefetch stream waits for compute to
    finish reading a slot before overwriting it with new data. Uses
    ``event.query()`` to skip the GPU-side dependency when the
    compute event is already signaled (almost always the case for
    LRU victims last read many blocks ago).
    """

    def __init__(
        self,
        template: list[PinnedParamBuffer],
        num_slots: int,
        device: torch.device,
    ) -> None:
        self._slots = [_GpuSlot(template, device) for _ in range(num_slots)]
        self._free: list[int] = list(range(num_slots))
        self._events: list[torch.cuda.Event | None] = [None] * num_slots

    def acquire(self) -> int:
        return self._free.pop()

    def release(self, slot_id: int) -> None:
        self._free.append(slot_id)

    def slot(self, slot_id: int) -> _GpuSlot:
        return self._slots[slot_id]

    def set_compute_event(self, slot_id: int, event: torch.cuda.Event) -> None:
        self._events[slot_id] = event

    def wait_if_needed(self, slot_id: int, stream: torch.cuda.Stream | None) -> None:
        ev = self._events[slot_id]
        if ev is not None:
            if stream is not None and not ev.query():
                stream.wait_event(ev)
            self._events[slot_id] = None


# ---------------------------------------------------------------------------
# Block store: pinned CPU + (optional) GPU pool
# ---------------------------------------------------------------------------


class _BlockPinnedStore:
    """Per-block pinned CPU + (when activated) per-slot GPU storage.

    Lifecycle:

    - ``__init__`` pins CPU only. Each frozen block parameter slot is
      replaced with a :class:`PinnedParamBuffer`'s ``cpu_param``
      Parameter so the model can run on CPU without extra storage when
      offloaded. Buffers (registered via ``register_buffer``) get a
      pinned CPU clone in place. No GPU allocation occurs.

    - :meth:`activate_pool` allocates a :class:`_GpuSlotPool` for
      bounded GPU residency. Block layouts must be homogeneous for
      pool reuse; heterogeneous configurations fall back to per-load
      ``cudaMalloc`` (slower).

    - :meth:`deactivate_pool` releases the pool. Block params/buffers
      may still reference GPU storage from the last activation cycle
      until callers run :meth:`evict_block` for each block.

    Buffers (registered via ``register_buffer``) are kept simple:
    per-buffer CPU clone, per-buffer ``.to(device)`` on load. They
    are typically tiny (norm eps, RoPE position tables) so a slab
    abstraction would be over-engineering.
    """

    def __init__(self, layers: list[nn.Module] | nn.ModuleList) -> None:
        self._layers = list(layers)
        # Per block: list of PinnedParamBuffer (one per frozen param).
        self._param_bufs: list[list[PinnedParamBuffer]] = []
        # Per block: list of (qual_name, submod, local_name) for direct
        # _parameters assignment on load/evict. Resolved once at init so
        # the hot path doesn't re-walk named_modules.
        self._param_locs: list[list[tuple[str, nn.Module, str]]] = []
        # Per block: list of (buffer_obj, cpu_clone). Buffers are not slabbed;
        # per-buffer .to() on load.
        self._buf_pairs: list[list[tuple[torch.Tensor, torch.Tensor]]] = []

        for layer in self._layers:
            modules_map = dict(layer.named_modules())
            block_bufs: list[PinnedParamBuffer] = []
            block_locs: list[tuple[str, nn.Module, str]] = []
            for qual_name, p in layer.named_parameters():
                if p.requires_grad:
                    continue
                buf = PinnedParamBuffer(qual_name, p)
                parts = qual_name.rsplit(".", 1)
                if len(parts) == 2:
                    submod, local_name = modules_map[parts[0]], parts[1]
                else:
                    submod, local_name = layer, qual_name
                # Repoint the model's param SLOT at the pinned cpu_param
                # so the block can run on CPU without extra storage when
                # offloaded. _parameters[leaf] swap (rather than p.data
                # assignment) is required for correctness with quanto
                # WeightQBytesTensor; the .data path is silently a no-op
                # for the inner _data/_scale storages.
                submod._parameters[local_name] = buf.cpu_param
                block_bufs.append(buf)
                block_locs.append((qual_name, submod, local_name))
            self._param_bufs.append(block_bufs)
            self._param_locs.append(block_locs)

            # Capture (buffer_obj, cpu_clone) — clone owns pinned CPU
            # storage so (a) cache_bytes accounting is honest and (b) the
            # non_blocking=True .to(device) calls in load_block aren't
            # silently demoted to synchronous (PyTorch requires pinned
            # source for true async H2D copies).
            buf_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
            for _name, b in layer.named_buffers():
                cpu_clone = b.data.clone(memory_format=torch.contiguous_format).pin_memory()
                b.data = cpu_clone
                buf_pairs.append((b, cpu_clone))
            self._buf_pairs.append(buf_pairs)

        self._device: torch.device | None = None
        self._pool: _GpuSlotPool | None = None
        self._block_to_slot: dict[int, int] = {}
        # Captured on first activate_pool() so a mismatched re-activation
        # raises rather than silently reusing the wrong pool.
        self._pool_config: tuple[int, torch.device] | None = None

    @property
    def cache_bytes(self) -> int:
        """Total pinned host bytes held across all blocks."""
        total = 0
        for block in self._param_bufs:
            for buf in block:
                total += buf.pinned_data.numel() * buf.pinned_data.element_size()
                if buf.pinned_scale is not None:
                    total += buf.pinned_scale.numel() * buf.pinned_scale.element_size()
        for block_pairs in self._buf_pairs:
            for _, cpu_clone in block_pairs:
                total += cpu_clone.numel() * cpu_clone.element_size()
        return total

    def activate_pool(self, num_gpu_slots: int, device: torch.device) -> None:
        """Allocate a homogeneous-block GPU slot pool (no-op for
        heterogeneous layouts; per-load alloc is used instead).

        Idempotent when the same ``(num_gpu_slots, device)`` is
        requested back-to-back. Raises ``ValueError`` if a second call
        requests a different configuration — the existing pool's slot
        layout would no longer match.
        """
        if self._pool_config is not None:
            existing = self._pool_config
            if existing != (num_gpu_slots, device):
                raise ValueError(
                    f"_BlockPinnedStore pool already activated with "
                    f"{existing}; cannot re-activate with ({num_gpu_slots}, "
                    f"{device}). Call deactivate_pool() first."
                )
            return
        self._device = device
        self._pool_config = (num_gpu_slots, device)
        if num_gpu_slots > 0 and self._param_bufs and self._blocks_are_homogeneous():
            self._pool = _GpuSlotPool(self._param_bufs[0], num_gpu_slots, device)
        else:
            if num_gpu_slots > 0 and self._param_bufs:
                logger.info("Blocks have heterogeneous structure; using per-load GPU allocation")

    def deactivate_pool(self) -> None:
        """Drop the GPU slot pool reference. Caller is responsible for
        having evicted any block→slot mappings via :meth:`evict_block`
        so the slot Parameters are no longer referenced from the model."""
        self._pool = None
        self._block_to_slot.clear()
        self._pool_config = None

    def _blocks_are_homogeneous(self) -> bool:
        """All blocks must have identical layouts (same param names,
        shapes, dtypes, quanto specs including activation_qtype and
        outer size/stride) for pool slot reuse to be safe.

        Outer ``size``/``stride`` and ``act_qt`` are part of the quanto
        wrapper that ``make_gpu_param`` reconstructs from a template;
        if any block has a different outer layout the slot's cached
        Parameter would describe a tensor that no longer matches its
        own backing storage, so pool reuse must fall back to per-load
        allocation.
        """
        if len(self._param_bufs) <= 1:
            return True
        ref = self._param_bufs[0]

        def _key(b: PinnedParamBuffer) -> tuple:
            return (
                b.name, b.pinned_data.shape, b.pinned_data.dtype, b.is_quanto,
                b.pinned_scale.shape if b.pinned_scale is not None else None,
                b.pinned_scale.dtype if b.pinned_scale is not None else None,
                b.qtype, b.axis, b.act_qt, b.size, b.stride,
            )

        ref_keys = [_key(b) for b in ref]
        for block in self._param_bufs[1:]:
            if len(block) != len(ref_keys):
                return False
            for ref_tup, b in zip(ref_keys, block, strict=True):
                if _key(b) != ref_tup:
                    return False
        return True

    # -- load / evict ---------------------------------------------------------

    def load_block(
        self,
        idx: int,
        layer: nn.Module,
        device: torch.device,
        non_blocking: bool = False,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        if self._pool is not None:
            self._load_pooled(idx, non_blocking, stream)
        else:
            self._load_alloc(idx, device, non_blocking)

    def _load_pooled(self, idx: int, non_blocking: bool, stream: torch.cuda.Stream | None) -> None:
        slot_id = self._block_to_slot.get(idx)
        if slot_id is None:
            slot_id = self._pool.acquire()
            self._block_to_slot[idx] = slot_id
        self._pool.wait_if_needed(slot_id, stream)
        slot = self._pool.slot(slot_id)
        slot.copy_from(self._param_bufs[idx], non_blocking=non_blocking)

        for qual_name, submod, local_name in self._param_locs[idx]:
            submod._parameters[local_name] = slot.get_param(qual_name)
        for mod_buf, cpu_data in self._buf_pairs[idx]:
            mod_buf.data = cpu_data.to(self._device, non_blocking=non_blocking)

    def _load_alloc(self, idx: int, device: torch.device, non_blocking: bool) -> None:
        """Heterogeneous-blocks fallback: allocate GPU storage per call.
        Slower (cudaMalloc/free per load) but works when blocks aren't
        layout-compatible."""
        for buf, (_qn, submod, local_name) in zip(
            self._param_bufs[idx], self._param_locs[idx], strict=True,
        ):
            submod._parameters[local_name] = buf.load_to_gpu(device, non_blocking=non_blocking)
        for mod_buf, cpu_data in self._buf_pairs[idx]:
            mod_buf.data = cpu_data.to(device, non_blocking=non_blocking)

    def evict_block_fast(self, idx: int) -> None:
        """Release GPU slot without restoring CPU params (hot training path).

        The pre_hook will reload params from pinned buffers before any access,
        so restoring CPU pointers is unnecessary during training.
        """
        if self._pool is not None:
            slot_id = self._block_to_slot.pop(idx, None)
            if slot_id is not None:
                self._pool.release(slot_id)

    def evict_block(self, idx: int, _layer: nn.Module) -> None:
        """Release GPU slot AND restore CPU params (teardown / validation path)."""
        for buf, (_qn, submod, local_name) in zip(
            self._param_bufs[idx], self._param_locs[idx], strict=True,
        ):
            submod._parameters[local_name] = buf.cpu_param
        for mod_buf, cpu_data in self._buf_pairs[idx]:
            mod_buf.data = cpu_data
        if self._pool is not None:
            slot_id = self._block_to_slot.pop(idx, None)
            if slot_id is not None:
                self._pool.release(slot_id)

    def mark_compute_done(self, idx: int, event: torch.cuda.Event) -> None:
        """Record that compute finished reading from *idx*'s GPU slot."""
        if self._pool is not None:
            slot_id = self._block_to_slot.get(idx)
            if slot_id is not None:
                self._pool.set_compute_event(slot_id, event)


# ---------------------------------------------------------------------------
# LRU tracker
# ---------------------------------------------------------------------------


class _BlockTracker:
    def __init__(self, num_layers: int) -> None:
        self.num_layers = num_layers
        self._on_gpu: set[int] = set()
        self._lru: OrderedDict[int, None] = OrderedDict()
        self.peak_gpu_blocks = 0

    def is_on_gpu(self, idx: int) -> bool:
        return idx in self._on_gpu

    def touch(self, idx: int) -> None:
        if idx in self._lru:
            self._lru.move_to_end(idx)

    def mark_on_gpu(self, idx: int) -> None:
        self._on_gpu.add(idx)
        self._lru.pop(idx, None)
        self._lru[idx] = None

    def mark_on_cpu(self, idx: int) -> None:
        self._on_gpu.discard(idx)
        self._lru.pop(idx, None)

    def pick_victim(self, protected: set[int]) -> int:
        for idx in self._lru:
            if idx not in protected:
                return idx
        raise RuntimeError("no evictable block")

    def clear(self) -> None:
        self._on_gpu.clear()
        self._lru.clear()


# ---------------------------------------------------------------------------
# LoRA / trainable param restore after module-level moves
# ---------------------------------------------------------------------------


def _move_trainable_to_device(layer: nn.Module, device: torch.device) -> None:
    for p in layer.parameters():
        if p.requires_grad:
            if p.data.device != device:
                p.data = p.data.to(device)
            if p.grad is not None and p.grad.device != device:
                p.grad = p.grad.to(device)


# ---------------------------------------------------------------------------
# Main offloader
# ---------------------------------------------------------------------------


class BlockOffloader:
    """Streams frozen transformer blocks between CPU and GPU.

    Frozen weights are kept in persistent pinned CPU buffers. Prefetch uses
    a background thread and dedicated CUDA stream to overlap DMA with compute.
    Quanto ``WeightQBytesTensor`` is decomposed into ``_data``/``_scale`` for
    DMA and reconstructed on GPU.

    A pre-allocated GPU buffer pool avoids CUDA malloc/free overhead during
    training/inference and provides explicit multi-stream safety via per-slot
    events. Trainable parameters (e.g. LoRA adapters added via PEFT) stay on
    GPU permanently while active, so backward through them is unaffected by
    the offload.

    Implements :class:`~ltx_core.memory.strategy.ModelStrategy` via the
    ``prepare`` / ``activate`` / ``deactivate`` / ``close`` lifecycle.
    ``close()`` is destructive — it moves the model to ``meta`` and
    releases pinned storage, breaking the offloader → forward-hook →
    block reference cycle. ``shard_orchestrator.py`` calls it between
    shards for exactly that reason.

    Caveats
    -------
    - **Cross-region and intra-block tied weights are rejected at
      :meth:`prepare`.** Frozen storage shared across blocks, or
      between a block and a non-block sibling, can't be preserved by
      slot-local streaming. Storage shared across two slots within the
      same block can't be preserved by ``_BlockPinnedStore``'s
      duplicate-removed iteration either. Non-block-internal ties
      (the standard ``tie_weights()`` embed↔head pattern) are handled
      correctly via the composed :class:`PinnedWeights`'s storage-key
      dedup. Models with unsupported tying must untie or use
      whole-model :class:`PinnedWeights` instead.
    - **Buffer mutations during forward are discarded on
      :meth:`deactivate`.** Both block-internal buffers and non-block
      buffers (via composed :class:`PinnedWeights`) get a pinned CPU
      copy that overwrites any GPU-side mutations on the round-trip.
      Suitable for inference of stateless modules and for buffers
      that only hold derived constants (RoPE tables, sinusoidal
      embeddings); not suitable for models that need persistent
      buffer state across calls (BatchNorm running stats updated in
      training mode, RNN/SSM hidden state, KV cache, etc.).

    Parameters
    ----------
    model:
        The model containing the block list(s) (may be PEFT-wrapped).
    target_device:
        The GPU device to use for compute.
    blocks_to_swap:
        Number of blocks to keep offloaded on CPU. Must be < total blocks.
    layers_attr:
        Dotted attribute path(s) to ``nn.ModuleList`` block lists in the model.
        A single string for models with one block list (e.g. ``"transformer_blocks"``),
        or a list for models with multiple (e.g. ``["transformer_blocks",
        "single_transformer_blocks"]``). For PEFT-wrapped models, include the
        prefix (e.g. ``"base_model.model.transformer_blocks"``).
    prefetch_count:
        How many blocks ahead to prefetch on a background thread.
    auto_setup:
        When ``True`` (default), runs ``prepare(); activate()`` immediately
        in the constructor — preserving the pre-lifecycle-split behavior
        for trainer/shard-orchestrator/etc. Pass ``False`` when you want
        to control the lifecycle yourself (e.g., handing off to
        :class:`~ltx_core.memory.model_cache.ModelCache`); in that case
        the factory must call ``prepare()`` before returning the
        offloader so the cache reads the correct ``cache_bytes``.
    """

    def __init__(
        self,
        model: nn.Module,
        target_device: torch.device,
        blocks_to_swap: int,
        layers_attr: str | list[str],
        prefetch_count: int = 3,
        *,
        auto_setup: bool = True,
    ) -> None:
        self._model: nn.Module | None = model
        self._target_device = target_device
        self._blocks_to_swap = blocks_to_swap
        self._prefetch_count = prefetch_count
        self._layers_attrs = [layers_attr] if isinstance(layers_attr, str) else list(layers_attr)

        # Lifecycle flags
        self._prepared = False
        self._active = False
        self._closed = False

        # Resources owned at "prepared" lifetime
        self._layers: list[nn.Module] | None = None
        self._block_leaf_names: set[str] | None = None
        self._store: _BlockPinnedStore | None = None
        # Non-block frozen params/buffers (everything outside the block
        # list — sibling modules AND direct frozen state on parent
        # modules like LTX's velocity_model.scale_shift_table) are
        # managed via composed PinnedWeights with a skip filter for the
        # block params. None when there's nothing non-block to pin
        # (pure block-only model).
        self._non_block_pinned: PinnedWeights | None = None

        # Resources owned at "active" lifetime
        self._tracker: _BlockTracker | None = None
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._executor: ThreadPoolExecutor | None = None
        self._stream: torch.cuda.Stream | None = None
        self._pending: dict[int, Future[None]] = {}
        self._prefetch_events: dict[int, torch.cuda.Event] = {}
        self._last_idx: int = -1

        if auto_setup:
            self.prepare()
            self.activate()

    # ------------------------------------------------------------------
    # ModelStrategy lifecycle
    # ------------------------------------------------------------------

    @property
    def cache_bytes(self) -> int:
        """Total pinned host bytes held: block store + non-block
        pinned weights. ``0`` before :meth:`prepare` and after
        :meth:`close`."""
        total = 0
        if self._store is not None:
            total += self._store.cache_bytes
        if self._non_block_pinned is not None:
            total += self._non_block_pinned.cache_bytes
        return total

    @property
    def closed(self) -> bool:
        return self._closed

    def prepare(self) -> None:
        """Resolve layers, pin frozen weights to CPU. Truly inactive:
        no GPU allocation, no hooks, no executor.

        Frozen block weights go into a :class:`_BlockPinnedStore` (per-block
        pinned CPU + on-demand slot pool when active). Frozen non-block
        siblings (patchifier, output projection, norms, etc.) are pinned
        via a composed :class:`PinnedWeights` so they leave GPU on
        :meth:`deactivate`. Trainable params (e.g. LoRA) stay on CPU
        until :meth:`activate` moves them to GPU.

        Idempotent if already prepared. Raises if closed.

        For :class:`~ltx_core.memory.model_cache.ModelCache` integration
        the factory must call ``prepare()`` before returning the handle
        so the cache can read the correct ``cache_bytes``::

            def factory():
                off = BlockOffloader(..., auto_setup=False)
                off.prepare()
                return off
        """
        if self._closed:
            raise RuntimeError("BlockOffloader is closed and cannot be re-prepared.")
        if self._prepared:
            return
        assert self._model is not None

        self._layers, self._block_leaf_names = self._resolve_all_layers()
        num_layers = len(self._layers)
        if self._blocks_to_swap >= num_layers:
            raise ValueError(
                f"blocks_to_swap ({self._blocks_to_swap}) must be < num_layers ({num_layers})"
            )

        # Detect cross-region tied frozen weights BEFORE any pinning runs
        # (pinning clones storage, which would silently break the sharing
        # invariant). Cross-block ties can't be preserved by slot-local
        # streaming, and block↔non-block ties can't be preserved across
        # the two pinning regimes either.
        self._detect_cross_region_tied_weights()

        # Move the entire model to CPU. Block params come along (will be
        # pinned by _BlockPinnedStore below); non-block frozen state is
        # also there for PinnedWeights to pin; trainable params come too
        # (PinnedWeights skips them; activate moves them to GPU).
        self._model.to("cpu")

        # Compute the skip-id sets so the composed PinnedWeights walks
        # the OUTER model but ignores parameters/buffers that
        # _BlockPinnedStore will own. Capture ids BEFORE
        # _BlockPinnedStore mutates the slots.
        block_param_ids = {
            id(p) for layer in self._layers for p in layer.parameters()
        }
        block_buffer_ids = {
            id(b) for layer in self._layers for b in layer.buffers()
        }

        # Compose PinnedWeights for everything outside the block list:
        # non-block sibling modules AND direct frozen params/buffers on
        # parent modules (e.g., LTX's velocity_model.scale_shift_table).
        # Construct only if there's actually non-block content to manage —
        # PinnedWeights raises on empty input.
        if self._has_non_block_pinnable_content(block_param_ids, block_buffer_ids):
            self._non_block_pinned = PinnedWeights(
                self._model,
                self._target_device,
                skip_param_ids=block_param_ids,
                skip_buffer_ids=block_buffer_ids,
            )
        else:
            self._non_block_pinned = None

        # Pin block weights (CPU only — pool allocated in activate()).
        self._store = _BlockPinnedStore(self._layers)

        self._prepared = True

    def _has_non_block_pinnable_content(
        self, skip_param_ids: set[int], skip_buffer_ids: set[int]
    ) -> bool:
        """True if the outer model has any frozen param or buffer that
        isn't in the block-list skip set. Used to decide whether to
        construct the composed PinnedWeights."""
        assert self._model is not None
        for p in self._model.parameters():
            if not p.requires_grad and id(p) not in skip_param_ids:
                return True
        for b in self._model.buffers():
            if id(b) not in skip_buffer_ids:
                return True
        return False

    # ------------------------------------------------------------------
    # Cross-region tied-weight detection
    # ------------------------------------------------------------------

    def _detect_cross_region_tied_weights(self) -> None:
        """Group all params across regions (each block + non_block) by
        storage identity; raise on any unsupported tying configuration.

        Three categories are unsupported:

        - **Cross-region ties** (block↔block, block↔non_block): the two
          pinning regimes (per-block ``_BlockPinnedStore`` and
          whole-non-block composed ``PinnedWeights``) can't coordinate
          to share storage.
        - **Mixed frozen/trainable ties** across any region boundary:
          the frozen side gets pinned and slot-swapped while the
          trainable side is moved separately on activate, breaking the
          sharing invariant silently.
        - **Intra-block ties** (two slots in the same block sharing
          storage): ``_BlockPinnedStore`` uses ``named_parameters()``
          with default duplicate removal and only swaps one alias slot,
          leaving the other pointing at non-pinned data. Reject rather
          than silently break.

        Non-block-internal ties go to :class:`PinnedWeights` which
        handles them via storage-key dedup.
        """
        assert self._layers is not None
        assert self._model is not None

        # Map each block param's id to its block index, so we can
        # classify any param in the model into its region in O(1).
        param_id_to_region: dict[int, str] = {}
        for block_idx, layer in enumerate(self._layers):
            for p in layer.parameters():
                param_id_to_region.setdefault(id(p), f"block:{block_idx}")

        # storage_key -> list of (region_label, qualified_name, requires_grad,
        #                         id(parent), leaf)
        groups: dict[tuple, list[tuple[str, str, bool, int, str]]] = {}
        modules_map = dict(self._model.named_modules(remove_duplicate=False))
        for qual_name, p in self._model.named_parameters(remove_duplicate=False):
            if p.numel() == 0:
                continue
            parts = qual_name.rsplit(".", 1)
            if len(parts) == 2:
                parent_obj, leaf = modules_map[parts[0]], parts[1]
            else:
                parent_obj, leaf = self._model, qual_name
            region = param_id_to_region.get(id(p), "non_block")
            skey = storage_key(p.data)
            groups.setdefault(skey, []).append(
                (region, qual_name, p.requires_grad, id(parent_obj), leaf)
            )

        for members in groups.values():
            regions = {region for region, _, _, _, _ in members}
            names = sorted(name for _, name, _, _, _ in members)
            if len(regions) > 1:
                raise ValueError(
                    f"BlockOffloader does not support tied parameters across "
                    f"streamed regions: storage shared by {names}. Slot-local "
                    "block streaming cannot preserve cross-region tying "
                    "(neither frozen↔frozen nor frozen↔trainable). Use "
                    "whole-model PinnedWeights, disable block streaming, or "
                    "untie the parameters."
                )
            # Intra-block ties: a single block region with multiple
            # distinct (parent, leaf) slot locations means the same
            # storage is referenced at multiple places within the block
            # and _BlockPinnedStore would only swap one of them.
            sole_region = next(iter(regions))
            if sole_region.startswith("block:"):
                slot_locs = {(pid, leaf) for _, _, _, pid, leaf in members}
                if len(slot_locs) > 1:
                    raise ValueError(
                        f"BlockOffloader does not support intra-block tied "
                        f"parameters: storage shared by {names} within "
                        f"{sole_region}. _BlockPinnedStore cannot preserve "
                        "the tying invariant — one alias would stay pointing "
                        "at non-pinned data. Untie the parameters or use "
                        "whole-model PinnedWeights instead."
                    )

        # Same scan for buffers. Block buffers are managed by
        # _BlockPinnedStore (clones + pin per layer); non-block buffers
        # by composed PinnedWeights. Two failure modes:
        #   - Cross-region: block buffer and non-block buffer share
        #     storage → two pinning regimes can't coordinate, alias
        #     breaks silently.
        #   - Intra-block: two distinct buffers within the same block
        #     (or across blocks) share storage → _BlockPinnedStore
        #     clones each independently, alias breaks silently.
        # Walk per block to map each buffer instance to ALL its block
        # regions (not just the first), so a buffer object shared
        # across blocks classifies as multi-region and gets rejected.
        buffer_id_to_regions: dict[int, set[str]] = {}
        for block_idx, layer in enumerate(self._layers):
            for b in layer.buffers():
                buffer_id_to_regions.setdefault(id(b), set()).add(f"block:{block_idx}")

        # storage_key -> list of (region, qualified_name, id(buffer))
        buf_groups: dict[tuple, list[tuple[str, str, int]]] = {}
        for qual_name, b in self._model.named_buffers(remove_duplicate=False):
            if b.numel() == 0:
                continue
            block_regions = buffer_id_to_regions.get(id(b))
            if block_regions:
                # Buffer object reachable inside the block list. If it
                # shows up in multiple blocks (same instance reused),
                # record once per region — the multi-region check below
                # will catch it.
                for region in block_regions:
                    buf_groups.setdefault(storage_key(b), []).append(
                        (region, qual_name, id(b))
                    )
            else:
                buf_groups.setdefault(storage_key(b), []).append(
                    ("non_block", qual_name, id(b))
                )

        for members in buf_groups.values():
            regions = {region for region, _, _ in members}
            names = sorted({name for _, name, _ in members})
            if len(regions) > 1:
                raise ValueError(
                    f"BlockOffloader does not support tied buffers across "
                    f"streamed regions: storage shared by {names}. The two "
                    "pinning regimes (per-block clone vs composed "
                    "PinnedWeights) can't coordinate to preserve the "
                    "alias. Untie the buffers or use whole-model "
                    "PinnedWeights instead."
                )
            sole_region = next(iter(regions))
            if sole_region.startswith("block:"):
                # Intra-block: distinct buffer objects sharing storage
                # within the same block region. _BlockPinnedStore would
                # clone each independently and break the alias.
                distinct_ids = {bid for _, _, bid in members}
                if len(distinct_ids) > 1:
                    raise ValueError(
                        f"BlockOffloader does not support intra-block tied "
                        f"buffers: storage shared by {names} within "
                        f"{sole_region}. _BlockPinnedStore clones each "
                        "buffer independently — the alias would break. "
                        "Untie the buffers or use whole-model PinnedWeights."
                    )

    def activate(self) -> nn.Module:
        """Allocate per-activation resources and return the model.

        Auto-prepares if constructed. Allocates GPU slot pool, CUDA
        stream/events, prefetch executor, registers forward hooks, and
        pre-loads the resident block window. Not re-entrant: nested
        calls raise ``RuntimeError``. On failure, rolls back to
        ``prepared`` so the caller can retry or close cleanly.
        """
        if self._closed:
            raise RuntimeError("BlockOffloader is closed and cannot be activated.")
        if self._active:
            raise RuntimeError(
                "BlockOffloader.activate() is not re-entrant. Call deactivate() "
                "before activating again."
            )
        if not self._prepared:
            self.prepare()
        assert self._model is not None
        assert self._layers is not None
        assert self._store is not None

        num_layers = len(self._layers)
        num_resident = num_layers - self._blocks_to_swap
        num_gpu_slots = num_resident + self._prefetch_count

        self._active = True
        try:
            # Bring frozen non-block weights to GPU first (whole-model
            # bulk DMA via the composed PinnedWeights). After this the
            # patchifier, output projection, norms, etc. are all GPU-
            # resident; only the block list streams.
            if self._non_block_pinned is not None:
                self._non_block_pinned.activate()

            # Move trainable params (LoRA, etc.) to GPU. Walks the whole
            # model, so both block-internal and non-block trainable
            # params are covered.
            _move_trainable_to_device(self._model, self._target_device)

            self._tracker = _BlockTracker(num_layers)
            self._executor = ThreadPoolExecutor(max_workers=1)
            self._stream = torch.cuda.Stream(device=self._target_device, priority=-1)
            self._pending = {}
            self._prefetch_events = {i: torch.cuda.Event() for i in range(num_layers)}
            self._last_idx = -1

            self._store.activate_pool(num_gpu_slots, self._target_device)

            # Pre-load initial resident window (synchronous).
            for idx in range(min(num_resident, num_layers)):
                self._store.load_block(idx, self._layers[idx], self._target_device)
                self._tracker.mark_on_gpu(idx)

            self._register_hooks(num_resident)

            # Seed peak to reflect the pre-loaded resident window. Since cb83965,
            # peak_gpu_blocks is only updated inside the forward-pre hook; without
            # this seed, peak stays at 0 between activate() and the first forward,
            # which matters for deactivate/activate cycles where a callback may
            # read peak before any new forward runs.
            self.reset_peak()

            logger.info(
                f"Block offloading active: {self._blocks_to_swap}/{num_layers} blocks on CPU, "
                f"{num_resident} resident on GPU, prefetch={self._prefetch_count}, "
                f"gpu_pool_slots={num_gpu_slots}"
            )
            return self._model
        except BaseException:
            # Best-effort rollback to prepared state. If rollback
            # itself fails, leave _active=True so a later close() will
            # re-attempt cleanup of the partially-installed resources
            # (hooks, pool slots, executor, stream). The original
            # activation exception is always re-raised.
            try:
                self._teardown_active_resources(suppress_prefetch_errors=True)
            except BaseException as rollback_exc:
                logger.error(
                    "BlockOffloader.activate() rollback failed; offloader stays "
                    "in active state with leaked resources — call close() to "
                    "reclaim. Original error will still propagate. rollback=%r",
                    rollback_exc,
                    exc_info=True,
                )
            else:
                self._active = False
            raise

    def deactivate(self) -> None:
        """Release GPU pool, hooks, executor, stream. Pinned CPU stays.

        Idempotent. Drains any pending prefetch futures so
        deactivate-then-activate cycles don't see stale CUDA work. If a
        pending prefetch future raises (CUDA OOM, mid-DMA error,
        etc.), best-effort cleanup still runs but the first such
        exception is re-raised after — :class:`ModelCache` treats this
        as a poisoned strategy.
        """
        if not self._active:
            return
        prefetch_exc = self._teardown_active_resources(suppress_prefetch_errors=False)
        # Cleanup itself succeeded (or we wouldn't reach this line);
        # mark inactive before surfacing any captured prefetch error so
        # ``close()`` doesn't try to re-deactivate.
        self._active = False
        if prefetch_exc is not None:
            raise prefetch_exc

    def close(self) -> None:
        """Destructive: deactivate, move model to ``meta``, release pinned.

        Idempotent. After ``close()``, the wrapped model is unusable —
        callers must rebuild a fresh BlockOffloader to use it again.
        ``shard_orchestrator.py`` calls ``close()`` between shards to
        break the offloader → forward-hook → block → offloader
        reference cycle that holds the previous shard's GPU + pinned
        memory across the rebind.
        """
        if self._closed:
            return
        if self._active:
            # Suppress prefetch errors during close — close() is the
            # destructive endpoint; surfacing a prefetch failure would
            # block the host-allocator flush. Logged in cleanup.
            self._teardown_active_resources(suppress_prefetch_errors=True)
            self._active = False
        # Close the inner non-block PinnedWeights first. Its close()
        # moves the wrapper (and therefore the referenced non-block
        # children) to meta and drops its pinned refs. The subsequent
        # outer .to("meta") then walks blocks, with non-block already
        # meta-ified.
        #
        # If inner close() raises, leave _non_block_pinned in place so a
        # retry can complete it — clearing it would orphan the only
        # handle to its pinned storage. Same retryability pattern as
        # PinnedWeights.close().
        if self._non_block_pinned is not None:
            self._non_block_pinned.close()
            self._non_block_pinned = None
        if self._model is not None:
            # Move to meta to break model-side references to pinned storage.
            # Let this raise without clearing state — preserves the ability
            # to retry close() or hand-clean (matches PinnedWeights.close()
            # semantics).
            self._model.to("meta")
        self._model = None
        self._layers = None
        self._block_leaf_names = None
        self._store = None
        self._prepared = False
        self._closed = True

    def __enter__(self) -> nn.Module:
        return self.activate()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.deactivate()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_all_layers(self) -> tuple[list[nn.Module], set[str]]:
        flat: list[nn.Module] = []
        leaf_names: set[str] = set()
        assert self._model is not None
        for attr_path in self._layers_attrs:
            module_list = _resolve_attr(self._model, attr_path)
            flat.extend(module_list)
            leaf_names.add(attr_path.split(".")[-1])
        return flat, leaf_names

    def _teardown_active_resources(
        self,
        *,
        suppress_prefetch_errors: bool,
    ) -> BaseException | None:
        """Reverses everything ``activate()`` allocates. Used by
        ``deactivate()``, ``activate()``'s rollback path, and ``close()``.

        Returns the first captured prefetch exception (or None if there
        was none / ``suppress_prefetch_errors=True``). Cleanup-itself
        failures (CUDA sync errors, hook removal failures, etc.) raise
        immediately and skip the return — leaving the caller's
        ``_active`` flag at True so close() can retry.
        """
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

        # Wait for pending prefetches and mark them as on-GPU so the
        # eviction-restore loop below saves their CPU state. Capture
        # the first exception (if any) without short-circuiting the
        # rest of the cleanup.
        first_prefetch_exc: BaseException | None = None
        if self._tracker is not None:
            for idx, future in self._pending.items():
                try:
                    future.result()
                except BaseException as exc:
                    if first_prefetch_exc is None:
                        first_prefetch_exc = exc
                    if suppress_prefetch_errors:
                        logger.warning("pending prefetch raised during cleanup: %r", exc)
                self._tracker.mark_on_gpu(idx)
        self._pending.clear()
        self._prefetch_events.clear()

        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

        if self._stream is not None:
            self._stream.synchronize()
            self._stream = None

        if self._tracker is not None and self._store is not None and self._layers is not None:
            torch.cuda.synchronize(device=self._target_device)
            # Restore CPU params for ALL blocks. Fast-evicted blocks still
            # have module params pointing at (now-stale) GPU slot data.
            for idx, layer in enumerate(self._layers):
                self._store.evict_block(idx, layer)
            self._tracker.clear()

        if self._store is not None:
            self._store.deactivate_pool()

        # Return frozen non-block weights to pinned CPU and move
        # trainable params off the GPU so the deactivated state truly
        # has no GPU footprint. Order matters: PinnedWeights.deactivate
        # restores Parameter slots before we strip trainable .data refs.
        if self._non_block_pinned is not None:
            self._non_block_pinned.deactivate()
        if self._model is not None:
            _move_trainable_to_device(self._model, torch.device("cpu"))

        self._tracker = None
        self._last_idx = -1

        # Return the captured prefetch exception (if any) so the caller
        # can decide whether to surface it (deactivate path) or
        # suppress it (activate-rollback / close path). Cleanup-itself
        # failures already raised mid-method.
        if suppress_prefetch_errors:
            return None
        return first_prefetch_exc

    # ------------------------------------------------------------------
    # Block transfer
    # ------------------------------------------------------------------

    def _evict_one(self, protected: set[int], compute_event: torch.cuda.Event | None = None) -> None:
        victim = self._tracker.pick_victim(protected=protected)
        if compute_event is not None:
            self._store.mark_compute_done(victim, compute_event)
        self._store.evict_block_fast(victim)
        self._tracker.mark_on_cpu(victim)

    def _do_prefetch(self, idx: int) -> None:
        with torch.cuda.stream(self._stream):
            self._store.load_block(idx, self._layers[idx], self._target_device, non_blocking=True, stream=self._stream)
            self._prefetch_events[idx].record(self._stream)

    def _submit_prefetch(self, idx: int, max_on_gpu: int) -> None:
        if idx < 0 or idx >= len(self._layers):
            return
        if self._tracker.is_on_gpu(idx) or idx in self._pending:
            return
        if len(self._tracker._on_gpu) + len(self._pending) >= max_on_gpu:
            return
        self._pending[idx] = self._executor.submit(self._do_prefetch, idx)

    def _ensure_on_gpu(self, idx: int) -> None:
        future = self._pending.pop(idx, None)
        if future is not None:
            future.result()
            ev = self._prefetch_events[idx]
            if not ev.query():
                torch.cuda.current_stream(self._target_device).wait_event(ev)
            self._tracker.mark_on_gpu(idx)
            return

        if not self._tracker.is_on_gpu(idx):
            self._store.load_block(idx, self._layers[idx], self._target_device)
            self._tracker.mark_on_gpu(idx)

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def _register_hooks(self, num_resident: int) -> None:
        idx_map: dict[int, int] = {id(layer): idx for idx, layer in enumerate(self._layers)}
        max_on_gpu = num_resident + self._prefetch_count
        pending_keys = self._pending  # local ref avoids dict attr lookup

        def _pre_hook(_module: nn.Module, _args: Any, *, idx: int) -> None:  # noqa: ANN401
            if self._tracker.is_on_gpu(idx):
                self._tracker.touch(idx)
            else:
                # Record compute-done event only when eviction is needed.
                # All kernels from previous blocks have been submitted, so
                # this event covers any block that ran before this hook.
                compute_event = torch.cuda.current_stream(self._target_device).record_event()
                while len(self._tracker._on_gpu) >= num_resident:
                    protected = {idx} | set(pending_keys.keys())
                    self._evict_one(protected, compute_event)
                self._ensure_on_gpu(idx)

            direction = 1 if idx >= self._last_idx else -1
            self._last_idx = idx
            for offset in range(1, self._prefetch_count + 1):
                self._submit_prefetch(idx + direction * offset, max_on_gpu)

            total = len(self._tracker._on_gpu) + len(pending_keys)
            self._tracker.peak_gpu_blocks = max(self._tracker.peak_gpu_blocks, total)

        for layer in self._layers:
            idx = idx_map[id(layer)]
            h = layer.register_forward_pre_hook(functools.partial(_pre_hook, idx=idx))
            self._hooks.append(h)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def peak_gpu_blocks(self) -> int:
        """Peak blocks on GPU (tracked + pending prefetches) since last reset."""
        return self._tracker.peak_gpu_blocks if self._tracker is not None else 0

    def reset_peak(self) -> None:
        if self._tracker is not None:
            self._tracker.peak_gpu_blocks = len(self._tracker._on_gpu) + len(self._pending)
