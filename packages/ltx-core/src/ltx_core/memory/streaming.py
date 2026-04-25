"""Block-level CPU offloading for memory-efficient training and inference.

Keeps most frozen transformer block weights on CPU in persistent pinned
memory buffers. Uses a background thread and dedicated CUDA stream to
prefetch upcoming blocks, overlapping DMA with compute.

Quanto ``WeightQBytesTensor`` weights are decomposed into their inner
``_data`` (int8) and ``_scale`` (float) components for pinned-buffer DMA,
then reconstructed on GPU via ``WeightQBytesTensor.create()``.

Uses LRU eviction so the pre_hook works regardless of traversal direction
(forward 0→47 or backward recomputation 47→0 with gradient checkpointing).

Trainable parameters (e.g. LoRA adapters with ``requires_grad=True``) stay
on GPU permanently so the offload doesn't disrupt backward. For inference
with frozen LoRA adapters, merge the LoRA into the base weights first.
"""

from __future__ import annotations

import functools
import logging
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import torch
from torch import nn

from ltx_core.memory._buffers import PinnedParamBuffer, _QUANTO_AVAILABLE

logger = logging.getLogger(__name__)

if _QUANTO_AVAILABLE:
    from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor


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


# Local alias keeps existing call sites in this module unchanged. The class
# itself lives in ``_buffers.py`` so ``pinned.py`` can share it without
# reaching into another module's private namespace.
_PinnedParamBuffer = PinnedParamBuffer


# ---------------------------------------------------------------------------
# Pre-allocated GPU buffer pool
# ---------------------------------------------------------------------------


class _PackedSlab:
    """Contiguous pinned CPU + GPU buffers holding many tensors of the same dtype.

    On each load only **one** ``copy_()`` is needed per slab instead of one per
    tensor.  Individual tensors are accessed as views into the flat buffer.
    """

    __slots__ = ("cpu_flat", "gpu_flat", "meta")

    def __init__(self, tensors: list[torch.Tensor], device: torch.device) -> None:
        self.meta: list[tuple[int, int, torch.Size, tuple[int, ...]]] = []
        total = 0
        for t in tensors:
            if not t.is_contiguous():
                raise ValueError("_PackedSlab only supports contiguous tensors")
            self.meta.append((total, t.numel(), t.size(), t.stride()))
            total += t.numel()
        self.cpu_flat = torch.empty(total, dtype=tensors[0].dtype, pin_memory=True)
        self.gpu_flat = torch.empty(total, dtype=tensors[0].dtype, device=device)
        for t, (off, n, _sz, _st) in zip(tensors, self.meta, strict=True):
            self.cpu_flat[off : off + n].copy_(t.reshape(-1))

    def gpu_view(self, i: int) -> torch.Tensor:
        off, n, sz, _st = self.meta[i]
        return self.gpu_flat.narrow(0, off, n).view(sz)


class _GpuSlot:
    """Pre-allocated GPU tensors for one block's frozen parameters and buffers.

    Uses packed slabs (one per dtype group) so that each ``load`` requires only
    2-3 ``copy_()`` calls instead of one per parameter.
    """

    __slots__ = ("buf_slab", "data_slab", "gpu_bufs", "gpu_params", "other_slab", "scale_slab")

    def __init__(
        self,
        template_params: list[_PinnedParamBuffer],
        template_bufs: dict[str, torch.Tensor],
        device: torch.device,
    ) -> None:
        # Separate tensors into dtype groups for slab packing
        data_tensors: list[torch.Tensor] = []
        scale_tensors: list[torch.Tensor] = []
        other_tensors: list[torch.Tensor] = []
        data_idx: list[int] = []
        scale_idx: list[int] = []
        other_idx: list[int] = []
        for i, buf in enumerate(template_params):
            if buf.is_quanto:
                data_idx.append(i)
                data_tensors.append(buf.pinned_data)
                scale_idx.append(i)
                scale_tensors.append(buf.pinned_scale)
            else:
                other_idx.append(i)
                other_tensors.append(buf.pinned_data)

        self.data_slab: _PackedSlab | None = _PackedSlab(data_tensors, device) if data_tensors else None
        self.scale_slab: _PackedSlab | None = _PackedSlab(scale_tensors, device) if scale_tensors else None
        self.other_slab: _PackedSlab | None = _PackedSlab(other_tensors, device) if other_tensors else None

        buf_list = list(template_bufs.values())
        self.buf_slab: _PackedSlab | None = _PackedSlab(buf_list, device) if buf_list else None
        self.gpu_bufs: list[torch.Tensor] = [self.buf_slab.gpu_view(i) for i in range(len(buf_list))] if self.buf_slab else []

        # Build nn.Parameter wrappers referencing views into the slabs
        self.gpu_params: list[nn.Parameter] = [None] * len(template_params)  # type: ignore[list-item]
        for slab_pos, param_idx in enumerate(data_idx):
            buf = template_params[param_idx]
            gd = self.data_slab.gpu_view(slab_pos)
            gs = self.scale_slab.gpu_view(slab_pos)
            qt = WeightQBytesTensor.create(buf.qtype, buf.axis, buf.size, buf.stride, gd, gs, buf.act_qt)
            self.gpu_params[param_idx] = nn.Parameter(qt, requires_grad=False)
        for slab_pos, param_idx in enumerate(other_idx):
            self.gpu_params[param_idx] = nn.Parameter(self.other_slab.gpu_view(slab_pos), requires_grad=False)

    def copy_from(self, pinned_slabs: _BlockPinnedSlabs, non_blocking: bool = False) -> None:
        """Copy all block data with 2-4 slab copies instead of 154 individual copies."""
        if self.data_slab is not None:
            self.data_slab.gpu_flat.copy_(pinned_slabs.data_flat, non_blocking=non_blocking)
        if self.scale_slab is not None:
            self.scale_slab.gpu_flat.copy_(pinned_slabs.scale_flat, non_blocking=non_blocking)
        if self.other_slab is not None:
            self.other_slab.gpu_flat.copy_(pinned_slabs.other_flat, non_blocking=non_blocking)
        if self.buf_slab is not None:
            self.buf_slab.gpu_flat.copy_(pinned_slabs.buf_flat, non_blocking=non_blocking)


class _BlockPinnedSlabs:
    """Packed pinned CPU slabs for one block, matching the GPU slot layout."""

    __slots__ = ("buf_flat", "data_flat", "other_flat", "scale_flat")

    def __init__(self, param_bufs: list[_PinnedParamBuffer], buf_tensors: list[torch.Tensor]) -> None:
        data_t = [b.pinned_data for b in param_bufs if b.is_quanto]
        scale_t = [b.pinned_scale for b in param_bufs if b.is_quanto]
        other_t = [b.pinned_data for b in param_bufs if not b.is_quanto]

        self.data_flat = self._pack(data_t)
        self.scale_flat = self._pack(scale_t)
        self.other_flat = self._pack(other_t)
        self.buf_flat = self._pack(buf_tensors)

    @staticmethod
    def _pack(tensors: list[torch.Tensor]) -> torch.Tensor | None:
        if not tensors:
            return None
        total = sum(t.numel() for t in tensors)
        flat = torch.empty(total, dtype=tensors[0].dtype, pin_memory=True)
        off = 0
        for t in tensors:
            if not t.is_contiguous():
                raise ValueError("_PackedSlab only supports contiguous tensors")
            flat[off : off + t.numel()].copy_(t.reshape(-1))
            off += t.numel()
        return flat


class _GpuPool:
    """Pool of pre-allocated GPU buffer slots.

    All blocks share the same parameter structure, so one template is used to
    create ``num_slots`` identical GPU buffer sets.  Slots are acquired on load
    and released on eviction.  Per-slot CUDA events enforce multi-stream safety:
    the prefetch stream waits for compute to finish reading a slot before
    overwriting it with new data.
    """

    def __init__(
        self,
        template_params: list[_PinnedParamBuffer],
        template_bufs: dict[str, torch.Tensor],
        num_slots: int,
        device: torch.device,
    ) -> None:
        self._slots = [_GpuSlot(template_params, template_bufs, device) for _ in range(num_slots)]
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
        """Ensure compute is done reading from this slot before it is reused.

        Uses ``event.query()`` to skip the GPU-side dependency when the
        compute event is already signaled (which is almost always the case
        for LRU victims that were last read 20+ blocks ago).
        """
        ev = self._events[slot_id]
        if ev is not None:
            if stream is not None and not ev.query():
                stream.wait_event(ev)
            self._events[slot_id] = None


# ---------------------------------------------------------------------------
# Block store: pinned CPU + GPU pool
# ---------------------------------------------------------------------------


class _BlockPinnedStore:
    """Manages pinned buffers for all frozen params and buffers in a set of blocks.

    When ``num_gpu_slots > 0``, a :class:`_GpuPool` is allocated to avoid CUDA
    malloc/free during training.  Otherwise falls back to per-load allocation.
    """

    def __init__(
        self,
        layers: list[nn.Module] | nn.ModuleList,
        num_gpu_slots: int = 0,
        device: torch.device | None = None,
    ) -> None:
        self._param_bufs: list[list[_PinnedParamBuffer]] = []
        self._module_bufs: list[dict[str, torch.Tensor]] = []
        for layer in layers:
            block_pbufs: list[_PinnedParamBuffer] = []
            for name, param in layer.named_parameters():
                if not param.requires_grad:
                    block_pbufs.append(_PinnedParamBuffer(name, param))
            self._param_bufs.append(block_pbufs)

        # Cache (submodule, local_name) per param for direct _parameters assignment
        self._param_locs: list[list[tuple[nn.Module, str]]] = []
        # Cache (buffer_tensor, cpu_clone) per block to avoid named_buffers() walks
        self._buf_pairs: list[list[tuple[torch.Tensor, torch.Tensor]]] = []
        self._module_bufs: list[dict[str, torch.Tensor]] = []
        for i, layer in enumerate(layers):
            modules_map = dict(layer.named_modules())
            locs: list[tuple[nn.Module, str]] = []
            for pb in self._param_bufs[i]:
                parts = pb.name.rsplit(".", 1)
                if len(parts) == 2:
                    locs.append((modules_map[parts[0]], parts[1]))
                else:
                    locs.append((layer, pb.name))
            self._param_locs.append(locs)
            buf_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
            buf_dict: dict[str, torch.Tensor] = {}
            for name, buf in layer.named_buffers():
                cpu_clone = buf.data.clone()
                buf_pairs.append((buf, cpu_clone))
                buf_dict[name] = cpu_clone
            self._buf_pairs.append(buf_pairs)
            self._module_bufs.append(buf_dict)

        # Build per-block packed pinned slabs for slab-based DMA
        self._pinned_slabs: list[_BlockPinnedSlabs] = []
        for i in range(len(layers)):
            buf_tensors = [cpu for (_mb, cpu) in self._buf_pairs[i]]
            self._pinned_slabs.append(_BlockPinnedSlabs(self._param_bufs[i], buf_tensors))

        self._pool: _GpuPool | None = None
        self._block_to_slot: dict[int, int] = {}
        if num_gpu_slots > 0 and device is not None and len(self._param_bufs) > 0:
            if self._blocks_are_homogeneous():
                self._pool = _GpuPool(self._param_bufs[0], self._module_bufs[0], num_gpu_slots, device)
            else:
                logger.info("Blocks have heterogeneous structure; using per-load GPU allocation")

    def _blocks_are_homogeneous(self) -> bool:
        if len(self._param_bufs) <= 1:
            return True
        ref_params = [(b.name, b.pinned_data.shape, b.pinned_data.dtype, b.is_quanto,
                        b.pinned_scale.shape if b.pinned_scale is not None else None)
                       for b in self._param_bufs[0]]
        ref_bufs = [(k, v.shape) for k, v in self._module_bufs[0].items()]
        for i in range(1, len(self._param_bufs)):
            block_bufs = self._param_bufs[i]
            if len(block_bufs) != len(ref_params):
                return False
            for (rn, rs, rd, rq, rss), b in zip(ref_params, block_bufs, strict=True):
                if b.name != rn or b.pinned_data.shape != rs or b.pinned_data.dtype != rd or b.is_quanto != rq:
                    return False
                if rss is not None and (b.pinned_scale is None or b.pinned_scale.shape != rss):
                    return False
            mod_bufs = [(k, v.shape) for k, v in self._module_bufs[i].items()]
            if mod_bufs != ref_bufs:
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

    # -- internal -------------------------------------------------------------

    def _load_pooled(self, idx: int, non_blocking: bool, stream: torch.cuda.Stream | None) -> None:
        slot_id = self._block_to_slot.get(idx)
        if slot_id is None:
            slot_id = self._pool.acquire()
            self._block_to_slot[idx] = slot_id

        self._pool.wait_if_needed(slot_id, stream)
        slot = self._pool.slot(slot_id)
        slot.copy_from(self._pinned_slabs[idx], non_blocking=non_blocking)

        for (submod, local_name), gpu_param in zip(self._param_locs[idx], slot.gpu_params, strict=True):
            submod._parameters[local_name] = gpu_param
        for (mod_buf, _cpu), gpu_val in zip(self._buf_pairs[idx], slot.gpu_bufs, strict=True):
            mod_buf.data = gpu_val

    def _load_alloc(self, idx: int, device: torch.device, non_blocking: bool) -> None:
        for (submod, local_name), buf in zip(self._param_locs[idx], self._param_bufs[idx], strict=True):
            submod._parameters[local_name] = buf.load_to_gpu(device, non_blocking=non_blocking)
        for (mod_buf, cpu_data), _name in zip(self._buf_pairs[idx], self._module_bufs[idx], strict=True):
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
        for (submod, local_name), buf in zip(self._param_locs[idx], self._param_bufs[idx], strict=True):
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
# LoRA param restore after module-level moves
# ---------------------------------------------------------------------------


def _move_lora_to_device(layer: nn.Module, device: torch.device) -> None:
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
    GPU permanently, so backward through them is unaffected by the offload.

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
    """

    def __init__(
        self,
        model: nn.Module,
        target_device: torch.device,
        blocks_to_swap: int,
        layers_attr: str | list[str],
        prefetch_count: int = 3,
    ) -> None:
        self._model = model
        self._target_device = target_device
        self._blocks_to_swap = blocks_to_swap
        self._prefetch_count = prefetch_count
        self._layers_attrs = [layers_attr] if isinstance(layers_attr, str) else list(layers_attr)

        self._layers: list[nn.Module] | None = None
        self._tracker: _BlockTracker | None = None
        self._store: _BlockPinnedStore | None = None
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._executor: ThreadPoolExecutor | None = None
        self._stream: torch.cuda.Stream | None = None
        self._pending: dict[int, Future[None]] = {}
        self._prefetch_events: dict[int, torch.cuda.Event] = {}
        self._last_idx: int = -1

        self.setup()

    # ------------------------------------------------------------------
    # Setup / teardown
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Initialize offloading state. Re-callable after ``teardown()``."""
        if self._tracker is not None or self._hooks:
            self.teardown()

        self._layers, block_leaf_names = self._resolve_all_layers()
        num_layers = len(self._layers)
        if self._blocks_to_swap >= num_layers:
            raise ValueError(f"blocks_to_swap ({self._blocks_to_swap}) must be < num_layers ({num_layers})")

        num_resident = num_layers - self._blocks_to_swap
        self._tracker = _BlockTracker(num_layers)
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._stream = torch.cuda.Stream(device=self._target_device, priority=-1)
        self._pending = {}
        self._prefetch_events = {i: torch.cuda.Event() for i in range(num_layers)}
        self._last_idx = -1

        # Move non-block modules to GPU
        parent_paths: set[str] = set()
        for attr_path in self._layers_attrs:
            parts = attr_path.split(".")
            parent_paths.add(".".join(parts[:-1]) if len(parts) > 1 else "")
        for parent_path in parent_paths:
            parent = _resolve_dotted(self._model, parent_path) if parent_path else self._model
            for name, child in parent.named_children():
                if name not in block_leaf_names:
                    child.to(self._target_device)

        # Move all blocks to CPU (LoRA stays on GPU)
        for layer in self._layers:
            layer.to("cpu")
            _move_lora_to_device(layer, self._target_device)

        # Create pinned buffers and GPU pool from the CPU state
        num_gpu_slots = num_resident + self._prefetch_count
        self._store = _BlockPinnedStore(self._layers, num_gpu_slots=num_gpu_slots, device=self._target_device)

        # Pre-load initial resident window (synchronous)
        for idx in range(min(num_resident, num_layers)):
            self._store.load_block(idx, self._layers[idx], self._target_device)
            self._tracker.mark_on_gpu(idx)

        self._register_hooks(num_resident)

        # Seed peak to reflect the pre-loaded resident window. Since cb83965,
        # peak_gpu_blocks is only updated inside the forward-pre hook; without
        # this seed, peak stays at 0 between setup() and the first forward,
        # which matters for teardown/setup cycles (validation) where a
        # post-validation callback may read peak before any new forward runs.
        self.reset_peak()

        logger.info(
            f"Block offloading active: {self._blocks_to_swap}/{num_layers} blocks on CPU, "
            f"{num_resident} resident on GPU, prefetch={self._prefetch_count}, "
            f"gpu_pool_slots={num_gpu_slots}"
        )

    def _resolve_all_layers(self) -> tuple[list[nn.Module], set[str]]:
        flat: list[nn.Module] = []
        leaf_names: set[str] = set()
        for attr_path in self._layers_attrs:
            module_list = _resolve_attr(self._model, attr_path)
            flat.extend(module_list)
            leaf_names.add(attr_path.split(".")[-1])
        return flat, leaf_names

    def teardown(self) -> None:
        """Remove hooks, wait for pending work, evict all blocks to CPU."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

        # Wait for pending prefetches and mark them as on-GPU so teardown saves them
        for idx, future in self._pending.items():
            future.result()
            if self._tracker is not None:
                self._tracker.mark_on_gpu(idx)
        self._pending.clear()
        self._prefetch_events.clear()

        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

        if self._stream is not None:
            self._stream.synchronize()
            self._stream = None

        if self._tracker is not None and self._store is not None:
            torch.cuda.synchronize(device=self._target_device)
            # Restore CPU params for ALL blocks. Fast-evicted blocks still
            # have module params pointing at (now-stale) GPU slot data.
            for idx, layer in enumerate(self._layers):
                self._store.evict_block(idx, layer)
            self._tracker.clear()

        self._tracker = None
        self._store = None
        self._layers = None

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

# Back-compat alias — old name from when this lived in ltx-trainer.
TrainingBlockOffloader = BlockOffloader
