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

from ltx_core.memory.buffers import PinnedParamBuffer

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


class GpuSlot:
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


class GpuSlotPool:
    """Pool of pre-allocated :class:`GpuSlot` instances.

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
        self._slots = [GpuSlot(template, device) for _ in range(num_slots)]
        self._free: list[int] = list(range(num_slots))
        self._events: list[torch.cuda.Event | None] = [None] * num_slots

    def acquire(self) -> int:
        return self._free.pop()

    def release(self, slot_id: int) -> None:
        self._free.append(slot_id)

    def slot(self, slot_id: int) -> GpuSlot:
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
# Block store: pinned CPU + GPU pool
# ---------------------------------------------------------------------------


class BlockPinnedStore:
    """Per-block pinned CPU + per-slot GPU storage for frozen weights.

    For each transformer block, one ``PinnedParamBuffer`` per frozen
    parameter holds the pinned-CPU clone (decomposing quanto into
    ``_data`` + ``_scale`` if applicable). The model's ``param.data``
    is repointed at the pinned buffer's ``cpu_param`` so the block
    can run on CPU without any extra storage when offloaded.

    When ``num_gpu_slots > 0``, a :class:`GpuSlotPool` is allocated
    to avoid CUDA malloc/free during training. Slot reuse via
    in-place ``copy_()`` keeps the GPU footprint bounded at
    ``num_slots × block_size`` regardless of model depth.

    Buffers (registered via ``register_buffer``) are kept simple:
    per-buffer CPU clone, per-buffer ``.to(device)`` on load. They
    are typically tiny (norm eps, RoPE position tables) so a slab
    abstraction would be over-engineering.
    """

    def __init__(
        self,
        layers: list[nn.Module] | nn.ModuleList,
        num_gpu_slots: int = 0,
        device: torch.device | None = None,
    ) -> None:
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
                # Repoint the model's param at the pinned cpu_param so the
                # block can run on CPU without extra storage when offloaded.
                p.data = buf.cpu_param.data
                block_bufs.append(buf)
                parts = qual_name.rsplit(".", 1)
                if len(parts) == 2:
                    submod, local_name = modules_map[parts[0]], parts[1]
                else:
                    submod, local_name = layer, qual_name
                block_locs.append((qual_name, submod, local_name))
            self._param_bufs.append(block_bufs)
            self._param_locs.append(block_locs)

            # Capture (buffer_obj, cpu_clone) — clone owns CPU storage so
            # the GPU-side .data swap on load doesn't lose the source.
            buf_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
            for _name, b in layer.named_buffers():
                cpu_clone = b.data.clone()
                b.data = cpu_clone
                buf_pairs.append((b, cpu_clone))
            self._buf_pairs.append(buf_pairs)

        self._device = device
        self._pool: GpuSlotPool | None = None
        self._block_to_slot: dict[int, int] = {}
        if num_gpu_slots > 0 and device is not None and self._param_bufs:
            if self._blocks_are_homogeneous():
                self._pool = GpuSlotPool(self._param_bufs[0], num_gpu_slots, device)
            else:
                logger.info("Blocks have heterogeneous structure; using per-load GPU allocation")

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


class BlockTracker:
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

    Caveats
    -------
    - **Tied weights are not deduplicated.** Two ``nn.Parameter`` objects
      sharing the same storage (rare in transformer block lists, but
      possible in some architectures) are cloned into separate pinned
      buffers, doubling pinned memory and breaking the tying invariant on
      GPU. LTX-2 transformer blocks have no tied weights so this is a
      latent concern; restore explicit dedup if a consumer hits it.

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
        self._tracker: BlockTracker | None = None
        self._store: BlockPinnedStore | None = None
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
        self._tracker = BlockTracker(num_layers)
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

        # Create pinned buffers and GPU pool from the CPU state.
        num_gpu_slots = num_resident + self._prefetch_count
        self._store = BlockPinnedStore(self._layers, num_gpu_slots=num_gpu_slots, device=self._target_device)

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
