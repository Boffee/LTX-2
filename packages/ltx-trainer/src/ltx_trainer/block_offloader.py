"""Block-level CPU offloading for memory-efficient LoRA training.

Keeps most frozen transformer block weights on CPU in persistent pinned
memory buffers. Uses a background thread and dedicated CUDA stream to
prefetch upcoming blocks, overlapping DMA with compute.

Quanto ``WeightQBytesTensor`` weights are decomposed into their inner
``_data`` (int8) and ``_scale`` (float) components for pinned-buffer DMA,
then reconstructed on GPU via ``WeightQBytesTensor.create()``.

Uses LRU eviction so the pre_hook works regardless of traversal direction
(forward 0→47 or backward recomputation 47→0).

LoRA parameters stay on GPU permanently.
"""

from __future__ import annotations

import functools
import logging
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)

_QUANTO_AVAILABLE = False
try:
    from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor

    _QUANTO_AVAILABLE = True
except ImportError:
    pass


def _resolve_attr(module: nn.Module, dotted_path: str) -> nn.ModuleList:
    obj: Any = module
    for part in dotted_path.split("."):
        obj = getattr(obj, part)
    if not isinstance(obj, nn.ModuleList):
        raise TypeError(f"Expected nn.ModuleList at '{dotted_path}', got {type(obj).__name__}")
    return obj


# ---------------------------------------------------------------------------
# Pinned buffer management for frozen (possibly quantized) parameters
# ---------------------------------------------------------------------------


class _PinnedParamBuffer:
    """Persistent pinned CPU buffer for a single frozen parameter."""

    __slots__ = ("name", "is_quanto", "pinned_data", "pinned_scale", "qtype", "axis", "size", "stride", "act_qt")

    def __init__(self, name: str, param: nn.Parameter) -> None:
        self.name = name
        t = param.data
        if _QUANTO_AVAILABLE and isinstance(t, WeightQBytesTensor):
            self.is_quanto = True
            self.pinned_data = t._data.clone().pin_memory()
            self.pinned_scale = t._scale.clone().pin_memory()
            self.qtype = t.qtype
            self.axis = t.axis
            self.size = t.size()
            self.stride = t.stride()
            self.act_qt = getattr(t, "activation_qtype", None)
        else:
            self.is_quanto = False
            self.pinned_data = t.data.clone().pin_memory()
            self.pinned_scale = None
            self.qtype = self.axis = self.size = self.stride = self.act_qt = None

    def load_to_gpu(self, device: torch.device, non_blocking: bool = False) -> nn.Parameter:
        if self.is_quanto:
            gd = self.pinned_data.to(device, non_blocking=non_blocking)
            gs = self.pinned_scale.to(device, non_blocking=non_blocking)
            qt = WeightQBytesTensor.create(self.qtype, self.axis, self.size, self.stride, gd, gs, self.act_qt)
            return nn.Parameter(qt, requires_grad=False)
        return nn.Parameter(self.pinned_data.to(device, non_blocking=non_blocking), requires_grad=False)

    def save_from_gpu(self, param: nn.Parameter) -> nn.Parameter:
        """Copy GPU param back to pinned buffer, return CPU param pointing to pinned storage."""
        t = param.data
        if self.is_quanto:
            self.pinned_data.copy_(t._data)
            self.pinned_scale.copy_(t._scale)
            qt = WeightQBytesTensor.create(self.qtype, self.axis, self.size, self.stride, self.pinned_data, self.pinned_scale, self.act_qt)
            return nn.Parameter(qt, requires_grad=False)
        self.pinned_data.copy_(t)
        return nn.Parameter(self.pinned_data, requires_grad=False)


class _BlockPinnedStore:
    """Manages pinned buffers for all frozen params in a set of blocks."""

    def __init__(self, layers: nn.ModuleList) -> None:
        self._buffers: list[list[_PinnedParamBuffer]] = []
        for layer in layers:
            block_bufs = []
            for name, param in layer.named_parameters():
                if not param.requires_grad:
                    block_bufs.append(_PinnedParamBuffer(name, param))
            self._buffers.append(block_bufs)

    def load_block(self, idx: int, layer: nn.Module, device: torch.device, non_blocking: bool = False) -> None:
        params = dict(layer.named_parameters())
        for buf in self._buffers[idx]:
            new_p = buf.load_to_gpu(device, non_blocking=non_blocking)
            torch.utils.swap_tensors(params[buf.name], new_p)

    def save_block(self, idx: int, layer: nn.Module) -> None:
        params = dict(layer.named_parameters())
        for buf in self._buffers[idx]:
            new_p = buf.save_from_gpu(params[buf.name])
            torch.utils.swap_tensors(params[buf.name], new_p)


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
        if len(self._on_gpu) > self.peak_gpu_blocks:
            self.peak_gpu_blocks = len(self._on_gpu)

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


class TrainingBlockOffloader:
    """Streams frozen transformer blocks between CPU and GPU for LoRA training.

    Frozen weights are kept in persistent pinned CPU buffers. Prefetch uses
    a background thread and dedicated CUDA stream to overlap DMA with compute.
    Quanto ``WeightQBytesTensor`` is decomposed into ``_data``/``_scale`` for
    DMA and reconstructed on GPU.

    Parameters
    ----------
    model:
        The model containing the block list (may be PEFT-wrapped).
    target_device:
        The GPU device to use for compute.
    blocks_to_swap:
        Number of blocks to keep offloaded on CPU. Must be < total blocks.
    layers_attr:
        Auto-detected if not provided.
    prefetch_count:
        How many blocks ahead to prefetch on a background thread.
    """

    _LAYERS_ATTR_CANDIDATES = [
        "base_model.model.transformer_blocks",
        "transformer_blocks",
    ]

    def __init__(
        self,
        model: nn.Module,
        target_device: torch.device,
        blocks_to_swap: int,
        layers_attr: str | None = None,
        prefetch_count: int = 2,
    ) -> None:
        self._model = model
        self._layers_attr = layers_attr or self._detect_layers_attr(model)
        self._target_device = target_device
        self._blocks_to_swap = blocks_to_swap
        self._prefetch_count = prefetch_count

        self._layers: nn.ModuleList | None = None
        self._tracker: _BlockTracker | None = None
        self._store: _BlockPinnedStore | None = None
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._executor: ThreadPoolExecutor | None = None
        self._stream: torch.cuda.Stream | None = None
        self._pending: dict[int, Future[torch.cuda.Event]] = {}
        self._last_idx: int = -1

        self.setup()

    @classmethod
    def _detect_layers_attr(cls, model: nn.Module) -> str:
        for path in cls._LAYERS_ATTR_CANDIDATES:
            try:
                _resolve_attr(model, path)
                return path
            except (AttributeError, TypeError):
                continue
        raise ValueError(f"Could not find transformer blocks at any of: {cls._LAYERS_ATTR_CANDIDATES}")

    # ------------------------------------------------------------------
    # Setup / teardown
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Initialize offloading state. Re-callable after ``teardown()``."""
        if self._tracker is not None or self._hooks:
            self.teardown()

        self._layers = _resolve_attr(self._model, self._layers_attr)
        num_layers = len(self._layers)
        if self._blocks_to_swap >= num_layers:
            raise ValueError(f"blocks_to_swap ({self._blocks_to_swap}) must be < num_layers ({num_layers})")

        num_resident = num_layers - self._blocks_to_swap
        self._tracker = _BlockTracker(num_layers)
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._stream = torch.cuda.Stream(device=self._target_device)
        self._pending = {}
        self._last_idx = -1

        # Move non-block modules to GPU
        layers_attr_parts = self._layers_attr.split(".")
        parent: Any = self._model
        for part in layers_attr_parts[:-1]:
            parent = getattr(parent, part)
        layers_leaf = layers_attr_parts[-1]
        for name, child in parent.named_children():
            if name != layers_leaf:
                child.to(self._target_device)

        # Move all blocks to CPU (LoRA stays on GPU)
        for layer in self._layers:
            layer.to("cpu")
            _move_lora_to_device(layer, self._target_device)

        # Create pinned buffers from the CPU state
        self._store = _BlockPinnedStore(self._layers)

        # Pre-load initial resident window (synchronous)
        for idx in range(min(num_resident, num_layers)):
            self._store.load_block(idx, self._layers[idx], self._target_device)
            self._tracker.mark_on_gpu(idx)

        # Also move buffers (input_scale, output_scale) for resident blocks
        for idx in range(min(num_resident, num_layers)):
            for b in self._layers[idx].buffers():
                if not b.data.is_cuda:
                    b.data = b.data.to(self._target_device)

        self._register_hooks(num_resident)

        logger.info(
            f"Block offloading active: {self._blocks_to_swap}/{num_layers} blocks on CPU, "
            f"{num_resident} resident on GPU, prefetch={self._prefetch_count}"
        )

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

        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

        if self._stream is not None:
            self._stream.synchronize()
            self._stream = None

        if self._tracker is not None and self._store is not None:
            torch.cuda.synchronize(device=self._target_device)
            for idx, layer in enumerate(self._layers):
                if self._tracker.is_on_gpu(idx):
                    self._store.save_block(idx, layer)
                    # Also move buffers back to CPU
                    for b in layer.buffers():
                        if b.data.is_cuda:
                            b.data = b.data.to("cpu")
            self._tracker.clear()

        self._tracker = None
        self._store = None
        self._layers = None

    # ------------------------------------------------------------------
    # Block transfer
    # ------------------------------------------------------------------

    def _evict_one(self, protected: set[int]) -> None:
        victim = self._tracker.pick_victim(protected=protected)
        self._store.save_block(victim, self._layers[victim])
        for b in self._layers[victim].buffers():
            if b.data.is_cuda:
                b.data = b.data.to("cpu")
        self._tracker.mark_on_cpu(victim)

    def _do_prefetch(self, idx: int) -> torch.cuda.Event:
        """Background thread: transfer block to GPU on the prefetch stream."""
        with torch.cuda.stream(self._stream):
            self._store.load_block(idx, self._layers[idx], self._target_device, non_blocking=True)
            for b in self._layers[idx].buffers():
                if not b.data.is_cuda:
                    b.data = b.data.to(self._target_device, non_blocking=True)
        return self._stream.record_event()

    def _submit_prefetch(self, idx: int, max_on_gpu: int) -> None:
        if idx < 0 or idx >= len(self._layers):
            return
        if self._tracker.is_on_gpu(idx) or idx in self._pending:
            return
        if len(self._tracker._on_gpu) + len(self._pending) >= max_on_gpu:
            return
        self._pending[idx] = self._executor.submit(self._do_prefetch, idx)

    def _ensure_on_gpu(self, idx: int) -> None:
        """Wait for pending prefetch or load synchronously."""
        future = self._pending.pop(idx, None)
        if future is not None:
            event = future.result()
            torch.cuda.current_stream(self._target_device).wait_event(event)
            self._tracker.mark_on_gpu(idx)
            return

        if not self._tracker.is_on_gpu(idx):
            self._store.load_block(idx, self._layers[idx], self._target_device)
            for b in self._layers[idx].buffers():
                if not b.data.is_cuda:
                    b.data = b.data.to(self._target_device)
            self._tracker.mark_on_gpu(idx)

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def _register_hooks(self, num_resident: int) -> None:
        idx_map: dict[int, int] = {id(layer): idx for idx, layer in enumerate(self._layers)}
        max_on_gpu = num_resident + self._prefetch_count

        def _pre_hook(module: nn.Module, _args: Any, *, idx: int) -> None:  # noqa: ANN401
            if self._tracker.is_on_gpu(idx):
                self._tracker.touch(idx)
            else:
                while len(self._tracker._on_gpu) >= num_resident:
                    protected = {idx} | set(self._pending.keys())
                    self._evict_one(protected)
                self._ensure_on_gpu(idx)

            direction = 1 if idx >= self._last_idx else -1
            self._last_idx = idx
            for offset in range(1, self._prefetch_count + 1):
                self._submit_prefetch(idx + direction * offset, max_on_gpu)

        for layer in self._layers:
            idx = idx_map[id(layer)]
            h = layer.register_forward_pre_hook(functools.partial(_pre_hook, idx=idx))
            self._hooks.append(h)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def peak_gpu_blocks(self) -> int:
        return self._tracker.peak_gpu_blocks if self._tracker is not None else 0

    def reset_peak(self) -> None:
        if self._tracker is not None:
            self._tracker.peak_gpu_blocks = len(self._tracker._on_gpu)
