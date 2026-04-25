"""Whole-model pinned-CPU weight cache for fast bulk DMA to GPU.

Holds a model's frozen weights in pinned CPU memory so subsequent GPU
loads are bulk DMA (~200 ms for a 12 GB Gemma at PCIe Gen5 x16) instead
of re-reading the safetensors from disk (~3-5 s per call).

Use case: a model that fits on GPU when active but should be evicted
between calls — text encoder during diffusion, VAE between encode and
decode phases, etc. Different from :class:`BlockOffloader`: no per-block
streaming, no forward hooks, no LRU. The whole model goes to GPU on
context entry; on exit, parameter ``.data`` is repointed back at the
pinned CPU storage so the GPU storage is released by refcount.

Caveats
-------
- The constructor *mutates* the wrapped ``model`` — its frozen
  ``.data`` tensors are repointed at pinned CPU buffers and its
  registered buffers are replaced with pinned copies. Only use the
  model via :meth:`on_gpu` after wrapping.
- Buffer mutations during forward (RNN/SSM state, KV cache,
  training-mode BatchNorm running stats) are *discarded* on exit.
  Suitable for inference of stateless modules; not suitable for
  models that need persistent buffer state across calls.
- Incompatible with ``torch.compile`` (compile traces capture tensor
  identity; ``.data`` swaps invalidate the trace).
- Wrap the model *before* DDP/FSDP — those wrappers manage ``.data``
  themselves and conflict with this class.
- ``on_gpu()`` is not re-entrant: nested calls raise ``RuntimeError``.
- Not thread-safe: concurrent callers on the same instance race on
  ``.data`` assignment.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from typing import Any

import torch
from torch import nn

from ltx_core.memory._buffers import PinnedParamBuffer

logger = logging.getLogger(__name__)


def _set_buffer(module: nn.Module, name: str, value: torch.Tensor, persistent: bool) -> None:
    """Replace a registered buffer in-place by its leaf name on
    ``module``, preserving the original ``persistent`` flag so
    ``state_dict()`` behavior survives the swap."""
    module.register_buffer(name, value, persistent=persistent)


class PinnedWeights:
    """Whole-model pinned-CPU weight cache with bulk GPU transfer.

    On construction, every frozen ``nn.Parameter`` is wrapped in a
    :class:`PinnedParamBuffer` (handling quanto decomposition where
    applicable) and the model's ``param.data`` is repointed at the
    pinned ``cpu_param``. The :meth:`on_gpu` context manager transfers
    every pinned buffer to ``target_device`` for the duration of the
    with-block, then repoints back to the pinned CPU storage on exit
    so the GPU storage is released by refcount.

    Trainable parameters (``requires_grad=True``) are not pinned;
    constructing on a model with no frozen parameters raises.

    Parameters
    ----------
    model:
        The model to cache. Should be on CPU when passed in (we won't
        move it for you — that lets the caller control build-time
        device).
    target_device:
        GPU device to bulk-transfer to in :meth:`on_gpu`.
    include_buffers:
        Also cache registered buffers (LayerNorm running stats, position
        embeddings stored as buffers, etc.). Default True. Set False
        for models with very large mutable buffers you'd rather rebuild
        on each call.
    """

    def __init__(
        self,
        model: nn.Module,
        target_device: torch.device,
        include_buffers: bool = True,
    ) -> None:
        self._model = model
        self._device = target_device
        self._include_buffers = include_buffers
        self._active = False  # guards re-entry of on_gpu() and teardown-while-active

        # Pin every frozen parameter via PinnedParamBuffer (quanto-aware).
        self._param_bufs: dict[str, PinnedParamBuffer] = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                continue
            self._param_bufs[name] = PinnedParamBuffer(name, p)

        if not self._param_bufs:
            raise ValueError(
                "PinnedWeights requires at least one frozen parameter to cache. "
                "All params on the wrapped model have requires_grad=True - for "
                "training flows use ltx_core.memory.BlockOffloader instead."
            )

        # Cache (param, buf) pairs once so per-call moves don't re-walk
        # named_parameters().
        self._frozen_params: list[tuple[nn.Parameter, PinnedParamBuffer]] = []
        for name, p in model.named_parameters():
            buf = self._param_bufs.get(name)
            if buf is not None:
                p.data = buf.cpu_param.data
                self._frozen_params.append((p, buf))

        # Cache buffers if requested. Capture each buffer's original
        # persistent flag so the swap doesn't silently demote it.
        self._buffer_pins: list[tuple[nn.Module, str, torch.Tensor, bool]] = []
        if include_buffers:
            for full_name, b in list(model.named_buffers()):
                parent = self._resolve_parent(model, full_name)
                leaf = full_name.rsplit(".", 1)[-1]
                persistent = leaf not in parent._non_persistent_buffers_set
                pinned = b.detach().clone(memory_format=torch.contiguous_format).pin_memory()
                _set_buffer(parent, leaf, pinned, persistent)
                self._buffer_pins.append((parent, leaf, pinned, persistent))

    @staticmethod
    def _resolve_parent(model: nn.Module, dotted_name: str) -> nn.Module:
        parent: Any = model
        parts = dotted_name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        return parent

    @property
    def pinned_bytes(self) -> int:
        """Total pinned CPU memory currently held."""
        total = 0
        for buf in self._param_bufs.values():
            total += buf.pinned_data.numel() * buf.pinned_data.element_size()
            if buf.pinned_scale is not None:
                total += buf.pinned_scale.numel() * buf.pinned_scale.element_size()
        for _, _, pinned, _ in self._buffer_pins:
            total += pinned.numel() * pinned.element_size()
        return total

    @contextlib.contextmanager
    def on_gpu(self) -> Iterator[nn.Module]:
        """Bulk-DMA pinned weights to GPU; yield model; restore on exit.

        On entry, every pinned param + buffer is transferred to
        ``target_device`` (per-tensor ``.to()``, non-blocking, then a
        single ``cuda.synchronize`` to make the writes visible).
        Inside the with-block, the model is fully GPU-resident. On
        exit, parameter ``.data`` is repointed back at the pinned CPU
        storage so GPU storage is released by refcount.

        Not re-entrant; nested calls raise ``RuntimeError``.
        """
        if self._active:
            raise RuntimeError(
                "PinnedWeights.on_gpu() is not re-entrant. The wrapped model "
                "is already inside an active on_gpu() context."
            )
        if not self._param_bufs:
            raise RuntimeError(
                "PinnedWeights has been torn down - pinned buffers are released."
            )
        self._active = True
        try:
            self._move_to_gpu()
            try:
                yield self._model
            finally:
                self._move_to_pinned()
        finally:
            self._active = False

    def _move_to_gpu(self) -> None:
        for p, buf in self._frozen_params:
            gpu_param = buf.load_to_gpu(self._device, non_blocking=True)
            p.data = gpu_param.data
        if self._include_buffers:
            for parent, leaf, pinned, persistent in self._buffer_pins:
                _set_buffer(parent, leaf, pinned.to(self._device, non_blocking=True), persistent)
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)

    def _move_to_pinned(self) -> None:
        for p, buf in self._frozen_params:
            p.data = buf.cpu_param.data
        if self._include_buffers:
            for parent, leaf, pinned, persistent in self._buffer_pins:
                _set_buffer(parent, leaf, pinned, persistent)

    def teardown(self) -> None:
        """Release pinned CPU buffers. The wrapped model is unusable
        after teardown. Raises if called while inside an ``on_gpu()``
        context."""
        if self._active:
            raise RuntimeError(
                "PinnedWeights.teardown() called while on_gpu() is active - "
                "exit the context first or the model would be left holding "
                "GPU tensors that cannot be restored."
            )
        self._param_bufs.clear()
        self._frozen_params.clear()
        self._buffer_pins.clear()
