"""Whole-model pinned-CPU weight cache for fast bulk DMA.

Holds a model's frozen weights in pinned CPU memory so that subsequent
GPU loads are bulk DMA (~200 ms for a 12 GB Gemma at PCIe Gen5 x16)
instead of re-reading the safetensors from disk (~3-5 s per call).

Use case: a model that fits on GPU when active but should be evicted
between calls — e.g., the text encoder during diffusion. Different from
``streaming.BlockOffloader``: no per-block streaming, no forward hooks,
no LRU. The whole model goes to GPU on context entry and back to its
pinned CPU buffers on exit.

Example
-------
>>> from ltx_core.memory import PinnedWeights
>>> import torch
>>>
>>> model = build_text_encoder(device="cpu")        # build on CPU
>>> cache = PinnedWeights(model, torch.device("cuda"))
>>> # Model now holds its weights in pinned CPU memory.
>>>
>>> with cache.on_gpu() as gpu_model:
...     embeddings = gpu_model.encode(prompt)
>>> # Weights repointed back to pinned CPU; GPU memory freed.
>>>
>>> # Next call: bulk DMA, no disk read.
>>> with cache.on_gpu() as gpu_model:
...     embeddings = gpu_model.encode(another_prompt)
>>>
>>> cache.teardown()                                # release pinned RAM
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from typing import Any

import torch
from torch import nn

from ltx_core.memory.streaming import _PinnedParamBuffer

logger = logging.getLogger(__name__)


def _set_buffer(model: nn.Module, dotted_name: str, value: torch.Tensor) -> None:
    """Replace a registered buffer in-place by its dotted name."""
    parent: Any = model
    parts = dotted_name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    parent.register_buffer(parts[-1], value, persistent=False)


class PinnedWeights:
    """Whole-model pinned-CPU weight cache with bulk GPU transfer.

    On construction, every frozen ``nn.Parameter`` (and optionally every
    registered buffer) of ``model`` is cloned into pinned CPU memory and
    the model's tensors are repointed at those pinned buffers. The
    :meth:`on_gpu` context manager bulk-DMAs everything to ``target_device``
    for the duration of the with-block, then repoints back to the pinned
    buffers on exit.

    Trainable parameters (``requires_grad=True``) are *not* pinned — they
    stay on whatever device they were on when ``PinnedWeights`` was built.
    For typical inference flows that's a non-issue (everything is frozen);
    for training flows use :class:`BlockOffloader` instead.

    Parameters
    ----------
    model:
        The model to cache. Should be on CPU when passed in (we won't move
        it for you — that lets the caller control build-time device).
    target_device:
        GPU device to bulk-transfer to in :meth:`on_gpu`.
    include_buffers:
        Also cache registered buffers (LayerNorm running stats, position
        embeddings stored as buffers, etc.). Default True. Set False for
        models with very large mutable buffers you'd rather rebuild.
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

        # Pin every frozen parameter; trainable params left untouched.
        self._param_pins: dict[str, _PinnedParamBuffer] = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                continue
            self._param_pins[name] = _PinnedParamBuffer(name, p)

        # Repoint each frozen param's .data at its pinned buffer.
        # Also reconstruct the quanto wrapper if applicable so the layer's
        # forward keeps seeing a quantized tensor (not a raw byte buffer).
        for name, p in model.named_parameters():
            if name not in self._param_pins:
                continue
            buf = self._param_pins[name]
            p.data = buf.cpu_param.data

        # Cache buffers if requested. Buffers are simple tensors; we just
        # clone+pin and reassign via register_buffer.
        self._buffer_pins: dict[str, torch.Tensor] = {}
        if include_buffers:
            for name, b in list(model.named_buffers()):
                pinned = b.detach().clone(memory_format=torch.contiguous_format).pin_memory()
                self._buffer_pins[name] = pinned
                _set_buffer(model, name, pinned)

    @property
    def pinned_bytes(self) -> int:
        """Total pinned CPU memory currently held."""
        total = 0
        for buf in self._param_pins.values():
            total += buf.pinned_data.numel() * buf.pinned_data.element_size()
            if buf.pinned_scale is not None:
                total += buf.pinned_scale.numel() * buf.pinned_scale.element_size()
        for b in self._buffer_pins.values():
            total += b.numel() * b.element_size()
        return total

    @contextlib.contextmanager
    def on_gpu(self) -> Iterator[nn.Module]:
        """Bulk-DMA pinned weights to GPU; yield model; restore on exit.

        On entry: every cached param + buffer is transferred to
        ``target_device`` (non-blocking, then a single ``cuda.synchronize``).
        Inside the with-block, the model is fully GPU-resident and can be
        called normally. On exit, ``.data`` is repointed back to the pinned
        CPU buffers — the GPU storage is released by Python refcount once
        no caller still holds a reference to it.
        """
        self._move_to_gpu()
        try:
            yield self._model
        finally:
            self._move_to_pinned()

    def _move_to_gpu(self) -> None:
        for name, p in self._model.named_parameters():
            if name not in self._param_pins:
                continue
            gpu_param = self._param_pins[name].load_to_gpu(self._device, non_blocking=True)
            p.data = gpu_param.data
        if self._include_buffers:
            for name in list(self._buffer_pins):
                _set_buffer(
                    self._model,
                    name,
                    self._buffer_pins[name].to(self._device, non_blocking=True),
                )
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)

    def _move_to_pinned(self) -> None:
        for name, p in self._model.named_parameters():
            if name not in self._param_pins:
                continue
            p.data = self._param_pins[name].cpu_param.data
        if self._include_buffers:
            for name, pinned in self._buffer_pins.items():
                _set_buffer(self._model, name, pinned)

    def teardown(self) -> None:
        """Release pinned CPU buffers. The wrapped model is unusable after."""
        self._param_pins.clear()
        self._buffer_pins.clear()
