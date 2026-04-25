"""Whole-model pinned-CPU weight cache for fast bulk DMA to GPU.

Holds a model's frozen weights in pinned CPU memory so that subsequent
GPU loads are bulk DMA (~200 ms for a 12 GB Gemma at PCIe Gen5 x16)
instead of re-reading the safetensors from disk (~3-5 s per call).

Use case: a model that fits on GPU when active but should be evicted
between calls — e.g., the text encoder during diffusion. Different from
:class:`BlockOffloader`: no per-block streaming, no forward hooks, no
LRU. The whole model goes to GPU on context entry; on exit, the GPU
slab is destroyed and parameter ``.data`` is repointed back at the
pinned CPU slab views, so GPU memory is fully released between calls.

Caveats
-------
- The constructor *mutates* the wrapped ``model`` — its frozen ``.data``
  tensors are repointed at views into the pinned slab and its registered
  buffers are replaced with pinned copies. Only use the model via
  :meth:`on_gpu` after wrapping.
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

from ltx_core.memory._buffers import GpuSlab, PinnedSlab

logger = logging.getLogger(__name__)


def _set_buffer(module: nn.Module, name: str, value: torch.Tensor, persistent: bool) -> None:
    """Replace a registered buffer in-place by its leaf name on ``module``,
    preserving the original ``persistent`` flag so ``state_dict()``
    behavior survives the swap."""
    module.register_buffer(name, value, persistent=persistent)


class PinnedWeights:
    """Whole-model pinned-CPU weight cache with bulk GPU transfer.

    On construction, every frozen ``nn.Parameter`` is packed into a
    single :class:`PinnedSlab` (the slab is the sole CPU pinned source
    of truth — there are no per-param clones), and registered buffers
    are individually pinned. The :meth:`on_gpu` context manager
    constructs a :class:`GpuSlab` on entry, bulk-DMAs the slab to GPU,
    repoints the model's params at the GPU slab views, runs the
    user code, then on exit repoints back to CPU slab views and drops
    the GpuSlab so its GPU storage is released.

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

        # Pack every frozen parameter into one PinnedSlab. Trainable
        # params are skipped — PinnedWeights is for inference / frozen-
        # base flows; use BlockOffloader for training where some params
        # need backward.
        frozen = [(n, p) for n, p in model.named_parameters() if not p.requires_grad]
        if not frozen:
            raise ValueError(
                "PinnedWeights requires at least one frozen parameter to cache. "
                "All params on the wrapped model have requires_grad=True — for "
                "training flows use ltx_core.memory.BlockOffloader instead."
            )
        self._slab: PinnedSlab | None = PinnedSlab(frozen)
        self._slab.install_into_params()

        # Cache (param, qual_name) tuples once so per-call moves don't
        # re-walk named_parameters() and don't re-classify which params
        # are slabbed.
        self._slab_params: list[tuple[nn.Parameter, str]] = [(p, n) for n, p in frozen]

        # Cache buffers individually. Capture each buffer's original
        # ``persistent`` flag so the swap doesn't silently demote it
        # — without this, a persistent buffer would drop out of
        # state_dict() after the first on_gpu() restoration.
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
        total = self._slab.pinned_bytes if self._slab is not None else 0
        for _, _, pinned, _ in self._buffer_pins:
            total += pinned.numel() * pinned.element_size()
        return total

    @contextlib.contextmanager
    def on_gpu(self) -> Iterator[nn.Module]:
        """Bulk-DMA pinned weights to GPU; yield model; restore on exit.

        On entry, a fresh :class:`GpuSlab` is constructed (allocating
        GPU memory equal to the slab size), the pinned slab is bulk-
        copied into it, and the model's frozen params are repointed at
        the GpuSlab views. On exit, params are repointed back at the
        pinned CPU slab views and the GpuSlab is dropped — its GPU
        storage is released by refcount, so memory is freed between
        calls.

        Not re-entrant; nested calls raise ``RuntimeError``.
        """
        if self._active:
            raise RuntimeError(
                "PinnedWeights.on_gpu() is not re-entrant. The wrapped model "
                "is already inside an active on_gpu() context."
            )
        if self._slab is None:
            raise RuntimeError(
                "PinnedWeights has been torn down — pinned buffers are released."
            )
        self._active = True
        # Local — explicitly NOT cached across calls. Caching would keep
        # the GPU allocation resident between on_gpu() invocations,
        # defeating the whole point of evicting a model when it isn't
        # being used.
        gpu_slab: GpuSlab | None = None
        try:
            gpu_slab = GpuSlab(self._slab, self._device)
            self._slab.bulk_to_gpu(gpu_slab, non_blocking=True)
            self._move_buffers_to_gpu()
            if self._device.type == "cuda":
                # Make non-blocking H2D copies visible to subsequent kernels.
                torch.cuda.synchronize(self._device)
            self._install_gpu_params(gpu_slab)
            try:
                yield self._model
            finally:
                self._restore_cpu_params()
                self._move_buffers_to_pinned()
        finally:
            self._active = False
            # Drop the GpuSlab reference; its GPU storage releases when
            # nothing else holds a reference to its tensors. The cached
            # gpu_param references inside the slab are no longer pointed
            # at by the model's _parameters dict (we just restored them
            # to CPU views), so the GPU memory is reclaimed.
            del gpu_slab

    def _install_gpu_params(self, gpu_slab: GpuSlab) -> None:
        for p, qual_name in self._slab_params:
            p.data = gpu_slab.get_view(qual_name)

    def _restore_cpu_params(self) -> None:
        assert self._slab is not None
        for p, qual_name in self._slab_params:
            p.data = self._slab.get_view(qual_name)

    def _move_buffers_to_gpu(self) -> None:
        if not self._include_buffers:
            return
        for parent, leaf, pinned, persistent in self._buffer_pins:
            _set_buffer(parent, leaf, pinned.to(self._device, non_blocking=True), persistent)

    def _move_buffers_to_pinned(self) -> None:
        if not self._include_buffers:
            return
        for parent, leaf, pinned, persistent in self._buffer_pins:
            _set_buffer(parent, leaf, pinned, persistent)

    def teardown(self) -> None:
        """Release pinned CPU buffers. The wrapped model is unusable
        after teardown. Raises if called while inside an ``on_gpu()``
        context — call would leave the model holding GPU tensors with
        no path back."""
        if self._active:
            raise RuntimeError(
                "PinnedWeights.teardown() called while on_gpu() is active - "
                "exit the context first or the model would be left holding "
                "GPU tensors that cannot be restored."
            )
        self._slab = None
        self._slab_params.clear()
        self._buffer_pins.clear()
