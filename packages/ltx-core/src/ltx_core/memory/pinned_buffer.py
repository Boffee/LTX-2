"""Per-parameter pinned-CPU storage primitive.

Internal to the ``block_offload`` subpackage. Shared by
:class:`PinnedWeights` (whole-model bulk pin) and :class:`BlockStreamer`
(per-block streaming). Both consumers reach this through the same
abstraction so the addition of new tensor types only requires writing
a new :class:`TensorAdapter`, not editing the consumers.

Per-parameter mechanics live in the tensor adapter
(:mod:`tensor_adapters` for plain tensors, :mod:`_quanto_adapter` for
quanto). :class:`PinnedParamBuffer` is a thin holder that pairs one
:class:`nn.Parameter` with the adapter that handles its tensor type
plus the pinned-host state that adapter produced.
"""

from __future__ import annotations

import logging
from collections.abc import Hashable
from typing import Any

import torch
from torch import nn

# Importing _quanto_adapter has the side effect of registering
# QuantoAdapter when optimum-quanto is installed, so it precedes the
# RegularAdapter fallback in select_adapter. The import must come after
# tensor_adapters defines register_adapter / select_adapter.
from . import _quanto_adapter  # noqa: F401 (registration side effect)
from .tensor_adapters import TensorAdapter, select_adapter

logger = logging.getLogger(__name__)


def storage_key(t: torch.Tensor) -> tuple[Any, ...]:
    """Identity key for tied-weight detection.

    Two tensors that produce the same key represent the same logical
    tensor backed by the same storage region with the same view layout
    and (for quanto) the same quant metadata; they can be deduplicated
    into a single :class:`PinnedParamBuffer`.

    Used by :class:`~block_offload.PinnedWeights` (for handle-level
    dedup of tied frozen params) and
    :func:`~block_offload.make_block_offloader` (for cross-region
    tied-weight detection across blocks and non-block modules).

    Dispatches to the matching adapter so each tensor type contributes
    its own identity components (regular: storage + view; quanto:
    storage of both inner tensors plus quant metadata).
    """
    return select_adapter(t).storage_key(t)


class PinnedParamBuffer:
    """Pinned host storage for one parameter, with GPU-load helpers.

    Construction picks an adapter via :func:`select_adapter` based on
    the parameter's tensor type, then uses the adapter to clone-and-pin
    the bytes and build the deactivated-state :class:`nn.Parameter`
    (:attr:`cpu_param`).

    The lifecycle methods (:meth:`allocate_gpu_storage`,
    :meth:`make_gpu_param`, :meth:`copy_to_gpu`, :meth:`copy_back`,
    :meth:`load_to_gpu`) all dispatch through the adapter. Consumers
    work with the opaque :class:`GpuState` returned by
    :meth:`allocate_gpu_storage`; the buffer round-trips that opaque
    handle through subsequent calls.

    ``copy_back`` is required when the GPU-side parameter has been
    mutated in place (training step, mutable buffer in train mode) and
    the deactivated host state must reflect those updates before the
    next activate. Callers set ``copy_back=True`` per-buffer at
    construction time when the slot is trainable or otherwise mutates.
    """

    __slots__ = ("adapter", "copy_back_enabled", "cpu_param", "name", "pinned_state")

    def __init__(
        self,
        name: str,
        param: nn.Parameter,
        *,
        copy_back: bool = False,
    ) -> None:
        self.name = name
        self.adapter: type[TensorAdapter] = select_adapter(param.data)
        self.pinned_state = self.adapter.clone_pin(param.data)
        self.cpu_param: nn.Parameter = self.adapter.cpu_param(self.pinned_state)
        self.copy_back_enabled: bool = copy_back

    def allocate_gpu_storage(self, device: torch.device) -> Any:
        """Allocate empty GPU storage mirroring this buffer's layout.
        Returns an opaque adapter-specific handle; pass it back to
        :meth:`make_gpu_param`, :meth:`copy_to_gpu`, and :meth:`copy_back`."""
        return self.adapter.alloc_gpu(self.pinned_state, device)

    def make_gpu_param(self, gpu_state: Any) -> nn.Parameter:
        """Build the GPU-side :class:`nn.Parameter` for this buffer.
        Adapter receives the paired pinned state so structured tensor
        types (quanto) can reconstruct their wrappers."""
        return self.adapter.gpu_param(self.pinned_state, gpu_state)

    def copy_to_gpu(self, gpu_state: Any, *, non_blocking: bool = False) -> None:
        """Bulk DMA pinned host bytes into pre-allocated GPU storage."""
        self.adapter.copy_to_gpu(self.pinned_state, gpu_state, non_blocking=non_blocking)

    def copy_back(self, gpu_state: Any) -> None:
        """Copy live GPU bytes back into pinned host state. No-op when
        :attr:`copy_back_enabled` is False (frozen params don't need it).
        Called on deactivate for buffers wrapping mutable params."""
        if self.copy_back_enabled:
            self.adapter.copy_back(gpu_state, self.pinned_state)

    def load_to_gpu(
        self, device: torch.device, non_blocking: bool = False
    ) -> nn.Parameter:
        """Convenience: allocate GPU storage and copy in one shot.
        Used by the no-pool fallback path; the pooled path uses
        :meth:`allocate_gpu_storage` + :meth:`make_gpu_param` once at
        slot construction and :meth:`copy_to_gpu` on each load."""
        gpu_state = self.allocate_gpu_storage(device)
        self.copy_to_gpu(gpu_state, non_blocking=non_blocking)
        return self.make_gpu_param(gpu_state)

    @property
    def cache_bytes(self) -> int:
        """Bytes this buffer consumes in pinned host memory."""
        return self.adapter.cache_bytes(self.pinned_state)

    @property
    def homogeneity_key(self) -> Hashable:
        """Identity tuple for layout homogeneity checks. Used by
        :class:`BlockStreamer` to verify all blocks share the same
        layout before allocating a single GPU pool slot."""
        return self.adapter.homogeneity_key(self.pinned_state)

    # ------------------------------------------------------------------
    # Backward-compat shims
    # ------------------------------------------------------------------
    # The pre-adapter PinnedParamBuffer exposed a fixed set of fields
    # (pinned_data, pinned_scale, is_quanto, qtype, axis, act_qt, size,
    # stride). Those leaked tensor-type-specific shape into consumers.
    # The adapter-based design replaces them with cache_bytes /
    # homogeneity_key methods. Keep the old names as read-only
    # properties so existing test code and any external probes keep
    # working during migration; remove in a later release.

    @property
    def pinned_data(self) -> torch.Tensor:
        return self.pinned_state.data  # type: ignore[attr-defined]

    @property
    def pinned_scale(self) -> torch.Tensor | None:
        return getattr(self.pinned_state, "scale", None)

    @property
    def is_quanto(self) -> bool:
        return self.adapter.__name__ == "QuantoAdapter"

    @property
    def qtype(self) -> Any:
        return getattr(self.pinned_state, "qtype", None)

    @property
    def axis(self) -> Any:
        return getattr(self.pinned_state, "axis", None)

    @property
    def act_qt(self) -> Any:
        return getattr(self.pinned_state, "act_qt", None)

    @property
    def size(self) -> Any:
        return getattr(self.pinned_state, "size", None)

    @property
    def stride(self) -> Any:
        return getattr(self.pinned_state, "stride", None)
