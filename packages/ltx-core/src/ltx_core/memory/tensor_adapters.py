"""Tensor-type adapters: per-type pin/move/wrap mechanics.

Different tensor subclasses need different machinery to move bytes
across the CPU↔GPU boundary while preserving correctness:

- Plain ``torch.Tensor`` (bf16/fp16/fp32): single contiguous storage,
  ``p.data = ...`` swap preserves :class:`nn.Parameter` identity, optimizer-safe.
- Quanto ``WeightQBytesTensor``: two pinned tensors (``_data`` + ``_scale``)
  plus quant metadata; the wrapper must be reconstructed on each move and
  installed via slot replacement (``module._parameters[leaf] = new_param``).

Each adapter encapsulates the mechanics for one tensor type. The rest
of the package (:class:`PinnedParamBuffer`, :class:`PinnedWeights`,
:class:`BlockStreamer`) is type-agnostic and dispatches through
:func:`select_adapter`.

This module is internal to :mod:`block_offload`. Adapters are registered
at module import time; new types can be added by writing a new adapter
class and calling :func:`register_adapter`.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, Protocol, TypeVar, runtime_checkable

import torch
from torch import nn

__all__ = [
    "ParamIdentity",
    "TensorAdapter",
    "register_adapter",
    "select_adapter",
]

# Adapter-specific opaque state types. The Protocol is generic over
# them so consumers (PinnedParamBuffer) can stay tensor-type-agnostic
# while each adapter pins its own concrete state shape.
PinnedStateT = TypeVar("PinnedStateT")
GpuStateT = TypeVar("GpuStateT")


ParamIdentity = Literal["original", "stable_replacement", "ephemeral_replacement"]
"""How an adapter handles :class:`nn.Parameter` identity across moves.

- ``"original"``: ``p.data`` swap, same Parameter object survives every cycle.
  Optimizer-safe — references held by ``optim.state`` stay valid.
- ``"stable_replacement"``: a fresh Parameter is installed on the slot, but
  the same object is reused across activate/deactivate cycles (e.g. one
  ``cpu_param`` for the deactivated state, one for the activated).
  Optimizer state attached at first install survives subsequent cycles.
- ``"ephemeral_replacement"``: a brand-new Parameter is constructed every
  activate. Optimizer state keyed by the old object is orphaned.
  Inference-only.
"""


@runtime_checkable
class TensorAdapter(Protocol[PinnedStateT, GpuStateT]):
    """Adapter encoding the mechanics of pinning, moving, and wrapping
    one tensor type. Adapters are stateless; they hold no per-param data.

    Each adapter declares its capabilities via class attributes
    (:attr:`param_identity`, :attr:`uses_pinned_host`,
    :attr:`supports_trainable`, etc.) so the composer can validate
    compatibility (e.g., reject ``trainable + ephemeral_replacement``).

    Generic over two opaque state types: ``PinnedStateT`` (the pinned
    host representation) and ``GpuStateT`` (the GPU storage). Each
    adapter pins these to its own concrete dataclasses; consumers
    round-trip the opaque types without inspecting them.
    """

    # ------- Capability flags -------

    param_identity: ClassVar[ParamIdentity]
    """How activate/deactivate handles Parameter identity. See :data:`ParamIdentity`."""

    uses_pinned_host: ClassVar[bool]
    """True if :meth:`clone_pin` allocates page-locked host memory.
    False for adapters that just hold the tensor in regular CPU memory."""

    supports_trainable: ClassVar[bool]
    """True if this adapter can host trainable params correctly across
    activate/deactivate cycles. Implies ``param_identity == "original"``
    AND that ``copy_back`` round-trips in-place updates AND that the
    user's pre-wrap Parameter object survives in-model. Currently no
    adapter supports trainable through :class:`PinnedParamBuffer` — the
    buffer's slot-replacement mechanism orphans the original Parameter.
    This flag exists for future identity-preserving paths and for
    composer-side compatibility checks."""

    moves_grad: ClassVar[bool]
    """True if the adapter handles ``param.grad`` migration alongside
    ``param.data``. Implementations that don't manage grads leave
    ``.grad`` as the user found it (typically GPU after backward)."""

    is_quanto: ClassVar[bool]
    """True for :class:`QuantoAdapter`, False for everyone else.
    Lets compat shims and adapter-specific code paths branch on
    quanto-ness without string comparison on adapter class names."""

    # ------- Per-tensor methods (stateless) -------

    @staticmethod
    def matches(t: torch.Tensor) -> bool:
        """True if this adapter handles tensor ``t``. Used by
        :func:`select_adapter` for dispatch. Implementations should be
        conservative — :class:`RegularAdapter` matches only plain
        ``torch.Tensor``, not unrecognized subclasses."""
        ...

    @staticmethod
    def storage_key(t: torch.Tensor) -> tuple:
        """Composite identity key for tied-weight detection. Two tensors
        with the same key share storage and quant metadata; different
        keys must not be deduped. Includes view layout (shape/stride/
        offset) so distinct views into the same buffer don't collapse."""
        ...

    @staticmethod
    def clone_pin(t: torch.Tensor) -> PinnedStateT:
        """Clone ``t`` into pinned (or regular) host memory. Returns
        opaque adapter-specific state used by subsequent operations."""
        ...

    @staticmethod
    def cpu_param(state: PinnedStateT) -> nn.Parameter:
        """Build a stable :class:`nn.Parameter` wrapping the host state.
        Used as the deactivated-state slot value
        (``module._parameters[leaf] = cpu_param``)."""
        ...

    @staticmethod
    def alloc_gpu(state: PinnedStateT, device: torch.device) -> GpuStateT:
        """Allocate empty GPU storage mirroring this state's layout.
        Returns opaque adapter-specific state."""
        ...

    @staticmethod
    def gpu_param(pinned: PinnedStateT, gpu_state: GpuStateT) -> nn.Parameter:
        """Build a stable :class:`nn.Parameter` wrapping the GPU state.
        Reused across many :meth:`copy_to_gpu` calls.

        Takes both the pinned host state and the GPU state because
        adapters with structured tensors (e.g. quanto) need metadata
        captured at pin time to reconstruct the GPU-side wrapper. Plain
        adapters ignore ``pinned``."""
        ...

    @staticmethod
    def copy_to_gpu(
        src: PinnedStateT, dst: GpuStateT, *, non_blocking: bool = False
    ) -> None:
        """Bulk DMA the pinned state's bytes into pre-allocated GPU storage."""
        ...

    @staticmethod
    def copy_back(src: GpuStateT, dst: PinnedStateT) -> None:
        """Copy live GPU bytes back into the pinned host state. Required
        when the model has mutated ``p.data`` on GPU (training step) and
        the deactivated state must reflect those updates."""
        ...

    @staticmethod
    def cache_bytes(state: PinnedStateT) -> int:
        """Total bytes this state consumes in host memory. Used by
        :class:`ModelCache` for budget accounting."""
        ...

    @staticmethod
    def homogeneity_key(state: PinnedStateT) -> Hashable:
        """Identity used to test that a list of states is
        layout-homogeneous (same dtype/shape/stride/quant-metadata).
        Required by :class:`BlockStreamer`'s GPU pool, which preallocates
        slots assuming all blocks share the same layout. Returns any
        hashable value — typically a tuple of layout components."""
        ...


# ---------------------------------------------------------------------------
# RegularAdapter — plain torch.Tensor (bf16/fp16/fp32, etc.)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _RegularPinned:
    """Pinned-CPU state for a regular tensor: one contiguous host buffer."""

    data: torch.Tensor


@dataclass(slots=True)
class _RegularGpu:
    """GPU state for a regular tensor: one contiguous device buffer."""

    data: torch.Tensor


class RegularAdapter:
    """Adapter for plain ``torch.Tensor`` (no subclass machinery).

    Builds fresh stable :class:`nn.Parameter` objects for the deactivated
    (pinned-CPU) and activated (GPU) states. Consumers slot-replace via
    ``module._parameters[leaf] = ...``; the same two Parameter objects
    are reused across all cycles, so this is ``stable_replacement``
    identity — not ``original``. PyTorch optimizers keyed by the user's
    *pre-wrap* Parameter become orphaned once :class:`PinnedParamBuffer`
    installs ``cpu_param`` in the slot. **This adapter is frozen-only.**

    A future identity-preserving path (true ``p.data = ...`` swap on the
    user's original Parameter, no slot replacement) would let
    :class:`PinnedWeights` handle trainables; until then, trainable
    params should be excluded from PinnedWeights via ``skip_slots``.

    Conservative on dispatch: only matches exactly
    ``type(t) is torch.Tensor`` (or ``nn.Parameter``). Unrecognized
    tensor subclasses fall through to other adapters or raise via
    :func:`select_adapter`.
    """

    param_identity: ClassVar[ParamIdentity] = "stable_replacement"
    uses_pinned_host: ClassVar[bool] = True
    supports_trainable: ClassVar[bool] = False
    moves_grad: ClassVar[bool] = False
    is_quanto: ClassVar[bool] = False

    @staticmethod
    def matches(t: torch.Tensor) -> bool:
        # Strict identity match on the base class. PEFT, FSDP, quanto,
        # DTensor, etc. are subclasses with extra state; a silent fallback
        # to RegularAdapter would clone-and-dequantize quanto or break
        # distributed placement. Each subclass needs its own adapter.
        return type(t) is torch.Tensor or type(t) is nn.Parameter

    @staticmethod
    def storage_key(t: torch.Tensor) -> tuple:
        return (
            "regular",
            t.data_ptr(),
            t.dtype,
            tuple(t.shape),
            t.stride(),
            t.storage_offset(),
        )

    @staticmethod
    def clone_pin(t: torch.Tensor) -> _RegularPinned:
        return _RegularPinned(
            data=t.data.clone(memory_format=torch.contiguous_format).pin_memory()
        )

    @staticmethod
    def cpu_param(state: _RegularPinned) -> nn.Parameter:
        return nn.Parameter(state.data, requires_grad=False)

    @staticmethod
    def alloc_gpu(state: _RegularPinned, device: torch.device) -> _RegularGpu:
        return _RegularGpu(data=torch.empty_like(state.data, device=device))

    @staticmethod
    def gpu_param(pinned: _RegularPinned, gpu_state: _RegularGpu) -> nn.Parameter:  # noqa: ARG004
        # pinned unused: regular tensors carry no metadata beyond storage.
        # Argument kept for Protocol parity with QuantoAdapter, which needs it.
        return nn.Parameter(gpu_state.data, requires_grad=False)

    @staticmethod
    def copy_to_gpu(
        src: _RegularPinned, dst: _RegularGpu, *, non_blocking: bool = False
    ) -> None:
        dst.data.copy_(src.data, non_blocking=non_blocking)

    @staticmethod
    def copy_back(gpu_state: _RegularGpu, dst: _RegularPinned) -> None:
        # Blocking copy: callers run this on deactivate, where they
        # need the host buffer up-to-date before the next activate.
        dst.data.copy_(gpu_state.data, non_blocking=False)

    @staticmethod
    def cache_bytes(state: _RegularPinned) -> int:
        return state.data.numel() * state.data.element_size()

    @staticmethod
    def homogeneity_key(state: _RegularPinned) -> tuple:
        return (state.data.dtype, tuple(state.data.shape), state.data.stride())


# ---------------------------------------------------------------------------
# Adapter registry / dispatch
# ---------------------------------------------------------------------------

# Adapters are tried in registration order. The first whose ``matches()``
# returns True wins. RegularAdapter is appended last as the conservative
# fallback for plain tensors. Subclass adapters (quanto, etc.) register
# themselves at import time and slot in front.
_ADAPTERS: list[type[TensorAdapter[Any, Any]]] = []


def register_adapter(adapter: type[TensorAdapter[Any, Any]]) -> None:
    """Register an adapter for use by :func:`select_adapter`. Adapters
    registered later take priority over earlier ones for ``matches()``
    dispatch — this lets specialized adapters (quanto, FP8 variants)
    precede :class:`RegularAdapter`."""
    if adapter not in _ADAPTERS:
        _ADAPTERS.insert(0, adapter)


def select_adapter(t: torch.Tensor) -> type[TensorAdapter[Any, Any]]:
    """Find the registered adapter that handles tensor ``t``.

    Tries adapters in reverse registration order (newest first), returning
    the first whose :meth:`TensorAdapter.matches` returns True. Raises
    :class:`NotImplementedError` if no adapter matches — the alpha library
    refuses to silently dequantize or otherwise mishandle unknown tensor
    subclasses.
    """
    for adapter in _ADAPTERS:
        if adapter.matches(t):
            return adapter
    raise NotImplementedError(
        f"No registered TensorAdapter for tensor type {type(t).__name__!r}. "
        f"Plain tensors are handled by RegularAdapter; tensor subclasses "
        f"need a dedicated adapter (see optimum.quanto integration in "
        f"_quanto_adapter.py for an example)."
    )


# Register the regular fallback last so subclass adapters take priority.
register_adapter(RegularAdapter)
