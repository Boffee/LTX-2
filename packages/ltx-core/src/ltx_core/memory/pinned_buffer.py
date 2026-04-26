"""Pinned-CPU storage primitive shared between ``block_offloader``
(per-block streaming) and ``pinned_weights`` (whole-model bulk).

Internal to the ``ltx_core.memory`` subpackage. Not part of the public
API, but lives in its own module so both consumers can reach it without
crossing each other's private namespaces.

Per-parameter pinned-CPU storage with optional quanto
``WeightQBytesTensor`` decomposition. ``PinnedParamBuffer`` clones each
frozen parameter's data into pinned CPU memory so subsequent CPU→GPU
transfers run as ``cuMemcpyAsync`` with full PCIe bandwidth.

Quanto note: a naive ``param.data.clone()`` on a ``WeightQBytesTensor``
falls back to the dispatch's plain-tensor handler, which silently
*dequantizes* the value to bf16/fp32. ``PinnedParamBuffer`` detects
quanto explicitly and decomposes into ``_data`` (int8/fp8) + ``_scale``
(fp16/fp32), pinning each separately and reconstructing the quantized
wrapper around the GPU tensors at load time.
"""

from __future__ import annotations

import logging
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


def storage_key(t: torch.Tensor) -> tuple[Any, ...]:
    """Identity key for tied-weight detection.

    Two tensors that produce the same key represent the same logical
    tensor backed by the same storage region with the same view layout
    and (for quanto) the same quant metadata; they can be deduplicated
    into a single :class:`PinnedParamBuffer`.

    Used by both :class:`~ltx_core.memory.PinnedWeights` (for handle-
    level dedup of tied frozen params) and
    :class:`~ltx_core.memory.BlockOffloader` (for cross-region tied-
    weight detection across blocks and non-block modules).

    Note: pure storage identity is not sufficient — two views into the
    same parent tensor with different shape/stride/offset must not
    dedup. The key incorporates view layout for that reason.
    """
    if _QUANTO_AVAILABLE and isinstance(t, WeightQBytesTensor):
        # NOTE: relies on the internal structure of WeightQBytesTensor
        # (`_data`, `_scale`, `qtype`, `axis`, `activation_qtype`) which
        # quanto does not expose as a public API. Pinned to the
        # WeightQBytesTensor layout in optimum-quanto as of the
        # version this repo depends on; if quanto refactors the
        # wrapper class this function will break at runtime — start
        # debugging here.
        return (
            "quanto",
            t._data.data_ptr(),
            t._data.dtype,
            tuple(t._data.shape),
            t._data.stride(),
            t._data.storage_offset(),
            t._scale.data_ptr(),
            t._scale.dtype,
            tuple(t._scale.shape),
            t._scale.stride(),
            t._scale.storage_offset(),
            t.qtype,
            t.axis,
            tuple(t.size()),
            t.stride(),
            getattr(t, "activation_qtype", None),
        )
    return (
        "plain",
        t.data_ptr(),
        t.dtype,
        tuple(t.shape),
        t.stride(),
        t.storage_offset(),
    )


class PinnedParamBuffer:
    """Pinned CPU storage for one frozen parameter, with GPU-load helpers.

    Construction clones ``param.data`` into pinned CPU memory. For
    quanto ``WeightQBytesTensor`` the inner ``_data`` and ``_scale`` are
    decomposed and pinned separately; the quantized wrapper is
    reconstructed from these tensors on demand.

    ``cpu_param`` is a stable ``nn.Parameter`` wrapping the pinned
    storage (or a quanto wrapper around it). Callers can repoint a
    model parameter at it via ``module._parameters[name] = buf.cpu_param``
    without churning Parameter identity across loads.

    ``allocate_gpu_storage(device)`` returns a tuple of empty GPU
    tensors mirroring this buffer's layout — used by GPU pool slots
    to pre-allocate. ``copy_to_gpu(...)`` then writes the pinned bytes
    into those GPU tensors via in-place ``copy_()`` (one or two calls
    per param depending on quanto).
    """

    __slots__ = (
        "act_qt", "axis", "cpu_param", "is_quanto", "name",
        "pinned_data", "pinned_scale", "qtype", "size", "stride",
    )

    def __init__(self, name: str, param: nn.Parameter) -> None:
        self.name = name
        t = param.data
        # Force contiguous_format on the clone: fp8-quanto leaves some layers'
        # internal _data buffers strided (likely via an internal transpose/view
        # in the block-by-block quantizer). The default preserve_format would
        # carry that non-contiguity through to pin_memory(), which can break
        # downstream uses that assume a 1-D contiguous pinned buffer. The
        # quanto tensor's own stride is stored separately (self.stride) and
        # re-applied on GPU reconstruction via WeightQBytesTensor.create.
        if _QUANTO_AVAILABLE and isinstance(t, WeightQBytesTensor):
            self.is_quanto = True
            self.pinned_data = t._data.clone(memory_format=torch.contiguous_format).pin_memory()
            self.pinned_scale = t._scale.clone(memory_format=torch.contiguous_format).pin_memory()
            self.qtype = t.qtype
            self.axis = t.axis
            self.size = t.size()
            self.stride = t.stride()
            self.act_qt = getattr(t, "activation_qtype", None)
            qt = WeightQBytesTensor.create(
                self.qtype, self.axis, self.size, self.stride,
                self.pinned_data, self.pinned_scale, self.act_qt,
            )
            self.cpu_param = nn.Parameter(qt, requires_grad=False)
        else:
            self.is_quanto = False
            self.pinned_data = t.data.clone(memory_format=torch.contiguous_format).pin_memory()
            self.pinned_scale = None
            self.qtype = self.axis = self.size = self.stride = self.act_qt = None
            self.cpu_param = nn.Parameter(self.pinned_data, requires_grad=False)

    def allocate_gpu_storage(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Allocate empty GPU tensors mirroring this param's pinned layout.
        Returns ``(gpu_data, gpu_scale_or_None)`` — the caller is
        responsible for filling them via ``copy_to_gpu()`` on each load."""
        gpu_data = torch.empty_like(self.pinned_data, device=device)
        gpu_scale = torch.empty_like(self.pinned_scale, device=device) if self.is_quanto else None
        return gpu_data, gpu_scale

    def make_gpu_param(self, gpu_data: torch.Tensor, gpu_scale: torch.Tensor | None) -> nn.Parameter:
        """Build a stable ``nn.Parameter`` wrapping the given GPU tensors,
        reconstructing the quanto wrapper if applicable. Called once per
        slot at slot construction; the returned Parameter is reused
        across many ``copy_to_gpu`` calls."""
        if self.is_quanto:
            assert gpu_scale is not None
            qt = WeightQBytesTensor.create(
                self.qtype, self.axis, self.size, self.stride, gpu_data, gpu_scale, self.act_qt,
            )
            return nn.Parameter(qt, requires_grad=False)
        return nn.Parameter(gpu_data, requires_grad=False)

    def copy_to_gpu(
        self,
        gpu_data: torch.Tensor,
        gpu_scale: torch.Tensor | None,
        non_blocking: bool = False,
    ) -> None:
        """Write the pinned bytes into pre-allocated GPU storage in place
        (1 ``copy_()`` for non-quanto, 2 for quanto)."""
        gpu_data.copy_(self.pinned_data, non_blocking=non_blocking)
        if self.is_quanto:
            assert gpu_scale is not None
            gpu_scale.copy_(self.pinned_scale, non_blocking=non_blocking)

    def make_meta_param(self) -> nn.Parameter:
        """Build a ``nn.Parameter`` on the ``meta`` device with this
        buffer's shape/dtype/quanto layout. Used by
        ``PinnedWeights.close()`` to surgically replace managed slots
        (matching ``model.to("meta")`` semantics) without touching
        slots owned by other strategies in composed setups."""
        meta_data = torch.empty_like(self.pinned_data, device="meta")
        meta_scale = (
            torch.empty_like(self.pinned_scale, device="meta")
            if self.is_quanto
            else None
        )
        return self.make_gpu_param(meta_data, meta_scale)

    def load_to_gpu(self, device: torch.device, non_blocking: bool = False) -> nn.Parameter:
        """Convenience: allocate GPU storage and copy in one shot.
        Used by the no-pool fallback path; the pooled path uses
        :meth:`allocate_gpu_storage` + :meth:`make_gpu_param` once at
        slot construction and :meth:`copy_to_gpu` on each load."""
        gpu_data, gpu_scale = self.allocate_gpu_storage(device)
        self.copy_to_gpu(gpu_data, gpu_scale, non_blocking=non_blocking)
        return self.make_gpu_param(gpu_data, gpu_scale)
