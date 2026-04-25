"""Pinned CPU buffer for a single frozen parameter — shared between
``streaming`` (per-block streaming) and ``pinned`` (whole-model bulk).

Internal to the ``ltx_core.memory`` subpackage. Not part of the public API,
but lives in its own module so both consumers can reach it without
crossing each other's private namespaces.

Quanto ``WeightQBytesTensor`` is decomposed into its ``_data`` and
``_scale`` components so the pinned buffer holds exactly two contiguous
CPU tensors (which can be DMA'd in one ``copy_()`` each) and the
quantized wrapper is reconstructed on GPU at load time.
"""

from __future__ import annotations

import torch
from torch import nn

_QUANTO_AVAILABLE = False
try:
    from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor

    _QUANTO_AVAILABLE = True
except ImportError:
    pass


class PinnedParamBuffer:
    """Pinned CPU storage for one frozen parameter, with GPU-load helper.

    Construction clones ``param.data`` into pinned CPU memory (decomposing
    quanto ``WeightQBytesTensor`` into ``_data`` + ``_scale`` if present).
    ``cpu_param`` is an ``nn.Parameter`` wrapping the pinned storage —
    callers can repoint a model parameter at it without reallocating.
    ``load_to_gpu(device)`` produces a fresh ``nn.Parameter`` whose
    storage lives on the target device.
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
        # in the block-by-block quantizer), and the default preserve_format
        # would carry that non-contiguity through to pin_memory(), tripping
        # the strict is_contiguous() assert later in PackedSlab._pack.
        # The quanto tensor's own stride is stored separately (self.stride)
        # and re-applied on GPU reconstruction via WeightQBytesTensor.create.
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

    def load_to_gpu(self, device: torch.device, non_blocking: bool = False) -> nn.Parameter:
        if self.is_quanto:
            gd = self.pinned_data.to(device, non_blocking=non_blocking)
            gs = self.pinned_scale.to(device, non_blocking=non_blocking)
            qt = WeightQBytesTensor.create(self.qtype, self.axis, self.size, self.stride, gd, gs, self.act_qt)
            return nn.Parameter(qt, requires_grad=False)
        return nn.Parameter(self.pinned_data.to(device, non_blocking=non_blocking), requires_grad=False)
