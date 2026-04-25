"""Pinned CPU + GPU storage primitives shared between ``streaming``
(per-block streaming) and ``pinned`` (whole-model bulk).

Internal to the ``ltx_core.memory`` subpackage. Not part of the public
API, but lives in its own module so both consumers can reach it without
crossing each other's private namespaces.

Two complementary types:

* :class:`PinnedSlab` — packed pinned-CPU storage for a group of
  frozen ``nn.Parameter`` s, grouped by dtype. Single source of
  truth: ``install_into_params()`` repoints each model param.data
  at an ``as_strided`` view into the slab.
* :class:`GpuSlab` — matching GPU-side storage with the same layout.
  ``PinnedSlab.bulk_to_gpu`` does one ``copy_()`` per dtype group;
  per-param view tensors on the GPU side are built once at
  construction and remain stable across loads.

Quanto ``WeightQBytesTensor`` is decomposed into its ``_data`` and
``_scale`` components so they pack into separate dtype groups; the
quantized wrapper is reconstructed via ``WeightQBytesTensor.create``
around the slab views on both CPU and GPU sides.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
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


# ---------------------------------------------------------------------------
# Packed slab storage with per-param views
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _StorageLayout:
    """Where a single tensor lives within a packed dtype buffer.

    ``offset`` and ``numel`` are in elements (not bytes). ``shape``/
    ``stride`` describe the original tensor layout — used by
    ``as_strided`` to construct the per-param view.
    """

    offset: int
    numel: int
    shape: torch.Size
    stride: tuple[int, ...]


@dataclass(frozen=True)
class _ParamSpec:
    """Reconstruction recipe for one nn.Parameter from slab views.

    Holds the per-tensor layout(s) plus quanto wrapper metadata so the
    same spec works on both the pinned-CPU and GPU side of the slab.
    """

    name: str
    is_quanto: bool
    data_dtype: torch.dtype
    data: _StorageLayout
    # Quanto-only fields:
    scale_dtype: torch.dtype | None = None
    scale: _StorageLayout | None = None
    qtype: Any = None
    axis: int | None = None
    outer_shape: torch.Size | None = None
    outer_stride: tuple[int, ...] | None = None
    activation_qtype: Any = None


def _build_view(buffer: torch.Tensor, layout: _StorageLayout) -> torch.Tensor:
    """Construct a view into ``buffer`` at the given layout via
    ``as_strided``. ``buffer`` must already be 1-D and contiguous;
    the view will reproduce the original tensor's shape and stride."""
    return torch.as_strided(buffer, layout.shape, layout.stride, layout.offset)


def _wrap_quanto(spec: _ParamSpec, data_view: torch.Tensor, scale_view: torch.Tensor) -> "WeightQBytesTensor":
    if not _QUANTO_AVAILABLE:
        raise RuntimeError("quanto wrapper requested but optimum.quanto is not installed")
    return WeightQBytesTensor.create(
        spec.qtype, spec.axis, spec.outer_shape, spec.outer_stride,
        data_view, scale_view, spec.activation_qtype,
    )


def _classify_param(name: str, param: nn.Parameter) -> tuple[_ParamSpec, torch.Tensor, torch.Tensor | None]:
    """Inspect a parameter and produce its (spec, data_tensor, scale_tensor)
    triple. ``spec.data.offset`` is left at 0 — the slab assigns the
    final offset during packing."""
    t = param.data
    if _QUANTO_AVAILABLE and isinstance(t, WeightQBytesTensor):
        d, s = t._data, t._scale
        if not d.is_contiguous() or not s.is_contiguous():
            # We deliberately do NOT silently reshape: the caller must
            # ensure contiguous source storage so as_strided views are
            # valid. fp8-quanto sometimes leaves _data strided; this
            # branch forces contiguous via clone.
            d = d.contiguous()
            s = s.contiguous()
        spec = _ParamSpec(
            name=name,
            is_quanto=True,
            data_dtype=d.dtype,
            data=_StorageLayout(0, d.numel(), d.size(), d.stride()),
            scale_dtype=s.dtype,
            scale=_StorageLayout(0, s.numel(), s.size(), s.stride()),
            qtype=t.qtype,
            axis=t.axis,
            outer_shape=t.size(),
            outer_stride=t.stride(),
            activation_qtype=getattr(t, "activation_qtype", None),
        )
        return spec, d, s
    if not t.is_contiguous():
        t = t.contiguous()
    spec = _ParamSpec(
        name=name,
        is_quanto=False,
        data_dtype=t.dtype,
        data=_StorageLayout(0, t.numel(), t.size(), t.stride()),
    )
    return spec, t, None


class PinnedSlab:
    """Packed pinned-CPU storage for a group of frozen ``nn.Parameter`` s,
    grouped by dtype.

    Construction:
      1. Classify each input param. Quanto params decompose into
         (data, scale) pairs; non-quanto params have only data.
      2. Dedupe by source storage pointer to handle tied/shared params.
      3. Group by dtype across all data + scale tensors.
      4. Allocate one pinned 1-D buffer per dtype, sized to the sum of
         numels in that group.
      5. Pack each tensor end-to-end into its dtype buffer; store the
         final offset in the param's ``_ParamSpec``.

    After construction, each input ``nn.Parameter.data`` can be
    repointed at its slab view via :meth:`install_into_params`. The
    view is an ``as_strided`` window into the slab — zero-copy, byte-
    identical to the original.

    The slab is the sole CPU pinned source. ``bulk_to_gpu`` does one
    ``copy_()`` per dtype buffer, transferring the packed bytes to a
    matching :class:`GpuSlab` in a single bandwidth-bound DMA.
    """

    def __init__(self, named_params: list[tuple[str, nn.Parameter]]) -> None:
        self._params: list[nn.Parameter] = [p for _, p in named_params]
        # Classify and dedupe.
        seen_storage: dict[int, str] = {}  # data_ptr → first-seen name (for tied weights)
        prelim_specs: list[_ParamSpec] = []
        prelim_data_tensors: list[torch.Tensor] = []
        prelim_scale_tensors: list[torch.Tensor | None] = []
        # Map from parameter index → canonical spec name. Parameters that
        # share storage with an already-seen one map back to the original.
        self._alias_to_canonical: dict[int, str] = {}

        for idx, (name, param) in enumerate(named_params):
            spec, data_t, scale_t = _classify_param(name, param)
            data_ptr = data_t.data_ptr()
            if data_ptr in seen_storage:
                # Tied weight: alias to the first occurrence; do NOT pack again.
                self._alias_to_canonical[idx] = seen_storage[data_ptr]
                continue
            seen_storage[data_ptr] = name
            self._alias_to_canonical[idx] = name
            prelim_specs.append(spec)
            prelim_data_tensors.append(data_t)
            prelim_scale_tensors.append(scale_t)

        # Group by dtype across data + scale tensors.
        # Each group becomes one pinned buffer.
        groups: OrderedDict[torch.dtype, list[tuple[_ParamSpec, str, torch.Tensor]]] = OrderedDict()
        # Each entry: (spec, "data" or "scale", source tensor)
        for spec, data_t, scale_t in zip(prelim_specs, prelim_data_tensors, prelim_scale_tensors):
            groups.setdefault(spec.data_dtype, []).append((spec, "data", data_t))
            if spec.is_quanto:
                groups.setdefault(spec.scale_dtype, []).append((spec, "scale", scale_t))

        # Allocate one pinned buffer per dtype group; pack tensors and
        # rewrite spec offsets.
        self._buffers: dict[torch.dtype, torch.Tensor] = {}
        # Final, offset-corrected specs keyed by name.
        self._specs: dict[str, _ParamSpec] = {}
        # Building staged updates so we can replace _ParamSpec immutably.
        spec_updates: dict[str, dict[str, _StorageLayout]] = {}

        for dtype, entries in groups.items():
            total = sum(t.numel() for _, _, t in entries)
            buf = torch.empty(total, dtype=dtype, pin_memory=True)
            self._buffers[dtype] = buf
            offset = 0
            for spec, role, tensor in entries:
                buf.narrow(0, offset, tensor.numel()).copy_(tensor.reshape(-1))
                slot = _StorageLayout(offset, tensor.numel(), tensor.size(), tensor.stride())
                spec_updates.setdefault(spec.name, {})[role] = slot
                offset += tensor.numel()

        # Materialize final _ParamSpec objects with offsets filled in.
        for spec in prelim_specs:
            updates = spec_updates[spec.name]
            kwargs = {
                "name": spec.name,
                "is_quanto": spec.is_quanto,
                "data_dtype": spec.data_dtype,
                "data": updates["data"],
            }
            if spec.is_quanto:
                kwargs.update({
                    "scale_dtype": spec.scale_dtype,
                    "scale": updates["scale"],
                    "qtype": spec.qtype,
                    "axis": spec.axis,
                    "outer_shape": spec.outer_shape,
                    "outer_stride": spec.outer_stride,
                    "activation_qtype": spec.activation_qtype,
                })
            self._specs[spec.name] = _ParamSpec(**kwargs)

        # Pre-build CPU views — avoids re-running as_strided + quanto wrapper
        # on each get_view call. Also wrap each view in a stable
        # ``nn.Parameter`` so callers that assign to ``module._parameters``
        # get a reference that doesn't churn across loads.
        self._cpu_views: dict[str, torch.Tensor] = {}
        self._cpu_params: dict[str, nn.Parameter] = {}
        for name, spec in self._specs.items():
            view = self._build_cpu_view(spec)
            self._cpu_views[name] = view
            self._cpu_params[name] = nn.Parameter(view, requires_grad=False)

    def _build_cpu_view(self, spec: _ParamSpec) -> torch.Tensor:
        data_buf = self._buffers[spec.data_dtype]
        data_view = _build_view(data_buf, spec.data)
        if not spec.is_quanto:
            return data_view
        scale_buf = self._buffers[spec.scale_dtype]  # type: ignore[index]
        scale_view = _build_view(scale_buf, spec.scale)  # type: ignore[arg-type]
        return _wrap_quanto(spec, data_view, scale_view)

    def install_into_params(self) -> None:
        """Repoint each tracked ``nn.Parameter.data`` at its slab view.

        After install, the model's frozen params are backed by the slab
        — they have no independent storage. The model can be used
        (CPU forward) but the bytes live in pinned memory.
        """
        for idx, param in enumerate(self._params):
            canonical = self._alias_to_canonical[idx]
            param.data = self._cpu_views[canonical]

    def bulk_to_gpu(self, gpu_slab: "GpuSlab", non_blocking: bool = True) -> None:
        """One ``copy_()`` per dtype buffer, pinned → GPU. Caller is
        responsible for a CUDA sync if required."""
        if not gpu_slab.is_compatible_with(self):
            raise ValueError("GpuSlab is not layout-compatible with this PinnedSlab")
        for dtype, src in self._buffers.items():
            gpu_slab._buffers[dtype].copy_(src, non_blocking=non_blocking)

    def get_view(self, name: str) -> torch.Tensor:
        """Return the cached CPU view for a named param (or alias)."""
        canonical = self._lookup_canonical(name)
        return self._cpu_views[canonical]

    def get_param(self, name: str) -> nn.Parameter:
        """Return a stable ``nn.Parameter`` wrapping the cached CPU view.

        The Parameter object is built once at slab construction; consumers
        assigning to ``submod._parameters[local_name]`` get a reference
        whose identity persists across loads (required for PEFT and to
        avoid Python ref churn on the hot path)."""
        canonical = self._lookup_canonical(name)
        return self._cpu_params[canonical]

    def _lookup_canonical(self, name: str) -> str:
        if name in self._cpu_views:
            return name
        for idx, p in enumerate(self._params):
            # Slow path; only hit on aliases looked up by alias name.
            if hasattr(p, "_param_name") and p._param_name == name:  # type: ignore[attr-defined]
                return self._alias_to_canonical[idx]
        raise KeyError(name)

    @property
    def specs(self) -> dict[str, _ParamSpec]:
        return self._specs

    @property
    def buffers(self) -> dict[torch.dtype, torch.Tensor]:
        return self._buffers

    @property
    def pinned_bytes(self) -> int:
        return sum(b.numel() * b.element_size() for b in self._buffers.values())

    def install_canonical_into(self, target_param: nn.Parameter, source_name: str) -> None:
        """Set ``target_param.data`` to the slab's view for ``source_name``.
        Used by callers that have a fresh ``nn.Parameter`` (e.g. PEFT-
        wrapped) and want it backed by this slab."""
        target_param.data = self._cpu_views[source_name]


class GpuSlab:
    """Packed GPU storage matching a :class:`PinnedSlab`'s layout.

    Allocates one GPU buffer per dtype the template has, mirroring the
    pinned slab's offsets exactly. Per-param view tensors (and quanto
    wrappers) are built ONCE at construction and cached. Subsequent
    ``PinnedSlab.bulk_to_gpu`` calls overwrite the underlying storage
    via ``copy_()``; the cached view tensors continue to point at the
    same GPU memory and remain valid references.

    Compatibility check (`is_compatible_with`) verifies a candidate
    pinned slab matches this GPU slab's layout exactly — same dtype
    buffers, same per-param specs (including quanto qtype/axis/scale
    layout). Required before reusing a GPU slot for a different
    block in a streaming pool.
    """

    def __init__(self, template: PinnedSlab, device: torch.device) -> None:
        self._device = device
        self._buffers: dict[torch.dtype, torch.Tensor] = {}
        for dtype, src in template.buffers.items():
            self._buffers[dtype] = torch.empty(src.numel(), dtype=dtype, device=device)
        self._template_specs: dict[str, _ParamSpec] = template.specs
        self._gpu_views: dict[str, torch.Tensor] = {}
        self._gpu_params: dict[str, nn.Parameter] = {}
        for name, spec in template.specs.items():
            view = self._build_gpu_view(spec)
            self._gpu_views[name] = view
            self._gpu_params[name] = nn.Parameter(view, requires_grad=False)

    def _build_gpu_view(self, spec: _ParamSpec) -> torch.Tensor:
        data_buf = self._buffers[spec.data_dtype]
        data_view = _build_view(data_buf, spec.data)
        if not spec.is_quanto:
            return data_view
        scale_buf = self._buffers[spec.scale_dtype]  # type: ignore[index]
        scale_view = _build_view(scale_buf, spec.scale)  # type: ignore[arg-type]
        return _wrap_quanto(spec, data_view, scale_view)

    def get_view(self, name: str) -> torch.Tensor:
        """Return the stable cached GPU view tensor for a named param.

        The returned reference is reusable across many ``bulk_to_gpu``
        loads — only the underlying storage bytes change."""
        return self._gpu_views[name]

    def get_param(self, name: str) -> nn.Parameter:
        """Return a stable ``nn.Parameter`` wrapping the cached GPU view.

        Used by :class:`BlockOffloader` to populate
        ``submod._parameters[local_name]`` without churning Parameter
        identity across block loads."""
        return self._gpu_params[name]

    def is_compatible_with(self, pinned: PinnedSlab) -> bool:
        """Strict layout check: same dtype set, same per-param specs.

        Two slabs are compatible iff their dtype groups have the same
        sizes and every param spec matches byte-for-byte. Required
        before bulk-copying ``pinned`` into ``self``."""
        if set(self._buffers) != set(pinned.buffers):
            return False
        for dtype, src in pinned.buffers.items():
            if self._buffers[dtype].numel() != src.numel():
                return False
        if set(self._template_specs) != set(pinned.specs):
            return False
        for name, spec in pinned.specs.items():
            if self._template_specs[name] != spec:
                return False
        return True

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def buffers(self) -> dict[torch.dtype, torch.Tensor]:
        return self._buffers


def make_named_params(named_params: Iterable[tuple[str, nn.Parameter]]) -> list[tuple[str, nn.Parameter]]:
    """Helper: materialize an iterable of (name, param) tuples,
    filtering out ``requires_grad=True`` entries (they are not slab-
    eligible — slabs only handle frozen weights)."""
    return [(n, p) for n, p in named_params if not p.requires_grad]
