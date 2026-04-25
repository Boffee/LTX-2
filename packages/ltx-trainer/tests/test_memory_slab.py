"""Tests for ``ltx_core.memory._buffers`` slab abstractions.

Lives in ltx-trainer/tests because that's where the project's pytest
infrastructure currently sits; ltx-core itself has no tests directory.
The tests exercise ltx-core types directly.

Phase 1 scope: PinnedSlab + GpuSlab work correctly in isolation. Tests
do NOT yet exercise BlockOffloader or PinnedWeights against the new
types — those migrations come in Phase 2/3.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from ltx_core.memory._buffers import (
    GpuSlab,
    PinnedSlab,
    _ParamSpec,
    _StorageLayout,
    make_named_params,
)


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


# ---------------------------------------------------------------------------
# PinnedSlab basic correctness
# ---------------------------------------------------------------------------


def _toy_module() -> nn.Module:
    m = nn.Sequential(
        nn.Linear(8, 16),
        nn.LayerNorm(16),
        nn.Linear(16, 4),
    ).cpu()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


class TestPinnedSlab:
    def test_packs_single_dtype(self) -> None:
        m = _toy_module()
        named = list(m.named_parameters())
        slab = PinnedSlab(named)
        assert set(slab.buffers) == {torch.float32}
        # All param numels packed into one buffer.
        expected = sum(p.numel() for _, p in named)
        assert slab.buffers[torch.float32].numel() == expected
        # Pinned bytes match the buffer.
        assert slab.pinned_bytes == expected * 4

    def test_install_repoints_param_data(self) -> None:
        m = _toy_module()
        named = list(m.named_parameters())
        # Snapshot param values before install.
        before = {n: p.data.clone() for n, p in named}
        slab = PinnedSlab(named)
        slab.install_into_params()
        # Each param.data should now be a view into the slab — same values,
        # but storage shared with the slab buffer.
        slab_buf = slab.buffers[torch.float32]
        for n, p in named:
            assert torch.equal(p.data, before[n]), f"{n} value drift after install"
            assert p.data.untyped_storage().data_ptr() == slab_buf.untyped_storage().data_ptr(), (
                f"{n} storage not aliased to slab"
            )
            assert p.data.shape == before[n].shape
            assert p.data.stride() == before[n].stride()

    def test_view_shape_and_stride_preserved(self) -> None:
        m = _toy_module()
        slab = PinnedSlab(list(m.named_parameters()))
        for name, spec in slab.specs.items():
            view = slab.get_view(name)
            assert view.shape == spec.data.shape
            assert view.stride() == spec.data.stride

    def test_dtype_grouping(self) -> None:
        # Mixed bf16 + fp32: one buffer per dtype.
        p1 = nn.Parameter(torch.randn(8, dtype=torch.bfloat16), requires_grad=False)
        p2 = nn.Parameter(torch.randn(16, dtype=torch.float32), requires_grad=False)
        slab = PinnedSlab([("a", p1), ("b", p2)])
        assert set(slab.buffers) == {torch.bfloat16, torch.float32}
        assert slab.buffers[torch.bfloat16].numel() == 8
        assert slab.buffers[torch.float32].numel() == 16

    def test_tied_params_share_one_slab_slot(self) -> None:
        # Two nn.Parameters wrapping the same storage (tied weights).
        shared = torch.randn(4, 8)
        p1 = nn.Parameter(shared, requires_grad=False)
        p2 = nn.Parameter(shared, requires_grad=False)
        slab = PinnedSlab([("a", p1), ("b", p2)])
        # Only one entry packed (the second is an alias of the first).
        assert len(slab.specs) == 1
        # Buffer holds 32 elements, not 64 — the tie was deduped.
        assert slab.buffers[torch.float32].numel() == 32
        # After install, both params share the same slab view.
        slab.install_into_params()
        assert p1.data.untyped_storage().data_ptr() == p2.data.untyped_storage().data_ptr()
        assert p1.data.data_ptr() == p2.data.data_ptr()

    def test_forward_after_install(self) -> None:
        m = _toy_module()
        named = list(m.named_parameters())
        slab = PinnedSlab(named)
        # Compute reference output BEFORE install (using original param storage).
        x = torch.randn(2, 8)
        ref = m(x).clone()
        slab.install_into_params()
        # After install, forward should produce the same output (same bytes,
        # different storage location).
        out = m(x)
        assert torch.equal(ref, out), "forward after install must be byte-identical"


# ---------------------------------------------------------------------------
# GpuSlab + bulk transfer
# ---------------------------------------------------------------------------


@CUDA
class TestGpuSlab:
    def test_construction_mirrors_pinned_layout(self) -> None:
        m = _toy_module()
        pinned = PinnedSlab(list(m.named_parameters()))
        gpu = GpuSlab(pinned, torch.device("cuda"))
        # Same dtype set, matching numels per group.
        assert set(gpu.buffers) == set(pinned.buffers)
        for dtype, b in pinned.buffers.items():
            assert gpu.buffers[dtype].numel() == b.numel()
            assert gpu.buffers[dtype].is_cuda

    def test_bulk_copy_matches_per_tensor(self) -> None:
        # Compare slab bulk-copy result to per-tensor .to(cuda) for byte equality.
        m = _toy_module()
        named = list(m.named_parameters())
        pinned = PinnedSlab(named)
        pinned.install_into_params()
        gpu = GpuSlab(pinned, torch.device("cuda"))
        pinned.bulk_to_gpu(gpu)
        torch.cuda.synchronize()
        for name, _ in named:
            ref = pinned.get_view(name).to("cuda")
            actual = gpu.get_view(name)
            assert torch.equal(ref, actual), f"{name} bytes differ after bulk DMA"

    def test_views_stable_across_loads(self) -> None:
        m = _toy_module()
        pinned = PinnedSlab(list(m.named_parameters()))
        pinned.install_into_params()
        gpu = GpuSlab(pinned, torch.device("cuda"))

        pinned.bulk_to_gpu(gpu)
        first_views = {n: gpu.get_view(n) for n, _ in m.named_parameters()}

        # Re-do the bulk copy. Views returned should be the SAME tensor
        # objects (identity), with the same storage pointer — only the
        # bytes have been overwritten in place.
        pinned.bulk_to_gpu(gpu)
        for n, v in first_views.items():
            again = gpu.get_view(n)
            assert again is v, f"{n} GPU view churned across loads"

    def test_compat_check_identifies_layout_mismatch(self) -> None:
        m1 = _toy_module()
        m2 = nn.Sequential(nn.Linear(8, 32), nn.LayerNorm(32)).cpu()
        for p in m2.parameters():
            p.requires_grad_(False)
        s1 = PinnedSlab(list(m1.named_parameters()))
        s2 = PinnedSlab(list(m2.named_parameters()))
        gpu = GpuSlab(s1, torch.device("cuda"))
        assert gpu.is_compatible_with(s1)
        assert not gpu.is_compatible_with(s2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_make_named_params_filters_trainable(self) -> None:
        m = nn.Linear(4, 4)  # default requires_grad=True
        for p in m.parameters():
            p.requires_grad_(True)
        named = make_named_params(m.named_parameters())
        assert named == []

        # Half frozen
        m.weight.requires_grad_(False)
        named = make_named_params(m.named_parameters())
        assert [n for n, _ in named] == ["weight"]
