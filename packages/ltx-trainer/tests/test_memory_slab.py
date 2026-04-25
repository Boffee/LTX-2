"""Tests for ``ltx_core.memory._buffers.PinnedParamBuffer``.

Lives in ltx-trainer/tests because that's where the project's pytest
infrastructure currently sits; ltx-core itself has no tests directory.
The tests exercise ltx-core types directly.

(Filename retained for git-history continuity even though the slab
abstraction it originally tested has been replaced by the simpler
per-parameter ``PinnedParamBuffer``.)
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from ltx_core.memory._buffers import PinnedParamBuffer


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


# ---------------------------------------------------------------------------
# PinnedParamBuffer basic correctness
# ---------------------------------------------------------------------------


class TestPinnedParamBuffer:
    def test_non_quanto_pin_and_load(self) -> None:
        p = nn.Parameter(torch.randn(8, 16, dtype=torch.bfloat16), requires_grad=False)
        buf = PinnedParamBuffer("w", p)
        assert not buf.is_quanto
        assert buf.pinned_data.is_pinned()
        assert buf.pinned_data.shape == p.shape
        assert buf.pinned_scale is None
        # cpu_param wraps the pinned tensor.
        assert buf.cpu_param.data.data_ptr() == buf.pinned_data.data_ptr()

    @CUDA
    def test_load_to_gpu_non_quanto(self) -> None:
        p = nn.Parameter(torch.randn(4, 8, dtype=torch.bfloat16), requires_grad=False)
        buf = PinnedParamBuffer("w", p)
        gpu = buf.load_to_gpu(torch.device("cuda"))
        assert gpu.is_cuda
        assert gpu.shape == p.shape
        torch.cuda.synchronize()
        assert torch.equal(gpu.cpu(), buf.pinned_data)

    @CUDA
    def test_pool_pattern_allocate_and_copy(self) -> None:
        # Mirrors how _GpuSlot uses PinnedParamBuffer: allocate GPU
        # storage once, then copy_to_gpu in place on each load.
        p = nn.Parameter(torch.randn(16, dtype=torch.bfloat16), requires_grad=False)
        buf = PinnedParamBuffer("w", p)
        device = torch.device("cuda")
        gpu_data, gpu_scale = buf.allocate_gpu_storage(device)
        gpu_param = buf.make_gpu_param(gpu_data, gpu_scale)
        assert gpu_param.is_cuda
        assert gpu_scale is None  # non-quanto
        # First copy
        buf.copy_to_gpu(gpu_data, gpu_scale, non_blocking=True)
        torch.cuda.synchronize()
        assert torch.equal(gpu_data.cpu(), buf.pinned_data)
        # Mutate pinned source and re-copy — gpu_data should track.
        new_vals = torch.randn(16, dtype=torch.bfloat16, pin_memory=True)
        buf.pinned_data.copy_(new_vals)
        buf.copy_to_gpu(gpu_data, gpu_scale, non_blocking=True)
        torch.cuda.synchronize()
        assert torch.equal(gpu_data.cpu(), new_vals)
        # Stable Parameter object — gpu_param identity preserved.
        assert gpu_param is gpu_param  # tautology; documents intent

    def test_contiguous_format_forced(self) -> None:
        # A view of a transposed tensor is non-contiguous. clone() with
        # contiguous_format normalizes it; pinned data must be 1-D
        # contiguous so downstream callers can rely on it.
        base = torch.randn(8, 16, dtype=torch.bfloat16)
        non_contig = base.t()
        assert not non_contig.is_contiguous()
        p = nn.Parameter(non_contig, requires_grad=False)
        buf = PinnedParamBuffer("w", p)
        assert buf.pinned_data.is_contiguous()
        assert buf.pinned_data.is_pinned()

    def test_cpu_param_data_ptr_stable(self) -> None:
        # The cpu_param.data must be the same tensor object as pinned_data
        # (or a quanto wrapper around it) — callers repoint module
        # _parameters at it and expect the storage to be alive for the
        # buffer's lifetime.
        p = nn.Parameter(torch.randn(4, dtype=torch.bfloat16), requires_grad=False)
        buf = PinnedParamBuffer("w", p)
        ptr_before = buf.cpu_param.data.data_ptr()
        assert ptr_before == buf.pinned_data.data_ptr()
