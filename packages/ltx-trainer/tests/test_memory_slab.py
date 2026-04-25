"""Tests for ``ltx_core.memory.buffers.PinnedParamBuffer``.

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

from ltx_core.memory.buffers import PinnedParamBuffer


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
        # Mirrors how GpuSlot uses PinnedParamBuffer: allocate GPU
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
        # Stable storage — gpu_param wraps the same GPU bytes as gpu_data.
        # GpuSlot relies on this: build the Parameter wrapper once at slot
        # construction, mutate underlying storage in place on each load.
        assert gpu_param.data_ptr() == gpu_data.data_ptr()

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

    @CUDA
    def test_slot_param_identity_stable_across_loads(self) -> None:
        # GpuSlot caches the Parameter wrapping its GPU storage; copy_from
        # must not churn that wrapper. Hooks repointing submod._parameters
        # at slot.get_param() observe a stable object across reloads — the
        # whole point of the pool-slot pattern over per-load allocation.
        from ltx_core.memory.streaming import GpuSlot

        p1 = nn.Parameter(torch.randn(8, dtype=torch.bfloat16), requires_grad=False)
        p2 = nn.Parameter(torch.randn(8, dtype=torch.bfloat16), requires_grad=False)
        block = [PinnedParamBuffer("a", p1), PinnedParamBuffer("b", p2)]
        slot = GpuSlot(block, torch.device("cuda"))

        a_first = slot.get_param("a")
        b_first = slot.get_param("b")
        slot.copy_from(block, non_blocking=False)
        torch.cuda.synchronize()
        assert slot.get_param("a") is a_first
        assert slot.get_param("b") is b_first
        slot.copy_from(block, non_blocking=False)
        torch.cuda.synchronize()
        assert slot.get_param("a") is a_first
        assert slot.get_param("b") is b_first


# ---------------------------------------------------------------------------
# Quanto path — only nontrivial branch in PinnedParamBuffer
# ---------------------------------------------------------------------------


class TestPinnedParamBufferQuanto:
    def test_pin_decomposes_data_and_scale(self) -> None:
        # Quanto WeightQBytesTensor must be decomposed into _data + _scale
        # and the cpu_param wrapper reconstructed from the pinned tensors.
        # A naive tensor.clone() would silently dequantize via the dispatch
        # fallback — that bug is the reason buffers.py exists.
        quanto = pytest.importorskip("optimum.quanto")
        from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor

        rows, cols = 4, 8
        data = torch.randint(-128, 127, (rows, cols), dtype=torch.int8)
        scale = torch.rand(rows, 1, dtype=torch.bfloat16)
        qt = WeightQBytesTensor.create(
            quanto.qint8, 0, (rows, cols), (cols, 1), data, scale, None,
        )
        p = nn.Parameter(qt, requires_grad=False)
        buf = PinnedParamBuffer("w", p)

        assert buf.is_quanto
        assert buf.pinned_data.is_pinned()
        assert buf.pinned_data.is_contiguous()
        assert buf.pinned_data.dtype == torch.int8
        assert buf.pinned_scale is not None
        assert buf.pinned_scale.is_pinned()
        assert buf.qtype is quanto.qint8
        assert buf.axis == 0
        assert tuple(buf.size) == (rows, cols)
        assert buf.stride == (cols, 1)
        assert buf.act_qt is None
        # cpu_param wraps a quanto tensor pointing at the pinned tensors.
        assert isinstance(buf.cpu_param.data, WeightQBytesTensor)
        assert buf.cpu_param.data._data.data_ptr() == buf.pinned_data.data_ptr()
        assert buf.cpu_param.data._scale.data_ptr() == buf.pinned_scale.data_ptr()

    @CUDA
    def test_load_to_gpu_round_trip(self) -> None:
        quanto = pytest.importorskip("optimum.quanto")
        from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor

        rows, cols = 4, 8
        data = torch.randint(-128, 127, (rows, cols), dtype=torch.int8)
        scale = torch.rand(rows, 1, dtype=torch.bfloat16)
        qt = WeightQBytesTensor.create(
            quanto.qint8, 0, (rows, cols), (cols, 1), data, scale, None,
        )
        p = nn.Parameter(qt, requires_grad=False)
        buf = PinnedParamBuffer("w", p)

        gpu_param = buf.load_to_gpu(torch.device("cuda"))
        torch.cuda.synchronize()
        assert isinstance(gpu_param.data, WeightQBytesTensor)
        assert gpu_param.data._data.is_cuda
        assert gpu_param.data._scale.is_cuda
        assert torch.equal(gpu_param.data._data.cpu(), buf.pinned_data)
        assert torch.equal(gpu_param.data._scale.cpu(), buf.pinned_scale)
