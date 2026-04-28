"""Unit tests for ``DiscreteSigmasTimestepSampler``.

Used for distillation training: training timesteps must be drawn from
the same fixed sigma set the model sees at inference. A bug here would
silently mis-train the distilled targets.
"""

from __future__ import annotations

import pytest
import torch

from ltx_trainer.timestep_samplers import DiscreteSigmasTimestepSampler


class TestDiscreteSigmasTimestepSampler:
    def test_sample_values_from_set(self) -> None:
        sigmas = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875]
        sampler = DiscreteSigmasTimestepSampler(sigmas)
        out = sampler.sample(batch_size=10_000)
        assert out.shape == (10_000,)
        # Every sample must be one of the input values (within fp32 epsilon).
        unique = set(out.unique().tolist())
        expected = set(sigmas)
        # All observed values are in the expected set.
        for v in unique:
            assert any(abs(v - e) < 1e-6 for e in expected), f"{v} not in {expected}"
        # With 10k samples, every sigma should appear at least once.
        assert len(unique) == len(expected)

    def test_sample_for_batch_shape(self) -> None:
        sampler = DiscreteSigmasTimestepSampler([0.5, 0.7])
        batch = torch.zeros(4, 16, 8)
        out = sampler.sample_for(batch)
        assert out.shape == (4,)

    def test_sample_for_wrong_ndim(self) -> None:
        sampler = DiscreteSigmasTimestepSampler([0.5])
        with pytest.raises(ValueError, match="3 dimensions"):
            sampler.sample_for(torch.zeros(4, 8))

    def test_empty_list_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            DiscreteSigmasTimestepSampler([])

    def test_zero_sigma_rejected(self) -> None:
        # The terminal 0.0 has a degenerate training target (clean = clean)
        # and must be excluded from the training sigma set.
        with pytest.raises(ValueError, match="must be > 0"):
            DiscreteSigmasTimestepSampler([1.0, 0.5, 0.0])

    def test_negative_sigma_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be > 0"):
            DiscreteSigmasTimestepSampler([1.0, -0.1])
