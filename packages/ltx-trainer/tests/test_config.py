"""Unit tests for the sharded-preprocessing config validators.

These tests verify that the ``LtxTrainerConfig`` model validator rejects
configurations that would silently break downstream (sharded + v2v, sharded
with shard_size < batch_size, etc.) and that ``tmpfs_conditions_dir``'s
path check is deferred to orchestrator entry.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ltx_trainer.config import LtxTrainerConfig


@pytest.fixture
def fake_model_path(tmp_path: Path) -> Path:
    p = tmp_path / "model.safetensors"
    p.touch()
    return p


@pytest.fixture
def metadata_file(tmp_path: Path) -> Path:
    rows = [{"media_path": f"c{i}.mp4", "caption": f"x {i}"} for i in range(5)]
    p = tmp_path / "metadata.json"
    p.write_text(json.dumps(rows))
    return p


def _base(fake_model_path: Path, tmp_path: Path) -> dict:
    """Minimal valid precomputed-mode config dict — tests mutate from here."""
    pre = tmp_path / "precomputed"
    pre.mkdir(exist_ok=True)
    return {
        "output_dir": str(tmp_path / "out"),
        "model": {
            "model_path": str(fake_model_path),
            "text_encoder_path": str(tmp_path),
            "training_mode": "lora",
        },
        "lora": {},
        "optimization": {"steps": 100, "batch_size": 1},
        "data": {"preprocessed_data_root": str(pre)},
    }


def _sharded(base: dict, metadata_file: Path, tmp_path: Path, **overrides) -> dict:
    """Convert a precomputed base into a sharded-mode dict with defaults."""
    base["data"] = {
        "dataset_metadata_file": str(metadata_file),
        "resolution_buckets": "512x512x25",
        "shard_size": 3,
        "shard_preprocessing_output_dir": str(tmp_path / "shard"),
    }
    (tmp_path / "shard").mkdir(exist_ok=True)
    for k, v in overrides.items():
        base["data"][k] = v
    return base


# ---------------------------------------------------------------------------
# Data source XOR
# ---------------------------------------------------------------------------


class TestDataSourceXor:
    def test_neither_raises(self, fake_model_path: Path, tmp_path: Path) -> None:
        cfg = _base(fake_model_path, tmp_path)
        cfg["data"] = {}
        with pytest.raises(ValueError, match="Either preprocessed_data_root or dataset_metadata_file"):
            LtxTrainerConfig(**cfg)

    def test_both_raises(
        self, fake_model_path: Path, metadata_file: Path, tmp_path: Path
    ) -> None:
        cfg = _base(fake_model_path, tmp_path)
        cfg["data"]["dataset_metadata_file"] = str(metadata_file)
        with pytest.raises(ValueError, match="mutually exclusive"):
            LtxTrainerConfig(**cfg)

    def test_precomputed_only(self, fake_model_path: Path, tmp_path: Path) -> None:
        LtxTrainerConfig(**_base(fake_model_path, tmp_path))

    def test_sharded_only(
        self, fake_model_path: Path, metadata_file: Path, tmp_path: Path
    ) -> None:
        base = _base(fake_model_path, tmp_path)
        LtxTrainerConfig(**_sharded(base, metadata_file, tmp_path))


# ---------------------------------------------------------------------------
# Sharded-mode required aux fields
# ---------------------------------------------------------------------------


class TestShardedRequiredFields:
    def test_missing_buckets(
        self, fake_model_path: Path, metadata_file: Path, tmp_path: Path
    ) -> None:
        cfg = _sharded(_base(fake_model_path, tmp_path), metadata_file, tmp_path)
        cfg["data"]["resolution_buckets"] = None
        with pytest.raises(ValueError, match="resolution_buckets is required"):
            LtxTrainerConfig(**cfg)

    def test_missing_shard_size(
        self, fake_model_path: Path, metadata_file: Path, tmp_path: Path
    ) -> None:
        cfg = _sharded(_base(fake_model_path, tmp_path), metadata_file, tmp_path)
        cfg["data"]["shard_size"] = None
        with pytest.raises(ValueError, match="shard_size is required"):
            LtxTrainerConfig(**cfg)

    def test_missing_output_dir(
        self, fake_model_path: Path, metadata_file: Path, tmp_path: Path
    ) -> None:
        cfg = _sharded(_base(fake_model_path, tmp_path), metadata_file, tmp_path)
        cfg["data"]["shard_preprocessing_output_dir"] = None
        with pytest.raises(ValueError, match="shard_preprocessing_output_dir is required"):
            LtxTrainerConfig(**cfg)

    def test_missing_text_encoder_path(
        self, fake_model_path: Path, metadata_file: Path, tmp_path: Path
    ) -> None:
        cfg = _sharded(_base(fake_model_path, tmp_path), metadata_file, tmp_path)
        cfg["model"]["text_encoder_path"] = None
        with pytest.raises(ValueError, match="text_encoder_path is required"):
            LtxTrainerConfig(**cfg)


# ---------------------------------------------------------------------------
# Sharded-mode strategy / batch_size compat
# ---------------------------------------------------------------------------


class TestShardedStrategyBatchSize:
    def test_video_to_video_rejected(
        self,
        fake_model_path: Path,
        metadata_file: Path,
        tmp_path: Path,
    ) -> None:
        cfg = _sharded(_base(fake_model_path, tmp_path), metadata_file, tmp_path)
        # Minimal v2v strategy shape + required validation reference_videos.
        cfg["training_strategy"] = {"name": "video_to_video"}
        cfg["validation"] = {
            "prompts": ["p"],
            "reference_videos": [str(metadata_file)],  # path only needs to exist
            "interval": 100,
            "reference_downscale_factor": 1,
        }
        with pytest.raises(ValueError, match="video_to_video"):
            LtxTrainerConfig(**cfg)

    def test_shard_size_lt_batch_size_rejected(
        self, fake_model_path: Path, metadata_file: Path, tmp_path: Path
    ) -> None:
        cfg = _sharded(_base(fake_model_path, tmp_path), metadata_file, tmp_path, shard_size=2)
        cfg["optimization"]["batch_size"] = 4
        with pytest.raises(ValueError, match="shard_size .* must be >= batch_size"):
            LtxTrainerConfig(**cfg)

    def test_shard_size_eq_batch_size_ok(
        self, fake_model_path: Path, metadata_file: Path, tmp_path: Path
    ) -> None:
        cfg = _sharded(_base(fake_model_path, tmp_path), metadata_file, tmp_path, shard_size=4)
        cfg["optimization"]["batch_size"] = 4
        LtxTrainerConfig(**cfg)  # must not raise


# ---------------------------------------------------------------------------
# tmpfs_conditions_dir is deferred — config load must not check path existence
# ---------------------------------------------------------------------------


class TestTmpfsDeferredValidation:
    def test_nonexistent_path_passes_config_load(
        self, fake_model_path: Path, metadata_file: Path, tmp_path: Path
    ) -> None:
        cfg = _sharded(
            _base(fake_model_path, tmp_path),
            metadata_file,
            tmp_path,
            tmpfs_conditions_dir="/nonexistent/path/xyz",
        )
        # Must not raise — existence is checked at orchestrator entry, not here.
        loaded = LtxTrainerConfig(**cfg)
        assert str(loaded.data.tmpfs_conditions_dir) == "/nonexistent/path/xyz"

    def test_default_is_dev_shm(
        self, fake_model_path: Path, metadata_file: Path, tmp_path: Path
    ) -> None:
        cfg = _sharded(_base(fake_model_path, tmp_path), metadata_file, tmp_path)
        loaded = LtxTrainerConfig(**cfg)
        assert str(loaded.data.tmpfs_conditions_dir) == "/dev/shm"


# ---------------------------------------------------------------------------
# Distilled-sigma training: discrete_sigmas mode + base_lora wiring
# ---------------------------------------------------------------------------


class TestDiscreteSigmasMode:
    def test_loads_with_sigmas_param(self, fake_model_path: Path, tmp_path: Path) -> None:
        cfg = _base(fake_model_path, tmp_path)
        cfg["flow_matching"] = {
            "timestep_sampling_mode": "discrete_sigmas",
            "timestep_sampling_params": {"sigmas": [1.0, 0.5, 0.25]},
        }
        loaded = LtxTrainerConfig(**cfg)
        assert loaded.flow_matching.timestep_sampling_mode == "discrete_sigmas"
        assert loaded.flow_matching.timestep_sampling_params["sigmas"] == [1.0, 0.5, 0.25]

    def test_unknown_mode_rejected(self, fake_model_path: Path, tmp_path: Path) -> None:
        cfg = _base(fake_model_path, tmp_path)
        cfg["flow_matching"] = {"timestep_sampling_mode": "made_up_mode"}
        with pytest.raises(ValueError):
            LtxTrainerConfig(**cfg)


class TestBaseLoraConfig:
    def test_loads_with_existing_path(self, fake_model_path: Path, tmp_path: Path) -> None:
        # fake_model_path is a real (empty) file from the fixture.
        cfg = _base(fake_model_path, tmp_path)
        cfg["model"]["base_lora"] = {"path": str(fake_model_path), "strength": 0.6}
        loaded = LtxTrainerConfig(**cfg)
        assert loaded.model.base_lora is not None
        assert loaded.model.base_lora.strength == 0.6

    def test_missing_path_rejected(self, fake_model_path: Path, tmp_path: Path) -> None:
        cfg = _base(fake_model_path, tmp_path)
        cfg["model"]["base_lora"] = {"path": str(tmp_path / "does_not_exist.safetensors")}
        with pytest.raises(ValueError, match="does not exist"):
            LtxTrainerConfig(**cfg)

    def test_negative_strength_rejected(self, fake_model_path: Path, tmp_path: Path) -> None:
        cfg = _base(fake_model_path, tmp_path)
        cfg["model"]["base_lora"] = {"path": str(fake_model_path), "strength": -0.1}
        with pytest.raises(ValueError):
            LtxTrainerConfig(**cfg)
