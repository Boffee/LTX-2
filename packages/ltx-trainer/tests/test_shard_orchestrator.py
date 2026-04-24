"""Unit tests for the sharded-preprocessing orchestrator.

These tests exercise pure-Python logic only — no GPU, model weights, or
real video files. Everything is verified against tempdirs, fake metadata,
and empty sentinel files.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ltx_trainer.config import LtxTrainerConfig
from ltx_trainer.shard_orchestrator import (
    ShardOrchestrator,
    _parse_resolution_buckets,
    _scheduler_default_from_total,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_model_path(tmp_path: Path) -> Path:
    """Empty placeholder checkpoint path — `ModelConfig` only requires the
    file to exist, not to be loadable."""
    p = tmp_path / "model.safetensors"
    p.touch()
    return p


@pytest.fixture
def metadata_file(tmp_path: Path) -> Path:
    """Minimal metadata JSON with 10 rows — enough to exercise shard merging."""
    rows = [{"media_path": f"clip{i}.mp4", "caption": f"caption {i}"} for i in range(10)]
    p = tmp_path / "metadata.json"
    p.write_text(json.dumps(rows))
    return p


def _make_config(
    fake_model_path: Path,
    metadata_file: Path,
    tmp_path: Path,
    *,
    shard_size: int = 3,
    batch_size: int = 1,
    total_steps: int = 100,
    scheduler_type: str = "linear",
    scheduler_params: dict | None = None,
    with_audio: bool = False,
) -> LtxTrainerConfig:
    """Build a valid sharded-mode config for the tests."""
    train_dir = tmp_path / "train"
    shard_dir = tmp_path / "shard"
    train_dir.mkdir(exist_ok=True)
    shard_dir.mkdir(exist_ok=True)
    return LtxTrainerConfig(
        output_dir=str(train_dir),
        model={
            "model_path": str(fake_model_path),
            "text_encoder_path": str(tmp_path),
            "training_mode": "lora",
        },
        lora={},
        optimization={
            "steps": total_steps,
            "batch_size": batch_size,
            "scheduler_type": scheduler_type,
            "scheduler_params": scheduler_params or {},
        },
        data={
            "dataset_metadata_file": str(metadata_file),
            "resolution_buckets": "512x512x25",
            "shard_size": shard_size,
            "shard_preprocessing_output_dir": str(shard_dir),
        },
        training_strategy={"with_audio": with_audio},
    )


# ---------------------------------------------------------------------------
# _parse_resolution_buckets
# ---------------------------------------------------------------------------


class TestParseResolutionBuckets:
    def test_single(self) -> None:
        assert _parse_resolution_buckets("768x768x25") == [(25, 768, 768)]

    def test_multiple(self) -> None:
        assert _parse_resolution_buckets("768x768x25;512x512x49") == [
            (25, 768, 768),
            (49, 512, 512),
        ]

    def test_malformed_raises(self) -> None:
        with pytest.raises(ValueError, match="expected format"):
            _parse_resolution_buckets("768x768")


# ---------------------------------------------------------------------------
# _scheduler_default_from_total
# ---------------------------------------------------------------------------


class TestSchedulerDefaultFromTotal:
    @pytest.mark.parametrize("kind", ["linear", "cosine", "polynomial"])
    def test_full_run_length(self, kind: str) -> None:
        assert _scheduler_default_from_total(kind, 500) == 500

    def test_cosine_with_restarts_is_quarter(self) -> None:
        # Matches the trainer's non-sharded default of `T_0 = steps // 4`.
        assert _scheduler_default_from_total("cosine_with_restarts", 500) == 125


# ---------------------------------------------------------------------------
# _shard_groups
# ---------------------------------------------------------------------------


class TestShardGroups:
    def test_merge_trailing_undersize(
        self, fake_model_path: Path, metadata_file: Path, tmp_path: Path
    ) -> None:
        # 10 samples, shard_size=3 → [3, 3, 3, 1]; trailing merge → [3, 3, 4].
        cfg = _make_config(fake_model_path, metadata_file, tmp_path, shard_size=3)
        orch = ShardOrchestrator(cfg)
        groups = orch._shard_groups(0)
        assert [len(g) for g in groups] == [3, 3, 4]

    def test_deterministic_per_cycle(
        self, fake_model_path: Path, metadata_file: Path, tmp_path: Path
    ) -> None:
        cfg = _make_config(fake_model_path, metadata_file, tmp_path, shard_size=3)
        orch = ShardOrchestrator(cfg)
        g0a = [[r["media_path"] for r in s] for s in orch._shard_groups(0)]
        g0b = [[r["media_path"] for r in s] for s in orch._shard_groups(0)]
        assert g0a == g0b, "Same (seed, cycle) must produce the same composition"

    def test_reshuffle_between_cycles(
        self, fake_model_path: Path, metadata_file: Path, tmp_path: Path
    ) -> None:
        cfg = _make_config(fake_model_path, metadata_file, tmp_path, shard_size=3)
        orch = ShardOrchestrator(cfg)
        g0 = [[r["media_path"] for r in s] for s in orch._shard_groups(0)]
        g1 = [[r["media_path"] for r in s] for s in orch._shard_groups(1)]
        assert g0 != g1, "Cycle bump must reshuffle"

    def test_shard_size_exceeds_total(
        self, fake_model_path: Path, metadata_file: Path, tmp_path: Path
    ) -> None:
        cfg = _make_config(fake_model_path, metadata_file, tmp_path, shard_size=100)
        orch = ShardOrchestrator(cfg)
        groups = orch._shard_groups(0)
        # One shard with all 10 samples, no merge artefacts.
        assert len(groups) == 1
        assert len(groups[0]) == 10


# ---------------------------------------------------------------------------
# _latest_saved_checkpoint
# ---------------------------------------------------------------------------


def _touch_pair(ckpt_dir: Path, step: int, *, state: bool = True, weights: bool = True) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    padded = f"{step:05d}"
    if weights:
        (ckpt_dir / f"lora_weights_step_{padded}.safetensors").touch()
    if state:
        (ckpt_dir / f"training_state_step_{padded}.pt").touch()


class TestLatestSavedCheckpoint:
    def test_empty_dir(
        self, fake_model_path: Path, metadata_file: Path, tmp_path: Path
    ) -> None:
        cfg = _make_config(fake_model_path, metadata_file, tmp_path)
        orch = ShardOrchestrator(cfg)
        step, weights = orch._latest_saved_checkpoint()
        assert step == 0
        assert weights is None

    def test_matched_pair(
        self, fake_model_path: Path, metadata_file: Path, tmp_path: Path
    ) -> None:
        cfg = _make_config(fake_model_path, metadata_file, tmp_path)
        orch = ShardOrchestrator(cfg)
        _touch_pair(orch._ckpt_dir, 100)
        step, weights = orch._latest_saved_checkpoint()
        assert step == 100
        assert weights is not None and "step_00100" in weights.name

    def test_orphan_weights_ignored(
        self, fake_model_path: Path, metadata_file: Path, tmp_path: Path
    ) -> None:
        # Simulate a kill between the weights-save and the state-save:
        # step-100 is a complete pair, step-200 has only weights.
        cfg = _make_config(fake_model_path, metadata_file, tmp_path)
        orch = ShardOrchestrator(cfg)
        _touch_pair(orch._ckpt_dir, 100)
        _touch_pair(orch._ckpt_dir, 200, state=False)
        step, weights = orch._latest_saved_checkpoint()
        assert step == 100, "Must not pick up orphan weights newer than the last state"
        assert weights is not None and "step_00100" in weights.name

    def test_orphan_state_refused(
        self, fake_model_path: Path, metadata_file: Path, tmp_path: Path
    ) -> None:
        # State without matching weights is an unexpected half-state.
        cfg = _make_config(fake_model_path, metadata_file, tmp_path)
        orch = ShardOrchestrator(cfg)
        _touch_pair(orch._ckpt_dir, 100, weights=False)
        step, weights = orch._latest_saved_checkpoint()
        assert step == 0
        assert weights is None

    def test_multiple_pairs_picks_latest(
        self, fake_model_path: Path, metadata_file: Path, tmp_path: Path
    ) -> None:
        cfg = _make_config(fake_model_path, metadata_file, tmp_path)
        orch = ShardOrchestrator(cfg)
        _touch_pair(orch._ckpt_dir, 50)
        _touch_pair(orch._ckpt_dir, 150)
        _touch_pair(orch._ckpt_dir, 100)
        step, weights = orch._latest_saved_checkpoint()
        assert step == 150


# ---------------------------------------------------------------------------
# _count_trainable_rows
# ---------------------------------------------------------------------------


def _touch_outputs(
    output_dir: Path, media_rel: str, *, latents: bool, conditions: bool, audio: bool = False
) -> None:
    stem_pt = Path(media_rel).with_suffix(".pt")
    if latents:
        p = output_dir / "latents" / stem_pt
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
    if conditions:
        p = output_dir / "conditions" / stem_pt
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
    if audio:
        p = output_dir / "audio_latents" / stem_pt
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()


class TestCountTrainableRows:
    def test_all_complete(
        self, fake_model_path: Path, metadata_file: Path, tmp_path: Path
    ) -> None:
        cfg = _make_config(fake_model_path, metadata_file, tmp_path)
        orch = ShardOrchestrator(cfg)
        rows = [{"media_path": "a.mp4", "caption": "x"}, {"media_path": "b.mp4", "caption": "y"}]
        for r in rows:
            _touch_outputs(orch._output_dir, r["media_path"], latents=True, conditions=True)
        assert orch._count_trainable_rows(rows) == 2

    def test_missing_latents(
        self, fake_model_path: Path, metadata_file: Path, tmp_path: Path
    ) -> None:
        cfg = _make_config(fake_model_path, metadata_file, tmp_path)
        orch = ShardOrchestrator(cfg)
        rows = [{"media_path": "a.mp4", "caption": "x"}]
        _touch_outputs(orch._output_dir, "a.mp4", latents=False, conditions=True)
        assert orch._count_trainable_rows(rows) == 0

    def test_missing_conditions(
        self, fake_model_path: Path, metadata_file: Path, tmp_path: Path
    ) -> None:
        cfg = _make_config(fake_model_path, metadata_file, tmp_path)
        orch = ShardOrchestrator(cfg)
        rows = [{"media_path": "a.mp4", "caption": "x"}]
        _touch_outputs(orch._output_dir, "a.mp4", latents=True, conditions=False)
        assert orch._count_trainable_rows(rows) == 0

    def test_missing_audio_with_audio_required(
        self, fake_model_path: Path, metadata_file: Path, tmp_path: Path
    ) -> None:
        cfg = _make_config(fake_model_path, metadata_file, tmp_path, with_audio=True)
        orch = ShardOrchestrator(cfg)
        assert orch._strategy.requires_audio is True
        rows = [{"media_path": "a.mp4", "caption": "x"}]
        _touch_outputs(orch._output_dir, "a.mp4", latents=True, conditions=True, audio=False)
        assert orch._count_trainable_rows(rows) == 0

    def test_missing_audio_without_audio(
        self, fake_model_path: Path, metadata_file: Path, tmp_path: Path
    ) -> None:
        cfg = _make_config(fake_model_path, metadata_file, tmp_path, with_audio=False)
        orch = ShardOrchestrator(cfg)
        rows = [{"media_path": "a.mp4", "caption": "x"}]
        _touch_outputs(orch._output_dir, "a.mp4", latents=True, conditions=True, audio=False)
        assert orch._count_trainable_rows(rows) == 1, "Audio missing is OK when not required"


# ---------------------------------------------------------------------------
# _build_shard_config
# ---------------------------------------------------------------------------


class TestBuildShardConfig:
    def test_forced_keep_last_n_and_state(
        self, fake_model_path: Path, metadata_file: Path, tmp_path: Path
    ) -> None:
        cfg = _make_config(fake_model_path, metadata_file, tmp_path)
        orch = ShardOrchestrator(cfg)
        sc = orch._build_shard_config(None, 50, skip_initial_validation=False)
        assert sc.checkpoints.keep_last_n == -1
        assert sc.checkpoints.save_training_state == "full"

    def test_skip_initial_validation_flag(
        self, fake_model_path: Path, metadata_file: Path, tmp_path: Path
    ) -> None:
        cfg = _make_config(fake_model_path, metadata_file, tmp_path)
        orch = ShardOrchestrator(cfg)
        sc_true = orch._build_shard_config(None, 50, skip_initial_validation=True)
        sc_false = orch._build_shard_config(None, 50, skip_initial_validation=False)
        assert sc_true.validation.skip_initial_validation is True
        # User's original default is False; orchestrator only writes True.
        assert sc_false.validation.skip_initial_validation is False

    def test_preprocessed_data_root_points_at_preprocess_dir(
        self, fake_model_path: Path, metadata_file: Path, tmp_path: Path
    ) -> None:
        cfg = _make_config(fake_model_path, metadata_file, tmp_path)
        orch = ShardOrchestrator(cfg)
        sc = orch._build_shard_config(None, 50, skip_initial_validation=False)
        assert sc.data.preprocessed_data_root == str(orch._output_dir)
        # Metadata-mode fields cleared so the trainer doesn't recurse.
        assert sc.data.dataset_metadata_file is None
        assert sc.data.shard_size is None
        assert sc.data.resolution_buckets is None
        assert sc.data.shard_preprocessing_output_dir is None

    def test_scheduler_params_linear(
        self, fake_model_path: Path, metadata_file: Path, tmp_path: Path
    ) -> None:
        cfg = _make_config(
            fake_model_path, metadata_file, tmp_path, scheduler_type="linear", total_steps=500
        )
        orch = ShardOrchestrator(cfg)
        sc = orch._build_shard_config(None, 100, skip_initial_validation=False)
        assert sc.optimization.scheduler_params == {"total_iters": 500}

    def test_scheduler_params_cosine(
        self, fake_model_path: Path, metadata_file: Path, tmp_path: Path
    ) -> None:
        cfg = _make_config(
            fake_model_path, metadata_file, tmp_path, scheduler_type="cosine", total_steps=500
        )
        orch = ShardOrchestrator(cfg)
        sc = orch._build_shard_config(None, 100, skip_initial_validation=False)
        assert sc.optimization.scheduler_params == {"T_max": 500}

    def test_scheduler_params_cosine_with_restarts(
        self, fake_model_path: Path, metadata_file: Path, tmp_path: Path
    ) -> None:
        cfg = _make_config(
            fake_model_path,
            metadata_file,
            tmp_path,
            scheduler_type="cosine_with_restarts",
            total_steps=500,
        )
        orch = ShardOrchestrator(cfg)
        sc = orch._build_shard_config(None, 100, skip_initial_validation=False)
        # T_0 = total_steps // 4 preserves the trainer's ~4-restart default.
        assert sc.optimization.scheduler_params == {"T_0": 125}

    def test_scheduler_params_user_override_wins(
        self, fake_model_path: Path, metadata_file: Path, tmp_path: Path
    ) -> None:
        cfg = _make_config(
            fake_model_path,
            metadata_file,
            tmp_path,
            scheduler_type="linear",
            total_steps=500,
            scheduler_params={"total_iters": 777},
        )
        orch = ShardOrchestrator(cfg)
        sc = orch._build_shard_config(None, 100, skip_initial_validation=False)
        assert sc.optimization.scheduler_params == {"total_iters": 777}

    def test_scheduler_params_step_is_skipped(
        self, fake_model_path: Path, metadata_file: Path, tmp_path: Path
    ) -> None:
        cfg = _make_config(
            fake_model_path, metadata_file, tmp_path, scheduler_type="step", total_steps=500
        )
        orch = ShardOrchestrator(cfg)
        sc = orch._build_shard_config(None, 100, skip_initial_validation=False)
        # step_size is a period, not a total; orchestrator doesn't inject.
        assert sc.optimization.scheduler_params == {}

    def test_revalidation_preserves_invariants(
        self, fake_model_path: Path, metadata_file: Path, tmp_path: Path
    ) -> None:
        # The mutated copy is round-tripped through model_validate; make sure
        # its XOR invariant (preprocessed_data_root XOR dataset_metadata_file)
        # still holds — otherwise validation would raise here.
        cfg = _make_config(fake_model_path, metadata_file, tmp_path)
        orch = ShardOrchestrator(cfg)
        sc = orch._build_shard_config(None, 50, skip_initial_validation=False)
        assert isinstance(sc, LtxTrainerConfig)
        assert sc.data.preprocessed_data_root is not None
        assert sc.data.dataset_metadata_file is None
