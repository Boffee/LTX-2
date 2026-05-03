"""Shard orchestrator: preprocess one shard at a time, train on it, repeat.

Used when the training config sets ``data.dataset_metadata_file`` instead of
``data.preprocessed_data_root``. On each iteration the orchestrator:

  1. Deterministically shuffles the dataset into ``shard_size`` chunks.
  2. Writes the current shard's rows to a temporary metadata file.
  3. Calls ``preprocess_dataset`` (from ``scripts/process_dataset.py``) on it —
     writing video/audio latents and text embeddings under
     ``shard_preprocessing_output_dir``.
  4. Runs ``LtxvTrainer`` with a per-shard config that points at the output
     directory, resumes from the last checkpoint, and has ``steps`` set to the
     cumulative target so the trainer stops at the shard's epoch boundary.
  5. Loops to the next shard, reshuffling at cycle boundaries (``seed + cycle``).

The preprocessing scripts are unmodified; the trainer has a single 3-line
consistency tweak in ``_create_scheduler`` (scheduler sizing now reads from
``scheduler_params``, matching the existing pattern used by
``cosine_with_restarts`` and ``step``). Everything else in this module is
orchestration on top of the two existing entry points.
"""

from __future__ import annotations

import contextlib
import gc
import json
import os
import random
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from ltx_trainer import logger
from ltx_trainer.config import LtxTrainerConfig
from ltx_trainer.trainer import LtxvTrainer
from ltx_trainer.training_strategies import get_training_strategy

# Match the columns process_dataset.py expects by default.
VIDEO_COLUMN = "media_path"
CAPTION_COLUMN = "caption"

# Prefix for the per-run tmpfs tempdir that backs <output>/conditions. Stable
# value lets us recognise dirs left behind by earlier runs that died via SIGKILL.
TMPFS_PREFIX = "ltx-shard-conditions-"

# For scheduler types whose curve shape is controlled by a single parameter,
# the name of that parameter. Injected into ``scheduler_params`` per shard so
# the curve is sized for the full run rather than per-shard.
#
# ``step`` is omitted: ``step_size`` is a period (every N steps, multiply LR
# by gamma) rather than a total run length, so "set it from total_steps" has
# no obvious right value. User configures it themselves via scheduler_params.
_SCHEDULER_TOTAL_KEY: dict[str, str] = {
    "linear": "total_iters",
    "cosine": "T_max",
    "cosine_with_warmup": "T_max",
    "polynomial": "total_iters",
    "cosine_with_restarts": "T_0",
}


def _scheduler_default_from_total(
    scheduler_type: str, total_steps: int, params: dict | None = None
) -> int:
    """Default value to inject for the scheduler's sizing parameter, given
    the full run length. Mirrors the trainer's non-sharded defaults: for
    most schedulers the curve length equals ``total_steps``, but
    ``cosine_with_restarts`` uses ``T_0 = steps // 4`` — a restart cycle
    length, not a run total — so we preserve that meaning. ``cosine_with_warmup``
    pins the cosine phase length, so the default subtracts the warmup."""
    if scheduler_type == "cosine_with_restarts":
        return total_steps // 4
    if scheduler_type == "cosine_with_warmup":
        warmup_steps = (params or {}).get("warmup_steps", min(300, max(1, total_steps // 20)))
        return max(1, total_steps - warmup_steps)
    return total_steps


def _ensure_scripts_on_path() -> None:
    """Expose ``packages/ltx-trainer/scripts/`` so ``from process_dataset import ...`` works.

    ``preprocess_dataset`` is a script, not a library module — when invoked via
    ``python scripts/train_sharded.py``, the scripts dir is already on
    ``sys.path[0]``. This helper makes the import work in other contexts too
    (tests, uv run) without moving the preprocessing code into the package.
    """
    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))


def _parse_resolution_buckets(s: str) -> list[tuple[int, int, int]]:
    """``"WxHxF;WxHxF;..."`` → ``[(frames, height, width), ...]``."""
    out: list[tuple[int, int, int]] = []
    for bucket_str in s.split(";"):
        parts = bucket_str.strip().split("x")
        if len(parts) != 3:
            raise ValueError(f"Invalid bucket '{bucket_str}': expected format 'WxHxF'")
        w, h, f = (int(p) for p in parts)
        out.append((f, h, w))
    return out


def tear_down_trainer(trainer: "LtxvTrainer") -> None:
    """Release shard-trainer GPU/pinned memory so the next shard can
    construct cleanly. Drops every trainer attr that holds GPU state,
    then drops the block offloader last so its streamer finalizer
    flushes the allocator cache *after* the optimizer/model
    allocations are also freed.
    """
    try:
        offloader = getattr(trainer, "_model_offloader", None)
        if offloader is not None:
            offloader.deactivate()  # remove hooks, return slots to pinned-CPU
        # Drop optimizer/scheduler/model refs first so their tensors are
        # freed back to the allocator before the streamer's finalizer
        # runs ``empty_cache()`` (which fires when ``_model_offloader``
        # is dropped below).
        for attr in ("_optimizer", "_lr_scheduler", "_transformer",
                     "_text_encoder", "_embeddings_processor",
                     "_vae_decoder", "_vae_encoder"):
            if hasattr(trainer, attr):
                setattr(trainer, attr, None)
        if offloader is not None:
            trainer._model_offloader = None  # triggers streamer finalize
        gc.collect()
    except Exception as e:  # noqa: BLE001
        logger.warning(f"⚠️  Trainer teardown encountered {type(e).__name__}: {e}")


class ShardOrchestrator:
    """Runs one cycle of (preprocess shard → train on shard) at a time."""

    def __init__(self, config: LtxTrainerConfig) -> None:
        # Sharded preprocessing is single-GPU only — preprocessing, the tmpfs
        # symlink, and the on-disk shard output are all process-singleton state.
        # Under `accelerate launch` / `torchrun`, every rank would spawn its own
        # orchestrator and trample each other. Bail immediately.
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size > 1:
            raise RuntimeError(
                f"Sharded preprocessing mode is single-GPU only (WORLD_SIZE={world_size}). "
                f"Run `python scripts/train_sharded.py CONFIG` directly, not via "
                f"`accelerate launch` or `torchrun`. For multi-GPU training, preprocess "
                f"the dataset up front with scripts/process_dataset.py and use "
                f"scripts/train.py with data.preprocessed_data_root instead."
            )

        self._cfg = config
        # Preprocessing output (latents on disk, conditions symlinked to tmpfs)
        # is rooted at data.shard_preprocessing_output_dir; checkpoints come out
        # under cfg.output_dir (the trainer's choice — see trainer.py:1085).
        # These are typically distinct in user configs.
        self._output_dir = Path(config.data.shard_preprocessing_output_dir).expanduser().resolve()
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._ckpt_dir = Path(config.output_dir).expanduser().resolve() / "checkpoints"
        self._metadata_file = Path(config.data.dataset_metadata_file)
        self._samples = self._load_metadata(self._metadata_file)
        self._resolution_buckets = _parse_resolution_buckets(config.data.resolution_buckets)
        # Resolve the strategy here so we get a typed `requires_audio` answer
        # (the orchestrator only sees the strategy's *config*, which exposes
        # different fields per type — TextToVideoConfig has `with_audio`,
        # VideoToVideoConfig does not). Same dispatch the trainer uses.
        self._strategy = get_training_strategy(config.training_strategy)

    @staticmethod
    def _load_metadata(dataset_file: Path) -> list[dict[str, str]]:
        suffix = dataset_file.suffix.lower()
        if suffix == ".csv":
            rows = pd.read_csv(dataset_file).to_dict("records")
        elif suffix == ".json":
            with open(dataset_file, encoding="utf-8") as f:
                rows = json.load(f)
        elif suffix == ".jsonl":
            with open(dataset_file, encoding="utf-8") as f:
                rows = [json.loads(line) for line in f]
        else:
            raise ValueError(f"Unsupported metadata format: {suffix}")

        for row in rows:
            if VIDEO_COLUMN not in row or CAPTION_COLUMN not in row:
                raise ValueError(f"Metadata row missing '{VIDEO_COLUMN}' or '{CAPTION_COLUMN}': {row}")
        if not rows:
            raise ValueError(f"Metadata file {dataset_file} contains no samples")
        return rows

    def _shard_groups(self, cycle: int) -> list[list[dict[str, str]]]:
        """Deterministic shuffle + chunk. Reshuffles per cycle via ``seed + cycle``."""
        shard_size = self._cfg.data.shard_size
        indices = list(range(len(self._samples)))
        rng = random.Random(self._cfg.seed + cycle)
        rng.shuffle(indices)
        groups = [indices[i : i + shard_size] for i in range(0, len(indices), shard_size)]
        # Merge an undersized trailing shard so every shard is >= shard_size.
        if len(groups) > 1 and len(groups[-1]) < shard_size:
            groups[-2].extend(groups.pop())
        return [[self._samples[i] for i in g] for g in groups]

    def _write_shard_metadata(self, rows: list[dict[str, str]]) -> Path:
        """Write shard rows to a temp JSON next to the original metadata file so
        relative media paths resolve against the same data root."""
        tmp = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            prefix="ltx-shard-",
            dir=str(self._metadata_file.parent),
            delete=False,
            encoding="utf-8",
        )
        with tmp as f:
            json.dump(rows, f)
        return Path(tmp.name)

    def _count_trainable_rows(self, rows: list[dict[str, str]]) -> int:
        """Count rows whose required preprocessed files all landed on disk.

        ``preprocess_dataset`` silently filters individual samples (videos with
        too few frames in ``MediaDataset._filter_valid_videos``, captions that
        fail to encode, audio extraction errors). The trainer's
        ``PrecomputedDataset`` further filters samples missing any data source.
        If every row in a shard is filtered, the trainer's DataLoader fails
        deep in the stack; surface a clear orchestrator error before that.
        """
        latents_dir = self._output_dir / "latents"
        conditions_dir = self._output_dir / "conditions"
        audio_dir = self._output_dir / "audio_latents"
        needs_audio = self._strategy.requires_audio

        count = 0
        for row in rows:
            media_rel = Path(row[VIDEO_COLUMN]).with_suffix(".pt")
            if not (latents_dir / media_rel).exists():
                continue
            if not (conditions_dir / media_rel).exists():
                continue
            if needs_audio and not (audio_dir / media_rel).exists():
                continue
            count += 1
        return count

    def _preprocess_shard(self, rows: list[dict[str, str]]) -> None:
        _ensure_scripts_on_path()
        from process_dataset import preprocess_dataset  # noqa: PLC0415

        tmp_meta = self._write_shard_metadata(rows)
        try:
            preprocess_dataset(
                dataset_file=str(tmp_meta),
                caption_column=CAPTION_COLUMN,
                video_column=VIDEO_COLUMN,
                resolution_buckets=self._resolution_buckets,
                batch_size=1,
                output_dir=str(self._output_dir),
                lora_trigger=None,
                vae_tiling=self._cfg.data.vae_tiling,
                vae_tile_size=self._cfg.data.vae_tile_size,
                vae_tile_overlap=self._cfg.data.vae_tile_overlap,
                decode=False,
                model_path=str(self._cfg.model.model_path),
                text_encoder_path=str(self._cfg.model.text_encoder_path),
                device="cuda",
                remove_llm_prefixes=False,
                with_audio=self._strategy.requires_audio,
                load_text_encoder_in_8bit=self._cfg.acceleration.load_text_encoder_in_8bit,
            )
        finally:
            tmp_meta.unlink(missing_ok=True)

    def _build_shard_config(
        self, load_checkpoint: Path | None, target_steps: int, *, skip_initial_validation: bool
    ) -> LtxTrainerConfig:
        """Config passed to each per-shard ``LtxvTrainer``. Points at the shard
        output dir, resumes from the previous shard's final checkpoint, and sets
        ``optimization.steps`` to the cumulative target so the trainer stops at
        the shard's epoch boundary.

        The scheduler's curve-length parameter (``total_iters`` / ``T_max`` /
        ``T_0``) is pinned to the full-run total via ``scheduler_params`` so the
        LR curve spans all shards rather than resetting per shard. ``steps``
        controls the stop point, ``scheduler_params`` controls the curve.

        ``checkpoints.keep_last_n`` is forced to -1 (keep all) for the per-shard
        trainer: per-instance retention can't see across shards, and the trainer's
        same-step interval-save + final-save pattern can unlink the file the
        orchestrator needs for handoff. Sharded runs accumulate all per-shard
        checkpoints under the output dir; clean up manually if disk fills.

        ``checkpoints.save_training_state`` is forced to "full" so the optimizer
        state survives the per-shard trainer reinstantiation. With the default
        "minimal", Adam's m/v moments would reset every shard boundary and the
        adaptive LR would spike (lr/sqrt(v+eps) with v=0). Adds ~4× the LoRA
        weight size to each saved state file (~500MB for default LoRA configs;
        switch to optimizer_type="adamw8bit" to cut that ~4×).

        ``validation.skip_initial_validation`` is forced True on every shard
        except the very first trainer invocation of a fresh run — otherwise the
        trainer's per-call initial validation fires at every shard boundary.
        """
        cfg = self._cfg.model_copy(deep=True)
        cfg.data.preprocessed_data_root = str(self._output_dir)
        cfg.data.dataset_metadata_file = None
        cfg.data.resolution_buckets = None
        cfg.data.shard_size = None
        cfg.data.shard_preprocessing_output_dir = None
        cfg.model.load_checkpoint = str(load_checkpoint) if load_checkpoint else None
        cfg.optimization.steps = target_steps
        cfg.checkpoints.keep_last_n = -1  # see docstring
        cfg.checkpoints.save_training_state = "full"  # see docstring
        if skip_initial_validation:
            cfg.validation.skip_initial_validation = True

        total_steps = self._cfg.optimization.steps
        key = _SCHEDULER_TOTAL_KEY.get(cfg.optimization.scheduler_type)
        if key is not None:
            params = dict(cfg.optimization.scheduler_params)
            if cfg.optimization.scheduler_type == "cosine_with_warmup":
                params.setdefault("warmup_steps", min(300, max(1, total_steps // 20)))
            default = _scheduler_default_from_total(cfg.optimization.scheduler_type, total_steps, params)
            params.setdefault(key, default)  # user override wins
            cfg.optimization.scheduler_params = params

        # Re-validate via model_validate so model_validator-level invariants
        # are re-checked on the mutated copy. Pydantic's model_copy(deep=True)
        # does NOT re-run model validators; our mutations could silently
        # violate a future cross-field invariant added to LtxTrainerConfig.
        return LtxTrainerConfig.model_validate(cfg.model_dump())

    def run(self, disable_progress_bars: bool = False) -> None:
        # Check completion before touching tmpfs — no point validating the
        # mount just to return.
        completed_steps, _ = self._latest_saved_checkpoint()
        total_steps = self._cfg.optimization.steps
        if completed_steps >= total_steps:
            logger.info(
                f"✅ Already at step {completed_steps} ≥ target {total_steps}; "
                f"nothing to do. (Adjust optimization.steps to extend training, "
                f"or remove {self._ckpt_dir} to start fresh.)"
            )
            return
        # Validate tmpfs root once up front — the per-shard context just
        # creates/cleans a tempdir under this root, so any misconfiguration
        # should fail here rather than on every shard iteration.
        tmpfs_root = Path(self._cfg.data.tmpfs_conditions_dir)
        if not tmpfs_root.is_dir():
            raise RuntimeError(
                f"tmpfs_conditions_dir does not exist or is not a directory: {tmpfs_root}. "
                f"Mount the tmpfs first (e.g. /dev/shm is the Linux default)."
            )
        logger.info(f"🗂  conditions tmpfs root: {tmpfs_root}")
        self._run_loop(disable_progress_bars)

    @contextlib.contextmanager
    def _shard_conditions_tmpfs(self) -> Iterator[Path]:
        """Create a fresh tmpfs tempdir for one shard's text embeddings and
        symlink ``<output>/conditions`` at it. On context exit,
        ``TemporaryDirectory`` wipes the tempdir and the symlink is removed,
        so the previous shard's conditions are gone before the next shard
        begins preprocessing.

        Conditions never touch persistent disk: ``preprocess_dataset`` writes
        through the symlink into the tempdir on tmpfs.

        Note: SIGKILL / OOM-kill bypass ``TemporaryDirectory.__exit__``, so a
        hard-killed run leaves one orphan tempdir under ``tmpfs_conditions_dir``
        forever. We don't auto-sweep on entry because independent sharded runs
        sharing the tmpfs would clobber each other's live tempdirs. tmpfs is
        wiped on reboot; otherwise
        ``rm -rf <tmpfs>/ltx-shard-conditions-*`` manually.
        """
        tmpfs_root = Path(self._cfg.data.tmpfs_conditions_dir)
        cond_link = self._output_dir / "conditions"
        with tempfile.TemporaryDirectory(dir=str(tmpfs_root), prefix=TMPFS_PREFIX) as cond_path_str:
            cond_path = Path(cond_path_str)
            self._install_conditions_symlink(cond_link, cond_path)
            try:
                yield cond_path
            finally:
                # Remove only the symlink; the tempdir itself is cleaned by
                # the ``with`` on TemporaryDirectory.
                if cond_link.is_symlink():
                    cond_link.unlink()

    @staticmethod
    def _install_conditions_symlink(link: Path, target: Path) -> None:
        """Ensure ``link → target``. Refuses to overwrite a real directory to
        avoid stomping on an earlier non-tmpfs run's cached conditions."""
        if link.is_symlink():
            link.unlink()
        elif link.exists():
            raise RuntimeError(
                f"{link} already exists and is not a symlink — refusing to overwrite. "
                f"Remove it manually if you want to switch to tmpfs_conditions_dir."
            )
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target, target_is_directory=True)

    def _latest_saved_checkpoint(self) -> tuple[int, Path | None]:
        """Return ``(step, weights_path)`` for the latest complete (weights + state)
        pair, or ``(0, None)`` if none. Weights-without-state orphans are ignored.

        We glob ``training_state_step_*.pt`` because the trainer writes weights
        before the state sidecar (``trainer.py:1121``/``1127`` then ``:1135``); a
        kill between those leaves orphan weights without optimizer/scheduler
        state. The sidecar is written atomically via tmp+rename in
        ``_save_training_state``, so its presence guarantees the matching
        weights file also landed.

        The caller passes the returned ``weights_path`` as ``model.load_checkpoint``
        rather than the checkpoints directory, so the trainer's ``_find_checkpoint``
        can't fall back to a newer orphan weights file (which would bypass the
        atomic-resume guarantee and make ``_resolve_resume_state`` silently return
        ``(0, None)``, losing optimizer state continuity).
        """
        if not self._ckpt_dir.is_dir():
            return 0, None
        latest_step = 0
        latest_state_path: Path | None = None
        for p in self._ckpt_dir.rglob("training_state_step_*.pt"):
            try:
                step = int(p.stem.split("step_")[1])
            except (IndexError, ValueError):
                continue
            if step > latest_step:
                latest_step, latest_state_path = step, p
        if latest_state_path is None:
            return 0, None
        padded = f"{latest_step:05d}"
        weights = next(latest_state_path.parent.glob(f"*_weights_step_{padded}.safetensors"), None)
        if weights is None:
            # Sidecar without matching weights is an unexpected half-state.
            # Refuse to resume rather than silently start fresh.
            logger.warning(
                f"⚠️  Found {latest_state_path.name} with no matching weights file. "
                f"Refusing to resume — starting from scratch."
            )
            return 0, None
        return latest_step, weights

    def _run_loop(self, disable_progress_bars: bool) -> None:
        total_steps = self._cfg.optimization.steps
        # One optimization step consumes batch_size * gradient_accumulation_steps
        # samples (the trainer loops `remaining_steps * grad_accum` batches).
        samples_per_step = self._cfg.optimization.batch_size * self._cfg.optimization.gradient_accumulation_steps

        # Resume: if any shard has already saved a (weights + state) pair under
        # this output dir, point the first shard's trainer at the matching
        # weights file so the trainer's resume path loads its sidecar state.
        # Skip any shards whose cumulative target was already reached.
        completed_steps, resume_weights = self._latest_saved_checkpoint()
        if completed_steps > 0:
            logger.info(f"🔁 Resuming orchestrator from step {completed_steps}")
            last_checkpoint: Path | None = resume_weights
        else:
            last_checkpoint = (
                Path(self._cfg.model.load_checkpoint) if self._cfg.model.load_checkpoint else None
            )

        # Initial validation should fire once at the start of a truly fresh
        # run (no checkpoint on disk) — let the user's skip_initial_validation
        # govern that first one. Every subsequent trainer invocation, including
        # any after a process restart, must skip it; otherwise we'd validate
        # at every shard boundary.
        seen_initial = completed_steps > 0

        cumulative_target = 0
        cycle = 0
        while cumulative_target < total_steps:
            groups = self._shard_groups(cycle)
            for shard_idx, shard_rows in enumerate(groups):
                # Steps budget tracks actual shard size: a merged trailing
                # shard (up to 2*shard_size-1 rows) gets proportionally more
                # training than a standard shard, and shard_size > total_samples
                # degenerates cleanly to a single-shard cycle. Divide by
                # samples_per_step (= batch_size * grad_accum) because the
                # trainer counts optimization steps, not batch iterations.
                steps_this_shard = max(1, len(shard_rows) // samples_per_step)
                cumulative_target = min(cumulative_target + steps_this_shard, total_steps)

                # Skip shards whose work is already covered by the latest checkpoint.
                if cumulative_target <= completed_steps:
                    continue

                logger.info(
                    f"🧩 Shard cycle={cycle} idx={shard_idx + 1}/{len(groups)} "
                    f"samples={len(shard_rows)} target_step={cumulative_target}/{total_steps}"
                )

                # Fresh tmpfs tempdir scoped to this shard — only one shard's
                # text embeddings occupy RAM at a time. Context exit wipes
                # the tempdir before the next shard starts.
                with self._shard_conditions_tmpfs():
                    self._preprocess_shard(shard_rows)

                    trainable = self._count_trainable_rows(shard_rows)
                    batch_size = self._cfg.optimization.batch_size
                    if trainable < batch_size:
                        raise RuntimeError(
                            f"Shard cycle={cycle} idx={shard_idx + 1} produced {trainable} "
                            f"trainable samples; need >= batch_size ({batch_size}) so the "
                            f"trainer's drop_last=True dataloader yields at least one batch. "
                            f"Of {len(shard_rows)} input rows, the rest were dropped during "
                            f"preprocessing (frame-count filter, encode failures, missing "
                            f"audio under with_audio=True, etc.). Check warnings above for "
                            f"per-sample errors; fix the metadata or the underlying media "
                            f"files and re-run."
                        )

                    shard_cfg = self._build_shard_config(
                        last_checkpoint,
                        cumulative_target,
                        skip_initial_validation=seen_initial,
                    )
                    trainer = LtxvTrainer(shard_cfg)
                    trainer.train(disable_progress_bars=disable_progress_bars)
                    # Re-resolve the latest complete (weights + state) pair so
                    # the next shard resumes from an atomic pair — handing
                    # over the checkpoints directory would let the trainer's
                    # _find_checkpoint glob pick an orphan weights file from
                    # a kill-in-flight write.
                    _, last_checkpoint = self._latest_saved_checkpoint()
                    seen_initial = True
                    # Tear down the shard's trainer explicitly before the
                    # next shard creates a new one. Forces ``gc.collect()``
                    # so any PEFT/accelerator-introduced cycles are broken
                    # immediately and the streamer's finalizer flushes the
                    # CUDA cache before shard 2 loads, instead of waiting
                    # for the cycle collector to run on its own schedule.
                    tear_down_trainer(trainer)
                    del trainer

                if cumulative_target >= total_steps:
                    logger.info("✅ Reached total step target — orchestrator done")
                    return
            cycle += 1
