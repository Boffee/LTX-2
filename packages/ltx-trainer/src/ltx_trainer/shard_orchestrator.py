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

The trainer and the preprocessing scripts are unchanged; this module just
orchestrates calls between them.
"""

from __future__ import annotations

import json
import random
import sys
import tempfile
from pathlib import Path

import pandas as pd

from ltx_trainer import logger
from ltx_trainer.config import LtxTrainerConfig
from ltx_trainer.trainer import LtxvTrainer

# Match the columns process_dataset.py expects by default.
VIDEO_COLUMN = "media_path"
CAPTION_COLUMN = "caption"


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


class ShardOrchestrator:
    """Runs one cycle of (preprocess shard → train on shard) at a time."""

    def __init__(self, config: LtxTrainerConfig) -> None:
        self._cfg = config
        self._output_dir = Path(config.data.shard_preprocessing_output_dir).expanduser().resolve()
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._metadata_file = Path(config.data.dataset_metadata_file)
        self._samples = self._load_metadata(self._metadata_file)
        self._resolution_buckets = _parse_resolution_buckets(config.data.resolution_buckets)

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
                vae_tiling=False,
                decode=False,
                model_path=str(self._cfg.model.model_path),
                text_encoder_path=str(self._cfg.model.text_encoder_path),
                device="cuda",
                remove_llm_prefixes=False,
                with_audio=getattr(self._cfg.training_strategy, "with_audio", False),
                load_text_encoder_in_8bit=self._cfg.acceleration.load_text_encoder_in_8bit,
            )
        finally:
            tmp_meta.unlink(missing_ok=True)

    def _build_shard_config(self, load_checkpoint: Path | None, target_steps: int) -> LtxTrainerConfig:
        """Config passed to each per-shard ``LtxvTrainer``. Points at the shard
        output dir, resumes from the previous shard's final checkpoint, and sets
        ``optimization.steps`` to the cumulative target so the trainer stops at
        the shard's epoch boundary."""
        cfg = self._cfg.model_copy(deep=True)
        cfg.data.preprocessed_data_root = str(self._output_dir)
        cfg.data.dataset_metadata_file = None
        cfg.data.resolution_buckets = None
        cfg.data.shard_size = None
        cfg.data.shard_preprocessing_output_dir = None
        cfg.model.load_checkpoint = str(load_checkpoint) if load_checkpoint else None
        cfg.optimization.steps = target_steps
        return cfg

    def run(self, disable_progress_bars: bool = False) -> None:
        total_steps = self._cfg.optimization.steps
        shard_size = self._cfg.data.shard_size
        batch_size = self._cfg.optimization.batch_size
        steps_per_shard = max(1, shard_size // batch_size)

        last_checkpoint: Path | None = (
            Path(self._cfg.model.load_checkpoint) if self._cfg.model.load_checkpoint else None
        )
        cumulative_target = 0
        cycle = 0
        while cumulative_target < total_steps:
            groups = self._shard_groups(cycle)
            for shard_idx, shard_rows in enumerate(groups):
                cumulative_target = min(cumulative_target + steps_per_shard, total_steps)
                logger.info(
                    f"🧩 Shard cycle={cycle} idx={shard_idx + 1}/{len(groups)} "
                    f"samples={len(shard_rows)} target_step={cumulative_target}/{total_steps}"
                )

                self._preprocess_shard(shard_rows)

                shard_cfg = self._build_shard_config(last_checkpoint, cumulative_target)
                trainer = LtxvTrainer(shard_cfg)
                last_checkpoint, _ = trainer.train(disable_progress_bars=disable_progress_bars)

                if cumulative_target >= total_steps:
                    logger.info("✅ Reached total step target — orchestrator done")
                    return
            cycle += 1
