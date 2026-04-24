"""Sharded offline-style preprocessing at training time.

Runs the offline preprocessing pipeline (VAE latents + text embeddings) on one
shard at a time and writes results to ``.precomputed/{latents,conditions}/``.
The trainer consumes the shard from disk, trains one epoch, then rotates to
the next shard which is preprocessed on demand.

File layout uses ``<stem>_<f>x<h>x<w>.pt`` with the *chosen* resolution bucket
as a filename suffix, so editing the bucket list only forces re-encoding for
videos whose nearest bucket actually changed; everything else is reused. The
suffix is mirrored on the conditions file (same bucket key as latents) so the
pairing logic in :class:`PrecomputedDataset` keeps working.

Difference from :mod:`online_dataset`: both latents and text embeddings are
written to disk (vs. embeddings kept only in RAM). This sets up for future
tmpfs-backed condition directories where the on-disk representation is mounted
in memory for faster retrieval.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset

from ltx_trainer import logger
from ltx_trainer.shard_manager import ShardManager
from ltx_trainer.video_preprocessing import (
    bucket_filename_suffix,
    find_nearest_bucket,
    max_frames_in_buckets,
    normalize_video_frames,
    resize_and_crop_video,
)
from ltx_trainer.video_utils import get_video_metadata, read_video

# Conventions (same as process_dataset.py defaults)
VIDEO_COLUMN = "media_path"
CAPTION_COLUMN = "caption"
PRECOMPUTED_DIR_NAME = ".precomputed"
LATENTS_DIR_NAME = "latents"
CONDITIONS_DIR_NAME = "conditions"


class ShardPreprocessingDataset(Dataset):
    """Dataset that preprocesses raw videos + captions per shard, writing to disk.

    Data source is a metadata file (CSV/JSON/JSONL) with columns ``media_path``
    (video paths) and ``caption`` (text). Samples are split into shards managed
    by :class:`ShardManager`. Before each epoch the trainer calls
    :meth:`encode_shard`, which encodes the shard's videos with the VAE and its
    captions with the text encoder, writing results to
    ``<output_dir>/.precomputed/{latents,conditions}/``. The dataset then serves
    the shard's samples by loading those files from disk.

    Parameters
    ----------
    dataset_file:
        Path to CSV/JSON/JSONL metadata file.
    output_dir:
        Directory under which the ``.precomputed/`` tree is written. Typically
        the dataset root — the result is a drop-in for :class:`PrecomputedDataset`.
    resolution_buckets:
        List of ``(frames, height, width)`` tuples. Each video is matched to the
        nearest bucket by aspect ratio, then resized and cropped to that exact
        size.
    shard_size:
        Samples per shard. ``None`` = preprocess the entire dataset in one shard.
    seed:
        Random seed for shard shuffling.
    """

    def __init__(
        self,
        dataset_file: str | Path,
        output_dir: str | Path,
        resolution_buckets: list[tuple[int, int, int]],
        shard_size: int | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self._dataset_file = Path(dataset_file)
        self._output_root = Path(output_dir).expanduser().resolve() / PRECOMPUTED_DIR_NAME
        self._latents_dir = self._output_root / LATENTS_DIR_NAME
        self._conditions_dir = self._output_root / CONDITIONS_DIR_NAME
        self._resolution_buckets = resolution_buckets
        self._max_frames = max_frames_in_buckets(resolution_buckets)

        self._samples = self._load_metadata()
        self._shard_files: list[tuple[Path, Path]] | None = None
        self._shards = ShardManager(
            total_samples=len(self._samples),
            shard_size=shard_size,
            seed=seed,
        )

    # ------------------------------------------------------------------
    # Metadata loading
    # ------------------------------------------------------------------

    def _load_metadata(self) -> list[dict[str, str]]:
        suffix = self._dataset_file.suffix.lower()
        if suffix == ".csv":
            rows = pd.read_csv(self._dataset_file).to_dict("records")
        elif suffix == ".json":
            with open(self._dataset_file, encoding="utf-8") as f:
                rows = json.load(f)
        elif suffix == ".jsonl":
            with open(self._dataset_file, encoding="utf-8") as f:
                rows = [json.loads(line) for line in f]
        else:
            raise ValueError(f"Unsupported metadata format: {suffix}")

        data_root = self._dataset_file.parent
        samples: list[dict[str, str]] = []
        for row in rows:
            if VIDEO_COLUMN not in row or CAPTION_COLUMN not in row:
                raise ValueError(
                    f"Metadata row missing '{VIDEO_COLUMN}' or '{CAPTION_COLUMN}' column: {row}"
                )
            rel = Path(row[VIDEO_COLUMN].strip())
            samples.append(
                {
                    "video_path": str(data_root / rel),
                    "relative_path": str(rel),
                    "caption": row[CAPTION_COLUMN],
                }
            )

        if not samples:
            raise ValueError(f"Metadata file {self._dataset_file} contains no samples")
        return samples

    # ------------------------------------------------------------------
    # Shard delegation to ShardManager
    # ------------------------------------------------------------------

    @property
    def total_samples(self) -> int:
        return self._shards.total_samples

    @property
    def num_shards(self) -> int:
        return self._shards.num_shards

    @property
    def min_shard_size(self) -> int:
        return self._shards.min_shard_size

    @property
    def shard_state(self) -> tuple[int, int]:
        return self._shards.state

    def advance_shard(self) -> None:
        """Rotate to the next shard. Caller must call ``encode_shard`` before iterating."""
        self._shards.advance()
        self._shard_files = None

    def restore_shard_state(self, cycle: int, shard_idx: int) -> None:
        """Restore shard position from a checkpoint. Caller must call ``encode_shard``."""
        self._shards.restore(cycle, shard_idx)
        self._shard_files = None

    def __len__(self) -> int:
        # When the shard is encoded, use its size — encode_shard may have skipped
        # bad samples, so the list can be smaller than the nominal shard size.
        # PyTorch's RandomSampler re-reads len() on each iter() call, so a new
        # DataLoader iteration picks up the post-skip count.
        if self._shard_files is not None:
            return len(self._shard_files)
        return self._shards.current_shard_size

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self._shard_files is None:
            raise RuntimeError("Shard not encoded yet — call encode_shard() before iterating")
        latents_path, conditions_path = self._shard_files[index]
        return {
            "latents": torch.load(latents_path, map_location="cpu", weights_only=True),
            "conditions": torch.load(conditions_path, map_location="cpu", weights_only=True),
            "idx": index,
        }

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode_shard(
        self,
        vae_encoder: nn.Module,
        text_encoder: nn.Module,
        embeddings_processor: nn.Module,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        """Preprocess the current shard to disk.

        For each sample in the shard, choose the nearest resolution bucket from
        container metadata, then write video latents to
        ``.precomputed/latents/<stem>_<f>x<h>x<w>.pt`` and text embeddings to
        ``.precomputed/conditions/<stem>_<f>x<h>x<w>.pt``. Samples whose output
        files already exist are skipped (idempotent re-visits, even across bucket
        list edits — only videos whose nearest bucket actually changed re-encode).
        Samples that fail to load/encode are skipped with a warning.
        """
        shard_indices = self._shards.current_shard
        shard_num = self._shards.state[1] + 1
        num_shards = self._shards.num_shards
        start = time.monotonic()

        self._latents_dir.mkdir(parents=True, exist_ok=True)
        self._conditions_dir.mkdir(parents=True, exist_ok=True)

        shard_files: list[tuple[Path, Path]] = []
        skipped = 0
        reused = 0
        for global_idx in shard_indices:
            sample = self._samples[global_idx]
            try:
                latents_path, conditions_path, bucket = self._resolve_paths(sample)
                if latents_path.exists() and conditions_path.exists():
                    reused += 1
                else:
                    if not latents_path.exists():
                        self._encode_video_to_disk(
                            sample, latents_path, bucket, vae_encoder, device, dtype
                        )
                    if not conditions_path.exists():
                        self._encode_text_to_disk(
                            sample, conditions_path, text_encoder, embeddings_processor
                        )
            except Exception as e:
                logger.warning(f"Skipping sample {global_idx} ({sample['video_path']}): {e}")
                skipped += 1
                continue
            shard_files.append((latents_path, conditions_path))

        self._shard_files = shard_files
        elapsed = time.monotonic() - start
        info_parts: list[str] = []
        if reused:
            info_parts.append(f"{reused} from cache")
        if skipped:
            info_parts.append(f"{skipped} skipped")
        info_str = f" ({', '.join(info_parts)})" if info_parts else ""
        logger.info(
            f"Preprocessed shard {shard_num}/{num_shards}: {len(shard_files)} samples{info_str} "
            f"→ {self._output_root} in {elapsed:.1f}s"
        )

        if not shard_files:
            raise RuntimeError(
                f"Shard {shard_num}/{num_shards} produced zero preprocessed samples — "
                f"all {len(shard_indices)} samples failed to encode"
            )

    def _resolve_paths(self, sample: dict[str, str]) -> tuple[Path, Path, tuple[int, int, int]]:
        """Pick the nearest bucket from container metadata and return output paths.

        Conditions are bucket-independent in *content* (text embeddings don't depend
        on resolution) but the filename still carries the bucket suffix to keep the
        latents/conditions pairing intact for downstream :class:`PrecomputedDataset`
        consumption.
        """
        video_path = Path(sample["video_path"])
        num_frames, h, w = get_video_metadata(video_path)
        bucket = find_nearest_bucket(num_frames, h, w, self._resolution_buckets)
        suffix = bucket_filename_suffix(bucket)

        rel = Path(sample["relative_path"])
        rel_with_bucket = rel.with_name(f"{rel.stem}_{suffix}.pt")
        latents_path = self._latents_dir / rel_with_bucket
        conditions_path = self._conditions_dir / rel_with_bucket
        return latents_path, conditions_path, bucket

    def _encode_video_to_disk(
        self,
        sample: dict[str, str],
        output_path: Path,
        bucket: tuple[int, int, int],
        vae_encoder: nn.Module,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        video_path = Path(sample["video_path"])
        target_f, target_h, target_w = bucket

        video, fps = read_video(video_path, max_frames=self._max_frames)
        video = resize_and_crop_video(video, target_h, target_w)
        video = video[:target_f]
        video = normalize_video_frames(video)

        # [F, C, H, W] → [1, C, F, H, W]
        video = video.permute(1, 0, 2, 3).unsqueeze(0)

        with torch.inference_mode():
            latents = vae_encoder(video.to(device=device, dtype=dtype))

        _, _, nf, lh, lw = latents.shape
        latent_data = {
            "latents": latents[0].cpu().contiguous(),
            "num_frames": nf,
            "height": lh,
            "width": lw,
            "fps": fps,
        }
        _atomic_save(latent_data, output_path)

    def _encode_text_to_disk(
        self,
        sample: dict[str, str],
        output_path: Path,
        text_encoder: nn.Module,
        embeddings_processor: nn.Module,
    ) -> None:
        with torch.inference_mode():
            hidden_states, mask = text_encoder.encode(sample["caption"], padding_side="left")
            video_embeds, audio_embeds = embeddings_processor.feature_extractor(
                hidden_states, mask, "left"
            )

        data: dict[str, Tensor] = {
            "video_prompt_embeds": video_embeds[0].cpu().contiguous(),
            "prompt_attention_mask": mask[0].cpu().contiguous(),
        }
        if audio_embeds is not None:
            data["audio_prompt_embeds"] = audio_embeds[0].cpu().contiguous()
        _atomic_save(data, output_path)


def _atomic_save(obj: dict[str, Any], path: Path) -> None:
    """Write atomically: write to ``.tmp`` then rename, so concurrent readers never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)
