"""Online per-shard encoding dataset for training without offline preprocessing.

Encodes raw videos + captions at training time, one shard at a time:
- Video latents: encoded with VAE, cached to disk (expensive, only done once)
- Text embeddings: encoded with text encoder, kept in memory only (avoids disk I/O)

On shard rotation, text embeddings are discarded and re-encoded for the next shard.
Video latents are loaded from disk cache on subsequent visits.
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
LATENT_CACHE_DIR_NAME = ".latent_cache"


class OnlineEncodingDataset(Dataset):
    """Dataset that encodes raw videos + captions per shard at training time.

    Data source is a metadata file (CSV/JSON/JSONL) with columns ``media_path``
    (video paths) and ``caption`` (text). Samples are split into shards managed
    by :class:`ShardManager`. The dataset does not load or encode anything until
    :meth:`encode_shard` is called — the trainer is responsible for calling it at
    startup and after each shard rotation.

    Parameters
    ----------
    dataset_file:
        Path to CSV/JSON/JSONL metadata file.
    resolution_buckets:
        List of ``(frames, height, width)`` tuples. Each video is matched to the
        nearest bucket by aspect ratio, then resized and cropped to that exact size.
    shard_size:
        Samples per shard. ``None`` = encode everything at once.
    seed:
        Random seed for shard shuffling.
    """

    def __init__(
        self,
        dataset_file: str | Path,
        resolution_buckets: list[tuple[int, int, int]],
        shard_size: int | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self._dataset_file = Path(dataset_file)
        self._resolution_buckets = resolution_buckets
        self._latent_cache_dir = self._dataset_file.parent / LATENT_CACHE_DIR_NAME
        self._max_frames = max_frames_in_buckets(resolution_buckets)

        self._samples = self._load_metadata()
        self._memory_cache: dict[int, dict[str, Any]] | None = None
        self._shards = ShardManager(total_samples=len(self._samples), shard_size=shard_size, seed=seed)

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
            video_path = data_root / Path(row[VIDEO_COLUMN].strip())
            samples.append({"video_path": str(video_path), "caption": row[CAPTION_COLUMN]})

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
        """Rotate to next shard. Caller must call ``encode_shard`` to load data."""
        self._shards.advance()
        self._memory_cache = None

    def restore_shard_state(self, cycle: int, shard_idx: int) -> None:
        """Restore shard position from checkpoint. Caller must call ``encode_shard`` to load data."""
        self._shards.restore(cycle, shard_idx)
        self._memory_cache = None

    def __len__(self) -> int:
        # When the cache is populated, use its size — encode_shard() may have skipped
        # bad samples, so the cache can be smaller than the nominal shard size.
        # PyTorch's RandomSampler re-reads len() on each iter() call, so a new
        # DataLoader iteration will pick up the post-skip count.
        if self._memory_cache is not None:
            return len(self._memory_cache)
        return self._shards.current_shard_size

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self._memory_cache is None:
            raise RuntimeError("Shard not encoded yet — call encode_shard() before iterating")
        return self._memory_cache[index]

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
        """Encode the current shard's videos + captions.

        Video latents are cached to disk (only encoded once per bucket config).
        Text embeddings are computed fresh and kept in memory only.
        Samples that fail to load/encode are skipped with a warning.
        """
        shard_indices = self._shards.current_shard
        shard_num = self._shards.state[1] + 1
        num_shards = self._shards.num_shards
        start = time.monotonic()

        cache: dict[int, dict[str, Any]] = {}
        skipped = 0
        local_idx = 0
        for global_idx in shard_indices:
            sample = self._samples[global_idx]
            try:
                latent_data = self._encode_or_load_video(sample, vae_encoder, device, dtype)
                text_data = self._encode_text(sample, text_encoder, embeddings_processor)
            except Exception as e:
                logger.warning(f"Skipping sample {global_idx} ({sample['video_path']}): {e}")
                skipped += 1
                continue
            cache[local_idx] = {
                "latents": latent_data,
                "conditions": text_data,
                "idx": global_idx,
            }
            local_idx += 1

        self._memory_cache = cache
        elapsed = time.monotonic() - start
        skip_info = f", {skipped} skipped" if skipped else ""
        logger.info(
            f"Encoded shard {shard_num}/{num_shards} ({len(cache)} samples{skip_info}) in {elapsed:.1f}s"
        )

        if not cache:
            raise RuntimeError(
                f"Shard {shard_num}/{num_shards} produced zero encoded samples — "
                f"all {len(shard_indices)} samples failed to encode"
            )

    def _encode_or_load_video(
        self,
        sample: dict[str, str],
        vae_encoder: nn.Module,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, Any]:
        """Encode video with VAE, or load from disk cache if already encoded.

        Reads container metadata (no frame decode) to pick the bucket, then keys
        the cache by ``(video_path, chosen_bucket)``. Adding or removing buckets
        only re-encodes videos whose nearest bucket actually changed.
        """
        video_path = Path(sample["video_path"])

        num_frames, h, w = get_video_metadata(video_path)
        bucket = find_nearest_bucket(num_frames, h, w, self._resolution_buckets)
        cache_path = self._get_latent_cache_path(video_path, bucket)

        if cache_path.exists():
            return torch.load(cache_path, map_location="cpu", weights_only=True)

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

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # Write atomically: write to .tmp then rename, so concurrent readers never
        # see a partially-written file.
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        torch.save(latent_data, tmp_path)
        tmp_path.replace(cache_path)
        return latent_data

    def _encode_text(
        self,
        sample: dict[str, str],
        text_encoder: nn.Module,
        embeddings_processor: nn.Module,
    ) -> dict[str, Tensor]:
        """Encode caption into text features. Kept in memory only."""
        with torch.inference_mode():
            hidden_states, mask = text_encoder.encode(sample["caption"], padding_side="left")
            video_embeds, audio_embeds = embeddings_processor.feature_extractor(hidden_states, mask, "left")

        result: dict[str, Tensor] = {
            "video_prompt_embeds": video_embeds[0].cpu().contiguous(),
            "prompt_attention_mask": mask[0].cpu().contiguous(),
        }
        if audio_embeds is not None:
            result["audio_prompt_embeds"] = audio_embeds[0].cpu().contiguous()
        return result

    # ------------------------------------------------------------------
    # Cache keying
    # ------------------------------------------------------------------

    def _get_latent_cache_path(self, video_path: Path, bucket: tuple[int, int, int]) -> Path:
        """Build a cache path keyed by video path AND the *chosen* bucket.

        The bucket dimensions live in the filename (``<stem>_<f>x<h>x<w>.pt``), so
        only videos whose nearest bucket changed need to be re-encoded when the
        bucket list is edited. Cached files for buckets that are still in the list
        keep being reused.
        """
        suffix = bucket_filename_suffix(bucket)
        return self._latent_cache_dir / f"{video_path.stem}_{suffix}.pt"
