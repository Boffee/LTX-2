"""Video preprocessing utilities shared between offline and online encoding.

Resolution buckets use the canonical `(frames, height, width)` tuple format
throughout this module, matching the convention in `scripts/process_videos.py`.
The string format used at the CLI and in config is `"WxHxF"` (width x height x frames).
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import Tensor
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import crop, resize

VAE_SPATIAL_FACTOR = 32
VAE_TEMPORAL_FACTOR = 8


def parse_resolution_buckets(resolution_buckets_str: str) -> list[tuple[int, int, int]]:
    """Parse resolution buckets from string format to list of (frames, height, width) tuples.

    Input format: ``"WxHxF"`` or ``"WxHxF;WxHxF;..."`` (width x height x frames).
    Output tuples are ordered ``(frames, height, width)`` for consistency with downstream code.
    """
    buckets: list[tuple[int, int, int]] = []
    for bucket_str in resolution_buckets_str.split(";"):
        parts = bucket_str.strip().split("x")
        if len(parts) != 3:
            raise ValueError(f"Invalid bucket '{bucket_str}': expected format 'WxHxF'")
        w, h, f = (int(p) for p in parts)

        if w % VAE_SPATIAL_FACTOR != 0 or h % VAE_SPATIAL_FACTOR != 0:
            raise ValueError(
                f"Bucket {bucket_str}: width and height must be multiples of {VAE_SPATIAL_FACTOR}, got {w}x{h}"
            )
        if f % VAE_TEMPORAL_FACTOR != 1:
            raise ValueError(
                f"Bucket {bucket_str}: number of frames must satisfy frames % {VAE_TEMPORAL_FACTOR} == 1, got {f}"
            )

        buckets.append((f, h, w))
    return buckets


def find_nearest_bucket(
    num_frames: int,
    height: int,
    width: int,
    buckets: list[tuple[int, int, int]],
) -> tuple[int, int, int]:
    """Find the nearest ``(frames, height, width)`` bucket for a video's dimensions.

    Prefers buckets with matching aspect ratio, then more frames, then larger area.
    Only considers buckets whose frame count is <= the video's available frames.
    """
    relevant = [b for b in buckets if b[0] <= num_frames]
    if not relevant:
        raise ValueError(
            f"No resolution buckets have <= {num_frames} frames. Available: {buckets}"
        )

    def distance(bucket: tuple[int, int, int]) -> tuple:
        bf, bh, bw = bucket
        return (
            abs(math.log(width / height) - math.log(bw / bh)),
            -bf,
            -(bh * bw),
        )

    return min(relevant, key=distance)


def resize_and_crop_video(video: Tensor, target_h: int, target_w: int) -> Tensor:
    """Resize and center-crop video frames ``[F, C, H, W]`` to target dimensions."""
    h, w = video.shape[2], video.shape[3]
    if w / h > target_w / target_h:
        # Wider than target — scale by height
        new_w = int(w * target_h / h)
        video = resize(video, [target_h, new_w], interpolation=InterpolationMode.BICUBIC)
    else:
        # Taller than target — scale by width
        new_h = int(h * target_w / w)
        video = resize(video, [new_h, target_w], interpolation=InterpolationMode.BICUBIC)

    h, w = video.shape[2], video.shape[3]
    top = (h - target_h) // 2
    left = (w - target_w) // 2
    return crop(video, top, left, target_h, target_w)


def normalize_video_frames(video: Tensor) -> Tensor:
    """Clamp to ``[0, 1]`` and normalize to ``[-1, 1]``. Input/output shape: ``[F, C, H, W]``."""
    return video.clamp(0, 1) * 2 - 1


def max_frames_in_buckets(buckets: list[tuple[int, int, int]]) -> int:
    """Return the maximum frame count across all buckets."""
    return max(b[0] for b in buckets)


def buckets_fingerprint(buckets: list[tuple[int, int, int]]) -> str:
    """Stable string fingerprint of a bucket list, for use in cache keys."""
    sorted_buckets = sorted(buckets)
    return ";".join(f"{f}x{h}x{w}" for f, h, w in sorted_buckets)


def bucket_filename_suffix(bucket: tuple[int, int, int]) -> str:
    """Format a single ``(frames, height, width)`` bucket as a filename suffix.

    Used to key per-video latent cache files by the *chosen* bucket rather than
    the entire bucket list. With this scheme, adding/removing buckets only forces
    re-encoding for videos whose nearest bucket actually changed; videos that
    still pick the same bucket are served from cache.
    """
    f, h, w = bucket
    return f"{f}x{h}x{w}"
