"""Shared shard state management for sharded datasets.

Handles the shuffling, grouping, rotation, and resume state for datasets
that process samples one shard at a time. Data loading (from disk or via
online encoding) is the responsibility of the dataset that owns a
:class:`ShardManager` — this class manages state only.
"""

from __future__ import annotations

import random

from ltx_trainer import logger


class ShardManager:
    """Manages shard state and rotation for sharded datasets.

    Samples are grouped into shards of ``shard_size`` at initialization. Shards
    rotate in order via :meth:`advance` and wrap around with a reshuffle at the
    end of a full cycle. Each cycle uses a deterministic seed (``seed + cycle``)
    so shard composition is reproducible on resume.

    Parameters
    ----------
    total_samples:
        Total number of samples in the dataset.
    shard_size:
        Samples per shard. ``None`` means one shard with all samples (no rotation).
    seed:
        Base seed used to shuffle samples into shards.
    """

    def __init__(self, total_samples: int, shard_size: int | None, seed: int = 42) -> None:
        self._total_samples = total_samples
        self._shard_size = shard_size
        self._seed = seed
        self._shard_groups: list[list[int]] = []
        self._current_shard_idx: int = 0
        self._shard_cycle: int = 0
        self._setup_shards()

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def total_samples(self) -> int:
        return self._total_samples

    @property
    def num_shards(self) -> int:
        return len(self._shard_groups)

    @property
    def current_shard(self) -> list[int]:
        """Sample indices in the currently active shard."""
        return self._shard_groups[self._current_shard_idx]

    @property
    def current_shard_size(self) -> int:
        return len(self._shard_groups[self._current_shard_idx])

    @property
    def min_shard_size(self) -> int:
        """Size of the smallest shard (useful for validation)."""
        return min(len(g) for g in self._shard_groups)

    @property
    def state(self) -> tuple[int, int]:
        """Current (cycle, shard_idx) — suitable for checkpointing."""
        return (self._shard_cycle, self._current_shard_idx)

    def has_sharding(self) -> bool:
        """True if there are multiple shards to rotate through."""
        return len(self._shard_groups) > 1

    # ------------------------------------------------------------------
    # Transitions
    # ------------------------------------------------------------------

    def advance(self) -> None:
        """Advance to the next shard. Wraps around with a reshuffle at the end of a cycle."""
        if not self.has_sharding():
            return

        self._current_shard_idx += 1
        if self._current_shard_idx >= len(self._shard_groups):
            self._current_shard_idx = 0
            self._shard_cycle += 1
            self._setup_shards()
            logger.info("All shards visited, reshuffling for next cycle")

    def restore(self, cycle: int, shard_idx: int) -> None:
        """Restore shard state from a checkpoint."""
        if not self.has_sharding():
            return
        self._shard_cycle = cycle
        self._setup_shards()
        self._current_shard_idx = min(shard_idx, len(self._shard_groups) - 1)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_shards(self) -> None:
        indices = list(range(self._total_samples))

        if self._shard_size is not None and self._shard_size < self._total_samples:
            rng = random.Random(self._seed + self._shard_cycle)
            rng.shuffle(indices)
            self._shard_groups = [
                indices[i : i + self._shard_size] for i in range(0, self._total_samples, self._shard_size)
            ]
            # Merge an undersized trailing shard into the previous one so every
            # shard is at least shard_size (avoids drop_last skipping samples).
            if len(self._shard_groups) > 1 and len(self._shard_groups[-1]) < self._shard_size:
                self._shard_groups[-2].extend(self._shard_groups.pop())
        else:
            self._shard_groups = [indices]

        self._current_shard_idx = 0
