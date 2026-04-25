"""Back-compat shim — the offloader has moved to :mod:`ltx_core.memory`.

This module previously held ``BlockOffloader``/``TrainingBlockOffloader``
directly. They now live in ``ltx_core.memory.streaming``; this file
re-exports them so any external import like
``from ltx_core.block_offloader import BlockOffloader`` keeps working.
New code should import from ``ltx_core.memory``.
"""

from ltx_core.memory.streaming import BlockOffloader, TrainingBlockOffloader  # noqa: F401

__all__ = ["BlockOffloader", "TrainingBlockOffloader"]
