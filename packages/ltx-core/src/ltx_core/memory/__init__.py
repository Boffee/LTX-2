"""GPU memory management utilities — model-agnostic, torch-only.

Two complementary offload strategies:

- :class:`BlockOffloader` — per-block streaming. Use for models whose
  individual blocks fit on GPU but the whole model does not. Hooks-based,
  prefetches upcoming blocks on a secondary CUDA stream, supports
  gradient checkpointing through autograd backward.

- :class:`PinnedWeights` — whole-model pinned-CPU bulk cache. Use for
  models that fit on GPU when active but should be evicted between
  calls (e.g., text encoder during diffusion). One CPU→GPU transfer
  per use; on exit, parameters are repointed back at the pinned CPU
  storage and the GPU storage is released by refcount.

Both classes share the underlying per-parameter pinned storage from
:class:`~ltx_core.memory.pinned_buffer.PinnedParamBuffer` (clone + pin
+ optional quanto ``WeightQBytesTensor`` decomposition), so quantized
models work with either.

Designed to be a self-contained subpackage so it can be lifted out
into its own library when a second consumer appears (no LTX imports
here).
"""

from ltx_core.memory.block_offloader import BlockOffloader, TrainingBlockOffloader  # noqa: F401
from ltx_core.memory.pinned_weights import PinnedWeights

__all__ = [
    "BlockOffloader",
    "PinnedWeights",
]
