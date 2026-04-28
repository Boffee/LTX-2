"""GPU memory management utilities — model-agnostic, torch-only.

Two complementary offload strategies:

- :func:`make_block_offloader` — per-block streaming. Use for models
  whose individual blocks fit on GPU but the whole model does not.
  Hooks-based, prefetches upcoming blocks on a secondary CUDA stream,
  supports gradient checkpointing through autograd backward. Returns
  a :class:`BlockStreamingStrategy` composing one
  :class:`BlockStreamer` per ``layers_attr`` path plus a non-block
  :class:`PinnedWeights` plus a :class:`TrainableMover`. For bespoke
  configurations (per-group ``blocks_to_swap``), construct the
  components directly and pass them to :class:`BlockStreamingStrategy`.

- :class:`PinnedWeights` — whole-model pinned-CPU bulk cache. Use for
  models that fit on GPU when active but should be evicted between
  calls (e.g., text encoder during diffusion). One CPU→GPU transfer
  per use; on deactivate, parameter slots are repointed at pinned
  CPU storage and the GPU storage is released by refcount.

Both classes share the underlying per-parameter pinned storage from
:class:`~ltx_core.memory.pinned_buffer.PinnedParamBuffer` (clone + pin
+ optional quanto ``WeightQBytesTensor`` decomposition), so quantized
models work with either.

Both :class:`PinnedWeights` and :class:`BlockStreamingStrategy`
implement the :class:`ModelStrategy` Protocol — the plug-in contract
for storage/placement strategies that :class:`ModelCache` consumes.
New strategies (disk-mmap, NVMe-paged, multi-GPU shard, etc.) just
satisfy the protocol.

All strategies pin in their constructor, so ``cache_bytes`` is final
immediately and :class:`ModelCache` can admit them without a
factory-side ``prepare()`` dance. ``activate()`` then brings
everything to GPU; ``deactivate()`` returns to pinned CPU.

:func:`make_block_offloader` produces a :class:`BlockStreamingStrategy`
that composes (in order):
  1. A non-block :class:`PinnedWeights` with a :class:`SlotOwnership`
     skip filter for everything outside the block list (sibling
     modules + direct parent-module state, e.g. an unembedding head
     or a learnable bias attached to the model root).
  2. A :class:`TrainableMover` for LoRA / adapter weights.
  3. One :class:`BlockStreamer` per ``layers_attr`` path.

Cross-region tied parameters (block ↔ non-block, cross-block, or
mixed trainable/frozen across regions) are detected at construction
and raise — slot-local block streaming cannot preserve such ties; use
whole-model :class:`PinnedWeights` instead.

:class:`ModelCache` manages the cached backing storage of multiple
strategies with LRU eviction, an active-set with refcounted leases, and
transactional admission. See its docstring for design notes.

Compatibility
-------------
- **``torch.compile`` is not supported** for managed modules.
  :class:`PinnedWeights` and :class:`BlockStreamer` swap parameter
  slots (``module._parameters[leaf] = new_param``) on every
  activate/deactivate, and :class:`BlockStreamer` registers
  forward-pre hooks that mutate slots on every block call. Both
  invalidate the tensor-identity assumptions ``torch.compile`` makes
  about its trace, producing recompiles or graph breaks at best,
  silent miscompilation at worst. Compile the surrounding code if
  needed, but never compile a module that's wrapped by these
  strategies.
- **Wrap before DDP/FSDP**, not after. Those wrappers manage parameter
  storage themselves and conflict with the slot-swap pattern.
- **Single-thread / sequential.** No internal locking; concurrent use
  on the same strategy or cache is undefined behavior.

Designed as a self-contained, model-agnostic library — pipeline-specific
glue (e.g. monkey-patching upstream pipeline classes to route
construction through the cache) belongs in the consumer, not here.
"""

from .block_compose import (
    BlockStreamingStrategy,
    TrainableMover,
    make_block_offloader,
)
from .block_streamer import BlockStreamer
from .model_cache import (
    ActivationError,
    DuplicateModelKeyError,
    ModelCache,
    ModelCacheError,
    ModelInUseError,
    ModelNotRegisteredError,
    ModelSpec,
    ModelTooLargeError,
)
from .pinned_weights import PinnedWeights
from .strategy import ModelStrategy, SlotOwnership

# `ModelCacheSnapshot`, `ModelCacheStats`, and `ModelInfo` are observability
# types — used by callers who introspect cache state, not the typical
# acquire/use path. Import them directly from
# `ltx_core.memory.model_cache` when needed.

__all__ = [
    "ActivationError",
    "BlockStreamer",
    "BlockStreamingStrategy",
    "DuplicateModelKeyError",
    "ModelCache",
    "ModelCacheError",
    "ModelInUseError",
    "ModelNotRegisteredError",
    "ModelSpec",
    "ModelStrategy",
    "ModelTooLargeError",
    "PinnedWeights",
    "SlotOwnership",
    "TrainableMover",
    "make_block_offloader",
]
