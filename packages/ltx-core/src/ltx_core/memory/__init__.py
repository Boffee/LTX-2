"""GPU memory management utilities — model-agnostic, torch-only.

Two complementary offload strategies:

- :class:`BlockOffloader` — per-block streaming. Use for models whose
  individual blocks fit on GPU but the whole model does not. Hooks-based,
  prefetches upcoming blocks on a secondary CUDA stream, supports
  gradient checkpointing through autograd backward.

- :class:`PinnedWeights` — whole-model pinned-CPU bulk cache. Use for
  models that fit on GPU when active but should be evicted between
  calls (e.g., text encoder during diffusion). One CPU→GPU transfer
  per use; on deactivate, parameter slots are repointed at pinned
  CPU storage and the GPU storage is released by refcount.

Both classes share the underlying per-parameter pinned storage from
:class:`~ltx_core.memory.pinned_buffer.PinnedParamBuffer` (clone + pin
+ optional quanto ``WeightQBytesTensor`` decomposition), so quantized
models work with either.

Both :class:`PinnedWeights` and :class:`BlockOffloader` implement the
:class:`ModelStrategy` Protocol — the plug-in contract for
storage/placement strategies that :class:`ModelCache` consumes. New
strategies (disk-mmap, NVMe-paged, multi-GPU shard, etc.) just satisfy
the protocol.

:class:`BlockOffloader` defaults to ``auto_setup=True`` which runs
``prepare(); activate()`` immediately so long-lived training callers
don't have to phase the lifecycle by hand. For
:class:`ModelCache` integration, factories pass ``auto_setup=False``
and call ``prepare()`` before returning the handle so the cache reads
the correct ``cache_bytes`` immediately::

    def factory():
        off = BlockOffloader(..., auto_setup=False)
        off.prepare()
        return off

The prepared state is fully GPU-inactive: block frozen weights live in
the per-block pinned store, non-block frozen siblings (patchifier,
output projection, norms, etc.) are pinned via composed
``PinnedWeights``, and trainable params sit on CPU. ``activate()``
brings everything to GPU; ``deactivate()`` returns it to pinned CPU.
Cross-region tied parameters (block ↔ non-block, cross-block, or
mixed trainable/frozen across regions) are detected at ``prepare()``
and raise — slot-local block streaming cannot preserve such ties;
use whole-model ``PinnedWeights`` instead.

:class:`ModelCache` manages the cached backing storage of multiple
strategies with LRU eviction, an active-set with refcounted leases, and
transactional admission. See its docstring for design notes.

Compatibility
-------------
- **``torch.compile`` is not supported** for managed modules.
  :class:`PinnedWeights` and :class:`BlockOffloader` swap parameter
  slots (``module._parameters[leaf] = new_param``) on every
  activate/deactivate, and :class:`BlockOffloader` registers
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

Designed as a self-contained subpackage so it can be lifted out into
its own library when a second consumer appears. The core strategy and
cache modules avoid pipeline imports; the optional
:mod:`~ltx_core.memory.pipeline_install` integration module imports
``ltx_pipelines`` lazily at install time and is the only piece tied
to the LTX repo layout.
"""

from ltx_core.memory.block_offloader import BlockOffloader
from ltx_core.memory.model_cache import (
    ActivationError,
    DuplicateModelKeyError,
    ModelCache,
    ModelCacheError,
    ModelEvictionError,
    ModelInUseError,
    ModelNotRegisteredError,
    ModelSpec,
    ModelTooLargeError,
)
from ltx_core.memory.pinned_weights import PinnedWeights
from ltx_core.memory.strategy import ModelStrategy

# `ModelCacheSnapshot`, `ModelCacheStats`, and `ModelInfo` are observability
# types — used by callers who introspect cache state, not the typical
# acquire/use path. Import them directly from
# `ltx_core.memory.model_cache` when needed.

__all__ = [
    "ActivationError",
    "BlockOffloader",
    "DuplicateModelKeyError",
    "ModelCache",
    "ModelCacheError",
    "ModelEvictionError",
    "ModelInUseError",
    "ModelNotRegisteredError",
    "ModelSpec",
    "ModelStrategy",
    "ModelTooLargeError",
    "PinnedWeights",
]
