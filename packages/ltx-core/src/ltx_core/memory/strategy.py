"""Public protocol for model storage/placement strategies.

A :class:`ModelStrategy` owns one model plus the resources needed to
make it usable for compute (pinned CPU buffers, GPU slot pools, forward
hooks, mmap regions, etc.). It is the plug-in contract used by
:class:`~block_offload.model_cache.ModelCache` so the manager does not
need to know how any particular strategy works.

Implementations in this package: :class:`~block_offload.PinnedWeights`
(whole-model bulk DMA between pinned CPU and GPU) and
:func:`~block_offload.make_block_offloader` (block-level streaming for
models too big for GPU). Future strategies (disk-mmap, NVMe-paged,
multi-GPU shard) just have to satisfy this protocol.

Lifecycle
---------
``__init__`` sets up backing storage (pinning, etc.) so
``cache_bytes`` is final immediately and the strategy is ready for
:class:`~block_offload.model_cache.ModelCache` admission →
``activate()`` (make model usable, returns the ``nn.Module``) →
``deactivate()`` (release transient compute resources, keep
``cache_bytes`` resident).

``activate()/deactivate()`` may be repeated as many times as you
want. The strategy is also a context manager:
``with strategy as model: ...`` is equivalent to ``activate()`` /
``deactivate()``.

There is no ``close()``. To release ``cache_bytes`` (typically
pinned host memory), drop the strategy reference (and the model
reference if you don't need it anymore). Python's refcount-based
GC frees pinned tensors immediately. Strategies release what they
own; ownership of the user's model is the user's concern.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import TracebackType
from typing import Literal, Protocol, runtime_checkable

from torch import nn


@dataclass(frozen=True, slots=True)
class SlotOwnership:
    """Identifies a parameter or buffer slot in the model tree by
    ``(parent_module, leaf_name, kind)``.

    Used as a slot-skip filter when one strategy manages a subset of
    a model's slots and a second strategy needs to ignore them. Unlike
    ``id(param)`` / ``id(buffer)``, this identity survives
    ``module._parameters[leaf] = new_param`` swaps — the parent module
    and leaf name are stable even when the Parameter/buffer object at
    that slot changes. That decouples filter consumers from
    construction order: a strategy can be built with the filter at any
    time, before or after the producing strategy has mutated slots.

    ``parent_id`` is ``id()`` of the parent module, which is stable for
    the module's lifetime and unique per submodule (Python guarantee
    while a reference is held).
    """

    parent_id: int
    leaf: str
    kind: Literal["param", "buffer"]


@runtime_checkable
class ModelStrategy(Protocol):
    """Storage/placement strategy for one model.

    ``cache_bytes`` is the only resource the manager budgets; everything
    else (GPU memory, hooks, executor threads) is owned by the strategy
    and not visible to callers.

    The ``@runtime_checkable`` decoration enables ``isinstance(x,
    ModelStrategy)`` for sanity checks and tests, but the check is
    structural and only verifies attribute *presence*, not signatures
    or types — and most context managers already provide
    ``__enter__``/``__exit__``, so passing this check is necessary but
    not sufficient. Treat it as a weak guard, not a contract verifier.
    """

    @property
    def cache_bytes(self) -> int:
        """Bytes charged against ``ModelCache.max_cache_bytes``.

        Typically pinned host memory, but a strategy may report any
        resource it wants the cache to budget against — staging buffers,
        mmap regions, etc. Strategies that don't consume cache budget
        (e.g. an "always on GPU" passthrough) should return 0.
        """
        ...

    def activate(self) -> nn.Module:
        """Make the model usable for compute and return the module to call.

        Implementations may move weights to GPU, allocate a slot pool,
        register forward hooks, install an mmap, or do nothing for
        always-resident strategies. Not necessarily re-entrant — call
        :meth:`deactivate` before activating again.
        """
        ...

    def deactivate(self) -> None:
        """Undo :meth:`activate`. ``cache_bytes`` remains held.

        Should be infallible under normal use: the cache treats a
        raising ``deactivate()`` as a poisoned strategy and drops it
        (without further cleanup attempts) since the strategy's
        internal state is unknown after the failure. After deactivate,
        the caller drops the strategy reference to release pinned
        memory — there is no separate ``close()`` step.
        """
        ...

    def __enter__(self) -> nn.Module:
        """Equivalent to :meth:`activate`."""
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        """Equivalent to :meth:`deactivate`."""
        ...
