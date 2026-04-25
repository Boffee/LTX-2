"""Public protocol for model storage/placement strategies.

A :class:`ModelStrategy` owns one model plus the resources needed to
make it usable for compute (pinned CPU buffers, GPU slot pools, forward
hooks, mmap regions, etc.). It is the plug-in contract used by
:class:`~ltx_core.memory.model_cache.ModelCache` so the manager does not
need to know how any particular strategy works.

Implementations in this package: :class:`~ltx_core.memory.PinnedWeights`
(whole-model bulk DMA between pinned CPU and GPU). A
:class:`~ltx_core.memory.BlockOffloader` (block-level streaming for
models too big for GPU) implementation is planned once its lifecycle is
split into the required ``activate`` / ``deactivate`` / ``close``
methods. Future strategies (disk-mmap, NVMe-paged, multi-GPU shard)
just have to satisfy this protocol.

Lifecycle
---------
``__init__`` (constructs and pins backing storage) →
``activate()`` (make model usable, returns the ``nn.Module``) →
``deactivate()`` (release transient compute resources, keep
``cache_bytes`` resident) → ``close()`` (release ``cache_bytes``;
the wrapped model is unusable afterward).

``close()`` is idempotent. ``activate()/deactivate()`` may be repeated
between construction and ``close()``. The strategy is also a context
manager: ``with strategy as model: ...`` is equivalent to
``activate()`` / ``deactivate()``.
"""

from __future__ import annotations

from types import TracebackType
from typing import Protocol, runtime_checkable

from torch import nn


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

    @property
    def closed(self) -> bool:
        """``True`` after :meth:`close`. A closed strategy is unusable."""
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
        raising ``deactivate()`` as a poisoned strategy and discards
        the entry (calls :meth:`close` on it) since the strategy's
        internal state is unknown after the failure.
        """
        ...

    def close(self) -> None:
        """Release ``cache_bytes``. Idempotent.

        After ``close()`` returns, the wrapped model may no longer be
        usable (typically its parameters have been moved to the ``meta``
        device to break storage references). Callers must request a
        fresh strategy to use the model again.
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
