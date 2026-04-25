"""LRU model cache over :class:`ModelStrategy` plug-ins.

Manages cached pinned-CPU (or other handle-owned) backing storage for
multiple independent models. When a new model needs more cache than is
free, the least-recently-used inactive entries are evicted until room is
available. Active entries (currently inside an ``acquire()`` /
``use()`` context) are never evicted.

Design highlights
-----------------
- **Strategy-agnostic.** The cache only talks to the
  :class:`ModelStrategy` protocol — three lifecycle methods plus
  ``cache_bytes`` accounting. Pluggable: today
  :class:`PinnedWeights` only; tomorrow ``BlockOffloader`` (after its
  lifecycle split), disk-mmap, NVMe-paged, multi-GPU shard.
- **Active-set with refcount.** Multiple keys can be active
  simultaneously (e.g. text encoder and embedding processor in the
  same call), and the same key can be acquired re-entrantly (refcount
  bump, the underlying strategy is not re-activated).
- **Transactional admission, with one caveat.** Activation failure on
  a freshly-built handle drops the poisoned entry, runs ``close()``,
  and propagates :class:`ActivationError`. Close failure during
  eviction propagates :class:`ModelEvictionError` rather than silently
  corrupting state. **Factory failure** preserves the registration but
  leaves any pre-eviction *committed* — the cache evicts inactive LRU
  entries to give the factory a predictable host-memory budget for
  pinning, and those evictions are not rolled back if the factory
  raises (rolling back would mean re-pinning the just-released
  weights, which can OOM the host allocator). The cache stays
  internally consistent; the cost is some warm cached entries
  disappearing.
- **No GPU budget.** The cache only enforces ``max_cache_bytes``
  (typically pinned host memory). Concurrent active models share the
  GPU at the caller's risk.
- **Single-thread.** No locking. Sequential callers only.

Inspired by ComfyUI's ``model_management.LoadedModel`` /
``current_loaded_models`` and accelerate's hook lifecycle, but
instance-owned (not global) so it's library-friendly and embeddable.
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
from collections import OrderedDict
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from ltx_core.memory.strategy import ModelStrategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Registration spec for a cached model.

    ``key`` is the cache identity — caller-chosen and must include any
    construction inputs that affect the resulting weights or structure
    (checkpoint hash, dtype, quantization config, LoRA stack, etc.). The
    cache does not introspect the factory; calling ``use()`` with the
    same key but a different factory silently returns the cached entry.

    ``estimated_cache_bytes`` is used to evict before building. The
    cache reconciles against the actual ``handle.cache_bytes`` after
    construction; if the actual exceeds the estimate enough to overflow
    the budget, the cache evicts more or rejects the admission.

    ``label`` is an optional human-readable name surfaced in logs and
    snapshots; defaults to ``None`` if omitted (in which case displays
    fall back to ``key``).
    """

    key: str
    estimated_cache_bytes: int
    factory: Callable[[], ModelStrategy]
    label: str | None = None


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """Per-key snapshot returned by :meth:`ModelCache.info`."""

    key: str
    label: str | None
    estimated_cache_bytes: int
    cache_bytes: int | None  # None when not built (registered but never used)
    cached: bool
    active_count: int


@dataclass
class ModelCacheStats:
    """Mutable counters tracking cache activity over its lifetime."""

    hits: int = 0
    misses: int = 0
    builds: int = 0
    evictions: int = 0
    bytes_evicted: int = 0
    factory_errors: int = 0
    activation_errors: int = 0
    close_errors: int = 0
    peak_cache_bytes: int = 0


@dataclass(frozen=True, slots=True)
class ModelCacheSnapshot:
    """Detached point-in-time view of the cache.

    The dataclass itself is frozen and ``stats`` is deep-copied at
    capture time so subsequent cache activity does not mutate it. The
    nested ``active_refcounts`` dict and ``stats`` instance are not
    additionally frozen, so callers should not mutate them.
    """

    max_cache_bytes: int
    used_cache_bytes: int
    registered_keys: tuple[str, ...]
    cached_keys_lru_to_mru: tuple[str, ...]
    active_refcounts: dict[str, int]
    stats: ModelCacheStats


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ModelCacheError(RuntimeError):
    """Base for all cache errors."""


class ModelNotRegisteredError(ModelCacheError):
    """``use(str)`` called for a key that has no registration."""


class DuplicateModelKeyError(ModelCacheError):
    """``register(spec)`` called for a key that is already registered
    (without ``replace=True``)."""


class ModelTooLargeError(ModelCacheError):
    """Even after evicting all inactive entries, the requested model
    cannot fit in the budget. Carries enough context to debug."""

    def __init__(
        self,
        *,
        key: str,
        required: int,
        used: int,
        limit: int,
        active_refcounts: dict[str, int],
    ) -> None:
        short = used + required - limit
        super().__init__(
            f"Cannot admit {key!r}: needs {required} bytes, "
            f"{used}/{limit} already used (short by {short}). "
            f"Active non-evictable entries: {active_refcounts}"
        )
        self.key = key
        self.required = required
        self.used = used
        self.limit = limit
        self.active_refcounts = active_refcounts


class ModelInUseError(ModelCacheError):
    """``evict(key)`` or ``clear()`` called while one or more entries
    are active."""


class ModelEvictionError(ModelCacheError):
    """An entry's ``close()`` raised during eviction. Cache state is
    consistent (the entry is removed) but the underlying strategy may
    have leaked resources."""


class ActivationError(ModelCacheError):
    """A strategy's ``activate()`` raised. For freshly-built handles,
    the cache discards the poisoned entry. For previously-cached
    entries, the entry remains cached but the active refcount is rolled
    back."""


# ---------------------------------------------------------------------------
# Internal entry state
# ---------------------------------------------------------------------------


@dataclass
class _Entry:
    spec: ModelSpec
    handle: ModelStrategy | None = None
    cache_bytes: int = 0           # actual, post-build
    active_count: int = 0
    active_module: nn.Module | None = None  # cached for re-entrant acquire


# ---------------------------------------------------------------------------
# ModelCache
# ---------------------------------------------------------------------------


class ModelCache:
    """LRU pool over :class:`ModelStrategy` instances.

    Parameters
    ----------
    max_cache_bytes:
        Total budget for cached strategy backing storage (typically
        pinned host memory). Must be ≥ 0.
    empty_host_cache:
        Optional callback invoked after every successful eviction (and
        after closing a rejected newly-built handle) to flush PyTorch's
        ``CachingHostAllocator`` so freed pinned pages return to the OS.
        If ``None`` and CUDA is available, defaults to
        ``torch._C._host_emptyCache`` when present. Pass a no-op
        callable to disable.
    """

    def __init__(
        self,
        max_cache_bytes: int,
        *,
        empty_host_cache: Callable[[], None] | None = None,
    ) -> None:
        if max_cache_bytes < 0:
            raise ValueError(f"max_cache_bytes must be >= 0, got {max_cache_bytes}")
        self._max_cache_bytes = max_cache_bytes
        self._empty_host_cache = self._resolve_host_cache_cb(empty_host_cache)
        # Insertion order is meaningful for snapshots only; LRU is tracked
        # separately so eviction order doesn't depend on registration order.
        self._entries: dict[str, _Entry] = {}
        # Inactive entries only. MRU at the right end.
        self._lru: OrderedDict[str, None] = OrderedDict()
        self._used_bytes = 0
        self._stats = ModelCacheStats()

    @staticmethod
    def _resolve_host_cache_cb(cb: Callable[[], None] | None) -> Callable[[], None] | None:
        if cb is not None:
            return cb
        # Best-effort default: PyTorch's CachingHostAllocator flush is
        # only present when CUDA is available and only on torch >= 2.x.
        host_empty: Any = getattr(torch._C, "_host_emptyCache", None)
        if callable(host_empty) and torch.cuda.is_available():
            return host_empty
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def max_cache_bytes(self) -> int:
        return self._max_cache_bytes

    @property
    def used_cache_bytes(self) -> int:
        return self._used_bytes

    def register(self, spec: ModelSpec, *, replace: bool = False) -> None:
        """Register a lazy model factory without building it.

        Idempotent only with ``replace=True``: re-registering an
        existing key without ``replace`` raises
        :class:`DuplicateModelKeyError`. With ``replace=True``, any
        existing cached entry is evicted first (which raises
        :class:`ModelInUseError` if the key is active).
        """
        if spec.estimated_cache_bytes < 0:
            raise ValueError(
                f"spec.estimated_cache_bytes must be >= 0, got {spec.estimated_cache_bytes}"
            )
        if spec.key in self._entries:
            if not replace:
                raise DuplicateModelKeyError(f"{spec.key!r} is already registered")
            self._evict_for_replace(spec.key)
        self._entries[spec.key] = _Entry(spec=spec)

    def unregister(self, key: str, *, evict: bool = True) -> None:
        """Drop a registration. If a built handle exists and
        ``evict=True`` (default), close it; otherwise raise
        :class:`ModelInUseError` if it's cached or active."""
        entry = self._entries.get(key)
        if entry is None:
            return
        if entry.active_count > 0:
            raise ModelInUseError(
                f"cannot unregister active model {key!r} (refcount={entry.active_count})"
            )
        if entry.handle is not None:
            if not evict:
                raise ModelInUseError(
                    f"{key!r} has a built handle; pass evict=True to close it"
                )
            self._evict_inactive(key)
        del self._entries[key]

    @contextlib.contextmanager
    def use(self, model: str | ModelSpec) -> Iterator[nn.Module]:
        """Acquire an active lease on a cached model.

        Accepts either a registered key (string) or a :class:`ModelSpec`
        (auto-registers if its key isn't already known). Yields the
        ``nn.Module`` for use; on context exit, releases the lease.

        Re-entrant for the same key: nested ``use()`` calls bump a
        per-key refcount and share the already-active module (the
        underlying strategy is *not* activated again).
        """
        spec = model if isinstance(model, ModelSpec) else None
        key = spec.key if spec is not None else model
        assert isinstance(key, str)

        entry = self._entries.get(key)
        if entry is None:
            if spec is None:
                raise ModelNotRegisteredError(
                    f"{key!r} is not registered; pass a ModelSpec to use() or call register() first"
                )
            self.register(spec)
            entry = self._entries[key]

        with self._activate_entry(entry):
            assert entry.active_module is not None
            yield entry.active_module

    def evict(self, key: str) -> None:
        """Manually evict one inactive cached entry. Raises
        :class:`ModelInUseError` if active, no-op if not cached."""
        entry = self._entries.get(key)
        if entry is None or entry.handle is None:
            return
        if entry.active_count > 0:
            raise ModelInUseError(f"cannot evict active model {key!r}")
        self._evict_inactive(key)

    def clear(self) -> None:
        """Evict all inactive entries. Registrations are preserved.
        Raises :class:`ModelInUseError` if any entry is active."""
        active = [k for k, e in self._entries.items() if e.active_count > 0]
        if active:
            raise ModelInUseError(f"cannot clear while models are active: {active}")
        for key in list(self._lru):
            self._evict_inactive(key)

    def info(self, key: str) -> ModelInfo:
        """Per-key snapshot. Raises :class:`ModelNotRegisteredError` if
        unknown."""
        entry = self._entries.get(key)
        if entry is None:
            raise ModelNotRegisteredError(f"{key!r} is not registered")
        return ModelInfo(
            key=entry.spec.key,
            label=entry.spec.label,
            estimated_cache_bytes=entry.spec.estimated_cache_bytes,
            cache_bytes=entry.cache_bytes if entry.handle is not None else None,
            cached=entry.handle is not None,
            active_count=entry.active_count,
        )

    def snapshot(self) -> ModelCacheSnapshot:
        """Whole-cache point-in-time view. Stats are deep-copied so
        the snapshot is genuinely immutable."""
        return ModelCacheSnapshot(
            max_cache_bytes=self._max_cache_bytes,
            used_cache_bytes=self._used_bytes,
            registered_keys=tuple(self._entries.keys()),
            cached_keys_lru_to_mru=tuple(self._lru.keys()),
            active_refcounts={k: e.active_count for k, e in self._entries.items() if e.active_count > 0},
            stats=dataclasses.replace(self._stats),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def _activate_entry(self, entry: _Entry) -> Iterator[None]:
        key = entry.spec.key
        # Re-entrant case: already active for this key. Bump refcount,
        # share the active module, do not re-activate the strategy.
        if entry.active_count > 0:
            entry.active_count += 1
            try:
                yield
            finally:
                entry.active_count -= 1
            return

        # First lease: ensure built, then activate.
        is_freshly_built = entry.handle is None
        if is_freshly_built:
            self._build_into_entry(entry)
        else:
            self._stats.hits += 1
            self._lru.pop(key, None)  # leaving inactive set

        assert entry.handle is not None
        try:
            module = entry.handle.activate()
        except BaseException as exc:
            self._stats.activation_errors += 1
            if is_freshly_built:
                # Drop the poisoned entry — no point caching a handle
                # whose activation just failed. close() may also fail;
                # log and surface the original.
                self._discard_freshly_built(entry, cause=exc)
            else:
                # Existing cached entry — keep it cached, but it's no
                # longer active. Put back into the LRU so it remains
                # eligible for eviction.
                self._lru[key] = None
            raise ActivationError(f"activate() failed for {key!r}") from exc

        entry.active_module = module
        entry.active_count = 1
        try:
            yield
        finally:
            entry.active_count -= 1
            if entry.active_count == 0:
                # Strategy contract: deactivate is expected to leave
                # the handle in a clean inactive state and not raise
                # under normal use. If it does raise, the strategy's
                # internal state is unknown — don't risk reusing it.
                # Close and discard so a subsequent acquire rebuilds.
                try:
                    entry.handle.deactivate()
                except BaseException:
                    self._discard_poisoned(entry)
                    raise
                entry.active_module = None
                # Active → inactive: re-enter LRU at MRU position.
                self._lru[key] = None

    def _build_into_entry(self, entry: _Entry) -> None:
        """Cache miss: evict to make room, build, reconcile actuals."""
        key = entry.spec.key
        self._stats.misses += 1
        estimate = entry.spec.estimated_cache_bytes
        # Pre-build eviction: trust the estimate.
        self._evict_until_room(key, estimate)
        try:
            handle = entry.spec.factory()
        except BaseException:
            self._stats.factory_errors += 1
            raise
        actual = handle.cache_bytes
        if actual < 0:
            # Misbehaving strategy. Don't admit it (would corrupt
            # _used_bytes accounting) — close and raise.
            self._close_uncached(handle)
            raise ModelCacheError(
                f"strategy.cache_bytes for {key!r} returned {actual} (must be >= 0)"
            )
        # Reconcile if actual overshot the estimate.
        if actual > estimate:
            try:
                self._evict_until_room(key, actual)
            except BaseException:
                # Can't fit the actual size — close the new handle and
                # re-raise. Don't mark as built.
                self._close_uncached(handle)
                raise
        entry.handle = handle
        entry.cache_bytes = actual
        self._used_bytes += actual
        self._stats.builds += 1
        self._stats.peak_cache_bytes = max(self._stats.peak_cache_bytes, self._used_bytes)
        # Freshly-built entries are about to be activated; do NOT add to
        # LRU yet. They'll be added on first deactivate.

    def _evict_until_room(self, incoming_key: str, required: int) -> None:
        """Evict inactive LRU entries until ``required`` bytes fit. If
        active entries block sufficient eviction, raise
        :class:`ModelTooLargeError`."""
        if required > self._max_cache_bytes:
            raise ModelTooLargeError(
                key=incoming_key,
                required=required,
                used=self._used_bytes,
                limit=self._max_cache_bytes,
                active_refcounts={
                    k: e.active_count for k, e in self._entries.items() if e.active_count > 0
                },
            )
        while self._used_bytes + required > self._max_cache_bytes:
            try:
                victim = next(iter(self._lru))
            except StopIteration:
                raise ModelTooLargeError(
                    key=incoming_key,
                    required=required,
                    used=self._used_bytes,
                    limit=self._max_cache_bytes,
                    active_refcounts={
                        k: e.active_count for k, e in self._entries.items() if e.active_count > 0
                    },
                ) from None
            self._evict_inactive(victim)

    def _evict_inactive(self, key: str) -> None:
        entry = self._entries[key]
        assert entry.handle is not None
        assert entry.active_count == 0
        handle = entry.handle
        bytes_freed = entry.cache_bytes
        # Detach from cache state first so a close() failure doesn't
        # leave us in an inconsistent state.
        entry.handle = None
        entry.cache_bytes = 0
        self._lru.pop(key, None)
        self._used_bytes -= bytes_freed
        try:
            handle.close()
        except BaseException as exc:
            self._stats.close_errors += 1
            raise ModelEvictionError(
                f"close() raised while evicting {key!r}"
            ) from exc
        finally:
            self._stats.evictions += 1
            self._stats.bytes_evicted += bytes_freed
            self._after_close()

    def _evict_for_replace(self, key: str) -> None:
        """``register(replace=True)`` path. Refuses to clobber an active
        entry."""
        entry = self._entries[key]
        if entry.active_count > 0:
            raise ModelInUseError(
                f"cannot replace active registration {key!r} (refcount={entry.active_count})"
            )
        if entry.handle is not None:
            self._evict_inactive(key)

    def _discard_poisoned(self, entry: _Entry) -> None:
        """Strategy raised in deactivate() — its internal state is now
        unknown. Drop the entry and best-effort close. Active state is
        cleared so the entry doesn't appear active in snapshots."""
        key = entry.spec.key
        handle = entry.handle
        bytes_to_free = entry.cache_bytes
        entry.handle = None
        entry.cache_bytes = 0
        entry.active_module = None
        entry.active_count = 0
        self._lru.pop(key, None)
        self._used_bytes -= bytes_to_free
        if handle is not None:
            try:
                handle.close()
            except BaseException as close_exc:
                self._stats.close_errors += 1
                logger.error(
                    "close() raised while discarding poisoned %r whose deactivate() "
                    "had already failed; original deactivate error will still propagate. "
                    "close error=%r",
                    key,
                    close_exc,
                    exc_info=True,
                )
        self._after_close()

    def _close_uncached(self, handle: ModelStrategy) -> None:
        """Close a handle that never made it into the cache (oversized
        actual, factory return value rejected, etc.). Best-effort: log
        close failures rather than raising over the original cause."""
        try:
            handle.close()
        except BaseException as exc:
            self._stats.close_errors += 1
            logger.warning(
                "close() raised on a never-cached handle; ignoring", exc_info=exc
            )
        finally:
            self._after_close()

    def _discard_freshly_built(self, entry: _Entry, *, cause: BaseException) -> None:
        """Activation just failed on a freshly-built handle. Roll back
        the cache state and try to close the handle. Any close failure
        is logged (don't mask the original activation cause)."""
        key = entry.spec.key
        handle = entry.handle
        bytes_to_free = entry.cache_bytes
        entry.handle = None
        entry.cache_bytes = 0
        # Was never added to LRU (freshly-built path skips it), but be
        # defensive in case future refactors change that.
        self._lru.pop(key, None)
        self._used_bytes -= bytes_to_free
        if handle is not None:
            try:
                handle.close()
            except BaseException as close_exc:
                self._stats.close_errors += 1
                logger.error(
                    "close() raised while discarding freshly-built %r whose activate() "
                    "had already failed; original activation error will still propagate. "
                    "close error=%r",
                    key,
                    close_exc,
                    exc_info=True,
                )
        self._after_close()
        # Use cause to silence ruff's unused-arg lint if any analyzer cares.
        _ = cause

    def _after_close(self) -> None:
        if self._empty_host_cache is None:
            return
        try:
            self._empty_host_cache()
        except Exception:
            logger.warning("empty_host_cache callback raised; ignoring", exc_info=True)
