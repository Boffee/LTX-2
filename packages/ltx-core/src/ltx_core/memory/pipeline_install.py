"""Monkey-patch installer for routing ltx_pipelines block construction
through a :class:`ModelCache`.

Designed for the rebase-friendly use case: keep all integration code in
this locally-owned subpackage so the upstream ``ltx-pipelines`` source
files don't need to change. One call at script startup::

    from ltx_core.memory import ModelCache
    from ltx_core.memory.pipeline_install import install_model_cache

    cache = ModelCache(max_cache_bytes=80 * 1024**3)
    install_model_cache(cache)

After install, every :class:`~ltx_pipelines.utils.blocks.DiffusionStage`
and :class:`~ltx_pipelines.utils.blocks.PromptEncoder` invocation
routes its model construction through ``cache``. Cache entries are
keyed by block-instance object identity (ComfyUI-style) so:

- One pipeline instance reused across many calls hits the cache
- Different pipeline instances each get their own entries
- When a block instance is garbage-collected, a ``weakref.finalize``
  hook auto-evicts its cache entries

V1 patches only the two block classes that already expose a
``_xxx_ctx`` context-manager seam designed for swapping. The
``__call__``-only blocks (``ImageConditioner``, ``VideoDecoder``,
etc.) are out of scope — they're either too small to be worth
caching, or their lazy-iterator return type makes lifetime
correctness fragile.

Streaming bypass
----------------
Calls with ``streaming_prefetch_count is not None`` fall through to
the original method (which uses :class:`LayerStreamingWrapper`).
Reason: the LTX transformer (``X0Model.velocity_model``) has direct
frozen parameters on the ``velocity_model`` module itself
(``scale_shift_table``), and :class:`BlockOffloader` rejects that
pattern at :meth:`prepare`. Caching streaming-mode models would
require either teaching :class:`BlockOffloader` to manage
direct-parent state, or composing a third strategy.
:class:`LayerStreamingWrapper`'s pinned-CPU pages are NOT counted
against the cache budget — keep that in mind for tight-budget
configurations that mix streaming and cached models.

Other skips
-----------
- ``DiffusionStage._torch_compile=True``: compile and slot-swap are
  fundamentally incompatible. Falls back to original.
- ``cache=None``: uninstalled.

Caveats
-------
- Cache key includes ``streaming_prefetch_count`` (selects PinnedWeights
  vs the streaming fallback path) but **not** other per-call kwargs.
  Calling the same block instance with kwargs that affect model
  construction would silently reuse the first-built model. For LTX-2
  ``Builder.build()`` ignores ``**kwargs`` so this is safe today.
- Default size estimates are 0 (no pre-eviction; rely on post-activate
  reconciliation). For tight-budget configurations where the total
  pinned working set may exceed ``max_cache_bytes``, pass non-zero
  ``transformer_size_estimate`` / ``text_encoder_size_estimate`` so
  the cache evicts before the next factory pins memory — otherwise
  the host allocator can spike during construction.
- Process-lifetime cache by default. Call ``uninstall_model_cache()``
  to restore the original block methods and evict installer-owned
  entries. Uninstall raises if any installer-owned entry is currently
  active (caller must finish their work and retry).
"""

from __future__ import annotations

import inspect
import itertools
import logging
import weakref
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any

import torch
from torch import nn

from ltx_core.memory.model_cache import ModelCache, ModelInUseError, ModelSpec
from ltx_core.memory.pinned_weights import PinnedWeights
from ltx_core.memory.strategy import ModelStrategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level installer state
# ---------------------------------------------------------------------------


_INSTALLED_CACHE: ModelCache | None = None
# Per-installer settings — re-evaluated at install time.
_TRANSFORMER_SIZE_ESTIMATE: int = 0
_TEXT_ENCODER_SIZE_ESTIMATE: int = 0
# Per-block-class WeakKeyDictionary[block_instance, monotonic int token].
# Tokens stay stable across the block's lifetime and let us derive cache
# keys without keeping the block alive (which would defeat finalize).
_TOKENS: dict[type, weakref.WeakKeyDictionary[Any, int]] = {}
_TOKEN_COUNTER = itertools.count()
# token -> set of cache keys associated with that block instance. The
# block-GC finalizer reads this to evict everything for that token.
_BLOCK_KEYS: dict[int, set[str]] = {}
# All keys ever produced by this installer — for clean uninstall without
# touching unrelated entries the caller may have added.
_INSTALLED_KEYS: set[str] = set()
# (cls, method_name) -> original unbound method, for restore on uninstall.
_ORIGINALS: dict[tuple[type, str], Callable[..., Any]] = {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def install_model_cache(
    cache: ModelCache | None,
    *,
    transformer_size_estimate: int = 0,
    text_encoder_size_estimate: int = 0,
) -> None:
    """Patch :class:`~ltx_pipelines.utils.blocks.DiffusionStage` and
    :class:`~ltx_pipelines.utils.blocks.PromptEncoder` to route model
    construction through ``cache``.

    Parameters
    ----------
    cache:
        The :class:`ModelCache` to route through. Pass ``None`` to
        uninstall (restore originals + evict installer-owned keys).
    transformer_size_estimate:
        Bytes to estimate for transformer entries at admission. Used by
        the cache for pre-eviction. Default 0 (no pre-eviction; rely on
        post-activate reconciliation). For tight-budget configurations,
        pass an actual estimate (e.g. ``24 * 1024**3``) so the cache
        evicts the previously-pinned model BEFORE the factory pins the
        next one — without it, peak host memory spikes to
        ``old_pinned + new_pinned`` during construction.
    text_encoder_size_estimate:
        Same, for text encoder entries.

    Idempotent: calling with the same cache is a no-op (just updates
    estimates). Calling with a different cache uninstalls the previous
    one first.

    Raises ``RuntimeError`` if the targeted block methods have signatures
    that don't match what the patcher expects (e.g., upstream refactor)
    — fail loudly rather than silently misbehave.
    """
    global _INSTALLED_CACHE, _TRANSFORMER_SIZE_ESTIMATE, _TEXT_ENCODER_SIZE_ESTIMATE

    if cache is None:
        uninstall_model_cache()
        return

    # Same cache → idempotent update.
    if _INSTALLED_CACHE is cache:
        _TRANSFORMER_SIZE_ESTIMATE = transformer_size_estimate
        _TEXT_ENCODER_SIZE_ESTIMATE = text_encoder_size_estimate
        return

    # Different cache → uninstall, then install fresh.
    if _INSTALLED_CACHE is not None:
        uninstall_model_cache()

    # Lazy import keeps the memory subpackage self-contained — pipelines
    # don't have to be installed for the rest of memory to work.
    from ltx_pipelines.utils.blocks import DiffusionStage, PromptEncoder

    _validate_signature(DiffusionStage, "_transformer_ctx", ("self", "streaming_prefetch_count"))
    _validate_signature(PromptEncoder, "_text_encoder_ctx", ("self", "streaming_prefetch_count"))

    _install_method_patch(DiffusionStage, "_transformer_ctx", _patched_transformer_ctx)
    _install_method_patch(PromptEncoder, "_text_encoder_ctx", _patched_text_encoder_ctx)

    _INSTALLED_CACHE = cache
    _TRANSFORMER_SIZE_ESTIMATE = transformer_size_estimate
    _TEXT_ENCODER_SIZE_ESTIMATE = text_encoder_size_estimate


class UninstallBusyError(RuntimeError):
    """Raised when ``uninstall_model_cache`` cannot evict one or more
    installer-owned entries because they are currently active. Caller
    must finish their work (exit any open ``cache.use`` contexts) and
    retry. Uninstall does NOT partially unwind on failure — the
    patches stay installed and the surviving entries stay tracked."""

    def __init__(self, busy_keys: list[str]) -> None:
        super().__init__(
            f"uninstall_model_cache failed: {len(busy_keys)} installer-owned "
            f"entries are still active: {busy_keys}. Exit any cache.use() "
            "contexts and retry."
        )
        self.busy_keys = busy_keys


def uninstall_model_cache() -> None:
    """Restore the original block methods and evict all
    installer-owned cache entries. Other cache entries (added by the
    caller directly via ``cache.use(...)``) are preserved.

    Idempotent — no-op if not currently installed.

    Raises :class:`UninstallBusyError` if any installer-owned entry is
    currently active. On failure: patches stay installed and busy keys
    stay tracked, but any non-busy installer-owned keys evicted before
    the first failure stay evicted (we don't try to roll back a partial
    eviction). The caller can finish their work and retry — only the
    busy keys will need a second eviction pass.
    """
    global _INSTALLED_CACHE

    if _INSTALLED_CACHE is None:
        return

    cache = _INSTALLED_CACHE

    # First pass: try to evict every installer-owned key. Collect
    # failures (typically ModelInUseError) without clearing tracking.
    busy: list[str] = []
    succeeded: list[str] = []
    for key in list(_INSTALLED_KEYS):
        try:
            cache.unregister(key, evict=True)
        except ModelInUseError:
            busy.append(key)
        except Exception:
            # Genuine eviction failure — log and treat as succeeded for
            # bookkeeping (the entry is in an unknown state but we
            # can't do better than dropping our reference to it).
            logger.warning(
                "uninstall_model_cache: unregister raised for %r; dropping tracking",
                key,
                exc_info=True,
            )
            succeeded.append(key)
        else:
            succeeded.append(key)

    # Drop bookkeeping for the keys we successfully evicted (or gave up on).
    for key in succeeded:
        _INSTALLED_KEYS.discard(key)
    # Prune _BLOCK_KEYS to drop succeeded entries too.
    for token, keys in list(_BLOCK_KEYS.items()):
        keys -= set(succeeded)
        if not keys:
            del _BLOCK_KEYS[token]

    if busy:
        # Some entries are active — refuse to uninstall. Patches remain
        # in place so subsequent calls go through the cache; failed
        # keys remain tracked so a retry catches them.
        raise UninstallBusyError(busy)

    # All clear — drop tokens and restore originals.
    _TOKENS.clear()
    for (cls, name), original in _ORIGINALS.items():
        setattr(cls, name, original)
    _ORIGINALS.clear()
    _INSTALLED_CACHE = None


# ---------------------------------------------------------------------------
# Internal helpers (also used by tests with stub block classes)
# ---------------------------------------------------------------------------


def _validate_signature(cls: type, method_name: str, expected_prefix: tuple[str, ...]) -> None:
    method = getattr(cls, method_name, None)
    if method is None:
        raise RuntimeError(f"{cls.__name__} has no method {method_name!r}")
    sig = inspect.signature(method)
    params = tuple(sig.parameters.keys())
    if params[: len(expected_prefix)] != expected_prefix:
        raise RuntimeError(
            f"{cls.__name__}.{method_name} signature changed: got {params}, "
            f"expected to start with {expected_prefix}. The pipeline_install "
            "patcher needs an update."
        )


def _install_method_patch(cls: type, method_name: str, patched: Callable[..., Any]) -> None:
    """Save the original method and replace it with ``patched``."""
    if (cls, method_name) in _ORIGINALS:
        return  # already patched
    _ORIGINALS[(cls, method_name)] = getattr(cls, method_name)
    setattr(cls, method_name, patched)


def _find_patched_ancestor(self: Any, method_name: str) -> tuple[type, Callable[..., Any]]:
    """Walk ``type(self).__mro__`` to find the closest ancestor whose
    method is patched. Returns (ancestor_class, original_method). Raises
    if no ancestor was patched (programmer error in tests).

    Subclass safety: the patch lives on the ancestor (e.g. DiffusionStage)
    but the instance is `MyDiffusionSubclass(DiffusionStage)`. Looking
    up `_ORIGINALS[(type(self), method_name)]` would KeyError; walking
    the MRO finds the entry on the actual patched class.
    """
    for cls in type(self).__mro__:
        key = (cls, method_name)
        if key in _ORIGINALS:
            return cls, _ORIGINALS[key]
    raise KeyError(
        f"no original recorded for {type(self).__name__}.{method_name} "
        f"(searched MRO {[c.__name__ for c in type(self).__mro__]})"
    )


def _get_or_create_token(cls: type, block: Any) -> int:
    """Get the monotonic token for ``block``, creating it if absent.

    Sets up a one-shot ``weakref.finalize`` on the block that, when the
    block is garbage-collected, evicts every cache key associated with
    it (so stale entries don't accumulate after pipelines are dropped).
    """
    bucket = _TOKENS.setdefault(cls, weakref.WeakKeyDictionary())
    existing = bucket.get(block)
    if existing is not None:
        return existing
    token = next(_TOKEN_COUNTER)
    bucket[block] = token

    # The finalizer captures only the token (an int) and the cache lookup
    # at finalize-time, NOT the block (which would defeat the GC).
    def _on_block_gc(token: int = token) -> None:
        keys = _BLOCK_KEYS.pop(token, None)
        if not keys:
            return
        cache = _INSTALLED_CACHE
        for k in list(keys):
            _INSTALLED_KEYS.discard(k)
            if cache is None:
                continue
            try:
                cache.unregister(k, evict=True)
            except Exception:
                logger.warning(
                    "block-GC eviction failed for %r", k, exc_info=True
                )

    weakref.finalize(block, _on_block_gc)
    return token


def _associate_key(token: int, key: str) -> None:
    """Track ``key`` against ``token`` so the block's GC finalizer can
    evict it later. Also adds to the global installer-owned set for
    uninstall cleanup."""
    _BLOCK_KEYS.setdefault(token, set()).add(key)
    _INSTALLED_KEYS.add(key)


# ---------------------------------------------------------------------------
# Patched methods
# ---------------------------------------------------------------------------


def _patched_transformer_ctx(
    self: Any,
    streaming_prefetch_count: int | None,
    **kwargs: Any,
) -> AbstractContextManager[nn.Module]:
    cache = _INSTALLED_CACHE
    _, original = _find_patched_ancestor(self, "_transformer_ctx")

    # Fall back to original when caching is impossible or incompatible:
    #   - cache uninstalled
    #   - torch.compile (slot swaps invalidate compile's tensor-identity tracking)
    #   - streaming mode (BlockOffloader rejects LTXModel's direct
    #     velocity_model.scale_shift_table; LayerStreamingWrapper handles it)
    if cache is None or getattr(self, "_torch_compile", False) or streaming_prefetch_count is not None:
        return original(self, streaming_prefetch_count, **kwargs)

    cls = type(self)
    token = _get_or_create_token(cls, self)
    key = f"{cls.__name__}:{token}:transformer:pinned"
    _associate_key(token, key)

    block_ref = weakref.ref(self)
    target_device = self._device

    def factory() -> ModelStrategy:
        block = block_ref()
        if block is None:
            # Should be unreachable: factory only runs while the cache
            # holds the spec, and the block-GC finalizer removes the
            # spec when the block dies.
            raise RuntimeError(
                f"{cls.__name__} instance was garbage-collected before "
                "factory ran (race in cache lifecycle)"
            )
        # Don't pass kwargs through — Builder.build() ignores them today
        # and retaining per-call objects (video_tools instances) in the
        # factory closure for the cache lifetime is wasteful. If a
        # future kwarg ever affects model construction it must be added
        # to the cache key (variant) so the cache rebuilds.
        model = block._build_transformer(device=torch.device("cpu"))
        return PinnedWeights(model, target_device)

    return cache.use(
        ModelSpec(
            key=key,
            estimated_cache_bytes=_TRANSFORMER_SIZE_ESTIMATE,
            factory=factory,
            label=f"{cls.__name__} transformer",
        )
    )


def _patched_text_encoder_ctx(
    self: Any,
    streaming_prefetch_count: int | None,
) -> AbstractContextManager[nn.Module]:
    cache = _INSTALLED_CACHE
    _, original = _find_patched_ancestor(self, "_text_encoder_ctx")

    # Streaming mode: same fallback as transformer. The Gemma text
    # encoder may not have direct parent params today, but staying
    # consistent (cache only the whole-model PinnedWeights path)
    # avoids the need to teach BlockOffloader about each backbone's
    # quirks.
    if cache is None or streaming_prefetch_count is not None:
        return original(self, streaming_prefetch_count)

    cls = type(self)
    token = _get_or_create_token(cls, self)
    key = f"{cls.__name__}:{token}:text_encoder:pinned"
    _associate_key(token, key)

    block_ref = weakref.ref(self)
    target_device = self._device
    dtype = self._dtype

    def factory() -> ModelStrategy:
        block = block_ref()
        if block is None:
            raise RuntimeError(
                f"{cls.__name__} instance was garbage-collected before "
                "factory ran (race in cache lifecycle)"
            )
        built = block._text_encoder_builder.build(device=torch.device("cpu"), dtype=dtype)
        # Match the upstream call which sets eval mode after build.
        built.train(False)
        return PinnedWeights(built, target_device)

    return cache.use(
        ModelSpec(
            key=key,
            estimated_cache_bytes=_TEXT_ENCODER_SIZE_ESTIMATE,
            factory=factory,
            label=f"{cls.__name__} text encoder",
        )
    )


__all__ = [
    "install_model_cache",
    "uninstall_model_cache",
    "UninstallBusyError",
]
