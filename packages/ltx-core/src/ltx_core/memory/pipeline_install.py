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
keyed by block-instance object identity so:

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

Strategy choice per component
-----------------------------
- **Transformer**: follows the pipeline's per-call
  ``streaming_prefetch_count`` kwarg. ``None`` →
  :class:`PinnedWeights` (whole-model bulk DMA); ``int`` →
  :func:`make_block_offloader` (per-block streaming). Cache key includes
  ``stream{N}`` vs ``pinned`` so toggling on the same block instance
  produces distinct entries. Streaming-mode caching relies on
  :func:`make_block_offloader` handling direct frozen parameters on parent
  modules (e.g. LTX's ``velocity_model.scale_shift_table``) via the
  composed-PinnedWeights skip filter.
- **Text encoder**: always :class:`PinnedWeights`. The pipeline's
  ``streaming_prefetch_count`` kwarg is ignored for the text encoder
  because text encoders fit on GPU under any realistic config and
  are used one-shot per prompt — per-block hook overhead doesn't
  amortize over a single forward pass. Hardcoding this also avoids
  the silent footgun where the pipeline's single-knob kwarg would
  inadvertently stream the text encoder when the caller really only
  wanted the diffusion transformer to stream.

Fallback to original
--------------------
- ``DiffusionStage._torch_compile=True``: compile and slot-swap are
  fundamentally incompatible. Falls back to original.
- ``cache=None``: uninstalled.

Caveats
-------
- Cache key is ``f"{cls.__name__}:{token}:{kind}:{variant}"`` — keyed
  by block-instance token + streaming variant only, **not** per-call
  kwargs. Calling the same block instance with kwargs that affect
  model construction would silently reuse the first-built model. For
  LTX-2 ``Builder.build()`` ignores ``**kwargs`` so this is safe
  today; if a future kwarg ever
  affects construction it must be added to the variant in the key
  so the cache rebuilds.
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

from ltx_core.memory.block_compose import make_block_offloader
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

    Strategy choice
    ---------------
    - **Transformer**: follows the pipeline's per-call
      ``streaming_prefetch_count`` kwarg. ``None`` →
      :class:`PinnedWeights` (whole-model bulk DMA); ``int`` →
      :func:`make_block_offloader` (per-block streaming).
    - **Text encoder**: always :class:`PinnedWeights`. Streaming a
      text encoder doesn't make sense — they fit on GPU, are used
      one-shot per prompt, and per-block hook overhead doesn't
      amortize. The pipeline's ``streaming_prefetch_count`` kwarg is
      ignored for the text encoder.

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


class UninstallBusyError(ModelInUseError):
    """Raised when ``uninstall_model_cache`` cannot evict one or more
    installer-owned entries because they are currently active.

    Subclasses :class:`~ltx_core.memory.model_cache.ModelInUseError` so
    callers can ``except ModelInUseError`` to catch any "entry is busy"
    family of error.

    On failure: patches stay installed; busy keys stay tracked; any
    non-busy installer-owned keys evicted before the first failure stay
    evicted (we don't roll back a partial eviction). The caller can
    finish their work and retry — only the busy keys will need a
    second eviction pass.
    """

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


def _resolve_block_list(model: nn.Module, layers_attr: str, owner_label: str) -> Any:
    """Walk the dotted ``layers_attr`` path on ``model``; raise a clear
    error if the path doesn't resolve. Same paths the upstream
    ``_streaming_model`` callsite uses, so failures here mean upstream
    reshaped the model and the patcher needs an update."""
    obj: Any = model
    for part in layers_attr.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError as exc:
            raise RuntimeError(
                f"pipeline_install: could not resolve streaming layers at "
                f"'{layers_attr}' for cached {owner_label} streaming mode "
                f"(failed at {part!r} on {type(obj).__name__}). The upstream "
                "model layout may have changed; the patcher's hard-coded "
                "layers_attr needs updating."
            ) from exc
    return obj


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

    # Fall back to original when caching is impossible:
    #   - cache uninstalled
    #   - torch.compile (slot swaps invalidate compile's tensor-identity
    #     tracking; both PinnedWeights and block-streaming hit this)
    if cache is None or getattr(self, "_torch_compile", False):
        return original(self, streaming_prefetch_count, **kwargs)

    cls = type(self)
    token = _get_or_create_token(cls, self)
    variant = (
        f"stream{streaming_prefetch_count}"
        if streaming_prefetch_count is not None
        else "pinned"
    )
    key = f"{cls.__name__}:{token}:transformer:{variant}"
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
        if streaming_prefetch_count is None:
            return PinnedWeights(model, target_device)
        # Streaming: make_block_offloader handles direct parent params
        # (e.g. LTX's velocity_model.scale_shift_table) via the
        # composed PinnedWeights with a skip filter, so streaming-mode
        # models are cacheable. Pinning happens inside the factory
        # so cache_bytes is final when ModelCache admits the entry.
        layers_attr = "velocity_model.transformer_blocks"
        layer_list = _resolve_block_list(model, layers_attr, cls.__name__)
        num_layers = len(layer_list)
        return make_block_offloader(
            model,
            target_device=target_device,
            layers_attr=layers_attr,
            blocks_to_swap=num_layers - 1,
            prefetch_count=streaming_prefetch_count,
        )

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

    if cache is None:
        return original(self, streaming_prefetch_count)

    # Always use PinnedWeights for the text encoder, regardless of the
    # pipeline's streaming_prefetch_count kwarg. Streaming a text
    # encoder doesn't make sense — text encoders fit on GPU under any
    # realistic config and are used one-shot per prompt; per-block
    # hook overhead doesn't amortize over a single forward pass.
    # Decoupling text-encoder strategy from the pipeline's
    # one-knob-fits-all kwarg also avoids a silent footgun where
    # passing streaming_prefetch_count for the diffusion transformer
    # would inadvertently stream the text encoder too.
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
