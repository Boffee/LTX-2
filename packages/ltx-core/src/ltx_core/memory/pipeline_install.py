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

Skipped per call:

- If ``DiffusionStage._torch_compile`` is True: compile and slot-swap
  are fundamentally incompatible. Falls back to the original
  ``_transformer_ctx``.

Caveats:

- Cache key includes ``streaming_prefetch_count`` (selects PinnedWeights
  vs BlockOffloader strategy) but **not** other per-call kwargs.
  Calling the same block instance with kwargs that affect model
  construction would silently reuse the first-built model. For LTX-2
  the typical kwargs (``video_tools``) only control runtime
  patchify/unpatchify, not model architecture, so this is safe.
- Pre-eviction at admission uses caller-supplied size estimates; the
  cache reconciles to actual ``cache_bytes`` after activate. If
  estimates are 0 (default), no pre-eviction happens — over-budget
  warnings come from the post-activate reconciliation path.
- Process-lifetime cache by default. Call ``uninstall_model_cache()``
  to restore the original block methods and evict installer-owned
  entries.
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

from ltx_core.memory.block_offloader import BlockOffloader
from ltx_core.memory.model_cache import ModelCache, ModelSpec
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

# layers_attr per block class for the streaming (BlockOffloader) path —
# matches what the upstream ``_streaming_model`` callsites use today.
_STREAMING_LAYERS_ATTR: dict[str, str] = {
    "DiffusionStage": "velocity_model.transformer_blocks",
    "PromptEncoder": "model.model.language_model.layers",
}


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
        post-activate reconciliation). Pass an actual estimate (e.g.
        ``24 * 1024**3``) for proactive eviction in tight-budget
        scenarios.
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


def uninstall_model_cache() -> None:
    """Restore the original block methods and evict all
    installer-owned cache entries. Other cache entries (added by the
    caller directly via ``cache.use(...)``) are preserved.

    Idempotent — no-op if not currently installed.
    """
    global _INSTALLED_CACHE

    if _INSTALLED_CACHE is None:
        return

    cache = _INSTALLED_CACHE

    # Evict only keys this installer created. Use unregister(evict=True)
    # so the spec/factory closure is dropped too — evict() alone leaves
    # the registration behind.
    for key in list(_INSTALLED_KEYS):
        try:
            cache.unregister(key, evict=True)
        except Exception:
            logger.warning("uninstall_model_cache: unregister failed for %r", key, exc_info=True)
    _INSTALLED_KEYS.clear()
    _BLOCK_KEYS.clear()
    _TOKENS.clear()

    # Restore originals.
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


def _build_streaming_offloader(
    model: nn.Module,
    target_device: torch.device,
    layers_attr: str,
    streaming_prefetch_count: int,
) -> BlockOffloader:
    """Construct a BlockOffloader matching the LayerStreamingWrapper
    semantics the original ``_streaming_model`` callsite used.

    ``LayerStreamingWrapper(prefetch_count=N)`` keeps ``1 + N`` blocks
    on GPU at a time. ``BlockOffloader`` parameterizes the same window
    via ``blocks_to_swap = num_layers - 1`` (one resident at a time)
    plus ``prefetch_count = N``.
    """
    layer_list: Any = model
    for part in layers_attr.split("."):
        layer_list = getattr(layer_list, part)
    num_layers = len(layer_list)
    off = BlockOffloader(
        model,
        target_device=target_device,
        blocks_to_swap=num_layers - 1,
        layers_attr=layers_attr,
        prefetch_count=streaming_prefetch_count,
        auto_setup=False,
    )
    off.prepare()
    return off


# ---------------------------------------------------------------------------
# Patched methods
# ---------------------------------------------------------------------------


def _patched_transformer_ctx(
    self: Any,
    streaming_prefetch_count: int | None,
    **kwargs: Any,
) -> AbstractContextManager[nn.Module]:
    cache = _INSTALLED_CACHE
    # Fall back to original when caching is incompatible (torch.compile
    # does tensor-identity tracking that slot swaps invalidate) or when
    # an explicit user wrapper is in place (their call wins).
    if (
        cache is None
        or getattr(self, "_torch_compile", False)
        or getattr(self, "_transformer_wrapper", None) is not None
    ):
        original = _ORIGINALS[(type(self), "_transformer_ctx")]
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
    layers_attr = _STREAMING_LAYERS_ATTR.get(cls.__name__, "transformer_blocks")

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
        model = block._build_transformer(device=torch.device("cpu"), **kwargs)
        if streaming_prefetch_count is None:
            return PinnedWeights(model, target_device)
        return _build_streaming_offloader(
            model, target_device, layers_attr, streaming_prefetch_count,
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
    if cache is None:
        original = _ORIGINALS[(type(self), "_text_encoder_ctx")]
        return original(self, streaming_prefetch_count)

    cls = type(self)
    token = _get_or_create_token(cls, self)
    variant = (
        f"stream{streaming_prefetch_count}"
        if streaming_prefetch_count is not None
        else "pinned"
    )
    key = f"{cls.__name__}:{token}:text_encoder:{variant}"
    _associate_key(token, key)

    block_ref = weakref.ref(self)
    target_device = self._device
    dtype = self._dtype
    layers_attr = _STREAMING_LAYERS_ATTR.get(cls.__name__, "model.model.language_model.layers")

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
        if streaming_prefetch_count is None:
            return PinnedWeights(built, target_device)
        return _build_streaming_offloader(
            built, target_device, layers_attr, streaming_prefetch_count,
        )

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
]
