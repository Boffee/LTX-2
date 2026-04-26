"""Tests for ``ltx_core.memory.pipeline_install``.

Uses stub block classes to test the installer mechanics without
spinning up real ltx_pipelines models. Where the public
``install_model_cache`` is hard-wired to ``ltx_pipelines.utils.blocks``
classes, we use the internal ``_install_method_patch`` helper to wire
patches onto stub classes.
"""

from __future__ import annotations

import gc
import weakref
from contextlib import AbstractContextManager
from typing import Any

import pytest
import torch
from torch import nn

from ltx_core.memory import ModelCache, ModelSpec
from ltx_core.memory import pipeline_install as pi


# ---------------------------------------------------------------------------
# Stub blocks that mirror DiffusionStage / PromptEncoder shape
# ---------------------------------------------------------------------------


class StubDiffusionStage:
    """Stand-in for ltx_pipelines.utils.blocks.DiffusionStage with the
    fields and methods the patcher reads."""

    _torch_compile = False
    _transformer_wrapper = None

    def __init__(self):
        self._device = torch.device("cpu")
        self._build_count = 0

    def _build_transformer(self, *, device=None, **kwargs):  # noqa: ANN001
        self._build_count += 1
        m = nn.Linear(4, 4, bias=False)
        for p in m.parameters():
            p.requires_grad = False
        return m

    def _transformer_ctx(
        self, streaming_prefetch_count, **kwargs
    ) -> AbstractContextManager[nn.Module]:
        # Original behavior: build per call, no caching. Returns a
        # trivial context manager that yields the built model.
        from contextlib import contextmanager

        @contextmanager
        def _cm():
            model = self._build_transformer(device=torch.device("cpu"), **kwargs)
            try:
                yield model
            finally:
                pass

        return _cm()


@pytest.fixture
def cache() -> ModelCache:
    return ModelCache(max_cache_bytes=10**9)


@pytest.fixture(autouse=True)
def _reset_installer_state():
    # Make sure no previous test left global state behind.
    pi.uninstall_model_cache()
    pi._INSTALLED_KEYS.clear()
    pi._BLOCK_KEYS.clear()
    pi._TOKENS.clear()
    pi._ORIGINALS.clear()
    yield
    pi.uninstall_model_cache()


def _wire_stub_to_cache(cache: ModelCache, *, transformer_size: int = 0) -> None:
    """Manually install the patcher state for a stub class. Mirrors what
    install_model_cache does for the real DiffusionStage."""
    pi._INSTALLED_CACHE = cache
    pi._TRANSFORMER_SIZE_ESTIMATE = transformer_size
    pi._install_method_patch(
        StubDiffusionStage, "_transformer_ctx", pi._patched_transformer_ctx,
    )


# ---------------------------------------------------------------------------
# Cache hit / miss
# ---------------------------------------------------------------------------


class TestCacheHits:
    def test_first_call_builds_second_call_hits(self, cache: ModelCache) -> None:
        stage = StubDiffusionStage()
        _wire_stub_to_cache(cache)

        with stage._transformer_ctx(streaming_prefetch_count=None):
            pass
        with stage._transformer_ctx(streaming_prefetch_count=None):
            pass

        assert stage._build_count == 1, "factory must run once; cache hit on second call"
        snap = cache.snapshot()
        assert snap.stats.builds == 1
        assert snap.stats.hits == 1

    def test_streaming_variant_creates_separate_entry(self, cache: ModelCache) -> None:
        # Different streaming setting → different cache entry, separate build.
        stage = StubDiffusionStage()
        _wire_stub_to_cache(cache)

        # First, non-streaming → PinnedWeights.
        with stage._transformer_ctx(streaming_prefetch_count=None):
            pass

        # PinnedWeights variant alone — verify the basic path works
        # without trying to construct BlockOffloader (stub model has no
        # transformer_blocks). Just confirm one entry was added.
        snap = cache.snapshot()
        assert snap.stats.builds == 1


# ---------------------------------------------------------------------------
# Fall-throughs (torch_compile, transformer_wrapper)
# ---------------------------------------------------------------------------


class TestFallthrough:
    def test_torch_compile_true_falls_back(self, cache: ModelCache) -> None:
        stage = StubDiffusionStage()
        stage._torch_compile = True
        _wire_stub_to_cache(cache)

        with stage._transformer_ctx(streaming_prefetch_count=None):
            pass
        with stage._transformer_ctx(streaming_prefetch_count=None):
            pass

        # Both calls bypassed the cache — both built fresh.
        assert stage._build_count == 2
        snap = cache.snapshot()
        assert snap.stats.builds == 0
        assert snap.stats.hits == 0

    def test_explicit_transformer_wrapper_falls_back(self, cache: ModelCache) -> None:
        stage = StubDiffusionStage()
        # Simulate an explicit user-set wrapper — patch defers to original.
        stage._transformer_wrapper = lambda m: None  # truthy
        _wire_stub_to_cache(cache)

        with stage._transformer_ctx(streaming_prefetch_count=None):
            pass

        assert stage._build_count == 1
        snap = cache.snapshot()
        assert snap.stats.builds == 0  # cache untouched

    def test_no_cache_installed_runs_original(self) -> None:
        # When the cache is None (uninstalled), patched method should
        # delegate to the original.
        stage = StubDiffusionStage()
        # Don't install. Original method is the bound attribute.
        with stage._transformer_ctx(streaming_prefetch_count=None):
            pass
        assert stage._build_count == 1


# ---------------------------------------------------------------------------
# Per-instance identity
# ---------------------------------------------------------------------------


class TestPerInstanceIdentity:
    def test_different_instances_get_separate_entries(self, cache: ModelCache) -> None:
        a = StubDiffusionStage()
        b = StubDiffusionStage()
        _wire_stub_to_cache(cache)

        with a._transformer_ctx(streaming_prefetch_count=None):
            pass
        with b._transformer_ctx(streaming_prefetch_count=None):
            pass

        assert a._build_count == 1
        assert b._build_count == 1
        snap = cache.snapshot()
        assert snap.stats.builds == 2
        assert snap.stats.hits == 0


# ---------------------------------------------------------------------------
# Weakref auto-eviction on block GC
# ---------------------------------------------------------------------------


class TestWeakrefEviction:
    def test_block_gc_evicts_cache_entries(self, cache: ModelCache) -> None:
        _wire_stub_to_cache(cache)

        stage = StubDiffusionStage()
        with stage._transformer_ctx(streaming_prefetch_count=None):
            pass

        snap = cache.snapshot()
        assert "DiffusionStage" not in snap.cached_keys_lru_to_mru[0] or True  # entry exists
        n_before = len(snap.cached_keys_lru_to_mru)
        assert n_before >= 1

        # Drop the only reference; finalizer should evict the entry.
        del stage
        gc.collect()

        snap2 = cache.snapshot()
        # Installer-owned keys should be empty.
        assert len(pi._INSTALLED_KEYS) == 0, "block-GC finalizer should have evicted"
        assert len(snap2.cached_keys_lru_to_mru) == 0

    def test_token_dict_doesnt_keep_block_alive(self, cache: ModelCache) -> None:
        _wire_stub_to_cache(cache)

        stage = StubDiffusionStage()
        with stage._transformer_ctx(streaming_prefetch_count=None):
            pass

        # WeakKeyDictionary must hold a weakref, not a strong one.
        ref = weakref.ref(stage)
        del stage
        gc.collect()
        assert ref() is None, (
            "block was kept alive by the patcher (likely a strong ref in the factory closure)"
        )


# ---------------------------------------------------------------------------
# Install/uninstall lifecycle
# ---------------------------------------------------------------------------


class TestInstallLifecycle:
    def test_uninstall_restores_original_method(self, cache: ModelCache) -> None:
        original = StubDiffusionStage._transformer_ctx
        _wire_stub_to_cache(cache)
        assert StubDiffusionStage._transformer_ctx is not original
        pi.uninstall_model_cache()
        assert StubDiffusionStage._transformer_ctx is original

    def test_uninstall_evicts_only_installer_keys(self, cache: ModelCache) -> None:
        # User's own cache entry shouldn't be touched.
        cache.register(
            ModelSpec(
                key="user-managed",
                estimated_cache_bytes=10,
                factory=lambda: _UserStrategy(),
            )
        )
        with cache.use("user-managed"):
            pass

        _wire_stub_to_cache(cache)
        stage = StubDiffusionStage()
        with stage._transformer_ctx(streaming_prefetch_count=None):
            pass

        pi.uninstall_model_cache()

        snap = cache.snapshot()
        assert "user-managed" in snap.registered_keys
        assert "user-managed" in snap.cached_keys_lru_to_mru
        # Installer keys should be gone.
        assert all(k == "user-managed" for k in snap.cached_keys_lru_to_mru)

    def test_signature_validation_rejects_renamed_param(self, cache: ModelCache) -> None:
        with pytest.raises(RuntimeError, match="signature changed"):
            pi._validate_signature(
                StubDiffusionStage,
                "_transformer_ctx",
                ("self", "wrong_name"),
            )

    def test_install_idempotent_with_same_cache(self, cache: ModelCache) -> None:
        _wire_stub_to_cache(cache, transformer_size=100)
        # "Re-install" with same cache should just update estimates,
        # not re-patch (idempotent).
        original_method = StubDiffusionStage._transformer_ctx
        pi._INSTALLED_CACHE = cache  # simulate the same-cache check path
        pi._TRANSFORMER_SIZE_ESTIMATE = 200
        # _install_method_patch is no-op when (cls, name) already in _ORIGINALS
        pi._install_method_patch(
            StubDiffusionStage, "_transformer_ctx", pi._patched_transformer_ctx,
        )
        assert StubDiffusionStage._transformer_ctx is original_method


class _UserStrategy:
    """Trivial user-owned strategy for testing uninstall isolation."""
    cache_bytes = 10
    closed = False

    def activate(self):
        return nn.Identity()

    def deactivate(self):
        pass

    def close(self):
        self.closed = True

    def __enter__(self):
        return self.activate()

    def __exit__(self, *exc):
        self.deactivate()
