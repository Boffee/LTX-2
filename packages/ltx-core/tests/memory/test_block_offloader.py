"""Tests for ``ltx_core.memory.block_offloader.BlockOffloader``.

Most lifecycle tests run on CPU (the offloader's setup/teardown logic
is device-agnostic); CUDA-specific tests gate on availability.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from ltx_core.memory import BlockOffloader, ModelStrategy

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _make_block_model(num_blocks: int = 4, width: int = 8) -> nn.Module:
    """Tiny transformer-shaped model: nn.ModuleList of Linear blocks
    plus a non-block embed/head for non-block-resident testing.
    All params frozen."""

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(width, width, bias=False)
            self.transformer_blocks = nn.ModuleList(
                [nn.Linear(width, width, bias=False) for _ in range(num_blocks)]
            )
            self.head = nn.Linear(width, width, bias=False)

        def forward(self, x):
            x = self.embed(x)
            for block in self.transformer_blocks:
                x = block(x)
            return self.head(x)

    m = TinyModel()
    for p in m.parameters():
        p.requires_grad = False
    return m


# ---------------------------------------------------------------------------
# ModelStrategy conformance
# ---------------------------------------------------------------------------


class TestModelStrategyConformance:
    @CUDA
    def test_isinstance_runtime_check(self) -> None:
        m = _make_block_model()
        off = BlockOffloader(m, torch.device("cuda"), blocks_to_swap=2, layers_attr="transformer_blocks")
        try:
            assert isinstance(off, ModelStrategy)
        finally:
            off.close()

    @CUDA
    def test_has_lifecycle_methods(self) -> None:
        m = _make_block_model()
        off = BlockOffloader(
            m, torch.device("cuda"), blocks_to_swap=2,
            layers_attr="transformer_blocks", auto_setup=False,
        )
        try:
            assert callable(off.prepare)
            assert callable(off.activate)
            assert callable(off.deactivate)
            assert callable(off.close)
            assert isinstance(off.cache_bytes, int)
            assert isinstance(off.closed, bool)
        finally:
            off.close()


# ---------------------------------------------------------------------------
# auto_setup default vs explicit lifecycle
# ---------------------------------------------------------------------------


class TestAutoSetup:
    @CUDA
    def test_auto_setup_true_yields_active_offloader(self) -> None:
        # Default behavior preserved: trainer-style construction
        # immediately yields a fully-active offloader, hooks installed,
        # blocks pre-loaded.
        m = _make_block_model()
        off = BlockOffloader(
            m, torch.device("cuda"), blocks_to_swap=2, layers_attr="transformer_blocks",
        )
        try:
            assert off._prepared
            assert off._active
            assert len(off._hooks) == 4  # one per block
        finally:
            off.close()

    @CUDA
    def test_auto_setup_false_stays_constructed(self) -> None:
        m = _make_block_model()
        off = BlockOffloader(
            m, torch.device("cuda"), blocks_to_swap=2,
            layers_attr="transformer_blocks", auto_setup=False,
        )
        try:
            assert not off._prepared
            assert not off._active
            assert off._store is None
            assert off.cache_bytes == 0
        finally:
            off.close()


# ---------------------------------------------------------------------------
# Lifecycle transitions
# ---------------------------------------------------------------------------


class TestLifecycle:
    @CUDA
    def test_constructed_to_prepared(self) -> None:
        m = _make_block_model()
        off = BlockOffloader(
            m, torch.device("cuda"), blocks_to_swap=2,
            layers_attr="transformer_blocks", auto_setup=False,
        )
        try:
            off.prepare()
            assert off._prepared
            assert not off._active
            assert off._store is not None
            assert off.cache_bytes > 0
            # Pre-prepare: no hooks, no executor, no stream.
            assert not off._hooks
            assert off._executor is None
            assert off._stream is None
        finally:
            off.close()

    @CUDA
    def test_prepared_to_active(self) -> None:
        m = _make_block_model()
        off = BlockOffloader(
            m, torch.device("cuda"), blocks_to_swap=2,
            layers_attr="transformer_blocks", auto_setup=False,
        )
        try:
            off.prepare()
            cache_bytes_prepared = off.cache_bytes
            returned = off.activate()
            assert returned is m
            assert off._active
            assert len(off._hooks) == 4
            assert off._executor is not None
            assert off._stream is not None
            # cache_bytes unchanged across activate (only pinned CPU counts).
            assert off.cache_bytes == cache_bytes_prepared
        finally:
            off.close()

    @CUDA
    def test_active_to_prepared_via_deactivate(self) -> None:
        m = _make_block_model()
        off = BlockOffloader(
            m, torch.device("cuda"), blocks_to_swap=2,
            layers_attr="transformer_blocks", auto_setup=False,
        )
        try:
            off.prepare()
            off.activate()
            cache_bytes_active = off.cache_bytes
            off.deactivate()
            # Active resources released, pinned store retained.
            assert not off._active
            assert off._prepared
            assert not off._hooks
            assert off._executor is None
            assert off._stream is None
            assert off.cache_bytes == cache_bytes_active
        finally:
            off.close()

    @CUDA
    def test_reactivation_reuses_pinned_store(self) -> None:
        m = _make_block_model()
        off = BlockOffloader(
            m, torch.device("cuda"), blocks_to_swap=2,
            layers_attr="transformer_blocks", auto_setup=False,
        )
        try:
            off.prepare()
            store_before = off._store
            off.activate()
            off.deactivate()
            off.activate()
            # Same pinned store across activate/deactivate cycles.
            assert off._store is store_before
            off.deactivate()
        finally:
            off.close()

    @CUDA
    def test_activate_auto_prepares(self) -> None:
        # Ergonomics: skipping prepare() and going straight to activate()
        # should auto-prepare for callers who don't care about phasing.
        m = _make_block_model()
        off = BlockOffloader(
            m, torch.device("cuda"), blocks_to_swap=2,
            layers_attr="transformer_blocks", auto_setup=False,
        )
        try:
            off.activate()
            assert off._prepared
            assert off._active
            off.deactivate()
        finally:
            off.close()

    @CUDA
    def test_activate_not_reentrant(self) -> None:
        m = _make_block_model()
        off = BlockOffloader(
            m, torch.device("cuda"), blocks_to_swap=2,
            layers_attr="transformer_blocks", auto_setup=False,
        )
        try:
            off.activate()
            with pytest.raises(RuntimeError, match="not re-entrant"):
                off.activate()
        finally:
            off.close()

    @CUDA
    def test_deactivate_when_not_active_is_noop(self) -> None:
        m = _make_block_model()
        off = BlockOffloader(
            m, torch.device("cuda"), blocks_to_swap=2,
            layers_attr="transformer_blocks", auto_setup=False,
        )
        try:
            off.deactivate()  # no error
            off.prepare()
            off.deactivate()  # still no error
        finally:
            off.close()


# ---------------------------------------------------------------------------
# close() destructiveness — matches PinnedWeights pattern
# ---------------------------------------------------------------------------


class TestClose:
    @CUDA
    def test_close_moves_model_to_meta(self) -> None:
        m = _make_block_model()
        off = BlockOffloader(
            m, torch.device("cuda"), blocks_to_swap=2, layers_attr="transformer_blocks",
        )
        off.close()
        for p in m.parameters():
            assert p.device.type == "meta"

    @CUDA
    def test_close_is_idempotent(self) -> None:
        m = _make_block_model()
        off = BlockOffloader(
            m, torch.device("cuda"), blocks_to_swap=2, layers_attr="transformer_blocks",
        )
        off.close()
        assert off.closed
        off.close()  # no error
        assert off.closed

    @CUDA
    def test_close_deactivates_first(self) -> None:
        m = _make_block_model()
        off = BlockOffloader(
            m, torch.device("cuda"), blocks_to_swap=2, layers_attr="transformer_blocks",
        )
        # Active when close() runs — should deactivate, then destroy.
        assert off._active
        off.close()
        assert not off._active
        assert off.closed

    @CUDA
    def test_activate_after_close_raises(self) -> None:
        m = _make_block_model()
        off = BlockOffloader(
            m, torch.device("cuda"), blocks_to_swap=2,
            layers_attr="transformer_blocks", auto_setup=False,
        )
        off.close()
        with pytest.raises(RuntimeError, match="closed"):
            off.activate()

    @CUDA
    def test_prepare_after_close_raises(self) -> None:
        m = _make_block_model()
        off = BlockOffloader(
            m, torch.device("cuda"), blocks_to_swap=2,
            layers_attr="transformer_blocks", auto_setup=False,
        )
        off.close()
        with pytest.raises(RuntimeError, match="closed"):
            off.prepare()


# ---------------------------------------------------------------------------
# Back-compat aliases
# ---------------------------------------------------------------------------


class TestBackCompat:
    @CUDA
    def test_setup_after_construct_prepares_and_activates(self) -> None:
        m = _make_block_model()
        off = BlockOffloader(
            m, torch.device("cuda"), blocks_to_swap=2,
            layers_attr="transformer_blocks", auto_setup=False,
        )
        try:
            off.setup()
            assert off._prepared
            assert off._active
        finally:
            off.close()

    @CUDA
    def test_setup_when_active_is_noop(self) -> None:
        m = _make_block_model()
        off = BlockOffloader(
            m, torch.device("cuda"), blocks_to_swap=2, layers_attr="transformer_blocks",
        )
        try:
            assert off._active
            off.setup()  # idempotent when already active
            assert off._active
        finally:
            off.close()

    @CUDA
    def test_setup_after_close_raises(self) -> None:
        m = _make_block_model()
        off = BlockOffloader(
            m, torch.device("cuda"), blocks_to_swap=2, layers_attr="transformer_blocks",
        )
        off.close()
        with pytest.raises(RuntimeError, match="closed"):
            off.setup()

    @CUDA
    def test_teardown_is_destructive(self) -> None:
        # Critical for shard_orchestrator.py: teardown() between shards
        # must break the forward-hook reference cycle by destroying
        # everything, not just deactivating.
        m = _make_block_model()
        off = BlockOffloader(
            m, torch.device("cuda"), blocks_to_swap=2, layers_attr="transformer_blocks",
        )
        off.teardown()
        # Destructive: model on meta, offloader closed.
        assert off.closed
        for p in m.parameters():
            assert p.device.type == "meta"


# ---------------------------------------------------------------------------
# Hook lifecycle
# ---------------------------------------------------------------------------


class TestHookLifecycle:
    @CUDA
    def test_hooks_removed_on_deactivate(self) -> None:
        m = _make_block_model()
        off = BlockOffloader(
            m, torch.device("cuda"), blocks_to_swap=2, layers_attr="transformer_blocks",
        )
        try:
            assert len(off._hooks) == 4
            off.deactivate()
            assert not off._hooks
            # PyTorch's per-module pre-hooks dict should be empty for the blocks.
            for block in m.transformer_blocks:
                assert len(block._forward_pre_hooks) == 0
        finally:
            off.close()

    @CUDA
    def test_hooks_removed_on_close(self) -> None:
        m = _make_block_model()
        off = BlockOffloader(
            m, torch.device("cuda"), blocks_to_swap=2, layers_attr="transformer_blocks",
        )
        # Capture references before close.
        blocks = list(m.transformer_blocks)
        off.close()
        for block in blocks:
            assert len(block._forward_pre_hooks) == 0


# ---------------------------------------------------------------------------
# Forward correctness across cycles (smoke test)
# ---------------------------------------------------------------------------


class TestForwardCorrectness:
    @CUDA
    def test_forward_matches_eager_baseline(self) -> None:
        # Smoke test: with the offloader active, forward should produce
        # the same output (modulo small floating-point noise from
        # bfloat16 round trips).
        torch.manual_seed(42)
        m = _make_block_model(num_blocks=4, width=8).cuda()
        x = torch.randn(2, 8, device="cuda")
        with torch.no_grad():
            expected = m(x)

        # Re-create on CPU and offload.
        torch.manual_seed(42)
        m_off = _make_block_model(num_blocks=4, width=8)
        off = BlockOffloader(
            m_off, torch.device("cuda"), blocks_to_swap=2, layers_attr="transformer_blocks",
        )
        try:
            with torch.no_grad():
                got = m_off(x)
            torch.cuda.synchronize()
            torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)
        finally:
            off.close()

    @CUDA
    def test_forward_after_deactivate_then_activate_cycle(self) -> None:
        # Cycle: build → use → deactivate → use again. Both forward
        # passes must produce identical outputs.
        torch.manual_seed(42)
        m = _make_block_model(num_blocks=4, width=8)
        off = BlockOffloader(
            m, torch.device("cuda"), blocks_to_swap=2,
            layers_attr="transformer_blocks", auto_setup=False,
        )
        try:
            x = torch.randn(2, 8, device="cuda")
            off.activate()
            with torch.no_grad():
                first = m(x)
            torch.cuda.synchronize()
            off.deactivate()

            off.activate()
            with torch.no_grad():
                second = m(x)
            torch.cuda.synchronize()
            torch.testing.assert_close(first, second)
            off.deactivate()
        finally:
            off.close()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_blocks_to_swap_must_be_lt_num_layers(self) -> None:
        m = _make_block_model(num_blocks=4)
        with pytest.raises(ValueError, match="blocks_to_swap"):
            BlockOffloader(
                m, torch.device("cpu"), blocks_to_swap=4,  # equal to num_layers
                layers_attr="transformer_blocks", auto_setup=False,
            ).prepare()


# ---------------------------------------------------------------------------
# ModelCache integration via prepared-factory pattern
# ---------------------------------------------------------------------------


class TestModelCacheIntegration:
    @CUDA
    def test_prepared_factory_works_with_model_cache(self) -> None:
        # The documented pattern for caching a BlockOffloader: factory
        # calls prepare() so the cache sees the correct cache_bytes
        # before activating.
        from ltx_core.memory import ModelCache, ModelSpec

        device = torch.device("cuda")

        def factory():
            m = _make_block_model(num_blocks=4, width=8)
            off = BlockOffloader(
                m, device, blocks_to_swap=2,
                layers_attr="transformer_blocks", auto_setup=False,
            )
            off.prepare()
            return off

        # Conservative estimate; cache will reconcile against actual.
        cache = ModelCache(max_cache_bytes=10_000_000)
        spec = ModelSpec(key="xformer", estimated_cache_bytes=1024, factory=factory)

        with cache.use(spec) as model:
            assert isinstance(model, nn.Module)
            x = torch.randn(2, 8, device=device)
            with torch.no_grad():
                _ = model(x)
            torch.cuda.synchronize()

        # After exit: deactivated, but pinned store retained.
        info = cache.info("xformer")
        assert info.cached
        assert info.cache_bytes is not None
        assert info.cache_bytes > 0
        assert info.active_count == 0

        # Second use is a cache hit — no rebuild.
        with cache.use("xformer"):
            pass
        snap = cache.snapshot()
        assert snap.stats.builds == 1
        assert snap.stats.hits == 1

        cache.clear()


# ---------------------------------------------------------------------------
# Activation rollback failure
# ---------------------------------------------------------------------------


class TestActivationRollbackFailure:
    @CUDA
    def test_active_flag_stays_true_when_rollback_fails(self, monkeypatch) -> None:
        # If _teardown_active_resources fails during rollback, _active
        # must NOT be set to False — leaving it True ensures a later
        # close() will re-attempt cleanup of partial resources.
        m = _make_block_model()
        off = BlockOffloader(
            m, torch.device("cuda"), blocks_to_swap=2,
            layers_attr="transformer_blocks", auto_setup=False,
        )
        try:
            off.prepare()

            # Simulate activate failing AFTER hooks are installed AND
            # rollback failing too.
            original_register_hooks = off._register_hooks
            original_teardown = off._teardown_active_resources

            def broken_register_hooks(*args, **kwargs):
                original_register_hooks(*args, **kwargs)
                raise RuntimeError("simulated activate failure")

            def broken_teardown(*, suppress_prefetch_errors):
                raise RuntimeError("simulated rollback failure")

            monkeypatch.setattr(off, "_register_hooks", broken_register_hooks)
            monkeypatch.setattr(off, "_teardown_active_resources", broken_teardown)

            with pytest.raises(RuntimeError, match="simulated activate failure"):
                off.activate()

            # Rollback failed — _active stays True so close() will
            # re-attempt cleanup of the leaked resources.
            assert off._active is True
        finally:
            # Restore original teardown so close() can do its work.
            monkeypatch.setattr(off, "_teardown_active_resources", original_teardown)
            off.close()


# ---------------------------------------------------------------------------
# Pending prefetch failure during deactivate
# ---------------------------------------------------------------------------


class TestPrefetchFailureOnDeactivate:
    @CUDA
    def test_prefetch_failure_propagates_after_cleanup(self) -> None:
        # If a pending prefetch future raises during deactivate, the
        # cleanup still completes (hooks removed, pool released, etc.)
        # but the first prefetch exception is re-raised so ModelCache
        # treats the strategy as poisoned.
        from concurrent.futures import Future

        m = _make_block_model()
        off = BlockOffloader(
            m, torch.device("cuda"), blocks_to_swap=2, layers_attr="transformer_blocks",
        )
        # Inject a pre-failed Future into _pending so deactivate's
        # drain loop encounters it.
        bad_future: Future[None] = Future()
        bad_future.set_exception(RuntimeError("simulated prefetch failure"))
        off._pending[0] = bad_future

        with pytest.raises(RuntimeError, match="simulated prefetch failure"):
            off.deactivate()

        # Even though we raised, cleanup completed: hooks gone, executor
        # gone, _active is False.
        assert not off._hooks
        assert off._executor is None
        assert not off._active

        # close() should still work cleanly (no hooks/executor to clean up).
        off.close()
        assert off.closed


# ---------------------------------------------------------------------------
# BlockPinnedStore.activate_pool idempotency
# ---------------------------------------------------------------------------


class TestActivatePoolIdempotency:
    @CUDA
    def test_same_config_idempotent(self) -> None:
        from ltx_core.memory.block_offloader import BlockPinnedStore

        m = _make_block_model()
        store = BlockPinnedStore(list(m.transformer_blocks))
        store.activate_pool(2, torch.device("cuda"))
        pool_first = store._pool
        store.activate_pool(2, torch.device("cuda"))  # same config — no-op
        assert store._pool is pool_first

    @CUDA
    def test_mismatched_config_raises(self) -> None:
        from ltx_core.memory.block_offloader import BlockPinnedStore

        m = _make_block_model()
        store = BlockPinnedStore(list(m.transformer_blocks))
        store.activate_pool(2, torch.device("cuda"))
        with pytest.raises(ValueError, match="already activated"):
            store.activate_pool(3, torch.device("cuda"))
        with pytest.raises(ValueError, match="already activated"):
            store.activate_pool(2, torch.device("cpu"))
