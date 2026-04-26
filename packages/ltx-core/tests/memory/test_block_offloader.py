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
# PR 3b: prepared state truly inactive — non-block on pinned CPU
# ---------------------------------------------------------------------------


class TestPreparedStateInactive:
    """Verifies that the 'prepared but not active' state has no GPU
    footprint — the payoff of PR 3b's non-block PinnedWeights composition.
    Without it, non-block siblings (embed, head, norms) would sit on
    target_device permanently, defeating ModelCache eviction."""

    @CUDA
    def test_prepared_has_no_params_on_target_device(self) -> None:
        m = _make_block_model(num_blocks=4, width=8)
        target = torch.device("cuda")
        off = BlockOffloader(
            m, target, blocks_to_swap=2,
            layers_attr="transformer_blocks", auto_setup=False,
        )
        try:
            off.prepare()
            # No frozen params on target device — block params are pinned
            # CPU, non-block params are pinned CPU via the inner
            # PinnedWeights, trainable (none here) would be on CPU too.
            for p in m.parameters():
                assert p.device != target, (
                    f"prepared state leaked GPU residency: {p.shape}@{p.device}"
                )
        finally:
            off.close()

    @CUDA
    def test_non_block_pinned_after_prepare(self) -> None:
        m = _make_block_model(num_blocks=4, width=8)
        off = BlockOffloader(
            m, torch.device("cuda"), blocks_to_swap=2,
            layers_attr="transformer_blocks", auto_setup=False,
        )
        try:
            off.prepare()
            # Non-block PinnedWeights instance was created (embed + head
            # are frozen and non-block).
            assert off._non_block_pinned is not None
            assert off._non_block_pinned.cache_bytes > 0
            # cache_bytes includes both block and non-block contributions.
            assert off.cache_bytes > off._store.cache_bytes
        finally:
            off.close()

    @CUDA
    def test_activate_brings_non_block_to_gpu(self) -> None:
        m = _make_block_model(num_blocks=4, width=8)
        target = torch.device("cuda")
        off = BlockOffloader(
            m, target, blocks_to_swap=2, layers_attr="transformer_blocks",
        )
        try:
            # After activate, embed and head (non-block) are on GPU.
            assert m.embed.weight.is_cuda
            assert m.head.weight.is_cuda
        finally:
            off.close()

    @CUDA
    def test_deactivate_returns_non_block_to_pinned(self) -> None:
        m = _make_block_model(num_blocks=4, width=8)
        target = torch.device("cuda")
        off = BlockOffloader(
            m, target, blocks_to_swap=2,
            layers_attr="transformer_blocks", auto_setup=False,
        )
        try:
            off.prepare()
            off.activate()
            assert m.embed.weight.is_cuda
            off.deactivate()
            # Back to pinned CPU, NOT on target device.
            assert m.embed.weight.device != target
            assert m.embed.weight.is_pinned()
            assert m.head.weight.is_pinned()
        finally:
            off.close()

    @CUDA
    def test_buffer_only_non_block_module(self) -> None:
        # A non-block sibling with only registered buffers (e.g., a
        # RoPE position table) and no learnable params. Previously
        # PinnedWeights would refuse to wrap it and we'd silently
        # leave the buffers on CPU — forward with CUDA inputs would
        # crash. Now PinnedWeights pins buffer-only modules and the
        # buffers correctly round-trip on activate/deactivate.
        class RopeTable(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("table", torch.randn(8, 4))

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.rope = RopeTable()
                self.transformer_blocks = nn.ModuleList(
                    [nn.Linear(4, 4, bias=False) for _ in range(4)]
                )

        m = M()
        for p in m.parameters():
            p.requires_grad = False
        target = torch.device("cuda")
        off = BlockOffloader(
            m, target, blocks_to_swap=2,
            layers_attr="transformer_blocks", auto_setup=False,
        )
        try:
            off.prepare()
            # PinnedWeights was constructed for the buffer-only RoPE
            # sibling — buffers are now pinned CPU.
            assert off._non_block_pinned is not None
            assert m.rope.table.is_pinned()
            off.activate()
            # On activate, the buffer is moved to GPU.
            assert m.rope.table.is_cuda
            off.deactivate()
            # And back to pinned CPU on deactivate.
            assert m.rope.table.is_pinned()
        finally:
            off.close()

    def test_block_only_model_has_no_non_block_pinned(self) -> None:
        # Edge case: model whose only top-level child IS the block list
        # (no patchifier/head/etc). Non-block wrapper is None; cache_bytes
        # comes purely from blocks.
        class BlockOnly(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer_blocks = nn.ModuleList(
                    [nn.Linear(4, 4, bias=False) for _ in range(4)]
                )

        m = BlockOnly()
        for p in m.parameters():
            p.requires_grad = False
        off = BlockOffloader(
            m, torch.device("cpu"), blocks_to_swap=2,
            layers_attr="transformer_blocks", auto_setup=False,
        )
        try:
            off.prepare()
            assert off._non_block_pinned is None
            assert off._non_block_wrapper is None
            assert off.cache_bytes > 0  # block bytes only
        finally:
            off.close()


# ---------------------------------------------------------------------------
# Cross-region tied-weight detection
# ---------------------------------------------------------------------------


class TestCrossRegionTiedDetection:
    def test_cross_block_tied_raises(self) -> None:
        # Two blocks share a frozen weight via tied storage. Slot-local
        # streaming can't preserve this; must raise at prepare().
        shared = torch.randn(8, 8)
        block_0 = nn.Linear(8, 8, bias=False)
        block_1 = nn.Linear(8, 8, bias=False)
        # Tie block_0 and block_1 weights — distinct Parameter wrappers,
        # same storage, exposed in named_parameters under different
        # qualified names.
        block_0.weight = nn.Parameter(shared, requires_grad=False)
        block_1.weight = nn.Parameter(shared, requires_grad=False)

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer_blocks = nn.ModuleList([block_0, block_1])

        m = M()
        for p in m.parameters():
            p.requires_grad = False
        off = BlockOffloader(
            m, torch.device("cpu"), blocks_to_swap=1,
            layers_attr="transformer_blocks", auto_setup=False,
        )
        with pytest.raises(ValueError, match="cross-region|tied frozen"):
            off.prepare()
        off.close()

    def test_block_to_non_block_tied_raises(self) -> None:
        shared = torch.randn(4, 4)
        block_0 = nn.Linear(4, 4, bias=False)
        block_0.weight = nn.Parameter(shared, requires_grad=False)
        head = nn.Linear(4, 4, bias=False)
        head.weight = nn.Parameter(shared, requires_grad=False)  # tied to block

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer_blocks = nn.ModuleList(
                    [block_0, nn.Linear(4, 4, bias=False)]
                )
                self.head = head

        m = M()
        for p in m.parameters():
            p.requires_grad = False
        off = BlockOffloader(
            m, torch.device("cpu"), blocks_to_swap=1,
            layers_attr="transformer_blocks", auto_setup=False,
        )
        with pytest.raises(ValueError, match="cross-region|tied frozen"):
            off.prepare()
        off.close()

    def test_mixed_trainable_frozen_cross_region_tied_raises(self) -> None:
        # A trainable block param tied to a frozen non-block param: if
        # not detected, the frozen side gets pinned/swapped while the
        # trainable side is moved separately on activate, silently
        # breaking the tie. Detection now ignores requires_grad.
        shared = torch.randn(4, 4)
        block_0 = nn.Linear(4, 4, bias=False)
        block_0.weight = nn.Parameter(shared, requires_grad=True)  # trainable
        head = nn.Linear(4, 4, bias=False)
        head.weight = nn.Parameter(shared, requires_grad=False)  # frozen, tied

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer_blocks = nn.ModuleList(
                    [block_0, nn.Linear(4, 4, bias=False)]
                )
                self.head = head

        m = M()
        # Make non-block-1 frozen so we don't trip blocks_to_swap validation
        for p in m.transformer_blocks[1].parameters():
            p.requires_grad = False
        off = BlockOffloader(
            m, torch.device("cpu"), blocks_to_swap=1,
            layers_attr="transformer_blocks", auto_setup=False,
        )
        with pytest.raises(ValueError, match="cross-region|tied"):
            off.prepare()
        off.close()

    def test_intra_block_tied_raises(self) -> None:
        # Two slots WITHIN one block share storage. _BlockPinnedStore
        # uses default named_parameters (remove_duplicate=True) and
        # would only swap one alias, leaving the other pointing at
        # non-pinned data. Detect+reject rather than silently break.
        shared = torch.randn(8, 8)

        class TiedBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn_q = nn.Linear(8, 8, bias=False)
                self.attn_k = nn.Linear(8, 8, bias=False)
                # Tie within the block: distinct Parameter objects, same storage.
                self.attn_q.weight = nn.Parameter(shared, requires_grad=False)
                self.attn_k.weight = nn.Parameter(shared, requires_grad=False)

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer_blocks = nn.ModuleList(
                    [TiedBlock(), nn.Linear(8, 8, bias=False)]
                )

        m = M()
        for p in m.parameters():
            p.requires_grad = False
        off = BlockOffloader(
            m, torch.device("cpu"), blocks_to_swap=1,
            layers_attr="transformer_blocks", auto_setup=False,
        )
        with pytest.raises(ValueError, match="intra-block tied"):
            off.prepare()
        off.close()

    def test_non_block_internal_tied_works(self) -> None:
        # Tied embed↔head WITHIN non-block region: PinnedWeights
        # composition handles this via its own dedup. Should not raise.
        embed = nn.Embedding(16, 8)
        head = nn.Linear(8, 16, bias=False)
        head.weight = embed.weight  # standard tie_weights() pattern

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = embed
                self.transformer_blocks = nn.ModuleList(
                    [nn.Linear(8, 8, bias=False) for _ in range(4)]
                )
                self.head = head

        m = M()
        for p in m.parameters():
            p.requires_grad = False
        off = BlockOffloader(
            m, torch.device("cpu"), blocks_to_swap=2,
            layers_attr="transformer_blocks", auto_setup=False,
        )
        try:
            off.prepare()
            # Non-block PinnedWeights deduped the tie — single slot.
            assert len(off._non_block_pinned._slots) == 1
            # Both names still tied at the Parameter level.
            assert m.embed.weight is m.head.weight
        finally:
            off.close()


# ---------------------------------------------------------------------------
# Direct-parent state rejection
# ---------------------------------------------------------------------------


class TestDirectParentStateRejection:
    def test_direct_frozen_param_on_root_raises(self) -> None:
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                # Direct frozen param on root (not via a child module).
                self.weight = nn.Parameter(torch.randn(4), requires_grad=False)
                self.transformer_blocks = nn.ModuleList(
                    [nn.Linear(4, 4, bias=False) for _ in range(4)]
                )

        m = M()
        for p in m.parameters():
            p.requires_grad = False
        off = BlockOffloader(
            m, torch.device("cpu"), blocks_to_swap=2,
            layers_attr="transformer_blocks", auto_setup=False,
        )
        with pytest.raises(ValueError, match="directly to a parent module"):
            off.prepare()
        off.close()

    def test_direct_param_on_ancestor_when_layers_attr_nested(self) -> None:
        # Nested layers_attr like "encoder.blocks": parent_paths is
        # {"encoder"}. Direct frozen state on the ROOT (an ancestor of
        # the parent) must still be detected — the wrapper only walks
        # encoder's children and would miss it.
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                # Direct frozen param on root, ABOVE the parent path.
                self.weight = nn.Parameter(torch.randn(4), requires_grad=False)
                self.encoder = nn.Module()
                self.encoder.blocks = nn.ModuleList(
                    [nn.Linear(4, 4, bias=False) for _ in range(4)]
                )

        m = M()
        for p in m.parameters():
            p.requires_grad = False
        off = BlockOffloader(
            m, torch.device("cpu"), blocks_to_swap=2,
            layers_attr="encoder.blocks", auto_setup=False,
        )
        with pytest.raises(ValueError, match="directly to a parent module"):
            off.prepare()
        off.close()

    def test_direct_trainable_tied_to_frozen_block_raises(self) -> None:
        # Mixed-tie edge case: direct trainable param on root sharing
        # storage with a frozen block param. The frozen side gets
        # pinned and slot-swapped; the trainable side is moved
        # separately on activate, breaking the tie. Cross-region
        # detection must catch this.
        shared = torch.randn(4, 4)
        block_0 = nn.Linear(4, 4, bias=False)
        block_0.weight = nn.Parameter(shared, requires_grad=False)

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                # Direct TRAINABLE param tied to block_0.weight.
                self.tied_w = nn.Parameter(shared, requires_grad=True)
                self.transformer_blocks = nn.ModuleList(
                    [block_0, nn.Linear(4, 4, bias=False)]
                )

        m = M()
        for p in m.transformer_blocks[1].parameters():
            p.requires_grad = False
        off = BlockOffloader(
            m, torch.device("cpu"), blocks_to_swap=1,
            layers_attr="transformer_blocks", auto_setup=False,
        )
        with pytest.raises(ValueError, match="cross-region|tied"):
            off.prepare()
        off.close()

    def test_direct_buffer_on_root_raises(self) -> None:
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("table", torch.randn(8))
                self.transformer_blocks = nn.ModuleList(
                    [nn.Linear(4, 4, bias=False) for _ in range(4)]
                )

        m = M()
        for p in m.parameters():
            p.requires_grad = False
        off = BlockOffloader(
            m, torch.device("cpu"), blocks_to_swap=2,
            layers_attr="transformer_blocks", auto_setup=False,
        )
        with pytest.raises(ValueError, match="directly to a parent module"):
            off.prepare()
        off.close()


# ---------------------------------------------------------------------------
# Block buffers are pinned (T2 fix)
# ---------------------------------------------------------------------------


class TestBlockBuffersPinned:
    def test_block_buffer_clone_is_pinned(self) -> None:
        # Block-internal buffers must use pin_memory(), otherwise
        # cache_bytes lies AND non_blocking=True H2D copies in
        # load_block silently demote to synchronous.
        class BlockWithBuffer(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 4, bias=False)
                self.register_buffer("table", torch.randn(8))

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer_blocks = nn.ModuleList(
                    [BlockWithBuffer() for _ in range(4)]
                )

        m = M()
        for p in m.parameters():
            p.requires_grad = False
        off = BlockOffloader(
            m, torch.device("cpu"), blocks_to_swap=2,
            layers_attr="transformer_blocks", auto_setup=False,
        )
        try:
            off.prepare()
            for block in m.transformer_blocks:
                assert block.table.is_pinned(), (
                    "block buffer should be pinned for honest cache_bytes "
                    "and to avoid silently-synchronous H2D copies"
                )
        finally:
            off.close()


# ---------------------------------------------------------------------------
# _BlockPinnedStore.activate_pool idempotency
# ---------------------------------------------------------------------------


class TestActivatePoolIdempotency:
    @CUDA
    def test_same_config_idempotent(self) -> None:
        from ltx_core.memory.block_offloader import _BlockPinnedStore

        m = _make_block_model()
        store = _BlockPinnedStore(list(m.transformer_blocks))
        store.activate_pool(2, torch.device("cuda"))
        pool_first = store._pool
        store.activate_pool(2, torch.device("cuda"))  # same config — no-op
        assert store._pool is pool_first

    @CUDA
    def test_mismatched_config_raises(self) -> None:
        from ltx_core.memory.block_offloader import _BlockPinnedStore

        m = _make_block_model()
        store = _BlockPinnedStore(list(m.transformer_blocks))
        store.activate_pool(2, torch.device("cuda"))
        with pytest.raises(ValueError, match="already activated"):
            store.activate_pool(3, torch.device("cuda"))
        with pytest.raises(ValueError, match="already activated"):
            store.activate_pool(2, torch.device("cpu"))
