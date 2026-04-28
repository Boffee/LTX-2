"""Whole-model pinned-CPU weight cache for fast bulk DMA to GPU.

Holds a model's frozen weights in pinned CPU memory so subsequent GPU
loads are bulk DMA (~200 ms for a 12 GB text encoder at PCIe Gen5 x16) instead
of re-reading the safetensors from disk (~3-5 s per call).

Use case: a model that fits on GPU when active but should be evicted
between calls — text encoder during diffusion, VAE between encode and
decode phases, etc. Different from :func:`make_block_offloader`: no per-block
streaming, no forward hooks, no LRU. The whole model goes to GPU on
:meth:`PinnedWeights.activate` and the GPU storage is released on
:meth:`PinnedWeights.deactivate` by repointing each module's parameter
slot back at a Parameter that wraps pinned CPU storage.

Implements :class:`~block_offload.strategy.ModelStrategy` so it plugs
into a model cache directly.

Cross-cutting compatibility caveats (``torch.compile`` incompatibility,
DDP/FSDP wrap-before requirement, single-thread contract) live in the
:mod:`~block_offload` package docstring.

Class-specific caveats
----------------------
- The constructor *mutates* the wrapped ``model`` — each frozen
  parameter slot (``module._parameters[leaf]``) is replaced with a
  Parameter wrapping pinned CPU storage, and registered buffers are
  replaced with pinned copies. Only use the model via :meth:`activate`
  or context-manager entry after wrapping.
- Slot replacement (rather than ``param.data`` swap) is required for
  correctness with quanto ``WeightQBytesTensor``: assigning
  ``param.data = new_quanto_tensor`` is a no-op for the inner ``_data``
  / ``_scale`` storages, so the model would silently keep referencing
  the original (non-pinned) quanto wrapper.
- Buffer mutations during forward (RNN/SSM state, KV cache,
  training-mode BatchNorm running stats) are *discarded* on
  :meth:`deactivate`. Suitable for inference of stateless modules; not
  suitable for models that need persistent buffer state across calls.
- **Caller owns lifecycle correctness.** Calling :meth:`activate`
  twice without an intervening :meth:`deactivate` double-allocates
  GPU storage. After :meth:`activate` raises, the strategy is
  poisoned — drop the strategy reference and rebuild.
- There is no ``close()``. Pinned memory is freed when the caller
  drops the strategy AND model references; Python's refcount-based
  GC reclaims the pinned tensors immediately. The strategy releases
  what it owns (its internal slot tracking); the user's model is the
  user's concern.
- Tied weights *are* deduplicated. Two parameter slots whose values
  share underlying storage — whether the standard ``tie_weights()``
  pattern (one ``Parameter`` under multiple names) or the rarer case
  of distinct quanto wrappers around shared inner ``_data`` — share a
  single :class:`PinnedParamBuffer` and a single Parameter wrapper on
  activation, preserving the tying invariant on GPU.
"""

from __future__ import annotations

import logging
from types import TracebackType
from typing import Any

import torch
from torch import nn

from .pinned_buffer import PinnedParamBuffer, storage_key
from .slot_graph import iter_buffer_slots, iter_param_slots
from .strategy import SlotOwnership

logger = logging.getLogger(__name__)


def _set_buffer(module: nn.Module, name: str, value: torch.Tensor, persistent: bool) -> None:
    """Replace a registered buffer in-place by its leaf name on
    ``module``, preserving the original ``persistent`` flag so
    ``state_dict()`` behavior survives the swap."""
    module.register_buffer(name, value, persistent=persistent)


class PinnedWeights:
    """Whole-model pinned-CPU weight cache with bulk GPU transfer.

    Implements :class:`~block_offload.strategy.ModelStrategy`.

    On construction, every frozen parameter slot is replaced with a
    Parameter wrapping pinned CPU storage (handling quanto decomposition
    and tied-weight dedup). :meth:`activate` allocates GPU tensors for
    each unique pinned buffer, swaps the matching Parameter into every
    slot that pointed at that buffer, and returns the model;
    :meth:`deactivate` swaps the slots back at the pinned-CPU
    Parameters so the GPU storage is released by refcount.

    Trainable parameters (``requires_grad=True``) are not pinned.
    Buffer-only modules (only registered buffers, no frozen params)
    are valid — common for sibling tables like RoPE/positional
    embeddings managed via :func:`make_block_offloader`'s non-block
    composition. Construction raises only if there is *nothing* to
    manage — neither frozen params nor (with ``include_buffers=True``)
    registered buffers.

    Parameters
    ----------
    model:
        The model to cache. Auto-moved to CPU at construction so
        ``pin_memory()`` succeeds.
    target_device:
        GPU device to bulk-transfer to in :meth:`activate`.
    include_buffers:
        Also cache registered buffers (LayerNorm running stats, position
        embeddings stored as buffers, etc.). Default True. Set False
        for models with very large mutable buffers you'd rather rebuild
        on each call.
    skip_slots:
        Optional set of :class:`SlotOwnership` tuples identifying
        ``(parent_module, leaf, kind)`` slots to skip during the
        walk. Used by composers like :class:`BlockStreamingStrategy`
        that want to hand the *outer* model to PinnedWeights but
        manage some subset of slots themselves (e.g., the streamed
        block list). Skipped slots are not pinned. Slot identity is
        based on
        ``(id(parent), leaf, kind)`` rather than ``id(param)`` so the
        filter survives slot-mutating strategies regardless of
        construction order.
    """

    def __init__(
        self,
        model: nn.Module,
        target_device: torch.device,
        include_buffers: bool = True,
        *,
        skip_slots: set[SlotOwnership] | None = None,
    ) -> None:
        self._model: nn.Module | None = model
        self._device = target_device
        self._include_buffers = include_buffers
        self._skip_slots: set[SlotOwnership] = skip_slots or set()

        # Auto-move to CPU so pin_memory() succeeds. Matches the
        # behavior of make_block_offloader — caller doesn't need to
        # remember the build-time device dance.
        model.to("cpu")

        # Tied-weight aware pinning. We walk both named_modules and
        # named_parameters with remove_duplicate=False so:
        #   - shared submodule aliases (m.a is m.b) get visited at every
        #     alias rather than just one canonical name
        #   - the standard tie_weights() pattern (one Parameter under
        #     multiple names) shows up at every name
        # We then group by storage identity and validate requires_grad
        # uniformity per group: a tied group with mixed
        # trainable/frozen members would silently break the tying
        # invariant if we pinned only the frozen members, so we raise.
        # All-trainable groups are skipped (PinnedWeights only manages
        # frozen weights). All-frozen groups become one PinnedParamBuffer
        # whose slot-location list is deduped by (id(parent), leaf) so
        # we don't double-write into shared submodules.
        # storage_key -> list of (name, param, parent_module, leaf)
        groups: dict[tuple[Any, ...], list[tuple[str, nn.Parameter, nn.Module, str]]] = {}
        for s in iter_param_slots(model):
            if s.slot in self._skip_slots:
                continue  # composer (e.g. BlockStreamingStrategy) owns this slot
            if s.param.numel() == 0:
                # Zero-sized tensors all share data_ptr()==0; key by id(p)
                # to keep them in independent groups rather than spuriously
                # collapsing them.
                skey = ("__empty__", id(s.param), s.name)
            else:
                skey = storage_key(s.param.data)
            groups.setdefault(skey, []).append((s.name, s.param, s.parent, s.leaf))

        # Per unique buffer: (PinnedParamBuffer, list of (parent, leaf)).
        # A tied storage group with any trainable member needs copy_back so
        # in-place updates on the GPU side round-trip back to pinned host
        # storage on deactivate. Mixed-grad ties are supported: storage
        # swap (used by RegularAdapter) preserves tying because all aliases
        # observe the same underlying GPU buffer during activate.
        self._slots: list[tuple[PinnedParamBuffer, list[tuple[nn.Module, str]]]] = []
        for members in groups.values():
            copy_back = any(p.requires_grad for _, p, _, _ in members)
            first_name, first_p = members[0][0], members[0][1]
            buf = PinnedParamBuffer(first_name, first_p, copy_back=copy_back)
            seen_locs: set[tuple[int, str]] = set()
            locs: list[tuple[nn.Module, str]] = []
            for _, _, parent, leaf in members:
                key = (id(parent), leaf)
                if key in seen_locs:
                    continue
                seen_locs.add(key)
                locs.append((parent, leaf))
            self._slots.append((buf, locs))

        # Buffer pinning — Phase 1: build templates only, NO slot
        # mutation. Same alias-aware grouping as parameters: shared
        # buffer instances visible at multiple (parent, leaf) paths
        # get one pinned clone shared across all locations, with each
        # location's persistent flag preserved independently (the
        # shared buffer might be persistent in one parent and
        # non-persistent in another).
        # Per unique buffer: (pinned_tensor, list of (parent, leaf, persistent))
        self._buffer_slots: list[
            tuple[torch.Tensor, list[tuple[nn.Module, str, bool]]]
        ] = []
        if include_buffers:
            buf_groups: dict[
                tuple[Any, ...],
                tuple[torch.Tensor, list[tuple[nn.Module, str, bool]]],
            ] = {}
            for s in iter_buffer_slots(model):
                if s.slot in self._skip_slots:
                    continue  # composer owns this buffer
                persistent = s.leaf not in s.parent._non_persistent_buffers_set
                if s.buffer.numel() == 0:
                    skey = ("__empty_buf__", id(s.buffer), s.name)
                else:
                    skey = storage_key(s.buffer)
                existing = buf_groups.get(skey)
                if existing is None:
                    pinned = s.buffer.detach().clone(memory_format=torch.contiguous_format).pin_memory()
                    buf_groups[skey] = (pinned, [(s.parent, s.leaf, persistent)])
                else:
                    seen_locs = {(id(p), leaf) for p, leaf, _ in existing[1]}
                    if (id(s.parent), s.leaf) not in seen_locs:
                        existing[1].append((s.parent, s.leaf, persistent))
            self._buffer_slots = list(buf_groups.values())

        # Phase 2: apply ALL slot mutations together, AFTER all
        # pinning succeeded. This makes __init__ strong-exception-safe
        # — a failure during buffer pinning above leaves the user's
        # model untouched (the local pinned tensors get GC'd as we
        # propagate). ModelCache cannot close a factory that never
        # returned, so constructor mutation must be all-or-nothing.
        for buf, locs in self._slots:
            for parent, leaf in locs:
                parent._parameters[leaf] = buf.cpu_param
        for pinned, locs in self._buffer_slots:
            for parent, leaf, persistent in locs:
                _set_buffer(parent, leaf, pinned, persistent)

        # Reject only if there is nothing at all to manage — neither
        # frozen params nor (when include_buffers=True) registered
        # buffers. Buffer-only modules (e.g., a pure RoPE/positional
        # table sibling) are valid: PinnedWeights still gives them
        # pinned-CPU storage and the activate/deactivate round-trip,
        # which is exactly what make_block_offloader non-block composition
        # needs.
        if not self._slots and not self._buffer_slots:
            raise ValueError(
                "PinnedWeights requires at least one frozen parameter or, "
                "when include_buffers=True, at least one registered buffer "
                "to cache. The wrapped model has neither — for training "
                "flows use block_offload.make_block_offloader instead, or "
                "leave the model unwrapped."
            )

    # ------------------------------------------------------------------
    # ModelStrategy protocol
    # ------------------------------------------------------------------

    @property
    def cache_bytes(self) -> int:
        """Total pinned host bytes held. Tied weights counted once."""
        total = 0
        for buf, _ in self._slots:
            total += buf.cache_bytes
        for pinned, _ in self._buffer_slots:
            total += pinned.numel() * pinned.element_size()
        return total

    def activate(self) -> nn.Module:
        """Bulk-DMA pinned weights to GPU and return the model.

        Per-tensor ``.to()`` (non-blocking), then a single
        ``cuda.synchronize`` to make the writes visible. Tied parameter
        slots all receive the same GPU Parameter.

        **Lifecycle is caller's responsibility.** Calling activate()
        twice without an intervening deactivate() double-allocates
        GPU storage. Don't.

        **Failure semantics (poison-on-failure):** if activation fails
        midway, the strategy is left in an undefined state — some
        slots may be GPU, some pinned-CPU. The caller's only valid
        next action is :meth:`deactivate` (which forces all slots
        back to pinned-CPU) followed by dropping the strategy
        reference.
        """
        assert self._model is not None
        self._move_to_gpu()
        return self._model

    def deactivate(self) -> None:
        """Repoint slots back at pinned-CPU Parameters. Idempotent —
        safe to call before activate or multiple times. After
        deactivate, drop the strategy reference to release pinned
        memory (and the model reference too if you don't need it
        anymore)."""
        self._move_to_pinned()

    def __enter__(self) -> nn.Module:
        return self.activate()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.deactivate()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _move_to_gpu(self) -> None:
        # One GPU Parameter per unique buffer. Tied slots all receive
        # the same Parameter object so the tying invariant survives on
        # device. GPU state is held until deactivate so copy_back can
        # round-trip in-place updates back into pinned host storage
        # (no-op for buffers with copy_back_enabled=False).
        self._gpu_states: dict[int, Any] = {}
        for buf, locs in self._slots:
            gpu_state = buf.allocate_gpu_storage(self._device)
            buf.copy_to_gpu(gpu_state, non_blocking=True)
            gpu_param = buf.make_gpu_param(gpu_state)
            self._gpu_states[id(buf)] = gpu_state
            for parent, leaf in locs:
                parent._parameters[leaf] = gpu_param
        if self._include_buffers:
            for pinned, locs in self._buffer_slots:
                gpu = pinned.to(self._device, non_blocking=True)
                for parent, leaf, persistent in locs:
                    _set_buffer(parent, leaf, gpu, persistent)
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)

    def _move_to_pinned(self) -> None:
        # Round-trip in-place GPU updates into pinned host storage for
        # any buffer that opted in (typically trainable slots). The
        # copy_back call is a no-op when copy_back_enabled is False.
        gpu_states = getattr(self, "_gpu_states", {})
        for buf, locs in self._slots:
            gpu_state = gpu_states.get(id(buf))
            if gpu_state is not None:
                buf.copy_back(gpu_state)
            for parent, leaf in locs:
                parent._parameters[leaf] = buf.cpu_param
        if hasattr(self, "_gpu_states"):
            self._gpu_states.clear()
        if self._include_buffers:
            for pinned, locs in self._buffer_slots:
                for parent, leaf, persistent in locs:
                    _set_buffer(parent, leaf, pinned, persistent)
