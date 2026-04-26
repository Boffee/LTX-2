"""Whole-model pinned-CPU weight cache for fast bulk DMA to GPU.

Holds a model's frozen weights in pinned CPU memory so subsequent GPU
loads are bulk DMA (~200 ms for a 12 GB Gemma at PCIe Gen5 x16) instead
of re-reading the safetensors from disk (~3-5 s per call).

Use case: a model that fits on GPU when active but should be evicted
between calls — text encoder during diffusion, VAE between encode and
decode phases, etc. Different from :class:`BlockOffloader`: no per-block
streaming, no forward hooks, no LRU. The whole model goes to GPU on
:meth:`PinnedWeights.activate` and the GPU storage is released on
:meth:`PinnedWeights.deactivate` by repointing each module's parameter
slot back at a Parameter that wraps pinned CPU storage.

Implements :class:`~ltx_core.memory.strategy.ModelStrategy` so it plugs
into a model cache directly.

Cross-cutting compatibility caveats (``torch.compile`` incompatibility,
DDP/FSDP wrap-before requirement, single-thread contract) live in the
:mod:`~ltx_core.memory` package docstring.

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
- :meth:`activate` is not re-entrant: nested calls raise ``RuntimeError``.
- :meth:`close` is destructive: it moves the wrapped model to the
  ``meta`` device to release storage references. The model object is
  unusable after ``close()``; callers must rebuild to use it again.
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

from ltx_core.memory.pinned_buffer import PinnedParamBuffer, storage_key

logger = logging.getLogger(__name__)


def _set_buffer(module: nn.Module, name: str, value: torch.Tensor, persistent: bool) -> None:
    """Replace a registered buffer in-place by its leaf name on
    ``module``, preserving the original ``persistent`` flag so
    ``state_dict()`` behavior survives the swap."""
    module.register_buffer(name, value, persistent=persistent)


class PinnedWeights:
    """Whole-model pinned-CPU weight cache with bulk GPU transfer.

    Implements :class:`~ltx_core.memory.strategy.ModelStrategy`.

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
    embeddings managed via :class:`BlockOffloader`'s non-block
    composition. Construction raises only if there is *nothing* to
    manage — neither frozen params nor (with ``include_buffers=True``)
    registered buffers.

    Parameters
    ----------
    model:
        The model to cache. Should be on CPU when passed in (we won't
        move it for you — that lets the caller control build-time
        device).
    target_device:
        GPU device to bulk-transfer to in :meth:`activate`.
    include_buffers:
        Also cache registered buffers (LayerNorm running stats, position
        embeddings stored as buffers, etc.). Default True. Set False
        for models with very large mutable buffers you'd rather rebuild
        on each call.
    skip_param_ids:
        Optional set of ``id(param)`` values to skip during the
        parameter walk. Used by composers like :class:`BlockOffloader`
        that want to hand the *outer* model to PinnedWeights but
        manage some subset of params themselves (e.g., the streamed
        block list). Skipped slots are not pinned and are not touched
        by :meth:`close`.
    skip_buffer_ids:
        Same idea, for registered buffers.
    """

    def __init__(
        self,
        model: nn.Module,
        target_device: torch.device,
        include_buffers: bool = True,
        *,
        skip_param_ids: set[int] | None = None,
        skip_buffer_ids: set[int] | None = None,
    ) -> None:
        self._model: nn.Module | None = model
        self._device = target_device
        self._include_buffers = include_buffers
        self._active = False  # guards re-entry of activate() and close-while-active
        self._closed = False
        self._skip_param_ids: set[int] = skip_param_ids or set()
        self._skip_buffer_ids: set[int] = skip_buffer_ids or set()

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
        modules_map = dict(model.named_modules(remove_duplicate=False))
        # storage_key -> list of (name, param, parent_module, leaf)
        groups: dict[tuple[Any, ...], list[tuple[str, nn.Parameter, nn.Module, str]]] = {}
        for name, p in model.named_parameters(remove_duplicate=False):
            if id(p) in self._skip_param_ids:
                continue  # composer (e.g. BlockOffloader) owns this slot
            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                parent_path, leaf = parts
                parent = modules_map[parent_path]
            else:
                parent, leaf = model, name
            if p.numel() == 0:
                # Zero-sized tensors all share data_ptr()==0; key by id(p)
                # to keep them in independent groups rather than spuriously
                # collapsing them.
                skey = ("__empty__", id(p), name)
            else:
                skey = storage_key(p.data)
            groups.setdefault(skey, []).append((name, p, parent, leaf))

        # Per unique buffer: (PinnedParamBuffer, list of (parent, leaf)).
        self._slots: list[tuple[PinnedParamBuffer, list[tuple[nn.Module, str]]]] = []
        for members in groups.values():
            grad_states = {p.requires_grad for _, p, _, _ in members}
            if len(grad_states) > 1:
                names = [n for n, _, _, _ in members]
                raise ValueError(
                    f"Tied storage spans both trainable and frozen parameters: "
                    f"{names}. PinnedWeights cannot pin a tied group with mixed "
                    "requires_grad without breaking the tying invariant. Untie "
                    "the parameters or freeze/unfreeze them consistently."
                )
            if True in grad_states:
                continue  # all trainable — PinnedWeights does not manage these
            first_name, first_p = members[0][0], members[0][1]
            buf = PinnedParamBuffer(first_name, first_p)
            seen_locs: set[tuple[int, str]] = set()
            locs: list[tuple[nn.Module, str]] = []
            for _, _, parent, leaf in members:
                key = (id(parent), leaf)
                if key in seen_locs:
                    continue
                seen_locs.add(key)
                locs.append((parent, leaf))
            self._slots.append((buf, locs))

        # Initial repoint: every frozen slot now references the pinned
        # cpu_param. Tied slots all reference the same Parameter object,
        # which is stronger than the pre-PinnedWeights tying invariant
        # (those may have been distinct Parameter objects sharing storage).
        for buf, locs in self._slots:
            for parent, leaf in locs:
                parent._parameters[leaf] = buf.cpu_param

        # Cache buffers if requested. Same alias-aware grouping as
        # parameters: shared buffer instances visible at multiple
        # (parent, leaf) paths get one pinned clone shared across all
        # locations, with each location's persistent flag preserved
        # independently (the shared buffer might be persistent in one
        # parent and non-persistent in another).
        # Per unique buffer: (pinned_tensor, list of (parent, leaf, persistent))
        self._buffer_slots: list[
            tuple[torch.Tensor, list[tuple[nn.Module, str, bool]]]
        ] = []
        if include_buffers:
            buf_groups: dict[
                tuple[Any, ...],
                tuple[torch.Tensor, list[tuple[nn.Module, str, bool]]],
            ] = {}
            for full_name, b in list(model.named_buffers(remove_duplicate=False)):
                if id(b) in self._skip_buffer_ids:
                    continue  # composer owns this buffer
                parent = self._resolve_parent(model, full_name)
                leaf = full_name.rsplit(".", 1)[-1]
                persistent = leaf not in parent._non_persistent_buffers_set
                if b.numel() == 0:
                    skey = ("__empty_buf__", id(b), full_name)
                else:
                    skey = storage_key(b)
                existing = buf_groups.get(skey)
                if existing is None:
                    pinned = b.detach().clone(memory_format=torch.contiguous_format).pin_memory()
                    buf_groups[skey] = (pinned, [(parent, leaf, persistent)])
                else:
                    pinned = existing[0]
                    seen_locs = {(id(p), l) for p, l, _ in existing[1]}
                    if (id(parent), leaf) not in seen_locs:
                        existing[1].append((parent, leaf, persistent))
                _set_buffer(parent, leaf, pinned, persistent)
            self._buffer_slots = list(buf_groups.values())

        # Reject only if there is nothing at all to manage — neither
        # frozen params nor (when include_buffers=True) registered
        # buffers. Buffer-only modules (e.g., a pure RoPE/positional
        # table sibling) are valid: PinnedWeights still gives them
        # pinned-CPU storage and the activate/deactivate round-trip,
        # which is exactly what BlockOffloader's non-block composition
        # needs.
        if not self._slots and not self._buffer_slots:
            raise ValueError(
                "PinnedWeights requires at least one frozen parameter or, "
                "when include_buffers=True, at least one registered buffer "
                "to cache. The wrapped model has neither — for training "
                "flows use ltx_core.memory.BlockOffloader instead, or "
                "leave the model unwrapped."
            )

    @staticmethod
    def _resolve_parent(model: nn.Module, dotted_name: str) -> nn.Module:
        parent: Any = model
        parts = dotted_name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        return parent

    # ------------------------------------------------------------------
    # ModelStrategy protocol
    # ------------------------------------------------------------------

    @property
    def cache_bytes(self) -> int:
        """Total pinned host bytes held. Tied weights counted once."""
        total = 0
        for buf, _ in self._slots:
            total += buf.pinned_data.numel() * buf.pinned_data.element_size()
            if buf.pinned_scale is not None:
                total += buf.pinned_scale.numel() * buf.pinned_scale.element_size()
        for pinned, _ in self._buffer_slots:
            total += pinned.numel() * pinned.element_size()
        return total

    @property
    def closed(self) -> bool:
        return self._closed

    def activate(self) -> nn.Module:
        """Bulk-DMA pinned weights to GPU and return the model.

        Per-tensor ``.to()`` (non-blocking), then a single
        ``cuda.synchronize`` to make the writes visible. Tied parameter
        slots all receive the same GPU Parameter.

        Not re-entrant; nested calls raise ``RuntimeError``. Call
        :meth:`deactivate` before activating again.
        """
        if self._closed:
            raise RuntimeError("PinnedWeights is closed and cannot be activated.")
        if self._active:
            raise RuntimeError(
                "PinnedWeights.activate() is not re-entrant. Call deactivate() "
                "before activating again."
            )
        assert self._model is not None
        self._active = True
        try:
            self._move_to_gpu()
        except BaseException:
            # Best-effort rollback so slots end up referencing pinned CPU
            # again. If rollback itself fails, log it but re-raise the
            # original exception so the caller sees the actual cause.
            try:
                self._move_to_pinned()
            except BaseException as rollback_exc:
                logger.error(
                    "PinnedWeights.activate() rollback failed; original error "
                    "will still propagate. rollback=%r",
                    rollback_exc,
                    exc_info=True,
                )
            self._active = False
            raise
        return self._model

    def deactivate(self) -> None:
        """Repoint slots back at pinned-CPU Parameters. GPU storage is
        released by refcount as soon as no other references remain."""
        if not self._active:
            return
        try:
            self._move_to_pinned()
        finally:
            self._active = False

    def close(self) -> None:
        """Release pinned CPU storage and invalidate managed slots.

        Walks the slots/buffer-slots this strategy manages and replaces
        each with a meta-device Parameter/buffer of matching shape and
        dtype. This breaks the model→pinned-tensor reference chain so
        the pinned pages can return to the host allocator, and matches
        the semantics of ``model.to("meta")`` for the slots we own.

        Surgical (per-slot) rather than wholesale ``model.to("meta")``
        because :class:`PinnedWeights` may be used in composed setups
        (e.g., :class:`BlockOffloader` hands the outer model with a
        skip filter — block params are owned by ``_BlockPinnedStore``
        and would be trampled if we touched the whole model).

        Trainable parameters and any slots passed via ``skip_param_ids``
        / ``skip_buffer_ids`` are NOT touched — they're not ours to
        meta-ify. For standalone use (no skip filter), trainable params
        survive close on whatever device they were on; callers who
        want full meta-ification should drop their model reference too
        and let GC finish the job.

        Idempotent. Raises ``RuntimeError`` if called while active;
        deactivate first. If a per-slot replacement raises (rare —
        usually a quanto subclass quirk), state is *not* cleared and
        the strategy is *not* marked closed so the caller can retry.
        """
        if self._closed:
            return
        if self._active:
            raise RuntimeError(
                "PinnedWeights.close() called while activate() is in effect - "
                "deactivate first or the model would be left holding stale "
                "GPU tensors that cannot be restored."
            )
        if self._model is not None:
            # Surgical meta-ification: walk only the slots we manage.
            # Let exceptions propagate without clearing state — preserves
            # retry / hand-clean possibility.
            for buf, locs in self._slots:
                meta_param = buf.make_meta_param()
                for parent, leaf in locs:
                    parent._parameters[leaf] = meta_param
            for _pinned, locs in self._buffer_slots:
                # We don't have a "pinned buffer's meta equivalent" helper —
                # build one inline. Buffers are plain tensors (no quanto
                # wrapper), so torch.empty_like + register_buffer suffices.
                if not locs:
                    continue
                # Use the first location's pinned tensor as the shape/dtype
                # template — all locations share the same pinned tensor by
                # construction.
                template = _pinned
                meta_buf = torch.empty_like(template, device="meta")
                for parent, leaf, persistent in locs:
                    _set_buffer(parent, leaf, meta_buf, persistent)
        self._model = None
        self._slots.clear()
        self._buffer_slots.clear()
        self._closed = True

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
        # device.
        for buf, locs in self._slots:
            gpu_param = buf.load_to_gpu(self._device, non_blocking=True)
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
        for buf, locs in self._slots:
            for parent, leaf in locs:
                parent._parameters[leaf] = buf.cpu_param
        if self._include_buffers:
            for pinned, locs in self._buffer_slots:
                for parent, leaf, persistent in locs:
                    _set_buffer(parent, leaf, pinned, persistent)
