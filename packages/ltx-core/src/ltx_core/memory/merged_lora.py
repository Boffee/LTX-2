"""Merged-LoRA strategy for stacked-LoRA inference.

For 3+ stacked LoRAs on a bf16/fp16 base, the PEFT-style routed
forward pays a per-LoRA cost on every step. This strategy takes
ComfyUI's path: merge LoRA deltas into the base weights at prefetch
time, run forward as a single matmul per layer regardless of K.

Built on the existing :class:`BlockStreamer` + :class:`PinnedWeights`
infrastructure. The novel piece is a ``post_load`` callback that
runs on the prefetch CUDA stream after each block's bytes are
DMA'd in:

    pinned bf16 base       --DMA-->  pool slot
    GPU LoRA factors       --addmm_--> slot.weights += B_cat @ A_cat

CUDA orders the merge after the DMA automatically (same stream).
The "free unmerge" property: when a block is evicted and re-loaded,
``slot.copy_from`` overwrites GPU bytes with pristine pinned base
bytes, so the next merge starts from clean base. No subtract-and-
restore, no drift.

Constraints
-----------
- Base must be bf16 or fp16. ``addmm_`` requires arithmetic-capable
  target dtype; fp8 and quanto are unsupported in v1.
- LoRA set is fixed during the active window. Switch combos via
  deactivate -> set_active -> activate.
- All LoRAs registered at construction (cache_bytes is final at
  admission per the package contract).
- Active list is ordered (Sequence, not set) for bf16-reproducible
  output.
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Any

import torch
from torch import nn

from .block_compose import _has_non_block_pinnable_content, _resolve_attr
from .block_streamer import BlockStreamer
from .pinned_weights import PinnedWeights
from .strategy import SlotOwnership

__all__ = [
    "LoRABundle",
    "LoRALayerFactors",
    "MergedLoRAStrategy",
]


# ---------------------------------------------------------------------------
# LoRA data types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class LoRALayerFactors:
    """LoRA factors for one target layer.

    ``A`` is shape ``(rank, in_dim)``; ``B`` is shape ``(out_dim, rank)``.
    The merged delta is ``scaling * (B @ A)`` -- a ``(out_dim, in_dim)``
    matrix added to the corresponding base weight. ``scaling`` follows
    the standard PEFT convention (``alpha / rank``).

    The strategy clones, dtype-casts, and pins these tensors at
    construction; the user-supplied tensors are not retained.
    """

    A: torch.Tensor
    B: torch.Tensor
    scaling: float


@dataclass(slots=True)
class LoRABundle:
    """All factors for one named adapter.

    ``blocks[block_idx][in_block_qualname]`` -> :class:`LoRALayerFactors`.
    Block indices align with positions in the ``layers_attr`` ModuleList.
    In-block qualnames are paths used by the :class:`BlockStreamer` slot
    (e.g., ``"attn.q_proj.weight"``, not the full
    ``"transformer_blocks.0.attn.q_proj.weight"``).
    """

    name: str
    blocks: dict[int, dict[str, LoRALayerFactors]]


def _concat_lora_factors(
    factors: list[tuple[torch.Tensor, torch.Tensor, float]],
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Concatenate N LoRA (A, B, strength) triples into a single factor pair.

    Returns ``(A_cat, B_cat)`` such that ``B_cat @ A_cat`` equals the sum of
    ``strength_i * B_i @ A_i`` for all inputs (within floating-point tolerance).
    Mixed ranks across LoRAs are handled naturally by the concatenation.
    """
    if not factors:
        return None
    as_list: list[torch.Tensor] = []
    bs_list: list[torch.Tensor] = []
    for a, b, strength in factors:
        as_list.append(a.to(device=device, dtype=dtype))
        bs_list.append(b.to(device=device, dtype=dtype) * strength)
    if len(as_list) == 1:
        return as_list[0], bs_list[0]
    return torch.cat(as_list, dim=0), torch.cat(bs_list, dim=1)


# Per-block pre-concatenated merge list: (qual, B_cat, A_cat) tuples
# ready for a single in-place addmm_ per target weight.
_MergePlan = dict[int, list[tuple[str, torch.Tensor, torch.Tensor]]]


# ---------------------------------------------------------------------------
# MergedLoRAStrategy
# ---------------------------------------------------------------------------


class MergedLoRAStrategy:
    """Top-level :class:`ModelStrategy` that merges stacked LoRAs into
    block weights at prefetch time.

    See module docstring for architecture. bf16/fp16 base only;
    raises at construction otherwise. All LoRAs must be passed at
    construction; ``set_active`` chooses the active subset for the
    next active window.
    """

    def __init__(
        self,
        model: nn.Module,
        target_device: torch.device,
        loras: Sequence[LoRABundle],
        *,
        layers_attr: str,
        blocks_to_swap: int,
        prefetch_count: int = 2,
    ) -> None:
        self._validate_base_dtype(model)
        base_dtype = next(model.parameters()).dtype

        self._model = model
        self._device = target_device

        blocks = list(_resolve_attr(model, layers_attr))
        if not blocks:
            raise ValueError(
                f"layers_attr={layers_attr!r} resolved to an empty ModuleList"
            )

        # Per-block frozen-target shape map for factor validation.
        target_shapes: list[dict[str, tuple[int, ...]]] = [
            {qual: tuple(p.shape)
             for qual, p in block.named_parameters()
             if not p.requires_grad}
            for block in blocks
        ]

        # Pin factors. Detects duplicate names while building.
        self._pinned: dict[str, dict[int, dict[str, LoRALayerFactors]]] = (
            self._pin_factors(loras, base_dtype, len(blocks), target_shapes)
        )

        self._active: tuple[str, ...] = ()
        # Per-block merge list, populated at activate, drained on
        # deactivate via ExitStack callback. Empty dict means "no
        # merges" (unactivated, or active=()).
        self._merge_plan: _MergePlan = {}
        self._teardown: contextlib.ExitStack | None = None

        self._streamer = BlockStreamer(
            blocks=blocks,
            target_device=target_device,
            blocks_to_swap=blocks_to_swap,
            prefetch_count=prefetch_count,
            name=f"BlockStreamer[{layers_attr}]",
            post_load=self._apply_active_loras,
        )
        skip: set[SlotOwnership] = set(self._streamer.slot_filter)
        self._non_block: PinnedWeights | None = None
        if _has_non_block_pinnable_content(model, skip):
            self._non_block = PinnedWeights(
                model, target_device, skip_slots=skip,
            )

    # ---------------------------------------------------------------- API

    @property
    def available(self) -> set[str]:
        """Names of all LoRAs registered at construction."""
        return set(self._pinned)

    def set_active(self, names: Sequence[str]) -> None:
        """Set the LoRAs to merge during the next active window.

        ``names`` is ordered -- the merge order is preserved because
        bf16 addition is non-associative. Pass ``()`` for base-only
        forward. Raises if called while active.
        """
        if self._teardown is not None:
            raise RuntimeError(
                "MergedLoRAStrategy.set_active() requires the strategy "
                "to be inactive. Call deactivate() first."
            )
        unknown = set(names) - self._pinned.keys()
        if unknown:
            raise ValueError(
                f"Unknown LoRA names: {sorted(unknown)}. "
                f"Registered: {sorted(self._pinned)}"
            )
        active = tuple(names)
        if len(set(active)) != len(active):
            raise ValueError(
                f"Duplicate LoRA names in active list: {active}. Each "
                f"LoRA can be active at most once per window."
            )
        self._active = active

    @property
    def active(self) -> tuple[str, ...]:
        """The currently configured active LoRA list."""
        return self._active

    # --------------------------------------------------- ModelStrategy

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def cache_bytes(self) -> int:
        total = self._streamer.cache_bytes
        if self._non_block is not None:
            total += self._non_block.cache_bytes
        for blocks in self._pinned.values():
            for layers in blocks.values():
                for f in layers.values():
                    total += f.A.numel() * f.A.element_size()
                    total += f.B.numel() * f.B.element_size()
        return total

    def activate(self) -> None:
        # ExitStack idiom from BlockStreamingStrategy: register cleanup
        # before each activation step, pop_all on success. Component
        # contracts make deactivate idempotent and safe-before-activate,
        # so register-then-activate is correct.
        self._merge_plan = self._build_merge_plan(self._active)
        with contextlib.ExitStack() as stack:
            stack.callback(self._merge_plan.clear)
            if self._non_block is not None:
                stack.callback(self._non_block.deactivate)
                self._non_block.activate()
            stack.callback(self._streamer.deactivate)
            self._streamer.activate()
            # Sync to close the cross-stream race with resident-block
            # initial merges (addmm_ kernels enqueued async on the
            # default stream during streamer.activate()).
            if self._device.type == "cuda":
                torch.cuda.synchronize(self._device)
            self._teardown = stack.pop_all()

    def deactivate(self) -> None:
        stack = self._teardown
        self._teardown = None
        if stack is not None:
            stack.close()

    def __enter__(self) -> nn.Module:
        self.activate()
        return self.model

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.deactivate()

    # ------------------------------------------------------- Internals

    @staticmethod
    def _validate_base_dtype(model: nn.Module) -> None:
        bad: list[tuple[str, torch.dtype]] = []
        for name, p in model.named_parameters():
            if p.dtype not in (torch.bfloat16, torch.float16):
                bad.append((name, p.dtype))
                if len(bad) > 3:
                    break
        if bad:
            raise ValueError(
                f"MergedLoRAStrategy requires bf16/fp16 base; found "
                f"{bad}. fp8 and quanto are unsupported in v1 (in-place "
                f"merge requires arithmetic-capable target dtype). For "
                f"fp8 base, use PEFT routed mode."
            )

    @staticmethod
    def _pin_factors(
        loras: Sequence[LoRABundle],
        base_dtype: torch.dtype,
        num_blocks: int,
        target_shapes: list[dict[str, tuple[int, ...]]],
    ) -> dict[str, dict[int, dict[str, LoRALayerFactors]]]:
        """Validate and pin user-supplied factors.

        Validation surfaces bad bundles at construction rather than
        as cryptic prefetch-future failures. Trainable factors are
        intentionally detached -- the strategy is inference-only.
        """
        pinned: dict[str, dict[int, dict[str, LoRALayerFactors]]] = {}
        for lora in loras:
            if lora.name in pinned:
                existing = sorted(pinned)
                raise ValueError(
                    f"LoRA names must be unique; {lora.name!r} appears "
                    f"more than once. Already registered: {existing}"
                )
            per_block: dict[int, dict[str, LoRALayerFactors]] = {}
            for block_idx, layer_factors in lora.blocks.items():
                if not 0 <= block_idx < num_blocks:
                    raise ValueError(
                        f"LoRA {lora.name!r}: block_idx={block_idx} out of "
                        f"range [0, {num_blocks})"
                    )
                block_targets = target_shapes[block_idx]
                per_layer: dict[str, LoRALayerFactors] = {}
                for qual, f in layer_factors.items():
                    expected = block_targets.get(qual)
                    if expected is None:
                        raise ValueError(
                            f"LoRA {lora.name!r} block {block_idx}: target "
                            f"{qual!r} is not a frozen param. Available "
                            f"frozen targets: {sorted(block_targets)}"
                        )
                    if not f.A.is_floating_point() or not f.B.is_floating_point():
                        raise ValueError(
                            f"LoRA {lora.name!r} block {block_idx} layer "
                            f"{qual!r}: factors must be floating-point; got "
                            f"A.dtype={f.A.dtype}, B.dtype={f.B.dtype}. "
                            f"Casting integer factors to bf16 silently "
                            f"truncates values."
                        )
                    # Combined shape guard: 2D + rank match + target
                    # compatibility, one diagnostic instead of three.
                    if (
                        f.A.dim() != 2 or f.B.dim() != 2
                        or f.A.shape[0] != f.B.shape[1]
                        or expected != (f.B.shape[0], f.A.shape[1])
                    ):
                        raise ValueError(
                            f"LoRA {lora.name!r} block {block_idx} layer "
                            f"{qual!r}: factor shape mismatch -- "
                            f"A.shape={tuple(f.A.shape)}, "
                            f"B.shape={tuple(f.B.shape)}, target shape "
                            f"{expected}. Expected A=(rank, in_dim), "
                            f"B=(out_dim, rank), B@A.shape == target."
                        )
                    # .cpu() handles factors stored on GPU (e.g., loaded
                    # from a GPU-resident model). Without it, .pin_memory()
                    # would raise.
                    per_layer[qual] = LoRALayerFactors(
                        A=f.A.detach().cpu().to(base_dtype).clone().pin_memory(),
                        B=f.B.detach().cpu().to(base_dtype).clone().pin_memory(),
                        scaling=float(f.scaling),
                    )
                per_block[block_idx] = per_layer
            pinned[lora.name] = per_block
        return pinned

    def _build_merge_plan(self, active: tuple[str, ...]) -> _MergePlan:
        """Pre-concatenate active LoRA factors on GPU per (block, target).

        For N active LoRAs targeting the same weight, this produces a
        single ``(A_cat, B_cat)`` pair so the prefetch callback does one
        ``addmm_`` per target instead of N.
        """
        # Collect per-(block, qual) factor triples from all active LoRAs.
        raw: dict[int, dict[str, list[tuple[torch.Tensor, torch.Tensor, float]]]] = {}
        for name in active:
            for block_idx, layer_factors in self._pinned[name].items():
                block_raw = raw.setdefault(block_idx, {})
                for qual, f in layer_factors.items():
                    block_raw.setdefault(qual, []).append((f.A, f.B, f.scaling))

        base_dtype = next(self._model.parameters()).dtype
        plan: _MergePlan = {}
        for block_idx, qual_factors in raw.items():
            bucket: list[tuple[str, torch.Tensor, torch.Tensor]] = []
            for qual, factors in qual_factors.items():
                pair = _concat_lora_factors(factors, base_dtype, self._device)
                if pair is not None:
                    a_cat, b_cat = pair
                    bucket.append((qual, b_cat, a_cat))
            plan[block_idx] = bucket

        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        return plan

    def _apply_active_loras(self, slot: Any, block_idx: int) -> None:  # noqa: ANN401
        """post_load callback: merge active LoRAs for this block via
        in-place addmm_ on the prefetch stream."""
        for qual, b_cat, a_cat in self._merge_plan.get(block_idx, ()):
            slot.get_param(qual).data.addmm_(b_cat, a_cat)
