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
    pinned LoRA factors    --DMA-->  GPU temporaries
    GPU temporaries        --addmm_--> slot.weights += B_cat @ A_cat

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
  deactivate -> set_loras -> activate.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Sequence
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
    "LoRA",
    "MergedLoRAStrategy",
]


# ---------------------------------------------------------------------------
# Public LoRA type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoRA:
    """A LoRA adapter from a flat safetensors state dict.

    ``state_dict`` uses the same keys returned by
    ``safetensors.torch.load_file()`` — e.g.
    ``"diffusion_model.transformer_blocks.0.attn.lora_A.weight"``.
    The strategy handles prefix stripping and A/B pairing internally.

    ``strength`` is the only user-facing multiplier (no alpha/rank
    scaling). 1.0 reproduces the LoRA's full effect.
    """

    state_dict: dict[str, torch.Tensor]
    strength: float = 1.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _default_key_transform(key: str) -> str:
    """Strip the common ``diffusion_model.`` prefix from ComfyUI LoRA keys."""
    prefix = "diffusion_model."
    return key[len(prefix) :] if key.startswith(prefix) else key


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


# Per-block pre-concatenated merge list: (qual, B_cat_pinned, A_cat_pinned)
# tuples ready for per-prefetch DMA + addmm_.
_MergePlan = dict[int, list[tuple[str, torch.Tensor, torch.Tensor]]]


# ---------------------------------------------------------------------------
# MergedLoRAStrategy
# ---------------------------------------------------------------------------


class MergedLoRAStrategy:
    """Merges stacked LoRAs into block weights at prefetch time.

    See module docstring for architecture. bf16/fp16 base only;
    raises at construction otherwise.

    LoRAs are set post-construction via :meth:`set_loras`, which
    accepts flat safetensors state dicts. The strategy handles A/B
    pairing, key matching, block decomposition, and CPU pinning
    internally.
    """

    def __init__(
        self,
        model: nn.Module,
        target_device: torch.device,
        *,
        layers_attr: str,
        blocks_to_swap: int,
        prefetch_count: int = 2,
        key_transform: Callable[[str], str] | None = _default_key_transform,
    ) -> None:
        self._validate_base_dtype(model)

        self._model = model
        self._device = target_device
        self._key_transform = key_transform

        blocks = list(_resolve_attr(model, layers_attr))
        if not blocks:
            raise ValueError(
                f"layers_attr={layers_attr!r} resolved to an empty ModuleList"
            )

        self._reverse_index = self._build_reverse_index(
            layers_attr, blocks,
        )

        self._merge_plan: _MergePlan = {}
        self._lora_factor_bytes: int = 0
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

    # ------------------------------------------------------------------ API

    def set_loras(self, loras: Sequence[LoRA]) -> None:
        """Replace all LoRAs. Must be called while deactivated.

        Processes flat state dicts: applies ``key_transform``, pairs
        A/B factors, matches to model parameters, concatenates per
        (block, target), and pins on CPU. The resulting merge plan is
        used by the next :meth:`activate` call.

        Pass an empty sequence to clear all LoRAs (base-only forward).
        """
        if self._teardown is not None:
            raise RuntimeError(
                "MergedLoRAStrategy.set_loras() requires the strategy "
                "to be inactive. Call deactivate() first."
            )
        self._merge_plan.clear()
        self._lora_factor_bytes = 0

        if not loras:
            return

        base_dtype = next(self._model.parameters()).dtype
        raw = self._pair_and_assign(loras, base_dtype)
        self._merge_plan, self._lora_factor_bytes = self._concat_and_pin(
            raw, base_dtype,
        )

    # ----------------------------------------------- ModelStrategy interface

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def cache_bytes(self) -> int:
        total = self._streamer.cache_bytes
        if self._non_block is not None:
            total += self._non_block.cache_bytes
        total += self._lora_factor_bytes
        return total

    def activate(self) -> None:
        with contextlib.ExitStack() as stack:
            if self._non_block is not None:
                stack.callback(self._non_block.deactivate)
                self._non_block.activate()
            stack.callback(self._streamer.deactivate)
            self._streamer.activate()
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

    # ----------------------------------------------------------- Internals

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
    def _build_reverse_index(
        layers_attr: str,
        blocks: list[nn.Module],
    ) -> dict[str, tuple[int, str, tuple[int, ...]]]:
        """Map ``full_qualname -> (block_idx, in_block_qual, shape)``
        for frozen parameters inside the block list."""
        index: dict[str, tuple[int, str, tuple[int, ...]]] = {}
        for block_idx, block in enumerate(blocks):
            for qual, p in block.named_parameters():
                if not p.requires_grad:
                    full = f"{layers_attr}.{block_idx}.{qual}"
                    index[full] = (block_idx, qual, tuple(p.shape))
        return index

    def _pair_and_assign(
        self,
        loras: Sequence[LoRA],
        base_dtype: torch.dtype,
    ) -> dict[int, dict[str, list[tuple[torch.Tensor, torch.Tensor, float]]]]:
        """Pair lora_A/lora_B keys, match to model params via reverse index.

        Returns ``raw[block_idx][in_block_qual] -> [(A, B, strength), ...]``
        ready for concatenation.
        """
        transform = self._key_transform
        raw: dict[int, dict[str, list[tuple[torch.Tensor, torch.Tensor, float]]]] = {}

        for lora in loras:
            a_tensors: dict[str, torch.Tensor] = {}
            b_tensors: dict[str, torch.Tensor] = {}
            for key, tensor in lora.state_dict.items():
                if key.endswith(".lora_A.weight"):
                    base_key = key[: -len(".lora_A.weight")]
                    a_tensors[base_key] = tensor
                elif key.endswith(".lora_B.weight"):
                    base_key = key[: -len(".lora_B.weight")]
                    b_tensors[base_key] = tensor

            a_only = set(a_tensors) - set(b_tensors)
            b_only = set(b_tensors) - set(a_tensors)
            if a_only or b_only:
                raise ValueError(
                    f"Unpaired LoRA factors: A-only={sorted(a_only)}, "
                    f"B-only={sorted(b_only)}. Each target needs both "
                    f".lora_A.weight and .lora_B.weight."
                )

            for base_key in a_tensors:
                a = a_tensors[base_key]
                b = b_tensors[base_key]
                target_key = f"{base_key}.weight"
                if transform is not None:
                    target_key = transform(target_key)

                entry = self._reverse_index.get(target_key)
                if entry is None:
                    continue  # non-block target, skip silently

                block_idx, in_block_qual, expected_shape = entry

                if not a.is_floating_point() or not b.is_floating_point():
                    raise ValueError(
                        f"LoRA factors for {target_key!r}: must be "
                        f"floating-point; got A.dtype={a.dtype}, "
                        f"B.dtype={b.dtype}."
                    )
                if (
                    a.dim() != 2
                    or b.dim() != 2
                    or a.shape[0] != b.shape[1]
                    or expected_shape != (b.shape[0], a.shape[1])
                ):
                    raise ValueError(
                        f"LoRA factor shape mismatch for {target_key!r}: "
                        f"A.shape={tuple(a.shape)}, B.shape={tuple(b.shape)}, "
                        f"target shape {expected_shape}. Expected "
                        f"A=(rank, in_dim), B=(out_dim, rank), "
                        f"B@A.shape == target."
                    )

                block_raw = raw.setdefault(block_idx, {})
                block_raw.setdefault(in_block_qual, []).append(
                    (a, b, lora.strength)
                )

        return raw

    def _concat_and_pin(
        self,
        raw: dict[int, dict[str, list[tuple[torch.Tensor, torch.Tensor, float]]]],
        base_dtype: torch.dtype,
    ) -> tuple[_MergePlan, int]:
        """Concatenate factors per (block, target) and pin on CPU."""
        plan: _MergePlan = {}
        total_bytes = 0

        for block_idx, qual_factors in raw.items():
            bucket: list[tuple[str, torch.Tensor, torch.Tensor]] = []
            for qual, factors in qual_factors.items():
                pair = _concat_lora_factors(factors, base_dtype, torch.device("cpu"))
                if pair is not None:
                    a_cat, b_cat = pair
                    a_pinned = a_cat.contiguous().pin_memory()
                    b_pinned = b_cat.contiguous().pin_memory()
                    total_bytes += a_pinned.numel() * a_pinned.element_size()
                    total_bytes += b_pinned.numel() * b_pinned.element_size()
                    bucket.append((qual, b_pinned, a_pinned))
            plan[block_idx] = bucket

        return plan, total_bytes

    def _apply_active_loras(self, slot: Any, block_idx: int) -> None:  # noqa: ANN401
        """post_load callback: DMA this block's factors from pinned CPU
        to GPU and merge via in-place addmm_ on the prefetch stream."""
        for qual, b_pinned, a_pinned in self._merge_plan.get(block_idx, ()):
            b_gpu = b_pinned.to(device=self._device, non_blocking=True)
            a_gpu = a_pinned.to(device=self._device, non_blocking=True)
            slot.get_param(qual).data.addmm_(b_gpu, a_gpu)
