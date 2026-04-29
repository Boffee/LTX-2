"""LoRA types and per-weight merge transform.

:class:`LoRA` holds a flat safetensors state dict with a strength
multiplier.  :class:`LoRATransform` holds pre-concatenated (A, B)
factor matrices in pinned CPU memory and applies the merge via
in-place ``addmm_`` after DMA.

:class:`~ltx_core.memory.BlockOffloader` is the consumer-facing API:
its ``set_loras`` method pairs factors from :class:`LoRA` state dicts,
creates one :class:`LoRATransform` per matched weight, and attaches
it to the corresponding :class:`~ltx_core.memory.PinnedParamBuffer`.
The transform fires automatically when the buffer copies to GPU.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

__all__ = [
    "FactorList",
    "KeyTransformT",
    "LoRA",
    "LoRATransform",
    "concat_lora_factors",
    "default_key_transform",
    "pair_and_validate",
]


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoRA:
    """A LoRA adapter from a flat safetensors state dict.

    ``state_dict`` uses the same keys returned by
    ``safetensors.torch.load_file()`` — e.g.
    ``"diffusion_model.transformer_blocks.0.attn.lora_A.weight"``.
    The consumer handles prefix stripping and A/B pairing internally.

    ``strength`` is the only user-facing multiplier (no alpha/rank
    scaling). 1.0 reproduces the LoRA's full effect.
    """

    state_dict: dict[str, torch.Tensor]
    strength: float = 1.0


class LoRATransform:
    """Per-weight LoRA factors applied after DMA to GPU.

    Stores concatenated (A, B) factor matrices in pinned CPU memory.
    Consumers call :meth:`apply` with the GPU tensor to merge in-place
    via ``addmm_``.  Multiple stacked LoRAs are pre-concatenated into
    a single (A_cat, B_cat) pair so the merge is always one ``addmm_``.
    """

    __slots__ = ("_a_cat", "_b_cat")

    def __init__(self, a_cat: torch.Tensor, b_cat: torch.Tensor) -> None:
        self._a_cat = a_cat.contiguous().pin_memory()
        self._b_cat = b_cat.contiguous().pin_memory()

    def apply(self, gpu_data: torch.Tensor) -> None:
        b = self._b_cat.to(device=gpu_data.device, non_blocking=True)
        a = self._a_cat.to(device=gpu_data.device, non_blocking=True)
        gpu_data.addmm_(b, a)

    @property
    def nbytes(self) -> int:
        return self._a_cat.nbytes + self._b_cat.nbytes


def default_key_transform(key: str) -> str:
    """Strip the common ``diffusion_model.`` prefix from ComfyUI LoRA keys."""
    prefix = "diffusion_model."
    return key[len(prefix) :] if key.startswith(prefix) else key


FactorList = list[tuple[torch.Tensor, torch.Tensor, float]]


def concat_lora_factors(
    factors: FactorList,
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


def pair_and_validate(
    loras: Sequence[LoRA],
    reverse_index: dict[str, tuple[int, ...]],
    key_transform: KeyTransformT,
) -> dict[str, FactorList]:
    """Pair lora_A/lora_B keys, validate shapes, return per-target factors.

    ``reverse_index`` maps model param qualified names to their expected
    shape.  ``key_transform`` is applied to LoRA state-dict base keys
    before lookup.

    Returns ``{target_qualname: [(A, B, strength), ...]}`` for matched
    targets.
    """
    raw: dict[str, FactorList] = {}

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

        for base_key, a in a_tensors.items():
            b = b_tensors[base_key]
            target_key = f"{base_key}.weight"
            if key_transform is not None:
                target_key = key_transform(target_key)

            expected_shape = reverse_index.get(target_key)
            if expected_shape is None:
                continue

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

            raw.setdefault(target_key, []).append(
                (a, b, lora.strength)
            )

    return raw


KeyTransformT = Callable[[str], str] | None
