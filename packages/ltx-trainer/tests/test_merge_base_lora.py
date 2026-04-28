"""Equivalence test for ``LtxvTrainer._merge_base_lora`` against the LTX
inference fuse path (``ltx_core.loader.fuse_loras.apply_loras``).

Both paths must produce numerically identical merged weights for the same
base + LoRA + strength so that a checkpoint trained on top of the merged
base can be reproduced at inference by re-applying the same LoRA at the
same strength.
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import nn

from ltx_core.loader.fuse_loras import apply_loras
from ltx_core.loader.primitives import LoraStateDictWithStrength, StateDict
from ltx_trainer.config import BaseLoraConfig
from ltx_trainer.trainer import LtxvTrainer


class _Tiny(nn.Module):
    """Three-linear stack with explicit attribute names so parameters are
    addressable as ``layer{i}.weight`` (mirroring the real transformer's
    ``named_parameters`` shape)."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.layer0 = nn.Linear(in_dim, out_dim, bias=False)
        self.layer1 = nn.Linear(in_dim, out_dim, bias=False)
        self.layer2 = nn.Linear(in_dim, out_dim, bias=False)


def test_merge_base_lora_matches_inference_fuse(tmp_path: Path) -> None:
    torch.manual_seed(0)
    in_dim, out_dim, rank, strength = 16, 24, 8, 0.6
    n_layers = 3

    # Two identical models — one for the trainer path, one as the
    # reference whose state dict we feed to apply_loras.
    model_trainer = _Tiny(in_dim, out_dim).to(torch.bfloat16)
    model_reference = _Tiny(in_dim, out_dim).to(torch.bfloat16)
    model_reference.load_state_dict(model_trainer.state_dict())

    # Synthetic LoRA factors keyed with the ``diffusion_model.`` prefix
    # the real distilled file uses.
    lora_tensors: dict[str, torch.Tensor] = {}
    for i in range(n_layers):
        a = torch.randn(rank, in_dim, dtype=torch.bfloat16) * 0.1
        b = torch.randn(out_dim, rank, dtype=torch.bfloat16) * 0.1
        lora_tensors[f"diffusion_model.layer{i}.lora_A.weight"] = a
        lora_tensors[f"diffusion_model.layer{i}.lora_B.weight"] = b

    lora_path = tmp_path / "lora.safetensors"
    save_file(lora_tensors, str(lora_path))

    # Trainer path: in-place merge into model_trainer.
    cfg = BaseLoraConfig(path=lora_path, strength=strength)
    LtxvTrainer._merge_base_lora(model_trainer, cfg)

    # Inference path: apply_loras returns a fused state dict. Build the
    # model state dict with the matching ``diffusion_model.`` prefix
    # since apply_loras keys the LoRA lookup by the model's own keys.
    ref_sd = {
        f"diffusion_model.layer{i}.weight": model_reference.get_parameter(f"layer{i}.weight").detach().clone()
        for i in range(n_layers)
    }
    model_state = StateDict(sd=ref_sd, device=torch.device("cpu"), size=0, dtype={torch.bfloat16})
    lora_state = StateDict(sd=lora_tensors, device=torch.device("cpu"), size=0, dtype={torch.bfloat16})
    fused_state = apply_loras(model_state, [LoraStateDictWithStrength(lora_state, strength)])

    # Each merged trainer weight must match the apply_loras result.
    for i in range(n_layers):
        got = model_trainer.get_parameter(f"layer{i}.weight")
        expected = fused_state.sd[f"diffusion_model.layer{i}.weight"]
        torch.testing.assert_close(got, expected, rtol=1e-2, atol=1e-3)
