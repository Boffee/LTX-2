"""Equivalence test: the ``LoRA`` factor pairing and key transform used by the
trainer (via ``BlockOffloader.set_loras``) must produce the same merged
weights as the inference fuse path (``ltx_core.loader.fuse_loras.apply_loras``).
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import nn

from ltx_core.loader.fuse_loras import apply_loras
from ltx_core.loader.primitives import LoraStateDictWithStrength, StateDict
from ltx_core.memory import LoRA


class _Tiny(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.layer0 = nn.Linear(in_dim, out_dim, bias=False)
        self.layer1 = nn.Linear(in_dim, out_dim, bias=False)
        self.layer2 = nn.Linear(in_dim, out_dim, bias=False)


def test_lora_factors_match_inference_fuse(tmp_path: Path) -> None:
    torch.manual_seed(0)
    in_dim, out_dim, rank, strength = 16, 24, 8, 0.6
    n_layers = 3

    model = _Tiny(in_dim, out_dim).to(torch.bfloat16)
    model_reference = _Tiny(in_dim, out_dim).to(torch.bfloat16)
    model_reference.load_state_dict(model.state_dict())

    lora_tensors: dict[str, torch.Tensor] = {}
    for i in range(n_layers):
        a = torch.randn(rank, in_dim, dtype=torch.bfloat16) * 0.1
        b = torch.randn(out_dim, rank, dtype=torch.bfloat16) * 0.1
        lora_tensors[f"diffusion_model.layer{i}.lora_A.weight"] = a
        lora_tensors[f"diffusion_model.layer{i}.lora_B.weight"] = b

    lora_path = tmp_path / "lora.safetensors"
    save_file(lora_tensors, str(lora_path))

    # Trainer path: construct LoRA (pairs, pins, strips diffusion_model. prefix),
    # then apply the same addmm_ that LoRATransform.apply() would.
    lora = LoRA(lora_tensors)
    params = dict(model.named_parameters())
    for target_key, (a, b) in lora.targets.items():
        p = params[target_key]
        p.data.addmm_(b.to(dtype=p.dtype), a.to(dtype=p.dtype), alpha=strength)

    # Inference path: apply_loras with diffusion_model.-prefixed state dict.
    ref_sd = {
        f"diffusion_model.layer{i}.weight": model_reference.get_parameter(f"layer{i}.weight").detach().clone()
        for i in range(n_layers)
    }
    model_state = StateDict(sd=ref_sd, device=torch.device("cpu"), size=0, dtype={torch.bfloat16})
    lora_state = StateDict(sd=lora_tensors, device=torch.device("cpu"), size=0, dtype={torch.bfloat16})
    fused_state = apply_loras(model_state, [LoraStateDictWithStrength(lora_state, strength)])

    for i in range(n_layers):
        got = model.get_parameter(f"layer{i}.weight")
        expected = fused_state.sd[f"diffusion_model.layer{i}.weight"]
        torch.testing.assert_close(got, expected, rtol=1e-2, atol=1e-3)
