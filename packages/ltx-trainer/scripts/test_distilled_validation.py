#!/usr/bin/env python3
# ruff: noqa: T201
"""Standalone test: base model + distilled LoRA, no trainable LoRA.

Isolates the validation pipeline from training to diagnose whether
noisy validation samples are caused by the trainable LoRA or by
something in the generation config / merge logic itself.

Usage:
    cd packages/ltx-trainer
    uv run python scripts/test_distilled_validation.py
"""

import sys
import torch
from pathlib import Path
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file

from ltx_core.memory import BlockOffloader
from ltx_trainer.model_loader import load_embeddings_processor, load_model, load_text_encoder
from ltx_trainer.progress import StandaloneSamplingProgress
from ltx_trainer.validation_sampler import (
    CachedPromptEmbeddings,
    GenerationConfig,
    ValidationSampler,
)
from ltx_trainer.video_utils import save_video

# --- Config (mirrors ltx2_3_av_lora_448_epoch_sharded.yaml) ---
MODEL_PATH = "/home/brian/models/lightricks/LTX-2.3/ltx-2.3-22b-dev.safetensors"
TEXT_ENCODER_PATH = "/home/brian/models/google/gemma-3-12b-it-qat-q4_0-unquantized"
BASE_LORA_PATH = "/home/brian/models/lightricks/ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
BASE_LORA_STRENGTH = 0.6

PROMPT = (
    "A 15-second cinematic action sequence: a sweeping aerial drone shot races over "
    "a collapsing skyscraper at dusk, plunges into a high-speed crane descent alongside "
    "a hero leaping between shattering glass facades, then transitions through a whip-pan "
    "into a slow-motion bullet-time orbit as a massive shockwave erupts with volumetric "
    "smoke, sparks, and debris VFX; the camera completes the sequence with a Steadicam "
    "push-in tracking the hero through cascading embers, lens flares, and atmospheric "
    "haze, accompanied by thunderous explosions, shattering glass, screaming wind, and "
    "a pounding orchestral score."
)
NEGATIVE_PROMPT = "worst quality, inconsistent motion, blurry, jittery, distorted"

WIDTH, HEIGHT, NUM_FRAMES = 448, 256, 361
FRAME_RATE = 24.0
SEED = 42
GUIDANCE_SCALE = 1.0
STG_SCALE = 0.0

DISTILLED_SIGMAS = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]
BLOCKS_TO_SWAP = 47

LORA_RANK = 64
LORA_ALPHA = 64
LORA_TARGET_MODULES = [
    "to_k", "to_q", "to_v", "to_out.0",
    "ff.net.0.proj", "ff.net.2",
    "audio_ff.net.0.proj", "audio_ff.net.2",
]

OUTPUT_DIR = Path("/home/brian/ltx-runs/ltx2_3_av_lora_448_distilled_r64/test_distilled_only")


def merge_base_lora(transformer: torch.nn.Module, lora_path: str, strength: float) -> int:
    lora_sd_raw = load_file(lora_path)
    prefix = "diffusion_model."
    lora_sd = {
        (k[len(prefix):] if k.startswith(prefix) else k): v
        for k, v in lora_sd_raw.items()
    }

    params = dict(transformer.named_parameters())
    cpu = torch.device("cpu")
    merged = 0
    for key, p in params.items():
        if not key.endswith(".weight"):
            continue
        base_key = key[:-len(".weight")]
        a_key = f"{base_key}.lora_A.weight"
        b_key = f"{base_key}.lora_B.weight"
        if a_key not in lora_sd or b_key not in lora_sd:
            continue
        a = lora_sd[a_key].to(device=cpu, dtype=p.dtype)
        b = lora_sd[b_key].to(device=cpu, dtype=p.dtype)
        p.data.addmm_(b * strength, a)
        merged += 1
    return merged


def load_trained_lora(transformer, lora_path: str):
    """Wrap with PEFT and load trained LoRA weights, same as the trainer."""
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=0.0,
        init_lora_weights=True,
    )
    transformer = get_peft_model(transformer, lora_config)
    sd = load_file(lora_path)
    sd = {k.replace("diffusion_model.", "", 1): v for k, v in sd.items()}
    set_peft_model_state_dict(transformer.get_base_model(), sd)
    return transformer


def main() -> None:
    trained_lora_path = sys.argv[1] if len(sys.argv) > 1 else None
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    print("Loading model components...")
    components = load_model(
        checkpoint_path=MODEL_PATH,
        device="cpu",
        dtype=torch.bfloat16,
        with_video_vae_encoder=False,
        with_video_vae_decoder=True,
        with_audio_vae_decoder=True,
        with_vocoder=True,
        with_text_encoder=False,
    )

    transformer = components.transformer.to(dtype=torch.bfloat16)

    # --- Phase 1: encode prompts on GPU before transformer claims VRAM ---
    print("Encoding prompts (8-bit text encoder)...")
    text_encoder = load_text_encoder(TEXT_ENCODER_PATH, device="cuda", dtype=torch.bfloat16, load_in_8bit=True)

    with torch.inference_mode():
        pos_hs, pos_mask = text_encoder.encode(PROMPT)
        pos_hs = tuple(h.cpu() for h in pos_hs)
        pos_mask = pos_mask.cpu()

    del text_encoder
    torch.cuda.empty_cache()

    embeddings_processor = load_embeddings_processor(MODEL_PATH, device="cpu")
    with torch.inference_mode():
        pos_out = embeddings_processor.process_hidden_states(pos_hs, pos_mask)
    v_ctx_pos = pos_out.video_encoding
    a_ctx_pos = pos_out.audio_encoding
    del pos_hs, pos_mask, pos_out

    v_ctx_neg, a_ctx_neg = None, None

    del embeddings_processor
    print("  Embeddings cached, text encoder freed")

    cached = CachedPromptEmbeddings(
        video_context_positive=v_ctx_pos,
        audio_context_positive=a_ctx_pos,
        video_context_negative=v_ctx_neg,
        audio_context_negative=a_ctx_neg,
    )

    # --- Phase 2: merge LoRA on CPU, then activate block offloading ---
    print(f"Merging distilled LoRA (strength={BASE_LORA_STRENGTH})...")
    merged = merge_base_lora(transformer, BASE_LORA_PATH, BASE_LORA_STRENGTH)
    print(f"  Merged {merged} targets")

    if trained_lora_path is not None:
        print(f"Loading trained LoRA from {trained_lora_path}...")
        transformer = load_trained_lora(transformer, trained_lora_path)
        base_transformer = transformer.get_base_model()
    else:
        base_transformer = transformer

    transformer.requires_grad_(False)

    print(f"Setting up block offloading ({BLOCKS_TO_SWAP} blocks on CPU)...")
    offloader = BlockOffloader(
        base_transformer,
        target_device=device,
        layers_attr="transformer_blocks",
        blocks_to_swap=BLOCKS_TO_SWAP,
    )
    offloader.activate()

    # --- Phase 3: generate ---
    num_steps = len(DISTILLED_SIGMAS) - 1

    gen_config = GenerationConfig(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_FRAMES,
        frame_rate=FRAME_RATE,
        num_inference_steps=num_steps,
        sigmas=DISTILLED_SIGMAS,
        guidance_scale=GUIDANCE_SCALE,
        seed=SEED,
        generate_audio=True,
        stg_scale=STG_SCALE,
        cached_embeddings=cached,
    )

    print(f"Generating {WIDTH}x{HEIGHT}x{NUM_FRAMES} @ {FRAME_RATE}fps...")
    print(f"  Sigmas: {DISTILLED_SIGMAS}")
    print(f"  CFG={GUIDANCE_SCALE}, STG={STG_SCALE}, seed={SEED}")

    with StandaloneSamplingProgress(num_steps=num_steps) as progress:
        sampler = ValidationSampler(
            transformer=transformer,
            vae_decoder=components.video_vae_decoder,
            vae_encoder=components.video_vae_encoder,
            audio_decoder=components.audio_vae_decoder,
            vocoder=components.vocoder,
            sampling_context=progress,
            skip_transformer_to_device=True,
        )
        video, audio = sampler.generate(config=gen_config, device=device)

    offloader.deactivate()

    if trained_lora_path is not None:
        stem = Path(trained_lora_path).stem
        output_name = f"with_{stem}.mp4"
    else:
        output_name = "distilled_only.mp4"
    output_path = OUTPUT_DIR / output_name
    audio_sr = components.vocoder.output_sampling_rate if components.vocoder else None
    save_video(
        video_tensor=video,
        output_path=output_path,
        fps=FRAME_RATE,
        audio=audio,
        audio_sample_rate=audio_sr,
    )
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
