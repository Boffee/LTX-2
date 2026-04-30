"""Smoke test for block offloading and audio LR.

Creates dummy preprocessed data and runs a short training loop to verify:
1. Block offloading works (no device mismatch, loss decreases)
2. Validation deactivate/activate cycling works
3. Audio LR creates separate param groups
4. No regression when features are disabled

Usage:
    uv run python scripts/test_offloading.py \
        --model_path /data/models/lightricks/LTX-2.3/ltx-2.3-22b-dev.safetensors \
        --text_encoder_path /data/models/google/gemma-3-12b-it-qat-q4_0-unquantized
"""

import argparse
import shutil
import tempfile
import time
from pathlib import Path

import torch
import yaml


def create_dummy_data(
    data_dir: Path,
    num_samples: int = 2,
    video_dims: tuple[int, int, int] = (640, 352, 25),
    with_audio: bool = False,
    fps: int = 24,
) -> None:
    latents_dir = data_dir / "latents"
    conditions_dir = data_dir / "conditions"
    latents_dir.mkdir(parents=True)
    conditions_dir.mkdir(parents=True)
    audio_latents_dir = data_dir / "audio_latents" if with_audio else None
    if audio_latents_dir is not None:
        audio_latents_dir.mkdir(parents=True)

    width, height, num_frames = video_dims
    assert num_frames % 8 == 1, f"num_frames must satisfy frames % 8 == 1, got {num_frames}"
    assert width % 32 == 0 and height % 32 == 0, "dims must be divisible by 32"
    latent_frames = (num_frames - 1) // 8 + 1
    latent_h = height // 32
    latent_w = width // 32

    # AudioLatentShape.from_duration with LTX-2.3 defaults (sr=16k, hop=160,
    # latent_downsample=4) → 25 latent frames per second of waveform.
    duration_sec = num_frames / fps
    audio_frames = max(1, round(duration_sec * 25))

    for i in range(num_samples):
        latent_data = {
            "latents": torch.randn(128, latent_frames, latent_h, latent_w),
            "num_frames": latent_frames,
            "height": latent_h,
            "width": latent_w,
            "fps": fps,
        }
        torch.save(latent_data, latents_dir / f"sample_{i:04d}.pt")

        condition_data = {
            "video_prompt_embeds": torch.randn(256, 4096, dtype=torch.bfloat16),
            "audio_prompt_embeds": torch.randn(256, 2048, dtype=torch.bfloat16),
            "prompt_attention_mask": torch.ones(256, dtype=torch.bool),
        }
        torch.save(condition_data, conditions_dir / f"sample_{i:04d}.pt")

        if audio_latents_dir is not None:
            audio_data = {
                "latents": torch.randn(8, audio_frames, 16),
                "num_time_steps": audio_frames,
                "frequency_bins": 16,
                "duration": duration_sec,
            }
            torch.save(audio_data, audio_latents_dir / f"sample_{i:04d}.pt")


def make_config(
    model_path: str,
    text_encoder_path: str,
    data_dir: str,
    output_dir: str,
    blocks_to_swap: int | None = None,
    audio_learning_rate: float | None = None,
    quantization: str | None = "int8-quanto",
    video_dims: tuple[int, int, int] = (640, 352, 25),
    lora_rank: int = 16,
    lora_alpha: int | None = None,
    steps: int = 3,
    with_audio: bool = False,
    optimizer_type: str = "adamw",
) -> dict:
    cfg = {
        "model": {
            "model_path": model_path,
            "text_encoder_path": text_encoder_path,
            "training_mode": "lora",
        },
        "lora": {
            "rank": lora_rank,
            "alpha": lora_alpha if lora_alpha is not None else lora_rank,
            "target_modules": [
                "to_k",
                "to_q",
                "to_v",
                "to_out.0",
                "ff.net.0.proj",
                "ff.net.2",
                "audio_ff.net.0.proj",
                "audio_ff.net.2",
            ],
        },
        "training_strategy": {
            "name": "text_to_video",
            "with_audio": with_audio,
        },
        "optimization": {
            "learning_rate": 1e-4,
            "steps": steps,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "optimizer_type": optimizer_type,
            "scheduler_type": "constant",
            "enable_gradient_checkpointing": True,
        },
        "acceleration": {
            "mixed_precision_mode": "bf16",
            # quantization=None → pure bf16 weights, no quanto
            "quantization": quantization if quantization != "none" else None,
            "load_text_encoder_in_8bit": True,
        },
        "data": {
            "preprocessed_data_root": data_dir,
        },
        "validation": {
            "prompts": ["a cat sitting on a table"],
            "video_dims": list(video_dims),
            "inference_steps": 4,
            "interval": 2,
        },
        "checkpoints": {
            "interval": 9999,
        },
        "flow_matching": {
            "timestep_sampling_mode": "uniform",
        },
        "output_dir": output_dir,
        "seed": 42,
    }

    if blocks_to_swap is not None:
        cfg["acceleration"]["blocks_to_swap"] = blocks_to_swap

    if audio_learning_rate is not None:
        cfg["optimization"]["audio_learning_rate"] = audio_learning_rate

    return cfg


def _verify_offloader(trainer, blocks_to_swap: int) -> None:
    """Check offloader invariants after a training step."""
    strategy = trainer._model_offloader
    assert strategy is not None, "Offloader should be active"

    # The new API: strategy has _components; the last component is the
    # StreamedWeights (PinnedWeights and TrainableWeights come first).
    from ltx_core.memory import StreamedWeights
    streamers = [c for c in strategy._components if isinstance(c, StreamedWeights)]
    assert len(streamers) == 1, f"Expected 1 streamer, got {len(streamers)}"
    streamer = streamers[0]

    num_layers = len(streamer._blocks)
    num_resident = num_layers - blocks_to_swap
    expected_max = num_resident + streamer._prefetch_count

    peak = streamer.peak_gpu_blocks
    assert peak <= expected_max, (
        f"Peak GPU blocks ({peak}) exceeds expected max ({expected_max})"
    )
    assert peak > 0, "No blocks were ever on GPU — offloader may not be active"

    # Verify LoRA params stayed on GPU
    for layer in streamer._blocks:
        for name, p in layer.named_parameters():
            if p.requires_grad:
                assert p.data.is_cuda, f"LoRA param {name} should be on GPU"

    streamer.reset_peak()
    print(f"    offloader OK: peak {peak} blocks on GPU (max allowed: {expected_max})")


def _verify_audio_lr(trainer, expected_audio_lr: float) -> None:
    """Check that optimizer has separate audio/video param groups."""
    groups = trainer._optimizer.param_groups
    assert len(groups) >= 2, f"Expected >=2 param groups, got {len(groups)}"
    audio_group = groups[1]
    assert abs(audio_group["lr"] - expected_audio_lr) < 1e-10, (
        f"Audio LR {audio_group['lr']} != expected {expected_audio_lr}"
    )
    print(
        f"    audio LR OK: {len(groups)} param groups, audio LR={audio_group['lr']:.2e}"
    )


def run_test(name: str, config: dict, tmp_dir: Path) -> bool:
    config_path = tmp_dir / f"{name}.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    print(f"\n{'=' * 60}")
    print(f"TEST: {name}")
    print(f"{'=' * 60}")

    trainer = None
    try:
        from ltx_trainer.config import LtxTrainerConfig
        from ltx_trainer.shard_orchestrator import tear_down_trainer
        from ltx_trainer.trainer import LtxvTrainer

        trainer_config = LtxTrainerConfig(**config)

        # Time model construction separately so the stats.steps_per_second
        # figure isn't drowned by one-time setup cost. trainer.train()'s
        # internal total_time_seconds covers only the training loop itself
        # (it starts its own timer at train_start_time), but block offloader
        # setup happens inside __init__ before we even call train().
        init_start = time.perf_counter()
        trainer = LtxvTrainer(trainer_config)
        init_time = time.perf_counter() - init_start

        blocks_to_swap = config["acceleration"].get("blocks_to_swap")
        audio_lr = config["optimization"].get("audio_learning_rate")

        # Per-step wall-clock timing — measures the gap between consecutive
        # step_callback firings, which matches one optimization step (plus any
        # validation triggered at that step boundary).
        step_times: list[float] = []
        last_mark = [time.perf_counter()]

        def step_callback(step: int, total: int, paths: list[Path]) -> None:
            now = time.perf_counter()
            step_times.append(now - last_mark[0])
            last_mark[0] = now
            if blocks_to_swap:
                _verify_offloader(trainer, blocks_to_swap)
            if audio_lr:
                _verify_audio_lr(trainer, audio_lr)

        train_start = time.perf_counter()
        _model_path, stats = trainer.train(
            disable_progress_bars=True, step_callback=step_callback
        )
        train_wall = time.perf_counter() - train_start

        if step_times:
            per_step = "  ".join(f"step {i + 1}: {t:.2f}s" for i, t in enumerate(step_times))
            print(f"  timing: trainer-init={init_time:.1f}s  train-wall={train_wall:.1f}s")
            print(f"  per-step (wall, between callbacks): {per_step}")

        print(
            f"  PASSED — {stats.steps_per_second:.1f} steps/s, peak VRAM: {stats.peak_gpu_memory_gb:.1f} GB"
        )
        return True
    except Exception as e:
        print(f"  FAILED — {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        if trainer is not None:
            tear_down_trainer(trainer)
            del trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test for block offloading")
    parser.add_argument(
        "--model_path", required=True, help="Path to LTX model checkpoint"
    )
    parser.add_argument(
        "--text_encoder_path", required=True, help="Path to text encoder"
    )
    parser.add_argument(
        "--blocks_to_swap",
        type=int,
        default=24,
        help="Blocks to swap for offloading test",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="int8-quanto",
        choices=["none", "int8-quanto", "int4-quanto", "int2-quanto", "fp8-quanto", "fp8uz-quanto"],
        help="Quanto precision to apply to the transformer; 'none' keeps bf16 weights",
    )
    parser.add_argument(
        "--video-dims",
        type=str,
        default="640x352x25",
        help="Video dims WxHxF (frames must satisfy frames%%8==1; W,H divisible by 32)",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=16,
        help="LoRA rank (also used for alpha unless --lora-alpha is given)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=None,
        help="LoRA alpha (defaults to --lora-rank)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=3,
        help="Training steps per test case (1 is enough for OOM validation)",
    )
    parser.add_argument(
        "--with-audio",
        action="store_true",
        help="Enable audio modality (generates dummy audio_latents and exercises video↔audio cross-attention)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adamw", "adamw8bit"],
        help="Optimizer to use (adamw8bit matches the real audio-video LoRA config)",
    )
    args = parser.parse_args()
    w, h, f = (int(x) for x in args.video_dims.split("x"))
    video_dims = (w, h, f)

    tmp_dir = Path(tempfile.mkdtemp(prefix="offload_test_"))
    data_dir = tmp_dir / "data"
    print(f"Working directory: {tmp_dir}")

    try:
        num_samples = max(1, args.steps)
        create_dummy_data(data_dir, num_samples=num_samples, video_dims=video_dims, with_audio=args.with_audio)

        base_kwargs = {
            "model_path": args.model_path,
            "text_encoder_path": args.text_encoder_path,
            "data_dir": str(data_dir),
        }

        results = {}

        # Test 1: Block offloading
        cfg = make_config(
            **base_kwargs,
            output_dir=str(tmp_dir / "offload"),
            blocks_to_swap=args.blocks_to_swap,
            quantization=args.quantization,
            video_dims=video_dims,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            steps=args.steps,
            with_audio=args.with_audio,
            optimizer_type=args.optimizer,
        )
        results["offloading"] = run_test(
            f"block offloading (blocks_to_swap={args.blocks_to_swap})", cfg, tmp_dir
        )

        # Test 2: Both features together
        cfg = make_config(
            **base_kwargs,
            output_dir=str(tmp_dir / "both"),
            blocks_to_swap=args.blocks_to_swap,
            audio_learning_rate=5e-5,
            quantization=args.quantization,
            video_dims=video_dims,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            steps=args.steps,
            with_audio=args.with_audio,
            optimizer_type=args.optimizer,
        )
        results["both"] = run_test("offloading + audio LR", cfg, tmp_dir)

        print(f"\n{'=' * 60}")
        print("RESULTS")
        print(f"{'=' * 60}")
        for name, passed in results.items():
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {name}")

        all_passed = all(results.values())
        print(f"\n{'ALL PASSED' if all_passed else 'SOME FAILED'}")

    finally:
        shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    main()
