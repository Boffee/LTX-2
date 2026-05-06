#!/usr/bin/env python

"""
Train LTXV models under the sharded-preprocessing orchestrator.

For large datasets where preprocessing everything up front is impractical
(text embeddings alone can exceed available disk/RAM), this entry point
alternates between ``preprocess_dataset`` and ``LtxvTrainer.train()``
one shard at a time. It reads the same YAML config as ``train.py`` but
requires ``data.dataset_metadata_file`` and the sharded-mode ancillary
fields to be set.

Basic usage:
    python scripts/train_sharded.py CONFIG_PATH [--disable-progress-bars]

For a standard preprocessed dataset (``data.preprocessed_data_root``),
use ``scripts/train.py`` instead.
"""

import os

# Must run before any torch-touching import — the allocator config is read
# at first CUDA init and ignored thereafter. Reduces fragmentation OOMs at
# high-resolution VAE preprocessing (e.g. 640x384x385). setdefault preserves
# a user override exported in the shell. Both names set: PYTORCH_ALLOC_CONF
# is the new name (PyTorch ≥ recent), PYTORCH_CUDA_ALLOC_CONF is the
# deprecated alias kept for older PyTorch builds.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Optional memory-history recording for diagnosing OOMs. Off by default
# (per-allocation stack capture has measurable overhead). When set, every
# alloc/free is recorded with a Python stack trace; on uncaught exception
# we dump a snapshot to the path for later inspection in PyTorch's memory
# viz: https://docs.pytorch.org/memory_viz
LTX_MEMORY_SNAPSHOT_PATH = os.environ.get("LTX_MEMORY_SNAPSHOT_PATH")

from pathlib import Path

import typer
import yaml

from ltx_trainer.config import LtxTrainerConfig
from ltx_trainer.shard_orchestrator import ShardOrchestrator

if LTX_MEMORY_SNAPSHOT_PATH:
    import torch
    torch.cuda.memory._record_memory_history(
        enabled="all",
        context="all",
        stacks="python",
        max_entries=200_000,
    )

app = typer.Typer(
    pretty_exceptions_enable=False,
    no_args_is_help=True,
    help="Train LTXV models under the sharded-preprocessing orchestrator.",
)


@app.command()
def main(
    config_path: str = typer.Argument(..., help="Path to YAML configuration file"),
    disable_progress_bars: bool = typer.Option(
        False,
        "--disable-progress-bars",
        help="Disable progress bars and log status messages instead.",
    ),
) -> None:
    """Run sharded-preprocessing training from the given configuration file."""
    config_path = Path(config_path)
    if not config_path.exists():
        typer.echo(f"Error: Configuration file {config_path} does not exist.")
        raise typer.Exit(code=1)

    with open(config_path, "r") as file:
        config_data = yaml.safe_load(file)

    try:
        trainer_config = LtxTrainerConfig(**config_data)
    except Exception as e:
        typer.echo(f"Error: Invalid configuration data: {e}")
        raise typer.Exit(code=1) from e

    if trainer_config.data.dataset_metadata_file is None:
        typer.echo(
            "Error: data.dataset_metadata_file must be set to use train_sharded.py. "
            "For preprocessed datasets, use scripts/train.py instead."
        )
        raise typer.Exit(code=1)

    try:
        ShardOrchestrator(trainer_config).run(disable_progress_bars=disable_progress_bars)
    except (Exception, KeyboardInterrupt):
        if LTX_MEMORY_SNAPSHOT_PATH:
            import torch
            try:
                torch.cuda.memory._dump_snapshot(LTX_MEMORY_SNAPSHOT_PATH)
                typer.echo(f"Memory snapshot written to {LTX_MEMORY_SNAPSHOT_PATH}")
            except Exception as snap_err:
                typer.echo(f"Failed to dump memory snapshot: {snap_err}")
        raise


if __name__ == "__main__":
    app()
