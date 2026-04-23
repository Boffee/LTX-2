"""Block-level CPU offloading for memory-efficient LoRA training.

Keeps most frozen transformer block weights on CPU and moves them to GPU on
demand using ``nn.Module.to()``, which correctly handles quantized tensors
(e.g. quanto QLinear).

Uses LRU eviction so the pre_hook works regardless of traversal direction
(forward 0→47 or backward recomputation 47→0).

LoRA parameters stay on GPU permanently — after each block move, they are
restored to GPU so the optimizer can access them.
"""

from __future__ import annotations

import functools
import logging
from collections import OrderedDict
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)


def _resolve_attr(module: nn.Module, dotted_path: str) -> nn.ModuleList:
    """Resolve a dotted attribute path like ``'base_model.model.transformer_blocks'``."""
    obj: Any = module
    for part in dotted_path.split("."):
        obj = getattr(obj, part)
    if not isinstance(obj, nn.ModuleList):
        raise TypeError(f"Expected nn.ModuleList at '{dotted_path}', got {type(obj).__name__}")
    return obj


class _BlockTracker:
    """Tracks which blocks are on GPU with LRU eviction order."""

    def __init__(self, num_layers: int) -> None:
        self.num_layers = num_layers
        self._on_gpu: set[int] = set()
        self._lru: OrderedDict[int, None] = OrderedDict()
        self.peak_gpu_blocks = 0

    def is_on_gpu(self, idx: int) -> bool:
        return idx in self._on_gpu

    def touch(self, idx: int) -> None:
        if idx in self._lru:
            self._lru.move_to_end(idx)

    def mark_on_gpu(self, idx: int) -> None:
        self._on_gpu.add(idx)
        self._lru.pop(idx, None)
        self._lru[idx] = None
        if len(self._on_gpu) > self.peak_gpu_blocks:
            self.peak_gpu_blocks = len(self._on_gpu)

    def mark_on_cpu(self, idx: int) -> None:
        self._on_gpu.discard(idx)
        self._lru.pop(idx, None)

    def pick_victim(self, protected: set[int]) -> int:
        for idx in self._lru:
            if idx not in protected:
                return idx
        raise RuntimeError("no evictable block")

    def clear(self) -> None:
        self._on_gpu.clear()
        self._lru.clear()


def _move_frozen_to_device(
    layer: nn.Module,
    device: torch.device | str,
    lora_device: torch.device | str | None = None,
) -> None:
    """Move a block to a device, then restore LoRA params to lora_device."""
    layer.to(device)
    if lora_device is not None:
        for p in layer.parameters():
            if p.requires_grad:
                if p.data.device != torch.device(lora_device):
                    p.data = p.data.to(lora_device)
                if p.grad is not None and p.grad.device != torch.device(lora_device):
                    p.grad = p.grad.to(lora_device)


class TrainingBlockOffloader:
    """Streams frozen transformer blocks between CPU and GPU for LoRA training.

    Uses ``nn.Module.to()`` for block transfers, which correctly handles
    quantized tensors (quanto QLinear, etc.). Eviction uses LRU ordering
    so it works for both forward and backward (gradient checkpointing
    recomputation) traversal.

    Parameters
    ----------
    model:
        The model containing the block list (may be PEFT-wrapped).
    target_device:
        The GPU device to use for compute.
    blocks_to_swap:
        Number of blocks to keep offloaded on CPU. Must be < total blocks.
    layers_attr:
        Dotted attribute path to the ``nn.ModuleList`` of sequential blocks.
        Auto-detected if not provided.
    """

    _LAYERS_ATTR_CANDIDATES = [
        "base_model.model.transformer_blocks",  # PEFT-wrapped
        "transformer_blocks",  # unwrapped
    ]

    def __init__(
        self,
        model: nn.Module,
        target_device: torch.device,
        blocks_to_swap: int,
        layers_attr: str | None = None,
    ) -> None:
        self._model = model
        self._layers_attr = layers_attr or self._detect_layers_attr(model)
        self._target_device = target_device
        self._blocks_to_swap = blocks_to_swap

        self._layers: nn.ModuleList | None = None
        self._tracker: _BlockTracker | None = None
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []

        self.setup()

    @classmethod
    def _detect_layers_attr(cls, model: nn.Module) -> str:
        for path in cls._LAYERS_ATTR_CANDIDATES:
            try:
                _resolve_attr(model, path)
                return path
            except (AttributeError, TypeError):
                continue
        raise ValueError(f"Could not find transformer blocks at any of: {cls._LAYERS_ATTR_CANDIDATES}")

    def setup(self) -> None:
        """Initialize offloading state. Re-callable after ``teardown()``."""
        if self._tracker is not None or self._hooks:
            self.teardown()

        self._layers = _resolve_attr(self._model, self._layers_attr)
        num_layers = len(self._layers)

        if self._blocks_to_swap >= num_layers:
            raise ValueError(f"blocks_to_swap ({self._blocks_to_swap}) must be < num_layers ({num_layers})")

        num_resident = num_layers - self._blocks_to_swap
        self._tracker = _BlockTracker(num_layers)

        # Move non-block modules to GPU using .to() (handles quanto tensors)
        layers_attr_parts = self._layers_attr.split(".")
        parent: Any = self._model
        for part in layers_attr_parts[:-1]:
            parent = getattr(parent, part)
        layers_leaf = layers_attr_parts[-1]
        for name, child in parent.named_children():
            if name != layers_leaf:
                child.to(self._target_device)

        # Move all blocks to CPU, keeping LoRA params on GPU
        for layer in self._layers:
            _move_frozen_to_device(layer, "cpu", lora_device=self._target_device)

        # Pre-load initial resident window
        for idx in range(min(num_resident, num_layers)):
            self._layers[idx].to(self._target_device)
            self._tracker.mark_on_gpu(idx)

        self._register_hooks(num_resident)

        logger.info(
            f"Block offloading active: {self._blocks_to_swap}/{num_layers} blocks on CPU, "
            f"{num_resident} resident on GPU"
        )

    def _register_hooks(self, num_resident: int) -> None:
        idx_map: dict[int, int] = {id(layer): idx for idx, layer in enumerate(self._layers)}

        def _pre_hook(
            module: nn.Module,
            _args: Any,  # noqa: ANN401
            *,
            idx: int,
        ) -> None:
            if self._tracker.is_on_gpu(idx):
                self._tracker.touch(idx)
                return

            while len(self._tracker._on_gpu) >= num_resident:
                victim = self._tracker.pick_victim(protected={idx})
                _move_frozen_to_device(self._layers[victim], "cpu", lora_device=self._target_device)
                self._tracker.mark_on_cpu(victim)

            module.to(self._target_device)
            self._tracker.mark_on_gpu(idx)

        for layer in self._layers:
            idx = idx_map[id(layer)]
            h = layer.register_forward_pre_hook(functools.partial(_pre_hook, idx=idx))
            self._hooks.append(h)

    @property
    def peak_gpu_blocks(self) -> int:
        return self._tracker.peak_gpu_blocks if self._tracker is not None else 0

    def reset_peak(self) -> None:
        if self._tracker is not None:
            self._tracker.peak_gpu_blocks = len(self._tracker._on_gpu)

    def teardown(self) -> None:
        """Remove hooks, evict all blocks to CPU."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

        if self._tracker is not None:
            torch.cuda.synchronize(device=self._target_device)
            for idx, layer in enumerate(self._layers):
                if self._tracker.is_on_gpu(idx):
                    _move_frozen_to_device(layer, "cpu", lora_device=self._target_device)
            self._tracker.clear()
            self._tracker = None

        self._layers = None
