# LTX-Trainer: Block-Level CPU Offloading + Audio LR

## Goal

Enable LTX-2.3 22B LoRA training on RTX 5090 (32GB VRAM) by adding:
1. **Block-level CPU offloading** — swap transformer blocks between CPU pinned memory and GPU
2. **Separate audio learning rate** — different LR for audio vs video pathway LoRA parameters

## Context

### Why offloading is needed

LTX-2.3 22B has 48 `BasicAVTransformerBlock` layers. In bf16 the transformer is ~42GB, in int8-quanto ~22GB. The existing `low_vram` config uses int8 quantization + 8-bit optimizer + gradient checkpointing + rank 16, which barely fits at ~31GB — no headroom for rank 64 or validation. CPU offloading keeps a subset of blocks in pinned CPU memory, streaming them to GPU on demand.

### Why block-level (not layer-level)

We evaluated three offloading strategies used in the community:
- **ai-toolkit**: wraps individual `nn.Linear` layers with ping-pong CUDA streams. Fine-grained but poor compute-to-transfer ratio (a single matmul completes in ~0.13ms, transfer takes ~0.14ms — constantly I/O bound). ~240 sync points per forward pass.
- **musubi-tuner**: swaps entire transformer blocks. Better overlap — block compute (5-20ms) exceeds block transfer (~8ms for int8), so prefetching hides the transfer. ~24 sync points per forward pass.
- **RamTorch**: replaces `nn.Linear` with CPU-resident versions using `share_memory()`. Can't use `pin_memory()` (mutually exclusive with `share_memory()`), and benchmarks show quantization adds overhead rather than helping.

Block-level wins because: fewer sync points (24 vs 240), better PCIe utilization (large DMA transfers), and prefetching reliably hides transfer behind compute. With gradient checkpointing (which doubles all transfers), the sync overhead gap widens further.

### Existing infrastructure

**`ltx-core` already has `layer_streaming.py`** — a production inference offloader with:
- `_LayerStore`: manages pinned CPU copies, tracks which layers are on GPU
- `_AsyncPrefetcher`: issues H2D transfers on a dedicated CUDA stream with per-layer events
- `LayerStreamingWrapper`: registers forward pre/post hooks on each block

This code is directly reusable. The training adaptation requires the specific changes described below.

## Architecture

### Model structure (ltx-core)

```
LTXModel (model/transformer/model.py)
├── transformer_blocks: nn.ModuleList[BasicAVTransformerBlock] (48 blocks)
│   └── BasicAVTransformerBlock (transformer.py:24)
│       ├── attn1, attn2, ff                    ← video self-attn, cross-attn, FFN
│       ├── scale_shift_table                   ← video adaln params
│       ├── audio_attn1, audio_attn2, audio_ff  ← audio self-attn, cross-attn, FFN
│       ├── audio_scale_shift_table             ← audio adaln params
│       ├── audio_to_video_attn                 ← cross-modal (Q:video, KV:audio)
│       ├── video_to_audio_attn                 ← cross-modal (Q:audio, KV:video)
│       └── scale_shift_table_a2v_ca_*          ← cross-modal adaln params
└── (other: embeddings, norms, proj_out — stay on GPU always)
```

Forward pass iterates sequentially (_process_transformer_blocks, model.py:339, loop at line 348):
```python
for block in self.transformer_blocks:
    if self._enable_gradient_checkpointing and self.training:
        video, audio = torch.utils.checkpoint.checkpoint(
            block, video, audio, perturbations, use_reentrant=False
        )
    else:
        video, audio = block(video=video, audio=audio, perturbations=perturbations)
```

### Per-block parameter count

Each `BasicAVTransformerBlock` contains ~390M parameters, calculated from the model dimensions (model.py:41-57):
- Video: 32 heads × 128 d_head = 4096 inner dim
- Audio: 32 heads × 64 d_head = 2048 inner dim

| Component | Params (M) |
|-----------|-----------|
| Video attn1 + attn2 (Q/K/V/out + RMSNorm each) | ~134 |
| Video FFN (4× expansion, GELUApprox) | ~134 |
| Audio attn1 + attn2 | ~34 |
| Audio FFN | ~34 |
| Cross-modal attentions (audio_to_video + video_to_audio) | ~50 |
| Scale/shift tables, norms, gating | ~5 |
| **Total per block** | **~390M** |

48 blocks × 390M = 18.7B params in blocks + ~3B in embeddings/projections ≈ 22B total.

Per-block memory:
- **bf16: ~780MB** per block
- **int8-quanto: ~390MB** per block

### Trainer structure (ltx-trainer)

```
LtxvTrainer (trainer.py)
├── __init__()
│   ├── _load_models()              → loads transformer, VAE, text encoder
│   ├── _setup_accelerator()        → Accelerator(mixed_precision, grad_accum)
│   ├── _collect_trainable_params() → _setup_lora() then [p for p if p.requires_grad]
│   ├── _load_checkpoint()          → loads checkpoint if configured
│   └── _prepare_models_for_training()
│       ├── gradient checkpointing setup
│       ├── accelerator.prepare(transformer)  ← device placement + wrapping
│       └── ← OFFLOADING HOOKS GO HERE (see Integration Point)
└── train()
    └── _init_optimizer()           → AdamW/AdamW8bit(self._trainable_params, lr=lr)
        └── accelerator.prepare(optimizer, scheduler)
```

Config uses Pydantic (config.py):
```python
class AccelerationConfig(ConfigBaseModel):
    mixed_precision_mode: Literal["no", "fp16", "bf16"] | None = "bf16"
    quantization: QuantizationOptions | None = None
    load_text_encoder_in_8bit: bool = False
    # ← ADD: blocks_to_swap

class OptimizationConfig(ConfigBaseModel):
    learning_rate: float = 5e-4
    optimizer_type: Literal["adamw", "adamw8bit"] = "adamw"
    # ← ADD: audio_learning_rate
```

## Implementation: Block-Level CPU Offloading

### Scope: LoRA training only

Block offloading is designed for LoRA training where base model weights are frozen and only small LoRA A/B matrices are trainable. The "skip trainable params" rule (below) means offloading would be a no-op for full fine-tuning. The config should validate: `blocks_to_swap > 0` requires `training_mode == "lora"`.

### What to reuse from `layer_streaming.py`

- `_LayerStore` — pinned memory management (needs one modification: skip trainable params)
- `_AsyncPrefetcher` — async CUDA stream + per-layer events (reuse as-is)

### Integration point

Offloading must be set up **before** `accelerator.prepare()` moves the model to GPU, because `_LayerStore.__init__` calls `tensor.data.pin_memory()` which requires tensors to be on CPU. We use `device_placement=[False]` to prevent Accelerate from overriding our custom device layout while still getting mixed-precision and model-wrapping benefits.

In `_prepare_models_for_training()` (trainer.py:679), the sequence is:

```python
def _prepare_models_for_training(self) -> None:
    # ... existing FSDP dtype cast, gradient checkpointing setup ...

    if self._config.acceleration.blocks_to_swap and self._config.acceleration.blocks_to_swap > 0:
        # Apply offloading BEFORE accelerator.prepare() while weights are on CPU.
        # _setup_lora() has already run, so LoRA params have requires_grad=True
        # and will be skipped by the offloader.
        self._block_offloader = TrainingBlockOffloader(
            model=self._transformer,
            layers_attr="transformer_blocks",
            target_device=self._accelerator.device,
            blocks_to_swap=self._config.acceleration.blocks_to_swap,
            prefetch_count=2,
        )
        # Tell Accelerate not to move the model — we handle device layout.
        self._transformer = self._accelerator.prepare(self._transformer, device_placement=[False])
    else:
        self._transformer = self._accelerator.prepare(self._transformer)
```

### Change 1: Drop post_hook, evict in pre_hook (ring-buffer)

**Problem:** The inference code evicts a block in `post_hook` (immediately after forward). During training with gradient checkpointing, `torch.utils.checkpoint.checkpoint` recomputes the forward inside backward. The sequence is:

```
checkpoint.backward for block i:
  1. Re-run block.forward()     ← pre_hook fires (move to GPU) ✓
                                ← post_hook fires (evict to CPU) ✗ BUG
  2. Backward through the ops   ← needs weights on GPU for grad_input = grad_out @ W.T
```

The post_hook evicts weights BEFORE backward can use them.

**Fix:** Remove post_hook entirely. Handle eviction in pre_hook using a ring-buffer pattern:

```python
def _pre_hook(module, _args, *, idx):
    # Evict the block outside our resident window
    evict_idx = (idx - num_resident) % num_layers
    if evict_idx != idx and store.is_on_gpu(evict_idx):
        store.evict_to_cpu(evict_idx, layers[evict_idx])

    # Ensure current block is on GPU
    prefetcher.wait(idx)
    if not store.is_on_gpu(idx):
        store.move_to_gpu(idx, module)

    # Record compute stream usage
    compute_stream = torch.cuda.current_stream(target_device)
    for param in itertools.chain(module.parameters(), module.buffers()):
        param.data.record_stream(compute_stream)

    # Prefetch next blocks
    for offset in range(1, prefetch_count + 1):
        prefetcher.prefetch((idx + offset) % num_layers)
```

`num_resident = num_layers - blocks_to_swap` (number of blocks on GPU at any given time).

**Note on GPU occupancy:** Peak GPU block count is `num_resident + prefetch_count`, not `num_resident`, because prefetched blocks accumulate before the first eviction triggers. With `prefetch_count=2`, that's 2 extra blocks × ~390MB = ~780MB above the naive estimate.

**Note on backward prefetching:** During backward, blocks are visited in reverse order (47→0). The prefetcher still prefetches forward (idx+1, idx+2), which means backward block transfers fall back to synchronous loads. Estimated overhead: `blocks_to_swap × ~8ms` per backward pass (~192ms for 24 swapped blocks at int8 over PCIe 5.0). This is ~10% wall-clock overhead per step. Optimizing backward prefetch direction (detecting reverse traversal via decreasing idx sequence) would roughly halve this and is a concrete follow-up.

### Change 2: Skip trainable (LoRA) params in move/evict

**Problem:** The inference code moves ALL parameters. During training, LoRA A/B matrices are trainable (`requires_grad=True`) and their gradients accumulate on GPU. If evicted to CPU, the optimizer step fails (device mismatch between param.data on CPU and param.grad on GPU).

**Fix:** In `_LayerStore.move_to_gpu()` and `evict_to_cpu()`, skip trainable params:

```python
for name, param in itertools.chain(layer.named_parameters(), layer.named_buffers()):
    if param.requires_grad:
        continue  # LoRA weights stay on GPU permanently
    param.data = pinned[name].to(self.target_device, non_blocking=non_blocking)
```

LoRA weights are tiny (~0.5MB per Linear at rank 64) — negligible VRAM cost. Similarly, in `__init__`, only pin frozen params:

```python
for name, tensor in itertools.chain(layer.named_parameters(), layer.named_buffers()):
    if tensor.requires_grad:
        continue  # Don't pin trainable params
    pinned_tensor = tensor.data.pin_memory()
    tensor.data = pinned_tensor
    pinned[name] = pinned_tensor
```

**Init ordering dependency:** This works because `_setup_lora()` (called from `_collect_trainable_params()` at trainer.py:99) runs before `_prepare_models_for_training()` (trainer.py:102). By the time the offloader is created, LoRA modules are already injected and their params have `requires_grad=True`.

### Pinned memory: all blocks

`_LayerStore` pins frozen tensors for all 48 blocks. This is correct for the ring-buffer design: the sliding window means every block travels between CPU and GPU across consecutive forward passes (block 0 is evicted during forward to make room for block 24, then needs to be reloaded from CPU at the start of the next forward pass). No block is truly permanent. Total pinned RAM: ~48 × ~390MB ≈ ~18.5GB at int8.

### Config additions

```python
class AccelerationConfig(ConfigBaseModel):
    # ... existing fields ...
    blocks_to_swap: int | None = Field(
        default=None,
        description="Number of transformer blocks to keep on CPU (0 or None = no offloading). "
                    "Blocks are streamed to GPU on demand using async prefetch. "
                    "Requires training_mode='lora'. "
                    "Validated at runtime against actual model layer count.",
        ge=0,
    )
```

Runtime validation in the trainer (not a static `le=47` in the schema):
```python
num_layers = len(transformer.transformer_blocks)
if blocks_to_swap >= num_layers:
    raise ValueError(f"blocks_to_swap ({blocks_to_swap}) must be < num_layers ({num_layers})")
```

### Memory math for RTX 5090

Each `BasicAVTransformerBlock` at int8-quanto ≈ **390MB** (bf16 ≈ 780MB).

| blocks_to_swap | Blocks on GPU (peak) | Model VRAM | Est. total VRAM |
|----------------|---------------------|------------|-----------------|
| 20 | 28 + 2 prefetch = 30 | ~11.7GB + ~2GB non-block = ~13.7GB | ~24.7GB |
| 24 | 24 + 2 = 26 | ~10.1GB + ~2GB = ~12.1GB | ~23.1GB |
| 30 | 18 + 2 = 20 | ~7.8GB + ~2GB = ~9.8GB | ~20.8GB |
| 36 | 12 + 2 = 14 | ~5.5GB + ~2GB = ~7.5GB | ~18.5GB |

Estimates assume: model blocks + ~2GB embeddings/norms/projections + ~8GB activations (checkpointed) + ~1GB optimizer (LoRA only) + ~2GB overhead.

With int8 + 30 blocks swapped: **~20.8GB**. Room for rank 64 and validation on 32GB.

### Quantization interaction

The `_LayerStore` pins tensors via `tensor.data.pin_memory()`. quanto-quantized tensors (QTensor) may not support `pin_memory()`. If so, transfer them without pinning — they'll use pageable memory (slightly slower async, but still works). Check with:

```python
try:
    pinned_tensor = tensor.data.pin_memory()
except RuntimeError:
    pinned_tensor = tensor.data  # fallback for quantized tensors
```

**Performance note:** Pageable memory copies are effectively synchronous even with `non_blocking=True`, which means the prefetcher's overlap benefit is lost for unpinnable tensors. Since int8-quanto is the primary use case, verify empirically whether quanto tensors can be pinned. If not, the async prefetch degrades to synchronous and the backward prefetch optimization becomes more important.

## Implementation: Separate Audio Learning Rate

### Motivation

Audio typically converges faster than video in joint AV training. Using a lower audio LR (e.g., 0.5× video LR) prevents audio overfitting.

### Implementation

In `_init_optimizer()` (trainer.py:771), split trainable params into audio vs video groups. Since this is LoRA training, only params with `requires_grad=True` are considered — these are exclusively LoRA A/B matrices inside the targeted modules (to_k, to_q, to_v, to_out.0).

```python
AUDIO_MODULES = {".audio_attn1.", ".audio_attn2.", ".audio_ff.",
                 ".video_to_audio_attn."}

def _init_optimizer(self) -> None:
    opt_cfg = self._config.optimization
    lr = opt_cfg.learning_rate
    audio_lr = opt_cfg.audio_learning_rate or lr

    audio_params = []
    video_params = []
    for name, param in self._transformer.named_parameters():
        if not param.requires_grad:
            continue
        if any(mod in name for mod in AUDIO_MODULES):
            audio_params.append(param)
        else:
            video_params.append(param)

    param_groups = [{"params": video_params, "lr": lr}]
    if audio_params and audio_lr != lr:
        param_groups.append({"params": audio_params, "lr": audio_lr})
    else:
        param_groups[0]["params"].extend(audio_params)

    if opt_cfg.optimizer_type == "adamw":
        optimizer = AdamW(param_groups)
    elif opt_cfg.optimizer_type == "adamw8bit":
        from bitsandbytes.optim import AdamW8bit
        optimizer = AdamW8bit(param_groups)
```

Uses explicit dotted path-component matches (e.g., `.audio_attn1.`) rather than loose substring matching, to avoid false positives.

### Which modules are "audio"

From `BasicAVTransformerBlock` (transformer.py:24-122), matched by LoRA target presence (to_k, to_q, to_v, to_out.0):

| Module | Role | LR group |
|--------|------|----------|
| `audio_attn1` | Audio self-attention | audio |
| `audio_attn2` | Audio cross-attention (text→audio) | audio |
| `audio_ff` | Audio feed-forward | audio (no LoRA targets by default — FFN has no to_k/to_q/to_v/to_out.0) |
| `video_to_audio_attn` | Cross-modal (Q:audio, KV:video) | audio |
| `audio_to_video_attn` | Cross-modal (Q:video, KV:audio) | video |
| `attn1`, `attn2`, `ff` | Video pathway | video |

`audio_to_video_attn` is kept at video LR: its Q projection operates on video features, and the module primarily serves video generation quality. `video_to_audio_attn` uses audio LR: its Q projection operates on audio features.

Note: `audio_ff` has no default LoRA targets (FeedForward uses `proj` and `net.0.proj`/`net.2`, not `to_k`/`to_q`/`to_v`/`to_out.0`). It is included in `AUDIO_MODULES` for completeness in case custom `target_modules` include FFN layers.

### Config addition

```python
class OptimizationConfig(ConfigBaseModel):
    # ... existing fields ...
    audio_learning_rate: float | None = Field(
        default=None,
        description="Learning rate for audio pathway LoRA parameters. If None, uses the main learning_rate.",
    )
```

## Testing

1. **Smoke test offloading**: Train 10 steps with `blocks_to_swap=24`, verify loss decreases and no device mismatch errors
2. **Compare with no offloading**: Run same config on a high-VRAM GPU without offloading, verify identical loss trajectory (within floating point noise)
3. **Gradient checkpointing interaction**: Verify blocks are on GPU during backward recomputation (add an assert in debug mode)
4. **LoRA weights stay on GPU**: After a training step, verify all `requires_grad=True` params have `.device == cuda`
5. **Audio LR**: Verify optimizer has 2 param groups with different LRs, both update during training
6. **Quantization + offloading**: Test with `int8-quanto` quantization enabled, verify no pin_memory errors
7. **Ring-buffer eviction**: Unit test that at each block index, exactly `num_resident + prefetch_count` blocks are on GPU
8. **Edge cases**: Test `blocks_to_swap=1` and `blocks_to_swap=47`

## Files to modify

| File | Change |
|------|--------|
| `ltx_trainer/config.py` | Add `blocks_to_swap` to `AccelerationConfig`, `audio_learning_rate` to `OptimizationConfig` |
| `ltx_trainer/trainer.py` | Apply offloader before `prepare()` with `device_placement=[False]`, split optimizer param groups |
| `ltx_trainer/block_offloader.py` (NEW) | Training-adapted block offloader (reuses `_LayerStore`, `_AsyncPrefetcher` from ltx-core with trainable-param skip) |
| `configs/ltx2_av_lora_low_vram.yaml` | Add example `blocks_to_swap: 30` config |

## Our training setup (for reference)

- **GPU**: RTX 5090 32GB
- **Model**: LTX-2.3 22B (`/data/models/lightricks/LTX-2.3/ltx-2.3-22b-dev.safetensors`)
- **Text encoder**: Gemma 3 12B qat-q4 (`/data/models/google/gemma-3-12b-it-qat-q4_0-unquantized`)
- **Dataset**: 19,288 video segments, 640×352, F=121-385, 24fps, with audio
- **Training**: LoRA rank 64, alpha 64, lr 1e-4, grad_accum 4, adamw8bit, int8-quanto
- **Target**: ~2000 optimizer steps
