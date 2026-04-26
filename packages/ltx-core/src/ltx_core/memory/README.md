# Memory

A model-agnostic GPU/CPU memory manager for PyTorch. Two pluggable
strategies for moving model weights between host and GPU, plus an LRU
cache that swaps multiple independent models in and out of GPU memory
the way ComfyUI does.

Self-contained, library-friendly: no dependencies beyond `torch` (plus
optional `optimum.quanto` for quantized models). Designed to be lifted
into its own package when a second consumer appears.

## What's in here

| Module | Role |
|---|---|
| `strategy.py` | `ModelStrategy` — the plug-in contract every strategy implements |
| `pinned_weights.py` | `PinnedWeights` — whole-model bulk pinned-CPU↔GPU strategy |
| `block_offloader.py` | `BlockOffloader` — block-level streaming for models bigger than GPU |
| `pinned_buffer.py` | `PinnedParamBuffer` — per-tensor pinning primitive (handles quanto) |
| `model_cache.py` | `ModelCache` — LRU pool over strategies with active-set leases |
| `pipeline_install.py` | Optional one-line monkey-patch installer (see [Integrations](#integrations)) |

## Why use this

You have multiple PyTorch models that don't all fit on GPU
simultaneously, and you want to swap them in and out efficiently
across many calls. Re-loading from disk every call is too slow
(seconds per gigabyte). Keeping all models resident on GPU is too
expensive. `torch.cuda.empty_cache()` plus `.to("meta")` gets you the
basics but leaves significant performance on the table — pinned host
memory does CPU↔GPU DMA at full PCIe bandwidth (~30 GB/s vs.
~3 GB/s from disk), and a single LRU cache across multiple models
matches ComfyUI's swap behavior.

This library gives you:

1. **Strategies** that pin a model's frozen weights to host RAM and
   bulk-DMA them to GPU on demand.
2. **A cache** that holds multiple pinned models, evicts least-recently-
   used inactive entries when a new model needs room, and tracks active
   leases so you can't accidentally evict something you're using.
3. **A clean plug-in contract** so you can write your own strategy
   (disk-mmap, NVMe-paged, multi-GPU shard) and it slots in.

## When to use what

| Situation | Use |
|---|---|
| Model fits on GPU when active; want fast eviction between calls | **`PinnedWeights`** — bulk DMA, ~200 ms for 12 GB at PCIe Gen5 x16 |
| Model too big for GPU even when active | **`BlockOffloader`** — streams transformer blocks via forward hooks |
| Multiple models swap in/out across a script | Wrap each in a strategy, hand to **`ModelCache`** |

## Quick start: PinnedWeights

```python
import torch
from ltx_core.memory import PinnedWeights

model = build_my_model()  # any nn.Module with frozen params
strategy = PinnedWeights(model, target_device=torch.device("cuda"))

# First use pays the pinning cost (clone + pin_memory).
# Subsequent uses skip pinning — bulk-DMA only.
with strategy as gpu_model:
    output = gpu_model(input_tensor)

with strategy as gpu_model:
    output = gpu_model(input_tensor_2)

strategy.close()  # destructive — moves model to "meta", releases pinned
```

`PinnedWeights` mutates the model in place: every frozen
`nn.Parameter` slot gets repointed at a Parameter wrapping pinned CPU
storage. After construction, only access the model through the
strategy's context manager (or `activate()` / `deactivate()`).

## Quick start: BlockOffloader

For models too big to fit on GPU even when active. Streams transformer
blocks through a small GPU-resident window using forward-pre hooks
and a CUDA-stream-based async prefetcher.

```python
import torch
from ltx_core.memory import BlockOffloader

# auto_setup=True (default) runs prepare() + activate() in __init__.
offloader = BlockOffloader(
    model,
    target_device=torch.device("cuda"),
    blocks_to_swap=24,   # offload N blocks; rest stay GPU-resident
    layers_attr="transformer_blocks",  # path to the nn.ModuleList of blocks
    prefetch_count=2,
)

# Forward through `model` normally; hooks stream blocks on demand.
output = model(input_tensor)

# Destructive close (also breaks the hook reference cycle):
offloader.close()
```

Trainable parameters (e.g. LoRA adapters) stay on GPU permanently
while the offloader is active — backward through them is unaffected
by the offload.

## Quick start: ModelCache

For multiple independent models swapping in and out of GPU.

```python
from ltx_core.memory import ModelCache, ModelSpec, PinnedWeights

cache = ModelCache(max_cache_bytes=80 * 1024**3)

# Register specs (lazy — factory only runs on first acquire / cache miss)
cache.register(ModelSpec(
    key="text_encoder",
    estimated_cache_bytes=12 * 1024**3,
    factory=lambda: PinnedWeights(build_text_encoder(), device),
))
cache.register(ModelSpec(
    key="diffusion_model",
    estimated_cache_bytes=24 * 1024**3,
    factory=lambda: PinnedWeights(build_diffusion_model(), device),
))

# First use builds via factory; subsequent uses hit the cache.
with cache.use("text_encoder") as enc:
    embeddings = enc.encode(prompt)

with cache.use("diffusion_model") as t:
    latent = t(...)

# When budget pressure forces eviction, LRU inactive entries go first.
# Active entries (currently inside `cache.use(...)`) are never evicted.
```

You can also auto-register at acquire time:

```python
spec = ModelSpec(key="vae", estimated_cache_bytes=500*1024**2,
                 factory=lambda: PinnedWeights(build_vae(), device))
with cache.use(spec) as vae:  # registers if missing, then uses
    decoded = vae.decode(latent)
```

## Architecture

```
                       ┌──────────────────┐
                       │   ModelCache     │  LRU pool, active-set leases,
                       │                  │  transactional admission
                       └────────┬─────────┘
                                │ uses (via ModelStrategy protocol)
                                ▼
            ┌───────────────────┴───────────────────┐
            │                                       │
   ┌────────▼─────────┐                  ┌──────────▼─────────┐
   │  PinnedWeights   │                  │   BlockOffloader   │
   │  whole-model DMA │                  │  per-block stream  │
   └────────┬─────────┘                  └──────────┬─────────┘
            │                                       │
            └─────────────┬─────────────────────────┘
                          ▼
                ┌──────────────────┐
                │ PinnedParamBuffer│  per-tensor pinned-CPU storage
                │  (quanto-aware)  │  shared primitive
                └──────────────────┘
```

`ModelStrategy` is the protocol every strategy implements —
`cache_bytes`, `activate()`, `deactivate()`, `close()`, plus the
context-manager dunders. `ModelCache` only talks to this protocol;
write a new strategy and it slots in:

```python
from contextlib import AbstractContextManager
from torch import nn

class MyStrategy:
    @property
    def cache_bytes(self) -> int: ...
    @property
    def closed(self) -> bool: ...
    def activate(self) -> nn.Module: ...
    def deactivate(self) -> None: ...
    def close(self) -> None: ...
    def __enter__(self) -> nn.Module: return self.activate()
    def __exit__(self, *exc) -> None: self.deactivate()
```

## Strategy lifecycle

Both built-in strategies follow the same conceptual lifecycle, with
`BlockOffloader` adding an explicit `prepare()` step because pinning
and GPU pool allocation are distinct phases:

```
PinnedWeights:    constructed → activate ↔ deactivate → close
BlockOffloader:   constructed → prepare → activate ↔ deactivate → close
```

`activate()` makes the model usable for compute. `deactivate()`
releases transient GPU resources (the `cache_bytes` worth of pinned
storage stays held). `close()` is the destructive endpoint — it moves
the model to the `meta` device and releases pinned storage; the
wrapped model is unusable afterward.

## Compatibility

- **`torch.compile` is not supported** for managed modules. Both
  strategies swap parameter slots (`module._parameters[leaf] = new_param`)
  on every activate/deactivate, and `BlockOffloader` registers
  forward-pre hooks that mutate slots on every block call. Both
  invalidate the tensor-identity assumptions `torch.compile` makes
  about its trace.
- **Wrap before DDP/FSDP**, not after. Those wrappers manage parameter
  storage themselves and conflict with the slot-swap pattern.
- **Single-thread / sequential.** No internal locking; concurrent use
  on the same strategy or cache is undefined behavior.
- **Buffer mutations during forward are discarded** on `deactivate()`.
  Suitable for inference of stateless modules; not suitable for models
  that need persistent buffer state across calls (BatchNorm running
  stats updated in training mode, RNN/SSM hidden state, KV cache).

## Tied weights

Both strategies handle the standard `tie_weights()` pattern (one
`Parameter` referenced under multiple names) plus the rarer case of
distinct quanto wrappers around shared inner `_data` storage.

`BlockOffloader` rejects (at `prepare()`) tied weights that span
streamed regions — block↔block, block↔non-block, or mixed
trainable/frozen across regions. Slot-local block streaming can't
preserve cross-region tying. Use whole-model `PinnedWeights` for those
models instead.

## Quanto support

Quanto-quantized models (`optimum.quanto.WeightQBytesTensor`) are
handled correctly by both strategies. `PinnedParamBuffer` decomposes
the wrapper into its inner `_data` (int8/fp8) and `_scale` (fp16/fp32)
tensors, pins each, and reconstructs the quanto wrapper around the GPU
storage on activation.

A naive `param.data.clone()` on a quanto tensor silently
*dequantizes* it via the dispatch fallback — the explicit decomposition
is required for correctness.

## Failure modes

The cache and strategies surface failures as typed exceptions rather
than silent corruption.

| Exception | When |
|---|---|
| `ModelTooLargeError` | Cache miss can't fit even after evicting all inactive entries (active entries blocking) |
| `ActivationError` | Strategy's `activate()` raised — the cache discards the entry; next acquire rebuilds |
| `ModelInUseError` | `evict()` / `clear()` / `unregister()` called while entry is active |
| `ModelEvictionError` | Strategy's `close()` raised during eviction — cache state is consistent (entry removed) but underlying resources may have leaked |
| `DuplicateModelKeyError` | `register()` called for an existing key without `replace=True` |
| `ModelNotRegisteredError` | `use(str)` called for an unknown key |

## Observability

```python
snap = cache.snapshot()
snap.used_cache_bytes        # current pinned-host bytes
snap.cached_keys_lru_to_mru  # tuple of currently-cached keys
snap.active_refcounts        # tuple of (key, refcount) for active entries
snap.stats.hits              # cache hit count
snap.stats.evictions         # total evictions
snap.stats.peak_cache_bytes  # high-water mark
```

`ModelCacheSnapshot`, `ModelCacheStats`, and `ModelInfo` are direct
imports from `ltx_core.memory.model_cache` (not re-exported at the
package level — they're observability types, not the typical
acquire/use path).

## Integrations

### `ltx-pipelines` (optional)

A monkey-patch installer routes `DiffusionStage._transformer_ctx` and
`PromptEncoder._text_encoder_ctx` through a `ModelCache` with zero
edits to the upstream `ltx-pipelines` source.

```python
from ltx_core.memory import ModelCache
from ltx_core.memory.pipeline_install import install_model_cache
from ltx_pipelines import TI2VidOneStagePipeline

cache = ModelCache(max_cache_bytes=80 * 1024**3)
install_model_cache(cache)

pipeline = TI2VidOneStagePipeline(...)
for prompt in prompts:
    pipeline(prompt)  # text encoder + transformer cached across calls
```

Cache entries are keyed by block-instance object identity, so reusing
a pipeline gets cache hits across calls; constructing a new pipeline
gets a fresh entry. When a pipeline is garbage-collected, its cache
entries are auto-evicted via `weakref.finalize`.

Streaming mode (`streaming_prefetch_count=N`) and
`torch_compile=True` fall back to the original (non-cached) path
because both are incompatible with the slot-swap pattern.

To uninstall:

```python
from ltx_core.memory.pipeline_install import uninstall_model_cache
uninstall_model_cache()
```

`uninstall_model_cache()` raises `UninstallBusyError` (subclass of
`ModelInUseError`) if any installer-owned entry is currently active —
exit your `cache.use(...)` contexts and retry.

## Inspiration

Shaped by ComfyUI's `model_management.py` (object-identity caching,
weakref finalizers, multi-model swapping) and HuggingFace accelerate's
hook lifecycle patterns, but instance-owned (not global) so it's
library-friendly and embeddable.
