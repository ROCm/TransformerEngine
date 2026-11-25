# Using KOSMOS's fused MoE in ROCm Transformer Engine

`transformer_engine.pytorch.kosmos.kosmos_moe` runs the routed-expert part of an MoE layer (token dispatch, both expert GEMMs with SwiGLU, combine, router-weighted sum) as one kernel in the forward and one in the backward, with the token all-to-all overlapped with the GEMMs.
Scope: bf16, an expert-parallel (EP) group of 8 GPUs in one node, gfx950 (MI350X / MI355X), PyTorch. The router stays yours: you pass the top-k expert ids and weights.

## 1. Build

KOSMOS is built into Transformer Engine by default when the build targets gfx950.

```bash
git clone --recursive https://github.com/ROCm/TransformerEngine.git
cd TransformerEngine
NVTE_FRAMEWORK=pytorch NVTE_ROCM_ARCH=gfx950 PYTORCH_ROCM_ARCH=gfx950 NVTE_FUSED_ATTN=0 MAX_JOBS=64 \
  pip install -e . --no-build-isolation -v
python -c "from transformer_engine.pytorch import kosmos; print(kosmos.is_available())"   # True on gfx950
```

`--recursive` fetches the HipKittens version KOSMOS needs (`3rdparty/hipkittens_kosmos`). `--no-build-isolation` is required.

## 2. Minimal use: the defaults are safe

```python
import torch
from transformer_engine.pytorch import kosmos

# Once, when the model is built or a checkpoint is loaded: KOSMOS's W1 row order.
w1 = torch.nn.Parameter(kosmos.to_kosmos_w1(w1_gate_up))   # [E_loc, 2I, H] bf16
w2 = torch.nn.Parameter(w2_down)                            # [E_loc, H, I] bf16

# Every step, on all 8 EP ranks:
#   x            [T, H]  bf16, this rank's tokens
#   topk_idx     [T, k]  int32 or int64 global expert ids (-1 drops a route, e.g. for padding tokens)
#   topk_weights [T, k]  fp32 router weights (they get a gradient)
y = kosmos.kosmos_moe(ep_group, x, topk_idx, topk_weights, w1, w2)   # [T, H] bf16
y.backward(dy)
```

With no other arguments the op is correct for **any** routing: nothing is ever dropped.
The two sizing knobs have defaults chosen so that you do not need to touch them to start:

| knob | default | what happens when a call does not fit |
|---|---|---|
| `capacity_rows` (receive rows per GPU) | 1.5 x the balanced load plus padding: `1.5 T k + 256 E_loc`, rounded up to 256 | the call still runs correctly, in rounds inside the same kernel, slower; `status()` bit 8 |
| `saved_rows` (rows a training call keeps for its backward) | equal to `capacity_rows` | the call keeps nothing and its backward recomputes (correct, slower); `status()` bit 32 |

So: start with the defaults, check `kosmos.status()` now and then (section 4), and tune only if it reports bits 8 or 32, or if you want to save memory.

Converting `W1` per call instead of once is slow and puts the gradient on the copy; passing an unconverted `W1` is not detected and gives wrong results. `kosmos.from_kosmos_w1` converts back for checkpoints.

## 3. In a training loop (Megatron-style fp32 `main_grad`)

```python
w1.main_grad = torch.zeros_like(w1, dtype=torch.float32)
w2.main_grad = torch.zeros_like(w2, dtype=torch.float32)

y = kosmos.kosmos_moe(ep_group, x, topk_idx, topk_weights, w1, w2,
                      fuse_wgrad_accumulation=True)   # the backward kernel adds dW into main_grad
```

With `main_grad` on both weights (both fp32 or both bf16) the kernel adds the weight gradients directly; if the weights carry Megatron's `grad_added_to_main_grad` attribute it is set, as Megatron's DDP expects.
Everything a backward needs is an ordinary PyTorch tensor saved by autograd, so the following work without anything else to call:

- pipeline parallelism: any number of microbatches in flight, backwards in any order;
- `torch.utils.checkpoint` (reentrant or not) around the layer;
- saved-tensor offload (`torch.autograd.graph.saved_tensors_hooks`, `save_on_cpu`);
- CUDA graphs: make **one eager call** per configuration first (it creates KOSMOS's buffers), then capture forwards and backwards; copy new inputs into the captured tensors before each replay.

**The one rule:** every EP rank must make the same `kosmos_moe` calls, forward and backward, in the same order, on one stream, with the same `save` and `saved_rows`. The kernels wait for their peers; a rank that skips or adds a call leaves the other seven waiting forever. Timing loops must stop after a step count all ranks agree on, never on each rank's own clock.

## 4. Watching the status

```python
st = kosmos.status(ep_group)          # host-blocking; OR of the status bits since the last call; resets them
if st & 8:  print("a call needed more receive rows than capacity_rows (ran in rounds)")
if st & 32: print("a call kept nothing for its backward (recomputed): rows > saved_rows, or rounds")
```

`status` raises on error bits by default (`raise_on=(1, 2, 64)`: 1 an expert id out of range, 2 a token routed twice to one expert, 64 an error). Bit 1 is also set by `-1` ids, so pass `raise_on=(2, 64)` if you pad with `-1`.
It synchronizes with the GPU, so call it every few hundred steps, not every step.

## 5. Tuning `capacity_rows` and `saved_rows`

Both are in units of **received rows on the busiest GPU of the EP group**, each local expert's rows padded up to a multiple of 256 (that is how KOSMOS lays them out). Measure it from your own routing:

```python
def busiest_rank_rows(topk_idx, num_experts, ep_group, pad=256):
    """Rows the busiest EP rank receives for this call, each local expert padded to `pad`."""
    ids = topk_idx[topk_idx >= 0].flatten().long()
    counts = torch.bincount(ids, minlength=num_experts)              # this rank's tokens per global expert
    torch.distributed.all_reduce(counts, group=ep_group)             # all ranks' tokens per global expert
    padded = (counts + pad - 1) // pad * pad
    per_rank = padded.view(torch.distributed.get_world_size(ep_group), -1).sum(dim=1)  # expert e lives on rank e // E_loc
    return int(per_rank.max())
```

Run it over a few hundred steps of real routing, take the maximum, and add a margin. Then:

| situation | `capacity_rows` | `saved_rows` | `save` |
|---|---|---|---|
| getting started | default | default | `"all"` |
| routing close to balanced, want less memory | the measured maximum plus a margin (e.g. 5%) | default (= capacity) | `"all"` |
| routing uneven (hot experts / GPUs), `status()` shows bit 8 | the measured maximum plus a margin; this removes the rounds | default, or lower to save memory (calls above it recompute) | `"all"` |
| memory tight | as above | lower | `"xp"` (recompute the first GEMM) or `"none"` (also re-send tokens) |

Things to know:

- Every rank allocates for the busiest rank. The keep-or-recompute decision must be the same on all ranks, so `saved_rows` is compared with the busiest rank's rows; under uneven routing every rank's saved buffers are sized for the hot one.
- `capacity_rows` is part of the configuration: KOSMOS allocates its fixed buffers once per distinct (`H`, `I`, `E_loc`, `k`, `T`, `capacity_rows`, ...) and every layer with the same configuration shares them. Use one value per shape, not per call.
- `save` and `saved_rows` may differ between calls (e.g. `save="xp"` on some layers), but each call must use the same values on all ranks.
- Measured on MI355X with a synthetic hot spot (one GPU receiving 2.4x the mean rows), the default capacity made every call run in rounds and recompute, 1.29x to 1.34x slower than with `capacity_rows` set for the hot GPU.

## 6. Memory

`kosmos.memory_report(ep_group)` lists, per configuration, what KOSMOS holds outside PyTorch's caching allocator and what one training call keeps until its backward.
Measured per GPU at H = I = 4096, 256 experts, top-8, 8192 tokens per GPU (MI355X):

| | default capacity (106,496 rows) | `capacity_rows=70656` |
|---|---|---|
| fixed per configuration, outside the caching allocator (KOSMOS's communication buffer and scratch) | 6.7 GiB | 4.8 GiB |
| per training call in flight, caching allocator, `save="all"` | 3.8 GiB | 2.7 GiB |
| same, `save="xp"` | 1.4 GiB | 1.1 GiB |
| same, `save="none"` | 0.57 GiB | 0.57 GiB |

"In flight" means forwards whose backward has not run yet, over all MoE layers of that configuration (with pipeline parallelism: layers per stage x microbatches in flight). That memory is ordinary PyTorch memory, freed when each backward finishes.

## 7. Other settings

- `wgcap` (default 256): workgroups per KOSMOS kernel; a lower value leaves CUs for kernels on other streams (e.g. pipeline or data-parallel collectives running at the same time, which is supported).
- `NVTE_KOSMOS_MOE_SYMMETRIC_MEMORY=auto|torch|kosmos`: where KOSMOS's communication buffer comes from. `auto` (default) uses PyTorch symmetric memory from PyTorch 2.13 on and KOSMOS's own allocation below it (PyTorch 2.10 cannot allocate it on ROCm; 2.12 does not free it). Same results and speed either way; every rank must use the same setting.
- `NVTE_LOG_KOSMOS_MOE_CONFIG=1`: logs each configuration and its memory on rank 0.
- `kosmos.destroy(ep_group)`: collective; frees KOSMOS's buffers (do not replay CUDA graphs that use them afterwards).

## 8. Not supported in this version

MXFP8, KOSMOS's fused router and shared expert, EP other than 8 GPUs in one node, JAX. They are planned for later versions.
