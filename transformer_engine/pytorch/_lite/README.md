# Transformer Engine Lite (`tealite`)

A pure-Python drop-in replacement for the `transformer_engine_torch` C++ extension
module. It targets **ROCm / AMD GPUs** and eliminates the need for C++ compilation
by dispatching to [AITER](https://github.com/ROCm/aiter) kernels (CK / Triton),
standalone Triton kernels, or PyTorch-native ops.

## Motivation

The full Transformer Engine build compiles hundreds of C++ / CUDA / HIP sources
via CMake. This takes significant time, couples tightly to toolchain versions, and
makes rapid iteration difficult on ROCm. The lite module sidesteps all of that:

- **No C++ compilation** -- build a wheel in seconds, not minutes.
- **No git submodules** -- AITER is an optional pip dependency (`pip install amd-aiter`).
- **Transparent activation** -- the module registers itself as
  `transformer_engine_torch` via `sys.modules`, so all existing TE code works
  without changes.

## Building the `tealite` Wheel

```bash
# From the repo root:
NVTE_LITE_ONLY=1 pip install .

# Or build the wheel without installing:
NVTE_LITE_ONLY=1 python setup.py bdist_wheel
```

This produces a wheel named **`tealite`** containing only Python and Triton
sources. A `LITE_BUILD` marker file is embedded in the package so that lite mode
activates automatically at import time -- no environment variable needed.

### Using lite mode with a full build

If you have a full Transformer Engine build installed, you can activate lite mode
at runtime instead:

```bash
NVTE_LITE=1 python train.py
```

## Runtime Backend Selection

Most subsystems follow a tiered fallback:

1. **AITER** (CK or Triton kernels from `amd-aiter`) -- best performance on MI300X
2. **Triton kernels** (bundled in `transformer_engine/pytorch/triton_kernels/`)
3. **PyTorch-native ops** -- always available, no extra dependencies

GEMM backend can be forced via `NVTE_LITE_GEMM_BACKEND={ck,triton,pytorch}`.

## Module Structure

```
_lite/
  __init__.py          # Public API -- mirrors transformer_engine_torch exports
  enums.py             # Pure-Python re-declarations of C++ enum types
  aiter_utils.py       # Shared AITER availability detection (lru_cache)

  # Compute kernels
  gemm.py              # GEMM dispatch (AITER CK/Triton, PyTorch matmul)
  attention.py          # Fused attention (AITER CK, flash-attn stub, SDPA)
  norms.py             # LayerNorm / RMSNorm (Triton, PyTorch)
  activations.py       # Activation functions (AITER fused gated, PyTorch)
  rope.py              # Rotary position embeddings (AITER, PyTorch)
  quantize.py          # FP8/MXFP8/MXFP4 quantization (Triton cast, PyTorch)
  softmax.py           # Scaled/masked softmax variants (PyTorch)
  dropout.py           # Dropout (PyTorch)
  transpose.py         # FP8 transpose ops

  # Structured / MOE
  permutation.py       # MOE token permutation (Triton sort, PyTorch gather)
  router.py            # MOE router ops -- topk, aux loss (PyTorch)
  padding.py           # Multi-row padding / unpadding

  # Distributed
  comm.py              # Comm-overlap stubs (not available; use torch.distributed)
  context_parallel.py  # THD <-> BSHD conversion helpers
  mori_ep.py           # Expert parallelism via MORI (dispatch/combine, autograd)

  # Optimizer
  multi_tensor.py      # Multi-tensor Adam, SGD, scale, L2 norm (PyTorch)

  # Misc
  misc.py              # Utility stubs
```

---

## Feature Status

Each section below compares the lite module against the full C++ build.

### GEMM

| Feature | Lite | Full Build |
|---------|------|------------|
| BF16 / FP16 / FP32 GEMM | AITER Triton or `torch.matmul` | cuBLAS / hipBLAS |
| Per-tensor FP8 x FP8 | AITER CK (`gemm_a8w8`) | cuBLAS |
| Per-row FP8 x FP8 | AITER Triton (`gemm_a8w8_per_token_scale`) | N/A |
| Block-scaled FP8 x FP8 | AITER CK/Triton (`gemm_a8w8_blockscale`) | cuBLAS |
| Mixed precision (FP16 x FP8) | AITER CK (`gemm_a16w8`) | cuBLAS |
| MXFP4 x MXFP4 | AITER CK/Triton (`gemm_a4w4`) | cuBLAS |
| Grouped GEMM | AITER Triton GMM | cuBLAS grouped |
| Bias epilogue | Yes | Yes |
| GELU epilogue | Yes | Yes |
| Accumulation epilogue | Yes | Yes |
| Multi-stream cuBLAS | No | Yes |

**Gaps:** No multi-stream execution. Performance depends on AITER kernel
maturity for each precision/shape combination. PyTorch fallback dequantizes to
BF16 before `torch.matmul`, losing the FP8 memory bandwidth advantage.

---

### Attention

| Feature | Lite | Full Build |
|---------|------|------------|
| Dense attention (BSHD, SBHD) | AITER CK / SDPA | CK / cuDNN / flash-attn |
| Variable-length (THD) | AITER CK varlen | CK / cuDNN |
| Causal masking | Yes | Yes |
| Padding masking | Yes | Yes |
| Sliding window | AITER only | Yes |
| ALiBi / bias types | AITER only | Yes |
| GQA (grouped query) | Yes (head expansion) | Yes |
| Dropout | Yes | Yes |
| KV cache copy | Yes | Yes |
| cuDNN backend | No | Yes |
| flash-attn package | Stub (NotImplementedError) | Yes |
| Softmax stats (LSE) | AITER only; SDPA returns dummy | Yes |

**Gaps:** No cuDNN attention backend. Flash-attn integration is stubbed out.
SDPA fallback does not return real softmax statistics (LSE), which can affect
numerics in some training configurations. Sliding window and bias types require
AITER -- no PyTorch fallback for those features.

---

### Activations

| Feature | Lite | Full Build |
|---------|------|------------|
| Non-gated (GeLU, ReLU, SiLU, QGeLU, SReLU) | PyTorch ops | CUDA kernels |
| Gated (GeGLU, SwiGLU, ReGLU, QGeGLU, SReGLU) | AITER fused or PyTorch | CUDA kernels |
| ClampedSwiGLU | PyTorch | CUDA kernel |
| All backward variants | Yes | Yes |
| Fused dbias + dact (non-gated) | Yes | Yes |
| Fused dbias + dact (gated) | No | Yes |
| Fused activation + FP8 quantization | No (quantize post-compute) | Yes (FULLY_FUSED, FUSED_AMAX_FP8, NVFP4) |

**Gaps:** No fused activation + quantization -- always a separate post-compute
step, meaning extra memory traffic. Gated dbias fusions are missing. Only SwiGLU
and GeGLU get AITER-fused forward kernels; the other 9 activations run as
unfused PyTorch ops.

---

### LayerNorm / RMSNorm

| Feature | Lite | Full Build |
|---------|------|------------|
| LayerNorm forward / backward | AITER Triton > TE Triton > PyTorch | CUDA tuned kernels |
| RMSNorm forward / backward | AITER Triton > TE Triton > PyTorch | CUDA tuned kernels |
| RMSNorm backward + add | Yes | Yes |
| Zero-centered gamma | Yes | Yes |
| Fused RMSNorm + FP8 quant (delayed) | AITER (`fused_rms_fp8_per_tensor_static_quant`) | CUDA kernel |
| Fused RMSNorm + FP8 quant (current, per-row) | AITER (`rmsnorm2d_fwd_with_dynamicquant`) | N/A |
| Fused RMSNorm + MXFP8 quant (block) | AITER (`fused_rms_fp8_group_quant`) — partial | CUDA kernel |
| Output quantization (generic) | Yes | Yes |
| cuDNN backend | No | Yes (optional) |
| Pre-tuned hidden sizes (28 sizes) | No (auto-tune) | Yes |
| Fused LayerNormLinear | No | Yes |
| Fused LayerNormMLP | No | Yes |
| SM margin (backward) | Ignored | Full per-stage control |
| Tensor / sequence parallelism | No | Yes |
| FSDP2 integration | No | Yes |

**Gaps:** No cuDNN backend or pre-tuned CUDA kernels. The compound fused modules
(`LayerNormLinear`, `LayerNormMLP`) are full-build-only -- these fuse norm +
projection into single kernels with FP8 and parallelism support. SM margin
control is ignored in the backward pass. No distributed parallelism integration.

The core norm operations are the strongest lite subsystem. AITER Triton kernels
are the primary backend with TE Triton and PyTorch fallbacks. The fused
RMSNorm+FP8 quantize path for CurrentScaling is a lite-only feature -- it fuses
norm and per-row quantize into a single kernel, which is not available in the
full C++ build (see [FP8 Training](#fp8-training) below).

---

### RoPE (Rotary Position Embeddings)

| Feature | Lite | Full Build |
|---------|------|------------|
| Basic RoPE (forward / backward) | AITER or PyTorch | CUDA kernel |
| QKV fused RoPE | Yes | Yes |
| Tensor formats (sbhd / bshd / thd) | None (single assumed layout) | All three |
| Interleaved cos/sin layout | No | Yes |
| Partial RoPE (`rotary_percent`) | No | Yes |
| `start_positions` (KV-cache inference) | No | Yes |
| `cu_seqlens` (THD ragged packing) | No | Yes |
| Context parallelism (`cp_size` / `cp_rank`) | Yes | Yes |
| Position interpolation (NTK-like) | No | Yes |
| `RotaryPositionEmbedding` module | No | Yes |

**Gaps:** Only the simplest layout works -- apply rotation to a dense tensor with
a single assumed layout. Missing `start_positions` blocks KV-cache inference,
missing `cu_seqlens` blocks variable-length batching. Context parallelism is
supported for multi-GPU training with `cp_size` / `cp_rank` parameters.

---

### Quantization

| Feature | Lite | Full Build |
|---------|------|------------|
| Per-tensor Float8 (e4m3 / e5m2) | Triton cast kernel | CUDA kernel |
| Per-row dynamic Float8 (CurrentScaling) | AITER (`dynamic_per_token_quant_fp8_i8`) | N/A |
| MXFP8 (block-scaled) | Triton cast kernel | CUDA kernel |
| MXFP4 | Triton cast kernel | CUDA kernel |
| Dequantize | Yes | Yes |
| Bias-grad + quantize | Yes | Yes |
| Multi-tensor quantize | Yes | Yes |
| Amax compute / update | Yes | Yes |
| Block-scaling partial amax / cast | Yes | Yes |
| Fused cast + transpose | Triton (noop variant) | CUDA kernel |
| FP8 recipe management | Via PyTorch quantizers | Full DelayedScaling + recipes |

**Gaps:** Minimal. The Triton cast kernels cover all major quantization formats.
When a `Float8CurrentScalingQuantizer` is used and AITER is available, all
quantize calls automatically use per-row dynamic scaling instead of per-tensor --
this is strictly better (higher precision, single kernel) and happens
transparently. Performance difference vs CUDA kernels varies by shape and dtype.

---

### FP8 Training

The lite module supports three FP8 scaling recipes, each with different
trade-offs. The CurrentScaling per-row path is a **lite-only optimization** that
is not available in the full C++ build.

| Recipe | Scaling granularity | Lite backend | Full Build |
|--------|-------------------|--------------|------------|
| `DelayedScaling` | Per-tensor (history window) | AITER fused norm+quant / Triton cast | CUDA kernels |
| `Float8CurrentScaling` | **Per-row dynamic** | AITER fused norm+quant / per-token GEMM | Per-tensor CUDA kernels |
| `MXFP8BlockScaling` | Per-block (128×128 or 1×128) | Triton cast / AITER block GEMM | CUDA kernels |
| `Float8BlockScaling` | Per-block (128×128) | Triton cast / AITER block GEMM | CUDA kernels |

#### CurrentScaling per-row fusion (lite-only)

The full C++ build implements `Float8CurrentScaling` as **per-tensor** current
scaling: scan the entire tensor for `amax`, compute one scale, then quantize.
This requires three kernel launches and three full memory passes:

```
Kernel 1: RMSNorm(input) → BF16 output    [read input, write BF16 to HBM]
Kernel 2: amax = max(abs(BF16 output))     [read BF16 from HBM, write scalar]
Kernel 3: FP8 = BF16 output × scale        [read BF16 from HBM, write FP8]
```

The lite module replaces this with AITER's `rmsnorm2d_fwd_with_dynamicquant`
which fuses norm + quantize into a **single kernel** with **per-row** scaling.
Each row computes its own `max(abs(...))` in registers and quantizes before the
data leaves SRAM. The BF16 intermediate never touches HBM:

```
Kernel 1: RMSNorm+Quant(input) → FP8 output + yscale(M,)   [1 read, 1 write]
```

This works because per-row scaling removes the global data dependency: row 0's
scale doesn't depend on row 1's data, so the fused kernel can process each row
independently.

**Forward path:**

| Step | Kernel | Input → Output |
|------|--------|---------------|
| Fused norm+quant | `rmsnorm2d_fwd_with_dynamicquant` | BF16 → FP8 + scale(M,) |
| GEMM | `gemm_a8w8_per_token_scale` | FP8 × FP8 → BF16 |

**Backward path (dgrad):**

| Step | Kernel | Input → Output |
|------|--------|---------------|
| Per-row quant dY | `dynamic_per_token_quant_fp8_i8` | BF16 → FP8 + scale(M,) |
| dgrad GEMM | `gemm_a8w8_per_token_scale` | FP8 × FP8 → BF16 |

**Backward path (wgrad):**

Per-row scales are along the reduction axis (M) — incompatible with per-token
GEMM. Falls back to per-tensor `gemm_a8w8_CK`, which is acceptable since the
reduction across tokens averages out outliers.

**Precision advantage:** Per-row scaling gives each token its own optimal scale
factor. A batch with high-magnitude outlier tokens no longer forces every token
to share a single scale driven by the outlier's magnitude. This is especially
beneficial for long-context training where activation magnitudes vary widely
across sequence positions.

**Detection is automatic:** When `Float8CurrentScaling` recipe is used in lite
mode and AITER is available, the per-row path activates transparently — no
configuration needed. The `Float8Tensor._scale_inv` field carries shape `(M,)`
instead of `(1,)`, and the GEMM dispatch detects this and routes to
`gemm_a8w8_per_token_scale`.

---

### MOE (Mixture of Experts)

| Feature | Lite | Full Build |
|---------|------|------------|
| Token permutation (forward / backward) | Triton sort + PyTorch gather | CUDA kernel |
| Token unpermutation | PyTorch gather + scatter | CUDA kernel |
| Top-k routing | Fused Triton kernel | CUDA fused kernel |
| Auxiliary load-balancing loss | Fused Triton kernel | CUDA fused kernel |
| Score functions (softmax, sigmoid) | Fused Triton kernel | CUDA fused kernel |

**Gaps:** Functionally complete. Router ops use a fused Triton kernel that
combines topk, scoring, and aux loss in a single pass. The full build uses fused
CUDA kernels. Performance difference is most visible at high expert counts.

---

### Communication / Distributed

| Feature | Lite | Full Build |
|---------|------|------------|
| Comm-overlap (AG/RS + GEMM) | **Not available** (stubs raise error) | Full support |
| NVSHMEM integration | **Not available** | Full support |
| Expert parallelism (EP) | MORI dispatch/combine | NCCL / NVSHMEM |
| `torch.distributed` | Works normally | Works normally |
| Tensor parallelism | No built-in support | Integrated in modules |
| Sequence parallelism | No built-in support | Integrated in modules |
| Context parallelism helpers | THD <-> BSHD conversion only | Full support |

**Gaps:** Comm-overlap APIs remain stubs. Multi-GPU training works via standard
`torch.distributed` (DDP, FSDP), but fused communication + compute overlap is
not available. Tensor and sequence parallelism have no built-in support.

Expert parallelism is now supported via the MORI integration (see below), which
bridges the most significant distributed gap for MoE workloads.

---

### Expert Parallelism (MORI)

The `mori_ep` module integrates AMD's [MORI](https://github.com/ROCm/mori)
(Modular RDMA Interface) library to provide high-performance distributed expert
parallelism for MoE pipelines. MORI handles token dispatch/combine across GPUs
using XGMI (intra-node) and RDMA (inter-node) without requiring C++ extensions.

**Requirements:** `pip install mori` (or build from source with ROCm 6.4+).
MORI shmem must be initialized after `torch.distributed.init_process_group()`.

| Feature | Lite (MORI) | Full Build |
|---------|-------------|------------|
| Token dispatch (flat layout) | `MoriExpertParallel.dispatch` | NCCL all-to-all |
| Token combine (flat layout) | `MoriExpertParallel.combine` | NCCL all-to-all |
| Per-expert layout (grouped GEMM) | `dispatch_standard_moe` / `combine_standard_moe` | Custom kernels |
| Layout conversion (flat <-> per-expert) | `convert_dispatch_to_standard` / `convert_standard_to_combine_input` | N/A |
| Autograd (flat layout training) | `MoriEPDispatch` / `MoriEPCombine` | Integrated in MoE module |
| Autograd (per-expert layout training) | `MoriEPDispatchStdMoE` / `MoriEPCombineStdMoE` | Integrated in MoE module |
| Routing map conversion (mask <-> index) | `mask_to_index` / `index_to_mask` | N/A (native format) |
| Intra-node transport (XGMI) | Yes | N/A (NCCL) |
| Inter-node transport (RDMA) | Yes (multiple kernel types) | N/A (NCCL) |
| FP8 quantized dispatch | `fp8_direct_cast` mode | Yes |
| Convenience dispatch+combine cycle | `dispatch_and_combine` | N/A |

**Kernel types:** `intra_node` (default), `inter_node`, `inter_node_v1`,
`inter_node_v1_ll`, `async_ll`. Standard MoE layout requires `intra_node` or
`inter_node_v1_ll`.

**EP gaps vs full build:**

- **No integration with TE's `MoE` module layer** -- MORI EP is a standalone
  primitive. The full build's `MoE` module handles EP dispatch/combine
  transparently within its forward pass; with lite, you must call the MORI APIs
  explicitly in your training loop.
- **No comm-overlap with expert GEMM** -- dispatch and GEMM run sequentially.
  The full build can overlap EP communication with expert computation.
- **No pipeline-parallel EP** -- only data-parallel expert parallelism is
  supported. No integration with pipeline stages or interleaved scheduling.
- **No heterogeneous expert placement** -- assumes uniform
  `num_experts_per_rank` across all ranks. The full build supports uneven expert
  distribution.
- **Standard MoE layout limited to two kernel types** -- `dispatch_standard_moe`
  / `combine_standard_moe` require MORI built with `ENABLE_STANDARD_MOE_ADAPT=ON`
  and only work with `intra_node` or `inter_node_v1_ll` kernels.

---

### Multi-Tensor Optimizer Ops

| Feature | Lite | Full Build |
|---------|------|------------|
| Multi-tensor Adam | PyTorch | C++ fused kernel |
| Multi-tensor SGD | PyTorch | C++ fused kernel |
| Multi-tensor scale | PyTorch | C++ fused kernel |
| Multi-tensor L2 norm | PyTorch | C++ fused kernel |
| FP8 Adam | PyTorch | C++ fused kernel |
| Capturable Adam | PyTorch | C++ fused kernel + CUDA graphs |

**Gaps:** All functionally correct but use per-tensor PyTorch loops instead of
fused multi-tensor C++ kernels. Optimizer step overhead is higher but typically
not the training bottleneck.

---

## Summary

| Subsystem | Functional Coverage | Performance | Key Backend |
|-----------|-------------------|-------------|-------------|
| GEMM | Full (incl. per-row FP8) | Good (AITER) | AITER CK/Triton |
| Attention | Full | Good (AITER) | AITER CK / SDPA |
| Norms | Full + fused norm+quant | Good (AITER) | AITER Triton / TE Triton |
| FP8 Training | Full (3 recipes) | **Best** (fused per-row) | AITER fused kernels |
| Activations | Full | Moderate | AITER (2 ops) / PyTorch |
| Quantization | Full + per-row dynamic | Good (AITER/Triton) | AITER / Triton cast |
| RoPE | Basic + CP | Moderate | AITER / PyTorch |
| MOE | Full | Good (Triton) | Triton fused router |
| Expert parallelism | Full (standalone) | Good (MORI) | MORI XGMI/RDMA |
| Comm-overlap | **None** | N/A | Stubs |
| Multi-tensor ops | Full | Lower | PyTorch loops |

The lite module provides **functional correctness** across all major compute
paths. Performance is competitive for GEMM, attention, norms, and quantization
where AITER or Triton kernels are available. The **FP8 CurrentScaling per-row
fusion** is a lite-only optimization that outperforms the full build's per-tensor
path by eliminating two HBM round-trips per norm+quantize operation. Expert
parallelism is available via MORI for distributed MoE workloads. The remaining
primary gaps are **comm-overlap** (not available) and **fused compound modules**
(LayerNormLinear, LayerNormMLP) which are full-build-only.
