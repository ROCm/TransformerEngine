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
| LayerNorm forward / backward | Triton or PyTorch | CUDA tuned kernels |
| RMSNorm forward / backward | Triton or PyTorch | CUDA tuned kernels |
| RMSNorm backward + add | Yes | Yes |
| Zero-centered gamma | Yes | Yes |
| Output quantization | Yes (generic quantizer) | Yes (per-tensor, block, MXFP8) |
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

The core norm operations themselves are the strongest lite subsystem -- Triton
kernels with `zero_centered_gamma` and quantizer support cover most single-GPU
use cases.

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
| Context parallelism (`cp_size` / `cp_rank`) | No | Yes |
| Position interpolation (NTK-like) | No | Yes |
| `RotaryPositionEmbedding` module | No | Yes |

**Gaps:** The most feature-limited lite subsystem. Only the simplest case works --
apply rotation to a dense tensor with a single assumed layout. Missing
`start_positions` blocks KV-cache inference, missing `cu_seqlens` blocks
variable-length batching, missing context parallelism blocks distributed
training. Suitable only for basic single-GPU training with uniform sequence
lengths.

---

### Quantization

| Feature | Lite | Full Build |
|---------|------|------------|
| Per-tensor Float8 (e4m3 / e5m2) | Triton cast kernel | CUDA kernel |
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
Performance difference vs CUDA kernels varies by shape and dtype. The
higher-level FP8 recipe and delayed-scaling infrastructure lives above `_lite` in
the PyTorch module layer and works with both backends.

---

### MOE (Mixture of Experts)

| Feature | Lite | Full Build |
|---------|------|------------|
| Token permutation (forward / backward) | Triton sort + PyTorch gather | CUDA kernel |
| Token unpermutation | PyTorch gather + scatter | CUDA kernel |
| Top-k routing | PyTorch (`torch.topk`) | CUDA fused kernel |
| Auxiliary load-balancing loss | PyTorch | CUDA fused kernel |
| Score functions | PyTorch (`F.softmax`) | CUDA fused kernel |

**Gaps:** Functionally complete but entirely PyTorch-native (except Triton sort
for permutation). The full build uses fused CUDA kernels for all router and
permutation ops. Performance difference is most visible at high expert counts.

---

### Communication / Distributed

| Feature | Lite | Full Build |
|---------|------|------------|
| Comm-overlap (AG/RS + GEMM) | **Not available** (stubs raise error) | Full support |
| NVSHMEM integration | **Not available** | Full support |
| `torch.distributed` | Works normally | Works normally |
| Tensor parallelism | No built-in support | Integrated in modules |
| Sequence parallelism | No built-in support | Integrated in modules |
| Context parallelism helpers | THD <-> BSHD conversion only | Full support |

**Gaps:** The most significant gap overall. All comm-overlap APIs are stubs.
Multi-GPU training works via standard `torch.distributed` (DDP, FSDP), but the
fused communication + compute overlap that TE provides for large-scale training
is not available. This primarily affects performance at scale rather than
correctness.

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
| GEMM | Full | Good (AITER) | AITER CK/Triton |
| Attention | Full | Good (AITER) | AITER CK / SDPA |
| Norms | Full | Good (Triton) | Triton kernels |
| Activations | Full | Moderate | AITER (2 ops) / PyTorch |
| Quantization | Full | Good (Triton) | Triton cast kernels |
| RoPE | Basic only | Moderate | AITER / PyTorch |
| MOE | Full | Moderate | Triton sort / PyTorch |
| Comm-overlap | **None** | N/A | Stubs |
| Multi-tensor ops | Full | Lower | PyTorch loops |

The lite module provides **functional correctness** across all major compute
paths. Performance is competitive for GEMM, attention, norms, and quantization
where AITER or Triton kernels are available. The primary gaps are **comm-overlap**
(not available), **RoPE** (missing advanced features), and **fused compound
modules** (LayerNormLinear, LayerNormMLP) which are full-build-only.
