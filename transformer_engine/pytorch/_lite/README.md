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

GEMM backend can be forced via `NVTE_LITE_GEMM_BACKEND={pytorch,triton,ck}`
(default `pytorch`, which prefers `torch._scaled_mm` and falls back to AITER).

## Environment Variables

| Variable | Scope | Values | Default | Purpose |
|----------|-------|--------|---------|---------|
| `NVTE_LITE_ONLY` | build-time | `0` / `1` | `0` | When `1`, `setup.py` produces the `tealite` wheel (Python + Triton only, no C++ extensions) and embeds a `LITE_BUILD` marker so lite mode activates automatically at import. |
| `NVTE_LITE` | runtime | `0` / `1` | `0` | When `1`, forces lite dispatch at import time on a full build — `transformer_engine.pytorch` registers `_lite` as `transformer_engine_torch` in `sys.modules` instead of loading the C++ extension. Set automatically by `tealite` wheels via the `LITE_BUILD` marker. |
| `NVTE_LITE_GEMM_BACKEND` | runtime | `pytorch`, `triton`, `ck` | `pytorch` | Forces the GEMM backend in `_lite/gemm.py`. `pytorch` prefers `torch._scaled_mm` (hipBLASLt-backed on ROCm) for FP8 and falls back to AITER for FP8 shapes `_scaled_mm` can't serve (per-row scale on the reduction axis, block-scaled, unsupported dtype combos), with dequantize + `torch.matmul` as last resort. `triton` and `ck` route directly to AITER's Triton or CK kernels respectively. Read once at module import. |
| `NVTE_LITE_AMAX_FUSED` | runtime | `0` / `1` | `1` | When `1` (default), `fused_amax_and_scale_update_after_reduction` dispatches to a single Triton multi-tensor-apply kernel that mirrors `delayed_scaling.cu`'s `kernel_bulk`. Set to `0` to fall back to the per-group Python loop (e.g. for debugging or on a system where the Triton kernel fails to load). |
| `NVTE_LITE_SKIP_FP8_DGRAD_FOR_NORM` | runtime | `0` / `1` | `0` | Opt-in optimization for `LayerNormLinear` / `LayerNormMLP` fused modules: when set, the dgrad GEMM emits BF16 instead of FP8 if the only downstream consumer is the norm backward (which would dequantize anyway). Eliminates a BF16 → FP8 → BF16 round-trip; DelayedScaling amax bookkeeping is preserved via a standalone reduction (`amax_utils.update_amax_from_bf16`). Scoped to `Float8Quantizer` and `Float8CurrentScalingQuantizer`; MXFP8 is skipped because per-block scales can't be reconstructed from amax alone. |
| `NVTE_LITE_DIAG` | runtime | `0` / `1` | `0` | Enables one-shot diagnostic prints from `_lite/{gemm,norms,attention,quantize}.py` (and `module/base.py`) capturing shapes, dtypes, scale layout, scaled-mm rejection reasons, etc. Off by default; intended for triaging numerical or dispatch issues. |

## Module Structure

```
_lite/
  __init__.py          # Public API -- mirrors transformer_engine_torch exports
  enums.py             # Pure-Python re-declarations of C++ enum types
  aiter_utils.py       # Shared AITER availability detection (lru_cache)
  amax_utils.py        # BF16 amax-update helper for skip-FP8-dgrad path

  # Compute kernels
  gemm.py              # GEMM dispatch (torch._scaled_mm, AITER CK/Triton, PyTorch matmul)
  grouped_gemm.py      # Grouped GEMM for MoE (AITER Triton GMM, BF16/FP16; FP8 NYI)
  attention.py         # Fused attention (AITER CK, flash-attn stub, SDPA)
  norms.py             # LayerNorm / RMSNorm (Triton, PyTorch)
  activations.py       # Activation functions (AITER fused gated, PyTorch)
  rope.py              # Rotary position embeddings (AITER, PyTorch)
  quantize.py          # FP8/MXFP8/MXFP4 quantization (Triton cast, PyTorch)
  softmax.py           # Scaled/masked softmax variants (PyTorch)
  dropout.py           # Dropout (PyTorch)
  transpose.py         # FP8 transpose ops

  # Compound modules (pure-Python autograd Functions, lazy-loaded to avoid
  # circular import with tex registration; see __init__.py)
  fused_layernorm_linear.py  # LayerNorm+Linear fused autograd Function
  fused_layernorm_mlp.py     # LayerNorm+MLP fused autograd Function

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
maturity for each precision/shape combination. The default `pytorch` backend
routes FP8×FP8 through `torch._scaled_mm` (hipBLASLt-backed on ROCm), which
keeps FP8 memory bandwidth — only when `_scaled_mm` rejects the combo
(per-row scale on the reduction axis, certain block-scaled or unsupported
dtype combos) does the GEMM fall through to dequantize + `torch.matmul`,
which loses the FP8 bandwidth advantage.

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

**FP8 attention flags (`fp8_dpa`, `fp8_mha`):** Not supported. AITER, PyTorch
SDPA, and the stubbed flash-attn path all operate on bf16/fp16 — there is no
FP8 attention kernel in lite. Setting either flag to `True` on the recipe
raises a clear `NotImplementedError` from `get_fused_attn_backend` pointing
back at `fp8_dpa=False / fp8_mha=False`. The default recipe (both flags
`False`, which is the default) continues to work — attention runs bf16 while
GEMMs use FP8. See `TestFP8AttentionFlags` for the contract.

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
| Fused activation + FP8 quantization (gated) | AITER (`act_mul_and_fp8_group_quant`) | Yes (FULLY_FUSED, FUSED_AMAX_FP8, NVFP4) |
| Fused activation + FP8 quantization (non-gated) | No (quantize post-compute) | Yes |

**Gaps:** Gated activations (SwiGLU, GeGLU, ReGLU) use AITER's
`act_mul_and_fp8_group_quant` to fuse activation + gate multiply + FP8 quantize
into a single kernel, eliminating the BF16 intermediate round-trip. This covers
both `Float8BlockQuantizer` (per-block scales, `group_size=block_len`) and
`Float8CurrentScalingQuantizer` (per-row scales, `group_size = output_hidden_dim`
so each row gets one scale). Non-gated activations (GeLU, ReLU, SiLU, etc.)
still run as separate ops with post-compute quantize. Gated dbias fusions are
missing. Activations outside the 6 with explicit paths (swiglu/geglu/reglu for
fused + gelu/silu/relu for basic) fall back to unfused PyTorch ops.

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
| Fused LayerNormLinear | Yes (pure-Python autograd Function) | Yes (CUDA) |
| Fused LayerNormMLP | Yes (pure-Python autograd Function) | Yes (CUDA) |
| SM margin (backward) | Ignored | Full per-stage control |

**Gaps:** No cuDNN backend or pre-tuned CUDA kernels. SM margin control is
ignored in the backward pass. Distributed-parallelism status (TP/SP/FSDP2)
for the fused compound modules is documented in the
[Communication / Distributed](#communication--distributed) section.

`LayerNormLinear` and `LayerNormMLP` are implemented as pure-Python
`torch.autograd.Function` subclasses in `_lite/fused_layernorm_linear.py` and
`_lite/fused_layernorm_mlp.py`. They reuse the AITER fused norm+quant path
when FP8 is active, then chain to the Linear/MLP GEMMs. This is not the same
thing as the full build's single CUDA kernel, but functionally covers the same
API surface — including `return_bias`, `return_layernorm_output`, and all
supported activations.

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
| FP8 recipe management (`fp8_autocast`, recipes) | Yes (pure Python, shared) | Yes |

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
| Grouped GEMM — BF16 / FP16 (fwd / dgrad / wgrad) | AITER Triton GMM (`gmm` / `ptgmm`) | cuBLAS grouped |
| Grouped GEMM — FP8 | **Not yet supported** (NYI) | cuBLAS grouped |
| `te.GroupedLinear` / `GroupedMLP` (BF16) | Yes — `tex.te_general_grouped_gemm` hot-swap | Yes |

**Gaps:** Router and permutation ops are functionally complete (fused Triton
kernel for topk/scoring/aux-loss in a single pass; full build uses fused CUDA
kernels). Performance difference is most visible at high expert counts.

The expert compute path for `GroupedLinear` / `GroupedMLP` is served by
`_lite/grouped_gemm.py`, which adapts AITER's Triton GMM kernels (`gmm`,
`ptgmm`) to the C++ `te_general_grouped_gemm` signature — no `_lite/`
GroupedLinear module is needed; the tex hot-swap is sufficient. **FP8 grouped
GEMM is not yet supported**: AITER's generic GMM family is BF16/FP16 only
(the `p`/`np` prefix is persistent vs non-persistent kernel, not per-tensor
scaling), and FP8 expert compute lives in AITER as a separate fused-MoE op
(`aiter.fused_moe`, `moe_op_gemm_a8w8_blockscale`) with a different API
shape. Run with `TE_FP8=0` for MoE training in lite mode until the Phase 2
dispatcher lands. See also the `TestGroupedLinear::test_fp8_forward` xfail
under [Known xfails](#known-xfails).

---

### Communication / Distributed

| Feature | Lite | Full Build |
|---------|------|------------|
| Comm-overlap (AG/RS + GEMM) | **Not available** (stubs raise error) | Full support |
| NVSHMEM integration | **Not available** | Full support |
| Expert parallelism (EP) | MORI dispatch/combine | NCCL / NVSHMEM |
| `torch.distributed` | Works normally | Works normally |
| FSDP2 integration | Yes — `use_fsdp2=True` wraps weights in `FSDPAGTensor` (1D mesh; HSDP / 2D mesh not yet plumbed) | Yes |
| Tensor parallelism | No built-in support (compound modules accept `tp_size`/`tp_group`/`parallel_mode` for API compat but ignore them; hardcoded `tp_size=1`) | Integrated in modules |
| Sequence parallelism (Megatron-style) | No built-in support (requires TP) | Integrated in modules |
| Context parallelism | RoPE + attention CP supported; THD <-> BSHD conversion helpers | Full support |

**Gaps:** Comm-overlap APIs remain stubs. Tensor parallelism and Megatron-style
sequence parallelism (which is a TP optimization) have no built-in support in
lite's fused compound modules — `LayerNormLinear` / `LayerNormMLP` accept the
related kwargs for API compatibility but hardcode `tp_size=1`. The multi-node
story for lite is therefore FSDP/HSDP-shaped, not TP-shaped. FSDP2 with a 1D
mesh is supported and tested; HSDP (2D mesh) requires `device_mesh` plumbing
through the compound modules and is not yet wired.

Expert parallelism is supported via the MORI integration (see below), which
bridges the most significant distributed gap for MoE workloads. Context
parallelism (sequence sharding without TP, e.g. Ulysses-style) is supported
in attention and RoPE.

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

## Running Tests

The lite module has a dedicated test suite at `tests/pytorch/test_lite.py`.
All tests run entirely in lite mode (the file sets `NVTE_LITE=1` before
importing TE, so the C++ extension is never loaded).

```bash
# Full lite test suite
pytest tests/pytorch/test_lite.py -v

# One test class
pytest tests/pytorch/test_lite.py::TestRecipeIntegration -v

# Tests filtered by name
pytest tests/pytorch/test_lite.py -k "current_scaling" -v
```

### Test Coverage

| Class | What it covers |
|-------|----------------|
| `TestImport` | Module loads, key symbols exist |
| `TestForward` | bf16 forward for Linear / LayerNormLinear / LayerNormMLP / LayerNorm / RMSNorm / TransformerLayer |
| `TestBackward` | Same modules with `loss.backward()` |
| `TestNumerical` | Lite output vs `torch.nn` reference (FP32 exact / BF16 close) |
| `TestTritonNorms` | Triton + AITER norm kernels, fused norm+quant for Float8Quantizer / Float8CurrentScalingQuantizer / MXFP8Quantizer |
| `TestQuantize` | FP8 quantize/dequantize (no-recursion), `bgrad_quantize`, CurrentScaling per-row |
| `TestMXFP8` | MXFP8 tensor detection, E8M0 scale conversion, roundtrip error |
| `TestGemm` | `generic_gemm` with all transpose combinations, bias epilogue, bias-gradient epilogue |
| `TestAttention` | Fused attention: BSHD / SBHD / THD layouts, causal / padding masks, GQA, bias |
| `TestMoERouter` | MoE router top-k, softmax / sigmoid score functions, aux-loss |
| `TestMoEPermutation` | Token permute / unpermute / roundtrip, gradient shapes |
| `TestMoEPadding` | Multi-row pad / unpad / roundtrip across dtypes |
| `TestLiteLayerNormLinear` | LayerNormLinear bf16 forward+backward, LayerNorm/RMSNorm variants, `return_layernorm_output` |
| `TestLiteLayerNormMLP` | LayerNormMLP bf16 forward+backward, non-gated + gated activations |
| `TestFusedGatedActQuant` | AITER fused gated act + block FP8 quantize (swiglu/geglu/reglu × Float8BlockQuantizer) |
| `TestFusedGatedActCurrentScaling` | AITER fused gated act + per-row FP8 quantize (swiglu/geglu/reglu × Float8CurrentScalingQuantizer, `group_size = N/2`) |
| `TestRecipeIntegration` | Full `te.autocast(recipe=...)` path for Linear / LayerNormLinear / LayerNormMLP / TransformerLayer × DelayedScaling / Float8CurrentScaling; multi-step loops; FP8 vs bf16 correlation |
| `TestLiteAPI` | Public symbol presence, tex function signatures, DType enum, module constructor kwargs, regression tests |
| `TestFP8Training` | `optimizer.step()`-driven training — overfit-a-batch (loss must drop), FP8 vs bf16 weight tracking, cache-invalidation |
| `TestFP8AttentionFlags` | `fp8_dpa=True` / `fp8_mha=True` raise clean `NotImplementedError`; default flags work |
| `TestGroupedLinear` | GroupedLinear forward+backward, output matches manual F.linear per chunk, uneven m_splits |

Total: **~285 tests** covering forward, backward, FP8 recipes (DelayedScaling /
Float8CurrentScaling end-to-end), API contracts, training loops, and MoE ops.
The suite is the primary gate against regressions in the lite build.

### Known xfails

- `TestGroupedLinear::test_fp8_forward` — FP8 GroupedLinear hits a dtype
  mismatch in the Triton GMM wrapper (`lhs=fp32` vs `bias=bf16`). This is a
  pre-existing issue in `triton_kernels/gmm/gmm_common.py`; out of scope for
  the lite adapter. The marker is `strict=True`, so if the Triton fix lands
  upstream the test will fail-loud (XPASS → FAIL) to force a deliberate flip.

### Adding new tests

- Any new kernel or dispatch path added to `_lite/` should get a regression
  test in `test_lite.py`. Prefer the test class closest to the feature
  (e.g. a new GEMM kernel → `TestGemm`, a new recipe-level feature →
  `TestRecipeIntegration`).
- Tests that exercise FP8 recipes should use the `_RECIPES_FWD_BWD` or
  `_RECIPES_FWD` helpers to parametrize across whatever recipes the hardware
  supports, so tests skip cleanly on unsupported hardware.
- FP8-vs-bf16 correlation (cosine similarity ≥ 0.9 for single modules,
  ≥ 0.75 for TransformerLayer) is the standard numerical check — catches
  silent wrong-dispatch and scale-broadcast bugs.

---

## Summary

| Subsystem | Functional Coverage | Performance | Key Backend |
|-----------|-------------------|-------------|-------------|
| GEMM | Full (incl. per-row FP8) | Good | `torch._scaled_mm` (hipBLASLt) / AITER CK / Triton |
| Attention | Full | Good (AITER) | AITER CK / SDPA |
| Norms | Full + fused norm+quant | Good (AITER) | AITER Triton / TE Triton |
| FP8 Training | Full (3 recipes) | **Best** (fused per-row) | AITER fused kernels |
| Activations | Full | Moderate | AITER (2 ops) / PyTorch |
| Quantization | Full + per-row dynamic | Good (AITER/Triton) | AITER / Triton cast |
| RoPE | Basic + CP | Moderate | AITER / PyTorch |
| MOE | BF16 full; FP8 grouped GEMM NYI | Good (Triton) | Triton fused router + AITER Triton GMM |
| Expert parallelism | Full (standalone) | Good (MORI) | MORI XGMI/RDMA |
| Comm-overlap | **None** | N/A | Stubs |
| Multi-tensor ops | Full | Lower | PyTorch loops |

The lite module provides **functional correctness** across all major compute
paths. Performance is competitive for GEMM, attention, norms, and quantization
where AITER or Triton kernels are available. The **FP8 CurrentScaling per-row
fusion** is a lite-only optimization that outperforms the full build's per-tensor
path by eliminating two HBM round-trips per norm+quantize operation. Expert
parallelism is available via MORI for distributed MoE workloads. The remaining
primary gaps are **comm-overlap** (not available), **tensor/sequence
parallelism** (no built-in support in lite's compound modules), **FP8 grouped
GEMM** (BF16/FP16 only — blocks FP8 MoE training, see the MOE section), and a
handful of FP8 attention paths (`fp8_dpa` / `fp8_mha` — see the Attention
section).
