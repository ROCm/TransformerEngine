---
name: ck-debugging
description: Triage, investigate, debug, and isolate CK/AITER Fused Attention failures in TransformerEngine as integration vs kernel issues.
---

# CK Fused Attention Debugging Guide (TransformerEngine, ROCm)

Use this playbook to quickly answer one question:
**Is the failure in TE↔CK integration, or in the CK/AITER kernel itself?**

---

## 1) File layout and integration surface

### Backend selection and dispatch (hipified — edit CUDA source, not `*_hip.cpp`)
| File | Role |
|---|---|
| `transformer_engine/common/fused_attn_rocm/fused_attn.cpp` | Runtime backend selection (`nvte_get_fused_attn_backend`), all `nvte_fused_attn_{fwd,bwd}*` entry points that dispatch to CK or AOTriton |
| `transformer_engine/common/fused_attn_rocm/fused_attn_ck.cpp` | CK eligibility (`is_ck_backend_supported`), type/mask/stride conversions, workspace allocation, calls into `ck_fused_attn::ck_attn_{fwd,bwd}` and `ck_attn_varlen_{fwd,bwd}` |
| `transformer_engine/common/fused_attn_rocm/fused_attn_aotriton.cpp` | AOTriton equivalent (useful for comparison when CK fails but AOTriton passes) |
| `transformer_engine/common/fused_attn_rocm/utils.{h,cpp}` | `generateMatrixStrides`, `NVTE_QKV_Matrix` enum — stride computation shared by CK and AOTriton |

### CK kernel wrappers (native ROCm — edit directly, NOT hipified)
| File | Role |
|---|---|
| `transformer_engine/common/ck_fused_attn/include/ck_fused_attn/ck_fused_attn.hpp` | Public API: `ck_attn_fwd`, `ck_attn_varlen_fwd`, `ck_attn_bwd`, `ck_attn_varlen_bwd` + `DType`, `MaskType`, `BiasType` enums |
| `transformer_engine/common/ck_fused_attn/src/ck_fused_attn_fwd.cpp` | Forward kernel dispatch (calls `fmha_fwd` from ck_tile) |
| `transformer_engine/common/ck_fused_attn/src/ck_fused_attn_bwd.cpp` | Backward kernel dispatch |
| `transformer_engine/common/ck_fused_attn/src/ck_fused_attn_utils.{hpp,cpp}` | dtype/bias helpers, `DISPATCH_DTYPE_16BIT` macro |

### Python-level attention dispatch
| File | Role |
|---|---|
| `transformer_engine/pytorch/attention/dot_product_attention/backends.py` | `get_available_attention_backends()` — Python-level backend selection |
| `transformer_engine/pytorch/attention/dot_product_attention/utils.py` | `AttentionLogging` class, `_NVTE_DEBUG`, `_NVTE_DEBUG_LEVEL` |
| `transformer_engine/pytorch/cpp_extensions/fused_attn.py` | `FusedAttnBackend` enum, `fused_attn_fwd`/`fused_attn_bwd` Python wrappers |

### Build system
| File | Role |
|---|---|
| `transformer_engine/common/CMakeLists.txt` | Top-level C++ build, `USE_FUSED_ATTN_CK` compile flag |
| `transformer_engine/common/ck_fused_attn/CMakeLists.txt` | CK wrapper build, links ck_tile and aiter |
| `transformer_engine/common/ck_fused_attn/aiter_prebuilt.cmake` | AITER prebuilt SO linking |

### Tests
| File | Role |
|---|---|
| `tests/pytorch/attention/test_attention.py` | Main attention tests: `test_dot_product_attention`, `test_dpa_mask`, `test_dpa_bias`, `test_dpa_sliding_window`, `test_dpa_alibi_slopes`, `test_dpa_qkv_layout`, `test_dpa_qkv_layout_thd` |
| `3rdparty/aiter/op_tests/test_mha.py` | AITER standalone Python MHA tests |
| `3rdparty/aiter/op_tests/cpp/mha/` | Standalone C++ MHA executables: `benchmark_mha_fwd`, `benchmark_mha_bwd` |

---

## 2) Environment variables reference

### Backend selection
| Env var | Default | Effect |
|---|---|---|
| `NVTE_FUSED_ATTN` | `1` (enabled) | Master toggle for all fused attention; set `0` to disable |
| `NVTE_FUSED_ATTN_CK` | follows `NVTE_FUSED_ATTN` | CK backend toggle |
| `NVTE_FUSED_ATTN_AOTRITON` | follows `NVTE_FUSED_ATTN` | AOTriton backend toggle |
| `NVTE_FLASH_ATTN` | `1` (enabled) | Flash attention toggle |

### CK kernel tuning
| Env var | Default | Effect |
|---|---|---|
| `NVTE_CK_USES_FWD_V3` | `1` | Use ASM v3 forward kernel (faster, narrower config support) |
| `NVTE_CK_USES_BWD_V3` | `1` | Use ASM v3 backward kernel |
| `NVTE_CK_IS_V3_ATOMIC_FP32` | `1` | Use fp32 atomics in bwd v3 (more accurate, slower) |
| `NVTE_CK_HOW_V3_BF16_CVT` | `1` | bf16 conversion method for v3 kernels |
| `NVTE_CK_ZERO_OUT_PAD` | `1` | Zero out padded positions in output |

### Debug/logging (all layers, use together for full trace)
| Env var | Layer | What it logs |
|---|---|---|
| `NVTE_DEBUG=1` + `NVTE_DEBUG_LEVEL=2` | Python (PyTorch) | Backend selection decisions, attention config |
| `NVTE_LOG_FUSED_ATTN_CONFIG=1` | C++ dispatch (`fused_attn.cpp`) | Shape, dtype, layout, mask, window for each fwd/bwd call |
| `NVTE_LOG_CK_CONFIG=1` | C++ CK glue (`fused_attn_ck.cpp`) | CK eligibility filter results, workspace sizes, strides, v3 flags |
| `NVTE_LOG_AOTRITON_CONFIG=1` | C++ AOTriton glue | AOTriton-specific dispatch logging |
| `CK_FUSED_ATTN_LOG_CONFIG=1` | CK kernel wrapper (`ck_fused_attn_fwd/bwd.cpp`) | fmha_traits, fmha_args, kernel name selected |

**Full debug command prefix:**
```bash
NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2 NVTE_LOG_FUSED_ATTN_CONFIG=1 NVTE_LOG_CK_CONFIG=1 CK_FUSED_ATTN_LOG_CONFIG=1 <test command>
```

### CI backend configs (`ci/_utils.sh::configure_fused_attn_env`)
| Mode | `NVTE_FUSED_ATTN` | `NVTE_FUSED_ATTN_CK` | `NVTE_FUSED_ATTN_AOTRITON` | `NVTE_FLASH_ATTN` |
|---|---|---|---|---|
| `auto` | unset | unset | unset | unset |
| `ck` | unset | unset | `0` | `0` |
| `aotriton` | unset | `0` | unset | `0` |
| `flash` | `0` | `0` | `0` | unset |
| `unfused` | `0` | unset | unset | `0` |

---

## 3) Gather minimum reproducibility context (before changing code)

Capture these from logs or user report:
- Forward vs backward failure (`fwd` / `bwd`)
- Exact shape/config: batch (`b`), seq lengths (`s_q`, `s_kv`), num heads (`h`), num kv heads (`hg`), head dim (`d_qk`, `d_v`)
- Data type(s): fp16 / bf16 (CK does not support fp8 in fused attn)
- QKV layout: which `NVTE_QKV_Layout` (e.g., `BSHD_BSHD_BSHD`, `BS3HD`, `THD_THD_THD`)
- Mask type: `NO_MASK`, `CAUSAL_MASK`, `PADDING_CAUSAL_MASK`, `CAUSAL_BOTTOM_RIGHT_MASK`, etc.
- Bias type: `NO_BIAS`, `POST_SCALE_BIAS`, `ALIBI`
- Dropout probability
- Sliding window size: `(window_size_left, window_size_right)`
- GQA/MQA details: `h` vs `hg` ratio
- GPU architecture (`gfx942`, `gfx950`, etc.) + ROCm version + TE commit
- Whether fallback backend (AOTriton/unfused) succeeds with same config

If config info is incomplete, request it first; otherwise debugging is noisy and slow.

When self-collecting logs, enable full logging:
```bash
NVTE_LOG_FUSED_ATTN_CONFIG=1 NVTE_LOG_CK_CONFIG=1 CK_FUSED_ATTN_LOG_CONFIG=1 <test command>
```

If a segfault occurs, rerun under `rocgdb`:
```bash
rocgdb --args python -m pytest <test> -x -s
# then: (gdb) run, wait for crash, (gdb) bt
```

---

## 4) CK eligibility checks (`is_ck_backend_supported`)

The function in `fused_attn_ck.cpp:23-152` applies these filters in order. When CK is rejected, `NVTE_LOG_CK_CONFIG=1` prints the reason. The filters are:

1. **GQA groups**: `num_gqa_groups > 0` and `num_attn_heads % num_gqa_groups == 0`
2. **Data type**: `q_dtype == kv_dtype` and both are fp16 or bf16 (no fp8)
3. **Bias type**: only `NO_BIAS`, `ALIBI`, or `POST_SCALE_BIAS` (no `PRE_SCALE_BIAS`)
4. **Head dim**: `head_dim_qk < 512` and `head_dim_v < 512`
5. **Causal + window**: if causal mask, window must be `(-1, 0)` or `(>=0, 0)`
6. **No mask + window**: if no mask, window must be `(-1, -1)` or `(>=0, >=0)`
7. **QKV packed + GQA**: MQA/GQA cannot use qkvpacked layouts (`3HD`, `H3D`)
8. **QKV packed + seqlen**: qkvpacked requires `s_q == s_kv`
9. **THD + padding**: ragged (THD) format requires a padding mask type
10. **Padding + bias**: padding mask cannot combine with `POST_SCALE_BIAS` or `ALIBI`

If CK is rejected, the runtime falls through to AOTriton, then to `NVTE_No_Backend` (which causes `NVTE_ERROR`).

---

## 5) Common error signatures and where they come from

### From dispatch layer (`fused_attn.cpp`)
- `"Invalid combination of data type and sequence length for rocm fused attention."` — no backend accepted the config. Check eligibility with `NVTE_LOG_CK_CONFIG=1`.
- `"qkv_layout not supported!"` — unknown layout enum value.
- `"window_size should be (-1, 0) or (>=0, 0) for attn_mask_type=..."` — window/mask mismatch.

### From CK glue (`fused_attn_ck.cpp`)
- `"NVTE_3HD NVTE_H3D should have h=hg."` — packed layout with GQA mismatch.
- `"Unexpected Aux_CTX_Tensors->size."` — wrong number of auxiliary tensors passed to fwd/bwd.
- `"Unexpected workspace_size."` — workspace allocation mismatch between first call (size query) and second call (execute).
- `"CK fused attn backend not compiled."` — `USE_FUSED_ATTN_CK` not set at build time.

### From CK kernel wrappers (`ck_fused_attn_fwd/bwd.cpp`)
- `"fused attn configs not supported in ck_fused_attn fwd pass."` — config doesn't match any compiled CK tile kernel.
- `"fused attn configs not supported in ck_fused_attn bwd pass."` — same for backward.
- `"Invalid dtype in ck_fused_attn."` — bad dtype conversion.
- `"Invalid bias_type in ck_fused_attn."` / `"Invalid bias_shape in ck_fused_attn."` — bias type/shape not recognized.
- `"Invalid type for 16 bit.."` — `DISPATCH_DTYPE_16BIT` macro failure.

### From HIP runtime
- `hipError_t` from `NVTE_CHECK_CUDA(...)` wrapping CK calls — usually a kernel launch failure or illegal memory access.

---

## 6) Reproduce in controlled CK-only path

### Path A: TE pytest with CK forced
```bash
# Force CK-only backend
export NVTE_FLASH_ATTN=0
export NVTE_FUSED_ATTN_AOTRITON=0
# Full logging
export NVTE_LOG_FUSED_ATTN_CONFIG=1 NVTE_LOG_CK_CONFIG=1 CK_FUSED_ATTN_LOG_CONFIG=1

pytest tests/pytorch/attention/test_attention.py::test_dot_product_attention -x -s -k "<filter>"
```

### Path B: AITER Python JIT (isolates from TE integration)
1. Install aiter: `cd 3rdparty/aiter && pip install -e .`
2. Use `3rdparty/aiter/op_tests/test_mha.py` or write a minimal reproducer.
3. Call MHA functions directly (e.g. `mha_fwd`, `fmha_v3_fwd`).

### Path C: Standalone C++ executables (maximum isolation)
1. Build:
   ```bash
   cd 3rdparty/aiter/op_tests/cpp/mha
   bash build_mha.sh fwd   # or: bwd, fwd_v3, bwd_v3, or no arg for all
   ```
2. Run with proper env:
   ```bash
   export LD_LIBRARY_PATH=<TE_ROOT>/transformer_engine/lib:${LD_LIBRARY_PATH}
   export AITER_ASM_DIR=$(realpath ../../../hsa)  # or equivalent absolute path
   ```
3. Use `-?` flag to list all arguments.
4. Example commands mapping to TE configs:
   ```bash
   # Forward: batch=4, heads=32, kv_heads=8, dim=128, seq=4096, causal, bf16
   ./benchmark_mha_fwd -prec=bf16 -b=4 -h=32 -h_k=8 -d=128 -s=4096 \
     -iperm=1 -operm=1 -mask=1 -lse=1 -mode=0 -kname=1 -v=1

   # Backward (same config)
   ./benchmark_mha_bwd -prec=bf16 -b=4 -h=32 -h_k=8 -d=128 -s=4096 \
     -iperm=1 -operm=1 -mask=1 -mode=0 -kname=1 -v=1
   ```
5. Key argument mappings:
   - `-iperm=1 -operm=1` → BSHD layout (TE default)
   - `-iperm=0 -operm=0` → SBHD layout
   - `-mask=0` → no mask, `-mask=1` → causal top-left, `-mask=2` → causal bottom-right
   - `-mask=t:L,R` → SWA top-left, `-mask=b:L,R` → SWA bottom-right
   - `-lse=1` → store LSE (TE always does this)
   - `-mode=0` → batch mode, `-mode=1` → group/varlen mode
   - `-bias=n` → no bias, `-bias=e` → elementwise, `-bias=a` → alibi
   - `-fwd_v3=1` / `-bwd_v3=1` → use ASM v3 kernels
   - `-v3_atomic_fp32=0|1` → bwd atomic precision

---

## 7) Decision tree: integration bug vs kernel bug

### Case 1: Fails in TE, passes in standalone `benchmark_mha_{fwd,bwd}` with equivalent config
→ **Likely TE integration bug**. Focus on:
- Argument marshaling in `fused_attn_ck.cpp`: type conversions (`nvte_to_ck_dtype`, `nvte_to_ck_bias_type`, `set_ck_mask`), stride computation (`generateMatrixStrides`), workspace layout
- Backend selection conditions in `fused_attn.cpp` — is the right config reaching CK?
- Padding removal/addition logic (`remove_padding`, `add_padding`, `add_padding_softmax_lse`)
- BSHD-to-THD conversion path (`bshd_to_thd`, `generate_cu_seqlen_padded`)

### Case 2: Fails both in TE and standalone
→ **Likely CK/AITER kernel issue** (or unsupported config). Produce a minimal standalone reproducer and hand off to AITER/CK team.

### Case 3: Passes in TE only when fallback backend (AOTriton) is chosen
→ **CK eligibility guard likely wrong**. Inspect filters in `is_ck_backend_supported`.

### Case 4: Numerical mismatch (passes but wrong values)
→ Compare CK output vs AOTriton output on same config. If CK-standalone also gives wrong values, kernel bug. If only TE-CK path gives wrong values, check:
- Stride ordering (batch vs head vs seq strides differ between batched and varlen paths)
- LSE storage format (padded vs unpadded, h×s_q vs s_q×h ordering)
- Workspace buffer reuse / overlap
- `NVTE_CK_ZERO_OUT_PAD` behavior

---

## 8) High-value integration checks

When the failure is TE-side, verify these in `fused_attn_ck.cpp`:

### Stride computation
- `generateMatrixStrides` in `utils.cpp` computes 4-element strides `[batch, head, seq, dim]` for each matrix
- Batched CK API (`ck_attn_fwd`) expects `stride_b, stride_h, stride_s` (3 strides, dim=1 implied)
- Varlen CK API (`ck_attn_varlen_fwd`) expects `stride_h, stride_s` (2 strides, no batch stride)
- When SBHD+padding triggers pad removal, the varlen strides are recomputed: `stride_h=q_stride[1]`, `stride_s=min(q_stride[0], q_stride[2])`

### Workspace allocation
- First call with `workspace==nullptr` queries size, second call executes
- Workspace sections are allocated sequentially: alibi slopes → softmax LSE → Q/K/V/O without-padding buffers → cu_seqlen_padded
- Mismatch between query and execute allocations causes `"Unexpected workspace_size"` errors

### Type/mask/bias mapping
- NVTE `CAUSAL_MASK` / `PADDING_CAUSAL_MASK` → CK `mask_top_left`
- NVTE `CAUSAL_BOTTOM_RIGHT_MASK` / `PADDING_CAUSAL_BOTTOM_RIGHT_MASK` → CK `mask_bottom_right`
- NVTE `NO_MASK` / `PADDING_MASK` with SWA `(>=0, >=0)` → CK `mask_bottom_right` (not `window_generic`)
- NVTE `POST_SCALE_BIAS` → CK `elementwise_bias`
- NVTE `ALIBI` → CK `alibi` (slope array auto-generated in workspace)

### Backward-specific
- `dq_acc_ptr` workspace for split-K accumulation: sized as `float * nsplits * h * max_tokens_q * d_qk`
- `dk_expanded_ptr` / `dv_expanded_ptr` for GQA: expanded to full head count, then reduced
- `dbias_expanded_ptr` → `dbias_ptr` reduction when bias dims differ
- Deterministic mode always set to `false` (TODO in source)
- LSE from forward must be passed correctly via `Aux_CTX_Tensors->tensors[0]`

---

## 9) Running TE tests for fused attention

### Single test
```bash
NVTE_FLASH_ATTN=0 NVTE_FUSED_ATTN_AOTRITON=0 \
  pytest tests/pytorch/attention/test_attention.py::test_dot_product_attention -x -s \
  -k "model_name_and_params"
```

### Key test functions
| Test | What it covers |
|---|---|
| `test_dot_product_attention` | Core fwd/bwd correctness across dtypes, models |
| `test_dpa_mask` | All mask types |
| `test_dpa_bias` / `test_dpa_bias_shapes` | Bias types and shapes |
| `test_dpa_sliding_window` | SWA with different window sizes |
| `test_dpa_alibi_slopes` | ALiBi attention bias |
| `test_dpa_qkv_layout` | All QKV layout variants |
| `test_dpa_qkv_layout_thd` | THD (ragged) layouts |
| `test_dpa_qkv_layout_thd_mqa_gqa` | THD + MQA/GQA |
| `test_gqa_mla_thd` | GQA/MLA with THD format, CK backend check |
| `test_mha_fp8_vs_f16` / `test_dpa_fp8_vs_f16` | FP8 vs FP16 comparison |

### CI invocation
```bash
# From ci/pytorch.sh — runs with specific backend
ci/pytorch.sh  # uses TEST_LEVEL, TEST_SGPU, TEST_FILTER
# Backend set via configure_fused_attn_env in ci/_utils.sh
```

---

## 10) Common pitfalls

1. **Stride mismatch between batched and varlen paths**: SBHD+padding triggers pad removal which changes the varlen stride computation. The `min(stride[0], stride[2])` logic can produce unexpected results for certain layouts.
2. **Workspace size queried with different params than execute call**: Any change to config between the two calls will cause workspace size mismatch.
3. **Treating unsupported config as runtime failure instead of eligibility failure**: If CK doesn't support a config, it should be caught by `is_ck_backend_supported`, not crash at kernel launch.
4. **Missing backward-only failures**: Always test both fwd and bwd. Some configs work in fwd but fail in bwd (e.g., due to expanded gradient buffers in GQA).
5. **Mismatch between TE-side defaults and standalone binary defaults**: TE always stores LSE (`-lse=1`), always uses `iperm=1 operm=1` for BSHD. Standalone defaults may differ.
6. **Comparing non-equivalent configs across TE and standalone paths**: Ensure mask type, window size, dropout, and all flags match exactly.
7. **v3 kernel fallback**: v3 ASM kernels support a narrower config range than CK fallback. If `NVTE_CK_USES_FWD_V3=1` but the config isn't supported by v3, the kernel wrapper falls back to CK tile. Check `CK_FUSED_ATTN_LOG_CONFIG=1` output for which kernel was actually selected.
8. **cu_seqlen_padded generation**: For BSHD+padding→THD conversion, `generate_cu_seqlen_padded` creates synthetic padded seqlens. If actual padding pattern doesn't match assumptions, results will be wrong.
9. **Build flag `USE_FUSED_ATTN_CK` not set**: If CK backend returns `false` for everything and `NVTE_LOG_CK_CONFIG` produces no output, check that the build included CK. The `is_ck_backend_supported` function returns `false` when compiled without `USE_FUSED_ATTN_CK`.

---

## 11) Output artifact requirements (always produce)

For each investigated failure, record:

**Concise handoff format:**
- **Config:** `B=?, Sq=?, Skv=?, H=?, Hg=?, Dqk=?, Dv=?, dtype=?, layout=?, causal=?, dropout=?, mask=?, bias=?, window=?`
- **TE result:** pass/fail + key error
- **Standalone result:** pass/fail + key error
- **Conclusion:** `integration` / `kernel` / `unsupported-config`
- **Owner:** TE vs AITER/CK

For comprehensive output, reference [TEMPLATE.md](TEMPLATE.md).
