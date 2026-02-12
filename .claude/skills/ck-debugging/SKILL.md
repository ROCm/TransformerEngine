---
name: ck-debugging
description: Triage, investigate, debug, and isolate CK/AITER Fused Attention failures in TransformerEngine as integration vs kernel issues.
---

# CK Fused Attention Debugging Guide (TransformerEngine, ROCm)

Use this playbook to quickly answer one question:
**Is the failure in TE↔CK integration, or in the CK/AITER kernel itself?**

## 1) Map the integration surface first
- Build-time CK args parsing/validation:
  - `transformer_engine/common/CMakeLists.txt`
  - `tools/check_aiter_mha_args_usage.py`
- CK fused-attn kernel wrappers/entry points:
  - `transformer_engine/common/ck_fused_attn/ck_fused_attn_*`
- CK backend preprocessing and dispatch glue:
  - `transformer_engine/common/fused_attn_rocm/fused_attn_ck.cpp`
- Runtime backend selection / fallback path:
  - `transformer_engine/common/fused_attn_rocm/fused_attn.cpp`

## 2) Gather minimum reproducibility context (before changing code)
Capture these from logs or user report:
- Forward vs backward failure (`fwd` / `bwd`)
- Exact shape/config: batch, seq lengths (`s_q`, `s_kv`), num heads, head dim
- Data type(s): fp16/bf16/fp8
- Mask/dropout/causal/windowing/alibi/padding settings
- GQA/MQA/group mode details if used
- GPU architecture + ROCm version + TE commit
- Whether fallback backend succeeds

When self-collecting logs (for example, rerunning a failing pytest), enable full config logging in the same command: `NVTE_LOG_FUSED_ATTN_CONFIG=1 NVTE_LOG_CK_CONFIG=1 CK_FUSED_ATTN_LOG_CONFIG=1 <test command>`.

If reproducing triggers a segmentation fault, rerun under `rocgdb` to capture a usable backtrace: `rocgdb --args python -m pytest ...` (then run and collect `bt`).

If config info is incomplete, request it first; otherwise debugging is noisy and slow.

## 3) Reproduce in controlled CK-only path
Preferred path (AITER Python JIT):
1. Start from `3rdparty/aiter/op_tests/test_mha.py` to reproduce through the same Python JIT interface used in many real flows.
2. Add a minimal wrapper test (for example, `test_te_reproducer`) that pins only the failing TE config.
3. Call the Python-level MHA functions directly (e.g. `mha_fwd` and `fmha_v3_fwd`).
4. Record the exact test invocation, pinned parameters, and first failing log line.

Secondary path (native executables for isolation/confirmation):
1. From `3rdparty/aiter/op_tests/cpp/mha`, build with `mha_build.sh`.
2. Keep env explicit when running:
   - `LD_LIBRARY_PATH=<TE_ROOT>/transformer_engine/lib:${LD_LIBRARY_PATH}`
   - `AITER_ASM_DIR=$(realpath 3rdparty/aiter/hsa)` (or equivalent absolute path)
3. Use `fwd.exe -?` / `bwd.exe -?` to confirm argument mapping.
4. Re-encode the same failing config in `fwd.exe` / `bwd.exe` and compare behavior vs Python JIT.
5. Keep in mind that TE always stores LSE, hence use `-lse=1`.
6. Record full commands to include in handoff.

## 4) Decision tree: integration bug vs kernel bug
1. **Fails in TE, but passes in `fwd.exe`/`bwd.exe` with equivalent config**
   - Likely TE integration bug.
   - Focus on argument marshaling/normalization in:
     - `fused_attn_ck.cpp`
     - `ck_fused_attn_*`
     - backend selection conditions in `fused_attn.cpp`

2. **Fails both in TE and standalone `fwd.exe`/`bwd.exe`**
   - Likely CK/AITER kernel issue (or unsupported config).
   - Produce a minimal standalone reproducer command and hand off.

3. **Passes in TE only when fallback backend is chosen**
   - CK eligibility/selection guard likely wrong.
   - Inspect backend capability checks and shape constraints in `fused_attn.cpp`.

## 5) High-value checks when it is integration-related
- Verify all expected CK args are present and in the right order/type.
- Check TE→CK conversions for:
  - layout / strides
  - sequence length semantics (`s_q` vs `s_kv`)
  - grouped-query mapping
  - mask/bias/dropout flags
  - causal/windowing flags
  - dtype/accumulator assumptions
- Confirm no silent defaulting for missing fields.
- Confirm runtime-selected backend matches intent (no accidental fallback/misroute).

## 6) Output artifact requirements (always produce)
For each investigated failure, record:
- TE reproducer summary (shapes, dtype, flags)
- Standalone command(s) tested (`fwd.exe`/`bwd.exe`) and result
- Classification: `integration` or `kernel`
- Owning component and next action

Suggested concise handoff format:
- **Config:** `B=?, Sq=?, Skv=?, H=?, D=?, dtype=?, causal=?, dropout=?, mask=?`
- **TE result:** pass/fail + key error
- **Standalone result:** pass/fail + key error
- **Conclusion:** integration vs kernel
- **Owner:** TE vs AITER/CK

For more comprehensive output formatting, reference [TEMPLATE.md](TEMPLATE.md)

## 7) Common pitfalls
- Mismatch between TE-side defaults and standalone binary defaults.
- Treating unsupported config as runtime failure instead of eligibility failure.
- Comparing non-equivalent configs across TE and standalone paths.
- Missing backward-only failures (always test both directions when applicable).