# CK/AITER Fused-Attn Debug Handoff Template

Use this template when handing off a failure investigation to TE or AITER/CK owners.

---

## 1) Summary
- **Classification:** `integration` | `kernel` | `unknown`
- **Direction:** `fwd` | `bwd` | `both`

## 2) Environment
- **TE commit:**
- **AITER commit/submodule ref:**
- **ROCm version:**
- **GPU architecture (gfx):**

## 3) Failing Configuration
- **Batch (B):**
- **Query seq (Sq):**
- **KV seq (Skv):**
- **Num heads (H):**
- **Head dim (D):**
- **DType(s):** fp16 / bf16 / fp8
- **Causal:** true/false
- **Dropout:**
- **Mask/Bias mode:**
- **Windowing/Alibi/Padding:**
- **GQA/MQA details:**

## 4) TE Reproducer
- **Backend intent:** CK only / auto / fallback allowed
- **Command or test entrypoint:**
- **Key env vars:**
- **Observed result:** pass/fail
- **First failing log line / error signature:**

## 5) Standalone AITER Reproducer (`fwd.exe` / `bwd.exe`)
- **Build location:** `3rdparty/aiter/op_tests/cpp/mha`
- **Build command:**
- **Runtime env:**
	- `LD_LIBRARY_PATH=<TE_ROOT>/transformer_engine/lib:${LD_LIBRARY_PATH}`
	- `AITER_ASM_DIR=$(realpath ../../../hsa)`
- **Exact standalone command(s):**
- **Observed result:** pass/fail
- **First failing log line / error signature:**

## 6) Equivalence Check (TE vs Standalone)
- **Are shape/dtype/flags exactly matched?** yes/no
- **Any default mismatch noticed?**
- **Notes:**

## 7) Conclusion and Ownership
- **Conclusion:** integration vs kernel vs unsupported-config
- **Likely owner:** TE (`fused_attn_ck.cpp` / `fused_attn.cpp` / `ck_fused_attn_*`) or AITER/CK kernel team
- **Requested next action:**

## 8) Artifacts
- **Logs attached:**
- **Minimal reproducer commands attached:**
- **Patch/commit links (if any):**

---

# Example (Filled)

## 1) Summary
- **Classification:** `integration`
- **Direction:** `bwd`

## 2) Environment
- **TE commit:** `abc1234`
- **AITER commit/submodule ref:** `def5678`
- **ROCm version:** 6.2.1
- **GPU architecture (gfx):** gfx942

## 3) Failing Configuration
- **Batch (B):** 4
- **Query seq (Sq):** 4096
- **KV seq (Skv):** 4096
- **Num heads (H):** 32
- **Head dim (D):** 128
- **DType(s):** bf16
- **Causal:** true
- **Dropout:** 0.0
- **Mask/Bias mode:** causal mask only
- **Windowing/Alibi/Padding:** none
- **GQA/MQA details:** none

## 4) TE Reproducer
- **Backend intent:** CK only
- **Command or test entrypoint:** `pytest tests/pytorch/fused_attn/test_fused_attn.py::test_bwd_case_x`
- **Key env vars:** CK backend forced; debug logging enabled
- **Observed result:** fail
- **First failing log line / error signature:** `invalid argument: ck_bwd workspace size mismatch`

## 5) Standalone AITER Reproducer (`fwd.exe` / `bwd.exe`)
- **Build location:** `3rdparty/aiter/op_tests/cpp/mha`
- **Build command:** `./mha_build.sh`
- **Runtime env:**
	- `LD_LIBRARY_PATH=<TE_ROOT>/transformer_engine/lib:${LD_LIBRARY_PATH}`
	- `AITER_ASM_DIR=$(realpath ../../../hsa)`
- **Exact standalone command(s):**
	- `./bwd.exe <equivalent args>`
	- `./fwd.exe <equivalent args>`
- **Observed result:** pass (both)
- **First failing log line / error signature:** N/A

## 6) Equivalence Check (TE vs Standalone)
- **Are shape/dtype/flags exactly matched?** yes
- **Any default mismatch noticed?** TE-side workspace/alignment default differs from standalone path
- **Notes:** likely marshaling/normalization issue before CK call

## 7) Conclusion and Ownership
- **Conclusion:** integration
- **Likely owner:** TE (`fused_attn_ck.cpp` argument preparation)
- **Requested next action:** inspect workspace-size and alignment mapping in TE→CK bwd path

## 8) Artifacts
- **Logs attached:** `te_fail.log`, `standalone_pass.log`
- **Minimal reproducer commands attached:** yes
- **Patch/commit links (if any):** none
