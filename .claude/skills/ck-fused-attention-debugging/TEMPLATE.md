# CK/AITER Fused-Attn Debug Handoff Template

Use this template when handing off a failure investigation to TE or AITER/CK owners.

---

## 1) Summary
- **Classification:** `integration` | `kernel` | `unsupported-config` | `unknown`
- **Direction:** `fwd` | `bwd` | `both`

## 2) Environment
- **TE commit:**
- **AITER commit/submodule ref:**
- **ROCm version:**
- **GPU architecture (gfx):**
- **Container image/tag (if applicable):**

## 3) Failing Configuration
- **Batch (B):**
- **Query seq (Sq):**
- **KV seq (Skv):**
- **Num Q heads (H):**
- **Num KV heads (Hg):**
- **Head dim QK (Dqk):**
- **Head dim V (Dv):**
- **DType(s):** fp16 / bf16
- **QKV Layout:** (e.g., `BSHD_BSHD_BSHD`, `BS3HD`, `THD_THD_THD`)
- **Mask type:** (e.g., `CAUSAL_MASK`, `PADDING_CAUSAL_MASK`, `NO_MASK`)
- **Bias type:** (e.g., `NO_BIAS`, `POST_SCALE_BIAS`, `ALIBI`)
- **Dropout:**
- **Window size:** `(left, right)`
- **GQA/MQA details:**

## 4) TE Reproducer
- **Backend intent:** CK only / auto / fallback allowed
- **Command or test entrypoint:**
- **Key env vars:**
  ```bash
  NVTE_FLASH_ATTN=0 NVTE_FUSED_ATTN_AOTRITON=0
  NVTE_LOG_FUSED_ATTN_CONFIG=1 NVTE_LOG_CK_CONFIG=1 CK_FUSED_ATTN_LOG_CONFIG=1
  ```
- **Observed result:** pass/fail
- **First failing log line / error signature:**
- **Backend actually selected (from log):**

## 5) Standalone AITER Reproducer (`benchmark_mha_fwd` / `benchmark_mha_bwd`)
- **Build location:** `3rdparty/aiter/op_tests/cpp/mha`
- **Build command:** `bash build_mha.sh` (or `bash build_mha.sh fwd` / `bwd`)
- **Runtime env:**
	- `LD_LIBRARY_PATH=<TE_ROOT>/transformer_engine/lib:${LD_LIBRARY_PATH}`
	- `AITER_ASM_DIR=$(realpath ../../../hsa)`
- **Exact standalone command(s):**
- **Observed result:** pass/fail
- **First failing log line / error signature:**

## 6) Equivalence Check (TE vs Standalone)
- **Are shape/dtype/flags exactly matched?** yes/no
- **Layout mapping verified?** (`-iperm=1 -operm=1` for BSHD, etc.)
- **LSE enabled?** (`-lse=1` — TE always stores LSE)
- **v3 kernel flags matched?** (`NVTE_CK_USES_FWD_V3` → `-fwd_v3=`, etc.)
- **Any default mismatch noticed?**
- **Notes:**

## 7) Conclusion and Ownership
- **Conclusion:** integration vs kernel vs unsupported-config
- **Likely owner:** TE (`fused_attn_ck.cpp` / `fused_attn.cpp` / `ck_fused_attn_*`) or AITER/CK kernel team
- **Specific area (if integration):**
  - [ ] Stride computation (`generateMatrixStrides` / stride mapping to CK API)
  - [ ] Type/mask/bias conversion (`nvte_to_ck_*`, `set_ck_mask`)
  - [ ] Workspace allocation/layout
  - [ ] Padding removal/addition
  - [ ] Backend eligibility check (`is_ck_backend_supported`)
  - [ ] Aux tensor handling (LSE, rng_state, bias in `Aux_CTX_Tensors`)
  - [ ] Other: ___
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
- **Num Q heads (H):** 32
- **Num KV heads (Hg):** 8
- **Head dim QK (Dqk):** 128
- **Head dim V (Dv):** 128
- **DType(s):** bf16
- **QKV Layout:** `BSHD_BSHD_BSHD`
- **Mask type:** `CAUSAL_MASK`
- **Bias type:** `NO_BIAS`
- **Dropout:** 0.0
- **Window size:** `(-1, 0)`
- **GQA/MQA details:** GQA with ratio 4:1

## 4) TE Reproducer
- **Backend intent:** CK only
- **Command or test entrypoint:** `pytest tests/pytorch/attention/test_attention.py::test_dot_product_attention -x -s -k "test_config"`
- **Key env vars:** `NVTE_FLASH_ATTN=0 NVTE_FUSED_ATTN_AOTRITON=0 NVTE_LOG_CK_CONFIG=1`
- **Observed result:** fail
- **First failing log line / error signature:** `invalid argument: ck_bwd workspace size mismatch`
- **Backend actually selected (from log):** NVTE_CK

## 5) Standalone AITER Reproducer
- **Build location:** `3rdparty/aiter/op_tests/cpp/mha`
- **Build command:** `bash build_mha.sh bwd`
- **Runtime env:**
	- `LD_LIBRARY_PATH=<TE_ROOT>/transformer_engine/lib:${LD_LIBRARY_PATH}`
	- `AITER_ASM_DIR=$(realpath ../../../hsa)`
- **Exact standalone command(s):**
	- `./benchmark_mha_bwd -prec=bf16 -b=4 -h=32 -h_k=8 -d=128 -s=4096 -iperm=1 -operm=1 -mask=1 -mode=0 -kname=1 -v=1`
- **Observed result:** pass
- **First failing log line / error signature:** N/A

## 6) Equivalence Check
- **Are shape/dtype/flags exactly matched?** yes
- **Layout mapping verified?** yes (`-iperm=1 -operm=1`)
- **LSE enabled?** yes (implicit in bwd)
- **v3 kernel flags matched?** yes (default v3 enabled)
- **Any default mismatch noticed?** TE-side workspace alignment default differs from standalone path
- **Notes:** Likely marshaling/normalization issue before CK call

## 7) Conclusion and Ownership
- **Conclusion:** integration
- **Likely owner:** TE (`fused_attn_ck.cpp` argument preparation)
- **Specific area:**
  - [x] Workspace allocation/layout
- **Requested next action:** Inspect workspace-size and alignment mapping in TE→CK bwd path for GQA expanded gradients

## 8) Artifacts
- **Logs attached:** `te_fail.log`, `standalone_pass.log`
- **Minimal reproducer commands attached:** yes
- **Patch/commit links (if any):** none
