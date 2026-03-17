# Debugging Mixed FP8 Triton GEMM Bug with Claude Code

## Executive Summary

Using Claude Code, we identified and root-caused a **Triton compiler bug** that produces silently wrong results when performing GEMM operations with mixed FP8 types (e4m3 x e5m2) on AMD gfx950 GPUs. The bug was traced to two defects in Triton's MFMA instruction lowering (`MFMA.cpp`), already fixed upstream in [triton-lang/triton PR #9567](https://github.com/triton-lang/triton/pull/9567) but not yet included in any PyTorch release as of PyTorch 2.11. The entire investigation -- from symptom discovery to root cause confirmation via GPU assembly analysis -- was completed in a single Claude Code session.

---

## 1. Symptom Discovery

### Context

TransformerEngine's Triton GEMM backend (`NVTE_USE_GEMM_TRITON=1`) supports FP8 tensor inputs via `tl.dot()`. During development of a consolidated test suite for `te_generic_gemm_triton()`, we parametrized tests across FP8 format combinations:

- Same format: e4m3 x e4m3, e5m2 x e5m2
- Mixed format: e4m3 x e5m2, e5m2 x e4m3

### Observation

Same-format FP8 tests passed with near-zero error (max diff ~0.0003), but **mixed-format tests failed catastrophically** with 98-99% of elements mismatched and errors on the order of the output magnitude:

```
Same format  - Max diff: 0.000324, Mean diff: 0.000031, Mismatched: 0/65536 (0.0%)
Mixed format - Max diff: 8.207232, Mean diff: 1.527544, Mismatched: 65409/65536 (99.8%)
```

This pattern held across both reference approaches:
- Triton vs PyTorch `torch.matmul` on dequantized tensors
- Triton vs C++ `tex.generic_gemm` (hipBLASLt backend)

The C++ backend produced correct results for mixed types, confirming the bug was in the Triton path.

### Key reasoning

The fact that same-format works but mixed-format fails is a strong signal. In same-format FP8, both operands have identical type metadata, so any bug that swaps or confuses operand types would be invisible. Mixed types break the symmetry and expose the bug.

---

## 2. Hypothesis Formation

### Understanding the operand swap

TransformerEngine's Triton GEMM follows the standard BLAS-to-Triton conversion: since BLAS uses column-major and Triton uses row-major, operands A and B are **swapped** before calling the kernel.

From `gemm_triton.py` lines 655-659:

```python
# BLAS column-major to Triton row-major conversion: swap A and B
a_row_major = B_flat.T if transb else B_flat
b_row_major = A_flat.T if transa else A_flat
a_scale_triton = b_scale_inv
b_scale_triton = a_scale_inv
```

FP8 dtype reinterpretation happens **before** the swap (lines 566-569):

```python
if a_fp8_dtype is not None:
    A_data = reinterpret_as_fp8_tensor(A_data, a_fp8_dtype)
if b_fp8_dtype is not None:
    B_data = reinterpret_as_fp8_tensor(B_data, b_fp8_dtype)
```

So after the swap, the PyTorch tensors carry the correct dtypes with them. The Triton kernel receives `a_row_major` with dtype e5m2 and `b_row_major` with dtype e4m3 (for user A=e4m3, B=e5m2 with TN layout). The kernel calls `tl.dot(a, b)` which should infer the types from the tensor dtypes.

### Hypothesis

The Triton compiler, when lowering `tl.dot()` to AMD MFMA instructions, might be encoding the operand format metadata incorrectly when the two FP8 types differ. On AMD gfx950, FP8 GEMM uses specialized MFMA instructions that encode each operand's format:

- `v_mfma_f32_32x32x16_{fp8|bf8}_{fp8|bf8}` -- format in the mnemonic
- `v_mfma_f32_32x32x64_f8f6f4` -- format in `cbsz` (src A) and `blgp` (src B) modifier fields

If the compiler assigns these format fields based on pre-swap types rather than actual tensor types, the hardware would misinterpret the data.

---

## 3. AMD MFMA Instruction Specification for FP8

To verify the hypothesis, we needed to understand exactly how AMD's Matrix Fused Multiply-Add (MFMA) instructions encode FP8 operand formats. AMD gfx950 (CDNA4) provides two families of FP8 MFMA instructions, each with different format encoding mechanisms.

### Instruction family 1: Named-format MFMA (CDNA3/CDNA4)

These instructions encode the operand formats directly in the instruction mnemonic:

```
v_mfma_f32_32x32x16_{srcA_fmt}_{srcB_fmt}  dst, src0, src1, src2
```

Available mnemonics for 32x32x16:

| Mnemonic | src0 (A) format | src1 (B) format |
|---|---|---|
| `v_mfma_f32_32x32x16_fp8_fp8` | E4M3 (fp8) | E4M3 (fp8) |
| `v_mfma_f32_32x32x16_fp8_bf8` | E4M3 (fp8) | E5M2 (bf8) |
| `v_mfma_f32_32x32x16_bf8_fp8` | E5M2 (bf8) | E4M3 (fp8) |
| `v_mfma_f32_32x32x16_bf8_bf8` | E5M2 (bf8) | E5M2 (bf8) |

Each combination is a distinct instruction opcode. The hardware knows each operand's format from the opcode alone. There are also 16x16x32 variants with the same naming convention.

**Operand layout** (32x32x16 variant):
- `src0` (A): 2 VGPRs (8 FP8 elements packed per 32-bit VGPR, 2 VGPRs = 16 elements along K)
- `src1` (B): 2 VGPRs (same packing)
- `src2`/`dst`: 16 VGPRs (32x32 FP32 accumulator tile, distributed across 64 wavefront lanes)

### Instruction family 2: Unified f8f6f4 MFMA (CDNA4 / gfx950 only)

This is a newer, more flexible instruction that supports multiple narrow formats through modifier fields rather than separate opcodes:

```
v_mfma_f32_32x32x64_f8f6f4  dst, src0, src1, src2  [cbsz:N] [blgp:M]
```

The `cbsz` and `blgp` fields are 3-bit modifier fields encoded in the instruction word:

- **`cbsz`**: Specifies the data format of **src0 (operand A)**
- **`blgp`**: Specifies the data format of **src1 (operand B)**

| Value | Format | Mantissa/Exponent | Bits | VGPRs for src (32x32x64) |
|---|---|---|---|---|
| 0 | FP8 (E4M3) | 4-bit exponent, 3-bit mantissa | 8 | 8 VGPRs |
| 1 | BF8 (E5M2) | 5-bit exponent, 2-bit mantissa | 8 | 8 VGPRs |
| 2 | FP6 (E3M2) | 3-bit exponent, 2-bit mantissa | 6 | 6 VGPRs |
| 3 | BF6 (E2M3) | 2-bit exponent, 3-bit mantissa | 6 | 6 VGPRs |
| 4 | FP4 (E2M1) | 2-bit exponent, 1-bit mantissa | 4 | 4 VGPRs |

When `cbsz` or `blgp` is 0 (the default), the modifier is omitted from the disassembly. So:
- `v_mfma_f32_32x32x64_f8f6f4 ... blgp:1` means: src0=FP8/E4M3 (default 0), src1=BF8/E5M2 (1)
- `v_mfma_f32_32x32x64_f8f6f4 ... cbsz:1` means: src0=BF8/E5M2 (1), src1=FP8/E4M3 (default 0)
- No modifier at all means: both src0 and src1 are FP8/E4M3 (both default 0)

The key difference from the named-format family: the f8f6f4 instruction uses a **single opcode** with modifier fields, whereas the named-format family uses **separate opcodes** per format combination. The number of src VGPRs varies by format (8 for 8-bit, 6 for 6-bit, 4 for 4-bit), which the hardware determines from cbsz/blgp.

There is also a 16x16x128 variant (`v_mfma_f32_16x16x128_f8f6f4`) with the same modifier encoding.

### How Triton maps types to instructions

In `MFMA.cpp`, the `getMfmaF8F6F4MatrixFormat()` function maps MLIR types to the integer values:

```cpp
static inline int32_t getMfmaF8F6F4MatrixFormat(Type t) {
  return llvm::TypeSwitch<Type, int32_t>(t)
      .Case<Float8E4M3FNType>([](Type) { return 0; })   // E4M3 -> cbsz/blgp = 0
      .Case<Float8E5M2Type>([](Type) { return 1; })      // E5M2 -> cbsz/blgp = 1
      .Case<Float6E3M2FNType>([](Type) { return 2; })    // E3M2 -> cbsz/blgp = 2
      .Case<Float6E2M3FNType>([](Type) { return 3; })    // E2M3 -> cbsz/blgp = 3
      .Case<Float4E2M1FNType>([](Type) { return 4; })    // E2M1 -> cbsz/blgp = 4
      .Default([](Type) { return -1; });
}
```

These are then assigned in `generateScaledMFMAOp()`:

```cpp
int32_t cbsz = getMfmaF8F6F4MatrixFormat(elemTypeA);  // format for src0
int32_t blgp = getMfmaF8F6F4MatrixFormat(elemTypeB);  // format for src1
```

This mapping is confirmed by the LLVM backend in `SIDefines.h` (`MFMAScaleFormats` enum) and the AMD CDNA4 ISA Reference Guide.

### When Triton selects each family

On gfx950, Triton's `AccelerateAMDMatmul.cpp` pass decides which instruction family to use:

1. For `tl.dot()` with FP8 operands, it first tries to use the f8f6f4 scaled instruction (family 2) via `BlockedToMFMA` with `withScale=true`. This succeeds when `BLOCK_SIZE_K >= 64` (the minimum K dimension for the 32x32x64 instruction).

2. If that fails (e.g., `BLOCK_SIZE_K < 64`), it falls back to the named-format instruction (family 1) with `BLOCK_SIZE_K >= 16`.

3. For `tl.dot_scaled()` (used by MXFP8), the `ScaledBlockedToScaledMFMAF8F6F4` pattern always selects family 2.

Both families require correct format-to-operand assignment to produce correct results.

### Source references

- AMD Instinct CDNA4 ISA Reference Guide (Section 12: Matrix Fused Multiply-Add)
- LLVM: `llvm/lib/Target/AMDGPU/SIDefines.h` -- `MFMAScaleFormats` enum
- LLVM: `llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp` -- `getMFMA_F8F6F4_WithFormatArgs()`
- Triton: `third_party/amd/lib/TritonAMDGPUToLLVM/DotOpToLLVM/MFMA.cpp` -- `getMfmaF8F6F4MatrixFormat()`
- Triton: `third_party/amd/lib/TritonAMDGPUTransforms/AccelerateAMDMatmul.cpp` -- instruction family selection
- Triton: `third_party/amd/lib/TritonAMDGPUTransforms/MfmaGroup.cpp` -- MFMA intrinsic database

---

## 4. Assembly-Level Verification

### Dumping GPU assembly

We used AMD's assembly dump mechanism to inspect the generated instructions:

```bash
AMDGCN_ENABLE_DUMP=1 python3 -c "
import os
os.environ['NVTE_USE_GEMM_TRITON'] = '1'
# ... set up FP8 tensors and call general_gemm ...
" 2>&1 > /tmp/mixed_fp8_dump.txt
```

Three test cases were run with fresh Triton caches (to force recompilation):

1. **Same format**: A=e4m3, B=e4m3
2. **Mixed**: A=e4m3, B=e5m2
3. **Mixed reversed**: A=e5m2, B=e4m3

### Extracting MFMA instructions

```bash
grep "v_mfma" /tmp/mixed_fp8_dump.txt | sort | uniq -c | sort -rn
```

### Results

| Case | 32x32x16 variant | 32x32x64 f8f6f4 modifier |
|---|---|---|
| e4m3 x e4m3 (same) | `fp8_fp8` (72 instances) | no modifier (48 instances) |
| A=e4m3, B=e5m2 | `bf8_fp8` (72 instances) | `blgp:1` (48 instances) |
| A=e5m2, B=e4m3 | `fp8_bf8` (72 instances) | `cbsz:1` (48 instances) |

### Analysis

For user A=e4m3, B=e5m2 with TN layout, the BLAS-to-Triton operand swap produces:
- Triton operand A (src0) = original B data = **e5m2 (bf8)**
- Triton operand B (src1) = original A data = **e4m3 (fp8)**

#### Named-format variant (32x32x16)

The assembly shows `v_mfma_f32_32x32x16_bf8_fp8`. Per the instruction spec (Section 3):
- `bf8` (first suffix) = src0 format = **e5m2**
- `fp8` (second suffix) = src1 format = **e4m3**

This **matches** the actual data in registers after the swap. The named-format variant appears correct.

#### Unified f8f6f4 variant (32x32x64)

The assembly shows `v_mfma_f32_32x32x64_f8f6f4 ... blgp:1`. Per the modifier spec (Section 3):
- `cbsz` (absent = 0) = src0 format = **FP8/E4M3**
- `blgp:1` = src1 format = **BF8/E5M2**

But after the swap, the actual data is:
- src0 contains **e5m2** data (needs `cbsz:1`)
- src1 contains **e4m3** data (needs `blgp:0`, the default)

**Expected encoding**: `cbsz:1` (src0 = e5m2), no `blgp` modifier (src1 = e4m3)
**Actual encoding**: `blgp:1` (src1 = e5m2), no `cbsz` modifier (src0 = e4m3)

The format tags are assigned to the **opposite** operands. The hardware reads src0 as E4M3 but the register contains E5M2 data, and vice versa for src1. Since E4M3 and E5M2 have different exponent/mantissa bit partitioning (4+3 vs 5+2), the numerical interpretation is completely wrong, explaining the near-100% mismatch.

#### Cross-checking the reversed case

For user A=e5m2, B=e4m3 (the reversed mixed case), after swap:
- src0 = original B = **e4m3 (fp8)**
- src1 = original A = **e5m2 (bf8)**

Expected: no modifier (src0 = e4m3, default), `blgp:1` (src1 = e5m2)
Actual: `cbsz:1` (src0 = e5m2) -- again the **opposite** of what the data requires.

Both mixed cases show the same pattern: the f8f6f4 modifier assigns the format of the **pre-swap** operands, not the post-swap ones.

---

## 5. Identifying the Triton Compiler Bug

### Examining Triton source

We fetched the MFMA lowering code from `triton-lang/triton`:

```
third_party/amd/lib/TritonAMDGPUToLLVM/DotOpToLLVM/MFMA.cpp
```

The format mapping function:

```cpp
static inline int32_t getMfmaF8F6F4MatrixFormat(Type t) {
  return llvm::TypeSwitch<Type, int32_t>(t)
      .Case<Float8E4M3FNType>([](Type) { return 0; })   // fp8
      .Case<Float8E5M2Type>([](Type) { return 1; })      // bf8
      ...
}
```

In `generateScaledMFMAOp`:

```cpp
int32_t cbsz = getMfmaF8F6F4MatrixFormat(elemTypeA);  // src A format
int32_t blgp = getMfmaF8F6F4MatrixFormat(elemTypeB);  // src B format
```

### The two bugs

**Bug 1: Intrinsic selection uses pre-swap types.** In `convertDot`, when `mfmaLayout.getIsTransposed()` is true, operands A and B are physically swapped before the MFMA intrinsic call. The code swaps the intrinsic element types for the named variant:

```cpp
if (mfmaLayout.getIsTransposed() && elemTyA != elemTyB) {
    std::swap(intrinsicElemTyA, intrinsicElemTyB);
}
```

But the `cbsz`/`blgp` assignment in `generateScaledMFMAOp` uses the original `elemTypeA`/`elemTypeB`, not the swapped types.

**Bug 2: Operand B packing uses A's type.** `operandB` is extracted using `aTensorTy.getElementType()` instead of `bTensorTy.getElementType()`:

```cpp
// BUGGY:
auto operandB = getValuesFromDotOperandLayoutStruct(
    loadedB, ..., aTensorTy.getElementType(), ...);
```

This means when types differ, B's data is loaded/converted using A's format, corrupting the data before it reaches the MFMA instruction. This is the fundamental bug -- it affects **all** instruction variants, not just the f8f6f4 one.

### Why same-format works

When both operands have the same FP8 type, `aTensorTy.getElementType() == bTensorTy.getElementType()`, so bug #2 is invisible. And `cbsz == blgp`, so bug #1 doesn't matter either. The symmetry masks both bugs.

---

## 6. Confirming the Upstream Fix

### Checking the installed Triton version

```bash
pip list | grep triton
# pytorch-triton-rocm  3.5.1+gitbfeb0668
# triton                3.4.0
```

The `pytorch-triton-rocm` package is built from Triton commit `bfeb0668` (Nov 4, 2025).

### Finding the fix

We searched the upstream `triton-lang/triton` repository and found **PR #9567**: "[AMD][BACKEND] Fix mixed types MFMA fp8 instruction selection", merged **February 27, 2026** (commit `eaaa75cf5`).

The fix addresses both bugs:

```cpp
// Fix 1: Swap intrinsic element types when transposed + mixed
Type intrinsicElemTyA = elemTyA;
Type intrinsicElemTyB = elemTyB;
if (mfmaLayout.getIsTransposed() && elemTyA != elemTyB) {
    std::swap(intrinsicElemTyA, intrinsicElemTyB);
}

// Fix 2: Use correct type for operandB
auto operandB = getValuesFromDotOperandLayoutStruct(
    loadedB, ..., bTensorTy.getElementType(), ...);  // was aTensorTy
```

### Checking PyTorch version availability

We checked the Triton commit pins across all PyTorch release branches:

| PyTorch Branch | Triton Pin Date | Includes Fix? |
|---|---|---|
| release/2.9 | Nov 2025 | No |
| release/2.10 | Dec 2025 | No |
| release/2.11 | Dec 2025 | No |
| main | Dec 2025 | No |

**No existing PyTorch release includes the fix.** Expected in PyTorch 2.12+.

---

## 7. Attempted Workaround: Restricting MFMA Variant

### Hypothesis

On gfx950, Triton auto-promotes `tl.dot()` with FP8 operands to the `v_mfma_f32_32x32x64_f8f6f4` scaled instruction when `BLOCK_SIZE_K >= 64`. We tried restricting `BLOCK_SIZE_K < 64` in the autotuner configs to force the older `v_mfma_f32_32x32x16` variant.

### Result

After restricting BLOCK_SIZE_K, the assembly confirmed only `v_mfma_f32_32x32x16_bf8_fp8` instructions were generated (no f8f6f4). However, **the tests still failed with 99.8% mismatch**.

This confirmed that bug #2 (operand B packing using A's element type) affects **both** instruction variants. The data is corrupted before reaching the instruction, so no instruction selection change can fix it.

### Conclusion

There is no workaround at the kernel configuration level. The fix must come from the Triton compiler.

---

## 8. Mitigations Applied

### Runtime guard in `gemm_triton.py`

Added a `ValueError` that catches mixed FP8 types before they reach `tl.dot()`:

```python
if (a_fp8_dtype is not None and b_fp8_dtype is not None
        and a_fp8_dtype != b_fp8_dtype):
    raise ValueError(
        "Mixed FP8 types are not supported in the Triton GEMM backend "
        "due to a Triton compiler bug (triton-lang/triton#9567)."
    )
```

### Recipe-level guard in `quantization.py`

Added a check in `check_recipe_support()` (called at `autocast()` entry) that rejects `Format.HYBRID` when `NVTE_USE_GEMM_TRITON=1`:

```python
if use_gemm_triton and recipe.fp8_format == Format.HYBRID:
    raise ValueError(
        "The Triton GEMM backend does not support Format.HYBRID because "
        "the backward pass produces mixed FP8 type GEMMs (e5m2 x e4m3)."
    )
```

This gives users a clear error at training setup time rather than a crash during backward pass. Users can switch to `Format.E4M3` (which uses e4m3 for both forward and backward) or disable the Triton backend.

### Test updates

Mixed FP8 tests changed from `xfail` to `skip` with documentation referencing the upstream fix.

---

## 9. Key References

| Reference | Description |
|---|---|
| [triton-lang/triton PR #9567](https://github.com/triton-lang/triton/pull/9567) | Upstream fix: "Fix mixed types MFMA fp8 instruction selection" |
| Commit `eaaa75cf5daf498a4aab29f135fa07977e318fb6` | Fix commit in Triton (merged 2026-02-27) |
| `MFMA.cpp` in `third_party/amd/lib/TritonAMDGPUToLLVM/DotOpToLLVM/` | Triton's MFMA instruction lowering |
| `gemm_triton.py` lines 564-569, 655-659 | TE's FP8 reinterpretation and operand swap |
| `quantization.py` `check_recipe_support()` | TE's recipe validation entry point |
| AMD CDNA4 ISA Reference | `cbsz`/`blgp` field encoding for `v_mfma_f32_32x32x64_f8f6f4` |
| `AMDGCN_ENABLE_DUMP=1` env var | Triggers GPU assembly dump for Triton kernels |

---

## 10. Debugging Timeline

| Step | Action | Tool/Command |
|---|---|---|
| 1 | Created parametrized test suite covering same and mixed FP8 formats | Write test file |
| 2 | Observed 99% mismatch for mixed FP8 only | `pytest -v` |
| 3 | Confirmed C++ backend handles mixed FP8 correctly | Triton vs C++ comparison tests |
| 4 | Hypothesized operand swap + format encoding mismatch | Code reading of `gemm_triton.py` |
| 5 | Dumped GPU assembly with `AMDGCN_ENABLE_DUMP=1` | `grep v_mfma` on dump files |
| 6 | Identified `blgp:1` on wrong operand in f8f6f4 instruction | Assembly analysis |
| 7 | Confirmed `cbsz`/`blgp` mapping from LLVM and Triton source | WebFetch of `MFMA.cpp` |
| 8 | Found upstream fix PR #9567 with exact two-bug diagnosis | Web search + GitHub API |
| 9 | Attempted BLOCK_SIZE_K restriction workaround | Modified autotuner configs |
| 10 | Confirmed workaround insufficient (bug #2 affects all variants) | Re-ran tests + assembly check |
| 11 | Checked all PyTorch release branches for fix availability | WebFetch of triton pin files |
| 12 | Applied runtime guards and documentation | Edited `gemm_triton.py`, `quantization.py` |
