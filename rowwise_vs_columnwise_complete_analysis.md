# Complete Analysis: Rowwise vs Columnwise in Transformer Engine

**Reference:** ROCm TransformerEngine commit `f141f34bff6cd775dd113ee5a96f66c9d0a44fc8`

---

## Table of Contents
1. [The Core Concept](#the-core-concept)
2. [The Hardware Restriction](#the-hardware-restriction)
3. [The Selection Logic](#the-selection-logic)
4. [Why It Works: The Mathematics](#why-it-works-the-mathematics)
5. [Mapping to Linear Layer Usage](#mapping-to-linear-layer-usage)

---

## The Core Concept

### Important: Different Meanings for Different Tensor Types

The terms "rowwise" and "columnwise" have **fundamentally different meanings** depending on the tensor type:

#### For Standard Float8Tensor (Per-tensor Scaling)

1. **Rowwise data** (`_data` attribute):
   - Normal PyTorch row-major memory layout
   - Shape: `[*batch_dims, M, K]` for a matrix
   - Used by default in PyTorch

2. **Columnwise data** (`_transpose` attribute):
   - **Actually transposed** memory layout
   - Shape: `[K, M, *batch_dims]` for the same matrix
   - Relationship: `columnwise_shape = (rowwise_shape[-1],) + rowwise_shape[:-1]`

**Key insight:** For Float8Tensor, columnwise IS the transpose of rowwise (different memory layout).

#### For MXFP8Tensor (Block Scaling)

1. **Rowwise data** (`_data` attribute):
   - Shape: `[M, K]` (same as Float8Tensor)
   - **Horizontal block scaling**: Each row divided into K/32 blocks
   - Scales stored as `[M, K/32]`

2. **Columnwise data** (`_columnwise_data` attribute):
   - Shape: `[M, K]` (**NOT transposed!**)
   - **Vertical block scaling**: Each column divided into M/32 blocks
   - Scales stored as `[M/32, K]`

**Key insight:** For MXFP8Tensor, rowwise and columnwise have the SAME shape but different quantization patterns.

### Visual Representation

#### Standard Float8Tensor (Transpose-based)
```
Matrix A:
┌─────────────┐
│ A in memory │ ← rowwise format (normal PyTorch layout)
│ [M, K]      │
└─────────────┘
       │
       │ store transpose
       ▼
┌─────────────┐
│ A^T in mem  │ ← columnwise format (transposed layout)
│ [K, M]      │
└─────────────┘
```

#### MXFP8Tensor (Same shape, different scaling)
```
Matrix A:
┌─────────────────────────┐
│ A rowwise [M, K]        │ ← Horizontal blocks (K/32 per row)
│ [━━|━━|━━|...|━━]      │
│ [━━|━━|━━|...|━━]      │
└─────────────────────────┘
       │
       │ different quantization
       ▼
┌─────────────────────────┐
│ A columnwise [M, K]     │ ← Vertical blocks (M/32 per column)
│ [↓ ↓ ↓ ... ↓]          │    (SAME SHAPE!)
│ [↓ ↓ ↓ ... ↓]          │
└─────────────────────────┘
```

### Examples

#### Standard Float8Tensor Example
```python
# Rowwise format
rowwise_data.shape = [2, 2048, 14336]  # [batch, M, K]

# Columnwise format (transposed!)
columnwise_data.shape = [14336, 2048, 2]  # [K, M, batch]

# Transformation for Float8Tensor:
# Last dim → first, everything else shifts right
# [2, 2048, 14336] → [14336, 2048, 2] (transposed storage)
```

#### MXFP8Tensor Example
```python
# Rowwise format
rowwise_data.shape = [2, 2048, 14336]  # [batch, M, K]
rowwise_scales.shape = [2, 2048, 448]  # 14336/32 = 448 blocks per row

# Columnwise format (NOT transposed!)
columnwise_data.shape = [2, 2048, 14336]  # SAME SHAPE!
columnwise_scales.shape = [2, 64, 14336]  # 2048/32 = 64 blocks per column

# Both have same memory layout but different quantization patterns
```

---

## The Hardware Restriction

### Hopper GPUs Only Support TN Layout for FP8

From `transformer_engine/common/gemm/cublaslt_gemm.cu` (line 117):
> "Hopper only supports TN GEMMs for FP8. 'Column-wise data' is transpose of data."

**Problem:** Three GEMM layouts exist (TN, NN, NT), but Hopper FP8 hardware only supports TN.

**Solution:** Store matrices in both formats and select the right one to convert any layout to TN.

### Why Both Formats Exist

**Memory cost:** 2× storage (both rowwise and columnwise)

**Compute benefit:**
- Zero-cost layout conversion (just swap pointers)
- No actual transpose operations needed
- Can use optimized FP8 Tensor Cores

**Optimization:** The Linear module only creates format(s) actually needed:
- Forward only → Only TN layout formats
- With backward → Additional formats for NN/NT layouts
- With activation recomputation → Different formats per pass

---

## The Selection Logic

### For Standard Float8Tensor (Per-tensor Scaling)

For Hopper FP8 GEMMs, the C++ backend converts ALL layouts to TN:

| Layout | A should be? | B should be? | Solution |
|--------|--------------|--------------|----------|
| **TN** | Transposed   | Not transposed | Use rowwise for both (already TN) |
| **NN** | Not transposed | Not transposed | Use A's columnwise (convert to TN) |
| **NT** | Not transposed | Transposed | Use both columnwise (convert to TN) |

**The Float8Tensor rule:**
- Want transpose in GEMM? → Use **rowwise** data
- Don't want transpose in GEMM? → Use **columnwise** data (it IS the transpose)

### For MXFP8Tensor (Block Scaling)

MXFP8 has completely different selection logic based on accumulation dimension:

| Layout | A accumulates along | B accumulates along | A uses | B uses |
|--------|-------------------|-------------------|---------|---------|
| **TN** | K dimension | K dimension | rowwise | rowwise |
| **NN** | M dimension | K dimension | columnwise | rowwise |
| **NT** | Batch dimension | Batch dimension | columnwise | columnwise |

**The MXFP8 rule:**
- Accumulating along K? → Use **rowwise** (horizontal blocks)
- Accumulating along M or batch? → Use **columnwise** (vertical blocks)

### Code Location

**Files (contain both Float8Tensor and MXFP8 logic):**
- `transformer_engine/common/gemm/cublaslt_gemm.cu` (NVIDIA cuBLAS)
- `transformer_engine/common/gemm/rocm_gemm.cu` (AMD hipBLASLt)

**Function:** `CanonicalizeGemmInput()`
- Lines 90-229 in cublaslt_gemm.cu
- Lines 200-291 in rocm_gemm.cu

The function handles three different scaling modes:
1. **Tensor scaling** (standard Float8Tensor) - lines 108-127, 167-186
2. **MXFP8 scaling** - lines 234-250, 272-285
3. **Block scaling** (Float8BlockScaling) - lines 142-165, 201-223

### Standard Float8Tensor Selection Logic

#### For Operand A (lines 108-127)

```cpp
// Default: use rowwise data
ret.A = A.data.dptr;              // rowwise
ret.transA = transA;              // As requested
ret.Atype = A.data.dtype;
ret.A_scale_inv = A.scale_inv.dptr;
ret.lda = is_A_transposed ? k : m;

// Special case for Hopper FP8 when A should NOT be transposed
if (!nvte_is_non_tn_fp8_gemm_supported() && !is_A_transposed) {
    // Convert NN/NT layouts to TN
    if (A.has_columnwise_data() && is_fp8_dtype(A.columnwise_data.dtype)) {
        ret.A = A.columnwise_data.dptr;    // Use transpose
        ret.transA = CUBLAS_OP_T;          // Force transpose flag
        ret.Atype = A.columnwise_data.dtype;
        ret.A_scale_inv = A.columnwise_scale_inv.dptr;
        ret.lda = k;
    }
}
```

**Float8Tensor selection for A:**
- If `transA = T` → Use **rowwise** [M,K], keep T flag
- If `transA = N` → Use **columnwise** [K,M] (transposed), change to T flag

#### For Operand B (lines 167-186)

```cpp
// Default: use rowwise data
ret.B = B.data.dptr;              // rowwise
ret.transB = transB;              // As requested
ret.Btype = B.data.dtype;
ret.B_scale_inv = B.scale_inv.dptr;
ret.ldb = is_B_transposed ? n : k;

// Special case for Hopper FP8 when B SHOULD be transposed
if (!nvte_is_non_tn_fp8_gemm_supported() && is_B_transposed) {
    // Convert NT/TT layouts to TN
    if (B.has_columnwise_data() && is_fp8_dtype(B.columnwise_data.dtype)) {
        ret.B = B.columnwise_data.dptr;    // Use transpose
        ret.transB = CUBLAS_OP_N;          // Force non-transpose flag
        ret.Btype = B.columnwise_data.dtype;
        ret.B_scale_inv = B.columnwise_scale_inv.dptr;
        ret.ldb = k;
    }
}
```

**Float8Tensor selection for B:**
- If `transB = N` → Use **rowwise** [M,K], keep N flag
- If `transB = T` → Use **columnwise** [K,M] (transposed), change to N flag

### Layout Conversion Table

| Requested Layout | transA | transB | A uses | B uses | Actual cuBLAS Call | Result |
|------------------|--------|--------|--------|--------|-------------------|--------|
| **TN** | T | N | rowwise | rowwise | T(rowwise) × N(rowwise) | T × N ✓ |
| **NN** | N | N | columnwise | rowwise | T(columnwise) × N(rowwise) | T × N ✓ |
| **NT** | N | T | columnwise | columnwise | T(columnwise) × N(columnwise) | T × N ✓ |
| **TT** | T | T | rowwise | columnwise | T(rowwise) × N(columnwise) | T × N ✓ |

**All FP8 GEMM layouts become TN for Hopper!**

---

## Why It Works: The Mathematics

### Key Property

Since columnwise data IS the transpose of rowwise data, applying BLAS transpose operations gives:

```
Let A_r = matrix in rowwise format
Let A_c = matrix in columnwise format = transpose(A_r)

cuBLAS operations (column-major convention):
  CUBLAS_OP_N(A_r) = A
  CUBLAS_OP_T(A_r) = A^T
  CUBLAS_OP_N(A_c) = A^T    (because A_c already IS the transpose)
  CUBLAS_OP_T(A_c) = A      (transpose of transpose cancels)
```

### Example 1: Forward Pass (TN layout)

**Requested:** `weight.T @ input`
- A = weight, transA = True
- B = input, transB = False

**Code executes:**
```cpp
// Lines 111, 170
ret.A = weight.data.dptr;      // rowwise
ret.transA = CUBLAS_OP_T;
ret.B = input.data.dptr;       // rowwise
ret.transB = CUBLAS_OP_N;
```

**BLAS call:** `T(weight_rowwise) × N(input_rowwise)` = weight.T @ input ✓

### Example 2: Backward dgrad (NN layout)

**Requested:** `weight @ grad_output`
- A = weight, transA = False
- B = grad_output, transB = False

**Code executes:**
```cpp
// Lines 119, 170
ret.A = weight.columnwise_data.dptr;  // columnwise (transpose)
ret.transA = CUBLAS_OP_T;             // Force T
ret.B = grad_output.data.dptr;        // rowwise
ret.transB = CUBLAS_OP_N;
```

**BLAS call:** `T(weight_columnwise) × N(grad_output_rowwise)`
- = `T(transpose(weight)) × grad_output`
- = `weight × grad_output` ✓

### Example 3: Backward wgrad (NT layout)

**Requested:** `input @ grad_output.T`
- A = input, transA = False
- B = grad_output, transB = True

**Code executes:**
```cpp
// Lines 119, 178
ret.A = input.columnwise_data.dptr;        // columnwise (transpose)
ret.transA = CUBLAS_OP_T;                  // Force T
ret.B = grad_output.columnwise_data.dptr;  // columnwise (transpose)
ret.transB = CUBLAS_OP_N;                  // Force N
```

**BLAS call:** `T(input_columnwise) × N(grad_output_columnwise)`
- = `T(transpose(input)) × transpose(grad_output)`
- = `input × grad_output.T` ✓

---

## Additional Scaling Modes

The code also handles two other FP8 scaling modes beyond standard tensor scaling:

### MXFP8 Scaling - Comprehensive Analysis

#### What is MXFP8?

MXFP8 (Microscaling FP8) is a block-wise scaling format that differs fundamentally from standard per-tensor FP8 scaling. From `transformer_engine/common/recipe/__init__.py:252-274`:

```python
@dataclass()
class MXFP8BlockScaling(Recipe):
    """
    Use the MXFP8 scaling factor strategy.

    In this strategy, tensors are scaled in blockwise fashion. Each group
    of 32 consecutive values is scaled together using their own scaling
    factor. The type of the scaling factor is E8M0 (8 bits of exponent,
    0 bits of mantissa), equivalent to scaling by a power of 2.

    Since the scaling happens in a particular direction (either rowwise
    or columnwise), in this recipe the quantized tensor and its transpose
    are not numerically equivalent.
    """
```

**Core MXFP8 characteristics:**
1. **Block size**: 32 consecutive elements per block
2. **Scale format**: E8M0 (8-bit exponent only, power of 2)
3. **Direction-dependent**: Scaling happens along specific dimensions
4. **Non-equivalent transpose**: A tensor and its transpose have different quantizations

#### MXFP8 Memory Layout

From `transformer_engine/pytorch/tensor/mxfp8_tensor.py:113-130`:

**Critical difference from standard FP8:** For MXFP8, both rowwise and columnwise data have the **SAME shape** `[M, K]` but with different scaling patterns.

```python
# For a matrix with shape [M, K]:

# Rowwise data and scales
data = torch.empty(shape, dtype=torch.uint8)  # Shape: [M, K]
scale_inv = torch.zeros([M, K//32])  # Scales along K dimension

# Columnwise data and scales
columnwise_data = torch.empty_like(data)  # SAME SHAPE: [M, K]
columnwise_scale_inv = torch.zeros([M//32, K])  # Scales along M dimension
```

**Rowwise MXFP8:**
```
Matrix [M, K] with rowwise scaling:
[━━━━━━━━━━━━━━━━━━━━━━] row 0: K elements → K/32 blocks
[━━━━━━━━━━━━━━━━━━━━━━] row 1: K elements → K/32 blocks
        ...
[━━━━━━━━━━━━━━━━━━━━━━] row M-1: K elements → K/32 blocks

Storage: data[M, K], scales[M, K/32]
Each row is independently scaled in blocks of 32 along K dimension
```

**Columnwise MXFP8:**
```
Matrix [M, K] with columnwise scaling:
↓ ↓ ↓ ↓ ... ↓
c c c c ... c
o o o o ... o  Each column: M elements → M/32 blocks
l l l l ... l
0 1 2 3 ... K-1

Storage: data[M, K] (NOT transposed!), scales[M/32, K]
Each column is independently scaled in blocks of 32 along M dimension
```

**Key insight:** MXFP8 "rowwise" and "columnwise" refer to the **direction of block scaling**, not the memory layout. Both have the same shape but different quantization patterns.

### MXFP8 Selection Logic (Different Rules!)

#### MXFP8 Uses Accumulation-Based Selection (lines 234-285 in rocm_gemm.cu)

**Fundamental difference:** MXFP8 rowwise and columnwise have the **same shape** but different scaling patterns.

```cpp
// From transformer_engine/common/gemm/rocm_gemm.cu:234-285
if (is_mxfp_scaling(A.scaling_mode)) {
    // MXFP8 selection for A based on accumulation dimension
    if (is_A_transposed) {
        // Will accumulate along K → use rowwise (horizontal blocks)
        ret.A = A.data.dptr;  // Shape [M, K], scales [M, K/32]
        ret.A_scale_inv = A.scale_inv.dptr;
    } else {
        // Will accumulate along M → use columnwise (vertical blocks)
        ret.A = A.columnwise_data.dptr;  // ALSO shape [M, K]!, scales [M/32, K]
        ret.A_scale_inv = A.columnwise_scale_inv.dptr;
    }
    ret.transA = transA;  // Keep original transpose flag!
}

if (is_mxfp_scaling(B.scaling_mode)) {
    // MXFP8 selection for B based on accumulation dimension
    if (is_B_transposed) {
        // Will accumulate along batch/other → use columnwise
        ret.B = B.columnwise_data.dptr;  // Shape [M, K], scales [M/32, K]
        ret.B_scale_inv = B.columnwise_scale_inv.dptr;
    } else {
        // Will accumulate along K → use rowwise
        ret.B = B.data.dptr;  // Shape [M, K], scales [M, K/32]
        ret.B_scale_inv = B.scale_inv.dptr;
    }
    ret.transB = transB;  // Keep original transpose flag!
}
```

**Key MXFP8 differences:**
1. Both formats have **same shape** `[M, K]` (no transpose!)
2. Rowwise: Horizontal 32-element blocks (scales along K)
3. Columnwise: Vertical 32-element blocks (scales along M)
4. Selection based on which dimension is accumulated in dot products
5. Keeps original transpose flags (doesn't convert to TN)

#### Complete MXFP8 GEMM Examples

Let's trace through all three GEMM operations with concrete dimensions:
- Weight: `W[1024, 768]` (out_features=1024, in_features=768)
- Input: `X[batch, 768]`
- Output gradient: `dY[batch, 1024]`

##### 1. Forward Pass (fprop): Y = X @ W^T (row-major view)

**What we want (row-major):** `Y[batch, out] = X[batch, in] @ W^T[in, out]`

**What BLAS sees (column-major):** Our row-major matrices appear transposed:
- Row-major `X[batch, 768]` → BLAS sees `X^T[768, batch]`
- Row-major `W[1024, 768]` → BLAS sees `W^T[768, 1024]`
- Row-major `Y[batch, 1024]` → BLAS sees `Y^T[1024, batch]`

**BLAS computation with layout="TN":**
```python
# Code: general_gemm(W, X, layout="TN")
# BLAS computes: C = op(A) @ op(B)
# With TN: C = A^T @ B
# So: Y^T = W^T^T @ X^T = W @ X^T
# Which equals: (X @ W^T)^T in row-major
# Result when read as row-major: Y = X @ W^T ✓
```

**MXFP8 Selection:**
- **Weight (W)**: `transA = T` → Uses **rowwise** MXFP8
  - Data shape: `[1024, 768]` (not transposed)
  - Scales shape: `[1024, 24]` (768/32 = 24 blocks per row)
  - Scaling pattern: Horizontal blocks along in_features dimension
- **Input (X)**: `transB = N` → Uses **rowwise** MXFP8
  - Data shape: `[batch, 768]`
  - Scales shape: `[batch, 24]` (768/32 = 24 blocks per row)
  - Scaling pattern: Horizontal blocks along in_features dimension

**Why:** Both accumulate along in_features (768) dimension

##### 2. Backward dgrad: dX = dY @ W (row-major view)

**What we want (row-major):** `dX[batch, in] = dY[batch, out] @ W[out, in]`

**BLAS computation with layout="NN":**
```python
# Code: general_gemm(W, dY, layout="NN")
# BLAS computes: C = A @ B (no transposes)
# So: dX^T = W^T @ dY^T
# Which equals: (dY @ W)^T in row-major
# Result when read as row-major: dX = dY @ W ✓
```

**MXFP8 Selection:**
- **Weight (W)**: `transA = N` → Uses **columnwise** MXFP8
  - Data shape: `[1024, 768]` (SAME shape, not transposed!)
  - Scales shape: `[32, 768]` (1024/32 = 32 blocks per column)
  - Scaling pattern: Vertical blocks along out_features dimension
- **Grad output (dY)**: `transB = N` → Uses **rowwise** MXFP8
  - Data shape: `[batch, 1024]`
  - Scales shape: `[batch, 32]` (1024/32 = 32 blocks per row)
  - Scaling pattern: Horizontal blocks along out_features dimension

**Why:** Both accumulate along out_features (1024) dimension

##### 3. Backward wgrad: dW = dY^T @ X (row-major view)

**What we want (row-major):** `dW[out, in] = dY^T[out, batch] @ X[batch, in]`

**BLAS computation with layout="NT":**
```python
# Code: general_gemm(X, dY, layout="NT")
# BLAS computes: C = A @ B^T
# So: dW^T = X^T @ dY^T^T = X^T @ dY
# Which equals: (dY^T @ X)^T in row-major
# Result when read as row-major: dW = dY^T @ X ✓
```

**MXFP8 Selection:**
- **Input (X)**: `transA = N` → Uses **columnwise** MXFP8
  - Data shape: `[batch, 768]` (NOT transposed!)
  - Scales shape: `[batch/32, 768]` (batch/32 blocks per column)
  - Scaling pattern: Vertical blocks along batch dimension
- **Grad output (dY)**: `transB = T` → Uses **columnwise** MXFP8
  - Data shape: `[batch, 1024]` (NOT transposed!)
  - Scales shape: `[batch/32, 1024]` (batch/32 blocks per column)
  - Scaling pattern: Vertical blocks along batch dimension

**Why:** Both accumulate along batch dimension

#### MXFP8 Selection Summary

| Pass | Row-Major Formula | BLAS Sees | MXFP8 Format | Scaling Along |
|------|------------------|-----------|--------------|---------------|
| **fprop** | Y = X @ W^T | Y^T = W @ X^T | | |
| | X[batch, in] | X^T[in, batch] | rowwise | in dimension |
| | W[out, in] | W^T[in, out] | rowwise | in dimension |
| **dgrad** | dX = dY @ W | dX^T = W^T @ dY^T | | |
| | W[out, in] | W^T[in, out] | **columnwise** | out dimension |
| | dY[batch, out] | dY^T[out, batch] | rowwise | out dimension |
| **wgrad** | dW = dY^T @ X | dW^T = X^T @ dY | | |
| | X[batch, in] | X^T[in, batch] | **columnwise** | batch dimension |
| | dY[batch, out] | dY^T[out, batch] | **columnwise** | batch dimension |

**The Key MXFP8 Insight:** MXFP8 selects formats based on which dimension is accumulated in the dot products:
1. **fprop**: Accumulates along `in_features` → both use rowwise
2. **dgrad**: Accumulates along `out_features` → appropriate scaling
3. **wgrad**: Accumulates along `batch` → both use columnwise

### Block Scaling (Float8BlockScaling - Hopper Only)

**Note:** Block Scaling is implemented for NVIDIA Hopper GPUs but not for AMD MI300 (though it could be implemented).

**Similar to standard Float8Tensor but always forces TN layout:**

```cpp
// For A:
if (is_A_transposed) {
    ret.A = A.data.dptr;
} else {
    ret.A = A.columnwise_data.dptr;
}
ret.transA = CUBLAS_OP_T;  // Always transpose
ret.A_scale_inv = is_A_transposed ? A.scale_inv.dptr : A.columnwise_scale_inv.dptr;

// For B:
if (is_B_transposed) {
    ret.B = B.columnwise_data.dptr;
} else {
    ret.B = B.data.dptr;
}
ret.transB = CUBLAS_OP_N;  // Never transpose
ret.B_scale_inv = is_B_transposed ? B.columnwise_scale_inv.dptr : B.scale_inv.dptr;
```

**Key difference:** Block scaling always results in TN layout (transA=T, transB=N) regardless of requested layout.

### Comparison of All Scaling Modes

| Scaling Mode | Platform | Rowwise/Columnwise Meaning | Force TN? | Key Characteristic |
|--------------|----------|---------------------------|-----------|-------------------|
| **Tensor Scaling (Float8Tensor)** | Hopper/MI300 | Columnwise = transposed `[K,M]` | Yes (Hopper only) | Single scale per tensor |
| **MXFP8 (MXFP8Tensor)** | Hopper/MI300 | Both same shape `[M,K]`, different scaling | **No** | E8M0 scales per 32 elements |
| **Block Scaling (Float8BlockScaling)** | **Hopper only** | Columnwise = transposed `[K,M]` | Yes (always) | FP32 scales per block |

**Selection Logic Summary:**

| Tensor Type | A Selection | B Selection | Basis |
|------------|-------------|-------------|--------|
| **Float8Tensor** | rowwise if transA=T, else columnwise | rowwise if transB=N, else columnwise | Avoid transpose ops |
| **MXFP8Tensor** | rowwise if transA=T, else columnwise | columnwise if transB=T, else rowwise | Accumulation dimension |
| **Float8BlockScaling** | rowwise if transA=T, else columnwise | columnwise if transB=T, else rowwise | Always force TN |

**Critical MXFP8 differences:**
- **Rowwise and columnwise have SAME shape `[M, K]`** (unlike standard FP8)
- Rowwise: Scales along K dimension (horizontal blocks)
- Columnwise: Scales along M dimension (vertical blocks)
- Uses E8M0 (power-of-2) scales, fixed 32-element blocks
- Doesn't convert to TN layout
- Both formats are different quantizations, not transposes

---

## Mapping to Linear Layer Usage

From `transformer_engine/pytorch/module/linear.py`:

### Forward Pass (fprop)

**Code (lines 267-271, 233):**
```python
weight_quantizer.set_usage(rowwise=True, columnwise=...)
inputmat.update_usage(rowwise_usage=True)
```

**GEMM:** `general_gemm(weightmat, inputmat_total, ..., layout='TN')`
- Layout: TN (transA=True, transB=False)
- Weight needs: **rowwise** (will be transposed)
- Input needs: **rowwise** (will NOT be transposed)

**Matches BLAS code:** Lines 111, 170 ✓

### Backward dgrad

**Code (lines 681):**
```python
weightmat.update_usage(columnwise_usage=True)
```

**GEMM:** `general_gemm(weight_fp8, grad_output, ..., layout='NN')`
- Layout: NN (transA=False, transB=False)
- Weight needs: **columnwise** (convert NN to TN)
- grad_output needs: **rowwise** (implicit)

**Matches BLAS code:** Lines 119, 170 ✓

### Backward wgrad

**Code (line 371):**
```python
inputmat.update_usage(rowwise_usage=False, columnwise_usage=True)
```

**GEMM:** `general_gemm(x, dy, ..., layout='NT')`
- Layout: NT (transA=False, transB=True)
- Input needs: **columnwise** (convert NT to TN)
- grad_output needs: **columnwise** (convert NT to TN)

**Matches BLAS code:** Lines 119, 178 ✓

### Summary Table

| Pass | GEMM Layout | Operand | Needs Transpose? | Uses | BLAS Code |
|------|-------------|---------|------------------|------|-----------|
| fprop | TN | weight | Yes (transA=T) | rowwise | Line 111 |
| fprop | TN | input | No (transB=N) | rowwise | Line 170 |
| dgrad | NN | weight | No (transA=N) | columnwise | Line 119 |
| dgrad | NN | grad_output | No (transB=N) | rowwise | Line 170 |
| wgrad | NT | input | No (transA=N) | columnwise | Line 119 |
| wgrad | NT | grad_output | Yes (transB=T) | columnwise | Line 178 |

**Pattern:**
- Needs transpose in GEMM → Use **rowwise**
- Doesn't need transpose in GEMM → Use **columnwise**

---

## Summary

### The Complete Picture

1. **Critical distinction - "rowwise" and "columnwise" have different meanings:**
   - **Float8Tensor**: Columnwise = transposed layout `[K,M]` vs rowwise `[M,K]`
   - **MXFP8Tensor**: Both have same shape `[M,K]` but different scaling patterns
   - **Float8BlockScaling**: Same as Float8Tensor (columnwise = transposed)

2. **Hardware restriction:**
   - Hopper/MI300 FP8 GEMMs only support TN layout in BLAS
   - Block Scaling (Float8BlockScaling) is Hopper-only, not implemented for MI300

3. **Standard FP8 (Float8Tensor) solution:**
   - Store both formats: rowwise `[M,K]` and columnwise `[K,M]` (transposed)
   - Select format to avoid transpose operations
   - Convert all layouts to TN via pointer swaps
   - Selection: Need transpose → rowwise, don't need transpose → columnwise

4. **MXFP8 (MXFP8Tensor) solution:**
   - Store both formats with **same shape** `[M,K]` but different quantizations
   - Rowwise: Horizontal 32-element blocks (scales along K)
   - Columnwise: Vertical 32-element blocks (scales along M)
   - Selection based on accumulation dimension:
     - fprop: Both use rowwise (accumulate along in_features)
     - dgrad: Weight columnwise, dY rowwise (accumulate along out_features)
     - wgrad: Both columnwise (accumulate along batch)
   - Doesn't convert to TN layout (keeps original transpose flags)

5. **Why each approach works:**
   - **Float8Tensor**: Columnwise IS the transpose, enables zero-cost layout conversion
   - **MXFP8Tensor**: Each format has optimal scaling for its accumulation pattern
   - **Float8BlockScaling**: Similar to Float8Tensor but always forces TN
   - All approaches trade memory (2× storage) for performance/accuracy

### Verified Code References

- `transformer_engine/common/gemm/cublaslt_gemm.cu`: Lines 90-229
- `transformer_engine/common/gemm/rocm_gemm.cu`: Lines 190-286
- `transformer_engine/pytorch/module/linear.py`: Usage patterns
- `transformer_engine/pytorch/tensor/mxfp8_tensor.py`: MXFP8 tensor implementation
- `transformer_engine/common/recipe/__init__.py`: MXFP8BlockScaling definition

**Commit:** `f141f34bff6cd775dd113ee5a96f66c9d0a44fc8` (ROCm fork)
