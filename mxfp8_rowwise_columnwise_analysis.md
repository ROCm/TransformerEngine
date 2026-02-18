# Complete Analysis: MXFP8 Rowwise vs Columnwise in Transformer Engine

**Reference:** ROCm TransformerEngine commit `f141f34bff6cd775dd113ee5a96f66c9d0a44fc8`

---

## Table of Contents
1. [What is MXFP8?](#what-is-mxfp8)
2. [MXFP8 vs Standard FP8 Scaling](#mxfp8-vs-standard-fp8-scaling)
3. [Understanding Rowwise and Columnwise in MXFP8](#understanding-rowwise-and-columnwise-in-mxfp8)
4. [Row-Major vs Column-Major Context](#row-major-vs-column-major-context)
5. [MXFP8 Selection Logic in GEMM](#mxfp8-selection-logic-in-gemm)
6. [Complete GEMM Examples](#complete-gemm-examples)
7. [Code Implementation Details](#code-implementation-details)
8. [Summary](#summary)

---

## What is MXFP8?

MXFP8 (Microscaling FP8) is a block-wise scaling format that differs fundamentally from standard per-tensor FP8 scaling.

### Key Characteristics

From `transformer_engine/common/recipe/__init__.py:252-274`:
```python
@dataclass()
class MXFP8BlockScaling(Recipe):
    """
    Use the MXFP8 scaling factor strategy.

    In this strategy, tensors are scaled in blockwise fashion. Each group
    of 32 consecutive values is scaled together using their own scaling
    factor. The type of the scaling factor is E8M0 (8 bits of exponent,
    0 bits of mantissa), equivalent to scaling by a power of 2.
    """
```

**Core features:**
1. **Block size**: 32 consecutive elements per block
2. **Scale format**: E8M0 (8-bit exponent only, power of 2)
3. **Direction-dependent**: Scaling happens along specific dimensions
4. **Non-equivalent transpose**: A tensor and its transpose have different quantizations

---

## MXFP8 vs Standard FP8 Scaling

### Standard FP8 (Per-tensor scaling)
- Single scaling factor for entire tensor
- Columnwise is actually transposed: rowwise `[M, K]`, columnwise `[K, M]`
- Can swap between rowwise/columnwise by just changing pointer
- Transpose doesn't change scaling

### MXFP8 (Block scaling)
- Multiple scaling factors (one per 32-element block)
- **Both formats have same shape `[M, K]`** but different scaling patterns
- **Critical difference** (from `recipe/__init__.py:261-267`):
  > "Since the scaling happens in a particular direction (either rowwise or columnwise), in this recipe the quantized tensor and its transpose are not numerically equivalent."
- Must store both versions separately as different quantizations

---

## Understanding Rowwise and Columnwise in MXFP8

### Important: These Terms Refer to Scaling Direction, Not Memory Layout

In TransformerEngine, "rowwise" and "columnwise" for MXFP8 refer to the **direction of block scaling**, not the memory layout. Unlike standard Float8Tensor, **both formats have the same shape**.

### Definitions

For a matrix `[M, K]` in row-major:

**Rowwise MXFP8:**
```
Matrix [M, K] with rowwise scaling:
[━━━━━━━━━━━━━━━━━━━━━━] row 0: K elements → K/32 blocks
[━━━━━━━━━━━━━━━━━━━━━━] row 1: K elements → K/32 blocks
[━━━━━━━━━━━━━━━━━━━━━━] row 2: K elements → K/32 blocks
        ...
[━━━━━━━━━━━━━━━━━━━━━━] row M-1: K elements → K/32 blocks

Storage: data[M, K], scales[M, K/32]
Each row is independently scaled in blocks of 32 along the K dimension
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
Each column is independently scaled in blocks of 32 along the M dimension
```

### Memory Layout - Critical Discovery

From `transformer_engine/pytorch/tensor/mxfp8_tensor.py:112-130`:
```python
# For a matrix with shape [M, K]:

# Rowwise data and scales
data = torch.empty(shape, dtype=torch.uint8)  # Shape: [M, K]
scale_inv = torch.zeros([M, K//32], dtype=torch.uint8)

# Columnwise data and scales
columnwise_data = torch.empty_like(data)  # SAME SHAPE: [M, K]!
columnwise_scale_inv = torch.zeros([M//32, K], dtype=torch.uint8)
```

**Key insight:** Both rowwise and columnwise have the **same data shape** `[M, K]` but different **scaling patterns**. This is fundamentally different from standard Float8Tensor where columnwise is actually transposed.

---

## Row-Major vs Column-Major Context

### The Two Perspectives

**PyTorch (Row-Major):**
- Stores matrices row-by-row in memory
- Forward pass: `Y = X @ W^T` where:
  - `X` is `[batch, in_features]`
  - `W` is `[out_features, in_features]`
  - `Y` is `[batch, out_features]`

**BLAS (Column-Major):**
- Stores matrices column-by-column in memory
- A row-major matrix appears transposed to BLAS
- PyTorch's row-major `X[M, K]` appears as `X^T[K, M]` to BLAS

### How BLAS Sees Our Row-Major Matrices

When we pass row-major matrices to BLAS:
- Row-major `X[batch, in]` → BLAS sees `X^T[in, batch]`
- Row-major `W[out, in]` → BLAS sees `W^T[in, out]`
- Row-major `Y[batch, out]` → BLAS sees `Y^T[out, batch]`

### Mathematical Operations

For linear layer with weight `W[out_features, in_features]`:

| Operation | Row-Major Formula | What BLAS Sees | BLAS Computation |
|-----------|------------------|----------------|------------------|
| **Forward** | `Y = X @ W^T` | `Y^T = W @ X^T` | `gemm(W, X, "TN")` |
| **Backward dgrad** | `dX = dY @ W` | `dX^T = W^T @ dY^T` | `gemm(W, dY, "NN")` |
| **Backward wgrad** | `dW = dY^T @ X` | `dW^T = X^T @ dY` | `gemm(X, dY, "NT")` |

---

## MXFP8 Selection Logic in GEMM

### The Selection Rule

From `transformer_engine/common/gemm/rocm_gemm.cu:234-285`:

```cpp
if (is_mxfp_scaling(A.scaling_mode)) {
    // MXFP8 selection for A
    if (is_A_transposed) {
        // Use rowwise when A needs transpose
        ret.A = A.data.dptr;  // rowwise data
    } else {
        // Use columnwise when A doesn't need transpose
        ret.A = A.columnwise_data.dptr;
    }
    ret.transA = transA;  // Keep original transpose flag!
}

if (is_mxfp_scaling(B.scaling_mode)) {
    // MXFP8 selection for B
    if (is_B_transposed) {
        // Use columnwise when B needs transpose
        ret.B = B.columnwise_data.dptr;
    } else {
        // Use rowwise when B doesn't need transpose
        ret.B = B.data.dptr;  // rowwise data
    }
    ret.transB = transB;  // Keep original transpose flag!
}
```

**Key differences from standard FP8:**
1. MXFP8 keeps the original transpose flags (doesn't convert to TN)
2. **Both rowwise and columnwise have the same shape `[M, K]`** (unlike standard FP8 where columnwise is `[K, M]`)
3. Selection is based on which dimension needs block-wise scaling for accumulation

---

## Complete GEMM Examples

Let's trace through all three GEMM operations with a concrete example:
- Weight: `W[1024, 768]` (out_features=1024, in_features=768)
- Input: `X[batch_size, 768]`
- Output gradient: `dY[batch_size, 1024]`

### 1. Forward Pass (fprop): Y = X @ W^T (row-major view)

**What we want (row-major):** `Y[batch, out] = X[batch, in] @ W^T[in, out]`
**Concrete example:** `Y[batch, 1024] = X[batch, 768] @ W^T[768, 1024]`

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

**Code** (`transformer_engine/pytorch/module/linear.py:305-316`):
```python
gemm_out = general_gemm(
    weightmat,      # W[1024, 768]
    inputmat_total, # X[batch, 768]
    layout="TN",    # Default layout for forward
    ...
)
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

**Why this selection?**
- The dot products accumulate along the 768 (in_features) dimension
- Both matrices need their in_features dimension scaled in blocks

### 2. Backward dgrad: dX = dY @ W (row-major view)

**What we want (row-major):** `dX[batch, in] = dY[batch, out] @ W[out, in]`
**Concrete example:** `dX[batch, 768] = dY[batch, 1024] @ W[1024, 768]`

**BLAS computation with layout="NN":**
```python
# Code: general_gemm(W, dY, layout="NN")
# BLAS computes: C = A @ B (no transposes)
# So: dX^T = W^T @ dY^T
# Which equals: (dY @ W)^T in row-major
# Result when read as row-major: dX = dY @ W ✓
```

**Code** (`transformer_engine/pytorch/module/linear.py:674-693`):
```python
gemm_out = general_gemm(
    weight_fp8,    # W[1024, 768]
    grad_output,   # dY[batch, 1024]
    layout="NN",   # No transposes
    ...
)
# Update weight usage for MXFP8
weight_fp8.update_usage(columnwise_usage=True)
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

**Why this selection?**
- The dot products accumulate along the 1024 (out_features) dimension
- Both matrices need their out_features dimension scaled in blocks

### 3. Backward wgrad: dW = dY^T @ X (row-major view)

**What we want (row-major):** `dW[out, in] = dY^T[out, batch] @ X[batch, in]`
**Concrete example:** `dW[1024, 768] = dY^T[1024, batch] @ X[batch, 768]`

**BLAS computation with layout="NT":**
```python
# Code: general_gemm(X, dY, layout="NT")
# BLAS computes: C = A @ B^T
# So: dW^T = X^T @ dY^T^T = X^T @ dY
# Which equals: (dY^T @ X)^T in row-major
# Result when read as row-major: dW = dY^T @ X ✓
```

**Code** (`transformer_engine/pytorch/module/linear.py:734-736, 766-769, 802-826`):
```python
# Setup quantizers for wgrad
inputmat_total.update_usage(columnwise_usage=True)
grad_output.update_usage(columnwise_usage=True)

# wgrad GEMM
dw = general_gemm(
    inputmat_total,  # X[batch, 768]
    grad_output,     # dY[batch, 1024]
    layout="NT",     # Note: arguments are (X, dY) not (dY, X)
    ...
)
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

**Why this selection?**
- The dot products accumulate along the batch dimension
- Both matrices need their batch dimension scaled in blocks
- This is why wgrad benefits from larger batch sizes for MXFP8 efficiency

### Summary: MXFP8 Selection by Accumulation Dimension

The MXFP8 scaling dimension selection becomes clearer when we understand the actual data flow:

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

**The Key Insight:**

MXFP8 needs to scale along the dimension that will be **accumulated** in the GEMM:

1. **fprop**: Both matrices accumulate along `in_features` → both use rowwise (scales along K)
2. **dgrad**: Weight accumulates along `out_features` → uses columnwise (scales along M)
3. **wgrad**: Both accumulate along `batch` → both use columnwise

This ensures that within each dot product computation, all elements share the same scale factor, maintaining numerical stability.

---

## Code Implementation Details

### MXFP8 Tensor Structure

From `transformer_engine/pytorch/tensor/mxfp8_tensor.py:113-130`:
```python
# MXFP8 block size constant
MXFP8_BLOCK_SCALING_SIZE = 32  # From constants.py

# For a matrix conceptually [M, K]:
# Rowwise format
scale_inv = torch.zeros(
    round_up_to_nearest_multiple(M, 128),
    round_up_to_nearest_multiple(K // MXFP8_BLOCK_SCALING_SIZE, 4),
    dtype=torch.uint8
)

# Columnwise format
columnwise_scale_inv = torch.zeros(
    round_up_to_nearest_multiple(M // MXFP8_BLOCK_SCALING_SIZE, 4),
    round_up_to_nearest_multiple(K, 128),
    dtype=torch.uint8
)
```

### GEMM Backend Selection

From `transformer_engine/common/gemm/rocm_gemm.cu:200-291`:
```cpp
GemmParam CanonicalizeGemmInput(...) {
    // Lines 234-250: MXFP8 handling for A
    if (is_mxfp_scaling(A.scaling_mode)) {
        // Note: Row-wise and column-wise data are scaled along different
        // dimensions (with matrix interpreted in row-major order).
        if (is_A_transposed) {
            NVTE_CHECK(A.has_data(), "Input A is missing row-wise usage");
            ret.A = A.data.dptr;
        } else {
            NVTE_CHECK(A.has_columnwise_data(), "Input A is missing column-wise usage");
            ret.A = A.columnwise_data.dptr;
        }
        ret.transA = transA;  // Keep original flag
        ret.A_scale_inv = is_A_transposed ? A.scale_inv.dptr : A.columnwise_scale_inv.dptr;
    }

    // Lines 272-285: MXFP8 handling for B
    if (is_mxfp_scaling(B.scaling_mode)) {
        if (is_B_transposed) {
            NVTE_CHECK(B.has_columnwise_data(), "Input B is missing column-wise usage");
            ret.B = B.columnwise_data.dptr;
        } else {
            NVTE_CHECK(B.has_data(), "Input B is missing row-wise usage");
            ret.B = B.data.dptr;
        }
        ret.transB = transB;  // Keep original flag
        ret.B_scale_inv = is_B_transposed ? B.columnwise_scale_inv.dptr : B.scale_inv.dptr;
    }
}
```

---

## Summary

### Key Takeaways

1. **MXFP8 uses block-wise scaling** with 32-element blocks and E8M0 (power-of-2) scales

2. **Rowwise vs Columnwise terminology** is always from row-major (PyTorch) perspective:
   - Rowwise: Scales along K dimension (horizontal blocks)
   - Columnwise: Scales along M dimension (vertical blocks), stored transposed

3. **Selection pattern for MXFP8** is based on which dimension is accumulated:
   - If accumulating along K: Use rowwise (scales along K)
   - If accumulating along M: Use columnwise (scales along M)
   - If accumulating along batch: Use columnwise for both

4. **The three GEMM passes** use different formats:
   - **fprop**: Both rowwise (accumulate along in_features)
   - **dgrad**: Weight columnwise, dY rowwise (accumulate along out_features)
   - **wgrad**: Both columnwise (accumulate along batch)

5. **Memory trade-off**: 2× storage for weights but better numerical accuracy (no double quantization)

6. **Hardware optimization**: Modern GPUs (Blackwell, MI300/MI350) have native MXFP8 support

### Critical Insight: Same Shape, Different Quantization

Unlike standard Float8Tensor where:
- Rowwise: `[M, K]`
- Columnwise: `[K, M]` (transposed)

MXFP8 has:
- Rowwise: `[M, K]` with horizontal 32-element blocks
- Columnwise: `[M, K]` (same shape!) with vertical 32-element blocks

This means MXFP8 "columnwise" is NOT a transpose but a different quantization pattern of the same data.

### Performance Implications

- **Memory overhead**: ~3% for scales (1 byte per 32 elements) + 2× data when both formats needed
- **Accuracy benefit**: Each GEMM uses optimally quantized data for its accumulation pattern
- **Batch size matters**: wgrad efficiency depends on batch size (needs batch/32 blocks)

### References

- Code: ROCm TransformerEngine commit `f141f34bff6cd775dd113ee5a96f66c9d0a44fc8`
- OCP Microscaling Formats (MX) Specification
- NVIDIA/AMD documentation on FP8 training strategies