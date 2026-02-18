# Complete MXFP8 Analysis: BLAS vs Triton Implementation

## Key Context

### MXFP8 Storage Reality (Confirmed by Testing)
- **Rowwise**: `[M, K]` with scales `[M, K//32]` (blocks along K dimension)
- **Columnwise**: `[M, K]` (SAME shape!) with scales `[M//32, K]` (blocks along M dimension)

Both formats have the **same shape** but different quantization patterns.

### Platform Differences
- **BLAS (Column-Major)**: Can apply transpose during GEMM computation
- **Triton (Row-Major)**: Needs data already in correct shape for `tl.dot_scaled`

---

## 1. Forward Pass (fprop): Y = X @ W^T

### Dimensions
- Weight: `W[1024, 768]` (out_features=1024, in_features=768)
- Input: `X[batch, 768]`
- Output: `Y[batch, 1024]`
- Operation: `Y = X @ W^T`

### BLAS Implementation (Column-Major)

**What BLAS sees:**
- Row-major `X[batch, 768]` → BLAS sees `X^T[768, batch]`
- Row-major `W[1024, 768]` → BLAS sees `W^T[768, 1024]`
- GEMM call: `gemm(W, X, "TN")`
- Computes: `W^T^T @ X^T = W @ X^T = (X @ W^T)^T`

**MXFP8 Selection:**
- Weight: `transA=T` → Uses **rowwise** `[1024, 768]`, scales `[1024, 24]`
- Input: `transB=N` → Uses **rowwise** `[batch, 768]`, scales `[batch, 24]`
- Both accumulate along in_features (768)

### Triton Implementation (Row-Major)

**What Triton needs:**
- Direct computation: `Y[batch, 1024] = X[batch, 768] @ W^T[768, 1024]`
- For `tl.dot_scaled`:
  - X needs: `[batch, 768]` with scales `[batch, 24]`
  - W^T needs: `[768, 1024]` with scales `[768, 32]`

**MXFP8 Selection Problem:**
- **Input X**: Can use rowwise ✓
  - Data: `[batch, 768]`, scales `[batch, 24]` - Perfect!
- **Weight W**: CANNOT get W^T correctly! ✗
  - W rowwise: `[1024, 768]` - wrong shape
  - W columnwise: `[1024, 768]` - still wrong shape (NOT transposed!)
  - Need W^T `[768, 1024]` but neither format provides it

**Conclusion:** fprop CANNOT be implemented without pre-transposed weights

---

## 2. Backward dgrad: dX = dY @ W

### Dimensions
- Weight: `W[1024, 768]`
- Grad output: `dY[batch, 1024]`
- Grad input: `dX[batch, 768]`
- Operation: `dX = dY @ W`

### BLAS Implementation (Column-Major)

**What BLAS sees:**
- Row-major `dY[batch, 1024]` → BLAS sees `dY^T[1024, batch]`
- Row-major `W[1024, 768]` → BLAS sees `W^T[768, 1024]`
- GEMM call: `gemm(W, dY, "NN")`
- Computes: `W^T @ dY^T = (dY @ W)^T`

**MXFP8 Selection:**
- Weight: `transA=N` → Uses **columnwise** `[1024, 768]`, scales `[32, 768]`
- Grad output: `transB=N` → Uses **rowwise** `[batch, 1024]`, scales `[batch, 32]`
- Both accumulate along out_features (1024)

### Triton Implementation (Row-Major)

**What Triton needs:**
- Direct computation: `dX[batch, 768] = dY[batch, 1024] @ W[1024, 768]`
- For `tl.dot_scaled`:
  - dY needs: `[batch, 1024]` with scales `[batch, 32]`
  - W needs: `[1024, 768]` with scales `[32, 768]`

**MXFP8 Selection:**
- **Grad output dY**: Can use rowwise ✓
  - Data: `[batch, 1024]`, scales `[batch, 32]` - Perfect!
- **Weight W**: Can use columnwise ✓
  - Data: `[1024, 768]`, scales `[32, 768]` - Perfect!

**Conclusion:** dgrad CAN be implemented! This is NN layout.

---

## 3. Backward wgrad: dW = dY^T @ X

### Dimensions
- Grad output: `dY[batch, 1024]`
- Input: `X[batch, 768]`
- Weight gradient: `dW[1024, 768]`
- Operation: `dW = dY^T @ X`

### BLAS Implementation (Column-Major)

**What BLAS sees:**
- Row-major `X[batch, 768]` → BLAS sees `X^T[768, batch]`
- Row-major `dY[batch, 1024]` → BLAS sees `dY^T[1024, batch]`
- GEMM call: `gemm(X, dY, "NT")`
- Computes: `X^T @ dY^T^T = X^T @ dY = (dY^T @ X)^T`

**MXFP8 Selection:**
- Input: `transA=N` → Uses **columnwise** `[batch, 768]`, scales `[batch/32, 768]`
- Grad output: `transB=T` → Uses **columnwise** `[batch, 1024]`, scales `[batch/32, 1024]`
- Both accumulate along batch dimension

### Triton Implementation (Row-Major)

**What Triton needs:**
- Direct computation: `dW[1024, 768] = dY^T[1024, batch] @ X[batch, 768]`
- For `tl.dot_scaled`:
  - dY^T needs: `[1024, batch]` with scales `[1024, batch/32]`
  - X needs: `[batch, 768]` with scales `[batch/32, 768]`

**MXFP8 Selection Problem:**
- **Grad output dY**: CANNOT get dY^T correctly! ✗
  - dY rowwise: `[batch, 1024]` - wrong shape
  - dY columnwise: `[batch, 1024]` - still wrong shape (NOT transposed!)
  - Need dY^T `[1024, batch]` but neither format provides it
- **Input X**: Can use columnwise ✓
  - Data: `[batch, 768]`, scales `[batch/32, 768]` - Correct pattern!

**Conclusion:** wgrad CANNOT be implemented without pre-transposed grad output

---

## Summary Table

| Operation | BLAS Support | Triton Support | Issue |
|-----------|--------------|----------------|-------|
| **fprop** | ✓ Works | ✗ Cannot | Need W^T but columnwise isn't transposed |
| **dgrad** | ✓ Works | ✓ Works! | NN layout - both operands available correctly |
| **wgrad** | ✓ Works | ✗ Cannot | Need dY^T but columnwise isn't transposed |

## Why BLAS Works But Triton Doesn't

**BLAS Success:**
1. Selects appropriate quantization pattern (rowwise or columnwise)
2. Passes transpose flag to BLAS routine
3. BLAS applies transpose during computation
4. Works because BLAS handles transpose as part of GEMM

**Triton Failure:**
1. `tl.dot_scaled` needs data already in correct shape
2. Cannot transpose after quantization (would break block structure)
3. Columnwise is NOT transposed (same shape as rowwise)
4. Only works when no transpose is needed (NN layout = dgrad only)

## Solution Requirements

To support all three operations in Triton, we need ONE of:

1. **Pre-transposed storage**: Store W^T and dY^T during quantization
2. **Custom kernel**: Replace `tl.dot_scaled` with transpose-aware implementation
3. **Dynamic requantization**: Transpose then requantize (defeats purpose)
4. **Accept limitation**: Only support dgrad (NN layout)

## Key Insight

The fundamental issue is that MXFP8 columnwise is a **different quantization pattern**, not a **transposed matrix**. This works for BLAS (which transposes during computation) but not for Triton (which needs pre-transposed data).

---

## Recommended Path Forward

### Short-term (Current Implementation)
- Support only dgrad (NN layout)
- Raise clear error for fprop and wgrad
- Document limitation prominently

### Medium-term
- Modify MXFP8Tensor to optionally store transposed versions
- For weights: Store both W and W^T quantized
- For activations: Quantize with transpose flag when needed

### Long-term
- Implement custom Triton kernel that handles transpose
- Or wait for `tl.dot_scaled` to support transpose flags