# MXFP8 Selection Logic for Triton Implementation

## Overview
This document defines the correct MXFP8 data/scale selection for Triton's row-major kernels, accounting for logical transpose capabilities.

---

## Selection Rules for Triton

### Key Principle
- Triton works in row-major (natural PyTorch layout)
- We can use logical transpose (view) without data movement
- `tl.dot_scaled` needs specific scale patterns along reduction dimension

### Selection Logic

| **Layout** | **transA** | **transB** | **A needs** | **A selection** | **B needs** | **B selection** |
|------------|------------|------------|-------------|-----------------|-------------|-----------------|
| **TN** | True | False | A^T: `[M,K]` from `[K,M]` | **columnwise** (stored as A^T) | B: `[K,N]` | **columnwise.T** (transpose view) |
| **NN** | False | False | A: `[M,K]` | **rowwise** | B: `[K,N]` | **columnwise.T** |
| **NT** | False | True | A: `[M,K]` | **rowwise** | B^T: `[K,N]` from `[N,K]` | **rowwise** (then transpose) |

---

## Detailed Analysis by Pass

### 1. Forward Pass (fprop): Y = X @ W^T

**Layout:** TN (transA=True, transB=False)
- Weight W: `[out, in]` → needs W^T: `[in, out]`
- Input X: `[batch, in]` → use as-is

**Selection:**
- **Weight (transA=True):** Use **columnwise**
  - Stored as `[in, out]` (already W^T!)
  - Scales: `[in/32, out]` ✓
- **Input (transB=False):** Use **rowwise**
  - Data: `[batch, in]`
  - Scales: `[batch, in/32]` ✓

**Why it works:** Both accumulate along `in_features` dimension

---

### 2. Backward dgrad: dX = dY @ W

**Layout:** NN (transA=False, transB=False)
- Weight W: `[out, in]` → use as-is
- Grad output dY: `[batch, out]` → use as-is

**Selection:**
- **Weight (transA=False):** Use **rowwise**
  - Data: `[out, in]`
  - Scales: `[out, in/32]` ✓
- **Grad output (transB=False):** Use **columnwise.T**
  - Stored as: `[out, batch]` → transpose to `[batch, out]`
  - Scales: `[out/32, batch]` → transpose to `[batch, out/32]` ✓

**Why it works:** Both accumulate along `out_features` dimension

---

### 3. Backward wgrad: dW = dY^T @ X

**Layout:** NT (transA=False, transB=True)
- Grad output dY: `[batch, out]` → needs dY^T: `[out, batch]`
- Input X: `[batch, in]` → use as-is

**Selection:**
- **Grad output (transA=False):** Use **columnwise**
  - Stored as `[out, batch]` (already dY^T!)
  - Scales: `[out/32, batch]` ✓
- **Input (transB=True):** Use **columnwise**
  - Stored as `[in, batch]` (already X^T!)
  - Scales: `[in/32, batch]` ✓

**Alternative (if batch dimension is small):**
- Could use rowwise for both and transpose, but columnwise is more efficient

**Why it works:** Both accumulate along `batch` dimension

---

## Implementation Strategy

### For each operand:

1. **Determine what shape we need** (considering transpose flags)
2. **Check if columnwise gives us that shape directly**
   - If yes → use columnwise
   - If no → check if columnwise.T gives us the shape
   - Otherwise → use rowwise (and transpose if needed)
3. **Ensure scales match** the expected pattern for `tl.dot_scaled`

### Code Pattern:

```python
def select_mxfp8_for_triton(tensor, need_transpose, is_first_operand):
    """
    Select the right MXFP8 format for Triton kernel.

    For first operand: need [M, K] with scales [M, K/32]
    For second operand: need [K, N] with scales [K/32, N]
    """
    if is_first_operand:
        # Need rowwise scaling pattern
        if not need_transpose:
            return tensor._rowwise_data, tensor._rowwise_scale_inv
        else:
            # Check if columnwise is already transposed
            if tensor._columnwise_data.shape == needed_shape:
                return tensor._columnwise_data, tensor._columnwise_scale_inv
            else:
                # Transpose rowwise
                return tensor._rowwise_data.T, tensor._rowwise_scale_inv.T
    else:
        # Need columnwise scaling pattern (along K dimension)
        # This is trickier - need scales [K/32, N]
        if not need_transpose:
            # Use columnwise and transpose it
            return tensor._columnwise_data.T, tensor._columnwise_scale_inv.T
        else:
            # Check if rowwise after transpose gives right pattern
            # Usually doesn't work well
            pass
```

---

## Comparison with C++ (BLAS) Selection

| Pass | Operation | C++ A selection | C++ B selection | Triton A selection | Triton B selection |
|------|-----------|-----------------|-----------------|--------------------|--------------------|
| **fprop** | Y = X @ W^T | W rowwise | X rowwise | W columnwise | X rowwise |
| **dgrad** | dX = dY @ W | W columnwise | dY rowwise | W rowwise | dY columnwise.T |
| **wgrad** | dW = dY^T @ X | X columnwise | dY columnwise | dY columnwise | X columnwise |

**Key differences:**
- C++ forces everything to TN layout for hardware
- Triton can use natural layouts with logical transpose
- Columnwise storage often gives us the transpose "for free"

---

## Summary

The key insight is that MXFP8's columnwise storage (which stores the transpose) combined with Triton's ability to handle logical transpose (view) operations allows us to match `tl.dot_scaled`'s requirements perfectly for many cases.

**General rule for Triton:**
1. First operand needs rowwise scaling → prefer rowwise or columnwise-that-gives-right-shape
2. Second operand needs columnwise scaling → prefer columnwise.T or format-that-gives-right-scales
3. Use logical transpose (views) whenever possible to avoid data movement