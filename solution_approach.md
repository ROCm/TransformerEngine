# Solution for Triton MXFP8 GEMM

## The Problem

1. **C++ BLAS approach**: Selects appropriate quantization (rowwise/columnwise) and lets BLAS handle transpose during GEMM
2. **Triton limitation**: `tl.dot_scaled` doesn't support transpose flags - needs data already in correct shape
3. **MXFP8 constraint**: Cannot transpose after quantization (would need requantization)

## Key Insights

1. MXFP8 rowwise and columnwise have **same shape** but different quantization patterns
2. C++ selects based on which dimension is accumulated (for numerical stability)
3. BLAS performs actual transpose during computation when transpose flag is set
4. Triton needs pre-transposed data with matching scale patterns

## Solution Options

### Option 1: Support Only NN Layout (Simple but Limited)
- Only support cases where no transposes are needed
- A uses rowwise, B uses columnwise
- Pros: Simple, works correctly
- Cons: Very limited - can't support fprop (TN) or wgrad (NT)

### Option 2: Pre-transpose and Requantize (Current Approach Issue)
- When transpose needed, transpose then requantize
- Problem: We don't have access to original FP32 data, only quantized
- Would need to dequantize → transpose → requantize (lossy)

### Option 3: Store Additional Transposed Versions
- For weights, store 4 versions:
  - W rowwise [out, in]
  - W columnwise [out, in]
  - W^T rowwise [in, out]
  - W^T columnwise [in, out]
- Pros: Correct quantization for all cases
- Cons: 4x storage for weights (unacceptable)

### Option 4: Hybrid Approach (RECOMMENDED)
- Recognize that for linear layers, we mainly need:
  - fprop: W^T (transposed weight)
  - dgrad: W (original weight)
  - wgrad: Both X and dY (activations, not weights)
- Store only what's needed:
  - W columnwise for dgrad (no transpose needed)
  - W^T rowwise for fprop (pre-transposed)
  - Activations use appropriate format dynamically
- This requires modifying MXFP8Tensor to support transpose during quantization

### Option 5: Custom Triton Kernel
- Modify the kernel to handle transpose internally
- Instead of using `tl.dot_scaled`, implement custom logic
- Pros: Most flexible
- Cons: Complex, potentially slower

## Recommended Implementation Path

For now, implement **Option 1** (NN-only support) to get something working, with clear error messages for unsupported layouts. This allows:
- Testing the basic MXFP8 functionality
- Validating numerical accuracy
- Building incrementally toward fuller support

```python
def select_mxfp8_for_triton(tensor, need_transpose, operand_idx):
    """Select MXFP8 format for Triton GEMM."""
    if need_transpose:
        raise NotImplementedError(
            f"MXFP8 with transpose not yet supported in Triton backend. "
            f"Operand {operand_idx} needs transpose but tl.dot_scaled requires "
            f"pre-transposed data with matching scale patterns."
        )

    if operand_idx == 0:  # First operand needs rowwise pattern
        return tensor._rowwise_data, tensor._rowwise_scale_inv
    else:  # Second operand needs columnwise pattern
        return tensor._columnwise_data, tensor._columnwise_scale_inv
```

## Future Work

1. Investigate if we can get FP32 data during quantization to support pre-transpose
2. Explore custom Triton kernels that handle transpose
3. Consider different storage strategies for weights vs activations
4. Benchmark memory vs compute tradeoffs for different approaches