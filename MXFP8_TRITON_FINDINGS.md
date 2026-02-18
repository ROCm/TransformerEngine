# MXFP8 Triton Implementation Findings

## Executive Summary

We discovered a critical misunderstanding about MXFP8 columnwise storage that explains the numerical accuracy issues. The MXFP8 "columnwise" data is **NOT** transposed - it has the same shape as rowwise but with different quantization patterns. This makes it incompatible with Triton's `tl.dot_scaled` for transpose cases.

## Key Discovery

### What We Expected (From Documentation)
- Rowwise: `[M, K]` with scales `[M, K//32]`
- Columnwise: `[K, M]` (transposed) with scales `[K, M//32]`

### What Actually Exists
- Rowwise: `[M, K]` with scales `[M, K//32]` (blocks along K)
- Columnwise: `[M, K]` (SAME shape!) with scales `[M//32, K]` (blocks along M)

Both have the **same shape** but **different quantization patterns**.

## The Fundamental Problem

### C++ BLAS Approach
1. Selects appropriate quantization pattern (rowwise/columnwise)
2. Passes transpose flag to BLAS
3. BLAS handles the actual transpose during GEMM computation

### Triton Limitation
1. `tl.dot_scaled` doesn't support transpose flags
2. Expects data already in the correct shape with matching scale patterns
3. Cannot transpose MXFP8 after quantization (would need requantization)

## Working Solution: NN Layout Only

Currently, we can only support NN layout (no transposes):

```python
# For NN layout: C = A @ B
# A: [M, K] → use rowwise (scales [M, K//32])
# B: [K, N] → use columnwise (scales [K//32, N])

if transa or transb:
    raise NotImplementedError(
        "MXFP8 with transpose not yet supported in Triton backend"
    )

A_data = A_mxfp8._rowwise_data
a_scale_inv = A_mxfp8._rowwise_scale_inv
B_data = B_mxfp8._columnwise_data
b_scale_inv = B_mxfp8._columnwise_scale_inv
```

## Test Results

### Supported Case
- **NN layout**: ✓ Works correctly
  - A uses rowwise: `[128, 256]` with scales `[128, 8]`
  - B uses columnwise: `[256, 128]` with scales `[8, 128]`
  - Scale patterns match `tl.dot_scaled` requirements

### Unsupported Cases
- **TN layout** (fprop): ✗ Cannot support
  - Would need W^T pre-quantized in rowwise format
- **NT layout** (wgrad): ✗ Cannot support
  - Would need B^T pre-quantized in columnwise format
- **TT layout**: ✗ Cannot support
  - Would need both operands pre-transposed

## Why This Matters

The three main GEMM operations in neural networks are:
1. **Forward pass (fprop)**: `Y = X @ W^T` (TN layout) - **Cannot support**
2. **Backward dgrad**: `dX = dY @ W` (NN layout) - **Can support**
3. **Backward wgrad**: `dW = dY^T @ X` (NT layout) - **Cannot support**

This severely limits the usefulness of the current implementation.

## Potential Solutions

### 1. Pre-transpose During Quantization (Recommended)
Modify MXFP8Tensor to support transpose during quantization:
```python
# Pseudo-code
W_mxfp8 = quantizer.quantize(W, store_transpose=True)
# Would store both W and W^T quantized versions
```

### 2. Custom Triton Kernel
Implement a kernel that handles transpose internally rather than using `tl.dot_scaled`.

### 3. Hybrid Storage
- Weights: Store both original and transposed versions
- Activations: Quantize dynamically as needed
- Trade-off: 2x memory for weights

### 4. Accept Limited Support
Only support specific operations that don't require transpose.

## Files Modified

1. **transformer_engine/pytorch/gemm_triton.py**
   - Added MXFP8TensorWrapper class
   - Updated selection logic to understand non-transposed columnwise
   - Added NotImplementedError for transpose cases

2. **Documentation Created**
   - `mxfp8_rowwise_columnwise_analysis.md`: Comprehensive analysis
   - `fprop_triton_analysis.md`: Forward pass specific analysis
   - `solution_approach.md`: Solution options
   - This file: Summary of findings

## Next Steps

1. **Immediate**: Document the limitation clearly in code and docs
2. **Short-term**: Investigate pre-transpose during quantization
3. **Long-term**: Consider custom Triton kernel for full support

## Conclusion

The MXFP8 Triton implementation currently only supports NN layout due to the discovery that columnwise data is not actually transposed. This is a fundamental limitation that requires architectural changes to fully resolve. The C++ BLAS backend doesn't have this issue because BLAS can handle transposes during computation, while Triton's `tl.dot_scaled` cannot.