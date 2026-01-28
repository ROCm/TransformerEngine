# Implementation Plan: Adding FP8 Support to te_generic_gemm_triton()

## Executive Summary

The goal is to extend `te_generic_gemm_triton()` to detect Float8Tensor inputs, extract their FP8 components, and pass them correctly to the underlying `te_gemm_triton()` kernel which already supports FP8. This will enable the high-level wrapper to work with FP8 tensors seamlessly.

## Architecture Overview

### Current State
- **Low-level**: `te_gemm_triton()` (lines 174-277) fully supports FP8 with explicit scale_inv parameters
- **High-level**: `te_generic_gemm_triton()` (lines 73-171) only handles regular torch.Tensor
- **Gap**: No Float8Tensor detection/extraction in the high-level wrapper

### Target State
`te_generic_gemm_triton()` will:
1. Detect Float8Tensor inputs for A, B, and D (output)
2. Extract `_data` (uint8), `_scale_inv`, and `_fp8_dtype` from Float8Tensor objects
3. Convert FP8 data to native PyTorch FP8 types for Triton kernel
4. Pass scale_inv parameters to the matmul kernel
5. Handle FP8 output creation if quantizer is provided

---

## Detailed Implementation Plan

### Phase 1: Float8Tensor Detection and Extraction

**Location**: Beginning of `te_generic_gemm_triton()` function (after line 93)

**Steps**:

1. **Import Float8Tensor classes**
   ```python
   from transformer_engine.pytorch.float8_tensor import Float8Tensor
   from transformer_engine.pytorch.tensor._internal.float8_tensor_base import Float8TensorBase
   ```

2. **Create detection flags and storage variables**
   - `input_a_fp8`: bool flag for whether A is Float8Tensor
   - `input_b_fp8`: bool flag for whether B is Float8Tensor
   - `output_fp8`: bool flag for whether D is Float8Tensor
   - `a_scale_inv`: torch.Tensor or None for A's scale inverse
   - `b_scale_inv`: torch.Tensor or None for B's scale inverse
   - `d_scale`: torch.Tensor or None for D's scale
   - `d_amax`: torch.Tensor or None for D's amax buffer
   - `a_fp8_dtype`: tex.DType for A's FP8 format
   - `b_fp8_dtype`: tex.DType for B's FP8 format
   - `d_fp8_dtype`: tex.DType for D's FP8 format

3. **Extract Float8Tensor components for input A**
   ```python
   if isinstance(A, (Float8Tensor, Float8TensorBase)):
       input_a_fp8 = True
       a_fp8_dtype = A._fp8_dtype
       a_scale_inv = A._scale_inv
       # Extract raw uint8 data and convert to native PyTorch FP8 type
       A_data = A._data
       A = reinterpret_as_fp8_tensor(A_data, a_fp8_dtype)
   else:
       input_a_fp8 = False
       a_fp8_dtype = None
       a_scale_inv = None
   ```

4. **Extract Float8Tensor components for input B** (similar pattern)

5. **Handle output D tensor**
   - If D is provided and is Float8Tensor: extract components
   - If D is None and quantizer is provided: create Float8Tensor output
   - If D is None and no quantizer: create regular tensor output

### Phase 2: Batch Dimension Handling with FP8

**Location**: After extraction, before calling matmul (around lines 116-135)

**Challenge**: Float8Tensor needs special handling during view/reshape operations

**Steps**:

1. **Flatten batch dimensions for FP8 inputs**
   - For Float8Tensor: operate on the converted native FP8 tensor (already extracted)
   - The converted tensor is now regular torch.float8_e4m3fnuz/e5m2fnuz, can use standard view()

2. **Apply transpose and view operations**
   ```python
   # A and B are column major following BLAS convention
   # Triton matmul function assumes row major layouts
   a_row_major = B.T if transb else B
   b_row_major = A.T if transa else A
   a_row_major = a_row_major.view(-1, a_row_major.shape[-1])
   b_row_major = b_row_major.view(-1, b_row_major.shape[-1])
   ```

3. **Swap scale_inv parameters for row-major layout**
   ```python
   # Since we swap A and B for row-major, also swap their scales
   a_scale_triton = b_scale_inv  # B's scale for Triton's 'a'
   b_scale_triton = a_scale_inv  # A's scale for Triton's 'b'
   ```

### Phase 3: Output Tensor Creation

**Location**: Lines 149-160

**Steps**:

1. **Calculate output shape** (already implemented in `getGemmOutputShape()`)

2. **Create output tensor based on type**:

   **Case 1: FP8 output with quantizer**
   ```python
   if quantizer is not None and output_dtype is not None:
       # Use quantizer to create FP8 output
       output_fp8 = True
       # Determine FP8 dtype from output_dtype or quantizer
       if is_fp8_dtype(output_dtype):
           d_fp8_dtype = output_dtype
       elif hasattr(quantizer, 'dtype'):
           d_fp8_dtype = quantizer.dtype

       # Create Float8Tensor using quantizer
       D = quantizer.make_empty(D_shape, dtype=A.dtype, device=A.device)
       d_scale = 1.0 / D._scale_inv  # Convert scale_inv to scale
       d_amax = torch.zeros(1, dtype=torch.float32, device=A.device)

       # Extract and convert the raw data for kernel
       D_data = D._data
       D_kernel = reinterpret_as_fp8_tensor(D_data, d_fp8_dtype)
   ```

   **Case 2: Regular output (no FP8)**
   ```python
   else:
       output_fp8 = False
       # Determine output dtype: use A's nominal dtype if FP8, else A's actual dtype
       if input_a_fp8:
           out_dtype = A.dtype  # Use the nominal dtype from Float8Tensor
       else:
           out_dtype = A.dtype
       D = torch.empty(D_shape, dtype=out_dtype, device=A.device)
       d_scale = None
       d_amax = None
   ```

3. **Prepare output for kernel**
   ```python
   d_row_major = D_kernel.view(-1, D_kernel.shape[-1]) if output_fp8 else D.view(-1, D.shape[-1])
   ```

### Phase 4: Kernel Invocation

**Location**: Lines 162-169

**Steps**:

1. **Determine FP8 flags**
   ```python
   input_fp8 = input_a_fp8 and input_b_fp8
   ```

2. **Call matmul kernel with correct parameters**
   ```python
   matmul(
       a_row_major,           # B in row-major (already converted to native FP8 if needed)
       b_row_major,           # A in row-major (already converted to native FP8 if needed)
       d_row_major,           # D in row-major (already converted to native FP8 if needed)
       a_scale_triton,        # B's scale_inv
       b_scale_triton,        # A's scale_inv
       d_scale,               # D's scale (not scale_inv!)
       bias,                  # Keep as None for now (DEFAULT epilogue)
       d_amax,                # D's amax buffer
       epilogue,              # 'DEFAULT'
       input_fp8,             # True if both A and B are FP8
       output_fp8             # True if D is FP8
   )
   ```

3. **Note**: The matmul kernel expects:
   - `a_scale` and `b_scale` as scale_inv (inverse of scale)
   - `c_scale` as scale (NOT scale_inv) for output
   - This is confirmed by lines 367-372 in the matmul_kernel

### Phase 5: Output Processing and Return

**Location**: Lines 170-171

**Steps**:

1. **Update Float8Tensor metadata if output is FP8**
   ```python
   if output_fp8:
       # Update the Float8Tensor's scale_inv based on computed amax
       # For delayed scaling, this would be handled by the quantizer/recipe
       # For now, the scale was already set during creation

       # Copy the kernel output back to the Float8Tensor's _data
       D._data.copy_(D_kernel.view(torch.uint8))

       # Update amax if using current scaling
       if hasattr(quantizer, 'amax') and d_amax is not None:
           quantizer.amax.copy_(d_amax)
   ```

2. **Reshape output back to original batch dimensions**
   ```python
   # D already has the correct shape from creation
   # No additional reshaping needed since we used D_shape
   ```

3. **Return values**
   ```python
   return D, bias, None, None
   ```

---

## Edge Cases and Considerations

### 1. Mixed Precision Operations

**Case**: FP8 input A, higher precision input B
- **Handling**: Not supported in initial implementation (input_fp8 requires both)
- **Behavior**: Fall back to dequantizing FP8 tensor and doing higher precision GEMM
- **Implementation**: Check `input_fp8 = input_a_fp8 and input_b_fp8`; if only one is FP8, could dequantize

### 2. Different FP8 Formats

**Case**: A is E4M3, B is E5M2
- **Handling**: Fully supported
- **Evidence**: Test suite includes ('fp8e5-fp8e4', 'fp32') and ('fp8e4-fp8e5', 'fp32')
- **Implementation**: Extract dtype from each tensor independently

### 3. Batch Dimensions with FP8

**Case**: Tensors with shape (batch, m, k) in FP8
- **Handling**: Flatten to 2D before calling matmul
- **Implementation**: Use `product()` function to flatten batch dims (lines 116-119)
- **Consideration**: Float8Tensor view operations preserve FP8 metadata

### 4. Transpose Flags with FP8

**Case**: transa=True or transb=True with Float8Tensor
- **Handling**: Extract data first, then transpose
- **Implementation**: After extracting `_data` and converting to native FP8, standard transpose works
- **Note**: Float8Tensor has `_transpose` cache, but we're using converted native FP8 tensor

### 5. Quantizer Integration

**Case**: quantizer parameter provided for output
- **Handling**: Use quantizer.make_empty() to create FP8 output
- **Type checking**: Support both Float8Quantizer and Float8CurrentScalingQuantizer
- **Scale management**: Extract scale from quantizer for output scaling

### 6. None vs Empty Tensor

**Case**: D=None vs D=torch.Tensor()
- **Handling**:
  - D=None: Create new output tensor
  - D=torch.Tensor() (empty): This pattern is used for optional parameters in the codebase
- **Implementation**: Check `D is None` or `D.data_ptr() == 0`

---

## Data Type Conversion Flow

### Input Flow (FP8 → Native PyTorch FP8)
```
Float8Tensor (nominal dtype: torch.float32)
  ├─ _data: torch.uint8 (raw FP8 bytes)
  ├─ _fp8_dtype: tex.DType.kFloat8E4M3 or kFloat8E5M2
  └─ _scale_inv: torch.float32

↓ (reinterpret_as_fp8_tensor)

torch.float8_e4m3fnuz or torch.float8_e5m2fnuz
  └─ (Triton kernel can handle native PyTorch FP8)
```

### Output Flow (Native PyTorch FP8 → Float8Tensor)
```
Triton kernel output: torch.float8_e4m3fnuz or torch.float8_e5m2fnuz

↓ (view as uint8)

torch.uint8 (raw bytes)

↓ (copy to Float8Tensor._data)

Float8Tensor
  ├─ _data: torch.uint8 (updated with kernel output)
  ├─ _fp8_dtype: tex.DType.kFloat8E4M3 or kFloat8E5M2
  └─ _scale_inv: torch.float32 (1 / scale used in kernel)
```

---

## Validation and Testing Strategy

### Unit Tests to Create

1. **Basic FP8 GEMM Test**
   - Input: Float8Tensor A (E4M3), Float8Tensor B (E4M3)
   - Output: Regular tensor (fp32)
   - Layout: TN (non-transposed)
   - Verify: Correctness against reference implementation

2. **Mixed FP8 Format Test**
   - Input: Float8Tensor A (E4M3), Float8Tensor B (E5M2)
   - Output: Regular tensor (bf16)
   - Verify: Handles different FP8 formats correctly

3. **Batch Dimension Test**
   - Input: Float8Tensor A (shape: [4, 128, 256]), Float8Tensor B (shape: [4, 256, 512])
   - Verify: Correct batch dimension handling and output shape

4. **Transpose Test**
   - Layout: NT, NN (avoid TT which is not allowed)
   - Verify: Transpose operations work correctly with Float8Tensor

5. **FP8 Output Test**
   - Input: Float8Tensor A, Float8Tensor B
   - Output: Float8Tensor D (with quantizer)
   - Verify: Output is correctly quantized and amax is computed

### Verification Steps

1. **Type Checking**
   - Verify output is Float8Tensor when quantizer is provided
   - Verify output is regular tensor when no quantizer

2. **Numerical Accuracy**
   - Compare against te_gemm_triton() with explicitly extracted parameters
   - Compare against reference implementation (torch.matmul with dequantized inputs)
   - Use appropriate tolerances (atol=5e-3 from test suite)

3. **Scale Handling**
   - Verify scale_inv is correctly passed to kernel
   - Verify output scaling is correct for FP8 output

4. **Edge Cases**
   - Empty tensors
   - Scalar dimensions (m=1, k=1, n=1)
   - Large batch dimensions

---

## Implementation Sequence

### Step 1: Add Float8Tensor Detection (Lines 94-115)
- Add imports
- Add detection logic for A, B, D
- Extract _fp8_dtype, _scale_inv, _data

### Step 2: Convert FP8 Data (Lines 100-115)
- Use reinterpret_as_fp8_tensor() to convert uint8 → native FP8
- Store original nominal dtype for output creation

### Step 3: Update Shape Calculations (Lines 116-125)
- Keep existing shape calculation logic
- Works with both regular tensors and converted FP8 tensors

### Step 4: Update Layout Conversion (Lines 129-141)
- Add scale_inv swapping for row-major layout
- Set a_scale_triton and b_scale_triton correctly

### Step 5: Update Output Creation (Lines 149-161)
- Add quantizer-based FP8 output creation
- Add regular output creation with correct dtype

### Step 6: Update Kernel Call (Lines 162-169)
- Pass all FP8 parameters correctly
- Set input_fp8 and output_fp8 flags

### Step 7: Add Output Processing (After line 169)
- Copy kernel output to Float8Tensor._data if needed
- Update amax if applicable

### Step 8: Testing and Validation
- Create test cases following test_gemm_triton.py pattern
- Verify all edge cases

---

## Dependencies and Prerequisites

### Required Functions (Already Available)
- `is_fp8_dtype(dtype)`: Check if dtype is FP8 (line 42)
- `reinterpret_as_fp8_tensor(a, dtype)`: Convert uint8 → native FP8 (lines 45-49)
- `torch_to_te_dtype(dtype)`: Convert torch dtype → tex.DType (lines 13-23)
- `te_to_torch_dtype(dtype)`: Convert tex.DType → torch dtype (lines 25-40)
- `getGemmOutputShape(A, transa, B, transb)`: Calculate output shape (lines 51-65)
- `product(shape)`: Calculate product of shape dimensions (lines 67-71)

### Required Imports
```python
from transformer_engine.pytorch.float8_tensor import Float8Tensor
from transformer_engine.pytorch.tensor._internal.float8_tensor_base import Float8TensorBase
from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer, Float8CurrentScalingQuantizer
```

---

## Backward Compatibility

### Maintaining Existing Behavior
1. **Regular Tensors**: All existing code paths remain unchanged
2. **Function Signature**: No changes to te_generic_gemm_triton() signature
3. **Return Values**: Same return structure (D, bias, None, None)
4. **Error Handling**: Same assertions and error conditions

### Gradual Feature Rollout
1. **Phase 1**: FP8 inputs, regular output (DEFAULT epilogue)
2. **Phase 2** (Future): FP8 output with quantizer
3. **Phase 3** (Future): BIAS and BGRADB epilogues with FP8

---

## Performance Considerations

### Overhead Analysis
1. **Type Checking**: Minimal overhead (isinstance checks)
2. **Data Extraction**: Zero-copy view operations
3. **Conversion**: reinterpret_as_fp8_tensor is a view, not a copy
4. **Additional Memory**: Only scale_inv and amax buffers (small)

### Optimization Opportunities
1. **Caching**: Float8Tensor already caches _transpose
2. **In-place Operations**: Use existing buffers when possible
3. **Device Placement**: All operations on GPU, no CPU transfers

---

## Risk Assessment and Mitigation

### High Risk
1. **Scale vs Scale_inv Confusion**
   - **Risk**: Kernel expects scale for output, scale_inv for input
   - **Mitigation**: Clear documentation, thorough testing
   - **Evidence**: Lines 367-372 show kernel loading scale vs scale_inv

### Medium Risk
1. **Batch Dimension Handling**
   - **Risk**: Incorrect flattening/unflattening with FP8
   - **Mitigation**: Use existing product() function, test with various batch sizes

2. **Transpose Operations**
   - **Risk**: Transpose breaking FP8 tensor structure
   - **Mitigation**: Extract data before transpose, use native FP8 tensors

### Low Risk
1. **Type Compatibility**
   - **Risk**: Float8Tensor vs Float8TensorBase differences
   - **Mitigation**: Check for both types, use base class attributes

---

## Success Criteria

### Functional Requirements
- [ ] Detects Float8Tensor inputs correctly
- [ ] Extracts FP8 components (_data, _scale_inv, _fp8_dtype)
- [ ] Converts uint8 data to native PyTorch FP8 types
- [ ] Passes scale_inv to kernel correctly
- [ ] Creates FP8 output when quantizer provided
- [ ] Maintains backward compatibility with regular tensors

### Performance Requirements
- [ ] Minimal overhead compared to te_gemm_triton()
- [ ] Zero-copy operations where possible
- [ ] No unnecessary device transfers

### Quality Requirements
- [ ] All unit tests pass
- [ ] Numerical accuracy within tolerance (atol=5e-3)
- [ ] Edge cases handled correctly
- [ ] Code follows existing patterns and style

---

## Critical Files for Implementation

The following files are most critical for implementing this plan:

- **transformer_engine/pytorch/gemm_triton.py** - Primary implementation file; contains te_generic_gemm_triton() function that needs modification (lines 73-171)

- **transformer_engine/pytorch/tensor/float8_tensor.py** - Float8Tensor class definition; needed to understand _data, _scale_inv, _fp8_dtype attributes and Float8Quantizer interface

- **transformer_engine/pytorch/tensor/_internal/float8_tensor_base.py** - Float8TensorBase class; provides core FP8 tensor attributes and methods that both Float8Tensor and internal variants share

- **tests/pytorch/test_gemm_triton.py** - Test patterns for FP8 GEMM; provides reference for creating test cases and understanding expected behavior with FP8 inputs

- **transformer_engine/pytorch/cpp_extensions/gemm.py** - Shows how hipBLASLt backend handles Float8Tensor; provides pattern for Float8Tensor detection and quantizer usage (lines 28-127)

---

## Implementation Status

**Completed in commit 5ab60234**: "Add FP8 support to te_generic_gemm_triton() wrapper"

Key changes made:
1. Float8TensorWrapper class - Mimics C++ TensorWrapper behavior
2. Updated te_generic_gemm_triton() - Detects and extracts Float8Tensor components
3. Fixed getGemmOutputShape() - Matches C++ backend implementation
4. Columnwise tensor handling - Correctly transposes with dimension reordering

This plan served as the design document for the implementation.
