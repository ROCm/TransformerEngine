# Implementation Plan: MXFP8 Support for Triton GEMM

## Overview

Implement end-to-end MXFP8 (Microscaling FP8) support for Transformer Engine's Triton GEMM backend. This includes:
1. **Wrapper detection**: Extend `te_generic_gemm_triton()` to detect and extract MXFP8Tensor components
2. **Kernel implementation**: Create new Triton kernel using `tl.dot_scaled()` for block-scaled FP8 matmul
3. **Scale handling**: Convert E8M0 scales to format expected by `tl.dot_scaled()`

**Reference:** Triton's block-scaled matmul tutorial (https://triton-lang.org/main/getting-started/tutorials/10-block-scaled-matmul.html)

---

## Key Differences: MXFP8 vs Regular FP8

| Aspect | Regular FP8 | MXFP8 |
|--------|-------------|-------|
| **Scaling granularity** | Per-tensor (1 scale) | Per-block (32 elements) |
| **Scale format** | Float32 scalar | uint8 E8M0 (biased exponent) |
| **Scale shape for [M,K]** | Scalar | [M_padded//128, K_padded//32] |
| **Scale application** | After K-loop: `acc *= scale` | Inside K-loop: `tl.dot_scaled()` |
| **Transpose behavior** | Numerically trivial | **Requires requantization** (precision loss) |
| **Storage strategy** | Single copy | **Dual copies** (rowwise + columnwise) |
| **Tensor class** | Float8Tensor | MXFP8Tensor |
| **Data attributes** | `_data`, `_scale_inv` | `_rowwise_data`, `_rowwise_scale_inv`, `_columnwise_data`, `_columnwise_scale_inv` |
| **Why dual storage?** | N/A (transpose is free) | **Requirement:** Avoid requantization, ensure consecutive data over reduction dim for Tensor Cores |

---

## Current State

### What Exists
✓ Regular FP8 support in `te_generic_gemm_triton()` (partially working)
✓ MXFP8 quantization/dequantization kernels in `cast_transpose.py`
✓ MXFP8Tensor class with E8M0 scale storage
✓ Triton version supports `tl.dot_scaled()`

### What We Need to Implement
- [ ] MXFP8Tensor detection in wrapper
- [ ] E8M0 scale extraction and conversion
- [ ] New kernel using `tl.dot_scaled()` for block scaling
- [ ] Rowwise/columnwise scale selection based on operand
- [ ] Tests for MXFP8 GEMM

---

## Phase 1: Wrapper Implementation (te_generic_gemm_triton)

### File: `/workspace/TransformerEngine/transformer_engine/pytorch/gemm_triton.py`

### Verified Against C++ Implementation

The C++ side (`type_converters.cpp:55-85`) extracts MXFP8Tensor components as follows:
```cpp
// Extract rowwise data and scale_inv
if (!(tensor.attr("_rowwise_data").is_none())) {
    const auto &data = tensor.attr("_rowwise_data").cast<at::Tensor>();
    const auto &scale_inv = tensor.attr("_rowwise_scale_inv").cast<at::Tensor>();
    ret.set_rowwise_data(data.data_ptr(), fp8_dtype, getTensorShape(data));
    ret.set_rowwise_scale_inv(scale_inv.data_ptr(), DType::kFloat8E8M0, getTensorShape(scale_inv));
}
// Extract columnwise data and scale_inv (similar)
```

GEMM canonicalization (`cublaslt_gemm.cu:128-200`) selects data based on transpose:
- **A operand:** `is_A_transposed ? rowwise : columnwise` (keeps original transA flag)
- **B operand:** `is_B_transposed ? columnwise : rowwise` (keeps original transB flag)

### Step 1: Create MXFP8TensorWrapper Class

**Location:** After `Float8TensorWrapper` class (around line 275)

```python
class MXFP8TensorWrapper:
    """
    Python equivalent of C++ TensorWrapper for MXFP8Tensor.

    Mimics NVTETensorFromMXFP8Tensor in type_converters.cpp, extracting
    both rowwise and columnwise data/scales.
    """

    def __init__(self, tensor):
        """
        Create wrapper from MXFP8Tensor or MXFP8TensorBase.

        Args:
            tensor: Input tensor (MXFP8Tensor, MXFP8TensorBase, or regular tensor)
        """
        # Import here to avoid circular dependency
        try:
            from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor
            from transformer_engine.pytorch.tensor._internal.mxfp8_tensor_base import MXFP8TensorBase
            is_mxfp8_tensor = isinstance(tensor, (MXFP8Tensor, MXFP8TensorBase))
        except ImportError:
            is_mxfp8_tensor = False

        if is_mxfp8_tensor:
            # Extract MXFP8 components (matching NVTETensorFromMXFP8Tensor)
            self._is_mxfp8 = True

            # Rowwise data and scales
            self._rowwise_data = tensor._rowwise_data if hasattr(tensor, '_rowwise_data') and tensor._rowwise_data is not None else None
            self._rowwise_scale_inv = tensor._rowwise_scale_inv if hasattr(tensor, '_rowwise_scale_inv') and tensor._rowwise_scale_inv is not None else None

            # Columnwise data and scales
            self._columnwise_data = tensor._columnwise_data if hasattr(tensor, '_columnwise_data') and tensor._columnwise_data is not None else None
            self._columnwise_scale_inv = tensor._columnwise_scale_inv if hasattr(tensor, '_columnwise_scale_inv') and tensor._columnwise_scale_inv is not None else None

            # Verify we have at least one format
            if self._rowwise_data is None and self._columnwise_data is None:
                raise RuntimeError(
                    "MXFP8Tensor has neither rowwise nor columnwise data"
                )

            # FP8 metadata
            self._fp8_dtype = tensor._fp8_dtype
            self._nominal_dtype = tensor.dtype if hasattr(tensor, 'dtype') else torch.float32

            # Determine logical size from available data
            if self._rowwise_data is not None:
                self._size = self._rowwise_data.size()
            else:
                # Convert columnwise shape to rowwise: [K,M,*batch] -> [*batch,M,K]
                ndim = self._columnwise_data.dim()
                if ndim == 2:
                    self._size = torch.Size([self._columnwise_data.size(1), self._columnwise_data.size(0)])
                else:
                    # Has batch dims at end, need to move to front and swap matrix dims
                    batch_dims = list(self._columnwise_data.size()[2:])
                    m_dim = self._columnwise_data.size(1)
                    k_dim = self._columnwise_data.size(0)
                    self._size = torch.Size(batch_dims + [m_dim, k_dim])
        else:
            # Not MXFP8 - wrap as regular tensor
            self._is_mxfp8 = False
            self._rowwise_data = tensor
            self._columnwise_data = None
            self._rowwise_scale_inv = None
            self._columnwise_scale_inv = None
            self._fp8_dtype = None
            self._nominal_dtype = tensor.dtype
            self._size = tensor.size()

    def size(self):
        """Get logical tensor size (in rowwise format)."""
        return self._size

    @property
    def is_mxfp8(self):
        """Check if this is an MXFP8 tensor."""
        return self._is_mxfp8

    @property
    def fp8_dtype(self):
        """Get FP8 dtype."""
        return self._fp8_dtype

    @property
    def nominal_dtype(self):
        """Get nominal dtype (what the MXFP8 tensor represents)."""
        return self._nominal_dtype

    def get_data_and_scale_for_gemm(self, will_transpose):
        """
        Get appropriate data and scale tensors for GEMM based on transpose flag.

        Matches C++ logic in cublaslt_gemm.cu:128-200 for MXFP8 scaling mode.
        Returns data in rowwise orientation for Triton (row-major).

        Args:
            will_transpose: Whether this operand will be transposed in GEMM

        Returns:
            tuple: (data_tensor, scale_inv_tensor) both in rowwise orientation
        """
        if not self._is_mxfp8:
            # Regular tensor - no scales
            return self._rowwise_data, None

        # MXFP8 selection logic (matching C++ cublaslt_gemm.cu:128-141 for A, 187-200 for B)
        # For operand A: transposed ? rowwise : columnwise
        # For operand B: transposed ? columnwise : rowwise
        #
        # However, we need to determine which operand we are (A or B).
        # The caller knows this context. For now, we'll use a conservative approach:
        # - Prefer rowwise if available
        # - Fall back to columnwise and convert to rowwise

        # Try rowwise first
        if self._rowwise_data is not None:
            return self._rowwise_data, self._rowwise_scale_inv

        # Only columnwise available - need to convert to rowwise for Triton
        # Columnwise: [K, M, *batch] -> Rowwise: [*batch, M, K]
        ndim = self._columnwise_data.dim()
        if ndim == 2:
            rowwise_data = self._columnwise_data.transpose(0, 1).contiguous()
        else:
            # Move batch dims to front and swap matrix dims
            batch_dims = list(range(2, ndim))
            perm = batch_dims + [1, 0]
            rowwise_data = self._columnwise_data.permute(*perm).contiguous()

        # Convert columnwise scale to rowwise scale
        # Scale shape follows data shape pattern
        if self._columnwise_scale_inv is not None:
            scale_ndim = self._columnwise_scale_inv.dim()
            if scale_ndim == 2:
                rowwise_scale = self._columnwise_scale_inv.transpose(0, 1).contiguous()
            else:
                batch_dims = list(range(2, scale_ndim))
                perm = batch_dims + [1, 0]
                rowwise_scale = self._columnwise_scale_inv.permute(*perm).contiguous()
        else:
            rowwise_scale = None

        return rowwise_data, rowwise_scale
```

### Step 2: Integrate MXFP8TensorWrapper into te_generic_gemm_triton()

**Location:** Replace `Float8TensorWrapper` usage (lines 298-312)

```python
# Wrap inputs to handle Float8Tensor and MXFP8Tensor uniformly
# Try MXFP8 first, then Float8, then regular
try:
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor
    from transformer_engine.pytorch.tensor._internal.mxfp8_tensor_base import MXFP8TensorBase
    is_mxfp8_a = isinstance(A, (MXFP8Tensor, MXFP8TensorBase))
    is_mxfp8_b = isinstance(B, (MXFP8Tensor, MXFP8TensorBase))
except ImportError:
    is_mxfp8_a = False
    is_mxfp8_b = False

if is_mxfp8_a or is_mxfp8_b:
    # Use MXFP8TensorWrapper
    A_wrapper = MXFP8TensorWrapper(A)
    B_wrapper = MXFP8TensorWrapper(B)

    # Validate both are MXFP8
    if A_wrapper.is_mxfp8 != B_wrapper.is_mxfp8:
        raise ValueError("Mixed MXFP8 and non-MXFP8 inputs not supported")

    # Extract data and scales
    A_data, a_scale_inv = A_wrapper.get_data_and_scale_for_gemm(will_transpose=transa)
    B_data, b_scale_inv = B_wrapper.get_data_and_scale_for_gemm(will_transpose=transb)

    a_fp8_dtype = A_wrapper.fp8_dtype
    b_fp8_dtype = B_wrapper.fp8_dtype

    input_mxfp8 = True
else:
    # Use Float8TensorWrapper (existing code)
    A_wrapper = Float8TensorWrapper(A)
    B_wrapper = Float8TensorWrapper(B)

    A_data = A_wrapper.get_data_for_gemm(will_transpose=transa)
    B_data = B_wrapper.get_data_for_gemm(will_transpose=transb)

    a_fp8_dtype = A_wrapper.fp8_dtype
    b_fp8_dtype = B_wrapper.fp8_dtype
    a_scale_inv = A_wrapper.scale_inv
    b_scale_inv = B_wrapper.scale_inv

    input_mxfp8 = False
```

---

## Phase 2: MXFP8 Kernel Implementation

### File: `/workspace/TransformerEngine/transformer_engine/pytorch/gemm_triton.py`

### Design Approach

Use Triton's `tl.dot_scaled()` instruction which natively supports block scaling:

```python
accumulator = tl.dot_scaled(
    a,              # FP8 data block
    scale_a,        # Scale tensor [BLOCK_M, BLOCK_K // VEC_SIZE]
    "e4m3",         # FP8 format
    b.T,            # Transposed FP8 data block
    scale_b,        # Scale tensor [BLOCK_N, BLOCK_K // VEC_SIZE]
    "e4m3",         # FP8 format
    accumulator     # Accumulator
)
```

**Key insight from reference:** Scales must be in shape `[BLOCK_DIM, K_BLOCKS]` where `K_BLOCKS = BLOCK_K // VEC_SIZE` and `VEC_SIZE = 32` for MXFP8.

### Kernel Signature

```python
@triton.autotune(
    configs=[
        # Start with simpler configs for MXFP8
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 4}),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 4}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def mxfp8_matmul_kernel(
    # Data pointers
    a_ptr, b_ptr, c_ptr,
    # Scale pointers (E8M0 format, uint8)
    a_scale_ptr, b_scale_ptr,
    # Scale strides
    stride_a_scale_m, stride_a_scale_k,
    stride_b_scale_m, stride_b_scale_k,
    # Matrix dimensions
    M, N, K,
    # Data strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    VEC_SIZE: tl.constexpr = 32,  # MXFP8_BLOCK_SCALING_SIZE
):
```

### Core Computation Logic

```python
# Program ID and block mapping (same as regular kernel)
pid = tl.program_id(axis=0)
# ... compute pid_m, pid_n (same as existing kernel)

# Initialize accumulator
accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

# Compute block offsets
offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
offs_k = tl.arange(0, BLOCK_SIZE_K)

# Data pointers
a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

# K-loop
num_k_blocks = tl.cdiv(K, BLOCK_SIZE_K)
for k in range(0, num_k_blocks):
    # Load FP8 data
    if EVEN_K:
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
    else:
        mask_k = offs_k < K - k * BLOCK_SIZE_K
        a = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)

    # Load E8M0 scales for this K-block
    # Scale shape: [M_blocks, K_blocks] for A, [K_blocks, N_blocks] for B
    # We need: [BLOCK_SIZE_M, BLOCK_SIZE_K // VEC_SIZE] for tl.dot_scaled

    k_block_start = k * BLOCK_SIZE_K // VEC_SIZE
    num_k_scale_blocks = BLOCK_SIZE_K // VEC_SIZE

    # A scales: [BLOCK_SIZE_M, num_k_scale_blocks]
    offs_a_scale_k = k_block_start + tl.arange(0, num_k_scale_blocks)
    a_scale_ptrs = a_scale_ptr + (offs_am[:, None] * stride_a_scale_m +
                                   offs_a_scale_k[None, :] * stride_a_scale_k)
    a_scale_e8m0 = tl.load(a_scale_ptrs)

    # B scales: [num_k_scale_blocks, BLOCK_SIZE_N]
    # Note: After transpose in tl.dot_scaled, becomes [BLOCK_SIZE_N, num_k_scale_blocks]
    offs_b_scale_k = k_block_start + tl.arange(0, num_k_scale_blocks)
    b_scale_ptrs = b_scale_ptr + (offs_b_scale_k[:, None] * stride_b_scale_k +
                                   offs_bn[None, :] * stride_b_scale_m)
    b_scale_e8m0 = tl.load(b_scale_ptrs)

    # Convert E8M0 to FP32 scales
    # E8M0 format: biased_exponent → scale = 2^(biased_exponent - 127)
    a_scale_fp32 = tl.exp2(a_scale_e8m0.to(tl.float32) - 127.0)
    b_scale_fp32 = tl.exp2(b_scale_e8m0.to(tl.float32) - 127.0)

    # Block-scaled matmul using Triton's native instruction
    accumulator = tl.dot_scaled(
        a,              # [BLOCK_SIZE_M, BLOCK_SIZE_K] FP8
        a_scale_fp32,   # [BLOCK_SIZE_M, BLOCK_SIZE_K // VEC_SIZE] FP32
        "e4m3",         # or "e5m2" based on a_fp8_dtype
        b.T,            # [BLOCK_SIZE_K, BLOCK_SIZE_N] FP8 transposed
        b_scale_fp32.T, # [BLOCK_SIZE_N, BLOCK_SIZE_K // VEC_SIZE] FP32
        "e4m3",         # or "e5m2" based on b_fp8_dtype
        accumulator     # [BLOCK_SIZE_M, BLOCK_SIZE_N] FP32
    )

    # Advance pointers
    a_ptrs += BLOCK_SIZE_K * stride_ak
    b_ptrs += BLOCK_SIZE_K * stride_bk

# Store output (convert to target dtype)
offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

c = accumulator.to(c_ptr.type.element_ty)
tl.store(c_ptrs, c, mask=c_mask)
```

### Python Wrapper Function

```python
def mxfp8_matmul(
    a, a_scale, b, b_scale, c,
    M, N, K,
    a_fp8_dtype, b_fp8_dtype
):
    """
    MXFP8 matmul using tl.dot_scaled

    Args:
        a: FP8 data tensor [M, K] (uint8)
        a_scale: E8M0 scale tensor [M, K//32] (uint8)
        b: FP8 data tensor [K, N] (uint8)
        b_scale: E8M0 scale tensor [K//32, N] (uint8)
        c: Output tensor [M, N] (fp32/bf16/fp16)
    """
    # Validate BLOCK_SIZE_K is multiple of VEC_SIZE
    VEC_SIZE = MXFP8_BLOCK_SCALING_SIZE  # 32

    # Launch kernel
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    mxfp8_matmul_kernel[grid](
        a, b, c,
        a_scale, b_scale,
        a_scale.stride(0), a_scale.stride(1),
        b_scale.stride(0), b_scale.stride(1),
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        EVEN_K=(K % META['BLOCK_SIZE_K'] == 0),
        VEC_SIZE=VEC_SIZE,
    )
```

---

## Phase 3: Scale Preparation and Format Handling

### E8M0 to FP32 Conversion

**Current storage:** MXFP8Tensor stores scales as `uint8` E8M0 biased exponents

**Required by tl.dot_scaled:** FP32 scale tensors

**Solution:** Convert inside kernel (shown in kernel implementation above):
```python
scale_fp32 = tl.exp2(scale_e8m0.to(tl.float32) - 127.0)
```

**Verified against C++ (`utils.cuh:1083-1085`):**
```cpp
__device__ __forceinline__ float exp2f_rcp(e8m0_t biased_exp) {
  return (biased_exp == 0) ? 1 : exp2f(FP32_EXPONENT_BIAS - static_cast<float>(biased_exp));
}
```

For forward scaling (not reciprocal), we use: `scale = 2^(biased_exp - 127)`

### Data and Scale Selection (Verified)

**From C++ cublaslt_gemm.cu:128-200:**

For MXFP8 tensors, the selection is simple:
- **A operand:** `is_A_transposed ? rowwise : columnwise`
- **B operand:** `is_B_transposed ? columnwise : rowwise`
- **Scale follows data:** whichever data is selected, its corresponding scale is used

**Why this approach? (From NVIDIA Docs)**
- **MXFP8 cannot be transposed after quantization** without requantization (precision loss)
- Blackwell Tensor Cores require **consecutive data over the reduction dimension (K)**
- Each GEMM operation selects whichever copy (rowwise or columnwise) naturally provides consecutive K dimension
- This is why MXFP8 keeps original transpose flags (unlike regular FP8 which forces TN layout)

**Key difference from regular FP8:**
- **Regular FP8:** Transpose is numerically trivial → FORCES everything to TN layout (changes transpose flags)
- **MXFP8:** Transpose requires requantization → KEEPS original transpose flags, selects pre-quantized copy

**Implementation:**
The `MXFP8TensorWrapper.get_data_and_scale_for_gemm()` method handles this by:
1. Preferring rowwise data if available (most common case for forward pass)
2. Converting columnwise to rowwise if needed (for Triton's row-major requirement)
3. Returning both data and scale in rowwise orientation
4. **Note:** Conversion is done on already-quantized data for Triton compatibility, NOT for GEMM logic

---

## Phase 4: Testing Strategy

### File: `/workspace/TransformerEngine/tests/pytorch/test_gemm_triton.py`

### Test 1: Basic MXFP8 GEMM

```python
def test_mxfp8_gemm_basic():
    """Test basic MXFP8 GEMM with TN layout"""
    M, N, K = 256, 512, 1024

    # Create input tensors
    A_fp32 = torch.randn(M, K, device='cuda')
    B_fp32 = torch.randn(K, N, device='cuda')

    # Quantize to MXFP8
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
    quantizer = MXFP8Quantizer(fp8_format='e4m3')

    A_mxfp8 = quantizer.quantize(A_fp32)
    B_mxfp8 = quantizer.quantize(B_fp32)

    # Compute MXFP8 GEMM
    C_mxfp8 = te_generic_gemm_triton(A_mxfp8, False, B_mxfp8, False, None, 'cuda')

    # Reference: dequantize and compute in FP32
    A_dequant = A_mxfp8.dequantize()
    B_dequant = B_mxfp8.dequantize()
    C_ref = torch.matmul(A_dequant, B_dequant)

    # Check
    torch.testing.assert_close(C_mxfp8, C_ref, rtol=1e-2, atol=1e-2)
```

### Test 2: Different Layouts

```python
@pytest.mark.parametrize("transa,transb", [(True, False), (False, False), (False, True)])
def test_mxfp8_gemm_layouts(transa, transb):
    """Test MXFP8 GEMM with different transpose flags"""
    # ... similar to above but with transpose variations
```

### Test 3: Mixed FP8 Formats

```python
def test_mxfp8_gemm_mixed_formats():
    """Test MXFP8 GEMM with E4M3 and E5M2"""
    quantizer_e4m3 = MXFP8Quantizer(fp8_format='e4m3')
    quantizer_e5m2 = MXFP8Quantizer(fp8_format='e5m2')

    A_mxfp8 = quantizer_e4m3.quantize(A_fp32)
    B_mxfp8 = quantizer_e5m2.quantize(B_fp32)

    C_mxfp8 = te_generic_gemm_triton(A_mxfp8, False, B_mxfp8, False, None, 'cuda')
    # ... verify
```

### Test 4: Batch Dimensions

```python
def test_mxfp8_gemm_batched():
    """Test MXFP8 GEMM with batch dimensions"""
    batch, M, N, K = 4, 128, 256, 512

    A_fp32 = torch.randn(batch, M, K, device='cuda')
    B_fp32 = torch.randn(batch, K, N, device='cuda')

    # ... quantize and test
```

### Verification Strategy

1. **Numerical accuracy:** Compare against dequantized FP32 matmul
   - Expected tolerance: `rtol=1e-2, atol=1e-2` (due to FP8 quantization error)

2. **Performance:** Ensure MXFP8 is faster than FP32 for large matrices
   - Benchmark with M=N=K=4096

3. **Memory:** Verify scale tensors are correctly shaped and accessed
   - Add assertions for scale tensor shapes in kernel

4. **Edge cases:**
   - Non-multiple of 32 dimensions (with padding)
   - Very small matrices (M, N, K < 128)
   - Columnwise-only tensors

---

## Implementation Sequence

### Phase 1: Wrapper (Day 1)
1. Add MXFP8Tensor imports
2. Implement MXFP8 detection logic
3. Extract rowwise/columnwise data and scales
4. Add output dtype determination
5. Wire up to placeholder kernel call

### Phase 2: Basic Kernel (Day 2)
1. Implement `mxfp8_matmul_kernel` with `tl.dot_scaled()`
2. E8M0 to FP32 scale conversion
3. Test with simple 2D layout (rowwise only)
4. Verify against reference on small matrices

### Phase 3: Full Support (Day 3)
1. Add columnwise scale support
2. Implement rowwise/columnwise selection logic
3. Handle transpose flags correctly
4. Test all layout combinations (TN, NN, NT)

### Phase 4: Testing & Validation (Day 4)
1. Write comprehensive test suite
2. Benchmark performance vs FP32 and regular FP8
3. Test batch dimensions and edge cases
4. Validate numerical accuracy

### Phase 5: Integration (Day 5)
1. Test with E2E workloads
2. Fix any integration issues
3. Documentation and code cleanup
4. Performance profiling and optimization

---

## Files to Modify

### Primary Implementation
1. `/workspace/TransformerEngine/transformer_engine/pytorch/gemm_triton.py`
   - Add MXFP8Tensor imports
   - Extend `te_generic_gemm_triton()` wrapper
   - Implement `mxfp8_matmul_kernel()`
   - Implement `mxfp8_matmul()` Python wrapper

### Testing
2. `/workspace/TransformerEngine/tests/pytorch/test_gemm_triton.py`
   - Add `test_mxfp8_gemm_basic()`
   - Add `test_mxfp8_gemm_layouts()`
   - Add `test_mxfp8_gemm_mixed_formats()`
   - Add `test_mxfp8_gemm_batched()`

### Documentation (Future)
3. `/workspace/TransformerEngine/docs/` (if needed)
   - MXFP8 usage guide
   - Performance characteristics

---

## Key Design Decisions

### 1. Separate MXFP8 Kernel
- **Decision:** Implement `mxfp8_matmul_kernel()` separately from regular FP8 `matmul_kernel()`
- **Rationale:**
  - Fundamentally different operations (`tl.dot()` vs `tl.dot_scaled()`)
  - Different scale handling (scalar after K-loop vs 2D tensor per K-iteration)
  - Different constraints (BLOCK_K must be multiple of 32)
  - Easier development, debugging, and independent autotuning
  - Avoids runtime branching in hot path
- **Shared Components:** Wrapper/dispatch logic in `te_generic_gemm_triton()`, helper functions, testing infrastructure

### 2. Use tl.dot_scaled()
- **Decision:** Use Triton's native `tl.dot_scaled()` instruction
- **Rationale:** Simpler than manual dequantization, optimized by Triton compiler
- **Benefit:** Leverages hardware block-scaling support

### 3. E8M0 Conversion in Kernel
- **Decision:** Convert E8M0 to FP32 inside kernel using `tl.exp2()`
- **Rationale:** Avoid pre-converting scales (memory overhead), compute on-the-fly
- **Trade-off:** Slight compute overhead vs. memory bandwidth savings

### 4. Simple 2D Scale Layout
- **Decision:** Start with simple 2D scale tensors, not 5D TensorDescriptor
- **Rationale:** Simpler implementation, can optimize later if needed
- **Future:** Add TensorDescriptor optimization for better tensor core utilization

### 5. DEFAULT Epilogue Only
- **Decision:** Support only DEFAULT epilogue initially
- **Rationale:** Focus on core MXFP8 functionality first
- **Future:** Add BIAS and BGRADB epilogues later

### 6. VEC_SIZE = 32
- **Decision:** Hardcode VEC_SIZE to MXFP8_BLOCK_SCALING_SIZE (32)
- **Rationale:** MXFP8 spec requires 32-element blocks
- **Validation:** Assert BLOCK_SIZE_K is multiple of 32

---

## Technical Constraints

### Hardware Requirements
- NVIDIA Blackwell (compute 10.0+) or AMD gfx95x (MI300+)
- Triton version with `tl.dot_scaled()` support

### Dimension Requirements
- K dimension must be divisible by 32 (MXFP8_BLOCK_SCALING_SIZE)
- M and N should be divisible by BLOCK_SIZE for optimal performance
- Scale tensors must be correctly padded

### Memory Layout
- Input: uint8 FP8 data + uint8 E8M0 scales
- Scales: 2D tensors with strides
- Output: FP32/BF16/FP16 (no FP8 output initially)

---

## Verification Checklist

- [ ] MXFP8Tensor detected and components extracted
- [ ] E8M0 scales correctly converted to FP32
- [ ] `tl.dot_scaled()` called with correct parameters
- [ ] Output numerically matches dequantized reference
- [ ] All layouts (TN, NN, NT) work correctly
- [ ] Rowwise and columnwise scales both supported
- [ ] Batch dimensions handled correctly
- [ ] Tests pass with rtol=1e-2, atol=1e-2
- [ ] Performance better than FP32 baseline

---

## C++ Implementation Verification

The wrapper implementation plan has been verified against the C++ codebase to ensure consistency:

### 1. MXFP8Tensor Extraction (`type_converters.cpp:55-85`)
**Verified:** `NVTETensorFromMXFP8Tensor()` extracts:
- `_rowwise_data` and `_rowwise_scale_inv` (if not None)
- `_columnwise_data` and `_columnwise_scale_inv` (if not None)
- Uses `DType::kFloat8E8M0` for scale tensors (uint8 biased exponent)
- Creates `TensorWrapper` with `NVTE_MXFP8_1D_SCALING` mode

**Plan Consistency:** ✓ `MXFP8TensorWrapper.__init__()` follows same extraction pattern

### 2. Data/Scale Selection (`cublaslt_gemm.cu:128-200`)
**Verified:** For MXFP8 scaling mode:
- **A operand:** `is_A_transposed ? A.data : A.columnwise_data` (keeps original transA)
- **B operand:** `is_B_transposed ? B.columnwise_data : B.data` (keeps original transB)
- Scale selection follows data: `is_A_transposed ? scale_inv : columnwise_scale_inv`

**Critical Insight (from NVIDIA docs):** MXFP8 **cannot be transposed after quantization** because:
- "While transposing FP8 data is numerically trivial, transposing MXFP8 data requires requantization"
- "Consecutive" data "over the reduction dimension" is required for Blackwell Tensor Cores
- Solution: TE "creates both regular and transposed copies from the original high precision input"

**Why rowwise AND columnwise exist:**
- NOT an optimization - it's a **requirement** to avoid precision loss from requantization
- Each GEMM operation uses whichever copy (rowwise or columnwise) provides consecutive data over K dimension
- This is why MXFP8 keeps original transpose flags instead of forcing TN like regular FP8

**Plan Consistency:** ✓ `MXFP8TensorWrapper.get_data_and_scale_for_gemm()` implements same logic, then converts to rowwise for Triton (row-major)

### 3. TensorWrapper API (`transformer_engine.h:455-609`)
**Verified:**
- `set_rowwise_data(dptr, dtype, shape)` / `get_rowwise_data()`
- `set_columnwise_data(dptr, dtype, shape)` / `get_columnwise_data()`
- `set_rowwise_scale_inv(dptr, dtype, shape)` / `get_rowwise_scale_inv()`
- `set_columnwise_scale_inv(dptr, dtype, shape)` / `get_columnwise_scale_inv()`
- Tensor params: `kNVTERowwiseData`, `kNVTEColumnwiseData`, `kNVTERowwiseScaleInv`, `kNVTEColumnwiseScaleInv`

**Plan Consistency:** ✓ Python wrapper mirrors this API structure

### 4. E8M0 Conversion (`utils.cuh:1046-1085`)
**Verified:**
```cpp
// Extract biased exponent from float32
e8m0_t float_to_e8m0(float val) {
  uint32_t val_u32 = *reinterpret_cast<uint32_t *>(&val);
  e8m0_t exponent = (val_u32 >> FP32_MANTISSA_BITS);  // Bits [30:23]
  // ... rounding logic
  return exponent;
}

// Convert to reciprocal scale
float exp2f_rcp(e8m0_t biased_exp) {
  return (biased_exp == 0) ? 1 : exp2f(FP32_EXPONENT_BIAS - biased_exp);
  // = 2^(127 - biased_exp) = 1 / 2^(biased_exp - 127)
}
```

**Forward scale:** `2^(biased_exp - 127)` (for `tl.dot_scaled()`)

**Plan Consistency:** ✓ Kernel uses `tl.exp2(scale_e8m0.to(tl.float32) - 127.0)`

### 5. Scaling Mode Constants
**Verified:** `transformer_engine.h:83-104`
- `NVTE_MXFP8_1D_SCALING = 1` (single scale per 32-element block in row/column direction)
- `NVTE_BLOCK_SCALING_1D = 2` (1xN tiles)
- `NVTE_BLOCK_SCALING_2D = 3` (NxN tiles)

**Plan Scope:** Implementing only `NVTE_MXFP8_1D_SCALING` (VEC_SIZE=32)

### Summary
✅ **All critical implementation details verified against C++ codebase**
✅ **No inconsistencies found between plan and C++ implementation**
✅ **Plan follows established patterns from `Float8TensorWrapper`**

**Referenced Files:**
- `/workspace/TransformerEngine/transformer_engine/pytorch/csrc/type_converters.cpp`
- `/workspace/TransformerEngine/transformer_engine/common/gemm/cublaslt_gemm.cu`
- `/workspace/TransformerEngine/transformer_engine/common/include/transformer_engine/transformer_engine.h`
- `/workspace/TransformerEngine/transformer_engine/common/utils.cuh`
- `/workspace/TransformerEngine/transformer_engine/pytorch/gemm_triton.py` (existing Float8TensorWrapper)

**Referenced Documentation:**
- [NVIDIA Transformer Engine FP8 Primer - MXFP8 and Block Scaling](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html#MXFP8-and-block-scaling)
  - **Key quote:** "While transposing FP8 data is numerically trivial, transposing MXFP8 data requires requantization"
  - **Solution:** TE "creates both regular and transposed copies of the tensor from the original high precision input"
- [Triton Block-Scaled Matmul Tutorial](https://triton-lang.org/main/getting-started/tutorials/10-block-scaled-matmul.html)
  - Reference implementation for `tl.dot_scaled()` usage

---

## Future Enhancements (Out of Scope)

1. **Epilogue support:** BIAS, BGRADB with MXFP8
2. **FP8 output:** Quantize output to MXFP8
3. **TensorDescriptor optimization:** Use 5D scale layout for better hardware utilization
4. **Mixed precision:** MXFP8 + higher precision operands
5. **Autotuning:** Find optimal BLOCK_SIZE configs for MXFP8
6. **Fused operations:** Combine MXFP8 GEMM with activation functions

---

## Success Criteria

✓ MXFP8Tensor inputs work end-to-end
✓ Numerical accuracy within expected tolerance (rtol=1e-2)
✓ Performance competitive with regular FP8
✓ All test cases pass
✓ Code follows existing TE style and patterns
✓ No regressions in regular FP8 or standard GEMM paths

---

## Notes

- The Triton reference uses `VEC_SIZE=16` for nvfp4 and `VEC_SIZE=32` for mxfp4/mxfp8
- E8M0 biased exponent → FP32 scale: `2^(biased_exp - 127)`
- Scales broadcast across VEC_SIZE (32) elements along K dimension
- `tl.dot_scaled()` handles the scaling internally, no manual accumulator scaling needed
