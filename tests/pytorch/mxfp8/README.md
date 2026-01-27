# MXFP8 GEMM Tests

Tests for MXFP8 (Microscaling FP8) support in Transformer Engine's Triton GEMM backend.

## Test Files

### `test_mxfp8_gemm_basic.py`
Basic wrapper and import tests:
- Test MXFP8 class imports
- Test MXFP8TensorWrapper with regular tensors
- Test basic FP32 GEMM for reference

### `test_mxfp8_kernel_direct.py`
Direct kernel test with simulated data:
- Test `mxfp8_matmul()` kernel with simulated FP8 data
- Test E8M0 scale handling
- Validate kernel produces non-zero output

## Running Tests

### Run all MXFP8 tests with pytest
```bash
cd /workspace/TransformerEngine
pytest tests/pytorch/mxfp8/ -v
```

### Run individual test files
```bash
python tests/pytorch/mxfp8/test_mxfp8_gemm_basic.py
python tests/pytorch/mxfp8/test_mxfp8_kernel_direct.py
```

### Run with pytest for specific test
```bash
pytest tests/pytorch/mxfp8/test_mxfp8_kernel_direct.py::test_mxfp8_kernel_with_simulated_data -v
```

## Notes

- These tests use **simulated FP8 data** (not real quantization from MXFP8Quantizer)
- For full end-to-end testing, use actual `MXFP8Tensor` instances with proper quantization
- Tests require CUDA-capable GPU (MI300+, MI350, or NVIDIA Blackwell)
- VEC_SIZE = 32 (MXFP8_BLOCK_SCALING_SIZE)

## Implementation Details

The MXFP8 implementation includes:
- **MXFP8TensorWrapper**: Extracts rowwise/columnwise data and E8M0 scales
- **mxfp8_matmul_kernel()**: Triton kernel using `tl.dot_scaled()`
- **E8M0 conversion**: `scale = 2^(biased_exponent - 127)`
- **Dual storage**: Rowwise + columnwise copies (MXFP8 cannot be transposed after quantization)

See `/root/.claude/plans/ancient-chasing-robin.md` for full implementation plan.
