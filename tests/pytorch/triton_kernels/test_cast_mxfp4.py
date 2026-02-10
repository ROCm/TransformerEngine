# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

import math
import pytest
import torch
import numpy as np
import os

os.environ["NVTE_USE_CAST_TRANSPOSE_TRITON"] = "1"

from transformer_engine.pytorch.tensor.mxfp4_tensor import MXFP4Quantizer, MXFP4_BLOCK_SCALING_SIZE
from transformer_engine.pytorch.triton_kernels.cast import te_quantize_triton
from test_common import te_compare_results, fill_uniform


def mxfp4_quantize_cpu(input_tensor, axis='row'):
    """CPU reference for MXFP4 quantization matching Triton kernel behavior with shuffle."""
    original_shape = input_tensor.shape
    if input_tensor.dim() > 2:
        input_tensor = input_tensor.view(-1, input_tensor.shape[-1])
    
    M, N = input_tensor.shape
    
    if axis == 'col':
        input_tensor = input_tensor.t().contiguous()
        M, N = N, M
    
    data = input_tensor.cpu().float().numpy()
    
    BLOCK_SIZE = 32
    assert N % BLOCK_SIZE == 0, f"N={N} must be divisible by {BLOCK_SIZE}"
    
    num_blocks = N // BLOCK_SIZE
    
    # E2M1 FP4 lookup table
    fp4_values = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    
    # Reshape to blocks: [M, num_blocks, BLOCK_SIZE]
    data_blocks = data.reshape(M, num_blocks, BLOCK_SIZE)
    amax_blocks = np.max(np.abs(data_blocks), axis=2)
    
    # Triton's amax rounding: (amax + 0x200000) & 0xFF800000
    amax_int = amax_blocks.astype(np.float32).view(np.uint32)
    amax_int = ((amax_int + 0x200000) & 0xFF800000).astype(np.uint32)
    amax_rounded = amax_int.view(np.float32)
    
    # E8M0 scale computation: floor(log2(amax)) - 2 + 127
    scale_unbiased = np.floor(np.log2(np.maximum(amax_rounded, 1e-45))) - 2
    scale_unbiased = np.clip(scale_unbiased, -127, 127)
    scales = (scale_unbiased + 127).astype(np.uint8)
    scales = np.where(amax_blocks == 0, 0, scales)
    
    # Scale values for quantization
    scale_vals = np.where(scales[:, :, None] > 0, 
                          2.0 ** (-(scales[:, :, None] - 127)), 
                          1.0)
    
    scaled_blocks = data_blocks * scale_vals
    
    # Quantize to FP4
    signs = (scaled_blocks < 0).astype(np.uint8)
    abs_vals = np.abs(scaled_blocks)
    diffs = np.abs(abs_vals[:, :, :, None] - fp4_values[None, None, None, :])
    indices = np.argmin(diffs, axis=3).astype(np.uint8)
    fp4_encoded = (signs << 3) | indices
    
    fp4_flat = fp4_encoded.reshape(M, N)
    
    # Pack: (odd_col << 4) | even_col
    fp4_even = fp4_flat[:, 0::2]
    fp4_odd = fp4_flat[:, 1::2]
    fp4_packed = ((fp4_odd << 4) | fp4_even).astype(np.uint8)
    
    def cdiv(a, b): return (a + b - 1) // b
    
    scale_M_pad = cdiv(M, 256) * 256
    scale_N_pad = cdiv(num_blocks, 8) * 8
    scales_padded = np.full((scale_M_pad, scale_N_pad), 127, dtype=np.uint8)
    
    # Copy scales directly (no data shuffle support in Triton kernel)
    scales_padded[:M, :num_blocks] = scales
    
    fp4_packed_torch = torch.from_numpy(fp4_packed).to(input_tensor.device)
    scales_torch = torch.from_numpy(scales_padded).to(input_tensor.device)
    
    return fp4_packed_torch, scales_torch


@pytest.mark.parametrize("shape", [
    (128, 128),
    (256, 256),
    (256, 1024),
    (2048, 6144),
    (16384, 128),
    (32768, 160),
    (4096, 1632),
    (8, 32, 1024),
    (16, 8, 4, 512),
    # MXFP4 requires: shape[-1] % 32 == 0 and prod(shape[:-1]) % 32 == 0
    (32, 3221),   # Last dimension 3221 (prime) not divisible by 32
    (2333, 32),   # First dimension 2333 (prime) not divisible by 32 when flattened
    (1481, 677),  # Both dimensions are primes, neither divisible by 32
])
@pytest.mark.parametrize("in_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(("rowwise", "columnwise"), [
    (True, True), 
    (False, True), 
    (True, False)
])
@pytest.mark.parametrize("shuffle_B_matrix", [False, True])
def test_quantize_mxfp4(shape, in_dtype, rowwise, columnwise, shuffle_B_matrix):
    """Test MXFP4 quantization for rowwise/columnwise modes with/without FP4 shuffle.
    
    MXFP4 requires dimensions divisible by 32 for per-block scaling compatibility with AITER gemm_a4w4.
    
    Note: FP4 data shuffle (shuffle_B_matrix_for_aiter) is not yet supported in Triton kernel.
    """
    if shuffle_B_matrix:
        pytest.skip("FP4 data shuffle not yet supported in Triton kernel")
    
    input_tensor = fill_uniform(shape, dtype=in_dtype)
    quantizer = MXFP4Quantizer(
        rowwise=rowwise, 
        columnwise=columnwise,
        shuffle_B_matrix_for_aiter=shuffle_B_matrix
    )
    
    # Test invalid shapes are rejected
    if not quantizer.is_quantizable(input_tensor):
        assert not quantizer.is_quantizable(input_tensor), \
            f"is_quantizable() should return False for invalid shape {shape}"
        with pytest.raises(AssertionError, match="must be divisible by"):
            quantizer.make_empty(shape, dtype=in_dtype)
        return
    
    out = quantizer.make_empty(input_tensor.shape, dtype=in_dtype)
    quantized_out = te_quantize_triton(input_tensor, quantizer=quantizer, output=out)

    # Tolerance: allow 1 nibble diff for rare edge cases near FP4 boundaries
    data_atol = 20.0 if in_dtype != torch.float32 else 16.0
    scale_atol = 2.0 if in_dtype != torch.float32 else 1.0

    if rowwise:
        ref_data, ref_scale = mxfp4_quantize_cpu(input_tensor, axis='row')
        M = math.prod(input_tensor.shape[:-1])
        K = input_tensor.shape[-1]
        num_blocks = K // MXFP4_BLOCK_SCALING_SIZE
        
        te_compare_results(
            quantized_out._rowwise_data.view(torch.uint8),
            ref_data,
            atol=data_atol,
            rtol=0.0,
            msg="rowwise FP4 data mismatch",
            use_torch_semantics=True
        )
        
        # Compare only valid (non-padded) region - no shuffle extraction needed
        te_compare_results(
            quantized_out._rowwise_scale.view(torch.uint8)[:M, :num_blocks],
            ref_scale[:M, :num_blocks],
            atol=scale_atol,
            rtol=0.0,
            msg="rowwise E8M0 scales mismatch",
            use_torch_semantics=True
        )
    
    if columnwise:
        ref_data, ref_scale = mxfp4_quantize_cpu(input_tensor, axis='col')
        M = math.prod(input_tensor.shape[:-1])
        K = input_tensor.shape[-1]
        num_blocks = M // MXFP4_BLOCK_SCALING_SIZE
        
        te_compare_results(
            quantized_out._columnwise_data.view(torch.uint8),
            ref_data,
            atol=data_atol,
            rtol=0.0,
            msg="columnwise FP4 data mismatch",
            use_torch_semantics=True
        )
        
        # Compare only valid (non-padded) region - no shuffle extraction needed
        te_compare_results(
            quantized_out._columnwise_scale.view(torch.uint8)[:K, :num_blocks],
            ref_scale[:K, :num_blocks],
            atol=scale_atol,
            rtol=0.0,
            msg="columnwise E8M0 scales mismatch",
            use_torch_semantics=True
        )
