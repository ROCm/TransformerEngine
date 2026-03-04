# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

"""
Validates:
1. FP4 data encoding (E2M1 format, nibble-wise comparison)
2. E8M0 scale factors (±1 exponent tolerance for AMD/NVIDIA differences)
3. Statistical validation (allow small % mismatches for rounding ambiguity)
4. Edge cases (zeros, powers of 2, boundary values)
"""

import math
import pytest
import torch
import numpy as np
import os
from typing import Tuple

from transformer_engine.pytorch.tensor.mxfp4_tensor import (
    MXFP4Quantizer, 
    MXFP4_BLOCK_SCALING_SIZE
)
from transformer_engine.pytorch.triton_kernels.cast import te_quantize_triton
from test_common import fill_uniform


# ============================================================================
# CPU Reference Implementation
# ============================================================================
def shuffle_scales(scales: torch.Tensor):
    scales_shuffled = scales.clone()
    sm, sn = scales_shuffled.shape
    scales_shuffled = scales_shuffled.view(sm // 32, 2, 16, sn // 8, 2, 4, 1)
    scales_shuffled = scales_shuffled.permute(0, 3, 5, 2, 4, 1, 6).contiguous()
    scales_shuffled = scales_shuffled.view(sm // 32, sn * 32)
    return scales_shuffled

def un_shuffle_scales(scales_shuffled: torch.Tensor):
    scales = scales_shuffled.clone()
    sm, sn = scales.shape
    scales = scales.view(sm * 32, sn // 32)
    sm, sn = scales.shape
    scales = scales.view(sm // 32, sn // 8, 4, 16, 2, 2, 1)
    scales = scales.permute(0, 5, 3, 1, 4, 2, 6).contiguous()
    scales = scales.view(sm, sn)
    return scales

def mxfp4_quantize_cpu(
    input_tensor: torch.Tensor, 
    axis: str = 'row',
    SHUFFLE: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    CPU reference for MXFP4 quantization matching HIP kernel.
    
    Implements:
    - E8M0 scale: floor(log2(amax_rounded)) - 2 + 127
    - FP4 encoding: Threshold-based nearest-neighbor to E2M1 values
    - Packing: (odd_nibble << 4) | even_nibble
    
    Args:
        input_tensor: Input tensor (BF16 or FP32)
        axis: 'row' for rowwise, 'col' for columnwise
    
    Returns:
        (fp4_packed, scales_padded): Packed FP4 data and E8M0 scales
    """
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
    
    num_blocks = math.ceil(N / BLOCK_SIZE)
    
    # Reshape to blocks: [M, num_blocks, BLOCK_SIZE]
    data_blocks = data.reshape(M, num_blocks, BLOCK_SIZE)
    amax_blocks = np.max(np.abs(data_blocks), axis=2, keepdims=False)
    
    # === E8M0 Scale Computation (matches HIP compute_e8m0_scale) ===
    # Step 1: Round amax mantissa
    amax_int = amax_blocks.astype(np.float32).view(np.uint32)
    amax_int = ((amax_int + 0x200000) & 0xFF800000).astype(np.uint32)
    amax_rounded = amax_int.view(np.float32)
    
    # Step 2: Compute scale exponent
    with np.errstate(divide='ignore', invalid='ignore'):
        scale_unbiased = np.floor(np.log2(np.maximum(amax_rounded, 1e-45))) - 2
    scale_unbiased = np.clip(scale_unbiased, -127, 127)
    scales = (scale_unbiased + 127).astype(np.uint8)
    scales = np.where(amax_blocks == 0, 127, scales)
    
    # Quantization scale: 2^(-scale_unbiased)
    scale_vals = np.where(
        amax_blocks[:, :, None] > 0,
        2.0 ** (-(scales[:, :, None].astype(np.float32) - 127)),
        1.0
    )
    
    scaled_blocks = data_blocks * scale_vals
    
    # === FP4 Encoding (matches Triton kernel's threshold-based lookup) ===
    # E2M1 representable values: {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
    # Thresholds at midpoints: {0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0}
    
    signs = (scaled_blocks < 0).astype(np.uint8)
    abs_vals = np.abs(scaled_blocks)
    
    indices = np.zeros_like(abs_vals, dtype=np.uint8)
    indices = np.where(abs_vals >= 0.25, 1, indices)  # → 0.5
    indices = np.where(abs_vals >= 0.75, 2, indices)  # → 1.0
    indices = np.where(abs_vals >= 1.25, 3, indices)  # → 1.5
    indices = np.where(abs_vals >= 1.75, 4, indices)  # → 2.0
    indices = np.where(abs_vals >= 2.5,  5, indices)  # → 3.0
    indices = np.where(abs_vals >= 3.5,  6, indices)  # → 4.0
    indices = np.where(abs_vals >= 5.0,  7, indices)  # → 6.0
    
    # Combine sign and magnitude: [sign(1bit)][magnitude(3bits)]
    fp4_encoded = (signs << 3) | indices
    
    # Flatten to [M, N]
    fp4_flat = fp4_encoded.reshape(M, N)
    
    # === Pack two FP4 nibbles per byte ===
    # Layout: (odd_col << 4) | even_col
    fp4_even = fp4_flat[:, 0::2]
    fp4_odd = fp4_flat[:, 1::2]
    fp4_packed = ((fp4_odd << 4) | fp4_even).astype(np.uint8)
    
    fp4_packed_torch = torch.from_numpy(fp4_packed).to(input_tensor.device)
    scales_torch = torch.from_numpy(scales).to(input_tensor.device)
    
    if SHUFFLE:
        scales_pad = scales_torch
        scaleM = (M + 255) // 256 * 256
        scaleN_valid = (N + 31) // 32
        scaleN = (scaleN_valid + 7) // 8 * 8
        scales_pad = torch.empty(
            (scaleM, scaleN), dtype=scales_pad.dtype, device=scales_pad.device
        )
        scales_pad[:M, :scaleN_valid] = scales_torch[:M, :scaleN_valid]
        scales_pad = shuffle_scales(scales_pad)
        scales_pad = scales_pad.view(scales_pad.shape[0] * 32, -1)
        scales_torch = scales_pad
    
    return fp4_packed_torch, scales_torch


# ============================================================================
# Validation Helpers
# ============================================================================

def compare_fp4_data_nibblewise(
    test: torch.Tensor,
    ref: torch.Tensor,
    msg: str,
    max_mismatch_rate: float = 0.05,
) -> None:
    """
    Compare FP4 packed data nibble-by-nibble.
    
    Allows small % of mismatches for rounding ambiguity at FP4 boundaries.
    
    Args:
        test: Test output (uint8 packed)
        ref: Reference output (uint8 packed)
        msg: Error message prefix
        max_mismatch_rate: Maximum allowed mismatch rate (default 5%)
    """
    test_np = test.cpu().numpy().astype(np.uint8)
    ref_np = ref.cpu().numpy().astype(np.uint8)
    
    # Extract nibbles
    test_even = test_np & 0x0F
    test_odd = (test_np >> 4) & 0x0F
    ref_even = ref_np & 0x0F
    ref_odd = (ref_np >> 4) & 0x0F
    
    # Count mismatches
    even_mismatches = np.sum(test_even != ref_even)
    odd_mismatches = np.sum(test_odd != ref_odd)
    total_mismatches = even_mismatches + odd_mismatches
    total_nibbles = test_np.size * 2
    
    mismatch_rate = total_mismatches / total_nibbles
    exact_match_rate = 1.0 - mismatch_rate
    
    print(f"\n{msg}:")
    print(f"  Exact nibble match: {exact_match_rate:.2%}")
    print(f"  Mismatches: {total_mismatches}/{total_nibbles}")
    
    assert mismatch_rate <= max_mismatch_rate, (
        f"{msg}: Mismatch rate {mismatch_rate:.2%} exceeds {max_mismatch_rate:.2%}\n"
        f"  Mismatches: {total_mismatches}/{total_nibbles}"
    )


def compare_e8m0_scales(
    test: torch.Tensor,
    ref: torch.Tensor,
    msg: str,
    max_diff: int = 1,
    max_mismatch_rate: float = 0.02,
    max_outliers: int = 0,
) -> None:
    """
    Compare E8M0 scales allowing ±1 exponent difference.
    
    Args:
        test: Test scales (uint8)
        ref: Reference scales (uint8)
        msg: Error message prefix
        max_diff: Maximum allowed difference (default 1)
        max_mismatch_rate: Maximum allowed mismatch rate (default 2%)
        max_outliers: Maximum outliers with diff > max_diff (default 0)
    """
    test_np = test.cpu().numpy().astype(np.int16)
    ref_np = ref.cpu().numpy().astype(np.int16)
    
    diff = np.abs(test_np - ref_np)
    
    exact_matches = np.sum(diff == 0)
    within_1 = np.sum(diff <= max_diff)
    outliers = np.sum(diff > max_diff)
    total = test_np.size
    
    exact_rate = exact_matches / total
    within_1_rate = within_1 / total
    outlier_rate = outliers / total
    
    scale_bias = float(np.mean(test_np - ref_np))
    
    print(f"\n{msg}:")
    print(f"  Exact match: {exact_rate:.2%}")
    print(f"  Within ±{max_diff}: {within_1_rate:.2%}")
    print(f"  Outliers (diff > {max_diff}): {outliers}/{total} ({outlier_rate:.2%})")
    print(f"  Scale bias: {scale_bias:.3f}")
    
    assert outliers <= max_outliers, (
        f"{msg}: {outliers} outliers exceeds limit {max_outliers}\n"
        f"  Outlier rate: {outlier_rate:.2%}"
    )
    
    mismatch_rate = 1.0 - within_1_rate
    assert mismatch_rate <= max_mismatch_rate, (
        f"{msg}: Mismatch rate {mismatch_rate:.2%} exceeds {max_mismatch_rate:.2%}"
    )


# ============================================================================
# Test Cases
# ============================================================================

@pytest.mark.parametrize("shape", [
    (128, 128),
    (256, 256),
    (1024, 2048),
    (2048, 6144),
    (4096, 4096),
    (16384, 128),
    (32768, 160),
    (8, 32, 1024),
    (16, 8, 4, 512),
])
@pytest.mark.parametrize("in_dtype", [
    torch.bfloat16,
    torch.float32,
])
@pytest.mark.parametrize(("rowwise", "columnwise", "shuffle_B_matrix_for_aiter"), [
    (True, False, False),
    (False, True, False),
    (True, True, False),
    (True, False, True),
    (False, True, True),
    (True, True, True),
])
def test_quantize_mxfp4_standard(shape, in_dtype, rowwise, columnwise, shuffle_B_matrix_for_aiter):
    """Standard MXFP4 quantization with statistical validation."""
    input_tensor = fill_uniform(shape, dtype=in_dtype)
    
    quantizer = MXFP4Quantizer(
        rowwise=rowwise,
        columnwise=columnwise,
        shuffle_B_matrix_for_aiter=shuffle_B_matrix_for_aiter
    )
    
    quantized_out = te_quantize_triton(input_tensor, quantizer=quantizer)
    
    M = math.prod(input_tensor.shape[:-1])
    K = input_tensor.shape[-1]
    
    if rowwise:
        ref_data, ref_scale = mxfp4_quantize_cpu(input_tensor, axis='row', SHUFFLE=shuffle_B_matrix_for_aiter)
        num_blocks = math.ceil(K / MXFP4_BLOCK_SCALING_SIZE)
        
        compare_fp4_data_nibblewise(
            quantized_out._rowwise_data.view(torch.uint8),
            ref_data,
            msg=f"Rowwise FP4 ({shape}, {in_dtype})",
            max_mismatch_rate=0.05,
        )
        y1_scales_triton = quantized_out._rowwise_scale_inv.view(torch.uint8)
        y1_scales_torch = ref_scale
        if shuffle_B_matrix_for_aiter:
            y1_scales_triton = un_shuffle_scales(
                y1_scales_triton.view(y1_scales_triton.shape[0] // 32, -1)
            )
            y1_scales_torch = un_shuffle_scales(
                y1_scales_torch.view(y1_scales_torch.shape[0] // 32, -1)
            )
        
        compare_e8m0_scales(
            y1_scales_triton[:M, :num_blocks],
            y1_scales_torch[:M, :num_blocks],
            msg=f"Rowwise E8M0 ({shape}, {in_dtype})",
            max_diff=1,
            max_mismatch_rate=1e-4,  # 0.01% mismatch rate: allows up to 0.01% of scales to differ by >±1
            max_outliers=1,  # Up to 1 absolute outlier allowed for hardware rounding differences
        )
    
    if columnwise:
        ref_data, ref_scale = mxfp4_quantize_cpu(input_tensor, axis='col', SHUFFLE=shuffle_B_matrix_for_aiter)
        num_blocks = math.ceil(M / MXFP4_BLOCK_SCALING_SIZE)
        
        compare_fp4_data_nibblewise(
            quantized_out._columnwise_data.view(torch.uint8),
            ref_data,
            msg=f"Columnwise FP4 ({shape}, {in_dtype})",
            max_mismatch_rate=0.05,
        )
        
        y1_scales_triton = quantized_out._columnwise_scale_inv.view(torch.uint8)
        y1_scales_torch = ref_scale
        if shuffle_B_matrix_for_aiter:
            y1_scales_triton = un_shuffle_scales(
                y1_scales_triton.view(y1_scales_triton.shape[0] // 32, -1)
            )
            y1_scales_torch = un_shuffle_scales(
                y1_scales_torch.view(y1_scales_torch.shape[0] // 32, -1)
            )
        compare_e8m0_scales(
            y1_scales_triton[:K, :num_blocks],
            y1_scales_torch[:K, :num_blocks],
            msg=f"Columnwise E8M0 ({shape}, {in_dtype})",
            max_diff=1,
            max_mismatch_rate=1e-4,  # 0.01% mismatch rate: allows up to 0.01% of scales to differ by >±1
            max_outliers=1,  # Up to 1 absolute outlier allowed for hardware rounding differences
        )


@pytest.mark.parametrize("edge_case", [
    "all_zeros",
    "very_small",
    "very_large",
])
def test_quantize_mxfp4_edge_cases(edge_case):
    """Test edge cases for E8M0 scale computation and FP4 encoding."""
    shape = (256, 1024)
    M, K = shape
    
    if edge_case == "all_zeros":
        input_tensor = torch.zeros(shape, dtype=torch.bfloat16, device='cuda')
        
    elif edge_case == "very_small":
        input_tensor = torch.full(shape, 1e-38, dtype=torch.bfloat16, device='cuda')
        
    elif edge_case == "very_large":
        input_tensor = torch.full(shape, 3e38, dtype=torch.bfloat16, device='cuda')
    
    quantizer = MXFP4Quantizer(rowwise=True, columnwise=False)
    quantized_out = te_quantize_triton(input_tensor, quantizer=quantizer)
    
    ref_data, ref_scale = mxfp4_quantize_cpu(input_tensor, axis='row', SHUFFLE=False)
    num_blocks = math.ceil(K / MXFP4_BLOCK_SCALING_SIZE)
    
    if edge_case == "all_zeros":
        scales = quantized_out._rowwise_scale_inv.view(torch.uint8)[:M, :num_blocks]
        assert torch.all(scales == 127), (
            f"Zero blocks should have scale=127, got: {scales.unique()}"
        )
        
        data = quantized_out._rowwise_data.view(torch.uint8)
        assert torch.all(data == 0), "Zero data should encode to FP4=0"
    else:
        compare_fp4_data_nibblewise(
            quantized_out._rowwise_data.view(torch.uint8),
            ref_data,
            msg=f"Edge case: {edge_case}",
            max_mismatch_rate=0.05,
        )
        
        compare_e8m0_scales(
            quantized_out._rowwise_scale_inv.view(torch.uint8)[:M, :num_blocks],
            ref_scale[:M, :num_blocks],
            msg=f"Edge case scales: {edge_case}",
            max_diff=1,
            max_mismatch_rate=1e-4,  # 0.01% mismatch rate: allows up to 0.01% of scales to differ by >±1
            max_outliers=1,  # Up to 1 absolute outlier allowed for hardware rounding differences
        )


@pytest.mark.parametrize("invalid_shape", [
    (32, 3221),
    (2333, 32),
    (1481, 677),
    (31, 64),
    (64, 31),
])
def test_quantize_mxfp4_invalid_shapes(invalid_shape):
    """Test that invalid shapes are rejected (M and K must be divisible by 32)."""
    input_tensor = fill_uniform(invalid_shape, dtype=torch.bfloat16)
    quantizer = MXFP4Quantizer(rowwise=True, columnwise=False)
    
    assert not quantizer.is_quantizable(input_tensor), (
        f"Shape {invalid_shape} should not be quantizable"
    )
    
    with pytest.raises(AssertionError, match="must be divisible by"):
        quantizer.make_empty(invalid_shape, dtype=torch.bfloat16)