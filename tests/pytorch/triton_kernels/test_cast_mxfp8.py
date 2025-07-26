# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

import math
import os
import struct
import pytest
import torch

from transformer_engine.pytorch.constants import MXFP8_BLOCK_SCALING_SIZE
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
from transformer_engine.pytorch.triton_kernels.cast import te_dequantize_triton, te_quantize_triton
from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer
from transformer_engine.pytorch.triton_kernels.common import get_fp8_max, te_dtype_to_torch_dtype
from transformer_engine.pytorch.utils import round_up_to_nearest_multiple
import transformer_engine_torch as tex
from test_common import compare_results, fill_uniform, get_tolerances


FP32_MANTISSA_BITS = 23
FP32_EXPONENT_BIAS = 127

def float_to_e8m0(val: float) -> int:
    if math.isnan(val):
        return 0xFF
    if math.isinf(val):
        return 0xFE
    if val == 0.0:
        return 0x00

    val_u32 = struct.unpack('<I', struct.pack('<f', val))[0]
    exponent = (val_u32 >> FP32_MANTISSA_BITS) & 0xFF
    mantissa = val_u32 & 0x7FFFFF

    if (mantissa > 0 and exponent != 0xFE) and not (exponent == 0 and mantissa <= 0x400000):
        exponent += 1
    
    return exponent

def exp2f_rcp(biased_exp: int) -> float:
    if biased_exp == 0:
        return 1.0
    return math.pow(2.0, FP32_EXPONENT_BIAS - float(biased_exp))
 

def scale_block_torch(
    input_tensor: torch.Tensor,
    output_rowwise: torch.Tensor,
    output_columnwise: torch.Tensor,
    output_scale_rowwise: torch.Tensor,
    output_scale_columnwise: torch.Tensor,
    scale_idx: int,
    i_min: int,
    i_max: int,
    j_min: int,
    j_max: int,
    out_dtype: tex.DType
):
    #row-wise
    if j_max - j_min != 1:
        for i in range(i_min, i_max):
            block_input = input_tensor[i, j_min:j_max]
            amax = torch.max(torch.abs(block_input)).item()

            # Calculate scale
            biased_exponent = float_to_e8m0(amax * 1/get_fp8_max(out_dtype))
            scale_reciprocal = exp2f_rcp(biased_exponent)
            
            # Store the biased exponent in the output_scales tensor
            # output_scales should be a tensor of appropriate type
            output_scale_rowwise[i][scale_idx[1]] = biased_exponent
            block_output = block_input.clone() # Work on a clone to avoid modifying input_tensor directly

            # Apply scaling
            block_output *= scale_reciprocal

            # Store the results back into the output tensor
            output_rowwise[i, j_min:j_max] = block_output.to(te_dtype_to_torch_dtype(out_dtype))
        
    #column-wise
    if i_max - i_min != 1:
        for j in range(j_min, j_max):
            block_input = input_tensor[i_min:i_max, j]
            amax = torch.max(torch.abs(block_input)).item()

            # Calculate scale
            biased_exponent = float_to_e8m0(amax * 1/get_fp8_max(out_dtype))
            scale_reciprocal = exp2f_rcp(biased_exponent)
            
            # Store the biased exponent in the output_scales tensor
            # output_scales should be a tensor of appropriate type
            output_scale_columnwise[scale_idx[0]][j] = biased_exponent
            block_output = block_input.clone() # Work on a clone to avoid modifying input_tensor directly

            # Apply scaling
            block_output *= scale_reciprocal

            # Store the results back into the output tensor
            output_columnwise[i_min:i_max, j] = block_output.to(te_dtype_to_torch_dtype(out_dtype))

def compute_ref_x1_torch(
    input_tensor: torch.Tensor, # Input data
    output_rowwise: torch.Tensor,     # Main quantized output
    output_columnwise: torch.Tensor,     # Main quantized output
    output_scale_rowwise: torch.Tensor, # Scales for each block
    output_scale_columnwise: torch.Tensor, # Scales for each block
    block_size_Y: int,
    block_size_X: int,
    out_dtype: tex.DType
):
    cols = input_tensor.shape[-1] if len(input_tensor.shape) > 0 else 1
    rows = input_tensor.numel() // cols
    input_tensor_2d_view = input_tensor.reshape(rows, cols)
    output_rowwise_2d_view = output_rowwise.reshape(rows, cols)
    output_columnwise_2d_view = output_columnwise.reshape(rows, cols)
    blocks_Y = (rows + block_size_Y - 1) // block_size_Y
    blocks_X = (cols + block_size_X - 1) // block_size_X

    for ii in range(blocks_Y):
        i_min = ii * block_size_Y
        i_max = min((ii + 1) * block_size_Y, rows)
        for jj in range(blocks_X):
            j_min = jj * block_size_X
            j_max = min((jj + 1) * block_size_X, cols)
            scale_idx = (ii, jj)
            scale_block_torch(
                input_tensor=input_tensor_2d_view,
                output_rowwise=output_rowwise_2d_view,
                output_columnwise=output_columnwise_2d_view,
                output_scale_rowwise=output_scale_rowwise,
                output_scale_columnwise=output_scale_columnwise,
                scale_idx=scale_idx,
                i_min=i_min,
                i_max=i_max,
                j_min=j_min,
                j_max=j_max,
                out_dtype=out_dtype
            )

@pytest.mark.parametrize("shape", 
                         [
                        (1, 16),
                        (16, 48),
                        (65, 96),
                        (128, 128),
                        (256, 256),
                        (993, 512),
                        (256, 65536),
                        (2048, 6144),
                        (16384, 128),
                        (32768, 160),
                        (4096, 1632),
                        (8, 32, 1024),
                        (16, 8, 4, 512),
                        ])
@pytest.mark.parametrize("in_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("out_dtype", [tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2])
@pytest.mark.parametrize("block_sizes", [[1, 32], [32, 1], [32, 32]])
def test_quantize_dequantize_mxfp8(shape, in_dtype, out_dtype, block_sizes):
    if ((shape[-1] % MXFP8_BLOCK_SCALING_SIZE != 0) or (math.prod(shape[:-1]) % MXFP8_BLOCK_SCALING_SIZE != 0)):
        pytest.skip(f"Incorrect shape {shape} for MXFP8. Tensor dims must be divisible by {MXFP8_BLOCK_SCALING_SIZE}")
    input_tensor = fill_uniform(shape, dtype=in_dtype)
    triton_quantizer = MXFP8Quantizer(out_dtype)
    rowwise_scale_inv_ref = torch.zeros(
            round_up_to_nearest_multiple(math.prod(shape[:-1]), 128),
            round_up_to_nearest_multiple(shape[-1] // MXFP8_BLOCK_SCALING_SIZE, 4),
            dtype=torch.uint8,
            device="cuda",
        )
    columnwise_scale_inv_ref = torch.zeros(
                round_up_to_nearest_multiple(math.prod(shape[:-1]) // MXFP8_BLOCK_SCALING_SIZE, 4),
                round_up_to_nearest_multiple(shape[-1], 128),
                dtype=torch.uint8,
                device="cuda",
            )
    torch_out_dtype = te_dtype_to_torch_dtype(out_dtype)
    quantized_out_rowwise_ref = torch.empty(shape, dtype=torch_out_dtype, device="cuda")
    quantized_out_columnwise_ref = torch.empty_like(quantized_out_rowwise_ref)
    compute_ref_x1_torch(input_tensor, 
                        quantized_out_rowwise_ref, 
                        quantized_out_columnwise_ref,  
                        rowwise_scale_inv_ref, 
                        columnwise_scale_inv_ref, block_sizes[0], block_sizes[1], out_dtype)
    rowwise = block_sizes[1] != 1
    colwise = block_sizes[0] != 1
    out = None
    # If either rowwise is 1 or colwise is 1 but not at the same time
    if rowwise ^ colwise:
        out = triton_quantizer.make_empty(input_tensor.shape, dtype=in_dtype)
        # Make columnwise data none, since we won't be calculating that.
        if rowwise:
            out._columnwise_data = None
        # Make rowwise data none, since we won't be calculating that.
        if colwise:
            out._rowwise_data = None
    quantized_out_triton  = te_quantize_triton(input_tensor, quantizer=triton_quantizer, output=out)

    cmp = "te"
    atol_fp8, rtol_fp8 = get_tolerances(torch_out_dtype)
    if rowwise:
        compare_results(cmp, quantized_out_triton._rowwise_data.view(torch_out_dtype),  quantized_out_rowwise_ref, atol_fp8, rtol_fp8, "rowwise data doesn't match")
        compare_results("torch", quantized_out_triton._rowwise_scale_inv,  rowwise_scale_inv_ref, 0.0, 0.0, "rowwise scale inv doesn't match")
    if colwise:
        compare_results(cmp, quantized_out_triton._columnwise_data.view(torch_out_dtype),  quantized_out_columnwise_ref, atol_fp8, rtol_fp8, "columnwise data doesn't match")
        compare_results("torch", quantized_out_triton._columnwise_scale_inv,  columnwise_scale_inv_ref, 0.0, 0.0, "colwise scale inv doesn't match")
