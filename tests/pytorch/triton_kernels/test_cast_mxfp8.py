# Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

import math
import os
import pytest
import torch

from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
from transformer_engine.pytorch.triton_kernels.cast import te_dequantize_triton, te_quantize_triton
from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer
from transformer_engine.pytorch.triton_kernels.common import get_fp8_max, te_dtype_to_torch_dtype
import transformer_engine_torch as tex
import struct
from test_common import compare_results, fill_uniform, get_tolerances

FP32_MANTISSA_BITS = 23
FP32_EXPONENT_BIAS = 127.0

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
    
    return min(exponent, 0xFF)

def exp2f_rcp(biased_exp: int) -> float:
    if biased_exp == 0:
        return 1.0
    return math.pow(2.0, FP32_EXPONENT_BIAS - float(biased_exp))

def scale_block_torch(
    input_tensor: torch.Tensor,
    output_c: torch.Tensor,
    output_scales: torch.Tensor, # This will now be a tensor
    scale_idx: int,
    i_min: int,
    i_max: int,
    j_min: int,
    j_max: int,
    cols: int,
):
    # Create a view of the original input that mimics the 2D structure
    input_2d_view = input_tensor.view(-1, cols)
    block_input = input_2d_view[i_min:i_max, j_min:j_max]

    # Find the absolute maximum value in the block using torch.max
    amax = torch.max(torch.abs(block_input)).item() # .item() gets Python scalar from 0-dim tensor

    # Calculate scale
    biased_exponent = float_to_e8m0(amax * get_fp8_max())
    scale_reciprocal = exp2f_rcp(biased_exponent)
    
    # Store the biased exponent in the output_scales tensor
    # output_scales should be a tensor of appropriate type, e.g., torch.int8 or torch.float32
    output_scales[scale_idx] = biased_exponent

    # Quantize elements in the block
    block_output = block_input.clone() # Work on a clone to avoid modifying input_tensor directly

    # Apply scaling
    block_output *= scale_reciprocal

    # Store the results back into the output_c tensor
    # Ensure output_c has the correct shape or can be indexed properly
    output_c_2d_view = output_c.view(-1, cols)
    output_c_2d_view[i_min:i_max, j_min:j_max] = block_output

def compute_ref_x1_torch(
    input_tensor: torch.Tensor, # Input data
    output_c: torch.Tensor,     # Main quantized output
    output_scales: torch.Tensor, # Scales for each block
    rows: int,
    cols: int,
    block_size_Y: int,
    block_size_X: int,
    scales_stride: int,
):
    """
    Computes a reference output (quantized) and optionally a bias sum
    by iterating over blocks and calling scale_block_torch.
    """
    blocks_Y = (rows + block_size_Y - 1) // block_size_Y
    blocks_X = (cols + block_size_X - 1) // block_size_X

    for ii in range(blocks_Y):
        i_min = ii * block_size_Y
        i_max = min((ii + 1) * block_size_Y, rows)
        for jj in range(blocks_X):
            j_min = jj * block_size_X
            j_max = min((jj + 1) * block_size_X, cols)
            scale_idx = ii * scales_stride + jj
            scale_block_torch (
                input_tensor=input_tensor,
                output_c=output_c,
                output_scales=output_scales,
                scale_idx=scale_idx,
                i_min=i_min,
                i_max=i_max,
                j_min=j_min,
                j_max=j_max,
                cols=cols,
            )

@pytest.mark.parametrize("shape", 
                         [
                        (128, 128),
                        (256, 256),
                        (256, 65536),
                        (2048, 6144),
                        (16384, 128),
                        (32768, 160),
                        (4096, 1632),
                        
                        ])
@pytest.mark.parametrize("in_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("out_dtype", [tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2])
def test_quantize(shape, in_dtype, out_dtype):
    input_tensor = fill_uniform(shape, dtype=in_dtype)
    triton_quantizer = MXFP8Quantizer(out_dtype)
    tex_quantizer = MXFP8Quantizer(out_dtype)

    quantized_out_triton  = te_quantize_triton(input_tensor, quantizer=triton_quantizer)
    dequantized_out_triton = te_dequantize_triton(quantized_out_triton, dtype=in_dtype)
    dequantized_out_colwise_triton = te_dequantize_triton(quantized_out_triton, dtype=in_dtype, use_rowwise_scaling=False)
    compare_results("te", dequantized_out_triton, input_tensor, 0.13, 0.01, 'Dequantized and original results do not match!')
    compare_results("te", dequantized_out_colwise_triton, dequantized_out_triton, 0.01, 0.01, 'Dequantized colwise and dequantized rowwise results do not match!')
    # quantized_out_tex = tex.quantize(input_tensor, quantizer=tex_quantizer)
    # torch_out_dtype = te_dtype_to_torch_dtype(out_dtype)
    
    # atol_q, rtol_q = get_tolerances(torch_out_dtype)
    # cmp = "te"
    # compare_results(
    #     cmp,
    #     quantized_out_triton._data.view(torch_out_dtype),
    #     quantized_out_tex._data.view(torch_out_dtype),
    #     atol_q,
    #     rtol_q,
    #     lambda msg: f"triton does not match tex <-> hip\n\n{msg}\n",
    # )
    # assert quantized_out_triton._transpose is not None, "Triton transpose is none!" 
    # assert quantized_out_tex._transpose is not None, "TEX transpose is none!" 
    # compare_results(
    #     cmp,
    #     quantized_out_triton._transpose.view(torch_out_dtype),
    #     quantized_out_tex._transpose.view(torch_out_dtype),
    #     atol_q,
    #     rtol_q,
    #     lambda msg: f"triton does not match tex <-> hip\n\n{msg}\n",
    # )
    
    # atol_scale, rtol_scale = get_tolerances(torch.float32)
    # assert torch.allclose(quantized_out_triton._get_quantizer().amax, quantized_out_tex._get_quantizer().amax, atol=atol_scale, rtol=rtol_scale), 'AMAX results do not match!'