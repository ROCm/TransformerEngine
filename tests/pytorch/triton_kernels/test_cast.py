# Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

import os
import pytest
import torch

from transformer_engine.pytorch.triton_kernels.cast import te_quantize_triton
from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer
from transformer_engine.pytorch.triton_kernels.common import te_dtype_to_torch_dtype
import transformer_engine_torch as tex
from test_common import compare_results, fill_uniform, get_tolerances

@pytest.mark.parametrize("shape", 
                         [
                        (16 ),
                        (16000 ),
                        (128, 128),
                        (256, 256),
                        (768, 1024),
                        (256, 65536),
                        (2048, 12288),
                        (65536, 128),
                        (65536, 160),
                        (16384, 1616),
                        (1, 128),
                        (1, 1296),
                        (1, 16),
                        (5, 160),
                        (5, 4, 3, 160),
                        (217, 256)
                        ])
@pytest.mark.parametrize("in_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("out_dtype", [tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2])
def test_quantize(shape, in_dtype, out_dtype):
    input_tensor = fill_uniform(shape, dtype=in_dtype)

    scale_tensor = torch.rand(1, dtype=torch.float32, device='cuda') * 3.0 - 2.0
    amax_tensor = torch.zeros(1, dtype=torch.float32, device='cuda')
    triton_quantizer = Float8Quantizer(scale=scale_tensor, amax=amax_tensor, fp8_dtype=out_dtype)

    quantized_out_triton  = te_quantize_triton(input_tensor, quantizer=triton_quantizer)
    
    tex_quantizer = Float8Quantizer(scale=scale_tensor, amax=amax_tensor, fp8_dtype=out_dtype)
    quantized_out_tex = tex.quantize(input_tensor, tex_quantizer)
    torch_out_dtype = te_dtype_to_torch_dtype(out_dtype)
    
    atol_q, rtol_q = get_tolerances(torch_out_dtype)
    cmp = "te"
    compare_results(
        cmp,
        quantized_out_triton._data.view(torch_out_dtype),
        quantized_out_tex._data.view(torch_out_dtype),
        atol_q,
        rtol_q,
        lambda msg: f"triton does not match tex <-> hip\n\n{msg}\n",
    )
    assert quantized_out_triton._transpose is not None, "Triton transpose is none!" 
    assert quantized_out_tex._transpose is not None, "TEX transpose is none!" 
    compare_results(
        cmp,
        quantized_out_triton._transpose.view(torch_out_dtype),
        quantized_out_tex._transpose.view(torch_out_dtype),
        atol_q,
        rtol_q,
        lambda msg: f"triton does not match tex <-> hip\n\n{msg}\n",
    )
    
    atol_scale, rtol_scale = get_tolerances(torch.float32)
    assert torch.allclose(quantized_out_triton._get_quantizer().scale, quantized_out_tex._get_quantizer().scale, atol=atol_scale, rtol=rtol_scale), 'Scale results do not match!'
    assert torch.allclose(quantized_out_triton._get_quantizer().amax, quantized_out_tex._get_quantizer().amax, atol=atol_scale, rtol=rtol_scale), 'AMAX results do not match!'


@pytest.mark.parametrize("t_shape",
                         [
                        (16 ),
                        (768, 1024),
                        (1, 128),
                        (5, 160),
                        (5, 4, 3, 160),
                        ])
@pytest.mark.parametrize("fp8_dtype", [tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2])
def test_quantize_bad_transpose(t_shape, fp8_dtype):
    """
    Non-regression test for gh-13121, testing whether te_quantize_triton
    correctly dispatches based off of transpose buffer validity.
    """
    # The input type and shape are arbitrary, but we choose only one so as to
    # avoid unnecessarily expanding the test parameter space.
    in_dtype = torch.float32
    shape = (128, 128)
    input_tensor = fill_uniform(shape, dtype=in_dtype)
    output_tensor = torch.empty(shape, dtype=in_dtype, device='cuda')

    scale_tensor = torch.rand(1, dtype=torch.float32, device='cuda') * 3.0 - 2.0
    amax_tensor = torch.zeros(1, dtype=torch.float32, device='cuda')
    quantizer = Float8Quantizer(scale=scale_tensor, amax=amax_tensor, fp8_dtype=fp8_dtype)

    quantized_output = quantizer(output_tensor)
    quantized_output._transpose_invalid = True
    quantized_output._transpose = torch.empty(t_shape, device='cuda')

    te_quantize_triton(input_tensor, quantizer=quantizer, output=quantized_output)
