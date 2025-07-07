# Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

import os
import pytest
import torch

from transformer_engine.pytorch.triton_kernels.cast import te_quantize_triton
from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer
import transformer_engine_torch as tex
from test_common import fill_uniform, get_tolerances

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

    atol, rtol = get_tolerances(in_dtype)
    assert torch.allclose(quantized_out_triton, quantized_out_tex, atol=atol, rtol=rtol), 'Quantized results do not match!'
    assert quantized_out_triton._transpose is not None, "Triton transpose is none!" 
    assert quantized_out_tex._transpose is not None, "TEX transpose is none!" 
    assert torch.allclose(
        quantized_out_triton._transpose, quantized_out_tex._transpose, atol=0.0, rtol=0.0
    ), 'Transposed quantized results do not match!'
    assert torch.allclose(quantized_out_triton._get_quantizer().scale, quantized_out_tex._get_quantizer().scale, atol=atol, rtol=rtol), 'Scale results do not match!'
    assert torch.allclose(quantized_out_triton._get_quantizer().amax, quantized_out_tex._get_quantizer().amax, atol=atol, rtol=rtol), 'AMAX results do not match!'