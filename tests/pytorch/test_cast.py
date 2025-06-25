# Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

import pytest
import torch

from transformer_engine.pytorch.cpp_extensions.cast import quantize_triton, dequantize_triton
from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer
import transformer_engine_torch as tex
from triton_kernels.test_common import compare_results, fill_uniform, get_tolerances

@pytest.mark.parametrize("shape", 
                         [
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
                        (217, 256),
                        ])
@pytest.mark.parametrize("in_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("out_dtype", [tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2])
def test_quantize(shape, in_dtype, out_dtype):
    input_tensor = fill_uniform(shape, dtype=in_dtype)

    scale_tensor = torch.rand(1, dtype=torch.float32, device='cuda') * 3.0 - 2.0
    amax_tensor = torch.zeros(1, dtype=torch.float32, device='cuda')
    quantizer = Float8Quantizer(scale=scale_tensor, amax=amax_tensor, fp8_dtype=out_dtype)

    output_tensor  = quantize_triton(input_tensor, quantizer=quantizer)
    
    quantizer2 = Float8Quantizer(scale=scale_tensor, amax=amax_tensor, fp8_dtype=out_dtype)
    te_quantized_out = quantizer2.quantize(input_tensor)
    print("Actual Output Tensor (first few elements):\n", te_quantized_out[:5, :5] if te_quantized_out.ndim >= 2 else te_quantized_out[:5])
    print(f"Output Tensor Shape: {te_quantized_out.shape}, Dtype: {te_quantized_out.dtype}")

    atol, rtol = get_tolerances(in_dtype)
    assert torch.allclose(output_tensor, te_quantized_out, atol=atol, rtol=rtol), 'Quantized results do not match!'

    dequantized_output_triton = dequantize_triton(output_tensor, in_dtype)
    te_dequantized_out = te_quantized_out.dequantize(dtype=in_dtype)
    assert torch.allclose(dequantized_output_triton, te_dequantized_out, atol=atol, rtol=rtol), 'Dequantized results do not match!'
   