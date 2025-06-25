# Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

import pytest
import torch

from transformer_engine.pytorch.cpp_extensions.cast import quantize_triton, dequantize_triton
from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer
from transformer_engine.pytorch.triton_kernels.common import te_dtype_to_torch_dtype, torch_dtype_to_te_dtype
from transformer_engine.pytorch.triton_kernels.layernorm import te_layernorm_fwd_triton
from transformer_engine.pytorch.triton_kernels.norm_common import get_fwd_ln_sm_margin
import transformer_engine_torch as tex
from triton_kernels.test_common import compare_results, fill_uniform, get_tolerances

@pytest.mark.parametrize("shape", 
                         [
                        (128, 128),
                        # (256, 256),
                        # (768, 1024),
                        # (256, 65536),
                        # (2048, 12288),
                        # (65536, 128),
                        # (65536, 160),
                        # (16384, 1616),
                        # (1, 128),
                        # (1, 1296),
                        # (1, 16),
                        # (5, 160),
                        # (217, 256),
                        ])
@pytest.mark.parametrize("in_dtype", [torch.float32])
@pytest.mark.parametrize("out_dtype", [tex.DType.kFloat8E4M3])
@pytest.mark.parametrize("zero_centered_gamma", [False])
def test_layernorm_fwd_triton(shape, in_dtype, out_dtype, zero_centered_gamma):
    M, N = shape
    input_tensor = fill_uniform(shape, dtype=in_dtype)
    gamma = fill_uniform((N, ), dtype=in_dtype)
    beta = fill_uniform((N, ), dtype=in_dtype)
    epsilon = 1e-5
    output_tensor_ref = torch.empty((M, N), dtype=te_dtype_to_torch_dtype(out_dtype), device="cuda")

    scale_tensor = torch.rand(1, dtype=torch.float32, device='cuda') * 3.0 - 2.0
    amax_tensor = torch.zeros(1, dtype=torch.float32, device='cuda')
    quantizer = Float8Quantizer(scale=scale_tensor, amax=amax_tensor, fp8_dtype=out_dtype)

    output_tensor, mu, rsigma  = te_layernorm_fwd_triton(input_tensor, weight=gamma, bias=beta, eps=epsilon, zero_centered_gamma=zero_centered_gamma, out_dtype=out_dtype, quantizer=quantizer)
    output_tensor_ref, mu_ref, rsigma_ref = tex.layernorm_fwd(
        input_tensor,
        gamma,
        beta,
        epsilon,
        output_tensor_ref,
        quantizer,
        out_dtype,
        get_fwd_ln_sm_margin(),
        zero_centered_gamma,
    )

    atol, rtol = get_tolerances(in_dtype)
    assert torch.allclose(output_tensor, output_tensor_ref, atol=atol, rtol=rtol), 'Quantized results do not match!'
    assert torch.allclose(mu, mu_ref, atol=atol, rtol=rtol), 'Mu results do not match!'
    assert torch.allclose(rsigma, rsigma_ref, atol=atol, rtol=rtol), 'RSigma results do not match!'
