# Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

import pytest
import torch
import triton
import triton.language as tl
import os

from transformer_engine.pytorch.triton_kernels.rmsnorm_triton import te_rmsnorm_fwd_noalloc_triton, te_rmsnorm_fwd_inf_triton
import transformer_engine_torch as tex

def sizeof(in_dtype):
    if in_dtype == torch.float32:
        return 4
    elif in_dtype == torch.float16 or in_dtype == torch.bfloat16:
        return 2
    elif in_dtype == torch.float8_e4m3fnuz or in_dtype == float8_e5m2fnuz:
        return 1
    return 1

def get_tolerances(in_dtype):
    if in_dtype == torch.float32:
        return 1e-6, 5e-6
    elif in_dtype == torch.float16:
        return 1e-5, 1e-3
    elif in_dtype == torch.bfloat16:
        return 1e-5, 1e-2
    elif in_dtype == torch.float8_e4m3fnuz or in_dtype == torch.float8_e5m2fnuz:
        return 1e-2, 1e-2
    else:
        raise RuntimeError("Invalid type")
    return 0, 0

# matrix size from tests/cpp/operator/test_rmsnorm.cu
@pytest.mark.parametrize("M, N", 
                         [(2048, 4096),
                          (768, 2048),
                          (256, 1024),
                          (128, 768),
                          (64, 512),
                          (173, 409),
                          (71, 3571),
                          (29, 17389),
                        ])
@pytest.mark.parametrize("in_dtype", [torch.float32, torch.float16, torch.bfloat16])
#TODO add fp8/bf8 once fp8 triton kernels are available
@pytest.mark.parametrize("out_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("zero_center_gamma", [False])
def test_cast_tranpose_triton(M, N, in_dtype, out_dtype, zero_center_gamma):
    if sizeof(in_dtype) < sizeof(out_dtype):
        pytest.skip("size of input dtype < size of output dtype")
    if (in_dtype==torch.float16 and out_dtype==torch.bfloat16) or (in_dtype==torch.bfloat16 and out_dtype==torch.float16):
        pytest.skip("hipified rmsnorm kernel does not support mixing fp16 and bf16")
    ## Uniform distribution between [-2.0, 1.0]
    input_tensor = torch.rand(M, N, dtype=torch.float32, device='cuda') * 3.0 - 2.0
    input_tensor = input_tensor.to(in_dtype)
    gamma_tensor = torch.rand(N, dtype=torch.float32, device='cuda') * 3.0 - 2.0
    gamma_tensor = gamma_tensor.to(in_dtype)

    epsilon = 1e-5
    # run the triton path
    ln_out_triton = torch.empty(M, N, dtype=out_dtype, device='cuda')
    ln_out_triton, rsigma_triton = te_rmsnorm_fwd_noalloc_triton(input_tensor, gamma_tensor, epsilon, ln_out_triton)
    # run the reference hipified kernel path
    fwd_ln_sm_margin = int(os.getenv("NVTE_FWD_LAYERNORM_SM_MARGIN", "0"))
    ln_out_hipified = torch.empty(M, N, dtype=out_dtype, device='cuda')
    ln_out_hipified, rsigma_hipified = tex.rmsnorm_fwd_noalloc(input_tensor, gamma_tensor, ln_out_hipified, epsilon, fwd_ln_sm_margin, zero_center_gamma)
    atol, rtol = get_tolerances(out_dtype)
    assert torch.allclose(ln_out_triton, ln_out_hipified, atol=atol, rtol=rtol), 'ln_out does not match'
    # rsigma is of type fp32
    assert torch.allclose(rsigma_triton, rsigma_hipified, atol=1e-6, rtol=5e-5), 'rsigma does not match'

 
