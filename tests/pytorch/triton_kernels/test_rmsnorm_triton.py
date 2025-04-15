# Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

import pytest
import torch
import triton
import triton.language as tl
import os

from transformer_engine.pytorch.triton_kernels.rmsnorm_triton import te_rmsnorm_fwd_fp8_noalloc_triton, te_rmsnorm_fwd_noalloc_triton, te_rmsnorm_fwd_inf_triton
from transformer_engine.pytorch import cpp_extensions as tex

def get_te_dtype(dtype):
    if dtype == torch.float32:
        return tex.DType.kFloat32
    if dtype == torch.float16:
        return tex.DType.kFloat16
    if dtype == torch.bfloat16:
        return tex.DType.kBFloat16
    if dtype == torch.float8_e4m3fnuz or dtype == torch.float8_e4m3fn:
        return tex.DType.kFloat8E4M3
    if dtype == torch.float8_e5m2fnuz or dtype == torch.float8_e5m2:
        return tex.DType.kFloat8E5M2

def sizeof(in_dtype):
    if in_dtype == torch.float32:
        return 4
    elif in_dtype == torch.float16 or in_dtype == torch.bfloat16:
        return 2
    elif (in_dtype == torch.float8_e4m3fnuz or in_dtype == torch.float8_e5m2fnuz
          or in_dtype == torch.float8_e4m3fn or in_dtype == torch.float8_e5m2):
        return 1
    return 1

def get_tolerances(in_dtype):
    if in_dtype == torch.float32:
        return 1e-6, 5e-6
    elif in_dtype == torch.float16:
        return 1e-5, 1e-3
    elif in_dtype == torch.bfloat16:
        return 1e-5, 1e-2
    elif (in_dtype == torch.float8_e4m3fnuz or in_dtype == torch.float8_e5m2fnuz
          or in_dtype == torch.float8_e4m3fn or in_dtype == torch.float8_e5m2):
        #TODO: different tolerances for FNUZ and OCP
        return 1e-2, 1e-2
    else:
        raise RuntimeError("Invalid type")

test_shapes = [(2048, 4096),
               (768, 2048),
               (256, 1024),
               (128, 768),
               (64, 512),
               (173, 409),
               (71, 3571),
               (29, 17389),]

test_dtypes = [torch.float32, torch.float16, torch.bfloat16]

all_boolean = [True, False]

# matrix size from tests/cpp/operator/test_rmsnorm.cu
@pytest.mark.parametrize("M, N", test_shapes)
@pytest.mark.parametrize("in_dtype", test_dtypes)
#TODO add fp8/bf8 once fp8 triton kernels are available
@pytest.mark.parametrize("out_dtype", test_dtypes)
@pytest.mark.parametrize("zero_centered_gamma", all_boolean)
def test_rmsnorm_fwd_fp8_noalloc_triton(M, N, in_dtype, out_dtype, zero_centered_gamma):
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
    ln_out_triton, rsigma_triton = te_rmsnorm_fwd_fp8_noalloc_triton(input_tensor, gamma_tensor, epsilon, ln_out_triton, out_dtype, zero_centered_gamma)
    # run the reference hipified kernel path
    scale_tensor = torch.empty(0, dtype=torch.float32, device='cuda')
    amax_tensor = torch.zeros(0, dtype=torch.float32, device='cuda')
    scale_inv_tensor = torch.empty(0, dtype=torch.float32, device='cuda')

    fwd_ln_sm_margin = int(os.getenv("NVTE_FWD_LAYERNORM_SM_MARGIN", "0"))
    ln_out_hipified = torch.empty(M, N, dtype=out_dtype, device='cuda')
    ln_out_hipified, rsigma_hipified = tex.rmsnorm_fwd_fp8_noalloc(input_tensor, gamma_tensor, epsilon, scale_tensor, ln_out_hipified, amax_tensor, scale_inv_tensor, get_te_dtype(out_dtype), fwd_ln_sm_margin, zero_centered_gamma)
    atol, rtol = get_tolerances(out_dtype)
    assert torch.allclose(ln_out_triton, ln_out_hipified, atol=atol, rtol=rtol), 'ln_out does not match'
    # rsigma is of type fp32
    assert torch.allclose(rsigma_triton, rsigma_hipified, atol=1e-6, rtol=5e-5), 'rsigma does not match'

@pytest.mark.parametrize("M, N", test_shapes)
@pytest.mark.parametrize("in_dtype", test_dtypes)
#TODO add fp8/bf8 once fp8 triton kernels are available
@pytest.mark.parametrize("zero_centered_gamma", all_boolean)
def test_rmsnorm_fwd_noalloc_triton(M, N, in_dtype, zero_centered_gamma):
    ## Uniform distribution between [-2.0, 1.0]
    input_tensor = torch.rand(M, N, dtype=torch.float32, device='cuda') * 3.0 - 2.0
    input_tensor = input_tensor.to(in_dtype)
    gamma_tensor = torch.rand(N, dtype=torch.float32, device='cuda') * 3.0 - 2.0
    gamma_tensor = gamma_tensor.to(in_dtype)

    epsilon = 1e-5
    # run the triton path
    ln_out_triton = torch.empty(M, N, dtype=in_dtype, device='cuda')
    ln_out_triton, rsigma_triton = te_rmsnorm_fwd_noalloc_triton(input_tensor, gamma_tensor, ln_out_triton, epsilon, zero_centered_gamma)

    # run the reference hipified kernel path
    fwd_ln_sm_margin = int(os.getenv("NVTE_FWD_LAYERNORM_SM_MARGIN", "0"))
    ln_out_hipified = torch.empty(M, N, dtype=in_dtype, device='cuda')
    ln_out_hipified, rsigma_hipified = tex.rmsnorm_fwd_noalloc(input_tensor, gamma_tensor, ln_out_hipified, epsilon, fwd_ln_sm_margin, zero_centered_gamma)
    atol, rtol = get_tolerances(in_dtype)
    assert torch.allclose(ln_out_triton, ln_out_hipified, atol=atol, rtol=rtol), 'ln_out does not match'
    # rsigma is of type fp32
    assert torch.allclose(rsigma_triton, rsigma_hipified, atol=1e-6, rtol=5e-5), 'rsigma does not match'

@pytest.mark.parametrize("M, N", test_shapes)
@pytest.mark.parametrize("in_dtype", test_dtypes)
#TODO add fp8/bf8 once fp8 triton kernels are available
@pytest.mark.parametrize("zero_centered_gamma", all_boolean)
def test_rmsnorm_fwd_inf_triton(M, N, in_dtype, zero_centered_gamma):
    ## Uniform distribution between [-2.0, 1.0]
    input_tensor = torch.rand(M, N, dtype=torch.float32, device='cuda') * 3.0 - 2.0
    input_tensor = input_tensor.to(in_dtype)
    gamma_tensor = torch.rand(N, dtype=torch.float32, device='cuda') * 3.0 - 2.0
    gamma_tensor = gamma_tensor.to(in_dtype)

    epsilon = 1e-5
    # run the triton path
    ln_out_triton = te_rmsnorm_fwd_inf_triton(input_tensor, gamma_tensor, epsilon, zero_centered_gamma)

    # run the reference hipified kernel path
    fwd_ln_sm_margin = int(os.getenv("NVTE_FWD_LAYERNORM_SM_MARGIN", "0"))
    ln_out_hipified = tex.rmsnorm_fwd_inf(input_tensor, gamma_tensor, epsilon, fwd_ln_sm_margin, zero_centered_gamma)
    atol, rtol = get_tolerances(in_dtype)
    assert torch.allclose(ln_out_triton, ln_out_hipified, atol=atol, rtol=rtol), 'ln_out does not match'
