# Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

import pytest
import torch
import os

from transformer_engine.pytorch.triton_kernels.rmsnorm_triton import te_rmsnorm_fwd_triton, te_rmsnorm_bwd_triton, get_num_sms
from transformer_engine.pytorch import cpp_extensions as tex


def get_te_dtype(dtype):
    if dtype == torch.float32:
        return tex.DType.kFloat32
    if dtype == torch.float16:
        return tex.DType.kFloat16
    if dtype == torch.bfloat16:
        return tex.DType.kBFloat16
    if dtype == torch.float8_e4m3fnuz:
        return tex.DType.kFloat8E4M3
    if dtype == torch.float8_e5m2fnuz:
        return tex.DType.kFloat8E5M2


# get size in bytes of given PyTorch type
def sizeof(in_dtype):
    return torch.finfo(in_dtype).bits // 8


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


# PyTorch implementation of `compareResults` C++ function from `tests/cpp/test_common.cu`.
# Arguments:
#     t: actual tensor
#     r: expected tensor
def te_compare_results(t, r, atol, rtol):
    assert t.dtype == r.dtype, f"Tensor dtypes don't match: {t.dtype} vs {r.dtype}."
    assert t.shape == r.shape, f"Tensor shapes don't match: {t.shape} vs {r.shape}."
    assert atol > 0, "Absolute tolerance must be positive."
    assert rtol > 0, "Relative tolerance must be positive."
    dtype = t.dtype
    t = t.cpu().to(torch.float32).to(torch.float64)
    r = r.cpu().to(torch.float32).to(torch.float64)
    diff = t - r
    atol_mismatch = torch.abs(diff) > atol
    nonzero_r = r != 0
    rtol_mismatch = torch.full_like(atol_mismatch, False)
    rtol_mismatch[nonzero_r] = torch.abs(diff[nonzero_r] / r[nonzero_r]) > rtol
    mismatch = torch.logical_and(atol_mismatch, torch.logical_or(torch.logical_not(nonzero_r), rtol_mismatch))
    has_mismatch = torch.any(mismatch).item()
    # for fp32 the floating point comparison is enough to error out
    if has_mismatch and dtype != torch.float32:
        # check if it is just a failure of round to nearest choosing different side of the real value
        # for non fp32 types
        mean = (t + r) / 2
        eps = 1e-6
        mean_one_plus_eps = mean * (1 + eps)
        mean_one_minus_eps = mean * (1 - eps)
        mean_gte_zero = mean >= 0
        mean_p = torch.where(mean_gte_zero, mean_one_plus_eps, mean_one_minus_eps)
        mean_m = torch.where(mean_gte_zero, mean_one_minus_eps, mean_one_plus_eps)
        cast_mean_p = mean_p.to(torch.float32).to(dtype).to(torch.float32).to(torch.float64)
        cast_mean_m = mean_m.to(torch.float32).to(dtype).to(torch.float32).to(torch.float64)
        min_tr = torch.minimum(t, r)
        max_tr = torch.maximum(t, r)
        round_check = torch.logical_not(torch.logical_and(cast_mean_m == min_tr, cast_mean_p == max_tr))
        mismatch = torch.logical_and(mismatch, round_check)
        has_mismatch = torch.any(mismatch).item()
    # TODO: improve assertion message adding more information
    assert not has_mismatch, "There are tensor mismatches."


def get_ln_sm_margin(sm_margin_type):
    try:
        sm_margin = max(int(os.getenv(f"NVTE_{sm_margin_type}_LAYERNORM_SM_MARGIN", "0")), 0)
    except ValueError:
        sm_margin = 0
    assert sm_margin >= 0
    return sm_margin


def get_fwd_ln_sm_margin():
    return get_ln_sm_margin("FWD")


def get_bwd_ln_sm_margin():
    return get_ln_sm_margin("BWD")


def get_inf_ln_sm_margin():
    return get_ln_sm_margin("INF")


test_shapes = [
    (2048, 4096),
    (768, 2048),
    (256, 1024),
    (128, 768),
    (64, 512),
    (173, 409),
    (71, 3571),
    (29, 17389),
]

# descriptive type strings to better identify pytest test cases
test_dtypes_str = ['fp32', 'fp16', 'bf16']
# add i prefix to identify input type
test_idtypes_str = ["i" + dtype_str for dtype_str in test_dtypes_str]
# add o prefix to identify output type
test_odtypes_str = ["o" + dtype_str for dtype_str in test_dtypes_str]

# convert descriptive type strings to torch types
def str_to_torch_dtype(dtype_str):
    return {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}[dtype_str[1:]]

all_boolean = [True, False]


# matrix size from tests/cpp/operator/test_rmsnorm.cu
@pytest.mark.parametrize("M, N", test_shapes)
@pytest.mark.parametrize("in_dtype", test_idtypes_str)
# TODO: add fp8/bf8 once fp8 triton kernels are available
@pytest.mark.parametrize("out_dtype", test_odtypes_str)
@pytest.mark.parametrize("zero_centered_gamma", all_boolean)
def test_rmsnorm_fwd_bwd_triton(M, N, in_dtype, out_dtype, zero_centered_gamma):
    in_dtype = str_to_torch_dtype(in_dtype)
    out_dtype = str_to_torch_dtype(out_dtype)

    # skip conditions
    if sizeof(in_dtype) < sizeof(out_dtype):
        pytest.skip("size of input dtype < size of output dtype")
    if (in_dtype==torch.float16 and out_dtype==torch.bfloat16) or (in_dtype==torch.bfloat16 and out_dtype==torch.float16):
        pytest.skip("hipified rmsnorm kernel does not support mixing fp16 and bf16")

    # generate input tensors
    ## Uniform distribution between [-2.0, 1.0]
    torch.manual_seed(0)
    input_tensor = torch.rand(M, N, dtype=torch.float32, device='cuda') * 3.0 - 2.0
    input_tensor = input_tensor.to(in_dtype)
    # in hipfied kernel cpp test, weight type == input_type
    gamma_tensor = torch.rand(N, dtype=torch.float32, device='cuda') * 3.0 - 2.0
    gamma_tensor = gamma_tensor.to(in_dtype)
    dz_tensor = torch.rand(M, N, dtype=torch.float32, device='cuda') * 3.0 - 2.0
    # in hipfied kernel cpp test, dz is of weight type
    dz_tensor = dz_tensor.to(in_dtype)

    # other parameters:
    epsilon = 1e-5
    fwd_ln_sm_margin = get_fwd_ln_sm_margin()
    bwd_ln_sm_margin = get_bwd_ln_sm_margin()

    # run the fwd triton path
    # in hipfied kernel cpp test, z is of out_type
    ln_out_triton = torch.empty(M, N, dtype=out_dtype, device='cuda')
    ln_out_triton, _, rsigma_triton = te_rmsnorm_fwd_triton(input_tensor, gamma_tensor, epsilon, ln_out_triton, None, get_te_dtype(out_dtype), fwd_ln_sm_margin, zero_centered_gamma)

    # run the fwd reference hipified kernel path
    # dummy fp8 meta
    scale_tensor = torch.empty(0, dtype=torch.float32, device='cuda')
    amax_tensor = torch.zeros(0, dtype=torch.float32, device='cuda')
    scale_inv_tensor = torch.empty(0, dtype=torch.float32, device='cuda')
    ln_out_hipified = torch.empty(M, N, dtype=out_dtype, device='cuda')
    ln_out_hipified, _, rsigma_hipified = tex.rmsnorm_fwd(input_tensor, gamma_tensor, epsilon, ln_out_hipified, None, get_te_dtype(out_dtype), fwd_ln_sm_margin, zero_centered_gamma)

    # assert on ln_out
    ln_out_atol = 1e-8
    _, ln_out_rtol = get_tolerances(out_dtype)
    torch.testing.assert_close(ln_out_triton, ln_out_hipified, atol=ln_out_atol, rtol=ln_out_rtol,
                               msg=lambda msg: f"ln_out does not match triton <-> hip\n\n{msg}\n")

    # assert on rsigma
    rsigma_atol, rsigma_rtol = 1e-6, 5e-5
    # rsigma is of type fp32
    torch.testing.assert_close(rsigma_triton, rsigma_hipified, atol=rsigma_atol, rtol=rsigma_rtol,
                               msg=lambda msg: f"rsigma does not match triton <-> hip\n\n{msg}\n")

    # run triton bwd
    dx_triton, dgamma_triton = te_rmsnorm_bwd_triton(dz_tensor, input_tensor, rsigma_triton, gamma_tensor, bwd_ln_sm_margin, zero_centered_gamma)

    # run hipified ref bwd
    dx_hipified, dgamma_hipified = tex.rmsnorm_bwd(dz_tensor, input_tensor, rsigma_hipified, gamma_tensor, bwd_ln_sm_margin, zero_centered_gamma)

    atol_bwd = 5e-6
    rtol_bwd = 1e-4

    # Backward pass assertions are done with `te_compare_results` instead of the usual `torch.testing.assert_close`
    # because the former is more strict and causes some failures. The goal of this test is to mimick the equivalent
    # C++ test, so TE comparison behavior and error tolerances are used.

    # assert on dx
    te_compare_results(dx_triton, dx_hipified, atol_bwd, rtol_bwd)

    # assert on dgamma
    te_compare_results(dgamma_triton, dgamma_hipified, atol_bwd, rtol_bwd)

# matrix size from tests/cpp/operator/test_rmsnorm.cu
@pytest.mark.parametrize("M, N", test_shapes)
@pytest.mark.parametrize("in_dtype", test_idtypes_str)
# TODO: add fp8/bf8 once fp8 triton kernels are available
@pytest.mark.parametrize("out_dtype", test_odtypes_str)
@pytest.mark.parametrize("zero_centered_gamma", all_boolean)
def test_rmsnorm_fwd_noalloc_triton(M, N, in_dtype, out_dtype, zero_centered_gamma):
    in_dtype = str_to_torch_dtype(in_dtype)
    out_dtype = str_to_torch_dtype(out_dtype)

    # skip conditions
    if sizeof(in_dtype) < sizeof(out_dtype):
        pytest.skip("size of input dtype < size of output dtype")
    if (in_dtype==torch.float16 and out_dtype==torch.bfloat16) or (in_dtype==torch.bfloat16 and out_dtype==torch.float16):
        pytest.skip("hipified rmsnorm kernel does not support mixing fp16 and bf16")

    # generate input tensors
    ## Uniform distribution between [-2.0, 1.0]
    torch.manual_seed(0)
    input_tensor = torch.rand(M, N, dtype=torch.float32, device='cuda') * 3.0 - 2.0
    input_tensor = input_tensor.to(in_dtype)
    # in hipfied kernel cpp test, weight type == input_type
    gamma_tensor = torch.rand(N, dtype=torch.float32, device='cuda') * 3.0 - 2.0
    gamma_tensor = gamma_tensor.to(in_dtype)
    dz_tensor = torch.rand(M, N, dtype=torch.float32, device='cuda') * 3.0 - 2.0
    # in hipfied kernel cpp test, dz is of weight type
    dz_tensor = dz_tensor.to(in_dtype)

    # other parameters:
    epsilon = 1e-5
    fwd_ln_sm_margin = get_fwd_ln_sm_margin()
    bwd_ln_sm_margin = get_bwd_ln_sm_margin()

    # run the fwd triton path
    # in hipfied kernel cpp test, z is of out_type
    ln_out_triton, _, rsigma_triton = te_rmsnorm_fwd_triton(input_tensor, gamma_tensor, epsilon, None, None, get_te_dtype(out_dtype), fwd_ln_sm_margin, zero_centered_gamma)

    # run the fwd reference hipified kernel path
    ln_out_hipified, _, rsigma_hipified = tex.rmsnorm_fwd(input_tensor, gamma_tensor, epsilon, None, None, get_te_dtype(out_dtype), fwd_ln_sm_margin, zero_centered_gamma)

    # assert on ln_out
    ln_out_atol = 1e-8
    _, ln_out_rtol = get_tolerances(out_dtype)
    torch.testing.assert_close(ln_out_triton, ln_out_hipified, atol=ln_out_atol, rtol=ln_out_rtol,
                               msg=lambda msg: f"ln_out does not match triton <-> hip\n\n{msg}\n")

    # assert on rsigma
    rsigma_atol, rsigma_rtol = 1e-6, 5e-5
    # rsigma is of type fp32
    torch.testing.assert_close(rsigma_triton, rsigma_hipified, atol=rsigma_atol, rtol=rsigma_rtol,
                               msg=lambda msg: f"rsigma does not match triton <-> hip\n\n{msg}\n")

def test_sm_margin():
    num_sms = get_num_sms()
    assert num_sms > 0
    assert get_num_sms(0) == num_sms
    assert get_num_sms(-5) == num_sms
    assert get_num_sms(1) == num_sms - 1
    assert get_num_sms(100 * num_sms) == 1
