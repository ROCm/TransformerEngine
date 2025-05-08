# Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information


import pytest
import torch

from transformer_engine.pytorch import cpp_extensions as tex
from transformer_engine.pytorch.triton_kernels.norm_common_triton import (
    get_fwd_ln_sm_margin,
    get_bwd_ln_sm_margin,
    get_inf_ln_sm_margin,
)
from transformer_engine.pytorch.triton_kernels.rmsnorm_triton import (
    te_rmsnorm_fwd_fp8_noalloc_triton,
    te_rmsnorm_fwd_noalloc_triton,
    te_rmsnorm_fwd_inf_triton,
    te_rmsnorm_bwd_triton,
)

from test_common_triton import (
    input_dtypes_str,
    output_dtypes_str,
    str_to_torch_dtype,
    skip_in_dtype_gt_out_dtype,
    skip_mixed_16bit_float_types,
    fill_uniform,
    get_te_dtype,
    get_tolerances,
    compare_results,
)


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
test_idtypes_str = input_dtypes_str(["fp32", "fp16"])
test_odtypes_str = output_dtypes_str(["fp8e4m3"])

all_boolean = [True, False]


# matrix size from tests/cpp/operator/test_rmsnorm.cu
@pytest.mark.parametrize("M, N", test_shapes)
@pytest.mark.parametrize("in_dtype_test", test_idtypes_str)
# TODO: add fp8/bf8 once fp8 triton kernels are available
@pytest.mark.parametrize("out_dtype_test", test_odtypes_str)
@pytest.mark.parametrize("zero_centered_gamma", all_boolean)
def test_rmsnorm_fwd_bwd_triton(M, N, in_dtype_test, out_dtype_test, zero_centered_gamma):
    in_dtype = str_to_torch_dtype(in_dtype_test)
    out_dtype = str_to_torch_dtype(out_dtype_test)

    # skip conditions
    skip_in_dtype_gt_out_dtype(in_dtype, out_dtype)
    skip_mixed_16bit_float_types(in_dtype, out_dtype)

    # generate input tensors
    input_tensor = fill_uniform((M, N), in_dtype)
    # in hipfied kernel cpp test, weight type == input_type
    gamma_tensor = fill_uniform(N, in_dtype)
    # in hipfied kernel cpp test, dz is of weight type
    dz_tensor = fill_uniform((M, N), in_dtype)

    # other parameters:
    epsilon = 1e-5
    fwd_ln_sm_margin = get_fwd_ln_sm_margin()
    bwd_ln_sm_margin = get_bwd_ln_sm_margin()

    if out_dtype_test[1:] == "fp8e4m3":
        scale = fill_uniform((1,), torch.float32)
        amax_triton = torch.full((1,), 0.0, device="cuda")
    else:
        scale = None
    # run the fwd triton path
    # in hipfied kernel cpp test, z is of out_type
    ln_out_triton = torch.empty(M, N, dtype=out_dtype, device='cuda')
    ln_out_triton, rsigma_triton = te_rmsnorm_fwd_fp8_noalloc_triton(input_tensor, gamma_tensor, epsilon, ln_out_triton, out_dtype, fwd_ln_sm_margin, zero_centered_gamma,
                                                                     scale=scale, amax=amax_triton)
    scale_inv_triton = 1 / scale

    # run the fwd reference hipified kernel path
    # dummy fp8 meta
    if out_dtype_test[1:] == "fp8e4m3":
        amax_hipified = torch.full((1,), 0.0, device="cuda")
        scale_inv_hipified = torch.full((1,), 0.0, device="cuda")
    else:
        amax_hipified = scale_inv_hipified = torch.empty(0, device="cuda")
    ln_out_hipified = torch.empty(M, N, dtype=out_dtype, device='cuda')
    ln_out_hipified, rsigma_hipified = tex.rmsnorm_fwd_fp8_noalloc(input_tensor, gamma_tensor, epsilon, scale, ln_out_hipified, amax_hipified, scale_inv_hipified, get_te_dtype(out_dtype), fwd_ln_sm_margin, zero_centered_gamma)

    # assert on ln_out
    atol, rtol = get_tolerances(out_dtype)
    compare_results(
        "torch",
        ln_out_triton.to(torch.float32),
        ln_out_hipified.to(torch.float32),
        atol,
        rtol,
        lambda msg: f"ln_out does not match triton <-> hip\n\n{msg}\n",
    )

    # assert on rsigma
    rsigma_atol, rsigma_rtol = 1e-6, 5e-5
    # rsigma is of type fp32
    compare_results(
        "torch",
        rsigma_triton,
        rsigma_hipified,
        rsigma_atol,
        rsigma_rtol,
        lambda msg: f"rsigma does not match triton <-> hip\n\n{msg}\n",
    )
    atol_stats, _ = get_tolerances(torch.float32)
    rtol_stats = 5e-5
    if out_dtype_test[1:] == "fp8e4m3":
        compare_results(
            "torch",
            amax_triton,
            amax_hipified,
            atol_stats,
            rtol_stats,
            lambda msg: f"amax does not match triton <-> hip\n\n{msg}\n",
        )
        compare_results(
            "torch",
            scale_inv_triton,
            scale_inv_hipified,
            atol_stats,
            rtol_stats,
            lambda msg: f"scale_inv does not match triton <-> hip\n\n{msg}\n",
        )

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
    compare_results(
        "te",
        dx_triton,
        dx_hipified,
        atol_bwd,
        rtol_bwd,
        lambda msg: f"dx does not match triton <-> hip\n\n{msg}\n"
    )

    # assert on dgamma
    compare_results(
        "te",
        dgamma_triton,
        dgamma_hipified,
        atol_bwd,
        rtol_bwd,
        lambda msg: f"dgamma does not match triton <-> hip\n\n{msg}\n",
    )


@pytest.mark.parametrize("M, N", test_shapes)
@pytest.mark.parametrize("in_dtype_test", test_idtypes_str)
@pytest.mark.parametrize("out_dtype_test", test_odtypes_str)
# TODO: add fp8/bf8 once fp8 triton kernels are available
@pytest.mark.parametrize("zero_centered_gamma", all_boolean)
def test_rmsnorm_fwd_noalloc_triton(M, N, in_dtype_test, out_dtype_test, zero_centered_gamma):
    in_dtype = str_to_torch_dtype(in_dtype_test)
    out_dtype = str_to_torch_dtype(out_dtype_test)

    # skip conditions
    skip_in_dtype_gt_out_dtype(in_dtype, out_dtype)
    skip_mixed_16bit_float_types(in_dtype, out_dtype)

    input_tensor = fill_uniform((M, N), in_dtype)
    gamma_tensor = fill_uniform(N, in_dtype)

    epsilon = 1e-5
    fwd_ln_sm_margin = get_fwd_ln_sm_margin()

    # run the triton path
    ln_out_triton = torch.empty(M, N, dtype=out_dtype, device='cuda')
    if out_dtype_test[1:] == "fp8e4m3":
        scale = fill_uniform((1,), torch.float32)
        amax_triton = torch.full((1,), 0.0, device="cuda")
        amax_hipified =  torch.full((1,), 0.0, device="cuda")
        scale_inv_hipified = torch.full((1,), 0.0, device="cuda")
    else:
        scale = torch.empty(0, device="cuda")
        amax_triton = torch.empty(0, device="cuda")
        amax_hipified = torch.empty(0, device="cuda")
        scale_inv_hipified = torch.empty(0, device="cuda")
    ln_out_triton, rsigma_triton = te_rmsnorm_fwd_fp8_noalloc_triton(input_tensor, gamma_tensor, epsilon, ln_out_triton, out_dtype, fwd_ln_sm_margin, zero_centered_gamma,
                                                                     scale=scale, amax=amax_triton)
    scale_inv_triton = 1 / scale
    # run the reference hipified kernel path
    ln_out_hipified = torch.empty(M, N, dtype=out_dtype, device='cuda')
    ln_out_hipified, rsigma_hipified = tex.rmsnorm_fwd_fp8_noalloc(input_tensor, gamma_tensor, 
                                                               epsilon, scale,
                                                               ln_out_hipified,
                                                               amax_hipified,
                                                               scale_inv_hipified,
                                                               get_te_dtype(out_dtype),
                                                               fwd_ln_sm_margin, zero_centered_gamma)
    atol, rtol = get_tolerances(out_dtype)
    compare_results(
        "torch",
        ln_out_triton.to(torch.float64),
        ln_out_hipified.to(torch.float64),
        atol,
        rtol,
        lambda msg: f"ln_out does not match triton <-> hip\n\n{msg}\n",
    )
    # rsigma is of type fp32
    compare_results(
        "torch",
        rsigma_triton,
        rsigma_hipified,
        1e-6,
        5e-5,
        lambda msg: f"rsigma does not match triton <-> hip\n\n{msg}\n",
    )

    if out_dtype_test[1:] == "fp8e4m3":
        compare_results(
            "torch",
            amax_triton,
            amax_hipified,
            atol,
            rtol,
            lambda msg: f"amax does not match triton <-> hip\n\n{msg}\n",
        )
        compare_results(
            "torch",
            scale_inv_triton,
            scale_inv_hipified,
            atol,
            rtol,
            lambda msg: f"scale_inv does not match triton <-> hip\n\n{msg}\n",
        )


@pytest.mark.parametrize("M, N", test_shapes)
@pytest.mark.parametrize("in_dtype", test_idtypes_str)
# TODO: add fp8/bf8 once fp8 triton kernels are available
@pytest.mark.parametrize("zero_centered_gamma", all_boolean)
def test_rmsnorm_fwd_inf_triton(M, N, in_dtype, zero_centered_gamma):
    in_dtype = str_to_torch_dtype(in_dtype)

    input_tensor = fill_uniform((M, N), in_dtype)
    gamma_tensor = fill_uniform(N, in_dtype)

    epsilon = 1e-5
    inf_ln_sm_margin = get_inf_ln_sm_margin()

    # run the triton path
    ln_out_triton = te_rmsnorm_fwd_inf_triton(input_tensor, gamma_tensor, epsilon, inf_ln_sm_margin, zero_centered_gamma)

    # run the reference hipified kernel path
    ln_out_hipified = tex.rmsnorm_fwd_inf(input_tensor, gamma_tensor, epsilon, inf_ln_sm_margin, zero_centered_gamma)
    atol, rtol = get_tolerances(in_dtype)
    compare_results(
        "torch",
        ln_out_triton,
        ln_out_hipified,
        atol,
        rtol,
        lambda msg: f"ln_out does not match triton <-> hip\n\n{msg}\n",
    )
