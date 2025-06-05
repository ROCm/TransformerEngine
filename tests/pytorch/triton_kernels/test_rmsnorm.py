# Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information


import pytest
import torch

from transformer_engine import pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch.triton_kernels.common import torch_dtype_to_te_dtype
from transformer_engine.pytorch.triton_kernels.norm_common import (
    get_fwd_ln_sm_margin,
    get_bwd_ln_sm_margin,
    get_inf_ln_sm_margin,
)
from transformer_engine.pytorch.triton_kernels.rmsnorm import (
    te_rmsnorm_bwd_triton,
)
from test_common import (
    input_dtypes_str,
    output_dtypes_str,
    str_to_torch_dtype,
    skip_in_dtype_gt_out_dtype,
    skip_mixed_16bit_float_types,
    fill_uniform,
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

test_types_str = ["fp32", "fp16", "bf16"]
test_idtypes_str = input_dtypes_str(test_types_str)
test_odtypes_str = output_dtypes_str(test_types_str)

all_boolean = [True, False]


# matrix size from tests/cpp/operator/test_rmsnorm.cu
@pytest.mark.parametrize("M, N", test_shapes)
@pytest.mark.parametrize("in_dtype", test_idtypes_str)
# TODO: add fp8/bf8 once fp8 triton kernels are available
@pytest.mark.parametrize("out_dtype", test_odtypes_str)
@pytest.mark.parametrize("zero_centered_gamma", all_boolean)
def test_rmsnorm_bwd_triton(M, N, in_dtype, out_dtype, zero_centered_gamma):
    in_dtype = str_to_torch_dtype(in_dtype)
    out_dtype = str_to_torch_dtype(out_dtype)

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

    # run the fwd reference hipified kernel path
    # dummy fp8 meta
    scale_tensor = torch.empty(0, dtype=torch.float32, device='cuda')
    amax_tensor = torch.zeros(0, dtype=torch.float32, device='cuda')
    scale_inv_tensor = torch.empty(0, dtype=torch.float32, device='cuda')
    ln_out_hipified = torch.empty(M, N, dtype=out_dtype, device='cuda')
    ln_out_hipified, _, rsigma_hipified = tex.rmsnorm_fwd(input_tensor, gamma_tensor, epsilon, ln_out_hipified, None, torch_dtype_to_te_dtype(out_dtype), fwd_ln_sm_margin, zero_centered_gamma)

    # run triton bwd
    dx_triton, dgamma_triton = te_rmsnorm_bwd_triton(dz_tensor, input_tensor, rsigma_hipified, gamma_tensor, bwd_ln_sm_margin, zero_centered_gamma)

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

# TODO: bring back fwd pytests after refactoring te_rmsnorm_fwd_triton
