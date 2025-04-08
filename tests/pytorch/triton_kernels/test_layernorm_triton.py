# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information


import os

import pytest
import torch

from transformer_engine.pytorch import cpp_extensions as tex
from transformer_engine.pytorch.triton_kernels.layernorm_triton import (
    te_layernorm_bwd_triton,
    te_layernorm_fwd_fp8_noalloc_triton,
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


test_idtypes_str = input_dtypes_str(["fp32", "fp16"])

test_odtypes_str = output_dtypes_str(["fp32", "bf16", "fp16"])

test_shapes = [
    (2048, 12288),
    (768, 1024),
    (256, 65536),
    (128, 6144),
    (64, 2304),
    (229, 541),
    (71, 3571),
    (29, 17389),
]

all_boolean = [False, True]


@pytest.mark.parametrize("in_dtype", test_idtypes_str)
@pytest.mark.parametrize("out_dtype", test_odtypes_str)
@pytest.mark.parametrize("M, N", test_shapes)
@pytest.mark.parametrize("zero_centered_gamma", all_boolean)
def test_layernorm_fwd_bwd_triton(in_dtype, out_dtype, M, N, zero_centered_gamma):
    # Get Torch data types:
    in_dtype = str_to_torch_dtype(in_dtype)
    out_dtype = str_to_torch_dtype(out_dtype)

    # Skip conditions:
    skip_in_dtype_gt_out_dtype(in_dtype, out_dtype)
    skip_mixed_16bit_float_types(in_dtype, out_dtype)

    # Generate tensors:
    x = fill_uniform((M, N), in_dtype)
    gamma = fill_uniform(N, in_dtype)
    beta = fill_uniform(N, in_dtype)
    dz = fill_uniform((M, N), in_dtype)

    epsilon = 1e-5

    # Run Triton forward.
    y_triton = torch.empty((M, N), dtype=out_dtype, device="cuda")
    y_triton, mu_triton, rsigma_triton = te_layernorm_fwd_fp8_noalloc_triton(
        x, gamma, beta, epsilon, y_triton, out_dtype, zero_centered_gamma
    )

    # Run Hipified forward reference.
    scale = amax = scale_inv = torch.empty(0, device="cuda")
    fwd_ln_sm_margin = int(os.getenv("NVTE_FWD_LAYERNORM_SM_MARGIN", "0"))
    y_hipified = torch.empty((M, N), dtype=out_dtype, device="cuda")
    y_hipified, mu_hipified, rsigma_hipified = tex.layernorm_fwd_fp8_noalloc(
        x,
        gamma,
        beta,
        epsilon,
        scale,
        y_hipified,
        amax,
        scale_inv,
        get_te_dtype(out_dtype),
        fwd_ln_sm_margin,
        zero_centered_gamma,
    )

    # Assert on mu and rsigma:
    atol_stats, _ = get_tolerances(torch.float32)
    rtol_stats = 5e-5
    compare_results(
        "torch",
        mu_triton,
        mu_hipified,
        atol_stats,
        rtol_stats,
        lambda msg: f"mu does not match triton <-> hip\n\n{msg}\n",
    )
    compare_results(
        "torch",
        rsigma_triton,
        rsigma_hipified,
        atol_stats,
        rtol_stats,
        lambda msg: f"rsigma does not match triton <-> hip\n\n{msg}\n",
    )

    # Assert on y:
    atol, rtol = get_tolerances(out_dtype)
    if out_dtype == torch.float32:
        # everything pass with default fp32 atol=1e-6
        # atol = 5e-7
        pass
    compare_results(
        "torch",
        y_triton,
        y_hipified,
        atol,
        rtol,
        lambda msg: f"y does not match triton <-> hip\n\n{msg}\n",
    )

    # Run Triton backward.
    dx_triton, dgamma_triton, dbeta_triton = te_layernorm_bwd_triton(
        dz,
        x,
        mu_triton,
        rsigma_triton,
        gamma,
        zero_centered_gamma,
    )

    # Run Hipified backward reference.
    dx_hipified, dgamma_hipified, dbeta_hipified = tex.layernorm_bwd(
        dz,
        x,
        mu_hipified,
        rsigma_hipified,
        gamma,
        fwd_ln_sm_margin,
        zero_centered_gamma,
    )

    # Assert on dx, dgamma and dbeta:
    atol_bwd = 1.5e-4
    rtol_bwd = 1e-4
    # TE comparison deals with fp16 rounding errors.
    bwd_cmp = "te" if in_dtype == out_dtype == torch.float16 else "torch"
    compare_results(
        bwd_cmp,
        dx_triton,
        dx_hipified,
        atol_bwd,
        rtol_bwd,
        lambda msg: f"dx does not match triton <-> hip\n\n{msg}\n",
    )
    compare_results(
        bwd_cmp,
        dgamma_triton,
        dgamma_hipified,
        atol_bwd,
        rtol_bwd,
        lambda msg: f"dgamma does not match triton <-> hip\n\n{msg}\n",
    )
    compare_results(
        bwd_cmp,
        dbeta_triton,
        dbeta_hipified,
        atol_bwd,
        rtol_bwd,
        lambda msg: f"dbeta does not match triton <-> hip\n\n{msg}\n",
    )
