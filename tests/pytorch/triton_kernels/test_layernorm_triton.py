# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information


import pytest
import torch

from transformer_engine.pytorch import cpp_extensions as tex
from transformer_engine.pytorch.triton_kernels.norm_common_triton import (
    get_fwd_ln_sm_margin,
    get_bwd_ln_sm_margin,
)
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
#test_idtypes_str = input_dtypes_str(["fp32"])

test_odtypes_str = output_dtypes_str(["fp32", "bf16", "fp16", "fp8e4m3"])
#test_odtypes_str = output_dtypes_str(["fp8e4m3"])

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


@pytest.mark.parametrize("in_dtype_test", test_idtypes_str)
@pytest.mark.parametrize("out_dtype_test", test_odtypes_str)
@pytest.mark.parametrize("M, N", test_shapes)
@pytest.mark.parametrize("zero_centered_gamma", all_boolean)
def test_layernorm_fwd_bwd_triton(in_dtype_test, out_dtype_test, M, N, zero_centered_gamma):
    # Get Torch data types:
    in_dtype = str_to_torch_dtype(in_dtype_test)
    out_dtype = str_to_torch_dtype(out_dtype_test)

    # Skip conditions:
    skip_in_dtype_gt_out_dtype(in_dtype, out_dtype)
    skip_mixed_16bit_float_types(in_dtype, out_dtype)

    # Generate tensors:
    x = fill_uniform((M, N), in_dtype)
    gamma = fill_uniform(N, in_dtype)
    beta = fill_uniform(N, in_dtype)
    dz = fill_uniform((M, N), in_dtype)
    if out_dtype_test[1:] == "fp8e4m3":
        scale = fill_uniform((1,), torch.float32)
    else:
        scale = torch.empty(0, device="cuda")

    epsilon = 1e-5

    # Run Triton forward.
    y_triton = torch.empty((M, N), dtype=out_dtype, device="cuda")
    if out_dtype_test[1:] == "fp8e4m3":
        amax_triton = torch.full((1,), 0.0, device="cuda")
        scale_inv_triton = torch.full((1,), 0.0, device="cuda")
    else:
        amax_triton = scale_inv_triton = None

    y_triton, mu_triton, rsigma_triton, scale_inv_triton = te_layernorm_fwd_fp8_noalloc_triton(
        x,
        gamma,
        beta,
        epsilon,
        scale,
        y_triton,
        amax_triton,
        scale_inv_triton,
        out_dtype,
        zero_centered_gamma
    )

    print(amax_triton)
    print(scale_inv_triton)

    # Run Hipified forward reference.
    y_hipified = torch.empty((M, N), dtype=out_dtype, device="cuda")
    if out_dtype_test[1:] == "fp8e4m3":
        amax_hipified = torch.full((1,), 0.0, device="cuda")
        scale_inv_hipified = torch.full((1,), 0.0, device="cuda")
    else:
        amax_hipified = scale_inv_hipified = torch.empty(0, device="cuda")
    y_hipified, mu_hipified, rsigma_hipified = tex.layernorm_fwd_fp8_noalloc(
        x,
        gamma,
        beta,
        epsilon,
        scale,
        y_hipified,
        amax_hipified,
        scale_inv_hipified,
        get_te_dtype(out_dtype),
        get_fwd_ln_sm_margin(),
        zero_centered_gamma,
    )

    print(amax_hipified)
    print(scale_inv_hipified)

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
        # Everything pass with default fp32 atol=1e-6. TE C++ test uses atol=5e-7, this tolerance causes
        # some minor failures in Triton test.
        # TODO: Investigate test failures when using atol=5e-7.
        # atol = 5e-7
        pass
    compare_results(
        "torch",
        y_triton.to(torch.float32),
        y_hipified.to(torch.float32),
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
        get_bwd_ln_sm_margin(),
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


if __name__ == "__main__":
    import sys

    in_dtype = torch.float16
    out_dtype = torch.float8_e4m3fnuz
    zero_centered_gamma = True
    run_triton = sys.argv[1] == "triton"
    run_hipified = sys.argv[1] == "te"
    M = int(sys.argv[2])
    N = int(sys.argv[3])
    run_bwd = sys.argv[4] == "bwd"
    mode = "forward only" if not run_bwd else "forward + backward"

    # Parse waves_per_eu and num_warps:
    waves_per_eu = None
    if run_triton:
        try:
            waves_per_eu = int(sys.argv[5])
        except IndexError:
            pass
    num_warps = None
    if run_triton:
        try:
            num_warps = int(sys.argv[6])
        except IndexError:
            pass

    # Generate tensors:
    x = fill_uniform((M, N), in_dtype)
    gamma = fill_uniform(N, in_dtype)
    beta = fill_uniform(N, in_dtype)
    if run_bwd:
        dz = fill_uniform((M, N), in_dtype)
    if out_dtype == torch.float8_e4m3fnuz:
        scale = fill_uniform((1,), torch.float32)
    else:
        scale = torch.empty(0, device="cuda")

    epsilon = 1e-5

    if run_triton:
        print(f"Running {mode} Triton implementation for shape {(M, N)}...")
        # Select waves_per_eu and num_warps.
        try:
            best_waves_per_eu, best_num_warps = {
                (2048, 12288): (1, 8),
                (768, 1024): (4, 8),
                (256, 65536): (2, 16),
                (128, 6144): (4, 4),
                (64, 2304): (4, 16),
                (229, 541): (2, 16),
                (71, 3571): (1, 16),
                (29, 17389): (2, 16),
                (76800, 1600): (4, 4),
            }[(M, N)]
        except KeyError:
            best_waves_per_eu, best_num_warps = 2, 8
        if waves_per_eu is None:
            waves_per_eu = best_waves_per_eu
        if num_warps is None:
            num_warps = best_num_warps
        # Run Triton forward.
        y_triton = torch.empty((M, N), dtype=out_dtype, device="cuda")
        if out_dtype == torch.float8_e4m3fnuz:
            amax_triton = torch.full((1,), 0.0, device="cuda")
            scale_inv_triton = torch.full((1,), 0.0, device="cuda")
        else:
            amax_triton = scale_inv_triton = None

        y_triton, mu_triton, rsigma_triton, scale_inv_triton = te_layernorm_fwd_fp8_noalloc_triton(
            x,
            gamma,
            beta,
            epsilon,
            scale,
            y_triton,
            amax_triton,
            scale_inv_triton,
            out_dtype,
            zero_centered_gamma,
            waves_per_eu=waves_per_eu,
            num_warps=num_warps,
        )
        if run_bwd:
            # Run Triton backward.
            dx_triton, dgamma_triton, dbeta_triton = te_layernorm_bwd_triton(
                dz,
                x,
                mu_triton,
                rsigma_triton,
                gamma,
                zero_centered_gamma,
            )

    if run_hipified:
        print(f"Running {mode} TE implementation for shape {(M, N)}...")
        # Run Hipified forward reference.
        y_hipified = torch.empty((M, N), dtype=out_dtype, device="cuda")
        if out_dtype == torch.float8_e4m3fnuz:
            amax_hipified = torch.full((1,), 0.0, device="cuda")
            scale_inv_hipified = torch.full((1,), 0.0, device="cuda")
        else:
            amax_hipified = scale_inv_hipified = torch.empty(0, device="cuda")

        y_hipified, mu_hipified, rsigma_hipified = tex.layernorm_fwd_fp8_noalloc(
            x,
            gamma,
            beta,
            epsilon,
            scale,
            y_hipified,
            amax_hipified,
            scale_inv_hipified,
            get_te_dtype(out_dtype),
            get_fwd_ln_sm_margin(),
            zero_centered_gamma,
        )
        if run_bwd:
            # Run Hipified backward reference.
            dx_hipified, dgamma_hipified, dbeta_hipified = tex.layernorm_bwd(
                dz,
                x,
                mu_hipified,
                rsigma_hipified,
                gamma,
                get_bwd_ln_sm_margin(),
                zero_centered_gamma,
            )