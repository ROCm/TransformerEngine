# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information


import pytest
import torch

from transformer_engine import pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch.triton_kernels.common import (
    is_fp8_torch_dtype,
    te_dtype_to_torch_dtype,
    torch_dtype_to_te_dtype,
    te_dtype_to_aten_dtype,
    te_dtype_to_enum_value,
)
from transformer_engine.pytorch.triton_kernels.norm_common import (
    get_fwd_ln_sm_margin,
    get_bwd_ln_sm_margin,
)
from transformer_engine.pytorch.triton_kernels.layernorm import (
    te_layernorm_bwd_triton,
    te_layernorm_fwd_fp8_noalloc_triton,
    te_layernorm_fwd_fp8_inf_ts_triton,
    te_layernorm_fwd_fp8_triton,
    te_layernorm_fwd_noalloc_triton,
    te_layernorm_fwd_inf_ts_triton,
    te_layernorm_fwd_triton,
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


test_idtypes_str = input_dtypes_str(["fp32", "fp16", "bf16"])

test_odtypes_str = output_dtypes_str(["fp32", "bf16", "fp16", "fp8e4"])

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
def test_layernorm_fwd_fp8_triton(in_dtype, out_dtype, M, N, zero_centered_gamma):
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
    if is_fp8_torch_dtype(out_dtype):
        scale = fill_uniform((1,), torch.float32)
    else:
        scale = torch.empty(0, device="cuda")

    epsilon = 1e-5

    # Run Triton forward.
    amax_triton = torch.zeros((1,), device="cuda") if is_fp8_torch_dtype(out_dtype) else None
    scale_inv_triton = torch.zeros((1,), device="cuda") if is_fp8_torch_dtype(out_dtype) else None
    sm_margin = 0
    y_triton, mu_triton, rsigma_triton = te_layernorm_fwd_fp8_triton(
        x,
        gamma,
        beta,
        epsilon,
        scale,
        amax_triton,
        scale_inv_triton,
        torch_dtype_to_te_dtype(out_dtype),
        sm_margin,
        zero_centered_gamma,
    )
    # Run Hipified forward reference.
    if is_fp8_torch_dtype(out_dtype):
        amax_hipified = torch.full((1,), 0.0, device="cuda")
        scale_inv_hipified = torch.full((1,), 0.0, device="cuda")
    else:
        amax_hipified = scale_inv_hipified = torch.empty(0, device="cuda")

    y_hipified, mu_hipified, rsigma_hipified = tex.layernorm_fwd_fp8(
        x,
        gamma,
        beta,
        epsilon,
        scale,
        amax_hipified,
        scale_inv_hipified,
        torch_dtype_to_te_dtype(out_dtype),
        get_fwd_ln_sm_margin(),
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
    if is_fp8_torch_dtype(out_dtype):
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
        #view y_triton and y_hipified as fp8
        y_triton = y_triton.view(out_dtype)
        y_hipified = y_hipified.view(out_dtype)

    # Assert on y:
    atol, rtol = get_tolerances(out_dtype)
    if out_dtype == torch.float32:
        # Everything pass with default fp32 atol=1e-6. TE C++ test uses atol=5e-7, this tolerance causes
        # some minor failures in Triton test.
        # TODO: Investigate test failures when using atol=5e-7.
        # atol = 5e-7
        pass
    # NOTE: DO NOT upcast inputs to fp32 if you are using te_compare for any precision other than fp32
    compare_results(
        "te" if is_fp8_torch_dtype(out_dtype) else "torch",
        y_triton,
        y_hipified,
        atol,
        rtol,
        lambda msg: f"y does not match triton <-> hip\n\n{msg}\n",
    )

@pytest.mark.parametrize("in_dtype", test_idtypes_str)
@pytest.mark.parametrize("out_dtype", test_odtypes_str)
@pytest.mark.parametrize("M, N", test_shapes)
@pytest.mark.parametrize("zero_centered_gamma", all_boolean)
def test_layernorm_fwd_fp8_inf_ts_triton(in_dtype, out_dtype, M, N, zero_centered_gamma):
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
    if is_fp8_torch_dtype(out_dtype):
        scale = fill_uniform((1,), torch.float32)
    else:
        scale = torch.empty(0, device="cuda")

    epsilon = 1e-5

    # Run Triton forward.
    amax_triton = torch.zeros((1,), device="cuda") if is_fp8_torch_dtype(out_dtype) else None
    scale_inv_triton = torch.zeros((1,), device="cuda") if is_fp8_torch_dtype(out_dtype) else None
    sm_margin = 0
    y_triton = te_layernorm_fwd_fp8_inf_ts_triton(
        x,
        gamma,
        beta,
        epsilon,
        scale,
        amax_triton,
        scale_inv_triton,
        0,
        te_dtype_to_enum_value(torch_dtype_to_te_dtype(out_dtype)),
        sm_margin,
        zero_centered_gamma,
    )

    # Run Hipified forward reference.
    if is_fp8_torch_dtype(out_dtype):
        amax_hipified = torch.full((1,), 0.0, device="cuda")
        scale_inv_hipified = torch.full((1,), 0.0, device="cuda")
    else:
        amax_hipified = scale_inv_hipified = torch.empty(0, device="cuda")

    y_hipified = torch.ops.tex_ts.layernorm_fwd_fp8_inf_ts(
        x,
        gamma,
        beta,
        epsilon,
        scale,
        amax_hipified,
        scale_inv_hipified,
        0,
        te_dtype_to_enum_value(torch_dtype_to_te_dtype(out_dtype)),
        get_fwd_ln_sm_margin(),
        zero_centered_gamma,
    )

    if is_fp8_torch_dtype(out_dtype):
        atol_stats, _ = get_tolerances(torch.float32)
        rtol_stats = 5e-5
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
        #view y_triton and y_hipified as fp8
        y_triton = y_triton.view(out_dtype)
        y_hipified = y_hipified.view(out_dtype)

    # Assert on y:
    atol, rtol = get_tolerances(out_dtype)
    if out_dtype == torch.float32:
        # Everything pass with default fp32 atol=1e-6. TE C++ test uses atol=5e-7, this tolerance causes
        # some minor failures in Triton test.
        # TODO: Investigate test failures when using atol=5e-7.
        # atol = 5e-7
        pass
    # NOTE: DO NOT upcast inputs to fp32 if you are using te_compare for any precision other than fp32
    compare_results(
        "te" if is_fp8_torch_dtype(out_dtype) else "torch",
        y_triton,
        y_hipified,
        atol,
        rtol,
        lambda msg: f"y does not match triton <-> hip\n\n{msg}\n",
    )

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
    if is_fp8_torch_dtype(out_dtype):
        scale = fill_uniform((1,), torch.float32)
    else:
        scale = torch.empty(0, device="cuda")

    epsilon = 1e-5

    # Run Triton forward.
    y_triton = torch.empty((M, N), dtype=out_dtype, device="cuda")
    amax_triton = torch.zeros((1,), device="cuda") if is_fp8_torch_dtype(out_dtype) else None
    scale_inv_triton = torch.zeros((1,), device="cuda") if is_fp8_torch_dtype(out_dtype) else None
    sm_margin = 0
    y_triton, mu_triton, rsigma_triton = te_layernorm_fwd_fp8_noalloc_triton(
        x,
        gamma,
        beta,
        epsilon,
        scale,
        y_triton,
        amax_triton,
        scale_inv_triton,
        torch_dtype_to_te_dtype(out_dtype),
        sm_margin,
        zero_centered_gamma,
    )

    # Run Hipified forward reference.
    y_hipified = torch.empty((M, N), dtype=out_dtype, device="cuda")
    if is_fp8_torch_dtype(out_dtype):
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
        torch_dtype_to_te_dtype(out_dtype),
        get_fwd_ln_sm_margin(),
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
    if is_fp8_torch_dtype(out_dtype):
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

    # Assert on y:
    atol, rtol = get_tolerances(out_dtype)
    if out_dtype == torch.float32:
        # Everything pass with default fp32 atol=1e-6. TE C++ test uses atol=5e-7, this tolerance causes
        # some minor failures in Triton test.
        # TODO: Investigate test failures when using atol=5e-7.
        # atol = 5e-7
        pass
    # NOTE: DO NOT upcast inputs to fp32 if you are using te_compare for any precision other than fp32
    compare_results(
        "te",
        y_triton,
        y_hipified,
        atol,
        rtol,
        lambda msg: f"y does not match triton <-> hip\n\n{msg}\n",
    )

    if not is_fp8_torch_dtype(out_dtype):
        # Run Triton backward.
        dx_triton, dgamma_triton, dbeta_triton = te_layernorm_bwd_triton(
            dz,
            x,
            mu_triton,
            rsigma_triton,
            gamma,
            sm_margin,
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
        bwd_cmp = "te"
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

@pytest.mark.parametrize("in_dtype", test_idtypes_str)
@pytest.mark.parametrize("M, N", test_shapes)
@pytest.mark.parametrize("zero_centered_gamma", all_boolean)
def test_layernorm_fwd_inf_ts_triton(in_dtype, M, N, zero_centered_gamma):
    # Get Torch data types:
    in_dtype = str_to_torch_dtype(in_dtype)

    # Generate tensors:
    x = fill_uniform((M, N), in_dtype)
    gamma = fill_uniform(N, in_dtype)
    beta = fill_uniform(N, in_dtype)
    dz = fill_uniform((M, N), in_dtype)
 
    epsilon = 1e-5
    sm_margin = 0
    y_triton = te_layernorm_fwd_inf_ts_triton(
        x,
        gamma,
        beta,
        epsilon,
        sm_margin,
        zero_centered_gamma,
    )
    y_hipified= torch.ops.tex_ts.layernorm_fwd_inf_ts(
        x,
        gamma,
        beta,
        epsilon,
        get_fwd_ln_sm_margin(),
        zero_centered_gamma,
    )
    atol, rtol = get_tolerances(in_dtype)
    if in_dtype == torch.float32:
        # Everything pass with default fp32 atol=1e-6. TE C++ test uses atol=5e-7, this tolerance causes
        # some minor failures in Triton test.
        # TODO: Investigate test failures when using atol=5e-7.
        # atol = 5e-7
        pass

    compare_results(
        "te",
        y_triton,
        y_hipified,
        atol,
        rtol,
        lambda msg: f"y does not match triton <-> hip\n\n{msg}\n",
    )

@pytest.mark.parametrize("in_dtype", test_idtypes_str)
@pytest.mark.parametrize("M, N", test_shapes)
@pytest.mark.parametrize("zero_centered_gamma", all_boolean)
def test_layernorm_fwd_noalloc_triton(in_dtype, M, N, zero_centered_gamma):
    # Get Torch data types:
    in_dtype = str_to_torch_dtype(in_dtype)

    # Generate tensors:
    x = fill_uniform((M, N), in_dtype)
    gamma = fill_uniform(N, in_dtype)
    beta = fill_uniform(N, in_dtype)
    dz = fill_uniform((M, N), in_dtype)
 
    epsilon = 1e-5
    sm_margin = 0

    y_triton = torch.empty(M, N, dtype=in_dtype, device='cuda')
    y_triton, mu_triton, rsigma_triton = te_layernorm_fwd_noalloc_triton(
        x,
        gamma,
        beta,
        y_triton,
        epsilon,
        sm_margin,
        zero_centered_gamma,
    )
    y_hipified = torch.empty(M, N, dtype=in_dtype, device='cuda')
    y_hipified, mu_hipified, rsigma_hipified = tex.layernorm_fwd_noalloc(
        x,
        gamma,
        beta,
        y_hipified,
        epsilon,
        get_fwd_ln_sm_margin(),
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
    # check the ln_out 
    atol, rtol = get_tolerances(in_dtype)
    if in_dtype == torch.float32:
        # Everything pass with default fp32 atol=1e-6. TE C++ test uses atol=5e-7, this tolerance causes
        # some minor failures in Triton test.
        # TODO: Investigate test failures when using atol=5e-7.
        # atol = 5e-7
        pass

    compare_results(
        "te",
        y_triton,
        y_hipified,
        atol,
        rtol,
        lambda msg: f"y does not match triton <-> hip\n\n{msg}\n",
    )

@pytest.mark.parametrize("in_dtype", test_idtypes_str)
@pytest.mark.parametrize("M, N", test_shapes)
@pytest.mark.parametrize("zero_centered_gamma", all_boolean)
def test_layernorm_fwd_triton(in_dtype, M, N, zero_centered_gamma):
    # Get Torch data types:
    in_dtype = str_to_torch_dtype(in_dtype)

    # Generate tensors:
    x = fill_uniform((M, N), in_dtype)
    gamma = fill_uniform(N, in_dtype)
    beta = fill_uniform(N, in_dtype)
    dz = fill_uniform((M, N), in_dtype)
 
    epsilon = 1e-5
    sm_margin = 0

    y_triton, mu_triton, rsigma_triton = te_layernorm_fwd_triton(
        x,
        gamma,
        beta,
        epsilon,
        sm_margin,
        zero_centered_gamma,
    )
    y_hipified, mu_hipified, rsigma_hipified = tex.layernorm_fwd(
        x,
        gamma,
        beta,
        epsilon,
        get_fwd_ln_sm_margin(),
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
    # check the ln_out 
    atol, rtol = get_tolerances(in_dtype)
    if in_dtype == torch.float32:
        # Everything pass with default fp32 atol=1e-6. TE C++ test uses atol=5e-7, this tolerance causes
        # some minor failures in Triton test.
        # TODO: Investigate test failures when using atol=5e-7.
        # atol = 5e-7
        pass

    compare_results(
        "te",
        y_triton,
        y_hipified,
        atol,
        rtol,
        lambda msg: f"y does not match triton <-> hip\n\n{msg}\n",
    )
