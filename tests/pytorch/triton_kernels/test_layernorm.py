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

def compute_ref_stats(data: torch.Tensor, epsilon: float):
    mu_computed = torch.mean(data.to(torch.float32), dim=1, keepdim=True)
    variance = torch.mean((data.to(torch.float32) - mu_computed).pow(2), dim=1, keepdim=True)
    rsigma_computed = 1.0 / torch.sqrt(variance + epsilon)
    return mu_computed, rsigma_computed

def compute_gamma_py(g_val, zero_centered_gamma):
        if zero_centered_gamma:
            return g_val + 1.0
        else:
            return g_val
        
def compute_ref_output(data: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
                       output: torch.Tensor,
                       mu: torch.Tensor, rsigma: torch.Tensor,
                       amax, scale: float, zero_centered_gamma: bool):
    data_float = data.to(torch.float32)
    mu_float = mu.to(torch.float32)
    rsigma_float = rsigma.to(torch.float32)
    gamma_float = gamma.to(torch.float32) # M
    beta_float = beta.to(torch.float32)   # M
    g_tensor = compute_gamma_py(gamma_float, zero_centered_gamma)
    tmp_unscaled = (data_float - mu_float) * rsigma_float * g_tensor + beta_float
    output.copy_((tmp_unscaled * scale).to(output.dtype))

    current_max = torch.max(torch.abs(tmp_unscaled))

    if amax.numel() == 1:
        amax.fill_(current_max)
    else:
        raise ValueError("amax must be a 1-element torch.Tensor.")

test_idtypes_str = input_dtypes_str(["fp32", "fp16", "bf16"])
# TODO: bring back fp8 after refactoring te_layernorm_fwd_triton
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

    # Run Hipified forward reference.
    y_hipified = torch.empty((M, N), dtype=out_dtype, device="cuda")
    scale_ref, amax_ref = 1.0, torch.zeros(1, dtype=torch.float32, device='cuda')
    y_ref = torch.empty(M, N, dtype=out_dtype, device="cuda")
    mu_ref, rsigma_ref = compute_ref_stats(x, epsilon)
    compute_ref_output(x, gamma, beta, y_ref, mu_ref, rsigma_ref, amax_ref, scale_ref, zero_centered_gamma)
    
    y_hipified, mu_hipified, rsigma_hipified = tex.layernorm_fwd(
        x,
        gamma,
        beta,
        epsilon,
        y_hipified,
        None,
        torch_dtype_to_te_dtype(out_dtype),
        get_fwd_ln_sm_margin(),
        zero_centered_gamma,
    )

    fwd_cmp = "te"
    atol_fwd, rtol_fwd = get_tolerances(out_dtype)
    
    # Uncommenting below lines will cause test failures.
    # if out_dtype == torch.float32:
    #     atol_fwd = 5e-7

    compare_results(
        fwd_cmp,
        y_hipified,
        y_ref,
        atol_fwd,
        rtol_fwd,
        lambda msg: f"y does not match hip <-> ref\n\n{msg}\n",
    )

    # Run Triton backward.
    dx_triton, dgamma_triton, dbeta_triton = te_layernorm_bwd_triton(
        dz,
        x,
        mu_hipified,
        rsigma_hipified,
        gamma,
        get_bwd_ln_sm_margin(),
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