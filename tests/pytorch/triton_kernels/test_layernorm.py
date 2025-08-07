# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information


import pytest
import torch

from transformer_engine import pytorch as te
from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer
from transformer_engine.pytorch.tensor.quantized_tensor import QuantizedTensor
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
    dtype_tols,
)

def compute_ref_stats(data: torch.Tensor, epsilon: float):
    mu_computed = torch.mean(data.to(torch.float32), dim=1, keepdim=True)
    variance = torch.mean((data.to(torch.float32) - mu_computed).pow(2), dim=1, keepdim=True)
    rsigma_computed = 1.0 / torch.sqrt(variance + epsilon)
    return mu_computed.squeeze(1), rsigma_computed.squeeze(1)

def compute_gamma_py(g_val, zero_centered_gamma):
    if zero_centered_gamma:
        return g_val + 1.0
    else:
        return g_val
        
def compute_ref_output(data: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
                       output: torch.Tensor,
                       mu: torch.Tensor, rsigma: torch.Tensor,
                       amax: torch.Tensor, scale: torch.Tensor, zero_centered_gamma: bool):
    data_float = data.to(torch.float32)
    mu_float = mu.unsqueeze(1).to(torch.float32)
    rsigma_float = rsigma.unsqueeze(1).to(torch.float32)
    gamma_float = gamma.to(torch.float32) # N
    beta_float = beta.to(torch.float32)   # N
    g_tensor = compute_gamma_py(gamma_float, zero_centered_gamma)
    tmp_unscaled = (data_float - mu_float) * rsigma_float * g_tensor + beta_float
    output.copy_((tmp_unscaled * scale).to(output.dtype))

    current_max = torch.max(torch.abs(tmp_unscaled))

    if amax.numel() == 1:
        amax.fill_(current_max)
    else:
        raise ValueError("amax must be a 1-element torch.Tensor.")

test_idtypes_str = input_dtypes_str(["fp32", "fp16", "bf16"])
test_odtypes_str = output_dtypes_str(["fp32", "fp16", "bf16", "fp8e4"])

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
@pytest.mark.parametrize("columnwise", [False, True])
def test_layernorm_fwd_bwd_triton(in_dtype, out_dtype, M, N, zero_centered_gamma, columnwise):

    # Get Torch data types:
    in_dtype = str_to_torch_dtype(in_dtype)
    out_dtype = str_to_torch_dtype(out_dtype)

    # Skip conditions:
    skip_in_dtype_gt_out_dtype(in_dtype, out_dtype)
    skip_mixed_16bit_float_types(in_dtype, out_dtype)
    if columnwise and not is_fp8_torch_dtype(out_dtype):
        pytest.skip("Columnwise only affects quantized calls.")

    # Generate tensors:
    x = fill_uniform((M, N), in_dtype)
    gamma = fill_uniform(N, in_dtype)
    beta = fill_uniform(N, in_dtype)
    dz = fill_uniform((M, N), in_dtype)

    epsilon = 1e-5

    # Run Hipified forward reference.
    quantizer_hipified = None
    quantizer_triton = None
    scale_ref, amax_ref = torch.tensor(1.0), torch.zeros(1, dtype=torch.float32, device='cuda')
    if is_fp8_torch_dtype(out_dtype):
        scale_triton = torch.rand(1, dtype=torch.float32, device='cuda') * 3.0 - 2.0
        amax_triton = torch.zeros(1, dtype=torch.float32, device='cuda')
        quantizer_triton = Float8Quantizer(
            scale=scale_triton,
            amax=amax_triton,
            fp8_dtype=torch_dtype_to_te_dtype(out_dtype),
            columnwise=columnwise
        )
        
        scale_hipified = scale_triton.clone()
        amax_hipified = amax_triton.clone()
        quantizer_hipified = Float8Quantizer(
            scale=scale_hipified,
            amax=amax_hipified,
            fp8_dtype=torch_dtype_to_te_dtype(out_dtype),
            columnwise=columnwise
        )

        scale_ref = scale_triton.clone()
        amax_ref = amax_triton.clone()
    
    y_ref = torch.empty(M, N, dtype=out_dtype, device="cuda")
    mu_ref, rsigma_ref = compute_ref_stats(x, epsilon)
    compute_ref_output(x, gamma, beta, y_ref, mu_ref, rsigma_ref, amax_ref, scale_ref, zero_centered_gamma)
    y_triton, mu_triton, rsigma_triton = te_layernorm_fwd_triton(
        input=x, 
        weight=gamma, 
        bias=beta,
        eps=epsilon, 
        ln_out=None,
        quantizer=quantizer_triton,
        out_dtype=torch_dtype_to_te_dtype(out_dtype),
        sm_margin=get_fwd_ln_sm_margin(),
        zero_centered_gamma=zero_centered_gamma, 
        )
    
    y_hipified, mu_hipified, rsigma_hipified = tex.layernorm_fwd(
        input=x,
        weight=gamma,
        bias=beta,
        eps=epsilon,
        ln_out=None,
        quantizer=quantizer_hipified,
        otype=torch_dtype_to_te_dtype(out_dtype),
        sm_margin=get_fwd_ln_sm_margin(),
        zero_centered_gamma=zero_centered_gamma
    )

    # Run forward
    fwd_cmp = "te"
    atol_fwd, rtol_fwd = get_tolerances(out_dtype)
    
    # Uncommenting this creates some fp32 test failures
    # if out_dtype == torch.float32:
    #     atol_fwd = 5e-7

    y_triton_transpose = None
    y_hipified_transpose = None
    if isinstance(y_triton, QuantizedTensor):
        amax_triton = y_triton._get_quantizer().amax
        scale_inv_triton = y_triton._scale_inv
        y_triton_transpose = y_triton._transpose.view(out_dtype) if columnwise else None
        y_triton_t_invalid = y_triton._transpose_invalid
        y_triton = y_triton._data.view(out_dtype)
        

    if isinstance(y_hipified, QuantizedTensor):
        amax_hipified = y_hipified._get_quantizer().amax
        scale_inv_hipified = y_hipified._scale_inv
        y_hipified_transpose = y_hipified._transpose.view(out_dtype) if columnwise else None
        y_hipified_t_invalid = y_hipified._transpose_invalid
        y_hipified = y_hipified._data.view(out_dtype)
    
    assert y_triton.dtype == out_dtype, f"Expected dtypes to match: {y_triton.dtype} != {out_dtype}"
    assert y_triton.dtype == y_hipified.dtype, f"Expected dtypes to match: {y_triton.dtype} != {y_hipified.dtype}"
    compare_results(
        fwd_cmp,
        y_triton,
        y_hipified,
        atol_fwd,
        rtol_fwd,
        lambda msg: f"y does not match triton <-> hip\n\n{msg}\n",
    )

    compare_results(
        fwd_cmp,
        y_hipified,
        y_ref,
        atol_fwd,
        rtol_fwd,
        lambda msg: f"y does not match hip <-> ref\n\n{msg}\n",
    )

    compare_results(
        fwd_cmp,
        y_triton,
        y_ref,
        atol_fwd,
        rtol_fwd,
        lambda msg: f"y does not match triton <-> ref\n\n{msg}\n",
    )
    if y_triton_transpose is not None:
        if columnwise:
            assert not y_triton_t_invalid, "Expected a valid transpose buffer."
            compare_results(
                fwd_cmp,
                y_triton_transpose,
                y_hipified_transpose,
                atol_fwd,
                rtol_fwd,
                lambda msg: f"y transpose does not match triton <-> hip\n\n{msg}\n",
            )
        else:
            assert y_triton_t_invalid, "Expected an invalid transpose buffer."
    atol_stats, _ = get_tolerances(torch.float32)
    rtol_stats = 5e-5

    compare_results(
        fwd_cmp,
        mu_triton,
        mu_ref,
        atol_stats,
        rtol_stats,
        lambda msg: f"mu does not match triton <-> ref\n\n{msg}\n",
    )
    compare_results(
        fwd_cmp,
        rsigma_triton,
        rsigma_ref,
        atol_stats,
        rtol_stats,
        lambda msg: f"rsigma does not match triton <-> ref\n\n{msg}\n",
    )
    if is_fp8_torch_dtype(out_dtype):
        compare_results(
            "torch",
            amax_triton,
            amax_ref,
            atol_stats,
            rtol_stats,
            lambda msg: f"amax does not match triton <-> ref\n\n{msg}\n",
        )
        compare_results(
            "torch",
            scale_inv_triton,
            1.0/scale_ref,
            atol_stats,
            rtol_stats,
            lambda msg: f"scale_inv does not match triton <-> ref\n\n{msg}\n",
        )

    # Run Triton backward.
    dx_triton, dgamma_triton, dbeta_triton = te_layernorm_bwd_triton(
        dz,
        x,
        mu_triton,
        rsigma_triton,
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


@pytest.mark.parametrize("columnwise", [False, True])
def test_layernorm_fwd_triton_clamp(columnwise):
    """
    Non-regression test for MLPerf divergence issue. We test to ensure that in
    the case of output values beyond the range of the used FP8 dtype, we clamp
    them appropriately.
    """
    # Arbitrary
    M, N = (128, 128)
    zero_centered_gamma = True
    in_dtype = str_to_torch_dtype("fp32")
    out_dtype = str_to_torch_dtype("fp32")
    input_tensor = torch.full((M, N), 1, dtype=in_dtype, device="cuda")
    bias_tensor = fill_uniform(N, in_dtype)
    epsilon = 1e-5
    fwd_ln_sm_margin = get_fwd_ln_sm_margin()

    quantization = 'fp8'
    fp8_dtype = tex.DType.kFloat8E4M3
    gamma_tensor = torch.tensor([2**20] + [0]*127, dtype=in_dtype, device="cuda")

    scale_triton=torch.full([1], 100, dtype=torch.float32, device="cuda")
    amax_triton=torch.empty([1], dtype=torch.float32, device="cuda")
    scale_hip = scale_triton.clone()
    amax_hip = amax_triton.clone()

    quantizer_triton = Float8Quantizer(scale_triton, amax_triton, fp8_dtype, columnwise=columnwise)
    quantizer_hip = Float8Quantizer(scale_hip, amax_hip, fp8_dtype, columnwise=columnwise)


    # run the triton path
    ln_out_triton, mu_triton, rsigma_triton = te_layernorm_fwd_triton(
        input_tensor,
        bias_tensor,
        gamma_tensor,
        epsilon,
        None,
        quantizer_triton, torch_dtype_to_te_dtype(out_dtype),
        fwd_ln_sm_margin,
        zero_centered_gamma
    )

    # run the reference hipified kernel path
    ln_out_hipified, mu_hipified, rsigma_hipified = tex.layernorm_fwd(
        input_tensor,
        bias_tensor,
        gamma_tensor,
        epsilon,
        None,
        quantizer_hip, torch_dtype_to_te_dtype(out_dtype),
        fwd_ln_sm_margin,
        zero_centered_gamma
    )
    tols = dtype_tols(out_dtype if quantization is None else fp8_dtype)
    atol = tols["atol"]
    rtol = tols["rtol"]
    compare_results(
        "te",
        ln_out_triton,
        ln_out_hipified,
        atol,
        rtol,
        lambda msg: f"ln_out does not match triton <-> hip\n\n{msg}\n",
    )
    # TODO(micky774): Remove when `compare_results` correctly handles NaN values
    compare_results(
        "te",
        ln_out_triton.isnan(),
        ln_out_hipified.isnan(),
        atol,
        rtol,
        lambda msg: f"ln_out NaNs do not match triton <-> hip\n\n{msg}\n",
    )

    # rsigma is of type fp32
    compare_results(
        "te",
        rsigma_triton,
        rsigma_hipified,
        1e-6,
        5e-5,
        lambda msg: f"rsigma does not match triton <-> hip\n\n{msg}\n",
    )
    compare_results(
        "te",
        mu_triton,
        mu_hipified,
        1e-6,
        5e-5,
        lambda msg: f"mu does not match triton <-> hip\n\n{msg}\n",
    )
    compare_results(
        "te",
        quantizer_triton.scale,
        quantizer_hip.scale,
        1e-6,
        5e-5,
        lambda msg: f"Quantizer scale does not match triton <-> hip\n\n{msg}\n",
    )
    compare_results(
        "te",
        quantizer_triton.amax,
        quantizer_hip.amax,
        1e-6,
        5e-5,
        lambda msg: f"Quantizer amax does not match triton <-> hip\n\n{msg}\n",
    )
    compare_results(
        "te",
        ln_out_triton._scale_inv,
        ln_out_hipified._scale_inv,
        1e-6,
        5e-5,
        lambda msg: f"Output scale inverse does not match triton <-> hip\n\n{msg}\n",
    )
    if columnwise:
        assert not ln_out_triton._transpose_invalid, "Expected a valid transpose buffer."
        compare_results(
            "te",
            ln_out_triton._transpose,
            ln_out_hipified._transpose,
            atol,
            rtol,
            lambda msg: f"Output transpose does not match triton <-> hip\n\n{msg}\n",
        )
    else:
        assert ln_out_triton._transpose_invalid, "Expected an invalid transpose buffer."
