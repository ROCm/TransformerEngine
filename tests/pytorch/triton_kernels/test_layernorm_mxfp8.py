# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information


import pytest
import torch

from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
from transformer_engine.pytorch.triton_kernels.cast import te_dequantize_triton
import transformer_engine_torch as tex
from transformer_engine.pytorch.triton_kernels.common import torch_dtype_to_te_dtype
from transformer_engine.pytorch.triton_kernels.norm_common import get_fwd_ln_sm_margin
from transformer_engine.pytorch.triton_kernels.layernorm import te_layernorm_fwd_triton
from test_common import (
    get_tolerances,
    input_dtypes_str,
    output_dtypes_str,
    str_to_torch_dtype,
    skip_in_dtype_gt_out_dtype,
    skip_mixed_16bit_float_types,
    fill_uniform,
    compare_results,
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
test_odtypes_str = output_dtypes_str(["fp8e4", "fp8e5"])

test_shapes = [
        (32, 32),
        (768, 2304),
        (2048, 12288),
]

all_boolean = [False, True]

@pytest.mark.parametrize("in_dtype", test_idtypes_str)
@pytest.mark.parametrize("out_dtype", test_odtypes_str)
@pytest.mark.parametrize("M, N", test_shapes)
@pytest.mark.parametrize("zero_centered_gamma", all_boolean)
def test_layernorm_fwd_triton(in_dtype, out_dtype, M, N, zero_centered_gamma):

    # Get Torch data types:
    in_dtype = str_to_torch_dtype(in_dtype)
    out_dtype = str_to_torch_dtype(out_dtype)
    te_out_dtype = torch_dtype_to_te_dtype(out_dtype)

    # Skip conditions:
    skip_in_dtype_gt_out_dtype(in_dtype, out_dtype)
    skip_mixed_16bit_float_types(in_dtype, out_dtype)

    # Generate tensors:
    x = fill_uniform((M, N), in_dtype)
    gamma = fill_uniform(N, in_dtype)
    beta = fill_uniform(N, in_dtype)
    scale_ref, amax_ref = torch.tensor(1.0), torch.zeros(1, dtype=torch.float32, device='cuda')
    epsilon = 1e-5

    # Run Hipified forward reference.
    quantizer_triton = MXFP8Quantizer(te_out_dtype)
    
    y_ref = torch.empty(M, N, dtype=in_dtype, device="cuda")
    mu_ref, rsigma_ref = compute_ref_stats(x, epsilon)
    compute_ref_output(x, gamma, beta, y_ref, mu_ref, rsigma_ref, amax_ref, scale_ref, zero_centered_gamma)
    y_triton, mu_triton, rsigma_triton = te_layernorm_fwd_triton(
        input=x, 
        weight=gamma, 
        bias=beta,
        eps=epsilon, 
        ln_out=None,
        quantizer=quantizer_triton,
        otype=torch_dtype_to_te_dtype(out_dtype),
        sm_margin=get_fwd_ln_sm_margin(),
        zero_centered_gamma=zero_centered_gamma, 
        )
    dequantized_out_rowwise_triton = te_dequantize_triton(y_triton, dtype=torch_dtype_to_te_dtype(in_dtype))
    y_triton._rowwise_data = None
    dequantized_out_colwise_triton = te_dequantize_triton(y_triton, dtype=torch_dtype_to_te_dtype(in_dtype))

    if te_out_dtype == tex.DType.kFloat8E5M2:
        atol = 1.25e-1
        rtol = 1.25e-1
    elif te_out_dtype == tex.DType.kFloat8E4M3:
        if in_dtype == torch.float16:
            atol = 7e-2
            rtol = 7e-2
        else:
            atol = 6.25e-2
            rtol = 6.25e-2
    
    # Run forward
    fwd_cmp = "te"
    
    compare_results(
        fwd_cmp,
        dequantized_out_rowwise_triton,
        y_ref,
        atol,
        rtol,
        lambda msg: f"rowwise dequantized output doesn't match reference output\n{msg}\n",
    )
    
    compare_results(
        fwd_cmp,
        dequantized_out_colwise_triton,
        y_ref,
        atol,
        rtol,
        lambda msg: f"colwise dequantized output doesn't match reference output\n{msg}\n",
    )

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

