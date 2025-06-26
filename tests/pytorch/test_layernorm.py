# Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

import pytest
import torch

from transformer_engine.pytorch.cpp_extensions.cast import quantize_triton, dequantize_triton
from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer
from transformer_engine.pytorch.triton_kernels.common import te_dtype_to_torch_dtype, torch_dtype_to_te_dtype
from transformer_engine.pytorch.triton_kernels.layernorm import te_layernorm_fwd_triton
from transformer_engine.pytorch.triton_kernels.norm_common import get_fwd_ln_sm_margin
import transformer_engine_torch as tex
from triton_kernels.test_common import compare_results, fill_uniform, get_tolerances

@pytest.mark.parametrize("shape", 
                         [
                        (128, 128),
                        (71, 229),
                        (29, 541),
                        (768, 6144),
                        (2048, 12288),
                        # (65536, 128),
                        # (65536, 160),
                        # (16384, 1616),
                        # (1, 128),
                        # (1, 1296),
                        # (1, 16),
                        # (5, 160),
                        # (217, 256),
                        ])
@pytest.mark.parametrize("in_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("out_dtype", [tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2])
@pytest.mark.parametrize("zero_centered_gamma", [True, False])
def test_layernorm_fwd_triton_quantized_fp8(shape, in_dtype, out_dtype, zero_centered_gamma):
    M, N = shape
    input_tensor = fill_uniform(shape, dtype=in_dtype)
    gamma = fill_uniform((N, ), dtype=in_dtype)
    beta = fill_uniform((N, ), dtype=in_dtype)
    epsilon = 1e-5
    output_tensor_ref = None

    scale_tensor = torch.rand(1, dtype=torch.float32, device='cuda') * 3.0 - 2.0
    amax_tensor = torch.zeros(1, dtype=torch.float32, device='cuda')
    quantizer = Float8Quantizer(scale=scale_tensor, amax=amax_tensor, fp8_dtype=out_dtype)
    
    output_tensor, mu, rsigma  = te_layernorm_fwd_triton(input_tensor, weight=gamma, bias=beta, eps=epsilon, zero_centered_gamma=zero_centered_gamma, out_dtype=out_dtype, quantizer=quantizer)
    
    quantizer2 = Float8Quantizer(scale=scale_tensor, amax=amax_tensor, fp8_dtype=out_dtype)
    output_tensor_ref, mu_ref, rsigma_ref = tex.layernorm_fwd(
        input_tensor,
        gamma,
        beta,
        epsilon,
        output_tensor_ref,
        quantizer2,
        out_dtype,
        get_fwd_ln_sm_margin(),
        zero_centered_gamma,
    )

    # atol, rtol = get_tolerances(in_dtype)
    print(output_tensor.dtype)
    print(output_tensor_ref.dtype)
    print(output_tensor._fp8_dtype)
    print(output_tensor_ref._fp8_dtype)
    # print("DEQUANTIZING OUTPUT")
    # print(output_tensor.dequantize(dtype=in_dtype))
    # print("DEQUANTIZING OUTPUT REF")
    # # print(output_tensor_ref.dequantize(dtype=in_dtype))
    # dequantized_out = output_tensor.dequantize(dtype=in_dtype)
    # dequantized_out_ref = output_tensor_ref.dequantize(dtype=in_dtype)
    # print(output_tensor_ref.dequantize(dtype=in_dtype))
    dequantized_out = output_tensor.dequantize(dtype=in_dtype)
    dequantized_out_ref = output_tensor_ref.dequantize(dtype=in_dtype)
    atol = 1e-2
    rtol = 1e-2

    # close_matches = torch.isclose(dequantized_out, dequantized_out_ref, atol=atol, rtol=rtol)
    # # close_matches will be: tensor([ True,  True, False,  True])
    # num_close_matches = torch.sum(close_matches).item()
    # total_elements = dequantized_out.numel()
    # percentage_close_match = (num_close_matches / total_elements) * 100
    # mismatched_indices = torch.nonzero(~close_matches, as_tuple=False)
    # print("Indices of mismatched elements:", mismatched_indices)

    # print(f"Number of close matches: {num_close_matches}")
    # print(f"Total elements: {total_elements}")
    # print(f"Percentage of close match: {percentage_close_match:.2f}%")

    # mismatched_indices_tuple = torch.nonzero(~close_matches, as_tuple=True)
    # row_indices = mismatched_indices_tuple[0]
    # col_indices = mismatched_indices_tuple[1]

    # out_vals = dequantized_out[row_indices, col_indices]
    # ref_vals = dequantized_out_ref[row_indices, col_indices]

    # diffs = torch.abs(out_vals - ref_vals)
    # tolerances = atol + rtol * torch.abs(ref_vals)
    # is_diff_greater = diffs > tolerances
    # print("Inspecting all mismatched elements at once:")
    # for i in range(len(row_indices)):
    #     print(f"Index: ({row_indices[i].item()}, {col_indices[i].item()})")
    #     print(f"  dequantized_out value: {out_vals[i].item()}")
    #     print(f"  dequantized_out_ref value: {ref_vals[i].item()}")
    #     print(f"  Difference: {diffs[i].item()}")
    #     print(f"  Tolerance: {tolerances[i].item()}")
    #     print(f"  Is difference > tolerance? {is_diff_greater[i].item()}")
    #     print("-" * 20)

    assert output_tensor.dtype == output_tensor_ref.dtype, f"Dtypes do not match: {output_tensor.dtype} vs {output_tensor_ref.dtype}"
    assert output_tensor._fp8_dtype == output_tensor_ref._fp8_dtype, f"FP8 dtypes do not match: {output_tensor._fp8_dtype} vs {output_tensor_ref._fp8_dtype}"
    # assert torch.equal(output_tensor._data, output_tensor_ref._data), 'Quantized results do not match!'
    assert torch.allclose(dequantized_out, dequantized_out_ref, atol=atol, rtol=rtol), 'Quantized results do not match!'
    # assert torch.allclose(mu, mu_ref, atol=atol, rtol=rtol), 'Mu results do not match!'
    # assert torch.allclose(rsigma, rsigma_ref, atol=atol, rtol=rtol), 'RSigma results do not match!'
