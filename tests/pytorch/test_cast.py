# Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

import math
import pytest
import torch
import functools
from typing import Optional, List, Tuple, Union

from transformer_engine.pytorch.cpp_extensions.cast import quantize
from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer
from transformer_engine.pytorch.triton_kernels.common import te_dtype_to_torch_dtype
import transformer_engine_torch as tex
from triton_kernels.test_common import compare_results, fill_uniform, get_tolerances
from transformer_engine.pytorch.triton_kernels.common import (
    torch_e4m3_type,
    torch_e5m2_type,
)

# Define the baseline reference implementation
def compute_ref_quantize(
    data: torch.Tensor,
    scale: torch.Tensor,
    out_dtype: tex.DType,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reference implementation for quantization (casting and scaling) in PyTorch.
    Mimics the C++ `compute_ref` function.
    """
    assert data.is_cuda, "Reference should operate on CUDA tensors for consistency with TE"
    assert scale.is_cuda, "Scale tensor must be on CUDA"
    assert scale.numel() == 1, "Currently only supports per-tensor scale (single value)"

    input_fp32 = data.to(torch.float32)

    # Compute AMAX
    amax = torch.max(torch.abs(input_fp32)).to(torch.float32).reshape(1)

    # Apply Scaling
    # The C++ `compute_ref` multiplies by `scale.item()`
    scaled_input = input_fp32 * scale.item() # Use .item() for scalar multiplication

    # Determine the PyTorch dtype for the target tex.DType for direct conversion
    # The crucial part here is that `torch.float8_e4m3` and `torch.float8_e5m2`
    # are custom dtypes registered by Transformer Engine itself.
    # So, `.to()` will handle the lossy conversion to/from these custom types.
    target_torch_dtype = None
    if out_dtype == tex.DType.kFloat8E4M3:
        target_torch_dtype = torch_e4m3_type
    elif out_dtype == tex.DType.kFloat8E5M2:
        target_torch_dtype = torch_e5m2_type
    elif out_dtype == tex.DType.kFloat16:
        target_torch_dtype = torch.float16
    elif out_dtype == tex.DType.kBFloat16:
        target_torch_dtype = torch.bfloat16
    elif out_dtype == tex.DType.kFloat32:
        target_torch_dtype = torch.float32
    else:
        raise ValueError(f"Unsupported output DType for reference: {out_dtype}")

    # Perform the lossy conversion to the target dtype, then convert back to float32 for comparison.
    # This simulates the precision loss of the quantization.
    output_tensor_ref = scaled_input.to(target_torch_dtype)

    # Compute reference scale_inv (based on the initial scale)
    ref_scale_inv = torch.tensor([1.0 / scale.item()], dtype=torch.float32, device='cuda')

    return output_tensor_ref, amax, ref_scale_inv

@pytest.mark.parametrize("shape", 
                         [
                        (128, 128),
                        # (256, 256),
                        # (768, 1024),
                        # (256, 65536),
                        # (2048, 12288),
                        # (65536, 128),
                        # (65536, 160),
                        # (16384, 1616),
                        # (1, 128),
                        # (1, 1296),
                        # (1, 16),
                        # (5, 160),
                        # (217, 256),
                        ])
# @pytest.mark.parametrize("in_dtype", [torch.float32, torch.float16, torch.bfloat16])
# @pytest.mark.parametrize("out_dtype", [tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2])
@pytest.mark.parametrize("in_dtype", [torch.float32])
@pytest.mark.parametrize("out_dtype", [tex.DType.kFloat8E4M3])
def test_quantize(shape, in_dtype, out_dtype):
    # input_tensor = torch.rand(M, N, dtype=torch.float32, device='cuda') * 3.0 - 2.0
    # input_tensor = input_tensor.to(in_dtype)
    full_size = math.prod(shape)
    input_tensor = fill_uniform(shape, dtype=in_dtype)

    # scale_tensor = torch.full((1,), 1.0, dtype=torch.float32, device='cuda')
    scale_tensor = torch.rand(1, dtype=torch.float32, device='cuda') * 3.0 - 2.0
    amax_tensor = torch.zeros(1, dtype=torch.float32, device='cuda')
    quantizer = Float8Quantizer(scale=scale_tensor, amax=amax_tensor, fp8_dtype=out_dtype)

    output_tensor  = quantize(input_tensor, quantizer=quantizer)
    
    print(f"\nScale Tensor: {scale_tensor.item()}")
    print(f"\nInput Tensor Shape: {input_tensor.shape}, Dtype: {input_tensor.dtype}")
    print(f"Output Tensor Shape: {output_tensor.shape}, Dtype: {output_tensor.dtype}")
    print("Quantized Output Tensor (first few elements):\n", output_tensor[:5, :5] if output_tensor.ndim >= 2 else output_tensor[:5])

    quantizer = output_tensor._get_quantizer()

    actual_amax = quantizer.amax
    actual_scale_inv = output_tensor._scale_inv
    atol_output, rtol_output = get_tolerances(te_dtype_to_torch_dtype(out_dtype))
    atol_amax, rtol_amax = get_tolerances(torch.float32)

    ref_output_tensor, ref_amax, ref_scale_inv = compute_ref_quantize(
        input_tensor,
        quantizer.scale,
        out_dtype
    )
    print("Reference Output Tensor (first few elements):\n", ref_output_tensor[:5, :5] if ref_output_tensor.ndim >= 2 else ref_output_tensor[:5])

    quantizer2 = Float8Quantizer(scale=scale_tensor, amax=amax_tensor, fp8_dtype=out_dtype)
    te_quantized_out = quantizer2.quantize(input_tensor)
    te_quantized_out = output_tensor.dequantize(dtype=torch.float32)
    print("Actual Output Tensor (first few elements):\n", te_quantized_out[:5, :5] if te_quantized_out.ndim >= 2 else te_quantized_out[:5])

    compare_results("torch", output_tensor, ref_output_tensor, atol_output, rtol_output, "output_tensor mismatch")
    compare_results("te", actual_amax, ref_amax, atol_amax, rtol_amax, "amax mismatch")
    compare_results("te", actual_scale_inv, ref_scale_inv, atol_amax, rtol_amax, "scale_inv mismatch")