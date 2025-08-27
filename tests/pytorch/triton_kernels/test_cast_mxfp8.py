# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

import math
import os
import struct
import pytest
import torch

from transformer_engine.pytorch.constants import MXFP8_BLOCK_SCALING_SIZE
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
from transformer_engine.pytorch.triton_kernels.cast import te_dequantize_triton, te_quantize_triton
from transformer_engine.pytorch.triton_kernels.common import te_dtype_to_torch_dtype
import transformer_engine_torch as tex
from test_common import compare_results, fill_uniform, get_tolerances

@pytest.mark.parametrize("shape",
                         [
                        (128, 128),
                        (256, 256),
                        (256, 65536),
                        (2048, 6144),
                        (16384, 128),
                        (32768, 160),
                        (4096, 1632),
                        (8, 32, 1024),
                        (16, 8, 4, 512),
                        ])
@pytest.mark.parametrize("in_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("fp8_dtype", [tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2])
@pytest.mark.parametrize(("rowwise", "columnwise"), ((True, True), (False, True), (True, False)))
def test_quantize_mxfp8(shape, in_dtype, fp8_dtype, rowwise, columnwise):
    torch_out_dtype = te_dtype_to_torch_dtype(fp8_dtype)
    input_tensor = fill_uniform(shape, dtype=in_dtype)

    triton_quantizer = MXFP8Quantizer(fp8_dtype, rowwise=rowwise, columnwise=columnwise)
    hip_quantizer = MXFP8Quantizer(fp8_dtype, rowwise=rowwise, columnwise=columnwise)

    out_triton = triton_quantizer.make_empty(input_tensor.shape, dtype=in_dtype)
    out_hip = triton_quantizer.make_empty(input_tensor.shape, dtype=in_dtype)

    quantized_out_triton  = te_quantize_triton(input_tensor, quantizer=triton_quantizer, output=out_triton)
    quantized_out_hip = tex.quantize(input_tensor, quantizer=hip_quantizer, output=out_hip)

    cmp = "te"
    atol_fp8, rtol_fp8 = get_tolerances(torch_out_dtype)
    if rowwise:
        compare_results(
            cmp,
            quantized_out_triton._rowwise_data.view(torch_out_dtype),
            quantized_out_hip._rowwise_data.view(torch_out_dtype),
            atol_fp8, rtol_fp8, "rowwise data doesn't match"
        )
        compare_results(
            "torch",
            quantized_out_triton._rowwise_scale_inv,
            quantized_out_hip._rowwise_scale_inv,
            0.0, 0.0,
            "rowwise scale inv doesn't match")
    if columnwise:
        compare_results(
            cmp,
            quantized_out_triton._columnwise_data.view(torch_out_dtype),
            quantized_out_hip._columnwise_data.view(torch_out_dtype),
            atol_fp8, rtol_fp8, "columnwise data doesn't match"
        )
        compare_results(
            "torch",
            quantized_out_triton._columnwise_scale_inv,
            quantized_out_hip._columnwise_scale_inv,
            0.0, 0.0, "columnwise scale inv doesn't match"
        )

@pytest.mark.parametrize("shape",
                         [
                        (128, 128),
                        (256, 256),
                        (256, 65536),
                        (2048, 6144),
                        (16384, 128),
                        (32768, 160),
                        (4096, 1632),
                        (8, 32, 1024),
                        (16, 8, 4, 512),
                        ])
@pytest.mark.parametrize("in_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("out_dtype", [tex.DType.kFloat32, tex.DType.kFloat16, tex.DType.kBFloat16])
@pytest.mark.parametrize("fp8_dtype", [tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2])
@pytest.mark.parametrize(("rowwise", "columnwise"), ((True, True), (False, True), (True, False)))
def test_dequantize_mxfp8(shape, in_dtype, out_dtype, fp8_dtype, rowwise, columnwise):
    # TODO(micky774): Remove when we support cloning from columnwise-only data
    if not rowwise:
        pytest.skip(
            "The test requires cloning an MXFP8Tensor, but that is only "
            "supported when there is rowwise data available."
        )

    quantizer = MXFP8Quantizer(fp8_dtype=fp8_dtype, rowwise=rowwise, columnwise=columnwise)
    in_triton = quantizer(fill_uniform(shape, in_dtype))
    in_hip = in_triton.clone()

    out_triton = te_dequantize_triton(in_triton, out_dtype)
    out_hip = tex.dequantize(in_hip, out_dtype)

    atol, rtol = get_tolerances(te_dtype_to_torch_dtype(out_dtype))
    compare_results('te', out_triton, out_hip, atol, rtol, "output doesn't match")