# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
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
from test_common import te_compare_results, fill_uniform, get_tolerances

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

    atol_fp8, rtol_fp8 = get_tolerances(torch_out_dtype)
    if rowwise:
        te_compare_results(
            quantized_out_triton._rowwise_data.view(torch_out_dtype),
            quantized_out_hip._rowwise_data.view(torch_out_dtype),
            atol_fp8, rtol_fp8,
            msg="rowwise data doesn't match"
        )
        te_compare_results(
            quantized_out_triton._rowwise_scale_inv,
            quantized_out_hip._rowwise_scale_inv,
            0.0, 0.0,
            msg="rowwise scale inv doesn't match",
            use_torch_semantics=True
        )
    if columnwise:
        te_compare_results(
            quantized_out_triton._columnwise_data.view(torch_out_dtype),
            quantized_out_hip._columnwise_data.view(torch_out_dtype),
            atol_fp8, rtol_fp8,
            msg="columnwise data doesn't match"
        )
        te_compare_results(
            quantized_out_triton._columnwise_scale_inv,
            quantized_out_hip._columnwise_scale_inv,
            0.0, 0.0,
            msg="columnwise scale inv doesn't match",
            use_torch_semantics=True
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
    te_compare_results(out_triton, out_hip, atol, rtol, "output doesn't match", use_torch_semantics=True)


@pytest.mark.parametrize("shape",
                         [
                        (128, 128),
                        (256, 256),
                        (2048, 6144),
                        (16384, 128),
                        (8, 32, 1024),
                        ])
@pytest.mark.parametrize("in_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("fp8_dtype", [tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2])
def test_mxfp8_rowwise_to_columnwise_restripe(shape, in_dtype, fp8_dtype):
    """Rowwise MXFP8 + update_usage(columnwise) matches dequant + columnwise quantize."""
    torch_out_dtype = te_dtype_to_torch_dtype(fp8_dtype)
    x = fill_uniform(shape, dtype=in_dtype)

    row_quantizer = MXFP8Quantizer(fp8_dtype, rowwise=True, columnwise=False)
    row_tensor = row_quantizer(x)

    hp = row_tensor.dequantize(dtype=in_dtype)
    col_quantizer = MXFP8Quantizer(fp8_dtype, rowwise=False, columnwise=True)
    col_ref = col_quantizer.make_empty(x.shape, dtype=in_dtype)
    te_quantize_triton(hp, quantizer=col_quantizer, output=col_ref)

    row_tensor.update_usage(rowwise_usage=True, columnwise_usage=True)
    assert row_tensor._columnwise_data is not None
    assert row_tensor._rowwise_data is not None

    atol_fp8, rtol_fp8 = get_tolerances(torch_out_dtype)
    te_compare_results(
        row_tensor._columnwise_data.view(torch_out_dtype),
        col_ref._columnwise_data.view(torch_out_dtype),
        atol_fp8, rtol_fp8,
        msg="restriped columnwise data doesn't match dequant+requant",
    )
    te_compare_results(
        row_tensor._columnwise_scale_inv,
        col_ref._columnwise_scale_inv,
        0.0, 0.0,
        msg="restriped columnwise scale inv doesn't match",
        use_torch_semantics=True,
    )


def test_mxfp8_update_usage_columnwise_only_drops_rowwise():
    x = fill_uniform((128, 256), dtype=torch.bfloat16)
    tensor = MXFP8Quantizer(tex.DType.kFloat8E4M3, rowwise=True, columnwise=False)(x)
    tensor.update_usage(rowwise_usage=False, columnwise_usage=True)
    assert tensor._rowwise_data is None
    assert tensor._columnwise_data is not None
    assert tensor._columnwise_scale_inv is not None
