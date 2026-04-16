# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""MXFP4 GEMM tests: native AITER a4w4 GEMM vs Python reference GEMM.

Requires the aiter package (ROCm gfx950 only).
"""

import pytest
import torch
import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch.tensor.mxfp4_tensor import MXFP4Quantizer
from transformer_engine.pytorch.custom_recipes.quantization_mxfp4 import MXFP4QuantizerRef


recipe_available, reason_for_no_recipe = te.is_mxfp4_available(return_reason=True)

try:
    import aiter  # noqa: F401

    _aiter_available = True
except ImportError:
    _aiter_available = False

BLOCK_SIZE = 32


def check_mxfp4_gemm_versus_reference(
    x_dtype: torch.dtype,
    w_dtype: torch.dtype,
    out_dtype: torch.dtype,
    M: int,
    K: int,
    N: int,
    accumulate: bool,
):
    te_dtype = tex.DType.kFloat4E2M1
    device = "cuda"
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    x = torch.randn((M, K), dtype=x_dtype, device=device)
    w = torch.randn((N, K), dtype=w_dtype, device=device)

    if accumulate:
        out = torch.randn((M, N), dtype=out_dtype, device=device)
    else:
        out = None

    # Native MXFP4 quantization (shuffled for AITER GEMM)
    x_quantizer = MXFP4Quantizer(
        fp4_dtype=te_dtype,
        rowwise=True,
        columnwise=True,
        shuffle_B_matrix_for_aiter=False,
        shuffle_scales=True,
        use_hadamard=False,
    )
    w_quantizer = MXFP4Quantizer(
        fp4_dtype=te_dtype,
        rowwise=True,
        columnwise=True,
        shuffle_B_matrix_for_aiter=True,
        shuffle_scales=True,
        use_hadamard=False,
    )

    # Reference quantization (plain layout, no shuffle)
    x_quantizer_ref = MXFP4Quantizer(
        fp4_dtype=te_dtype,
        rowwise=True,
        columnwise=True,
        shuffle_B_matrix_for_aiter=False,
        shuffle_scales=False,
        use_hadamard=False,
    )
    w_quantizer_ref = MXFP4Quantizer(
        fp4_dtype=te_dtype,
        rowwise=True,
        columnwise=True,
        shuffle_B_matrix_for_aiter=False,
        shuffle_scales=False,
        use_hadamard=False,
    )

    x_mxfp4 = x_quantizer.make_empty((M, K), dtype=x_dtype, device=device, requires_grad=False)
    x_mxfp4 = x_quantizer.update_quantized(x, x_mxfp4)
    w_mxfp4 = w_quantizer.make_empty((N, K), dtype=w_dtype, device=device, requires_grad=False)
    w_mxfp4 = w_quantizer.update_quantized(w, w_mxfp4)

    x_mxfp4_ref = x_quantizer_ref.make_empty((M, K), dtype=x_dtype, device=device, requires_grad=False)
    x_mxfp4_ref = x_quantizer_ref.update_quantized(x, x_mxfp4_ref)
    w_mxfp4_ref = w_quantizer_ref.make_empty((N, K), dtype=w_dtype, device=device, requires_grad=False)
    w_mxfp4_ref = w_quantizer_ref.update_quantized(w, w_mxfp4_ref)

    # Extract un-shuffled quantized data for the reference GEMM
    qx_data = x_mxfp4_ref._rowwise_data.view(dtype=torch.uint8)[:M, :]
    qw_data = w_mxfp4_ref._rowwise_data.view(dtype=torch.uint8)[:N, :]
    sx_native = x_mxfp4_ref._rowwise_scale_inv
    sw_native = w_mxfp4_ref._rowwise_scale_inv

    expected_scale_cols = K // BLOCK_SIZE
    sx_trimmed = sx_native[:M, :expected_scale_cols]
    sw_trimmed = sw_native[:N, :expected_scale_cols]

    # Reference GEMM: dequantize to FP32, torch.mm, cast to out_dtype
    ref_quantizer = MXFP4QuantizerRef(rowwise=True, columnwise=True)
    y_ref = ref_quantizer.qgemm(
        qx=qx_data,
        qw=qw_data,
        out_dtype=out_dtype,
        sx=sx_trimmed,
        sw=sw_trimmed,
        bias=None,
        out=out.clone() if accumulate else None,
        accumulate=accumulate,
    )

    # Native AITER GEMM via general_gemm
    from transformer_engine.pytorch.cpp_extensions.gemm import general_gemm

    y_native, *_ = general_gemm(
        w_mxfp4,
        x_mxfp4,
        out_dtype=out_dtype,
        quantization_params=None,
        bias=None,
        use_split_accumulator=False,
        out=out.clone() if accumulate else None,
        accumulate=accumulate,
    )

    assert y_ref is not y_native
    assert not torch.isnan(y_ref.float()).all(), "All reference elements are NaN"

    y_ref = torch.where(y_ref.isnan(), torch.zeros_like(y_ref), y_ref)
    y_native = torch.where(y_native.isnan(), torch.zeros_like(y_native), y_native)

    torch.testing.assert_close(y_native, y_ref, atol=8e-3, rtol=8e-3)


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.skipif(not _aiter_available, reason="aiter package not available")
@pytest.mark.parametrize(
    "M, K, N",
    [
        (128, 128, 128),
        (256, 256, 256),
        (256, 1024, 256),
        (1024, 1024, 1024),
        (4096, 512, 3072),
    ],
)
@pytest.mark.parametrize("x_dtype", [torch.bfloat16], ids=str)
@pytest.mark.parametrize("w_dtype", [torch.bfloat16], ids=str)
@pytest.mark.parametrize("out_dtype", [torch.bfloat16], ids=str)
@pytest.mark.parametrize("accumulate", [True, False], ids=["accumulate", "no_accumulate"])
def test_mxfp4_gemm_versus_reference(
    M: int,
    K: int,
    N: int,
    x_dtype: torch.dtype,
    w_dtype: torch.dtype,
    out_dtype: torch.dtype,
    accumulate: bool
):
    check_mxfp4_gemm_versus_reference(
        x_dtype=x_dtype,
        w_dtype=w_dtype,
        out_dtype=out_dtype,
        M=M,
        K=K,
        N=N,
        accumulate=accumulate,
    )
