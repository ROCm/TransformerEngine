# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""MXFP4 GEMM tests.

This module tests native MXFP4 GEMM implementations against Python reference GEMMs:
- AITER a4w4 kernels
- hipBLASLt F4F4 kernels

Requires the aiter package (ROCm gfx950 only).
Requires the hipBLASLt>=1.3.0 (ROCm gfx950 only).
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


def _quantize_pair(x, w, M, K, N, x_dtype, w_dtype, device, *, shuffle: bool, swizzled_scales=None):
    """Quantize (x, w) to MXFP4.
    """
    te_dtype = tex.DType.kFloat4E2M1
    if swizzled_scales is None:
        swizzled_scales = shuffle
    x_quantizer = MXFP4Quantizer(
        fp4_dtype=te_dtype,
        rowwise=True,
        columnwise=True,
        shuffle_rowwise_data=False,
        shuffle_columnwise_data=shuffle,
        with_gemm_swizzled_scales=swizzled_scales,
        use_hadamard=False,
    )
    w_quantizer = MXFP4Quantizer(
        fp4_dtype=te_dtype,
        rowwise=True,
        columnwise=True,
        shuffle_rowwise_data=shuffle,
        shuffle_columnwise_data=shuffle,
        with_gemm_swizzled_scales=swizzled_scales,
        use_hadamard=False,
    )
    x_mxfp4 = x_quantizer.make_empty((M, K), dtype=x_dtype, device=device, requires_grad=False)
    x_mxfp4 = x_quantizer.update_quantized(x, x_mxfp4)
    w_mxfp4 = w_quantizer.make_empty((N, K), dtype=w_dtype, device=device, requires_grad=False)
    w_mxfp4 = w_quantizer.update_quantized(w, w_mxfp4)
    return x_mxfp4, w_mxfp4


def _reference_gemm(x, w, M, K, N, out_dtype, out, accumulate, device):
    """MXFP4 reference GEMM: quantize plain, dequant-matmul via MXFP4QuantizerRef."""
    te_dtype = tex.DType.kFloat4E2M1
    ref_quantizer_cfg = dict(
        fp4_dtype=te_dtype,
        rowwise=True,
        columnwise=True,
        shuffle_rowwise_data=False,
        shuffle_columnwise_data=False,
        with_gemm_swizzled_scales=False,
        use_hadamard=False,
    )
    x_ref_q = MXFP4Quantizer(**ref_quantizer_cfg)
    w_ref_q = MXFP4Quantizer(**ref_quantizer_cfg)
    x_ref = x_ref_q.make_empty((M, K), dtype=x.dtype, device=device, requires_grad=False)
    x_ref = x_ref_q.update_quantized(x, x_ref)
    w_ref = w_ref_q.make_empty((N, K), dtype=w.dtype, device=device, requires_grad=False)
    w_ref = w_ref_q.update_quantized(w, w_ref)

    qx_data = x_ref._rowwise_data.view(dtype=torch.uint8)[:M, :]
    qw_data = w_ref._rowwise_data.view(dtype=torch.uint8)[:N, :]
    expected_scale_cols = K // BLOCK_SIZE
    sx_trimmed = x_ref._rowwise_scale_inv[:M, :expected_scale_cols]
    sw_trimmed = w_ref._rowwise_scale_inv[:N, :expected_scale_cols]

    ref_quantizer = MXFP4QuantizerRef(rowwise=True, columnwise=True)
    return ref_quantizer.qgemm(
        qx=qx_data,
        qw=qw_data,
        m_params=None,  # MMParams not used in reference
        out_dtype=out_dtype,
        sx=sx_trimmed,
        sw=sw_trimmed,
        bias=None,
        out=out.clone() if accumulate else None,
        accumulate=accumulate,
    )


def _native_gemm(w_mxfp4, x_mxfp4, out_dtype, out, accumulate):
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
    return y_native


def check_mxfp4_gemm_versus_reference(
    x_dtype: torch.dtype,
    w_dtype: torch.dtype,
    out_dtype: torch.dtype,
    M: int,
    K: int,
    N: int,
    accumulate: bool,
    *,
    shuffle: bool,
    swizzled_scales=None,
):
    device = "cuda"
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    x = torch.randn((M, K), dtype=x_dtype, device=device)
    w = torch.randn((N, K), dtype=w_dtype, device=device)
    out = torch.randn((M, N), dtype=out_dtype, device=device) if accumulate else None

    x_mxfp4, w_mxfp4 = _quantize_pair(
        x, w, M, K, N, x_dtype, w_dtype, device, shuffle=shuffle, swizzled_scales=swizzled_scales
    )

    y_ref = _reference_gemm(x, w, M, K, N, out_dtype, out, accumulate, device)
    y_native = _native_gemm(w_mxfp4, x_mxfp4, out_dtype, out, accumulate)

    assert y_ref is not y_native
    assert not torch.isnan(y_ref.float()).all(), "All reference elements are NaN"

    y_ref = torch.where(y_ref.isnan(), torch.zeros_like(y_ref), y_ref)
    y_native = torch.where(y_native.isnan(), torch.zeros_like(y_native), y_native)

    torch.testing.assert_close(y_native, y_ref, atol=8e-3, rtol=8e-3)


_SHAPES = [
    (128, 128, 128),
    (256, 256, 256),
    (256, 1024, 256),
    (1024, 1024, 1024),
    (4096, 512, 3072),
]


# AITER a4w4 backend: shuffled FP4 data + swizzled scales. Requires the aiter package.
@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.skipif(not _aiter_available, reason="aiter package not available")
@pytest.mark.parametrize("M, K, N", _SHAPES)
@pytest.mark.parametrize("x_dtype", [torch.bfloat16], ids=str)
@pytest.mark.parametrize("w_dtype", [torch.bfloat16], ids=str)
@pytest.mark.parametrize("out_dtype", [torch.bfloat16], ids=str)
@pytest.mark.parametrize("accumulate", [True, False], ids=["accumulate", "no_accumulate"])
def test_mxfp4_gemm_aiter_versus_reference(
    M: int,
    K: int,
    N: int,
    x_dtype: torch.dtype,
    w_dtype: torch.dtype,
    out_dtype: torch.dtype,
    accumulate: bool,
    monkeypatch,
):
    """AITER a4w4 MXFP4 GEMM vs the Python MXFP4 reference."""
    monkeypatch.setenv("NVTE_ROCM_USE_HIPBLASLT_MXFP4", "0")
    check_mxfp4_gemm_versus_reference(
        x_dtype=x_dtype,
        w_dtype=w_dtype,
        out_dtype=out_dtype,
        M=M,
        K=K,
        N=N,
        accumulate=accumulate,
        shuffle=True,
    )


# hipBLASLt F4F4 backend: plain FP4 data, with plain (VEC32_UE8M0) or pre-swizzled
@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize("M, K, N", _SHAPES)
@pytest.mark.parametrize("x_dtype", [torch.bfloat16], ids=str)
@pytest.mark.parametrize("w_dtype", [torch.bfloat16], ids=str)
@pytest.mark.parametrize("out_dtype", [torch.bfloat16], ids=str)
@pytest.mark.parametrize("accumulate", [True, False], ids=["accumulate", "no_accumulate"])
@pytest.mark.parametrize("use_swizzled", [False, True], ids=["plain_scales", "swizzled_scales"])
def test_mxfp4_gemm_hipblaslt_versus_reference(
    M: int,
    K: int,
    N: int,
    x_dtype: torch.dtype,
    w_dtype: torch.dtype,
    out_dtype: torch.dtype,
    accumulate: bool,
    use_swizzled: bool,
    monkeypatch,
):
    """hipBLASLt MXFP4 GEMM vs the Python MXFP4 reference, plain or pre-swizzled scales."""
    if K % 256 != 0:
        pytest.skip("hipBLASLt MXFP4 currently requires K to be a multiple of 256")
    if accumulate:
        pytest.skip("hipBLASLt MXFP4 does not support accumulate yet")
    monkeypatch.setenv("NVTE_ROCM_USE_HIPBLASLT_MXFP4", "1")
    check_mxfp4_gemm_versus_reference(
        x_dtype=x_dtype,
        w_dtype=w_dtype,
        out_dtype=out_dtype,
        M=M,
        K=K,
        N=N,
        accumulate=accumulate,
        shuffle=False,
        swizzled_scales=use_swizzled,
    )
