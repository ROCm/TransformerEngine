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


def _mxfp4_quant(src, *, shuffle_rowwise=False, shuffle_columnwise=False, swizzled_scales=False):
    """Quantize a BF16 tensor to MXFP4 with both row-wise and column-wise buffers."""
    q = MXFP4Quantizer(
        fp4_dtype=tex.DType.kFloat4E2M1,
        rowwise=True,
        columnwise=True,
        shuffle_rowwise_data=shuffle_rowwise,
        shuffle_columnwise_data=shuffle_columnwise,
        with_gemm_swizzled_scales=swizzled_scales,
        use_hadamard=False,
    )
    t = q.make_empty(tuple(src.shape), dtype=src.dtype, device=src.device, requires_grad=False)
    return q.update_quantized(src, t)


def _consumed_klast(tq, use_rowwise: bool, rows: int, k: int):
    """Extract the contraction-last packed buffer + plain block scales that the GEMM consumes for
    an operand: the row-wise buffer for a transposed operand, the column-wise buffer for a
    non-transposed one. Both are physically ``(rows, K/2)`` with scales ``(rows, K/32)``."""
    if use_rowwise:
        data = tq._rowwise_data.view(dtype=torch.uint8)[:rows, : k // 2]
        scale = tq._rowwise_scale_inv[:rows, : k // 32]
    else:
        data = tq._columnwise_data.view(dtype=torch.uint8)[:rows, : k // 2]
        scale = tq._columnwise_scale_inv[:rows, : k // 32]
    return data, scale


def check_mxfp4_gemm_versus_reference(
    m, n, k, out_dtype, *, a_quant, b_quant, layout="TN", accumulate=False
):
    """MXFP4 GEMM (AITER a4w4 or hipBLASLt F4F4) vs the Python MXFP4 reference.
    """
    from transformer_engine.pytorch.cpp_extensions.gemm import general_gemm

    device = "cuda"
    transa = layout[0] == "T"
    transb = layout[1] == "T"
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    a_shape = (m, k) if transa else (k, m)
    b_shape = (k, n) if transb else (n, k)
    A_src = torch.randn(a_shape, dtype=torch.bfloat16, device=device)
    B_src = torch.randn(b_shape, dtype=torch.bfloat16, device=device)
    out = torch.randn((n, m), dtype=out_dtype, device=device) if accumulate else None

    # --- Native operands (quantized per the caller-supplied config) ---
    A_q = _mxfp4_quant(A_src, **a_quant)
    B_q = _mxfp4_quant(B_src, **b_quant)

    y_native, *_ = general_gemm(
        A_q,
        B_q,
        out_dtype=out_dtype,
        quantization_params=None,
        bias=None,
        use_split_accumulator=False,
        layout=layout,
        out=out.clone() if accumulate else None,
        accumulate=accumulate,
    )

    # --- Reference: plain re-quant of the same source, dequant the consumed buffer ---
    A_ref = _mxfp4_quant(A_src, swizzled_scales=False)
    B_ref = _mxfp4_quant(B_src, swizzled_scales=False)
    qA, sA = _consumed_klast(A_ref, transa, m, k)          # Aeff (m, k)
    qB, sB = _consumed_klast(B_ref, not transb, n, k)      # Beff (n, k)
    ref = MXFP4QuantizerRef(rowwise=True, columnwise=True)
    y_ref = ref.qgemm(
        qx=qB, qw=qA, m_params=None, out_dtype=out_dtype, sx=sB, sw=sA,
        bias=None, out=out.clone() if accumulate else None, accumulate=accumulate, layout=layout,
    )  # Beff @ Aeff^T -> (n, m), matching general_gemm's output orientation

    assert y_ref is not y_native
    assert not torch.isnan(y_ref.float()).all(), "All reference elements are NaN"
    y_native = torch.where(y_native.isnan(), torch.zeros_like(y_native), y_native)
    y_ref = torch.where(y_ref.isnan(), torch.zeros_like(y_ref), y_ref)
    torch.testing.assert_close(y_native, y_ref, atol=8e-3, rtol=8e-3)


_SHAPES = [
    (128, 128, 128),
    (256, 256, 256),
    (256, 1024, 256),
    (1024, 1024, 1024),
    (4096, 512, 3072),
]


# AITER a4w4 backend: shuffled FP4 data + swizzled scales, TN only. Requires the aiter package.
@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.skipif(not _aiter_available, reason="aiter package not available")
@pytest.mark.parametrize("M, K, N", _SHAPES)
@pytest.mark.parametrize("out_dtype", [torch.bfloat16], ids=str)
@pytest.mark.parametrize("accumulate", [True, False], ids=["accumulate", "no_accumulate"])
def test_mxfp4_gemm_aiter_versus_reference(
    M: int,
    K: int,
    N: int,
    out_dtype: torch.dtype,
    accumulate: bool,
    monkeypatch,
):
    """AITER a4w4 MXFP4 GEMM vs the Python MXFP4 reference (TN; A = weight (N,K), B = act (M,K))."""
    monkeypatch.setenv("NVTE_ROCM_USE_HIPBLASLT_MXFP4", "0")
    # a4w4 wants the weight row+column-shuffled and the activation column-shuffled, scales swizzled.
    check_mxfp4_gemm_versus_reference(
        m=N, n=M, k=K, out_dtype=out_dtype, layout="TN", accumulate=accumulate,
        a_quant=dict(shuffle_rowwise=True, shuffle_columnwise=True, swizzled_scales=True),
        b_quant=dict(shuffle_rowwise=False, shuffle_columnwise=True, swizzled_scales=True),
    )


# hipBLASLt F4F4 backend: plain FP4 data, plain or pre-swizzled UE8M0 scales
@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize("M, K, N", _SHAPES)
@pytest.mark.parametrize("out_dtype", [torch.bfloat16], ids=str)
@pytest.mark.parametrize("layout", ["TN", "NN", "NT", "TT"])
@pytest.mark.parametrize("use_swizzled", [False, True], ids=["plain_scales", "swizzled_scales"])
def test_mxfp4_gemm_hipblaslt_versus_reference(
    M: int,
    K: int,
    N: int,
    out_dtype: torch.dtype,
    layout: str,
    use_swizzled: bool,
    monkeypatch,
):
    """hipBLASLt GEMM MXFP4 vs the Python MXFP4 reference."""
    if K % 256 != 0:
        pytest.skip("hipBLASLt MXFP4 currently requires K to be a multiple of 256")
    monkeypatch.setenv("NVTE_ROCM_USE_HIPBLASLT_MXFP4", "1")
    # hipBLASLt consumes plain FP4 data; scales plain or pre-swizzled per use_swizzled.
    quant = dict(swizzled_scales=use_swizzled)
    check_mxfp4_gemm_versus_reference(
        m=M, n=N, k=K, out_dtype=out_dtype, layout=layout, a_quant=quant, b_quant=quant,
    )
