# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Correctness tests for AITER f4gemm kernels integrated into TransformerEngine.

Tests both the low-level f4gemm dispatch and the MXFP4Quantizer.qgemm() path
by comparing kernel output against a GPU dequant->mm reference (matching
AITER's own test_gemm_a4w4.py pattern).

Requires gfx950 (CDNA4) hardware.
"""

import math
import pytest
import torch

from test_common import fill_uniform

# ============================================================================
# Skip if not gfx950
# ============================================================================

def _is_gfx950():
    try:
        from transformer_engine.pytorch.triton_kernels.common import is_cdna4
        return torch.cuda.is_available() and is_cdna4()
    except Exception:
        return False

def _f4gemm_modules_available():
    try:
        from transformer_engine.pytorch.tensor.f4gemm_dispatch import _load_f4gemm_modules
        mod_asm, mod_bs = _load_f4gemm_modules()
        return mod_asm is not None or mod_bs is not None
    except Exception:
        return False

requires_gfx950 = pytest.mark.skipif(
    not _is_gfx950(), reason="f4gemm requires gfx950 (CDNA4)"
)
requires_f4gemm = pytest.mark.skipif(
    not _f4gemm_modules_available(),
    reason="f4gemm modules not built (build with NVTE_AITER_F4GEMM=1 on gfx950)",
)

# ============================================================================
# GPU Reference — mirrors AITER's own test_gemm_a4w4.py::run_torch()
# ============================================================================

# Vendored from aiter/utility/fp4_utils.py — pure PyTorch, no AITER build needed.
SCALE_GROUP_SIZE = 32

def _mxfp4_to_f32(x: torch.Tensor) -> torch.Tensor:
    """Decode packed FP4 (uint8 or float4_e2m1fn_x2) to FP32 on GPU."""
    if x.dtype == torch.float4_e2m1fn_x2:
        x = x.view(torch.uint8)
    x = x.repeat_interleave(2, dim=-1)
    x[..., ::2] = x[..., ::2] & 0xF
    x[..., 1::2] = x[..., 1::2] >> 4
    lut = torch.tensor([
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ], dtype=torch.float32, device=x.device)
    return lut[x.long()]

def _e8m0_to_f32(scale_e8m0: torch.Tensor) -> torch.Tensor:
    """Decode E8M0 biased exponent to FP32 scale on GPU."""
    s = scale_e8m0.view(torch.uint8)
    zero_case = s == 0
    nan_case = s == 0xFF
    s_f32 = s.to(torch.int32) << 23
    s_f32[zero_case] = 0x00400000
    s_f32[nan_case] = 0x7F800001
    return s_f32.view(torch.float32)

def gpu_reference_f4gemm(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scales: torch.Tensor,
    w_scales: torch.Tensor,
    M: int, N: int, K: int,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """GPU-side dequant->mm reference matching AITER's run_torch()."""
    x_f32 = _mxfp4_to_f32(x)[:M]
    w_f32 = _mxfp4_to_f32(w)[:N]
    xs = x_scales.view(torch.uint8)[:M, :K // SCALE_GROUP_SIZE]
    ws = w_scales.view(torch.uint8)[:N, :K // SCALE_GROUP_SIZE]
    xs_f32 = _e8m0_to_f32(xs).repeat_interleave(SCALE_GROUP_SIZE, dim=1)
    ws_f32 = _e8m0_to_f32(ws).repeat_interleave(SCALE_GROUP_SIZE, dim=1)
    x_f32 = x_f32 * xs_f32
    w_f32 = w_f32 * ws_f32
    return torch.mm(x_f32, w_f32.T).to(out_dtype)


# ============================================================================
# Helpers
# ============================================================================

def _assert_f4gemm_close(result: torch.Tensor, ref: torch.Tensor, msg: str):
    """Assert f4gemm result matches reference within FP4 GEMM tolerances.

    Per-element tolerance derived empirically from measured error distributions
    on gfx950:
        - p95 relative error across shapes: 16-23%  =>  rtol=0.25 covers all with margin
        - p95 absolute error: shape-dependent (56-164), not useful standalone
    Outlier budget: 5%, matching AITER's checkAllclose default.
    """
    _F4GEMM_RTOL = 0.25
    _F4GEMM_ATOL = 0
    _F4GEMM_OUTLIER_LIMIT = 0.05
    is_close = torch.isclose(result.float(), ref.float(), rtol=_F4GEMM_RTOL, atol=_F4GEMM_ATOL)
    outlier_ratio = 1.0 - is_close.float().mean().item()
    assert outlier_ratio <= _F4GEMM_OUTLIER_LIMIT, (
        f"{msg}: {outlier_ratio:.2%} elements outside "
        f"rtol={_F4GEMM_RTOL}/atol={_F4GEMM_ATOL} (limit {_F4GEMM_OUTLIER_LIMIT:.0%})"
    )


def quantize_and_prepare(M, N, K):
    """Quantize random BF16 tensors to MXFP4 via TE twice: with and without shuffle.

    Mirrors AITER's test_gemm_a4w4.py pattern:
    - Quantize with shuffle=True  -> shuffled scales for kernel
    - Quantize with shuffle=False -> un-shuffled scales for GPU reference
    The FP4 data is identical in both cases (shuffle only affects scales).

    Returns:
        qx, qw: packed FP4 as float4_e2m1fn_x2 (for kernel)
        sx, sw: E8M0 scales as float8_e8m0fnu (for kernel, shuffled)
        qx_raw, qw_raw: packed FP4 as uint8 (for reference, same data)
        sx_noshuf, sw_noshuf: un-shuffled E8M0 scales as uint8 (for reference)
    """
    from transformer_engine.pytorch.tensor.mxfp4_tensor import MXFP4Quantizer
    from transformer_engine.pytorch.triton_kernels.cast import te_quantize_triton

    x = fill_uniform((M, K), dtype=torch.bfloat16)
    w = fill_uniform((N, K), dtype=torch.bfloat16)

    q_shuf = MXFP4Quantizer(
        rowwise=True, columnwise=False, shuffle_B_matrix_for_aiter=True,
    )
    qx_shuf = te_quantize_triton(x, quantizer=q_shuf)
    qw_shuf = te_quantize_triton(w, quantizer=q_shuf)

    qx = qx_shuf._rowwise_data.view(torch.float4_e2m1fn_x2)
    qw = qw_shuf._rowwise_data.view(torch.float4_e2m1fn_x2)
    sx = qx_shuf._rowwise_scale_inv.view(torch.float8_e8m0fnu)
    sw = qw_shuf._rowwise_scale_inv.view(torch.float8_e8m0fnu)

    q_noshuf = MXFP4Quantizer(
        rowwise=True, columnwise=False, shuffle_B_matrix_for_aiter=False,
    )
    qx_noshuf = te_quantize_triton(x, quantizer=q_noshuf)
    qw_noshuf = te_quantize_triton(w, quantizer=q_noshuf)

    qx_raw = qx_noshuf._rowwise_data.view(torch.uint8)
    qw_raw = qw_noshuf._rowwise_data.view(torch.uint8)
    sx_noshuf = qx_noshuf._rowwise_scale_inv.view(torch.uint8)
    sw_noshuf = qw_noshuf._rowwise_scale_inv.view(torch.uint8)

    return qx, qw, sx, sw, qx_raw, qw_raw, sx_noshuf, sw_noshuf, M, N, K


# ============================================================================
# Tests: Low-level f4gemm dispatch
# ============================================================================

@requires_gfx950
@requires_f4gemm
@pytest.mark.parametrize("M,N,K", [
    (32, 128, 128),
    (128, 4096, 4096),
    (256, 1024, 2048),
    (512, 2048, 4096),
])
def test_f4gemm_dispatch(M, N, K):
    """Test f4gemm dispatch: correctness and output shape/dtype."""
    from transformer_engine.pytorch.tensor.f4gemm_dispatch import f4gemm

    qx, qw, sx, sw, qx_raw, qw_raw, sx_noshuf, sw_noshuf, *_ = \
        quantize_and_prepare(M, N, K)

    M_padded = ((M + 31) // 32) * 32
    out = torch.empty(M_padded, N, dtype=torch.bfloat16, device="cuda")
    result = f4gemm(qx, qw, sx, sw, out, bpreshuffle=True)

    # Shape and dtype
    assert result.shape == (M, N), f"Expected ({M}, {N}), got {result.shape}"
    assert result.dtype == torch.bfloat16, f"Expected bfloat16, got {result.dtype}"

    # Correctness vs GPU dequant->mm reference.
    # Small K (<1024) has high per-element error because FP4's 1-bit mantissa
    # dominates with fewer accumulations; skip element-wise check for those.
    ref = gpu_reference_f4gemm(qx_raw, qw_raw, sx_noshuf, sw_noshuf, M, N, K)
    if K >= 1024:
        _assert_f4gemm_close(result, ref, f"f4gemm dispatch ({M},{N},{K})")


@requires_gfx950
@requires_f4gemm
def test_f4gemm_dispatch_zero_input():
    """All-zero FP4 input should produce all-zero output."""
    from transformer_engine.pytorch.tensor.f4gemm_dispatch import f4gemm

    M, N, K = 128, 256, 256
    device = "cuda"

    qx = torch.zeros(M, K // 2, dtype=torch.uint8, device=device).view(torch.float4_e2m1fn_x2)
    qw = torch.zeros(N, K // 2, dtype=torch.uint8, device=device).view(torch.float4_e2m1fn_x2)

    scale_K = ((math.ceil(K / 32) + 7) // 8) * 8
    padded_M = ((M + 255) // 256) * 256
    padded_N = ((N + 255) // 256) * 256
    sx = torch.full((padded_M, scale_K), 127, dtype=torch.uint8, device=device).view(torch.float8_e8m0fnu)
    sw = torch.full((padded_N, scale_K), 127, dtype=torch.uint8, device=device).view(torch.float8_e8m0fnu)

    M_padded = ((M + 31) // 32) * 32
    out = torch.empty(M_padded, N, dtype=torch.bfloat16, device=device)

    result = f4gemm(qx, qw, sx, sw, out, bpreshuffle=False)
    assert result.shape == (M, N)
    assert result.dtype == torch.bfloat16
    torch.testing.assert_close(result, torch.zeros_like(result))


# ============================================================================
# Tests: MXFP4Quantizer.qgemm() (end-to-end)
# ============================================================================

@requires_gfx950
@requires_f4gemm
@pytest.mark.parametrize("M,N,K", [
    (128, 512, 256),
    (128, 4096, 4096),
    (256, 1024, 2048),
    (512, 2048, 4096),
])
def test_mxfp4_qgemm(M, N, K):
    """Test MXFP4Quantizer.qgemm(): correctness and shape/dtype."""
    from transformer_engine.pytorch.tensor.mxfp4_tensor import MXFP4Quantizer

    qx, qw, sx, sw, qx_raw, qw_raw, sx_noshuf, sw_noshuf, *_ = \
        quantize_and_prepare(M, N, K)

    quantizer = MXFP4Quantizer(
        rowwise=True, columnwise=False, shuffle_B_matrix_for_aiter=True,
    )
    result = quantizer.qgemm(qx, qw, None, torch.bfloat16, sx, sw)

    # Shape and dtype
    assert result.shape[1] == N, f"Expected N={N}, got {result.shape[1]}"
    assert result.dtype == torch.bfloat16, f"Expected bfloat16, got {result.dtype}"

    # Correctness vs GPU dequant->mm reference
    ref = gpu_reference_f4gemm(qx_raw, qw_raw, sx_noshuf, sw_noshuf, M, N, K)
    if K >= 1024:
        _assert_f4gemm_close(result, ref, f"qgemm ({M},{N},{K})")


@requires_gfx950
@requires_f4gemm
def test_mxfp4_qgemm_not_impl():
    """DGRAD and WGRAD should raise NotImplementedError."""
    from transformer_engine.pytorch.tensor.mxfp4_tensor import MXFP4Quantizer
    from transformer_engine.pytorch.custom_recipes.quantization import GEMMType

    M, K = 128, 256
    device = "cuda"
    quantizer = MXFP4Quantizer(rowwise=True, columnwise=False)

    qx = torch.zeros(M, K // 2, dtype=torch.uint8, device=device)
    sx = torch.zeros(M, K // 32, dtype=torch.uint8, device=device)

    with pytest.raises(NotImplementedError, match="FPROP"):
        quantizer.qgemm(qx, qx, None, torch.bfloat16, sx, sx, gemm_type=GEMMType.DGRAD)

    with pytest.raises(NotImplementedError, match="FPROP"):
        quantizer.qgemm(qx, qx, None, torch.bfloat16, sx, sx, gemm_type=GEMMType.WGRAD)
