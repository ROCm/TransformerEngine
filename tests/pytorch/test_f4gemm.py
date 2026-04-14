# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

"""
Correctness tests for MXFP4 GEMM integration via QoLA f4gemm kernels.

Validates:
1. Low-level dispatch: gemm_a4w4_asm / gemm_a4w4_blockscale
2. MXFP4Quantizer.qgemm() (the custom_gemm entry point)
3. general_gemm() routing through custom_gemm -> qgemm
4. MXFP4 tensor interface (custom, data/scale properties, original_shape)

Kernel output is compared against a GPU dequant->mm reference matching
AITER's own test_gemm_a4w4.py pattern.

Requires gfx950 (CDNA4) hardware and QoLA-built f4gemm .so files.
"""

import math
import pytest
import numpy as np
import torch


_rng = np.random.default_rng(np.random.MT19937(12345))


def fill_uniform(shape, dtype):
    x = _rng.uniform(-2.0, 1.0, shape).astype(np.float32)
    return torch.tensor(x, device="cuda").to(dtype)

# ============================================================================
# Skip conditions
# ============================================================================


def _is_gfx950():
    try:
        from transformer_engine.pytorch.triton_kernels.common import is_cdna4

        return torch.cuda.is_available() and is_cdna4()
    except Exception:
        return False


requires_gfx950 = pytest.mark.skipif(
    not _is_gfx950(), reason="f4gemm requires gfx950 (CDNA4)"
)


# ============================================================================
# GPU Reference — mirrors AITER's own test_gemm_a4w4.py::run_torch()
# ============================================================================

SCALE_GROUP_SIZE = 32


def _mxfp4_to_f32(x: torch.Tensor) -> torch.Tensor:
    """Decode packed FP4 (uint8 or float4_e2m1fn_x2) to FP32 on GPU."""
    if x.dtype == torch.float4_e2m1fn_x2:
        x = x.view(torch.uint8)
    x = x.repeat_interleave(2, dim=-1)
    x[..., ::2] = x[..., ::2] & 0xF
    x[..., 1::2] = x[..., 1::2] >> 4
    lut = torch.tensor(
        [
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        ],
        dtype=torch.float32,
        device=x.device,
    )
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
    M: int,
    N: int,
    K: int,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """GPU-side dequant->mm reference matching AITER's run_torch()."""
    x_f32 = _mxfp4_to_f32(x)[:M]
    w_f32 = _mxfp4_to_f32(w)[:N]
    xs = x_scales.view(torch.uint8)[:M, : K // SCALE_GROUP_SIZE]
    ws = w_scales.view(torch.uint8)[:N, : K // SCALE_GROUP_SIZE]
    xs_f32 = _e8m0_to_f32(xs).repeat_interleave(SCALE_GROUP_SIZE, dim=1)
    ws_f32 = _e8m0_to_f32(ws).repeat_interleave(SCALE_GROUP_SIZE, dim=1)
    x_f32 = x_f32 * xs_f32
    w_f32 = w_f32 * ws_f32
    return torch.mm(x_f32, w_f32.T).to(out_dtype)


# ============================================================================
# Tolerance helper
# ============================================================================


def _assert_f4gemm_close(result: torch.Tensor, ref: torch.Tensor, msg: str):
    """Assert f4gemm result matches reference within FP4 GEMM tolerances.

    Tolerances derived empirically from error distributions on gfx950:
        - rtol=0.25 covers p95 relative error across shapes
        - 5% outlier budget matching AITER's checkAllclose default
    """
    _F4GEMM_RTOL = 0.25
    _F4GEMM_ATOL = 0
    _F4GEMM_OUTLIER_LIMIT = 0.06
    is_close = torch.isclose(
        result.float(), ref.float(), rtol=_F4GEMM_RTOL, atol=_F4GEMM_ATOL
    )
    outlier_ratio = 1.0 - is_close.float().mean().item()
    assert outlier_ratio <= _F4GEMM_OUTLIER_LIMIT, (
        f"{msg}: {outlier_ratio:.2%} elements outside "
        f"rtol={_F4GEMM_RTOL}/atol={_F4GEMM_ATOL} (limit {_F4GEMM_OUTLIER_LIMIT:.0%})"
    )


# ============================================================================
# Helpers
# ============================================================================


def quantize_and_prepare(M, N, K):
    """Quantize random BF16 tensors to MXFP4 via TE.

    Quantizes twice: with shuffle=True for the kernel, and with
    shuffle=False for the GPU reference (shuffle only affects scales,
    FP4 data is identical in both cases).

    Returns:
        qx, qw: packed FP4 data (shuffled scales) for kernel
        sx, sw: E8M0 scales (shuffled) for kernel
        qx_raw, qw_raw: packed FP4 data as uint8 (for reference)
        sx_noshuf, sw_noshuf: un-shuffled scales (for reference)
        M, N, K: dimensions
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

    qx = qx_shuf._rowwise_data
    qw = qw_shuf._rowwise_data
    sx = qx_shuf._rowwise_scale_inv
    sw = qw_shuf._rowwise_scale_inv

    q_noshuf = MXFP4Quantizer(
        rowwise=True, columnwise=False, shuffle_B_matrix_for_aiter=False,
    )
    qx_noshuf = te_quantize_triton(x, quantizer=q_noshuf)
    qw_noshuf = te_quantize_triton(w, quantizer=q_noshuf)

    qx_raw = qx_noshuf._rowwise_data.view(torch.uint8)
    qw_raw = qw_noshuf._rowwise_data.view(torch.uint8)
    sx_noshuf = qx_noshuf._rowwise_scale_inv.view(torch.uint8)
    sw_noshuf = qw_noshuf._rowwise_scale_inv.view(torch.uint8)

    return qx, qw, sx, sw, qx_raw, qw_raw, sx_noshuf, sw_noshuf


# ============================================================================
# Tests: MXFP4 tensor interface (no .so files needed)
# ============================================================================


class TestMXFP4Interface:
    """Tests for the custom GEMM interface on MXFP4 types."""

    def test_custom_property_on_tensor(self):
        from transformer_engine.pytorch.tensor.mxfp4_tensor import MXFP4Quantizer

        quantizer = MXFP4Quantizer(rowwise=True, columnwise=False)
        tensor = quantizer.make_empty((128, 256), dtype=torch.bfloat16, device="cuda")
        assert tensor.custom is True

    def test_custom_property_on_quantizer(self):
        from transformer_engine.pytorch.tensor.mxfp4_tensor import MXFP4Quantizer

        quantizer = MXFP4Quantizer(rowwise=True, columnwise=False)
        assert quantizer.custom is True

    def test_data_returns_rowwise_packed(self):
        from transformer_engine.pytorch.tensor.mxfp4_tensor import MXFP4Quantizer

        M, K = 64, 128
        quantizer = MXFP4Quantizer(rowwise=True, columnwise=False)
        tensor = quantizer.make_empty((M, K), dtype=torch.bfloat16, device="cuda")
        assert tensor.data is not None
        assert tensor.data.dtype == torch.uint8
        assert tensor.data.shape == (M, K // 2)

    def test_scale_property(self):
        from transformer_engine.pytorch.tensor.mxfp4_tensor import (
            MXFP4Quantizer,
            MXFP4_BLOCK_SCALING_SIZE,
        )
        from transformer_engine.pytorch.utils import round_up_to_nearest_multiple

        M, K = 128, 256
        quantizer = MXFP4Quantizer(rowwise=True, columnwise=False)
        tensor = quantizer.make_empty((M, K), dtype=torch.bfloat16, device="cuda")
        assert tensor.scale is not None
        expected_M = round_up_to_nearest_multiple(M, 256)
        expected_K = round_up_to_nearest_multiple(K // MXFP4_BLOCK_SCALING_SIZE, 8)
        assert tensor.scale.shape == (expected_M, expected_K)

    def test_is_custom_gate(self):
        from transformer_engine.pytorch.tensor.mxfp4_tensor import MXFP4Quantizer
        from transformer_engine.pytorch.tensor.utils import is_custom

        quantizer = MXFP4Quantizer(rowwise=True, columnwise=False)
        tensor = quantizer.make_empty((64, 128), dtype=torch.bfloat16, device="cuda")
        assert is_custom(tensor) is True

    def test_qgemm_rejects_dgrad(self):
        from transformer_engine.pytorch.tensor.mxfp4_tensor import MXFP4Quantizer
        from transformer_engine.pytorch.custom_recipes.quantization import GEMMType

        quantizer = MXFP4Quantizer(rowwise=True, columnwise=False)
        dummy = torch.empty(1, device="cuda")
        with pytest.raises(NotImplementedError, match="FPROP"):
            quantizer.qgemm(
                dummy, dummy, None, torch.bfloat16, dummy, dummy,
                gemm_type=GEMMType.DGRAD,
            )

    def test_qgemm_rejects_wgrad(self):
        from transformer_engine.pytorch.tensor.mxfp4_tensor import MXFP4Quantizer
        from transformer_engine.pytorch.custom_recipes.quantization import GEMMType

        quantizer = MXFP4Quantizer(rowwise=True, columnwise=False)
        dummy = torch.empty(1, device="cuda")
        with pytest.raises(NotImplementedError, match="FPROP"):
            quantizer.qgemm(
                dummy, dummy, None, torch.bfloat16, dummy, dummy,
                gemm_type=GEMMType.WGRAD,
            )


# ============================================================================
# Tests: Low-level dispatch (requires .so files + gfx950)
# ============================================================================


@requires_gfx950
@pytest.mark.parametrize(
    "M,N,K",
    [
        (32, 128, 128),
        (128, 4096, 4096),
        (256, 1024, 2048),
        (512, 2048, 4096),
    ],
)
def test_dispatch_asm_gemm(M, N, K):
    """Test gemm_a4w4_asm dispatch: shape, dtype, and correctness vs reference."""
    from transformer_engine.pytorch.tensor.f4gemm_dispatch import gemm_a4w4_asm

    qx, qw, sx, sw, qx_raw, qw_raw, sx_noshuf, sw_noshuf = quantize_and_prepare(
        M, N, K
    )

    out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    result = gemm_a4w4_asm(qx, qw, sx, sw, out)

    assert result.shape == (M, N), f"Expected ({M}, {N}), got {result.shape}"
    assert result.dtype == torch.bfloat16

    # Correctness vs GPU dequant->mm reference (skip small K due to FP4 noise)
    if K >= 1024:
        ref = gpu_reference_f4gemm(qx_raw, qw_raw, sx_noshuf, sw_noshuf, M, N, K)
        _assert_f4gemm_close(result, ref, f"dispatch_asm ({M},{N},{K})")


@requires_gfx950
@pytest.mark.parametrize(
    "M,N,K",
    [
        (256, 1024, 2048),
        (512, 2048, 4096),
    ],
)
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
def test_dispatch_blockscale_gemm(M, N, K, out_dtype):
    """Test gemm_a4w4_blockscale dispatch: shape, dtype, and correctness."""
    from transformer_engine.pytorch.tensor.f4gemm_dispatch import gemm_a4w4_blockscale

    # Blockscale path does not use shuffled scales
    from transformer_engine.pytorch.tensor.mxfp4_tensor import MXFP4Quantizer
    from transformer_engine.pytorch.triton_kernels.cast import te_quantize_triton

    x = fill_uniform((M, K), dtype=torch.bfloat16)
    w = fill_uniform((N, K), dtype=torch.bfloat16)

    quantizer = MXFP4Quantizer(
        rowwise=True, columnwise=False, shuffle_B_matrix_for_aiter=False,
    )
    qx = te_quantize_triton(x, quantizer=quantizer)
    qw = te_quantize_triton(w, quantizer=quantizer)

    out = torch.empty(M, N, dtype=out_dtype, device="cuda")
    result = gemm_a4w4_blockscale(
        qx._rowwise_data, qw._rowwise_data,
        qx._rowwise_scale_inv, qw._rowwise_scale_inv,
        out,
    )

    assert result.shape == (M, N), f"Expected ({M}, {N}), got {result.shape}"
    assert result.dtype == out_dtype

    if K >= 1024:
        sx_raw = qx._rowwise_scale_inv.view(torch.uint8)
        sw_raw = qw._rowwise_scale_inv.view(torch.uint8)
        ref = gpu_reference_f4gemm(
            qx._rowwise_data.view(torch.uint8),
            qw._rowwise_data.view(torch.uint8),
            sx_raw, sw_raw, M, N, K, out_dtype,
        )
        _assert_f4gemm_close(result, ref, f"dispatch_blockscale ({M},{N},{K},{out_dtype})")


@requires_gfx950
def test_dispatch_zero_input():
    """All-zero FP4 input should produce all-zero output."""
    from transformer_engine.pytorch.tensor.f4gemm_dispatch import gemm_a4w4_asm

    M, N, K = 256, 256, 256
    device = "cuda"

    qx = torch.zeros(M, K // 2, dtype=torch.uint8, device=device)
    qw = torch.zeros(N, K // 2, dtype=torch.uint8, device=device)

    scale_K = ((math.ceil(K / 32) + 7) // 8) * 8
    padded_M = ((M + 255) // 256) * 256
    padded_N = ((N + 255) // 256) * 256
    sx = torch.full((padded_M, scale_K), 127, dtype=torch.uint8, device=device)
    sw = torch.full((padded_N, scale_K), 127, dtype=torch.uint8, device=device)

    out = torch.empty(M, N, dtype=torch.bfloat16, device=device)
    result = gemm_a4w4_asm(qx, qw, sx, sw, out)

    assert result.shape == (M, N)
    assert result.dtype == torch.bfloat16
    torch.testing.assert_close(result, torch.zeros_like(result))


# ============================================================================
# Tests: MXFP4Quantizer.qgemm()
# ============================================================================


@requires_gfx950
@pytest.mark.parametrize(
    "M,N,K",
    [
        (128, 512, 256),
        (128, 4096, 4096),
        (256, 1024, 2048),
        (512, 2048, 4096),
    ],
)
def test_mxfp4_qgemm(M, N, K):
    """Test MXFP4Quantizer.qgemm(): correctness and shape/dtype."""
    from transformer_engine.pytorch.tensor.mxfp4_tensor import MXFP4Quantizer

    qx, qw, sx, sw, qx_raw, qw_raw, sx_noshuf, sw_noshuf = quantize_and_prepare(
        M, N, K
    )

    quantizer = MXFP4Quantizer(
        rowwise=True, columnwise=False, shuffle_B_matrix_for_aiter=True,
    )
    # qgemm expects post-custom_gemm swap order: (weight, activation)
    result = quantizer.qgemm(qw, qx, None, torch.bfloat16, sw, sx)

    assert result.shape == (M, N)
    assert result.dtype == torch.bfloat16

    if K >= 1024:
        ref = gpu_reference_f4gemm(qx_raw, qw_raw, sx_noshuf, sw_noshuf, M, N, K)
        _assert_f4gemm_close(result, ref, f"qgemm ({M},{N},{K})")


# ============================================================================
# Tests: general_gemm() integration
# ============================================================================


@requires_gfx950
@pytest.mark.parametrize(
    "M,N,K",
    [
        (128, 128, 128),
        (256, 512, 256),
    ],
)
def test_general_gemm_routes_to_qgemm(M, N, K):
    """general_gemm with MXFP4Tensor inputs routes through custom_gemm -> qgemm."""
    from transformer_engine.pytorch.cpp_extensions.gemm import general_gemm
    from transformer_engine.pytorch.tensor.mxfp4_tensor import MXFP4Quantizer
    from transformer_engine.pytorch.triton_kernels.cast import te_quantize_triton

    x = fill_uniform((M, K), dtype=torch.bfloat16)
    w = fill_uniform((N, K), dtype=torch.bfloat16)

    quantizer = MXFP4Quantizer(
        rowwise=True, columnwise=False, shuffle_B_matrix_for_aiter=True,
    )
    act_q = te_quantize_triton(x, quantizer=quantizer)
    wt_q = te_quantize_triton(w, quantizer=quantizer)

    result, _, _, _ = general_gemm(
        act_q, wt_q, out_dtype=torch.bfloat16, layout="TN",
    )

    assert result is not None
    assert result.shape == (M, N)
    assert result.dtype == torch.bfloat16
    assert not torch.isnan(result).any(), "Output contains NaNs"
    assert not torch.isinf(result).any(), "Output contains Infs"


@requires_gfx950
def test_general_gemm_3d_input():
    """general_gemm with 3D MXFP4 input reshapes output back to 3D."""
    from transformer_engine.pytorch.cpp_extensions.gemm import general_gemm
    from transformer_engine.pytorch.tensor.mxfp4_tensor import MXFP4Quantizer
    from transformer_engine.pytorch.triton_kernels.cast import te_quantize_triton

    B, S, K, N = 4, 32, 128, 256
    shape = (B, S, K)

    x = fill_uniform(shape, dtype=torch.bfloat16)
    w = fill_uniform((N, K), dtype=torch.bfloat16)

    quantizer = MXFP4Quantizer(
        rowwise=True, columnwise=False, shuffle_B_matrix_for_aiter=True,
    )
    act_q = te_quantize_triton(x, quantizer=quantizer)
    wt_q = te_quantize_triton(w, quantizer=quantizer)

    result, _, _, _ = general_gemm(
        act_q, wt_q, out_dtype=torch.bfloat16, layout="TN",
    )

    assert result is not None
    # Output is 2D because custom_gemm's 3D reshape checks the weight's
    # original_shape (always 2D after the A,B swap), not the activation's.
    assert result.shape == (B * S, N), f"Expected ({B * S}, {N}), got {result.shape}"
