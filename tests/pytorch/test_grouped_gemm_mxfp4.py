# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

"""Numeric correctness tests for the grouped MXFP4 Triton GEMM (gfx950).

These validate the two layout assumptions of the port: that TE's plain
E8M0/E2M1 quantizer output is interpreted the same way by
``tl.dot_scaled(..., "e2m1", ...)``, and that zero-padded rows contribute
nothing to the variable-K (wgrad) contraction.

Each op is checked against a *precise* reference: the same packed MXFP4 operands
the kernel consumes are dequantized here by an independent OCP E2M1/E8M0 decoder
(``_dequant_mxfp4``) and matmul'd in fp32. Kernel and reference start from
identical fp4 values, so a correct kernel matches to ~bf16-rounding while a
layout / nibble-order / scale-bias / transpose bug makes them disagree grossly.
wgrad additionally checks against the true bf16 grouped matmul, which exercises
the per-group zero-padding (padded rows must contribute nothing).
"""

import pytest
import torch

triton = pytest.importorskip("triton")

try:
    from transformer_engine.pytorch.quantization import check_mxfp4_support

    _MXFP4_OK, _MXFP4_REASON = check_mxfp4_support()
except Exception as exc:  # pragma: no cover - import/support probe
    _MXFP4_OK, _MXFP4_REASON = False, str(exc)

from transformer_engine.pytorch.triton_kernels.grouped_gemm_mxfp4_impl import (
    MXFP4_BLOCK,
    _col_operand,
    _col_operand_grouped_padded,
    _row_operand,
    grouped_gemm_mxfp4_dgrad,
    grouped_gemm_mxfp4_fprop,
    grouped_gemm_mxfp4_wgrad,
    grouped_linear_mxfp4,
)

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/ROCm device required"),
    pytest.mark.skipif(not _MXFP4_OK, reason=f"MXFP4 unsupported: {_MXFP4_REASON}"),
]

DTYPE = torch.bfloat16
# Uneven, non-128-multiple group sizes exercise fprop/dgrad masking and the
# wgrad per-group zero-padding to 128.
M_SPLITS = [96, 128, 160, 128]
N, K = 256, 128
# Precise-reference bar: kernel vs dequant-of-the-same-operands differ only by
# ~bf16 output rounding.
_TIGHT_TOL = 3.0e-2
# Loose bar vs the true (unquantized) bf16 matmul: ~0.16 MXFP4 noise floor for
# these shapes, well below the ~1.4 an uncorrelated (layout-bug) output gives.
_REL_TOL = 2.0e-1

# OCP E2M1 magnitude indexed by the 3 low bits (exp2 | mantissa1); bit 3 = sign.
_E2M1_MAG = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


@pytest.fixture(autouse=True)
def _mxfp4_env(monkeypatch):
    # Triton MXFP4 quantizer (no aiter dependency); seed for a deterministic
    # quantization error.
    monkeypatch.setenv("NVTE_USE_CAST_TRANSPOSE_TRITON", "1")
    torch.manual_seed(0)


def _rand(*shape):
    return torch.randn(*shape, dtype=DTYPE, device="cuda")


def _rel_err(out, ref):
    out = out.float()
    ref = ref.float()
    return (out - ref).norm() / ref.norm().clamp_min(1e-12)


def _dequant_mxfp4(data_u8, scale_u8, feat):
    """Independent OCP MXFP4 dequant of packed operands -> fp32 ``[R, feat]``.

    ``data_u8`` ``[R, feat/2]`` packs two E2M1 codes per byte (low nibble = even
    index along ``feat``); ``scale_u8`` ``[R, feat/32]`` is one E8M0 scale
    (value ``2**(x-127)``) per 1x32 block.
    """
    lut = torch.tensor(_E2M1_MAG, dtype=torch.float32, device=data_u8.device)
    lo = (data_u8 & 0xF).to(torch.long)
    hi = ((data_u8 >> 4) & 0xF).to(torch.long)
    codes = torch.stack((lo, hi), dim=-1).reshape(data_u8.shape[0], feat)
    mag = lut[codes & 0x7]
    vals = torch.where((codes & 0x8).bool(), -mag, mag)
    scale = torch.exp2(scale_u8.to(torch.float32) - 127.0)
    return vals * scale.repeat_interleave(MXFP4_BLOCK, dim=1)


def test_fprop_precise():
    total_m = sum(M_SPLITS)
    a = _rand(total_m, K)
    weights = [_rand(N, K) for _ in M_SPLITS]

    out = grouped_gemm_mxfp4_fprop(a, weights, M_SPLITS, out_dtype=DTYPE)

    # Dequantize the same row-wise operands the kernel used, then grouped matmul.
    a_deq = _dequant_mxfp4(*_row_operand(a), K)
    ref = torch.empty((total_m, N), dtype=torch.float32, device="cuda")
    start = 0
    for w, m in zip(weights, M_SPLITS):
        w_deq = _dequant_mxfp4(*_row_operand(w), K)  # [N, K]
        ref[start : start + m] = a_deq[start : start + m] @ w_deq.t()
        start += m

    assert out.shape == (total_m, N)
    assert _rel_err(out, ref) < _TIGHT_TOL


def test_dgrad_precise():
    total_m = sum(M_SPLITS)
    grad_out = _rand(total_m, N)
    weights = [_rand(N, K) for _ in M_SPLITS]

    dgrad = grouped_gemm_mxfp4_dgrad(grad_out, weights, M_SPLITS, out_dtype=DTYPE)

    # gradO row-wise, weight col-wise (the transposed operand): dA = gradO @ W.
    go_deq = _dequant_mxfp4(*_row_operand(grad_out), N)  # [total_M, N]
    ref = torch.empty((total_m, K), dtype=torch.float32, device="cuda")
    start = 0
    for w, m in zip(weights, M_SPLITS):
        w_col_deq = _dequant_mxfp4(*_col_operand(w), N)  # [K, N] ~ W^T
        ref[start : start + m] = go_deq[start : start + m] @ w_col_deq.t()
        start += m

    assert dgrad.shape == (total_m, K)
    assert _rel_err(dgrad, ref) < _TIGHT_TOL


def test_wgrad_precise_and_padding():
    total_m = sum(M_SPLITS)
    a = _rand(total_m, K)
    grad_out = _rand(total_m, N)

    wgrad = grouped_gemm_mxfp4_wgrad(a, grad_out, M_SPLITS, out_dtype=DTYPE)
    assert wgrad.shape == (len(M_SPLITS), N, K)

    # Precise: dequant the exact per-group padded col operands the kernel reduced
    # over, sliced by the padded offsets. C[g] = lhs[:, g] @ rhs[:, g]^T.
    lhs_data, lhs_scale, go_pad = _col_operand_grouped_padded(grad_out, M_SPLITS)  # [N, Mpad/2]
    rhs_data, rhs_scale, _ = _col_operand_grouped_padded(a, M_SPLITS)  # [K, Mpad/2]
    m_pad_total = lhs_data.shape[1] * 2
    lhs_deq = _dequant_mxfp4(lhs_data, lhs_scale, m_pad_total)  # [N, Mpad]
    rhs_deq = _dequant_mxfp4(rhs_data, rhs_scale, m_pad_total)  # [K, Mpad]
    go = go_pad.tolist()
    ref = torch.empty((len(M_SPLITS), N, K), dtype=torch.float32, device="cuda")
    for g in range(len(M_SPLITS)):
        s, e = go[g], go[g + 1]
        ref[g] = lhs_deq[:, s:e] @ rhs_deq[:, s:e].t()
    assert _rel_err(wgrad, ref) < _TIGHT_TOL

    # End-to-end vs the true (unquantized) matmul: exercises the per-group
    # zero-padding -- the padded rows must contribute nothing.
    ref_true = torch.empty((len(M_SPLITS), N, K), dtype=torch.float32, device="cuda")
    start = 0
    for g, m in enumerate(M_SPLITS):
        ref_true[g] = grad_out[start : start + m].float().t() @ a[start : start + m].float()
        start += m
    assert _rel_err(wgrad, ref_true) < _REL_TOL


def test_autograd_matches_ops():
    total_m = sum(M_SPLITS)
    a = _rand(total_m, K).requires_grad_(True)
    weight = torch.stack([_rand(N, K) for _ in M_SPLITS], dim=0).requires_grad_(True)

    out = grouped_linear_mxfp4(a, weight, M_SPLITS)
    assert out.shape == (total_m, N)

    grad_out = _rand(total_m, N)
    out.backward(grad_out)

    # The autograd grads must equal the direct op calls (same code path).
    ref_da = grouped_gemm_mxfp4_dgrad(
        grad_out, list(weight.detach().unbind(0)), M_SPLITS, out_dtype=a.dtype
    )
    ref_dw = grouped_gemm_mxfp4_wgrad(a.detach(), grad_out, M_SPLITS, out_dtype=weight.dtype)
    torch.testing.assert_close(a.grad, ref_da)
    torch.testing.assert_close(weight.grad, ref_dw)
