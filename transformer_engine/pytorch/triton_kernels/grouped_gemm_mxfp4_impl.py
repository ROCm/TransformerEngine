# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

"""TE-side quantization bridge + dispatch for grouped MXFP4 GEMM (gfx950).

This is the glue between TransformerEngine's MXFP4 quantizer and the ported
Triton block-scaled grouped kernels in ``grouped_gemm_mxfp4.py``. It reuses
TE's :class:`MXFP4Quantizer` (rather than porting Primus-Turbo's fused grouped
dual-quant) to produce the packed FP4 operands the kernels consume, then builds
the per-group offsets and calls the kernels.

Three grouped ops, paired exactly like the dense MXFP4 recipe (all NT):

    fprop : C   = A_row      @ W_row^T        (contract K,  fwd kernel)
    dgrad : dA  = gradO_row  @ W_col^T        (contract N,  fwd kernel)
    wgrad : dW  = gradO_col  @ A_col^T        (contract M,  variable-K kernel)

Operand layout the kernels expect (plain, un-swizzled OCP MXFP4):
  * data  : E2M1 packed two-per-byte along the contraction axis (``uint8``);
  * scale : one E8M0 (``uint8``) per 1x32 logical-element block, contiguous
            along the block axis.
TE's ``MXFP4Quantizer`` emits exactly this in its row-wise ``_rowwise_data`` /
``_rowwise_scale_inv`` and col-wise ``_columnwise_data`` / ``_columnwise_scale_inv``
members when the shuffle / GEMM-swizzle flags are all off. The scale buffers are
over-allocated (rows to 256, block axis to 8); we slice the live ``[:, :F/32]``
region and make it contiguous so the kernel sees tight strides.

RHT (Hadamard) on the wgrad operands and the ``main_grad`` beta=1 accumulate
fusion are deferred to the autograd layer; this module runs plain
MX (``use_hadamard=False``) and returns a fresh weight-gradient tensor.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from ..tensor.mxfp4_tensor import MXFP4Quantizer
from ..utils import round_up_to_nearest_multiple
from .grouped_gemm_mxfp4 import (
    _is_gfx950,
    grouped_gemm_mxfp4_triton_kernel,
    grouped_gemm_mxfp4_variable_k_triton_kernel,
)

# Logical contraction tile the kernels step by (see grouped_gemm_mxfp4.py).
BLOCK_SIZE_K = 128
# Per-group wgrad M padding multiple; must be >= the largest wgrad BLOCK_K the
# kernel may pick. 256 lets the variable-K autotune try BLOCK_K=256 (at the cost
# of more padding for small groups). Default 128 = unchanged behavior.
WGRAD_PAD_MULTIPLE = int(os.environ.get("NVTE_MXFP4_WGRAD_PAD", str(BLOCK_SIZE_K)))
# One E8M0 scale per 32-element block.
MXFP4_BLOCK = 32


def _quantizer(*, rowwise: bool, columnwise: bool) -> MXFP4Quantizer:
    """MXFP4 quantizer configured for the plain (un-swizzled) Triton layout."""
    return MXFP4Quantizer(
        rowwise=rowwise,
        columnwise=columnwise,
        shuffle_rowwise_data=False,
        shuffle_columnwise_data=False,
        with_gemm_swizzled_scales=False,
        use_hadamard=False,
    )


def _prefix_offsets(m_splits: Sequence[int], device: torch.device) -> torch.Tensor:
    """Tight prefix-sum offsets ``[0, m0, m0+m1, ...]`` as int64 on device."""
    offs = [0]
    for m in m_splits:
        offs.append(offs[-1] + int(m))
    return torch.tensor(offs, dtype=torch.int64, device=device)


def _check_contract(dim: int, name: str) -> None:
    if dim % BLOCK_SIZE_K != 0:
        raise ValueError(
            f"grouped MXFP4 GEMM requires the contraction dim ({name}={dim}) to be a"
            f" multiple of {BLOCK_SIZE_K}."
        )


def _check_splits(
    m_splits: Sequence[int], total_rows: int, num_experts: Optional[int] = None
) -> None:
    """Validate ``m_splits`` before it reaches the raw kernel (memory safety).

    Bad splits would otherwise make the Triton kernel read past ``group_offs`` or
    store past the output. Checks: non-negative splits, ``sum == total_rows``, and
    (when given) ``len == num_experts``.
    """
    if any(int(m) < 0 for m in m_splits):
        raise ValueError(f"m_splits must be non-negative, got {list(m_splits)}")
    got = sum(int(m) for m in m_splits)
    if got != total_rows:
        raise ValueError(f"sum(m_splits)={got} must equal the grouped row count {total_rows}")
    if num_experts is not None and len(m_splits) != num_experts:
        raise ValueError(
            f"len(m_splits)={len(m_splits)} must equal the number of experts {num_experts}"
        )


def _require_gfx950() -> None:
    # The kernels use the CDNA4 scaled-FP4 MFMA (tl.dot_scaled e2m1); no other arch has it.
    if not _is_gfx950():
        raise RuntimeError(
            "grouped MXFP4 GEMM requires gfx950 (CDNA4); the current device lacks the"
            " scaled-FP4 MFMA used by tl.dot_scaled(..., \"e2m1\", ...)."
        )


def _row_operand(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Row-wise MXFP4: ``x`` [M, F] -> (data [M, F/2] u8, scale [M, F/32] u8).

    ``M`` is a free (masked) GEMM dim of any size; it is zero-padded to a
    32-multiple for the quantizer (which requires it) and sliced back.
    """
    M, Feat = x.shape
    M_pad = round_up_to_nearest_multiple(M, MXFP4_BLOCK)
    if M_pad != M:
        x = F.pad(x.contiguous(), (0, 0, 0, M_pad - M))
    q = _quantizer(rowwise=True, columnwise=False).quantize(x.contiguous())
    data = q._rowwise_data.view(torch.uint8)[:M]
    scale = q._rowwise_scale_inv[:M, : Feat // MXFP4_BLOCK].contiguous()
    return data, scale


def _col_operand(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Col-wise MXFP4: ``x`` [M, F] -> (data [F, M/2] u8, scale [F, M/32] u8).

    Requests both directions (matching the production transpose-quant path); the
    row-wise result is discarded. Both dims are zero-padded to a 32-multiple for
    the quantizer and sliced back (both are free/masked in the GEMM).
    """
    M, Feat = x.shape
    M_pad = round_up_to_nearest_multiple(M, MXFP4_BLOCK)
    Feat_pad = round_up_to_nearest_multiple(Feat, MXFP4_BLOCK)
    if M_pad != M or Feat_pad != Feat:
        x = F.pad(x.contiguous(), (0, Feat_pad - Feat, 0, M_pad - M))
    q = _quantizer(rowwise=True, columnwise=True).quantize(x.contiguous())
    data = q._columnwise_data.view(torch.uint8)[:Feat, : M // 2]
    scale = q._columnwise_scale_inv[:Feat, : M // MXFP4_BLOCK].contiguous()
    return data, scale


def _row_col_operand(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One cast_transpose: MXFP4 row-wise *and* col-wise from a single quantize.

    ``x`` [M, F] -> (row_data [M, F/2], row_scale [M, F/32], col_data [F, M/2],
    col_scale [F, M/32]), all ``uint8``. Lets a caller that needs both layouts of
    the same tensor (e.g. a weight consumed row-wise by fprop and col-wise by
    dgrad) pay the transpose-quant once instead of twice. Both dims are padded to
    a 32-multiple for the quantizer and sliced back.
    """
    M, Feat = x.shape
    M_pad = round_up_to_nearest_multiple(M, MXFP4_BLOCK)
    Feat_pad = round_up_to_nearest_multiple(Feat, MXFP4_BLOCK)
    if M_pad != M or Feat_pad != Feat:
        x = F.pad(x.contiguous(), (0, Feat_pad - Feat, 0, M_pad - M))
    q = _quantizer(rowwise=True, columnwise=True).quantize(x.contiguous())
    row_data = q._rowwise_data.view(torch.uint8)[:M, : Feat // 2]
    row_scale = q._rowwise_scale_inv[:M, : Feat // MXFP4_BLOCK].contiguous()
    col_data = q._columnwise_data.view(torch.uint8)[:Feat, : M // 2]
    col_scale = q._columnwise_scale_inv[:Feat, : M // MXFP4_BLOCK].contiguous()
    return row_data, row_scale, col_data, col_scale


def _quantize_weights_row_col(
    weights: Sequence[torch.Tensor],
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """Per-expert weight cast_transpose once -> stacked row and col MXFP4 operands.

    Returns ``((row_data [G,N,K/2], row_scale [G,N,K/32]),
    (col_data [G,K,N/2], col_scale [G,K,N/32]))`` -- row for fprop, col for dgrad,
    both from a single quantize per expert.
    """
    rows_d, rows_s, cols_d, cols_s = [], [], [], []
    for w in weights:
        rd, rs, cd, cs = _row_col_operand(w)
        rows_d.append(rd)
        rows_s.append(rs)
        cols_d.append(cd)
        cols_s.append(cs)
    row = (torch.stack(rows_d, dim=0), torch.stack(rows_s, dim=0))
    col = (torch.stack(cols_d, dim=0), torch.stack(cols_s, dim=0))
    return row, col


def _col_operand_grouped_padded(
    x: torch.Tensor,
    m_splits: Sequence[int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-group col-wise MXFP4 with each group's M padded to WGRAD_PAD_MULTIPLE.

    ``x`` [total_M, F] grouped along M -> (data [F, total_pad/2] u8,
    scale [F, total_pad/32] u8, go_pad [G+1] int64). Padding rows are zeros, so
    they contribute nothing to the wgrad contraction over M.
    """
    Feat = x.shape[1]
    datas: List[torch.Tensor] = []
    scales: List[torch.Tensor] = []
    offs = [0]
    start = 0
    for m in m_splits:
        m = int(m)
        xg = x[start : start + m]
        start += m
        m_pad = round_up_to_nearest_multiple(m, WGRAD_PAD_MULTIPLE)
        if m_pad != m:
            xg = F.pad(xg, (0, 0, 0, m_pad - m))
        data, scale = _col_operand(xg)  # (F, m_pad/2), (F, m_pad/32)
        datas.append(data)
        scales.append(scale)
        offs.append(offs[-1] + m_pad)
    data = torch.cat(datas, dim=1).contiguous()
    scale = torch.cat(scales, dim=1).contiguous()
    go_pad = torch.tensor(offs, dtype=torch.int64, device=x.device)
    return data, scale, go_pad


def grouped_gemm_mxfp4_fprop(
    a: torch.Tensor,
    weights: List[torch.Tensor],
    m_splits: Sequence[int],
    *,
    out_dtype: torch.dtype = torch.bfloat16,
    num_cu: Optional[int] = None,
    weight_row: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    """Grouped MXFP4 forward: ``C[g] = A[g] @ W[g]^T`` (contract K).

    Args:
        a: [total_M, K] activations, grouped along M by ``m_splits``.
        weights: list of G per-expert weight tensors, each [N, K].
        m_splits: per-group token counts (len G).
        weight_row: optional pre-quantized row-wise weights
            ``(data [G, N, K/2], scale [G, N, K/32])`` -- skips the weight cast
            (e.g. shared with dgrad's col-wise cast, see ``_quantize_weights_row_col``).

    Returns:
        [total_M, N] output in ``out_dtype``.
    """
    K = a.shape[1]
    _require_gfx950()
    _check_contract(K, "K")

    if weight_row is not None:
        b_data, b_scale = weight_row
        N = b_data.shape[1]
        _check_splits(m_splits, a.shape[0], b_data.shape[0])
    else:
        N = weights[0].shape[0]
        _check_splits(m_splits, a.shape[0], len(weights))
        b_datas, b_scales = [], []
        for w in weights:
            d, s = _row_operand(w)  # (N, K/2), (N, K/32)
            b_datas.append(d)
            b_scales.append(s)
        b_data = torch.stack(b_datas, dim=0)  # (G, N, K/2)
        b_scale = torch.stack(b_scales, dim=0)  # (G, N, K/32)

    a_data, a_scale = _row_operand(a)  # (total_M, K/2), (total_M, K/32)
    group_offs = _prefix_offsets(m_splits, a.device)
    return grouped_gemm_mxfp4_triton_kernel(
        a_data,
        a_scale,
        b_data,
        b_scale,
        group_offs,
        N,
        K,
        group_offs_out=group_offs,
        out_dtype=out_dtype,
        num_cu=num_cu,
    )


def grouped_gemm_mxfp4_dgrad(
    grad_out: torch.Tensor,
    weights: Optional[List[torch.Tensor]],
    m_splits: Sequence[int],
    *,
    out_dtype: torch.dtype = torch.bfloat16,
    num_cu: Optional[int] = None,
    weight_col: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    """Grouped MXFP4 dgrad: ``dA[g] = gradO[g] @ W[g]`` (contract N).

    Uses the forward NT kernel with the col-wise weight as the transposed
    operand: ``B[g] = W_col`` has shape [K, N/2], so the kernel's free dim is K
    (in_features) and its contraction is N (out_features).

    Args:
        grad_out: [total_M, N] output gradient, grouped along M by ``m_splits``.
        weights: list of G per-expert weight tensors, each [N, K]. May be ``None``
            when ``weight_col`` supplies the pre-quantized weights.
        weight_col: optional pre-quantized col-wise weights
            ``(data [G, K, N/2], scale [G, K, N/32])`` -- skips the weight cast
            (e.g. shared with fprop's row-wise cast, see ``_quantize_weights_row_col``).

    Returns:
        [total_M, K] input gradient in ``out_dtype``.
    """
    N = grad_out.shape[1]
    _require_gfx950()
    _check_contract(N, "N")

    if weight_col is not None:
        b_data, b_scale = weight_col
        K = b_data.shape[1]
        _check_splits(m_splits, grad_out.shape[0], b_data.shape[0])
    else:
        assert weights is not None, "dgrad needs weights or weight_col"
        K = weights[0].shape[1]
        _check_splits(m_splits, grad_out.shape[0], len(weights))
        b_datas, b_scales = [], []
        for w in weights:
            d, s = _col_operand(w)  # (K, N/2), (K, N/32)
            b_datas.append(d)
            b_scales.append(s)
        b_data = torch.stack(b_datas, dim=0)  # (G, K, N/2)
        b_scale = torch.stack(b_scales, dim=0)  # (G, K, N/32)

    go_data, go_scale = _row_operand(grad_out)  # (total_M, N/2), (total_M, N/32)
    group_offs = _prefix_offsets(m_splits, grad_out.device)
    # Kernel free dim = b.shape[-2] = K; kernel contraction = b.shape[-1]*2 = N.
    return grouped_gemm_mxfp4_triton_kernel(
        go_data,
        go_scale,
        b_data,
        b_scale,
        group_offs,
        K,
        N,
        group_offs_out=group_offs,
        out_dtype=out_dtype,
        num_cu=num_cu,
    )


def grouped_gemm_mxfp4_wgrad(
    a: torch.Tensor,
    grad_out: torch.Tensor,
    m_splits: Sequence[int],
    *,
    out_dtype: torch.dtype = torch.bfloat16,
    num_cu: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
    accumulate: bool = False,
) -> torch.Tensor:
    """Grouped MXFP4 wgrad: ``dW[g] = gradO[g]^T @ A[g]`` (contract M, variable-K).

    Both operands are col-wise, per-group padded to a BLOCK_SIZE_K-multiple M so
    the reduction over M needs no masking.

    Args:
        a: [total_M, K] activations, grouped along M by ``m_splits``.
        grad_out: [total_M, N] output gradient, grouped along M by ``m_splits``.
        out: [G, N, K] buffer to write into instead of allocating; with
            ``accumulate=True`` the wgrad is added into it (beta=1, e.g. main_grad).

    Returns:
        [G, N, K] weight gradient in ``out_dtype``.
    """
    N = grad_out.shape[1]
    K = a.shape[1]
    G = len(m_splits)
    _require_gfx950()
    if a.shape[0] != grad_out.shape[0]:
        raise ValueError(f"a rows ({a.shape[0]}) must equal grad_out rows ({grad_out.shape[0]})")
    _check_splits(m_splits, a.shape[0])

    lhs_data, lhs_scale, go_pad = _col_operand_grouped_padded(grad_out, m_splits)  # (N, Mpad/2)
    rhs_data, rhs_scale, _ = _col_operand_grouped_padded(a, m_splits)  # (K, Mpad/2)

    return grouped_gemm_mxfp4_variable_k_triton_kernel(
        lhs_data,
        lhs_scale,
        rhs_data,
        rhs_scale,
        go_pad,
        N,
        K,
        G,
        out_dtype=out_dtype,
        num_cu=num_cu,
        out=out,
        accumulate=accumulate,
    )


class _GroupedGemmMXFP4Func(torch.autograd.Function):
    """Autograd for grouped MXFP4 linear: ``out[g] = a[g] @ weight[g]^T``.

    The weight is cast_transposed once in forward (``_quantize_weights_row_col``):
    the row-wise layout feeds fprop and the col-wise layout is stashed for dgrad,
    so the weight quant is paid once per step instead of twice. Activation and
    grad_out still cast per pass -- their col layouts are per-group padded for the
    variable-K wgrad reduction, so they don't share with the un-padded row cast.
    """

    @staticmethod
    def forward(ctx, a, weight, m_splits):
        weights = list(torch.unbind(weight, dim=0))
        w_row, w_col = _quantize_weights_row_col(weights)
        out = grouped_gemm_mxfp4_fprop(
            a, weights, m_splits, out_dtype=a.dtype, weight_row=w_row
        )
        ctx.save_for_backward(a, weight, w_col[0], w_col[1])
        ctx.m_splits = m_splits
        return out

    @staticmethod
    def backward(ctx, grad_out):
        a, weight, w_col_data, w_col_scale = ctx.saved_tensors
        m_splits = ctx.m_splits
        grad_out = grad_out.contiguous()
        grad_a = grad_weight = None
        if ctx.needs_input_grad[0]:
            grad_a = grouped_gemm_mxfp4_dgrad(
                grad_out, None, m_splits, out_dtype=a.dtype,
                weight_col=(w_col_data, w_col_scale),
            )
        if ctx.needs_input_grad[1]:
            grad_weight = grouped_gemm_mxfp4_wgrad(a, grad_out, m_splits, out_dtype=weight.dtype)
        return grad_a, grad_weight, None


def grouped_linear_mxfp4(
    a: torch.Tensor,
    weight: torch.Tensor,
    m_splits: Sequence[int],
) -> torch.Tensor:
    """Autograd-enabled grouped MXFP4 linear.

    Args:
        a: [total_M, K] activations, grouped along M by ``m_splits``.
        weight: [G, N, K] stacked per-expert weights.
        m_splits: per-group token counts (len G).

    Returns:
        [total_M, N] output.
    """
    return _GroupedGemmMXFP4Func.apply(a, weight, tuple(int(m) for m in m_splits))
