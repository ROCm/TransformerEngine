# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information
#
# Adapted from AITER (ROCm/aiter @ 7d2f6a51a, aiter/ops/triton/_triton_kernels/
# chunk_delta_attn and gated_delta_rule/prefill/chunk_delta_h.py), itself adapted
# from flash-linear-attention: Copyright (c) 2023-2026, Songlin Yang, Yu Zhang,
# Zhiyuan Li. Both are MIT licensed.

# pylint: disable=possibly-used-before-assignment

"""Framework-agnostic Triton kernels for Kimi Delta Attention (KDA), forward only.

Two implementations of the same forward pass live here:

* The default pipeline -- gate cumsum, intra-chunk attention (``Aqk``/``Akk``
  and the triangular inverse), W/U recompute, the inter-chunk state recurrence,
  and the output projection. General shapes: GVA, any ``K <= 256``, chunk 32/64.
* FlashKDA -- a two-kernel split (per-chunk prepare, then a sequential
  recurrence that keeps the state in registers), optionally segmented into an
  affine scan for occupancy. ``K == V == 128``, ``C == 32``, no GVA, bf16,
  Kimi's sigmoid gate with in-kernel l2norm and beta sigmoid.

Kernel bodies follow AITER. Differences, all needed to drive the same kernels
from both PyTorch and JAX (``triton_extensions``):

* No ``@triton.heuristics`` / ``@triton.autotune``. Flags are explicit
  constexprs and launch configs come from ``kda_launch_config`` -- the
  configs AITER runs with its autotuning off (its default).
* Tensor parameters are ordered inputs first, outputs last, which is how the
  JAX bridge binds them.
* Kernels that read ``chunk_indices`` return early on chunks past the end of
  their sequence, so a varlen caller can pad ``chunk_indices`` to a static
  upper bound (see ``PADDED_CHUNK``).
* ``_kda_inter_solve_kernel`` writes the upper-triangular blocks of ``Akk``
  as zeros instead of relying on a zero-initialized buffer.
"""

import functools
import math

import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

RCP_LN2: float = math.log2(math.e)  # 1/ln(2), for log2-space gate arithmetic

# Local chunk index written into padded varlen ``chunk_indices`` rows. Every
# kernel that reads ``chunk_indices`` returns once ``i_t * BT >= T``, which a
# sentinel this large guarantees for any sequence.
PADDED_CHUNK: int = 1 << 24
_PADDED_CHUNK_C = tl.constexpr(PADDED_CHUNK)

# FlashKDA restrictions. K is consumed as two 64-wide halves so the recurrent
# state fits a pair of [64, BW] register tiles; C = 32 because the intra-chunk
# decay is re-centered on the chunk midpoint, which keeps it in fp32 for about
# +-70 log2 units, and the Kimi gate spends ~3.6 of those per token.
FLASH_KDA_K: int = 128
FLASH_KDA_CHUNK: int = 32
# Width of the diagonal blocks the WY inverse is built from.
FLASH_KDA_INV_BLOCK: int = 16

# Sub-chunk width of the default intra kernel.
KDA_SUB_CHUNK: int = 16

SOLVE_TRIL_DOT_PRECISION = tl.constexpr("ieee")


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------


@triton.jit
def exp(x):
    """Natural exp in fp32."""
    return tl.exp(x.to(tl.float32))


@triton.jit
def exp2(x):
    """Base-2 exp in fp32."""
    return tl.math.exp2(x.to(tl.float32))


@triton.jit
def softplus(x):
    """log(1 + exp(x)), falling back to the identity above x=20 so exp cannot overflow."""
    return tl.where(x < 20.0, tl.log(1.0 + tl.exp(x)), x)


# ---------------------------------------------------------------------------
# Default pipeline: elementwise prologue
# ---------------------------------------------------------------------------


@triton.jit
def _kda_l2norm_kernel(
    X,
    Y,
    eps,
    T,
    D: tl.constexpr,
    BD: tl.constexpr,
    BT: tl.constexpr,
):
    """L2 normalize per row, D <= 512 (BT rows per program)."""
    xoffset = tl.program_id(0).to(tl.int64) * BT
    row_idx = xoffset + tl.arange(0, BT)[:, None]
    xmask = row_idx < T
    col_idx = tl.arange(0, BD)[None, :]
    cmask = col_idx < D
    mask = xmask & cmask
    x = tl.load(X + col_idx + D * row_idx, mask=mask, other=0.0).to(tl.float32)
    sumsq = tl.sum(tl.where(xmask, x * x, 0.0), axis=1)
    rstd = tl.rsqrt(sumsq + eps)
    y = x * rstd[:, None]
    tl.store(Y + col_idx + D * row_idx, y.to(Y.dtype.element_ty), mask=mask)


@triton.jit
def _kda_beta_sigmoid_kernel(
    x,
    y,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = offs < n_elements
    b_x = tl.load(x + offs, mask=mask, other=0).to(tl.float32)
    b_y = tl.sigmoid(b_x)
    tl.store(y + offs, b_y.to(y.dtype.element_ty), mask=mask)


# ---------------------------------------------------------------------------
# Default pipeline: gate cumsum
# ---------------------------------------------------------------------------


@triton.jit
def _kda_gate_cumsum_kernel(
    s,
    A_log,
    dt_bias,
    cu_seqlens,
    chunk_indices,
    o,
    scale,
    lower_bound,
    T,
    H: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
):
    """Chunk-local cumsum of the fused gate (-exp(A)*softplus or lb*sigmoid(exp(A)*g))."""
    i_s, i_t, i_bh = (
        tl.program_id(0),
        tl.program_id(1).to(tl.int64),
        tl.program_id(2).to(tl.int64),
    )
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        T = eos - bos
        if i_t * BT >= T:
            return
    else:
        bos, eos = i_b * T, i_b * T + T

    o_t = i_t * BT + tl.arange(0, BT)
    o_s = i_s * BS + tl.arange(0, BS)
    m_s = (o_t[:, None] < T) & (o_s[None, :] < S)
    p_s = s + (bos * H + i_h) * S + o_t[:, None] * (H * S) + o_s[None, :]
    p_o = o + (bos * H + i_h) * S + o_t[:, None] * (H * S) + o_s[None, :]

    b_s = tl.load(p_s, mask=m_s, other=0.0).to(tl.float32)

    if HAS_BIAS:
        b_bias = tl.load(dt_bias + i_h * S + o_s, mask=o_s < S, other=0.0).to(tl.float32)
        b_s = b_s + b_bias[None, :]

    b_A = tl.load(A_log + i_h).to(tl.float32)
    if not USE_LOWER_BOUND:
        b_gate = -exp(b_A) * softplus(b_s)
    else:
        b_gate = lower_bound * tl.sigmoid(exp(b_A) * b_s)

    b_o = tl.cumsum(b_gate, axis=0)

    if HAS_SCALE:
        b_o *= scale

    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_s)


@triton.jit
def _kda_local_cumsum_kernel(
    s,
    cu_seqlens,
    chunk_indices,
    o,
    scale,
    T,
    H: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Chunk-local cumsum of a precomputed (log-space) gate ``[B, T, H, S]``."""
    i_s, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(
            tl.int32
        )
        T = eos - bos
        if i_t * BT >= T:
            return
    else:
        bos, eos = i_b * T, i_b * T + T

    o_t = i_t * BT + tl.arange(0, BT)
    o_s = i_s * BS + tl.arange(0, BS)
    msk = (o_t < T)[:, None] & (o_s < S)[None, :]
    s_base = s + (bos * H + i_h) * S
    o_base = o + (bos * H + i_h) * S
    stride_t = H * S
    offs = o_t[:, None] * stride_t + o_s[None, :]
    b_s = tl.load(s_base + offs, mask=msk, other=0.0).to(tl.float32)
    b_o = tl.cumsum(b_s, axis=0)
    if HAS_SCALE:
        b_o *= scale
    tl.store(o_base + offs, b_o.to(o_base.dtype.element_ty), mask=msk)


# ---------------------------------------------------------------------------
# Default pipeline: intra-chunk attention
# ---------------------------------------------------------------------------


@triton.jit
def _kda_intra_token_parallel_kernel(
    q,
    k,
    g,
    beta,
    cu_seqlens,
    Aqk,
    Akk,
    scale,
    N,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BH: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Diagonal ``Akk`` blocks + diagonal ``Aqk`` blocks, one program per token (non-safe gate)."""
    i_tg, i_hg = tl.program_id(0).to(tl.int64), tl.program_id(1)

    if IS_VARLEN:
        i_n = 0
        left, right = 0, N
        for _ in range(20):
            if left < right:
                mid = (left + right) // 2
                if i_tg < tl.load(cu_seqlens + mid + 1).to(tl.int32):
                    right = mid
                else:
                    left = mid + 1
        i_n = left
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        T = eos - bos
        i_t = i_tg - bos
    else:
        bos = (i_tg // T) * T
        i_t = i_tg % T

    if i_t >= T:
        return

    i_c = i_t // BT
    i_s = (i_t % BT) // BC
    i_tc = i_c * BT
    i_ts = i_tc + i_s * BC

    G: tl.constexpr = HV // H

    q += bos * H * K
    k += bos * H * K
    g += bos * HV * K
    Aqk += bos * HV * BT
    Akk += bos * HV * BC
    beta += bos * HV

    o_hv = i_hg * BH + tl.arange(0, BH)
    o_h = o_hv // G
    m_hv = o_hv < HV

    p_beta = beta + i_t * HV + o_hv
    b_beta = tl.load(p_beta, mask=m_hv, other=0.0).to(tl.float32)

    for j in range(i_ts, min(i_t + 1, min(T, i_ts + BC))):  # pylint: disable=nested-min-max
        b_Aqk_j = tl.zeros([BH], dtype=tl.float32)
        b_Akk_j = tl.zeros([BH], dtype=tl.float32)
        for i_k in range(tl.cdiv(K, BK)):
            o_k = i_k * BK + tl.arange(0, BK)
            m_k = o_k < K
            m_hk = m_hv[:, None] & m_k[None, :]
            p_qk = o_h[:, None] * K + o_k[None, :]

            b_q = tl.load(q + i_t * H * K + p_qk, mask=m_hk, other=0).to(tl.float32)
            b_k = tl.load(k + i_t * H * K + p_qk, mask=m_hk, other=0).to(tl.float32)
            b_kj = tl.load(k + j * H * K + p_qk, mask=m_hk, other=0).to(tl.float32)

            p_g = g + i_t * HV * K + o_hv[:, None] * K + o_k[None, :]
            p_gj = g + j * HV * K + o_hv[:, None] * K + o_k[None, :]
            b_g = tl.load(p_g, mask=m_hk, other=0.0).to(tl.float32)
            b_gj = tl.load(p_gj, mask=m_hk, other=0.0).to(tl.float32)

            b_kgj = tl.where(m_k[None, :], b_kj * exp2(b_g - b_gj), 0.0)
            b_Aqk_j += tl.sum(b_q * b_kgj, axis=1)
            b_Akk_j += tl.sum(b_k * b_beta[:, None] * b_kgj, axis=1)

        b_Aqk_j *= scale
        b_Akk_j *= tl.where(j < i_t, 1.0, 0.0)
        tl.store(
            Aqk + i_t * HV * BT + o_hv * BT + j % BT,
            b_Aqk_j.to(Aqk.dtype.element_ty),
            mask=m_hv,
        )
        tl.store(
            Akk + i_t * HV * BC + o_hv * BC + j - i_ts,
            b_Akk_j.to(Akk.dtype.element_ty),
            mask=m_hv,
        )


@triton.jit
def _kda_intra_sub_chunk_kernel(
    q,
    k,
    g,
    beta,
    cu_seqlens,
    chunk_indices,
    Aqk,
    Akk,
    scale,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Diagonal blocks with a sub-chunk midpoint pivot, plus their inverse (safe gate)."""
    i_t, i_i, i_bh = (
        tl.program_id(0).to(tl.int64),
        tl.program_id(1),
        tl.program_id(2).to(tl.int64),
    )
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    i_ti = i_t * BT + i_i * BC
    if i_ti >= T:
        return

    o_c = i_ti + tl.arange(0, BC)
    m_c = o_c < T

    q = q + (bos * H + i_h) * K
    k = k + (bos * H + i_h) * K
    g = g + (bos * HV + i_hv) * K
    beta = beta + bos * HV + i_hv
    Aqk = Aqk + (bos * HV + i_hv) * BT
    Akk = Akk + (bos * HV + i_hv) * BC

    p_beta = beta + o_c * HV
    b_beta = tl.load(p_beta, mask=m_c, other=0.0)

    b_Aqk = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk = tl.zeros([BC, BC], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        m_ck = m_c[:, None] & m_k[None, :]
        p_q = q + o_c[:, None] * (H * K) + o_k[None, :]
        p_k = k + o_c[:, None] * (H * K) + o_k[None, :]
        p_g = g + o_c[:, None] * (HV * K) + o_k[None, :]
        b_q = tl.load(p_q, mask=m_ck, other=0.0)
        b_k = tl.load(p_k, mask=m_ck, other=0.0)
        b_g = tl.load(p_g, mask=m_ck, other=0.0)

        # Reference gate at the mid-point of this sub-chunk.
        b_gn = tl.gather(
            b_g,
            tl.full([1, BK], min(BC // 2, T - i_ti - 1), dtype=tl.int16),
            axis=0,
        )

        b_gm = (b_g - b_gn).to(tl.float32)
        b_gq = tl.where(m_c[:, None], exp2(b_gm), 0.0)
        b_gk = tl.where(m_c[:, None], exp2(-b_gm), 0.0)

        b_kgt = tl.trans(b_k * b_gk)
        b_Aqk += tl.dot(b_q * b_gq, b_kgt)
        b_Akk += tl.dot(b_k * b_gq, b_kgt)

    b_Aqk *= scale
    b_Akk *= b_beta[:, None]

    o_i = tl.arange(0, BC)
    m_Aqk = o_i[:, None] >= o_i[None, :]
    m_Akk = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]

    b_Aqk = tl.where(m_Aqk, b_Aqk, 0.0)
    b_Akk = tl.where(m_Akk, b_Akk, 0.0)

    m_Aqk_st = m_c[:, None] & (o_i[None, :] < BT)
    m_Akk_st = m_c[:, None] & (o_i[None, :] < BC)
    p_Aqk = Aqk + o_c[:, None] * (HV * BT) + (i_i * BC + o_i)[None, :]
    p_Akk = Akk + o_c[:, None] * (HV * BC) + o_i[None, :]
    tl.store(p_Aqk, b_Aqk.to(Aqk.dtype.element_ty), mask=m_Aqk_st)

    # Forward substitution (in-place into register, then store)
    tl.store(p_Akk, b_Akk.to(Akk.dtype.element_ty), mask=m_Akk_st)
    tl.debug_barrier()

    b_Ai = -b_Akk
    for i in range(2, min(BC, T - i_ti)):
        b_a = -tl.load(Akk + (i_ti + i) * HV * BC + o_i)
        b_a = tl.where(o_i < i, b_a, 0.0)
        b_a += tl.sum(b_a[:, None] * b_Ai, 0)
        b_Ai = tl.where((o_i == i)[:, None], b_a, b_Ai)
    b_Ai += m_I
    tl.store(p_Akk, b_Ai.to(Akk.dtype.element_ty), mask=m_Akk_st)


@triton.jit
def _kda_inter_solve_kernel(
    q,
    k,
    g,
    beta,
    Akkd,
    Aqk_diag,  # pylint: disable=unused-argument
    cu_seqlens,
    chunk_indices,
    Aqk,
    Akk,
    scale,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    NC: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_SAFE_GATE: tl.constexpr,
):
    """Off-diagonal ``Aqk``/``Akk`` blocks, then the full chunk triangular inverse into ``Akk``.

    ``Aqk`` must be the buffer the intra kernel wrote the diagonal blocks into:
    this kernel only adds the off-diagonal ones. ``Aqk_diag`` is that same
    buffer and is never read here -- it exists so a framework that binds inputs
    and outputs separately (JAX) can alias it to ``Aqk``. ``Akk``'s
    upper-triangular blocks are written as zeros -- the W/U kernel reads them
    unmasked.
    """
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if i_t * BT >= T:
        return

    i_tc0 = i_t * BT
    i_tc1 = i_t * BT + BC
    i_tc2 = i_t * BT + 2 * BC
    i_tc3 = i_t * BT + 3 * BC

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    g += (bos * HV + i_hv) * K
    Aqk += (bos * HV + i_hv) * BT
    Akk += (bos * HV + i_hv) * BT
    Akkd += (bos * HV + i_hv) * BC

    o_i = tl.arange(0, BC)
    m_tc1 = (i_tc1 + o_i) < T
    m_tc2 = (i_tc2 + o_i) < T
    m_tc3 = (i_tc3 + o_i) < T
    o_c0 = i_tc0 + o_i
    o_c1 = i_tc1 + o_i
    o_c2 = i_tc2 + o_i
    o_c3 = i_tc3 + o_i
    m_tc0 = o_c0 < T
    m_A0 = m_tc0[:, None] & (o_i[None, :] < BT)
    m_A1 = m_tc1[:, None] & (o_i[None, :] < BT)
    m_A2 = m_tc2[:, None] & (o_i[None, :] < BT)
    m_A3 = m_tc3[:, None] & (o_i[None, :] < BT)

    b_Aqk10 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk10 = tl.zeros([BC, BC], dtype=tl.float32)

    b_Aqk20 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk20 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk21 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk21 = tl.zeros([BC, BC], dtype=tl.float32)

    b_Aqk30 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk30 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk31 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk31 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk32 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk32 = tl.zeros([BC, BC], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K

        m_ck0 = m_tc0[:, None] & m_k[None, :]
        p_k0 = k + o_c0[:, None] * (H * K) + o_k[None, :]
        p_g0 = g + o_c0[:, None] * (HV * K) + o_k[None, :]
        b_k0 = tl.load(p_k0, mask=m_ck0, other=0.0).to(tl.float32)
        b_g0 = tl.load(p_g0, mask=m_ck0, other=0.0).to(tl.float32)

        if i_tc1 < T:
            m_ck1 = m_tc1[:, None] & m_k[None, :]
            p_q1 = q + o_c1[:, None] * (H * K) + o_k[None, :]
            p_k1 = k + o_c1[:, None] * (H * K) + o_k[None, :]
            p_g1 = g + o_c1[:, None] * (HV * K) + o_k[None, :]
            b_q1 = tl.load(p_q1, mask=m_ck1, other=0.0).to(tl.float32)
            b_k1 = tl.load(p_k1, mask=m_ck1, other=0.0).to(tl.float32)
            b_g1 = tl.load(p_g1, mask=m_ck1, other=0.0).to(tl.float32)
            b_gn1 = tl.load(g + i_tc1 * HV * K + o_k, mask=m_k, other=0).to(tl.float32)
            b_gqn = tl.where(m_tc1[:, None], exp2(b_g1 - b_gn1[None, :]), 0)
            b_kgt = tl.trans(b_k0 * exp2(b_gn1[None, :] - b_g0))
            b_Aqk10 += tl.dot(b_q1 * b_gqn, b_kgt)
            b_Akk10 += tl.dot(b_k1 * b_gqn, b_kgt)

            if NC >= 3 and i_tc2 < T:
                m_ck2 = m_tc2[:, None] & m_k[None, :]
                p_q2 = q + o_c2[:, None] * (H * K) + o_k[None, :]
                p_k2 = k + o_c2[:, None] * (H * K) + o_k[None, :]
                p_g2 = g + o_c2[:, None] * (HV * K) + o_k[None, :]
                b_q2 = tl.load(p_q2, mask=m_ck2, other=0.0).to(tl.float32)
                b_k2 = tl.load(p_k2, mask=m_ck2, other=0.0).to(tl.float32)
                b_g2 = tl.load(p_g2, mask=m_ck2, other=0.0).to(tl.float32)
                b_gn2 = tl.load(g + i_tc2 * HV * K + o_k, mask=m_k, other=0).to(tl.float32)
                b_gqn2 = tl.where(m_tc2[:, None], exp2(b_g2 - b_gn2[None, :]), 0)
                b_qg2 = b_q2 * b_gqn2
                b_kg2 = b_k2 * b_gqn2
                b_kgt = tl.trans(b_k0 * exp2(b_gn2[None, :] - b_g0))
                b_Aqk20 += tl.dot(b_qg2, b_kgt)
                b_Akk20 += tl.dot(b_kg2, b_kgt)
                b_kgt = tl.trans(b_k1 * exp2(b_gn2[None, :] - b_g1))
                b_Aqk21 += tl.dot(b_qg2, b_kgt)
                b_Akk21 += tl.dot(b_kg2, b_kgt)

                if NC >= 4 and i_tc3 < T:
                    m_ck3 = m_tc3[:, None] & m_k[None, :]
                    p_q3 = q + o_c3[:, None] * (H * K) + o_k[None, :]
                    p_k3 = k + o_c3[:, None] * (H * K) + o_k[None, :]
                    p_g3 = g + o_c3[:, None] * (HV * K) + o_k[None, :]
                    b_q3 = tl.load(p_q3, mask=m_ck3, other=0.0).to(tl.float32)
                    b_k3 = tl.load(p_k3, mask=m_ck3, other=0.0).to(tl.float32)
                    b_g3 = tl.load(p_g3, mask=m_ck3, other=0.0).to(tl.float32)
                    b_gn3 = tl.load(g + i_tc3 * HV * K + o_k, mask=m_k, other=0).to(tl.float32)
                    b_gqn3 = tl.where(m_tc3[:, None], exp2(b_g3 - b_gn3[None, :]), 0)
                    b_qg3 = b_q3 * b_gqn3
                    b_kg3 = b_k3 * b_gqn3
                    b_kgt = tl.trans(b_k0 * exp2(b_gn3[None, :] - b_g0))
                    b_Aqk30 += tl.dot(b_qg3, b_kgt)
                    b_Akk30 += tl.dot(b_kg3, b_kgt)
                    b_kgt = tl.trans(b_k1 * exp2(b_gn3[None, :] - b_g1))
                    b_Aqk31 += tl.dot(b_qg3, b_kgt)
                    b_Akk31 += tl.dot(b_kg3, b_kgt)
                    b_kgt = tl.trans(b_k2 * exp2(b_gn3[None, :] - b_g2))
                    b_Aqk32 += tl.dot(b_qg3, b_kgt)
                    b_Akk32 += tl.dot(b_kg3, b_kgt)

    if i_tc1 < T:
        p_Aqk10 = Aqk + o_c1[:, None] * (HV * BT) + o_i[None, :]
        tl.store(p_Aqk10, (b_Aqk10 * scale).to(Aqk.dtype.element_ty), mask=m_A1)
        p_b1 = beta + bos * HV + i_hv + o_c1 * HV
        b_b1 = tl.load(p_b1, mask=m_tc1, other=0.0).to(tl.float32)
        b_Akk10 *= b_b1[:, None]
    if NC >= 3 and i_tc2 < T:
        p_Aqk20 = Aqk + o_c2[:, None] * (HV * BT) + o_i[None, :]
        p_Aqk21 = Aqk + o_c2[:, None] * (HV * BT) + (o_i + BC)[None, :]
        tl.store(p_Aqk20, (b_Aqk20 * scale).to(Aqk.dtype.element_ty), mask=m_A2)
        tl.store(p_Aqk21, (b_Aqk21 * scale).to(Aqk.dtype.element_ty), mask=m_A2)
        p_b2 = beta + bos * HV + i_hv + o_c2 * HV
        b_b2 = tl.load(p_b2, mask=m_tc2, other=0.0).to(tl.float32)
        b_Akk20 *= b_b2[:, None]
        b_Akk21 *= b_b2[:, None]
    if NC >= 4 and i_tc3 < T:
        p_Aqk30 = Aqk + o_c3[:, None] * (HV * BT) + o_i[None, :]
        p_Aqk31 = Aqk + o_c3[:, None] * (HV * BT) + (o_i + BC)[None, :]
        p_Aqk32 = Aqk + o_c3[:, None] * (HV * BT) + (o_i + 2 * BC)[None, :]
        tl.store(p_Aqk30, (b_Aqk30 * scale).to(Aqk.dtype.element_ty), mask=m_A3)
        tl.store(p_Aqk31, (b_Aqk31 * scale).to(Aqk.dtype.element_ty), mask=m_A3)
        tl.store(p_Aqk32, (b_Aqk32 * scale).to(Aqk.dtype.element_ty), mask=m_A3)
        p_b3 = beta + bos * HV + i_hv + o_c3 * HV
        b_b3 = tl.load(p_b3, mask=m_tc3, other=0.0).to(tl.float32)
        b_Akk30 *= b_b3[:, None]
        b_Akk31 *= b_b3[:, None]
        b_Akk32 *= b_b3[:, None]

    p_Akk00 = Akkd + o_c0[:, None] * (HV * BC) + o_i[None, :]
    p_Akk11 = Akkd + o_c1[:, None] * (HV * BC) + o_i[None, :]
    b_Ai00 = tl.load(p_Akk00, mask=m_A0, other=0.0).to(tl.float32)
    b_Ai11 = tl.load(p_Akk11, mask=m_A1, other=0.0).to(tl.float32)
    if NC >= 3:
        p_Akk22 = Akkd + o_c2[:, None] * (HV * BC) + o_i[None, :]
        b_Ai22 = tl.load(p_Akk22, mask=m_A2, other=0.0).to(tl.float32)
    if NC >= 4:
        p_Akk33 = Akkd + o_c3[:, None] * (HV * BC) + o_i[None, :]
        b_Ai33 = tl.load(p_Akk33, mask=m_A3, other=0.0).to(tl.float32)

    if not USE_SAFE_GATE:
        m_A = o_i[:, None] > o_i[None, :]
        m_I = o_i[:, None] == o_i[None, :]

        b_Ai00 = -tl.where(m_A, b_Ai00, 0)
        b_Ai11 = -tl.where(m_A, b_Ai11, 0)
        if NC >= 3:
            b_Ai22 = -tl.where(m_A, b_Ai22, 0)
        if NC >= 4:
            b_Ai33 = -tl.where(m_A, b_Ai33, 0)

        for i in range(2, min(BC, T - i_tc0)):
            b_a00 = -tl.load(Akkd + (i_tc0 + i) * HV * BC + o_i)
            b_a00 = tl.where(o_i < i, b_a00, 0.0)
            b_a00 += tl.sum(b_a00[:, None] * b_Ai00, 0)
            b_Ai00 = tl.where((o_i == i)[:, None], b_a00, b_Ai00)
        for i in range(BC + 2, min(2 * BC, T - i_tc0)):
            b_a11 = -tl.load(Akkd + (i_tc0 + i) * HV * BC + o_i)
            b_a11 = tl.where(o_i < i - BC, b_a11, 0.0)
            b_a11 += tl.sum(b_a11[:, None] * b_Ai11, 0)
            b_Ai11 = tl.where((o_i == i - BC)[:, None], b_a11, b_Ai11)
        if NC >= 3:
            for i in range(2 * BC + 2, min(3 * BC, T - i_tc0)):
                b_a22 = -tl.load(Akkd + (i_tc0 + i) * HV * BC + o_i)
                b_a22 = tl.where(o_i < i - 2 * BC, b_a22, 0.0)
                b_a22 += tl.sum(b_a22[:, None] * b_Ai22, 0)
                b_Ai22 = tl.where((o_i == i - 2 * BC)[:, None], b_a22, b_Ai22)
        if NC >= 4:
            for i in range(3 * BC + 2, min(4 * BC, T - i_tc0)):
                b_a33 = -tl.load(Akkd + (i_tc0 + i) * HV * BC + o_i)
                b_a33 = tl.where(o_i < i - 3 * BC, b_a33, 0.0)
                b_a33 += tl.sum(b_a33[:, None] * b_Ai33, 0)
                b_Ai33 = tl.where((o_i == i - 3 * BC)[:, None], b_a33, b_Ai33)

        b_Ai00 += m_I
        b_Ai11 += m_I
        if NC >= 3:
            b_Ai22 += m_I
        if NC >= 4:
            b_Ai33 += m_I

    b_Ai10 = -tl.dot(
        tl.dot(b_Ai11, b_Akk10, input_precision=SOLVE_TRIL_DOT_PRECISION),
        b_Ai00,
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    if NC >= 3:
        b_Ai21 = -tl.dot(
            tl.dot(b_Ai22, b_Akk21, input_precision=SOLVE_TRIL_DOT_PRECISION),
            b_Ai11,
            input_precision=SOLVE_TRIL_DOT_PRECISION,
        )
        b_Ai20 = -tl.dot(
            b_Ai22,
            tl.dot(b_Akk20, b_Ai00, input_precision=SOLVE_TRIL_DOT_PRECISION)
            + tl.dot(b_Akk21, b_Ai10, input_precision=SOLVE_TRIL_DOT_PRECISION),
            input_precision=SOLVE_TRIL_DOT_PRECISION,
        )
    if NC >= 4:
        b_Ai32 = -tl.dot(
            tl.dot(b_Ai33, b_Akk32, input_precision=SOLVE_TRIL_DOT_PRECISION),
            b_Ai22,
            input_precision=SOLVE_TRIL_DOT_PRECISION,
        )
        b_Ai31 = -tl.dot(
            b_Ai33,
            tl.dot(b_Akk31, b_Ai11, input_precision=SOLVE_TRIL_DOT_PRECISION)
            + tl.dot(b_Akk32, b_Ai21, input_precision=SOLVE_TRIL_DOT_PRECISION),
            input_precision=SOLVE_TRIL_DOT_PRECISION,
        )
        b_Ai30 = -tl.dot(
            b_Ai33,
            tl.dot(b_Akk30, b_Ai00, input_precision=SOLVE_TRIL_DOT_PRECISION)
            + tl.dot(b_Akk31, b_Ai10, input_precision=SOLVE_TRIL_DOT_PRECISION)
            + tl.dot(b_Akk32, b_Ai20, input_precision=SOLVE_TRIL_DOT_PRECISION),
            input_precision=SOLVE_TRIL_DOT_PRECISION,
        )

    # Upper-triangular blocks are zero; written here so Akk needs no zero-init.
    b_zero = tl.zeros([BC, BC], dtype=tl.float32).to(Akk.dtype.element_ty)
    for i_c in tl.static_range(NC):
        o_r = i_t * BT + i_c * BC + o_i
        m_r = (o_r < T)[:, None] & (o_i[None, :] < BT)
        for i_z in tl.static_range(i_c + 1, NC):
            p_z = Akk + o_r[:, None] * (HV * BT) + (o_i + i_z * BC)[None, :]
            tl.store(p_z, b_zero, mask=m_r)

    p_Akk00 = Akk + o_c0[:, None] * (HV * BT) + o_i[None, :]
    p_Akk10 = Akk + o_c1[:, None] * (HV * BT) + o_i[None, :]
    p_Akk11 = Akk + o_c1[:, None] * (HV * BT) + (o_i + BC)[None, :]
    tl.store(p_Akk00, b_Ai00.to(Akk.dtype.element_ty), mask=m_A0)
    tl.store(p_Akk10, b_Ai10.to(Akk.dtype.element_ty), mask=m_A1)
    tl.store(p_Akk11, b_Ai11.to(Akk.dtype.element_ty), mask=m_A1)
    if NC >= 3:
        p_Akk20 = Akk + o_c2[:, None] * (HV * BT) + o_i[None, :]
        p_Akk21 = Akk + o_c2[:, None] * (HV * BT) + (o_i + BC)[None, :]
        p_Akk22 = Akk + o_c2[:, None] * (HV * BT) + (o_i + 2 * BC)[None, :]
        tl.store(p_Akk20, b_Ai20.to(Akk.dtype.element_ty), mask=m_A2)
        tl.store(p_Akk21, b_Ai21.to(Akk.dtype.element_ty), mask=m_A2)
        tl.store(p_Akk22, b_Ai22.to(Akk.dtype.element_ty), mask=m_A2)
    if NC >= 4:
        p_Akk30 = Akk + o_c3[:, None] * (HV * BT) + o_i[None, :]
        p_Akk31 = Akk + o_c3[:, None] * (HV * BT) + (o_i + BC)[None, :]
        p_Akk32 = Akk + o_c3[:, None] * (HV * BT) + (o_i + 2 * BC)[None, :]
        p_Akk33 = Akk + o_c3[:, None] * (HV * BT) + (o_i + 3 * BC)[None, :]
        tl.store(p_Akk30, b_Ai30.to(Akk.dtype.element_ty), mask=m_A3)
        tl.store(p_Akk31, b_Ai31.to(Akk.dtype.element_ty), mask=m_A3)
        tl.store(p_Akk32, b_Ai32.to(Akk.dtype.element_ty), mask=m_A3)
        tl.store(p_Akk33, b_Ai33.to(Akk.dtype.element_ty), mask=m_A3)


# ---------------------------------------------------------------------------
# Default pipeline: W/U recompute
# ---------------------------------------------------------------------------


@triton.jit
def _kda_recompute_w_u_kernel(
    k,
    v,
    beta,
    A,
    gk,
    cu_seqlens,
    chunk_indices,
    w,
    u,
    kg,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """w = A @ (k*beta*exp2(gk)), u = A @ (v*beta), kg = k*exp2(gk_last - gk)."""
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        T = eos - bos
        if i_t * BT >= T:
            return
    else:
        bos, eos = i_b * T, i_b * T + T

    k += (bos * H + i_h) * K
    v += (bos * HV + i_hv) * V
    u += (bos * HV + i_hv) * V
    w += (bos * HV + i_hv) * K
    gk += (bos * HV + i_hv) * K
    beta += bos * HV + i_hv
    A += (bos * HV + i_hv) * BT
    kg += (bos * HV + i_hv) * K

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    p_b = beta + o_t * HV
    b_b = tl.load(p_b, mask=m_t, other=0.0)

    o_A = tl.arange(0, BT)
    m_A = m_t[:, None] & (o_A[None, :] < BT)
    p_A = A + o_t[:, None] * (HV * BT) + o_A[None, :]
    b_A = tl.load(p_A, mask=m_A, other=0.0)

    # u = A @ (v * beta)
    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = m_t[:, None] & (o_v[None, :] < V)
        p_v = v + o_t[:, None] * (HV * V) + o_v[None, :]
        p_u = u + o_t[:, None] * (HV * V) + o_v[None, :]
        b_v = tl.load(p_v, mask=m_v, other=0.0)
        b_vb = (b_v * b_b[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), mask=m_v)

    # w = A @ (k * beta * exp2(gk)); kg = k * exp2(gk_last - gk)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        m_tk = m_t[:, None] & m_k[None, :]
        p_w = w + o_t[:, None] * (HV * K) + o_k[None, :]
        p_k = k + o_t[:, None] * (H * K) + o_k[None, :]
        p_gk = gk + o_t[:, None] * (HV * K) + o_k[None, :]

        b_k = tl.load(p_k, mask=m_tk, other=0.0)
        b_gk = tl.load(p_gk, mask=m_tk, other=0.0).to(tl.float32)
        b_kb = b_k * b_b[:, None]
        b_kb *= exp2(b_gk)

        last_idx = min(i_t * BT + BT, T) - 1
        b_gn = tl.load(gk + last_idx * HV * K + o_k, mask=m_k, other=0.0).to(tl.float32)
        b_kg = b_k * tl.where(
            (i_t * BT + tl.arange(0, BT) < T)[:, None],
            exp2(b_gn[None, :] - b_gk),
            0,
        )
        p_kg = kg + o_t[:, None] * (HV * K) + o_k[None, :]
        tl.store(p_kg, b_kg.to(p_kg.dtype.element_ty), mask=m_tk)

        b_w = tl.dot(b_A, b_kb.to(b_k.dtype))
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), mask=m_tk)


# ---------------------------------------------------------------------------
# Default pipeline: inter-chunk state recurrence
# ---------------------------------------------------------------------------


@triton.jit
def _kda_fwd_h_kernel(
    k,
    v,
    w,
    gk,
    h0,
    cu_seqlens,
    chunk_offsets,
    h,
    v_new,
    ht,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    TRANSPOSE_STATE: tl.constexpr,
):
    """Per-chunk hidden states ``h`` and ``v_new = u - w @ h`` for a per-K (vector) log2 gate.

    ``H`` here is the value-head count. ``h`` is indexed by global chunk,
    ``chunk_offsets[n]`` being sequence ``n``'s first one.
    """
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(
            tl.int32
        )
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    b_h1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 128:
        b_h3 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 192:
        b_h4 = tl.zeros([64, BV], dtype=tl.float32)

    # Widen to int64 before scaling by K/V: a 64-head K=V=128 model overflows
    # int32 at ~131k tokens.
    h += (boh * H + i_h).to(tl.int64) * K * V
    v += (bos * H + i_h).to(tl.int64) * V
    k += (bos * H + i_h).to(tl.int64) * K
    w += (bos * H + i_h).to(tl.int64) * K
    v_new += (bos * H + i_h).to(tl.int64) * V
    stride_v = H * V
    stride_h = H * K * V
    stride_k = H * K
    if USE_INITIAL_STATE:
        h0 = h0 + i_nh.to(tl.int64) * K * V
    if STORE_FINAL_STATE:
        ht = ht + i_nh.to(tl.int64) * K * V

    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    o_k1 = tl.arange(0, 64)
    m_k1 = o_k1 < K
    o_k2 = 64 + o_k1
    m_k2 = o_k2 < K
    o_k3 = 128 + o_k1
    m_k3 = o_k3 < K
    o_k4 = 192 + o_k1
    m_k4 = o_k4 < K
    m_h1 = m_k1[:, None] & m_v[None, :]
    m_h2 = m_k2[:, None] & m_v[None, :]
    m_h3 = m_k3[:, None] & m_v[None, :]
    m_h4 = m_k4[:, None] & m_v[None, :]

    # TRANSPOSE_STATE describes the state buffers as V-first ``[V, K]``.
    if USE_INITIAL_STATE:
        if TRANSPOSE_STATE:
            p_h0_1 = h0 + o_k1[:, None] + o_v[None, :] * K
        else:
            p_h0_1 = h0 + o_k1[:, None] * V + o_v[None, :]
        b_h1 += tl.load(p_h0_1, mask=m_h1, other=0.0).to(tl.float32)
        if K > 64:
            if TRANSPOSE_STATE:
                p_h0_2 = h0 + o_k2[:, None] + o_v[None, :] * K
            else:
                p_h0_2 = h0 + o_k2[:, None] * V + o_v[None, :]
            b_h2 += tl.load(p_h0_2, mask=m_h2, other=0.0).to(tl.float32)
        if K > 128:
            if TRANSPOSE_STATE:
                p_h0_3 = h0 + o_k3[:, None] + o_v[None, :] * K
            else:
                p_h0_3 = h0 + o_k3[:, None] * V + o_v[None, :]
            b_h3 += tl.load(p_h0_3, mask=m_h3, other=0.0).to(tl.float32)
        if K > 192:
            if TRANSPOSE_STATE:
                p_h0_4 = h0 + o_k4[:, None] + o_v[None, :] * K
            else:
                p_h0_4 = h0 + o_k4[:, None] * V + o_v[None, :]
            b_h4 += tl.load(p_h0_4, mask=m_h4, other=0.0).to(tl.float32)

    for i_t in range(NT):
        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        m_tk1 = m_t[:, None] & m_k1[None, :]
        m_tv = m_t[:, None] & m_v[None, :]

        h_t = h + i_t.to(tl.int64) * stride_h
        p_h1 = h_t + o_k1[:, None] * V + o_v[None, :]
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), mask=m_h1)
        if K > 64:
            p_h2 = h_t + o_k2[:, None] * V + o_v[None, :]
            tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), mask=m_h2)
        if K > 128:
            p_h3 = h_t + o_k3[:, None] * V + o_v[None, :]
            tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), mask=m_h3)
        if K > 192:
            p_h4 = h_t + o_k4[:, None] * V + o_v[None, :]
            tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), mask=m_h4)

        p_w = w + o_t[:, None] * stride_k + o_k1[None, :]
        b_w = tl.load(p_w, mask=m_tk1, other=0.0)
        b_v = tl.dot(b_w, b_h1.to(b_w.dtype))
        if K > 64:
            p_w = w + o_t[:, None] * stride_k + o_k2[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & m_k2[None, :], other=0.0)
            b_v = tl.dot(b_w, b_h2.to(b_w.dtype), acc=b_v)
        if K > 128:
            p_w = w + o_t[:, None] * stride_k + o_k3[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & m_k3[None, :], other=0.0)
            b_v = tl.dot(b_w, b_h3.to(b_w.dtype), acc=b_v)
        if K > 192:
            p_w = w + o_t[:, None] * stride_k + o_k4[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & m_k4[None, :], other=0.0)
            b_v = tl.dot(b_w, b_h4.to(b_w.dtype), acc=b_v)
        p_v = v + o_t[:, None] * stride_v + o_v[None, :]
        b_v = tl.load(p_v, mask=m_tv, other=0.0) - b_v

        p_v = v_new + o_t[:, None] * stride_v + o_v[None, :]
        tl.store(p_v, b_v.to(p_v.dtype.element_ty), mask=m_tv)

        last_idx = min((i_t + 1) * BT, T) - 1
        b_gk_last1 = tl.load(
            gk + (bos + last_idx) * H * K + i_h * K + o_k1,
            mask=m_k1,
            other=0.0,
        )
        b_h1 *= tl.math.exp2(b_gk_last1)[:, None]
        if K > 64:
            b_gk_last2 = tl.load(
                gk + (bos + last_idx) * H * K + i_h * K + o_k2,
                mask=m_k2,
                other=0.0,
            )
            b_h2 *= tl.math.exp2(b_gk_last2)[:, None]
        if K > 128:
            b_gk_last3 = tl.load(
                gk + (bos + last_idx) * H * K + i_h * K + o_k3,
                mask=m_k3,
                other=0.0,
            )
            b_h3 *= tl.math.exp2(b_gk_last3)[:, None]
        if K > 192:
            b_gk_last4 = tl.load(
                gk + (bos + last_idx) * H * K + i_h * K + o_k4,
                mask=m_k4,
                other=0.0,
            )
            b_h4 *= tl.math.exp2(b_gk_last4)[:, None]
        b_v = b_v.to(k.dtype.element_ty)

        p_k = k + o_k1[:, None] + o_t[None, :] * stride_k
        b_k = tl.load(p_k, mask=m_k1[:, None] & m_t[None, :], other=0.0)
        b_h1 = tl.dot(b_k, b_v, acc=b_h1)
        if K > 64:
            p_k = k + o_k2[:, None] + o_t[None, :] * stride_k
            b_k = tl.load(p_k, mask=m_k2[:, None] & m_t[None, :], other=0.0)
            b_h2 = tl.dot(b_k, b_v, acc=b_h2)
        if K > 128:
            p_k = k + o_k3[:, None] + o_t[None, :] * stride_k
            b_k = tl.load(p_k, mask=m_k3[:, None] & m_t[None, :], other=0.0)
            b_h3 = tl.dot(b_k, b_v, acc=b_h3)
        if K > 192:
            p_k = k + o_k4[:, None] + o_t[None, :] * stride_k
            b_k = tl.load(p_k, mask=m_k4[:, None] & m_t[None, :], other=0.0)
            b_h4 = tl.dot(b_k, b_v, acc=b_h4)

    if STORE_FINAL_STATE:
        if TRANSPOSE_STATE:
            p_ht = ht + o_k1[:, None] + o_v[None, :] * K
        else:
            p_ht = ht + o_k1[:, None] * V + o_v[None, :]
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), mask=m_h1)
        if K > 64:
            if TRANSPOSE_STATE:
                p_ht = ht + o_k2[:, None] + o_v[None, :] * K
            else:
                p_ht = ht + o_k2[:, None] * V + o_v[None, :]
            tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), mask=m_h2)
        if K > 128:
            if TRANSPOSE_STATE:
                p_ht = ht + o_k3[:, None] + o_v[None, :] * K
            else:
                p_ht = ht + o_k3[:, None] * V + o_v[None, :]
            tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), mask=m_h3)
        if K > 192:
            if TRANSPOSE_STATE:
                p_ht = ht + o_k4[:, None] + o_v[None, :] * K
            else:
                p_ht = ht + o_k4[:, None] * V + o_v[None, :]
            tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), mask=m_h4)


# ---------------------------------------------------------------------------
# Default pipeline: output
# ---------------------------------------------------------------------------


@triton.jit
def _kda_gla_fwd_o_kernel(
    q,
    v,
    g,
    h,
    A,
    cu_seqlens,
    chunk_indices,
    o,
    scale,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """o = (q * exp2(g)) @ h * scale + tril(A) @ v_new."""
    i_v, i_t, i_bh = (
        tl.program_id(0),
        tl.program_id(1).to(tl.int64),
        tl.program_id(2).to(tl.int64),
    )
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        i_tg = i_t.to(tl.int64)
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        T = eos - bos
        if i_t * BT >= T:
            return
    else:
        NT = tl.cdiv(T, BT)
        i_tg = (i_b * NT + i_t).to(tl.int64)
        bos = (i_b * T).to(tl.int64)

    m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]

    q += (bos * H + i_h) * K
    g += (bos * HV + i_hv) * K
    v += (bos * HV + i_hv) * V
    o += (bos * HV + i_hv) * V
    h += (i_tg * HV + i_hv).to(tl.int64) * K * V
    A += (bos * HV + i_hv) * BT

    b_o = tl.zeros([BT, BV], dtype=tl.float32)

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        m_tk = m_t[:, None] & m_k[None, :]
        p_q = q + o_t[:, None] * (H * K) + o_k[None, :]
        p_g = g + o_t[:, None] * (HV * K) + o_k[None, :]
        p_h = h + o_k[:, None] * V + o_v[None, :]
        m_h = m_k[:, None] & m_v[None, :]

        b_q = tl.load(p_q, mask=m_tk, other=0.0)
        b_g = tl.load(p_g, mask=m_tk, other=0.0).to(tl.float32)
        b_qg = (b_q * exp2(b_g)).to(b_q.dtype)
        b_h = tl.load(p_h, mask=m_h, other=0.0)
        b_o += tl.dot(b_qg, b_h.to(b_qg.dtype))

    b_o *= scale

    o_A = tl.arange(0, BT)
    m_tv = m_t[:, None] & m_v[None, :]
    m_A = m_t[:, None] & (o_A[None, :] < BT)
    p_v = v + o_t[:, None] * (HV * V) + o_v[None, :]
    p_o = o + o_t[:, None] * (HV * V) + o_v[None, :]
    p_A = A + o_t[:, None] * (HV * BT) + o_A[None, :]

    b_v = tl.load(p_v, mask=m_tv, other=0.0)
    b_A = tl.load(p_A, mask=m_A, other=0.0)
    b_A = tl.where(m_s, b_A, 0.0).to(b_v.dtype)
    b_o += tl.dot(b_A, b_v)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_tv)


# ---------------------------------------------------------------------------
# FlashKDA
# ---------------------------------------------------------------------------


@triton.jit
def _flash_kda_prepare_kernel(
    q,
    k,
    g_raw,
    beta_raw,
    A_log,
    dt_bias,
    cu_seqlens,
    chunk_indices,
    ws_kd,
    ws_qd,
    ws_kr,
    ws_gt,
    ws_inv_mqk,
    scale,
    lower_bound,
    T,
    NT,
    TOTAL_TILES,
    H: tl.constexpr,
    K: tl.constexpr,
    C: tl.constexpr,
    BC: tl.constexpr,
    NUM_DOUBLING: tl.constexpr,
    NUM_MERGE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    CM_QKG: tl.constexpr,
    CM_WS: tl.constexpr,
):
    """K1, per chunk: decayed q/k, gate total, Mqk, and (I - L)^-1 into the workspace."""
    i_t = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        # program_id(0) is the global chunk index; chunk_indices maps it to
        # (sequence, chunk-within-sequence).
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int64)
        i_tl = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T_seq = eos - bos
        g_tile = i_t
    else:
        i_tl = i_t
        bos = i_b * T
        T_seq = T
        g_tile = i_b * NT + i_t

    t_off = i_tl * C
    if t_off >= T_seq:
        return
    actual_len = tl.minimum(C, T_seq - t_off)

    o_c = tl.arange(0, C)
    o_k = tl.arange(0, K)
    m_c = o_c < actual_len
    m_ck = m_c[:, None]

    base = (bos + t_off) * H + i_h
    qk_off = base * K + o_c[:, None] * (H * K) + o_k[None, :]

    b_q = tl.load(q + qk_off, mask=m_ck, other=0.0, cache_modifier=CM_QKG).to(tl.float32)
    b_k = tl.load(k + qk_off, mask=m_ck, other=0.0, cache_modifier=CM_QKG).to(tl.float32)

    # L2 normalize rows, matching the l2norm kernel's eps placement.
    b_q = b_q * (1.0 / tl.sqrt(tl.sum(b_q * b_q, axis=1) + 1e-6))[:, None]
    b_k = b_k * (1.0 / tl.sqrt(tl.sum(b_k * b_k, axis=1) + 1e-6))[:, None]

    # Gate: lower_bound * sigmoid(exp(A_log) * (g + dt_bias)), then chunk-local
    # cumsum into log2 space; same expression as the default gate cumsum.
    b_g = tl.load(g_raw + qk_off, mask=m_ck, other=0.0, cache_modifier=CM_QKG).to(tl.float32)
    if HAS_BIAS:
        b_g = b_g + tl.load(dt_bias + i_h * K + o_k).to(tl.float32)[None, :]
    b_A = tl.load(A_log + i_h).to(tl.float32)
    b_gate = lower_bound * tl.sigmoid(exp(b_A) * b_g)
    LOG2_E: tl.constexpr = 1.4426950408889634
    b_gcum = tl.cumsum(b_gate, axis=0) * LOG2_E
    b_gcum = tl.where(m_ck, b_gcum, 0.0)

    # gcum is non-positive and decreasing down the chunk, so every exponent
    # below is <= 0 and exp2 cannot overflow.
    last_row = actual_len - 1
    b_g_last = tl.sum(tl.where(o_c[:, None] == last_row, b_gcum, 0.0), axis=0)
    b_g_total = exp2(b_g_last)

    # Decay measured from the chunk start; only ever multiplies the state.
    b_exp_g = exp2(b_gcum)

    # Stored early to free registers before the inversion. Workspace is
    # [H, TOTAL_TILES, ...]; tail rows are zeroed because K2 accumulates over
    # all C rows.
    ws_idx = i_h * TOTAL_TILES + g_tile
    ck_off = ws_idx * C * K + o_c[:, None] * K + o_k[None, :]
    tl.store(
        ws_kd + ck_off,
        tl.where(m_ck, b_k * b_exp_g, 0.0).to(tl.bfloat16),
        cache_modifier=CM_WS,
    )
    tl.store(
        ws_qd + ck_off,
        tl.where(m_ck, b_q * b_exp_g * scale, 0.0).to(tl.bfloat16),
        cache_modifier=CM_WS,
    )
    tl.store(
        ws_kr + ck_off,
        tl.where(m_ck, b_k * exp2(b_g_last[None, :] - b_gcum), 0.0).to(tl.bfloat16),
        cache_modifier=CM_WS,
    )
    tl.store(ws_gt + ws_idx * K + o_k, b_g_total, cache_modifier=CM_WS)

    p_beta = beta_raw + (bos + t_off) * H + i_h + o_c * H
    b_beta = tl.sigmoid(tl.load(p_beta, mask=m_c, other=0.0).to(tl.float32))

    # Decay *differences* within a chunk are O(1) near the diagonal but cannot be
    # factored as exp2(gcum[i]) * exp2(-gcum[j]); re-centering on the chunk
    # midpoint keeps both factors inside fp32 for C <= 32.
    o_mid = tl.minimum(C // 2, actual_len - 1)
    b_gp = tl.sum(tl.where(o_c[:, None] == o_mid, b_gcum, 0.0), axis=0)
    b_gm = b_gcum - b_gp[None, :]
    b_dec = exp2(b_gm)
    b_inc = exp2(-b_gm)
    b_k_piv = tl.where(m_ck, b_k * b_dec, 0.0).to(tl.bfloat16)
    b_q_piv = tl.where(m_ck, b_q * b_dec * scale, 0.0).to(tl.bfloat16)
    b_k_inv = tl.where(m_ck, b_k * b_inc, 0.0).to(tl.bfloat16)

    # The WY form needs (I + tril(diag(beta) K K^T, -1))^-1. L is negated here so
    # the doubling below is the plain Neumann series in L.
    o_i = tl.arange(0, C)
    b_L = tl.dot(b_k_piv, tl.trans(b_k_inv))
    b_L = tl.where(o_i[:, None] > o_i[None, :], -b_L * b_beta[:, None], 0.0)

    cc_off = ws_idx * 2 * C * C + o_i[:, None] * C + o_i[None, :]
    b_Mqk = tl.dot(b_q_piv, tl.trans(b_k_inv))
    b_Mqk = tl.where(o_i[:, None] >= o_i[None, :], b_Mqk, 0.0)
    tl.store(ws_inv_mqk + cc_off + C * C, b_Mqk, cache_modifier=CM_WS)

    # Invert the BC-wide diagonal blocks via (I+D)(I+D^2)(I+D^4)..., then fold
    # the sub-diagonal blocks back in, doubling the block width each round. The
    # block split bounds the intermediates (flat doubling over 32 rows is 400%
    # wrong even in fp32). input_precision is spelled out: an unannotated fp32
    # dot resolves to tf32, which gfx942 truncates to 19 bits.
    b_D = tl.where(o_i[:, None] // BC == o_i[None, :] // BC, b_L, 0.0)
    b_INV = tl.where(o_i[:, None] == o_i[None, :], 1.0, 0.0) + b_D
    b_Dp = tl.dot(b_D, b_D, input_precision="ieee")
    for _ in tl.static_range(NUM_DOUBLING):
        b_INV = b_INV + tl.dot(b_INV, b_Dp, input_precision="ieee")
        b_Dp = tl.dot(b_Dp, b_Dp, input_precision="ieee")

    w = BC
    for _ in tl.static_range(NUM_MERGE):
        m_off = (o_i[:, None] // (2 * w) == o_i[None, :] // (2 * w)) & (
            o_i[:, None] // w != o_i[None, :] // w
        )
        b_off = tl.where(m_off, b_L, 0.0)
        b_INV = b_INV + tl.dot(
            b_INV,
            tl.dot(b_off, b_INV, input_precision="ieee"),
            input_precision="ieee",
        )
        w = 2 * w

    tl.store(ws_inv_mqk + cc_off, b_INV, cache_modifier=CM_WS)


@triton.jit
def _flash_kda_segment_kernel(
    ws_kd,
    ws_qd,
    ws_kr,
    ws_gt,
    ws_inv_mqk,
    v_input,
    beta_raw,
    h_in,
    seg_chunk_base,
    seg_nchunks,
    seg_tok_base,
    seg_tok_end,
    seg_seq,
    seg_is_last,
    out,
    h_out,
    final_state,
    TOTAL_TILES,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    W: tl.constexpr,
    C: tl.constexpr,
    BW: tl.constexpr,
    INIT_IDENTITY: tl.constexpr,
    HAS_H_IN: tl.constexpr,
    HAS_V: tl.constexpr,
    COMPUTE_OUTPUT: tl.constexpr,
    STORE_H_OUT: tl.constexpr,
    STORE_FINAL: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    CM_OUT: tl.constexpr,
):
    """K2: delta-rule recurrence over one segment of chunks, state in registers.

    The update is affine in the incoming state, ``h' = A_seg h + b_seg``, so one
    kernel serves all three passes of the segmented schedule:

      * ``h=0``, real ``v``            -> ``b_seg``
      * ``h=I``, ``v=0`` (``W = K``)   -> ``A_seg``
      * ``h=h_in`` from the scan, ``COMPUTE_OUTPUT`` -> the output

    Segments with ``nchunks == 0`` (padding) run no iterations.
    """
    i_w = tl.program_id(0).to(tl.int64)
    i_sh = tl.program_id(1).to(tl.int64)
    i_seg, i_h = i_sh // H, i_sh % H

    chunk_base = tl.load(seg_chunk_base + i_seg).to(tl.int64)
    n_chunks = tl.load(seg_nchunks + i_seg)
    tok_base = tl.load(seg_tok_base + i_seg).to(tl.int64)
    tok_end = tl.load(seg_tok_end + i_seg).to(tl.int64)

    o_c = tl.arange(0, C)
    o_k1 = tl.arange(0, 64)
    o_k2 = 64 + tl.arange(0, 64)
    o_w = i_w * BW + tl.arange(0, BW)
    m_w = o_w < W

    if INIT_IDENTITY:
        b_h1 = tl.where(o_k1[:, None] == o_w[None, :], 1.0, 0.0)
        b_h2 = tl.where(o_k2[:, None] == o_w[None, :], 1.0, 0.0)
    elif HAS_H_IN:
        s_off = (i_seg * H + i_h) * K * W + o_w[None, :]
        b_h1 = tl.load(h_in + s_off + o_k1[:, None] * W, mask=m_w[None, :], other=0.0)
        b_h2 = tl.load(h_in + s_off + o_k2[:, None] * W, mask=m_w[None, :], other=0.0)
        b_h1 = b_h1.to(tl.float32)
        b_h2 = b_h2.to(tl.float32)
    else:
        b_h1 = tl.zeros([64, BW], dtype=tl.float32)
        b_h2 = tl.zeros([64, BW], dtype=tl.float32)

    for j in range(n_chunks):
        ws_idx = i_h * TOTAL_TILES + chunk_base + j
        ck = ws_idx * C * K
        t0 = tok_base + j * C
        m_c = (t0 + o_c) < tok_end
        m_cw = m_c[:, None] & m_w[None, :]

        b_h1_bf = b_h1.to(tl.bfloat16)
        b_h2_bf = b_h2.to(tl.bfloat16)

        b_kd1 = tl.load(ws_kd + ck + o_c[:, None] * K + o_k1[None, :])
        b_kd2 = tl.load(ws_kd + ck + o_c[:, None] * K + o_k2[None, :])
        b_tmp = tl.dot(b_kd1, b_h1_bf) + tl.dot(b_kd2, b_h2_bf)

        vo_off = t0 * H * V + i_h * V + o_c[:, None] * (H * V) + o_w[None, :]
        if HAS_V:
            b_v = tl.load(v_input + vo_off, mask=m_cw, other=0.0).to(tl.float32)
        else:
            b_v = tl.zeros([C, BW], dtype=tl.float32)

        p_beta = beta_raw + t0 * H + i_h + o_c * H
        b_beta = tl.sigmoid(tl.load(p_beta, mask=m_c, other=0.0).to(tl.float32))

        # Tail rows must stay zero: U feeds the state update over all C rows.
        b_u = tl.where(m_c[:, None], (b_v - b_tmp) * b_beta[:, None], 0.0)

        cc = ws_idx * 2 * C * C + o_c[:, None] * C + o_c[None, :]
        b_inv = tl.load(ws_inv_mqk + cc)
        b_U = tl.dot(b_inv, b_u.to(ws_inv_mqk.dtype.element_ty))

        if COMPUTE_OUTPUT:
            b_mqk = tl.load(ws_inv_mqk + cc + C * C)
            b_qd1 = tl.load(ws_qd + ck + o_c[:, None] * K + o_k1[None, :])
            b_qd2 = tl.load(ws_qd + ck + o_c[:, None] * K + o_k2[None, :])
            b_o = tl.dot(b_qd1, b_h1_bf) + tl.dot(b_qd2, b_h2_bf)
            b_o += tl.dot(b_mqk, b_U.to(b_mqk.dtype))
            tl.store(
                out + vo_off,
                b_o.to(out.dtype.element_ty),
                mask=m_cw,
                cache_modifier=CM_OUT,
            )

        b_gt1 = tl.load(ws_gt + ws_idx * K + o_k1).to(tl.float32)
        b_gt2 = tl.load(ws_gt + ws_idx * K + o_k2).to(tl.float32)

        # The recurrence's own matmuls stay in bf16; the state accumulates in fp32.
        b_U_bf = b_U.to(tl.bfloat16)
        b_kr1_t = tl.load(ws_kr + ck + o_k1[:, None] + o_c[None, :] * K)
        b_kr2_t = tl.load(ws_kr + ck + o_k2[:, None] + o_c[None, :] * K)

        b_h1 = b_h1 * b_gt1[:, None] + tl.dot(b_kr1_t, b_U_bf).to(tl.float32)
        b_h2 = b_h2 * b_gt2[:, None] + tl.dot(b_kr2_t, b_U_bf).to(tl.float32)

    if STORE_H_OUT:
        s_off = (i_seg * H + i_h) * K * W + o_w[None, :]
        e_ty = h_out.dtype.element_ty
        tl.store(h_out + s_off + o_k1[:, None] * W, b_h1.to(e_ty), mask=m_w[None, :])
        tl.store(h_out + s_off + o_k2[:, None] * W, b_h2.to(e_ty), mask=m_w[None, :])

    # Not merged into one condition: when STORE_FINAL is false final_state is
    # null and seg_is_last must not be read.
    if STORE_FINAL:  # noqa: SIM102
        if tl.load(seg_is_last + i_seg) == 1:
            i_n = tl.load(seg_seq + i_seg).to(tl.int64)
            dt_s = final_state.dtype.element_ty
            if STATE_V_FIRST:
                f_off = (i_n * H + i_h) * V * K + o_w[None, :] * K
                tl.store(final_state + f_off + o_k1[:, None], b_h1.to(dt_s), mask=m_w[None, :])
                tl.store(final_state + f_off + o_k2[:, None], b_h2.to(dt_s), mask=m_w[None, :])
            else:
                f_off = (i_n * H + i_h) * K * V + o_w[None, :]
                tl.store(final_state + f_off + o_k1[:, None] * V, b_h1.to(dt_s), mask=m_w[None, :])
                tl.store(final_state + f_off + o_k2[:, None] * V, b_h2.to(dt_s), mask=m_w[None, :])


@triton.jit
def _flash_kda_seg_scan_kernel(
    A_seg,
    b_seg,
    h0,
    seq_seg_off,
    h_in,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    HAS_H0: tl.constexpr,
):
    """Serial scan across a sequence's segments: ``h <- A_seg h + b_seg``."""
    i_v = tl.program_id(0).to(tl.int64)
    i_nh = tl.program_id(1).to(tl.int64)
    i_n, i_h = i_nh // H, i_nh % H

    s0 = tl.load(seq_seg_off + i_n).to(tl.int64)
    s1 = tl.load(seq_seg_off + i_n + 1).to(tl.int64)

    o_k1 = tl.arange(0, 64)
    o_k2 = 64 + tl.arange(0, 64)
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V

    if HAS_H0:
        base = (i_n * H + i_h) * K * V + o_v[None, :]
        b_h1 = tl.load(h0 + base + o_k1[:, None] * V, mask=m_v[None, :], other=0.0).to(tl.float32)
        b_h2 = tl.load(h0 + base + o_k2[:, None] * V, mask=m_v[None, :], other=0.0).to(tl.float32)
    else:
        b_h1 = tl.zeros([64, BV], dtype=tl.float32)
        b_h2 = tl.zeros([64, BV], dtype=tl.float32)

    for s in range(s0, s1):
        hb = (s * H + i_h) * K * V + o_v[None, :]
        tl.store(h_in + hb + o_k1[:, None] * V, b_h1, mask=m_v[None, :])
        tl.store(h_in + hb + o_k2[:, None] * V, b_h2, mask=m_v[None, :])

        ab = (s * H + i_h) * K * K
        b_A11 = tl.load(A_seg + ab + o_k1[:, None] * K + o_k1[None, :])
        b_A12 = tl.load(A_seg + ab + o_k1[:, None] * K + o_k2[None, :])
        b_A21 = tl.load(A_seg + ab + o_k2[:, None] * K + o_k1[None, :])
        b_A22 = tl.load(A_seg + ab + o_k2[:, None] * K + o_k2[None, :])

        b_h1_bf = b_h1.to(tl.bfloat16)
        b_h2_bf = b_h2.to(tl.bfloat16)
        b_n1 = tl.dot(b_A11, b_h1_bf) + tl.dot(b_A12, b_h2_bf)
        b_n2 = tl.dot(b_A21, b_h1_bf) + tl.dot(b_A22, b_h2_bf)

        b_h1 = b_n1 + tl.load(b_seg + hb + o_k1[:, None] * V, mask=m_v[None, :], other=0.0)
        b_h2 = b_n2 + tl.load(b_seg + hb + o_k2[:, None] * V, mask=m_v[None, :], other=0.0)


# ---------------------------------------------------------------------------
# Varlen planning (TE addition; AITER plans on the host)
# ---------------------------------------------------------------------------


@triton.jit
def _kda_varlen_plan_kernel(
    cu_seqlens,
    chunk_indices,
    chunk_offsets,
    seg_desc,
    seq_seg_off,
    N,
    MAX_CHUNKS,
    MAX_SEGS,
    BT: tl.constexpr,
    CHUNKS_PER_SEG: tl.constexpr,
    BLOCK: tl.constexpr,
    WITH_SEGMENTS: tl.constexpr,
):
    """Chunk (and FlashKDA segment) tables for varlen input, built on the device.

    Each program fills ``BLOCK`` slots of the tables by walking the ``N``
    sequences once. Slots past the real count are padding -- ``PADDED_CHUNK``
    chunks, and zero-chunk segments that never end a sequence -- so the table
    sizes can be static upper bounds and nothing is read back to the host.
    Program 0 also writes the ``N + 1`` offsets.

    ``chunk_indices`` is ``[MAX_CHUNKS, 2]`` (sequence, chunk within sequence);
    ``seg_desc`` is ``[6, MAX_SEGS]`` (first global chunk, chunk count, first
    token, one-past-last token, sequence, is-last-of-sequence).
    """
    pid = tl.program_id(0)
    o = pid * BLOCK + tl.arange(0, BLOCK)

    c_seq = tl.zeros([BLOCK], dtype=tl.int32)
    c_local = tl.full([BLOCK], _PADDED_CHUNK_C, dtype=tl.int32)
    s_base = tl.zeros([BLOCK], dtype=tl.int32)
    s_cnt = tl.zeros([BLOCK], dtype=tl.int32)
    s_tok0 = tl.zeros([BLOCK], dtype=tl.int32)
    s_tok1 = tl.zeros([BLOCK], dtype=tl.int32)
    s_seq = tl.zeros([BLOCK], dtype=tl.int32)
    s_last = tl.zeros([BLOCK], dtype=tl.int32)

    c_off = tl.zeros([], dtype=tl.int32)
    s_off = tl.zeros([], dtype=tl.int32)
    for n in range(N):
        bos = tl.load(cu_seqlens + n).to(tl.int32)
        eos = tl.load(cu_seqlens + n + 1).to(tl.int32)
        nch = (eos - bos + BT - 1) // BT
        m = (o >= c_off) & (o < c_off + nch)
        c_seq = tl.where(m, n, c_seq)
        c_local = tl.where(m, o - c_off, c_local)
        if pid == 0:
            tl.store(chunk_offsets + n, c_off.to(chunk_offsets.dtype.element_ty))
        if WITH_SEGMENTS:
            nseg = tl.maximum((nch + CHUNKS_PER_SEG - 1) // CHUNKS_PER_SEG, 1)
            ms = (o >= s_off) & (o < s_off + nseg)
            ls = o - s_off
            c0 = ls * CHUNKS_PER_SEG
            cnt = tl.maximum(tl.minimum(nch - c0, CHUNKS_PER_SEG), 0)
            s_base = tl.where(ms, c_off + c0, s_base)
            s_cnt = tl.where(ms, cnt, s_cnt)
            s_tok0 = tl.where(ms, bos + c0 * BT, s_tok0)
            s_tok1 = tl.where(ms, tl.minimum(bos + (c0 + cnt) * BT, eos), s_tok1)
            s_seq = tl.where(ms, n, s_seq)
            s_last = tl.where(ms & (ls == nseg - 1), 1, s_last)
            if pid == 0:
                tl.store(seq_seg_off + n, s_off.to(seq_seg_off.dtype.element_ty))
            s_off += nseg
        c_off += nch

    if pid == 0:
        tl.store(chunk_offsets + N, c_off.to(chunk_offsets.dtype.element_ty))
        if WITH_SEGMENTS:
            tl.store(seq_seg_off + N, s_off.to(seq_seg_off.dtype.element_ty))

    ci_ty = chunk_indices.dtype.element_ty
    m_c = o < MAX_CHUNKS
    tl.store(chunk_indices + o * 2, c_seq.to(ci_ty), mask=m_c)
    tl.store(chunk_indices + o * 2 + 1, c_local.to(ci_ty), mask=m_c)
    if WITH_SEGMENTS:
        m_s = o < MAX_SEGS
        tl.store(seg_desc + o, s_base, mask=m_s)
        tl.store(seg_desc + MAX_SEGS + o, s_cnt, mask=m_s)
        tl.store(seg_desc + 2 * MAX_SEGS + o, s_tok0, mask=m_s)
        tl.store(seg_desc + 3 * MAX_SEGS + o, s_tok1, mask=m_s)
        tl.store(seg_desc + 4 * MAX_SEGS + o, s_seq, mask=m_s)
        tl.store(seg_desc + 5 * MAX_SEGS + o, s_last, mask=m_s)


KDA_VARLEN_PLAN_BLOCK: int = 256


# ---------------------------------------------------------------------------
# Launch configuration (framework-agnostic)
# ---------------------------------------------------------------------------


class KDALaunchConfig:
    """Tile parameters, ``num_warps`` and ``num_stages`` for one kernel launch."""

    __slots__ = ("kwargs", "num_warps", "num_stages")

    def __init__(self, kwargs=None, num_warps=4, num_stages=3):
        self.kwargs = dict(kwargs or {})
        self.num_warps = num_warps
        self.num_stages = num_stages

    def __repr__(self):
        return (
            f"KDALaunchConfig({self.kwargs}, num_warps={self.num_warps},"
            f" num_stages={self.num_stages})"
        )


# What AITER launches with its autotuning off (the default). Keyed by arch;
# "default" covers devices without a measured entry and must be launchable
# anywhere (the wide output tile needs more LDS than gfx942 has). Kernels AITER
# launches without an autotuner get the HIP backend's num_stages default (2);
# autotuned ones get triton.Config's (3) unless their config says otherwise.
_KDA_CONFIGS = {
    "default": {
        "l2norm": KDALaunchConfig({"BT": 32}, num_warps=4, num_stages=2),
        "beta_sigmoid": KDALaunchConfig({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=2),
        "gate_cumsum": KDALaunchConfig({"BS": 64}, num_warps=2, num_stages=3),
        "local_cumsum": KDALaunchConfig({"BS": 32}, num_warps=2, num_stages=3),
        "intra_token_parallel": KDALaunchConfig({"BH": 1, "BK": 64}, num_warps=4, num_stages=3),
        "intra_sub_chunk": KDALaunchConfig({}, num_warps=1, num_stages=3),
        "inter_solve": KDALaunchConfig({"BK": 32}, num_warps=1, num_stages=3),
        "recompute_w_u": KDALaunchConfig({"BK": 64, "BV": 64}, num_warps=4, num_stages=3),
        "fwd_h": KDALaunchConfig({"BV": 32}, num_warps=2, num_stages=2),
        "gla_fwd_o": KDALaunchConfig({"BK": 64, "BV": 64}, num_warps=4, num_stages=1),
        "flash_prepare": KDALaunchConfig({}, num_warps=2, num_stages=1),
        "flash_segment": KDALaunchConfig({"BW": 16}, num_warps=2, num_stages=2),
        # num_warps of the scan and the Gluon kernels are chosen per call.
        "flash_seg_scan": KDALaunchConfig({}, num_stages=2),
        "flash_gluon_k1": KDALaunchConfig({}, num_stages=2),
        # Gluon K2 pass A: (BW, num_warps) wide/narrow, wide taken above
        # MIN_BLOCKS_PER_CU blocks per CU.
        "flash_gluon_k2_wide": KDALaunchConfig(
            {"BW": 32, "MIN_BLOCKS_PER_CU": 0}, num_warps=2, num_stages=2
        ),
        "flash_gluon_k2_narrow": KDALaunchConfig({"BW": 32}, num_warps=2, num_stages=2),
    },
    "gfx942": {
        "gla_fwd_o": KDALaunchConfig({"BK": 32, "BV": 128}, num_warps=4, num_stages=2),
    },
    "gfx950": {
        "gla_fwd_o": KDALaunchConfig({"BK": 64, "BV": 128}, num_warps=8, num_stages=3),
        "flash_gluon_k2_wide": KDALaunchConfig(
            {"BW": 64, "MIN_BLOCKS_PER_CU": 2}, num_warps=4, num_stages=2
        ),
        "flash_gluon_k2_narrow": KDALaunchConfig({"BW": 32}, num_warps=2, num_stages=2),
    },
}


@functools.lru_cache(maxsize=None)
def kda_device_arch(device_index: int = 0) -> str:
    """gfx arch of a device, e.g. ``"gfx950"``."""
    props = triton.runtime.driver.active.utils.get_device_properties(device_index)
    return str(props["arch"]).split(":")[0]


@functools.lru_cache(maxsize=None)
def kda_num_cus(device_index: int = 0) -> int:
    """Compute-unit count of a device."""
    props = triton.runtime.driver.active.utils.get_device_properties(device_index)
    return props["multiprocessor_count"]


def kda_launch_config(name: str, arch: str) -> KDALaunchConfig:
    """Launch config of kernel ``name`` on ``arch``."""
    return _KDA_CONFIGS.get(arch, {}).get(name) or _KDA_CONFIGS["default"][name]


# ---------------------------------------------------------------------------
# FlashKDA scheduling (framework-agnostic)
# ---------------------------------------------------------------------------

# Blocks pass A should end up with, in units of the CU count.
_SEG_TARGET_BLOCKS = 3
_SEG_MAX_SEGMENTS = 16
_SEG_MAX_CHUNKS = 32
_SEG_MIN_CHUNKS = 64

_SCAN_BV_NARROW = 16
_SCAN_BV_WIDE = 32


def flash_kda_supported(
    K: int,
    V: int,
    H: int,
    HV: int,
    qv_bf16: bool,
    chunk_size: int,
    safe_gate: bool,
    use_gate_in_kernel: bool,
    use_qk_l2norm_in_kernel: bool,
    use_beta_sigmoid_in_kernel: bool,
    has_lower_bound: bool,
    has_A_log: bool,
) -> bool:
    """Whether the FlashKDA path can serve a call."""
    return (
        chunk_size == FLASH_KDA_CHUNK
        and safe_gate
        and K == FLASH_KDA_K
        and V == FLASH_KDA_K
        and HV == H
        and qv_bf16
        and use_gate_in_kernel
        and use_qk_l2norm_in_kernel
        and use_beta_sigmoid_in_kernel
        and has_lower_bound
        and has_A_log
    )


def flash_kda_choose_chunks_per_seg(
    n_chunks_max: int, n_seqs: int, H: int, V: int, num_cus: int
) -> int:
    """Segment length in chunks, or ``n_chunks_max`` to disable segmentation.

    K2 gets ``n_segments * H * (V / BW)`` blocks, so with one segment per
    sequence a low head count leaves most of the device idle. Segmenting buys
    blocks by turning one pass into three, so it is only taken when there is
    idle capacity and enough depth to amortize the extra passes.
    """
    blocks = n_seqs * H * max(1, V // 32)
    if blocks > num_cus:
        return n_chunks_max
    if n_chunks_max < _SEG_MIN_CHUNKS:
        return n_chunks_max
    segs = min(
        _SEG_MAX_SEGMENTS,
        max(_SEG_TARGET_BLOCKS * num_cus / blocks, n_chunks_max / _SEG_MAX_CHUNKS),
    )
    return max(1, min(n_chunks_max, 1 << round(math.log2(n_chunks_max / segs))))


def flash_kda_scan_bv(n_seqs: int, H: int, V: int, num_cus: int) -> tuple:
    """``(BV, num_warps)`` of the cross-segment scan."""
    if (V // _SCAN_BV_NARROW) * n_seqs * H > num_cus:
        return _SCAN_BV_WIDE, 2
    return _SCAN_BV_NARROW, 4


def flash_kda_gluon_k2_schedule(W: int, num_segs: int, H: int, arch: str, num_cus: int) -> tuple:
    """``(BW, num_warps, num_stages)`` for the Gluon fused pass A, which does not autotune."""
    wide = kda_launch_config("flash_gluon_k2_wide", arch)
    bw = wide.kwargs["BW"]
    blocks = (W // bw) * num_segs * H
    if W % bw == 0 and blocks >= wide.kwargs["MIN_BLOCKS_PER_CU"] * num_cus:
        return bw, wide.num_warps, wide.num_stages
    narrow = kda_launch_config("flash_gluon_k2_narrow", arch)
    return narrow.kwargs["BW"], narrow.num_warps, narrow.num_stages


def flash_kda_fixed_segments(B: int, T: int, C: int, chunks_per_seg: int) -> tuple:
    """Host-side segment descriptors for ``B`` equal-length sequences of ``T`` tokens.

    Returns ``(desc, seq_seg_off, num_segs)`` as plain lists: ``desc`` is six
    rows (chunk base, chunk count, first token, one-past-last token, sequence,
    is-last) of ``num_segs`` entries, ``seq_seg_off`` has ``B + 1`` entries.
    """
    rows = [[], [], [], [], [], []]
    seq_seg_off = [0]
    g_chunk = 0
    nch = triton.cdiv(T, C)
    nseg = max(1, triton.cdiv(nch, chunks_per_seg))
    for i in range(B):
        bos, eos = i * T, i * T + T
        for s in range(nseg):
            c0 = s * chunks_per_seg
            m = min(chunks_per_seg, nch - c0)
            rows[0].append(g_chunk + c0)
            rows[1].append(m)
            rows[2].append(bos + c0 * C)
            rows[3].append(min(bos + (c0 + m) * C, eos))
            rows[4].append(i)
            rows[5].append(1 if s == nseg - 1 else 0)
        seq_seg_off.append(len(rows[0]))
        g_chunk += nch
    return rows, seq_seg_off, len(rows[0])


def kda_varlen_max_chunks(total_tokens: int, n_seqs: int, chunk_size: int) -> int:
    """Upper bound on ``sum_i cdiv(len_i, chunk_size)`` over ``n_seqs`` sequences."""
    return (total_tokens + n_seqs * (chunk_size - 1)) // chunk_size
