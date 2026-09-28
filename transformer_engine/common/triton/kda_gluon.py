# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information
#
# Adapted from AITER (ROCm/aiter @ 7d2f6a51a,
# aiter/ops/triton/_gluon_kernels/gfx950/chunk_delta_attn), MIT licensed, plus
# the XCD remap of pass A from AITER branch zain/kda/xcd-remap (a3c3c2fdd).

"""Gluon (gfx950) replacements for FlashKDA's prepare kernel and fused pass A.

Framework-agnostic; see ``kda.py`` for the Triton kernels these stand in for.
Tensor parameters are ordered inputs first, outputs last, as in ``kda.py``.
Importing this module requires a Triton with Gluon; callers guard the import.
"""

import functools
import math

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from transformer_engine.common.triton.kda import remap_xcd

_BLK_WARP_K: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [1, 2], [1, 0])
_BLK1: gl.constexpr = gl.BlockedLayout([1], [64], [2], [0])
_BLK_CC: gl.constexpr = gl.BlockedLayout([1, 1], [4, 16], [2, 1], [1, 0])

_MMA_F16: gl.constexpr = gl.amd.AMDMFMALayout(
    version=4, instr_shape=[16, 16, 4], transposed=True, warps_per_cta=[2, 1]
)
_AF16: gl.constexpr = gl.DotOperandLayout(0, _MMA_F16, 1)
_BF16: gl.constexpr = gl.DotOperandLayout(1, _MMA_F16, 1)

_MMA_B16: gl.constexpr = gl.amd.AMDMFMALayout(
    version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[2, 1]
)
_A8_16: gl.constexpr = gl.DotOperandLayout(0, _MMA_B16, 8)
_B8_16: gl.constexpr = gl.DotOperandLayout(1, _MMA_B16, 8)

_SH_A: gl.constexpr = gl.SwizzledSharedLayout(8, 1, 16, [1, 0])
_SH_B: gl.constexpr = gl.SwizzledSharedLayout(8, 1, 16, [0, 1])
_SH_CC_F: gl.constexpr = gl.SwizzledSharedLayout(1, 2, 8, [0, 1])

# Warps the K1 layouts above are built for; the launch must match.
K1_NUM_WARPS: int = math.prod(_MMA_F16.warps_per_cta)


@gluon.jit
def _add(a, b):
    return a + b


@gluon.jit
def _exp(x):
    return gl.exp(x.to(gl.float32))


@gluon.jit
def _exp2(x):
    return gl.exp2(x.to(gl.float32))


@gluon.jit
def _sigmoid(x):
    return gl.extra.libdevice.fast_dividef(1.0, 1.0 + _exp(-x.to(gl.float32)))


@gluon.jit
def _l2norm(x):
    f = x.to(gl.float32)
    return f * gl.rsqrt(gl.sum(f * f, axis=1) + 1e-6)[:, None]


@gluon.jit
def _via_lds(x, shared: gl.constexpr, dot: gl.constexpr):
    return gl.allocate_shared_memory(x.dtype, x.shape, shared, x).load(dot)


@gluon.jit
def _dot_f32(a, b_op, a_op: gl.constexpr, acc_layout: gl.constexpr, N: gl.constexpr):
    return gl.amd.cdna4.mfma(
        gl.convert_layout(a, a_op),
        b_op,
        gl.zeros([a.shape[0], N], gl.float32, acc_layout),
    )


@gluon.jit
def flash_kda_k1_prepare_gluon(
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
    H: gl.constexpr,
    K: gl.constexpr,
    C: gl.constexpr,
    BC: gl.constexpr,
    IS_VARLEN: gl.constexpr,
    HAS_BIAS: gl.constexpr,
    CM_WS: gl.constexpr,
    CM_LOAD: gl.constexpr,
):
    """FlashKDA K1 on gfx950; same workspace contract as ``_flash_kda_prepare_kernel``."""
    gl.static_assert(C == 32 and K == 128)
    NUM_DOUBLING: gl.constexpr = BC.bit_length() - 2
    NUM_MERGE: gl.constexpr = (C // BC).bit_length() - 1

    i_t = gl.program_id(0).to(gl.int64)
    i_bh = gl.program_id(1).to(gl.int64)
    i_b = i_bh // H
    i_h = i_bh % H

    if IS_VARLEN:
        i_n = gl.load(chunk_indices + i_t * 2).to(gl.int64)
        i_tl = gl.load(chunk_indices + i_t * 2 + 1).to(gl.int64)
        bos = gl.load(cu_seqlens + i_n).to(gl.int64)
        eos = gl.load(cu_seqlens + i_n + 1).to(gl.int64)
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
    actual_len = gl.minimum(C, T_seq - t_off)

    o_c = gl.arange(0, C, layout=gl.SliceLayout(1, _BLK_WARP_K))
    o_k = gl.arange(0, K, layout=gl.SliceLayout(0, _BLK_WARP_K))
    o_k_v = gl.arange(0, K, layout=_BLK1)
    o_i_r = gl.arange(0, C, layout=gl.SliceLayout(1, _MMA_B16))
    o_i_c = gl.arange(0, C, layout=gl.SliceLayout(0, _MMA_B16))

    m_ck = (o_c < actual_len)[:, None]
    m_beta = o_i_r < actual_len

    base = (bos + t_off) * H + i_h
    qk_off = (base * K + o_c[:, None].to(gl.int64) * (H * K) + o_k[None, :]).to(gl.int32)

    b_q_raw = gl.amd.cdna4.buffer_load(ptr=q, offsets=qk_off, mask=m_ck, other=0.0, cache=CM_LOAD)
    b_k_raw = gl.amd.cdna4.buffer_load(ptr=k, offsets=qk_off, mask=m_ck, other=0.0, cache=CM_LOAD)
    if HAS_BIAS:
        bias = gl.amd.cdna4.buffer_load(ptr=dt_bias, offsets=(i_h * K).to(gl.int32) + o_k)[None, :]

    b_g = gl.amd.cdna4.buffer_load(
        ptr=g_raw, offsets=qk_off, mask=m_ck, other=0.0, cache=CM_LOAD
    ).to(gl.float32)
    if HAS_BIAS:
        b_g = b_g + bias
    b_A = gl.load(A_log + i_h)
    b_gate = lower_bound * _sigmoid(_exp(b_A) * b_g)
    log2_e: gl.constexpr = 1.4426950408889634
    b_gcum = gl.associative_scan(b_gate, 0, _add) * log2_e
    b_gcum = gl.where(m_ck, b_gcum, 0.0)

    b_g_last = gl.sum(gl.where(o_c[:, None] == actual_len - 1, b_gcum, 0.0), axis=0)
    b_g_total = _exp2(b_g_last)
    b_exp_g = _exp2(b_gcum)

    b_q = _l2norm(b_q_raw)
    b_k = _l2norm(b_k_raw)

    ws_idx = i_h * TOTAL_TILES + g_tile
    ck_off = (ws_idx * (C * K) + o_c[:, None].to(gl.int64) * K + o_k[None, :]).to(gl.int32)
    gl.amd.cdna4.buffer_store(
        gl.where(m_ck, b_k * b_exp_g, 0.0).to(ws_kd.dtype.element_ty),
        ws_kd,
        ck_off,
        cache=CM_WS,
    )
    gl.amd.cdna4.buffer_store(
        gl.where(m_ck, b_q * b_exp_g * scale, 0.0).to(ws_qd.dtype.element_ty),
        ws_qd,
        ck_off,
        cache=CM_WS,
    )
    b_kr_val = gl.where(m_ck, b_k * _exp2(b_g_last[None, :] - b_gcum), 0.0).to(gl.bfloat16)
    gl.amd.cdna4.buffer_store(b_kr_val.to(ws_kr.dtype.element_ty), ws_kr, ck_off, cache=CM_WS)
    gl.amd.cdna4.buffer_store(
        gl.convert_layout(b_g_total, _BLK1),
        ws_gt,
        (ws_idx * K).to(gl.int32) + o_k_v,
        cache=CM_WS,
    )

    b_beta = _sigmoid(
        gl.amd.cdna4.buffer_load(
            ptr=beta_raw,
            offsets=(base.to(gl.int32) + o_i_r * H),
            mask=m_beta,
            other=0.0,
        ).to(gl.float32)
    )

    o_mid = gl.minimum(C // 2, actual_len - 1)
    b_gp = gl.sum(gl.where(o_c[:, None] == o_mid, b_gcum, 0.0), axis=0)
    b_gm = b_gcum - b_gp[None, :]
    b_dec = _exp2(b_gm)
    b_inc = _exp2(-b_gm)
    b_k_piv = gl.where(m_ck, b_k * b_dec, 0.0).to(gl.bfloat16)
    b_q_piv = gl.where(m_ck, b_q * b_dec * scale, 0.0).to(gl.bfloat16)
    b_k_inv = gl.where(m_ck, b_k * b_inc, 0.0).to(gl.bfloat16)

    b_kinv_b = _via_lds(gl.permute(b_k_inv, 1, 0), _SH_B, _B8_16)

    b_L = gl.amd.cdna4.mfma(
        _via_lds(b_k_piv, _SH_A, _A8_16),
        b_kinv_b,
        gl.zeros([C, C], gl.float32, _MMA_B16),
    )
    b_L = gl.where(o_i_r[:, None] > o_i_c[None, :], -b_L * b_beta[:, None], 0.0)

    b_Mqk = gl.amd.cdna4.mfma(
        _via_lds(b_q_piv, _SH_A, _A8_16),
        b_kinv_b,
        gl.zeros([C, C], gl.float32, _MMA_B16),
    )
    b_Mqk = gl.where(o_i_r[:, None] >= o_i_c[None, :], b_Mqk, 0.0)

    o_r_cc = gl.arange(0, C, layout=gl.SliceLayout(1, _BLK_CC))
    o_c_cc = gl.arange(0, C, layout=gl.SliceLayout(0, _BLK_CC))
    cc_off_raw = (ws_idx * (2 * C * C) + o_r_cc[:, None].to(gl.int64) * C + o_c_cc[None, :]).to(
        gl.int32
    )
    gl.amd.cdna4.buffer_store(
        gl.convert_layout(b_Mqk.to(ws_inv_mqk.dtype.element_ty), _BLK_CC),
        ws_inv_mqk,
        cc_off_raw + C * C,
        cache=CM_WS,
    )

    if BC == C:
        b_D = b_L
    else:
        b_D = gl.where(o_i_r[:, None] // BC == o_i_c[None, :] // BC, b_L, 0.0)
    b_INV = gl.where(o_i_r[:, None] == o_i_c[None, :], 1.0, 0.0) + b_D
    b_INV = gl.convert_layout(b_INV, _MMA_F16)
    b_Dp = _dot_f32(b_D, _via_lds(b_D, _SH_CC_F, _BF16), _AF16, _MMA_F16, C)
    for _ in gl.static_range(NUM_DOUBLING):
        dp_b = _via_lds(b_Dp, _SH_CC_F, _BF16)
        b_INV = b_INV + _dot_f32(b_INV, dp_b, _AF16, _MMA_F16, C)
        b_Dp = _dot_f32(b_Dp, dp_b, _AF16, _MMA_F16, C)

    w = BC
    for _ in gl.static_range(NUM_MERGE):
        ne_w = o_i_r[:, None] // w != o_i_c[None, :] // w
        if 2 * w < C:
            m_off = (o_i_r[:, None] // (2 * w) == o_i_c[None, :] // (2 * w)) & ne_w
        else:
            m_off = ne_w
        b_off = gl.where(m_off, b_L, 0.0)
        inner = _dot_f32(b_off, _via_lds(b_INV, _SH_CC_F, _BF16), _AF16, _MMA_F16, C)
        b_INV = b_INV + _dot_f32(b_INV, _via_lds(inner, _SH_CC_F, _BF16), _AF16, _MMA_F16, C)
        w = 2 * w

    gl.amd.cdna4.buffer_store(
        gl.convert_layout(b_INV.to(ws_inv_mqk.dtype.element_ty), _BLK_CC),
        ws_inv_mqk,
        cc_off_raw,
        cache=CM_WS,
    )


KW = 8
KW_BIG = 8


@functools.lru_cache(maxsize=None)
def flash_kda_k2_layouts(nw, kw=KW, kw_big=KW_BIG):
    """Layout constexprs of ``flash_kda_k2_ab_fused_gluon`` for ``nw`` warps.

    ``instr_shape[0:2] = [16, 16]`` with ``transposed=False`` makes an MFMA
    accumulator a legal B operand, so the state can be the accumulator of
    ``dot(kr^T, U)`` and the B operand of ``dot(kd, h)`` without a round trip.
    """
    mma = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 4 * kw],
        transposed=False,
        warps_per_cta=[1, nw],
    )
    mma_b = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 4 * kw_big],
        transposed=False,
        warps_per_cta=[1, nw],
    )
    return {
        "MMA": mma,
        "A_OP": gl.DotOperandLayout(0, mma, kw),
        "B_OP": gl.DotOperandLayout(1, mma, kw),
        "MMA_B": mma_b,
        "A_OP_B": gl.DotOperandLayout(0, mma_b, kw_big),
        "B_OP_B": gl.DotOperandLayout(1, mma_b, kw_big),
        "BLK": gl.BlockedLayout([1, 8], [4, 16], [nw, 1], [1, 0]),
        "SH_KR": gl.SwizzledSharedLayout(8, 1, 16, [0, 1]),
    }


@gluon.jit
def _k2_sigmoid(x):
    return gl.extra.libdevice.fast_dividef(1.0, 1.0 + gl.exp(-x.to(gl.float32)))


@gluon.jit
def _recur(
    h,
    kd_a,
    inv_a,
    kr_a,
    gt,
    beta,
    v,
    m_c,
    C: gl.constexpr,
    BW: gl.constexpr,
    MMA: gl.constexpr,
    B_OP: gl.constexpr,
    MMA_B: gl.constexpr,
    B_OP_B: gl.constexpr,
    INV_TY: gl.constexpr,
    HAS_V: gl.constexpr,
):
    h_op = gl.convert_layout(h.to(gl.bfloat16), B_OP_B)
    tmp = gl.convert_layout(
        gl.amd.cdna4.mfma(kd_a, h_op, gl.zeros([C, BW], gl.float32, MMA_B)), MMA
    )
    if HAS_V:
        u = (v - tmp) * beta[:, None]
    else:
        u = (-tmp) * beta[:, None]
    # Tail rows must stay zero: U feeds the state update over all C rows.
    u = gl.where(m_c[:, None], u, 0.0)
    big_u = gl.amd.cdna4.mfma(
        inv_a, gl.convert_layout(u.to(INV_TY), B_OP), gl.zeros([C, BW], gl.float32, MMA)
    )
    h_next = gl.amd.cdna4.mfma(
        kr_a, gl.convert_layout(big_u.to(gl.bfloat16), B_OP), h * gt[:, None]
    )
    return h_next, big_u, h_op


@gluon.jit
def _kr_operand(
    kr_raw,
    K: gl.constexpr,
    C: gl.constexpr,
    SH_KR: gl.constexpr,
    A_OP: gl.constexpr,
):
    return gl.allocate_shared_memory(gl.bfloat16, [K, C], SH_KR, gl.permute(kr_raw, 1, 0)).load(
        A_OP
    )


@gluon.jit
def flash_kda_k2_ab_fused_gluon(
    ws_kd,
    ws_kr,
    ws_gt,
    ws_inv_mqk,
    v_input,
    beta_raw,
    seg_chunk_base,
    seg_nchunks,
    seg_tok_base,
    seg_tok_end,
    h_out_b,
    h_out_a,
    TOTAL_TILES,
    H: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    C: gl.constexpr,
    BW: gl.constexpr,
    MMA: gl.constexpr,
    A_OP: gl.constexpr,
    B_OP: gl.constexpr,
    MMA_B: gl.constexpr,
    A_OP_B: gl.constexpr,
    B_OP_B: gl.constexpr,
    BLK: gl.constexpr,
    SH_KR: gl.constexpr,
    NUM_XCDS: gl.constexpr,
):
    """Both pass-A recurrences (``b_seg`` and ``A_seg``) in one launch, sharing operand loads.

    The two chains are independent, so the scheduler interleaves them, which
    covers the serial dependence each has on its own. Requires ``K == V``.
    """
    # Keeps a (segment, head)'s V blocks on one XCD, so their re-reads of the
    # chunk workspace share an L2. See the Triton K2 for the full reasoning.
    n_w = gl.num_programs(0)
    pid = remap_xcd(gl.program_id(1) * n_w + gl.program_id(0), n_w * gl.num_programs(1), NUM_XCDS)
    i_w = (pid % n_w).to(gl.int64)
    i_sh = (pid // n_w).to(gl.int64)
    i_seg = i_sh // H
    i_h = i_sh % H

    chunk_base = gl.load(seg_chunk_base + i_seg).to(gl.int64)
    n_chunks = gl.load(seg_nchunks + i_seg)
    tok_base = gl.load(seg_tok_base + i_seg).to(gl.int64)
    tok_end = gl.load(seg_tok_end + i_seg).to(gl.int64)

    o_c_ab = gl.arange(0, C, layout=gl.SliceLayout(1, A_OP_B))
    o_k_ab = gl.arange(0, K, layout=gl.SliceLayout(0, A_OP_B))
    o_c_a = gl.arange(0, C, layout=gl.SliceLayout(1, A_OP))
    o_cc_a = gl.arange(0, C, layout=gl.SliceLayout(0, A_OP))
    o_c_s = gl.arange(0, C, layout=gl.SliceLayout(1, BLK))
    o_k_s = gl.arange(0, K, layout=gl.SliceLayout(0, BLK))
    o_c_m = gl.arange(0, C, layout=gl.SliceLayout(1, MMA))
    o_k_m = gl.arange(0, K, layout=gl.SliceLayout(1, MMA))
    o_w_m = i_w * BW + gl.arange(0, BW, layout=gl.SliceLayout(0, MMA))

    kd_off = (o_c_ab[:, None] * K + o_k_ab[None, :]).to(gl.int32)
    inv_off = (o_c_a[:, None] * C + o_cc_a[None, :]).to(gl.int32)
    kr_off = (o_c_s[:, None] * K + o_k_s[None, :]).to(gl.int32)
    gt_off = o_k_m.to(gl.int32)
    beta_off = (i_h + o_c_m * H).to(gl.int32)
    v_off = (i_h * V + o_c_m[:, None] * (H * V) + o_w_m[None, :]).to(gl.int32)

    h_b = gl.zeros([K, BW], gl.float32, MMA)
    h_a = gl.where(o_k_m[:, None] == o_w_m[None, :], 1.0, 0.0)

    ws0 = i_h * TOTAL_TILES + chunk_base
    inv_ty: gl.constexpr = ws_inv_mqk.dtype.element_ty

    for j in range(n_chunks):
        ws_idx = ws0 + j
        ck = ws_idx * (C * K)
        t0 = tok_base + j * C
        tb = t0 * H
        m_c = (t0 + o_c_m) < tok_end

        kd_a = gl.amd.cdna4.buffer_load(ptr=ws_kd + ck, offsets=kd_off)
        inv_a = gl.amd.cdna4.buffer_load(ptr=ws_inv_mqk + ws_idx * (2 * C * C), offsets=inv_off)
        gt = gl.amd.cdna4.buffer_load(ptr=ws_gt + ws_idx * K, offsets=gt_off)
        kr_raw = gl.amd.cdna4.buffer_load(ptr=ws_kr + ck, offsets=kr_off)
        beta = _k2_sigmoid(
            gl.amd.cdna4.buffer_load(ptr=beta_raw + tb, offsets=beta_off, mask=m_c, other=0.0)
        )
        b_v = gl.amd.cdna4.buffer_load(
            ptr=v_input + tb * V, offsets=v_off, mask=m_c[:, None], other=0.0
        ).to(gl.float32)

        kr_a = _kr_operand(kr_raw, K, C, SH_KR, A_OP)
        # Written one after the other so the scheduler has two independent MFMA
        # chains to interleave.
        h_b, _u, _h = _recur(h_b, kd_a, inv_a, kr_a, gt, beta, b_v, m_c, C, BW,
                             MMA, B_OP, MMA_B, B_OP_B, inv_ty, True)  # fmt: skip
        h_a, _u, _h = _recur(h_a, kd_a, inv_a, kr_a, gt, beta, b_v, m_c, C, BW,
                             MMA, B_OP, MMA_B, B_OP_B, inv_ty, False)  # fmt: skip

    s_base = (i_seg * H + i_h) * (K * V)
    s_off = (o_k_m[:, None] * V + o_w_m[None, :]).to(gl.int32)
    gl.amd.cdna4.buffer_store(h_b.to(h_out_b.dtype.element_ty), h_out_b + s_base, s_off)
    gl.amd.cdna4.buffer_store(h_a.to(h_out_a.dtype.element_ty), h_out_a + s_base, s_off)
