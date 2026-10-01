# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information
#
# Adapted from AITER (ROCm/aiter @ 7d2f6a51a,
# aiter/ops/triton/_gluon_kernels/gfx950/chunk_delta_attn), MIT licensed, plus
# the XCD remap of pass A from AITER branch zain/kda/xcd-remap (a3c3c2fdd).

"""Gluon (gfx950) replacements for FlashKDA's prepare kernel, fused pass A and pass C.

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
# One whole C = 32 column per thread, for the gate's cumulative sum.
_BLK_COL: gl.constexpr = gl.BlockedLayout([32, 1], [1, 64], [1, 2], [0, 1])

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

_LOG2E = gl.constexpr(1.4426950408889634)

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
def _sigmoid_log2(z):
    """sigmoid(x) given z = -x * log2(e). exp2 lowers to a bare v_exp_f32, where exp adds
    a denormal-range rescale that the sigmoid does not need."""
    return gl.extra.libdevice.fast_dividef(1.0, 1.0 + gl.exp2(z))


@gluon.jit
def _sigmoid(x):
    return _sigmoid_log2(x.to(gl.float32) * -_LOG2E)


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
def _dot_acc_f32(a, b_op, a_op: gl.constexpr, acc):
    """``acc + a @ b`` with ``acc`` as the MFMA accumulator."""
    return gl.amd.cdna4.mfma(gl.convert_layout(a, a_op), b_op, acc)


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

    # The gate is cumulated over rows, so it is computed with each thread owning
    # a whole column: the scan and the row picks below are then in-thread.
    o_c_col = gl.arange(0, C, layout=gl.SliceLayout(1, _BLK_COL))
    o_k_col = gl.arange(0, K, layout=gl.SliceLayout(0, _BLK_COL))
    g_off = (base * K + o_c_col[:, None].to(gl.int64) * (H * K) + o_k_col[None, :]).to(gl.int32)
    b_g = gl.amd.cdna4.buffer_load(ptr=g_raw, offsets=g_off, cache=CM_LOAD).to(gl.float32)
    # sigmoid(e^A * (g + bias)) with -log2(e) * e^A folded into one scalar.
    rate = _exp(gl.load(A_log + i_h)) * -_LOG2E
    if HAS_BIAS:
        bias = gl.amd.cdna4.buffer_load(ptr=dt_bias, offsets=(i_h * K).to(gl.int32) + o_k_col)
        b_z = b_g * rate + (bias * rate)[None, :]
    else:
        b_z = b_g * rate
    # Tail rows (g read unmasked) get a zero gate, so the cumulative gate is
    # constant from row actual_len - 1 on: the last row is row C - 1, and the
    # pivot row min(C / 2, actual_len - 1) is row C / 2.
    b_gate = gl.where(
        o_c_col[:, None] < actual_len, (lower_bound * _LOG2E) * _sigmoid_log2(b_z), 0.0
    )
    b_gcum_col = gl.associative_scan(b_gate, 0, _add)
    b_g_last_col = gl.sum(gl.where(o_c_col[:, None] == C - 1, b_gcum_col, 0.0), axis=0)
    b_gp_col = gl.sum(gl.where(o_c_col[:, None] == C // 2, b_gcum_col, 0.0), axis=0)
    b_gcum = gl.convert_layout(b_gcum_col, _BLK_WARP_K)
    b_g_last = gl.convert_layout(b_g_last_col, gl.SliceLayout(0, _BLK_WARP_K))
    b_g_total = _exp2(b_g_last_col)
    b_exp_g = _exp2(b_gcum)

    # Tail rows of q and k load as 0 and stay 0 through the norm. Every factor
    # they meet below is finite (the cumulative gate is constant over the
    # tail), so the workspace tiles' tail rows come out zero without a mask.
    b_q = _l2norm(b_q_raw)
    b_k = _l2norm(b_k_raw)

    ws_idx = i_h * TOTAL_TILES + g_tile
    ck_off = (ws_idx * (C * K) + o_c[:, None].to(gl.int64) * K + o_k[None, :]).to(gl.int32)
    gl.amd.cdna4.buffer_store(
        (b_k * b_exp_g).to(ws_kd.dtype.element_ty),
        ws_kd,
        ck_off,
        cache=CM_WS,
    )
    gl.amd.cdna4.buffer_store(
        (b_q * b_exp_g * scale).to(ws_qd.dtype.element_ty),
        ws_qd,
        ck_off,
        cache=CM_WS,
    )
    b_kr_val = (b_k * _exp2(b_g_last[None, :] - b_gcum)).to(gl.bfloat16)
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

    b_gp = gl.convert_layout(b_gp_col, gl.SliceLayout(0, _BLK_WARP_K))
    b_gm = b_gcum - b_gp[None, :]
    b_dec = _exp2(b_gm)
    b_inc = _exp2(-b_gm)
    b_k_piv = (b_k * b_dec).to(gl.bfloat16)
    b_q_piv = (b_q * b_dec * scale).to(gl.bfloat16)
    b_k_inv = (b_k * b_inc).to(gl.bfloat16)

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
    # I + D; D is strictly lower triangular, so its diagonal is free for the 1s.
    b_INV = gl.where(o_i_r[:, None] == o_i_c[None, :], 1.0, b_D)
    b_INV = gl.convert_layout(b_INV, _MMA_F16)
    b_Dp = _dot_f32(b_D, _via_lds(b_D, _SH_CC_F, _BF16), _AF16, _MMA_F16, C)
    for _ in gl.static_range(NUM_DOUBLING):
        dp_b = _via_lds(b_Dp, _SH_CC_F, _BF16)
        b_INV = _dot_acc_f32(b_INV, dp_b, _AF16, b_INV)
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
        b_INV = _dot_acc_f32(b_INV, _via_lds(inner, _SH_CC_F, _BF16), _AF16, b_INV)
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
        # Sources of the global -> LDS copies: 128 (or 32) bits per thread and
        # no lane replication, as buffer_load_to_shared requires.
        "BLK": gl.BlockedLayout([1, 8], [4, 16], [nw, 1], [1, 0]),
        "BLK_CC": gl.BlockedLayout([1, 2], [4, 16], [nw, 1], [1, 0]),
        # gt and beta: one element per thread, loaded to registers.
        "BLK_1D": gl.BlockedLayout([1], [64], [nw], [0]),
        "SH_WS": gl.SwizzledSharedLayout(8, 1, 16, [1, 0]),
        "SH_PLAIN": gl.SwizzledSharedLayout(1, 1, 1, [1, 0]),
        "SH_1D": gl.SwizzledSharedLayout(1, 1, 1, [0]),
    }


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
def _issue_chunk(
    ws_kd,
    ws_kr,
    ws_gt,
    ws_inv_mqk,
    v_input,
    beta_raw,
    ws_qd,
    s_kd,
    s_kr,
    s_inv,
    s_v,
    s_qd,
    s_mqk,
    ws_idx,
    t0,
    tok_end,
    ws_off,
    cc_off,
    v_off,
    o_c_v,
    o_k_g,
    o_c_g,
    beta_off,
    H: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    C: gl.constexpr,
    WITH_OUTPUT: gl.constexpr,
):
    """Start fetching one chunk: its tiles by async copy into LDS, and ``gt`` /
    ``beta`` (returned) into registers, one element per thread.

    ``gt`` and ``beta`` are too small to copy without lane replication, which
    the async copy cannot lower. They are issued first because ``vmcnt``
    retires in order: waiting on them must not also wait on the copies.
    Tail rows of ``beta`` and ``v`` are masked here; ``beta`` masked to 0 is
    harmless because the recurrence zeroes those rows of U anyway.
    ``WITH_OUTPUT`` (pass C) also fetches the output projection's ``qd`` and
    ``Mqk`` tiles into ``s_qd`` / ``s_mqk``.
    """
    tb = t0 * H
    gt = gl.amd.cdna4.buffer_load(ptr=ws_gt + ws_idx * K, offsets=o_k_g)
    beta = gl.amd.cdna4.buffer_load(
        ptr=beta_raw + tb, offsets=beta_off, mask=(t0 + o_c_g) < tok_end, other=0.0
    )
    ck = ws_idx * (C * K)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(s_kd, ws_kd + ck, ws_off)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(s_kr, ws_kr + ck, ws_off)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(s_inv, ws_inv_mqk + ws_idx * (2 * C * C), cc_off)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        s_v, v_input + tb * V, v_off, mask=((t0 + o_c_v) < tok_end)[:, None], other=0.0
    )
    if WITH_OUTPUT:
        gl.amd.cdna4.async_copy.buffer_load_to_shared(s_qd, ws_qd + ck, ws_off)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            s_mqk, ws_inv_mqk + ws_idx * (2 * C * C) + C * C, cc_off
        )
    gl.amd.cdna4.async_copy.commit_group()
    return gt, beta


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
    BLK_CC: gl.constexpr,
    BLK_1D: gl.constexpr,
    SH_WS: gl.constexpr,
    SH_PLAIN: gl.constexpr,
    SH_1D: gl.constexpr,
    NUM_XCDS: gl.constexpr,
):
    """Both pass-A recurrences (``b_seg`` and ``A_seg``) in one launch, sharing operand loads.

    The two chains are independent, so the scheduler interleaves them, which
    covers the serial dependence each has on its own. Requires ``K == V``.

    Chunk j+1's operands are fetched while chunk j computes: its tiles by
    async global -> LDS copy into the other half of a double buffer, so the
    prefetch costs no registers (a register prefetch halves occupancy).
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

    BLK_V: gl.constexpr = gl.BlockedLayout(
        [1, 8], [64 // (BW // 8), BW // 8], [gl.num_warps(), 1], [1, 0]
    )
    o_c_s = gl.arange(0, C, layout=gl.SliceLayout(1, BLK))
    o_k_s = gl.arange(0, K, layout=gl.SliceLayout(0, BLK))
    o_r_cc = gl.arange(0, C, layout=gl.SliceLayout(1, BLK_CC))
    o_c_cc = gl.arange(0, C, layout=gl.SliceLayout(0, BLK_CC))
    o_c_v = gl.arange(0, C, layout=gl.SliceLayout(1, BLK_V))
    o_w_v = i_w * BW + gl.arange(0, BW, layout=gl.SliceLayout(0, BLK_V))
    o_k_g = gl.arange(0, K, layout=BLK_1D)
    o_c_g = gl.arange(0, C, layout=BLK_1D)
    o_c_m = gl.arange(0, C, layout=gl.SliceLayout(1, MMA))
    o_k_m = gl.arange(0, K, layout=gl.SliceLayout(1, MMA))
    o_w_m = i_w * BW + gl.arange(0, BW, layout=gl.SliceLayout(0, MMA))

    ws_off = (o_c_s[:, None] * K + o_k_s[None, :]).to(gl.int32)
    cc_off = (o_r_cc[:, None] * C + o_c_cc[None, :]).to(gl.int32)
    v_off = (i_h * V + o_c_v[:, None] * (H * V) + o_w_v[None, :]).to(gl.int32)
    beta_off = (i_h + o_c_g * H).to(gl.int32)

    inv_ty: gl.constexpr = ws_inv_mqk.dtype.element_ty
    s_kd = gl.allocate_shared_memory(gl.bfloat16, [2, C, K], SH_WS)
    s_kr = gl.allocate_shared_memory(gl.bfloat16, [2, C, K], SH_WS)
    s_inv = gl.allocate_shared_memory(inv_ty, [2, C, C], SH_PLAIN)
    s_v = gl.allocate_shared_memory(v_input.dtype.element_ty, [2, C, BW], SH_PLAIN)
    # gt / beta go register -> LDS once they land, so the per-thread copies
    # of the MMA slice layout are only materialized at their use. beta's
    # sigmoid is applied on the way in, one element per thread rather than
    # once per row copy.
    s_gt = gl.allocate_shared_memory(gl.float32, [2, K], SH_1D)
    s_beta = gl.allocate_shared_memory(gl.float32, [2, C], SH_1D)

    h_b = gl.zeros([K, BW], gl.float32, MMA)
    h_a = gl.where(o_k_m[:, None] == o_w_m[None, :], 1.0, 0.0)

    ws0 = i_h * TOTAL_TILES + chunk_base
    gt_n = gl.zeros([K], gl.float32, BLK_1D)
    beta_n = gl.zeros([C], gl.float32, BLK_1D)
    if n_chunks > 0:
        gt_n, beta_n = _issue_chunk(ws_kd, ws_kr, ws_gt, ws_inv_mqk, v_input, beta_raw, None,
                                    s_kd.index(0), s_kr.index(0), s_inv.index(0), s_v.index(0),
                                    None, None, ws0, tok_base, tok_end, ws_off, cc_off, v_off,
                                    o_c_v, o_k_g, o_c_g, beta_off, H, K, V, C, False)  # fmt: skip

    for j in range(n_chunks):
        cur = j % 2
        # Chunk j has landed in half `cur` (every thread's copies, once past
        # the barrier), and every thread is done reading the other half, which
        # held chunk j-1, before it is refilled.
        gl.amd.cdna4.async_copy.wait_group(0)
        s_gt.index(cur).store(gt_n)
        s_beta.index(cur).store(_sigmoid(beta_n))
        gl.barrier()
        if j + 1 < n_chunks:
            gt_n, beta_n = _issue_chunk(ws_kd, ws_kr, ws_gt, ws_inv_mqk, v_input, beta_raw, None,
                                        s_kd.index(1 - cur), s_kr.index(1 - cur),
                                        s_inv.index(1 - cur), s_v.index(1 - cur), None, None,
                                        ws0 + j + 1, tok_base + (j + 1) * C, tok_end, ws_off,
                                        cc_off, v_off, o_c_v, o_k_g, o_c_g, beta_off, H, K, V, C,
                                        False)  # fmt: skip

        kd_a = gl.amd.cdna4.async_copy.load_shared_relaxed(s_kd.index(cur), A_OP_B)
        # ws_kr is [C, K] in memory; the permuted view is the kr^T A operand.
        kr_a = gl.amd.cdna4.async_copy.load_shared_relaxed(s_kr.index(cur).permute((1, 0)), A_OP)
        inv_a = gl.amd.cdna4.async_copy.load_shared_relaxed(s_inv.index(cur), A_OP)
        b_v = gl.amd.cdna4.async_copy.load_shared_relaxed(s_v.index(cur), MMA).to(gl.float32)
        gt = s_gt.index(cur).load(gl.SliceLayout(1, MMA))
        beta = s_beta.index(cur).load(gl.SliceLayout(1, MMA))
        m_c = (tok_base + j * C + o_c_m) < tok_end
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


@gluon.jit
def flash_kda_k2_c_gluon(
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
    final_state,
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
    BLK_CC: gl.constexpr,
    BLK_1D: gl.constexpr,
    SH_WS: gl.constexpr,
    SH_PLAIN: gl.constexpr,
    SH_1D: gl.constexpr,
    HAS_H_IN: gl.constexpr,
    STORE_FINAL: gl.constexpr,
    STATE_V_FIRST: gl.constexpr,
    CM_OUT: gl.constexpr,
    NUM_XCDS: gl.constexpr,
):
    """Pass C (the output pass) on gfx950: the recurrence of one segment from its incoming state.

    Same contract as ``_flash_kda_segment_kernel`` with ``COMPUTE_OUTPUT``,
    ``HAS_V`` and without ``STORE_H_OUT``. Chunk j+1's tiles are prefetched
    into LDS while chunk j computes, as in the fused pass A. The output is
    stored straight from the MFMA layout: converting it for a wider store
    costs an LDS round trip and enough scratch to cost occupancy at BW=128.
    """
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
    # Read up front, not after the loop: past the output stores the compiler
    # must assume these words may have changed and loads them per lane, which
    # leaves the final-state store a waterfall loop over a non-uniform pointer.
    # When STORE_FINAL is false final_state is null and these must not be read.
    if STORE_FINAL:
        is_last = gl.load(seg_is_last + i_seg)
        i_n = gl.load(seg_seq + i_seg).to(gl.int64)

    BLK_V: gl.constexpr = gl.BlockedLayout(
        [1, 8], [64 // (BW // 8), BW // 8], [gl.num_warps(), 1], [1, 0]
    )
    o_c_s = gl.arange(0, C, layout=gl.SliceLayout(1, BLK))
    o_k_s = gl.arange(0, K, layout=gl.SliceLayout(0, BLK))
    o_r_cc = gl.arange(0, C, layout=gl.SliceLayout(1, BLK_CC))
    o_c_cc = gl.arange(0, C, layout=gl.SliceLayout(0, BLK_CC))
    o_c_v = gl.arange(0, C, layout=gl.SliceLayout(1, BLK_V))
    o_w_v = i_w * BW + gl.arange(0, BW, layout=gl.SliceLayout(0, BLK_V))
    o_k_g = gl.arange(0, K, layout=BLK_1D)
    o_c_g = gl.arange(0, C, layout=BLK_1D)
    o_c_m = gl.arange(0, C, layout=gl.SliceLayout(1, MMA))
    o_k_m = gl.arange(0, K, layout=gl.SliceLayout(1, MMA))
    o_w_m = i_w * BW + gl.arange(0, BW, layout=gl.SliceLayout(0, MMA))

    ws_off = (o_c_s[:, None] * K + o_k_s[None, :]).to(gl.int32)
    cc_off = (o_r_cc[:, None] * C + o_c_cc[None, :]).to(gl.int32)
    v_off = (i_h * V + o_c_v[:, None] * (H * V) + o_w_v[None, :]).to(gl.int32)
    o_off = (i_h * V + o_c_m[:, None] * (H * V) + o_w_m[None, :]).to(gl.int32)
    beta_off = (i_h + o_c_g * H).to(gl.int32)
    s_off = (o_k_m[:, None] * V + o_w_m[None, :]).to(gl.int32)

    inv_ty: gl.constexpr = ws_inv_mqk.dtype.element_ty
    out_ty: gl.constexpr = out.dtype.element_ty
    s_kd = gl.allocate_shared_memory(gl.bfloat16, [2, C, K], SH_WS)
    s_qd = gl.allocate_shared_memory(gl.bfloat16, [2, C, K], SH_WS)
    s_kr = gl.allocate_shared_memory(gl.bfloat16, [2, C, K], SH_WS)
    s_inv = gl.allocate_shared_memory(inv_ty, [2, C, C], SH_PLAIN)
    s_mqk = gl.allocate_shared_memory(inv_ty, [2, C, C], SH_PLAIN)
    s_v = gl.allocate_shared_memory(v_input.dtype.element_ty, [2, C, BW], SH_PLAIN)
    s_gt = gl.allocate_shared_memory(gl.float32, [2, K], SH_1D)
    s_beta = gl.allocate_shared_memory(gl.float32, [2, C], SH_1D)

    if HAS_H_IN:
        h = gl.amd.cdna4.buffer_load(ptr=h_in + (i_seg * H + i_h) * (K * V), offsets=s_off)
        h = h.to(gl.float32)
    else:
        h = gl.zeros([K, BW], gl.float32, MMA)

    ws0 = i_h * TOTAL_TILES + chunk_base
    gt_n = gl.zeros([K], gl.float32, BLK_1D)
    beta_n = gl.zeros([C], gl.float32, BLK_1D)
    if n_chunks > 0:
        gt_n, beta_n = _issue_chunk(ws_kd, ws_kr, ws_gt, ws_inv_mqk, v_input, beta_raw, ws_qd,
                                    s_kd.index(0), s_kr.index(0), s_inv.index(0), s_v.index(0),
                                    s_qd.index(0), s_mqk.index(0), ws0, tok_base, tok_end, ws_off,
                                    cc_off, v_off, o_c_v, o_k_g, o_c_g, beta_off, H, K, V, C,
                                    True)  # fmt: skip

    for j in range(n_chunks):
        cur = j % 2
        # As in pass A: chunk j is in half `cur`, and half 1 - cur is free.
        gl.amd.cdna4.async_copy.wait_group(0)
        s_gt.index(cur).store(gt_n)
        s_beta.index(cur).store(_sigmoid(beta_n))
        gl.barrier()
        if j + 1 < n_chunks:
            gt_n, beta_n = _issue_chunk(ws_kd, ws_kr, ws_gt, ws_inv_mqk, v_input, beta_raw, ws_qd,
                                        s_kd.index(1 - cur), s_kr.index(1 - cur),
                                        s_inv.index(1 - cur), s_v.index(1 - cur),
                                        s_qd.index(1 - cur), s_mqk.index(1 - cur), ws0 + j + 1,
                                        tok_base + (j + 1) * C, tok_end, ws_off, cc_off, v_off,
                                        o_c_v, o_k_g, o_c_g, beta_off, H, K, V, C,
                                        True)  # fmt: skip

        t0 = tok_base + j * C
        kd_a = gl.amd.cdna4.async_copy.load_shared_relaxed(s_kd.index(cur), A_OP_B)
        inv_a = gl.amd.cdna4.async_copy.load_shared_relaxed(s_inv.index(cur), A_OP)
        b_v = gl.amd.cdna4.async_copy.load_shared_relaxed(s_v.index(cur), MMA).to(gl.float32)
        gt = s_gt.index(cur).load(gl.SliceLayout(1, MMA))
        beta = s_beta.index(cur).load(gl.SliceLayout(1, MMA))
        m_c = (t0 + o_c_m) < tok_end

        # _recur's steps, split around the output projection: updating the
        # state after it keeps the next state and kr out of registers meanwhile
        # (1.1-1.2x on this kernel).
        h_op = gl.convert_layout(h.to(gl.bfloat16), B_OP_B)
        tmp = gl.convert_layout(
            gl.amd.cdna4.mfma(kd_a, h_op, gl.zeros([C, BW], gl.float32, MMA_B)), MMA
        )
        u = gl.where(m_c[:, None], (b_v - tmp) * beta[:, None], 0.0)
        big_u = gl.amd.cdna4.mfma(
            inv_a, gl.convert_layout(u.to(inv_ty), B_OP), gl.zeros([C, BW], gl.float32, MMA)
        )

        # o = qd @ h + Mqk @ U, h being the state entering the chunk.
        qd_a = gl.amd.cdna4.async_copy.load_shared_relaxed(s_qd.index(cur), A_OP_B)
        mqk_a = gl.amd.cdna4.async_copy.load_shared_relaxed(s_mqk.index(cur), A_OP)
        o = gl.convert_layout(
            gl.amd.cdna4.mfma(qd_a, h_op, gl.zeros([C, BW], gl.float32, MMA_B)), MMA
        )
        o = gl.amd.cdna4.mfma(mqk_a, gl.convert_layout(big_u.to(inv_ty), B_OP), o)
        gl.amd.cdna4.buffer_store(o.to(out_ty), out + t0 * (H * V), o_off,
                                  mask=m_c[:, None], cache=CM_OUT)  # fmt: skip

        kr_a = gl.amd.cdna4.async_copy.load_shared_relaxed(s_kr.index(cur).permute((1, 0)), A_OP)
        h = gl.amd.cdna4.mfma(kr_a, gl.convert_layout(big_u.to(gl.bfloat16), B_OP), h * gt[:, None])

    if STORE_FINAL:  # noqa: SIM102
        if is_last == 1:
            if STATE_V_FIRST:
                f_off = (o_w_m[None, :] * K + o_k_m[:, None]).to(gl.int32)
            else:
                f_off = s_off
            gl.amd.cdna4.buffer_store(
                h.to(final_state.dtype.element_ty), final_state + (i_n * H + i_h) * (K * V), f_off
            )
