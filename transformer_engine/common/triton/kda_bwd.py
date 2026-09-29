# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information
#
# Adapted from flash-linear-attention (fla-org/flash-linear-attention @ 954438d,
# fla/ops/kda/{chunk_bwd,chunk_intra,gate}.py, fla/ops/common/{chunk_delta_h,gate}.py,
# fla/ops/utils/cumsum.py, fla/modules/l2norm.py): Copyright (c) 2023-2026,
# Songlin Yang, Yu Zhang, Zhiyuan Li. MIT licensed.

# pylint: disable=possibly-used-before-assignment

"""Framework-agnostic Triton kernels for the Kimi Delta Attention (KDA) backward pass.

The backward works on the default pipeline's intermediates (``g`` cumsum in
log2 space, ``Aqk``, ``Akk``, ``w``/``u``/``kg``, ``h``, ``v_new``; see
``kda.py``), which the caller recomputes from the saved inputs. Stages:

1. ``dAqk = do @ v_new^T * scale``, ``dv = Aqk^T @ do`` (``_kda_bwd_dav_kernel``).
2. The reverse-time state gradient ``dh`` per chunk, ``dh0``, and ``dv``
   through the state (``_kda_bwd_dhu_kernel``). With too few states to fill
   the GPU it shares a launch with the recomputed forward recurrence
   (``_kda_fwd_h_bwd_dhu_kernel``), after the ``dv`` half of stage 1 and
   before its ``dAqk`` half.
3. ``dq``/``dk``/``dg``/``dbeta``/``dv`` through the WY representation and the
   inter-chunk terms, plus ``dAkk`` (``_kda_bwd_wy_dqkg_kernel``).
4. The intra-chunk terms of ``dq``/``dk``/``dg``/``dbeta`` from ``dAqk`` and
   ``dAkk`` (``_kda_bwd_intra_kernel``).
5. A reverse chunk-local cumsum turns the gradient w.r.t. the cumulative gate
   into one w.r.t. the per-token gate, and the same kernel undoes the gate
   activation (``_kda_bwd_gate_cumsum_kernel``); l2norm and beta sigmoid are
   then undone elementwise.

Kernel bodies follow fla, with the same TE-side conventions as ``kda.py``: no
heuristics or autotuning, tensor parameters inputs first and outputs last, and
an early exit on padded varlen chunks. The state gradient keeps ``dh`` in the
``[K, V]`` layout of the forward's ``h``; ``TRANSPOSE_STATE`` only describes
``dht`` / ``dh0``. ``_kda_bwd_dhu_kernel`` reads ``qg = q * exp2(g)`` from
the recompute (``_kda_recompute_w_u_kernel``) rather than gating ``q`` on its
serial chunk chain.
"""

import triton
import triton.language as tl

from transformer_engine.common.triton.kda import _kda_fwd_h, exp, exp2, remap_xcd, softplus


@triton.jit
def _kda_bwd_dav_kernel(
    v,
    A,
    do,
    cu_seqlens,
    chunk_indices,
    dA,
    dv,
    scale,
    T,
    HV: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    COMPUTE_DA: tl.constexpr,
    COMPUTE_DV: tl.constexpr,
):
    """dA = tril(do @ v^T) * scale and dv = tril(A)^T @ do, per chunk (``v`` is ``v_new``).

    ``COMPUTE_DA`` / ``COMPUTE_DV`` select the outputs, so ``dv`` (which needs no
    ``v_new``) can be produced ahead of ``_kda_fwd_h_bwd_dhu_kernel``.
    """
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_hv = i_bh // HV, i_bh % HV
    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        T = eos - bos
        if i_t * BT >= T:
            return
    else:
        bos, eos = i_b * T, i_b * T + T

    do += (bos * HV + i_hv) * V
    if COMPUTE_DA:
        v += (bos * HV + i_hv) * V
        dA += (bos * HV + i_hv) * BT
    if COMPUTE_DV:
        dv += (bos * HV + i_hv) * V

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    o_A = tl.arange(0, BT)
    if COMPUTE_DV:
        m_AT = (o_A[:, None] < BT) & m_t[None, :]
        # b_A[s, t] = A[t, s]
        p_A = A + (bos * HV + i_hv) * BT + o_A[:, None] + o_t[None, :] * (HV * BT)
        b_A = tl.load(p_A, mask=m_AT, other=0.0)
        m_A = (o_t[:, None] <= o_t[None, :]) & (m_t[:, None] & m_t)
        b_A = tl.where(m_A, b_A, 0).to(do.dtype.element_ty)

    b_dA = tl.zeros([BT, BT], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = o_v < V
        m_vT = m_v[:, None] & m_t[None, :]
        m_tv = m_t[:, None] & m_v[None, :]
        p_do = do + o_t[:, None] * (HV * V) + o_v[None, :]
        b_do = tl.load(p_do, mask=m_tv, other=0.0)
        if COMPUTE_DA:
            p_v = v + o_v[:, None] + o_t[None, :] * (HV * V)
            b_v = tl.load(p_v, mask=m_vT, other=0.0)
            b_dA = tl.dot(b_do, b_v, b_dA)
        if COMPUTE_DV:
            p_dv = dv + o_t[:, None] * (HV * V) + o_v[None, :]
            b_dv = tl.dot(b_A.to(b_do.dtype), b_do)
            tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), mask=m_tv)

    if COMPUTE_DA:
        m_dA = m_t[:, None] & (o_A[None, :] < BT)
        p_dA = dA + o_t[:, None] * (HV * BT) + o_A[None, :]
        b_dA = tl.where(o_t[:, None] >= o_t, b_dA * scale, 0.0)
        tl.store(p_dA, b_dA.to(p_dA.dtype.element_ty), mask=m_dA)


@triton.jit
def _kda_bwd_dhu(
    i_v,
    i_nh,
    qg,
    g,
    k,
    w,
    dht,
    do,
    dv,
    cu_seqlens,
    chunk_offsets,
    dh,
    dh0,
    dv2,
    scale,
    T,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    TRANSPOSE_STATE: tl.constexpr,
):
    """Body of ``_kda_bwd_dhu_kernel`` for program ``(i_v, i_nh)``."""
    i_nh = i_nh.to(tl.int64)
    i_n, i_hv = i_nh // HV, i_nh % HV
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int64)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    b_dh1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        b_dh2 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 128:
        b_dh3 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 192:
        b_dh4 = tl.zeros([64, BV], dtype=tl.float32)

    qg += (bos * HV + i_hv) * K
    g += (bos * HV + i_hv) * K
    k += (bos * HV + i_hv) * K
    w += (bos * HV + i_hv) * K
    do += (bos * HV + i_hv) * V
    dv += (bos * HV + i_hv) * V
    dv2 += (bos * HV + i_hv) * V
    dh += (boh * HV + i_hv) * K * V
    if USE_INITIAL_STATE:
        dh0 += i_nh * K * V
    if USE_FINAL_STATE_GRADIENT:
        dht += i_nh * K * V

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

    if USE_FINAL_STATE_GRADIENT:
        if TRANSPOSE_STATE:
            p_dht = dht + o_k1[:, None] + o_v[None, :] * K
        else:
            p_dht = dht + o_k1[:, None] * V + o_v[None, :]
        b_dh1 += tl.load(p_dht, mask=m_h1, other=0.0).to(tl.float32)
        if K > 64:
            if TRANSPOSE_STATE:
                p_dht = dht + o_k2[:, None] + o_v[None, :] * K
            else:
                p_dht = dht + o_k2[:, None] * V + o_v[None, :]
            b_dh2 += tl.load(p_dht, mask=m_h2, other=0.0).to(tl.float32)
        if K > 128:
            if TRANSPOSE_STATE:
                p_dht = dht + o_k3[:, None] + o_v[None, :] * K
            else:
                p_dht = dht + o_k3[:, None] * V + o_v[None, :]
            b_dh3 += tl.load(p_dht, mask=m_h3, other=0.0).to(tl.float32)
        if K > 192:
            if TRANSPOSE_STATE:
                p_dht = dht + o_k4[:, None] + o_v[None, :] * K
            else:
                p_dht = dht + o_k4[:, None] * V + o_v[None, :]
            b_dh4 += tl.load(p_dht, mask=m_h4, other=0.0).to(tl.float32)

    for i_t in range(NT - 1, -1, -1):
        i_t64 = i_t.to(tl.int64)
        o_t = i_t64 * BT + tl.arange(0, BT)
        m_t = o_t < T
        m_tv = m_t[:, None] & m_v[None, :]

        dh_t = dh + i_t64 * HV * K * V
        tl.store(
            dh_t + o_k1[:, None] * V + o_v[None, :],
            b_dh1.to(dh.dtype.element_ty),
            mask=m_h1,
        )
        if K > 64:
            tl.store(
                dh_t + o_k2[:, None] * V + o_v[None, :],
                b_dh2.to(dh.dtype.element_ty),
                mask=m_h2,
            )
        if K > 128:
            tl.store(
                dh_t + o_k3[:, None] * V + o_v[None, :],
                b_dh3.to(dh.dtype.element_ty),
                mask=m_h3,
            )
        if K > 192:
            tl.store(
                dh_t + o_k4[:, None] * V + o_v[None, :],
                b_dh4.to(dh.dtype.element_ty),
                mask=m_h4,
            )

        last_idx = min((i_t64 + 1) * BT, T) - 1
        b_do = tl.load(do + o_t[:, None] * (HV * V) + o_v[None, :], mask=m_tv, other=0.0)

        # dv2 = dv + kg @ dh
        b_k = tl.load(
            k + o_t[:, None] * (HV * K) + o_k1[None, :],
            mask=m_t[:, None] & m_k1[None, :],
            other=0.0,
        )
        b_gl1 = tl.load(g + last_idx * HV * K + o_k1, mask=m_k1, other=0.0).to(tl.float32)
        b_dv = tl.dot(b_k, b_dh1.to(b_k.dtype))
        if K > 64:
            b_k = tl.load(
                k + o_t[:, None] * (HV * K) + o_k2[None, :],
                mask=m_t[:, None] & m_k2[None, :],
                other=0.0,
            )
            b_gl2 = tl.load(g + last_idx * HV * K + o_k2, mask=m_k2, other=0.0).to(tl.float32)
            b_dv = tl.dot(b_k, b_dh2.to(b_k.dtype), b_dv)
        if K > 128:
            b_k = tl.load(
                k + o_t[:, None] * (HV * K) + o_k3[None, :],
                mask=m_t[:, None] & m_k3[None, :],
                other=0.0,
            )
            b_gl3 = tl.load(g + last_idx * HV * K + o_k3, mask=m_k3, other=0.0).to(tl.float32)
            b_dv = tl.dot(b_k, b_dh3.to(b_k.dtype), b_dv)
        if K > 192:
            b_k = tl.load(
                k + o_t[:, None] * (HV * K) + o_k4[None, :],
                mask=m_t[:, None] & m_k4[None, :],
                other=0.0,
            )
            b_gl4 = tl.load(g + last_idx * HV * K + o_k4, mask=m_k4, other=0.0).to(tl.float32)
            b_dv = tl.dot(b_k, b_dh4.to(b_k.dtype), b_dv)
        p_dv = dv + o_t[:, None] * (HV * V) + o_v[None, :]
        b_dv += tl.load(p_dv, mask=m_tv, other=0.0)
        p_dv2 = dv2 + o_t[:, None] * (HV * V) + o_v[None, :]
        tl.store(p_dv2, b_dv.to(p_dv2.dtype.element_ty), mask=m_tv)

        # dh = dh * exp2(g_last) + qg^T @ do * scale - w^T @ dv2
        m_kt = m_k1[:, None] & m_t[None, :]
        b_qg = tl.load(qg + o_k1[:, None] + o_t[None, :] * (HV * K), mask=m_kt, other=0.0)
        b_w = tl.load(w + o_k1[:, None] + o_t[None, :] * (HV * K), mask=m_kt, other=0.0)
        b_dh1 *= exp2(b_gl1)[:, None]
        b_dh1 += tl.dot(b_qg, b_do.to(b_qg.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))
        if K > 64:
            m_kt = m_k2[:, None] & m_t[None, :]
            b_qg = tl.load(qg + o_k2[:, None] + o_t[None, :] * (HV * K), mask=m_kt, other=0.0)
            b_w = tl.load(w + o_k2[:, None] + o_t[None, :] * (HV * K), mask=m_kt, other=0.0)
            b_dh2 *= exp2(b_gl2)[:, None]
            b_dh2 += tl.dot(b_qg, b_do.to(b_qg.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))
        if K > 128:
            m_kt = m_k3[:, None] & m_t[None, :]
            b_qg = tl.load(qg + o_k3[:, None] + o_t[None, :] * (HV * K), mask=m_kt, other=0.0)
            b_w = tl.load(w + o_k3[:, None] + o_t[None, :] * (HV * K), mask=m_kt, other=0.0)
            b_dh3 *= exp2(b_gl3)[:, None]
            b_dh3 += tl.dot(b_qg, b_do.to(b_qg.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))
        if K > 192:
            m_kt = m_k4[:, None] & m_t[None, :]
            b_qg = tl.load(qg + o_k4[:, None] + o_t[None, :] * (HV * K), mask=m_kt, other=0.0)
            b_w = tl.load(w + o_k4[:, None] + o_t[None, :] * (HV * K), mask=m_kt, other=0.0)
            b_dh4 *= exp2(b_gl4)[:, None]
            b_dh4 += tl.dot(b_qg, b_do.to(b_qg.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))

    if USE_INITIAL_STATE:
        if TRANSPOSE_STATE:
            p_dh0 = dh0 + o_k1[:, None] + o_v[None, :] * K
        else:
            p_dh0 = dh0 + o_k1[:, None] * V + o_v[None, :]
        tl.store(p_dh0, b_dh1.to(p_dh0.dtype.element_ty), mask=m_h1)
        if K > 64:
            if TRANSPOSE_STATE:
                p_dh0 = dh0 + o_k2[:, None] + o_v[None, :] * K
            else:
                p_dh0 = dh0 + o_k2[:, None] * V + o_v[None, :]
            tl.store(p_dh0, b_dh2.to(p_dh0.dtype.element_ty), mask=m_h2)
        if K > 128:
            if TRANSPOSE_STATE:
                p_dh0 = dh0 + o_k3[:, None] + o_v[None, :] * K
            else:
                p_dh0 = dh0 + o_k3[:, None] * V + o_v[None, :]
            tl.store(p_dh0, b_dh3.to(p_dh0.dtype.element_ty), mask=m_h3)
        if K > 192:
            if TRANSPOSE_STATE:
                p_dh0 = dh0 + o_k4[:, None] + o_v[None, :] * K
            else:
                p_dh0 = dh0 + o_k4[:, None] * V + o_v[None, :]
            tl.store(p_dh0, b_dh4.to(p_dh0.dtype.element_ty), mask=m_h4)


@triton.jit
def _kda_bwd_dhu_kernel(
    qg,
    g,
    k,
    w,
    dht,
    do,
    dv,
    cu_seqlens,
    chunk_offsets,
    dh,
    dh0,
    dv2,
    scale,
    T,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    TRANSPOSE_STATE: tl.constexpr,
    NUM_XCDS: tl.constexpr,
):
    """Reverse-time recurrence of the per-chunk state gradient.

    ``dh[t]`` is the gradient w.r.t. ``h[t]`` (the state entering chunk ``t``);
    ``dv2 = dv + kg @ dh[t]`` adds the path through the state to ``v_new``'s
    gradient. ``qg`` (``q * exp2(g)``) and ``k`` (``kg``) are per value head and
    ``g`` is the chunk-local log2 cumsum.
    """
    # The V blocks of one state share their qg/kg/w/g loads: keep them on one XCD.
    NV = tl.num_programs(0)
    pid = remap_xcd(tl.program_id(1) * NV + tl.program_id(0), NV * tl.num_programs(1), NUM_XCDS)
    _kda_bwd_dhu(
        pid % NV,
        pid // NV,
        qg,
        g,
        k,
        w,
        dht,
        do,
        dv,
        cu_seqlens,
        chunk_offsets,
        dh,
        dh0,
        dv2,
        scale,
        T,
        HV,
        K,
        V,
        BT,
        BV,
        USE_INITIAL_STATE,
        USE_FINAL_STATE_GRADIENT,
        IS_VARLEN,
        TRANSPOSE_STATE,
    )


@triton.jit
def _kda_fwd_h_bwd_dhu_kernel(
    qg,
    g,
    k,
    v,
    w,
    h0,
    dht,
    do,
    dv,
    cu_seqlens,
    chunk_offsets,
    h,
    v_new,
    dh,
    dh0,
    dv2,
    scale,
    T,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    TRANSPOSE_STATE: tl.constexpr,
    NUM_XCDS: tl.constexpr,
):
    """``_kda_fwd_h_kernel`` (recompute) and ``_kda_bwd_dhu_kernel`` in one launch.

    The two recurrences are independent (``dhu`` needs ``dv = Aqk^T @ do`` but
    not ``v_new``) and each is a serial chain over chunks that leaves most CUs
    idle when there are few states, so they share the grid: program ``i_v <
    cdiv(V, BV)`` runs ``fwd_h`` on V block ``i_v``, the rest run ``dhu``.
    ``qg``, ``k`` (``kg``), ``v`` (``u``) and ``g`` (the log2 gate cumsum) are as
    for the two kernels; the final state is not stored.
    """
    NV: tl.constexpr = (V + BV - 1) // BV
    # A state's fwd_h (and dhu) V blocks share their loads: keep them on one XCD.
    n_v = tl.num_programs(0)
    pid = remap_xcd(tl.program_id(1) * n_v + tl.program_id(0), n_v * tl.num_programs(1), NUM_XCDS)
    i_v, i_nh = pid % n_v, pid // n_v
    if i_v < NV:
        _kda_fwd_h(
            i_v,
            i_nh,
            k,
            v,
            w,
            g,
            h0,
            cu_seqlens,
            chunk_offsets,
            h,
            v_new,
            None,
            T,
            HV,
            K,
            V,
            BT,
            BV,
            USE_INITIAL_STATE,
            False,
            IS_VARLEN,
            TRANSPOSE_STATE,
        )
    else:
        _kda_bwd_dhu(
            i_v - NV,
            i_nh,
            qg,
            g,
            k,
            w,
            dht,
            do,
            dv,
            cu_seqlens,
            chunk_offsets,
            dh,
            dh0,
            dv2,
            scale,
            T,
            HV,
            K,
            V,
            BT,
            BV,
            USE_INITIAL_STATE,
            USE_FINAL_STATE_GRADIENT,
            IS_VARLEN,
            TRANSPOSE_STATE,
        )


@triton.jit
def _kda_bwd_wy_dqkg_kernel(
    q,
    k,
    v,
    v_new,
    g,
    beta,
    A,
    h,
    do,
    dh,
    dv,
    cu_seqlens,
    chunk_indices,
    dq,
    dk,
    dv2,
    dg,
    db,
    dA,
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
    """Inter-chunk dq/dk/dg, the WY-representation gradients dv/dbeta/dg, and dAkk.

    ``A`` is ``Akk`` (the inverted WY matrix); ``dv`` is ``v_new``'s gradient
    from ``_kda_bwd_dhu_kernel``. ``dq``/``dk``/``dg`` are per value head, fp32.
    ``dA`` is the gradient w.r.t. the strictly lower ``beta * k k^T`` block.
    """
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1)
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        i_tg = i_t.to(tl.int64)
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        T = (eos - bos).to(tl.int32)
        if i_t * BT >= T:
            return
    else:
        NT = tl.cdiv(T, BT)
        i_tg = (i_b * NT + i_t).to(tl.int64)
        bos, eos = (i_b * T).to(tl.int64), (i_b * T + T).to(tl.int64)

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_last = o_t == min(T, i_t * BT + BT) - 1

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    v += (bos * HV + i_hv) * V
    v_new += (bos * HV + i_hv) * V
    g += (bos * HV + i_hv) * K
    beta += bos * HV + i_hv
    A += (bos * HV + i_hv) * BT
    h += (i_tg * HV + i_hv) * K * V
    do += (bos * HV + i_hv) * V
    dh += (i_tg * HV + i_hv) * K * V
    dq += (bos * HV + i_hv) * K
    dk += (bos * HV + i_hv) * K
    dv += (bos * HV + i_hv) * V
    dv2 += (bos * HV + i_hv) * V
    dg += (bos * HV + i_hv) * K
    db += bos * HV + i_hv
    dA += (bos * HV + i_hv) * BT

    p_beta = beta + o_t * HV
    b_beta = tl.load(p_beta, mask=m_t, other=0.0)

    o_A = tl.arange(0, BT)
    m_AT = (o_A[:, None] < BT) & m_t[None, :]
    # b_A[s, t] = A[t, s]
    p_A = A + o_A[:, None] + o_t[None, :] * (HV * BT)
    b_A = tl.load(p_A, mask=m_AT, other=0.0)

    b_dA = tl.zeros([BT, BT], dtype=tl.float32)
    b_db = tl.zeros([BT], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        m_tk = m_t[:, None] & m_k[None, :]

        p_k = k + o_t[:, None] * (H * K) + o_k[None, :]
        p_g = g + o_t[:, None] * (HV * K) + o_k[None, :]
        b_k = tl.load(p_k, mask=m_tk, other=0.0)
        b_g = tl.load(p_g, mask=m_tk, other=0.0).to(tl.float32)

        p_gn = g + (min(T, i_t * BT + BT) - 1).to(tl.int64) * HV * K + o_k
        b_gn = tl.load(p_gn, mask=m_k, other=0).to(tl.float32)

        b_dq = tl.zeros([BT, BK], dtype=tl.float32)
        b_dk = tl.zeros([BT, BK], dtype=tl.float32)
        b_dw = tl.zeros([BT, BK], dtype=tl.float32)
        b_dgk = tl.zeros([BK], dtype=tl.float32)

        for i_v in range(tl.cdiv(V, BV)):
            o_v = i_v * BV + tl.arange(0, BV)
            m_tv = m_t[:, None] & (o_v[None, :] < V)
            m_h = (o_v[:, None] < V) & m_k[None, :]
            p_v_new = v_new + o_t[:, None] * (HV * V) + o_v[None, :]
            p_do = do + o_t[:, None] * (HV * V) + o_v[None, :]
            # [BV, BK] views of the [K, V] states.
            p_h = h + o_v[:, None] + o_k[None, :] * V
            p_dh = dh + o_v[:, None] + o_k[None, :] * V
            p_dv = dv + o_t[:, None] * (HV * V) + o_v[None, :]
            b_v_new = tl.load(p_v_new, mask=m_tv, other=0.0)
            b_do = tl.load(p_do, mask=m_tv, other=0.0)
            b_h = tl.load(p_h, mask=m_h, other=0.0)
            b_dh = tl.load(p_dh, mask=m_h, other=0.0)
            b_dv = tl.load(p_dv, mask=m_tv, other=0.0)

            b_dgk += tl.sum(b_h * b_dh, axis=0)
            b_dq = tl.dot(b_do, b_h.to(b_do.dtype), b_dq)
            b_dk = tl.dot(b_v_new, b_dh.to(b_v_new.dtype), b_dk)
            b_dw = tl.dot(b_dv.to(b_v_new.dtype), b_h.to(b_v_new.dtype), b_dw)
            tl.debug_barrier()  # fla: required for correctness
            if i_k == 0:
                p_v = v + o_t[:, None] * (HV * V) + o_v[None, :]
                p_dv2 = dv2 + o_t[:, None] * (HV * V) + o_v[None, :]
                b_v = tl.load(p_v, mask=m_tv, other=0.0)
                b_dA = tl.dot(b_dv, tl.trans(b_v), b_dA)
                b_dvb = tl.dot(b_A, b_dv)
                b_dv2 = b_dvb * b_beta[:, None]
                b_db += tl.sum(b_dvb * b_v, 1)
                tl.store(p_dv2, b_dv2.to(p_dv2.dtype.element_ty), mask=m_tv)

        b_gk_exp = exp2(b_g)
        b_gb = b_gk_exp * b_beta[:, None]
        b_dgk *= exp2(b_gn)
        b_dq = b_dq * b_gk_exp * scale
        b_dk = b_dk * tl.where(m_t[:, None], exp2(b_gn[None, :] - b_g), 0)

        b_kg = b_k * b_gk_exp

        b_dw = -b_dw.to(b_A.dtype)
        b_dA = tl.dot(b_dw, tl.trans(b_kg.to(b_A.dtype)), b_dA)

        b_dkgb = tl.dot(b_A, b_dw)
        b_db += tl.sum(b_dkgb * b_kg, 1)

        p_q = q + o_t[:, None] * (H * K) + o_k[None, :]
        b_q = tl.load(p_q, mask=m_tk, other=0.0)
        b_kdk = b_k * b_dk
        b_dgk += tl.sum(b_kdk, axis=0)
        b_dg = b_q * b_dq - b_kdk + m_last[:, None] * b_dgk + b_kg * b_dkgb * b_beta[:, None]
        b_dk = b_dk + b_dkgb * b_gb

        p_dq = dq + o_t[:, None] * (HV * K) + o_k[None, :]
        p_dk = dk + o_t[:, None] * (HV * K) + o_k[None, :]
        p_dg = dg + o_t[:, None] * (HV * K) + o_k[None, :]
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), mask=m_tk)
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=m_tk)
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), mask=m_tk)

    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_dA = tl.where(m_A, b_dA * b_beta[None, :], 0)
    b_dA = tl.dot(b_dA.to(b_A.dtype), b_A)
    b_dA = tl.dot(b_A, b_dA.to(b_A.dtype))
    b_dA = tl.where(m_A, -b_dA, 0)

    m_dA = m_t[:, None] & (o_A[None, :] < BT)
    p_dA = dA + o_t[:, None] * (HV * BT) + o_A[None, :]
    p_db = db + o_t * HV
    tl.store(p_dA, b_dA.to(p_dA.dtype.element_ty), mask=m_dA)
    tl.store(p_db, b_db.to(p_db.dtype.element_ty), mask=m_t)


@triton.jit
def _kda_bwd_intra_kernel(
    q,
    k,
    g,
    beta,
    dAqk,
    dAkk,
    dq,
    dk,
    dg,
    cu_seqlens,
    chunk_indices,
    dq2,
    dk2,
    dg2,
    db,
    B,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    SAFE_GATE: tl.constexpr,
    NUM_XCDS: tl.constexpr,
):
    """Intra-chunk dq/dk/dg/dbeta from dAqk and dAkk, one ``BC`` sub-chunk and ``BK`` slice each.

    ``dq2 = dq + ...`` etc. accumulate onto the inter-chunk terms; ``db`` is
    ``[NK, B*T, HV]`` partials, one per ``BK`` slice.
    """
    # The NK * NC programs of a chunk re-read each other's k/g/q tiles; launched
    # round-robin they would sit on different XCDs (L2s), so keep them on one.
    n_kc, n_t = tl.num_programs(0), tl.num_programs(1)
    pid = remap_xcd(
        (tl.program_id(2) * n_t + tl.program_id(1)) * n_kc + tl.program_id(0),
        n_kc * n_t * tl.num_programs(2),
        NUM_XCDS,
    )
    i_kc = pid % n_kc
    i_t, i_bh = (pid // n_kc % n_t).to(tl.int64), (pid // (n_kc * n_t)).to(tl.int64)
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)
    i_k, i_i = i_kc // NC, i_kc % NC

    n_all = B * T
    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
    else:
        bos, eos = i_b * T, i_b * T + T
    T = eos - bos

    i_ti = i_t * BT + i_i * BC
    if i_ti >= T:
        return

    o_k = i_k * BK + tl.arange(0, BK)
    m_k = o_k < K

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    g += (bos * HV + i_hv) * K
    beta += bos * HV + i_hv

    dAqk += (bos * HV + i_hv) * BT
    dAkk += (bos * HV + i_hv) * BT
    dq += (bos * HV + i_hv) * K
    dq2 += (bos * HV + i_hv) * K
    dk += (bos * HV + i_hv) * K
    dk2 += (bos * HV + i_hv) * K
    dg += (bos * HV + i_hv) * K
    dg2 += (bos * HV + i_hv) * K
    db += (i_k * n_all + bos) * HV + i_hv

    o_i = tl.arange(0, BC)
    o_c = i_ti + o_i
    m_c = o_c < T
    m_ck = m_c[:, None] & m_k[None, :]
    m_dAf = m_c[:, None] & (o_i[None, :] < BT)
    m_dAt = (o_i[:, None] < BT) & m_c[None, :]
    p_g = g + o_c[:, None] * (HV * K) + o_k[None, :]
    b_g = tl.load(p_g, mask=m_ck, other=0.0).to(tl.float32)

    p_b = beta + o_c * HV
    b_b = tl.load(p_b, mask=m_c, other=0.0)

    # Rows of this sub-chunk against earlier sub-chunks.
    b_dq2 = tl.zeros([BC, BK], dtype=tl.float32)
    b_dk2 = tl.zeros([BC, BK], dtype=tl.float32)
    if i_i > 0:
        p_gn = g + i_ti * HV * K + o_k
        b_gn = tl.load(p_gn, mask=m_k, other=0).to(tl.float32)[None, :]
        for i_j in range(0, i_i):
            o_j = i_t * BT + i_j * BC + o_i
            m_jk = (o_j < T)[:, None] & m_k[None, :]
            p_k = k + o_j[:, None] * (H * K) + o_k[None, :]
            p_gk = g + o_j[:, None] * (HV * K) + o_k[None, :]
            p_dAqk = dAqk + o_c[:, None] * (HV * BT) + (i_j * BC + o_i)[None, :]
            p_dAkk = dAkk + o_c[:, None] * (HV * BT) + (i_j * BC + o_i)[None, :]
            b_k = tl.load(p_k, mask=m_jk, other=0.0)
            b_gk = tl.load(p_gk, mask=m_jk, other=0.0)
            b_kg = b_k * exp2(b_gn - b_gk)
            b_dAqk = tl.load(p_dAqk, mask=m_dAf, other=0.0)
            b_dAkk = tl.load(p_dAkk, mask=m_dAf, other=0.0)
            b_dq2 = tl.dot(b_dAqk, b_kg, b_dq2)
            b_dk2 = tl.dot(b_dAkk, b_kg, b_dk2)
        b_gqn = exp2(b_g - b_gn)
        b_dq2 *= b_gqn
        b_dk2 *= b_gqn

    # Rows of this sub-chunk against its own diagonal block.
    o_dA = (i_ti + o_i) * HV * BT + i_i * BC
    m_dA = (i_ti + o_i) < T
    p_kj = k + i_ti * H * K + o_k
    p_gkj = g + i_ti * HV * K + o_k

    p_q = q + o_c[:, None] * (H * K) + o_k[None, :]
    p_k = k + o_c[:, None] * (H * K) + o_k[None, :]
    b_q = tl.load(p_q, mask=m_ck, other=0.0)
    b_k = tl.load(p_k, mask=m_ck, other=0.0)

    if SAFE_GATE:
        # Pivot on the sub-chunk midpoint; the bounded gate keeps both exponents in range.
        p_gn = g + (i_ti + min(BC // 2, T - i_ti - 1)) * HV * K + o_k
        b_gn = tl.load(p_gn, mask=m_k, other=0).to(tl.float32)[None, :]

        p_dAqk = dAqk + o_c[:, None] * (HV * BT) + (i_i * BC + o_i)[None, :]
        p_dAkk = dAkk + o_c[:, None] * (HV * BT) + (i_i * BC + o_i)[None, :]
        b_dAqk_d = tl.load(p_dAqk, mask=m_dAf, other=0.0).to(tl.float32)
        b_dAkk_d = tl.load(p_dAkk, mask=m_dAf, other=0.0).to(tl.float32)

        m_i_d = (
            (o_i[:, None] >= o_i[None, :])
            & ((i_ti + o_i[:, None]) < T)
            & ((i_ti + o_i[None, :]) < T)
        )
        m_j_d = (i_ti + o_i[:, None]) < T
        b_dAqk_d = tl.where(m_i_d, b_dAqk_d, 0.0)
        b_dAkk_d = tl.where(m_i_d, b_dAkk_d, 0.0)
        b_g_d = tl.where(m_j_d, b_g - b_gn, 0.0)
        exp_g_d = tl.where(m_j_d, exp2(b_g_d), 0.0)
        exp_neg_g_d = tl.where(m_j_d, exp2(-b_g_d), 0.0)

        b_k_exp = b_k * exp_neg_g_d
        b_dq2 += tl.dot(b_dAqk_d, b_k_exp) * exp_g_d
        b_dk2 += tl.dot(b_dAkk_d, b_k_exp) * exp_g_d
    else:
        for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
            b_dAqk = tl.load(dAqk + o_dA + j, mask=m_dA, other=0)
            b_dAkk = tl.load(dAkk + o_dA + j, mask=m_dA, other=0)
            b_kj = tl.load(p_kj, mask=m_k, other=0).to(tl.float32)
            b_gkj = tl.load(p_gkj, mask=m_k, other=0).to(tl.float32)
            m_i = o_i[:, None] >= j
            b_gqk = exp2(b_g - b_gkj[None, :])
            b_dq2 += tl.where(m_i, b_dAqk[:, None] * b_kj[None, :] * b_gqk, 0.0)
            b_dk2 += tl.where(m_i, b_dAkk[:, None] * b_kj[None, :] * b_gqk, 0.0)
            p_kj += H * K
            p_gkj += HV * K

    b_db = tl.sum(b_dk2 * b_k, 1)
    b_dk2 *= b_b[:, None]

    p_dq = dq + o_c[:, None] * (HV * K) + o_k[None, :]
    p_dq2 = dq2 + o_c[:, None] * (HV * K) + o_k[None, :]
    p_db = db + o_c * HV

    b_dg2 = b_q * b_dq2
    b_dq2 = b_dq2 + tl.load(p_dq, mask=m_ck, other=0.0)
    tl.store(p_dq2, b_dq2.to(p_dq2.dtype.element_ty), mask=m_ck)
    tl.store(p_db, b_db.to(p_db.dtype.element_ty), mask=m_c)

    tl.debug_barrier()
    # Columns of this sub-chunk against later sub-chunks.
    b_dkt = tl.zeros([BC, BK], dtype=tl.float32)

    NC = min(NC, tl.cdiv(T - i_t * BT, BC))
    if i_i < NC - 1:
        p_gn = g + (min(i_ti + BC, T) - 1) * HV * K + o_k
        b_gn = tl.load(p_gn, mask=m_k, other=0).to(tl.float32)[None, :]
        for i_j in range(i_i + 1, NC):
            o_j = i_t * BT + i_j * BC + o_i
            m_j = o_j < T
            m_jk = m_j[:, None] & m_k[None, :]
            m_dAj = (o_i[:, None] < BT) & m_j[None, :]
            p_q = q + o_j[:, None] * (H * K) + o_k[None, :]
            p_k = k + o_j[:, None] * (H * K) + o_k[None, :]
            p_gk = g + o_j[:, None] * (HV * K) + o_k[None, :]
            p_b = beta + o_j * HV
            p_dAqk = dAqk + (i_i * BC + o_i)[:, None] + o_j[None, :] * (HV * BT)
            p_dAkk = dAkk + (i_i * BC + o_i)[:, None] + o_j[None, :] * (HV * BT)
            b_bj = tl.load(p_b, mask=m_j, other=0.0)
            b_qj = tl.load(p_q, mask=m_jk, other=0.0)
            b_kb = tl.load(p_k, mask=m_jk, other=0.0) * b_bj[:, None]
            b_gk = tl.load(p_gk, mask=m_jk, other=0.0).to(tl.float32)
            b_dAqk = tl.load(p_dAqk, mask=m_dAj, other=0.0)
            b_dAkk = tl.load(p_dAkk, mask=m_dAj, other=0.0)
            b_gkn = exp2(b_gk - b_gn)
            b_qg = b_qj * tl.where(m_j[:, None], b_gkn, 0)
            b_kbg = b_kb * tl.where(m_j[:, None], b_gkn, 0)
            # fp32 operands: bf16 loses too much here (fla).
            b_dkt = tl.dot(b_dAqk, b_qg, b_dkt)
            b_dkt = tl.dot(b_dAkk, b_kbg, b_dkt)
        b_dkt *= exp2(b_gn - b_g)

    o_dA = i_ti * HV * BT + i_i * BC + o_i
    p_qj = q + i_ti * H * K + o_k
    p_kj = k + i_ti * H * K + o_k
    p_gkj = g + i_ti * HV * K + o_k
    p_bj = beta + i_ti * HV

    if SAFE_GATE:
        p_gn = g + (i_ti + min(BC // 2, T - i_ti - 1)) * HV * K + o_k
        b_gn = tl.load(p_gn, mask=m_k, other=0).to(tl.float32)[None, :]

        p_dAqk = dAqk + (i_i * BC + o_i)[:, None] + o_c[None, :] * (HV * BT)
        p_dAkk = dAkk + (i_i * BC + o_i)[:, None] + o_c[None, :] * (HV * BT)
        b_dAqk_d = tl.load(p_dAqk, mask=m_dAt, other=0.0).to(tl.float32)
        b_dAkk_d = tl.load(p_dAkk, mask=m_dAt, other=0.0).to(tl.float32)

        m_i_d = (
            (o_i[:, None] <= o_i[None, :])
            & ((i_ti + o_i[:, None]) < T)
            & ((i_ti + o_i[None, :]) < T)
        )
        m_j_d = (i_ti + o_i[:, None]) < T
        b_dAqk_d = tl.where(m_i_d, b_dAqk_d, 0.0)
        b_dAkk_d = tl.where(m_i_d, b_dAkk_d, 0.0)
        b_g_d = tl.where(m_j_d, b_g - b_gn, 0.0)
        exp_g_d = tl.where(m_j_d, exp2(b_g_d), 0.0)
        exp_neg_g_d = tl.where(m_j_d, exp2(-b_g_d), 0.0)

        b_q_exp = b_q * exp_g_d
        b_kb_exp = b_k * b_b[:, None] * exp_g_d
        b_dkt += tl.dot(b_dAqk_d, b_q_exp) * exp_neg_g_d
        b_dkt += tl.dot(b_dAkk_d, b_kb_exp) * exp_neg_g_d
    else:
        for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
            b_dAqk = tl.load(dAqk + o_dA + j * HV * BT)
            b_dAkk = tl.load(dAkk + o_dA + j * HV * BT)
            b_qj = tl.load(p_qj, mask=m_k, other=0).to(tl.float32)
            b_kbj = tl.load(p_kj, mask=m_k, other=0).to(tl.float32) * tl.load(p_bj)
            b_gkj = tl.load(p_gkj, mask=m_k, other=0).to(tl.float32)
            m_i = o_i[:, None] <= j
            b_gkq = exp2(b_gkj[None, :] - b_g)
            b_dkt += tl.where(m_i, b_dAqk[:, None] * b_qj[None, :] * b_gkq, 0.0)
            b_dkt += tl.where(m_i, b_dAkk[:, None] * b_kbj[None, :] * b_gkq, 0.0)
            p_qj += H * K
            p_kj += H * K
            p_gkj += HV * K
            p_bj += HV

    p_dk = dk + o_c[:, None] * (HV * K) + o_k[None, :]
    p_dk2 = dk2 + o_c[:, None] * (HV * K) + o_k[None, :]
    p_dg = dg + o_c[:, None] * (HV * K) + o_k[None, :]
    p_dg2 = dg2 + o_c[:, None] * (HV * K) + o_k[None, :]

    b_dg2 += (b_dk2 - b_dkt) * b_k + tl.load(p_dg, mask=m_ck, other=0.0)
    b_dk2 += tl.load(p_dk, mask=m_ck, other=0.0)
    b_dk2 += b_dkt

    tl.store(p_dk2, b_dk2.to(p_dk2.dtype.element_ty), mask=m_ck)
    tl.store(p_dg2, b_dg2.to(p_dg2.dtype.element_ty), mask=m_ck)


@triton.jit
def _kda_bwd_gate_cumsum_kernel(
    s,
    g,
    A_log,
    dt_bias,
    cu_seqlens,
    chunk_indices,
    o,
    dA,
    dbias,
    lower_bound,
    T,
    H: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_GATE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
):
    """Chunk-local reverse cumsum of ``s`` ``[B, T, H, S]`` (the adjoint of the forward's
    gate cumsum), then with ``USE_GATE`` the backward of the fused gate activation.

    The activation is ``-exp(A_log) * softplus(g + bias)``, or ``lower_bound *
    sigmoid(exp(A_log) * (g + bias))``. ``o`` is written in its own dtype.
    ``dA`` ``[NT, B * H, cdiv(S, BS)]`` and ``dbias`` ``[NT, B * H, S]`` receive
    per-program partial sums of the ``A_log`` and ``dt_bias`` gradients (zeros
    from programs on padded varlen chunks).
    """
    i_s, i_t, i_bh = (
        tl.program_id(0),
        tl.program_id(1).to(tl.int64),
        tl.program_id(2).to(tl.int64),
    )
    i_p = (i_t * tl.num_programs(2) + i_bh) * tl.num_programs(0) + i_s
    i_b, i_h = i_bh // H, i_bh % H
    o_s = i_s * BS + tl.arange(0, BS)
    o_db = (i_p - i_s) // tl.num_programs(0) * S + o_s
    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        T = eos - bos
        if i_t * BT >= T:
            if USE_GATE:
                tl.store(dA + i_p, 0.0)
                if HAS_BIAS:
                    tl.store(dbias + o_db, tl.zeros([BS], dtype=tl.float32), mask=o_s < S)
            return
    else:
        bos, eos = i_b * T, i_b * T + T

    o_t = i_t * BT + tl.arange(0, BT)
    m_s = (o_t[:, None] < T) & (o_s[None, :] < S)
    offs = (bos * H + i_h) * S + o_t[:, None] * (H * S) + o_s[None, :]
    b_s = tl.load(s + offs, mask=m_s, other=0.0).to(tl.float32)
    b_o = tl.cumsum(b_s, axis=0, reverse=True)

    if USE_GATE:
        b_A = tl.load(A_log + i_h).to(tl.float32)
        b_g = tl.load(g + offs, mask=m_s, other=0.0).to(tl.float32)
        if HAS_BIAS:
            b_bias = tl.load(dt_bias + i_h * S + o_s, mask=o_s < S, other=0.0).to(tl.float32)
            b_g = b_g + b_bias[None, :]
        if not USE_LOWER_BOUND:
            b_A = -exp(b_A)
            b_yg = b_A * softplus(b_g)
            b_dg = b_A * (b_o * tl.sigmoid(b_g))
            b_dA = tl.sum(tl.sum(tl.where(m_s, b_o * b_yg, 0.0), 1), 0)
        else:
            b_A = exp(b_A)
            b_sig = tl.sigmoid(b_A * b_g)
            b_dg = b_o * (lower_bound * b_sig * (1.0 - b_sig)) * b_A
            b_dA = tl.sum(tl.sum(tl.where(m_s, b_dg * b_g, 0.0), 1), 0)
        b_o = b_dg
        tl.store(dA + i_p, b_dA)
        if HAS_BIAS:
            tl.store(dbias + o_db, tl.sum(tl.where(m_s, b_dg, 0.0), 0), mask=o_s < S)

    tl.store(o + offs, b_o.to(o.dtype.element_ty), mask=m_s)


@triton.jit
def _kda_bwd_l2norm_kernel(
    X,
    DY,
    DX,
    eps,
    T,
    D: tl.constexpr,
    BD: tl.constexpr,
    BT: tl.constexpr,
):
    """Backward of the row L2 normalization, recomputing the norm from ``X``."""
    xoffset = tl.program_id(0).to(tl.int64) * BT
    row_idx = xoffset + tl.arange(0, BT)[:, None]
    col_idx = tl.arange(0, BD)[None, :]
    mask = (row_idx < T) & (col_idx < D)
    x = tl.load(X + col_idx + D * row_idx, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(DY + col_idx + D * row_idx, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, axis=1) + eps)[:, None]
    y = x * rstd
    dx = dy * rstd - tl.sum(dy * y, axis=1)[:, None] * y * rstd
    tl.store(DX + col_idx + D * row_idx, dx.to(DX.dtype.element_ty), mask=mask)


@triton.jit
def _kda_bwd_beta_sigmoid_kernel(
    x,
    dy,
    dx,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """dx = dy * sigmoid(x) * (1 - sigmoid(x))."""
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = offs < n_elements
    b_x = tl.load(x + offs, mask=mask, other=0).to(tl.float32)
    b_dy = tl.load(dy + offs, mask=mask, other=0).to(tl.float32)
    b_y = tl.sigmoid(b_x)
    tl.store(dx + offs, (b_dy * b_y * (1.0 - b_y)).to(dx.dtype.element_ty), mask=mask)
