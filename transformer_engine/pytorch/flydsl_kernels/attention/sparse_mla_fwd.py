###############################################################################
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Ported from Primus-Turbo `primus_turbo/flydsl/attention/sparse_mla_fwd.py`
# (AMD-AGI/Primus-Turbo @ 502f3f35ce8e6090b2f729d1b478ce65a509f831), which is
# itself adapted from FlyDSL (https://github.com/ROCm/FlyDSL) and carries the
# Apache-2.0 terms. TE ships no LICENSE-APACHE, so the terms for the adapted
# portions are those at https://github.com/ROCm/FlyDSL (LICENSE-APACHE).
#
# See LICENSE for license information.
###############################################################################

"""DeepSeek-V4 sparse-MLA attention forward (flydsl, gfx950/MI355X).

Decode-shaped: grid is ``(num_tokens, num_heads // BLOCK_H)``, one query token
per workgroup. MLA's K and V are the same 576-wide latent per KV position, so a
single LDS tile is staged per KV block and read two ways (contiguous for QK,
transposed via ``ds_read_tr16`` for PV).

.. warning::

   **The QK product covers only dims [0, 512).** ``d_qk == 576`` is a storage
   stride only -- the trailing 64 dims (the decoupled RoPE sub-head in standard
   MLA) are read by neither Q nor KV, and ``STRIDE = D_LDS = 528 < 576`` means
   the LDS tile cannot hold them. Scores are ``q[..., :512] @ kv[..., :512].T``.

   This is upstream behavior, verified on gfx950 across all four dispatch paths
   (matches a 512-dim reference with LSE error 0.0; misses a 576-dim reference
   by ~40%). Callers must fold RoPE into the 512-wide latent themselves; this is
   **not** a drop-in MLA forward.

Kept deliberately close to upstream so re-syncs stay cheap: the kernel body is
byte-identical to the pinned commit. Only the header, this docstring, and the
host-side dispatch (moved to ``attention_wrappers.py``) differ.

This module imports ``flydsl`` at import time and must therefore be imported
lazily only after FlyDSL availability has been confirmed.
"""

from __future__ import annotations

import collections
import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import ArithValue
from flydsl.expr.utils.arith import _to_raw as _raw
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

# ============================================================================
# kernel implementation
# ============================================================================

_LOG2E = 1.4426950408889634
BLOCK_H = 64
HPW = 16
WAVES = BLOCK_H // HPW  # 4
THREADS = WAVES * 64  # 256
D = 512
D_LDS = 528  # padded D stride: %32=8 gives 16 distinct tr16 read banks (conflict-free diag)
DQK = 576
TILE_K = 16
TKP = 16  # kv rows = TILE_K (PV MFMA is K=16 now; no zero-pad needed)
KS = D // 32  # 16 QK MFMA K-steps
DT = D // 16  # 32 PV d-tiles


def build_fwd(
    topk_len,
    scale,
    has_sink=True,
    banded=False,
    qk2=False,
    pool_P=0,
    pool_cr=0,
    pf_pv=4,
    single_buffer=False,
    xcd_remap=0,
    peel=False,
    hoist_sink=True,
    fused_store=False,
    pv_early=False,
    pf_qk=3,
    dyn_pool=False,
    bh=BLOCK_H,
    pstore=False,
    fast_path=True,
):
    elem = fx.BFloat16
    # Per-build head-block: bh=128 (1 WG/token, 8 waves) collapses the two bh=64 WGs that
    # each redundantly gather+store the SAME shared MLA latent to LDS, halving KV store work.
    BLOCK_H = bh
    WAVES = BLOCK_H // HPW
    THREADS = WAVES * 64
    CHUNKS = THREADS // TILE_K  # d-chunks storing one KV tile (256->16, 512->32)
    ECHUNK = D // CHUNKS  # elems per chunk (256->32, 512->16)
    V8PT = ECHUNK // 8  # v8 KV regs gathered/stored per thread (256->4, 512->2)
    pv2x = not banded  # 2-tile-batch PV K=32 (non-banded gather path)
    # No XOR swizzle for the shared MLA latent: one swizzle can't make both the QK
    # contiguous-v8 read and the PV transpose-tr16 read conflict-free (K==V shares one
    # LDS). D_LDS=528 padding is the layout -> _swz is identity.
    STRIDE = D_LDS  # kv-row LDS stride (elements)
    # pstore: 4-buffer parity PRESTORE pipeline for the many-tile pv2x gather path. Pair
    # pr gathers+stores pair pr+1 into the OTHER parity buffers first, so those reg->LDS
    # ds_writes overlap this pair's PV MFMA drain; collapses 3->1 barriers/pair.
    pstore = bool(pstore) and pv2x
    NUM_TILES = (topk_len + TILE_K - 1) // TILE_K
    W = 128  # SWA window for the closed-form banded (cr0) path: kv in [token-127..token]
    LDS_ELEMS = TKP * STRIDE  # 16 kv rows x padded D per KV LDS tile
    # single_buffer: one KV LDS tile + an extra WAR barrier per tile (pro many-tile gather
    # only). pstore uses 4 parity buffers.
    NBUF = 4 if pstore else (1 if (single_buffer and not pv2x) else 2)
    # fast_path (first-pair fixed-max): default-ON accelerated softmax for the rescale-bound cr4
    # shapes (pro cr4 = plain path, flash cr4 = pstore path; both pv2x/non-banded). softmax is
    # shift-invariant so using the first pair's row-max as a fixed bound is math-exact; the fixed
    # alpha=1 lets the compiler fold the per-tile o*=alpha rescale -> nomax speed at higher
    # precision. Banded (cr0/cr128) are gather/memory-bound (no rescale head) so left out.
    _fmax0 = bool(fast_path and pv2x and not banded)
    allocator = SmemAllocator(None, arch=get_hip_arch(), global_sym_name="mla_fwd_smem")
    kv_off = allocator._align(allocator.ptr, 16)
    mask_off = allocator._align(kv_off + NBUF * LDS_ELEMS * 2, 16)
    # The QK/PV operand-swap removes the per-tile P-repack, so no lds_p region.
    allocator.ptr = allocator._align(mask_off + NBUF * TILE_K * 4, 16)

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def k_fn(
        Q: fx.Tensor,
        KV: fx.Tensor,
        TOPK: fx.Tensor,
        SINK: fx.Tensor,
        O: fx.Tensor,
        LSE: fx.Tensor,
        T: fx.Int32,
        H: fx.Int32,
        NKV: fx.Int32,
    ):
        v8 = Vec.make_type(8, elem)
        v4 = Vec.make_type(4, elem)
        v4f = Vec.make_type(4, fx.Float32)
        lds_kv = SmemPtr(allocator.get_base(), kv_off, elem.ir_type, shape=(NBUF * LDS_ELEMS,)).get()
        lds_mask = SmemPtr(allocator.get_base(), mask_off, fx.Float32.ir_type, shape=(NBUF * TILE_K,)).get()

        tid = fx.Index(gpu.thread_idx.x)
        lane = tid % fx.Index(64)
        wave = tid // fx.Index(64)
        lo = lane % fx.Index(16)
        grp = lane // fx.Index(16)

        def _swz(col, row):
            return col  # identity: no swizzle for MLA single-latent KV (see build_fwd note)

        # XCD-aware token remap (bit-identical WG->token permutation): each XCD owns a
        # contiguous token block so overlapping banded SWA windows (token t attends kv
        # [t-127..t]) stay L2-resident instead of round-robining across 8 L2 slices.
        if const_expr(xcd_remap):
            NXCD = 8
            raw = fx.Index(gpu.block_idx.x)
            token = (raw % fx.Index(NXCD)) * (fx.Index(T) // fx.Index(NXCD)) + raw // fx.Index(NXCD)
        else:
            token = fx.Index(gpu.block_idx.x)
        hg = fx.Index(gpu.block_idx.y)
        Hn = fx.Index(H)
        head_wave_base = hg * fx.Index(BLOCK_H) + wave * fx.Index(HPW)
        head_A = head_wave_base + lo

        q_rsrc = buffer_ops.create_buffer_resource(
            Q, max_size=False, num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(DQK * 2))
        )
        kv_rsrc = buffer_ops.create_buffer_resource(
            KV, max_size=False, num_records_bytes=_raw(fx.Index(NKV) * fx.Index(DQK * 2))
        )
        tk_rsrc = buffer_ops.create_buffer_resource(
            TOPK, max_size=False, num_records_bytes=_raw(fx.Index(T) * fx.Index(topk_len * 4))
        )
        o_rsrc = buffer_ops.create_buffer_resource(
            O, max_size=False, num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(D * 2))
        )
        lse_rsrc = buffer_ops.create_buffer_resource(
            LSE, max_size=False, num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(4))
        )
        if const_expr(has_sink):
            sink_rsrc = buffer_ops.create_buffer_resource(
                SINK, max_size=False, num_records_bytes=_raw(Hn * fx.Index(4))
            )

        c_log2e = fx.Float32(_LOG2E)
        c_scale = fx.Float32(scale)
        c_sl = fx.Float32(scale * _LOG2E)  # scale folded into exp2 base (unscaled score space)
        c_neg_inf = fx.Float32(float("-inf"))
        c_big_neg = fx.Float32(-3.0e38)  # finite running-max init: avoids -inf-(-inf)=NaN
        c_zero = fx.Float32(0.0)
        _mb = [c_zero]  # fixed-max per-row bound holder (set by the fast_path first-pair pre-step)

        # ---- Q register-resident (A operand): head=head_A ----
        q_row = token * Hn * fx.Index(DQK) + head_A * fx.Index(DQK)
        q_packs = [
            buffer_ops.buffer_load(
                q_rsrc, q_row + fx.Index(ks * 32) + grp * fx.Index(8), vec_width=8, dtype=elem
            )
            for ks in range_constexpr(KS)
        ]

        # hoist_sink: the per-head sink is loop-invariant and only used in the epilogue.
        # Issue its VMEM load at kernel top so it flies during the KV loop instead of
        # stalling the epilogue. head=head_wave_base+lo.
        sink_top = None
        if const_expr(has_sink and hoist_sink):
            sink_top = fx.Float32(
                buffer_ops.buffer_load(sink_rsrc, head_wave_base + lo, vec_width=1, dtype=fx.Float32)
            )

        g_row = tid // fx.Index(CHUNKS)
        g_within = tid % fx.Index(CHUNKS)
        tk_row = token * fx.Index(topk_len)

        def load_topk(tbase):
            # one gathered-row topk index for this thread's g_row (int32, 1 VGPR to carry)
            return fx.Int32(
                buffer_ops.buffer_load(tk_rsrc, tk_row + tbase + g_row, vec_width=1, dtype=fx.Int32)
            )

        def gather_load(idx):
            # Cross-tile prefetch: load this gathered KV row into VGPRs (no LDS store yet).
            # idx is prefetched a tile ahead; the loads are carried in iter_args and stored
            # to LDS next iteration so they fly during this tile's compute (latency-bound).
            valid = ArithValue(idx >= fx.Int32(0))
            src = fx.Index(valid.select(idx, fx.Int32(0)))
            return [
                buffer_ops.buffer_load(
                    kv_rsrc,
                    src * fx.Index(DQK) + g_within * fx.Index(ECHUNK) + fx.Index(c * 8),
                    vec_width=8,
                    dtype=elem,
                )
                for c in range_constexpr(V8PT)
            ]

        def gather_store(vvs, idx, buf_off, mbuf_off):
            # store the prefetched KV regs to LDS + publish the per-row mask.
            valid = ArithValue(idx >= fx.Int32(0))
            for c in range_constexpr(V8PT):
                Vec(vvs[c]).store(
                    lds_kv,
                    [
                        buf_off
                        + g_row * fx.Index(STRIDE)
                        + _swz(g_within * fx.Index(ECHUNK) + fx.Index(c * 8), g_row)
                    ],
                )
            if g_within == fx.Index(0):
                m = fx.Float32(valid.select(_raw(c_zero), _raw(c_neg_inf)))
                Vec.from_elements([m], fx.Float32).store(lds_mask, [mbuf_off + g_row])

        # ---- Closed-form banded path (no topk HBM load / no scatter): rank=t*TILE_K+g_row.
        # rank<W: SWA, kv=token-(W-1)+rank valid iff kv>=0. W<=rank<W+P: causal pool
        # kv=T+(rank-W) valid iff token>=cr*ps+(cr-1). rank>=W+P: padding masked -inf.
        tok_i32 = fx.Int32(token)
        grow_i32 = fx.Int32(g_row)
        t_tokens = fx.Int32(T)

        def kv_and_valid_b(t_val):
            rank = fx.Int32(t_val) * fx.Int32(TILE_K) + grow_i32
            swa_kv = tok_i32 - fx.Int32(W - 1) + rank
            if const_expr(pool_P == 0):
                return swa_kv, ArithValue(swa_kv >= fx.Int32(0))
            is_swa = ArithValue(rank < fx.Int32(W))
            swa_ok = ArithValue(swa_kv >= fx.Int32(0))
            ps = rank - fx.Int32(W)
            pool_kv = t_tokens + ps
            in_pool = ArithValue(rank < fx.Int32(W + pool_P))
            pool_ok = ArithValue(tok_i32 >= (fx.Int32(pool_cr) * ps + fx.Int32(pool_cr - 1)))
            pool_valid = ArithValue(arith.AndIOp(_raw(in_pool), _raw(pool_ok)).result)
            kv = fx.Int32(is_swa.select(_raw(swa_kv), _raw(pool_kv)))
            valid = ArithValue(is_swa.select(_raw(swa_ok), _raw(pool_valid)))
            return kv, valid

        def gather_load_b(t_val):
            kv, valid = kv_and_valid_b(t_val)
            src = fx.Index(valid.select(kv, fx.Int32(0)))
            return [
                buffer_ops.buffer_load(
                    kv_rsrc,
                    src * fx.Index(DQK) + g_within * fx.Index(ECHUNK) + fx.Index(c * 8),
                    vec_width=8,
                    dtype=elem,
                )
                for c in range_constexpr(V8PT)
            ]

        def gather_store_b(vvs, t_val, buf_off, mbuf_off):
            _, valid = kv_and_valid_b(t_val)
            for c in range_constexpr(V8PT):
                Vec(vvs[c]).store(
                    lds_kv,
                    [
                        buf_off
                        + g_row * fx.Index(STRIDE)
                        + _swz(g_within * fx.Index(ECHUNK) + fx.Index(c * 8), g_row)
                    ],
                )
            if g_within == fx.Index(0):
                m = fx.Float32(valid.select(_raw(c_zero), _raw(c_neg_inf)))
                Vec.from_elements([m], fx.Float32).store(lds_mask, [mbuf_off + g_row])

        # Softmax reduces along kv=grp*4+i: 4 in-register values (i) plus a cross-group
        # all-reduce over grp via pure-VALU permlane16/32_swap (xor16/xor32), no LDS. Each
        # returns the (own, peer) pair as an llvm struct so combine reduces with no cndmask.
        _pair_ty = ir.Type.parse("!llvm.struct<(i32, i32)>")

        def _rswap(x, op):
            v_i32 = _raw(ArithValue(_raw(x)).bitcast(fx.Int32.ir_type))
            sw = op(_pair_ty, v_i32, v_i32, False, True)
            a = llvm.extractvalue(fx.Int32.ir_type, sw, [0])
            b = llvm.extractvalue(fx.Int32.ir_type, sw, [1])
            af = fx.Float32(_raw(ArithValue(a).bitcast(fx.Float32.ir_type)))
            bf = fx.Float32(_raw(ArithValue(b).bitcast(fx.Float32.ir_type)))
            return af, bf

        def crossgrp_max(x):
            a, b = _rswap(x, rocdl.permlane16_swap)
            m = fx.Float32(arith.MaxNumFOp(_raw(a), _raw(b)).result)
            a2, b2 = _rswap(m, rocdl.permlane32_swap)
            return fx.Float32(arith.MaxNumFOp(_raw(a2), _raw(b2)).result)

        def crossgrp_sum(x):
            a, b = _rswap(x, rocdl.permlane16_swap)
            m = fx.Float32(arith.AddFOp(_raw(a), _raw(b)).result)
            a2, b2 = _rswap(m, rocdl.permlane32_swap)
            return fx.Float32(arith.AddFOp(_raw(a2), _raw(b2)).result)

        c_zero_v4 = Vec.filled(4, 0.0, fx.Float32)
        PF_QK = pf_qk
        PF_PV = pf_pv

        # ---- QK / softmax / PV compute closures ----
        def _qk_s(buf_off, kvp=None):
            # QK over full K=512 for the KV tile at LDS[buf_off]; returns the 4 per-lane
            # scores S[head=lo, kv=grp*4+i] (mask not yet added).
            _kvp = lds_kv if kvp is None else kvp

            def _bv(ks):
                return Vec.load(
                    v8,
                    _kvp,
                    [buf_off + lo * fx.Index(STRIDE) + _swz(fx.Index(ks * 32) + grp * fx.Index(8), lo)],
                )

            bvq = [_bv(k) for k in range_constexpr(PF_QK)]
            if const_expr(qk2):
                # 2-accumulator: split the 16-deep MFMA RAW chain into two 8-deep
                # independent chains (even/odd ks) to halve the QK operand bubble.
                acc0 = Vec.filled(4, 0.0, fx.Float32)
                acc1 = Vec.filled(4, 0.0, fx.Float32)
                for ks in range_constexpr(KS):
                    if ks + PF_QK < KS:
                        bvq.append(_bv(ks + PF_QK))
                    if ks % 2 == 0:
                        acc0 = rocdl.mfma_f32_16x16x32_bf16(v4f, [_raw(bvq[ks]), q_packs[ks], acc0])
                    else:
                        acc1 = rocdl.mfma_f32_16x16x32_bf16(v4f, [_raw(bvq[ks]), q_packs[ks], acc1])
                return [
                    fx.Float32(_raw(Vec(acc0)[i])) + fx.Float32(_raw(Vec(acc1)[i]))
                    for i in range_constexpr(4)
                ]
            acc = Vec.filled(4, 0.0, fx.Float32)
            for ks in range_constexpr(KS):
                if ks + PF_QK < KS:
                    bvq.append(_bv(ks + PF_QK))
                acc = rocdl.mfma_f32_16x16x32_bf16(v4f, [_raw(bvq[ks]), q_packs[ks], acc])
            return [fx.Float32(_raw(Vec(acc)[i])) for i in range_constexpr(4)]

        def _softmax(s4, mask4, m_run, l_run):
            # online-softmax update for one KV tile (unscaled score space).
            s_i = [s4[i] + fx.Float32(_raw(Vec(mask4)[i])) for i in range_constexpr(4)]
            # lmax = max over i on the PV critical path (lmax->crossgrp_max->m_new->exp2->
            # pB->PV MFMA). MaxNumF is associative for non-NaN floats so this is bit-exact
            # (unlike lsum, kept a fixed-order chain for determinism).
            lmax = s_i[0]
            for i in range_constexpr(1, 4):
                lmax = fx.Float32(arith.MaxNumFOp(_raw(lmax), _raw(s_i[i])).result)
            rmax = crossgrp_max(lmax)  # true rowmax over all 16 kv for head=lo
            m_new = fx.Float32(arith.MaxNumFOp(_raw(m_run), _raw(rmax)).result)
            alpha = fx.Float32(rocdl.exp2(fx.Float32.ir_type, _raw((m_run - m_new) * c_sl)))
            p = [
                fx.Float32(rocdl.exp2(fx.Float32.ir_type, _raw((s_i[i] - m_new) * c_sl)))
                for i in range_constexpr(4)
            ]
            lsum = p[0]
            for i in range_constexpr(1, 4):
                lsum = fx.Float32(arith.AddFOp(_raw(lsum), _raw(p[i])).result)
            # Deferred kv-sum: l_run is a per-grp partial; cross-grp sum done at epilogue.
            l_new = fx.Float32(alpha * l_run + lsum)
            return m_new, alpha, p, l_new

        def _store_o(o_v4, scal_v, dt):
            # scale + pack + store one output d-tile
            # (o[dt][i]=O[head=lo,d=dt*16+grp*4+i], 4 consecutive d/lane).
            head_i = head_wave_base + lo
            ov = Vec(o_v4) * Vec(scal_v)
            base = (
                token * Hn * fx.Index(D) + head_i * fx.Index(D) + fx.Index(dt * 16) + grp * fx.Index(4)
            ) * fx.Index(2)
            pk0 = rocdl.cvt_pk_bf16_f32(_raw(Vec(ov)[0]), _raw(Vec(ov)[1]))
            pk1 = rocdl.cvt_pk_bf16_f32(_raw(Vec(ov)[2]), _raw(Vec(ov)[3]))
            buffer_ops.buffer_store(
                _raw(Vec.from_elements([fx.Int32(_raw(pk0)), fx.Int32(_raw(pk1))], fx.Int32)),
                o_rsrc,
                base,
                offset_is_bytes=True,
            )

        def _tr(pv_base, dt):
            ptr = buffer_ops.create_llvm_ptr(_raw(pv_base + fx.Int64(dt * 32)), address_space=3)
            return _raw(Vec(rocdl.ds_read_tr16_b64(v4, ptr).result).bitcast(fx.Int16))

        def _pv_prefetch(buf_off):
            # Issue the first PF_PV PV tr16 ds_reads. They depend only on buf_off, not
            # softmax, so with pv_early they hoist before the softmax bridge to overlap
            # LDS-read latency with the VALU bridge. Returns the i64 LDS base + pre-reads.
            lo_d4 = lo // fx.Index(4)
            lo_m4 = lo % fx.Index(4)
            pv_base = fx.Int64(
                (fx.Index(buf_off) + (grp * fx.Index(4) + lo_d4) * fx.Index(STRIDE) + lo_m4 * fx.Index(4))
                * fx.Index(2)
                + fx.Index(kv_off)
            )

            trq = [_tr(pv_base, dt) for dt in range_constexpr(PF_PV)]
            return pv_base, trq

        def _pv_compute(pv_base, trq, p, alpha, o_acc, store_scal=None):
            # PV: K=16, A=V(ds_read_tr16), B=P; fuses o_acc*alpha rescale. tr16 reads are
            # prefetched PF_PV d-tiles ahead (32 MFMAs independent, wall is ds_read latency).
            # store_scal: scale+pack+store each d-tile inline as its MFMA retires (peeled).
            do_store = store_scal is not None
            pB = _raw(
                Vec.from_elements([fx.BFloat16(_raw(p[i])) for i in range_constexpr(4)], elem).bitcast(
                    fx.Int16
                )
            )
            alpha_v = Vec.from_elements([alpha, alpha, alpha, alpha], fx.Float32)
            if const_expr(do_store):
                scal_v = Vec.from_elements([store_scal, store_scal, store_scal, store_scal], fx.Float32)

            new_o = [None] * DT
            for dt in range_constexpr(DT):
                if dt + PF_PV < DT:
                    trq.append(_tr(pv_base, dt + PF_PV))
                o_r = Vec(o_acc[dt]) * Vec(alpha_v)
                res = rocdl.mfma_f32_16x16x16bf16_1k(v4f, [trq[dt], pB, o_r])
                if const_expr(do_store):
                    _store_o(res, scal_v, dt)
                else:
                    new_o[dt] = res
            return new_o

        def _pv_o(buf_off, p, alpha, o_acc, store_scal=None):
            pv_base, trq = _pv_prefetch(buf_off)
            return _pv_compute(pv_base, trq, p, alpha, o_acc, store_scal=store_scal)

        def _softmax32(s_a, s_b, mask_a, mask_b, m_run, l_run):
            # combined online softmax over 32 kv (2 tiles): one crossgrp_max, both p at the
            # same m_new (no alpha-fold), deferred kv-sum (per-grp partial).
            sa = [s_a[i] + fx.Float32(_raw(Vec(mask_a)[i])) for i in range_constexpr(4)]
            sb = [s_b[i] + fx.Float32(_raw(Vec(mask_b)[i])) for i in range_constexpr(4)]
            if const_expr(_fmax0):
                # fixed-max: P=exp2((S - m_bound)*c_sl), alpha=const 1.0 -> compiler FOLDS the
                # o*=alpha rescale (that fold, not a skip, is why fixed-max hits nomax speed).
                # _mb[0] = the first-pair max bound set by the fast_path pre-step.
                one = fx.Float32(1.0)
                mb = _mb[0]
                p_a = [
                    fx.Float32(rocdl.exp2(fx.Float32.ir_type, _raw((sa[i] - mb) * c_sl)))
                    for i in range_constexpr(4)
                ]
                p_b = [
                    fx.Float32(rocdl.exp2(fx.Float32.ir_type, _raw((sb[i] - mb) * c_sl)))
                    for i in range_constexpr(4)
                ]
                lsum = p_a[0]
                for i in range_constexpr(1, 4):
                    lsum = fx.Float32(arith.AddFOp(_raw(lsum), _raw(p_a[i])).result)
                for i in range_constexpr(4):
                    lsum = fx.Float32(arith.AddFOp(_raw(lsum), _raw(p_b[i])).result)
                return mb, one, p_a, p_b, fx.Float32(l_run + lsum)  # m=mb so lse=mb+log(l) correct
            lmax = sa[0]
            for i in range_constexpr(1, 4):
                lmax = fx.Float32(arith.MaxNumFOp(_raw(lmax), _raw(sa[i])).result)
            for i in range_constexpr(4):
                lmax = fx.Float32(arith.MaxNumFOp(_raw(lmax), _raw(sb[i])).result)
            rmax = crossgrp_max(lmax)
            m_new = fx.Float32(arith.MaxNumFOp(_raw(m_run), _raw(rmax)).result)
            alpha = fx.Float32(rocdl.exp2(fx.Float32.ir_type, _raw((m_run - m_new) * c_sl)))
            p_a = [
                fx.Float32(rocdl.exp2(fx.Float32.ir_type, _raw((sa[i] - m_new) * c_sl)))
                for i in range_constexpr(4)
            ]
            p_b = [
                fx.Float32(rocdl.exp2(fx.Float32.ir_type, _raw((sb[i] - m_new) * c_sl)))
                for i in range_constexpr(4)
            ]
            lsum = p_a[0]
            for i in range_constexpr(1, 4):
                lsum = fx.Float32(arith.AddFOp(_raw(lsum), _raw(p_a[i])).result)
            for i in range_constexpr(4):
                lsum = fx.Float32(arith.AddFOp(_raw(lsum), _raw(p_b[i])).result)
            l_new = fx.Float32(alpha * l_run + lsum)
            return m_new, alpha, p_a, p_b, l_new

        def _v32_base(buf_a, buf_b, koff=kv_off):
            lo_d4 = lo // fx.Index(4)
            lo_m4 = lo % fx.Index(4)

            def _base(bo):
                return fx.Int64(
                    (fx.Index(bo) + (grp * fx.Index(4) + lo_d4) * fx.Index(STRIDE) + lo_m4 * fx.Index(4))
                    * fx.Index(2)
                    + fx.Index(koff)
                )

            return _base(buf_a), _base(buf_b)

        def _v32(base_a, base_b, dt):
            pa = buffer_ops.create_llvm_ptr(_raw(base_a + fx.Int64(dt * 32)), address_space=3)
            pb = buffer_ops.create_llvm_ptr(_raw(base_b + fx.Int64(dt * 32)), address_space=3)
            va = Vec(rocdl.ds_read_tr16_b64(v4, pa).result).bitcast(fx.Int16)
            vb = Vec(rocdl.ds_read_tr16_b64(v4, pb).result).bitcast(fx.Int16)
            return _raw(
                Vec.from_elements([va[0], va[1], va[2], va[3], vb[0], vb[1], vb[2], vb[3]], fx.Int16).bitcast(
                    elem
                )
            )

        def _pv_o_k32(base_a, base_b, trq, p_a, ab, p_b, aprod, o_acc):
            # PV K=32 over 2 tiles: o = o*aprod + [p_a*ab | p_b] @ [V_a | V_b].
            # trq = PF_PV prefetched V32 (read before softmax to overlap its VALU bridge).
            # fast_path/fixed-max: aprod is const 1.0 -> the compiler folds o*=aprod (rescale-free).
            pB = _raw(
                Vec.from_elements(
                    [fx.BFloat16(_raw(p_a[i] * ab)) for i in range_constexpr(4)]
                    + [fx.BFloat16(_raw(p_b[i])) for i in range_constexpr(4)],
                    elem,
                )
            )
            aprod_v = Vec.from_elements([aprod, aprod, aprod, aprod], fx.Float32)
            new_o = [None] * DT
            for dt in range_constexpr(DT):
                if dt + PF_PV < DT:
                    trq.append(_v32(base_a, base_b, dt + PF_PV))
                new_o[dt] = rocdl.mfma_f32_16x16x32_bf16(v4f, [trq[dt], pB, Vec(o_acc[dt]) * Vec(aprod_v)])
            return new_o

        # ---- Epilogue closures (shared by the plain and peeled paths) ----
        def _epi_scalars(m_run, l_run):
            # l_run must already be the cross-grp summed denominator. Returns per-head store
            # scale + LSE inputs. Pure scalar chain (sink fold exp2, rcp) independent of PV
            # o_acc, so in the peeled path it overlaps the last tile's 32-MFMA drain.
            mrs = fx.Float32(m_run * c_scale)
            head_i = head_wave_base + lo
            if const_expr(has_sink):
                if const_expr(hoist_sink):
                    sink = sink_top
                else:
                    sink = fx.Float32(
                        buffer_ops.buffer_load(sink_rsrc, head_i, vec_width=1, dtype=fx.Float32)
                    )
                mf = fx.Float32(arith.MaxNumFOp(_raw(mrs), _raw(sink)).result)
                af = fx.Float32(_raw(ArithValue(_raw((mrs - mf) * c_log2e)).exp2()))
                st = fx.Float32(_raw(ArithValue(_raw((sink - mf) * c_log2e)).exp2()))
                l_t = fx.Float32(l_run * af + st)
                m_f = mf
            else:
                af = fx.Float32(1.0)
                l_t = l_run
                m_f = mrs
            lp = ArithValue(l_t > c_zero)
            safe = fx.Float32(lp.select(_raw(l_t), _raw(fx.Float32(1.0))))
            inv = fx.Float32(rocdl.rcp(fx.Float32.ir_type, _raw(safe)))
            scal = fx.Float32(af * inv)
            return m_f, l_t, lp, scal

        def _write_lse(m_f, l_t, lp):
            # LSE is per head=lo (identical across grp) -> only grp==0 writes it.
            head_i = head_wave_base + lo
            lse_val = fx.Float32(m_f) + fmath.log(l_t)
            lse_out = fx.Float32(lp.select(_raw(lse_val), _raw(c_neg_inf)))
            buffer_ops.buffer_store(
                lse_out,
                lse_rsrc,
                (token * Hn + head_i) * fx.Index(4),
                mask=_raw(ArithValue(grp == fx.Index(0))),
                offset_is_bytes=True,
            )

        def _write_out(m_f, l_t, lp, scal, o_final):
            scal_v = Vec.from_elements([scal, scal, scal, scal], fx.Float32)
            for dt in range_constexpr(DT):
                _store_o(o_final[dt], scal_v, dt)
            _write_lse(m_f, l_t, lp)

        # ---- Per-tile online-softmax loop ----
        if const_expr(banded):
            kv0 = gather_load_b(fx.Index(0))
            init = [c_big_neg, c_zero] + [c_zero_v4 for _ in range_constexpr(DT)] + list(kv0)
        else:
            idxA0 = load_topk(fx.Index(0))
            kv0 = gather_load(idxA0)
            idxB0 = load_topk(fx.Index(TILE_K))
            init = [c_big_neg, c_zero] + [c_zero_v4 for _ in range_constexpr(DT)] + list(kv0) + [idxA0, idxB0]

        # peel_last: unroll the last tile after the scf.for so the epilogue scalar chain
        # interleaves with its PV MFMA drain. dyn_pool: WG-uniform trip for the cr128 pool
        # tail; skipping fully-masked tiles (s=-inf -> alpha=1, p=0) is a bit-exact no-op.
        peel_last = bool(banded and peel)
        if const_expr(dyn_pool):
            n_sw = W // TILE_K
            n_pool_tiles = (pool_P + TILE_K - 1) // TILE_K
            total_tiles = fx.Index(n_sw)
            for j in range_constexpr(n_pool_tiles):
                thr = pool_cr * (j * TILE_K) + (pool_cr - 1)
                okj = ArithValue(tok_i32 >= fx.Int32(thr))
                total_tiles = total_tiles + fx.Index(_raw(okj.select(_raw(fx.Int32(1)), _raw(fx.Int32(0)))))
            t_last_dyn = total_tiles - fx.Index(1)
            loop_ub = t_last_dyn if peel_last else total_tiles
        else:
            n_loop = (NUM_TILES - 1) if peel_last else NUM_TILES
            loop_ub = fx.Index(n_loop)
            t_last_dyn = fx.Index(NUM_TILES - 1)
        if const_expr(pv2x and pstore):
            # 2-tile-batch PV K=32 + 4-buffer parity PRESTORE pipeline: pair pr reads buffers
            # stored by the previous iteration's prestore, then prestores pair pr+1 into the
            # other buffers so those ds_writes overlap its PV drain (one barrier/pair, WAR).
            LE = fx.Index(LDS_ELEMS)
            MK = fx.Index(TILE_K)
            TWO = fx.Index(2)
            kv1 = gather_load(idxB0)
            gather_store(kv0, idxA0, fx.Index(0), fx.Index(0))  # prologue: pair0 -> bufs 0,1
            gather_store(kv1, idxB0, LE, MK)
            if const_expr(_fmax0):
                # fast_path: first-pair fixed-max bound (see plain path). Pair0 is already in bufs
                # {0,LE}; QK it once for the crossgrp-max -> _mb[0], then the loop's _softmax32 runs
                # the fixed-max branch (alpha=1 folded -> rescale-free, nomax speed, high precision).
                gpu.barrier()
                sa_f0 = _qk_s(fx.Index(0))
                sb_f0 = _qk_s(LE)
                ma_f0 = Vec.load(v4f, lds_mask, [grp * fx.Index(4)])
                mb_f0 = Vec.load(v4f, lds_mask, [MK + grp * fx.Index(4)])
                lm_f0 = c_big_neg
                for i in range_constexpr(4):
                    lm_f0 = fx.Float32(
                        arith.MaxNumFOp(_raw(lm_f0), _raw(sa_f0[i] + fx.Float32(_raw(Vec(ma_f0)[i])))).result
                    )
                for i in range_constexpr(4):
                    lm_f0 = fx.Float32(
                        arith.MaxNumFOp(_raw(lm_f0), _raw(sb_f0[i] + fx.Float32(_raw(Vec(mb_f0)[i])))).result
                    )
                # Floor the fixed-max bound to a finite value: a fully-masked first pair
                # gives -inf -> exp2(+inf) -> NaN. maxnumf(.,0) keeps the alpha=1 fold.
                _mb[0] = fx.Float32(arith.MaxNumFOp(_raw(crossgrp_max(lm_f0)), _raw(c_zero)).result)
            idx2 = load_topk(TWO * MK)
            idx3 = load_topk(fx.Index(3) * MK)
            xinit = [c_big_neg, c_zero] + [c_zero_v4 for _ in range_constexpr(DT)] + [idx2, idx3]
            xres = xinit
            for pr, ia in range(fx.Index(0), fx.Index(NUM_TILES // 2), fx.Index(1), init=xinit):
                m_run = ia[0]
                l_run = ia[1]
                o_acc = [ia[2 + dt] for dt in range_constexpr(DT)]
                idxc = ia[2 + DT]
                idxd = ia[2 + DT + 1]  # pair pr+1 tile indices
                par = pr % TWO
                cur = (TWO * par) * LE
                curb = cur + LE
                mca = (TWO * par) * MK
                mcb = mca + MK
                npar = (pr + fx.Index(1)) % TWO
                nxt = (TWO * npar) * LE
                nxtb = nxt + LE
                mna = (TWO * npar) * MK
                mnb = mna + MK
                gpu.barrier()  # cur bufs visible; prev reads done (WAR for nxt)
                kv_na = gather_load(idxc)
                kv_nb = gather_load(idxd)
                gather_store(kv_na, idxc, nxt, mna)  # prestore pair pr+1 -> other bufs
                gather_store(kv_nb, idxd, nxtb, mnb)  # ds_writes overlap the PV below
                s_a = _qk_s(cur)
                s_b = _qk_s(curb)
                mask_a = Vec.load(v4f, lds_mask, [mca + grp * fx.Index(4)])
                mask_b = Vec.load(v4f, lds_mask, [mcb + grp * fx.Index(4)])
                base_a, base_b = _v32_base(cur, curb)
                trq = [_v32(base_a, base_b, dt) for dt in range_constexpr(PF_PV)]
                m_new, alpha, p_a, p_b, l_new = _softmax32(s_a, s_b, mask_a, mask_b, m_run, l_run)
                new_o = _pv_o_k32(base_a, base_b, trq, p_a, fx.Float32(1.0), p_b, alpha, o_acc)
                idxe = load_topk((pr * TWO + fx.Index(4)) * MK)
                idxf = load_topk((pr * TWO + fx.Index(5)) * MK)
                xres = yield [m_new, l_new] + list(new_o) + [idxe, idxf]
            m_run = xres[0]
            l_sum = crossgrp_sum(xres[1])
            o_acc = [xres[2 + dt] for dt in range_constexpr(DT)]
            m_f, l_t, lp, scal = _epi_scalars(m_run, l_sum)
            _write_out(m_f, l_t, lp, scal, o_acc)
        if const_expr(pv2x and not pstore):
            if const_expr(_fmax0):
                # CHEAP fixed-max: bound = crossgrp-max over the FIRST pair only (1 QK, no full
                # pre-pass). softmax is shift-invariant so any bound is mathematically exact; the
                # main pass then runs at nomax speed (alpha=1 folded) with far better precision
                # than no-max (pair-0 max already caps the bulk of the score range).
                gather_store(kv0, idxA0, fx.Index(0), fx.Index(0))
                kv1_f0 = gather_load(idxB0)
                gather_store(kv1_f0, idxB0, fx.Index(LDS_ELEMS), fx.Index(TILE_K))
                gpu.barrier()
                sa_f0 = _qk_s(fx.Index(0))
                sb_f0 = _qk_s(fx.Index(LDS_ELEMS))
                ma_f0 = Vec.load(v4f, lds_mask, [grp * fx.Index(4)])
                mb_f0 = Vec.load(v4f, lds_mask, [fx.Index(TILE_K) + grp * fx.Index(4)])
                lm_f0 = c_big_neg
                for i in range_constexpr(4):
                    lm_f0 = fx.Float32(
                        arith.MaxNumFOp(_raw(lm_f0), _raw(sa_f0[i] + fx.Float32(_raw(Vec(ma_f0)[i])))).result
                    )
                for i in range_constexpr(4):
                    lm_f0 = fx.Float32(
                        arith.MaxNumFOp(_raw(lm_f0), _raw(sb_f0[i] + fx.Float32(_raw(Vec(mb_f0)[i])))).result
                    )
                # Floor the fixed-max bound to a finite value (see the other fast_path):
                # a fully-masked first pair -> -inf -> NaN.
                _mb[0] = fx.Float32(arith.MaxNumFOp(_raw(crossgrp_max(lm_f0)), _raw(c_zero)).result)
                gpu.barrier()
            # 2-tile-batch PV K=32 (non-banded)
            idx2 = load_topk(fx.Index(2) * fx.Index(TILE_K))
            idx3 = load_topk(fx.Index(3) * fx.Index(TILE_K))
            xinit = (
                [c_big_neg, c_zero]
                + [c_zero_v4 for _ in range_constexpr(DT)]
                + list(kv0)
                + [idxA0, idxB0, idx2, idx3]
            )
            xres = xinit
            b1 = fx.Index(LDS_ELEMS)
            for pr, ia in range(fx.Index(0), fx.Index(NUM_TILES // 2), fx.Index(1), init=xinit):
                m_run = ia[0]
                l_run = ia[1]
                o_acc = [ia[2 + dt] for dt in range_constexpr(DT)]
                kv_a = [ia[2 + DT + c] for c in range_constexpr(V8PT)]
                idxa = ia[2 + DT + V8PT]
                idxb = ia[2 + DT + V8PT + 1]
                idxc = ia[2 + DT + V8PT + 2]
                idxd = ia[2 + DT + V8PT + 3]
                gpu.barrier()
                gather_store(kv_a, idxa, fx.Index(0), fx.Index(0))
                gpu.barrier()
                kv_b = gather_load(idxb)
                s_a = _qk_s(fx.Index(0))
                gather_store(kv_b, idxb, b1, fx.Index(TILE_K))
                gpu.barrier()
                kv_c = gather_load(idxc)
                s_b = _qk_s(b1)
                mask_a = Vec.load(v4f, lds_mask, [fx.Index(0) + grp * fx.Index(4)])
                mask_b = Vec.load(v4f, lds_mask, [fx.Index(TILE_K) + grp * fx.Index(4)])
                # pv_early: hoist PF_PV V32 tr16 reads ahead of softmax so their LDS-read
                # latency overlaps the pure-VALU softmax bridge (permlane-max + exp2).
                base_a, base_b = _v32_base(fx.Index(0), b1)
                trq = [_v32(base_a, base_b, dt) for dt in range_constexpr(PF_PV)]
                m_new, alpha, p_a, p_b, l_new = _softmax32(s_a, s_b, mask_a, mask_b, m_run, l_run)
                new_o = _pv_o_k32(base_a, base_b, trq, p_a, fx.Float32(1.0), p_b, alpha, o_acc)
                idxe = load_topk((pr * fx.Index(2) + fx.Index(4)) * fx.Index(TILE_K))
                idxf = load_topk((pr * fx.Index(2) + fx.Index(5)) * fx.Index(TILE_K))
                xres = yield [m_new, l_new] + list(new_o) + list(kv_c) + [idxc, idxd, idxe, idxf]
            m_run = xres[0]
            l_sum = crossgrp_sum(xres[1])
            o_acc = [xres[2 + dt] for dt in range_constexpr(DT)]
            m_f, l_t, lp, scal = _epi_scalars(m_run, l_sum)
            _write_out(m_f, l_t, lp, scal, o_acc)
        if const_expr(not pv2x):
            loop_results = init
            for t, iter_args in range(fx.Index(0), loop_ub, fx.Index(1), init=init):
                m_run = iter_args[0]
                l_run = iter_args[1]
                o_acc = [iter_args[2 + dt] for dt in range_constexpr(DT)]
                kv_cur = [iter_args[2 + DT + c] for c in range_constexpr(V8PT)]
                if const_expr(not banded):
                    idxA = iter_args[2 + DT + V8PT]
                    idxB = iter_args[2 + DT + V8PT + 1]

                if const_expr(single_buffer):
                    buf_off = fx.Index(0)
                    mbuf_off = fx.Index(0)
                else:
                    buf_off = (t % fx.Index(2)) * fx.Index(LDS_ELEMS)
                    mbuf_off = (t % fx.Index(2)) * fx.Index(TILE_K)
                if const_expr(banded):
                    gather_store_b(kv_cur, t, buf_off, mbuf_off)
                    gpu.barrier()
                    kv_next = gather_load_b(t + fx.Index(1))
                else:
                    gather_store(kv_cur, idxA, buf_off, mbuf_off)
                    gpu.barrier()
                    kv_next = gather_load(idxB)
                    idxB2 = load_topk((t + fx.Index(2)) * fx.Index(TILE_K))

                mask4 = Vec.load(v4f, lds_mask, [mbuf_off + grp * fx.Index(4)])
                s4 = _qk_s(buf_off)
                if const_expr(pv_early):
                    # Hoist the PF_PV tr16 reads ahead of the softmax permlane bridge so
                    # their LDS-read latency overlaps the pure-VALU bridge.
                    pv_base, trq = _pv_prefetch(buf_off)
                    m_new, alpha, p, l_new = _softmax(s4, mask4, m_run, l_run)
                    new_o = _pv_compute(pv_base, trq, p, alpha, o_acc)
                else:
                    m_new, alpha, p, l_new = _softmax(s4, mask4, m_run, l_run)
                    new_o = _pv_o(buf_off, p, alpha, o_acc)
                if const_expr(single_buffer):
                    # WAR: all reads of the single KV buffer (QK bv + PV tr16) must retire
                    # before the next iteration overwrites it with tile t+1's KV.
                    gpu.barrier()

                _carry = [m_new, l_new] + list(new_o) + list(kv_next)
                if const_expr(not banded):
                    _carry = _carry + [idxB, idxB2]
                loop_results = yield _carry

            # head = head_wave_base + lo (single head per lane, replicated across grp).
            if peel_last:
                # Peeled last KV tile: overlap the epilogue scalar chain with PV.
                m_run = loop_results[0]
                l_run_p = loop_results[1]
                o_acc = [loop_results[2 + dt] for dt in range_constexpr(DT)]
                kv_cur = [loop_results[2 + DT + c] for c in range_constexpr(V8PT)]
                t_last = t_last_dyn  # runtime (total_tiles-1) when dyn_pool, else NUM_TILES-1
                # banded => single_buffer is always False here -> double-buffered offset.
                buf_off = (t_last % fx.Index(2)) * fx.Index(LDS_ELEMS)
                mbuf_off = (t_last % fx.Index(2)) * fx.Index(TILE_K)
                gather_store_b(kv_cur, t_last, buf_off, mbuf_off)
                gpu.barrier()
                mask4 = Vec.load(v4f, lds_mask, [mbuf_off + grp * fx.Index(4)])
                s4 = _qk_s(buf_off)
                if const_expr(pv_early):
                    pv_base, trq = _pv_prefetch(buf_off)
                m_new, alpha, p, l_new = _softmax(s4, mask4, m_run, l_run_p)
                # Cross-grp sum + scale chain issued BEFORE the PV MFMAs: independent of
                # o_acc, so its permlane/exp2/rcp latency hides behind the PV drain.
                l_sum = crossgrp_sum(l_new)
                m_f, l_t, lp, scal = _epi_scalars(m_new, l_sum)
                if const_expr(fused_store):
                    # scale+pack+store fused into the PV loop -> O stores spread across
                    # the MFMA drain (only the LSE store remains after PV).
                    if const_expr(pv_early):
                        _pv_compute(pv_base, trq, p, alpha, o_acc, store_scal=scal)
                    else:
                        _pv_o(buf_off, p, alpha, o_acc, store_scal=scal)
                    _write_lse(m_f, l_t, lp)
                else:
                    if const_expr(pv_early):
                        new_o = _pv_compute(pv_base, trq, p, alpha, o_acc)
                    else:
                        new_o = _pv_o(buf_off, p, alpha, o_acc)
                    _write_out(m_f, l_t, lp, scal, new_o)
            else:
                m_run = loop_results[0]
                # l_run is a per-grp partial (deferred kv-sum) -> cross-grp sum ONCE here.
                l_sum = crossgrp_sum(loop_results[1])
                o_acc = [loop_results[2 + dt] for dt in range_constexpr(DT)]
                m_f, l_t, lp, scal = _epi_scalars(m_run, l_sum)
                _write_out(m_f, l_t, lp, scal, o_acc)

    _launch_kw = {}

    @flyc.jit
    def launch(Q, KV, TOPK, SINK, O, LSE, T, H, NKV, stream):
        allocator.finalized = False
        with ir.InsertionPoint(CompilationContext.get_current().gpu_module_body):
            allocator.finalize()
        gy = fx.Index(H) // fx.Index(BLOCK_H)
        k_fn(Q, KV, TOPK, SINK, O, LSE, T, H, NKV, **_launch_kw).launch(
            grid=(fx.Index(T), gy, 1), block=(THREADS, 1, 1), stream=stream
        )

    def _compile(*a):
        return flyc.compile(launch, *a)

    launch.compile = _compile
    return launch


# ============================================================================
# build / compile caches
# ============================================================================

SparseMLASpec = collections.namedtuple(
    "SparseMLASpec",
    [
        "topk_len",
        "scale",
        "has_sink",
        "banded",
        "qk2",
        "pool_P",
        "pool_cr",
        "pf_pv",
        "single_buffer",
        "xcd_remap",
        "peel",
        "hoist_sink",
        "fused_store",
        "pv_early",
        "pf_qk",
        "dyn_pool",
        "bh",
        "pstore",
        "fast_path",
    ],
)
SparseMLASpec.__doc__ = """Full kernel specialization key.

Every field is a ``const_expr`` knob consumed by :func:`build_fwd`, so two specs
differing anywhere must compile to distinct binaries. ``fast_path`` is included
deliberately: upstream omits it from its compiled-artifact key, which would hand
back a stale binary if the flag were ever varied at runtime.
"""


@functools.lru_cache(maxsize=None)
def _build_cached(spec):
    """Trace and build the kernel for one specialization."""
    return build_fwd(
        spec.topk_len,
        spec.scale,
        has_sink=spec.has_sink,
        banded=spec.banded,
        qk2=spec.qk2,
        pool_P=spec.pool_P,
        pool_cr=spec.pool_cr,
        pf_pv=spec.pf_pv,
        single_buffer=spec.single_buffer,
        xcd_remap=spec.xcd_remap,
        peel=spec.peel,
        hoist_sink=spec.hoist_sink,
        fused_store=spec.fused_store,
        pv_early=spec.pv_early,
        pf_qk=spec.pf_qk,
        dyn_pool=spec.dyn_pool,
        bh=spec.bh,
        pstore=spec.pstore,
        fast_path=spec.fast_path,
    )


# Compiled artifacts cannot go through ``lru_cache``: compiling needs the live
# tensor arguments, which are unhashable. FlyDSL specializes on operand
# dtype/rank -- shapes arrive as scalar kernel arguments -- so keying on the
# specialization tuple alone is sufficient.
_COMPILED_CACHE: dict = {}


def get_compiled_fwd(spec, args):
    """Return the compiled launcher for ``spec``, compiling on first use."""
    compiled = _COMPILED_CACHE.get(spec)
    if compiled is None:
        compiled = _build_cached(spec).compile(*args)
        _COMPILED_CACHE[spec] = compiled
    return compiled
