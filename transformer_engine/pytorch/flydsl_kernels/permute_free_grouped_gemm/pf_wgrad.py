# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Permute-free MoE weight-gradient (wgrad) grouped GEMM in FlyDSL.

Contract: the gradient operand is the *block-padded* ``[R_block, N]`` buffer
and ``SORTED`` maps each padded slot to a received-token row::

    dW[e][n, k] = sum_{routed slot s of e, valid} grad[grad_base[e] + local(s), n]
                                                 * x[SORTED[s], k]

where ``grad_base[e] = block_start[e] * block_size_m`` and padding slots
(``SORTED[s] == num_recv_tokens``) contribute zero.

Each 32-slot step is DMA'd into LDS as ``[slot, feature]`` tiles and read back transposed with
``ds_read_tr16_b64``, so the slot becomes the MFMA contraction axis. The workgroup's warps fill
the LDS tile together, then each computes its own ``(block_n/warps_n) x (block_k/warps_k)``
sub-tile.
"""

from __future__ import annotations

import functools
import inspect

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly as fly_dialect
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import rocdl as _rocdl_ops
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import AddressSpace, PointerType, T

from ..tensor_shim import ptr_rsrc

__all__ = ["compile_moe_wgrad_v2", "WGRAD_BLOCK_M"]

# MFMA atom dims (16x16x32 bf16).
WMMA_M = 16  # output rows  -> N (grad feature)
WMMA_N = 16  # output cols  -> K (x feature)
WMMA_K = 32  # contraction  -> token-slot tile
C_FRAG = 4   # f32 values per lane for the C (dW) fragment
WARP_SIZE = 64

# Coalesced fill vector width (bf16 elems). 8 bf16 = 16 B = one global_load_dwordx4.
FILL_V = 8

WGRAD_BLOCK_M = 32  # contraction (slot) step; matches the align block_size
NUM_BUF = 3         # triple-buffered DMA pipeline (pipe_stages == 3)


def _align_up(x, align):
    return (x + align - 1) // align * align


# Older FlyDSL takes the raw buffer ops' ``aux`` as an i32 operand; newer, as an optional attribute.
_AUX_IS_OPERAND = (
    inspect.signature(_rocdl_ops.raw_ptr_buffer_store).parameters["aux"].kind
    is not inspect.Parameter.KEYWORD_ONLY
)


def _zero_soffset_aux():
    """Zero ``soffset`` (and ``aux``, where it is an operand) for a raw buffer op."""
    z = fx.Int32(0).ir_value()
    return (z, z) if _AUX_IS_OPERAND else (z,)


def raw_buffer_load(rsrc, offset, dtype):
    """Element-offset ``rocdl.raw_ptr_buffer_load``."""
    ty = dtype.ir_type
    byte_off = fx.Int32(offset) * (ty.width // 8)
    return dtype(
        _rocdl_ops.raw_ptr_buffer_load(ty, rsrc, byte_off.ir_value(), *_zero_soffset_aux())
    )


def buffer_load_i32(rsrc, offset):
    return raw_buffer_load(rsrc, offset, fx.Int32)


def raw_buffer_store(data, rsrc, offset):
    byte_off = fx.Int32(offset) * (type(data).width // 8)
    _rocdl_ops.raw_ptr_buffer_store(
        data.ir_value(), rsrc, byte_off.ir_value(), *_zero_soffset_aux()
    )



@functools.lru_cache(maxsize=None)
def compile_moe_wgrad_v2(
    *,
    block_n: int = 64,
    block_k: int = 64,
    warps_n: int = 1,
    warps_k: int = 1,
    accumulate: bool = False,
    swap_gather: bool = False,
    out_dtype: str = "bf16",
):
    # ``out_dtype="fp32"`` lets an fp32 ``main_grad`` accumulate in-kernel (no bf16 scratch).
    if out_dtype not in ("bf16", "fp32"):
        raise ValueError(f"moe_wgrad out_dtype must be 'bf16' or 'fp32', got {out_dtype!r}")
    # ``swap_gather`` (FC2): token-gather ``GRAD`` and walk ``X`` contiguously, the reverse of FC1,
    # so the same kernel emits the native ``dW2[E, H, F]`` layout without a transpose.
    if block_n % (warps_n * WMMA_M) != 0 or block_k % (warps_k * WMMA_N) != 0:
        raise ValueError("block_n/block_k must be multiples of warps_*16")

    n_threads = warps_n * warps_k * WARP_SIZE
    if (WGRAD_BLOCK_M * block_n) % (n_threads * FILL_V) != 0:
        raise ValueError("32*block_n must be a multiple of n_threads*FILL_V")
    if (WGRAD_BLOCK_M * block_k) % (n_threads * FILL_V) != 0:
        raise ValueError("32*block_k must be a multiple of n_threads*FILL_V")

    WN = block_n // warps_n          # per-warp grad-feature span
    WK = block_k // warps_k          # per-warp x-feature span
    M_STEPS = WN // WMMA_M           # grad-feature atoms per warp (MFMA-M)
    N_STEPS = WK // WMMA_N           # x-feature atoms   per warp (MFMA-N)
    NACC = M_STEPS * N_STEPS

    # The DMA fill can't pad LDS rows, so bank conflicts are broken by an XOR chunk swizzle instead.
    SG = block_n                     # grad LDS row stride (bf16 elems)
    SX = block_k                     # x LDS row stride
    CPR_G_SWZ = block_n // FILL_V    # feature chunks per grad row
    CPR_X_SWZ = block_k // FILL_V    # feature chunks per x row

    G_TILE_ELEMS = WGRAD_BLOCK_M * SG
    X_TILE_ELEMS = WGRAD_BLOCK_M * SX
    G_FILLS = (WGRAD_BLOCK_M * block_n) // (n_threads * FILL_V)
    X_FILLS = (WGRAD_BLOCK_M * block_k) // (n_threads * FILL_V)

    out_is_f32 = out_dtype == "fp32"
    KERNEL_NAME = (
        f"moe_wgrad_routelist_bf16_{block_n}x{block_k}_w{warps_n}x{warps_k}"
        f"{'_o32' if out_is_f32 else ''}{'_acc' if accumulate else ''}"
        f"{'_sg' if swap_gather else ''}_dsz_s3_dsa_v2"
    )

    # Hand-laid LDS offsets: buffer_load_lds needs a raw addrspace(3) global (see ``_gep_lds``).
    g_lds_off = 0
    x_lds_off = _align_up(g_lds_off + G_TILE_ELEMS * 2 * NUM_BUF, 16)
    LDS_TOTAL_BYTES = _align_up(x_lds_off + X_TILE_ELEMS * 2 * NUM_BUF, 128)
    LDS_SYM = KERNEL_NAME + "_lds"

    @flyc.kernel(known_block_size=[n_threads, 1, 1])
    def wgrad_kernel(
        dW: fx.Pointer,          # [E, N, K] bf16 output
        X: fx.Pointer,           # [num_recv_tokens, K] bf16 (received-token activations)
        GRAD: fx.Pointer,        # [R_block, N] bf16 (block-padded per-slot gradient)
        SORTED: fx.Pointer,      # [padded] i32 token row per slot (sentinel = num_recv_tokens)
        BLOCK_START: fx.Pointer,      # [E] i32 (block units)
        BLOCKS_PER_EXPERT: fx.Pointer,  # [E] i32
        GRAD_BASE: fx.Pointer,  # [E] i32 first-slot row = block_start[e] * block_size_m
        N: fx.Int32,
        K: fx.Int32,
        num_recv_tokens: fx.Int32,
        sorted_numel: fx.Int32,  # SORTED (sorted_slot_ids) element count, to bound its SRD
    ):
        c0 = fx.Index(0)

        dW_rsrc = ptr_rsrc(dW)
        x_rsrc = ptr_rsrc(X)
        grad_rsrc = ptr_rsrc(GRAD)
        # Bounded so slot-id look-ahead past the last expert reads hardware zero.
        sorted_rsrc = ptr_rsrc(SORTED, num_records_bytes=fx.Index(sorted_numel * 4))
        bstart_rsrc = ptr_rsrc(BLOCK_START)
        bpe_rsrc = ptr_rsrc(BLOCKS_PER_EXPERT)
        gbase_rsrc = ptr_rsrc(GRAD_BASE)

        smem_raw_ptr = _llvm.mlir_addressof(ir.Type.parse("!llvm.ptr<3>"), LDS_SYM)

        tid = fx.Int32(fx.thread_idx.x)
        n_tile = fx.Int32(fx.block_idx.x)   # along N (grad feature)
        k_tile = fx.Int32(fx.block_idx.y)   # along K (x feature)
        expert = fx.Int32(fx.block_idx.z)   # expert id

        wid = tid // WARP_SIZE
        lane = tid % WARP_SIZE
        wn_id = wid // warps_k                 # warp row (grad feature)
        wk_id = wid % warps_k                  # warp col (x feature)
        lane_n = lane % 16                     # MFMA C col (k_feat)
        lane_m_base = lane // 16               # 0..3
        tr_k_group = (lane % 16) // 4          # 0..3
        tr_col_sub = lane % 4                  # 0..3

        warp_n_base = wn_id * fx.Int32(WN)     # grad-feature col base of this warp
        warp_k_base = wk_id * fx.Int32(WK)     # x-feature col base of this warp

        # One alias scope per (operand, ring slot), noalias to the rest, so LLVM keeps other slots'
        # fills in flight instead of draining vmcnt to 0 before each transpose read.
        _ALIAS_DOMAIN = '#llvm.alias_scope_domain<id = "moe_wgrad_lds">'
        _SCOPE_IDS = tuple(
            [f"g{r}" for r in range(NUM_BUF)] + [f"x{r}" for r in range(NUM_BUF)]
        )

        def _scope_attr(ids):
            inner = ", ".join(
                f'#llvm.alias_scope<id = "{sid}", domain = {_ALIAS_DOMAIN}>'
                for sid in ids
            )
            return ir.Attribute.parse(f"[{inner}]")

        _MY_SCOPE = {sid: _scope_attr((sid,)) for sid in _SCOPE_IDS}
        _NOALIAS_SCOPE = {
            sid: _scope_attr(tuple(o for o in _SCOPE_IDS if o != sid))
            for sid in _SCOPE_IDS
        }

        def _g_sid(slot):
            return f"g{slot % NUM_BUF}"

        def _x_sid(slot):
            return f"x{slot % NUM_BUF}"

        def _scope_kw(sid):
            """alias/noalias metadata kwargs for one (operand, ring slot), or {}."""
            if sid is None:
                return {}
            return {
                "alias_scopes": _MY_SCOPE[sid],
                "noalias_scopes": _NOALIAS_SCOPE[sid],
            }

        N_idx = fx.Index(N)
        K_idx = fx.Index(K)
        nrecv_idx = fx.Index(num_recv_tokens)

        # Bound the gathered operand to [num_recv, feat] so padding/sentinel rows read hardware 0,
        # which zeroes their outer product (the DMA fill can't mask them in registers).
        if swap_gather:
            grad_rsrc = ptr_rsrc(GRAD, num_records_bytes=nrecv_idx * N_idx * 2)
        else:
            x_rsrc = ptr_rsrc(X, num_records_bytes=nrecv_idx * K_idx * 2)

        n_block_base = n_tile * block_n
        k_block_base = k_tile * block_k
        n_base_idx = fx.Index(n_block_base)
        k_base_idx = fx.Index(k_block_base)

        bstart = buffer_load_i32(bstart_rsrc, expert)
        nblocks = buffer_load_i32(bpe_rsrc, expert)
        gbase = buffer_load_i32(gbase_rsrc, expert)
        base_slot = fx.Index(bstart) * WGRAD_BLOCK_M
        num_slots = fx.Index(nblocks) * WGRAD_BLOCK_M

        # Rebase the contiguous operand's SRD per expert in i64 so its i32 offsets don't wrap past
        # 2 GiB. readfirstlane keeps the base in an SGPR; a VGPR base waterfalls (~3.7x slower).
        gbase_u = rocdl.readfirstlane(T.i32, gbase)
        _pool_ptr_ty = PointerType.get(
            elem_ty=fx.BFloat16.ir_type, address_space=AddressSpace.Global, alignment=16
        )
        if swap_gather:
            _contig_base = fx.Int64(fx.ptrtoint(X))
            x_rsrc = ptr_rsrc(
                fx.inttoptr(
                    _pool_ptr_ty, _contig_base + fx.Int64(gbase_u) * fx.Int64(K) * fx.Int64(2)
                )
            )
        else:
            _contig_base = fx.Int64(fx.ptrtoint(GRAD))
            grad_rsrc = ptr_rsrc(
                fx.inttoptr(
                    _pool_ptr_ty, _contig_base + fx.Int64(gbase_u) * fx.Int64(N) * fx.Int64(2)
                )
            )

        CPR_G = block_n // FILL_V
        CPR_X = block_k // FILL_V

        def _tile_idx(i, cpr):
            chunk = tid + fx.Int32(i * n_threads)
            slot_i32 = chunk // fx.Int32(cpr)
            feat_i32 = (chunk % fx.Int32(cpr)) * fx.Int32(FILL_V)
            return fx.Index(slot_i32), fx.Index(feat_i32), slot_i32, feat_i32

        def load_slot_ids(s_base_idx):
            g_ids = [
                buffer_load_i32(sorted_rsrc, base_slot + s_base_idx + _tile_idx(i, CPR_G)[0])
                for i in range_constexpr(G_FILLS)
            ]
            x_ids = [
                buffer_load_i32(sorted_rsrc, base_slot + s_base_idx + _tile_idx(i, CPR_X)[0])
                for i in range_constexpr(X_FILLS)
            ]
            return g_ids, x_ids

        def _dma_one(rsrc, ids, slot_base_idx, cpr, feat_base_idx, dim_idx, lds_off,
                     buf_byte, n_fills, clamp_row, sid=None):
            # Global->LDS DMA; the LDS base must be wave-uniform, and lane L lands at base + L*16B.
            for i in range_constexpr(n_fills):
                phys = tid + fx.Int32(i * n_threads)
                slot = phys // fx.Int32(cpr)
                chunk = phys % fx.Int32(cpr)
                slot_idx = fx.Index(slot)
                token = fx.Index(ids[i])
                valid = (slot_base_idx + slot_idx < num_slots) & (token < nrecv_idx)
                if const_expr(clamp_row):
                    # Expert-relative walk (SRD rebased): clamp overrun to row 0.
                    row_idx = valid.select(slot_base_idx + slot_idx, c0)
                else:
                    # Gather: look-ahead slots read token 0 (a valid row), so force them OOB too.
                    row_idx = valid.select(token, nrecv_idx)
                swz = slot & fx.Int32(cpr - 1)
                glob_feat = (chunk ^ swz) * fx.Int32(FILL_V)
                voff_elem = row_idx * dim_idx + feat_base_idx + fx.Index(glob_feat)
                voff_byte = fx.Int32(voff_elem * 2)
                lds_perlane = fx.Int32(lds_off) + buf_byte + phys * fx.Int32(FILL_V * 2)
                lds_base = rocdl.readfirstlane(T.i32, lds_perlane)
                rocdl.raw_ptr_buffer_load_lds(
                    rsrc, _gep_lds(smem_raw_ptr, lds_base), fx.Int32(FILL_V * 2),
                    voff_byte, fx.Int32(0), fx.Int32(0), fx.Int32(1),
                    **_scope_kw(sid),
                )

        def dma_fill(g_ids, x_ids, slot_base_idx, g_buf_byte, x_buf_byte, wbuf=None):
            # ``wbuf``: ring slot being filled (selects the alias scope).
            g_sid = _g_sid(wbuf) if wbuf is not None else None
            x_sid = _x_sid(wbuf) if wbuf is not None else None
            _dma_one(
                grad_rsrc, g_ids, slot_base_idx, CPR_G_SWZ, n_base_idx, N_idx,
                g_lds_off, g_buf_byte, G_FILLS, clamp_row=not swap_gather, sid=g_sid,
            )
            _dma_one(
                x_rsrc, x_ids, slot_base_idx, CPR_X_SWZ, k_base_idx, K_idx,
                x_lds_off, x_buf_byte, X_FILLS, clamp_row=swap_gather, sid=x_sid,
            )

        def _dma_barrier(keep=0):
            # Land the DMA'd tile (leaving ``keep`` loads in flight) and this wave's ds_reads.
            rocdl.s_waitcnt(vmcnt=keep, lgkmcnt=0)
            rocdl.s_barrier()

        mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(WMMA_M, WMMA_N, WMMA_K, fx.BFloat16))
        acc_ty = fx.Vector.make_type(C_FRAG, fx.Float32)

        def mfma(a, b, c):
            return fly_dialect.mma_atom_call_ssa([acc_ty], mma_atom, a, b, c)

        def compute(accs, g_buf_byte, x_buf_byte, dma_prefetch=None, slot=None):
            g_read_sid = _g_sid(slot) if slot is not None else None
            x_read_sid = _x_sid(slot) if slot is not None else None

            def read_a(mi):
                return _tr_read_frag_swz(
                    smem_raw_ptr, g_lds_off, SG, CPR_G_SWZ, warp_n_base, mi * WMMA_M,
                    lane_m_base, tr_k_group, tr_col_sub, g_buf_byte,
                    alias_kw=_scope_kw(g_read_sid),
                )

            def read_b(nj):
                return _tr_read_frag_swz(
                    smem_raw_ptr, x_lds_off, SX, CPR_X_SWZ, warp_k_base, nj * WMMA_N,
                    lane_m_base, tr_k_group, tr_col_sub, x_buf_byte,
                    alias_kw=_scope_kw(x_read_sid),
                )

            new_accs = [None] * NACC
            b_frags = [None] * N_STEPS

            if const_expr(dma_prefetch is not None):
                # All reads before the next DMA, else LLVM puts vmcnt(0) before the first ds_read.
                a_frags = [read_a(mi) for mi in range_constexpr(M_STEPS)]
                for nj in range_constexpr(N_STEPS):
                    b_frags[nj] = read_b(nj)
                rocdl.sched_barrier(0)
                dma_prefetch()
                rocdl.sched_barrier(0)
                rocdl.s_setprio(1)
                for mi in range_constexpr(M_STEPS):
                    for nj in range_constexpr(N_STEPS):
                        idx = mi * N_STEPS + nj
                        new_accs[idx] = mfma(a_frags[mi], b_frags[nj], accs[idx])
                    rocdl.sched_barrier(0)
                rocdl.s_setprio(0)
                return new_accs

            # A is read one fragment ahead; B is read once, during mi == 0, and then reused.
            a_next = read_a(0)
            b_frags[0] = read_b(0)
            rocdl.s_setprio(1)
            for mi in range_constexpr(M_STEPS):
                a_cur = a_next
                if const_expr(mi + 1 < M_STEPS):
                    a_next = read_a(mi + 1)
                for nj in range_constexpr(N_STEPS):
                    if const_expr(mi == 0 and nj + 1 < N_STEPS):
                        b_frags[nj + 1] = read_b(nj + 1)
                    idx = mi * N_STEPS + nj
                    new_accs[idx] = mfma(a_cur, b_frags[nj], accs[idx])
                rocdl.sched_barrier(0)
            rocdl.s_setprio(0)
            return new_accs

        acc_init = [fx.Vector.filled(C_FRAG, 0.0, fx.Float32) for _ in range(NACC)]

        # NUM_BUF-slot LDS ring with prefetch distance D. The loop is unrolled by NUM_BUF so every
        # buffer offset is a compile-time constant. KEEP leaves the newest tile's DMA in flight.
        D = NUM_BUF - 1
        PER_TILE_DMA = G_FILLS + X_FILLS
        KEEP = (NUM_BUF - 2) * PER_TILE_DMA

        def g_byte(j):
            return fx.Int32((j % NUM_BUF) * G_TILE_ELEMS * 2)

        def x_byte(j):
            return fx.Int32((j % NUM_BUF) * X_TILE_ELEMS * 2)

        ID_SET = G_FILLS + X_FILLS

        # Slot ids are loaded NUM_BUF tiles ahead to hide the SORTED-load latency.
        def pack_idsets(sets):
            out = []
            for g_ids, x_ids in sets:
                out += g_ids + x_ids
            return out

        def unpack_idsets(args, base):
            sets = []
            for s in range_constexpr(NUM_BUF):
                off = base + s * ID_SET
                g_ids = [args[off + i] for i in range(G_FILLS)]
                x_ids = [args[off + G_FILLS + i] for i in range(X_FILLS)]
                sets.append((g_ids, x_ids))
            return sets

        def mk_prefetch(g_ids, x_ids, pf_slot, wbuf):
            return lambda: dma_fill(
                g_ids, x_ids, pf_slot, g_byte(wbuf), x_byte(wbuf), wbuf=wbuf
            )

        # Prologue: stage tiles 0..D-1.
        for j in range_constexpr(D):
            j_slot = fx.Index(j * WMMA_K)
            gidj, xidj = load_slot_ids(j_slot)
            dma_fill(gidj, xidj, j_slot, g_byte(j), x_byte(j), wbuf=j)
        _dma_barrier()

        init_sets = [
            load_slot_ids(fx.Index((D + j) * WMMA_K))
            for j in range_constexpr(NUM_BUF)
        ]

        ntiles = fx.Int32(num_slots) // WMMA_K
        group_end = fx.Index((ntiles // NUM_BUF) * (NUM_BUF * WMMA_K))

        for s_iv, state in range(
            c0, group_end, NUM_BUF * WMMA_K, init=acc_init + pack_idsets(init_sets)
        ):
            s_base = fx.Index(s_iv)
            accs = state[:NACC]
            cur_sets = unpack_idsets(state, NACC)

            # Sub-tile j reads buffer j and prefetches tile j+D into buffer (j+D) % NUM_BUF.
            for j in range_constexpr(NUM_BUF):
                g_ids_j, x_ids_j = cur_sets[j]
                pf_slot = s_base + (j + D) * WMMA_K
                accs = compute(
                    accs, g_byte(j), x_byte(j),
                    dma_prefetch=mk_prefetch(g_ids_j, x_ids_j, pf_slot, j + D),
                    slot=j,
                )
                _dma_barrier(KEEP)

            nxt_sets = [
                load_slot_ids(s_base + (NUM_BUF + D + j) * WMMA_K)
                for j in range_constexpr(NUM_BUF)
            ]
            results = yield accs + pack_idsets(nxt_sets)

        accs = results[:NACC]

        # Tail: the last ``rem`` tiles are already staged in buffers 0..rem-1.
        _dma_barrier()
        rem = fx.Uint32(ntiles - (ntiles // NUM_BUF) * NUM_BUF)

        def _tail(j, accs):
            if const_expr(j < NUM_BUF - 1):
                if rem > j:
                    accs = _tail(j + 1, compute(accs, g_byte(j), x_byte(j), slot=j))
            return accs

        accs = _tail(0, accs)

        # Each dW element belongs to exactly one workgroup, so the ``accumulate`` RMW is race-free.
        out_ty = fx.Float32 if out_is_f32 else fx.BFloat16
        E_NK_row = fx.Index(expert) * N_idx * K_idx
        for mi in range_constexpr(M_STEPS):
            for nj in range_constexpr(N_STEPS):
                acc = fx.Vector(accs[mi * N_STEPS + nj])
                c_n = k_block_base + warp_k_base + fx.Int32(nj * WMMA_N) + lane_n
                c_n_idx = fx.Index(c_n)
                k_ok = c_n_idx < K_idx
                for ii in range_constexpr(C_FRAG):
                    n_out = (
                        n_block_base + warp_n_base + fx.Int32(mi * WMMA_M)
                        + lane_m_base * C_FRAG + fx.Int32(ii)
                    )
                    n_out_idx = fx.Index(n_out)
                    if (n_out_idx < N_idx) & k_ok:
                        val = acc[ii]  # f32 accumulator
                        out_off = E_NK_row + n_out_idx * K_idx + c_n_idx
                        if const_expr(accumulate):
                            val = val + raw_buffer_load(dW_rsrc, out_off, out_ty).to(fx.Float32)
                        raw_buffer_store(val.to(out_ty), dW_rsrc, out_off)

    @flyc.jit
    def launch_wgrad(
        dW: fx.Pointer,
        X: fx.Pointer,
        GRAD: fx.Pointer,
        SORTED: fx.Pointer,
        BLOCK_START: fx.Pointer,
        BLOCKS_PER_EXPERT: fx.Pointer,
        GRAD_BASE: fx.Pointer,
        N: fx.Int32,
        K: fx.Int32,
        num_recv_tokens: fx.Int32,
        num_experts: fx.Int32,
        sorted_numel: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            _llvm.GlobalOp(
                global_type=ir.Type.parse(f"!llvm.array<{LDS_TOTAL_BYTES} x i8>"),
                sym_name=LDS_SYM,
                linkage=ir.Attribute.parse("#llvm.linkage<external>"),
                addr_space=3,
                alignment=1024,
            )
        gx = (N + block_n - 1) // block_n
        gy = (K + block_k - 1) // block_k
        gz = num_experts
        wgrad_kernel._func.__name__ = KERNEL_NAME
        wgrad_kernel(
            dW, X, GRAD, SORTED, BLOCK_START, BLOCKS_PER_EXPERT, GRAD_BASE,
            N, K, num_recv_tokens, sorted_numel,
        ).launch(grid=(gx, gy, gz), block=(n_threads, 1, 1), stream=stream)

    return launch_wgrad


def _gep_lds(base_ptr, byte_i32):
    """Byte-offset GEP off the LDS global; ``buffer_load_lds`` needs one to set M0 correctly."""
    return _llvm.getelementptr(
        ir.Type.parse("!llvm.ptr<3>"), base_ptr, [fx.Int32(byte_i32).ir_value()],
        [-(2 ** 31)], T.i8, None,
    )


def _tr_read_frag_swz(
    smem_base, lds_off, stride, cpr, warp_col_base, col_const,
    lane_m_base, tr_k_group, tr_col_sub, buf_byte, alias_kw=None,
):
    """Transpose-read one fragment, undoing the fill's ``chunk ^ (slot & (cpr - 1))`` swizzle."""
    col_run = warp_col_base + tr_col_sub * fx.Int32(4)
    feat_log = col_run + fx.Int32(col_const)          # logical feature column
    chunk = feat_log // fx.Int32(FILL_V)
    within = feat_log % fx.Int32(FILL_V)
    row_lo = lane_m_base * fx.Int32(8) + tr_k_group   # contraction slot (0..31)

    def _read(slot):
        swz = slot & fx.Int32(cpr - 1)
        phys_chunk = chunk ^ swz
        phys_feat = phys_chunk * fx.Int32(FILL_V) + within
        elem = slot * fx.Int32(stride) + phys_feat
        byte = elem * fx.Int32(2) + fx.Int32(lds_off) + buf_byte
        raw = rocdl.ds_read_tr16_b64(
            T.vec(4, T.bf16), _gep_lds(smem_base, byte), **(alias_kw or {})
        ).result
        return fx.Vector(raw, (4,), fx.BFloat16)

    lo = _read(row_lo)
    hi = _read(row_lo + fx.Int32(4))
    return lo.shuffle(hi, [0, 1, 2, 3, 4, 5, 6, 7])

