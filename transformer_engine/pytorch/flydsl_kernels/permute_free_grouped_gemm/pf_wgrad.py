# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Permute-free MoE weight-gradient (wgrad) grouped GEMM in FlyDSL.

Contract: the gradient operand is the *block-padded* ``[em_max, N]`` buffer
and ``SORTED`` maps each padded slot to a received-token row::

    dW[e][n, k] = sum_{routed slot s of e, valid} grad[grad_base[e] + local(s), n]
                                                 * x[SORTED[s], k]

where ``grad_base[e] = block_start[e] * block_size_m`` is the expert's block-padded first-slot
row, ``local(s)`` is the slot's offset within expert ``e``'s block-padded range, and padding
slots (``SORTED[s] == num_recv_tokens``) are masked to zero. The grad walk is a plain
contiguous row scan (no ``SORTED`` indirection); ``SORTED`` is only consulted for the ``x``
gather token and the padding mask.

The contraction tile is staged through LDS and transposed on-read:

  1. Coalesced fill: each 32-slot contraction step loads ``grad[slot, n_feat]`` and
     ``x[token(slot), k_feat]`` into LDS as ``[slot(row), feature(col)]`` tiles with
     wide vector loads along the contiguous feature axis.
  2. Hardware transpose-read: ``ds_read_tr16_b64`` reads the ``[slot, feature]`` tile
     transposed into the MFMA A/B fragment layout ``[feature, slot]`` -- so the token
     slot becomes the matrix-core contraction axis with no VGPR shuffle and no strided
     global gather.

A workgroup of ``warps_n x warps_k`` warps computes one ``block_n x block_k`` output
tile. All warps **cooperatively fill** the shared LDS contraction tile once per step
(amortizing the global gather), then each warp owns a ``(block_n/warps_n) x
(block_k/warps_k)`` sub-tile of 16x16 MFMA atoms, transpose-reading its own feature
columns out of the shared tile. ``warps_n = warps_k = 1`` reduces to the single-warp v2.

Fixed configuration (no runtime toggles):
  - bf16 inputs, bf16 output (FC1: compact grad + token-gathered ``x``)
  - optional ``accumulate``: overwrite (default) or read-modify-write into ``dW``
  - DMA + XOR chunk swizzle fill, 3-stage LDS pipeline, LLVM DMA alias scopes
"""

from __future__ import annotations

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import scf
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, const_expr, gpu, ptrtoint, range_constexpr, rocdl, vector
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch
from flydsl.utils.smem_allocator import SmemAllocator

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


@functools.lru_cache(maxsize=None)
def compile_moe_wgrad_v2(
    *,
    block_n: int = 64,
    block_k: int = 64,
    warps_n: int = 1,
    warps_k: int = 1,
    accumulate: bool = False,
):
    if block_n % (warps_n * WMMA_M) != 0 or block_k % (warps_k * WMMA_N) != 0:
        raise ValueError("block_n/block_k must be multiples of warps_*16")

    n_threads = warps_n * warps_k * WARP_SIZE
    if (WGRAD_BLOCK_M * block_n) % (n_threads * FILL_V) != 0:
        raise ValueError("32*block_n must be a multiple of n_threads*FILL_V")
    if (WGRAD_BLOCK_M * block_k) % (n_threads * FILL_V) != 0:
        raise ValueError("32*block_k must be a multiple of n_threads*FILL_V")

    gpu_arch = get_rocm_arch()
    WN = block_n // warps_n          # per-warp grad-feature span
    WK = block_k // warps_k          # per-warp x-feature span
    M_STEPS = WN // WMMA_M           # grad-feature atoms per warp (MFMA-M)
    N_STEPS = WK // WMMA_N           # x-feature atoms   per warp (MFMA-N)
    NACC = M_STEPS * N_STEPS

    # DMA fill writes each lane's 16B contiguously into LDS (no per-row pad possible),
    # so the swizzle path uses an un-padded stride and breaks bank conflicts with an
    # XOR chunk-swizzle instead (see the fill + transpose-read below).
    SG = block_n                     # grad LDS row stride (bf16 elems)
    SX = block_k                     # x LDS row stride
    # Swizzle granule = FILL_V bf16 (one 16B DMA unit). The XOR maps the feature chunk
    # index with the slot so consecutive contraction slots land on distinct banks.
    CPR_G_SWZ = block_n // FILL_V    # feature chunks per grad row
    CPR_X_SWZ = block_k // FILL_V    # feature chunks per x row

    G_TILE_ELEMS = WGRAD_BLOCK_M * SG
    X_TILE_ELEMS = WGRAD_BLOCK_M * SX
    G_FILLS = (WGRAD_BLOCK_M * block_n) // (n_threads * FILL_V)
    X_FILLS = (WGRAD_BLOCK_M * block_k) // (n_threads * FILL_V)

    KERNEL_NAME = (
        f"moe_wgrad_routelist_bf16_{block_n}x{block_k}_w{warps_n}x{warps_k}"
        f"{'_acc' if accumulate else ''}_dsz_s3_dsa_v2"
    )

    # LDS allocation: NUM_BUF-buffered grad tile + x tile (2 bytes/bf16).
    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem")
    g_lds_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = g_lds_off + G_TILE_ELEMS * 2 * NUM_BUF
    x_lds_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = x_lds_off + X_TILE_ELEMS * 2 * NUM_BUF
    # The DMA path backs LDS with a raw llvm addrspace(3) global (buffer_load_lds needs a
    # real global for M0; the memref allocator does not work). Same offsets/size.
    LDS_TOTAL_BYTES = allocator._align(allocator.ptr, 128)
    LDS_SYM = KERNEL_NAME + "_lds"

    @flyc.kernel(known_block_size=[n_threads, 1, 1])
    def wgrad_kernel(
        dW: fx.Pointer,          # [E, N, K] bf16 output
        X: fx.Pointer,           # [num_recv_tokens, K] bf16 (received-token activations)
        GRAD: fx.Pointer,        # [em_max, N] bf16 (block-padded per-slot gradient)
        SORTED: fx.Pointer,      # [padded] i32 received-token row per slot (sentinel = num_recv_tokens)
        BLOCK_START: fx.Pointer,      # [E] i32 (block units)
        BLOCKS_PER_EXPERT: fx.Pointer,  # [E] i32
        GRAD_BASE: fx.Pointer,  # [E] i32 (block-padded first-slot row = block_start[e] * block_size_m)
        N: fx.Int32,
        K: fx.Int32,
        num_recv_tokens: fx.Int32,
    ):
        bf16 = T.bf16
        c0 = arith.constant(0, index=True)

        dW_rsrc = ptr_rsrc(dW)
        x_rsrc = ptr_rsrc(X)
        grad_rsrc = ptr_rsrc(GRAD)
        sorted_rsrc = ptr_rsrc(SORTED)
        bstart_rsrc = ptr_rsrc(BLOCK_START)
        bpe_rsrc = ptr_rsrc(BLOCKS_PER_EXPERT)
        gbase_rsrc = ptr_rsrc(GRAD_BASE)

        # DMA path: raw addrspace(3) global backs LDS; reads + DMA both GEP off it.
        smem_raw_ptr = _llvm.mlir_addressof(ir.Type.parse("!llvm.ptr<3>"), LDS_SYM)

        tid = fx.Int32(gpu.thread_id("x"))
        n_tile = fx.Int32(gpu.block_id("x"))   # along N (grad feature)
        k_tile = fx.Int32(gpu.block_id("y"))   # along K (x feature)
        expert = fx.Int32(gpu.block_id("z"))   # expert id

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

        # One alias scope per (operand, ring slot): the grad tile and the x tile each own
        # NUM_BUF disjoint LDS byte ranges. A fill of slot r and a read of slot r share a
        # scope so their RAW dependency survives; every other pair (different slot, or the
        # other operand) is marked noalias, which is what lets SIInsertWaitcnts keep older
        # in-flight fills streaming instead of draining vmcnt to 0 before each transpose read.
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

        N_idx = arith.index_cast(T.index, N)
        K_idx = arith.index_cast(T.index, K)
        nrecv_idx = arith.index_cast(T.index, num_recv_tokens)

        # DMA fill cannot mask padding slots in registers, so bound the token-gathered
        # ``x`` operand's resource to its real [num_recv, K] extent: the sentinel token
        # (== num_recv) and any pipeline overrun then read out-of-bounds -> hardware
        # returns 0. A zeroed gathered column makes the padding slot's outer product
        # ``grad (x) 0 == 0``, so the paired contiguous-walk grad operand may safely read
        # garbage (clamped to row 0) for those slots.
        x_addr_i64 = arith.index_cast(T.i64, ptrtoint(X))
        x_nrec_bytes = nrecv_idx * K_idx * arith.index(2)
        x_rsrc = buffer_ops.create_buffer_resource_from_addr(
            x_addr_i64, num_records_bytes=x_nrec_bytes
        )

        n_block_base = n_tile * block_n
        k_block_base = k_tile * block_k
        n_base_idx = arith.index_cast(T.index, n_block_base)
        k_base_idx = arith.index_cast(T.index, k_block_base)

        # Per-expert routed-slot range. ``base_slot`` is the wgrad-align slot offset into
        # ``SORTED`` (holds the received-token row for the ``x`` gather); ``grad_base_idx`` is
        # expert ``e``'s block-padded first-slot row into the ``[em_max, N]`` grad buffer.
        bstart = buffer_load_i32(bstart_rsrc, expert)
        nblocks = buffer_load_i32(bpe_rsrc, expert)
        gbase = buffer_load_i32(gbase_rsrc, expert)
        base_slot = arith.index_cast(T.index, bstart) * arith.index(WGRAD_BLOCK_M)
        num_slots = arith.index_cast(T.index, nblocks) * arith.index(WGRAD_BLOCK_M)
        grad_base_idx = arith.index_cast(T.index, gbase)

        CPR_G = block_n // FILL_V
        CPR_X = block_k // FILL_V

        def _tile_idx(i, cpr):
            chunk = tid + fx.Int32(i * n_threads)
            slot_i32 = chunk // fx.Int32(cpr)
            feat_i32 = (chunk % fx.Int32(cpr)) * fx.Int32(FILL_V)
            slot_idx = arith.index_cast(T.index, slot_i32)
            feat_idx = arith.index_cast(T.index, feat_i32)
            return slot_idx, feat_idx, slot_i32, feat_i32

        # ``load_slot_ids`` issues the (indirect) ``sorted`` index loads. These are
        # prefetched *two* steps ahead and carried across the loop as iter_args.
        def load_slot_ids(s_base_idx):
            g_ids = [
                buffer_load_i32_idx(sorted_rsrc, base_slot + s_base_idx + _tile_idx(i, CPR_G)[0])
                for i in range_constexpr(G_FILLS)
            ]
            x_ids = [
                buffer_load_i32_idx(sorted_rsrc, base_slot + s_base_idx + _tile_idx(i, CPR_X)[0])
                for i in range_constexpr(X_FILLS)
            ]
            return g_ids, x_ids

        def _dma_one(rsrc, ids, slot_base_idx, cpr, feat_base_idx, dim_idx, lds_off,
                     buf_byte, n_fills, clamp_row, sid=None):
            # Issue the global->LDS DMA for one operand: each lane streams FILL_V bf16
            # from ``global[row, swizzled_feat]`` straight into contiguous LDS (no VGPR
            # staging). The LDS destination base is wave-uniform (readfirstlane); the
            # hardware spreads lane L to base + L*16B, reconstructing the contiguous
            # ``phys_linear * FILL_V`` layout the swizzled transpose read expects.
            for i in range_constexpr(n_fills):
                phys = tid + fx.Int32(i * n_threads)
                slot = phys // fx.Int32(cpr)
                chunk = phys % fx.Int32(cpr)
                slot_idx = arith.index_cast(T.index, slot)
                token = arith.index_cast(T.index, ids[i])
                in_range = arith.cmpi(
                    arith.CmpIPredicate.ult, slot_base_idx + slot_idx, num_slots
                )
                valid = arith.andi(
                    in_range, arith.cmpi(arith.CmpIPredicate.ult, token, nrecv_idx)
                )
                if const_expr(clamp_row):
                    # grad: contiguous route walk; clamp overrun to row 0 for fault safety
                    # (its padding contribution is cancelled by the zeroed x column).
                    row_idx = valid.select(grad_base_idx + slot_base_idx + slot_idx, c0)
                else:
                    # x: gather by received-token; sentinel/OOB row -> hardware 0.
                    row_idx = token
                swz = slot & fx.Int32(cpr - 1)
                glob_feat = (chunk ^ swz) * fx.Int32(FILL_V)
                glob_feat_idx = arith.index_cast(T.index, glob_feat)
                voff_elem = row_idx * dim_idx + feat_base_idx + glob_feat_idx
                voff_byte = arith.index_cast(T.i32, voff_elem * arith.index(2))
                lds_perlane = fx.Int32(lds_off) + buf_byte + phys * fx.Int32(FILL_V * 2)
                lds_base = rocdl.readfirstlane(T.i32, lds_perlane)
                rocdl.raw_ptr_buffer_load_lds(
                    rsrc, _gep_lds(smem_raw_ptr, lds_base), fx.Int32(FILL_V * 2),
                    voff_byte, fx.Int32(0), fx.Int32(0), fx.Int32(1),
                    **_scope_kw(sid),
                )

        def dma_fill(g_ids, x_ids, slot_base_idx, g_buf_byte, x_buf_byte, wbuf=None):
            # FC1: token-gather ``x`` (clamp_row=False); compact grad route walk (clamp_row=True).
            # ``wbuf`` is the Python ring-slot index this tile is being staged into (for alias scopes).
            g_sid = _g_sid(wbuf) if wbuf is not None else None
            x_sid = _x_sid(wbuf) if wbuf is not None else None
            _dma_one(
                grad_rsrc, g_ids, slot_base_idx, CPR_G_SWZ, n_base_idx, N_idx,
                g_lds_off, g_buf_byte, G_FILLS, clamp_row=True, sid=g_sid,
            )
            _dma_one(
                x_rsrc, x_ids, slot_base_idx, CPR_X_SWZ, k_base_idx, K_idx,
                x_lds_off, x_buf_byte, X_FILLS, clamp_row=False, sid=x_sid,
            )

        def _dma_barrier(keep=0):
            # DMA lands on vmcnt (global load); drain it before the workgroup barrier so
            # all waves observe the freshly-staged LDS tile. ``keep`` leaves that many
            # vmem ops in flight (graduated wait) -- for the 3-stage pipeline this keeps the
            # just-issued tile's DMA streaming across the barrier so it overlaps the next
            # iteration's MFMA too (only the tile read next is fully drained). lgkmcnt(0)
            # retires this wave's ds_reads before the buffer is recycled NUM_BUF steps on.
            asm = f"s_waitcnt vmcnt({keep}) lgkmcnt(0)\ns_barrier"
            _llvm.InlineAsmOp(
                res=None, operands_=[], asm_string=asm,
                constraints="", has_side_effects=True, is_align_stack=False,
            )

        def compute(accs, g_buf_byte, x_buf_byte, dma_prefetch=None, slot=None):
            # A fragments are read one-per-mi to keep VGPR pressure low. B fragments are
            # loop-invariant across mi, fetched once and interleaved with the mi==0 MFMAs
            # so their ds_read latency overlaps compute. The MFMA region runs at raised
            # priority so the matrix pipe stays fed while reads are in flight.
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
                # Burst *all* current-tile transpose reads first, then issue the next-tile
                # DMA. Keeping the reads ahead of the DMA in program order stops the compiler
                # from planting an ``s_waitcnt vmcnt(0)`` in front of the first ds_read;
                # the DMA then streams under the MFMA burst instead.
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
                        new_accs[idx] = rocdl.mfma_f32_16x16x32_bf16(
                            T.vec(C_FRAG, T.f32),
                            [a_frags[mi], b_frags[nj], accs[idx], 0, 0, 0],
                        )
                    rocdl.sched_barrier(0)
                rocdl.s_setprio(0)
                return new_accs

            a_next = read_a(0)
            b_frags[0] = read_b(0)  # first B needed for the very first MFMA
            rocdl.s_setprio(1)
            for mi in range_constexpr(M_STEPS):
                a_cur = a_next
                if const_expr(mi + 1 < M_STEPS):
                    a_next = read_a(mi + 1)  # prefetch next A while MFMA-ing current
                for nj in range_constexpr(N_STEPS):
                    # Prefetch the next B fragment during the first mi only; its ds_read
                    # then overlaps this step's MFMA and all later mi reuse the resident frag.
                    if const_expr(mi == 0 and nj + 1 < N_STEPS):
                        b_frags[nj + 1] = read_b(nj + 1)
                    idx = mi * N_STEPS + nj
                    new_accs[idx] = rocdl.mfma_f32_16x16x32_bf16(
                        T.vec(C_FRAG, T.f32), [a_cur, b_frags[nj], accs[idx], 0, 0, 0]
                    )
                rocdl.sched_barrier(0)
            rocdl.s_setprio(0)
            return new_accs

        acc_init = [
            arith.constant_vector(0.0, T.vec(C_FRAG, T.f32))
            for _ in range(NACC)
        ]
        acc_iter_args = NACC

        # NUM_BUF-stage ping-pong pipeline: MFMA the current LDS buffer while the next
        # step's global gather is in flight. Prefetch distance D == NUM_BUF - 1: stage D
        # tiles up front and keep the most-recently-issued tile's DMA in flight across the
        # barrier (graduated ``vmcnt``) so it overlaps two MFMA steps instead of one.
        D = NUM_BUF - 1
        PER_TILE_DMA = G_FILLS + X_FILLS
        # Graduated ``vmcnt`` kept in flight across each sub-tile's barrier: leaves the
        # just-issued tile's DMA streaming (only the tile read next is fully drained).
        KEEP = (NUM_BUF - 2) * PER_TILE_DMA
        acc_ty = T.vec(C_FRAG, T.f32)

        # Static NUM_BUF-buffer ring, distance-D, slot-loop unrolled by NUM_BUF. Sub-tile j
        # always reads buffer j and prefetches tile (base+j+D) into buffer (j+D) % NUM_BUF --
        # both Python constants, so every LDS byte offset is compile-time constant and the
        # backend can prove read(buf j) never aliases the in-flight DMA write(buf (j+D)).
        def g_byte(j):
            return fx.Int32((j % NUM_BUF) * G_TILE_ELEMS * 2)

        def x_byte(j):
            return fx.Int32((j % NUM_BUF) * X_TILE_ELEMS * 2)

        ID_SET = G_FILLS + X_FILLS

        # Slot ids are carried NUM_BUF-ahead so their ~500-cycle SORTED-load latency is
        # retired before ``dma_fill`` consumes them.
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

        # Prologue: stage tiles 0..D-1 into buffers 0..D-1 (distance-D), then drain fully.
        for j in range_constexpr(D):
            j_slot = arith.index(j * WMMA_K)
            gidj, xidj = load_slot_ids(j_slot)
            dma_fill(gidj, xidj, j_slot, g_byte(j), x_byte(j), wbuf=j)
        _dma_barrier()

        # Initial carried id-sets: iteration 0 prefetches tiles D..D+NUM_BUF-1.
        init_sets = [
            load_slot_ids(arith.index((D + j) * WMMA_K))
            for j in range_constexpr(NUM_BUF)
        ]

        ntiles = fx.Int32(arith.index_cast(T.i32, num_slots)) // fx.Int32(WMMA_K)
        group_end = arith.index_cast(
            T.index, (ntiles // fx.Int32(NUM_BUF)) * fx.Int32(NUM_BUF * WMMA_K)
        )
        step_big = arith.index(NUM_BUF * WMMA_K)

        loop = scf.ForOp(
            c0, group_end, step_big, iter_args=acc_init + pack_idsets(init_sets)
        )
        with ir.InsertionPoint(loop.body):
            s_base = loop.induction_variable
            accs = [loop.body.arguments[1 + i] for i in range(NACC)]
            cur_sets = unpack_idsets(loop.body.arguments, 1 + acc_iter_args)

            # sub-tile j: read buffer j (tile s_base+j), prefetch tile s_base+j+D into
            # buffer (j+D) % NUM_BUF using its carried id-set.
            for j in range_constexpr(NUM_BUF):
                g_ids_j, x_ids_j = cur_sets[j]
                pf_slot = s_base + arith.index((j + D) * WMMA_K)
                accs = compute(
                    accs, g_byte(j), x_byte(j),
                    dma_prefetch=mk_prefetch(g_ids_j, x_ids_j, pf_slot, j + D),
                    slot=j,
                )
                _dma_barrier(KEEP)

            nxt_sets = [
                load_slot_ids(s_base + arith.index((NUM_BUF + D + j) * WMMA_K))
                for j in range_constexpr(NUM_BUF)
            ]
            scf.YieldOp(accs + pack_idsets(nxt_sets))

        accs = [loop.results[i] for i in range(NACC)]

        # Tail: rem in 0..NUM_BUF-1 leftover tiles, already prefetched into buffers
        # 0..rem-1 by the last group. Drain any in-flight DMA, then consume read-only.
        _dma_barrier()
        rem = ntiles - (ntiles // fx.Int32(NUM_BUF)) * fx.Int32(NUM_BUF)

        def _tail(j, accs):
            if const_expr(j >= NUM_BUF - 1):
                return accs
            has_j = arith.cmpi(arith.CmpIPredicate.ugt, rem, fx.Int32(j))
            tif = scf.IfOp(has_j, results_=[acc_ty] * NACC, has_else=True)
            with ir.InsertionPoint(tif.then_block):
                a2 = compute(accs, g_byte(j), x_byte(j), dma_prefetch=None, slot=j)
                scf.YieldOp(_tail(j + 1, a2))
            with ir.InsertionPoint(tif.else_block):
                scf.YieldOp(accs)
            return [tif.results[i] for i in range(NACC)]

        accs = _tail(0, accs)

        # Epilogue: C[m=n_feat, n=k_feat], lane holds 4 rows. Each dW element is owned by
        # exactly one workgroup (grid = N x K x E over disjoint output tiles), so when
        # ``accumulate`` is set the read-modify-write into the destination is race-free.
        E_NK_row = arith.index_cast(T.index, expert) * N_idx * K_idx
        for mi in range_constexpr(M_STEPS):
            for nj in range_constexpr(N_STEPS):
                acc_idx = mi * N_STEPS + nj
                acc = accs[acc_idx]
                c_n = k_block_base + warp_k_base + fx.Int32(nj * WMMA_N) + lane_n
                c_n_idx = arith.index_cast(T.index, c_n)
                k_ok = arith.cmpi(arith.CmpIPredicate.ult, c_n_idx, K_idx)
                for ii in range_constexpr(C_FRAG):
                    n_out = (
                        n_block_base + warp_n_base + fx.Int32(mi * WMMA_M)
                        + lane_m_base * C_FRAG + fx.Int32(ii)
                    )
                    n_out_idx = arith.index_cast(T.index, n_out)
                    in_bounds = arith.andi(
                        arith.cmpi(arith.CmpIPredicate.ult, n_out_idx, N_idx), k_ok
                    )
                    store_if = scf.IfOp(in_bounds, results_=[], has_else=False)
                    with ir.InsertionPoint(store_if.then_block):
                        val = vector.extract(
                            acc, static_position=[ii], dynamic_position=[]
                        )  # f32 accumulator
                        out_off = E_NK_row + n_out_idx * K_idx + c_n_idx
                        if const_expr(accumulate):
                            prev = buffer_ops.buffer_load(
                                dW_rsrc, out_off, vec_width=1, dtype=bf16
                            )
                            val = arith.addf(val, arith.extf(T.f32, prev))
                        store_val = arith.truncf(bf16, val)
                        buffer_ops.buffer_store(store_val, dW_rsrc, out_off)
                        scf.YieldOp([])

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
            N, K, num_recv_tokens,
        ).launch(grid=(gx, gy, gz), block=(n_threads, 1, 1), stream=stream)

    return launch_wgrad


def _gep_lds(base_ptr, byte_i32):
    """GEP an LDS !llvm.ptr<3> by a runtime i8 byte offset off a real LDS base pointer.

    ``buffer_load_lds`` derives its M0 write base from the LDS pointer, which the backend
    only lowers correctly when the pointer is a GEP off a genuine addrspace(3) global.
    We route the reads through the same base so alias analysis ties the DMA writes to the
    transpose reads.
    """
    return _llvm.getelementptr(
        ir.Type.parse("!llvm.ptr<3>"), rocdl._to_ir(base_ptr), [rocdl._to_ir(byte_i32)],
        [-(2 ** 31)], T.i8, None,
    )


def _tr_read_frag_swz(
    smem_base, lds_off, stride, cpr, warp_col_base, col_const,
    lane_m_base, tr_k_group, tr_col_sub, buf_byte, alias_kw=None,
):
    """Swizzled transpose-read for the DMA fill (un-padded LDS).

    The DMA writes each contraction tile contiguously (stride == feature span), so bank
    conflicts on the transpose read are broken by an XOR *chunk* swizzle: the physical
    feature chunk of logical ``(slot, feat)`` is ``(feat // FILL_V) XOR (slot & (cpr - 1))``
    -- the *same* map the fill applies to the global gather column, so the read lands on
    exactly the element the DMA staged.
    """
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


def buffer_load_i32(rsrc, off_i32):
    return buffer_ops.buffer_load(rsrc, off_i32, vec_width=1, dtype=T.i32)


def buffer_load_i32_idx(rsrc, off_idx):
    return buffer_ops.buffer_load(rsrc, off_idx, vec_width=1, dtype=T.i32)
