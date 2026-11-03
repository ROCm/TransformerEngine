# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

"""FlyDSL MXFP4 TN GEMM for gfx1250.

D[M, N] = a[M, K] @ b[N, K]^T with K-contiguous packed FP4 operands and E8M0 scales along K. TE's
MXFP4 columnwise storage is a materialized transpose, so this one kernel serves TN/NN/NT.
Tiles are staged with TDM into a 3-stage LDS pipeline and C is written back through LDS by TDM.
"""

import functools

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr

from .exceptions import FlyDSLUnsupportedError

MXFP4_BLOCK_SIZE = 32
SCALE_WORD_K = 4 * MXFP4_BLOCK_SIZE
BLOCK_M = 256
BLOCK_N = 256
BLOCK_K = 128
NUM_WAVES = 8
WAVE_32 = 32
NUM_THREADS = NUM_WAVES * WAVE_32
NUM_STAGES = 3  # LDS pages per operand
# K-loop tile j is loaded by wave group j % 2, two waves per operand
TDM_GROUP_WAVES = NUM_WAVES // 2
TDM_WAVES = 2
PRO_SCALE_WAVES = 2
# Scales stream through SCALE_STAGES LDS pages of SCALE_GROUP K tiles
SCALE_GROUP = 8
SCALE_STAGES = 2
LDS_B_START = 64 * 1024
C_HALF_ROWS = BLOCK_M // 2  # C is stored in two halves
C_WAVE_ROWS = C_HALF_ROWS // NUM_WAVES
# Logical 16 B chunk c of row r is stored at chunk c ^ ((r >> 2) & 3) to avoid LDS bank conflicts
SWZ_CHUNKS = 4
SCHED_TAIL_MMAS = 8
# Each lane's scale words are made contiguous so they load as ds_load_b128
SCALE_LANE_ROWS_A = 32
SCALE_LANE_ROWS_B = 64
OUT_KINDS = {torch.float32: 0, torch.bfloat16: 1, torch.float16: 2}


# Scale words (one BLOCK_K tile of a row) stored [rows // block_rows][K_iters][block_rows],
# viewed by logical (row, k_tile)
def iter_major_scale_words(t, rows, words_per_row, block_rows, lane_rows):
    src = fx.get_iter(t)
    ptr = fx.recast_iter(fx.PointerType.get(fx.Int32.ir_type, src.memspace, 4), src)
    reps = block_rows // lane_rows
    return fx.make_view(
        ptr,
        fx.make_layout(
            (((lane_rows, reps), rows // block_rows), words_per_row),
            (((reps, 1), words_per_row * block_rows), block_rows),
        ),
    )


@flyc.kernel
def gemm_kernel(
    A: fx.Tensor,
    B: fx.Tensor,
    A_s: fx.Tensor,
    B_s: fx.Tensor,
    C: fx.Tensor,
    M: fx.Int32,
    N: fx.Int32,
    K: fx.Int32,
    K_iters: fx.Constexpr[int],
    OUT_KIND: fx.Constexpr[int],
):

    def as_words(t, rows, words_per_row):
        src = fx.get_iter(t)
        ptr = fx.recast_iter(fx.PointerType.get(fx.Int32.ir_type, src.memspace, 4), src)
        return fx.make_view(ptr, fx.make_layout((rows, words_per_row), (words_per_row, 1)))

    out_fx = (fx.Float32, fx.BFloat16, fx.Float16)[OUT_KIND]
    out_bytes = 4 if OUT_KIND == 0 else 2
    c_stage_bytes = C_HALF_ROWS * BLOCK_N * out_bytes

    tid = fx.thread_idx.x
    bid = fx.block_idx.x
    blocks_n = N // BLOCK_N
    bid_m = bid // blocks_n
    bid_n = bid % blocks_n

    # wave-uniform, so TDM descriptors derived from it stay in SGPRs
    lane = tid % 32
    wave = fx.Int32(fx.rocdl.readfirstlane(fx.Int32.ir_type, tid // 32))
    lane_mod16, khalf = lane % 16, lane // 16
    wm, wn = wave % 2, wave // 2

    k_words = K // 8
    A_words = as_words(A, M, k_words)
    B_words = as_words(B, N, k_words)

    # physical chunks of this lane's logical chunks khalf and khalf + 2 (same for all its rows)
    swz = (lane_mod16 // 4) % SWZ_CHUNKS
    chunk_lo = khalf ^ swz
    chunk_hi = chunk_lo ^ 2
    a_lds_lo = (wm * 16 + lane_mod16) * 16 + 4 * chunk_lo
    a_lds_hi = (wm * 16 + lane_mod16) * 16 + 4 * chunk_hi
    b_lds_lo = (wn * 16 + lane_mod16) * 16 + 4 * chunk_lo
    b_lds_hi = (wn * 16 + lane_mod16) * 16 + 4 * chunk_hi
    as_lds_base = (wm * 16 + lane_mod16) * (BLOCK_M // SCALE_LANE_ROWS_A)
    bs_lds_base = (wn * 16 + lane_mod16) * (BLOCK_N // SCALE_LANE_ROWS_B)

    # tile k lives in LDS page k % NUM_STAGES
    smem = fx.SharedAllocator(static=False)
    lds_A = smem.allocate(fx.Array[fx.Int32, NUM_STAGES * BLOCK_M * 16]).peek()
    # B starts at 64 KB so all its pages are reachable with immediate offsets from one base register
    smem.allocate(fx.Array[fx.Int32, LDS_B_START // 4 - NUM_STAGES * BLOCK_M * 16])
    lds_B = smem.allocate(fx.Array[fx.Int32, NUM_STAGES * BLOCK_N * 16]).peek()
    box_A = fx.make_layout((BLOCK_M, 16), (16, 1))
    sA = fx.make_view(
        lds_A.ptr, fx.make_layout(((BLOCK_M, 16), NUM_STAGES), ((16, 1), BLOCK_M * 16))
    )
    box_B = fx.make_layout((BLOCK_N, 16), (16, 1))
    sB = fx.make_view(
        lds_B.ptr, fx.make_layout(((BLOCK_N, 16), NUM_STAGES), ((16, 1), BLOCK_N * 16))
    )

    num_groups = (K_iters + SCALE_GROUP - 1) // SCALE_GROUP
    tail_tiles = K_iters - (num_groups - 1) * SCALE_GROUP
    lds_As = smem.allocate(fx.Array[fx.Int32, SCALE_STAGES * SCALE_GROUP * BLOCK_M]).peek()
    lds_Bs = smem.allocate(fx.Array[fx.Int32, SCALE_STAGES * SCALE_GROUP * BLOCK_N]).peek()

    # C is staged from LDS offset 0 after the K loop
    if smem.allocated_bytes < c_stage_bytes:
        smem.allocate(c_stage_bytes - smem.allocated_bytes)
    lds_C_ptr = fx.recast_iter(out_fx, smem.base_ptr)

    sA_base_ptr = fx.get_iter(sA)
    sB_base_ptr = fx.get_iter(sB)
    sAs_base_ptr = lds_As.ptr
    sBs_base_ptr = lds_Bs.ptr

    # K-loop TDM atoms over this wave's half tile; later K tiles are reached with imm_offset
    tdm_wave = wave % TDM_WAVES
    HALF_ROWS = BLOCK_M // TDM_WAVES
    half_box = fx.make_layout((HALF_ROWS, 16), (16, 1))
    gA_half = fx.make_view(
        fx.add_offset(fx.get_iter(A_words), (bid_m * BLOCK_M + tdm_wave * HALF_ROWS) * k_words),
        half_box,
    )
    gB_half = fx.make_view(
        fx.add_offset(fx.get_iter(B_words), (bid_n * BLOCK_N + tdm_wave * HALF_ROWS) * k_words),
        half_box,
    )
    tdm_A = fx.rocdl.make_tdm_atom(
        gA_half, [None, None], strides=[fx.Int64(k_words), None], num_warps=1
    )
    tdm_B = fx.rocdl.make_tdm_atom(
        gB_half, [None, None], strides=[fx.Int64(k_words), None], num_warps=1
    )

    def lds_half(base_ptr, page):
        return fx.make_view(
            fx.add_offset(base_ptr, page * BLOCK_M * 16 + tdm_wave * HALF_ROWS * 16), half_box
        )

    # prologue atoms: whole tiles and scales
    tdm_A1w, cA1w = fx.rocdl.cdna5.make_tiled_tdm_atom(
        fx.rocdl.TensorLoad(), A_words, box_A, (BLOCK_M, 16), num_warps=1
    )
    blk_cA1w = fx.zipped_divide(cA1w, (BLOCK_M, 16))[None, (bid_m, None)]
    tAsA1w, tAgA1w = fx.rocdl.cdna5.tdm_partition(tdm_A1w, 0, fx.make_layout(1, 1), sA, blk_cA1w)
    tdm_B1w, cB1w = fx.rocdl.cdna5.make_tiled_tdm_atom(
        fx.rocdl.TensorLoad(), B_words, box_B, (BLOCK_N, 16), num_warps=1
    )
    blk_cB1w = fx.zipped_divide(cB1w, (BLOCK_N, 16))[None, (bid_n, None)]
    tBsB1w, tBgB1w = fx.rocdl.cdna5.tdm_partition(tdm_B1w, 0, fx.make_layout(1, 1), sB, blk_cB1w)

    # scale group g of a row block is contiguous: 1-D TDM, bumped to group g with imm_offset
    def scale_ptr(t):
        src = fx.get_iter(t)
        return fx.recast_iter(fx.PointerType.get(fx.Int32.ir_type, src.memspace, 4), src)

    sA_blk = fx.add_offset(scale_ptr(A_s), bid_m * K_iters * BLOCK_M)
    sB_blk = fx.add_offset(scale_ptr(B_s), bid_n * K_iters * BLOCK_N)

    def scale_box(tiles, block_rows):
        return fx.make_layout(tiles * block_rows, 1)

    # prologue: scale groups 0 and 1, half per wave
    sc_wave = wave % PRO_SCALE_WAVES
    pro_tiles = min(SCALE_STAGES * SCALE_GROUP, K_iters)
    pro_half_A = pro_tiles * BLOCK_M // PRO_SCALE_WAVES
    pro_half_B = pro_tiles * BLOCK_N // PRO_SCALE_WAVES
    gAs_pro = fx.make_view(
        fx.add_offset(sA_blk, sc_wave * pro_half_A), fx.make_layout(pro_half_A, 1)
    )
    gBs_pro = fx.make_view(
        fx.add_offset(sB_blk, sc_wave * pro_half_B), fx.make_layout(pro_half_B, 1)
    )
    tdm_As_pro = fx.rocdl.make_tdm_atom(gAs_pro, [None], num_warps=1)
    tdm_Bs_pro = fx.rocdl.make_tdm_atom(gBs_pro, [None], num_warps=1)
    lAs_pro = fx.make_view(
        fx.add_offset(sAs_base_ptr, sc_wave * pro_half_A), fx.make_layout(pro_half_A, 1)
    )
    lBs_pro = fx.make_view(
        fx.add_offset(sBs_base_ptr, sc_wave * pro_half_B), fx.make_layout(pro_half_B, 1)
    )

    gAs_grp = fx.make_view(sA_blk, scale_box(SCALE_GROUP, BLOCK_M))
    gBs_grp = fx.make_view(sB_blk, scale_box(SCALE_GROUP, BLOCK_N))
    tdm_As_grp = fx.rocdl.make_tdm_atom(gAs_grp, [None], num_warps=1)
    tdm_Bs_grp = fx.rocdl.make_tdm_atom(gBs_grp, [None], num_warps=1)
    gAs_tail = fx.make_view(sA_blk, scale_box(tail_tiles, BLOCK_M))
    gBs_tail = fx.make_view(sB_blk, scale_box(tail_tiles, BLOCK_N))
    tdm_As_tail = fx.rocdl.make_tdm_atom(gAs_tail, [None], num_warps=1)
    tdm_Bs_tail = fx.rocdl.make_tdm_atom(gBs_tail, [None], num_warps=1)

    # whole tiles, for K-loop iterations that also load a scale group
    gA_full = fx.make_view(fx.add_offset(fx.get_iter(A_words), bid_m * BLOCK_M * k_words), box_A)
    gB_full = fx.make_view(fx.add_offset(fx.get_iter(B_words), bid_n * BLOCK_N * k_words), box_B)
    tdm_A_full = fx.rocdl.make_tdm_atom(
        gA_full, [None, None], strides=[fx.Int64(k_words), None], num_warps=1
    )
    tdm_B_full = fx.rocdl.make_tdm_atom(
        gB_full, [None, None], strides=[fx.Int64(k_words), None], num_warps=1
    )

    # C store: each wave writes C_WAVE_ROWS rows of each half
    c_wave_box = fx.make_layout((C_WAVE_ROWS, BLOCK_N), (BLOCK_N, 1))
    c_wave_row = bid_m * BLOCK_M + wave * C_WAVE_ROWS
    gC_wave = [
        fx.make_view(
            fx.add_offset(fx.get_iter(C), (c_wave_row + h * C_HALF_ROWS) * N + bid_n * BLOCK_N),
            c_wave_box,
        )
        for h in range(2)
    ]
    tdm_C = fx.rocdl.make_tdm_atom(
        gC_wave[0], [None, None], strides=[fx.Int64(N), None], num_warps=1
    )
    lds_C_wave = fx.make_view(fx.add_offset(lds_C_ptr, wave * C_WAVE_ROWS * BLOCK_N), c_wave_box)

    bC = fx.slice(fx.zipped_divide(C, (BLOCK_M, BLOCK_N)), (None, (bid_m, bid_n)))

    mma_atom = fx.make_mma_atom(fx.rocdl.WMMAScale(16, 16, 128, fx.Float4E2M1FN))
    tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((2, 4, 1), (1, 2, 0)))
    thr_mma = tiled_mma.thr_slice(tid)

    copy_atom_i32 = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)
    copy_atom_c = fx.make_copy_atom(
        fx.UniversalCopy32b() if OUT_KIND == 0 else fx.UniversalCopy16b(), out_fx
    )

    # double-buffered register fragments: tile k is read into set k % 2
    frag_A = [fx.make_rmem_tensor(fx.make_layout((8, 8, 1), (1, 8, 0)), fx.Int32) for _ in range(2)]
    frag_B = [fx.make_rmem_tensor(fx.make_layout((8, 4, 1), (1, 8, 0)), fx.Int32) for _ in range(2)]
    frag_sA = [fx.make_rmem_tensor((1, 8, 1), fx.Int32) for _ in range(2)]
    frag_sB = [fx.make_rmem_tensor((1, 4, 1), fx.Int32) for _ in range(2)]
    frag_acc = thr_mma.make_fragment_C(bC)

    # tiles issued per wave class, to derive each class's tensorcnt wait for tile t
    WAVE_CLASSES = ((0, 2), (2, 4), (4, NUM_WAVES))
    tdm_issued = [[0], [1], [0]]  # prologue ops

    def wait_tile(t):
        counts = [sum(1 for x in tiles if x > t) for tiles in tdm_issued]
        if counts[0] == counts[1] == counts[2]:
            fx.rocdl.s_wait_tensorcnt(counts[0])
        else:
            if wave < WAVE_CLASSES[0][1]:
                fx.rocdl.s_wait_tensorcnt(counts[0])
            else:
                if wave < WAVE_CLASSES[1][1]:
                    fx.rocdl.s_wait_tensorcnt(counts[1])
                else:
                    fx.rocdl.s_wait_tensorcnt(counts[2])

    # With scale_group g the group's waves load the A tile, A scale group g, the B tile and
    # B scale group g, still one TDM op per wave; the scale op shares tile j's wait slot
    def issue_tdm(j, scale_group=None):
        page = j % NUM_STAGES
        group = j % 2
        k_off = fx.Int64(j * BLOCK_K // 2)
        if scale_group is None:
            if wave // TDM_GROUP_WAVES == group:
                if wave % TDM_GROUP_WAVES < TDM_WAVES:
                    fx.copy(tdm_A, gA_half, lds_half(sA_base_ptr, page), imm_offset=k_off)
                else:
                    fx.copy(tdm_B, gB_half, lds_half(sB_base_ptr, page), imm_offset=k_off)
        else:
            g = scale_group
            spage = g % SCALE_STAGES
            tail = g == num_groups - 1 and tail_tiles < SCALE_GROUP
            atom_As, atom_Bs = (tdm_As_tail, tdm_Bs_tail) if tail else (tdm_As_grp, tdm_Bs_grp)
            g_As, g_Bs = (gAs_tail, gBs_tail) if tail else (gAs_grp, gBs_grp)
            tiles = tail_tiles if tail else SCALE_GROUP
            l_As = fx.make_view(
                fx.add_offset(sAs_base_ptr, spage * SCALE_GROUP * BLOCK_M),
                scale_box(tiles, BLOCK_M),
            )
            l_Bs = fx.make_view(
                fx.add_offset(sBs_base_ptr, spage * SCALE_GROUP * BLOCK_N),
                scale_box(tiles, BLOCK_N),
            )
            s_offA = fx.Int64(g * SCALE_GROUP * BLOCK_M * 4)
            s_offB = fx.Int64(g * SCALE_GROUP * BLOCK_N * 4)
            if wave // TDM_GROUP_WAVES == group:
                gw = wave % TDM_GROUP_WAVES
                if gw < 2:
                    if gw == 0:
                        fx.copy(
                            tdm_A_full,
                            gA_full,
                            fx.make_view(fx.add_offset(sA_base_ptr, page * BLOCK_M * 16), box_A),
                            imm_offset=k_off,
                        )
                    else:
                        fx.copy(atom_As, g_As, l_As, imm_offset=s_offA)
                else:
                    if gw == 2:
                        fx.copy(
                            tdm_B_full,
                            gB_full,
                            fx.make_view(fx.add_offset(sB_base_ptr, page * BLOCK_N * 16), box_B),
                            imm_offset=k_off,
                        )
                    else:
                        fx.copy(atom_Bs, g_Bs, l_Bs, imm_offset=s_offB)
        for c in ((0, 1) if group == 0 else (2,)):
            tdm_issued[c].append(j)

    # one TDM op per wave: tiles 0 and 1 and the scales
    def issue_prologue_tdm():
        if wave < 4:
            if wave == 0:
                fx.copy(tdm_A1w, tAgA1w[None, 0], tAsA1w[None, 0])
            else:
                if wave == 1:
                    fx.copy(tdm_B1w, tBgB1w[None, 0], tBsB1w[None, 0])
                else:
                    if wave == 2:
                        fx.copy(tdm_A1w, tAgA1w[None, 1], tAsA1w[None, 1])
                    else:
                        fx.copy(tdm_B1w, tBgB1w[None, 1], tBsB1w[None, 1])
        else:
            if wave < 4 + PRO_SCALE_WAVES:
                fx.copy(tdm_As_pro, gAs_pro, lAs_pro)
            else:
                fx.copy(tdm_Bs_pro, gBs_pro, lBs_pro)

    def lds_to_regs(k):
        page, r = k % NUM_STAGES, k % 2
        # one copy per swizzled chunk; registers stay in logical chunk order
        for h, a_off, b_off in ((0, a_lds_lo, b_lds_lo), (1, a_lds_hi, b_lds_hi)):
            src_A = fx.make_view(
                fx.add_offset(sA_base_ptr, page * BLOCK_M * 16 + a_off),
                fx.make_layout((4, 8), (1, 32 * 16)),
            )
            src_B = fx.make_view(
                fx.add_offset(sB_base_ptr, page * BLOCK_N * 16 + b_off),
                fx.make_layout((4, 4), (1, 64 * 16)),
            )
            fx.copy(
                copy_atom_i32,
                src_A,
                fx.make_view(
                    fx.add_offset(fx.get_iter(frag_A[r]), 4 * h), fx.make_layout((4, 8), (1, 8))
                ),
            )
            fx.copy(
                copy_atom_i32,
                src_B,
                fx.make_view(
                    fx.add_offset(fx.get_iter(frag_B[r]), 4 * h), fx.make_layout((4, 4), (1, 8))
                ),
            )
        s_off = (k // SCALE_GROUP) % SCALE_STAGES * SCALE_GROUP + k % SCALE_GROUP
        src_sA = fx.make_view(
            fx.add_offset(sAs_base_ptr, s_off * BLOCK_M + as_lds_base),
            fx.make_layout((4, 2), (1, 4)),
        )
        src_sB = fx.make_view(
            fx.add_offset(sBs_base_ptr, s_off * BLOCK_N + bs_lds_base),
            fx.make_layout((4, 1), (1, 4)),
        )
        fx.copy(
            copy_atom_i32,
            src_sA,
            fx.make_view(fx.get_iter(frag_sA[r]), fx.make_layout((4, 2), (1, 4))),
        )
        fx.copy(
            copy_atom_i32,
            src_sB,
            fx.make_view(fx.get_iter(frag_sB[r]), fx.make_layout((4, 1), (1, 4))),
        )

    A_REPS, B_REPS = 8, 4
    n_dsrd = 2 * (A_REPS + B_REPS) + (A_REPS + B_REPS) // 2
    n_mma = A_REPS * B_REPS

    # interleave tile k+1's LDS reads with gemm(k)'s WMMAs
    def hot_loop_scheduler():
        n_spread = n_mma - SCHED_TAIL_MMAS
        issued = 0
        for i in range_constexpr(n_spread):
            want = (i + 1) * n_dsrd // n_spread
            if want > issued:
                fx.rocdl.sched_dsrd(want - issued)
                issued = want
            fx.rocdl.sched_mfma(1)
        fx.rocdl.sched_mfma(SCHED_TAIL_MMAS)
        fx.rocdl.sched_barrier(0)

    def gemm_step(k):
        r = k % 2
        fx.gemm(mma_atom, frag_acc, [frag_A[r], frag_sA[r]], [frag_B[r], frag_sB[r]], frag_acc)

    frag_acc.fill(0)

    issue_prologue_tdm()
    wait_tile(0)
    fx.barrier()
    lds_to_regs(0)

    for k in range_constexpr(K_iters):
        if k + 1 < K_iters:
            # keep WMMAs from being scheduled across the barrier
            fx.rocdl.sched_barrier(0)
            wait_tile(k + 1)
            fx.rocdl.s_wait_dscnt(0)
            fx.barrier()
            fx.rocdl.sched_barrier(0)
            if k == 0 and NUM_STAGES - 1 < K_iters:
                issue_tdm(NUM_STAGES - 1)
            if k + NUM_STAGES < K_iters:
                # scale group g >= SCALE_STAGES: issued at k = SCALE_GROUP * (g - 1)
                g_next = k // SCALE_GROUP + 1
                if k % SCALE_GROUP == 0 and SCALE_STAGES <= g_next < num_groups:
                    issue_tdm(k + NUM_STAGES, scale_group=g_next)
                else:
                    issue_tdm(k + NUM_STAGES)
            lds_to_regs(k + 1)
            gemm_step(k)
            hot_loop_scheduler()
        else:
            gemm_step(k)

    if const_expr(OUT_KIND == 0):
        frag_c = frag_acc
    else:
        frag_c = fx.make_fragment_like(frag_acc, out_fx)
        frag_c.store(frag_acc.load().to(out_fx))

    # accumulator (v, j, i) is tile row wm*16 + 8*khalf + v + 32*i, column wn*16 + lane_mod16 + 64*j
    fx.rocdl.s_wait_dscnt(0)
    fx.barrier()
    lane_c = (wm * 16 + 8 * khalf) * BLOCK_N + wn * 16 + lane_mod16
    for h in range_constexpr(2):
        if h > 0:
            fx.rocdl.s_wait_tensorcnt(0)
            fx.barrier()
        # sched_barrier(0x7FF) keeps the stores from being merged into ds_store_2addr
        for ii in range_constexpr(4):
            for j in range_constexpr(4):
                for v in range_constexpr(8):
                    src_c = fx.make_view(
                        fx.add_offset(fx.get_iter(frag_c), v + 8 * j + 32 * (4 * h + ii)),
                        fx.make_layout(1, 1),
                    )
                    dst_c = fx.make_view(
                        fx.add_offset(lds_C_ptr, lane_c + (32 * ii + v) * BLOCK_N + 64 * j),
                        fx.make_layout(1, 1),
                    )
                    fx.copy(copy_atom_c, src_c, dst_c)
                    fx.rocdl.sched_barrier(0x7FF)
        fx.rocdl.s_wait_dscnt(0)
        fx.barrier()
        fx.copy(tdm_C, lds_C_wave, gC_wave[h])
    fx.rocdl.s_wait_tensorcnt(0)


@flyc.jit
def launch_gemm(
    A: fx.Tensor,
    B: fx.Tensor,
    A_s: fx.Tensor,
    B_s: fx.Tensor,
    C: fx.Tensor,
    M: fx.Int32,
    N: fx.Int32,
    K: fx.Int32,
    K_iters: fx.Constexpr[int],
    OUT_KIND: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    grid_tot = (M // BLOCK_M) * (N // BLOCK_N)
    gemm_kernel(A, B, A_s, B_s, C, M, N, K, K_iters, OUT_KIND).launch(
        grid=(grid_tot, 1, 1), block=(NUM_THREADS, 1, 1), stream=stream
    )


@flyc.kernel
def shuffle_scales_kernel(
    scales: fx.Tensor,
    scales_shuf: fx.Tensor,
    rows: fx.Int32,
    words_per_row: fx.Int32,
    src_row_words: fx.Int32,
    block_rows: fx.Constexpr[int],
    lane_rows: fx.Constexpr[int],
):
    # row-major [rows][src_row_words] (TE's padded buffer) -> iteration-major, one thread per word
    tid = fx.thread_idx.x
    bid_r = fx.block_idx.x
    bid_k = fx.block_idx.y

    src = fx.get_iter(scales)
    src_ptr = fx.recast_iter(fx.PointerType.get(fx.Int32.ir_type, src.memspace, 4), src)
    src_words = fx.make_view(src_ptr, fx.make_layout((rows, words_per_row), (src_row_words, 1)))
    dst_words = iter_major_scale_words(scales_shuf, rows, words_per_row, block_rows, lane_rows)

    src_tile = fx.slice(fx.zipped_divide(src_words, (block_rows, 1)), (None, (bid_r, bid_k)))
    dst_tile = fx.slice(fx.zipped_divide(dst_words, (block_rows, 1)), (None, (bid_r, bid_k)))

    copy_atom_i32 = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32)
    tile_mn, tv_layout = fx.make_layout_tv(
        fx.make_layout((block_rows, 1), (1, 1)), fx.make_layout((1, 1), (1, 1))
    )
    thr_copy = fx.make_tiled_copy(copy_atom_i32, tv_layout, tile_mn).get_slice(tid)
    part_src = thr_copy.partition_S(src_tile)
    part_dst = thr_copy.partition_D(dst_tile)
    frag = fx.make_fragment_like(part_src)
    fx.copy(copy_atom_i32, part_src, frag)
    fx.copy(copy_atom_i32, frag, part_dst)


@flyc.jit
def shuffle_scales(
    scales: fx.Tensor,
    scales_shuf: fx.Tensor,
    rows: fx.Int32,
    K: fx.Int32,
    src_row_words: fx.Int32,
    block_rows: fx.Constexpr[int],
    lane_rows: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    words_per_row = K // SCALE_WORD_K
    grid_r = rows // block_rows
    shuffle_scales_kernel(
        scales, scales_shuf, rows, words_per_row, src_row_words, block_rows, lane_rows
    ).launch(grid=(grid_r, words_per_row, 1), block=(block_rows, 1, 1), stream=stream)


@functools.lru_cache(maxsize=None)
def _swizzle_index(rows: int, K: int, device: torch.device) -> torch.Tensor:
    swz = (torch.arange(rows, device=device) >> 2) & (SWZ_CHUNKS - 1)
    src_chunk = torch.arange(SWZ_CHUNKS, device=device)[None, :] ^ swz[:, None]
    return src_chunk[:, None, :, None].expand(rows, K // BLOCK_K, SWZ_CHUNKS, 4)


def swizzle_operand(q, rows, K):
    """Copy of packed FP4 operand q [rows, K/2] with the LDS chunk swizzle applied."""
    words = q.view(torch.int32).view(rows, K // BLOCK_K, SWZ_CHUNKS, 4)
    return words.gather(2, _swizzle_index(rows, K, q.device)).view(torch.uint8).view(rows, K // 2)


def _shuffle_scale_words(scale, rows, K, block_rows, lane_rows, stream):
    """Reorder TE E8M0 scales (possibly padded) into the kernel's iteration-major order."""
    if scale.dtype != torch.uint8 or scale.ndim != 2:
        raise FlyDSLUnsupportedError(
            f"FlyDSL MXFP4 expects 2D uint8 E8M0 scales, got {scale.dtype} {tuple(scale.shape)}"
        )
    if scale.shape[0] < rows or scale.shape[1] < K // MXFP4_BLOCK_SIZE:
        raise FlyDSLUnsupportedError(
            f"FlyDSL MXFP4 scales {tuple(scale.shape)} do not cover "
            f"({rows}, {K // MXFP4_BLOCK_SIZE})"
        )
    if scale.stride(1) != 1 or scale.stride(0) % 4 != 0 or scale.data_ptr() % 4 != 0:
        raise FlyDSLUnsupportedError(
            f"FlyDSL MXFP4 needs word-aligned row-major scales, got strides {scale.stride()}"
        )
    out = torch.empty(rows, K // MXFP4_BLOCK_SIZE, dtype=torch.uint8, device=scale.device)
    shuffle_scales(scale, out, rows, K, scale.stride(0) // 4, block_rows, lane_rows, stream=stream)
    return out


def check_mxfp4_shape(m, n, k):
    """Raise FlyDSLUnsupportedError unless the kernel supports C[m, n] with reduction depth k."""
    if m % BLOCK_M or n % BLOCK_N:
        raise FlyDSLUnsupportedError(
            f"FlyDSL MXFP4 needs M, N multiples of {BLOCK_M}, got M={m}, N={n}"
        )
    if k % BLOCK_K or k < 2 * BLOCK_K:
        raise FlyDSLUnsupportedError(
            f"FlyDSL MXFP4 needs K a multiple of {BLOCK_K}, at least {2 * BLOCK_K}, got K={k}"
        )


def mxfp4_matmul(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    D: torch.Tensor,
    stream=None,
):
    """D[M, N] = dequant(a, a_scale) @ dequant(b, b_scale)^T.

    a [M, K/2] and b [N, K/2] packed FP4; scales E8M0 covering [M, K/32] and [N, K/32];
    D float32, bfloat16 or float16.
    """
    if a.ndim != 2 or b.ndim != 2 or D.ndim != 2:
        raise ValueError(
            "FlyDSL MXFP4 expects rank-2 a, b, D; got "
            f"{tuple(a.shape)}, {tuple(b.shape)}, {tuple(D.shape)}"
        )
    m, k2 = a.shape
    n, kb2 = b.shape
    k = 2 * k2
    if kb2 != k2:
        raise ValueError(f"Incompatible MXFP4 operands: a{tuple(a.shape)} and b{tuple(b.shape)}")
    if tuple(D.shape) != (m, n):
        raise ValueError(f"D shape {tuple(D.shape)} != expected {(m, n)}")
    if D.dtype not in OUT_KINDS:
        raise FlyDSLUnsupportedError(
            f"FlyDSL MXFP4 output must be float32, bfloat16 or float16, got {D.dtype}"
        )
    if not (a.is_contiguous() and b.is_contiguous() and D.is_contiguous()):
        raise FlyDSLUnsupportedError("FlyDSL MXFP4 requires contiguous a, b and D")
    if len({a.device, b.device, a_scale.device, b_scale.device, D.device}) != 1:
        raise ValueError("a, b, scales and D must be on the same device")
    check_mxfp4_shape(m, n, k)

    if stream is None:
        stream = torch.cuda.current_stream(a.device)
    a_u8 = a.view(torch.uint8)
    b_u8 = b.view(torch.uint8)
    a_scale_k = _shuffle_scale_words(a_scale, m, k, BLOCK_M, SCALE_LANE_ROWS_A, stream)
    b_scale_k = _shuffle_scale_words(b_scale, n, k, BLOCK_N, SCALE_LANE_ROWS_B, stream)
    with torch.cuda.stream(stream):
        a_k = swizzle_operand(a_u8, m, k)
        b_k = swizzle_operand(b_u8, n, k)
    launch_gemm(
        a_k, b_k, a_scale_k, b_scale_k, D, m, n, k, k // BLOCK_K, OUT_KINDS[D.dtype], stream=stream
    )
    return D


__all__ = [
    "BLOCK_M",
    "BLOCK_N",
    "BLOCK_K",
    "MXFP4_BLOCK_SIZE",
    "check_mxfp4_shape",
    "mxfp4_matmul",
]
