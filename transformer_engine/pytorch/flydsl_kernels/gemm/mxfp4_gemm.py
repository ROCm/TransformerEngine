# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

"""FlyDSL MXFP4 TN GEMM for gfx1250.

D[M, N] = a[M, K] @ b[N, K]^T with K-contiguous packed FP4 operands and E8M0 scales along K. TE's
MXFP4 columnwise storage is a materialized transpose, so this one kernel serves TN/NN/NT.
Tiles are staged with TDM into a 3-stage LDS pipeline and C is written back through LDS by TDM.
"""

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr

from .exceptions import FlyDSLUnsupportedError

MXFP4_BLOCK_SIZE = 32
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
# A/B LDS rows: 16 words padded to 20 by the TDM, so ds_load_b128 reads avoid bank conflicts
ROW_WORDS = 16
LDS_ROW_WORDS = 20
SCHED_TAIL_MMAS = 8
# Scales are TE's row-major words (one per row and K tile), kept in LDS as [row][SCALE_STAGES *
# SCALE_GROUP] with an odd row pitch so the per-row reads avoid bank conflicts
LDS_SCALE_PITCH = 17
OUT_KINDS = {torch.float32: 0, torch.bfloat16: 1, torch.float16: 2}


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
    As_sw: fx.Int32,
    Bs_sw: fx.Int32,
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

    # this lane's 16 B chunks khalf and khalf + 2
    a_lds_lo = (wm * 16 + lane_mod16) * LDS_ROW_WORDS + 4 * khalf
    a_lds_hi = a_lds_lo + 8
    b_lds_lo = (wn * 16 + lane_mod16) * LDS_ROW_WORDS + 4 * khalf
    b_lds_hi = b_lds_lo + 8
    as_lds_base = (wm * 16 + lane_mod16) * LDS_SCALE_PITCH
    bs_lds_base = (wn * 16 + lane_mod16) * LDS_SCALE_PITCH

    # tile k lives in LDS page k % NUM_STAGES
    smem = fx.SharedAllocator(static=False)
    PAGE_A, PAGE_B = BLOCK_M * LDS_ROW_WORDS, BLOCK_N * LDS_ROW_WORDS
    lds_A = smem.allocate(fx.Array[fx.Int32, NUM_STAGES * PAGE_A]).peek()
    # B starts at 64 KB so all its pages are reachable with immediate offsets from one base register
    smem.allocate(fx.Array[fx.Int32, LDS_B_START // 4 - NUM_STAGES * PAGE_A])
    lds_B = smem.allocate(fx.Array[fx.Int32, NUM_STAGES * PAGE_B]).peek()
    box_A = fx.make_layout((BLOCK_M, ROW_WORDS), (LDS_ROW_WORDS, 1))
    sA = fx.make_view(
        lds_A.ptr,
        fx.make_layout(((BLOCK_M, ROW_WORDS), NUM_STAGES), ((LDS_ROW_WORDS, 1), PAGE_A)),
    )
    box_B = fx.make_layout((BLOCK_N, ROW_WORDS), (LDS_ROW_WORDS, 1))
    sB = fx.make_view(
        lds_B.ptr,
        fx.make_layout(((BLOCK_N, ROW_WORDS), NUM_STAGES), ((LDS_ROW_WORDS, 1), PAGE_B)),
    )
    pad = dict(pad_interval=ROW_WORDS, pad_amount=LDS_ROW_WORDS - ROW_WORDS)

    num_groups = (K_iters + SCALE_GROUP - 1) // SCALE_GROUP
    tail_tiles = K_iters - (num_groups - 1) * SCALE_GROUP
    lds_As = smem.allocate(fx.Array[fx.Int32, BLOCK_M * LDS_SCALE_PITCH]).peek()
    lds_Bs = smem.allocate(fx.Array[fx.Int32, BLOCK_N * LDS_SCALE_PITCH]).peek()

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
        gA_half, [None, None], strides=[fx.Int64(k_words), None], num_warps=1, **pad
    )
    tdm_B = fx.rocdl.make_tdm_atom(
        gB_half, [None, None], strides=[fx.Int64(k_words), None], num_warps=1, **pad
    )

    def lds_half(base_ptr, page):
        return fx.make_view(
            fx.add_offset(base_ptr, page * PAGE_A + tdm_wave * HALF_ROWS * LDS_ROW_WORDS),
            half_box,
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

    # scale group g of a row block is a [rows][SCALE_GROUP] box at column g * SCALE_GROUP
    def scale_ptr(t):
        src = fx.get_iter(t)
        return fx.recast_iter(fx.PointerType.get(fx.Int32.ir_type, src.memspace, 4), src)

    sA_blk = fx.add_offset(scale_ptr(A_s), bid_m * BLOCK_M * As_sw)
    sB_blk = fx.add_offset(scale_ptr(B_s), bid_n * BLOCK_N * Bs_sw)

    def scale_atom(view, sw, tiles, width):
        return fx.rocdl.make_tdm_atom(
            view,
            [None, tiles],
            strides=[fx.Int64(sw), None],
            num_warps=1,
            pad_interval=width,
            pad_amount=LDS_SCALE_PITCH - width,
        )

    # prologue: scale groups 0 and 1, half the rows per wave
    sc_wave = wave % PRO_SCALE_WAVES
    PRO_ROWS = BLOCK_M // PRO_SCALE_WAVES
    PRO_COLS = SCALE_STAGES * SCALE_GROUP
    pro_box = fx.make_layout((PRO_ROWS, PRO_COLS), (PRO_COLS, 1))
    gAs_pro = fx.make_view(fx.add_offset(sA_blk, sc_wave * PRO_ROWS * As_sw), pro_box)
    gBs_pro = fx.make_view(fx.add_offset(sB_blk, sc_wave * PRO_ROWS * Bs_sw), pro_box)
    tdm_As_pro = scale_atom(gAs_pro, As_sw, K_iters, PRO_COLS)
    tdm_Bs_pro = scale_atom(gBs_pro, Bs_sw, K_iters, PRO_COLS)
    pro_lds_off = sc_wave * PRO_ROWS * LDS_SCALE_PITCH
    lAs_pro = fx.make_view(fx.add_offset(sAs_base_ptr, pro_lds_off), pro_box)
    lBs_pro = fx.make_view(fx.add_offset(sBs_base_ptr, pro_lds_off), pro_box)

    grp_box = fx.make_layout((BLOCK_M, SCALE_GROUP), (SCALE_GROUP, 1))
    gAs_grp = fx.make_view(sA_blk, grp_box)
    gBs_grp = fx.make_view(sB_blk, grp_box)
    tdm_As_grp = scale_atom(gAs_grp, As_sw, None, SCALE_GROUP)
    tdm_Bs_grp = scale_atom(gBs_grp, Bs_sw, None, SCALE_GROUP)
    tail_col = (num_groups - 1) * SCALE_GROUP
    gAs_tail = fx.make_view(fx.add_offset(sA_blk, tail_col), grp_box)
    gBs_tail = fx.make_view(fx.add_offset(sB_blk, tail_col), grp_box)
    tdm_As_tail = scale_atom(gAs_tail, As_sw, tail_tiles, SCALE_GROUP)
    tdm_Bs_tail = scale_atom(gBs_tail, Bs_sw, tail_tiles, SCALE_GROUP)

    # whole tiles, for K-loop iterations that also load a scale group
    gA_full = fx.make_view(fx.add_offset(fx.get_iter(A_words), bid_m * BLOCK_M * k_words), box_A)
    gB_full = fx.make_view(fx.add_offset(fx.get_iter(B_words), bid_n * BLOCK_N * k_words), box_B)
    tdm_A_full = fx.rocdl.make_tdm_atom(
        gA_full, [None, None], strides=[fx.Int64(k_words), None], num_warps=1, **pad
    )
    tdm_B_full = fx.rocdl.make_tdm_atom(
        gB_full, [None, None], strides=[fx.Int64(k_words), None], num_warps=1, **pad
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
    copy_atom_s32 = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32)
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
            l_As = fx.make_view(fx.add_offset(sAs_base_ptr, spage * SCALE_GROUP), grp_box)
            l_Bs = fx.make_view(fx.add_offset(sBs_base_ptr, spage * SCALE_GROUP), grp_box)
            s_offA = s_offB = fx.Int64(0 if tail else g * SCALE_GROUP * 4)
            if wave // TDM_GROUP_WAVES == group:
                gw = wave % TDM_GROUP_WAVES
                if gw < 2:
                    if gw == 0:
                        fx.copy(
                            tdm_A_full,
                            gA_full,
                            fx.make_view(fx.add_offset(sA_base_ptr, page * PAGE_A), box_A),
                            imm_offset=k_off,
                        )
                    else:
                        fx.copy(atom_As, g_As, l_As, imm_offset=s_offA)
                else:
                    if gw == 2:
                        fx.copy(
                            tdm_B_full,
                            gB_full,
                            fx.make_view(fx.add_offset(sB_base_ptr, page * PAGE_B), box_B),
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
        for h, a_off, b_off in ((0, a_lds_lo, b_lds_lo), (1, a_lds_hi, b_lds_hi)):
            src_A = fx.make_view(
                fx.add_offset(sA_base_ptr, page * PAGE_A + a_off),
                fx.make_layout((4, 8), (1, 32 * LDS_ROW_WORDS)),
            )
            src_B = fx.make_view(
                fx.add_offset(sB_base_ptr, page * PAGE_B + b_off),
                fx.make_layout((4, 4), (1, 64 * LDS_ROW_WORDS)),
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
        s_col = (k // SCALE_GROUP) % SCALE_STAGES * SCALE_GROUP + k % SCALE_GROUP
        fx.copy(
            copy_atom_s32,
            fx.make_view(
                fx.add_offset(sAs_base_ptr, as_lds_base + s_col),
                fx.make_layout((1, 8), (1, 32 * LDS_SCALE_PITCH)),
            ),
            fx.make_view(fx.get_iter(frag_sA[r]), fx.make_layout((1, 8), (1, 1))),
        )
        # B rows are 64 apart; sched_barrier keeps them from merging into ds_load_2addr_stride64
        for j in range_constexpr(4):
            fx.copy(
                copy_atom_s32,
                fx.make_view(
                    fx.add_offset(sBs_base_ptr, bs_lds_base + s_col + j * 64 * LDS_SCALE_PITCH),
                    fx.make_layout(1, 1),
                ),
                fx.make_view(fx.add_offset(fx.get_iter(frag_sB[r]), j), fx.make_layout(1, 1)),
            )
            fx.rocdl.sched_barrier(0x7FF)

    A_REPS, B_REPS = 8, 4
    n_dsrd = 3 * (A_REPS + B_REPS)
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
    As_sw: fx.Int32,
    Bs_sw: fx.Int32,
    K_iters: fx.Constexpr[int],
    OUT_KIND: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    grid_tot = (M // BLOCK_M) * (N // BLOCK_N)
    gemm_kernel(A, B, A_s, B_s, C, M, N, K, As_sw, Bs_sw, K_iters, OUT_KIND).launch(
        grid=(grid_tot, 1, 1), block=(NUM_THREADS, 1, 1), stream=stream
    )


def _scale_row_words(scale, rows, K):
    """Row stride in words of TE's row-major E8M0 scales (possibly padded), read in place."""
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
    return scale.stride(0) // 4


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
    launch_gemm(
        a.view(torch.uint8),
        b.view(torch.uint8),
        a_scale,
        b_scale,
        D,
        m,
        n,
        k,
        _scale_row_words(a_scale, m, k),
        _scale_row_words(b_scale, n, k),
        k // BLOCK_K,
        OUT_KINDS[D.dtype],
        stream=stream,
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
