# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

"""FlyDSL MXFP4 TN GEMM for gfx1250.

D[M, N] = a[M, K] @ b[N, K]^T (+ beta * D) with K-contiguous packed FP4 operands and E8M0 scales
along K. TE's MXFP4 columnwise storage is a materialized transpose, so this one kernel serves
TN/NN/NT. Tiles are staged with TDM into a 3-stage LDS pipeline and C is written back through LDS
by TDM.
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
SCALE_GROUP = 6
SCALE_STAGES = 2
LDS_B_START = 64 * 1024
C_HALF_ROWS = BLOCK_M // 2  # C is stored in two halves
C_WAVE_ROWS = C_HALF_ROWS // NUM_WAVES
# A/B LDS rows: 16 words padded to 20 by the TDM, so ds_load_b128 reads avoid bank conflicts
ROW_WORDS = 16
LDS_ROW_WORDS = 20
SCHED_TAIL_MMAS = 8
K_LOOP_BODY = SCALE_GROUP
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
    beta: fx.Float32,
    K_SHORT: fx.Constexpr[int],
    K_REM: fx.Constexpr[int],
    OUT_KIND: fx.Constexpr[int],
    ACC: fx.Constexpr[int],
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
    K_iters = K // BLOCK_K
    # global offsets in 64 bits: rows * k_words and rows * N exceed 2^31 for large shapes
    k_words64, N64 = fx.Int64(k_words), fx.Int64(N)
    A_words = as_words(A, M, k_words)
    B_words = as_words(B, N, k_words)

    # this lane's 16 B chunk khalf (khalf + 2 is 8 words on)
    a_lds_lo = (wm * 16 + lane_mod16) * LDS_ROW_WORDS + 4 * khalf
    b_lds_lo = (wn * 16 + lane_mod16) * LDS_ROW_WORDS + 4 * khalf
    as_lds_base = (wm * 16 + lane_mod16) * SCALE_GROUP
    bs_lds_base = (wn * 16 + lane_mod16) * SCALE_GROUP

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

    PAGE_S = BLOCK_M * SCALE_GROUP
    lds_As = smem.allocate(fx.Array[fx.Int32, SCALE_STAGES * PAGE_S]).peek()
    lds_Bs = smem.allocate(fx.Array[fx.Int32, SCALE_STAGES * PAGE_S]).peek()

    # C is staged from LDS offset 0 after the K loop
    if smem.allocated_bytes < c_stage_bytes:
        smem.allocate(c_stage_bytes - smem.allocated_bytes)
    lds_C_ptr = fx.recast_iter(out_fx, smem.base_ptr)
    if const_expr(ACC):
        # C half 0 is prefetched here, half 1 at offset 0; each half is stored from where it landed
        lds_Cpre = smem.allocate(fx.Array[fx.Int32, c_stage_bytes // 4]).peek()
        lds_C_half = (fx.recast_iter(out_fx, lds_Cpre.ptr), lds_C_ptr)
    else:
        lds_C_half = (lds_C_ptr, lds_C_ptr)

    sA_base_ptr = fx.get_iter(sA)
    sB_base_ptr = fx.get_iter(sB)
    sAs_base_ptr = lds_As.ptr
    sBs_base_ptr = lds_Bs.ptr

    # K-loop TDM atoms over this wave's half tile; later K tiles are reached with imm_offset
    tdm_wave = wave % TDM_WAVES
    HALF_ROWS = BLOCK_M // TDM_WAVES
    half_box = fx.make_layout((HALF_ROWS, 16), (16, 1))
    gA_half = fx.make_view(
        fx.add_offset(
            fx.get_iter(A_words), fx.Int64(bid_m * BLOCK_M + tdm_wave * HALF_ROWS) * k_words64
        ),
        half_box,
    )
    gB_half = fx.make_view(
        fx.add_offset(
            fx.get_iter(B_words), fx.Int64(bid_n * BLOCK_N + tdm_wave * HALF_ROWS) * k_words64
        ),
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

    # prologue atoms: whole tiles 0 and 1
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

    # scale offsets stay 32-bit (rows * K / 128 words; the host checks they fit)
    sA_blk = fx.add_offset(scale_ptr(A_s), bid_m * BLOCK_M * As_sw)
    sB_blk = fx.add_offset(scale_ptr(B_s), bid_n * BLOCK_N * Bs_sw)

    grp_box = fx.make_layout((BLOCK_M, SCALE_GROUP), (SCALE_GROUP, 1))

    gAs_grp = fx.make_view(sA_blk, grp_box)
    gBs_grp = fx.make_view(sB_blk, grp_box)
    tdm_As_grp = fx.rocdl.make_tdm_atom(
        gAs_grp, [None, None], strides=[fx.Int64(As_sw), None], num_warps=1
    )
    tdm_Bs_grp = fx.rocdl.make_tdm_atom(
        gBs_grp, [None, None], strides=[fx.Int64(Bs_sw), None], num_warps=1
    )

    def copy_scale_group(blk, sw, lds_base, g, full=False, page=None):
        page = g % SCALE_STAGES if page is None else page
        lds = fx.make_view(fx.add_offset(lds_base, page * PAGE_S), grp_box)
        if const_expr(full):
            atom, view = (tdm_As_grp, gAs_grp) if blk is sA_blk else (tdm_Bs_grp, gBs_grp)
            fx.copy(atom, view, lds, imm_offset=fx.Int64(g) * (SCALE_GROUP * 4))
        else:
            view = fx.make_view(fx.add_offset(blk, g * SCALE_GROUP), grp_box)
            atom = fx.rocdl.make_tdm_atom(
                view, [None, K_iters - g * SCALE_GROUP], strides=[fx.Int64(sw), None], num_warps=1
            )
            fx.copy(atom, view, lds)

    # whole tiles, for K-loop iterations that also load a scale group
    gA_full = fx.make_view(
        fx.add_offset(fx.get_iter(A_words), fx.Int64(bid_m * BLOCK_M) * k_words64), box_A
    )
    gB_full = fx.make_view(
        fx.add_offset(fx.get_iter(B_words), fx.Int64(bid_n * BLOCK_N) * k_words64), box_B
    )
    tdm_A_full = fx.rocdl.make_tdm_atom(
        gA_full, [None, None], strides=[fx.Int64(k_words), None], num_warps=1, **pad
    )
    tdm_B_full = fx.rocdl.make_tdm_atom(
        gB_full, [None, None], strides=[fx.Int64(k_words), None], num_warps=1, **pad
    )

    bC = fx.slice(fx.zipped_divide(C, (BLOCK_M, BLOCK_N)), (None, (bid_m, bid_n)))

    mma_atom = fx.make_mma_atom(fx.rocdl.WMMAScale(16, 16, 128, fx.Float4E2M1FN))
    tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((2, 4, 1), (1, 2, 0)))
    thr_mma = tiled_mma.thr_slice(tid)

    copy_atom_i32 = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)
    copy_atom_s32 = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32)
    copy_atom_c = fx.make_copy_atom(
        fx.UniversalCopy32b() if OUT_KIND == 0 else fx.UniversalCopy16b(), out_fx
    )

    def new_frags():
        return (
            fx.make_rmem_tensor(fx.make_layout((8, 8, 1), (1, 8, 0)), fx.Int32),
            fx.make_rmem_tensor(fx.make_layout((8, 4, 1), (1, 8, 0)), fx.Int32),
            fx.make_rmem_tensor(fx.make_layout((1, 8, 1), (0, 1, 0)), fx.Int32),
            fx.make_rmem_tensor(fx.make_layout((1, 4, 1), (0, 1, 0)), fx.Int32),
        )

    A_REPS, B_REPS = 8, 4
    acc_tile = fx.make_layout((8, 1, 1), (1, 0, 0))
    scale_tile = fx.make_layout((1, 1, 1), (0, 0, 0))
    frag_acc = [
        [fx.make_rmem_tensor(acc_tile, fx.Float32) for _ in range(B_REPS)] for _ in range(A_REPS)
    ]

    def frag_view(t, off, layout):
        return fx.make_view(fx.add_offset(fx.get_iter(t), off), layout)

    def frag_view_ptr(ptr, off):
        return fx.make_view(fx.add_offset(ptr, off), fx.make_layout(1, 1))

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
            if wave < 6:
                if wave == 4:
                    copy_scale_group(sA_blk, As_sw, sAs_base_ptr, 0)
                else:
                    if const_expr(K_SHORT == 0 or K_SHORT > SCALE_GROUP):
                        copy_scale_group(sA_blk, As_sw, sAs_base_ptr, 1)
            else:
                if wave == 6:
                    copy_scale_group(sB_blk, Bs_sw, sBs_base_ptr, 0)
                else:
                    if const_expr(K_SHORT == 0 or K_SHORT > SCALE_GROUP):
                        copy_scale_group(sB_blk, Bs_sw, sBs_base_ptr, 1)

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

    def gemm_step(src):
        frag_A, frag_B, frag_sA, frag_sB = src
        for i in range_constexpr(A_REPS):
            for j in range_constexpr(B_REPS):
                fx.gemm(
                    mma_atom,
                    frag_acc[i][j],
                    [frag_view(frag_A, 8 * i, acc_tile), frag_view(frag_sA, i, scale_tile)],
                    [frag_view(frag_B, 8 * j, acc_tile), frag_view(frag_sB, j, scale_tile)],
                    frag_acc[i][j],
                )

    def rt_ptrs0():
        return [fx.get_iter(v) for v in (gA_half, gB_half, gA_full, gB_full)] + [sA_blk, sB_blk]

    def rt_advance(ptrs, tiles):
        tw = tiles * (BLOCK_K // 8)
        return [fx.add_offset(p, tw) for p in ptrs[:4]] + [
            fx.add_offset(p, tiles) for p in ptrs[4:]
        ]

    def rt_views(ptrs):
        layouts = (half_box, half_box, box_A, box_B, grp_box, grp_box)
        return tuple(fx.make_view(p, l) for p, l in zip(ptrs, layouts))

    def issue_tdm_rt(kb, vw, ks, scales, exact):
        vA_half, vB_half, vA_full, vB_full, vAs, vBs = vw
        js = ks + NUM_STAGES
        page, group = js % NUM_STAGES, js % 2
        k_off = fx.Int64(js * (BLOCK_K // 2))
        if wave // TDM_GROUP_WAVES == group:
            gw = wave % TDM_GROUP_WAVES
            if const_expr(scales):
                gi = ks // SCALE_GROUP + 1
                gpage = (kb // SCALE_GROUP + gi) % SCALE_STAGES
                s_off = fx.Int64(gi * SCALE_GROUP * 4)
                if gw < 2:
                    if gw == 0:
                        fx.copy(
                            tdm_A_full,
                            vA_full,
                            fx.make_view(fx.add_offset(sA_base_ptr, page * PAGE_A), box_A),
                            imm_offset=k_off,
                        )
                    else:
                        if const_expr(exact):
                            g = kb // SCALE_GROUP + gi
                            copy_scale_group(sA_blk, As_sw, sAs_base_ptr, g, page=gpage)
                        else:
                            lds = fx.make_view(fx.add_offset(sAs_base_ptr, gpage * PAGE_S), grp_box)
                            fx.copy(tdm_As_grp, vAs, lds, imm_offset=s_off)
                else:
                    if gw == 2:
                        fx.copy(
                            tdm_B_full,
                            vB_full,
                            fx.make_view(fx.add_offset(sB_base_ptr, page * PAGE_B), box_B),
                            imm_offset=k_off,
                        )
                    else:
                        if const_expr(exact):
                            g = kb // SCALE_GROUP + gi
                            copy_scale_group(sB_blk, Bs_sw, sBs_base_ptr, g, page=gpage)
                        else:
                            lds = fx.make_view(fx.add_offset(sBs_base_ptr, gpage * PAGE_S), grp_box)
                            fx.copy(tdm_Bs_grp, vBs, lds, imm_offset=s_off)
            else:
                if gw < TDM_WAVES:
                    fx.copy(tdm_A, vA_half, lds_half(sA_base_ptr, page), imm_offset=k_off)
                else:
                    fx.copy(tdm_B, vB_half, lds_half(sB_base_ptr, page), imm_offset=k_off)

    def lds_to_regs_rt(r, dst, bases):
        a_lo, b_lo, s_a, s_b, s_d = bases
        page = r % NUM_STAGES
        frag_A, frag_B, frag_sA, frag_sB = dst
        for h in range_constexpr(2):
            fx.copy(
                copy_atom_i32,
                fx.make_view(
                    fx.add_offset(sA_base_ptr, a_lo + (page * PAGE_A + 8 * h)),
                    fx.make_layout((4, 8), (1, 32 * LDS_ROW_WORDS)),
                ),
                frag_view(frag_A, 4 * h, fx.make_layout((4, 8), (1, 8))),
            )
            fx.copy(
                copy_atom_i32,
                fx.make_view(
                    fx.add_offset(sB_base_ptr, b_lo + (page * PAGE_B + 8 * h)),
                    fx.make_layout((4, 4), (1, 64 * LDS_ROW_WORDS)),
                ),
                frag_view(frag_B, 4 * h, fx.make_layout((4, 4), (1, 8))),
            )
        if const_expr(r // SCALE_GROUP % SCALE_STAGES == 1):
            s_a, s_b = s_a + s_d, s_b + s_d
        col = r % SCALE_GROUP
        for i in range_constexpr(A_REPS):
            fx.copy(
                copy_atom_s32,
                frag_view_ptr(sAs_base_ptr, s_a + (col + i * 32 * SCALE_GROUP)),
                frag_view(frag_sA, i, fx.make_layout(1, 1)),
            )
            fx.rocdl.sched_barrier(0x7FF)
        for j in range_constexpr(B_REPS):
            fx.copy(
                copy_atom_s32,
                frag_view_ptr(sBs_base_ptr, s_b + (col + j * 64 * SCALE_GROUP)),
                frag_view(frag_sB, j, fx.make_layout(1, 1)),
            )
            fx.rocdl.sched_barrier(0x7FF)

    def wait_tile_rt(ks, pending):
        if const_expr(pending == 0):
            fx.rocdl.s_wait_tensorcnt(0)
        else:
            if wave // TDM_GROUP_WAVES == (ks + 2) % 2:
                fx.rocdl.s_wait_tensorcnt(1)
            else:
                fx.rocdl.s_wait_tensorcnt(0)

    # C store: each wave writes C_WAVE_ROWS rows of each half
    def make_c_tdm():
        c_wave_box = fx.make_layout((C_WAVE_ROWS, BLOCK_N), (BLOCK_N, 1))
        c_wave_row = bid_m * BLOCK_M + wave * C_WAVE_ROWS
        gC_wave = [
            fx.make_view(
                fx.add_offset(
                    fx.get_iter(C),
                    fx.Int64(c_wave_row + h * C_HALF_ROWS) * N64 + fx.Int64(bid_n * BLOCK_N),
                ),
                c_wave_box,
            )
            for h in range(2)
        ]
        tdm_C = fx.rocdl.make_tdm_atom(gC_wave[0], [None, None], strides=[N64, None], num_warps=1)
        lds_C_wave = [
            fx.make_view(fx.add_offset(p, wave * C_WAVE_ROWS * BLOCK_N), c_wave_box)
            for p in lds_C_half
        ]
        return gC_wave, tdm_C, lds_C_wave

    if const_expr(ACC):
        gC_wave, tdm_C, lds_C_wave = make_c_tdm()

    def load_c_half(h):
        fx.copy(tdm_C, gC_wave[h], lds_C_wave[h])

    def k_step_rt(ks, cur, bases, issue=None, scales=False, exact=False, pending=1, c_load=False):
        fx.rocdl.sched_barrier(0)
        wait_tile_rt(ks, pending)
        fx.rocdl.s_wait_dscnt(0)
        fx.barrier()
        fx.rocdl.sched_barrier(0)
        if const_expr(c_load):
            # after the last A/B wait, so it never holds up a K tile
            load_c_half(0)
        if const_expr(issue is not None):
            issue_tdm_rt(issue[0], issue[1], ks, scales, exact)
        nxt = new_frags()
        lds_to_regs_rt(ks + 1, nxt, bases)
        gemm_step(cur)
        hot_loop_scheduler()
        return nxt

    def carry_values(fr):
        frag_A, frag_B, frag_sA, frag_sB = fr
        return (
            [frag_view(frag_A, 8 * i, acc_tile).load() for i in range_constexpr(A_REPS)]
            + [frag_view(frag_B, 8 * j, acc_tile).load() for j in range_constexpr(B_REPS)]
            + [frag_sA.load(), frag_sB.load()]
            + [acc.load() for acc_row in frag_acc for acc in acc_row]
        )

    def from_carry(vals):
        fr = new_frags()
        frag_A, frag_B, frag_sA, frag_sB = fr
        for i in range_constexpr(A_REPS):
            frag_view(frag_A, 8 * i, acc_tile).store(vals[i])
        for j in range_constexpr(B_REPS):
            frag_view(frag_B, 8 * j, acc_tile).store(vals[A_REPS + j])
        frag_sA.store(vals[A_REPS + B_REPS])
        frag_sB.store(vals[A_REPS + B_REPS + 1])
        accs = vals[A_REPS + B_REPS + 2 :]
        for i in range_constexpr(A_REPS):
            for j in range_constexpr(B_REPS):
                frag_acc[i][j].store(accs[i * B_REPS + j])
        return fr

    for acc_row in frag_acc:
        for acc in acc_row:
            acc.fill(0)

    issue_prologue_tdm()
    if wave // TDM_WAVES == 1:
        fx.rocdl.s_wait_tensorcnt(1)
    else:
        fx.rocdl.s_wait_tensorcnt(0)
    fx.barrier()
    bases = (a_lds_lo, b_lds_lo, as_lds_base, bs_lds_base, fx.Int32(PAGE_S))
    cur = new_frags()
    lds_to_regs_rt(0, cur, bases)

    fx.rocdl.sched_barrier(0)
    fx.rocdl.s_wait_tensorcnt(0)
    fx.rocdl.s_wait_dscnt(0)
    fx.barrier()
    fx.rocdl.sched_barrier(0)
    if const_expr(ACC and K_SHORT == 2):
        # no K step follows
        load_c_half(0)
    views0 = rt_views(rt_ptrs0())
    for ks in range_constexpr(-1, 1):
        if const_expr(K_SHORT == 0 or ks + NUM_STAGES < K_SHORT):
            issue_tdm_rt(0, views0, ks, False, False)
    nxt = new_frags()
    lds_to_regs_rt(1, nxt, bases)
    gemm_step(cur)
    hot_loop_scheduler()
    cur = nxt

    if const_expr(K_SHORT == 0):
        # kb & 2^30 is zero, but keeps the carried LDS bases from being hoisted
        n_body = K_iters // K_LOOP_BODY - 2
        init = carry_values(cur) + list(bases) + rt_ptrs0()
        for i, state in range(0, n_body, 1, init=init):
            nc = len(state) - 11
            cur = from_carry(state[:nc])
            a_lo, b_lo, s_a, s_b, s_d = (fx.Int32(v) for v in state[nc : nc + 5])
            bases = (a_lo, b_lo, s_a, s_b, s_d)
            ptrs = list(state[nc + 5 :])
            kb = fx.Int32(i) * K_LOOP_BODY
            issue = (kb, rt_views(ptrs))
            for u in range_constexpr(K_LOOP_BODY):
                at_group = (1 + u) % SCALE_GROUP == 0
                cur = k_step_rt(1 + u, cur, bases, issue, scales=at_group)
            zero = kb & (1 << 30)
            bases_next = [a_lo + zero, b_lo + zero, s_a + s_d, s_b + s_d, -s_d]
            results = yield carry_values(cur) + bases_next + rt_advance(ptrs, K_LOOP_BODY)
        nc = len(results) - 11
        cur = from_carry(results[:nc])
        bases = tuple(fx.Int32(v) for v in results[nc : nc + 5])
        kb = n_body * K_LOOP_BODY
        issue = (kb, rt_views(list(results[nc + 5 :])))
        R = 2 * SCALE_GROUP + K_REM - 4
    else:
        issue = (0, views0)
        R = max(K_SHORT - 4, 0)
    for u in range_constexpr(R):
        at_group = (1 + u) % SCALE_GROUP == 0 and u < R - 3
        cur = k_step_rt(1 + u, cur, bases, issue, scales=at_group, exact=u >= R - 8)
    ks_last = R + 3 if K_SHORT == 0 else K_SHORT - 1
    for ks in range_constexpr(max(ks_last - 2, 1), ks_last):
        last = ks == ks_last - 1
        cur = k_step_rt(ks, cur, bases, pending=int(not last), c_load=ACC and last)
    if const_expr(ACC):
        # the K loop is done reading A/B
        fx.rocdl.s_wait_dscnt(0)
        fx.barrier()
        fx.rocdl.sched_barrier(0)
        load_c_half(1)
    gemm_step(cur)
    x = bases[0] - wm * 16 * LDS_ROW_WORDS
    lane_mod16, khalf = x // LDS_ROW_WORDS, x % LDS_ROW_WORDS // 4

    if const_expr(not ACC):
        gC_wave, tdm_C, lds_C_wave = make_c_tdm()

    if const_expr(OUT_KIND == 0):
        frag_c = frag_acc
    else:
        frag_c = [[fx.make_fragment_like(acc, out_fx) for acc in acc_row] for acc_row in frag_acc]
        if const_expr(not ACC):
            for i in range_constexpr(A_REPS):
                for j in range_constexpr(B_REPS):
                    frag_c[i][j].store(frag_acc[i][j].load().to(out_fx))

    # accumulator (v, j, i) is tile row wm*16 + 8*khalf + v + 32*i, column wn*16 + lane_mod16 + 64*j
    def c_offset(ii, j, v):
        return lane_c + (32 * ii + v) * BLOCK_N + 64 * j

    # each thread reads C where it scatters D, so no barrier is needed in between
    def add_beta_c(h):
        for ii in range_constexpr(4):
            for j in range_constexpr(4):
                acc = frag_acc[4 * h + ii][j]
                c_old = fx.make_fragment_like(acc, out_fx)
                for v in range_constexpr(8):
                    src_c = fx.make_view(
                        fx.add_offset(lds_C_half[h], c_offset(ii, j, v)), fx.make_layout(1, 1)
                    )
                    fx.copy(copy_atom_c, src_c, frag_view(c_old, v, fx.make_layout(1, 1)))
                    # a merged ds_load_2addr is split with an s_wait_dscnt 0
                    fx.rocdl.sched_barrier(0x7FF)
                c_vals = c_old.load()
                if const_expr(OUT_KIND != 0):
                    c_vals = c_vals.to(fx.Float32)
                acc.store(acc.load() + c_vals * beta)
                if const_expr(OUT_KIND != 0):
                    frag_c[4 * h + ii][j].store(acc.load().to(out_fx))

    if const_expr(ACC):
        fx.rocdl.s_wait_tensorcnt(1)
    fx.rocdl.s_wait_dscnt(0)
    fx.barrier()
    lane_c = (wm * 16 + 8 * khalf) * BLOCK_N + wn * 16 + lane_mod16
    for h in range_constexpr(2):
        if h > 0:
            if const_expr(ACC):
                # C half 1, not the half 0 store (TDM ops complete in order)
                fx.rocdl.s_wait_tensorcnt(1)
            else:
                fx.rocdl.s_wait_tensorcnt(0)
            fx.barrier()
        if const_expr(ACC):
            add_beta_c(h)
        # sched_barrier(0x7FF) keeps the stores from being merged into ds_store_2addr
        for ii in range_constexpr(4):
            for j in range_constexpr(4):
                for v in range_constexpr(8):
                    src_c = frag_view(frag_c[4 * h + ii][j], v, fx.make_layout(1, 1))
                    dst_c = fx.make_view(
                        fx.add_offset(lds_C_half[h], c_offset(ii, j, v)),
                        fx.make_layout(1, 1),
                    )
                    fx.copy(copy_atom_c, src_c, dst_c)
                    fx.rocdl.sched_barrier(0x7FF)
        fx.rocdl.s_wait_dscnt(0)
        fx.barrier()
        fx.copy(tdm_C, lds_C_wave[h], gC_wave[h])
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
    beta: fx.Float32,
    K_SHORT: fx.Constexpr[int],
    K_REM: fx.Constexpr[int],
    OUT_KIND: fx.Constexpr[int],
    ACC: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    grid_tot = (M // BLOCK_M) * (N // BLOCK_N)
    gemm_kernel(
        A, B, A_s, B_s, C, M, N, K, As_sw, Bs_sw, beta, K_SHORT, K_REM, OUT_KIND, ACC
    ).launch(grid=(grid_tot, 1, 1), block=(NUM_THREADS, 1, 1), stream=stream)


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
    if rows * (scale.stride(0) // 4) > 2**31 - 1:
        raise FlyDSLUnsupportedError(f"FlyDSL MXFP4 scale offsets exceed 32 bits ({rows} rows)")
    return scale.stride(0) // 4


def k_loop_constexprs(K):
    """(K_SHORT, K_REM): the K tile count if the kernel needs no K loop, else K tiles mod
    K_LOOP_BODY, so that kernels are compiled per remainder, not per K."""
    k_iters = K // BLOCK_K
    if k_iters < 2 * K_LOOP_BODY:
        return k_iters, 0
    return 0, k_iters % K_LOOP_BODY


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
    accumulate: bool = False,
    beta: float = 1.0,
):
    """D[M, N] = dequant(a, a_scale) @ dequant(b, b_scale)^T (+ beta * D if ``accumulate``).

    a [M, K/2] and b [N, K/2] packed FP4; scales E8M0 covering [M, K/32] and [N, K/32];
    D float32, bfloat16 or float16. With ``accumulate``, D's existing values are read back
    and ``beta * D`` is added to the FP32 accumulator before narrowing to D's dtype.
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
        float(beta),
        *k_loop_constexprs(k),
        OUT_KINDS[D.dtype],
        # beta == 0 does not read D, as in hipBLASLt
        int(accumulate and float(beta) != 0.0),
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
