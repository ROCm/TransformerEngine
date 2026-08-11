# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""FlyDSL FP32 4-wave GEMM kernel for Transformer Engine.

The kernel specializes on K at compile time because the K32 loop is fully
hand-unrolled. M/N are runtime launch dimensions. The private optimized core
consumes A and B as FP32 tensors shaped [M, K] and [N, K], and writes FP32 C
shaped [M, N]. The public ``fp32_matmul`` entry point accepts Transformer
Engine's TN contract and performs the required private adaptation.

This module imports ``flydsl`` at import time and must therefore be imported
lazily only after FlyDSL availability has been confirmed.
"""

import functools

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

# Transformer Engine-local FlyDSL utilities.
from .gemm_common_utils import require_block_tiling
from .fp16_gemm_utils import (
    G2SLoader,
    S2RLoader,
    compute_global_swizzle,
    make_byte_buffer_tensor as make_fp32_byte_buffer_tensor,
    pack_i32x4_i32x8,
    swizzle_128,
    xcd_swizzle,
    barrier
)


_BLOCK_M = 256
_BLOCK_N = 256
_BLOCK_K = 32

BLOCK_M = _BLOCK_M
BLOCK_N = _BLOCK_N
BLOCK_K = _BLOCK_K

NUM_THREADS = 256
WARP_SIZE = 64
NUM_WAVES = NUM_THREADS // WARP_SIZE

SUBTILE_M = 64 
SUBTILE_N = 64

MFMA_M = 16
MFMA_N = 16

SUBTILES_PER_WAVE = 4
MFMA_M_PER_SUBTILE = SUBTILE_M // MFMA_M
MFMA_N_PER_SUBTILE = SUBTILE_N // MFMA_N
ACCS_PER_WAVE = SUBTILES_PER_WAVE * MFMA_M_PER_SUBTILE * MFMA_N_PER_SUBTILE

ELEM_BYTES = 4
VEC_BYTES = 16

LDS_ELEMS_A = BLOCK_M * BLOCK_K
LDS_ELEMS_B = BLOCK_N * BLOCK_K
LDS_BYTES_A = LDS_ELEMS_A * ELEM_BYTES
LDS_BYTES_B = LDS_ELEMS_B * ELEM_BYTES

LOAD_PASSES_A = LDS_BYTES_A // (NUM_THREADS * VEC_BYTES)
LOAD_PASSES_B = LDS_BYTES_B // (NUM_THREADS * VEC_BYTES)
LOAD_PASSES_A_SUBTILE = LOAD_PASSES_A // 2
LOAD_PASSES_B_SUBTILE = LOAD_PASSES_B // 2
PASSES_PER_A_MI = LOAD_PASSES_A_SUBTILE // MFMA_M_PER_SUBTILE

LDS_SYM_A0 = "fp32_pp_smem_a0"
LDS_SYM_A1 = "fp32_pp_smem_a1"
LDS_SYM_B0 = "fp32_pp_smem_b0"
LDS_SYM_B1 = "fp32_pp_smem_b1"
LDS_ALIAS_DOMAIN = '#llvm.alias_scope_domain<id = "fp32_pp_lds">'
SCOPE_IDS = ("a0", "a1", "b0", "b1")

assert BLOCK_K == 32
# DO NOT CHANGE THE FOLLOWING LINE.
assert NUM_THREADS == 256
assert LOAD_PASSES_A * NUM_THREADS * VEC_BYTES == LDS_BYTES_A
assert LOAD_PASSES_B * NUM_THREADS * VEC_BYTES == LDS_BYTES_B
assert LOAD_PASSES_A % 2 == 0
assert LOAD_PASSES_B % 2 == 0


def _compile_kernel(K: int, use_xcd_remap: bool = True, epilogue: str = "DEFAULT"):
    """Build the specialized 4-wave kernel for compile-time ``K``.

    ``K`` must contain at least four K32 tiles. Runtime M/N are expected to
    be exact multiples of ``BLOCK_M``/``BLOCK_N``; the kernel has no edge masks.

    ``epilogue`` selects the fused post-GEMM stages, resolved at compile time
    so the store loop stays branch-free:

        DEFAULT        plain matmul
        BIAS           + per-output-feature bias vector (indexed by N)
        GELU_AUX       (reserved) GELU with saved pre-activation aux
        GELU_AUX_BIAS  (reserved) bias then GELU with saved aux

    Only DEFAULT and BIAS are implemented; the GELU modes are accepted so the
    store-loop structure and dispatch signature are already in place.
    """
    if epilogue not in ("DEFAULT", "BIAS", "GELU_AUX", "GELU_AUX_BIAS"):
        raise ValueError(f"Unsupported FP32 epilogue: {epilogue}")
    if epilogue in ("GELU_AUX", "GELU_AUX_BIAS"):
        raise NotImplementedError(
            f"FP32 epilogue {epilogue} is reserved but not yet implemented"
        )
    has_bias = epilogue in ("BIAS", "GELU_AUX_BIAS")
    has_gelu = epilogue in ("GELU_AUX", "GELU_AUX_BIAS")

    BLOCK_M, BLOCK_N, BLOCK_K = _BLOCK_M, _BLOCK_N, _BLOCK_K
    NUM_THREADS = 256
    WARP_SIZE = 64

    SUBTILE_M = 64
    SUBTILE_N = 64

    MFMA_M = 16
    MFMA_N = 16

    SUBTILES_PER_WAVE = 4
    MFMA_M_PER_SUBTILE = SUBTILE_M // MFMA_M
    MFMA_N_PER_SUBTILE = SUBTILE_N // MFMA_N
    ACCS_PER_WAVE = SUBTILES_PER_WAVE * MFMA_M_PER_SUBTILE * MFMA_N_PER_SUBTILE

    ELEM_BYTES = 4
    VEC_BYTES = 16

    LDS_ELEMS_A = BLOCK_M * BLOCK_K
    LDS_ELEMS_B = BLOCK_N * BLOCK_K
    LDS_BYTES_A = LDS_ELEMS_A * ELEM_BYTES
    LDS_BYTES_B = LDS_ELEMS_B * ELEM_BYTES

    LOAD_PASSES_A = LDS_BYTES_A // (NUM_THREADS * VEC_BYTES)
    LOAD_PASSES_B = LDS_BYTES_B // (NUM_THREADS * VEC_BYTES)
    LOAD_PASSES_A_SUBTILE = LOAD_PASSES_A // 2
    LOAD_PASSES_B_SUBTILE = LOAD_PASSES_B // 2

    assert K % BLOCK_K == 0, f"K must be a multiple of {BLOCK_K}, got {K}"
    NUM_K_TILES = K // BLOCK_K
    assert NUM_K_TILES >= 4, f"K={K} gives {NUM_K_TILES} K32 tiles; the two-page pipeline needs at least 4"

    LDS_ELEMS_HALF = (BLOCK_M // 2) * BLOCK_K
    LDS_BYTES_HALF = LDS_ELEMS_HALF * ELEM_BYTES
    LOAD_PASSES_HALF = LDS_BYTES_HALF // (NUM_THREADS * VEC_BYTES)
    assert LOAD_PASSES_HALF == LOAD_PASSES_A_SUBTILE == LOAD_PASSES_B_SUBTILE

    @fx.struct
    class SharedStorage:
        # Each logical 256x64 FP32 page is two independent 128x64 half-pages.
        # Store LDS as bytes so BufferCopyLDS128b sees i8 on both source and
        # destination. Each half-page remains exactly 16 KiB.
        a0_0: fx.Array[fx.Uint8, LDS_BYTES_HALF, 16]
        a0_1: fx.Array[fx.Uint8, LDS_BYTES_HALF, 16]
        a1_0: fx.Array[fx.Uint8, LDS_BYTES_HALF, 16]
        a1_1: fx.Array[fx.Uint8, LDS_BYTES_HALF, 16]
        b0_0: fx.Array[fx.Uint8, LDS_BYTES_HALF, 16]
        b0_1: fx.Array[fx.Uint8, LDS_BYTES_HALF, 16]
        b1_0: fx.Array[fx.Uint8, LDS_BYTES_HALF, 16]
        b1_1: fx.Array[fx.Uint8, LDS_BYTES_HALF, 16]

    @flyc.kernel(known_block_size=[NUM_THREADS, 1, 1])
    def kernel_gemm(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        Bias: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
    ):
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        lds_a0 = (lds.a0_0, lds.a0_1)
        lds_a1 = (lds.a1_0, lds.a1_1)
        lds_b0 = (lds.b0_0, lds.b0_1)
        lds_b1 = (lds.b1_0, lds.b1_1)

        # A/B arrive as contiguous uint8 byte views. Keeping staging byte-addressed
        # preserves the original 16-byte G2L instruction cadence and vmcnt values.
        gA = make_fp32_byte_buffer_tensor(A)
        gB = make_fp32_byte_buffer_tensor(B)
        a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
        b_div = fx.logical_divide(gB, fx.make_layout(1, 1))
        tx = gpu.thread_id("x")

        num_blocks_m = c_m // BLOCK_M
        num_blocks_n = c_n // BLOCK_N

        if const_expr(use_xcd_remap):
            pid_m, pid_n = xcd_swizzle(num_blocks_m, num_blocks_n)
        else:
            pid_m, pid_n = divmod(fx.block_idx.x, num_blocks_n)

        bx_m = pid_m * BLOCK_M
        by_n = pid_n * BLOCK_N

        # The flattened/XCD-swizzled block coordinates are i32, while global
        # address arithmetic below is expressed in MLIR index type.  Convert
        # once here and use these index-typed tile bases for every address.
        bx_m_idx = fx.Index(bx_m)
        by_n_idx = fx.Index(by_n)

        # Keep wave/lane arithmetic in i32. compute_global_swizzle() combines
        # these values with i32 constants, so Index-typed coordinates would make
        # arith.addi receive mixed operand types.
        tx_i32 = fx.Int32(tx)
        wave_id = tx_i32 // fx.Int32(WARP_SIZE)
        lane = tx_i32 % fx.Int32(WARP_SIZE)

        # The utility mapping is identical to the previous manual staging:
        # each step contributes one contiguous 16-byte vector per thread, while
        # the global K coordinate is XOR-unswizzled for the physical LDS slot.
        gl_off_a = compute_global_swizzle(lane, wave_id, K * ELEM_BYTES, LOAD_PASSES_HALF, preshuffled=False)
        gl_off_b = compute_global_swizzle(lane, wave_id, K * ELEM_BYTES, LOAD_PASSES_HALF, preshuffled=False)
        a_g2s = G2SLoader(a_div, gl_off_a, LOAD_PASSES_HALF, fx.Uint8.ir_type, wave_id)
        b_g2s = G2SLoader(b_div, gl_off_b, LOAD_PASSES_HALF, fx.Uint8.ir_type, wave_id)
        s2r = S2RLoader(fx.Int32(0), 1)

        layout_lane16 = fx.make_layout((4, 16), (16, 1))
        coord_lane16 = fx.idx2crd(fx.Int32(lane), layout_lane16)
        lane_div_16 = fx.get(coord_lane16, 0)
        lane_mod_16 = fx.get(coord_lane16, 1)

        # Per-CTA tile base in C elements, folded into each store's linear
        # coordinate below (matching the scale-load addressing on this build;
        # add_offset on a dynamic Index is unsupported here).
        c_tile_base_elems = bx_m_idx * fx.Index(c_n) + by_n_idx
        gC = fx.rocdl.make_buffer_tensor(C, max_size=True)
        c_div = fx.logical_divide(gC, fx.make_layout(1, 1))
        c_store_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)

        # Bias is a length-N fp32 vector indexed by the global output-feature
        # (N) coordinate and broadcast across the M/token rows. const_expr folds
        # this compile-time flag at trace time so the setup (and load_bias) are
        # inlined into the kernel scope with no runtime dispatch branch.
        if const_expr(has_bias):
            gBias = fx.rocdl.make_buffer_tensor(Bias, max_size=True)
            bias_div = fx.logical_divide(gBias, fx.make_layout(1, 1))
            bias_ld_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)

            def load_bias(col):
                reg = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Float32)
                fx.copy(bias_ld_atom, fx.slice(bias_div, (None, fx.Int32(col))), reg)
                return fx.memref_load_vec(reg)[0]

        PIN_ACC_BASE = 0

        def _reg_list(prefix, start, end):
            return ",".join(f"~{{{prefix}{r}}}" for r in range(start, end + 1))

        def reserve_pinned_accumulators():
            # Reserve a fixed physical AGPR bank for all accumulators. In the
            # SSA-lowered path, the compiler generated heavy AGPR <-> VGPR traffic,
            # including v_accvgpr_mov/read sequences, s_nop stalls, and accumulator
            # spills. Pinning each f32x4 accumulator to a stable AGPR range keeps the
            # scaled MFMA accumulation in place and avoids those transfers and spills.
            #
            # ACCS_PER_WAVE = 64 accumulator objects and each object is f32x4,
            # so the physical bank is exactly 64 * 4 = 256 AGPRs: a[0:255].
            clobbers = _reg_list("a", PIN_ACC_BASE, PIN_ACC_BASE + ACCS_PER_WAVE * 4 - 1)
            llvm.InlineAsmOp(
                None,
                [],
                "",
                clobbers,
                has_side_effects=True,
            )

        def zero_pinned_accumulators():
            for ai in range_constexpr(ACCS_PER_WAVE * 4):
                llvm.InlineAsmOp(
                    None,
                    [],
                    f"v_accvgpr_write_b32 a[{PIN_ACC_BASE + ai}], 0",
                    f"~{{a{PIN_ACC_BASE + ai}}}",
                    has_side_effects=True,
                )

        def _inline_asm_i32(asm_string, constraints, operands=None):
            op = llvm.InlineAsmOp(
                T.i32,
                operands or [],
                asm_string,
                constraints,
                has_side_effects=True,
            )
            return _one_i32_result(op)

        def _one_i32_result(op):
            # Accept the result attribute names exposed by the supported MLIR Python bindings.
            return getattr(op, "result", getattr(op, "res", op.results[0]))

        def read_pinned_accumulator(acc_idx):
            acc_pin = PIN_ACC_BASE + acc_idx * 4
            r0 = _inline_asm_i32(f"v_accvgpr_read_b32 $0, a[{acc_pin + 0}]", "=v")
            r1 = _inline_asm_i32(f"v_accvgpr_read_b32 $0, a[{acc_pin + 1}]", "=v")
            r2 = _inline_asm_i32(f"v_accvgpr_read_b32 $0, a[{acc_pin + 2}]", "=v")
            r3 = _inline_asm_i32(f"v_accvgpr_read_b32 $0, a[{acc_pin + 3}]", "=v")
            return Vec.from_elements([r0, r1, r2, r3], fx.Int32).bitcast(fx.Float32)

        def read_physical_accumulator_slot(slot_idx):
            acc_pin = PIN_ACC_BASE + slot_idx * 4
            r0 = _inline_asm_i32(f"v_accvgpr_read_b32 $0, a[{acc_pin + 0}]", "=v")
            r1 = _inline_asm_i32(f"v_accvgpr_read_b32 $0, a[{acc_pin + 1}]", "=v")
            r2 = _inline_asm_i32(f"v_accvgpr_read_b32 $0, a[{acc_pin + 2}]", "=v")
            r3 = _inline_asm_i32(f"v_accvgpr_read_b32 $0, a[{acc_pin + 3}]", "=v")
            return Vec.from_elements([r0, r1, r2, r3], fx.Int32).bitcast(fx.Float32)

        def hot_loop_scheduler_q_refill_2n():
            # Eight refill VMEM operations overlap four independent 8-MFMA
            # groups (two K16 halves x two N-halves).
            for _ in range_constexpr(4):
                rocdl.sched_vmem(2)
                rocdl.sched_mfma(32)
            rocdl.sched_barrier(0)

        def hot_loop_scheduler_q0_refill_a1_2n():
            # Eight refill VMEM operations and eight distributed A-bottom LDS
            # reads overlap four independent 8-MFMA K32 groups.
            for _ in range_constexpr(4):
                rocdl.sched_vmem(2)
                rocdl.sched_dsrd(2)
                rocdl.sched_mfma(32)
            rocdl.sched_barrier(0)

        def hot_loop_scheduler_q_prefetch_4n():
            # Eight two-read prefetch groups overlap four complete-quadrant
            # 16-MFMA groups (two K16 halves for each of Q2 and Q3).
            for _ in range_constexpr(4):
                rocdl.sched_dsrd(4)
                rocdl.sched_mfma(64)
            rocdl.sched_barrier(0)

        def stage_a_subtile_pass(k_base, subtile, pass_in_subtile, lds_a):
            # One pass writes 256 threads * 16 B = 4 KiB. Four passes fill one
            # 128x64 half-page (16 KiB). Each half has its own LDS base.
            global_base = (bx_m_idx + fx.Index(subtile * (BLOCK_M // 2))) * fx.Index(K * ELEM_BYTES) + k_base * fx.Index(ELEM_BYTES)
            a_g2s.load_one(lds_a[subtile], fx.Int32(global_base), pass_in_subtile)

        def stage_b_subtile_pass(k_base, subtile, pass_in_subtile, lds_b):
            global_base = (by_n_idx + fx.Index(subtile * (BLOCK_N // 2))) * fx.Index(K * ELEM_BYTES) + k_base * fx.Index(ELEM_BYTES)
            b_g2s.load_one(lds_b[subtile], fx.Int32(global_base), pass_in_subtile)

        def stage_a_subtile(k_base, subtile, lds_a):
            for pass_in_subtile in range_constexpr(LOAD_PASSES_HALF):
                stage_a_subtile_pass(k_base, subtile, pass_in_subtile, lds_a)

        def stage_b_subtile(k_base, subtile, lds_b):
            for pass_in_subtile in range_constexpr(LOAD_PASSES_HALF):
                stage_b_subtile_pass(k_base, subtile, pass_in_subtile, lds_b)

        def load_frag_half_at_byte_base(lds_page, row_byte_base, half):
            # Issue exactly one 16-byte LDS read for one 16-byte half of the wave operand tile.
            # Keeping the halves separate allows steady-state Q0 to schedule one
            # A-bottom ds_read_b128 in each refill/MFMA chunk.
            k_col = reg_lds_k_col0 if half == 0 else reg_lds_k_col1
            return s2r.load_one(lds_page, fx.Int32(row_byte_base + k_col))

        def pack_frag_halves(x0, x1):
            return pack_i32x4_i32x8(x0, x1)

        def load_frag_at_byte_base(lds_page, row_byte_base):
            # Default complete-fragment path used outside the dedicated Q0 schedule.
            x0 = load_frag_half_at_byte_base(lds_page, row_byte_base, 0)
            x1 = load_frag_half_at_byte_base(lds_page, row_byte_base, 1)
            return pack_frag_halves(x0, x1)

        def load_b_frag(lds_b, local_row, half):
            # B is [N, K]. Each 128-row half-page has a local row origin of 0.
            half_row = local_row - fx.Index(half * (BLOCK_N // 2))
            return load_frag_at_byte_base(lds_b[half], half_row * fx.Index(BLOCK_K * ELEM_BYTES))

        def _acc_idx(subtile_id, mi, ni):
            return subtile_id * MFMA_M_PER_SUBTILE * MFMA_N_PER_SUBTILE + mi * MFMA_N_PER_SUBTILE + ni

        def _fp32_k4_operand(full_frag, k32_half, k4):
            # A/B 16x32 FP32 wave fragments are i32x8: one FP32 value per
            # VGPR and eight K4 MFMA steps per logical K32 tile.  Keep the
            # existing two-half schedule by grouping four K4 steps per half.
            return Vec(full_frag)[k32_half * 4 + k4]

        def _pinned_fp32_mfma_once(acc_idx, a_k4, b_k4):
            acc_pin = PIN_ACC_BASE + acc_idx * 4
            llvm.InlineAsmOp(
                None,
                [arith._to_raw(a_k4), arith._to_raw(b_k4)],
                (
                    f"v_mfma_f32_16x16x4_f32 "
                    f"a[{acc_pin}:{acc_pin + 3}], "
                    f"$0, $1, "
                    f"a[{acc_pin}:{acc_pin + 3}]"
                ),
                (
                    f"v,v,~{{a{acc_pin}}},~{{a{acc_pin + 1}}},"
                    f"~{{a{acc_pin + 2}}},~{{a{acc_pin + 3}}}"
                ),
                has_side_effects=True,
            )

        def pinned_mfma(acc_idx, a_frag, b_frag):
            """Accumulate one logical 16x16x32 FP32 product into pinned AGPRs."""
            for k32_half in range_constexpr(2):
                for k4 in range_constexpr(4):
                    _pinned_fp32_mfma_once(
                        acc_idx,
                        _fp32_k4_operand(a_frag, k32_half, k4),
                        _fp32_k4_operand(b_frag, k32_half, k4),
                    )

        def pinned_final_mfma(dst_slot, old_acc_idx, a_frag, b_frag):
            # The final logical K32 update is eight in-place K4 FP32 MFMAs.
            assert dst_slot == old_acc_idx
            pinned_mfma(old_acc_idx, a_frag, b_frag)

        def mfma_4n(acc_base, a_frag, b0, b1, b2, b3):
            pinned_mfma(acc_base + 0, a_frag, b0)
            pinned_mfma(acc_base + 1, a_frag, b1)
            pinned_mfma(acc_base + 2, a_frag, b2)
            pinned_mfma(acc_base + 3, a_frag, b3)

        def mfma_2n(acc_base, a_frag, b0, b1):
            pinned_mfma(acc_base + 0, a_frag, b0)
            pinned_mfma(acc_base + 1, a_frag, b1)

        def mfma_2n_4mi_k32(subtile_id, n_base, k32_half, a0, a1, a2, a3, b0, b1):
            """Issue four K4 steps (one K16 half) for a 4x2 accumulator slab."""
            a_frags = (a0, a1, a2, a3)
            b_frags = (b0, b1)
            for k4 in range_constexpr(4):
                for mi in range_constexpr(4):
                    a_k4 = _fp32_k4_operand(a_frags[mi], k32_half, k4)
                    for nj in range_constexpr(2):
                        _pinned_fp32_mfma_once(
                            _acc_idx(subtile_id, mi, n_base + nj),
                            a_k4,
                            _fp32_k4_operand(b_frags[nj], k32_half, k4),
                        )

        def mfma_4n_4mi_k32(subtile_id, k32_half, a0, a1, a2, a3, b0, b1, b2, b3):
            """Issue four K4 steps (one K16 half) for a complete 4x4 quadrant."""
            a_frags = (a0, a1, a2, a3)
            b_frags = (b0, b1, b2, b3)
            for k4 in range_constexpr(4):
                for mi in range_constexpr(4):
                    a_k4 = _fp32_k4_operand(a_frags[mi], k32_half, k4)
                    for ni in range_constexpr(4):
                        _pinned_fp32_mfma_once(
                            _acc_idx(subtile_id, mi, ni),
                            a_k4,
                            _fp32_k4_operand(b_frags[ni], k32_half, k4),
                        )

        def store_acc_vector_for_logical_idx(logical_acc_idx, acc):
            subtile_id = logical_acc_idx // (MFMA_M_PER_SUBTILE * MFMA_N_PER_SUBTILE)
            local_idx = logical_acc_idx % (MFMA_M_PER_SUBTILE * MFMA_N_PER_SUBTILE)
            sm = subtile_id // 2
            sn = subtile_id % 2
            mi = local_idx // MFMA_N_PER_SUBTILE
            ni = local_idx % MFMA_N_PER_SUBTILE

            subtile_m_idx = reg_subtile_m_idx0 + fx.Index(sm * 2)
            subtile_n_idx = reg_subtile_n_idx0 + fx.Index(sn * 2)
            row_base = subtile_m_idx * SUBTILE_M + fx.Index(mi * MFMA_M) + lane_div_16 * 4
            col = subtile_n_idx * SUBTILE_N + fx.Index(ni * MFMA_N) + lane_mod_16

            # Bias depends only on the output-feature (N) coordinate, so read it
            # once per column (global index by_n_idx + col) and reuse across the
            # four M rows below.
            if const_expr(has_bias):
                bias_value = load_bias(by_n_idx + col)

            for ii in range_constexpr(4):
                row = row_base + fx.Index(ii)
                c_idx = c_tile_base_elems + row * fx.Index(c_n) + col

                # Epilogue stages run on the fp32 accumulator; output is fp32
                # so there is no dtype narrowing. GELU will slot in here later.
                value = Vec(acc)[ii]
                if const_expr(has_bias):
                    value = value + bias_value

                reg = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Float32)
                fx.memref_store_vec(Vec.filled(1, value, fx.Float32), reg)
                fx.copy(c_store_atom, reg, fx.slice(c_div, (None, fx.Int32(c_idx))))


        # Explicit register coordinates for HK-style four-quadrant mapping.
        # BLOCK_M/BLOCK_N are 256x256.  Four waves map to warp positions
        # inside each 128x128 quadrant:
        #   cA: (warp_m,     warp_n)
        #   cB: (warp_m,     warp_n + 2)
        #   cC: (warp_m + 2, warp_n)
        #   cD: (warp_m + 2, warp_n + 2)
        reg_k_col0 = lane_div_16 * 16
        reg_k_col1 = 64 + lane_div_16 * 16

        # Every fragment row differs only by multiples of 16, so row % 16 is
        # always lane_mod_16. Hoist the logical->physical XOR mapping once.
        _, reg_lds_k_col0 = swizzle_128(lane_mod_16, reg_k_col0)
        _, reg_lds_k_col1 = swizzle_128(lane_mod_16, reg_k_col1)

        reg_subtile_m_idx0 = wave_id // 2
        reg_subtile_n_idx0 = wave_id % 2

        reserve_pinned_accumulators()
        zero_pinned_accumulators()

        def load_b_subtile_ni_regs(lds_b, sn, ni):
            subtile_n_idx = reg_subtile_n_idx0 + fx.Index(sn * 2)
            b_row_addr = subtile_n_idx * fx.Index(SUBTILE_N) + fx.Index(ni * MFMA_N) + lane_mod_16
            return load_b_frag(lds_b, b_row_addr, sn)

        def load_b_subtile_regs(lds_b, sn):
            return (
                load_b_subtile_ni_regs(lds_b, sn, 0),
                load_b_subtile_ni_regs(lds_b, sn, 1),
                load_b_subtile_ni_regs(lds_b, sn, 2),
                load_b_subtile_ni_regs(lds_b, sn, 3),
            )

        def load_a_subtile_mi_half(lds_a, sm, mi, half):
            subtile_m_idx = reg_subtile_m_idx0 + fx.Index(sm * 2)
            a_row_addr = subtile_m_idx * fx.Index(SUBTILE_M) + fx.Index(mi * MFMA_M) + lane_mod_16
            half_row = a_row_addr - fx.Index(sm * (BLOCK_M // 2))
            row_byte_base = half_row * fx.Index(BLOCK_K * ELEM_BYTES)
            return load_frag_half_at_byte_base(lds_a[sm], row_byte_base, half)

        def load_a_subtile_mi_regs(lds_a, sm, mi):
            x0 = load_a_subtile_mi_half(lds_a, sm, mi, 0)
            x1 = load_a_subtile_mi_half(lds_a, sm, mi, 1)
            return pack_frag_halves(x0, x1)

        def load_a_subtile_regs(lds_a, sm):
            return (
                load_a_subtile_mi_regs(lds_a, sm, 0),
                load_a_subtile_mi_regs(lds_a, sm, 1),
                load_a_subtile_mi_regs(lds_a, sm, 2),
                load_a_subtile_mi_regs(lds_a, sm, 3),
            )

        def hk_one_k_with_refill(
            k128,
            cur_a,
            cur_b,
            next_a,
            next_b,
            refill_a,
            refill_b,
            a0_regs,
            b0_regs,
        ):

            # Wait only far enough for the current page; the next-page refill may remain in flight.
            barrier(vmcnt=2 * LOAD_PASSES_A_SUBTILE + 2 * LOAD_PASSES_B_SUBTILE, lgkmcnt=0)
            rocdl.sched_barrier(0)

            # A-top and B-left are both carried as complete 64-row register tiles,
            # so their LDS half-pages can be refilled immediately.
            a00, a01, a02, a03 = a0_regs
            b00, b01, b02, b03 = b0_regs

            b10 = load_b_subtile_ni_regs(cur_b, 1, 0)
            b11 = load_b_subtile_ni_regs(cur_b, 1, 1)
            b12 = load_b_subtile_ni_regs(cur_b, 1, 2)
            b13 = load_b_subtile_ni_regs(cur_b, 1, 3)

            # Refill the current ping-pong page with K+2, alternating A and B passes.
            k_refill = fx.Index((k128 + 2) * BLOCK_K)

            # Q0: interleave the current tile's A-bottom LDS reads with K+2
            # refills and Q0 compute. Compute is K16-half-major across all 16
            # independent accumulators, eliminating the two-deep same-AGPR
            # dependency chains produced by pinned_mfma().
            rocdl.sched_barrier(0)
            a10_x0 = load_a_subtile_mi_half(cur_a, 1, 0, 0)
            stage_a_subtile_pass(k_refill, 0, 0, refill_a)
            mfma_2n_4mi_k32(0, 0, 0, a00, a01, a02, a03, b00, b01)

            a10_x1 = load_a_subtile_mi_half(cur_a, 1, 0, 1)
            stage_b_subtile_pass(k_refill, 0, 0, refill_b)
            mfma_2n_4mi_k32(0, 2, 0, a00, a01, a02, a03, b02, b03)

            a11_x0 = load_a_subtile_mi_half(cur_a, 1, 1, 0)
            stage_a_subtile_pass(k_refill, 0, 1, refill_a)
            # K32 slice 0 already covers K[0:16].

            a11_x1 = load_a_subtile_mi_half(cur_a, 1, 1, 1)
            stage_b_subtile_pass(k_refill, 0, 1, refill_b)
            # Keep this refill/LDS-read slot compute-free.

            a12_x0 = load_a_subtile_mi_half(cur_a, 1, 2, 0)
            stage_a_subtile_pass(k_refill, 0, 2, refill_a)
            mfma_2n_4mi_k32(0, 0, 1, a00, a01, a02, a03, b00, b01)

            a12_x1 = load_a_subtile_mi_half(cur_a, 1, 2, 1)
            stage_b_subtile_pass(k_refill, 0, 2, refill_b)
            mfma_2n_4mi_k32(0, 2, 1, a00, a01, a02, a03, b02, b03)

            a13_x0 = load_a_subtile_mi_half(cur_a, 1, 3, 0)
            stage_a_subtile_pass(k_refill, 0, 3, refill_a)
            # K32 slice 1 already covers K[16:32].

            a13_x1 = load_a_subtile_mi_half(cur_a, 1, 3, 1)
            stage_b_subtile_pass(k_refill, 0, 3, refill_b)
            # Keep this refill/LDS-read slot compute-free.

            hot_loop_scheduler_q0_refill_a1_2n()

            # Retire the eight distributed A-bottom LDS reads before K+2 refills
            # overwrite the current page's A-bottom half-page. Keep this wait as
            # late as possible to maximize read/compute overlap.
            rocdl.sched_barrier(0)
            barrier(lgkmcnt=0)
            rocdl.sched_barrier(0)

            a10 = pack_frag_halves(a10_x0, a10_x1)
            a11 = pack_frag_halves(a11_x0, a11_x1)
            a12 = pack_frag_halves(a12_x0, a12_x1)
            a13 = pack_frag_halves(a13_x0, a13_x1)

            rocdl.sched_barrier(0)
            stage_b_subtile_pass(k_refill, 1, 0, refill_b)
            mfma_2n_4mi_k32(1, 0, 0, a00, a01, a02, a03, b10, b11)

            stage_a_subtile_pass(k_refill, 1, 0, refill_a)
            mfma_2n_4mi_k32(1, 2, 0, a00, a01, a02, a03, b12, b13)

            stage_b_subtile_pass(k_refill, 1, 1, refill_b)
            # K32 slice 0 already covers K[0:16].

            stage_a_subtile_pass(k_refill, 1, 1, refill_a)
            # Keep this refill slot compute-free.

            stage_b_subtile_pass(k_refill, 1, 2, refill_b)
            mfma_2n_4mi_k32(1, 0, 1, a00, a01, a02, a03, b10, b11)

            stage_a_subtile_pass(k_refill, 1, 2, refill_a)
            mfma_2n_4mi_k32(1, 2, 1, a00, a01, a02, a03, b12, b13)

            stage_b_subtile_pass(k_refill, 1, 3, refill_b)
            # K32 slice 1 already covers K[16:32].

            stage_a_subtile_pass(k_refill, 1, 3, refill_a)
            # Keep this refill slot compute-free.
            hot_loop_scheduler_q_refill_2n()

            # Leave exactly the K+2 refill and scale loads outstanding. The following
            # LDS reads consume the already-ready next page, not the page being refilled.
            rocdl.sched_barrier(0)
            barrier(vmcnt=2 * LOAD_PASSES_A_SUBTILE + 2 * LOAD_PASSES_B_SUBTILE, lgkmcnt=0)
            rocdl.sched_barrier(0)

            next_a00 = load_a_subtile_mi_regs(next_a, 0, 0)
            mfma_4n_4mi_k32(2, 0, a10, a11, a12, a13, b00, b01, b02, b03)

            next_a01 = load_a_subtile_mi_regs(next_a, 0, 1)
            # K32 slice 0 already covers K[0:16].

            next_a02 = load_a_subtile_mi_regs(next_a, 0, 2)
            mfma_4n_4mi_k32(2, 1, a10, a11, a12, a13, b00, b01, b02, b03)

            next_a03 = load_a_subtile_mi_regs(next_a, 0, 3)
            # K32 slice 1 already covers K[16:32].

            next_b00 = load_b_subtile_ni_regs(next_b, 0, 0)
            mfma_4n_4mi_k32(3, 0, a10, a11, a12, a13, b10, b11, b12, b13)

            next_b01 = load_b_subtile_ni_regs(next_b, 0, 1)
            # K32 slice 0 already covers K[0:16].

            next_b02 = load_b_subtile_ni_regs(next_b, 0, 2)
            mfma_4n_4mi_k32(3, 1, a10, a11, a12, a13, b10, b11, b12, b13)

            next_b03 = load_b_subtile_ni_regs(next_b, 0, 3)
            # K32 slice 1 already covers K[16:32].

            hot_loop_scheduler_q_prefetch_4n()

            next_a0_regs = (next_a00, next_a01, next_a02, next_a03)
            next_b0_regs = (next_b00, next_b01, next_b02, next_b03)

            return next_a0_regs, next_b0_regs

        def hk_one_k_tail_with_next(cur_a, cur_b, next_a, next_b, a0_regs, b0_regs):
            barrier(vmcnt=2 * LOAD_PASSES_A_SUBTILE + 2 * LOAD_PASSES_B_SUBTILE, lgkmcnt=0)

            a00, a01, a02, a03 = a0_regs
            b00, b01, b02, b03 = b0_regs

            b10 = load_b_subtile_ni_regs(cur_b, 1, 0)
            b11 = load_b_subtile_ni_regs(cur_b, 1, 1)
            b12 = load_b_subtile_ni_regs(cur_b, 1, 2)
            b13 = load_b_subtile_ni_regs(cur_b, 1, 3)

            mfma_4n(_acc_idx(0, 0, 0), a00, b00, b01, b02, b03)
            mfma_4n(_acc_idx(0, 1, 0), a01, b00, b01, b02, b03)
            mfma_4n(_acc_idx(0, 2, 0), a02, b00, b01, b02, b03)
            mfma_4n(_acc_idx(0, 3, 0), a03, b00, b01, b02, b03)

            rocdl.sched_barrier(0)
            barrier(lgkmcnt=0)
            rocdl.sched_barrier(0)

            a10 = load_a_subtile_mi_regs(cur_a, 1, 0)
            a11 = load_a_subtile_mi_regs(cur_a, 1, 1)
            a12 = load_a_subtile_mi_regs(cur_a, 1, 2)
            a13 = load_a_subtile_mi_regs(cur_a, 1, 3)

            mfma_4n(_acc_idx(1, 0, 0), a00, b10, b11, b12, b13)
            mfma_4n(_acc_idx(1, 1, 0), a01, b10, b11, b12, b13)
            mfma_4n(_acc_idx(1, 2, 0), a02, b10, b11, b12, b13)
            mfma_4n(_acc_idx(1, 3, 0), a03, b10, b11, b12, b13)

            rocdl.sched_barrier(0)
            barrier(LOAD_PASSES_A_SUBTILE + LOAD_PASSES_B_SUBTILE, lgkmcnt=0)
            rocdl.sched_barrier(0)

            next_a00 = load_a_subtile_mi_regs(next_a, 0, 0)
            mfma_4n(_acc_idx(2, 0, 0), a10, b00, b01, b02, b03)

            next_a01 = load_a_subtile_mi_regs(next_a, 0, 1)
            mfma_4n(_acc_idx(2, 1, 0), a11, b00, b01, b02, b03)

            next_a02 = load_a_subtile_mi_regs(next_a, 0, 2)
            mfma_4n(_acc_idx(2, 2, 0), a12, b00, b01, b02, b03)

            next_a03 = load_a_subtile_mi_regs(next_a, 0, 3)
            mfma_4n(_acc_idx(2, 3, 0), a13, b00, b01, b02, b03)

            next_b00 = load_b_subtile_ni_regs(next_b, 0, 0)
            mfma_4n(_acc_idx(3, 0, 0), a10, b10, b11, b12, b13)

            next_b01 = load_b_subtile_ni_regs(next_b, 0, 1)
            mfma_4n(_acc_idx(3, 1, 0), a11, b10, b11, b12, b13)

            next_b02 = load_b_subtile_ni_regs(next_b, 0, 2)
            mfma_4n(_acc_idx(3, 2, 0), a12, b10, b11, b12, b13)

            next_b03 = load_b_subtile_ni_regs(next_b, 0, 3)
            mfma_4n(_acc_idx(3, 3, 0), a13, b10, b11, b12, b13)

            hot_loop_scheduler_q_prefetch_4n()

            next_a0_regs = (next_a00, next_a01, next_a02, next_a03)
            next_b0_regs = (next_b00, next_b01, next_b02, next_b03)

            return next_a0_regs, next_b0_regs

        def hk_one_k_final(cur_a, cur_b, a0_regs, b0_regs):
            barrier(vmcnt=0, lgkmcnt=0)

            a00, a01, a02, a03 = a0_regs
            b00, b01, b02, b03 = b0_regs

            # Materialize the remaining final-page A/B fragments once.  The
            # subsequent schedule is entirely register/AGPR traffic.
            b10 = load_b_subtile_ni_regs(cur_b, 1, 0)
            b11 = load_b_subtile_ni_regs(cur_b, 1, 1)
            b12 = load_b_subtile_ni_regs(cur_b, 1, 2)
            b13 = load_b_subtile_ni_regs(cur_b, 1, 3)

            rocdl.sched_barrier(0)
            barrier(lgkmcnt=0)
            rocdl.sched_barrier(0)

            a10 = load_a_subtile_mi_regs(cur_a, 1, 0)
            a11 = load_a_subtile_mi_regs(cur_a, 1, 1)
            a12 = load_a_subtile_mi_regs(cur_a, 1, 2)
            a13 = load_a_subtile_mi_regs(cur_a, 1, 3)

            rocdl.sched_barrier(0)
            barrier(lgkmcnt=0)
            rocdl.sched_barrier(0)

            a_frags = (a00, a01, a02, a03, a10, a11, a12, a13)
            b_frags = (b00, b01, b02, b03, b10, b11, b12, b13)

            # Rolling final-page epilogue.
            #
            # Finalize accumulators in their own physical AGPR slots, but delay
            # each AGPR read/store until several independent final MFMAs have
            # been issued. 
            #
            #   MFMA 0, MFMA 1, MFMA 2, MFMA 3, drain 0,
            #   MFMA 4, drain 1, MFMA 5, drain 2, ...
            #
            # The buffer stores are only issued here; they may remain in flight
            # while later MFMAs and accumulator drains continue.
            FINAL_EPILOGUE_DEPTH = 4
            pending = []

            for old_acc_idx in range_constexpr(ACCS_PER_WAVE):
                subtile_id = old_acc_idx // (MFMA_M_PER_SUBTILE * MFMA_N_PER_SUBTILE)
                local_idx = old_acc_idx % (MFMA_M_PER_SUBTILE * MFMA_N_PER_SUBTILE)
                sm = subtile_id // 2
                sn = subtile_id % 2
                mi = local_idx // MFMA_N_PER_SUBTILE
                ni = local_idx % MFMA_N_PER_SUBTILE

                a_frag_idx = sm * MFMA_M_PER_SUBTILE + mi
                b_frag_idx = sn * MFMA_N_PER_SUBTILE + ni

                # Final MFMA remains in-place.  The logical accumulator's own
                # AGPR slot is unique and cannot conflict with another pending
                # result, so no ad-hoc physical-slot permutation is needed.
                pinned_final_mfma(
                    old_acc_idx,
                    old_acc_idx,
                    a_frags[a_frag_idx],
                    b_frags[b_frag_idx],
                )
                pending.append(old_acc_idx)

                # Drain the oldest completed result only after enough newer
                # independent MFMAs have supplied the MFMA->AGPR-read spacing.
                if len(pending) == FINAL_EPILOGUE_DEPTH:
                    drain_acc_idx = pending.pop(0)
                    acc = read_physical_accumulator_slot(drain_acc_idx)
                    store_acc_vector_for_logical_idx(drain_acc_idx, acc)

            # Flush the final results after all final-page MFMAs have issued.
            for drain_acc_idx in pending:
                acc = read_physical_accumulator_slot(drain_acc_idx)
                store_acc_vector_for_logical_idx(drain_acc_idx, acc)

        # Prologue: stage K0/K1 data into ping-pong LDS pages.
        stage_a_subtile(fx.Index(0), 0, lds_a0)
        stage_b_subtile(fx.Index(0), 0, lds_b0)
        stage_b_subtile(fx.Index(0), 1, lds_b0)
        stage_a_subtile(fx.Index(0), 1, lds_a0)

        stage_a_subtile(fx.Index(BLOCK_K), 0, lds_a1)
        stage_b_subtile(fx.Index(BLOCK_K), 0, lds_b1)
        stage_b_subtile(fx.Index(BLOCK_K), 1, lds_b1)
        stage_a_subtile(fx.Index(BLOCK_K), 1, lds_a1)

        rocdl.sched_barrier(0)
        barrier(vmcnt=3 * LOAD_PASSES_A_SUBTILE + 4 * LOAD_PASSES_B_SUBTILE)
        rocdl.sched_barrier(0)

        a0_regs = load_a_subtile_regs(lds_a0, 0)

        rocdl.sched_barrier(0)
        barrier(vmcnt=3 * LOAD_PASSES_A_SUBTILE + 3 * LOAD_PASSES_B_SUBTILE)
        rocdl.sched_barrier(0)

        b0_regs = load_b_subtile_regs(lds_b0, 0)

        # Main HK loop: exactly one logical K32 per iteration.
        # Even k consumes and refills LDS0; odd k does the same for LDS1.
        for k128 in range_constexpr(NUM_K_TILES - 2):
            if (k128 % 2) == 0:
                a0_regs, b0_regs = hk_one_k_with_refill(
                    k128,
                    lds_a0,
                    lds_b0,
                    lds_a1,
                    lds_b1,
                    lds_a0,
                    lds_b0,
                    a0_regs,
                    b0_regs,
                )
            else:
                a0_regs, b0_regs = hk_one_k_with_refill(
                    k128,
                    lds_a1,
                    lds_b1,
                    lds_a0,
                    lds_b0,
                    lds_a1,
                    lds_b1,
                    a0_regs,
                    b0_regs,
                )

        # Common two-page tail. The penultimate tile uses Q2/Q3 carry-prefetch
        # to prepare A-top/B-left for the final tile, but performs no K+2 refill.
        if (NUM_K_TILES % 2) == 0:
            a0_regs, b0_regs = hk_one_k_tail_with_next(
                lds_a0,
                lds_b0,
                lds_a1,
                lds_b1,
                a0_regs,
                b0_regs,
            )
            hk_one_k_final(lds_a1, lds_b1, a0_regs, b0_regs)
        else:
            a0_regs, b0_regs = hk_one_k_tail_with_next(
                lds_a1,
                lds_b1,
                lds_a0,
                lds_b0,
                a0_regs,
                b0_regs,
            )
            hk_one_k_final(lds_a0, lds_b0, a0_regs, b0_regs)


    @flyc.jit
    def launch_gemm(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        Bias: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        # The integration only dispatches aligned shapes; no partial-tile masking exists.
        grid_x = (c_m // BLOCK_M) * (c_n // BLOCK_N)
        kernel_gemm(
            A,
            B,
            C,
            Bias,
            c_m,
            c_n,
            value_attrs={"rocdl.waves_per_eu": 1, "rocdl.flat_work_group_size": "256,256"},
        ).launch(grid=(grid_x, 1, 1), block=(NUM_THREADS, 1, 1), stream=stream)

    return launch_gemm

@functools.lru_cache(maxsize=None)
def _cached_launch(K: int, use_xcd_remap: bool = True, epilogue: str = "DEFAULT"):
    return _compile_kernel(K, use_xcd_remap=use_xcd_remap, epilogue=epilogue)



def fp32_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    stream=None,
    *,
    epilogue: str = "DEFAULT",
    bias: torch.Tensor = None,
):
    """TE-facing TN FP32 GEMM adapter.

    Public/backend contract:
        a: [M, K] FP32
        b: [K, N] FP32
        c: [M, N] FP32 output

    The optimized core streams both operands with K contiguous and therefore
    privately consumes B as [N, K]. In the normal TE TN path, ``b`` is a
    transpose view of contiguous rowwise weight storage, so ``b.T`` is already
    contiguous and does not require a physical transpose.
    """
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(
            f"FlyDSL FP32 TN expects rank-2 operands, got A{tuple(a.shape)} "
            f"and B{tuple(b.shape)}"
        )

    m, k = a.shape
    kb, n = b.shape
    if kb != k:
        raise ValueError(
            f"Inner dimensions do not match: A{tuple(a.shape)} and B{tuple(b.shape)}"
        )
    if a.dtype != torch.float32 or b.dtype != torch.float32:
        raise TypeError(
            "FlyDSL FP32 GEMM expects both operands to have torch.float32 dtype, "
            f"got {a.dtype} and {b.dtype}"
        )
    if tuple(c.shape) != (m, n):
        raise ValueError(f"C shape {tuple(c.shape)} != expected {(m, n)}")
    if c.dtype != torch.float32:
        raise TypeError(
            f"The current FlyDSL FP32 kernel stores torch.float32 output, got {c.dtype}"
        )
    if a.device != b.device or a.device != c.device:
        raise ValueError(
            f"A, B, and C must be on the same device, got "
            f"{a.device}, {b.device}, and {c.device}"
        )
    if not c.is_contiguous():
        raise ValueError("FlyDSL FP32 GEMM requires contiguous output storage")

    b_hk = b.transpose(0, 1).contiguous()
    doGemm(a, b_hk, c, stream=stream, epilogue=epilogue, bias=bias)


def doGemm(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    stream=None,
    use_xcd_remap: bool = True,
    epilogue: str = "DEFAULT",
    bias: torch.Tensor = None,
):
    """Launch the private K-specialized FP32 core.

    A and B are shaped [M, K] and [N, K]; C is shaped [M, N]. M and N
    remain runtime values, while K selects the cached compile-time specialization.
    """
    M_runtime, K_runtime = A.shape
    N_runtime, Kb_runtime = B.shape
    assert K_runtime == Kb_runtime, f"A.K={K_runtime} != B.K={Kb_runtime}"
    assert A.dtype == torch.float32 and B.dtype == torch.float32
    assert C.dtype == torch.float32
    require_block_tiling(
        M_runtime,
        N_runtime,
        K_runtime,
        block_m=_BLOCK_M,
        block_n=_BLOCK_N,
        block_k=_BLOCK_K,
        label="FP32 GEMM",
    )
    assert C.shape == (M_runtime, N_runtime)

    if epilogue not in ("DEFAULT", "BIAS", "GELU_AUX", "GELU_AUX_BIAS"):
        raise ValueError(f"Unsupported FP32 epilogue: {epilogue}")
    needs_bias = epilogue in ("BIAS", "GELU_AUX_BIAS")
    if needs_bias:
        if bias is None:
            raise ValueError(f"FP32 epilogue {epilogue} requires a bias tensor")
        # Bias is indexed by the output-feature (N) axis and broadcast over M.
        if bias.dtype != torch.float32:
            raise TypeError(f"FP32 bias must be float32, got {bias.dtype}")
        if bias.numel() != N_runtime:
            raise ValueError(
                f"FP32 bias length {bias.numel()} != N (out_features) {N_runtime}"
            )
        if bias.device != A.device:
            raise ValueError("bias must be on the same device as A, B, and C")
    elif bias is not None:
        raise ValueError(f"FP32 epilogue {epilogue} does not accept a bias tensor")

    if stream is None:
        stream = torch.cuda.current_stream()

    A_arg = A.contiguous().view(torch.uint8).view(-1)
    B_arg = B.contiguous().view(torch.uint8).view(-1)
    C_arg = C.view(-1)
    # DEFAULT keeps the kernel signature uniform with a dummy 1-element bias.
    if needs_bias:
        Bias_arg = bias.contiguous().view(-1)
    else:
        Bias_arg = torch.zeros(1, dtype=torch.float32, device=A.device)
    launch = _cached_launch(int(K_runtime), bool(use_xcd_remap), epilogue)
    launch(A_arg, B_arg, C_arg, Bias_arg, M_runtime, N_runtime, stream=stream)
