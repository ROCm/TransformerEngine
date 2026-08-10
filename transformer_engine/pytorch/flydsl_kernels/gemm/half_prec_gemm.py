# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""FlyDSL half-precision (FP16/BF16) TN/NN/NT 4-wave GEMM kernel for Transformer Engine.

FP16 and BF16 share one source-level kernel generator: the algorithm is
identical and only the ``v_mfma_f32_16x16x32_{f16,bf16}`` opcode differs. That
opcode is selected at compile time via ``mfma_suffix``, so each (dtype, K,
output, layout) combination still compiles to its own cached binary with no
runtime dtype branch.

All supported layouts share the same generator while compiling to separate
cached binaries:

    TN: A [M,K] normal read,    B [N,K] normal read
    NN: A [M,K] normal read,    B [K,N] transpose read
    NT: A [K,M] transpose read, B [K,N] transpose read

The layout is a Python-only cache key. Global addressing and LDS fragment
reconstruction are selected while building each specialization, so no runtime
layout branch is emitted in the GEMM kernel.

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
from .exceptions import FlyDSLUnsupportedError
from .gemm_common_utils import require_block_tiling
from .fp16_gemm_utils import (
    G2SLoader,
    S2RLoader,
    compute_global_transpose_swizzle,
    compute_global_swizzle,
    make_byte_buffer_tensor,
    pack_i32x4_i32x8,
    swizzle_128,
    xcd_swizzle,
    barrier
)

# FP16 and BF16 differ only in the MFMA opcode suffix.
_MFMA_SUFFIX = {torch.float16: "f16", torch.bfloat16: "bf16"}


_BLOCK_M = 256
_BLOCK_N = 256
_BLOCK_K = 64

# Public metadata consumed by wrappers.
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

ELEM_BYTES = 2
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

assert BLOCK_K == 64
# DO NOT CHANGE THE FOLLOWING LINE.
assert NUM_THREADS == 256
assert LOAD_PASSES_A * NUM_THREADS * VEC_BYTES == LDS_BYTES_A
assert LOAD_PASSES_B * NUM_THREADS * VEC_BYTES == LDS_BYTES_B
assert LOAD_PASSES_A % 2 == 0
assert LOAD_PASSES_B % 2 == 0


def _compile_kernel(
    K: int,
    output_dtype: torch.dtype,
    layout: str,
    mfma_suffix: str,
    use_xcd_remap: bool = True,
):
    """Build one compile-time-specialized TN, NN, or NT half-precision kernel.

    ``mfma_suffix`` selects the ``v_mfma_f32_16x16x32_{f16,bf16}`` opcode.
    ``K`` must contain at least four K64 tiles. Runtime M/N are expected to
    be exact multiples of ``BLOCK_M``/``BLOCK_N``; the kernel has no edge masks.
    """
    if layout not in ("TN", "NN", "NT"):
        raise ValueError(f"Unsupported half-precision kernel layout: {layout}")
    if mfma_suffix not in ("f16", "bf16"):
        raise ValueError(f"Unsupported MFMA suffix: {mfma_suffix}")

    a_transpose_read = layout == "NT"
    b_transpose_read = layout in ("NN", "NT")

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

    ELEM_BYTES = 2
    VEC_BYTES = 16

    if output_dtype == torch.float16:
        output_element_bytes = 2
        output_fx_dtype = fx.Float16
    elif output_dtype == torch.bfloat16:
        output_element_bytes = 2
        output_fx_dtype = fx.BFloat16
    elif output_dtype == torch.float32:
        output_element_bytes = 4
        output_fx_dtype = fx.Float32
    else:
        raise TypeError(
            "FlyDSL half-precision GEMM output dtype must be torch.float16, "
            f"torch.bfloat16, or torch.float32, got {output_dtype}"
        )

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
    assert NUM_K_TILES >= 4, f"K={K} gives {NUM_K_TILES} K64 tiles; the two-page pipeline needs at least 4"

    LDS_ELEMS_HALF = (BLOCK_M // 2) * BLOCK_K
    LDS_BYTES_HALF = LDS_ELEMS_HALF * ELEM_BYTES
    LOAD_PASSES_HALF = LDS_BYTES_HALF // (NUM_THREADS * VEC_BYTES)
    assert LOAD_PASSES_HALF == LOAD_PASSES_A_SUBTILE == LOAD_PASSES_B_SUBTILE

    # Resolve layout-specific addressing and fragment reads before capture.
    Q0_SCHED_DSRD = 4 if a_transpose_read else 2
    PREFETCH_SCHED_DSRD = 8 if a_transpose_read else 4

    if a_transpose_read:
        def _a_leading_dim_bytes(c_m):
            return c_m * ELEM_BYTES

        def _a_global_base_bytes(k_base, subtile, c_m, bx_m_idx):
            return (
                k_base * fx.Index(c_m * ELEM_BYTES)
                + (bx_m_idx + fx.Index(subtile * (BLOCK_M // 2)))
                * fx.Index(ELEM_BYTES)
            )

        def _load_a_half(
            load_transposed_frag_half,
            load_frag_half_at_byte_base,
            lds_a,
            sm,
            mi,
            half,
            reg_subtile_m_idx0,
            lane_mod_16,
        ):
            del load_frag_half_at_byte_base, lane_mod_16
            subtile_m_idx = reg_subtile_m_idx0 + fx.Index(sm * 2)
            local_m_tile = (
                subtile_m_idx * fx.Index(SUBTILE_M)
                + fx.Index(mi * MFMA_M)
                - fx.Index(sm * (BLOCK_M // 2))
            )
            return load_transposed_frag_half(lds_a[sm], local_m_tile, half)
    else:
        def _a_leading_dim_bytes(c_m):
            del c_m
            return K * ELEM_BYTES

        def _a_global_base_bytes(k_base, subtile, c_m, bx_m_idx):
            del c_m
            return (
                (bx_m_idx + fx.Index(subtile * (BLOCK_M // 2)))
                * fx.Index(K * ELEM_BYTES)
                + k_base * fx.Index(ELEM_BYTES)
            )

        def _load_a_half(
            load_transposed_frag_half,
            load_frag_half_at_byte_base,
            lds_a,
            sm,
            mi,
            half,
            reg_subtile_m_idx0,
            lane_mod_16,
        ):
            del load_transposed_frag_half
            subtile_m_idx = reg_subtile_m_idx0 + fx.Index(sm * 2)
            a_row_addr = (
                subtile_m_idx * fx.Index(SUBTILE_M)
                + fx.Index(mi * MFMA_M)
                + lane_mod_16
            )
            half_row = a_row_addr - fx.Index(sm * (BLOCK_M // 2))
            return load_frag_half_at_byte_base(
                lds_a[sm],
                half_row * fx.Index(BLOCK_K * ELEM_BYTES),
                half,
            )

    if b_transpose_read:
        def _b_leading_dim_bytes(c_n):
            return c_n * ELEM_BYTES

        def _b_global_base_bytes(k_base, subtile, c_n, by_n_idx):
            return (
                k_base * fx.Index(c_n * ELEM_BYTES)
                + (by_n_idx + fx.Index(subtile * (BLOCK_N // 2)))
                * fx.Index(ELEM_BYTES)
            )

        def _load_b_ni(
            load_transposed_frag,
            load_normal_b_frag,
            lds_b,
            sn,
            ni,
            reg_subtile_n_idx0,
            lane_mod_16,
        ):
            del load_normal_b_frag, lane_mod_16
            subtile_n_idx = reg_subtile_n_idx0 + fx.Index(sn * 2)
            local_n_tile = (
                subtile_n_idx * fx.Index(SUBTILE_N)
                + fx.Index(ni * MFMA_N)
                - fx.Index(sn * (BLOCK_N // 2))
            )
            return load_transposed_frag(lds_b[sn], local_n_tile)
    else:
        def _b_leading_dim_bytes(c_n):
            del c_n
            return K * ELEM_BYTES

        def _b_global_base_bytes(k_base, subtile, c_n, by_n_idx):
            del c_n
            return (
                (by_n_idx + fx.Index(subtile * (BLOCK_N // 2)))
                * fx.Index(K * ELEM_BYTES)
                + k_base * fx.Index(ELEM_BYTES)
            )

        def _load_b_ni(
            load_transposed_frag,
            load_normal_b_frag,
            lds_b,
            sn,
            ni,
            reg_subtile_n_idx0,
            lane_mod_16,
        ):
            del load_transposed_frag
            subtile_n_idx = reg_subtile_n_idx0 + fx.Index(sn * 2)
            b_row_addr = (
                subtile_n_idx * fx.Index(SUBTILE_N)
                + fx.Index(ni * MFMA_N)
                + lane_mod_16
            )
            return load_normal_b_frag(lds_b, b_row_addr, sn)

    # Resolve global staging maps before FlyDSL captures ``kernel_gemm``.
    # Half-precision uses K64, so each transpose-read half-page is two independent
    # [K64, X64] slices with 128-byte physical rows.
    if a_transpose_read:
        def _a_global_offsets(lane, wave_id, c_m):
            return compute_global_transpose_swizzle(
                lane,
                wave_id,
                _a_leading_dim_bytes(c_m),
                LOAD_PASSES_HALF,
            )
    else:
        def _a_global_offsets(lane, wave_id, c_m):
            del c_m
            return compute_global_swizzle(
                lane,
                wave_id,
                K * ELEM_BYTES,
                LOAD_PASSES_HALF,
                preshuffled=False,
            )

    if b_transpose_read:
        def _b_global_offsets(lane, wave_id, c_n):
            return compute_global_transpose_swizzle(
                lane,
                wave_id,
                _b_leading_dim_bytes(c_n),
                LOAD_PASSES_HALF,
            )
    else:
        def _b_global_offsets(lane, wave_id, c_n):
            del c_n
            return compute_global_swizzle(
                lane,
                wave_id,
                K * ELEM_BYTES,
                LOAD_PASSES_HALF,
                preshuffled=False,
            )

    @fx.struct
    class SharedStorage:
        # Preserve the passing TN byte-staging contract exactly.  A half-precision K64
        # half-page is 128 rows x 128 bytes = 16 KiB.
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
        c_m: fx.Int32,
        c_n: fx.Int32,
    ):
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        lds_a0 = (lds.a0_0, lds.a0_1)
        lds_a1 = (lds.a1_0, lds.a1_1)
        lds_b0 = (lds.b0_0, lds.b0_1)
        lds_b1 = (lds.b1_0, lds.b1_1)

        # A/B arrive as contiguous uint8 byte views of the original
        # row-major half-precision tensors. This preserves the validated 16-byte
        # BufferCopyLDS128b path and byte-based address arithmetic.
        gA = make_byte_buffer_tensor(A)
        gB = make_byte_buffer_tensor(B)
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

        # Offsets are always bytes.  TN uses the original 128-byte XOR
        # swizzle.  NN/NT stage K-major 16-bit data as two [K64, X64] slices for
        # ds_read_b64_tr_b16; the layout choice was resolved before capture.
        gl_off_a = _a_global_offsets(lane, wave_id, c_m)
        gl_off_b = _b_global_offsets(lane, wave_id, c_n)

        a_g2s = G2SLoader(
            a_div,
            gl_off_a,
            LOAD_PASSES_HALF,
            fx.Uint8.ir_type,
            wave_id,
        )
        b_g2s = G2SLoader(
            b_div,
            gl_off_b,
            LOAD_PASSES_HALF,
            fx.Uint8.ir_type,
            wave_id,
        )
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
        c_store_atom = fx.make_copy_atom(
            fx.rocdl.BufferCopy32b() if output_element_bytes == 4 else fx.rocdl.BufferCopy16b(),
            output_fx_dtype,
        )

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
            # groups (two K32 slices x two N-halves).
            for _ in range_constexpr(4):
                rocdl.sched_vmem(2)
                rocdl.sched_mfma(8)
            rocdl.sched_barrier(0)

        def hot_loop_scheduler_q0_refill_a1_2n():
            # Eight refill VMEM operations and eight distributed A-bottom LDS
            # reads overlap four independent 8-MFMA K32 groups.
            for _ in range_constexpr(4):
                rocdl.sched_vmem(2)
                rocdl.sched_dsrd(Q0_SCHED_DSRD)
                rocdl.sched_mfma(8)
            rocdl.sched_barrier(0)

        def hot_loop_scheduler_q_prefetch_4n():
            # Eight two-read prefetch groups overlap four complete-quadrant
            # 16-MFMA groups (two K32 slices for each of Q2 and Q3).
            for _ in range_constexpr(4):
                rocdl.sched_dsrd(PREFETCH_SCHED_DSRD)
                rocdl.sched_mfma(16)
            rocdl.sched_barrier(0)

        def stage_a_subtile_pass(k_base, subtile, pass_in_subtile, lds_a):
            # One pass writes 256 threads * 16 B = 4 KiB. Four passes fill one
            # 128x64 half-page (16 KiB). Each half has its own LDS base.
            global_base = _a_global_base_bytes(
                k_base, subtile, c_m, bx_m_idx
            )
            a_g2s.load_one(lds_a[subtile], fx.Int32(global_base), pass_in_subtile)

        def stage_b_subtile_pass(k_base, subtile, pass_in_subtile, lds_b):
            global_base = _b_global_base_bytes(
                k_base, subtile, c_n, by_n_idx
            )
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
            return load_frag_at_byte_base(
                lds_b[half],
                half_row * fx.Index(BLOCK_K * ELEM_BYTES),
            )

        def load_transposed_frag_half(lds_page, local_x_tile, half):
            # Half-precision uses v_mfma_f32_16x16x32_{f16,bf16}, not the MXFP8 K128
            # instruction.  A 128-X half-page is therefore two independent
            # swizzled [K64, X64] half-precision slices. One ds_read_b64_tr_b16 returns
            # four half-precision values/lane; two reads form one K32 MFMA fragment.
            local_x_i32 = fx.Int32(local_x_tile)
            slice_idx = local_x_i32 // fx.Int32(64)
            x_in_slice = local_x_i32 % fx.Int32(64)
            lane_div16_i32 = fx.Int32(lane_div_16)
            lane_in16_i32 = fx.Int32(lane_mod_16)

            source_k = (
                lane_div16_i32 * fx.Int32(8)
                + lane_in16_i32 // fx.Int32(4)
            )
            source_x_byte = (
                x_in_slice * fx.Int32(ELEM_BYTES)
                + (lane_in16_i32 % fx.Int32(4)) * fx.Int32(8)
            )

            physical_k, physical_x = swizzle_128(source_k, source_x_byte)
            slice_base = slice_idx * fx.Int32(64 * 128)
            base = slice_base + physical_k * fx.Int32(128) + physical_x
            other = base ^ fx.Int32(0x220)
            immediate_offset = 0 if half == 0 else 0x1000
            return s2r.load_one_transpose(
                lds_page,
                base,
                other,
                immediate_offset=immediate_offset,
            )

        def load_transposed_frag(lds_page, local_x_tile):
            x0 = load_transposed_frag_half(lds_page, local_x_tile, 0)
            x1 = load_transposed_frag_half(lds_page, local_x_tile, 1)
            return pack_frag_halves(x0, x1)

        def _acc_idx(subtile_id, mi, ni):
            return subtile_id * MFMA_M_PER_SUBTILE * MFMA_N_PER_SUBTILE + mi * MFMA_N_PER_SUBTILE + ni

        def _k32_frag(full_frag, k32):
            # A/B 16x64 half-precision wave fragments are i32x8. Each K32 MFMA
            # consumes one contiguous i32x4 slice (eight half-precision values/lane).
            lo = k32 * 4
            v = Vec(full_frag)
            return Vec.from_elements(
                [v[lo], v[lo + 1], v[lo + 2], v[lo + 3]],
                fx.Int32,
            )

        def _pinned_mfma_once(acc_idx, a_k32, b_k32):
            acc_pin = PIN_ACC_BASE + acc_idx * 4
            llvm.InlineAsmOp(
                None,
                [arith._to_raw(a_k32), arith._to_raw(b_k32)],
                (
                    f"v_mfma_f32_16x16x32_{mfma_suffix} "
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
            """Accumulate one logical 16x16x64 half-precision product into pinned AGPRs."""
            for k32 in range_constexpr(2):
                _pinned_mfma_once(
                    acc_idx,
                    _k32_frag(a_frag, k32),
                    _k32_frag(b_frag, k32),
                )

        def pinned_final_mfma(dst_slot, old_acc_idx, a_frag, b_frag):
            # The final logical K64 update is two in-place K32 MFMAs.
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

        def mfma_2n_4mi_k32(subtile_id, n_base, k32, a0, a1, a2, a3, b0, b1):
            """Issue one K32 slice for a 4x2 accumulator slab."""
            a_frags = (a0, a1, a2, a3)
            b_frags = (b0, b1)
            for mi in range_constexpr(4):
                a_k32 = _k32_frag(a_frags[mi], k32)
                for nj in range_constexpr(2):
                    _pinned_mfma_once(
                        _acc_idx(subtile_id, mi, n_base + nj),
                        a_k32,
                        _k32_frag(b_frags[nj], k32),
                    )

        def mfma_4n_4mi_k32(subtile_id, k32, a0, a1, a2, a3, b0, b1, b2, b3):
            """Issue one K32 slice for a complete 4x4 quadrant."""
            a_frags = (a0, a1, a2, a3)
            b_frags = (b0, b1, b2, b3)
            for mi in range_constexpr(4):
                a_k32 = _k32_frag(a_frags[mi], k32)
                for ni in range_constexpr(4):
                    _pinned_mfma_once(
                        _acc_idx(subtile_id, mi, ni),
                        a_k32,
                        _k32_frag(b_frags[ni], k32),
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
            for ii in range_constexpr(4):
                row = row_base + fx.Index(ii)
                c_idx = c_tile_base_elems + row * fx.Index(c_n) + col
                value = Vec(acc)[ii]
                if const_expr(output_dtype != torch.float32):
                    value = value.to(output_fx_dtype)
                reg = fx.make_rmem_tensor(fx.make_layout(1, 1), output_fx_dtype)
                fx.memref_store_vec(Vec.filled(1, value, output_fx_dtype), reg)
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
            return _load_b_ni(
                load_transposed_frag,
                load_b_frag,
                lds_b,
                sn,
                ni,
                reg_subtile_n_idx0,
                lane_mod_16,
            )

        def load_b_subtile_regs(lds_b, sn):
            return (
                load_b_subtile_ni_regs(lds_b, sn, 0),
                load_b_subtile_ni_regs(lds_b, sn, 1),
                load_b_subtile_ni_regs(lds_b, sn, 2),
                load_b_subtile_ni_regs(lds_b, sn, 3),
            )

        def load_a_subtile_mi_half(lds_a, sm, mi, half):
            return _load_a_half(
                load_transposed_frag_half,
                load_frag_half_at_byte_base,
                lds_a,
                sm,
                mi,
                half,
                reg_subtile_m_idx0,
                lane_mod_16,
            )

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
            # refills and Q0 compute. Compute is K32-major across all 16
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
            # K32 slice 0 already covers K[0:32].

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
            # K32 slice 1 already covers K[32:64].

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
            # K32 slice 0 already covers K[0:32].

            stage_a_subtile_pass(k_refill, 1, 1, refill_a)
            # Keep this refill slot compute-free.

            stage_b_subtile_pass(k_refill, 1, 2, refill_b)
            mfma_2n_4mi_k32(1, 0, 1, a00, a01, a02, a03, b10, b11)

            stage_a_subtile_pass(k_refill, 1, 2, refill_a)
            mfma_2n_4mi_k32(1, 2, 1, a00, a01, a02, a03, b12, b13)

            stage_b_subtile_pass(k_refill, 1, 3, refill_b)
            # K32 slice 1 already covers K[32:64].

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
            # K32 slice 0 already covers K[0:32].

            next_a02 = load_a_subtile_mi_regs(next_a, 0, 2)
            mfma_4n_4mi_k32(2, 1, a10, a11, a12, a13, b00, b01, b02, b03)

            next_a03 = load_a_subtile_mi_regs(next_a, 0, 3)
            # K32 slice 1 already covers K[32:64].

            next_b00 = load_b_subtile_ni_regs(next_b, 0, 0)
            mfma_4n_4mi_k32(3, 0, a10, a11, a12, a13, b10, b11, b12, b13)

            next_b01 = load_b_subtile_ni_regs(next_b, 0, 1)
            # K32 slice 0 already covers K[0:32].

            next_b02 = load_b_subtile_ni_regs(next_b, 0, 2)
            mfma_4n_4mi_k32(3, 1, a10, a11, a12, a13, b10, b11, b12, b13)

            next_b03 = load_b_subtile_ni_regs(next_b, 0, 3)
            # K32 slice 1 already covers K[32:64].

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

        # Main HK loop: exactly one logical K64 per iteration.
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
            c_m,
            c_n,
            value_attrs={"rocdl.waves_per_eu": 1, "rocdl.flat_work_group_size": "256,256"},
        ).launch(grid=(grid_x, 1, 1), block=(NUM_THREADS, 1, 1), stream=stream)

    return launch_gemm

@functools.lru_cache(maxsize=None)
def _cached_launch(
    K: int,
    output_dtype: torch.dtype,
    layout: str,
    mfma_suffix: str,
    use_xcd_remap: bool = True,
):
    return _compile_kernel(
        K,
        output_dtype,
        layout,
        mfma_suffix,
        use_xcd_remap=use_xcd_remap,
    )


def _half_prec_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    *,
    layout: str,
    m: int,
    n: int,
    k: int,
    input_dtype: torch.dtype,
    label: str,
    stream=None,
):
    """Validate operands and launch a half-precision TN/NN/NT specialization.

    ``input_dtype`` is the required FP16/BF16 operand dtype and ``label`` is the
    human-readable kernel name used in error messages.
    """
    if layout not in ("TN", "NN", "NT"):
        raise ValueError(f"Unsupported {label} layout: {layout}")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(
            f"FlyDSL {label} expects rank-2 operands, got A{tuple(a.shape)} "
            f"and B{tuple(b.shape)}"
        )
    if a.dtype != input_dtype or b.dtype != input_dtype:
        raise TypeError(
            f"FlyDSL {label} GEMM expects {input_dtype} operands, "
            f"got A={a.dtype}, B={b.dtype}"
        )
    if not a.is_contiguous() or not b.is_contiguous():
        raise FlyDSLUnsupportedError(
            f"FlyDSL {label} {layout} requires original contiguous row-major "
            f"operands, got A stride={tuple(a.stride())}, "
            f"B stride={tuple(b.stride())}"
        )

    m = int(m)
    n = int(n)
    k = int(k)

    expected_shapes = {
        "TN": ((m, k), (n, k)),
        "NN": ((m, k), (k, n)),
        "NT": ((k, m), (k, n)),
    }
    expected_a, expected_b = expected_shapes[layout]
    if tuple(a.shape) != expected_a or tuple(b.shape) != expected_b:
        raise ValueError(
            f"FlyDSL {label} {layout} physical operands do not match contract: "
            f"A{tuple(a.shape)} expected {expected_a}; "
            f"B{tuple(b.shape)} expected {expected_b}"
        )

    if tuple(c.shape) != (m, n):
        raise ValueError(f"C shape {tuple(c.shape)} != expected {(m, n)}")
    if c.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(
            f"FlyDSL {label} output must be float16, bfloat16, or float32, "
            f"got {c.dtype}"
        )
    if a.device != b.device or a.device != c.device:
        raise ValueError(
            f"A, B, and C must be on the same device, got "
            f"{a.device}, {b.device}, and {c.device}"
        )
    if not c.is_contiguous():
        raise ValueError(f"FlyDSL {label} GEMM requires contiguous output storage")

    doGemm(
        a,
        b,
        c,
        layout=layout,
        m=m,
        n=n,
        k=k,
        input_dtype=input_dtype,
        label=label,
        stream=stream,
    )


def fp16_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    *,
    layout: str,
    m: int,
    n: int,
    k: int,
    stream=None,
):
    """Launch the wrapper-selected FP16 TN/NN/NT specialization."""
    _half_prec_matmul(
        a,
        b,
        c,
        layout=layout,
        m=m,
        n=n,
        k=k,
        input_dtype=torch.float16,
        label="FP16",
        stream=stream,
    )


def bf16_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    *,
    layout: str,
    m: int,
    n: int,
    k: int,
    stream=None,
):
    """Launch the wrapper-selected BF16 TN/NN/NT specialization."""
    _half_prec_matmul(
        a,
        b,
        c,
        layout=layout,
        m=m,
        n=n,
        k=k,
        input_dtype=torch.bfloat16,
        label="BF16",
        stream=stream,
    )


def doGemm(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    *,
    layout: str,
    m: int,
    n: int,
    k: int,
    input_dtype: torch.dtype,
    label: str,
    stream=None,
    use_xcd_remap: bool = True,
):
    """Launch one cached K/output/layout-specialized half-precision core.

    A and B are passed unchanged from ``gemm_wrappers.py``. Their pointers
    reference the original rowwise allocations:

      TN: A backing [M,K], B backing [N,K]
      NN: A backing [M,K], B backing [K,N]
      NT: A backing [K,M], B backing [K,N]

    NN/NT orientation is implemented by compile-time global addressing and
    ``ds_read_b64_tr_b16`` only.
    """
    if layout not in ("TN", "NN", "NT"):
        raise ValueError(f"Unsupported {label} layout: {layout}")

    M_runtime = int(m)
    N_runtime = int(n)
    K_runtime = int(k)

    if A.dtype != input_dtype or B.dtype != input_dtype:
        raise TypeError(
            f"{label} {layout} requires {input_dtype} inputs, "
            f"got {A.dtype} and {B.dtype}"
        )
    if C.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"Unsupported {label} output dtype: {C.dtype}")

    require_block_tiling(
        M_runtime,
        N_runtime,
        K_runtime,
        block_m=_BLOCK_M,
        block_n=_BLOCK_N,
        block_k=_BLOCK_K,
        label=f"{label} GEMM",
    )

    if tuple(C.shape) != (M_runtime, N_runtime):
        raise ValueError(
            f"C shape {tuple(C.shape)} != expected {(M_runtime, N_runtime)}"
        )

    if stream is None:
        stream = torch.cuda.current_stream()

    launch = _cached_launch(
        K_runtime,
        C.dtype,
        layout,
        _MFMA_SUFFIX[input_dtype],
        bool(use_xcd_remap),
    )
    # Preserve the original validated byte-addressed G2L path. These are
    # metadata-only dtype/flatten views of the already-contiguous row-major
    # tensors selected by gemm_wrappers.py; no transpose or copy is performed.
    A_arg = A.view(torch.uint8).view(-1)
    B_arg = B.view(torch.uint8).view(-1)
    C_arg = C.view(-1)

    launch(
        A_arg,
        B_arg,
        C_arg,
        M_runtime,
        N_runtime,
        stream=stream,
    )
