# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Flash Attention forward kernel for gfx1250 (CDNA5, WMMA, wave32).

Minimal-but-correct implementation using:
- WMMA 16x16x32 bf16/f16 → f32 for both GEMM stages.
- wave32 (4 waves per workgroup = 128 threads).
- Regular cooperative buffer loads for K/V (no TDM).
- Q preloaded to LDS (persistent across KV loop).
- Online softmax with ds_bpermute cross-lane reduction.
- Causal masking and GQA support.

Tile shape: BLOCK_M=64, BLOCK_N=32, HEAD_DIM=128.
Grid: (ceil(seq_len / BLOCK_M) * num_heads, batch, 1).
Block: (128, 1, 1) = 4 waves.

Layout: Q/K/V/O are 1D flattened from BSHD [batch, seq_len, num_heads, head_dim].
"""

import math as host_math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl, vector
from flydsl.expr import math as fmath
from flydsl.expr.typing import T
from flydsl.expr.arith import ArithValue
from flydsl.expr.arith import _to_raw as _raw
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr, check_smem_capacity
from .gemm_common_gfx1250 import (
    extract_lds_base_idx,
    lds_load_b128_raw,
    lds_transpose_load_raw,
)

_LOG2E = host_math.log2(host_math.e)

# WMMA parameters for gfx1250
WMMA_M, WMMA_N, WMMA_K = 16, 16, 32
WAVE_SIZE = 32
NUM_WAVES = 4
BLOCK_SIZE = NUM_WAVES * WAVE_SIZE  # 128 threads

# Tile sizes
BLOCK_M = 64  # Q rows per workgroup (4 waves × 16 rows)
BLOCK_N = 32  # KV columns per iteration (= WMMA_K, so GEMM2 has 1 K-step)

# LDS padding (bytes) to avoid bank conflicts
LDS_PAD = 8  # pad in elements (bf16)


def build_flash_attn_gfx1250_module(
    num_heads: int,
    head_dim: int = 128,
    causal: bool = True,
    dtype_str: str = "bf16",
    sm_scale: float = None,
    num_kv_heads: int = None,
    waves_per_eu: int = 2,
):
    """Build a flash attention forward kernel for gfx1250.

    Returns a JitFunction: launch_fn(q_flat, k_flat, v_flat, o_flat, batch_size, seq_len, stream=...)
    """
    gpu_arch = get_hip_arch()
    if num_kv_heads is None:
        num_kv_heads = num_heads
    assert num_heads % num_kv_heads == 0
    assert head_dim % 32 == 0 and head_dim >= 64, f"head_dim must be a multiple of 32 and >= 64, got {head_dim}"
    assert dtype_str in ("bf16", "f16")

    GQA_RATIO = num_heads // num_kv_heads
    HEAD_DIM = head_dim
    if sm_scale is None:
        sm_scale = 1.0 / host_math.sqrt(head_dim)
    SM_SCALE_LOG2E = sm_scale * _LOG2E

    is_bf16 = dtype_str == "bf16"
    elem_bytes = 2
    wmma_op = rocdl.wmma_f32_16x16x32_bf16 if is_bf16 else rocdl.wmma_f32_16x16x32_f16

    # LDS strides (in elements, including padding)
    LDS_Q_STRIDE = HEAD_DIM + LDS_PAD   # stride per Q row
    LDS_KV_STRIDE = HEAD_DIM + LDS_PAD  # stride per K/V row

    # LDS sizes (bytes)
    LDS_Q_SIZE = BLOCK_M * LDS_Q_STRIDE * elem_bytes
    LDS_KV_SIZE = BLOCK_N * LDS_KV_STRIDE * elem_bytes
    # P (softmax output) stored in LDS for GEMM2 reload: [BLOCK_M, BLOCK_N] bf16
    LDS_P_STRIDE = BLOCK_N + LDS_PAD
    LDS_P_SIZE = BLOCK_M * LDS_P_STRIDE * elem_bytes

    # LDS layout: [Q | K/V | P]
    LDS_Q_OFFSET = 0
    LDS_KV_OFFSET = LDS_Q_SIZE
    LDS_P_OFFSET = LDS_Q_SIZE + LDS_KV_SIZE
    LDS_TOTAL = LDS_Q_SIZE + LDS_KV_SIZE + LDS_P_SIZE

    # WMMA repetition counts per wave (each wave handles 16 Q rows)
    WMMA_M_PER_WAVE = 1  # 16 rows / WMMA_M=16
    # GEMM1: S[16, 32] = Q[16, 128] × K^T[128, 32]
    GEMM1_N_REP = BLOCK_N // WMMA_N  # 32/16 = 2
    GEMM1_K_STEPS = HEAD_DIM // WMMA_K  # 128/32 = 4
    # GEMM2: O[16, 128] += P[16, 32] × V[32, 128]
    GEMM2_N_REP = HEAD_DIM // WMMA_N  # 128/16 = 8
    GEMM2_K_STEPS = BLOCK_N // WMMA_K  # 32/32 = 1

    # Global strides (in elements)
    STRIDE_Q_TOKEN = num_heads * HEAD_DIM  # [B, S, H, D] flattened
    STRIDE_KV_TOKEN = num_kv_heads * HEAD_DIM

    arena = SmemAllocator(None, arch=gpu_arch, global_sym_name="flash_attn_gfx1250_arena")
    arena.ptr = LDS_TOTAL
    check_smem_capacity(LDS_TOTAL, gpu_arch)

    # Number of LDS elements (in elem_ty units) for SmemPtr shape
    LDS_TOTAL_ELEMS = LDS_TOTAL // elem_bytes

    compile_hints = dict(
        waves_per_eu=waves_per_eu,
        fast_fp_math=True,
        unsafe_fp_math=True,
    )

    @flyc.kernel
    def flash_attn_gfx1250_kernel(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,
        i32_batch: fx.Int32,
        i32_seq_len: fx.Int32,
    ):
        tx = gpu.thread_id("x")
        bx = gpu.block_id("x")  # encodes (q_tile_idx * num_heads + head_idx)
        by = gpu.block_id("y")  # batch index

        seq_len = arith.index_cast(T.index, i32_seq_len.ir_value())

        # Decode block ID: bx = q_tile_idx * num_heads + head_idx
        head_idx = bx % arith.index(num_heads)
        q_tile_idx = bx // arith.index(num_heads)
        kv_head_idx = head_idx // arith.index(GQA_RATIO)
        q_start = q_tile_idx * arith.index(BLOCK_M)

        # Wave/lane decomposition
        wave_id = tx // arith.index(WAVE_SIZE)
        lane_id = tx % arith.index(WAVE_SIZE)
        lane16 = lane_id % arith.index(16)
        lane_kgrp = lane_id // arith.index(16)  # 0 or 1

        # Wave's Q row offset within the workgroup tile
        wave_m_base = wave_id * arith.index(WMMA_M)  # each wave: 16 rows

        # Element type
        elem_ty = T.bf16 if is_bf16 else T.f16
        from flydsl.expr.typing import Numeric as _Numeric
        elem_dtype = _Numeric.from_ir_type(elem_ty)

        # ---- LDS setup ----
        arena_base = arena.get_base()
        lds_smem = SmemPtr(arena_base, 0, elem_ty, shape=(LDS_TOTAL_ELEMS,))
        lds_base = extract_lds_base_idx(lds_smem)

        # ---- Buffer resources ----
        q_nrec = arith.index_cast(T.index, i32_batch.ir_value()) * seq_len * arith.index(STRIDE_Q_TOKEN * elem_bytes)
        kv_nrec = arith.index_cast(T.index, i32_batch.ir_value()) * seq_len * arith.index(STRIDE_KV_TOKEN * elem_bytes)
        q_rsrc = buffer_ops.create_buffer_resource(Q, num_records_bytes=q_nrec)
        k_rsrc = buffer_ops.create_buffer_resource(K, num_records_bytes=kv_nrec)
        v_rsrc = buffer_ops.create_buffer_resource(V, num_records_bytes=kv_nrec)
        o_rsrc = buffer_ops.create_buffer_resource(O, num_records_bytes=q_nrec)

        # ---- Cooperative Q load to LDS ----
        # Each of 128 threads loads a portion of Q[BLOCK_M, HEAD_DIM]
        # Total elements: 64 × 128 = 8192, each thread loads 8192/128 = 64 elements = 8 dwords
        q_batch_offset = by * seq_len * arith.index(STRIDE_Q_TOKEN)
        q_base = q_batch_offset + q_start * arith.index(STRIDE_Q_TOKEN) + head_idx * arith.index(HEAD_DIM)

        # Each thread loads 8 × dword (8 bf16 pairs = 16 bf16 elements) per iteration, 4 iterations
        ELEMS_PER_LOAD = 8  # 8 bf16 = 16 bytes = 1 dwordx4
        Q_LOADS_PER_THREAD = (BLOCK_M * HEAD_DIM) // (BLOCK_SIZE * ELEMS_PER_LOAD)
        for li in range_constexpr(Q_LOADS_PER_THREAD):
            flat_idx = (tx + arith.index(li * BLOCK_SIZE)) * arith.index(ELEMS_PER_LOAD)
            row = flat_idx // arith.index(HEAD_DIM)
            col = flat_idx % arith.index(HEAD_DIM)
            # Global load (offset in i32 elements: byte_offset / 4)
            g_offset = q_base + row * arith.index(STRIDE_Q_TOKEN) + col
            g_i32_off = fx.Int32(g_offset * arith.index(elem_bytes) // arith.index(4))
            data = buffer_ops.buffer_load(q_rsrc, g_i32_off, vec_width=ELEMS_PER_LOAD // 2,
                                          dtype=T.i32)
            # LDS store
            lds_offset = arith.index(LDS_Q_OFFSET) + row * arith.index(LDS_Q_STRIDE * elem_bytes) + col * arith.index(elem_bytes)
            lds_ptr = buffer_ops.create_llvm_ptr(
                _raw(arith.index_cast(T.i32, lds_base + lds_offset)), address_space=3)
            from flydsl._mlir.dialects import llvm as llvm_d
            llvm_d.store(data, lds_ptr)

        gpu.barrier()

        # ---- Precompute Q lane bases for GEMM1 A-operand ----
        q_lds_base = lds_base + arith.index(LDS_Q_OFFSET)
        q_row_off = (wave_m_base + lane16) * arith.index(LDS_Q_STRIDE * elem_bytes)
        q_k_off = lane_kgrp * arith.index(8 * elem_bytes)
        q_lane_base = q_row_off + q_k_off

        def load_q_frag(ks):
            """Load Q A-fragment for K-step ks."""
            k_byte_off = arith.index(ks * WMMA_K * elem_bytes)
            off0 = q_lane_base + k_byte_off
            off1 = off0 + arith.index(32)
            v0 = fx.Vector(lds_load_b128_raw(q_lds_base, off0)).bitcast(elem_dtype)
            v1 = fx.Vector(lds_load_b128_raw(q_lds_base, off1)).bitcast(elem_dtype)
            return v0.shuffle(v1, list(range(16)))

        # ---- Precompute K/V B-operand lane bases ----
        kv_lds_base = lds_base + arith.index(LDS_KV_OFFSET)
        lane8 = lane16 % arith.index(8)
        lane_ngrp = lane16 // arith.index(8)
        kv_k_off = (lane_kgrp * arith.index(8) + lane8) * arith.index(LDS_KV_STRIDE * elem_bytes)
        kv_n_off = lane_ngrp * arith.index(8 * elem_bytes)

        def load_kv_frag_tr(wn, ks=0):
            """Load B-fragment from K/V in LDS using transpose load."""
            vec8_ty = ir.VectorType.get([8], elem_ty)
            n_col = arith.index(wn * WMMA_N * elem_bytes) + kv_n_off
            b_base = kv_k_off + n_col
            results = []
            for k_half in range_constexpr(2):
                k_row_off = (ks * WMMA_K + k_half * 16) * LDS_KV_STRIDE * elem_bytes
                elem_off = b_base + arith.index(k_row_off)
                v = lds_transpose_load_raw(vec8_ty, kv_lds_base, elem_off)
                results.append(fx.Vector(v))
            return results[0].shuffle(results[1], list(range(16)))

        # ---- P (softmax output) LDS helpers ----
        p_lds_base = lds_base + arith.index(LDS_P_OFFSET)
        p_row_off = (wave_m_base + lane16) * arith.index(LDS_P_STRIDE * elem_bytes)
        p_lane_base = p_row_off + lane_kgrp * arith.index(8 * elem_bytes)

        def store_p_to_lds(acc0, acc1):
            """Store 2 S accumulators (after softmax) as bf16 P[16,32] to LDS."""
            # acc0 = S[16,0:15], acc1 = S[16,16:31]
            # Lane layout: lane l holds C[l%16, (l/16)*8 : (l/16)*8+7]
            # Store: row = lane16, col_base = lane_kgrp*8 + tile_n_offset
            # Tile 0 (cols 0-15): store at col offset 0
            h0 = arith.trunc_f(T.vec(8, elem_ty), acc0)
            i32_0 = vector.bitcast(T.vec(4, T.i32), h0)
            off0 = p_lane_base  # col = lane_kgrp*8, within tile 0
            from flydsl._mlir.dialects import llvm as llvm_d
            ptr0 = buffer_ops.create_llvm_ptr(_raw(arith.index_cast(T.i32, p_lds_base + off0)), address_space=3)
            llvm_d.store(i32_0, ptr0)
            # Tile 1 (cols 16-31): store at col offset 16*elem_bytes
            h1 = arith.trunc_f(T.vec(8, elem_ty), acc1)
            i32_1 = vector.bitcast(T.vec(4, T.i32), h1)
            off1 = p_lane_base + arith.index(16 * elem_bytes)
            ptr1 = buffer_ops.create_llvm_ptr(_raw(arith.index_cast(T.i32, p_lds_base + off1)), address_space=3)
            llvm_d.store(i32_1, ptr1)

        def load_p_frag():
            """Reload P[16,32] as WMMA A-fragment (1 K-step since BLOCK_N=32=WMMA_K)."""
            off0 = p_lane_base
            off1 = p_lane_base + arith.index(32)  # +32 bytes = +16 bf16
            v0 = fx.Vector(lds_load_b128_raw(p_lds_base, off0)).bitcast(elem_dtype)
            v1 = fx.Vector(lds_load_b128_raw(p_lds_base, off1)).bitcast(elem_dtype)
            return v0.shuffle(v1, list(range(16)))

        # ---- Cross-lane reduction via ds_bpermute ----
        def reduce_pair_max(local_val):
            """Reduce max across lane pairs (lane l <-> lane l^16)."""
            pair_addr = (lane_id ^ arith.index(16)) * arith.index(4)
            pair_byte = fx.Int32(pair_addr)
            local_i32 = arith.bitcast(T.i32, local_val)
            peer_i32 = rocdl.ds_bpermute(T.i32, pair_byte, local_i32)
            peer_val = arith.bitcast(T.f32, peer_i32)
            return arith.maximumf(local_val, peer_val)

        def reduce_pair_sum(local_val):
            """Reduce sum across lane pairs (lane l <-> lane l^16)."""
            pair_addr = (lane_id ^ arith.index(16)) * arith.index(4)
            pair_byte = fx.Int32(pair_addr)
            local_i32 = arith.bitcast(T.i32, local_val)
            peer_i32 = rocdl.ds_bpermute(T.i32, pair_byte, local_i32)
            peer_val = arith.bitcast(T.f32, peer_i32)
            return local_val + peer_val

        # ---- Initialize O accumulators and softmax state ----
        o_accs = [arith.constant_vector(0.0, T.vec(8, T.f32)) for _ in range(GEMM2_N_REP)]
        m_val = arith.constant(float('-inf'), type=T.f32)  # row max (log2-scaled)
        l_val = arith.constant(0.0, type=T.f32)  # row sum

        # ---- KV loop ----
        kv_batch_offset = by * seq_len * arith.index(STRIDE_KV_TOKEN)
        num_kv_tiles = seq_len // arith.index(BLOCK_N)

        # Causal: limit number of KV tiles based on Q position
        if const_expr(causal):
            # Last Q row in this tile determines max KV tile
            last_q_row = q_start + wave_m_base + arith.index(WMMA_M - 1)
            max_kv_tile = (last_q_row + arith.index(BLOCK_N)) // arith.index(BLOCK_N)
            # min(num_kv_tiles, max_kv_tile) via select
            nkv_i32 = arith.index_cast(T.i32, num_kv_tiles)
            mkv_i32 = arith.index_cast(T.i32, max_kv_tile)
            use_max = arith.cmpi(arith.CmpIPredicate.slt, mkv_i32, nkv_i32)
            num_kv_tiles_eff_i32 = arith.select(use_max, mkv_i32, nkv_i32)
        else:
            num_kv_tiles_eff_i32 = arith.index_cast(T.i32, num_kv_tiles)

        # For loop over KV tiles
        from flydsl._mlir.dialects import scf
        zero_i32 = arith.constant(0, type=T.i32)
        one_i32 = arith.constant(1, type=T.i32)

        # Pack loop-carried values
        init_iter_args = [m_val, l_val] + o_accs

        for_op = scf.ForOp(
            _raw(zero_i32), _raw(num_kv_tiles_eff_i32), _raw(one_i32),
            [_raw(v) for v in init_iter_args],
        )

        with ir.InsertionPoint(for_op.body):
            kv_tile_idx = ir.Value(for_op.induction_variable)
            iter_m = ir.Value(for_op.inner_iter_args[0])
            iter_l = ir.Value(for_op.inner_iter_args[1])
            iter_o = [ir.Value(for_op.inner_iter_args[2 + i]) for i in range(GEMM2_N_REP)]

            kv_start = arith.index_cast(T.index, kv_tile_idx) * arith.index(BLOCK_N)

            # ---- Cooperative K load to LDS ----
            # 128 threads load K[BLOCK_N=32, HEAD_DIM=128] = 4096 elements
            # Each thread loads 4096/128 = 32 elements = 4 dwordx4 (16 bytes each)
            kv_base = kv_batch_offset + kv_start * arith.index(STRIDE_KV_TOKEN) + kv_head_idx * arith.index(HEAD_DIM)
            KV_ELEMS_TOTAL = BLOCK_N * HEAD_DIM
            KV_ELEMS_PER_THREAD = KV_ELEMS_TOTAL // BLOCK_SIZE
            KV_LOADS = KV_ELEMS_PER_THREAD // ELEMS_PER_LOAD

            for li in range_constexpr(KV_LOADS):
                flat_idx = (tx + arith.index(li * BLOCK_SIZE)) * arith.index(ELEMS_PER_LOAD)
                row = flat_idx // arith.index(HEAD_DIM)
                col = flat_idx % arith.index(HEAD_DIM)
                g_offset = kv_base + row * arith.index(STRIDE_KV_TOKEN) + col
                g_i32_off = fx.Int32(g_offset * arith.index(elem_bytes) // arith.index(4))
                data = buffer_ops.buffer_load(k_rsrc, g_i32_off, vec_width=ELEMS_PER_LOAD // 2,
                                              dtype=T.i32)
                lds_offset = arith.index(LDS_KV_OFFSET) + row * arith.index(LDS_KV_STRIDE * elem_bytes) + col * arith.index(elem_bytes)
                from flydsl._mlir.dialects import llvm as llvm_d
                lds_ptr = buffer_ops.create_llvm_ptr(
                    _raw(arith.index_cast(T.i32, lds_base + lds_offset)), address_space=3)
                llvm_d.store(data, lds_ptr)

            gpu.barrier()

            # ---- GEMM1: S[16,32] = Q[16,128] × K^T[128,32] ----
            s_accs = [arith.constant_vector(0.0, T.vec(8, T.f32)) for _ in range(GEMM1_N_REP)]

            for ks in range_constexpr(GEMM1_K_STEPS):
                q_frag = load_q_frag(ks)
                for wn in range_constexpr(GEMM1_N_REP):
                    k_frag = load_kv_frag_tr(wn, ks)
                    s_accs[wn] = wmma_op(
                        T.vec(8, T.f32), k_frag, q_frag, s_accs[wn],
                        reuseA=False, reuseB=False,
                    ).result

            # ---- Apply sm_scale (log2e) ----
            sm_scale_vec = arith.constant_vector(SM_SCALE_LOG2E, T.vec(8, T.f32))
            for wn in range_constexpr(GEMM1_N_REP):
                s_accs[wn] = arith.mulf(s_accs[wn], sm_scale_vec)

            # ---- Causal mask ----
            if const_expr(causal):
                # Row index: q_start + wave_m_base + lane16
                q_row = q_start + wave_m_base + lane16
                # Column range for this tile: kv_start + [0..31]
                # For acc[wn], columns = kv_start + wn*16 + lane_kgrp*8 + [0..7]
                neg_inf_vec = arith.constant_vector(float('-inf'), T.vec(8, T.f32))
                for wn in range_constexpr(GEMM1_N_REP):
                    col_base = kv_start + arith.index(wn * WMMA_N) + lane_kgrp * arith.index(8)
                    # Mask: set to -inf if col > q_row
                    # Check each of 8 elements
                    for ei in range_constexpr(8):
                        col_i = col_base + arith.index(ei)
                        mask_cond = arith.cmpi(arith.CmpIPredicate.sgt,
                                               arith.index_cast(T.i32, col_i),
                                               arith.index_cast(T.i32, q_row))
                        elem = vector.extract(s_accs[wn], static_position=[ei], dynamic_position=[])
                        masked = arith.select(mask_cond, arith.constant(float('-inf'), type=T.f32), elem)
                        s_accs[wn] = vector.insert(masked, s_accs[wn], static_position=[ei], dynamic_position=[])

            # ---- Online softmax ----
            # Compute local row max across 8 elements per accumulator
            def vec8_max(v):
                """Horizontal max of vec<8×f32>."""
                val = vector.extract(v, static_position=[0], dynamic_position=[])
                for i in range_constexpr(1, 8):
                    e = vector.extract(v, static_position=[i], dynamic_position=[])
                    val = arith.maximumf(val, e)
                return val

            # Local max: max across all GEMM1 accumulators for this lane
            local_max = vec8_max(s_accs[0])
            for wn in range_constexpr(1, GEMM1_N_REP):
                local_max = arith.maximumf(local_max, vec8_max(s_accs[wn]))

            # Cross-lane max (lane l <-> lane l^16 share same row)
            row_max = reduce_pair_max(local_max)

            # New running max
            m_new = arith.maximumf(iter_m, row_max)

            # exp2(old_max - new_max) for rescaling
            alpha = fmath.exp2(arith.subf(iter_m, m_new))
            alpha_vec = vector.broadcast(T.vec(8, T.f32), alpha)

            # Rescale running O and l
            new_l_scaled = arith.mulf(iter_l, alpha)
            new_o = [arith.mulf(iter_o[i], alpha_vec) for i in range(GEMM2_N_REP)]

            # exp2(S - m_new) for P
            m_new_vec = vector.broadcast(T.vec(8, T.f32), m_new)
            p_vecs = []
            for wn in range_constexpr(GEMM1_N_REP):
                p_vec = fmath.exp2(arith.subf(s_accs[wn], m_new_vec))
                p_vecs.append(p_vec)

            # Local row sum of exp values
            def vec8_sum(v):
                val = vector.extract(v, static_position=[0], dynamic_position=[])
                for i in range_constexpr(1, 8):
                    e = vector.extract(v, static_position=[i], dynamic_position=[])
                    val = arith.addf(val, e)
                return val

            local_sum = vec8_sum(p_vecs[0])
            for wn in range_constexpr(1, GEMM1_N_REP):
                local_sum = arith.addf(local_sum, vec8_sum(p_vecs[wn]))

            # Cross-lane sum
            row_sum = reduce_pair_sum(local_sum)
            new_l = arith.addf(new_l_scaled, row_sum)

            # ---- Store P to LDS for GEMM2 ----
            store_p_to_lds(p_vecs[0], p_vecs[1])
            gpu.barrier()

            # ---- Cooperative V load to LDS (reuse KV region) ----
            # Overwrite K in LDS with V
            v_base = kv_batch_offset + kv_start * arith.index(STRIDE_KV_TOKEN) + kv_head_idx * arith.index(HEAD_DIM)
            for li in range_constexpr(KV_LOADS):
                flat_idx = (tx + arith.index(li * BLOCK_SIZE)) * arith.index(ELEMS_PER_LOAD)
                row = flat_idx // arith.index(HEAD_DIM)
                col = flat_idx % arith.index(HEAD_DIM)
                g_offset = v_base + row * arith.index(STRIDE_KV_TOKEN) + col
                g_i32_off = fx.Int32(g_offset * arith.index(elem_bytes) // arith.index(4))
                data = buffer_ops.buffer_load(v_rsrc, g_i32_off, vec_width=ELEMS_PER_LOAD // 2,
                                              dtype=T.i32)
                lds_offset = arith.index(LDS_KV_OFFSET) + row * arith.index(LDS_KV_STRIDE * elem_bytes) + col * arith.index(elem_bytes)
                from flydsl._mlir.dialects import llvm as llvm_d
                lds_ptr = buffer_ops.create_llvm_ptr(
                    _raw(arith.index_cast(T.i32, lds_base + lds_offset)), address_space=3)
                llvm_d.store(data, lds_ptr)

            gpu.barrier()

            # ---- GEMM2: O[16,128] += P[16,32] × V[32,128] ----
            p_frag = load_p_frag()
            for wn in range_constexpr(GEMM2_N_REP):
                v_frag = load_kv_frag_tr(wn, 0)  # ks=0, BLOCK_N=32=WMMA_K
                new_o[wn] = wmma_op(
                    T.vec(8, T.f32), v_frag, p_frag, new_o[wn],
                    reuseA=False, reuseB=False,
                ).result

            gpu.barrier()

            # Yield loop-carried values
            yield_vals = [_raw(m_new), _raw(new_l)] + [_raw(v) for v in new_o]
            scf.YieldOp(yield_vals)

        # ---- Extract final values from loop ----
        final_m = ir.Value(for_op.results[0])
        final_l = ir.Value(for_op.results[1])
        final_o = [ir.Value(for_op.results[2 + i]) for i in range(GEMM2_N_REP)]

        # ---- Final rescale: O /= l ----
        inv_l = arith.divf(arith.constant(1.0, type=T.f32), final_l)
        inv_l_vec = vector.broadcast(T.vec(8, T.f32), inv_l)
        for i in range_constexpr(GEMM2_N_REP):
            final_o[i] = arith.mulf(final_o[i], inv_l_vec)

        # ---- Store O to global memory ----
        # O layout: same as Q [B, S, H, D]. Each lane stores 8 bf16 elements.
        # Accumulator layout: lane l has row (l%16), cols (l/16)*8 + wn*16 + [0..7]
        o_batch_offset = by * seq_len * arith.index(STRIDE_Q_TOKEN)
        o_row = q_start + wave_m_base + lane16
        o_col_base = lane_kgrp * arith.index(8)

        # Only store if row is within bounds
        o_in_bounds = arith.cmpi(arith.CmpIPredicate.slt,
                                 arith.index_cast(T.i32, o_row),
                                 arith.index_cast(T.i32, seq_len))
        if o_in_bounds:
            for wn in range_constexpr(GEMM2_N_REP):
                col = o_col_base + arith.index(wn * WMMA_N)
                g_offset = o_batch_offset + o_row * arith.index(STRIDE_Q_TOKEN) + head_idx * arith.index(HEAD_DIM) + col
                g_i32_off = fx.Int32(g_offset * arith.index(elem_bytes) // arith.index(4))
                # Convert f32 → bf16/f16 and store 8 elements (16 bytes)
                h_vec = arith.trunc_f(T.vec(8, elem_ty), final_o[wn])
                i32_vec = vector.bitcast(T.vec(4, T.i32), h_vec)
                buffer_ops.buffer_store(i32_vec, o_rsrc, g_i32_off)

    # ---- JIT launcher ----
    @flyc.jit
    def _launch_flash_attn_gfx1250(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,
        batch_size: fx.Int32,
        seq_len: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        # Finalize shared memory allocation (emits memref.global in gpu.module)
        from flydsl.compiler.kernel_function import CompilationContext
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            arena.finalized = False
            arena.finalize()

        # Grid: (num_q_tiles * num_heads, batch, 1)
        num_q_tiles = (seq_len + BLOCK_M - 1) // BLOCK_M
        grid_x = num_q_tiles * num_heads
        launcher = flash_attn_gfx1250_kernel(Q, K, V, O, batch_size, seq_len)
        launcher.launch(grid=(grid_x, batch_size, 1), block=(BLOCK_SIZE, 1, 1), stream=stream)

    return _launch_flash_attn_gfx1250
