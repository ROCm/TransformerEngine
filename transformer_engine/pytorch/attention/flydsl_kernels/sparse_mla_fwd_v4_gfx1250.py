# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""DeepSeek V4 Sparse MLA forward kernel for gfx1250 (WMMA, wave32).

Implements the DSv4 sparse attention forward pass:
- Lightning indexer pre-selects TOPK KV tokens per query (external)
- Kernel gathers KV via topk_indices and computes attention over selected tokens
- Attention sink: optional per-head learnable logit that dilutes softmax
- MQA mode: single KV head shared across all Q heads (MLA latent space)

Key trick: V = K_lora (first D_V elements of KV), no extra global load needed.

Grid: (total_tokens, cdiv(num_heads, BLOCK_H))
Block: (WAVE_SIZE=32, 1, 1) — single wave per workgroup

Tile parameters:
- BLOCK_H = 16: Q heads per workgroup (= WMMA_M)
- TILE_K = 32: gathered KV tokens per inner loop step (= WMMA_K)
"""

import functools
import math as host_math
from typing import Optional

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import scf
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl, vector
from flydsl.expr import math as fmath
from flydsl.expr.arith import _to_raw as _raw
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr, check_smem_capacity
from .gemm_common_gfx1250 import (
    extract_lds_base_idx,
    lds_load_b128_raw,
    lds_transpose_load_raw,
)

WMMA_M, WMMA_N, WMMA_K = 16, 16, 32
WAVE_SIZE = 32
NUM_WAVES = 4
BLOCK_SIZE = NUM_WAVES * WAVE_SIZE  # 128 threads — parallelizes the cooperative
                                    # load + transposed-K scatter 4x vs 1 wave.
BLOCK_H = 16  # heads per workgroup = WMMA_M
TILE_K = 32   # gathered tokens per inner loop step = WMMA_K

_LOG2E = host_math.log2(host_math.e)


def build_sparse_mla_fwd_v4_gfx1250(
    num_heads: int = 64,
    kv_lora_rank: int = 512,
    d_rope: int = 64,
    topk: int = 640,
    has_sink: bool = True,
    scale: float = None,
    waves_per_eu: int = 1,
):
    """Build a DSv4 sparse MLA forward kernel for gfx1250.

    Returns a callable: launch_fn(q, kv, topk_indices, sink, o, lse, stream=...)

    Args:
        num_heads: Number of Q attention heads.
        kv_lora_rank: Dimension of the KV latent (D_V). Also the output dim.
        d_rope: RoPE dimension appended to the latent.
        topk: Number of KV tokens selected by the lightning indexer.
        has_sink: Whether attention sink is used.
        scale: Attention scale (default: 1/sqrt(D_QK)).
    """
    gpu_arch = get_hip_arch()

    D_V = kv_lora_rank
    D_ROPE = d_rope
    D_QK = D_V + D_ROPE
    TOPK = topk
    HAS_SINK = has_sink

    if scale is None:
        scale = 1.0 / host_math.sqrt(D_QK)
    SCALE_LOG2E = scale * _LOG2E

    assert D_V % WMMA_N == 0, f"D_V={D_V} must be multiple of WMMA_N={WMMA_N}"
    assert D_ROPE % WMMA_K == 0 or D_ROPE % WMMA_N == 0, f"D_ROPE={D_ROPE} must align"
    assert TOPK % TILE_K == 0, f"TOPK={TOPK} must be multiple of TILE_K={TILE_K}"

    elem_bytes = 2  # bf16
    is_bf16 = True
    wmma_op = rocdl.wmma_f32_16x16x32_bf16

    # WMMA tile counts
    # Score GEMM: S[BH=16, TK=32] = Q[16, D] × K^T[D, 32]
    SCORE_N_TILES = TILE_K // WMMA_N   # 32/16 = 2
    SCORE_K_STEPS_LORA = D_V // WMMA_K  # 512/32 = 16
    SCORE_K_STEPS_ROPE = D_ROPE // WMMA_K  # 64/32 = 2

    # Acc GEMM: O[BH=16, D_V] += P[16, 32] × V[32, D_V]
    ACC_N_TILES = D_V // WMMA_N  # 512/16 = 32
    ACC_K_STEPS = TILE_K // WMMA_K  # 32/32 = 1
    # Split the acc N-tiles across the waves: wave w computes tiles
    # [w*ACC_TILES_PER_WAVE : (w+1)*ACC_TILES_PER_WAVE]. Score+softmax stay
    # per-wave (each wave needs the full P), only the P@V acc is partitioned.
    assert ACC_N_TILES % NUM_WAVES == 0, (
        f"ACC_N_TILES ({ACC_N_TILES}) must be divisible by NUM_WAVES ({NUM_WAVES}); "
        f"D_V={D_V} too small for {NUM_WAVES} waves"
    )
    ACC_TILES_PER_WAVE = ACC_N_TILES // NUM_WAVES

    # Number of TOPK tiles
    NUM_TOPK_TILES = TOPK // TILE_K

    # LDS layout: [Q | V (=KV natural [ktok,d]) | K (transposed [d,ktok]) | P | Valid]
    # Q tile [BLOCK_H, D_QK] — loaded once at start, persistent
    LDS_Q_STRIDE = D_QK + 8  # pad
    LDS_Q_SIZE = BLOCK_H * LDS_Q_STRIDE * elem_bytes
    # V tile [TILE_K, D_QK] natural layout (ktok rows) — for acc gemm O=P@V
    LDS_KV_STRIDE = D_QK + 8  # pad to avoid bank conflicts
    LDS_KV_SIZE = TILE_K * LDS_KV_STRIDE * elem_bytes
    # K tile [D_QK, TILE_K] transposed layout (d rows) — for score gemm S=Q@K^T.
    # ds_load_tr's transpose direction is tied to the physical row/col layout, so
    # the score genuinely needs d physically at the row stride (separate copy).
    LDS_KT_STRIDE = TILE_K + 8  # pad
    LDS_KT_SIZE = D_QK * LDS_KT_STRIDE * elem_bytes
    # P tile [BLOCK_H, TILE_K] for softmax output → WMMA A-operand reload
    LDS_P_STRIDE = TILE_K + 8
    LDS_P_SIZE = BLOCK_H * LDS_P_STRIDE * elem_bytes
    # Validity flags [TILE_K] i32 — 0=valid, 1=invalid
    LDS_VALID_SIZE = TILE_K * 4  # TILE_K i32 values

    LDS_Q_OFFSET = 0
    LDS_KV_OFFSET = LDS_Q_SIZE
    LDS_KT_OFFSET = LDS_Q_SIZE + LDS_KV_SIZE
    LDS_P_OFFSET = LDS_Q_SIZE + LDS_KV_SIZE + LDS_KT_SIZE
    LDS_VALID_OFFSET = LDS_Q_SIZE + LDS_KV_SIZE + LDS_KT_SIZE + LDS_P_SIZE
    LDS_TOTAL = LDS_VALID_OFFSET + LDS_VALID_SIZE

    # Strides for Q and O tensors: [T, H, D]
    STRIDE_Q_H = D_QK
    STRIDE_Q_T = num_heads * D_QK
    STRIDE_KV_T = 1 * D_QK  # MQA: 1 KV head
    STRIDE_O_H = D_V
    STRIDE_O_T = num_heads * D_V

    arena = SmemAllocator(None, arch=gpu_arch, global_sym_name="sparse_mla_v4_arena")
    arena.ptr = LDS_TOTAL
    check_smem_capacity(LDS_TOTAL, gpu_arch)

    compile_hints = dict(
        waves_per_eu=waves_per_eu,
        fast_fp_math=True,
        unsafe_fp_math=True,
    )

    @flyc.kernel
    def sparse_mla_fwd_v4_kernel(
        Q: fx.Tensor,        # [T, H, D_QK] bf16
        GKV: fx.Tensor,      # [T, TOPK, D_QK] bf16 (pre-gathered)
        VMask: fx.Tensor,    # [T, TOPK] int32 (0=valid, 1=invalid)
        Sink: fx.Tensor,     # [H] fp32
        O: fx.Tensor,        # [T, H, D_V] bf16
        LSE: fx.Tensor,      # [T, H] fp32
        i32_total_tokens: fx.Int32,
    ):
        token_idx = gpu.block_id("x")
        head_group_idx = gpu.block_id("y")
        tx = gpu.thread_id("x")

        from flydsl.expr.typing import T
        from flydsl.expr.typing import Numeric as _Numeric

        total_tokens = arith.index_cast(T.index, i32_total_tokens.ir_value())
        head_base = head_group_idx * arith.index(BLOCK_H)

        # Wave/lane decomposition (128 threads = 4 waves of 32)
        wave_id = tx // arith.index(WAVE_SIZE)
        lane_in_wave = tx % arith.index(WAVE_SIZE)
        lane16 = lane_in_wave % arith.index(16)
        lane_kgrp = lane_in_wave // arith.index(16)  # 0 or 1

        elem_ty = ir.BF16Type.get()
        elem_dtype = _Numeric.from_ir_type(elem_ty)

        # ---- LDS setup ----
        arena_base = arena.get_base()
        lds_smem = SmemPtr(arena_base, 0, T.bf16, shape=(LDS_TOTAL // elem_bytes,))
        lds_base = extract_lds_base_idx(lds_smem)
        q_lds_base = lds_base + arith.index(LDS_Q_OFFSET)
        kv_lds_base = lds_base + arith.index(LDS_KV_OFFSET)
        kt_lds_base = lds_base + arith.index(LDS_KT_OFFSET)
        p_lds_base = lds_base + arith.index(LDS_P_OFFSET)
        valid_lds_base = lds_base + arith.index(LDS_VALID_OFFSET)

        # ---- Buffer resources ----
        from flydsl.expr.typing import T
        q_rsrc = buffer_ops.create_buffer_resource(Q, max_size=True)
        gkv_rsrc = buffer_ops.create_buffer_resource(GKV, max_size=True)
        vmask_rsrc = buffer_ops.create_buffer_resource(VMask, max_size=True)
        o_rsrc = buffer_ops.create_buffer_resource(O, max_size=True)

        # ---- Q base offset (for WMMA on wave 0) ----
        q_base_offset = token_idx * arith.index(STRIDE_Q_T) + (head_base + lane16) * arith.index(STRIDE_Q_H)

        # ---- Cooperative Q load to LDS (ALL 128 threads) ----
        ELEMS_PER_LOAD = 8  # dwordx4
        Q_TOTAL_ELEMS = BLOCK_H * D_QK
        Q_LOADS_PER_THREAD = (Q_TOTAL_ELEMS + BLOCK_SIZE * ELEMS_PER_LOAD - 1) // (BLOCK_SIZE * ELEMS_PER_LOAD)
        q_global_base = token_idx * arith.index(STRIDE_Q_T) + head_base * arith.index(STRIDE_Q_H)
        for li in range_constexpr(Q_LOADS_PER_THREAD):
            flat_idx = (tx + arith.index(li * BLOCK_SIZE)) * arith.index(ELEMS_PER_LOAD)
            q_in_bounds = arith.cmpi(arith.CmpIPredicate.slt, flat_idx, arith.index(Q_TOTAL_ELEMS))
            if q_in_bounds:
                row = flat_idx // arith.index(D_QK)
                col = flat_idx % arith.index(D_QK)
                g_offset = q_global_base + row * arith.index(STRIDE_Q_H) + col
                g_i32_off = fx.Int32(g_offset * arith.index(elem_bytes) // arith.index(4))
                data = buffer_ops.buffer_load(q_rsrc, g_i32_off, vec_width=ELEMS_PER_LOAD // 2, dtype=T.i32)
                lds_offset = row * arith.index(LDS_Q_STRIDE * elem_bytes) + col * arith.index(elem_bytes)
                from flydsl._mlir.dialects import llvm as llvm_d
                lds_ptr = buffer_ops.create_llvm_ptr(
                    _raw(arith.index_cast(T.i32, q_lds_base + lds_offset)), address_space=3)
                llvm_d.store(data, lds_ptr)
        rocdl.s_wait_dscnt(0)
        gpu.barrier()

        # ---- Precompute Q A-operand lane bases from LDS ----
        q_row_off = lane16 * arith.index(LDS_Q_STRIDE * elem_bytes)
        q_k_off = lane_kgrp * arith.index(8 * elem_bytes)
        q_lane_base = q_row_off + q_k_off

        # ---- Initialize O accumulators and softmax state ----
        # Each wave maintains only its slice (ACC_TILES_PER_WAVE) of the D_V tiles.
        o_accs = [arith.constant_vector(0.0, T.vec(8, T.f32)) for _ in range(ACC_TILES_PER_WAVE)]
        m_val = arith.constant(float('-inf'), type=T.f32)
        l_val = arith.constant(0.0, type=T.f32)

        iter_m = m_val
        iter_l = l_val
        iter_o = list(o_accs)

        for tile_iter in range_constexpr(NUM_TOPK_TILES):
            tile_base = arith.index(tile_iter * TILE_K)
            zero_i32 = arith.constant(0, type=T.i32)

            # ---- Phase 1: Load gathered KV tile to LDS (ALL 128 threads) ----
            # GKV is [T, TOPK, D_QK] bf16 pre-gathered. Load tile [TILE_K, D_QK].
            # GKV base for this token + tile: token_idx * TOPK * D_QK + tile_base * D_QK
            STRIDE_GKV_TOKEN = TOPK * D_QK
            gkv_base = token_idx * arith.index(STRIDE_GKV_TOKEN) + tile_base * arith.index(D_QK)
            ELEMS_PER_LOAD = 8  # dwordx4
            TOTAL_KV_ELEMS = TILE_K * D_QK
            KV_LOADS_PER_THREAD = (TOTAL_KV_ELEMS + BLOCK_SIZE * ELEMS_PER_LOAD - 1) // (BLOCK_SIZE * ELEMS_PER_LOAD)

            for li in range_constexpr(KV_LOADS_PER_THREAD):
                flat_idx = (tx + arith.index(li * BLOCK_SIZE)) * arith.index(ELEMS_PER_LOAD)
                in_bounds = arith.cmpi(arith.CmpIPredicate.slt, flat_idx, arith.index(TOTAL_KV_ELEMS))
                if in_bounds:
                    row = flat_idx // arith.index(D_QK)   # ktok
                    col = flat_idx % arith.index(D_QK)    # d (start of 8-elem chunk)
                    g_offset = gkv_base + row * arith.index(D_QK) + col
                    g_i32_off = fx.Int32(g_offset * arith.index(elem_bytes) // arith.index(4))
                    data = buffer_ops.buffer_load(gkv_rsrc, g_i32_off, vec_width=ELEMS_PER_LOAD // 2, dtype=T.i32)
                    from flydsl._mlir.dialects import llvm as llvm_d
                    # Store natural [ktok, d] for V (acc gemm).
                    lds_offset = row * arith.index(LDS_KV_STRIDE * elem_bytes) + col * arith.index(elem_bytes)
                    lds_ptr = buffer_ops.create_llvm_ptr(
                        _raw(arith.index_cast(T.i32, kv_lds_base + lds_offset)), address_space=3)
                    llvm_d.store(data, lds_ptr)
                    # Store transposed [d, ktok] for K (score gemm): scatter 8 d-values
                    # to K_lds rows (col+0..col+7), same ktok column.
                    data_vec = fx.Vector(data).bitcast(elem_dtype)  # 8 bf16
                    for di in range_constexpr(ELEMS_PER_LOAD):
                        elem = vector.extract(data_vec.ir_value(), static_position=[di], dynamic_position=[])
                        kt_offset = (col + arith.index(di)) * arith.index(LDS_KT_STRIDE * elem_bytes) + row * arith.index(elem_bytes)
                        kt_ptr = buffer_ops.create_llvm_ptr(
                            _raw(arith.index_cast(T.i32, kt_lds_base + kt_offset)), address_space=3)
                        llvm_d.store(elem, kt_ptr)

            # Also load validity mask for this tile to LDS
            # VMask is [T, TOPK] int32. Load TILE_K entries.
            vmask_base = token_idx * arith.index(TOPK) + tile_base
            # Only first TILE_K threads need to load (1 i32 each)
            is_vmask_loader = arith.cmpi(arith.CmpIPredicate.slt, tx, arith.index(TILE_K))
            if is_vmask_loader:
                v_off = vmask_base + tx
                v_flag = buffer_ops.buffer_load(vmask_rsrc, fx.Int32(v_off), vec_width=1, dtype=T.i32)
                from flydsl._mlir.dialects import llvm as llvm_d
                vptr = buffer_ops.create_llvm_ptr(
                    _raw(arith.index_cast(T.i32, valid_lds_base + tx * arith.index(4))), address_space=3)
                llvm_d.store(v_flag, vptr)

            rocdl.s_wait_dscnt(0)
            gpu.barrier()

            # ---- Score computation: S[16, 32] = Q_lora·K_lora^T + Q_rope·K_rope^T ----
            # B-operand lane bases (transpose load from LDS)
            lane8 = lane16 % arith.index(8)
            lane_ngrp = lane16 // arith.index(8)
            # For acc V (KV natural [ktok, d], stride LDS_KV_STRIDE)
            kv_k_lane = (lane_kgrp * arith.index(8) + lane8) * arith.index(LDS_KV_STRIDE * elem_bytes)
            kv_n_lane = lane_ngrp * arith.index(8 * elem_bytes)
            # For score K (transposed kt_lds [d, ktok], stride LDS_KT_STRIDE)
            kt_k_lane = (lane_kgrp * arith.index(8) + lane8) * arith.index(LDS_KT_STRIDE * elem_bytes)
            kt_n_lane = lane_ngrp * arith.index(8 * elem_bytes)

            # All waves redundantly compute the full score (the 4 SIMDs are otherwise
            # idle during the score phase, so the redundancy is free wall-clock; each
            # wave needs the full P for its acc slice).
            s_accs = [arith.constant_vector(0.0, T.vec(8, T.f32)) for _ in range_constexpr(SCORE_N_TILES)]

            # Score GEMM: Q_lora[16, D_V] × K_lora^T[D_V, 32]
            for ks in range_constexpr(SCORE_K_STEPS_LORA):
                q_k_byte_off = arith.index(ks * WMMA_K * elem_bytes)
                q_off0 = q_lane_base + q_k_byte_off
                q_off1 = q_off0 + arith.index(32)
                q_lo_v = fx.Vector(lds_load_b128_raw(q_lds_base, q_off0)).bitcast(elem_dtype)
                q_hi_v = fx.Vector(lds_load_b128_raw(q_lds_base, q_off1)).bitcast(elem_dtype)
                q_frag = q_lo_v.shuffle(q_hi_v, list(range(16)))
                for wn in range_constexpr(SCORE_N_TILES):
                    vec8_ty = ir.VectorType.get([8], elem_ty)
                    n_col = arith.index(wn * WMMA_N * elem_bytes) + kt_n_lane
                    b_base = kt_k_lane + n_col
                    results = []
                    for k_half in range_constexpr(2):
                        k_row_off = (ks * WMMA_K + k_half * 16) * LDS_KT_STRIDE * elem_bytes
                        elem_off = b_base + arith.index(k_row_off)
                        v = lds_transpose_load_raw(vec8_ty, kt_lds_base, elem_off)
                        results.append(fx.Vector(v))
                    k_frag = results[0].shuffle(results[1], list(range(16)))
                    rocdl.s_wait_dscnt(0)
                    s_accs[wn] = wmma_op(
                        T.vec(8, T.f32), k_frag, q_frag, s_accs[wn],
                        signA=False, signB=False, modC=0, reuseA=False, reuseB=False,
                    ).result

            # Score GEMM: Q_rope[16, D_ROPE] × K_rope^T[D_ROPE, 32]
            for ks in range_constexpr(SCORE_K_STEPS_ROPE):
                q_k_byte_off = arith.index((D_V + ks * WMMA_K) * elem_bytes)
                q_off0 = q_lane_base + q_k_byte_off
                q_off1 = q_off0 + arith.index(32)
                q_lo_v = fx.Vector(lds_load_b128_raw(q_lds_base, q_off0)).bitcast(elem_dtype)
                q_hi_v = fx.Vector(lds_load_b128_raw(q_lds_base, q_off1)).bitcast(elem_dtype)
                q_frag = q_lo_v.shuffle(q_hi_v, list(range(16)))
                for wn in range_constexpr(SCORE_N_TILES):
                    vec8_ty = ir.VectorType.get([8], elem_ty)
                    n_col = arith.index(wn * WMMA_N * elem_bytes) + kt_n_lane
                    b_base = kt_k_lane + n_col
                    results = []
                    for k_half in range_constexpr(2):
                        k_row_off = (D_V + ks * WMMA_K + k_half * 16) * LDS_KT_STRIDE * elem_bytes
                        elem_off = b_base + arith.index(k_row_off)
                        v = lds_transpose_load_raw(vec8_ty, kt_lds_base, elem_off)
                        results.append(fx.Vector(v))
                    k_frag = results[0].shuffle(results[1], list(range(16)))
                    rocdl.s_wait_dscnt(0)
                    s_accs[wn] = wmma_op(
                        T.vec(8, T.f32), k_frag, q_frag, s_accs[wn],
                        signA=False, signB=False, modC=0, reuseA=False, reuseB=False,
                    ).result

            # ---- Apply scale (log2e) ----
            sm_scale_vec = arith.constant_vector(SCALE_LOG2E, T.vec(8, T.f32))
            for wn in range_constexpr(SCORE_N_TILES):
                s_accs[wn] = arith.mulf(s_accs[wn], sm_scale_vec)

            # ---- Mask invalid topk entries (read validity from LDS) ----
            for wn in range_constexpr(SCORE_N_TILES):
                for ei in range_constexpr(8):
                    # Column index for this element → which KV row
                    col_idx = lane_kgrp * arith.index(8) + arith.index(wn * WMMA_N + ei)
                    # Read validity from LDS
                    from flydsl._mlir.dialects import llvm as llvm_d
                    vptr = buffer_ops.create_llvm_ptr(
                        _raw(arith.index_cast(T.i32, valid_lds_base + col_idx * arith.index(4))), address_space=3)
                    v_flag = llvm_d.load(ir.IntegerType.get_signless(32), vptr)
                    is_inv = arith.cmpi(arith.CmpIPredicate.ne, v_flag, arith.constant(0, type=T.i32))
                    elem = vector.extract(s_accs[wn], static_position=[ei], dynamic_position=[])
                    masked = arith.select(is_inv, arith.constant(float('-inf'), type=T.f32), elem)
                    s_accs[wn] = vector.insert(masked, s_accs[wn], static_position=[ei], dynamic_position=[])

            # ---- Online softmax ----
            def vec8_max(v):
                val = vector.extract(v, static_position=[0], dynamic_position=[])
                for i in range_constexpr(1, 8):
                    e = vector.extract(v, static_position=[i], dynamic_position=[])
                    val = arith.maximumf(val, e)
                return val

            def vec8_sum(v):
                val = vector.extract(v, static_position=[0], dynamic_position=[])
                for i in range_constexpr(1, 8):
                    e = vector.extract(v, static_position=[i], dynamic_position=[])
                    val = arith.addf(val, e)
                return val

            # Cross-lane reduction (lane l <-> lane l^16 share same row)
            def reduce_pair(local_val, op="max"):
                pair_addr = (lane_in_wave ^ arith.index(16)) * arith.index(4)
                pair_byte = fx.Int32(pair_addr)
                local_i32 = arith.bitcast(T.i32, local_val)
                peer_i32 = rocdl.ds_bpermute(T.i32, pair_byte, local_i32)
                peer_val = arith.bitcast(T.f32, peer_i32)
                if const_expr(op == "max"):
                    return arith.maximumf(local_val, peer_val)
                return arith.addf(local_val, peer_val)

            # Row max
            local_max = vec8_max(s_accs[0])
            for wn in range_constexpr(1, SCORE_N_TILES):
                local_max = arith.maximumf(local_max, vec8_max(s_accs[wn]))
            row_max = reduce_pair(local_max, "max")

            m_new = arith.maximumf(iter_m, row_max)
            alpha = fmath.exp2(arith.subf(iter_m, m_new))
            alpha_vec = vector.broadcast(T.vec(8, T.f32), alpha)

            # Rescale running state (only this wave's acc slice)
            new_l = arith.mulf(iter_l, alpha)
            new_o = [arith.mulf(iter_o[i], alpha_vec) for i in range_constexpr(ACC_TILES_PER_WAVE)]

            # P = exp2(S - m_new)
            m_new_vec = vector.broadcast(T.vec(8, T.f32), m_new)
            p_vecs = []
            for wn in range_constexpr(SCORE_N_TILES):
                p_vec = fmath.exp2(arith.subf(s_accs[wn], m_new_vec))
                p_vecs.append(p_vec)

            # Row sum
            local_sum = vec8_sum(p_vecs[0])
            for wn in range_constexpr(1, SCORE_N_TILES):
                local_sum = arith.addf(local_sum, vec8_sum(p_vecs[wn]))
            row_sum = reduce_pair(local_sum, "sum")
            new_l = arith.addf(new_l, row_sum)

            # ---- Store P to LDS for WMMA A-operand reload ----
            p_row_off = lane16 * arith.index(LDS_P_STRIDE * elem_bytes)
            p_lane_base = p_row_off + lane_kgrp * arith.index(8 * elem_bytes)
            for wn in range_constexpr(SCORE_N_TILES):
                h_vec = arith.trunc_f(T.vec(8, elem_ty), p_vecs[wn])
                i32_vec = vector.bitcast(T.vec(4, T.i32), h_vec)
                off = p_lane_base + arith.index(wn * WMMA_N * elem_bytes)
                from flydsl._mlir.dialects import llvm as llvm_d
                ptr = buffer_ops.create_llvm_ptr(
                    _raw(arith.index_cast(T.i32, p_lds_base + off)), address_space=3)
                llvm_d.store(i32_vec, ptr)

            rocdl.s_wait_dscnt(0)
            gpu.barrier()

            # ---- O accumulation: O[16, D_V] += P[16, 32] × V[32, D_V] ----
            # P is in LDS as [16, 32+pad], load as WMMA A-operand
            # V = K_lora in LDS as [32, D_QK+pad], first D_V cols, load as B-operand (transpose)
            p_a_lane_base = p_row_off + lane_kgrp * arith.index(8 * elem_bytes)

            # Load P A-fragment (TILE_K=32=WMMA_K, single K-step)
            p_off0 = p_a_lane_base
            p_off1 = p_a_lane_base + arith.index(32)  # +32 bytes = +16 bf16
            p_v0 = fx.Vector(lds_load_b128_raw(p_lds_base, p_off0)).bitcast(elem_dtype)
            p_v1 = fx.Vector(lds_load_b128_raw(p_lds_base, p_off1)).bitcast(elem_dtype)
            p_frag = p_v0.shuffle(p_v1, list(range(16)))

            # Accumulate P × V — this wave owns D_V tiles [wave*ATPW : +ATPW].
            wave_acc_base = wave_id * arith.index(ACC_TILES_PER_WAVE)
            for wn_local in range_constexpr(ACC_TILES_PER_WAVE):
                wn_glob = wave_acc_base + arith.index(wn_local)  # runtime tile index
                # Load V B-fragment (transpose from KV LDS, first D_V cols)
                vec8_ty = ir.VectorType.get([8], elem_ty)
                n_col = wn_glob * arith.index(WMMA_N * elem_bytes) + kv_n_lane
                b_base = kv_k_lane + n_col  # V = first D_V cols of KV
                results = []
                for k_half in range_constexpr(2):
                    k_row_off = k_half * 16 * LDS_KV_STRIDE * elem_bytes
                    elem_off = b_base + arith.index(k_row_off)
                    v = lds_transpose_load_raw(vec8_ty, kv_lds_base, elem_off)
                    results.append(fx.Vector(v))
                v_frag = results[0].shuffle(results[1], list(range(16)))

                rocdl.s_wait_dscnt(0)
                new_o[wn_local] = wmma_op(
                    T.vec(8, T.f32), v_frag, p_frag, new_o[wn_local],
                    signA=False, signB=False, modC=0, reuseA=False, reuseB=False,
                ).result

            gpu.barrier()
            rocdl.s_wait_dscnt(0)

            # Update state for next iteration
            iter_m = m_new
            iter_l = new_l
            iter_o = new_o

        # ---- Extract final loop values ----
        final_m = iter_m
        final_l = iter_l
        final_o = iter_o

        # ---- Sink epilogue ----
        if const_expr(HAS_SINK):
            from flydsl.expr.typing import T
            sink_rsrc = buffer_ops.create_buffer_resource(Sink, max_size=True)
            head_idx = head_base + lane16
            sink_val = buffer_ops.buffer_load(sink_rsrc, fx.Int32(head_idx), vec_width=1, dtype=T.f32)
            # Scale sink to log2e domain
            sink_log2 = arith.mulf(sink_val, arith.constant(_LOG2E, type=T.f32))
            m_final = arith.maximumf(final_m, sink_log2)
            alpha_fix = fmath.exp2(arith.subf(final_m, m_final))
            sink_exp = fmath.exp2(arith.subf(sink_log2, m_final))
            l_total = arith.addf(arith.mulf(final_l, alpha_fix), sink_exp)
            inv_l = arith.divf(arith.constant(1.0, type=T.f32), l_total)
            alpha_inv_l = arith.mulf(alpha_fix, inv_l)
            alpha_inv_l_vec = vector.broadcast(T.vec(8, T.f32), alpha_inv_l)
            for i in range_constexpr(ACC_TILES_PER_WAVE):
                final_o[i] = arith.mulf(final_o[i], alpha_inv_l_vec)
            # LSE = m_final/log2e + log(l_total) = m_final/log2e + log(l_total)
            inv_log2e = arith.constant(1.0 / _LOG2E, type=T.f32)
            m_natural = arith.mulf(m_final, inv_log2e)
            from flydsl.expr import math as _fmath
            final_lse = arith.addf(m_natural, _fmath.log(l_total))
        else:
            from flydsl.expr.typing import T
            inv_l = arith.divf(arith.constant(1.0, type=T.f32), final_l)
            inv_l_vec = vector.broadcast(T.vec(8, T.f32), inv_l)
            for i in range_constexpr(ACC_TILES_PER_WAVE):
                final_o[i] = arith.mulf(final_o[i], inv_l_vec)
            inv_log2e = arith.constant(1.0 / _LOG2E, type=T.f32)
            m_natural = arith.mulf(final_m, inv_log2e)
            from flydsl.expr import math as _fmath
            final_lse = arith.addf(m_natural, _fmath.log(final_l))

        # ---- Store O and LSE ----
        from flydsl.expr.typing import T
        o_base = token_idx * arith.index(STRIDE_O_T) + (head_base + lane16) * arith.index(STRIDE_O_H)
        o_col_base = lane_kgrp * arith.index(8)
        # This wave stores only its D_V tile slice [wave*ATPW : +ATPW].
        wave_acc_base = wave_id * arith.index(ACC_TILES_PER_WAVE)
        for wn_local in range_constexpr(ACC_TILES_PER_WAVE):
            wn_glob = wave_acc_base + arith.index(wn_local)
            col = o_col_base + wn_glob * arith.index(WMMA_N)
            g_offset = o_base + col
            g_i32_off = fx.Int32(g_offset * arith.index(elem_bytes) // arith.index(4))
            h_vec = arith.trunc_f(T.vec(8, elem_ty), final_o[wn_local])
            i32_vec = vector.bitcast(T.vec(4, T.i32), h_vec)
            buffer_ops.buffer_store(i32_vec, o_rsrc, g_i32_off)

        # Store LSE[token, head]
        lse_rsrc = buffer_ops.create_buffer_resource(LSE, max_size=True)
        lse_offset = token_idx * arith.index(num_heads) + head_base + lane16
        # Only lane_kgrp=0 stores (both groups have the same LSE for their row)
        is_lane0 = arith.cmpi(arith.CmpIPredicate.eq, lane_kgrp, arith.index(0))
        if is_lane0:
            buffer_ops.buffer_store(final_lse, lse_rsrc, fx.Int32(lse_offset))

    # ---- JIT launcher ----
    @flyc.jit
    def _launch_sparse_mla_fwd_v4(
        Q: fx.Tensor,
        GKV: fx.Tensor,
        VMask: fx.Tensor,
        Sink: fx.Tensor,
        O: fx.Tensor,
        LSE: fx.Tensor,
        total_tokens: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        from flydsl.compiler.kernel_function import CompilationContext
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            arena.finalized = False
            arena.finalize()

        num_head_groups = (num_heads + BLOCK_H - 1) // BLOCK_H
        launcher = sparse_mla_fwd_v4_kernel(Q, GKV, VMask, Sink, O, LSE, total_tokens)
        launcher.launch(
            grid=(total_tokens, num_head_groups, 1),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    return _launch_sparse_mla_fwd_v4


# ---- PyTorch wrapper ----

@functools.lru_cache(maxsize=64)
def _build_sparse_mla_fwd_v4_cached(num_heads, kv_lora_rank, d_rope, topk, has_sink, scale):
    """Build (and cache) the compiled kernel for a given shape signature.

    Module-level so the lru_cache persists across calls. Defining the cache
    inside the wrapper recreates it every call, so the kernel was rebuilt on
    every invocation (~55 ms fixed cost). scale is part of the key (None and
    floats are both hashable).
    """
    return build_sparse_mla_fwd_v4_gfx1250(
        num_heads=num_heads,
        kv_lora_rank=kv_lora_rank,
        d_rope=d_rope,
        topk=topk,
        has_sink=has_sink,
        scale=scale,
    )


def sparse_mla_fwd_v4(
    q: torch.Tensor,
    kv: torch.Tensor,
    topk_indices: torch.Tensor,
    attn_sink: Optional[torch.Tensor] = None,
    kv_lora_rank: int = 512,
    scale: float = None,
) -> tuple:
    """DeepSeek V4 Sparse MLA Forward (gfx1250).

    Args:
        q: [T, H, D_QK] bf16 — query tensor
        kv: [T, 1, D_QK] bf16 — KV latent tensor (MQA, 1 head)
        topk_indices: [T, TOPK] int32 — selected token indices (-1 = invalid)
        attn_sink: [H] fp32 — per-head sink logit (None = no sink)
        kv_lora_rank: D_V dimension (default 512)
        scale: attention scale (default 1/sqrt(D_QK))

    Returns:
        (o, lse): output [T, H, D_V] bf16, logsumexp [T, H] fp32
    """
    T_tok, H, D_QK = q.shape
    TOPK = topk_indices.shape[1]
    D_V = kv_lora_rank
    D_ROPE = D_QK - D_V
    has_sink = attn_sink is not None

    o = torch.empty(T_tok, H, D_V, dtype=q.dtype, device=q.device)
    lse = torch.empty(T_tok, H, dtype=torch.float32, device=q.device)

    if attn_sink is None:
        attn_sink = torch.zeros(H, dtype=torch.float32, device=q.device)

    # Cached at module scope so the executable is built once per shape and
    # reused across calls (rebuilding per call cost a fixed ~55 ms).
    exe = _build_sparse_mla_fwd_v4_cached(H, D_V, D_ROPE, TOPK, has_sink, scale)

    # Pre-gather KV in PyTorch to avoid indirect loads in kernel
    invalid = (topk_indices < 0) | (topk_indices >= T_tok)
    safe_idx = topk_indices.clamp(0, T_tok - 1).long()
    gathered_kv = kv.squeeze(1)[safe_idx]  # [T, TOPK, D_QK] bf16
    # Validity mask: 0=valid, 1=invalid (int32 per TOPK entry)
    valid_mask = invalid.int()  # [T, TOPK] int32

    q_flat = q.contiguous().reshape(-1)
    gkv_flat = gathered_kv.contiguous().reshape(-1)
    vmask_flat = valid_mask.contiguous().reshape(-1)
    o_flat = o.reshape(-1)
    lse_flat = lse.reshape(-1)

    stream = torch.cuda.current_stream(q.device)
    exe(q_flat, gkv_flat, vmask_flat, attn_sink, o_flat, lse_flat, T_tok, stream=stream)

    return o, lse


# ---- PyTorch reference (for validation) ----

def ref_sparse_mla_fwd_v4(
    q: torch.Tensor,
    kv: torch.Tensor,
    topk_indices: torch.Tensor,
    attn_sink: Optional[torch.Tensor],
    kv_lora_rank: int = 512,
    scale: float = None,
) -> tuple:
    """Reference implementation in PyTorch (float32 accumulation)."""
    T_tok, H, D_QK = q.shape
    D_V = kv_lora_rank
    TOPK = topk_indices.shape[1]

    if scale is None:
        scale = 1.0 / host_math.sqrt(D_QK)

    # Gather KV
    invalid = (topk_indices < 0) | (topk_indices >= T_tok)
    safe_idx = topk_indices.clamp(0, T_tok - 1).long()
    gathered_kv = kv.squeeze(1).float()[safe_idx]  # [T, TOPK, D_QK]

    # Score
    S = torch.einsum("thd,tkd->thk", q.float(), gathered_kv) * scale
    S.masked_fill_(invalid[:, None, :].expand_as(S), float('-inf'))

    # Softmax with optional sink
    if attn_sink is not None:
        sink_expanded = attn_sink.view(1, H, 1).expand(T_tok, H, 1)
        S_ext = torch.cat([S, sink_expanded], dim=-1)
        lse = torch.logsumexp(S_ext, dim=-1)
    else:
        lse = torch.logsumexp(S, dim=-1)

    P = torch.exp(S - lse.unsqueeze(-1))
    P.masked_fill_(invalid[:, None, :].expand_as(P), 0.0)

    V = gathered_kv[..., :D_V]
    O = torch.einsum("thk,tkd->thd", P, V)

    return O.to(q.dtype), lse
