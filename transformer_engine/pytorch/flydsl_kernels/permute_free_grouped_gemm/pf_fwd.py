# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Permute-free MoE forward gather grouped-GEMM: MegaMOE's fast bf16 GEMM + route-list gather.

This is a port of the MegaMOE grouped bf16 GEMM (32x32x16 MFMA, 8-wave / 512-thread
workgroup, deep distance-2 3-buffer DMA ring, XCD swizzle with ``GROUP_M`` front-loading)
into Transformer Engine FlyDSL, with a *single* change: the A operand is fetched
through a per-row gather index instead of a contiguous pool.

The gather is folded into MegaMOE's pipeline rather than implemented as a separate
pre-permute: we keep Mega's DMA/LDS/MFMA path verbatim and only redirect the A fetch
through ``sorted_slot_ids``, preserving Mega's throughput while keeping the permute-free
memory model (no pre-permutation, gather-on-demand).

Contract (mirrors MegaMOE ``grouped_gemm_bf16_only`` + permute-free routing metadata):
  * ``A``            [num_recv, K] bf16   received-token activations, UNPERMUTED (gather source)
  * ``B``            [E, N, K]    bf16    per-expert weights, NT (contiguous inner K)
  * ``C``            [em_max, N]  bf16    block-padded route-ordered output (expert-major, in place)
  * ``sorted_slot_ids`` [em_max]  i32     received-token row per padded slot (sentinel = num_recv)
  * ``expert_ids``   [num_m_blocks] i32   expert id per ``BLOCK_M`` output block (padding tail:
                                          ``-1``; those blocks early-exit in-kernel)
  * ``num_tile_blocks`` [1]      i32      real (non-padding) ``BLOCK_M`` block count (device)

Gated activation and route-prob apply live in standalone Triton helpers, not here.
"""

from __future__ import annotations

import functools

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, range_constexpr
from flydsl.expr.buffer_ops import (
    buffer_load,
    create_buffer_resource,
    extract_base_index,
)
from flydsl.expr.typing import AddressSpace, PointerType

from ..gemm.half_prec_gemm import BLOCK_K, dense_mma_pipeline_bf16
from ..gemm.fp16_gemm_utils import G2SLoader, ceildiv, make_bf16_buffer_tensor, swizzle_128
from ..gemm.pf_gemm_utils import (
    Mfma32x32x16,
    S2RLoaderBf16,
    StoreCBf16,
    _i64,
    _make_shared_storage,
    compute_global_gather_swizzle_bf16,
    compute_global_swizzle_bf16,
    make_value_attrs,
    xcd_remap_pid,
)

__all__ = ["compile_grouped_gemm_gather_bf16", "grouped_gemm_gather_bf16"]


def compute_global_identity_swizzle_bf16(lane_id, wave_id, K, n_rounds, sorted_row_base):
    """Per-lane global A offsets reading the *route row directly* (identity gather).

    Same flat-buffer pipeline as :func:`compute_global_gather_swizzle_bf16` but the source row is
    ``sorted_row_base + row`` computed arithmetically instead of loaded from a gather table. This
    is the FC2 route-read (``index_a_by_route_pos``) path: it needs no ``sorted_slot_ids`` tensor,
    so it stays inside a HIP graph capture (no per-call identity allocation) while reusing the
    proven whole-buffer / base-0 gather tile (avoiding the per-tile A-rebase multi-block hazard).
    """
    offsets = []
    n_waves = fx.block_dim.x // 64
    for r in range_constexpr(n_rounds):
        row = lane_id // 8 + wave_id * 8 + r * (n_waves * 8)
        col_byte = (lane_id % 8) * 16
        _, c = swizzle_128(row, col_byte)
        offsets.append((sorted_row_base + row) * K + c // 2)
    return offsets


def gemm_bf16_nt_gather_tile(
    A,
    B_T,
    C,
    c_m,
    c_n,
    lds,
    sorted_res,
    sorted_row_base,
    block_n,
    *,
    K,
    BLOCK_M,
    BLOCK_N,
    out_fp16=False,
    nt_vmcnt=3,
    b_group_base,
    gather=True,
):
    """One NT tile of the grouped GEMM with the A rows *gathered* via ``sorted_slot_ids``.

    Identical to MegaMOE's ``gemm_bf16_nt_tile`` except:
      * the two LDS A half-tiles use *gather* swizzles (rows redirected through
        ``sorted_slot_ids[sorted_row_base + tile_row]``), so the A base offset is 0 and only the
        K-step rides the load ``soffset``;
      * ``C`` is already rebased to this tile's row block by the caller, so the store block_m is 0.
    """
    assert BLOCK_M >= 128 and BLOCK_N >= 256 and BLOCK_M % 128 == 0 and BLOCK_N % 256 == 0
    assert K % BLOCK_K == 0, f"bf16 NT gather needs K % {BLOCK_K} == 0 (got K={K})"
    N_TILES_A = BLOCK_M // 128
    N_TILES_B = BLOCK_N // 256
    LDS_BLOCK_M = BLOCK_M // 2
    LDS_BLOCK_N = BLOCK_N // 2
    N_LDS_STEPS_A = LDS_BLOCK_M // 64
    N_LDS_STEPS_B = LDS_BLOCK_N // 64
    N_LDS_ROUNDS = max(N_LDS_STEPS_A, N_LDS_STEPS_B)

    lane_id = fx.thread_idx.x % 64
    wave_id = fx.thread_idx.x // 64
    wave_m = wave_id // 4
    wave_n = wave_id % 4

    # A rows carried by the gather offsets -> global base is 0 (K rides soffset only).
    A0_gl_offset = fx.Int32(0)
    A1_gl_offset = fx.Int32(0)
    B0_gl_offset = (block_n * BLOCK_N) * K
    B1_gl_offset = (block_n * BLOCK_N + LDS_BLOCK_N) * K
    if b_group_base is not None:
        B0_gl_offset = B0_gl_offset + b_group_base
        B1_gl_offset = B1_gl_offset + b_group_base

    # ``A`` MUST be a flat 1D buffer view (built by the caller): a linear gather offset
    # (src_row*K + col) indexes row-major elements. A raw 2D tensor's logical_divide/slice
    # indexes the outer (row) dim, so a flat offset runs off the end -> garbage.
    gA = make_bf16_buffer_tensor(A)
    gB = make_bf16_buffer_tensor(B_T)
    a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
    b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

    # Two gather swizzles: the lo half covers tile rows [0, LDS_BLOCK_M), the hi half
    # [LDS_BLOCK_M, BLOCK_M); each redirects its tile row through sorted_slot_ids. The two
    # halves gather independent rows, so they need distinct loaders (a_g2s / a_g2s_hi).
    if gather:
        gl_off_a_lo = compute_global_gather_swizzle_bf16(
            lane_id, wave_id, K, N_LDS_ROUNDS, sorted_res, sorted_row_base
        )
        gl_off_a_hi = compute_global_gather_swizzle_bf16(
            lane_id, wave_id, K, N_LDS_ROUNDS, sorted_res, sorted_row_base + fx.Int32(LDS_BLOCK_M)
        )
    else:
        # FC2 route-read: identity (row = route position), no sorted_slot_ids load.
        gl_off_a_lo = compute_global_identity_swizzle_bf16(
            lane_id, wave_id, K, N_LDS_ROUNDS, sorted_row_base
        )
        gl_off_a_hi = compute_global_identity_swizzle_bf16(
            lane_id, wave_id, K, N_LDS_ROUNDS, sorted_row_base + fx.Int32(LDS_BLOCK_M)
        )
    gl_off_b = compute_global_swizzle_bf16(lane_id, wave_id, K, N_LDS_ROUNDS)

    mfma = Mfma32x32x16(N_TILES_A, N_TILES_B)
    a_g2s = G2SLoader(a_div, gl_off_a_lo, N_LDS_STEPS_A, fx.BFloat16.ir_type, wave_id)
    a_g2s_hi = G2SLoader(a_div, gl_off_a_hi, N_LDS_STEPS_A, fx.BFloat16.ir_type, wave_id)
    b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, fx.BFloat16.ir_type, wave_id)
    a_s2r = S2RLoaderBf16(wave_m, N_TILES_A)
    b_s2r = S2RLoaderBf16(wave_n, N_TILES_B)
    _out_ty = fx.Float16 if out_fp16 else fx.BFloat16
    store_c = StoreCBf16(C, c_m, c_n, _out_ty)

    dense_mma_pipeline_bf16(
        lds,
        a_g2s,
        b_g2s,
        a_s2r,
        b_s2r,
        mfma,
        store_c,
        A0_gl_offset,
        A1_gl_offset,
        B0_gl_offset,
        B1_gl_offset,
        BLOCK_K,
        BLOCK_K,
        fx.Int32(0),  # store block_m: C is already rebased to this tile
        block_n,
        wave_m,
        wave_n,
        K,
        BLOCK_M,
        BLOCK_N,
        nt_vmcnt,
        a_g2s_hi=a_g2s_hi,
    )


@functools.lru_cache(maxsize=256)
def compile_grouped_gemm_gather_bf16(
    K,
    BLOCK_M=256,
    BLOCK_N=256,
    GROUP_M=4,
    num_xcd=1,
    nt_vmcnt=3,  # gfx950 G2S LDS hazard: vmcnt>=4 races (nondeterministic); 3 is det
    waves_per_eu=2,
    agpr_alloc=0,
    out_fp16=False,
    gather=True,
):
    """Compile (cached) the gathering grouped BF16 GEMM launcher for one ``(K, tile)`` combo.

    Grid is over-launched to the padded output pool; each block early-exits past the real tile
    range (``num_tile_blocks``) or when ``expert_ids[block_m] < 0`` (PF padding tail). Mirrors
    MegaMOE's ``compile_grouped_gemm_bf16`` (NT) exactly apart from the gather A fetch, the extra
    ``SORTED`` argument, and the padding-block guard. Returns the flyc launch callable.

    The FC2 route-read (``index_a_by_route_pos``) case reuses this same gathering kernel with an
    identity ``SORTED`` (``SORTED[s] = s``), so no separate no-gather tile is needed.
    """
    SharedStorage = _make_shared_storage(BLOCK_M, BLOCK_N)

    @flyc.kernel(known_block_size=[512, 1, 1])
    def grouped_gemm_gather_k(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        TILE_TO_GROUP: fx.Tensor,
        NUM_TILE_BLOCKS: fx.Int32,
        SORTED: fx.Tensor,
        A_ELEMS: fx.Int32,
        c_n: fx.Int32,
    ):
        n_blocks = ceildiv(c_n, BLOCK_N)
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        group_res = create_buffer_resource(TILE_TO_GROUP, max_size=True)
        sorted_res = create_buffer_resource(SORTED, max_size=True)
        # Real (non-padding) BLOCK_M tile count as a scalar (host-known, capture-safe -- avoids a
        # per-call device tensor that a HIP graph capture forbids). Host-known int scalar.
        real_tiles = NUM_TILE_BLOCKS
        # XCD-swizzle over the REAL tile range only (front-loaded); swizzling the full padded
        # pool scatters real tiles -> ~2x slower.
        real_grid = real_tiles * n_blocks

        pool_ptr_ty = PointerType.get(
            elem_ty=fx.BFloat16.ir_type, address_space=AddressSpace.Global, alignment=16
        )
        # Flat 1D view of the whole received-token buffer, bounded to A_ELEMS elements. The
        # gather offsets index this row-major buffer directly; the buffer resource clamps the
        # padding sentinel (src_row == num_recv) to an OOB read of 0. (int32 element count:
        # holds up to 2^31 elems; larger A needs a per-lane i64 rebase like Mega's fp8 path.)
        a_base = fx.arith.ArithValue(arith.index_cast(fx.T.i64(), extract_base_index(A)), signed=True)
        A_flat = fx.make_view(fx.inttoptr(pool_ptr_ty, a_base), fx.make_layout(A_ELEMS, 1))

        def _emit():
            pid = xcd_remap_pid(fx.block_idx.x, real_grid, num_xcd)
            num_pid_m = real_tiles
            num_pid_in_group = GROUP_M * n_blocks
            group_id = pid // num_pid_in_group
            pid_in_group = pid % num_pid_in_group
            first_pid_m = group_id * GROUP_M
            remaining_m = num_pid_m - first_pid_m
            group_size_m = arith.select(remaining_m < GROUP_M, remaining_m, fx.Int32(GROUP_M))
            block_m = first_pid_m + (pid_in_group % group_size_m)
            block_n = pid_in_group // group_size_m
            g_idx = buffer_load(group_res, block_m, vec_width=1, dtype=fx.T.i32())
            # PF padding blocks mark expert_ids=-1; skip the full Mega pipeline.
            if g_idx >= fx.Int32(0):
                gbase = g_idx * fx.Int32(K) * c_n
                # Worst-case pool (cap*N > 2^31): rebase C per tile in int64, int32 in-resource
                # offset. Mirrors the fused nt path.
                c_byte_off = _i64(block_m * fx.Int32(BLOCK_M)) * _i64(c_n) * fx.Int64(2)
                c_base = fx.arith.ArithValue(arith.index_cast(fx.T.i64(), extract_base_index(C)), signed=True)
                C_tile = fx.make_view(
                    fx.inttoptr(pool_ptr_ty, c_base + c_byte_off),
                    fx.make_layout(fx.Int32(BLOCK_M) * c_n, 1),
                )
                sorted_row_base = block_m * fx.Int32(BLOCK_M)
                gemm_bf16_nt_gather_tile(
                    A_flat,
                    B,
                    C_tile,
                    fx.Int32(BLOCK_M),
                    c_n,
                    lds,
                    sorted_res,
                    sorted_row_base,
                    block_n,
                    K=K,
                    BLOCK_M=BLOCK_M,
                    BLOCK_N=BLOCK_N,
                    out_fp16=out_fp16,
                    nt_vmcnt=nt_vmcnt,
                    b_group_base=gbase,
                    gather=gather,
                )

        if fx.block_idx.x < real_grid:
            _emit()

    @flyc.jit
    def launch(A, B, C, TILE_TO_GROUP, NUM_TILE_BLOCKS: fx.Int32, SORTED, A_ELEMS: fx.Int32, c_m: fx.Int32, c_n: fx.Int32, stream: fx.Stream):
        grid_x = ceildiv(c_m, BLOCK_M) * ceildiv(c_n, BLOCK_N)
        grouped_gemm_gather_k(
            A,
            B,
            C,
            TILE_TO_GROUP,
            NUM_TILE_BLOCKS,
            SORTED,
            A_ELEMS,
            c_n,
            value_attrs=make_value_attrs(waves_per_eu, agpr_alloc, "512,512"),
        ).launch(grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)

    return launch


def grouped_gemm_gather_bf16(
    A,  # [num_recv, K] bf16   received-token activations (UNPERMUTED gather source)
    weight,  # [E, N, K] bf16   per-expert B (NT)
    output,  # [em_max, N] bf16   block-padded route-ordered C (in place)
    expert_ids,  # [num_m_blocks] i32   expert per BLOCK_M output block
    num_tile_blocks,  # int   real BLOCK_M block count (host scalar; capture-safe)
    sorted_slot_ids,  # [em_max] i32   received-token row per padded slot (sentinel = num_recv)
    *,
    BLOCK_M=256,
    BLOCK_N=256,
    GROUP_M=4,
    num_xcd=1,
    nt_vmcnt=3,
    waves_per_eu=2,
    agpr_alloc=0,
    gather=True,
):
    """Host entry: grouped bf16 NT GEMM. With ``gather=True`` (FC1) ``C[pos] = A[SORTED[pos]]
    @ B[expert]^T``; with ``gather=False`` (FC2 route-read) ``A`` is **block-padded route-ordered**
    ``[em_max, K]`` read at the route slot (``sorted_slot_ids`` unused, may be a dummy).

    ``output`` is written in place over its full padded ``[em_max, N]`` extent (padding rows carry
    dead values, ignored by downstream stages keyed on the same routing metadata). ``c_m`` is the
    padded slot count (``output.shape[0]``); the grid self-bounds to ``num_tile_blocks``.
    """
    assert A.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16
    assert output.dtype == torch.bfloat16
    c_m = int(output.shape[0])
    E, N, K = weight.shape
    assert K == A.shape[1], f"weight K={K} != A K={A.shape[1]}"
    assert output.shape[1] == N, f"output N={output.shape[1]} != weight N={N}"

    A = A.contiguous()
    a_elems = int(A.numel())
    if a_elems >= (1 << 31):
        raise ValueError(
            f"A has {a_elems} elems (>= 2^31); the flat-view gather bound is int32. "
            "Large-A support needs a per-lane i64 SRD rebase (TODO)."
        )
    weight_flat = weight.reshape(E * N, K).contiguous().view(-1)
    expert_ids_i32 = expert_ids.to(torch.int32)
    if gather:
        sorted_arg = sorted_slot_ids.to(torch.int32)
    else:
        # FC2 route-read: the kernel synthesizes the identity index on-device, so SORTED is
        # unread. Reuse a live tensor as the (unused) arg to avoid a capture-illegal allocation.
        sorted_arg = expert_ids_i32
    launch = compile_grouped_gemm_gather_bf16(
        K=K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        GROUP_M=GROUP_M,
        num_xcd=num_xcd,
        nt_vmcnt=int(nt_vmcnt),
        waves_per_eu=int(waves_per_eu),
        agpr_alloc=int(agpr_alloc),
        gather=bool(gather),
    )
    launch(
        A,
        weight_flat,
        output,
        expert_ids_i32,
        int(num_tile_blocks),
        sorted_arg,
        a_elems,
        c_m,
        N,
        stream=torch.cuda.current_stream(),
    )
    return output
