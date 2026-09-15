# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

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
  * ``C``            [R_block, N]  bf16    block-padded route-ordered output (expert-major, in place)
  * ``sorted_slot_ids`` [R_block]  i32     received-token row per padded slot (sentinel = num_recv)
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
from flydsl.expr import arith
from flydsl.expr.typing import AddressSpace, PointerType

from ..gemm.fp16_gemm_utils import ceildiv
from ..gemm.gemm_common_utils import _i64, make_value_attrs, require_launch_size
from ..gemm.pf_gemm_utils import (
    RouteI32Loader,
    _make_shared_storage,
    gemm_bf16_nt_gather_tile,
    xcd_remap_pid,
)

__all__ = ["compile_grouped_gemm_gather_bf16", "grouped_gemm_gather_bf16"]


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
        group_res = RouteI32Loader(TILE_TO_GROUP)
        sorted_res = RouteI32Loader(SORTED)
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
        a_base = fx.arith.ArithValue(arith.index_cast(fx.T.i64, fx.ptrtoint(fx.get_iter(A))), signed=True)
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
            g_idx = group_res.load(block_m)
            # PF padding blocks mark expert_ids=-1; skip the full Mega pipeline.
            if g_idx >= fx.Int32(0):
                gbase = g_idx * fx.Int32(K) * c_n
                # Worst-case pool (cap*N > 2^31): rebase C per tile in int64, int32 in-resource
                # offset. Mirrors the fused nt path.
                c_byte_off = _i64(block_m * fx.Int32(BLOCK_M)) * _i64(c_n) * fx.Int64(2)
                c_base = fx.arith.ArithValue(arith.index_cast(fx.T.i64, fx.ptrtoint(fx.get_iter(C))), signed=True)
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
    output,  # [R_block, N] bf16   block-padded route-ordered C (in place)
    expert_ids,  # [num_m_blocks] i32   expert per BLOCK_M output block
    num_tile_blocks,  # int   real BLOCK_M block count (host scalar; capture-safe)
    sorted_slot_ids,  # [R_block] i32   received-token row per padded slot (sentinel = num_recv)
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
    ``[R_block, K]`` read at the route slot (``sorted_slot_ids`` unused, may be a dummy).

    ``output`` is written in place over its full padded ``[R_block, N]`` extent (padding rows carry
    dead values, ignored by downstream stages keyed on the same routing metadata). ``c_m`` is the
    padded slot count (``output.shape[0]``); the grid self-bounds to ``num_tile_blocks``.
    """
    assert A.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16
    assert output.dtype == torch.bfloat16
    c_m = int(output.shape[0])
    E, N, K = weight.shape
    assert K == A.shape[1], f"weight K={K} != A K={A.shape[1]}"
    assert output.shape[1] == N, f"output N={output.shape[1]} != weight N={N}"

    # The kernel computes the per-expert weight offset (``expert_id * N * K``) in 32-bit, and
    # FlyDSL passes each operand's flattened size to the kernel in 32-bit, so ``weight`` and ``A``
    # must fit 2^31 -- keep the hard guard (``A`` is also bounded by ``A_ELEMS`` below).
    # ``output`` is safe: the kernel addresses it in 64-bit, one tile at a time
    # (``c_base + c_byte_off``), so it stays correct above 2 GiB.
    require_launch_size(
        "permute-free fwd",
        ("A", A),
        ("weight", weight),
    )

    # The tile core contracts over K in BLOCK_K=64 steps and needs >= 2 K-tiles. This is
    # enforced by a bare ``assert`` inside the traced kernel (stripped under ``python -O``,
    # yielding garbage), so validate host-side here where it always runs.
    if K % 64 != 0 or K < 128:
        raise ValueError(
            f"permute-free fwd requires K % 64 == 0 and K >= 128 (BLOCK_K=64, >= 2 K-tiles), got K={K}."
        )

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
