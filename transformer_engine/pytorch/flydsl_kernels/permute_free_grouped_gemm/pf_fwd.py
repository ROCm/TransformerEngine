# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Permute-free MoE forward gather grouped-GEMM: MegaMOE's fast bf16 GEMM + route-list gather.

MegaMOE's grouped bf16 GEMM pipeline, unchanged except that A rows are fetched through
``sorted_slot_ids`` instead of from a pre-permuted buffer.

Contract:
  * ``A``            [num_recv, K] bf16   received-token activations, UNPERMUTED (gather source)
  * ``B``            [E, N, K]    bf16    per-expert weights (contiguous inner K)
  * ``C``            [R_block, N]  bf16    block-padded route-ordered output (in place)
  * ``sorted_slot_ids`` [R_block]  i32     received-token row per padded slot (sentinel = num_recv)
  * ``expert_ids``   [num_m_blocks] i32   expert id per ``BLOCK_M`` output block (-1 = padding)
  * ``num_tile_blocks`` int               real (non-padding) ``BLOCK_M`` block count (host)

Gated activation and route-prob apply live in standalone Triton helpers, not here.
"""

from __future__ import annotations

import functools

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr.typing import AddressSpace, PointerType

from ..gemm.fp16_gemm_utils import ceildiv
from ..gemm.gemm_common_utils import make_value_attrs, require_launch_size
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

    The grid is over-launched to the padded pool; blocks past ``num_tile_blocks`` or with
    ``expert_ids[block_m] < 0`` exit early.
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
        # Host scalar, not a device tensor, so the launch stays graph-capture safe.
        real_tiles = NUM_TILE_BLOCKS
        # Swizzle over real tiles only; swizzling the padded pool scatters them (~2x slower).
        real_grid = real_tiles * n_blocks

        pool_ptr_ty = PointerType.get(
            elem_ty=fx.BFloat16.ir_type, address_space=AddressSpace.Global, alignment=16
        )
        # Flat A view bounded to A_ELEMS (int32) so the sentinel row (num_recv) reads OOB zero.
        a_base = fx.Int64(fx.ptrtoint(fx.get_iter(A)))
        A_flat = fx.make_view(fx.inttoptr(pool_ptr_ty, a_base), fx.make_layout(A_ELEMS, 1))

        def _emit():
            pid = xcd_remap_pid(fx.block_idx.x, real_grid, num_xcd)
            num_pid_m = real_tiles
            num_pid_in_group = GROUP_M * n_blocks
            group_id = pid // num_pid_in_group
            pid_in_group = pid % num_pid_in_group
            first_pid_m = group_id * GROUP_M
            remaining_m = num_pid_m - first_pid_m
            group_size_m = (remaining_m < GROUP_M).select(remaining_m, fx.Int32(GROUP_M))
            block_m = first_pid_m + (pid_in_group % group_size_m)
            block_n = pid_in_group // group_size_m
            g_idx = group_res.load(block_m)
            # Padding blocks have expert_ids == -1.
            if g_idx >= fx.Int32(0):
                gbase = g_idx * fx.Int32(K) * c_n
                # Rebase C per tile in i64 so pools beyond 2^31 elements stay addressable.
                c_byte_off = fx.Int64(block_m * fx.Int32(BLOCK_M)) * fx.Int64(c_n) * fx.Int64(2)
                c_base = fx.Int64(fx.ptrtoint(fx.get_iter(C)))
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
    weight,  # [E, N, K] bf16   per-expert weights (shared with dgrad/wgrad)
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
    """Host entry: grouped bf16 fprop ``C[pos] = A[row(pos)] @ B[expert]^T``.

    ``gather=True`` (FC1) reads ``A[SORTED[pos]]``; ``gather=False`` (FC2) reads block-padded
    route-ordered ``A`` at ``pos``. ``output`` is fully overwritten; padding rows hold garbage.
    """
    assert A.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16
    assert output.dtype == torch.bfloat16
    c_m = int(output.shape[0])
    E, N, K = weight.shape
    assert K == A.shape[1], f"weight K={K} != A K={A.shape[1]}"
    assert output.shape[1] == N, f"output N={output.shape[1]} != weight N={N}"

    # Weight offsets and A's flat view are int32; ``output`` is rebased per tile in i64.
    require_launch_size(
        "permute-free fwd",
        ("A", A),
        ("weight", weight),
    )

    # The in-kernel K assert is stripped under ``python -O``, so check here.
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
        # SORTED is unread; pass a live tensor rather than allocating (capture-illegal).
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
