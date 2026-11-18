# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Permute-free MoE data-gradient (dgrad) grouped-GEMM: MegaMOE's bf16 NN GEMM.

    dXrow[s, k] = sum_n grad[src(s), n] * W[e][n, k]

* route-read (FC1, ``gather=False``): ``grad`` is route-ordered; row ``s`` is read directly.
* gather (FC2, ``gather=True``): ``grad`` is token-ordered; slot ``s`` reads ``grad[SORTED[s]]``
  (sentinel ``num_recv`` reads zero).

Contract:
  * ``grad_y`` [rows, N]     bf16   incoming grad (rows = R_block route-read / num_recv gather)
  * ``weight`` [E, N, K]     bf16   per-expert weights (shared with forward)
  * ``dx``     [R_block, K]   bf16   block-padded route-ordered input grad per slot (in place)
  * ``expert_ids``      [num_m_blocks] i32   expert id per BLOCK_M slot block
  * ``num_tile_blocks`` int  real (non-padding) BLOCK_M block count (host)
  * ``sorted_slot_ids`` [R_block] i32  gather index (gather=True); unused for route-read
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
    gemm_bf16_nn_gather_tile,
    xcd_remap_pid,
)

__all__ = ["compile_grouped_gemm_dgrad_bf16", "grouped_gemm_dgrad_bf16"]


def _dgrad_tile(
    gather, GY_flat, GY_tile, WEIGHT, DX_tile, lds, sorted_res, sorted_row_base, block_n, gbase,
    *, N, Kout, BLOCK_M, BLOCK_N, out_fp16, nt_vmcnt,
):
    """Emit one NN dgrad tile; ``gather`` is resolved at trace time, not in device control flow."""
    A = GY_flat if gather else GY_tile
    gemm_bf16_nn_gather_tile(
        A, WEIGHT, DX_tile, fx.Int32(Kout), lds, sorted_res, sorted_row_base, block_n,
        Kc=N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, out_fp16=out_fp16, nt_vmcnt=nt_vmcnt,
        b_group_base=gbase, gather=gather,
    )


@functools.lru_cache(maxsize=256)
def compile_grouped_gemm_dgrad_bf16(
    N,  # contraction dim (forward intermediate feature)
    Kout,  # output cols (forward hidden input feature)
    BLOCK_M=256,
    BLOCK_N=256,
    GROUP_M=4,
    num_xcd=1,
    nt_vmcnt=3,
    waves_per_eu=2,
    agpr_alloc=0,
    out_fp16=False,
    gather=False,
):
    """Compile (cached) the grouped BF16 NN dgrad launcher for one ``(N, Kout, tile)`` combo."""
    SharedStorage = _make_shared_storage(BLOCK_M, BLOCK_N)

    @flyc.kernel(known_block_size=[512, 1, 1])
    def grouped_gemm_dgrad_k(
        GRAD_Y: fx.Tensor,
        WEIGHT: fx.Tensor,
        DX: fx.Tensor,
        TILE_TO_GROUP: fx.Tensor,
        NUM_TILE_BLOCKS: fx.Int32,
        SORTED: fx.Tensor,
        A_ELEMS: fx.Int32,
    ):
        n_blocks = ceildiv(fx.Int32(Kout), BLOCK_N)
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        group_res = RouteI32Loader(TILE_TO_GROUP)
        sorted_res = RouteI32Loader(SORTED)
        # Host scalar, not a device tensor, so the launch stays graph-capture safe.
        real_tiles = NUM_TILE_BLOCKS
        real_grid = real_tiles * n_blocks

        pool_ptr_ty = PointerType.get(
            elem_ty=fx.BFloat16.ir_type, address_space=AddressSpace.Global, alignment=16
        )
        gy_base = fx.Int64(fx.ptrtoint(fx.get_iter(GRAD_Y)))
        dx_base = fx.Int64(fx.ptrtoint(fx.get_iter(DX)))
        # Flat gather view bounded to A_ELEMS so the sentinel row reads zero (unused by route-read).
        GY_flat = fx.make_view(fx.inttoptr(pool_ptr_ty, gy_base), fx.make_layout(A_ELEMS, 1))

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
                gbase = g_idx * fx.Int32(N) * fx.Int32(Kout)
                sorted_row_base = block_m * fx.Int32(BLOCK_M)

                dx_byte_off = fx.Int64(block_m * fx.Int32(BLOCK_M)) * fx.Int64(Kout) * fx.Int64(2)
                DX_tile = fx.make_view(
                    fx.inttoptr(pool_ptr_ty, dx_base + dx_byte_off),
                    fx.make_layout(fx.Int32(BLOCK_M) * fx.Int32(Kout), 1),
                )
                # Route-read grad tile, rebased per tile in i64 (unused by gather).
                gy_byte_off = fx.Int64(block_m * fx.Int32(BLOCK_M)) * fx.Int64(N) * fx.Int64(2)
                GY_tile = fx.make_view(
                    fx.inttoptr(pool_ptr_ty, gy_base + gy_byte_off),
                    fx.make_layout(fx.Int32(BLOCK_M) * fx.Int32(N), 1),
                )

                _dgrad_tile(
                    gather, GY_flat, GY_tile, WEIGHT, DX_tile, lds, sorted_res, sorted_row_base,
                    block_n, gbase, N=N, Kout=Kout, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                    out_fp16=out_fp16, nt_vmcnt=nt_vmcnt,
                )

        if fx.block_idx.x < real_grid:
            _emit()

    @flyc.jit
    def launch(GRAD_Y, WEIGHT, DX, TILE_TO_GROUP, NUM_TILE_BLOCKS: fx.Int32, SORTED, A_ELEMS: fx.Int32, c_m: fx.Int32, stream: fx.Stream):
        grid_x = ceildiv(c_m, BLOCK_M) * ceildiv(fx.Int32(Kout), BLOCK_N)
        grouped_gemm_dgrad_k(
            GRAD_Y,
            WEIGHT,
            DX,
            TILE_TO_GROUP,
            NUM_TILE_BLOCKS,
            SORTED,
            A_ELEMS,
            value_attrs=make_value_attrs(waves_per_eu, agpr_alloc, "512,512"),
        ).launch(grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)

    return launch


def grouped_gemm_dgrad_bf16(
    grad_y,  # [rows, N] bf16   incoming grad (rows = R_block route-read / num_recv gather)
    weight,  # [E, N, K] bf16    per-expert weights (shared with forward)
    dx,  # [R_block, K] bf16     block-padded route-ordered input grad per slot (in place)
    expert_ids,  # [num_m_blocks] i32
    num_tile_blocks,  # int   real BLOCK_M block count (host scalar; capture-safe)
    sorted_slot_ids=None,  # [R_block] i32   gather index (required when gather=True)
    *,
    gather=False,
    BLOCK_M=256,
    BLOCK_N=256,
    GROUP_M=4,
    num_xcd=1,
    nt_vmcnt=3,
    waves_per_eu=2,
    agpr_alloc=0,
):
    """Host entry: grouped bf16 NN dgrad ``dXrow[s] = grad[src(s)] @ W[expert]``.

    ``dx`` is fully overwritten; padding rows hold garbage.
    """
    assert grad_y.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16
    assert dx.dtype == torch.bfloat16
    E, N, K = weight.shape
    c_m = int(dx.shape[0])
    assert grad_y.shape[1] == N, f"grad_y N={grad_y.shape[1]} != weight N={N}"
    assert dx.shape[1] == K, f"dx K={dx.shape[1]} != weight K={K}"
    # Weight offsets are int32; ``dx`` and route-read ``grad_y`` are rebased per tile in i64.
    require_launch_size(
        "permute-free dgrad",
        ("weight", weight),
    )
    # The in-kernel N assert is stripped under ``python -O``, so check here.
    if N % 64 != 0 or N < 128:
        raise ValueError(
            f"permute-free dgrad requires N % 64 == 0 and N >= 128 (contraction BLOCK_K=64, "
            f">= 2 tiles), got N={N}."
        )
    if not gather:
        assert grad_y.shape[0] == c_m, (
            f"route-read dgrad expects grad_y rows == dx rows ({grad_y.shape[0]} != {c_m}); "
            "did you mean gather=True (token-space grad)?"
        )

    grad_y = grad_y.contiguous()
    if gather:
        a_elems = int(grad_y.numel())
        if a_elems >= (1 << 31):
            raise ValueError(
                f"grad_y has {a_elems} elems (>= 2^31); the flat-view bound is int32. "
                "Large-A support needs a per-lane i64 SRD rebase (TODO)."
            )
    else:
        a_elems = int(BLOCK_M) * int(N)
    expert_ids_i32 = expert_ids.to(torch.int32)
    if gather:
        assert sorted_slot_ids is not None, "gather=True dgrad requires sorted_slot_ids"
        sorted_arg = sorted_slot_ids.to(torch.int32)
    else:
        # SORTED is unread; pass a live tensor rather than allocating (capture-illegal).
        sorted_arg = expert_ids_i32

    weight_flat = weight.reshape(E * N, K)
    if not weight_flat.is_contiguous():
        weight_flat = weight_flat.contiguous()
    weight_flat = weight_flat.view(-1)
    launch = compile_grouped_gemm_dgrad_bf16(
        N=N,
        Kout=K,
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
        grad_y,
        weight_flat,
        dx,
        expert_ids_i32,
        int(num_tile_blocks),
        sorted_arg,
        a_elems,
        c_m,
        stream=torch.cuda.current_stream(),
    )
    return dx
