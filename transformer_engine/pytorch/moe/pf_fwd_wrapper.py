# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL permute-free MoE fwd/dgrad gather-GEMM op wrapper."""

from __future__ import annotations

import torch

__all__ = [
    "flydsl_moe_fwd",
]

# Tile geometry swept offline in primus-turbo's bench_mega_moe; ``block_m`` comes from the align.
_PF_BLOCK_N = 256
_PF_GROUP_M = 4
_PF_DGRAD_FC1_GROUP_M = 8
_PF_NUM_XCD = 1


def _run_gather_gemm(
    A, B, C, sorted_slot_ids, expert_ids, *, block_m,
    transpose_b, index_a_by_route_pos,
):
    """Dispatch to the permute-free fwd (or, with ``transpose_b``, dgrad) gather-GEMM kernel."""
    block_m = int(block_m)
    R_block = int(sorted_slot_ids.shape[0])
    if R_block % block_m != 0:
        raise ValueError(
            f"permute-free gather-GEMM expects R_block ({R_block}) divisible by block_m ({block_m}); "
            "check routing align block_size_m."
        )
    num_tile_blocks = R_block // block_m
    expert_ids_i32 = expert_ids.to(torch.int32)
    if expert_ids_i32.numel() != num_tile_blocks:
        raise ValueError(
            f"permute-free gather-GEMM expects expert_ids length {num_tile_blocks} "
            f"(R_block={R_block}, BLOCK_M={block_m}), got {expert_ids_i32.numel()}; "
            f"routing align block_size_m must match block_m={block_m}"
        )

    if transpose_b:
        from ..flydsl_kernels.permute_free_grouped_gemm.pf_dgrad import grouped_gemm_dgrad_bf16

        dgrad_group_m = _PF_DGRAD_FC1_GROUP_M if index_a_by_route_pos else _PF_GROUP_M
        grouped_gemm_dgrad_bf16(
            A, B, C, expert_ids_i32, num_tile_blocks, sorted_slot_ids,
            gather=not index_a_by_route_pos,
            BLOCK_M=block_m, BLOCK_N=_PF_BLOCK_N, GROUP_M=dgrad_group_m, num_xcd=_PF_NUM_XCD,
        )
        return

    from ..flydsl_kernels.permute_free_grouped_gemm.pf_fwd import grouped_gemm_gather_bf16

    grouped_gemm_gather_bf16(
        A, B, C, expert_ids_i32, num_tile_blocks, sorted_slot_ids,
        gather=not index_a_by_route_pos,
        BLOCK_M=block_m, BLOCK_N=_PF_BLOCK_N, GROUP_M=_PF_GROUP_M, num_xcd=_PF_NUM_XCD,
    )


def flydsl_moe_fwd(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    sorted_slot_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    *,
    block_m: int,
    index_a_by_route_pos: bool = False,
    dgrad: bool = False,
) -> None:
    """Route-list gather-GEMM, writing block-padded route-ordered ``C[R_block, WIDTH_N]`` in place.

    ``A`` is token-ordered and gathered via ``sorted_slot_ids`` (FC1), or block-padded
    route-ordered when ``index_a_by_route_pos=True`` (FC2). ``B`` is ``[E, N_OUT, K]``.
    ``dgrad=True`` runs the data-gradient GEMM with the same ``B``.
    """
    assert A.dtype == B.dtype == C.dtype == torch.bfloat16
    assert A.stride(1) == 1, "A must be contiguous along the contraction (K)"
    assert B.stride(2) == 1, "B must be [E, N, K] with contiguous inner K"

    _run_gather_gemm(
        A, B, C, sorted_slot_ids, expert_ids,
        block_m=block_m,
        transpose_b=dgrad, index_a_by_route_pos=index_a_by_route_pos,
    )
