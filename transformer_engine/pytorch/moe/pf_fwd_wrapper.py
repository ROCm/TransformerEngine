# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL permute-free MoE route-list forward gather-GEMM op wrapper.

Mirrors the route-list forward gather-GEMM contract: the fwd/dgrad launch consumes
``sorted_slot_ids`` and ``expert_ids`` from the routing align (plus ``block_m`` for tile
geometry). ``block_start`` is not passed here; it is built alongside the align for wgrad only.
Writes the block-padded ``[em_max, WIDTH_N]`` slot output in place.

Forward gather-GEMM is FlyDSL-only (MegaMOE-ported plain GEMM).
"""

from __future__ import annotations

import torch

__all__ = [
    "flydsl_moe_fwd",
]

# MegaMOE hand-tuned bf16 grouped-GEMM tile geometry (offline-swept in primus-turbo's
# bench_mega_moe: BLOCK_N=256, GROUP_M=4 fwd / GROUP_M=8 FC1 NN dgrad, num_xcd=1).
# ``block_m`` comes from the route-list align (128 or 256); N-tile and grouping are fixed here.
_PF_BLOCK_N = 256
_PF_GROUP_M = 4
_PF_DGRAD_FC1_GROUP_M = 8  # bench_mega_moe grouped_gemm_combine L1-dgrad (NN) sweep
_PF_NUM_XCD = 1


def _run_gather_gemm(
    A, B, C, sorted_slot_ids, expert_ids, *, num_recv_tokens, block_m,
    transpose_b, index_a_by_route_pos,
):
    """Dispatch fwd/dgrad to the permute-free FlyDSL gather-GEMM kernels.

    Covers FC1 gather (``index_a_by_route_pos=False``), FC2 route-read
    (``index_a_by_route_pos=True``), and dgrad (``transpose_b``). Uses MegaMOE's fixed tile
    geometry (32x32x16 MFMA, ``BLOCK_N=256``, ``GROUP_M=4``, ``num_xcd=1``).
    """
    block_m = int(block_m)
    em_max = int(sorted_slot_ids.shape[0])
    if em_max % block_m != 0:
        raise ValueError(
            f"permute-free gather-GEMM expects em_max ({em_max}) divisible by block_m ({block_m}); "
            "check routing align block_size_m."
        )
    num_tile_blocks = em_max // block_m
    expert_ids_i32 = expert_ids.to(torch.int32)
    if expert_ids_i32.numel() != num_tile_blocks:
        raise ValueError(
            f"permute-free gather-GEMM expects expert_ids length {num_tile_blocks} "
            f"(em_max={em_max}, BLOCK_M={block_m}), got {expert_ids_i32.numel()}; "
            f"routing align block_size_m must match block_m={block_m}"
        )

    if transpose_b:
        # dgrad: NN GEMM contracting the incoming grad against the forward weight [E, N, K]
        # route-read from block-padded route-ordered grad; FC2 dgrad gathers token-ordered
        # grad [num_recv, N] into block-padded route-ordered dX [em_max, K].
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
    num_recv_tokens: int,
    block_m: int,
    index_a_by_route_pos: bool = False,
    dgrad: bool = False,
) -> None:
    """Route-list gather-GEMM, writing block-padded route-ordered ``C[em_max, WIDTH_N]`` in place.

    ``A`` is token-ordered ``[num_recv, K]`` gathered via ``sorted_slot_ids`` when
    ``index_a_by_route_pos=False`` (FC1), or block-padded route-ordered read by route slot when
    ``index_a_by_route_pos=True`` (FC2). ``B`` is ``[num_experts, N_OUT, K]`` (contiguous inner
    ``K``). Set ``dgrad=True`` for the data-gradient path (same ``B`` layout). Gated activation
    and route-prob apply are **not** fused here; use the standalone helpers in
    :mod:`pf_helper_kernels`.
    """
    assert A.dtype == B.dtype == C.dtype == torch.bfloat16
    assert A.stride(1) == 1, "A must be contiguous along the contraction (K)"
    assert B.stride(2) == 1, "B must be [E, N, K] with contiguous inner K"

    _run_gather_gemm(
        A, B, C, sorted_slot_ids, expert_ids,
        num_recv_tokens=num_recv_tokens, block_m=block_m,
        transpose_b=dgrad, index_a_by_route_pos=index_a_by_route_pos,
    )
