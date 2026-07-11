# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Route-list MoE gather-GEMM (bf16): decoupled token gather and compact route store.

``sorted_slot_ids[pos]`` holds the **received-token row** for the ``A`` gather; the output
row is the compact route index ``out_row(pos) = route_start[e] + (pos - block_start[e]*BM)``.
Padding slots use sentinel ``num_recv_tokens`` in ``sorted_slot_ids`` and are masked out.

Two modes (``INDEX_A_BY_ROUTE_POS``):
- fwd: gather ``A`` by received-token row (``sorted_slot_ids``), store to compact ``out_row``.
- dgrad: read the compact grad row (``out_row``) as the ``A`` operand, store to ``out_row``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import triton
import triton.language as tl

from .pid_preprocessing import pid_grid, remap_xcd, get_num_xcds

_USE_PERSISTENT = True


@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
    }
)
@triton.jit
def _route_list_moe_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    sorted_slot_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    block_start_ptr,
    route_start_ptr,
    N,
    K,
    num_recv_tokens,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    INDEX_A_BY_ROUTE_POS: tl.constexpr,
    SCATTER_TO_TOKEN: tl.constexpr,
    compute_type: tl.constexpr,
    NUM_XCDS: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    num_pid_m = tl.cdiv(num_tokens_post_padded, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    grid_mn = num_pid_n * num_pid_m
    if pid >= grid_mn:
        return
    pid = remap_xcd(pid, grid_mn, NUM_XCDS)
    pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M)

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts < 0:
        return

    pos = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    slot = tl.load(sorted_slot_ids_ptr + pos)
    token_mask = slot < num_recv_tokens

    block_start = tl.load(block_start_ptr + off_experts).to(tl.int64)
    route_start = tl.load(route_start_ptr + off_experts).to(tl.int64)
    out_row = route_start + (pos - block_start * BLOCK_SIZE_M)

    if INDEX_A_BY_ROUTE_POS:
        a_row = out_row
    else:
        a_row = slot.to(tl.int64)

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + a_row[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + off_experts * stride_be + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if EVEN_K:
            a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
            b = tl.load(b_ptrs)
        else:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                other=0.0,
            )
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    if SCATTER_TO_TOKEN:
        # Fused dgrad scatter: accumulate each per-route dX directly onto its received-token
        # row. Padded/sentinel positions are masked out, so padding is never scattered and no
        # compact intermediate + separate index_add is needed. C is fp32 for accurate
        # multi-route atomic accumulation.
        scatter_row = slot.to(tl.int64)
        c_ptrs = c_ptr + stride_cm * scatter_row[:, None] + stride_cn * offs_cn[None, :]
        # relaxed ordering is sufficient: this is pure accumulation and dA is only read after
        # the kernel completes (implicit device sync). scope must be "gpu" -- the same token
        # row is updated from multiple CTAs (a token routed to several experts lands in
        # different blocks), so a "cta"-scoped atomic would race across workgroups.
        tl.atomic_add(c_ptrs, accumulator, mask=c_mask, sem="relaxed", scope="gpu")
    else:
        c_ptrs = c_ptr + stride_cm * out_row[:, None] + stride_cn * offs_cn[None, :]
        tl.store(c_ptrs, accumulator.to(compute_type), mask=c_mask, cache_modifier=".wt")


@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
    }
)
@triton.jit
def _route_list_moe_persistent_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    sorted_slot_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    block_start_ptr,
    route_start_ptr,
    N,
    K,
    num_recv_tokens,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    INDEX_A_BY_ROUTE_POS: tl.constexpr,
    SCATTER_TO_TOKEN: tl.constexpr,
    compute_type: tl.constexpr,
    NUM_XCDS: tl.constexpr,
):
    start_pid = tl.program_id(axis=0)
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    num_pid_m = tl.cdiv(num_tokens_post_padded, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    tile_id = start_pid
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n
    num_valid_tiles = tl.cdiv((num_tiles - tile_id), NUM_SMS)

    for _ in range(0, num_valid_tiles):
        tile_id_remapped = remap_xcd(tile_id, num_tiles, NUM_XCDS)
        pid_m, pid_n = pid_grid(tile_id_remapped, num_pid_m, num_pid_n, GROUP_SIZE_M)

        off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
        if off_experts >= 0:
            pos = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
            slot = tl.load(sorted_slot_ids_ptr + pos)
            token_mask = slot < num_recv_tokens

            block_start = tl.load(block_start_ptr + off_experts).to(tl.int64)
            route_start = tl.load(route_start_ptr + off_experts).to(tl.int64)
            out_row = route_start + (pos - block_start * BLOCK_SIZE_M)

            if INDEX_A_BY_ROUTE_POS:
                a_row = out_row
            else:
                a_row = slot.to(tl.int64)

            offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
            a_ptrs = a_ptr + a_row[:, None] * stride_am + offs_k[None, :] * stride_ak
            b_ptrs = (
                b_ptr
                + off_experts * stride_be
                + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
            )
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
                if EVEN_K:
                    a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
                    b = tl.load(b_ptrs)
                else:
                    a = tl.load(
                        a_ptrs,
                        mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                        other=0.0,
                    )
                    b = tl.load(
                        b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0
                    )
                accumulator += tl.dot(a, b)
                a_ptrs += BLOCK_SIZE_K * stride_ak
                b_ptrs += BLOCK_SIZE_K * stride_bk

            offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
            if SCATTER_TO_TOKEN:
                # Fused dgrad scatter: accumulate each per-route dX onto its received-token
                # row; padded/sentinel positions are masked out. C is fp32 for accurate
                # multi-route atomic accumulation.
                scatter_row = slot.to(tl.int64)
                c_ptrs = c_ptr + stride_cm * scatter_row[:, None] + stride_cn * offs_cn[None, :]
                # relaxed ordering is sufficient (pure accumulation, dA read after kernel
                # completes); scope must be "gpu" since the same token row is updated from
                # multiple CTAs (a token routed to several experts lands in different blocks).
                tl.atomic_add(c_ptrs, accumulator, mask=c_mask, sem="relaxed", scope="gpu")
            else:
                c_ptrs = c_ptr + stride_cm * out_row[:, None] + stride_cn * offs_cn[None, :]
                tl.store(c_ptrs, accumulator.to(compute_type), mask=c_mask)

        tile_id += NUM_SMS


def fused_route_list_moe(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    sorted_slot_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    block_start: torch.Tensor,
    route_start: torch.Tensor,
    *,
    num_recv_tokens: int,
    compute_type: tl.dtype,
    config: Optional[Dict[str, Any]] = None,
    index_a_by_route_pos: bool = False,
    scatter_to_token: bool = False,
) -> None:
    """Launch route-list gather-GEMM in place.

    By default writes the compact ``C[num_routes, N]`` (fwd/dgrad). When
    ``scatter_to_token=True`` (dgrad), ``C`` is instead a compact ``[num_recv_tokens, N]``
    (fp32) buffer and each per-route result is atomic-accumulated onto its received-token row,
    fusing the scatter and skipping the block-padded / sentinel positions.
    """
    if config is None:
        raise ValueError("route-list MoE kernel requires an explicit tile config.")

    assert sorted_slot_ids.stride(0) == 1

    em = sorted_slot_ids.shape[0]
    if _USE_PERSISTENT:
        num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count * 2
        grid = lambda meta: (  # noqa: E731
            min(
                num_sms,
                triton.cdiv(em, meta["BLOCK_SIZE_M"]) * triton.cdiv(B.shape[1], meta["BLOCK_SIZE_N"]),
            ),
        )
        _route_list_moe_persistent_kernel[grid](
            A,
            B,
            C,
            sorted_slot_ids,
            expert_ids,
            num_tokens_post_padded,
            block_start,
            route_start,
            B.shape[1],
            A.shape[1],
            num_recv_tokens,
            A.stride(0),
            A.stride(1),
            B.stride(0),
            B.stride(2),
            B.stride(1),
            C.stride(0),
            C.stride(1),
            NUM_SMS=num_sms,
            INDEX_A_BY_ROUTE_POS=index_a_by_route_pos,
            SCATTER_TO_TOKEN=scatter_to_token,
            compute_type=compute_type,
            NUM_XCDS=get_num_xcds(),
            **config,
        )
    else:
        grid = lambda meta: (  # noqa: E731
            triton.cdiv(em, meta["BLOCK_SIZE_M"]) * triton.cdiv(B.shape[1], meta["BLOCK_SIZE_N"]),
        )
        _route_list_moe_kernel[grid](
            A,
            B,
            C,
            sorted_slot_ids,
            expert_ids,
            num_tokens_post_padded,
            block_start,
            route_start,
            B.shape[1],
            A.shape[1],
            num_recv_tokens,
            A.stride(0),
            A.stride(1),
            B.stride(0),
            B.stride(2),
            B.stride(1),
            C.stride(0),
            C.stride(1),
            INDEX_A_BY_ROUTE_POS=index_a_by_route_pos,
            SCATTER_TO_TOKEN=scatter_to_token,
            compute_type=compute_type,
            NUM_XCDS=get_num_xcds(),
            **config,
        )
