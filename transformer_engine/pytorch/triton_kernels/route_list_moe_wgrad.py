# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Route-list fused MoE wgrad: ``grad[route]^T @ x[token(route)]`` grouped by expert.

``sorted_slot_ids[route_pos]`` holds the received-token row for the ``x`` gather; the grad
operand is read from the compact ``[num_routes, N]`` buffer at
``grad_row = route_start[e] + (route_pos - block_start[e]*CONTRACT_M)``. Padding slots use
sentinel ``num_recv_tokens`` and are masked out.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import triton
import triton.language as tl

from .pid_preprocessing import get_num_xcds  # noqa: F401


@triton.jit
def _route_list_wgrad_accumulate_tile(
    x_ptr,
    grad_ptr,
    sorted_slot_ids_ptr,
    base_slot,
    route_start_e,
    num_slots,
    pid_n,
    pid_k,
    N,
    K,
    num_recv_tokens,
    stride_xm,
    stride_xk,
    stride_gm,
    stride_gn,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    CONTRACT_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_m = tl.arange(0, CONTRACT_M)

    accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K), dtype=tl.float32)
    for m0 in range(0, num_slots, CONTRACT_M):
        row = m0 + offs_m
        in_range = row < num_slots
        route_pos = base_slot + row
        token = tl.load(
            sorted_slot_ids_ptr + route_pos,
            mask=in_range,
            other=num_recv_tokens,
        ).to(tl.int64)
        token_mask = in_range & (token < num_recv_tokens)
        grad_row = (route_start_e + row).to(tl.int64)

        x_ptrs = x_ptr + token[:, None] * stride_xm + offs_k[None, :] * stride_xk
        if EVEN_K:
            x = tl.load(x_ptrs, mask=token_mask[:, None], other=0.0)
        else:
            x = tl.load(
                x_ptrs,
                mask=token_mask[:, None] & (offs_k[None, :] < K),
                other=0.0,
            )

        g_ptrs = grad_ptr + grad_row[None, :] * stride_gm + offs_n[:, None] * stride_gn
        if EVEN_N:
            g = tl.load(g_ptrs, mask=token_mask[None, :], other=0.0)
        else:
            g = tl.load(
                g_ptrs,
                mask=token_mask[None, :] & (offs_n[:, None] < N),
                other=0.0,
            )
        accumulator += tl.dot(g, x)

    return accumulator


@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
        "EVEN_N": lambda args: args["N"] % args["BLOCK_SIZE_N"] == 0,
    }
)
@triton.jit
def _route_list_moe_wgrad_kernel(
    x_ptr,
    grad_ptr,
    dw_ptr,
    sorted_slot_ids_ptr,
    block_start_ptr,
    blocks_per_expert_ptr,
    route_start_ptr,
    N,
    K,
    num_recv_tokens,
    stride_xm,
    stride_xk,
    stride_gm,
    stride_gn,
    stride_dwe,
    stride_dwn,
    stride_dwk,
    num_experts,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    CONTRACT_M: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_per_expert = num_pid_n * num_pid_k
    grid_nk = num_experts * num_pid_per_expert
    if pid >= grid_nk:
        return

    expert = pid // num_pid_per_expert
    pid_in_e = pid % num_pid_per_expert
    pid_n = pid_in_e // num_pid_k
    pid_k = pid_in_e % num_pid_k

    block_start = tl.load(block_start_ptr + expert).to(tl.int64)
    nblocks = tl.load(blocks_per_expert_ptr + expert).to(tl.int64)
    route_start_e = tl.load(route_start_ptr + expert).to(tl.int64)
    base_slot = block_start * CONTRACT_M
    num_slots = nblocks * CONTRACT_M

    acc = _route_list_wgrad_accumulate_tile(
        x_ptr,
        grad_ptr,
        sorted_slot_ids_ptr,
        base_slot,
        route_start_e,
        num_slots,
        pid_n,
        pid_k,
        N,
        K,
        num_recv_tokens,
        stride_xm,
        stride_xk,
        stride_gm,
        stride_gn,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        CONTRACT_M,
        EVEN_K,
        EVEN_N,
    )

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    dw_ptrs = (
        dw_ptr
        + expert * stride_dwe
        + offs_n[:, None] * stride_dwn
        + offs_k[None, :] * stride_dwk
    )
    mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
    tl.store(dw_ptrs, acc.to(dw_ptr.dtype.element_ty), mask=mask)


def get_default_route_list_wgrad_config(contract_m: int) -> Dict[str, Any]:
    """Tile config for the route-list Triton wgrad kernel."""
    return {
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "CONTRACT_M": contract_m,
        "GROUP_SIZE_M": 8,
        "num_warps": 8,
        "num_stages": 2,
    }


def fused_route_list_moe_wgrad(
    x: torch.Tensor,
    grad: torch.Tensor,
    dw: torch.Tensor,
    sorted_slot_ids: torch.Tensor,
    block_start: torch.Tensor,
    blocks_per_expert: torch.Tensor,
    route_start: torch.Tensor,
    *,
    num_recv_tokens: int,
    config: Optional[Dict[str, Any]] = None,
    contract_m: int = 32,
) -> None:
    """Fused route-list wgrad into ``dw[e, N, K]`` in place.

    ``x`` is ``[num_recv_tokens, K]`` (gathered by received-token row); ``grad`` is the
    compact ``[num_routes, N]`` per-route gradient (read via ``route_start``/``block_start``).
    """
    num_experts = dw.shape[0]
    n, k = dw.shape[1], dw.shape[2]
    wgrad_config = config or get_default_route_list_wgrad_config(contract_m)
    if "CONTRACT_M" not in wgrad_config:
        wgrad_config = {**wgrad_config, "CONTRACT_M": contract_m}

    grid = lambda meta: (  # noqa: E731
        num_experts
        * triton.cdiv(n, meta["BLOCK_SIZE_N"])
        * triton.cdiv(k, meta["BLOCK_SIZE_K"]),
    )
    _route_list_moe_wgrad_kernel[grid](
        x,
        grad,
        dw,
        sorted_slot_ids,
        block_start,
        blocks_per_expert,
        route_start,
        n,
        k,
        num_recv_tokens,
        x.stride(0),
        x.stride(1),
        grad.stride(0),
        grad.stride(1),
        dw.stride(0),
        dw.stride(1),
        dw.stride(2),
        num_experts,
        **wgrad_config,
    )
