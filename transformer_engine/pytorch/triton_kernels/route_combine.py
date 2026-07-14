# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Token-parallel gather-combine for the permute-free MoE path (Triton).

The permute-free token combine sums, for each received token, the per-route contributions of
the experts it was routed to. The compact route buffer ``[em_max, N]`` is expert-sorted, so a
token's routes live in different expert blocks. Doing the sum as a *scatter* (each route
atomic-adds onto its token row) serializes on hot token rows -- a token's top-k routes land in
different CTAs and collide on the same fp32 cell. That atomic contention is the dominant cost
of the fused FC2-forward / FC1-dgrad scatter.

This kernel flips the reduction into a contention-free **gather**: one program owns an output
token row and pulls its (<= topk) route rows from the compact buffer, summing them locally in
fp32 with zero atomics. It relies on the token->routes inverse map
(``build_route_inverse_map``), which is built sync-free during the align.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gather_combine_kernel(
    src_ptr,  # compact [em_max, N] (route order; valid rows [0, num_routes))
    token_routes_ptr,  # [T, MAXK] int32: compact route positions per token
    token_count_ptr,  # [T] int32: number of routes for each token
    out_ptr,  # [T, N] out
    N,
    stride_sm,
    stride_sn,
    stride_om,
    stride_on,
    MAXK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    compute_type: tl.constexpr,
):
    t = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    cnt = tl.load(token_count_ptr + t)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    # Sum the token's route rows (expert-ascending -> deterministic). Columns >= cnt are
    # unused padding and skipped, so no padded/garbage row is ever gathered.
    for j in range(0, MAXK):
        if j < cnt:
            r = tl.load(token_routes_ptr + t * MAXK + j).to(tl.int64)
            v = tl.load(src_ptr + r * stride_sm + offs_n * stride_sn, mask=n_mask, other=0.0)
            acc += v.to(tl.float32)
    tl.store(out_ptr + t * stride_om + offs_n * stride_on, acc.to(compute_type), mask=n_mask)


def route_gather_combine(
    src: torch.Tensor,
    token_routes: torch.Tensor,
    token_route_count: torch.Tensor,
    num_recv_tokens: int,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
    block_n: int = 512,
) -> torch.Tensor:
    """Combine per-route rows into per-token rows via a contention-free gather.

    Parameters
    ----------
    src:
        Compact per-route buffer ``[em_max, N]`` (route order; only ``[0, num_routes)`` valid).
    token_routes / token_route_count:
        Inverse map from :func:`build_route_inverse_map` (``[T, MAXK]`` route positions and the
        per-token route count).
    num_recv_tokens:
        Number of output token rows ``T``.

    Returns
    -------
    torch.Tensor
        ``[num_recv_tokens, N]`` in ``out_dtype`` -- each row is the fp32 sum of its token's
        route rows, cast once. No atomics, no host sync.
    """
    if src.dim() != 2:
        raise ValueError(f"src must be [em_max, N], got {tuple(src.shape)}.")
    if not src.is_contiguous():
        src = src.contiguous()
    n = src.shape[1]
    maxk = int(token_routes.shape[1])
    out = torch.empty((num_recv_tokens, n), dtype=out_dtype, device=src.device)
    compute_type = tl.bfloat16 if out_dtype == torch.bfloat16 else tl.float32
    grid = (num_recv_tokens, triton.cdiv(n, block_n))
    _gather_combine_kernel[grid](
        src,
        token_routes,
        token_route_count,
        out,
        n,
        src.stride(0),
        src.stride(1),
        out.stride(0),
        out.stride(1),
        MAXK=maxk,
        BLOCK_N=block_n,
        compute_type=compute_type,
    )
    return out
