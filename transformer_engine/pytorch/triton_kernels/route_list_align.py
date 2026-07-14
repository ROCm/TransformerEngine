# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Fused route-list alignment (Triton, host-sync-free).

Builds the expert-sorted, block-padded route-list buffers from a boolean ``routing_map``
without any host sync -- no ``nonzero`` (data-dependent size), no ``argsort``, no
``.item()``.

The build is launch-bound (E is tiny), so the work is packed into three small Triton
kernels instead of a chain of ~15 elementwise/scan torch ops:

1. ``_counts_within_kernel`` -- one program per expert computes that expert's token count
   and the exclusive within-expert rank of each routed cell (token-ascending).
2. ``_expert_meta_kernel`` -- from the per-expert counts, derives ``blocks_per_expert``,
   ``block_start``, ``route_start``, ``num_tokens_post_padded`` and the per-block
   ``expert_ids`` in a single launch (every program recomputes the tiny prefix sums from
   ``counts`` locally, so there is no cross-kernel scalar dependency).
3. ``_route_list_place_kernel`` -- scatters each routed cell into its deterministic slot so
   ``sorted_slot_ids`` (block-padded) and ``route_to_token`` (compact) stay consistent.

The index buffers are over-allocated to static, shape-derived upper bounds so their sizes
never depend on device data. The real block-padded extent ``em`` is returned as a device
scalar.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _counts_within_kernel(
    routing_map_ptr,  # [T, E] (bool/int8), True where token t feeds local expert e
    within_ptr,  # [E, T] int32 out: exclusive within-expert rank of each routed cell
    counts_ptr,  # [E] int32 out: routed-token count per expert
    T,
    stride_t,
    stride_e,
    BLOCK_T: tl.constexpr,
):
    e = tl.program_id(axis=0)
    offs = tl.arange(0, BLOCK_T)
    mask = offs < T
    vals = tl.load(routing_map_ptr + offs * stride_t + e * stride_e, mask=mask, other=0).to(
        tl.int32
    )
    # Exclusive prefix sum over tokens => within-expert rank; total => per-expert count.
    incl = tl.cumsum(vals, axis=0)
    excl = incl - vals
    tl.store(within_ptr + e * T + offs, excl, mask=mask)
    tl.store(counts_ptr + e, tl.sum(vals, axis=0))


@triton.jit
def _expert_meta_kernel(
    counts_ptr,  # [E] int32
    blocks_per_expert_ptr,  # [E] int32 out
    block_start_ptr,  # [E] int32 out (block units)
    route_start_ptr,  # [E] int32 out (compact route units)
    expert_ids_ptr,  # [blocks_max] int32 out (expert owning each block, -1 past the end)
    ntpp_ptr,  # [1] int32 out: block-padded token extent
    E,
    blocks_max,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    offs_e = tl.arange(0, BLOCK_E)
    mask_e = offs_e < E
    counts = tl.load(counts_ptr + offs_e, mask=mask_e, other=0)
    blocks_per_expert = (counts + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    cblocks = tl.cumsum(blocks_per_expert, axis=0)  # inclusive prefix over experts
    block_start = cblocks - blocks_per_expert
    route_start = tl.cumsum(counts, axis=0) - counts
    total_blocks = tl.max(tl.where(mask_e, cblocks, 0), axis=0)

    pid = tl.program_id(axis=0)
    if pid == 0:
        tl.store(blocks_per_expert_ptr + offs_e, blocks_per_expert, mask=mask_e)
        tl.store(block_start_ptr + offs_e, block_start, mask=mask_e)
        tl.store(route_start_ptr + offs_e, route_start, mask=mask_e)
        tl.store(ntpp_ptr, total_blocks * BLOCK_SIZE_M)

    # expert_ids[b] = #{e : cblocks[e] <= b} (== searchsorted(cblocks, b, right=True)),
    # then -1 for blocks past the real extent.
    offs_b = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = offs_b < blocks_max
    cblocks_valid = tl.where(mask_e, cblocks, (1 << 30))
    le = (cblocks_valid[None, :] <= offs_b[:, None]).to(tl.int32)
    expert_ids = tl.sum(le, axis=1)
    expert_ids = tl.where(offs_b < total_blocks, expert_ids, -1)
    tl.store(expert_ids_ptr + offs_b, expert_ids, mask=mask_b)


@triton.jit
def _route_list_place_kernel(
    routing_map_ptr,  # [T, E] (bool/int8), True where token t feeds local expert e
    within_ptr,  # [E, T] int32, exclusive within-expert rank of each routed cell
    block_start_ptr,  # [E] int32: first block index of each expert (block units)
    route_start_ptr,  # [E] int32: first compact route index of each expert
    sorted_slot_ids_ptr,  # [em_max] int32, sentinel-init (T)
    route_to_token_ptr,  # [routes_max] int32, sentinel-init (T)
    token_routes_ptr,  # [T, MAXK] int32 out: token->route positions (only if BUILD_INVERSE)
    token_count_ptr,  # [T] int32 out: routes per token (only if BUILD_INVERSE)
    T,
    E,
    stride_t,
    stride_e,
    BLOCK_SIZE_M: tl.constexpr,
    BUILD_INVERSE: tl.constexpr,
    MAXK: tl.constexpr,
):
    # One row of the map per program; place the (few) routed cells deterministically. Because
    # the same per-token expert scan already computes each routed cell's compact route position
    # ``pos = route_start[e] + within[e, t]``, the token->routes inverse map (used by the
    # contention-free gather-combine) is emitted here too when ``BUILD_INVERSE`` -- folding what
    # was a second per-token kernel launch into this one (expert-ascending, no atomics).
    t = tl.program_id(axis=0)
    if t >= T:
        return
    j = tl.zeros((), dtype=tl.int32)
    for e in range(0, E):
        is_routed = tl.load(routing_map_ptr + t * stride_t + e * stride_e)
        if is_routed != 0:
            w = tl.load(within_ptr + e * T + t)
            bs = tl.load(block_start_ptr + e)
            rs = tl.load(route_start_ptr + e)
            pos = rs + w
            tl.store(sorted_slot_ids_ptr + bs * BLOCK_SIZE_M + w, t)
            tl.store(route_to_token_ptr + pos, t)
            if BUILD_INVERSE:
                tl.store(token_routes_ptr + t * MAXK + j, pos)
                j += 1
    if BUILD_INVERSE:
        tl.store(token_count_ptr + t, j)


def route_list_scan(
    routing_map: torch.Tensor,
    *,
    num_experts: int,
):
    """Block-size-independent scan: per-expert token counts + within-expert ranks.

    Returned ``(counts [E], within [E, T])`` can be reused across multiple block sizes
    (e.g. the fwd/dgrad ``BLOCK_SIZE_M`` and the wgrad ``CONTRACT_M``), so the scan is
    only paid once per routing map. Pass them to :func:`route_list_align` via ``scan=``.
    """
    device = routing_map.device
    T = int(routing_map.size(0))
    E = int(num_experts)
    within = torch.empty((E, T), dtype=torch.int32, device=device)
    counts = torch.empty((E,), dtype=torch.int32, device=device)
    _counts_within_kernel[(E,)](
        routing_map,
        within,
        counts,
        T,
        routing_map.stride(0),
        routing_map.stride(1),
        BLOCK_T=triton.next_power_of_2(max(T, 1)),
    )
    return counts, within


def route_list_align(
    routing_map: torch.Tensor,
    *,
    num_experts: int,
    block_size: int,
    scan=None,
    max_routes_per_token: int | None = None,
    build_inverse_map: bool = False,
):
    """Sync-free fused build of the route-list align buffers.

    Parameters
    ----------
    scan:
        Optional ``(counts, within)`` from :func:`route_list_scan` for this ``routing_map``.
        When supplied the block-independent scan kernel is skipped (shared across block
        sizes); otherwise it is computed here.
    max_routes_per_token:
        Host-known upper bound on the number of experts any token routes to (e.g. the router
        top-k). When provided, the static over-allocation bound is tightened from the dense
        ``T * num_experts`` to ``T * min(max_routes_per_token, num_experts)`` -- still
        sync-free, but shrinking the padded buffers by ``num_experts / max_routes_per_token``.
    build_inverse_map:
        When True, the place kernel also emits the token->routes inverse map (used by the
        contention-free gather-combine in FC2 fwd / FC1 dgrad) in the same launch, so no
        separate inverse-map kernel is needed. Block-independent, so build it on the fwd align
        only.

    Returns
    -------
    ``(sorted_slot_ids, expert_ids, num_tokens_post_padded, block_start, blocks_per_expert,
       route_start, route_to_token, token_routes, token_route_count)`` -- index tensors
    ``int32``; ``num_tokens_post_padded`` (``[1]``) is a device scalar. ``token_routes``
    (``[T, min(topk, E)]``) and ``token_route_count`` (``[T]``) are ``None`` unless
    ``build_inverse_map``.
    """
    if routing_map.dtype != torch.bool:
        routing_map = routing_map.bool()
    routing_map = routing_map.contiguous()
    device = routing_map.device
    T = int(routing_map.size(0))
    E = int(num_experts)

    # Static (sync-free) upper bounds from shapes only. Each token routes to at most
    # ``min(max_routes_per_token, E)`` experts, so that tightens the dense ``T * E`` bound.
    max_per_token = E if max_routes_per_token is None else min(int(max_routes_per_token), E)
    routes_max = T * max_per_token
    blocks_max = (routes_max + block_size - 1) // block_size + E
    em_max = blocks_max * block_size

    # Per-expert count + exclusive within-expert rank (one program per expert). Reused
    # across block sizes when the caller passes a precomputed scan.
    if scan is None:
        counts, within = route_list_scan(routing_map, num_experts=E)
    else:
        counts, within = scan

    # Per-expert placement metadata + per-block expert ids (single launch).
    blocks_per_expert = torch.empty((E,), dtype=torch.int32, device=device)
    block_start = torch.empty((E,), dtype=torch.int32, device=device)
    route_start = torch.empty((E,), dtype=torch.int32, device=device)
    expert_ids = torch.empty((blocks_max,), dtype=torch.int32, device=device)
    num_tokens_post_padded = torch.empty((1,), dtype=torch.int32, device=device)
    block_b = 256
    _expert_meta_kernel[(triton.cdiv(blocks_max, block_b),)](
        counts,
        blocks_per_expert,
        block_start,
        route_start,
        expert_ids,
        num_tokens_post_padded,
        E,
        blocks_max,
        BLOCK_SIZE_M=block_size,
        BLOCK_E=triton.next_power_of_2(max(E, 1)),
        BLOCK_B=block_b,
    )

    # Optional token->routes inverse map, emitted by the same place kernel. Width is the
    # tightened per-token bound; zero-init so any unused tail column is a safe (in-range)
    # index (the gather masks columns >= count anyway).
    if build_inverse_map:
        maxk = max(max_per_token, 1)
        token_routes = torch.zeros((T, maxk), dtype=torch.int32, device=device)
        token_route_count = torch.empty((T,), dtype=torch.int32, device=device)
    else:
        maxk = 1
        token_routes = torch.empty((1,), dtype=torch.int32, device=device)  # unused stub
        token_route_count = token_routes

    # Scatter each routed cell into its deterministic (block-padded / compact) slot. Both
    # sentinel buffers are carved from a single fill (one launch); the place kernel then
    # overwrites the routed slots in each contiguous view.
    sentinel = torch.full((em_max + routes_max,), T, dtype=torch.int32, device=device)
    sorted_slot_ids = sentinel[:em_max]
    route_to_token = sentinel[em_max:]
    _route_list_place_kernel[(T,)](
        routing_map,
        within,
        block_start,
        route_start,
        sorted_slot_ids,
        route_to_token,
        token_routes,
        token_route_count,
        T,
        E,
        routing_map.stride(0),
        routing_map.stride(1),
        BLOCK_SIZE_M=block_size,
        BUILD_INVERSE=build_inverse_map,
        MAXK=maxk,
    )

    return (
        sorted_slot_ids,
        expert_ids,
        num_tokens_post_padded,
        block_start,
        blocks_per_expert,
        route_start,
        route_to_token,
        token_routes if build_inverse_map else None,
        token_route_count if build_inverse_map else None,
    )
