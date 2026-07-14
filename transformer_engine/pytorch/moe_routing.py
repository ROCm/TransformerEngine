# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""MoE routing metadata for permute-free grouped GEMM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class MoERoutingMetadata:
    """Routing tensors for the route-list gather-in-GEMM MoE path.

    Caller contract (permute-free FC1, post-dispatch):
    - Pass received activations ``[num_recv_tokens, hidden]`` (no permute, no topk).
    - Provide a boolean ``routing_map`` ``[num_recv_tokens, num_local_experts]`` marking
      which local expert(s) each received token feeds.
    - TE builds one expert-sorted ``sorted_slot_ids`` (received-token row per route) and runs
      the gather-GEMM. FC1 output is compact ``[num_routes, out_features]`` in expert order;
      the router-weight combine and the token reduction happen upstream in Megatron.

    Parameters
    ----------
    routing_map:
        Boolean mask ``[num_recv_tokens, num_local_experts]``, True where a received token
        feeds a local expert. ``num_routes = routing_map.sum()``. This is the *only* field
        the caller needs to provide.
    num_experts:
        Local expert count on this rank. Optional -- defaults to ``routing_map.size(1)``.
    topk:
        Upstream router top-k -- the host-known maximum number of local experts any received
        token can feed. Optional. When provided, ``prepare_moe_align`` tightens the (still
        sync-free) static over-allocation of the block-padded route buffers from the dense
        ``num_recv_tokens * num_experts`` bound down to
        ``num_recv_tokens * min(topk, num_experts)``, shrinking ``em_max`` (and the zero-init
        it costs) by up to ``num_experts / topk``. Leave ``None`` to keep the dense bound.

    The remaining fields are lazily populated by ``prepare_moe_align`` and cached for
    FC1 fwd/dgrad reuse (all in the expert-sorted, block-padded route-list layout):

    The align buffers are built sync-free and over-allocated to static, shape-derived
    upper bounds; the real extents are carried as device scalars so no host sync is
    needed to construct them.

    sorted_slot_ids:
        ``[em_max]`` received-token row to gather for each block-padded position (sentinel
        ``num_recv_tokens`` for padding and for the over-allocated tail).
    expert_ids:
        ``[blocks_max]`` local expert owning each ``BLOCK_SIZE_M`` block (``-1`` past the
        real block count; those blocks are never visited).
    num_tokens_post_padded:
        ``[1]`` device scalar = real ``em`` (block-padded route count). Bounds the kernel.
    block_start:
        ``[num_experts]`` per-expert first block index (block units).
    route_start:
        ``[num_experts]`` per-expert first *compact* route index (``cumsum(counts) - counts``).
        Maps a block-padded position to its compact output row.
    route_to_token:
        ``[routes_max]`` received-token row for each compact route (first ``num_routes``
        entries valid); used by the dgrad scatter-add back to ``[num_recv_tokens, K]``.
    token_routes / token_route_count:
        Inverse of ``route_to_token`` (token -> its compact route positions), built sync-free
        for the contention-free gather-combine that replaces the atomic scatter in the token
        combine (FC2 fwd) and the FC1 input-grad reduction (FC1 dgrad). ``token_routes`` is
        ``[num_recv_tokens, min(topk, num_experts)]`` (int32); for token ``t`` the first
        ``token_route_count[t]`` entries are its route positions (expert-ascending), the rest
        are unused padding.
    block_size_m:
        ``BLOCK_SIZE_M`` used to build the fwd/dgrad align buffers.
    wgrad_*:
        Separate block-``CONTRACT_M`` align buffers for the route-list wgrad kernel.
    route_counts / route_within:
        Cached block-size-independent scan (per-expert counts and within-expert ranks)
        shared by the fwd/dgrad and wgrad align builds.
    """

    routing_map: torch.Tensor
    num_experts: Optional[int] = None
    topk: Optional[int] = None
    sorted_slot_ids: Optional[torch.Tensor] = None
    expert_ids: Optional[torch.Tensor] = None
    num_tokens_post_padded: Optional[torch.Tensor] = None
    block_start: Optional[torch.Tensor] = None
    route_start: Optional[torch.Tensor] = None
    route_to_token: Optional[torch.Tensor] = None
    token_routes: Optional[torch.Tensor] = None
    token_route_count: Optional[torch.Tensor] = None
    block_size_m: Optional[int] = None
    wgrad_sorted_slot_ids: Optional[torch.Tensor] = None
    wgrad_block_start: Optional[torch.Tensor] = None
    wgrad_blocks_per_expert: Optional[torch.Tensor] = None
    wgrad_block_size: Optional[int] = None
    # Block-size-independent scan (per-expert counts + within-expert ranks), shared by the
    # fwd/dgrad and wgrad align builds so it is computed once per routing map.
    route_counts: Optional[torch.Tensor] = None
    route_within: Optional[torch.Tensor] = None

    def __post_init__(self):
        # num_experts is redundant with the routing map width (one column per local
        # expert), so the caller can pass just ``routing_map``.
        if self.num_experts is None:
            self.num_experts = int(self.routing_map.size(1))

    @property
    def num_recv_tokens(self) -> int:
        """Number of received tokens (rows of ``routing_map`` / the activation buffer)."""
        return int(self.routing_map.size(0))


@dataclass
class PermuteFreeMetadata(MoERoutingMetadata):
    """Routing metadata for the permute-free grouped GEMM, tagged with a direction.

    Extends :class:`MoERoutingMetadata` with a single ``route_space`` flag that selects
    which of the two permute-free GEMM directions a :class:`GroupedLinear` should run:

    - ``route_space=False`` (FC1): the input lives in **received-token order**
      ``[num_recv_tokens, in]``. The forward *gathers* per expert (``index_a_by_route_pos=
      False``) into the compact/padded ``[em_max, out]`` route buffer; the dgrad
      *scatters* the input gradient back to token rows.
    - ``route_space=True`` (FC2): the input is already in **route order**
      ``[em_max, in]`` (FC1's output). The forward reads by route position
      (``index_a_by_route_pos=True``) and *scatters to token* (``scatter_to_token=True``),
      emitting ``[num_recv_tokens, out]`` directly (the fused combine); the dgrad gathers
      the token-space grad back into the compact route buffer.

    The align buffers are identical for both directions, so a single built metadata can be
    reused for FC1 and FC2 (e.g. via ``dataclasses.replace(meta, route_space=True)``),
    avoiding a duplicate align build.
    """

    route_space: bool = False


def routing_map_to_topk(
    probs: torch.Tensor,
    routing_map: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert TE router outputs to vLLM-style ``topk_ids`` / ``topk_weights``.

    Parameters
    ----------
    probs:
        Router probabilities, shape ``[num_tokens, num_experts]``.
    routing_map:
        Boolean mask, shape ``[num_tokens, num_experts]``, True where routed.

    Returns
    -------
    topk_ids:
        ``int32`` tensor, shape ``[num_tokens, topk]``.
    topk_weights:
        ``float32`` tensor, shape ``[num_tokens, topk]``.
    """
    if probs.shape != routing_map.shape:
        raise ValueError(
            f"probs shape {probs.shape} must match routing_map shape {routing_map.shape}."
        )
    if routing_map.dtype != torch.bool:
        routing_map = routing_map.bool()

    topk = int(routing_map.sum(dim=1).max().item())
    if topk == 0:
        raise ValueError("routing_map has no routed experts.")

    masked = probs.masked_fill(~routing_map, float("-inf"))
    topk_weights, topk_ids = torch.topk(masked, k=topk, dim=-1)
    # Rows with fewer than topk routes may include -inf from padding; zero those weights.
    valid = routing_map.gather(1, topk_ids)
    topk_weights = topk_weights.masked_fill(~valid, 0.0)
    return topk_ids.to(torch.int32), topk_weights.to(torch.float32)


def index_map_to_topk_weights(
    probs: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """Gather router weights for an index routing map ``[num_tokens, topk]``."""
    if probs.dim() != 2 or topk_ids.dim() != 2:
        raise ValueError("probs and topk_ids must be 2D tensors.")
    if probs.size(0) != topk_ids.size(0):
        raise ValueError("probs and topk_ids must have the same num_tokens dimension.")
    gathered = probs.gather(1, topk_ids.to(torch.long))
    return gathered.to(torch.float32)
