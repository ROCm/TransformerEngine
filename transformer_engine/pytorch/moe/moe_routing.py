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

    Parameters
    ----------
    routing_map:
        Boolean mask ``[num_recv_tokens, num_local_experts]``, True where a received token
        feeds a local expert. ``num_routes = routing_map.sum()``.
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
        ``[T * min(topk, E)]`` received-token row to gather for each block-padded position (sentinel
        ``num_recv_tokens`` for padding and for the over-allocated tail).
    expert_ids:
        ``[blocks_max]`` local expert owning each ``BLOCK_SIZE_M`` block (``-1`` past the
        real block count; those blocks are never visited).
    slot_expert_ids:
        ``[T * min(topk, E)]`` per-slot local expert id, derived from ``expert_ids`` and
        ``block_size_m`` (``expert_ids[slot // block_size_m]``). Populated by
        ``prepare_moe_align`` for the standalone gated-activation kernels.
    num_tokens_post_padded:
        ``[1]`` device scalar = real ``em`` (block-padded route count). Bounds the kernel.
    block_start:
        ``[num_experts]`` per-expert first block index (block units). Expert ``e``'s block-padded
        slots start at ``block_start[e] * block_size_m``.
    token_routes / token_route_count:
        Token -> its block-padded slot positions, built sync-free for the contention-free
        gather-combine that replaces the atomic scatter in the token combine (FC2 fwd) and the
        FC1 input-grad reduction (FC1 dgrad). ``token_routes`` is
        ``[num_recv_tokens, min(topk, num_experts)]`` (int32); for token ``t`` the first
        ``token_route_count[t]`` entries are its padded slots (expert-ascending), the rest
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
    slot_expert_ids: Optional[torch.Tensor] = None
    num_tokens_post_padded: Optional[torch.Tensor] = None
    block_start: Optional[torch.Tensor] = None
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

    Extends :class:`MoERoutingMetadata`

    - ``route_space=False`` (FC1): the input lives in **received-token order**
      ``[num_recv_tokens, in]``. The forward *gathers* per expert (``index_a_by_route_pos=
      False``) into the compact/padded ``[T * min(topk, E), out]`` route buffer; the dgrad combines
      the input gradient back to token rows (contention-free gather-combine).
    - ``route_space=True`` (FC2): the input is already in **route order**
      ``[T * min(topk, E), in]`` (FC1's output). The forward reads by route position
      (``index_a_by_route_pos=True``) and combines each token's routes back to
      ``[num_recv_tokens, out]`` (contention-free gather-combine); the dgrad gathers
      the token-space grad back into the compact route buffer.

    The align buffers are identical for both directions, so a single built metadata can be
    reused for FC1 and FC2 (e.g. via ``dataclasses.replace(meta, route_space=True)``),
    avoiding a duplicate align build.

    Activation hint (optional):

    activation:
        Gated activation for the FC2 standalone pass -- ``"silu"`` or ``"gelu"``.
        ``None`` leaves activation to the caller. FC1 emits raw ``2F``; this hint is
        consumed on the FC2 direction (``route_space=True``) to run
        :func:`permute_free_gated_act_recompute` before the plain FC2 GEMM.

    (The per-route gating probabilities are *not* carried here: they need a gradient, so they
    are passed as a separate autograd tensor argument to the module rather than as metadata.)
    """

    route_space: bool = False
    activation: Optional[str] = None