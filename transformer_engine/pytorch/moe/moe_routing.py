# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""MoE routing metadata for permute-free grouped GEMM.

``R_block`` is the block-padded route buffer row count (a static upper bound); buffers are
``[R_block, F]`` and ``num_tokens_post_padded`` holds the real padded extent on device.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class WgradAlign:
    """Lazily-built block-``CONTRACT_M`` align buffers for the route-list wgrad kernel.

    Mutable so a metadata and its ``dataclasses.replace`` copy (FC1/FC2) share one build, even
    though it is built in the backward, after the copy was made.
    """

    sorted_slot_ids: Optional[torch.Tensor] = None
    block_start: Optional[torch.Tensor] = None
    blocks_per_expert: Optional[torch.Tensor] = None
    block_size: Optional[int] = None


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
        Optional max local experts per token. Shrinks the static route-buffer bound from
        ``num_recv_tokens * num_experts`` to ``num_recv_tokens * min(topk, num_experts)``.
    num_routes:
        Optional exact route count (``routing_map.sum()``). Sizes ``R_block`` exactly, but makes
        buffer shapes data-dependent, which breaks CUDA graph capture. Routes beyond an
        undercounted value are dropped; ``NVTE_PERMUTE_FREE_VALIDATE_ROUTES=1`` checks it (syncs).

    The remaining fields are filled lazily by ``prepare_moe_align`` (sync-free unless
    ``num_routes`` is set), sized to static upper bounds with real extents on device:

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
        ``[1]`` device scalar = actual ``R_block`` (real padded route count). Bounds the kernel.
    block_start:
        ``[num_experts]`` per-expert first block index (block units). Expert ``e``'s block-padded
        slots start at ``block_start[e] * block_size_m``.
    token_routes / token_route_count:
        ``[num_recv_tokens, min(topk, num_experts)]`` int32 slot positions per token (first
        ``token_route_count[t]`` valid), used by the atomic-free combines in FC2 fwd / FC1 dgrad.
    block_size_m:
        ``BLOCK_SIZE_M`` used to build the fwd/dgrad align buffers.
    wgrad_align:
        Shared wgrad align holder (see :class:`WgradAlign`).
    route_counts / route_within:
        Cached per-expert counts and within-expert ranks, shared by all align builds.
    """

    routing_map: torch.Tensor
    num_experts: Optional[int] = None
    topk: Optional[int] = None
    num_routes: Optional[int] = None
    sorted_slot_ids: Optional[torch.Tensor] = None
    expert_ids: Optional[torch.Tensor] = None
    slot_expert_ids: Optional[torch.Tensor] = None
    num_tokens_post_padded: Optional[torch.Tensor] = None
    block_start: Optional[torch.Tensor] = None
    token_routes: Optional[torch.Tensor] = None
    token_route_count: Optional[torch.Tensor] = None
    block_size_m: Optional[int] = None
    wgrad_align: Optional[WgradAlign] = None
    route_counts: Optional[torch.Tensor] = None
    route_within: Optional[torch.Tensor] = None
    # ``routing_map`` id/version at the last align build; a swap or in-place copy_ invalidates it.
    align_routing_id: Optional[int] = None
    align_routing_version: Optional[int] = None

    def __post_init__(self):
        if self.num_experts is None:
            self.num_experts = int(self.routing_map.size(1))
        # A ``replace`` copy passes the existing holder through, so FC1 and FC2 share it.
        if self.wgrad_align is None:
            self.wgrad_align = WgradAlign()

    @property
    def num_recv_tokens(self) -> int:
        """Number of received tokens (rows of ``routing_map`` / the activation buffer)."""
        return int(self.routing_map.size(0))


@dataclass
class PermuteFreeMetadata(MoERoutingMetadata):
    """Routing metadata for the permute-free grouped GEMM, tagged with a direction.

    - ``route_space=False`` (FC1): token-ordered input ``[num_recv_tokens, in]`` is gathered
      into block-padded route-ordered output ``[R_block, out]``.
    - ``route_space=True`` (FC2): block-padded route-ordered input ``[R_block, in]`` is combined
      back to token-ordered output ``[num_recv_tokens, out]``.

    The align buffers are direction-independent, so FC2 can reuse FC1's metadata via
    ``dataclasses.replace(meta, route_space=True)``.

    activation:
        Optional gated activation (``"silu"`` / ``"gelu"``) applied to the input on the FC2
        direction via :func:`permute_free_gated_act_fwd`. Route probs need a gradient, so they
        are a separate module input, not metadata.
    """

    route_space: bool = False
    activation: Optional[str] = None
