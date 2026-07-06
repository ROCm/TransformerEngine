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
    """Routing tensors for gather-in-GEMM MoE paths (vLLM/AITER layout).

    Caller contract (permute-free GroupedLinear):
    - Skip ``moe_permute`` before FC1; pass unpermuted activations ``[M, hidden]``.
    - FC1 output layout is ``[M, topk, out_features]`` (not expert-contiguous).
    - Reuse the same metadata for FC2; reduce with ``moe_unpermute`` or ``moe_sum``.

    Parameters
    ----------
    topk_ids:
        Expert indices, shape ``[num_tokens, topk]``, ``int32``.
    topk_weights:
        Router weights per selected expert, shape ``[num_tokens, topk]``, ``float32``.
    num_experts:
        Total expert count (local experts on this rank).
    sorted_token_ids:
        Optional cache from ``prepare_moe_align`` for FC1/FC2 reuse.
    expert_ids:
        Optional cache from ``prepare_moe_align``.
    num_tokens_post_padded:
        Optional cache from ``prepare_moe_align``.
    block_size_m:
        Optional cache of the ``BLOCK_SIZE_M`` used to build the align buffers.
        The dgrad GEMM must reuse the same value so the per-block ``expert_ids``
        and grid stay consistent.
    mul_routed_weight:
        If ``True``, multiply GEMM output by ``topk_weights`` inside the kernel.
    """

    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    num_experts: int
    sorted_token_ids: Optional[torch.Tensor] = None
    expert_ids: Optional[torch.Tensor] = None
    num_tokens_post_padded: Optional[torch.Tensor] = None
    block_size_m: Optional[int] = None
    mul_routed_weight: bool = False


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
