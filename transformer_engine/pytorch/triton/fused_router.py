# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""PyTorch wrapper functions for fused MoE router Triton kernels."""

import torch
import triton

from transformer_engine.common.triton.fused_router import (
    _fused_topk_score_fwd_kernel,
    _fused_topk_score_bwd_kernel,
    _fused_score_aux_loss_fwd_kernel,
    _fused_score_aux_loss_bwd_kernel,
)

_SCORE_FN_MAP = {"sigmoid": 0, "softmax": 1}


def fused_topk_with_score_function_fwd(
    logits: torch.Tensor,
    topk: int,
    use_pre_softmax: bool,
    scaling_factor: float,
    score_function: str,
    expert_bias: torch.Tensor | None,
    num_groups: int | None = None,
    group_topk: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused score-function + top-k forward via Triton.

    Returns (probs, routing_map, intermediate_output).
    """
    num_tokens, num_experts = logits.shape
    block_e = triton.next_power_of_2(num_experts)
    score_fn = _SCORE_FN_MAP[score_function]
    has_bias = expert_bias is not None
    sf = scaling_factor if scaling_factor is not None and scaling_factor > 0 else 1.0

    use_group_topk = (
        group_topk is not None and group_topk > 0
        and num_groups is not None and num_groups > 0
    )
    ng = num_groups if use_group_topk else 1
    gt = group_topk if use_group_topk else 1
    gs = num_experts // ng if use_group_topk else num_experts
    block_g = triton.next_power_of_2(ng)

    probs = torch.empty_like(logits)
    routing_map_i8 = torch.empty(
        num_tokens, num_experts, dtype=torch.int8, device=logits.device,
    )
    intermediate = torch.empty_like(logits)

    grid = (num_tokens,)
    _fused_topk_score_fwd_kernel[grid](
        logits,
        probs,
        routing_map_i8,
        intermediate,
        expert_bias if has_bias else logits,  # dummy ptr when unused
        num_tokens,
        sf,
        NUM_EXPERTS=num_experts,
        TOPK=topk,
        SCORE_FN=score_fn,
        USE_PRE_SOFTMAX=use_pre_softmax,
        HAS_BIAS=has_bias,
        USE_GROUP_TOPK=use_group_topk,
        NUM_GROUPS=ng,
        GROUP_TOPK=gt,
        GROUP_SIZE=gs,
        BLOCK_E=block_e,
        BLOCK_G=block_g,
    )

    routing_map = routing_map_i8.to(torch.bool)
    return probs, routing_map, intermediate


def fused_topk_with_score_function_bwd(
    num_tokens: int,
    num_experts: int,
    routing_map: torch.Tensor,
    intermediate_output: torch.Tensor,
    grad_probs: torch.Tensor,
    topk: int,
    use_pre_softmax: bool,
    scaling_factor: float,
    score_function: str,
) -> torch.Tensor:
    """Fused score-function + top-k backward via Triton."""
    block_e = triton.next_power_of_2(num_experts)
    score_fn = _SCORE_FN_MAP[score_function]
    sf = scaling_factor if scaling_factor is not None and scaling_factor > 0 else 1.0

    routing_map_i8 = routing_map.to(torch.int8)
    grad_logits = torch.empty(
        num_tokens, num_experts,
        dtype=intermediate_output.dtype, device=grad_probs.device,
    )

    grid = (num_tokens,)
    _fused_topk_score_bwd_kernel[grid](
        routing_map_i8,
        intermediate_output,
        grad_probs,
        grad_logits,
        num_tokens,
        sf,
        NUM_EXPERTS=num_experts,
        TOPK=topk,
        SCORE_FN=score_fn,
        USE_PRE_SOFTMAX=use_pre_softmax,
        BLOCK_E=block_e,
    )

    return grad_logits


def fused_score_for_moe_aux_loss_fwd(
    logits: torch.Tensor,
    topk: int,
    score_function: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused score computation for aux loss via Triton."""
    num_tokens, num_experts = logits.shape
    block_e = triton.next_power_of_2(num_experts)
    score_fn = _SCORE_FN_MAP[score_function]

    scores = torch.empty_like(logits)
    routing_map_i8 = torch.empty(
        num_tokens, num_experts, dtype=torch.int8, device=logits.device,
    )
    intermediate = torch.empty_like(logits)

    grid = (num_tokens,)
    _fused_score_aux_loss_fwd_kernel[grid](
        logits,
        scores,
        routing_map_i8,
        intermediate,
        num_tokens,
        NUM_EXPERTS=num_experts,
        TOPK=topk,
        SCORE_FN=score_fn,
        BLOCK_E=block_e,
    )

    routing_map = routing_map_i8.to(torch.bool)
    return scores, routing_map, intermediate


def fused_score_for_moe_aux_loss_bwd(
    num_tokens: int,
    num_experts: int,
    intermediate_output: torch.Tensor,
    grad_scores: torch.Tensor,
    topk: int,
    score_function: str,
) -> torch.Tensor:
    """Fused score backward for aux loss via Triton."""
    block_e = triton.next_power_of_2(num_experts)
    score_fn = _SCORE_FN_MAP[score_function]

    grad_logits = torch.empty(
        num_tokens, num_experts,
        dtype=intermediate_output.dtype, device=grad_scores.device,
    )

    grid = (num_tokens,)
    _fused_score_aux_loss_bwd_kernel[grid](
        intermediate_output,
        grad_scores,
        grad_logits,
        num_tokens,
        NUM_EXPERTS=num_experts,
        SCORE_FN=score_fn,
        BLOCK_E=block_e,
    )

    return grad_logits
