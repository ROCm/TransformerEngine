# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""MOE router operations -- PyTorch-native implementations."""

import torch
import torch.nn.functional as F


def fused_topk_with_score_function_fwd(logits, topk, use_pre_softmax, num_groups,
                                        group_topk, scaling_factor, score_function,
                                        expert_bias):
    """Fused topk with score function forward."""
    if use_pre_softmax:
        probs = F.softmax(logits, dim=-1)
        scores = probs
    else:
        scores = logits

    if expert_bias is not None:
        scores = scores + expert_bias

    # Select top-k experts per token
    topk_values, topk_indices = torch.topk(scores, k=topk, dim=-1)

    if not use_pre_softmax:
        # Compute softmax over selected experts
        topk_values = F.softmax(topk_values, dim=-1)

    # Normalize routing weights
    if scaling_factor > 0:
        topk_values = topk_values * scaling_factor

    return topk_values, topk_indices


def fused_topk_with_score_function_bwd(num_tokens, num_experts, routing_map,
                                        intermediate_output, grad_probs, topk,
                                        use_pre_softmax, scaling_factor, score_function):
    """Fused topk with score function backward."""
    grad_logits = torch.zeros(num_tokens, num_experts,
                              device=grad_probs.device, dtype=grad_probs.dtype)
    # Scatter gradients back to selected expert positions
    grad_logits.scatter_(1, routing_map, grad_probs)
    return grad_logits


def fused_score_for_moe_aux_loss_fwd(logits, topk, score_function):
    """Compute scores for MOE auxiliary loss."""
    scores = F.softmax(logits, dim=-1)
    return scores


def fused_score_for_moe_aux_loss_bwd(num_tokens, num_experts, intermediate_output,
                                      grad_scores, topk, score_function):
    """Backward of scores for MOE auxiliary loss."""
    # Softmax backward
    dot = (intermediate_output * grad_scores).sum(dim=-1, keepdim=True)
    grad_logits = intermediate_output * (grad_scores - dot)
    return grad_logits


def fused_moe_aux_loss_fwd(probs, tokens_per_expert, total_num_tokens, num_experts,
                            num_rows, num_cols, topk, coeff):
    """MOE auxiliary (load balancing) loss forward."""
    # Standard load-balancing loss: coeff * num_experts * sum(f_i * P_i)
    # f_i = fraction of tokens routed to expert i
    # P_i = average routing probability for expert i
    f = tokens_per_expert.float() / total_num_tokens
    p = probs.mean(dim=0)
    loss = coeff * num_experts * (f * p).sum()
    return loss


def fused_moe_aux_loss_bwd(const_buf, tokens_per_expert, num_rows, num_cols, grad_aux_loss):
    """MOE auxiliary loss backward."""
    # d(loss)/d(probs) = coeff * num_experts * f_i / num_tokens
    grad_probs = const_buf * grad_aux_loss
    return grad_probs
