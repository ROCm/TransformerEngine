# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""MOE router operations -- Triton-fused with PyTorch-native fallback."""

import torch
import torch.nn.functional as F


_EPSILON = 1e-9

# ---------------------------------------------------------------------------
# Lazy Triton import
# ---------------------------------------------------------------------------
_triton_router = None
_triton_attempted = False


def _try_load_triton_router():
    global _triton_router, _triton_attempted
    if _triton_attempted:
        return _triton_router
    _triton_attempted = True
    try:
        from transformer_engine.pytorch.triton import fused_router
        _triton_router = fused_router
    except (ImportError, RuntimeError):
        pass
    return _triton_router


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------

def fused_topk_with_score_function_fwd(logits, topk, use_pre_softmax, num_groups,
                                        group_topk, scaling_factor, score_function,
                                        expert_bias):
    """Fused topk with score function forward.

    Uses a single Triton kernel when available (no group_topk).
    Falls back to PyTorch-native for group_topk or when Triton is unavailable.

    Returns
    -------
    (probs, routing_map, intermediate_output)
    """
    triton_mod = _try_load_triton_router()
    if triton_mod is not None and logits.is_cuda:
        return triton_mod.fused_topk_with_score_function_fwd(
            logits, topk, use_pre_softmax, scaling_factor,
            score_function, expert_bias, num_groups, group_topk,
        )

    return _fused_topk_fwd_pytorch(
        logits, topk, use_pre_softmax, num_groups, group_topk,
        scaling_factor, score_function, expert_bias,
    )


def fused_topk_with_score_function_bwd(num_tokens, num_experts, routing_map,
                                        intermediate_output, grad_probs, topk,
                                        use_pre_softmax, scaling_factor, score_function):
    """Fused topk with score function backward."""
    triton_mod = _try_load_triton_router()
    if triton_mod is not None and grad_probs.is_cuda:
        return triton_mod.fused_topk_with_score_function_bwd(
            num_tokens, num_experts, routing_map, intermediate_output,
            grad_probs, topk, use_pre_softmax, scaling_factor, score_function,
        )

    return _fused_topk_bwd_pytorch(
        num_tokens, num_experts, routing_map, intermediate_output,
        grad_probs, topk, use_pre_softmax, scaling_factor, score_function,
    )


# ---------------------------------------------------------------------------
# Aux-loss score functions
# ---------------------------------------------------------------------------

def fused_score_for_moe_aux_loss_fwd(logits, topk, score_function):
    """Compute scores for MOE auxiliary loss."""
    triton_mod = _try_load_triton_router()
    if triton_mod is not None and logits.is_cuda:
        return triton_mod.fused_score_for_moe_aux_loss_fwd(
            logits, topk, score_function,
        )

    return _score_aux_loss_fwd_pytorch(logits, topk, score_function)


def fused_score_for_moe_aux_loss_bwd(num_tokens, num_experts, intermediate_output,
                                      grad_scores, topk, score_function):
    """Backward of scores for MOE auxiliary loss."""
    triton_mod = _try_load_triton_router()
    if triton_mod is not None and grad_scores.is_cuda:
        return triton_mod.fused_score_for_moe_aux_loss_bwd(
            num_tokens, num_experts, intermediate_output,
            grad_scores, topk, score_function,
        )

    return _score_aux_loss_bwd_pytorch(
        intermediate_output, grad_scores, score_function,
    )


# ---------------------------------------------------------------------------
# Aux-loss (unchanged -- already minimal, no fusion opportunity)
# ---------------------------------------------------------------------------

def fused_moe_aux_loss_fwd(probs, tokens_per_expert, total_num_tokens, num_experts,
                            num_rows, num_cols, topk, coeff):
    """MOE auxiliary (load balancing) loss forward.

    Returns
    -------
    (aux_loss, Const_buf)
        Matches the C++ interface.  Const_buf is a scalar tensor holding the
        pre-computed gradient coefficient used by the backward pass.
    """
    f = tokens_per_expert.float() / total_num_tokens
    p = probs.mean(dim=0)
    loss = coeff * num_experts * (f * p).sum()

    # Const_buf = (num_experts * coeff) / topk / total_num_tokens^2
    c_coeff = (num_experts * coeff) / topk / (total_num_tokens * total_num_tokens)
    const_buf = torch.tensor(c_coeff, dtype=torch.float32, device=probs.device)

    return loss, const_buf


def fused_moe_aux_loss_bwd(Const_buf=None, tokens_per_expert=None,
                            num_rows=None, num_cols=None, grad_aux_loss=None,
                            **kwargs):
    """MOE auxiliary loss backward.

    grad_probs[j, i] = Const_buf * tokens_per_expert[i] * grad_aux_loss
    """
    # Const_buf is a scalar, tokens_per_expert is [num_cols], grad_aux_loss is scalar
    # Output: [num_rows, num_cols]
    grad_row = Const_buf * tokens_per_expert.float() * grad_aux_loss  # [num_cols]
    grad_probs = grad_row.unsqueeze(0).expand(num_rows, num_cols).contiguous()
    return grad_probs


# =========================================================================== #
#  PyTorch-native fallbacks
# =========================================================================== #

def _fused_topk_fwd_pytorch(logits, topk, use_pre_softmax, num_groups,
                             group_topk, scaling_factor, score_function,
                             expert_bias):
    """PyTorch-native forward (supports group_topk)."""
    num_tokens, num_experts = logits.shape

    if score_function == "sigmoid":
        use_pre_softmax = False
        scores = torch.sigmoid(logits)
        intermediate_output = scores.clone()
        if expert_bias is not None:
            scores = scores + expert_bias
    elif score_function == "softmax":
        if use_pre_softmax:
            scores = F.softmax(logits, dim=-1)
            intermediate_output = scores.clone()
        else:
            scores = logits.clone()
            intermediate_output = torch.zeros_like(logits)
    else:
        raise ValueError(f"score_function must be 'softmax' or 'sigmoid', got '{score_function}'")

    use_group_topk = (
        group_topk is not None and group_topk > 0
        and num_groups is not None and num_groups > 0
    )
    if use_group_topk:
        group_size = num_experts // num_groups
        group_scores = torch.zeros(num_tokens, num_groups, device=logits.device, dtype=scores.dtype)
        for g in range(num_groups):
            g_start = g * group_size
            g_end = g_start + group_size
            g_vals, _ = torch.topk(scores[:, g_start:g_end], k=topk // group_topk, dim=-1)
            group_scores[:, g] = g_vals.sum(dim=-1)
        _, top_group_indices = torch.topk(group_scores, k=group_topk, dim=-1)
        mask = torch.zeros_like(scores, dtype=torch.bool)
        for g_idx in range(group_topk):
            g = top_group_indices[:, g_idx]
            for offset in range(group_size):
                mask[torch.arange(num_tokens, device=logits.device), g * group_size + offset] = True
        masked_scores = torch.where(mask, scores, torch.tensor(float('-inf'), device=logits.device))
        topk_values, topk_indices = torch.topk(masked_scores, k=topk, dim=-1)
    else:
        topk_values, topk_indices = torch.topk(scores, k=topk, dim=-1)

    if score_function == "sigmoid" and expert_bias is not None:
        topk_values = topk_values - expert_bias[topk_indices]

    if score_function == "softmax" and not use_pre_softmax:
        topk_values = F.softmax(topk_values, dim=-1)
        intermediate_output.scatter_(1, topk_indices, topk_values)

    if score_function == "sigmoid" and topk > 1:
        score_sum = topk_values.sum(dim=-1, keepdim=True) + _EPSILON
        topk_values = topk_values / score_sum

    if scaling_factor is not None and scaling_factor > 0:
        topk_values = topk_values * scaling_factor

    probs = torch.zeros(num_tokens, num_experts, device=logits.device, dtype=logits.dtype)
    probs.scatter_(1, topk_indices, topk_values.to(logits.dtype))

    routing_map = torch.zeros(num_tokens, num_experts, device=logits.device, dtype=torch.bool)
    routing_map.scatter_(1, topk_indices, True)

    return probs, routing_map, intermediate_output


def _fused_topk_bwd_pytorch(num_tokens, num_experts, routing_map,
                             intermediate_output, grad_probs, topk,
                             use_pre_softmax, scaling_factor, score_function):
    """PyTorch-native backward."""
    scaling_factor_val = scaling_factor if scaling_factor is not None and scaling_factor > 0 else 1.0
    grad = grad_probs * routing_map.float() * scaling_factor_val

    if score_function == "sigmoid":
        if topk > 1:
            fwd_out = intermediate_output * routing_map.float()
            sum_fwd = fwd_out.sum(dim=-1, keepdim=True) + _EPSILON
            out_x_grad = (fwd_out * grad).sum(dim=-1, keepdim=True)
            grad = torch.where(
                routing_map,
                grad / sum_fwd - out_x_grad / (sum_fwd * sum_fwd),
                torch.zeros_like(grad),
            )
        grad = grad * routing_map.float()
        grad = grad * intermediate_output * (1.0 - intermediate_output)

    elif score_function == "softmax":
        if not use_pre_softmax:
            out_x_grad = (intermediate_output * grad * routing_map.float()).sum(dim=-1, keepdim=True)
            grad = torch.where(
                routing_map,
                intermediate_output * (grad - out_x_grad),
                torch.zeros_like(grad),
            )
        else:
            grad = grad * routing_map.float()
            dot = (intermediate_output * grad).sum(dim=-1, keepdim=True)
            grad = intermediate_output * (grad - dot)

    return grad


def _score_aux_loss_fwd_pytorch(logits, topk, score_function):
    """PyTorch-native aux-loss score forward."""
    if score_function == "sigmoid":
        scores = torch.sigmoid(logits)
    else:
        scores = F.softmax(logits, dim=-1)
    intermediate_output = scores.clone()

    _, topk_indices = torch.topk(scores, k=topk, dim=-1)
    routing_map = torch.zeros_like(logits, dtype=torch.bool)
    routing_map.scatter_(1, topk_indices, True)

    return scores, routing_map, intermediate_output


def _score_aux_loss_bwd_pytorch(intermediate_output, grad_scores, score_function):
    """PyTorch-native aux-loss score backward."""
    if score_function == "sigmoid":
        grad_logits = grad_scores * intermediate_output * (1.0 - intermediate_output)
    else:
        dot = (intermediate_output * grad_scores).sum(dim=-1, keepdim=True)
        grad_logits = intermediate_output * (grad_scores - dot)
    return grad_logits
