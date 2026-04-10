# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Triton JIT kernels for fused MoE router operations.

Fuses score-function (sigmoid/softmax) + top-k + post-processing into a
single kernel launch, matching the C++ ``fused_topk_with_score_function``
kernel behaviour.
"""

import triton
import triton.language as tl


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

@triton.jit
def _iterative_topk(
    scores,
    valid,
    offs,
    selected,
    topk_vals,
    TOPK: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    """Select top-TOPK from *scores* at *valid* & unselected positions.

    Updates *selected* (0/1 int32 mask) and *topk_vals* in-place and returns
    the updated pair.
    """
    for _k in tl.static_range(TOPK):
        avail = valid & (selected == 0)
        masked = tl.where(avail, scores, float("-inf"))
        best = tl.max(masked, axis=0)
        is_best = (masked == best) & avail
        candidate = tl.where(is_best, offs, BLOCK_E)
        winner = tl.min(candidate, axis=0)
        newly = (offs == winner)
        selected = tl.where(newly, 1, selected)
        topk_vals = tl.where(newly, scores, topk_vals)
    return selected, topk_vals


# --------------------------------------------------------------------------- #
#  Forward kernel
# --------------------------------------------------------------------------- #

@triton.jit
def _fused_topk_score_fwd_kernel(
    logits_ptr,
    probs_ptr,
    routing_map_ptr,
    intermediate_ptr,
    bias_ptr,
    num_tokens,
    scaling_factor,
    # compile-time constants ------------------------------------------------
    NUM_EXPERTS: tl.constexpr,
    TOPK: tl.constexpr,
    SCORE_FN: tl.constexpr,       # 0 = sigmoid, 1 = softmax
    USE_PRE_SOFTMAX: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    USE_GROUP_TOPK: tl.constexpr,
    NUM_GROUPS: tl.constexpr,     # ignored when USE_GROUP_TOPK == False
    GROUP_TOPK: tl.constexpr,     # ignored when USE_GROUP_TOPK == False
    GROUP_SIZE: tl.constexpr,     # NUM_EXPERTS // NUM_GROUPS
    BLOCK_E: tl.constexpr,        # next_power_of_2(NUM_EXPERTS)
    BLOCK_G: tl.constexpr,        # next_power_of_2(NUM_GROUPS)
):
    """One program-instance per token."""
    pid = tl.program_id(0)
    if pid >= num_tokens:
        return

    offs = tl.arange(0, BLOCK_E)
    valid = offs < NUM_EXPERTS
    base = pid * NUM_EXPERTS

    # -- load logits --------------------------------------------------------
    logits = tl.load(logits_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)

    # -- Step 1: score function ---------------------------------------------
    if SCORE_FN == 0:  # sigmoid
        scores = tl.sigmoid(logits)
        intermediate = scores  # saved for backward
        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs, mask=valid, other=0.0).to(tl.float32)
            routing_scores = scores + bias
        else:
            routing_scores = scores
    else:  # softmax
        if USE_PRE_SOFTMAX:
            max_val = tl.max(tl.where(valid, logits, float("-inf")), axis=0)
            exp_l = tl.exp(logits - max_val)
            exp_l = tl.where(valid, exp_l, 0.0)
            sum_exp = tl.sum(exp_l, axis=0)
            scores = exp_l / sum_exp
            intermediate = scores
            routing_scores = scores
        else:
            routing_scores = logits
            scores = logits
            intermediate = tl.zeros([BLOCK_E], dtype=tl.float32)

    # -- Step 2: top-k (with optional group selection) ----------------------
    if USE_GROUP_TOPK:
        # 2a. Score each group: sum of top-(TOPK//GROUP_TOPK) within group
        g_offs = tl.arange(0, BLOCK_G)
        g_valid = g_offs < NUM_GROUPS
        group_scores = tl.zeros([BLOCK_G], dtype=tl.float32)

        for g in tl.static_range(NUM_GROUPS):
            in_group = ((offs // GROUP_SIZE) == g) & valid
            g_sel = tl.zeros([BLOCK_E], dtype=tl.int32)
            g_sum = tl.zeros([1], dtype=tl.float32)  # accumulator
            for _ik in tl.static_range(TOPK // GROUP_TOPK):
                g_avail = in_group & (g_sel == 0)
                g_masked = tl.where(g_avail, routing_scores, float("-inf"))
                g_best = tl.max(g_masked, axis=0)
                g_is_best = (g_masked == g_best) & g_avail
                g_cand = tl.where(g_is_best, offs, BLOCK_E)
                g_win = tl.min(g_cand, axis=0)
                g_sel = tl.where(offs == g_win, 1, g_sel)
                g_sum += g_best
            group_scores = tl.where(g_offs == g, g_sum, group_scores)

        # 2b. Select top GROUP_TOPK groups
        selected_groups = tl.zeros([BLOCK_G], dtype=tl.int32)
        for _gk in tl.static_range(GROUP_TOPK):
            ga = g_valid & (selected_groups == 0)
            gm = tl.where(ga, group_scores, float("-inf"))
            gb = tl.max(gm, axis=0)
            gib = (gm == gb) & ga
            gc = tl.where(gib, g_offs, BLOCK_G)
            gw = tl.min(gc, axis=0)
            selected_groups = tl.where(g_offs == gw, 1, selected_groups)

        # 2c. Build expert mask from selected groups
        expert_mask = tl.zeros([BLOCK_E], dtype=tl.int32)
        for g in tl.static_range(NUM_GROUPS):
            g_is_sel = tl.sum(tl.where(g_offs == g, selected_groups, 0), axis=0)
            in_g = ((offs // GROUP_SIZE) == g) & valid
            expert_mask = tl.where(in_g & (g_is_sel > 0), 1, expert_mask)

        # 2d. Top-k over experts in selected groups
        topk_scores_for_sel = tl.where(expert_mask != 0, routing_scores, float("-inf"))
    else:
        topk_scores_for_sel = routing_scores

    selected = tl.zeros([BLOCK_E], dtype=tl.int32)
    topk_vals = tl.zeros([BLOCK_E], dtype=tl.float32)
    selected, topk_vals = _iterative_topk(
        topk_scores_for_sel, valid, offs, selected, topk_vals,
        TOPK=TOPK, BLOCK_E=BLOCK_E,
    )
    sel_bool = selected != 0

    # -- Step 3: post-processing --------------------------------------------
    if SCORE_FN == 0:  # sigmoid
        if HAS_BIAS:
            topk_vals = tl.where(sel_bool, topk_vals - bias, topk_vals)
        if TOPK > 1:
            s = tl.sum(tl.where(sel_bool, topk_vals, 0.0), axis=0) + 1e-9
            topk_vals = tl.where(sel_bool, topk_vals / s, 0.0)
    else:  # softmax
        if not USE_PRE_SOFTMAX:
            sel_logits = tl.where(sel_bool, topk_vals, float("-inf"))
            mx = tl.max(sel_logits, axis=0)
            e = tl.exp(sel_logits - mx)
            e = tl.where(sel_bool, e, 0.0)
            topk_vals = e / tl.sum(e, axis=0)
            intermediate = tl.where(sel_bool, topk_vals, intermediate)

    # scaling
    topk_vals = tl.where(sel_bool, topk_vals * scaling_factor, 0.0)

    # -- Step 4: store outputs ----------------------------------------------
    tl.store(probs_ptr + base + offs,
             tl.where(sel_bool & valid, topk_vals, 0.0).to(logits.dtype),
             mask=valid)
    tl.store(routing_map_ptr + base + offs,
             selected.to(tl.int8),
             mask=valid)
    tl.store(intermediate_ptr + base + offs,
             tl.where(valid, intermediate, 0.0).to(logits.dtype),
             mask=valid)


# --------------------------------------------------------------------------- #
#  Backward kernel (unchanged — group topk doesn't affect backward)
# --------------------------------------------------------------------------- #

@triton.jit
def _fused_topk_score_bwd_kernel(
    routing_map_ptr,
    intermediate_ptr,
    grad_probs_ptr,
    grad_logits_ptr,
    num_tokens,
    scaling_factor,
    # compile-time constants ------------------------------------------------
    NUM_EXPERTS: tl.constexpr,
    TOPK: tl.constexpr,
    SCORE_FN: tl.constexpr,
    USE_PRE_SOFTMAX: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    """One program-instance per token."""
    pid = tl.program_id(0)
    if pid >= num_tokens:
        return

    offs = tl.arange(0, BLOCK_E)
    valid = offs < NUM_EXPERTS
    base = pid * NUM_EXPERTS

    grad = tl.load(grad_probs_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)
    sel_i8 = tl.load(routing_map_ptr + base + offs, mask=valid, other=0)
    sel = sel_i8 != 0
    fwd_out = tl.load(intermediate_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)

    # scale selected grads
    grad = tl.where(sel, grad * scaling_factor, grad)

    if SCORE_FN == 0:  # sigmoid
        if TOPK > 1:
            fwd_sel = tl.where(sel, fwd_out, 0.0)
            s = tl.sum(fwd_sel, axis=0) + 1e-9
            og = tl.sum(tl.where(sel, fwd_sel * grad, 0.0), axis=0)
            grad = tl.where(sel, grad / s - og / (s * s), 0.0)
        # mask unselected
        grad = tl.where(sel, grad, 0.0)
        # sigmoid derivative
        grad = grad * fwd_out * (1.0 - fwd_out)
    else:  # softmax
        if not USE_PRE_SOFTMAX:
            og = tl.sum(tl.where(sel, fwd_out * grad, 0.0), axis=0)
            grad = tl.where(sel, fwd_out * (grad - og), 0.0)
        else:
            grad = tl.where(sel, grad, 0.0)
            dot = tl.sum(tl.where(valid, fwd_out * grad, 0.0), axis=0)
            grad = fwd_out * (grad - dot)

    tl.store(grad_logits_ptr + base + offs,
             tl.where(valid, grad, 0.0).to(fwd_out.dtype),
             mask=valid)


# --------------------------------------------------------------------------- #
#  Aux-loss score forward kernel
# --------------------------------------------------------------------------- #

@triton.jit
def _fused_score_aux_loss_fwd_kernel(
    logits_ptr,
    scores_ptr,
    routing_map_ptr,
    intermediate_ptr,
    num_tokens,
    NUM_EXPERTS: tl.constexpr,
    TOPK: tl.constexpr,
    SCORE_FN: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    """Score computation for auxiliary loss — one program per token."""
    pid = tl.program_id(0)
    if pid >= num_tokens:
        return

    offs = tl.arange(0, BLOCK_E)
    valid = offs < NUM_EXPERTS
    base = pid * NUM_EXPERTS

    logits = tl.load(logits_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)

    if SCORE_FN == 0:  # sigmoid
        scores = tl.sigmoid(logits)
    else:  # softmax
        mx = tl.max(tl.where(valid, logits, float("-inf")), axis=0)
        e = tl.exp(logits - mx)
        e = tl.where(valid, e, 0.0)
        scores = e / tl.sum(e, axis=0)

    # top-k for routing map
    selected = tl.zeros([BLOCK_E], dtype=tl.int32)
    topk_vals = tl.zeros([BLOCK_E], dtype=tl.float32)
    selected, topk_vals = _iterative_topk(
        scores, valid, offs, selected, topk_vals,
        TOPK=TOPK, BLOCK_E=BLOCK_E,
    )

    tl.store(scores_ptr + base + offs,
             tl.where(valid, scores, 0.0).to(logits.dtype), mask=valid)
    tl.store(routing_map_ptr + base + offs,
             selected.to(tl.int8), mask=valid)
    tl.store(intermediate_ptr + base + offs,
             tl.where(valid, scores, 0.0).to(logits.dtype), mask=valid)


# --------------------------------------------------------------------------- #
#  Aux-loss score backward kernel
# --------------------------------------------------------------------------- #

@triton.jit
def _fused_score_aux_loss_bwd_kernel(
    intermediate_ptr,
    grad_scores_ptr,
    grad_logits_ptr,
    num_tokens,
    NUM_EXPERTS: tl.constexpr,
    SCORE_FN: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    """Score backward for auxiliary loss — one program per token."""
    pid = tl.program_id(0)
    if pid >= num_tokens:
        return

    offs = tl.arange(0, BLOCK_E)
    valid = offs < NUM_EXPERTS
    base = pid * NUM_EXPERTS

    fwd_out = tl.load(intermediate_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)
    g = tl.load(grad_scores_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)

    if SCORE_FN == 0:  # sigmoid
        grad = g * fwd_out * (1.0 - fwd_out)
    else:  # softmax
        dot = tl.sum(tl.where(valid, fwd_out * g, 0.0), axis=0)
        grad = fwd_out * (g - dot)

    tl.store(grad_logits_ptr + base + offs,
             tl.where(valid, grad, 0.0).to(fwd_out.dtype), mask=valid)
