# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Fused per-route routing-probability apply for the permute-free MoE path (Triton).

In the permute-free expert path each route ``r`` (a compact, expert-sorted position in the
FC1 output ``[em_max, H]``) carries a scalar gating probability
``prob[r] = dispatched_probs[token(r), expert(r)]`` that must scale the FC1 activation
before FC2. The naive way to build the padded ``[em_max]`` prob vector is a PyTorch advanced
index ``dispatched_probs[token, expert]`` whose autograd backward is
``aten::_index_put_impl_`` -- a generic, fp32, atomics-heavy scatter over the *padded*
``routes_max`` that dominates real MoE traces.

This module folds the gather + multiply (and its gradient) into two small Triton kernels
that reuse the route-list align buffers TE already builds for FC1:

- forward:  ``weighted[r, :] = activation[r, :] * dispatched_probs[token(r), expert(r)]``
  gathering the scalar prob inside the kernel; no ``[em_max]`` prob tensor is materialized.
- backward: ``grad_act[r, :] = grad_weighted[r, :] * prob[r]`` and
  ``grad_prob[r] = sum_h grad_weighted[r, h] * activation[r, h]`` scattered (masked, bounded
  to real routes) into ``grad_dispatched_probs`` with a single atomic add per route -- no
  ``index_put``.

Sentinel handling matches the rest of the route-list path: padded / over-allocated routes
carry ``token == num_recv_tokens`` in ``route_to_token`` and are masked out, so the tail is
never gathered or scattered.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

if TYPE_CHECKING:
    from ..moe_routing import MoERoutingMetadata


@triton.jit
def _route_prob_apply_fwd_kernel(
    act_ptr,
    probs_ptr,
    token_ptr,
    expert_ptr,
    out_ptr,
    num_routes_bound,
    num_recv_tokens,
    E,
    H,
    stride_am,
    stride_ah,
    stride_pm,
    stride_pe,
    stride_om,
    stride_oh,
    BLOCK_H: tl.constexpr,
):
    r = tl.program_id(axis=0)
    if r >= num_routes_bound:
        return
    token = tl.load(token_ptr + r).to(tl.int64)
    valid = token < num_recv_tokens
    expert = tl.load(expert_ptr + r).to(tl.int64)
    prob = tl.load(probs_ptr + token * stride_pm + expert * stride_pe, mask=valid, other=0.0)
    prob = prob.to(tl.float32)

    for h0 in range(0, H, BLOCK_H):
        offs_h = h0 + tl.arange(0, BLOCK_H)
        hmask = offs_h < H
        a = tl.load(act_ptr + r * stride_am + offs_h * stride_ah, mask=hmask, other=0.0)
        w = a.to(tl.float32) * prob
        tl.store(out_ptr + r * stride_om + offs_h * stride_oh, w.to(a.dtype), mask=hmask)


@triton.jit
def _route_prob_apply_bwd_kernel(
    grad_out_ptr,
    act_ptr,
    probs_ptr,
    token_ptr,
    expert_ptr,
    grad_act_ptr,
    grad_probs_ptr,
    num_routes_bound,
    num_recv_tokens,
    E,
    H,
    stride_gom,
    stride_goh,
    stride_am,
    stride_ah,
    stride_pm,
    stride_pe,
    stride_gam,
    stride_gah,
    stride_gpm,
    stride_gpe,
    BLOCK_H: tl.constexpr,
):
    r = tl.program_id(axis=0)
    if r >= num_routes_bound:
        return
    token = tl.load(token_ptr + r).to(tl.int64)
    valid = token < num_recv_tokens
    expert = tl.load(expert_ptr + r).to(tl.int64)
    prob = tl.load(probs_ptr + token * stride_pm + expert * stride_pe, mask=valid, other=0.0)
    prob = prob.to(tl.float32)

    acc = tl.zeros((), dtype=tl.float32)
    for h0 in range(0, H, BLOCK_H):
        offs_h = h0 + tl.arange(0, BLOCK_H)
        hmask = offs_h < H
        go = tl.load(grad_out_ptr + r * stride_gom + offs_h * stride_goh, mask=hmask, other=0.0)
        a = tl.load(act_ptr + r * stride_am + offs_h * stride_ah, mask=hmask, other=0.0)
        go_f = go.to(tl.float32)
        # dL/d activation = grad_out * prob
        ga = go_f * prob
        tl.store(grad_act_ptr + r * stride_gam + offs_h * stride_gah, ga.to(go.dtype), mask=hmask)
        # dL/d prob = sum_h grad_out * activation
        acc += tl.sum(go_f * a.to(tl.float32))

    # Single masked atomic add per route into grad_dispatched_probs[token, expert].
    gp_ptr = grad_probs_ptr + token * stride_gpm + expert * stride_gpe
    tl.atomic_add(gp_ptr, acc, mask=valid, sem="relaxed", scope="gpu")


def _expert_per_route(metadata: "MoERoutingMetadata", routes_max: int) -> torch.Tensor:
    """Compact per-route local-expert id, derived from the align ``route_start`` (no grad)."""
    route_start = metadata.route_start
    device = route_start.device
    num_experts = int(route_start.shape[0])
    r_idx = torch.arange(routes_max, device=device)
    # expert(r) = (#experts whose compact start <= r) - 1; duplicate starts (empty experts)
    # and the over-allocated tail are clamped to a valid expert -- those routes are masked
    # out in the kernels by the route_to_token sentinel anyway.
    expert = torch.searchsorted(route_start.to(torch.int64), r_idx, right=True) - 1
    return expert.clamp_(0, num_experts - 1).to(torch.int32)


class _ApplyRouteProbs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, activation, dispatched_probs, token, expert, num_recv_tokens):
        routes_max = int(token.shape[0])
        em_max, H = activation.shape
        out = torch.empty_like(activation)
        BLOCK_H = min(1024, triton.next_power_of_2(H))
        grid = (routes_max,)
        _route_prob_apply_fwd_kernel[grid](
            activation,
            dispatched_probs,
            token,
            expert,
            out,
            routes_max,
            num_recv_tokens,
            dispatched_probs.shape[1],
            H,
            activation.stride(0),
            activation.stride(1),
            dispatched_probs.stride(0),
            dispatched_probs.stride(1),
            out.stride(0),
            out.stride(1),
            BLOCK_H=BLOCK_H,
        )
        ctx.save_for_backward(activation, dispatched_probs, token, expert)
        ctx.num_recv_tokens = num_recv_tokens
        return out

    @staticmethod
    def backward(ctx, grad_out):
        activation, dispatched_probs, token, expert = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        routes_max = int(token.shape[0])
        _em_max, H = activation.shape
        # Zeroed so the inert tail rows [routes_max, em_max) (and empty routes) carry a 0
        # gradient into the activation producer; the kernel writes only [0, routes_max).
        grad_act = torch.zeros_like(activation)
        # Accumulate the prob gradient in fp32 (multi-route atomic adds into the same cell).
        grad_probs = torch.zeros_like(dispatched_probs, dtype=torch.float32)
        BLOCK_H = min(1024, triton.next_power_of_2(H))
        grid = (routes_max,)
        _route_prob_apply_bwd_kernel[grid](
            grad_out,
            activation,
            dispatched_probs,
            token,
            expert,
            grad_act,
            grad_probs,
            routes_max,
            ctx.num_recv_tokens,
            dispatched_probs.shape[1],
            H,
            grad_out.stride(0),
            grad_out.stride(1),
            activation.stride(0),
            activation.stride(1),
            dispatched_probs.stride(0),
            dispatched_probs.stride(1),
            grad_act.stride(0),
            grad_act.stride(1),
            grad_probs.stride(0),
            grad_probs.stride(1),
            BLOCK_H=BLOCK_H,
        )
        return grad_act, grad_probs.to(dispatched_probs.dtype), None, None, None


def apply_route_probs(
    activation: torch.Tensor,
    dispatched_probs: torch.Tensor,
    metadata: "MoERoutingMetadata",
) -> torch.Tensor:
    """Scale each route's FC1 activation by its gating probability, sync-free and fused.

    Parameters
    ----------
    activation:
        FC1 activation output ``[em_max, H]`` in the compact/padded route layout (rows
        ``[0, num_routes)`` valid, tail inert).
    dispatched_probs:
        Post-dispatch gating probs ``[num_recv_tokens, num_local_experts]`` (differentiable).
    metadata:
        ``MoERoutingMetadata`` whose FC1 align buffers are already built (``route_to_token``,
        ``route_start``). Call after the FC1 forward (or ``prepare_moe_align``).

    Returns
    -------
    torch.Tensor
        ``weighted`` ``[em_max, H]``: ``activation`` with each valid route scaled by its prob.
        Gradients flow to both ``activation`` and ``dispatched_probs`` (the latter via a fast,
        bounded, masked atomic scatter -- no ``index_put``).
    """
    route_to_token = metadata.route_to_token
    route_start = metadata.route_start
    if route_to_token is None or route_start is None:
        raise RuntimeError(
            "apply_route_probs requires the FC1 align buffers; call after the FC1 forward "
            "(or prepare_moe_align) so route_to_token / route_start are populated."
        )
    routes_max = int(route_to_token.shape[0])
    token = route_to_token.to(torch.int32)
    expert = _expert_per_route(metadata, routes_max)
    num_recv_tokens = metadata.num_recv_tokens
    return _ApplyRouteProbs.apply(activation, dispatched_probs, token, expert, num_recv_tokens)
