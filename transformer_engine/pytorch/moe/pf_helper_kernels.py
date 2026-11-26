# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Route-list MoE Triton helpers (bf16): align/scan, gated-act, gather-combine.

The grouped GEMMs themselves are FlyDSL (``pf_fwd_wrapper``, ``pf_wgrad_wrapper``).
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


# --- Fused gated-activation epilogue helpers (exp2-based, no libdevice) ---
@triton.jit
def _tanh(x):
    # tanh via exp2 (1.44269504089 = 1/ln2): tanh(x) = 2*sigmoid(2x) - 1.
    return 2.0 / (1.0 + tl.exp2(-2.0 * x * 1.44269504089)) - 1.0


@triton.jit
def _silu(x):
    # x * sigmoid(x), sigmoid via exp2.
    return x / (1.0 + tl.exp2(-(x * 1.44269504089)))


@triton.jit
def _gelu_tanh(x):
    # tanh approximation of GELU (matches PyTorch approximate='tanh').
    inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    return 0.5 * x * (1.0 + _tanh(inner))


# Passed as a constexpr so each activation compiles to its own kernel.
_ACT_IDS = {"silu": 0, "gelu": 1}


@triton.jit
def _apply_activation(x, ACTIVATION: tl.constexpr):
    if ACTIVATION == 0:
        return _silu(x)
    else:
        return _gelu_tanh(x)


# --- Activation derivatives (for the fused gated-activation backward) ---
@triton.jit
def _silu_grad(x):
    # d/dx [x*sigmoid(x)] = sigmoid(x) * (1 + x*(1 - sigmoid(x))).
    s = 1.0 / (1.0 + tl.exp2(-(x * 1.44269504089)))
    return s + x * s * (1.0 - s)


@triton.jit
def _gelu_tanh_grad(x):
    # d/dx [0.5*x*(1+tanh(inner))], inner = c*(x + 0.044715 x^3), c = sqrt(2/pi).
    inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    t = _tanh(inner)
    dinner = 0.7978845608028654 * (1.0 + 3.0 * 0.044715 * x * x)
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * dinner


@triton.jit
def _apply_activation_grad(x, ACTIVATION: tl.constexpr):
    if ACTIVATION == 0:
        return _silu_grad(x)
    else:
        return _gelu_tanh_grad(x)


# --- Fused activation + derivative: one exp2 for both (the backward is exp2-bound) ---
@triton.jit
def _silu_and_grad(x):
    # s = sigmoid(x); silu = x*s; d/dx silu = s + x*s*(1-s).
    s = 1.0 / (1.0 + tl.exp2(-(x * 1.44269504089)))
    act = x * s
    return act, s + act * (1.0 - s)


@triton.jit
def _gelu_tanh_and_grad(x):
    inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    t = _tanh(inner)
    dinner = 0.7978845608028654 * (1.0 + 3.0 * 0.044715 * x * x)
    return 0.5 * x * (1.0 + t), 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * dinner


@triton.jit
def _apply_activation_and_grad(x, ACTIVATION: tl.constexpr):
    if ACTIVATION == 0:
        return _silu_and_grad(x)
    else:
        return _gelu_tanh_and_grad(x)


def _gated_act_bwd_autotune_configs() -> list:
    """``BLOCK_M`` x ``BLOCK_H`` tile / warp configs for the latency-bound gated-act kernels."""
    return [
        triton.Config({"BLOCK_M": bm, "BLOCK_H": bh}, num_warps=w)
        for bm, bh, w in (
            # BLOCK_H / (warps * 64) >= 8 gives dwordx4 accesses; wider-warp points give dwordx2.
            (1, 1024, 4),
            (1, 1024, 2),
            (2, 1024, 2),
            (2, 1024, 4),
            (2, 512, 4),
            (4, 1024, 8),
            (8, 512, 4),
            (16, 256, 4),
            (16, 512, 8),
            (32, 256, 8),
            (32, 128, 4),
            (8, 256, 4),
            (4, 512, 4),
            (64, 128, 8),
        )
    ]


@triton.autotune(configs=_gated_act_bwd_autotune_configs(), key=["F", "HAS_PROBS"])
@triton.jit
def _gated_act_prob_bwd_kernel(
    grad_out_ptr,  # [T * min(topk, E), F]  grad wrt the fused FC1 output (= act(g)*u*prob)
    preact_ptr,  # [T * min(topk, E), 2F] raw pre-activation [gate | up]
    probs_ptr,  # [num_recv_tokens, E]
    token_ptr,  # [routes_max] route -> received-token row
    expert_ptr,  # [routes_max] route -> local expert
    dpre_ptr,  # [T * min(topk, E), 2F] out: grad wrt the raw 2F GEMM output
    grad_probs_ptr,  # [num_recv_tokens, E] out (fp32)
    act_out_ptr,  # [T * min(topk, E), F] out: fused fc2_input = act(g)*u*prob (EMIT_ACT only)
    nbound_ptr,  # [1] int32 device scalar: dynamic upper bound on route slots
    num_recv_tokens,
    F,
    stride_gom,
    stride_goh,
    stride_prem,
    stride_preh,
    stride_pm,
    stride_pe,
    stride_dprem,
    stride_dpreh,
    stride_gpm,
    stride_gpe,
    stride_aom,
    stride_aoh,
    ACTIVATION: tl.constexpr,
    HAS_PROBS: tl.constexpr,
    EMIT_ACT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Backward of the gated-activation (+ optional route-prob) FC1 epilogue.

    Emits ``dpre = [d_gate | d_up]`` and, with probs, ``dp = sum_f grad_out * act(gate) * up``
    as a plain store (each ``(token, expert)`` cell has exactly one route). ``EMIT_ACT`` also
    stores the forward ``act(gate) * up [* prob]`` for the FC2 wgrad. Programs past
    ``nbound_ptr`` (the real padded route extent) exit early.
    """
    pid = tl.program_id(axis=0)
    num_routes_bound = tl.load(nbound_ptr)
    if pid * BLOCK_M >= num_routes_bound:
        return
    r_offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    r_mask = r_offs < num_routes_bound
    token = tl.load(token_ptr + r_offs, mask=r_mask, other=num_recv_tokens).to(tl.int64)
    valid = token < num_recv_tokens
    expert = tl.load(expert_ptr + r_offs, mask=r_mask, other=0).to(tl.int64)
    if HAS_PROBS:
        prob = tl.load(
            probs_ptr + token * stride_pm + expert * stride_pe, mask=valid, other=0.0
        ).to(tl.float32)

    dp_acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for h0 in range(0, F, BLOCK_H):
        offs = h0 + tl.arange(0, BLOCK_H)
        hmask = offs < F
        m = r_mask[:, None] & hmask[None, :]
        prow = r_offs[:, None] * stride_prem
        g = tl.load(
            preact_ptr + prow + offs[None, :] * stride_preh, mask=m, other=0.0
        ).to(tl.float32)
        u = tl.load(
            preact_ptr + prow + (offs[None, :] + F) * stride_preh, mask=m, other=0.0
        ).to(tl.float32)
        go = tl.load(
            grad_out_ptr + r_offs[:, None] * stride_gom + offs[None, :] * stride_goh,
            mask=m,
            other=0.0,
        ).to(tl.float32)

        act_g, dact_g = _apply_activation_and_grad(g, ACTIVATION)
        if HAS_PROBS:
            dp_acc += tl.sum(go * act_g * u, axis=1)
            da = go * prob[:, None]
        else:
            da = go
        d_up = da * act_g
        d_gate = da * u * dact_g
        drow = r_offs[:, None] * stride_dprem
        tl.store(
            dpre_ptr + drow + offs[None, :] * stride_dpreh,
            d_gate.to(dpre_ptr.dtype.element_ty),
            mask=m,
        )
        tl.store(
            dpre_ptr + drow + (offs[None, :] + F) * stride_dpreh,
            d_up.to(dpre_ptr.dtype.element_ty),
            mask=m,
        )
        if EMIT_ACT:
            fi = act_g * u
            if HAS_PROBS:
                fi = fi * prob[:, None]
            tl.store(
                act_out_ptr + r_offs[:, None] * stride_aom + offs[None, :] * stride_aoh,
                fi.to(act_out_ptr.dtype.element_ty),
                mask=m,
            )

    if HAS_PROBS:
        tl.store(
            grad_probs_ptr + token * stride_gpm + expert * stride_gpe,
            dp_acc,
            mask=valid,
        )


def fused_gated_act_prob_bwd(
    grad_out: torch.Tensor,
    preact: torch.Tensor,
    token: torch.Tensor,
    expert: torch.Tensor,
    *,
    num_recv_tokens: int,
    activation: str,
    dispatched_probs: Optional[torch.Tensor] = None,
    num_routes_bound: Optional[torch.Tensor] = None,
    emit_act: bool = False,
):
    """Backward of the fused gated-activation (+ route-prob) FC1 epilogue.

    Parameters
    ----------
    grad_out:
        ``[T * min(topk, E), F]`` grad wrt the fused FC1 output (route/padded layout).
    preact:
        ``[T * min(topk, E), 2F]`` raw pre-activation ``[gate | up]`` saved by the forward.
    token, expert:
        ``[routes_max]`` per-route received-token row / local-expert id (int32).
    dispatched_probs:
        ``[num_recv_tokens, E]`` gating probs, or ``None`` if the forward did not fuse the
        route-prob multiply. When given, its gradient is returned.
    num_routes_bound:
        Optional ``[1]`` int32 device bound on real route slots (e.g. ``num_tokens_post_padded``)
        so tail programs exit early. ``None`` processes all ``routes_max``.
    emit_act:
        Also return the forward ``fc2_input = act(gate) * up [* prob]`` (bf16), saving the FC2
        wgrad a separate recompute pass.

    Returns
    -------
    (dpre, grad_probs, fc2_input)
        ``dpre`` ``[R_block, 2F]`` bf16; ``grad_probs`` ``[num_recv_tokens, E]`` in
        ``dispatched_probs.dtype`` or ``None``; ``fc2_input`` ``[R_block, F]`` bf16 or ``None``.
    """
    R_block, F = grad_out.shape
    if preact.shape[0] != R_block or preact.shape[1] != 2 * F:
        raise ValueError(
            f"preact must be [R_block, 2F]={(R_block, 2 * F)}, got {tuple(preact.shape)}."
        )
    routes_max = int(token.shape[0])
    has_probs = dispatched_probs is not None

    dpre = torch.empty((R_block, 2 * F), dtype=torch.bfloat16, device=grad_out.device)
    if emit_act:
        act_out = torch.empty((R_block, F), dtype=torch.bfloat16, device=grad_out.device)
        stride_aom, stride_aoh = act_out.stride(0), act_out.stride(1)
    else:
        act_out = None
        stride_aom = stride_aoh = 0
    if has_probs:
        grad_probs = torch.zeros(dispatched_probs.shape, dtype=torch.float32, device=grad_out.device)
        probs_ptr = dispatched_probs
        stride_pm, stride_pe = dispatched_probs.stride(0), dispatched_probs.stride(1)
        stride_gpm, stride_gpe = grad_probs.stride(0), grad_probs.stride(1)
    else:
        grad_probs = None
        probs_ptr = grad_out  # unused
        stride_pm = stride_pe = stride_gpm = stride_gpe = 0

    act_id = _ACT_IDS[activation]
    if num_routes_bound is None:
        nbound = torch.tensor([routes_max], dtype=torch.int32, device=grad_out.device)
    else:
        nbound = num_routes_bound
    grid = lambda meta: (triton.cdiv(routes_max, meta["BLOCK_M"]),)  # noqa: E731
    _gated_act_prob_bwd_kernel[grid](
        grad_out,
        preact,
        probs_ptr,
        token,
        expert,
        dpre,
        grad_probs if has_probs else grad_out,
        act_out if emit_act else dpre,
        nbound,
        num_recv_tokens,
        F,
        grad_out.stride(0),
        grad_out.stride(1),
        preact.stride(0),
        preact.stride(1),
        stride_pm,
        stride_pe,
        dpre.stride(0),
        dpre.stride(1),
        stride_gpm,
        stride_gpe,
        stride_aom,
        stride_aoh,
        ACTIVATION=act_id,
        HAS_PROBS=has_probs,
        EMIT_ACT=emit_act,
    )
    if has_probs:
        grad_probs = grad_probs.to(dispatched_probs.dtype)
    return dpre, grad_probs, act_out


@triton.autotune(configs=_gated_act_bwd_autotune_configs(), key=["F", "HAS_PROBS"])
@triton.jit
def _gated_act_prob_fwd_kernel(
    preact_ptr,  # [routes_max, 2F] raw pre-activation [gate | up]
    probs_ptr,  # [num_recv_tokens, E]
    token_ptr,  # [routes_max] route -> received-token row
    expert_ptr,  # [routes_max] route -> local expert
    act_ptr,  # [routes_max, F] out: act(gate) * up * prob
    nbound_ptr,  # [1] int32 device scalar: dynamic upper bound on route slots
    num_recv_tokens,
    F,
    stride_prem,
    stride_preh,
    stride_pm,
    stride_pe,
    stride_am,
    stride_ah,
    ACTIVATION: tl.constexpr,
    HAS_PROBS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Compute ``act(gate) * up [* prob]`` from the ``[gate | up]`` preact."""
    pid = tl.program_id(axis=0)
    num_routes_bound = tl.load(nbound_ptr)
    if pid * BLOCK_M >= num_routes_bound:
        return
    r_offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    r_mask = r_offs < num_routes_bound
    if HAS_PROBS:
        token = tl.load(token_ptr + r_offs, mask=r_mask, other=num_recv_tokens).to(tl.int64)
        valid = token < num_recv_tokens
        expert = tl.load(expert_ptr + r_offs, mask=r_mask, other=0).to(tl.int64)
        prob = tl.load(
            probs_ptr + token * stride_pm + expert * stride_pe, mask=valid, other=0.0
        ).to(tl.float32)

    for h0 in range(0, F, BLOCK_H):
        offs = h0 + tl.arange(0, BLOCK_H)
        hmask = offs < F
        m = r_mask[:, None] & hmask[None, :]
        prow = r_offs[:, None] * stride_prem
        g = tl.load(
            preact_ptr + prow + offs[None, :] * stride_preh, mask=m, other=0.0
        ).to(tl.float32)
        u = tl.load(
            preact_ptr + prow + (offs[None, :] + F) * stride_preh, mask=m, other=0.0
        ).to(tl.float32)
        act = _apply_activation(g, ACTIVATION) * u
        if HAS_PROBS:
            act = act * prob[:, None]
        tl.store(
            act_ptr + r_offs[:, None] * stride_am + offs[None, :] * stride_ah,
            act.to(act_ptr.dtype.element_ty),
            mask=m,
        )


def fused_gated_act_prob_fwd(
    preact: torch.Tensor,
    token: torch.Tensor,
    expert: torch.Tensor,
    *,
    num_recv_tokens: int,
    activation: str,
    dispatched_probs: Optional[torch.Tensor] = None,
    num_routes_bound: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute ``act(gate) * up [* prob]`` (``[routes_max, F]`` bf16) from the ``2F`` preact.

    Arguments match :func:`fused_gated_act_prob_bwd`.
    """
    routes_max, two_f = preact.shape
    if two_f % 2 != 0:
        raise ValueError(f"preact must be [routes_max, 2F], got a {two_f}-wide last dim.")
    F = two_f // 2
    has_probs = dispatched_probs is not None

    act = torch.empty((routes_max, F), dtype=torch.bfloat16, device=preact.device)
    if has_probs:
        probs_ptr = dispatched_probs
        stride_pm, stride_pe = dispatched_probs.stride(0), dispatched_probs.stride(1)
    else:
        probs_ptr = preact  # unused
        stride_pm = stride_pe = 0

    act_id = _ACT_IDS[activation]
    if num_routes_bound is None:
        nbound = torch.tensor([routes_max], dtype=torch.int32, device=preact.device)
    else:
        nbound = num_routes_bound
    grid = lambda meta: (triton.cdiv(routes_max, meta["BLOCK_M"]),)  # noqa: E731
    _gated_act_prob_fwd_kernel[grid](
        preact,
        probs_ptr,
        token,
        expert,
        act,
        nbound,
        num_recv_tokens,
        F,
        preact.stride(0),
        preact.stride(1),
        stride_pm,
        stride_pe,
        act.stride(0),
        act.stride(1),
        ACTIVATION=act_id,
        HAS_PROBS=has_probs,
    )
    return act



@triton.jit
def _gather_combine_kernel(
    src_ptr,  # block-padded [R_block, N] (padded slot order; padding rows carry dead values)
    token_routes_ptr,  # [T, MAXK] int32: block-padded slot positions per token
    token_count_ptr,  # [T] int32: number of routes for each token
    out_ptr,  # [T, N] out
    N,
    stride_sm,
    stride_sn,
    stride_om,
    stride_on,
    MAXK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    compute_type: tl.constexpr,
):
    t = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    cnt = tl.load(token_count_ptr + t)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    # Expert-ascending sum (deterministic); columns >= cnt are padding and never read.
    for j in range(0, MAXK):
        if j < cnt:
            r = tl.load(token_routes_ptr + t * MAXK + j).to(tl.int64)
            v = tl.load(src_ptr + r * stride_sm + offs_n * stride_sn, mask=n_mask, other=0.0)
            acc += v.to(tl.float32)
    tl.store(out_ptr + t * stride_om + offs_n * stride_on, acc.to(compute_type), mask=n_mask)


def route_gather_combine(
    src: torch.Tensor,
    token_routes: torch.Tensor,
    token_route_count: torch.Tensor,
    num_recv_tokens: int,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
    block_n: Optional[int] = None,
) -> torch.Tensor:
    """Sum each token's route rows into a token-ordered output (no atomics, no host sync).

    Parameters
    ----------
    src:
        Block-padded route-ordered ``[R_block, N]`` buffer.
    token_routes / token_route_count:
        ``[T, MAXK]`` slot positions per token and the per-token route count.
    num_recv_tokens:
        Number of output token rows ``T``.
    block_n:
        N-tile width; ``None`` picks ``min(next_pow2(N), 4096)``.

    Returns
    -------
    torch.Tensor
        ``[num_recv_tokens, N]`` in ``out_dtype`` (summed in fp32).
    """
    if src.dim() != 2:
        raise ValueError(f"src must be [R_block, N], got {tuple(src.shape)}.")
    if not src.is_contiguous():
        src = src.contiguous()
    n = src.shape[1]
    if block_n is None:
        block_n = min(triton.next_power_of_2(n), 4096)
    maxk = int(token_routes.shape[1])
    out = torch.empty((num_recv_tokens, n), dtype=out_dtype, device=src.device)
    compute_type = tl.bfloat16 if out_dtype == torch.bfloat16 else tl.float32
    grid = (num_recv_tokens, triton.cdiv(n, block_n))
    _gather_combine_kernel[grid](
        src,
        token_routes,
        token_route_count,
        out,
        n,
        src.stride(0),
        src.stride(1),
        out.stride(0),
        out.stride(1),
        MAXK=maxk,
        BLOCK_N=block_n,
        compute_type=compute_type,
    )
    return out


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
    # block_start is over padded block token counts
    block_start = cblocks - blocks_per_expert
    total_blocks = tl.max(tl.where(mask_e, cblocks, 0), axis=0)

    pid = tl.program_id(axis=0)
    if pid == 0:
        tl.store(blocks_per_expert_ptr + offs_e, blocks_per_expert, mask=mask_e)
        tl.store(block_start_ptr + offs_e, block_start, mask=mask_e)
        tl.store(ntpp_ptr, total_blocks * BLOCK_SIZE_M)

    # expert_ids[b] = #{e : cblocks[e] <= b} then -1 for blocks past the real extent.
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
    sorted_slot_ids_ptr,  # [T * min(topk, E)] int32, sentinel-init (T)
    token_routes_ptr,  # [T, MAXK] int32 out: token->route positions (only if BUILD_INVERSE)
    token_count_ptr,  # [T] int32 out: routes per token (only if BUILD_INVERSE)
    T,
    E,
    R_block,
    stride_t,
    stride_e,
    BLOCK_SIZE_M: tl.constexpr,
    BUILD_INVERSE: tl.constexpr,
    MAXK: tl.constexpr,
):
    # One token per program; with ``BUILD_INVERSE`` also emits its slots (expert-ascending).
    t = tl.program_id(axis=0)
    if t >= T:
        return
    j = tl.zeros((), dtype=tl.int32)
    for e in range(0, E):
        is_routed = tl.load(routing_map_ptr + t * stride_t + e * stride_e)
        if is_routed != 0:
            w = tl.load(within_ptr + e * T + t)
            bs = tl.load(block_start_ptr + e)
            slot = bs * BLOCK_SIZE_M + w
            # Masked so an undersized ``topk``/``num_routes`` hint drops routes, not memory.
            tl.store(sorted_slot_ids_ptr + slot, t, mask=slot < R_block)
            if BUILD_INVERSE:
                tl.store(token_routes_ptr + t * MAXK + j, slot, mask=j < MAXK)
                j += 1
    if BUILD_INVERSE:
        tl.store(token_count_ptr + t, tl.minimum(j, MAXK))


def route_list_scan(
    routing_map: torch.Tensor,
    *,
    num_experts: int,
):
    """Per-expert counts and within-expert ranks ``(counts [E], within [E, T])``.

    Block-size independent, so one scan can feed every :func:`route_list_align` via ``scan=``.
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
    topk: int | None = None,
    num_routes: int | None = None,
    build_inverse_map: bool = False,
):
    """Sync-free fused build of the route-list align buffers.

    Parameters
    ----------
    scan:
        Optional precomputed :func:`route_list_scan` result for this ``routing_map``.
    topk, num_routes:
        Optional route-buffer bounds; see :class:`MoERoutingMetadata`.
    build_inverse_map:
        Also emit ``token_routes`` / ``token_route_count``. Block-size independent, so build it
        on one align only.

    Returns
    -------
    ``(sorted_slot_ids, expert_ids, num_tokens_post_padded, block_start, blocks_per_expert,
       token_routes, token_route_count)`` -- index tensors ``int32``; ``num_tokens_post_padded``
    (``[1]``) is a device scalar. ``token_routes`` (``[T, min(topk, E)]``) and
    ``token_route_count`` (``[T]``) are ``None`` unless ``build_inverse_map``.
    """
    if routing_map.dtype != torch.bool:
        routing_map = routing_map.bool()
    routing_map = routing_map.contiguous()
    device = routing_map.device
    T = int(routing_map.size(0))
    E = int(num_experts)

    # Sync-free bound from shapes; ``+ E`` covers one partial block per expert.
    max_per_token = E if topk is None else min(int(topk), E)
    routes_max = T * max_per_token if num_routes is None else int(num_routes)
    blocks_max = (routes_max + block_size - 1) // block_size + E
    R_block = blocks_max * block_size

    if scan is None:
        counts, within = route_list_scan(routing_map, num_experts=E)
    else:
        counts, within = scan

    blocks_per_expert = torch.empty((E,), dtype=torch.int32, device=device)
    block_start = torch.empty((E,), dtype=torch.int32, device=device)
    expert_ids = torch.empty((blocks_max,), dtype=torch.int32, device=device)
    num_tokens_post_padded = torch.empty((1,), dtype=torch.int32, device=device)
    block_b = 256
    _expert_meta_kernel[(triton.cdiv(blocks_max, block_b),)](
        counts,
        blocks_per_expert,
        block_start,
        expert_ids,
        num_tokens_post_padded,
        E,
        blocks_max,
        BLOCK_SIZE_M=block_size,
        BLOCK_E=triton.next_power_of_2(max(E, 1)),
        BLOCK_B=block_b,
    )

    # Zero-init so unused columns hold an in-range index.
    if build_inverse_map:
        maxk = max(max_per_token, 1)
        token_routes = torch.zeros((T, maxk), dtype=torch.int32, device=device)
        token_route_count = torch.empty((T,), dtype=torch.int32, device=device)
    else:
        maxk = 1
        token_routes = torch.empty((1,), dtype=torch.int32, device=device)  # unused stub
        token_route_count = token_routes

    # Unwritten slots keep the sentinel T.
    sorted_slot_ids = torch.full((R_block,), T, dtype=torch.int32, device=device)
    _route_list_place_kernel[(T,)](
        routing_map,
        within,
        block_start,
        sorted_slot_ids,
        token_routes,
        token_route_count,
        T,
        E,
        R_block,
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
        token_routes if build_inverse_map else None,
        token_route_count if build_inverse_map else None,
    )
