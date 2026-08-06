# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Route-list MoE Triton helpers (bf16): align/scan, gated-act, gather-combine.

Grouped GEMM for FC1/FC2 forward, backward (dgrad), and wgrad is FlyDSL-only
(``pf_fwd_wrapper``, ``pf_wgrad_wrapper``). This module retains Triton kernels for
routing metadata construction, gated-activation recompute/bwd, and the token-order
gather-combine pass that follows the compact route-list GEMM outputs.
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


# Activation selector for the fused epilogue. Keep the set small and explicit; add a helper
# above and an entry here to extend. Passed to the kernel as a compile-time int so each
# activation specializes to its own instance (no runtime branch in the hot path).
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


# --- Fused activation + derivative (for the gated-activation backward) ---
# The backward needs *both* act(x) and act'(x) on the same input. Computing them via
# the separate helpers above evaluates the transcendental twice (sigmoid for silu,
# tanh for gelu). These return the pair from a single transcendental -- the backward
# kernel is VALU-bound on exp2, so sharing it is the dominant win.
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
    """Tile/warp configs for the gated-act-bwd autotuner.

    Each program owns a ``BLOCK_M`` x ``BLOCK_H`` route/feature tile. The kernel is
    latency-bound (the per-route work is tiny and the memory unit is never stalled), so
    the sweep brackets the route-tile height ``BLOCK_M`` (more routes/block = more loads
    in flight to hide latency) against the feature width ``BLOCK_H`` and the warp count,
    trading memory-level parallelism against VGPR/occupancy.
    """
    return [
        triton.Config({"BLOCK_M": bm, "BLOCK_H": bh}, num_warps=w)
        for bm, bh, w in (
            # ``num_warps`` such that BLOCK_H/(warps*64) >= 8 gives 128-bit (dwordx4)
            # global loads/stores; the (1,1024,2) / (2,1024,2) points below hit that,
            # while the wider-warp points keep dwordx2 -- the tuner picks per shape.
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
    nbound_ptr,  # [1] int32 device scalar: dynamic upper bound on compact routes
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
    """Backward of the fused gated-activation (+ optional route-prob) FC1 epilogue.

    Each program owns a ``BLOCK_M`` x ``BLOCK_H`` tile of routes x gate features. Given
    ``grad_out`` (F), the saved pre-activation ``[gate | up]`` (2F) and the per-route prob
    ``p``, it emits the 2F grad wrt the raw GEMM output (``dpre = [d_gate | d_up]``) and,
    when probs are fused, reduces ``dp = sum_f grad_out * act(gate) * up`` per route and
    scatters it to ``grad_dispatched_probs[token, expert]``.

    ``act(gate)`` and ``act'(gate)`` are produced together from a single transcendental
    (the kernel is VALU-heavy on ``exp2``). Each ``(token, expert)`` cell is written by
    exactly one route, so the prob-grad scatter is a plain store (no atomics). Padded /
    over-allocated routes carry the ``token == num_recv_tokens`` sentinel and are masked
    out of the prob load/store; their ``dpre`` rows are inert (ignored downstream).

    When ``EMIT_ACT`` is set the kernel *also* stores the ``F``-wide forward activation
    ``fc2_input = act(gate)*up[*prob]`` to ``act_out_ptr`` -- the same buffer the FC2 wgrad
    would otherwise recompute in a second full pass over the ``2F`` preact. Since ``act(gate)``,
    ``up`` and ``prob`` are already live in registers here, this is one extra multiply + store
    and removes the redundant ``2F`` HBM read (see :func:`fused_gated_act_prob_fwd`).

    The compact route buffers are statically over-allocated to the worst-case
    ``routes_max = T * topk`` (sync-free shape bound), but the real routes occupy only the
    dense head ``[0, num_routes)`` -- under expert parallelism this can be ~topk*E_local/E
    times smaller. ``nbound_ptr`` carries the actual (block-padded) route extent as a device
    scalar so tail programs beyond it exit before touching HBM, instead of grinding through
    the padding at full memory bandwidth.
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
            # Re-emit the F-wide forward activation from values already in registers, so the
            # FC2 wgrad can consume it directly instead of re-streaming the 2F preact.
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
    grad_probs_shape: Optional[torch.Size] = None,
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
        Optional ``[1]`` int32 device scalar giving a (block-padded) upper bound on the number
        of *real* compact routes. The route buffers are statically sized to the worst case
        ``routes_max = T * topk``, but under expert parallelism only a small dense head is
        populated; passing the actual extent (e.g. ``num_tokens_post_padded`` from the routing
        metadata) lets tail programs exit early instead of streaming the padding through HBM.
        When ``None`` the full static ``routes_max`` is used (no early exit).
    emit_act:
        When ``True`` the kernel additionally re-materialises the ``F``-wide forward activation
        ``fc2_input = act(gate)*up[*prob]`` (``[T * min(topk, E), F]`` bf16) from the values it
        already computes, so the FC2 wgrad can consume it directly instead of re-reading the ``2F``
        preact in a separate recompute pass. Returned as the third element (``None`` otherwise).

    Returns
    -------
    (dpre, grad_probs, fc2_input)
        ``dpre`` is ``[T * min(topk, E), 2F]`` (bf16), grad wrt the raw GEMM output; ``grad_probs`` is
        ``[num_recv_tokens, E]`` (matching ``dispatched_probs.dtype``) or ``None``; ``fc2_input`` is
        ``[T * min(topk, E), F]`` (bf16) when ``emit_act`` else ``None``.
    """
    em_max, F = grad_out.shape
    if preact.shape[0] != em_max or preact.shape[1] != 2 * F:
        raise ValueError(
            f"preact must be [T * min(topk, E), 2F]={ (em_max, 2 * F) }, got {tuple(preact.shape)}."
        )
    routes_max = int(token.shape[0])
    has_probs = dispatched_probs is not None

    dpre = torch.empty((em_max, 2 * F), dtype=torch.bfloat16, device=grad_out.device)
    if emit_act:
        act_out = torch.empty((em_max, F), dtype=torch.bfloat16, device=grad_out.device)
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
    nbound_ptr,  # [1] int32 device scalar: dynamic upper bound on compact routes
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
    """Recompute the fused gated-activation FC1 output ``act(gate) * up * prob`` from preact.

    This is the *forward* counterpart of :func:`_gated_act_prob_bwd_kernel`: given the saved
    ``2F`` pre-activation ``[gate | up]`` and the per-route prob it re-materialises the
    ``F``-wide activation that the forward fused into the FC1 epilogue, so the backward can
    checkpoint only the ``2F`` preact (never the ``F``-wide act) and rebuild it just-in-time
    for the FC2 wgrad. Padded / over-allocated routes carry the ``token == num_recv_tokens``
    sentinel and get ``prob == 0`` (their act rows are ignored downstream). ``nbound_ptr``
    bounds tail programs to the real compact route extent (sync-free, EP-friendly early exit).
    """
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
    """Recompute the fused gated-activation FC1 output from the saved pre-activation.

    Re-materialises ``act = act(gate) * up * prob`` (``[routes_max, F]``, bf16) from the
    ``[routes_max, 2F]`` preact ``[gate | up]`` -- the inverse of checkpointing the ``F``-wide
    activation. Feeding this transient buffer to the *unchanged* FC2 wgrad keeps that kernel at
    full (stored-act) speed while only the ``2F`` preact is persisted across the fwd/bwd
    boundary. See :func:`fused_gated_act_prob_bwd` for the argument contract (token / expert /
    ``num_routes_bound`` are the same per-route routing arrays).
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
    src_ptr,  # block-padded [em_max, N] (padded slot order; padding rows carry dead values)
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
    # Sum the token's route rows (expert-ascending -> deterministic). Columns >= cnt are
    # unused padding and skipped, so no padded/garbage slot is ever gathered (this masked
    # padded->token reduction is the sole output-side masking point in the block-padded path).
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
    """Combine per-route rows into per-token rows via a contention-free gather.

    Parameters
    ----------
    src:
        Compact per-route buffer ``[T * min(topk, E), N]`` (route order; only ``[0, num_routes)`` valid).
    token_routes / token_route_count:
        Token->routes inverse map from the align place kernel (``[T, MAXK]`` route positions
        and the per-token route count).
    num_recv_tokens:
        Number of output token rows ``T``.
    block_n:
        N-tile width (``BLOCK_N``). ``None`` (default) picks ``min(next_pow2(N), 4096)``: wide
        tiles issue larger contiguous per-route loads and cut redundant index loads.

    Returns
    -------
    torch.Tensor
        ``[num_recv_tokens, N]`` in ``out_dtype`` -- each row is the fp32 sum of its token's
        route rows, cast once. No atomics, no host sync.
    """
    if src.dim() != 2:
        raise ValueError(f"src must be [T * min(topk, E), N], got {tuple(src.shape)}.")
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
    stride_t,
    stride_e,
    BLOCK_SIZE_M: tl.constexpr,
    BUILD_INVERSE: tl.constexpr,
    MAXK: tl.constexpr,
):
    # One row of the map per program; place the (few) routed cells deterministically into their
    # block-padded slot ``bs*BLOCK_SIZE_M + within[e, t]``. The token->routes inverse map (used by
    # the contention-free gather-combine) is emitted here too when ``BUILD_INVERSE`` -- folding
    # what was a second per-token kernel launch into this one (expert-ascending, no atomics).
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
            tl.store(sorted_slot_ids_ptr + slot, t)
            if BUILD_INVERSE:
                # Block-padded canonical layout: every intermediate route tensor (the GEMM
                # output, the FC2 route-read input, the activation) is indexed by the padded
                # slot ``bs*BLOCK_SIZE_M + w``. The token->routes inverse map therefore stores the
                # padded slot so the final padded->token gather-combine reads the correct rows.
                tl.store(token_routes_ptr + t * MAXK + j, slot)
                j += 1
    if BUILD_INVERSE:
        tl.store(token_count_ptr + t, j)


def route_list_scan(
    routing_map: torch.Tensor,
    *,
    num_experts: int,
):
    """Block-size-independent scan: per-expert token counts + within-expert ranks.

    Returned ``(counts [E], within [E, T])`` can be reused across multiple block sizes
    (e.g. the fwd/dgrad ``BLOCK_SIZE_M`` and the wgrad ``CONTRACT_M``), so the scan is
    only paid once per routing map. Pass them to :func:`route_list_align` via ``scan=``.
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
    build_inverse_map: bool = False,
):
    """Sync-free fused build of the route-list align buffers.

    Parameters
    ----------
    scan:
        Optional ``(counts, within)`` from :func:`route_list_scan` for this ``routing_map``.
        When supplied the block-independent scan kernel is skipped (shared across block
        sizes); otherwise it is computed here.
    topk:
        Host-known upper bound on the number of experts any token routes to (the router
        top-k). When provided, the static over-allocation bound is tightened from the dense
        ``T * num_experts`` to ``T * min(topk, num_experts)`` -- still
        sync-free, but shrinking the padded buffers by ``num_experts / topk``.
    build_inverse_map:
        When True, the place kernel also emits the token->routes inverse map (used by the
        contention-free gather-combine in FC2 fwd / FC1 dgrad) in the same launch, so no
        separate inverse-map kernel is needed. Block-independent, so build it on the fwd align
        only.

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

    # Static (sync-free) upper bounds from shapes only. Each token routes to at most
    # ``min(topk, E)`` experts, so that tightens the dense ``T * E`` bound.
    max_per_token = E if topk is None else min(int(topk), E)
    routes_max = T * max_per_token
    blocks_max = (routes_max + block_size - 1) // block_size + E
    em_max = blocks_max * block_size

    # Per-expert count + exclusive within-expert rank (one program per expert). Reused
    # across block sizes when the caller passes a precomputed scan.
    if scan is None:
        counts, within = route_list_scan(routing_map, num_experts=E)
    else:
        counts, within = scan

    # Per-expert placement metadata + per-block expert ids (single launch).
    blocks_per_expert = torch.empty((E,), dtype=torch.int32, device=device)  # [E]: ceil(count[e]/block_m)
    block_start = torch.empty((E,), dtype=torch.int32, device=device)  # [E]: first M-block index of expert e
    expert_ids = torch.empty((blocks_max,), dtype=torch.int32, device=device)  # [blocks_max]: owner of M-block b (-1 past end)
    num_tokens_post_padded = torch.empty((1,), dtype=torch.int32, device=device)  # [1] device scalar: real em (padded route count)
    block_b = 256
    # From per-expert route counts, derive the expert-sorted block layout: how many M-blocks
    # each expert needs, where each expert's slots start (block_start), which expert owns each
    # M-block (expert_ids), and the real padded route extent (num_tokens_post_padded).
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

    # Optional token->routes inverse map, emitted by the same place kernel. Width is the
    # tightened per-token bound; zero-init so any unused tail column is a safe (in-range)
    # index (the gather masks columns >= count anyway).
    if build_inverse_map:
        maxk = max(max_per_token, 1)
        token_routes = torch.zeros((T, maxk), dtype=torch.int32, device=device)
        token_route_count = torch.empty((T,), dtype=torch.int32, device=device)
    else:
        maxk = 1
        token_routes = torch.empty((1,), dtype=torch.int32, device=device)  # unused stub
        token_route_count = token_routes

    # One program per token: for each routed (t, e), write sorted_slot_ids[slot] = t where
    # slot = block_start[e] * block_m + within[e, t]. Unwritten slots stay at sentinel T.
    # Optionally emits token_routes / token_route_count (token -> padded slot indices).
    sorted_slot_ids = torch.full((em_max,), T, dtype=torch.int32, device=device)
    _route_list_place_kernel[(T,)](
        routing_map,
        within,
        block_start,
        sorted_slot_ids,
        token_routes,
        token_route_count,
        T,
        E,
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
