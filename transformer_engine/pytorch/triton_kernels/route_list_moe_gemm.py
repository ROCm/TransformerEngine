# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Route-list MoE gather-GEMM (bf16): decoupled token gather and compact route store.

``sorted_slot_ids[pos]`` holds the **received-token row** for the ``A`` gather; the output
row is the compact route index ``out_row(pos) = route_start[e] + (pos - block_start[e]*BM)``.
Padding slots use sentinel ``num_recv_tokens`` in ``sorted_slot_ids`` and are masked out.

Two modes (``INDEX_A_BY_ROUTE_POS``):
- fwd: gather ``A`` by received-token row (``sorted_slot_ids``), store to compact ``out_row``.
- dgrad: read the compact grad row (``out_row``) as the ``A`` operand, store to ``out_row``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import triton
import triton.language as tl

from .pid_preprocessing import pid_grid, remap_xcd, get_num_xcds


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
ACT_SILU: tl.constexpr = 0
ACT_GELU: tl.constexpr = 1
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


@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
    }
)
@triton.jit
def _route_list_moe_persistent_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    sorted_slot_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    block_start_ptr,
    route_start_ptr,
    probs_ptr,
    preact_ptr,
    N,
    K,
    num_recv_tokens,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_pm,
    stride_pe,
    stride_prem,
    stride_pren,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    INDEX_A_BY_ROUTE_POS: tl.constexpr,
    GATED_ACT: tl.constexpr,
    ACTIVATION: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    SAVE_PREACT: tl.constexpr,
    compute_type: tl.constexpr,
    NUM_XCDS: tl.constexpr,
):
    start_pid = tl.program_id(axis=0)
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    num_pid_m = tl.cdiv(num_tokens_post_padded, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    tile_id = start_pid
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n
    num_valid_tiles = tl.cdiv((num_tiles - tile_id), NUM_SMS)

    for _ in range(0, num_valid_tiles):
        tile_id_remapped = remap_xcd(tile_id, num_tiles, NUM_XCDS)
        pid_m, pid_n = pid_grid(tile_id_remapped, num_pid_m, num_pid_n, GROUP_SIZE_M)

        off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
        if off_experts >= 0:
            pos = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
            slot = tl.load(sorted_slot_ids_ptr + pos)
            token_mask = slot < num_recv_tokens

            block_start = tl.load(block_start_ptr + off_experts).to(tl.int64)
            route_start = tl.load(route_start_ptr + off_experts).to(tl.int64)
            out_row = route_start + (pos - block_start * BLOCK_SIZE_M)

            if INDEX_A_BY_ROUTE_POS:
                a_row = out_row
            else:
                a_row = slot.to(tl.int64)

            if GATED_ACT:
                # Interleave the gate half [0, N/2) and up half [N/2, N) so a single N-block
                # holds matching (gate, up) pairs [g0,u0,g1,u1,...]; the epilogue reshape+split
                # then separates them with no permute. Requires the weight output dim laid out
                # as [gate | up].
                i = tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
                pair_idx = pid_n * (BLOCK_SIZE_N // 2) + i // 2
                offs_half = pair_idx % (N // 2)
                offs_bn = (offs_half + (i % 2) * (N // 2)) % N
            else:
                offs_bn = (
                    pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
                ) % N
            a_ptrs = a_ptr + a_row[:, None] * stride_am + offs_k[None, :] * stride_ak
            b_ptrs = (
                b_ptr
                + off_experts * stride_be
                + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
            )
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
                if EVEN_K:
                    a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
                    b = tl.load(b_ptrs)
                else:
                    a = tl.load(
                        a_ptrs,
                        mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                        other=0.0,
                    )
                    b = tl.load(
                        b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0
                    )
                accumulator += tl.dot(a, b)
                a_ptrs += BLOCK_SIZE_K * stride_ak
                b_ptrs += BLOCK_SIZE_K * stride_bk

            if GATED_ACT:
                if SAVE_PREACT:
                    # Stash the raw 2F pre-activation (gate|up) for the backward: the
                    # interleaved accumulator columns map back to their natural weight
                    # columns ``offs_bn`` (gate half [0, N/2), up half [N/2, N)), so this
                    # store lands the pre-activation in plain [gate | up] layout.
                    pre_ptrs = (
                        preact_ptr
                        + out_row[:, None] * stride_prem
                        + offs_bn[None, :] * stride_pren
                    )
                    # Mask wrapped lanes in the last N-tile (pair_idx >= N/2 aliases columns).
                    pre_mask = token_mask[:, None] & (pair_idx[None, :] < (N // 2))
                    tl.store(pre_ptrs, accumulator.to(compute_type), mask=pre_mask)
                # act(gate) * up over the interleaved columns -> N/2-wide output row. The
                # per-route gating prob (dispatched_probs[token, expert]) is applied *after*
                # the nonlinearity (silu/gelu is nonlinear, and FC2 is linear, so the correct
                # combine is w * act(gate) * up).
                g, u = accumulator.reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // 2, 2).split()
                act = _apply_activation(g, ACTIVATION) * u
                if MUL_ROUTED_WEIGHT:
                    prob = tl.load(
                        probs_ptr + slot * stride_pm + off_experts * stride_pe,
                        mask=token_mask,
                        other=0.0,
                    ).to(tl.float32)
                    act = act * prob[:, None]
                offs_cn = pid_n * (BLOCK_SIZE_N // 2) + tl.arange(0, BLOCK_SIZE_N // 2)
                c_mask = token_mask[:, None] & (offs_cn[None, :] < (N // 2))
                c_ptrs = c_ptr + stride_cm * out_row[:, None] + stride_cn * offs_cn[None, :]
                tl.store(c_ptrs, act.to(compute_type), mask=c_mask)
            else:
                offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
                c_ptrs = c_ptr + stride_cm * out_row[:, None] + stride_cn * offs_cn[None, :]
                tl.store(c_ptrs, accumulator.to(compute_type), mask=c_mask)

        tile_id += NUM_SMS


def fused_route_list_moe(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    sorted_slot_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    block_start: torch.Tensor,
    route_start: torch.Tensor,
    *,
    num_recv_tokens: int,
    compute_type: tl.dtype,
    config: Optional[Dict[str, Any]] = None,
    index_a_by_route_pos: bool = False,
    activation: Optional[str] = None,
    dispatched_probs: Optional[torch.Tensor] = None,
    preact_out: Optional[torch.Tensor] = None,
) -> None:
    """Launch route-list gather-GEMM in place.

    Writes the compact ``C[num_routes, N]`` (fwd/dgrad). The scatter back to token order is a
    separate contention-free gather-combine pass (:func:`route_gather_combine`), not fused here.

    ``activation`` (``"silu"``/``"gelu"``) enables the fused **gated** activation epilogue: the
    GEMM output ``N`` is the gate+up width (``2F``), ``C`` is the ``F``-wide activated buffer,
    and ``act(gate) * up`` is computed in-kernel. When ``dispatched_probs`` (``[num_recv_tokens,
    E]``) is given, the per-route gating prob is multiplied in *after* the activation. Only
    supported on the persistent kernel.
    """
    if config is None:
        raise ValueError("route-list MoE kernel requires an explicit tile config.")

    assert sorted_slot_ids.stride(0) == 1

    gated_act = activation is not None
    act_id = _ACT_IDS[activation] if gated_act else ACT_SILU
    mul_routed_weight = dispatched_probs is not None
    probs = dispatched_probs if dispatched_probs is not None else C
    stride_pm = dispatched_probs.stride(0) if dispatched_probs is not None else 0
    stride_pe = dispatched_probs.stride(1) if dispatched_probs is not None else 0

    save_preact = preact_out is not None
    if save_preact and not gated_act:
        raise ValueError("preact_out is only valid with a fused gated activation.")
    preact = preact_out if save_preact else C
    stride_prem = preact_out.stride(0) if save_preact else 0
    stride_pren = preact_out.stride(1) if save_preact else 0

    em = sorted_slot_ids.shape[0]
    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count * 2
    grid = lambda meta: (  # noqa: E731
        min(
            num_sms,
            triton.cdiv(em, meta["BLOCK_SIZE_M"]) * triton.cdiv(B.shape[1], meta["BLOCK_SIZE_N"]),
        ),
    )
    _route_list_moe_persistent_kernel[grid](
        A,
        B,
        C,
        sorted_slot_ids,
        expert_ids,
        num_tokens_post_padded,
        block_start,
        route_start,
        probs,
        preact,
        B.shape[1],
        A.shape[1],
        num_recv_tokens,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(0),
        C.stride(1),
        stride_pm,
        stride_pe,
        stride_prem,
        stride_pren,
        NUM_SMS=num_sms,
        INDEX_A_BY_ROUTE_POS=index_a_by_route_pos,
        GATED_ACT=gated_act,
        ACTIVATION=act_id,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        SAVE_PREACT=save_preact,
        compute_type=compute_type,
        NUM_XCDS=get_num_xcds(),
        **config,
    )

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
    ACTIVATION: tl.constexpr,
    HAS_PROBS: tl.constexpr,
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

    Returns
    -------
    (dpre, grad_probs)
        ``dpre`` is ``[T * min(topk, E), 2F]`` (bf16), grad wrt the raw GEMM output; ``grad_probs`` is
        ``[num_recv_tokens, E]`` (matching ``dispatched_probs.dtype``) or ``None``.
    """
    em_max, F = grad_out.shape
    if preact.shape[0] != em_max or preact.shape[1] != 2 * F:
        raise ValueError(
            f"preact must be [T * min(topk, E), 2F]={ (em_max, 2 * F) }, got {tuple(preact.shape)}."
        )
    routes_max = int(token.shape[0])
    has_probs = dispatched_probs is not None

    dpre = torch.empty((em_max, 2 * F), dtype=torch.bfloat16, device=grad_out.device)
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
        ACTIVATION=act_id,
        HAS_PROBS=has_probs,
    )
    if has_probs:
        grad_probs = grad_probs.to(dispatched_probs.dtype)
    return dpre, grad_probs
