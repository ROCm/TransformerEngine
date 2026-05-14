# Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Triton score-relu-reduce kernel for the lightning-indexer hybrid backend.

The hybrid backend computes the four projections (C_q, H_q, H_k, W_o) via
``jnp.einsum`` (which lowers to hipBLASLt bf16 GEMMs) and then hands the
results to this kernel for the score matmul + ReLU + per-(t, h) weighted
H-reduction:

    scores = relu(einsum("...thi,...si->...ths", H_q, H_k))   # never written
    O      = einsum("...ths,...th->...ts", scores, W_o)

The kernel keeps each per-head score tile in registers, avoiding the
(B, oH, T, H, S) HBM round-trip that an einsum-only implementation pays
on the pre-relu score tensor.
"""

import functools

import jax
import jax.numpy as jnp
import triton
import triton.language as tl

from jax import core
from jax.extend import core as extend_core
from jax.interpreters import mlir, xla

from .utils import triton_call_lowering


def _score_reduce_autotune_configs():
    # The kernel is dominated by Hq reads (one (BLOCK_T, d_i) load per H
    # iteration). Bigger BLOCK_T ⇒ fewer T tiles ⇒ less total Hq traffic.
    # Bigger BLOCK_S ⇒ more Hk reuse but bigger per-CTA footprint.
    #
    # BLOCK_T=512 was tried and consistently failed to launch on MI355X
    # (resource exhaustion — VGPR/LDS budget for 64-iter H-loop with that
    # large an accumulator). Capped at 256.
    cfgs = []
    for bt in (64, 128, 256):
        for bs in (32, 64, 128):
            for num_warps in (4, 8):
                for num_stages in (1, 2):
                    cfgs.append(triton.Config(
                        {"BLOCK_T": bt, "BLOCK_S": bs},
                        num_warps=num_warps, num_stages=num_stages,
                    ))
    # A few skinny / fat shapes the regular grid above won't hit.
    cfgs += [
        triton.Config({"BLOCK_T": 32,  "BLOCK_S": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_T": 32,  "BLOCK_S": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_T": 256, "BLOCK_S": 32},  num_warps=8, num_stages=2),
    ]
    return cfgs


@triton.autotune(configs=_score_reduce_autotune_configs(), key=["H", "d_i"])
@triton.jit
def _score_reduce_kernel(
    Hq_ptr,       # (B, oH, T_t, H, d_i) — produced by einsum("...tc,hci->...thi")
    Hk_ptr,       # (B, oH, T_s, d_i)
    W_o_ptr,      # (B, oH, T_t, H)
    O_ptr,        # (B, oH, T_t, T_s)
    B: tl.constexpr,
    oH: tl.constexpr,
    T_t: tl.constexpr,
    T_s: tl.constexpr,
    H: tl.constexpr,
    d_i: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """Compute one (BLOCK_T, BLOCK_S) tile of O for one (b, h_outer) slice.

    Grid order: (cdiv(T_s, BLOCK_S), cdiv(T_t, BLOCK_T), B * oH).

    S is the fastest-dispatching axis so consecutive CTAs share (B*oH, T)
    and vary only in S — they all read the same per-head Hq slab, hitting
    L2 instead of HBM. Hq layout is the natural einsum output
    (..., T, H, d_i); per-head loads are strided in T (stride H*d_i).
    """
    pid_s = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_bh = tl.program_id(2)

    # int64 indexing — Hq alone has B*oH*T*H*d_i = 4.3 B elements at T=S=4096,
    # exceeds int32 range.
    b = (pid_bh // oH).to(tl.int64)
    h_outer = (pid_bh % oH).to(tl.int64)

    rt = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    rs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    rdi = tl.arange(0, d_i)

    rt_mask = rt < T_t
    rs_mask = rs < T_s

    hq_base = b * (oH * T_t * H * d_i) + h_outer * (T_t * H * d_i)
    hk_base = b * (oH * T_s * d_i) + h_outer * (T_s * d_i)
    wo_base = b * (oH * T_t * H) + h_outer * (T_t * H)
    o_base = b * (oH * T_t * T_s) + h_outer * (T_t * T_s)

    # Load the (BLOCK_S, d_i) Hk slab once — it is loop-invariant over H.
    hk_ptrs = Hk_ptr + hk_base + rs[:, None] * d_i + rdi[None, :]
    Hk_tile = tl.load(hk_ptrs, mask=rs_mask[:, None], other=0.0)
    Hk_T = tl.trans(Hk_tile)  # (d_i, BLOCK_S)

    acc = tl.zeros((BLOCK_T, BLOCK_S), dtype=tl.float32)

    for h in range(H):
        hq_ptrs = (Hq_ptr + hq_base
                   + rt[:, None] * (H * d_i) + h * d_i + rdi[None, :])
        Hq_h = tl.load(hq_ptrs, mask=rt_mask[:, None], other=0.0)

        wo_ptrs = W_o_ptr + wo_base + rt * H + h
        w_h = tl.load(wo_ptrs, mask=rt_mask, other=0.0)

        score = tl.dot(Hq_h, Hk_T)
        score = tl.maximum(score, 0.0)
        acc += score * w_h[:, None].to(tl.float32)

    o_ptrs = O_ptr + o_base + rt[:, None] * T_s + rs[None, :]
    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty),
             mask=rt_mask[:, None] & rs_mask[None, :])


_score_reduce_p = extend_core.Primitive("te_indexer_score_reduce_triton")
_score_reduce_p.multiple_results = True


@_score_reduce_p.def_abstract_eval
def _score_reduce_abstract(Hq, Hk, W_o, *, out_dtype):
    del W_o
    # Hq layout: (B, oH, T_t, H, d_i)
    B, oH, T_t, _H, _d_i = Hq.shape
    T_s = Hk.shape[2]
    return [core.ShapedArray((B, oH, T_t, T_s), out_dtype)]


_score_reduce_p.def_impl(functools.partial(xla.apply_primitive, _score_reduce_p))


def _score_reduce_lowering(ctx, Hq, Hk, W_o, *, out_dtype):
    del out_dtype
    Hq_aval = ctx.avals_in[0]
    Hk_aval = ctx.avals_in[1]
    B, oH, T_t, H, d_i = Hq_aval.shape
    T_s = Hk_aval.shape[2]

    def grid_fn(merged_kwargs):
        bt = merged_kwargs.get("BLOCK_T", 64)
        bs = merged_kwargs.get("BLOCK_S", 64)
        # S as grid_x (fastest-dispatching) so per-(B*oH, T-tile) S workgroups
        # cluster in time and hit L2 on the shared Hq slab.
        return (triton.cdiv(T_s, bs), triton.cdiv(T_t, bt), B * oH)

    return triton_call_lowering(
        ctx,
        _score_reduce_kernel,
        Hq, Hk, W_o,
        grid=grid_fn,
        num_warps=4,
        num_stages=2,
        constexprs={
            "B": B,
            "oH": oH,
            "T_t": T_t,
            "T_s": T_s,
            "H": H,
            "d_i": d_i,
        },
    )


mlir.register_lowering(_score_reduce_p, _score_reduce_lowering, platform="rocm")
mlir.register_lowering(_score_reduce_p, _score_reduce_lowering, platform="cuda")


# --- Backward: dHq + dW_o kernel ----------------------------------------------
#
# FlashAttention-style: residuals saved from forward = (Hq, Hk, W_o). The
# (T, H, S) score tensor is recomputed inside this kernel from Hq @ Hk^T,
# so we never store H = relu(scores) -- which is 549 GB at the production
# 4096^2 shape.
#
# Math:
#   scores[t, h, s] = sum_i Hq[t, h, i] * Hk[s, i]
#   H_relu[t, h, s] = max(scores, 0)
#   O[t, s]         = sum_h H_relu[t, h, s] * W_o[t, h]
#
# Cotangents:
#   dW_o[t, h]  = sum_s dO[t, s] * H_relu[t, h, s]
#   dH[t,h,s]   = dO[t, s] * W_o[t, h]
#   dscores     = dH * (scores > 0)            # ReLU mask
#   dHq[t,h,i]  = sum_s dscores[t,h,s] * Hk[s, i]
#   dHk[s,i]    = sum_t sum_h dscores[t,h,s] * Hq[t, h, i]
#
# Kernel A (this one): computes dHq and dW_o.
#   Grid: (cdiv(T_t, BLOCK_T), B * oH). Each CTA owns BLOCK_T rows of dHq
#   and dW_o exclusively -- no atomics needed since the full S range is
#   reduced inside one CTA.
#
# Kernel B (next section): computes dHk. Grid (cdiv(T_s, BLOCK_S), B * oH);
# each CTA owns BLOCK_S rows of dHk and reduces over all T inside.


@triton.jit
def _score_reduce_dHq_dWo_kernel(
    Hq_ptr,    # (B, oH, T_t, H, d_i) bf16
    Hk_ptr,    # (B, oH, T_s, d_i) bf16
    W_o_ptr,   # (B, oH, T_t, H) bf16
    dO_ptr,    # (B, oH, T_t, T_s) fp32   (caller upcasts)
    dHq_ptr,   # (B, oH, T_t, H, d_i) bf16  OUTPUT
    dWo_ptr,   # (B, oH, T_t, H)      bf16  OUTPUT
    B: tl.constexpr,
    oH: tl.constexpr,
    T_t: tl.constexpr,
    T_s: tl.constexpr,
    H: tl.constexpr,
    d_i: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """Per-CTA: produces dHq[BLOCK_T, H, d_i] and dW_o[BLOCK_T, H].

    Outer-h loop / inner-s loop. For each h, we accumulate dHq and dW_o
    contributions over the full S range, then store and move on.
    """
    pid_t = tl.program_id(0)
    pid_bh = tl.program_id(1)

    # int64 indexing — production tensors exceed int32 range
    b = (pid_bh // oH).to(tl.int64)
    h_outer = (pid_bh % oH).to(tl.int64)

    rt = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    rdi = tl.arange(0, d_i)
    rt_mask = rt < T_t

    hq_base = b * (oH * T_t * H * d_i) + h_outer * (T_t * H * d_i)
    hk_base = b * (oH * T_s * d_i) + h_outer * (T_s * d_i)
    wo_base = b * (oH * T_t * H) + h_outer * (T_t * H)
    do_base = b * (oH * T_t * T_s) + h_outer * (T_t * T_s)

    for h in range(H):
        # Load Hq[..., rt, h, :] -> [BLOCK_T, d_i] bf16
        hq_ptrs = Hq_ptr + hq_base + rt[:, None] * (H * d_i) + h * d_i + rdi[None, :]
        Hq_h = tl.load(hq_ptrs, mask=rt_mask[:, None], other=0.0)

        # Load W_o[..., rt, h] -> [BLOCK_T] fp32
        wo_ptrs = W_o_ptr + wo_base + rt * H + h
        w_h = tl.load(wo_ptrs, mask=rt_mask, other=0.0).to(tl.float32)

        dHq_acc = tl.zeros((BLOCK_T, d_i), dtype=tl.float32)
        dWo_acc = tl.zeros((BLOCK_T,), dtype=tl.float32)

        for s_start in range(0, T_s, BLOCK_S):
            rs = s_start + tl.arange(0, BLOCK_S)
            rs_mask = rs < T_s

            # Load Hk[..., rs, :] -> [BLOCK_S, d_i] bf16
            hk_ptrs = Hk_ptr + hk_base + rs[:, None] * d_i + rdi[None, :]
            Hk_chunk = tl.load(hk_ptrs, mask=rs_mask[:, None], other=0.0)

            # Load dO[..., rt, rs] -> [BLOCK_T, BLOCK_S] (caller upcast to fp32)
            do_ptrs = dO_ptr + do_base + rt[:, None] * T_s + rs[None, :]
            dO_chunk = tl.load(
                do_ptrs,
                mask=rt_mask[:, None] & rs_mask[None, :],
                other=0.0,
            )

            # Recompute scores[BLOCK_T, BLOCK_S] = Hq_h @ Hk_chunk^T, in fp32
            scores = tl.dot(Hq_h, tl.trans(Hk_chunk))   # [BLOCK_T, BLOCK_S]
            relu_mask = scores > 0
            h_relu = tl.where(relu_mask, scores, 0.0)

            # dW_o accumulator: sum_s dO * H_relu
            dWo_acc += tl.sum(dO_chunk * h_relu, axis=1)

            # dscores = (dO * w_h) * relu_mask
            dH = dO_chunk * w_h[:, None]
            dscores = tl.where(relu_mask, dH, 0.0)

            # dHq_acc += dscores @ Hk_chunk  -> [BLOCK_T, d_i]
            dHq_acc += tl.dot(dscores.to(Hk_chunk.dtype), Hk_chunk)

        # Store dHq[..., rt, h, :]
        dhq_ptrs = dHq_ptr + hq_base + rt[:, None] * (H * d_i) + h * d_i + rdi[None, :]
        tl.store(
            dhq_ptrs,
            dHq_acc.to(dHq_ptr.dtype.element_ty),
            mask=rt_mask[:, None],
        )

        # Store dW_o[..., rt, h]
        dwo_ptrs = dWo_ptr + wo_base + rt * H + h
        tl.store(dwo_ptrs, dWo_acc.to(dWo_ptr.dtype.element_ty), mask=rt_mask)


_score_reduce_dHq_dWo_p = extend_core.Primitive("te_indexer_score_reduce_dHq_dWo")
_score_reduce_dHq_dWo_p.multiple_results = True


@_score_reduce_dHq_dWo_p.def_abstract_eval
def _score_reduce_dHq_dWo_abstract(Hq, Hk, W_o, dO):
    del Hk, dO
    return [
        core.ShapedArray(Hq.shape, Hq.dtype),   # dHq
        core.ShapedArray(W_o.shape, W_o.dtype), # dW_o
    ]


_score_reduce_dHq_dWo_p.def_impl(
    functools.partial(xla.apply_primitive, _score_reduce_dHq_dWo_p)
)


def _score_reduce_dHq_dWo_lowering(ctx, Hq, Hk, W_o, dO):
    Hq_aval = ctx.avals_in[0]
    Hk_aval = ctx.avals_in[1]
    B, oH, T_t, H, d_i = Hq_aval.shape
    T_s = Hk_aval.shape[2]
    BLOCK_T = 32 if T_t >= 32 else T_t
    BLOCK_S = 32 if T_s >= 32 else T_s

    return triton_call_lowering(
        ctx,
        _score_reduce_dHq_dWo_kernel,
        Hq, Hk, W_o, dO,
        grid=(triton.cdiv(T_t, BLOCK_T), B * oH),
        num_warps=4,
        num_stages=2,
        constexprs={
            "B": B, "oH": oH, "T_t": T_t, "T_s": T_s,
            "H": H, "d_i": d_i,
            "BLOCK_T": BLOCK_T, "BLOCK_S": BLOCK_S,
        },
    )


mlir.register_lowering(_score_reduce_dHq_dWo_p, _score_reduce_dHq_dWo_lowering, platform="rocm")
mlir.register_lowering(_score_reduce_dHq_dWo_p, _score_reduce_dHq_dWo_lowering, platform="cuda")


# --- Backward: dHk kernel -----------------------------------------------------


@triton.jit
def _score_reduce_dHk_kernel(
    Hq_ptr,    # (B, oH, T_t, H, d_i) bf16
    Hk_ptr,    # (B, oH, T_s, d_i) bf16
    W_o_ptr,   # (B, oH, T_t, H) bf16
    dO_ptr,    # (B, oH, T_t, T_s) fp32
    dHk_ptr,   # (B, oH, T_s, d_i) bf16  OUTPUT
    B: tl.constexpr,
    oH: tl.constexpr,
    T_t: tl.constexpr,
    T_s: tl.constexpr,
    H: tl.constexpr,
    d_i: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """Per-CTA: produces dHk[BLOCK_S, d_i].

    Outer-h loop / inner-t loop, accumulating dHk over all T and H.
    """
    pid_s = tl.program_id(0)
    pid_bh = tl.program_id(1)

    b = (pid_bh // oH).to(tl.int64)
    h_outer = (pid_bh % oH).to(tl.int64)

    rs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    rdi = tl.arange(0, d_i)
    rs_mask = rs < T_s

    hq_base = b * (oH * T_t * H * d_i) + h_outer * (T_t * H * d_i)
    hk_base = b * (oH * T_s * d_i) + h_outer * (T_s * d_i)
    wo_base = b * (oH * T_t * H) + h_outer * (T_t * H)
    do_base = b * (oH * T_t * T_s) + h_outer * (T_t * T_s)

    # Load Hk[..., rs, :] once -- needed for score recompute every iteration
    hk_ptrs = Hk_ptr + hk_base + rs[:, None] * d_i + rdi[None, :]
    Hk_tile = tl.load(hk_ptrs, mask=rs_mask[:, None], other=0.0)
    Hk_T = tl.trans(Hk_tile)  # [d_i, BLOCK_S]

    dHk_acc = tl.zeros((BLOCK_S, d_i), dtype=tl.float32)

    for h in range(H):
        for t_start in range(0, T_t, BLOCK_T):
            rt = t_start + tl.arange(0, BLOCK_T)
            rt_mask = rt < T_t

            # Load Hq[..., rt, h, :] -> [BLOCK_T, d_i]
            hq_ptrs = Hq_ptr + hq_base + rt[:, None] * (H * d_i) + h * d_i + rdi[None, :]
            Hq_h = tl.load(hq_ptrs, mask=rt_mask[:, None], other=0.0)

            # Load W_o[..., rt, h] -> [BLOCK_T]
            wo_ptrs = W_o_ptr + wo_base + rt * H + h
            w_h = tl.load(wo_ptrs, mask=rt_mask, other=0.0).to(tl.float32)

            # Load dO[..., rt, rs] -> [BLOCK_T, BLOCK_S]
            do_ptrs = dO_ptr + do_base + rt[:, None] * T_s + rs[None, :]
            dO_chunk = tl.load(
                do_ptrs,
                mask=rt_mask[:, None] & rs_mask[None, :],
                other=0.0,
            )

            # Recompute scores[BLOCK_T, BLOCK_S]
            scores = tl.dot(Hq_h, Hk_T)
            relu_mask = scores > 0

            # dscores = (dO * w_h[:, None]) * relu_mask
            dH = dO_chunk * w_h[:, None]
            dscores = tl.where(relu_mask, dH, 0.0)

            # dHk[s, i] += sum_t dscores[t, s] * Hq_h[t, i]
            # = (dscores^T @ Hq_h)[s, i]
            dHk_acc += tl.dot(tl.trans(dscores).to(Hq_h.dtype), Hq_h)

    dhk_ptrs = dHk_ptr + hk_base + rs[:, None] * d_i + rdi[None, :]
    tl.store(
        dhk_ptrs,
        dHk_acc.to(dHk_ptr.dtype.element_ty),
        mask=rs_mask[:, None],
    )


_score_reduce_dHk_p = extend_core.Primitive("te_indexer_score_reduce_dHk")
_score_reduce_dHk_p.multiple_results = True


@_score_reduce_dHk_p.def_abstract_eval
def _score_reduce_dHk_abstract(Hq, Hk, W_o, dO):
    del Hq, W_o, dO
    return [core.ShapedArray(Hk.shape, Hk.dtype)]


_score_reduce_dHk_p.def_impl(
    functools.partial(xla.apply_primitive, _score_reduce_dHk_p)
)


def _score_reduce_dHk_lowering(ctx, Hq, Hk, W_o, dO):
    Hq_aval = ctx.avals_in[0]
    Hk_aval = ctx.avals_in[1]
    B, oH, T_t, H, d_i = Hq_aval.shape
    T_s = Hk_aval.shape[2]
    BLOCK_T = 32 if T_t >= 32 else T_t
    BLOCK_S = 32 if T_s >= 32 else T_s

    return triton_call_lowering(
        ctx,
        _score_reduce_dHk_kernel,
        Hq, Hk, W_o, dO,
        grid=(triton.cdiv(T_s, BLOCK_S), B * oH),
        num_warps=4,
        num_stages=2,
        constexprs={
            "B": B, "oH": oH, "T_t": T_t, "T_s": T_s,
            "H": H, "d_i": d_i,
            "BLOCK_T": BLOCK_T, "BLOCK_S": BLOCK_S,
        },
    )


mlir.register_lowering(_score_reduce_dHk_p, _score_reduce_dHk_lowering, platform="rocm")
mlir.register_lowering(_score_reduce_dHk_p, _score_reduce_dHk_lowering, platform="cuda")


# --- Public score_reduce_triton with custom_vjp ------------------------------


@functools.partial(jax.custom_vjp, nondiff_argnums=(3,))
def _score_reduce_with_vjp(Hq, Hk, W_o, out_dtype):
    return _score_reduce_p.bind(Hq, Hk, W_o, out_dtype=out_dtype)[0]


def _score_reduce_fwd(Hq, Hk, W_o, out_dtype):
    out = _score_reduce_p.bind(Hq, Hk, W_o, out_dtype=out_dtype)[0]
    return out, (Hq, Hk, W_o)


def _score_reduce_bwd(out_dtype, residuals, dO):
    del out_dtype
    Hq, Hk, W_o = residuals
    # Backward kernels work in fp32; upcast dO once.
    dO_f32 = dO.astype(jnp.float32)
    dHq, dW_o = _score_reduce_dHq_dWo_p.bind(Hq, Hk, W_o, dO_f32)
    dHk, = _score_reduce_dHk_p.bind(Hq, Hk, W_o, dO_f32)
    return dHq, dHk, dW_o


_score_reduce_with_vjp.defvjp(_score_reduce_fwd, _score_reduce_bwd)


def score_reduce_triton(Hq, Hk, W_o, *, out_dtype=None):
    """Triton fused score-matmul + relu + per-(t, h) weighted H-reduction.

    Replaces the pattern:

        scores = relu(jnp.einsum("...thi,...si->...ths", Hq, Hk))   # never write
        O      = jnp.einsum("...ths,...th->...ts", scores, W_o)

    with a single kernel that holds the per-head score tile in registers,
    avoiding the (B, oH, T, H, S) HBM round-trip an einsum+XLA chain pays.

    Differentiable via two backward kernels (FlashAttention-style: residuals
    are just (Hq, Hk, W_o); the (T, H, S) score tensor is recomputed inside
    backward, never materialized).

    Args:
        Hq:  (B, oH, T_t, H, d_i)
        Hk:  (B, oH, T_s, d_i)
        W_o: (B, oH, T_t, H)
        out_dtype: defaults to Hq.dtype.

    Returns:
        O: (B, oH, T_t, T_s)
    """
    if Hq.ndim != 5:
        raise ValueError(
            f"Hq must be rank-5 (B, oH, T_t, H, d_i); got shape {Hq.shape}"
        )
    if Hk.ndim != 4:
        raise ValueError(
            f"Hk must be rank-4 (B, oH, T_s, d_i); got shape {Hk.shape}"
        )
    if W_o.ndim != 4:
        raise ValueError(
            f"W_o must be rank-4 (B, oH, T_t, H); got shape {W_o.shape}"
        )

    B, oH, T_t, H, d_i = Hq.shape
    Bk, oHk, T_s, d_i_k = Hk.shape
    Bw, oHw, T_t_w, H_w = W_o.shape
    if (Bk, oHk) != (B, oH):
        raise ValueError(
            f"(B, oH) mismatch: Hq has {(B, oH)}, Hk has {(Bk, oHk)}"
        )
    if d_i != d_i_k:
        raise ValueError(f"d_i mismatch: Hq has {d_i}, Hk has {d_i_k}")
    if (Bw, oHw, T_t_w, H_w) != (B, oH, T_t, H):
        raise ValueError(
            f"W_o shape {W_o.shape} does not match expected "
            f"(B={B}, oH={oH}, T_t={T_t}, H={H})"
        )

    if out_dtype is None:
        out_dtype = Hq.dtype

    return _score_reduce_with_vjp(Hq, Hk, W_o, jnp.dtype(out_dtype))


# --- Streaming top-k variant ----------------------------------------------------
#
# Same einsum-projected (Hq, Hk, W_o) inputs, but fuses top-k indices into the
# kernel: one CTA per (B, oH, T_t) query token, score row never materialized.
#
# Algorithm (mirrors TileLang dsa_sparse_finetune/indexer_topk_reducesum):
#   - Maintain a 2K-sized buffer of (score_bits, index) packed uint64
#   - Stream over T_s in BLOCK_S chunks; each chunk computes BLOCK_S new scores
#   - Place chunk into buffer[K:K+BLOCK_S], zero buffer[K+BLOCK_S:2K]
#   - tl.sort descending; top half is the running top-K
#   - After all chunks: buffer[:K] is the answer
#
# tl.sort returns values only, so we pack (score_bits << 32) | index into uint64.
# Post-ReLU scores are >= 0, so fp32 bit pattern is monotone in value.


@triton.jit
def _score_topk_kernel(
    Hq_ptr,        # (B, oH, T_t, H, d_i) bf16
    Hk_ptr,        # (B, oH, T_s, d_i) bf16
    W_o_ptr,       # (B, oH, T_t, H) bf16
    Topk_idx_ptr,  # (B, oH, T_t, K) int32 OUTPUT
    B: tl.constexpr,
    oH: tl.constexpr,
    T_t: tl.constexpr,
    T_s: tl.constexpr,
    H: tl.constexpr,
    d_i: tl.constexpr,
    K: tl.constexpr,
    S_PAD: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """Per-CTA: one query token's full top-K via streaming bitonic merge.

    Grid: (T_t, B * oH).
    """
    pid_t = tl.program_id(0)
    pid_bh = tl.program_id(1)
    # int64 indexing — Hq alone has B*oH*T*H*d_i = 4.3 B elements at T=S=4096.
    b = (pid_bh // oH).to(tl.int64)
    h_outer = (pid_bh % oH).to(tl.int64)
    pid_t_64 = pid_t.to(tl.int64)

    rh = tl.arange(0, H)
    rdi = tl.arange(0, d_i)

    # Pre-load Hq[b, h_outer, pid_t, :, :] -> [H, d_i] once
    hq_base = b * (oH * T_t * H * d_i) + h_outer * (T_t * H * d_i) + pid_t_64 * (H * d_i)
    Hq_token = tl.load(Hq_ptr + hq_base + rh[:, None] * d_i + rdi[None, :])

    # Pre-load w_o[b, h_outer, pid_t, :] -> [H] once
    wo_base = b * (oH * T_t * H) + h_outer * (T_t * H) + pid_t_64 * H
    w_o = tl.load(W_o_ptr + wo_base + rh).to(tl.float32)

    hk_base = b * (oH * T_s * d_i) + h_outer * (T_s * d_i)

    TOP_BUF: tl.constexpr = 2 * K
    INNER: tl.constexpr = K // BLOCK_S        # chunks per sort
    N_OUTER: tl.constexpr = S_PAD // K        # number of sorts per CTA
    top_packed = tl.zeros((TOP_BUF,), dtype=tl.uint64)

    rs_buf = tl.arange(0, TOP_BUF)
    rs_chunk = tl.arange(0, BLOCK_S)
    Hq_T = tl.trans(Hq_token)  # [d_i, H]

    # Two-level loop: fill the bottom K slots over INNER chunks, then sort.
    # Net: N_OUTER sorts instead of S_PAD/BLOCK_S sorts (4x fewer at production
    # shape). The previous round's "losers" (bottom-K after each sort) are
    # naturally overwritten by the next INNER chunks; correctness holds because
    # those losers are by definition below the running top-K threshold.
    for o in tl.static_range(N_OUTER):
        for i in tl.static_range(INNER):
            c = o * INNER + i
            s_start = c * BLOCK_S
            rs = s_start + rs_chunk
            rs_mask = rs < T_s

            # Load Hk_chunk[BLOCK_S, d_i]
            hk_ptrs = Hk_ptr + hk_base + rs[:, None] * d_i + rdi[None, :]
            Hk_chunk = tl.load(hk_ptrs, mask=rs_mask[:, None], other=0.0)

            # Score matmul: [BLOCK_S, d_i] @ [d_i, H] -> [BLOCK_S, H]
            logits = tl.dot(Hk_chunk, Hq_T)
            logits = tl.maximum(logits, 0.0)

            # Weighted H-reduce: sum(logits * w_o[None, :], axis=1) -> [BLOCK_S]
            # Note: w_o can be negative, so chunk_scores can be negative even after ReLU.
            chunk_scores = tl.sum(logits * w_o[None, :], axis=1)

            # Convert fp32 to "sortable uint32" so uint comparison matches fp32
            # comparison across the full sign range:
            #   positive: flip sign bit
            #   negative: flip all bits
            # See https://stereopsis.com/radix.html
            bits = chunk_scores.to(tl.uint32, bitcast=True)
            sign = bits >> 31
            flip_mask = (0 - sign.to(tl.int32)).to(tl.uint32) | 0x80000000
            sortable = bits ^ flip_mask
            # OOR positions get sortable=0 (smallest possible, sorts to bottom)
            sortable = tl.where(rs_mask, sortable, 0)

            # Pack (sortable_score_bits << 32) | index into uint64
            chunk_packed = (sortable.to(tl.uint64) << 32) | rs.to(tl.uint64)

            # Scatter chunk_packed into top_packed[K + i*BLOCK_S : K + (i+1)*BLOCK_S]
            chunk_offset = K + i * BLOCK_S
            in_chunk_slot = (rs_buf >= chunk_offset) & (rs_buf < chunk_offset + BLOCK_S)
            chunk_gather_idx = tl.where(in_chunk_slot, rs_buf - chunk_offset, 0).to(tl.int32)
            gathered = tl.gather(chunk_packed, chunk_gather_idx, axis=0)
            top_packed = tl.where(in_chunk_slot, gathered, top_packed)

        # All INNER chunks placed -> sort once
        top_packed = tl.sort(top_packed, descending=True)

    # Extract top K indices: gather positions [0, K) from the sorted buffer,
    # take low 32 bits.
    rk = tl.arange(0, K)
    top_k_packed = tl.gather(top_packed, rk, axis=0)
    top_k_idx = (top_k_packed & 0xFFFFFFFF).to(tl.int32)

    out_base = b * (oH * T_t * K) + h_outer * (T_t * K) + pid_t_64 * K
    tl.store(Topk_idx_ptr + out_base + rk, top_k_idx)


_score_topk_p = extend_core.Primitive("te_indexer_score_topk_triton")
_score_topk_p.multiple_results = True


def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


@_score_topk_p.def_abstract_eval
def _score_topk_abstract(Hq, Hk, W_o, *, k):
    del Hk, W_o
    B, oH, T_t, _H, _d_i = Hq.shape
    return [core.ShapedArray((B, oH, T_t, k), jnp.int32)]


_score_topk_p.def_impl(functools.partial(xla.apply_primitive, _score_topk_p))


def _score_topk_lowering(ctx, Hq, Hk, W_o, *, k):
    Hq_aval = ctx.avals_in[0]
    Hk_aval = ctx.avals_in[1]
    B, oH, T_t, H, d_i = Hq_aval.shape
    T_s = Hk_aval.shape[2]
    S_PAD = _next_pow2(T_s)
    # BLOCK_S must be <= K (so chunk fits in TOP_BUF[K:K+BLOCK_S]) and
    # divide S_PAD evenly. Cap at 128 so the [BLOCK_S, H] fp32 logits
    # intermediate stays in registers.
    BLOCK_S = min(128, k, S_PAD)

    return triton_call_lowering(
        ctx,
        _score_topk_kernel,
        Hq, Hk, W_o,
        grid=(T_t, B * oH),
        num_warps=4,
        num_stages=2,
        constexprs={
            "B": B, "oH": oH, "T_t": T_t, "T_s": T_s,
            "H": H, "d_i": d_i,
            "K": k, "S_PAD": S_PAD,
            "BLOCK_S": BLOCK_S,
        },
    )


mlir.register_lowering(_score_topk_p, _score_topk_lowering, platform="rocm")
mlir.register_lowering(_score_topk_p, _score_topk_lowering, platform="cuda")


def score_topk_triton(Hq, Hk, W_o, *, k):
    """Fused score-relu-reduce + streaming top-k.

    Computes the same scores as ``score_reduce_triton`` but never materializes the
    (B, oH, T_t, T_s) score matrix — instead, returns the top-k indices into the
    T_s axis directly.

    Args:
        Hq:  (B, oH, T_t, H, d_i)
        Hk:  (B, oH, T_s, d_i)
        W_o: (B, oH, T_t, H)
        k:   number of top scores to return per (b, oH, T_t) row. Must be a
             power of 2 and <= T_s.

    Returns:
        Topk_idx: (B, oH, T_t, k) int32 — top-k indices into T_s axis, in
        descending score order.

    Notes:
        Streaming: maintains a 2K candidate buffer and bitonic-sorts on each
        chunk. For k >> S/8 (e.g., k=S/2), this is algorithmically slower than a
        single full-row sort but matches the TileLang reference structure and
        generalizes to large S without per-CTA registers scaling with S.
    """
    if Hq.ndim != 5:
        raise ValueError(f"Hq must be rank-5; got shape {Hq.shape}")
    if Hk.ndim != 4:
        raise ValueError(f"Hk must be rank-4; got shape {Hk.shape}")
    if W_o.ndim != 4:
        raise ValueError(f"W_o must be rank-4; got shape {W_o.shape}")

    B, oH, T_t, H, d_i = Hq.shape
    Bk, oHk, T_s, d_i_k = Hk.shape
    Bw, oHw, T_t_w, H_w = W_o.shape
    if (Bk, oHk) != (B, oH):
        raise ValueError(f"(B, oH) mismatch: Hq has {(B, oH)}, Hk has {(Bk, oHk)}")
    if d_i != d_i_k:
        raise ValueError(f"d_i mismatch: Hq has {d_i}, Hk has {d_i_k}")
    if (Bw, oHw, T_t_w, H_w) != (B, oH, T_t, H):
        raise ValueError(f"W_o shape {W_o.shape} != expected (B, oH, T_t, H)")

    if k <= 0 or (k & (k - 1)) != 0:
        raise ValueError(f"k must be a positive power of 2; got {k}")
    if k > T_s:
        raise ValueError(f"k={k} must be <= T_s={T_s}")

    return _score_topk_p.bind(Hq, Hk, W_o, k=k)[0]
