# Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
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


# --- Chunked score-tile kernel for hybrid bwd --------------------------------
#
# Produces dscores_chunk[B, oH, T, H_CHUNK, T_s] and dW_o_chunk[B, oH, T, H_CHUNK]
# for ONE h-chunk. Caller loops over H/H_CHUNK chunks and feeds dscores_chunk
# to hipBLASLt einsums for dHq/dHk reductions. Bounds peak materialization to
# H/H_CHUNK fraction of the full (B, oH, T, H, T_s) score tensor.
#
# Fuses score recompute + relu + mask + dO*W_o broadcast in registers --
# nothing of size (B, oH, T, H, T_s) ever lands in HBM at full size. dW_o is
# reduced inline (sum_s of h_relu * dO) so h_relu also never materializes.


_HBWD_BLOCK_T = 64
_HBWD_BLOCK_S = 64


@triton.jit
def _score_dscores_chunk_kernel(
    Hq_chunk_ptr,        # input  (B, oH, T,   H_CHUNK, d_i) bf16
    Hk_ptr,              # input  (B, oH, T_s, d_i)         bf16
    W_o_chunk_ptr,       # input  (B, oH, T,   H_CHUNK)     bf16
    dO_ptr,              # input  (B, oH, T,   T_s)         fp32
    dscores_chunk_ptr,   # output (B, oH, T,   H_CHUNK, T_s) bf16
    dWo_chunk_ptr,       # output (B, oH, T,   H_CHUNK)     bf16
    B: tl.constexpr,
    oH: tl.constexpr,
    T: tl.constexpr,
    T_s: tl.constexpr,
    H_CHUNK: tl.constexpr,
    d_i: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """One CTA handles (T_tile, h_in) for one (b, h_outer). Loops over s_chunks.

    Each CTA writes its T_tile rows of (dscores_chunk[..., h_in, :],
    dW_o_chunk[..., h_in]). dW_o is reduced in registers (sum over s) so
    h_relu never lands in HBM -- we compute it on-the-fly and consume it.
    """
    pid_t = tl.program_id(0)
    pid_h_bh = tl.program_id(1)
    h_in = pid_h_bh % H_CHUNK
    pid_bh = pid_h_bh // H_CHUNK
    b = (pid_bh // oH).to(tl.int64)
    h_outer = (pid_bh % oH).to(tl.int64)
    h_in_64 = h_in.to(tl.int64)

    rt = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    rdi = tl.arange(0, d_i)
    rt_mask = rt < T

    hq_base = b * (oH * T * H_CHUNK * d_i) + h_outer * (T * H_CHUNK * d_i)
    hk_base = b * (oH * T_s * d_i) + h_outer * (T_s * d_i)
    wo_base = b * (oH * T * H_CHUNK) + h_outer * (T * H_CHUNK)
    do_base = b * (oH * T * T_s) + h_outer * (T * T_s)
    ds_base = b * (oH * T * H_CHUNK * T_s) + h_outer * (T * H_CHUNK * T_s)

    # Load Hq[..., t_tile, h_in, :] -> [BLOCK_T, d_i] once per CTA
    hq_ptrs = (Hq_chunk_ptr + hq_base
               + rt[:, None] * (H_CHUNK * d_i)
               + h_in_64 * d_i
               + rdi[None, :])
    Hq_h = tl.load(hq_ptrs, mask=rt_mask[:, None], other=0.0)

    # Load W_o[..., t_tile, h_in] -> [BLOCK_T] once per CTA
    wo_ptrs = W_o_chunk_ptr + wo_base + rt * H_CHUNK + h_in_64
    w_h = tl.load(wo_ptrs, mask=rt_mask, other=0.0).to(tl.float32)

    # dW_o accumulator: sum_s (h_relu * dO) -- reduced in regs
    dWo_acc = tl.zeros((BLOCK_T,), dtype=tl.float32)

    for s_start in range(0, T_s, BLOCK_S):
        rs = s_start + tl.arange(0, BLOCK_S)
        rs_mask = rs < T_s

        # Load Hk[..., s_chunk, :] and dO[..., t_tile, s_chunk]
        hk_ptrs = Hk_ptr + hk_base + rs[:, None] * d_i + rdi[None, :]
        Hk_chunk = tl.load(hk_ptrs, mask=rs_mask[:, None], other=0.0)

        do_ptrs = dO_ptr + do_base + rt[:, None] * T_s + rs[None, :]
        dO_chunk = tl.load(
            do_ptrs,
            mask=rt_mask[:, None] & rs_mask[None, :],
            other=0.0,
        )

        # scores tile in registers (never lands in HBM at full size)
        scores = tl.dot(Hq_h, tl.trans(Hk_chunk))
        relu_mask = scores > 0
        h_relu = tl.where(relu_mask, scores, 0.0)

        # dW_o contribution: sum_s (h_relu * dO)
        dWo_acc += tl.sum(h_relu * dO_chunk, axis=1)

        # dscores tile = relu_mask * (dO * W_o)
        dscores = tl.where(relu_mask, dO_chunk * w_h[:, None], 0.0)

        # Store dscores tile to HBM (bf16). Total dscores_chunk size is
        # H_CHUNK x smaller than the full (B,oH,T,H,T_s) tensor.
        ds_ptrs = (dscores_chunk_ptr + ds_base
                   + rt[:, None] * (H_CHUNK * T_s)
                   + h_in_64 * T_s
                   + rs[None, :])
        tl.store(
            ds_ptrs,
            dscores.to(dscores_chunk_ptr.dtype.element_ty),
            mask=rt_mask[:, None] & rs_mask[None, :],
        )

    # Store dW_o[..., t_tile, h_in]
    dwo_out_ptrs = dWo_chunk_ptr + wo_base + rt * H_CHUNK + h_in_64
    tl.store(
        dwo_out_ptrs,
        dWo_acc.to(dWo_chunk_ptr.dtype.element_ty),
        mask=rt_mask,
    )


_score_dscores_chunk_p = extend_core.Primitive("te_indexer_score_dscores_chunk")
_score_dscores_chunk_p.multiple_results = True


@_score_dscores_chunk_p.def_abstract_eval
def _score_dscores_chunk_abstract(Hq_chunk, Hk, W_o_chunk, dO):
    del Hk, W_o_chunk
    B, oH, T, H_CHUNK, _ = Hq_chunk.shape
    T_s = dO.shape[-1]
    return [
        core.ShapedArray((B, oH, T, H_CHUNK, T_s), Hq_chunk.dtype),  # dscores
        core.ShapedArray((B, oH, T, H_CHUNK), Hq_chunk.dtype),       # dW_o
    ]


_score_dscores_chunk_p.def_impl(
    functools.partial(xla.apply_primitive, _score_dscores_chunk_p)
)


def _score_dscores_chunk_lowering(ctx, Hq_chunk, Hk, W_o_chunk, dO):
    Hq_aval = ctx.avals_in[0]
    dO_aval = ctx.avals_in[3]
    B, oH, T, H_CHUNK, d_i = Hq_aval.shape
    T_s = dO_aval.shape[-1]
    BLOCK_T = _HBWD_BLOCK_T if T >= _HBWD_BLOCK_T else T
    BLOCK_S = _HBWD_BLOCK_S if T_s >= _HBWD_BLOCK_S else T_s
    n_t_tiles = (T + BLOCK_T - 1) // BLOCK_T

    return triton_call_lowering(
        ctx,
        _score_dscores_chunk_kernel,
        Hq_chunk, Hk, W_o_chunk, dO,
        grid=(n_t_tiles, B * oH * H_CHUNK),
        num_warps=4,
        num_stages=2,
        constexprs={
            "B": B, "oH": oH, "T": T, "T_s": T_s,
            "H_CHUNK": H_CHUNK, "d_i": d_i,
            "BLOCK_T": BLOCK_T, "BLOCK_S": BLOCK_S,
        },
    )


mlir.register_lowering(_score_dscores_chunk_p, _score_dscores_chunk_lowering, platform="rocm")
mlir.register_lowering(_score_dscores_chunk_p, _score_dscores_chunk_lowering, platform="cuda")


# --- Public score_reduce_triton with custom_vjp ------------------------------


@functools.partial(jax.custom_vjp, nondiff_argnums=(3,))
def _score_reduce_with_vjp(Hq, Hk, W_o, out_dtype):
    return _score_reduce_p.bind(Hq, Hk, W_o, out_dtype=out_dtype)[0]


def _score_reduce_fwd(Hq, Hk, W_o, out_dtype):
    out = _score_reduce_p.bind(Hq, Hk, W_o, out_dtype=out_dtype)[0]
    return out, (Hq, Hk, W_o)


_BWD_H_CHUNK = 8  # peak (B, oH, T, H_CHUNK, T_s) tile -- bounds materialization


def _score_reduce_bwd(out_dtype, residuals, dO):
    del out_dtype
    Hq, Hk, W_o = residuals
    B, oH, T, H, d_i = Hq.shape

    # Hybrid scheme with bounded materialization:
    #   For each h-chunk of size H_CHUNK (driven by lax.scan, NOT Python
    #   unroll, so intermediates are freed between iterations):
    #     1. Triton kernel fuses (score recompute + relu + mask + dO*W_o
    #        broadcast) and writes dscores_chunk[B,oH,T,H_CHUNK,T_s] to HBM.
    #        h_relu is consumed in-register to also produce dWo_chunk
    #        without ever materializing the (B,oH,T,H,T_s) h_relu tensor.
    #     2. hipBLASLt einsums on dscores_chunk give dHq_chunk and a partial
    #        dHk contribution.
    # Peak HBM intermediate stays at H_CHUNK/H fraction of the full score.
    #
    # The fully-fused Triton bwd variants (v2/v3/v4) remain in this file for
    # reference -- they don't materialize the score tensor either but are
    # slower than the hipBLASLt-based reductions used here (~2x at 4096^2).
    if H % _BWD_H_CHUNK == 0:
        H_CHUNK = _BWD_H_CHUNK
    else:
        H_CHUNK = 1
        for c in (4, 2):
            if H % c == 0:
                H_CHUNK = c
                break
    n_chunks = H // H_CHUNK

    Hq_r = Hq.reshape(B, oH, T, n_chunks, H_CHUNK, d_i)
    Wo_r = W_o.reshape(B, oH, T, n_chunks, H_CHUNK)
    # Move chunk axis to leading for scan over axis 0.
    Hq_s = jnp.moveaxis(Hq_r, -3, 0)   # (n_chunks, B, oH, T, H_CHUNK, d_i)
    Wo_s = jnp.moveaxis(Wo_r, -2, 0)   # (n_chunks, B, oH, T, H_CHUNK)

    def step(dHk_acc, chunk):
        Hq_c, Wo_c = chunk
        # Triton: dscores_chunk + dWo_chunk; no full (B,oH,T,H,T_s) tensor
        # ever exists in HBM.
        dscores_c, dWo_c = _score_dscores_chunk_p.bind(Hq_c, Hk, Wo_c, dO)
        dHq_c = jnp.einsum("...ths,...si->...thi", dscores_c, Hk)
        dHk_c = jnp.einsum("...ths,...thi->...si", dscores_c, Hq_c)
        new_dHk_acc = dHk_acc + dHk_c.astype(jnp.float32)
        return new_dHk_acc, (dHq_c, dWo_c)

    init = jnp.zeros(Hk.shape, dtype=jnp.float32)
    dHk_acc, (dHq_chunks, dWo_chunks) = jax.lax.scan(
        step, init, (Hq_s, Wo_s),
    )
    # dHq_chunks: (n_chunks, B, oH, T, H_CHUNK, d_i)
    # dWo_chunks: (n_chunks, B, oH, T, H_CHUNK)
    dHq = jnp.moveaxis(dHq_chunks, 0, -3).reshape(B, oH, T, H, d_i)
    dWo = jnp.moveaxis(dWo_chunks, 0, -2).reshape(B, oH, T, H)
    dHk = dHk_acc.astype(Hk.dtype)

    return dHq.astype(Hq.dtype), dHk, dWo.astype(W_o.dtype)


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


# Autotune sweep for _score_topk_kernel.
#
# BLOCK_T: number of query tokens per CTA. BLOCK_T>1 amortizes the Hk_chunk
# load across BLOCK_T queries — the single biggest lever at large T_s. At
# BLOCK_T=1 (original), each CTA reloads all of Hk for its (b, oH) slab,
# causing L2 thrash. BLOCK_T=2 halves Hk HBM traffic; BLOCK_T=4 quarters it,
# but grows per-CTA register pressure (Hq_token, top_packed, logits all
# scale with BLOCK_T).
#
# BLOCK_S knobs the inner-chunk size; bigger BLOCK_S = better matmul
# arithmetic intensity, but bigger per-CTA transient footprint
# (logits[BLOCK_S, BLOCK_T*H] fp32 + Hk_chunk[BLOCK_S, d_i] bf16).
#
# Constraint: BLOCK_S must divide K (so INNER = K // BLOCK_S is an integer
# >= 1). Configs whose BLOCK_S exceeds K or doesn't divide K are filtered
# out at lowering time — otherwise jaxlib's autotuner would time them as
# zero-work (fast) and pick a bogus winner that returns all-zero indices.
_SCORE_TOPK_CONFIGS = [
    triton.Config({"BLOCK_S": bs, "BLOCK_T": bt}, num_warps=nw, num_stages=ns)
    for bt in (1, 2)
    for bs in (32, 64, 128, 256)
    for nw in (4, 8)
    for ns in (1, 2)
] + [
    # BLOCK_T=4 only at smaller BLOCK_S — at BLOCK_S=256 the logits
    # intermediate [256, 4*H=256] fp32 = 256 KB overflows reliably.
    triton.Config({"BLOCK_S": bs, "BLOCK_T": 4}, num_warps=nw, num_stages=ns)
    for bs in (32, 64, 128)
    for nw in (4, 8)
    for ns in (1, 2)
]


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
    BLOCK_T: tl.constexpr,
):
    """Per-CTA: BLOCK_T consecutive query tokens, all sharing Hk loads.

    Grid: (cdiv(T_t, BLOCK_T), B * oH). Each CTA does:
      - Pre-load Hq[..., rt, :, :] for BLOCK_T contiguous query tokens
      - For each S chunk: load Hk_chunk ONCE, do one [BLOCK_S, d_i] @
        [d_i, BLOCK_T*H] matmul, weighted-H-reduce per T
      - Maintain a single 1D top buffer of size BLOCK_T*2K, with T encoded
        in the top 8 bits of each packed entry. After global sort desc,
        per-T entries stay grouped together so per-T top-K can be sliced
        from fixed offsets.

    Note on layout (1D vs 2D top buffer):
      A 2D [BLOCK_T, 2K] top buffer with per-row sort is the natural
      design, but `tl.gather + tl.sort(dim=1)` on uint64 2D tensors trips
      `TritonGPUOptimizeThreadLocality` on the AMD backend (gfx950, Triton
      3.4.0). The 1D-with-encoded-T workaround sidesteps this — it pays a
      ~1.5x sort-cost penalty (one sort of BLOCK_T*2K vs BLOCK_T sorts of
      2K) for BLOCK_T=2, but unblocks Hk-load amortization across queries.
    """
    pid_t = tl.program_id(0)
    pid_bh = tl.program_id(1)
    # int64 indexing — Hq has B*oH*T*H*d_i = 4.3 B elements at T=S=4096.
    b = (pid_bh // oH).to(tl.int64)
    h_outer = (pid_bh % oH).to(tl.int64)

    rh = tl.arange(0, H)
    rdi = tl.arange(0, d_i)
    rs_chunk = tl.arange(0, BLOCK_S)
    rk = tl.arange(0, K)
    rt_local = tl.arange(0, BLOCK_T)

    rt = pid_t * BLOCK_T + rt_local
    rt_64 = rt.to(tl.int64)
    rt_mask = rt < T_t

    # Load Hq[b, h_outer, rt, :, :] -> [BLOCK_T, H, d_i].
    hq_base = b * (oH * T_t * H * d_i) + h_outer * (T_t * H * d_i)
    Hq_token = tl.load(
        Hq_ptr + hq_base
        + rt_64[:, None, None] * (H * d_i)
        + rh[None, :, None] * d_i
        + rdi[None, None, :],
        mask=rt_mask[:, None, None],
        other=0.0,
    )

    # Load w_o[b, h_outer, rt, :] -> [BLOCK_T, H]
    wo_base = b * (oH * T_t * H) + h_outer * (T_t * H)
    w_o = tl.load(
        W_o_ptr + wo_base + rt_64[:, None] * H + rh[None, :],
        mask=rt_mask[:, None],
        other=0.0,
    ).to(tl.float32)

    # Flatten Hq for one big matmul per Hk_chunk: [BLOCK_T * H, d_i] -> trans
    Hq_flat = tl.reshape(Hq_token, (BLOCK_T * H, d_i))
    Hq_T = tl.trans(Hq_flat)  # [d_i, BLOCK_T * H]
    w_o_flat = tl.reshape(w_o, (BLOCK_T * H,))

    hk_base = b * (oH * T_s * d_i) + h_outer * (T_s * d_i)

    TOP_BUF: tl.constexpr = 2 * K
    INNER: tl.constexpr = K // BLOCK_S        # chunks per sort
    N_OUTER: tl.constexpr = S_PAD // K        # number of sorts per CTA
    BIG_BUF: tl.constexpr = BLOCK_T * TOP_BUF

    # Initialize 1D top buffer with t-encoding pre-applied so per-T regions
    # stay grouped after global sort. Each slot at position rb gets:
    #   t_pos = rb // TOP_BUF      -> which T this slot belongs to
    #   t_enc = BLOCK_T - t_pos    -> 1..BLOCK_T (never 0 → never collides with
    #                                  reserved init pattern)
    #   packed = (t_enc << 56) | 0  -> score=0 (sortable=0), index=0
    # Real candidates also get tagged with their t_enc; after global sort
    # desc, all entries with t_enc=BLOCK_T (i.e. t=0) come first, then
    # t_enc=BLOCK_T-1, etc. Within each t group, ordered by score then index.
    rb = tl.arange(0, BIG_BUF)
    rb_t = rb // TOP_BUF           # [BIG_BUF] in [0, BLOCK_T)
    rb_pos = rb % TOP_BUF          # [BIG_BUF] in [0, TOP_BUF)
    t_enc_per_slot = (BLOCK_T - rb_t).to(tl.uint64)
    top_packed = t_enc_per_slot << 56

    # Pre-compute the per-slot (t, pos)-to-flat-chunk-index map used in
    # scatter: for each rb, identify the (t, j) in chunk_packed_flat to pull
    # from. j depends on `chunk_offset` (varies per inner iter), so the
    # gather index is recomputed each iter.

    for o in tl.static_range(N_OUTER):
        for i in tl.static_range(INNER):
            c = o * INNER + i
            s_start = c * BLOCK_S
            rs = s_start + rs_chunk     # [BLOCK_S]
            rs_mask = rs < T_s

            # Load Hk_chunk[BLOCK_S, d_i] ONCE — shared across BLOCK_T queries.
            hk_ptrs = Hk_ptr + hk_base + rs[:, None] * d_i + rdi[None, :]
            Hk_chunk = tl.load(hk_ptrs, mask=rs_mask[:, None], other=0.0)

            # One big matmul: [BLOCK_S, d_i] @ [d_i, BLOCK_T*H] -> [BLOCK_S, BLOCK_T*H]
            logits = tl.dot(Hk_chunk, Hq_T)
            logits = tl.maximum(logits, 0.0)

            # Weighted reduce over H per (s, t):
            #   chunk_scores[s, t] = sum_h logits[s, t*H + h] * w_o[t, h]
            weighted = logits * w_o_flat[None, :]
            weighted_3d = tl.reshape(weighted, (BLOCK_S, BLOCK_T, H))
            chunk_scores = tl.sum(weighted_3d, axis=2)  # [BLOCK_S, BLOCK_T]
            chunk_scores_T = tl.trans(chunk_scores)      # [BLOCK_T, BLOCK_S]

            # Radix-flip: fp32 bit pattern -> sortable uint32 across full sign
            # range (positives: flip sign bit; negatives: flip all bits).
            # See https://stereopsis.com/radix.html
            bits = chunk_scores_T.to(tl.uint32, bitcast=True)
            sign = bits >> 31
            flip_mask = (0 - sign.to(tl.int32)).to(tl.uint32) | 0x80000000
            sortable = bits ^ flip_mask
            sortable = tl.where(rs_mask[None, :], sortable, 0)

            # Pack: (t_enc<<56) | (sortable<<24) | (index in low 24 bits).
            # 24-bit index supports T_s up to 16M, far above our regime.
            t_enc_chunk = (BLOCK_T - rt_local).to(tl.uint64)  # [BLOCK_T]
            rs_2d = tl.broadcast_to(rs[None, :], (BLOCK_T, BLOCK_S))
            chunk_packed_2d = (
                (t_enc_chunk[:, None] << 56)
                | (sortable.to(tl.uint64) << 24)
                | rs_2d.to(tl.uint64)
            )  # [BLOCK_T, BLOCK_S]
            # Flatten to 1D for the scatter (1D gather + 1D sort sidesteps
            # the AMD-backend bug with 2D gather+sort combos).
            chunk_packed_flat = tl.reshape(chunk_packed_2d, (BLOCK_T * BLOCK_S,))

            # Scatter into top_packed[t*TOP_BUF + K+i*BLOCK_S : ...] for each t.
            # For each rb in [0, BIG_BUF):
            #   t = rb // TOP_BUF
            #   pos = rb % TOP_BUF
            #   in_slot = (pos >= K + i*BLOCK_S) & (pos < K + (i+1)*BLOCK_S)
            #   flat_idx = t * BLOCK_S + (pos - (K + i*BLOCK_S))
            chunk_offset = K + i * BLOCK_S
            in_slot = (rb_pos >= chunk_offset) & (rb_pos < chunk_offset + BLOCK_S)
            j = rb_pos - chunk_offset
            flat_idx = tl.where(in_slot, rb_t * BLOCK_S + j, 0).to(tl.int32)
            gathered = tl.gather(chunk_packed_flat, flat_idx, axis=0)
            top_packed = tl.where(in_slot, gathered, top_packed)

        # 1D sort of the entire buffer. Per-T regions stay grouped via t_enc.
        top_packed = tl.sort(top_packed, descending=True)

    # Extract per-T top K. After sort desc, t=0's top K is at positions
    # [0, K), t=1's at [TOP_BUF, TOP_BUF+K), etc. — i.e. base = t*TOP_BUF.
    out_idx = rt_local[:, None] * TOP_BUF + rk[None, :]  # [BLOCK_T, K]
    out_idx_flat = tl.reshape(out_idx, (BLOCK_T * K,)).to(tl.int32)
    top_k_packed_flat = tl.gather(top_packed, out_idx_flat, axis=0)
    top_k_packed = tl.reshape(top_k_packed_flat, (BLOCK_T, K))
    # Strip the t_enc and sortable bits, keep low 24 bits (index).
    top_k_idx = (top_k_packed & 0xFFFFFF).to(tl.int32)

    out_base = b * (oH * T_t * K) + h_outer * (T_t * K)
    out_ptrs = Topk_idx_ptr + out_base + rt_64[:, None] * K + rk[None, :]
    tl.store(out_ptrs, top_k_idx, mask=rt_mask[:, None])


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

    # Build a K-filtered autotuner around the plain JIT kernel. We do this at
    # lowering time (rather than decorating the kernel at definition) because
    # configs with BLOCK_S > k or BLOCK_S that doesn't divide k would compile
    # to a kernel where INNER = k // BLOCK_S = 0 — i.e. a no-op that's fastest
    # in the autotune timing race. Filtering ensures the runtime picker only
    # sees configs that actually do the work.
    #
    # Also filter BLOCK_T configs that don't evenly divide T_t — we mask the
    # tail but unnecessary padding hurts L1/L2 efficiency.
    valid_configs = [
        c for c in _SCORE_TOPK_CONFIGS
        if c.kwargs["BLOCK_S"] <= k
        and k % c.kwargs["BLOCK_S"] == 0
        and T_t % c.kwargs["BLOCK_T"] == 0
    ]
    if not valid_configs:
        raise ValueError(
            f"No valid BLOCK_S/BLOCK_T config for k={k}, T_t={T_t}"
        )

    autotuned_kernel = triton.autotune(
        configs=valid_configs,
        key=["H", "d_i", "T_s", "K"],
    )(_score_topk_kernel)

    def grid_fn(merged_kwargs):
        bt = merged_kwargs.get("BLOCK_T", 1)
        return (triton.cdiv(T_t, bt), B * oH)

    return triton_call_lowering(
        ctx,
        autotuned_kernel,
        Hq, Hk, W_o,
        grid=grid_fn,
        constexprs={
            "B": B, "oH": oH, "T_t": T_t, "T_s": T_s,
            "H": H, "d_i": d_i,
            "K": k, "S_PAD": S_PAD,
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
