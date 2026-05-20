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
        for c in (8, 4, 2):
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
