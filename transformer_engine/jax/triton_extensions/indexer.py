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

FP8 score matmul
----------------
These ops never quantize. ``Hq`` / ``Hk`` are consumed in whatever precision
they arrive in -- e4m3 or bf16 -- alongside the per-row scales ``Sq`` / ``Ks``,
so a caller holding already-quantized index-q / index-k can hand them straight
in without a dequantize/requantize round trip. ``tl.dot`` specializes on the
operand dtype, so one kernel serves both precisions; omitting the scales means
"already unit-scaled" and passes all-ones.

The expected quantization (see :func:`quantize_e4m3`, which callers may use but
are not required to) is per-row along the contracted ``d_i`` axis: ``Sq`` per
(b, oH, t, h) and ``Ks`` per (b, oH, s), i.e. the row scale of each matmul
operand. Because ReLU is positive-homogeneous and both scales are strictly
positive, dequantization commutes with it and factors out of the H-reduction:

    O[t, s] = sum_h relu(sq[t,h] * ks[s] * (q_hat . k_hat)) * W_o[t,h]
            = ks[s] * sum_h relu(q_hat . k_hat) * (W_o[t,h] * sq[t,h])

So ``Sq`` folds into the per-head weight and ``Ks`` is a single per-column
post-multiply -- no separate dequantization pass over the operands.

Gradients follow the operand dtype: with e4m3 ``Hq`` / ``Hk`` the cotangents
come back as e4m3 too. They are the chain-rule cotangents w.r.t. the *quantized*
operands (``dHq_true * sq``), which renormalizes them into the operand's own
range rather than leaving a raw unscaled fp8 gradient. ``Sq`` / ``Ks`` are
quantization metadata and are treated as constants (zero cotangent).
"""

import functools
import os

import jax
import jax.numpy as jnp
import triton
import triton.language as tl

from jax import core
from jax.extend import core as extend_core
from jax.interpreters import mlir, xla

from transformer_engine.jax.util import get_jnp_float8_e4m3_type

from .utils import triton_call_lowering


def _autotune_disabled():
    """True when ``NVTE_INDEXER_DISABLE_AUTOTUNE=1``.

    When set, each kernel's lowering collapses its autotune sweep to the first
    (still prune-valid) config, so no time is spent compiling and benchmarking
    every candidate. Intended for the test suite — a full sweep at large k/T_s
    costs many minutes and only picks the fastest config, not a more correct
    one. Read at lowering time so a test fixture can toggle it per process."""
    return os.environ.get("NVTE_INDEXER_DISABLE_AUTOTUNE", "0") == "1"


# --- FP8 operand quantization ------------------------------------------------
#
# e4m3 is the score-matmul operand format. On gfx942 the hardware format is the
# "fnuz" variant (exponent bias 8); everywhere else it is OCP e4m3. Both are
# already mapped to Triton type strings in ``utils.get_triton_dtype``, so the
# kernels below are dtype-polymorphic and need no per-variant handling.


def fp8_dtype():
    """e4m3 dtype matching the current device's hardware format."""
    return jnp.dtype(get_jnp_float8_e4m3_type())


def is_fp8(dtype):
    """Whether ``dtype`` is one of the 8-bit float formats."""
    dtype = jnp.dtype(dtype)
    return jnp.issubdtype(dtype, jnp.floating) and dtype.itemsize == 1


def fp8_dot_supported(d_i):
    """Whether the score matmul is worth running in fp8 for this inner dim.

    The narrowest e4m3 MFMA/WGMMA tile contracts 32 elements, so a ``d_i``
    below that would have Triton pad the K axis with zeros -- costing more than
    the fp8 speedup buys. Callers should stay in bf16 there.
    """
    return d_i >= 32


def quantize_e4m3(x, *, axis=-1):
    """Symmetric per-row e4m3 quantization along ``axis``.

    Offered for callers that hold unquantized operands; the score ops do not
    call it themselves. Returns ``(x_q, scale)`` such that
    ``x ~= x_q * expand_dims(scale, axis)``. ``scale`` is fp32 and strictly
    positive, which is what lets it commute with the ReLU downstream (see the
    module docstring).

    Differentiable as a straight-through estimator: the scale is held constant
    (``stop_gradient``), so a cotangent w.r.t. ``x_q`` flows back to ``x`` as
    ``g / scale`` without the spurious term the amax reduction would otherwise
    contribute, and the e4m3 rounding is not modelled.
    """
    dtype = fp8_dtype()
    fp8_max = float(jnp.finfo(dtype).max)
    x_f32 = x.astype(jnp.float32)
    amax = jnp.max(jnp.abs(x_f32), axis=axis)
    # An all-zero row has no scale worth choosing; 1.0 keeps it strictly
    # positive (so the ReLU factoring stays valid) and round-trips to zero.
    scale = jax.lax.stop_gradient(jnp.where(amax > 0, amax / fp8_max, 1.0))
    x_q = jnp.clip(x_f32 / jnp.expand_dims(scale, axis), -fp8_max, fp8_max)
    return x_q.astype(dtype), scale


def _validate_scales(Hq, Hk, Sq, Ks):
    """Fill in unit scales for any omitted, and check the supplied ones.

    The scales index the non-contracted axes of their operand: ``Sq`` is
    ``Hq.shape[:-1]`` and ``Ks`` is ``Hk.shape[:-1]``.
    """
    if Sq is None:
        Sq = jnp.ones(Hq.shape[:-1], jnp.float32)
    elif Sq.shape != Hq.shape[:-1]:
        raise ValueError(
            f"Sq shape {Sq.shape} does not match Hq rows {Hq.shape[:-1]}"
        )
    if Ks is None:
        Ks = jnp.ones(Hk.shape[:-1], jnp.float32)
    elif Ks.shape != Hk.shape[:-1]:
        raise ValueError(
            f"Ks shape {Ks.shape} does not match Hk rows {Hk.shape[:-1]}"
        )
    return Sq.astype(jnp.float32), Ks.astype(jnp.float32)


def _score_reduce_autotune_configs():
    # The kernel is dominated by Hq reads (one (BLOCK_T, d_i) load per H
    # iteration). Bigger BLOCK_T ⇒ fewer T tiles ⇒ less total Hq traffic.
    # Bigger BLOCK_S ⇒ more Hk reuse but bigger per-CTA footprint.
    #
    # BLOCK_T=512 was tried and consistently failed to launch on MI355X
    # (resource exhaustion — VGPR/LDS budget for 64-iter H-loop with that
    # large an accumulator). Capped at 256.
    cfgs = [
        triton.Config({"BLOCK_T": 32,  "BLOCK_S": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_T": 32,  "BLOCK_S": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_T": 256, "BLOCK_S": 32},  num_warps=8, num_stages=2),
    ]
    cfgs += [
        triton.Config({"BLOCK_T": bt, "BLOCK_S": bs, "matrix_instr_nonkdim": nk_dim, "waves_per_eu": wpe}, num_warps=nw, num_stages=ns)
        for bt in (32, 64)
        for bs in (256, 512)
        for nk_dim in (16,)
        for wpe in (0, 2)   # 0 means let backend compiler decide
        for nw in (4,)
        for ns in (2,)
    ]
    return cfgs


@triton.autotune(configs=_score_reduce_autotune_configs(), key=["H", "d_i"])
@triton.jit
def _score_reduce_kernel(
    Hq_ptr,       # (B, oH, T_t, H, d_i) — produced by einsum("...tc,hci->...thi")
    Hk_ptr,       # (B, oH, T_s, d_i)
    W_o_ptr,      # (B, oH, T_t, H)
    Sq_ptr,       # (B, oH, T_t, H)  fp32 — Hq row scales (all-ones when bf16)
    Ks_ptr,       # (B, oH, T_s)     fp32 — Hk row scales (all-ones when bf16)
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
    ks_base = b * (oH * T_s) + h_outer * T_s
    o_base = b * (oH * T_t * T_s) + h_outer * (T_t * T_s)

    # Load the (BLOCK_S, d_i) Hk slab once — it is loop-invariant over H.
    hk_ptrs = Hk_ptr + hk_base + rs[:, None] * d_i + rdi[None, :]
    Hk_tile = tl.load(hk_ptrs, mask=rs_mask[:, None], other=0.0)
    Hk_T = tl.trans(Hk_tile)  # (d_i, BLOCK_S)

    # Hk row scales — also loop-invariant over H, applied once after the loop.
    ks = tl.load(Ks_ptr + ks_base + rs, mask=rs_mask, other=0.0)

    acc = tl.zeros((BLOCK_T, BLOCK_S), dtype=tl.float32)

    for h in range(H):
        hq_ptrs = (Hq_ptr + hq_base
                   + rt[:, None] * (H * d_i) + h * d_i + rdi[None, :])
        Hq_h = tl.load(hq_ptrs, mask=rt_mask[:, None], other=0.0)

        wo_ptrs = W_o_ptr + wo_base + rt * H + h
        w_h = tl.load(wo_ptrs, mask=rt_mask, other=0.0).to(tl.float32)
        # Fold the Hq row scale into the output weight: relu is
        # positive-homogeneous, so sq can be pulled out through it.
        sq_h = tl.load(Sq_ptr + wo_base + rt * H + h, mask=rt_mask, other=0.0)

        score = tl.dot(Hq_h, Hk_T)
        score = tl.maximum(score, 0.0)
        acc += score * (w_h * sq_h)[:, None]

    acc = acc * ks[None, :]

    o_ptrs = O_ptr + o_base + rt[:, None] * T_s + rs[None, :]
    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty),
             mask=rt_mask[:, None] & rs_mask[None, :])


_score_reduce_p = extend_core.Primitive("te_indexer_score_reduce_triton")
_score_reduce_p.multiple_results = True


@_score_reduce_p.def_abstract_eval
def _score_reduce_abstract(Hq, Hk, W_o, Sq, Ks, *, out_dtype):
    del W_o, Sq, Ks
    # Hq layout: (B, oH, T_t, H, d_i)
    B, oH, T_t, _H, _d_i = Hq.shape
    T_s = Hk.shape[2]
    return [core.ShapedArray((B, oH, T_t, T_s), out_dtype)]


_score_reduce_p.def_impl(functools.partial(xla.apply_primitive, _score_reduce_p))


def _score_reduce_lowering(ctx, Hq, Hk, W_o, Sq, Ks, *, out_dtype):
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

    saved_configs = _score_reduce_kernel.configs
    if _autotune_disabled():
        _score_reduce_kernel.configs = saved_configs[:1]
    try:
        return triton_call_lowering(
            ctx,
            _score_reduce_kernel,
            Hq, Hk, W_o, Sq, Ks,
            grid=grid_fn,
            constexprs={
                "B": B,
                "oH": oH,
                "T_t": T_t,
                "T_s": T_s,
                "H": H,
                "d_i": d_i,
            },
        )
    finally:
        _score_reduce_kernel.configs = saved_configs


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


def _score_dscores_chunk_autotune_configs():
    # matrix_instr_nonkdim is pinned to 16. On Triton 3.7.0 / gfx950 the
    # H_CHUNK-unrolled score matmul crashes the compiler (uncatchable
    # std::bad_alloc / heap corruption the autotuner cannot skip) for both
    # nonkdim=32 and the backend-default nonkdim (0); only 16x16 MFMA tiles
    # compile. num_stages is pinned to 1: pipelining the s-loop crashes LLVM
    # codegen on the same toolchain.
    cfgs = []
    cfgs += [
        triton.Config(
            {"BLOCK_T": bt, "BLOCK_S": bs,
             "matrix_instr_nonkdim": 16, "waves_per_eu": wpe},
            num_warps=nw, num_stages=1)
        for bt in (32, 64, 128)
        for bs in (128, 256)
        for wpe in (0, 2)
        for nw in (4, 8)
    ]
    # larger BLOCK_S for long T_s
    cfgs += [
        triton.Config({"BLOCK_T": bt, "BLOCK_S": 512, "matrix_instr_nonkdim": 16},
                      num_warps=4, num_stages=1)
        for bt in (32, 64)
    ]
    return cfgs


@triton.autotune(configs=_score_dscores_chunk_autotune_configs(),
                 key=["T", "T_s", "H_CHUNK", "d_i"])
@triton.jit
def _score_dscores_chunk_kernel(
    Hq_chunk_ptr,        # input  (B, oH, T,   H_CHUNK, d_i) bf16 or e4m3
    Hk_ptr,              # input  (B, oH, T_s, d_i)         bf16 or e4m3
    W_o_chunk_ptr,       # input  (B, oH, T,   H_CHUNK)     bf16
    dO_ptr,              # input  (B, oH, T,   T_s)         fp32
    Sq_chunk_ptr,        # input  (B, oH, T,   H_CHUNK)     fp32 — Hq row scales
    Ks_ptr,              # input  (B, oH, T_s)              fp32 — Hk row scales
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
    """One CTA handles (T_tile, all H_CHUNK heads) for one (b, h_outer).

    Grid: (cdiv(T, BLOCK_T), B * oH). For each s-chunk we load dO_chunk and
    Hk_chunk ONCE and reuse them across every head in the chunk -- the key
    saving vs the original (which spun a separate CTA per head, each re-reading
    dO/Hk). dW_o is reduced in registers (sum over s) per head, so h_relu never
    lands in HBM.
    """
    pid_t = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = (pid_bh // oH).to(tl.int64)
    h_outer = (pid_bh % oH).to(tl.int64)

    rt = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    rdi = tl.arange(0, d_i)
    rhc = tl.arange(0, H_CHUNK)
    rt_mask = rt < T

    hq_base = b * (oH * T * H_CHUNK * d_i) + h_outer * (T * H_CHUNK * d_i)
    hk_base = b * (oH * T_s * d_i) + h_outer * (T_s * d_i)
    wo_base = b * (oH * T * H_CHUNK) + h_outer * (T * H_CHUNK)
    do_base = b * (oH * T * T_s) + h_outer * (T * T_s)
    ks_base = b * (oH * T_s) + h_outer * T_s
    ds_base = b * (oH * T * H_CHUNK * T_s) + h_outer * (T * H_CHUNK * T_s)

    # Per-head dW_o accumulators packed as (BLOCK_T, H_CHUNK), reduced over s.
    dWo_acc = tl.zeros((BLOCK_T, H_CHUNK), dtype=tl.float32)

    for s_start in range(0, T_s, BLOCK_S):
        rs = s_start + tl.arange(0, BLOCK_S)
        rs_mask = rs < T_s

        # Load Hk[..., s_chunk, :] and dO[..., t_tile, s_chunk] ONCE per s-chunk
        # -- shared across all H_CHUNK heads below.
        hk_ptrs = Hk_ptr + hk_base + rs[:, None] * d_i + rdi[None, :]
        Hk_chunk = tl.load(hk_ptrs, mask=rs_mask[:, None], other=0.0)
        Hk_T = tl.trans(Hk_chunk)  # (d_i, BLOCK_S)

        ks_chunk = tl.load(Ks_ptr + ks_base + rs, mask=rs_mask, other=0.0)

        do_ptrs = dO_ptr + do_base + rt[:, None] * T_s + rs[None, :]
        dO_chunk = tl.load(
            do_ptrs, mask=rt_mask[:, None] & rs_mask[None, :], other=0.0,
        )

        for h in tl.static_range(H_CHUNK):
            # Hq/w for head h (small, L2-resident across s-chunks).
            Hq_h = tl.load(
                Hq_chunk_ptr + hq_base + rt[:, None] * (H_CHUNK * d_i)
                + h * d_i + rdi[None, :],
                mask=rt_mask[:, None], other=0.0,
            )
            w_h = tl.load(
                W_o_chunk_ptr + wo_base + rt * H_CHUNK + h,
                mask=rt_mask, other=0.0,
            ).to(tl.float32)
            sq_h = tl.load(
                Sq_chunk_ptr + wo_base + rt * H_CHUNK + h,
                mask=rt_mask, other=0.0,
            )

            # Scores here are the *quantized* dot q_hat.k_hat; the true score is
            # sq_h * ks * scores. Both scales are positive, so the relu mask is
            # the same either way and only the magnitude below needs rescaling.
            scores = tl.dot(Hq_h, Hk_T)  # (BLOCK_T, BLOCK_S)
            relu_mask = scores > 0
            h_relu = tl.where(relu_mask, scores, 0.0) * ks_chunk[None, :]

            # dW_o[..., h] += sum_s (h_relu * dO); accumulate into column h.
            dwo_h = tl.sum(h_relu * dO_chunk, axis=1) * sq_h  # (BLOCK_T,)
            dWo_acc += tl.where(rhc[None, :] == h, dwo_h[:, None], 0.0)

            # dscores[..., h, s] = relu_mask * (dO * W_o)
            dscores = tl.where(relu_mask, dO_chunk * w_h[:, None], 0.0)
            ds_ptrs = (dscores_chunk_ptr + ds_base
                       + rt[:, None] * (H_CHUNK * T_s) + h * T_s + rs[None, :])
            tl.store(
                ds_ptrs, dscores.to(dscores_chunk_ptr.dtype.element_ty),
                mask=rt_mask[:, None] & rs_mask[None, :],
            )

    # Store dW_o[..., t_tile, :] for all heads.
    dwo_out_ptrs = dWo_chunk_ptr + wo_base + rt[:, None] * H_CHUNK + rhc[None, :]
    tl.store(
        dwo_out_ptrs, dWo_acc.to(dWo_chunk_ptr.dtype.element_ty),
        mask=rt_mask[:, None],
    )


_score_dscores_chunk_p = extend_core.Primitive("te_indexer_score_dscores_chunk")
_score_dscores_chunk_p.multiple_results = True


@_score_dscores_chunk_p.def_abstract_eval
def _score_dscores_chunk_abstract(Hq_chunk, Hk, W_o_chunk, dO, Sq_chunk, Ks):
    del Hk, Sq_chunk, Ks
    B, oH, T, H_CHUNK, _ = Hq_chunk.shape
    T_s = dO.shape[-1]
    # Both outputs are cotangents, so they follow W_o's (bf16) dtype -- Hq_chunk
    # is e4m3 on the fp8 path and would silently narrow them.
    return [
        core.ShapedArray((B, oH, T, H_CHUNK, T_s), W_o_chunk.dtype),  # dscores
        core.ShapedArray((B, oH, T, H_CHUNK), W_o_chunk.dtype),       # dW_o
    ]


_score_dscores_chunk_p.def_impl(
    functools.partial(xla.apply_primitive, _score_dscores_chunk_p)
)


def _score_dscores_chunk_lowering(ctx, Hq_chunk, Hk, W_o_chunk, dO, Sq_chunk, Ks):
    Hq_aval = ctx.avals_in[0]
    dO_aval = ctx.avals_in[3]
    B, oH, T, H_CHUNK, d_i = Hq_aval.shape
    T_s = dO_aval.shape[-1]

    # Grid: (T-tiles, B*oH) -- one CTA per (T_tile, b, h_outer) covers all
    # H_CHUNK heads (dO/Hk shared across heads). Depends on autotuned BLOCK_T.
    def grid_fn(merged_kwargs):
        bt = merged_kwargs.get("BLOCK_T", _HBWD_BLOCK_T)
        return ((T + bt - 1) // bt, B * oH)

    saved_configs = _score_dscores_chunk_kernel.configs
    if _autotune_disabled():
        _score_dscores_chunk_kernel.configs = saved_configs[:1]
    try:
        return triton_call_lowering(
            ctx,
            _score_dscores_chunk_kernel,
            Hq_chunk, Hk, W_o_chunk, dO, Sq_chunk, Ks,
            grid=grid_fn,
            constexprs={
                "B": B, "oH": oH, "T": T, "T_s": T_s,
                "H_CHUNK": H_CHUNK, "d_i": d_i,
            },
        )
    finally:
        _score_dscores_chunk_kernel.configs = saved_configs


mlir.register_lowering(_score_dscores_chunk_p, _score_dscores_chunk_lowering, platform="rocm")
mlir.register_lowering(_score_dscores_chunk_p, _score_dscores_chunk_lowering, platform="cuda")


# --- Public score_reduce_triton with custom_vjp ------------------------------


@functools.partial(jax.custom_vjp, nondiff_argnums=(5,))
def _score_reduce_with_vjp(Hq, Hk, W_o, Sq, Ks, out_dtype):
    return _score_reduce_p.bind(Hq, Hk, W_o, Sq, Ks, out_dtype=out_dtype)[0]


def _score_reduce_fwd(Hq, Hk, W_o, Sq, Ks, out_dtype):
    out = _score_reduce_p.bind(Hq, Hk, W_o, Sq, Ks, out_dtype=out_dtype)[0]
    return out, (Hq, Hk, W_o, Sq, Ks)


_BWD_H_CHUNK = 8  # peak (B, oH, T, H_CHUNK, T_s) tile -- bounds materialization


def _score_reduce_bwd(out_dtype, residuals, dO):
    del out_dtype
    Hq, Hk, W_o, Sq, Ks = residuals
    B, oH, T, H, d_i = Hq.shape
    # The score recompute inside the Triton kernel consumes Hq / Hk in their
    # stored precision (with Sq / Ks), but the two dscores reductions below are
    # hipBLASLt GEMMs and need dequantized operands. W_o is the bf16 anchor for
    # that intermediate precision -- Hq / Hk may be e4m3.
    compute_dtype = W_o.dtype
    fp8_operands = is_fp8(Hq.dtype)

    def _dequant(x, scale):
        if not fp8_operands:
            return x
        return (x.astype(jnp.float32) * scale[..., None]).astype(compute_dtype)

    Hk_deq = _dequant(Hk, Ks)

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
    Sq_r = Sq.reshape(B, oH, T, n_chunks, H_CHUNK)
    # Move chunk axis to leading for scan over axis 0.
    Hq_s = jnp.moveaxis(Hq_r, -3, 0)   # (n_chunks, B, oH, T, H_CHUNK, d_i)
    Wo_s = jnp.moveaxis(Wo_r, -2, 0)   # (n_chunks, B, oH, T, H_CHUNK)
    Sq_s = jnp.moveaxis(Sq_r, -2, 0)   # (n_chunks, B, oH, T, H_CHUNK)

    def step(dHk_acc, chunk):
        Hq_c, Wo_c, Sq_c = chunk
        # Triton: dscores_chunk + dWo_chunk; no full (B,oH,T,H,T_s) tensor
        # ever exists in HBM.
        dscores_c, dWo_c = _score_dscores_chunk_p.bind(Hq_c, Hk, Wo_c, dO, Sq_c, Ks)
        # Dequantize only the current chunk, so peak stays at H_CHUNK/H of Hq.
        Hq_c_deq = _dequant(Hq_c, Sq_c)
        dHq_c = jnp.einsum("...ths,...si->...thi", dscores_c, Hk_deq)
        dHk_c = jnp.einsum("...ths,...thi->...si", dscores_c, Hq_c_deq)
        new_dHk_acc = dHk_acc + dHk_c.astype(jnp.float32)
        return new_dHk_acc, (dHq_c, dWo_c)

    init = jnp.zeros(Hk.shape, dtype=jnp.float32)
    dHk_acc, (dHq_chunks, dWo_chunks) = jax.lax.scan(
        step, init, (Hq_s, Wo_s, Sq_s),
    )
    # dHq_chunks: (n_chunks, B, oH, T, H_CHUNK, d_i)
    # dWo_chunks: (n_chunks, B, oH, T, H_CHUNK)
    dHq = jnp.moveaxis(dHq_chunks, 0, -3).reshape(B, oH, T, H, d_i)
    dWo = jnp.moveaxis(dWo_chunks, 0, -2).reshape(B, oH, T, H)
    dHk = dHk_acc

    # The einsums above produce cotangents w.r.t. the *dequantized* operands.
    # Chain through Hq = Hq_q * sq to get the cotangent w.r.t. the operand we
    # were actually handed. When the operands are e4m3 this also renormalizes
    # the gradient into their range, instead of storing a raw unscaled fp8
    # value. With unit scales it is an exact no-op.
    dHq = (dHq.astype(jnp.float32) * Sq[..., None]).astype(Hq.dtype)
    dHk = (dHk.astype(jnp.float32) * Ks[..., None]).astype(Hk.dtype)

    # Sq / Ks are quantization metadata, held constant by every scaling recipe
    # (and by quantize_e4m3's stop_gradient), so they take a zero cotangent.
    return (
        dHq,
        dHk,
        dWo.astype(W_o.dtype),
        jnp.zeros_like(Sq),
        jnp.zeros_like(Ks),
    )


_score_reduce_with_vjp.defvjp(_score_reduce_fwd, _score_reduce_bwd)


def score_reduce_triton(Hq, Hk, W_o, *, Sq=None, Ks=None, out_dtype=None):
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
        Hq:  (B, oH, T_t, H, d_i) — e4m3 or bf16, consumed as given.
        Hk:  (B, oH, T_s, d_i)    — same dtype as Hq.
        W_o: (B, oH, T_t, H)
        Sq:  (B, oH, T_t, H) fp32 row scales for Hq, or None for unit scales.
        Ks:  (B, oH, T_s) fp32 row scales for Hk, or None for unit scales.
        out_dtype: defaults to W_o.dtype.

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
    # Both operands feed one tl.dot, which specializes on a single input dtype.
    if Hq.dtype != Hk.dtype:
        raise ValueError(
            f"Hq and Hk must share a dtype; got {Hq.dtype} and {Hk.dtype}"
        )
    Sq, Ks = _validate_scales(Hq, Hk, Sq, Ks)

    if out_dtype is None:
        # Not Hq.dtype: that would make an e4m3 operand yield an e4m3 output.
        out_dtype = W_o.dtype

    return _score_reduce_with_vjp(Hq, Hk, W_o, Sq, Ks, jnp.dtype(out_dtype))


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
# >= 1). Configs whose BLOCK_S exceeds K or doesn't divide K must be pruned
# (see _prune_topk_configs) — otherwise the autotuner would time them
# as zero-work (fast) and pick a bogus winner that returns all-zero indices.
_SCORE_TOPK_CONFIGS = [
    triton.Config({"BLOCK_S": bs, "BLOCK_T": bt, "waves_per_eu": wpe}, num_warps=nw, num_stages=ns)
    for bt in (1, 2)
    for bs in (32, 64, 128, 256)
    for wpe in (0, 2, 4)
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


def _prune_topk_configs(configs, named_args, **kwargs):
    """early_config_prune for _score_topk_kernel. Keep only configs where
    BLOCK_S divides K (INNER = K//BLOCK_S >= 1) and BLOCK_T divides T_t. The
    runtime values arrive in named_args or kwargs depending on call style."""
    vals = {**named_args, **kwargs}
    k = vals["K"]
    T_t = vals["T_t"]
    return [
        c for c in configs
        if c.kwargs["BLOCK_S"] <= k
        and k % c.kwargs["BLOCK_S"] == 0
        and T_t % c.kwargs["BLOCK_T"] == 0
    ]


@triton.autotune(
    configs=_SCORE_TOPK_CONFIGS,
    key=["H", "d_i", "T_s", "K"],
    prune_configs_by={"early_config_prune": _prune_topk_configs},
)
@triton.jit
def _score_topk_kernel(
    Hq_ptr,        # (B, oH, T_t, H, d_i) bf16 or e4m3
    Hk_ptr,        # (B, oH, T_s, d_i) bf16 or e4m3
    W_o_ptr,       # (B, oH, T_t, H) bf16
    Sq_ptr,        # (B, oH, T_t, H) fp32 — Hq row scales
    Ks_ptr,        # (B, oH, T_s)    fp32 — Hk row scales
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

    # Load w_o[b, h_outer, rt, :] -> [BLOCK_T, H], with the Hq row scale folded
    # in (relu is positive-homogeneous, so sq pulls out through it).
    wo_base = b * (oH * T_t * H) + h_outer * (T_t * H)
    w_o = tl.load(
        W_o_ptr + wo_base + rt_64[:, None] * H + rh[None, :],
        mask=rt_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    sq = tl.load(
        Sq_ptr + wo_base + rt_64[:, None] * H + rh[None, :],
        mask=rt_mask[:, None],
        other=0.0,
    )
    w_o = w_o * sq

    # Flatten Hq for one big matmul per Hk_chunk: [BLOCK_T * H, d_i] -> trans
    Hq_flat = tl.reshape(Hq_token, (BLOCK_T * H, d_i))
    Hq_T = tl.trans(Hq_flat)  # [d_i, BLOCK_T * H]
    w_o_flat = tl.reshape(w_o, (BLOCK_T * H,))

    hk_base = b * (oH * T_s * d_i) + h_outer * (T_s * d_i)
    ks_base = b * (oH * T_s) + h_outer * T_s

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
            # Hk row scale — per-s, so it rescales whole score columns and does
            # affect the ranking; must be applied before the radix flip below.
            ks_chunk = tl.load(Ks_ptr + ks_base + rs, mask=rs_mask, other=0.0)
            chunk_scores = chunk_scores * ks_chunk[:, None]
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


# --- Single-sort top-k (for S_PAD that fits in registers) -------------
#
# The streaming kernel above sorts a 2K buffer N_OUTER = S_PAD/K times. When k
# is a large fraction of T_s (e.g. k = T_s/2), that's several sorts of the 2K
# buffer. If all S_PAD candidates fit in registers, scattering them into one
# BLOCK_T*S_PAD buffer and doing a SINGLE descending sort is ~2x less sort work.
_SINGLE_SORT_MAX = 4096


# Configs for the single-sort kernel. BLOCK_S must divide S_PAD; BLOCK_T must
# divide T_t (pruned below). Tuned winner on gfx950 is BLOCK_T=1, BLOCK_S=128.
_SINGLE_TOPK_CONFIGS = [
    triton.Config({"BLOCK_S": bs, "BLOCK_T": bt, "waves_per_eu": wpe}, num_warps=nw, num_stages=1)
    for bs in (64, 128, 256)
    for bt in (1, 2)
    for wpe in (0, 2, 3, 4)
    for nw in (4, 8)
]


def _prune_single_topk_configs(configs, named_args, **kwargs):
    """early_config_prune for _score_topk_single_kernel. Keep only configs where
    BLOCK_S divides S_PAD (the static chunk loop tiles it exactly) and BLOCK_T
    divides T_t."""
    vals = {**named_args, **kwargs}
    S_PAD = vals["S_PAD"]
    T_t = vals["T_t"]
    return [
        c for c in configs
        if S_PAD % c.kwargs["BLOCK_S"] == 0
        and T_t % c.kwargs["BLOCK_T"] == 0
    ]


@triton.autotune(
    configs=_SINGLE_TOPK_CONFIGS,
    key=["H", "d_i", "T_s", "K"],
    prune_configs_by={"early_config_prune": _prune_single_topk_configs},
)
@triton.jit
def _score_topk_single_kernel(
    Hq_ptr, Hk_ptr, W_o_ptr, Sq_ptr, Ks_ptr, Topk_idx_ptr,
    B: tl.constexpr, oH: tl.constexpr, T_t: tl.constexpr, T_s: tl.constexpr,
    H: tl.constexpr, d_i: tl.constexpr, K: tl.constexpr, S_PAD: tl.constexpr,
    BLOCK_S: tl.constexpr, BLOCK_T: tl.constexpr,
):
    """Like ``_score_topk_kernel`` but holds all S_PAD candidates and sorts once.

    Grid: (cdiv(T_t, BLOCK_T), B * oH). Buffer is BLOCK_T*S_PAD packed uint64
    with T encoded in the high bits (same 1D-sort-groups-per-T trick). Requires
    BLOCK_S | S_PAD (so the static chunk loop tiles S_PAD exactly); no BLOCK_S|K
    constraint is needed since there is no 2K streaming buffer.
    """
    pid_t = tl.program_id(0)
    pid_bh = tl.program_id(1)
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

    hq_base = b * (oH * T_t * H * d_i) + h_outer * (T_t * H * d_i)
    Hq_token = tl.load(
        Hq_ptr + hq_base + rt_64[:, None, None] * (H * d_i)
        + rh[None, :, None] * d_i + rdi[None, None, :],
        mask=rt_mask[:, None, None], other=0.0)
    wo_base = b * (oH * T_t * H) + h_outer * (T_t * H)
    w_o = tl.load(W_o_ptr + wo_base + rt_64[:, None] * H + rh[None, :],
                  mask=rt_mask[:, None], other=0.0).to(tl.float32)
    sq = tl.load(Sq_ptr + wo_base + rt_64[:, None] * H + rh[None, :],
                 mask=rt_mask[:, None], other=0.0)
    w_o = w_o * sq
    Hq_flat = tl.reshape(Hq_token, (BLOCK_T * H, d_i))
    Hq_T = tl.trans(Hq_flat)
    w_o_flat = tl.reshape(w_o, (BLOCK_T * H,))
    hk_base = b * (oH * T_s * d_i) + h_outer * (T_s * d_i)
    ks_base = b * (oH * T_s) + h_outer * T_s

    N_CHUNK: tl.constexpr = S_PAD // BLOCK_S
    BIG: tl.constexpr = BLOCK_T * S_PAD
    rb = tl.arange(0, BIG)
    rb_t = rb // S_PAD
    rb_pos = rb % S_PAD
    t_enc_per_slot = (BLOCK_T - rb_t).to(tl.uint64)
    top_packed = (t_enc_per_slot << 56) | rb_pos.to(tl.uint64)

    for c in tl.static_range(N_CHUNK):
        rs = c * BLOCK_S + rs_chunk
        rs_mask = rs < T_s
        hk_ptrs = Hk_ptr + hk_base + rs[:, None] * d_i + rdi[None, :]
        Hk_chunk = tl.load(hk_ptrs, mask=rs_mask[:, None], other=0.0)
        logits = tl.dot(Hk_chunk, Hq_T)
        logits = tl.maximum(logits, 0.0)
        weighted = logits * w_o_flat[None, :]
        weighted_3d = tl.reshape(weighted, (BLOCK_S, BLOCK_T, H))
        chunk_scores = tl.sum(weighted_3d, axis=2)
        ks_chunk = tl.load(Ks_ptr + ks_base + rs, mask=rs_mask, other=0.0)
        chunk_scores = chunk_scores * ks_chunk[:, None]
        chunk_scores_T = tl.trans(chunk_scores)
        bits = chunk_scores_T.to(tl.uint32, bitcast=True)
        sign = bits >> 31
        flip_mask = (0 - sign.to(tl.int32)).to(tl.uint32) | 0x80000000
        sortable = bits ^ flip_mask
        sortable = tl.where(rs_mask[None, :], sortable, 0)
        t_enc_chunk = (BLOCK_T - rt_local).to(tl.uint64)
        rs_2d = tl.broadcast_to(rs[None, :], (BLOCK_T, BLOCK_S))
        chunk_packed_2d = ((t_enc_chunk[:, None] << 56)
                           | (sortable.to(tl.uint64) << 24) | rs_2d.to(tl.uint64))
        chunk_packed_flat = tl.reshape(chunk_packed_2d, (BLOCK_T * BLOCK_S,))
        chunk_offset = c * BLOCK_S
        in_slot = (rb_pos >= chunk_offset) & (rb_pos < chunk_offset + BLOCK_S)
        j = rb_pos - chunk_offset
        flat_idx = tl.where(in_slot, rb_t * BLOCK_S + j, 0).to(tl.int32)
        gathered = tl.gather(chunk_packed_flat, flat_idx, axis=0)
        top_packed = tl.where(in_slot, gathered, top_packed)

    top_packed = tl.sort(top_packed, descending=True)  # SINGLE sort
    out_idx = rt_local[:, None] * S_PAD + rk[None, :]
    out_idx_flat = tl.reshape(out_idx, (BLOCK_T * K,)).to(tl.int32)
    top_k_packed_flat = tl.gather(top_packed, out_idx_flat, axis=0)
    top_k_packed = tl.reshape(top_k_packed_flat, (BLOCK_T, K))
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
def _score_topk_abstract(Hq, Hk, W_o, Sq, Ks, *, k):
    del Hk, W_o, Sq, Ks
    B, oH, T_t, _H, _d_i = Hq.shape
    return [core.ShapedArray((B, oH, T_t, k), jnp.int32)]


_score_topk_p.def_impl(functools.partial(xla.apply_primitive, _score_topk_p))


def _score_topk_lowering(ctx, Hq, Hk, W_o, Sq, Ks, *, k):
    Hq_aval = ctx.avals_in[0]
    Hk_aval = ctx.avals_in[1]
    B, oH, T_t, H, d_i = Hq_aval.shape
    T_s = Hk_aval.shape[2]
    S_PAD = _next_pow2(T_s)

    # Both kernels are self-autotuned (@triton.autotune at definition); invalid
    # configs are dropped by their early_config_prune hooks, so the lowering just
    # picks the right kernel and launches. Single-sort path when all S_PAD
    # candidates fit in registers (~1.5x faster: one sort instead of S_PAD/K
    # streaming sorts of a 2K buffer); streaming kernel for very large T_s.
    autotuned_kernel = (_score_topk_single_kernel if S_PAD <= _SINGLE_SORT_MAX
              else _score_topk_kernel)

    def grid_fn(merged_kwargs):
        bt = merged_kwargs.get("BLOCK_T", 1)
        return (triton.cdiv(T_t, bt), B * oH)

    constexprs = {
        "B": B, "oH": oH, "T_t": T_t, "T_s": T_s,
        "H": H, "d_i": d_i,
        "K": k, "S_PAD": S_PAD,
    }

    # Apply the kernel's early_config_prune ourselves. triton_call_lowering hands
    # every config to jaxlib's runtime autotuner, which picks the fastest by
    # timing -- not correctness -- and does not run the prune hook. An invalid
    # config (e.g. a BLOCK_S that doesn't divide S_PAD, leaving the static chunk
    # loop with zero iterations) does no work, "wins" on speed, and returns the
    # kernel's uninitialized buffer. Prune here, before lowering, so only configs
    # valid for this S_PAD/K/T_t reach the autotuner.
    valid_configs = autotuned_kernel.early_config_prune(autotuned_kernel.configs, constexprs)
    if _autotune_disabled():
        valid_configs = valid_configs[:1]
    saved_configs = autotuned_kernel.configs
    autotuned_kernel.configs = valid_configs
    try:
        return triton_call_lowering(
            ctx,
            autotuned_kernel,
            Hq, Hk, W_o, Sq, Ks,
            grid=grid_fn,
            constexprs=constexprs,
        )
    finally:
        autotuned_kernel.configs = saved_configs


mlir.register_lowering(_score_topk_p, _score_topk_lowering, platform="rocm")
mlir.register_lowering(_score_topk_p, _score_topk_lowering, platform="cuda")


def score_topk_triton(Hq, Hk, W_o, *, k, Sq=None, Ks=None):
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
        Sq, Ks: per-row operand scales, or None for unit scales; see
             ``score_reduce_triton``. ``Ks`` rescales whole score columns and
             so does affect the ranking — it is applied before selection.

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
    if Hq.dtype != Hk.dtype:
        raise ValueError(
            f"Hq and Hk must share a dtype; got {Hq.dtype} and {Hk.dtype}"
        )
    Sq, Ks = _validate_scales(Hq, Hk, Sq, Ks)

    return _score_topk_p.bind(Hq, Hk, W_o, Sq, Ks, k=k)[0]
