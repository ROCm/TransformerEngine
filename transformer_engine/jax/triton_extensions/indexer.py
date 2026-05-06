# Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Raw-Triton low-rank indexer kernel + JAX primitive.

Math (matches the reference in transformer_engine.jax.indexer):

    C_q = Q @ W_dq                                            # (..., T, d_c)
    H_q = einsum("...tc,hci->...thi", C_q, W_uq)              # (..., T, H, d_i)
    H_k = K @ W_k                                              # (..., S, d_i)
    H   = relu(einsum("...thi,...si->...ths", H_q, H_k))       # (..., T, H, S)
    O   = einsum("...ths,...ht->...ts", H, weights)            # (..., T, S)

Q is the hidden state (rank-4 BHSD: B × outer-H × T × d). W_dq is a low-rank
down-projection (d → d_c) and W_uq is the per-(indexer-head) up-projection.
The kernel loops over indexer heads internally; the outer (B, outer-H) dims
are flattened into the grid's first axis.

FP8 mode: Q / K / W_uq / W_dq / W_k are all FP8 e4m3 (same dtype). The
five per-tensor scales (scale_q, scale_k, scale_wq, scale_wd, scale_wk)
fold into a single fp32 scalar applied at the end (ReLU is scale-invariant
under positive scaling). Three intermediate amax-based re-quantizations
(Cq, Hk, Hq per-head) keep the inner matmuls in fp8 too.
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


# Autotune sweep: BLOCK_T × BLOCK_S × num_warps × num_stages. Profiling
# showed num_warps=4 with the prior default (BLOCK_T=128) saturated VGPR
# (256/thread), forcing 1 wave/SIMD; smaller tiles or num_warps=8 cut VGPR
# in half and gave a 6× speedup at the d=512 fp8 config. Each config below
# launches at its own grid (cdiv(T_t, BLOCK_T) × cdiv(T_s, BLOCK_S)) — the
# triton_call_lowering helper supports per-config grids via a callable
# `grid` argument.
def _autotune_configs():
    configs = []
    for block_t in (16, 32):
        for block_s in (16, 32):
            for block_d in (16, 32):
                for num_warps in (4, 8):
                    for num_stages in (1, 2):
                        configs.append(triton.Config(
                            {"BLOCK_T": block_t, "BLOCK_S": block_s,
                             "BLOCK_D": block_d},
                            num_warps=num_warps, num_stages=num_stages,
                        ))
    return configs

_AUTOTUNE_CONFIGS = _autotune_configs()
# Re-run the benchmark when any of these constexprs change. T_t/T_s only
# affect grid size; their optimal config is dominated by per-CTA shape and
# the precision (IS_FP8).
_AUTOTUNE_KEY = ["IS_FP8", "d", "d_c", "H", "d_i"]


# Max representable value of FP8 e4m3 (used for per-tile inter-quantization).
# Triton requires module-level constants referenced inside @jit kernels to be
# wrapped in tl.constexpr explicitly.
_FP8_E4M3_MAX = tl.constexpr(448.0)
# Floor on per-tile amax to avoid divide-by-zero when a tile is all-zero.
_FP8_AMAX_EPS = tl.constexpr(1e-30)


@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=_AUTOTUNE_KEY)
@triton.jit
def _indexer_kernel(
    Q_ptr,
    K_ptr,
    W_uq_ptr,     # (H, d_c, d_i)  - replicated across (B, oH)
    W_dq_ptr,     # (d, d_c)       - replicated; same dtype as Q
    W_k_ptr,      # (d, d_i)       - replicated
    weights_ptr,
    scale_ptr,    # 0-D fp32 tensor: combined scale sq*sk*swq*swk (1.0 if non-FP8)
    O_ptr,
    B: tl.constexpr,
    oH: tl.constexpr,
    T_t: tl.constexpr,
    T_s: tl.constexpr,
    d: tl.constexpr,
    d_c: tl.constexpr,
    H: tl.constexpr,
    d_i: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_FP8: tl.constexpr,
):
    """Compute one (BLOCK_T, BLOCK_S) tile of O for one (b, h_outer) slice.

    Grid: (B * oH, cdiv(T_t, BLOCK_T), cdiv(T_s, BLOCK_S))

    Pipeline:
      C_q = Q @ W_dq                                  (down-projection, d-tiled)
      Hk  = K @ W_k                                   (key projection, d-tiled)
      for h in range(H):                              (loop over indexer heads)
          Hq  = C_q @ W_uq[:, h, :]                   (per-head up-projection)
          Hi  = relu(Hq @ Hk^T)                       (per-head score)
          acc += Hi * weights[:, h]                   (weighted accumulate)

    The two d-contracting GEMMs (Q@W_dq and K@W_k) are tiled along d in
    chunks of BLOCK_D. This keeps the W_dq / W_k tiles loaded into LDS at
    BLOCK_D × {d_c, d_i} instead of d × {d_c, d_i}, freeing registers /
    LDS for the inner per-head loop.

    FP8 mode (IS_FP8=True): all five matrices share the fp8 dtype. Every
    MFMA is native fp8: the d-tiled Q@W_dq and K@W_k dots, then the inner
    C_q@W_uq[h] and Hq@Hk^T dots after per-tile amax re-quantization of
    Cq/Hk/Hq. The per-tile amax scales fold into the accumulator (Hq inside
    the loop, Cq/Hk after) along with the user's combined per-tensor scale.
    """
    pid_bh = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_s = tl.program_id(2)

    b = pid_bh // oH
    h_outer = pid_bh % oH

    rt = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    rs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    rdc = tl.arange(0, d_c)
    rdi = tl.arange(0, d_i)

    rt_mask = rt < T_t
    rs_mask = rs < T_s

    in_dtype = Q_ptr.dtype.element_ty
    q_base = b * (oH * T_t * d) + h_outer * (T_t * d)
    k_base = b * (oH * T_s * d) + h_outer * (T_s * d)

    # d-tiled accumulators for Q @ W_dq → (BLOCK_T, d_c) and K @ W_k →
    # (BLOCK_S, d_i). fp32 accumulators; quantization happens after the loop.
    # Requires d % BLOCK_D == 0.
    Cq_dot = tl.zeros((BLOCK_T, d_c), dtype=tl.float32)
    Hk_dot = tl.zeros((BLOCK_S, d_i), dtype=tl.float32)
    for d_off in range(0, d, BLOCK_D):
        rd = d_off + tl.arange(0, BLOCK_D)

        q_ptrs = Q_ptr + q_base + rt[:, None] * d + rd[None, :]
        Q_chunk = tl.load(q_ptrs, mask=rt_mask[:, None], other=0.0)
        k_ptrs = K_ptr + k_base + rs[:, None] * d + rd[None, :]
        K_chunk = tl.load(k_ptrs, mask=rs_mask[:, None], other=0.0)
        wdq_ptrs = W_dq_ptr + rd[:, None] * d_c + rdc[None, :]
        Wdq_chunk = tl.load(wdq_ptrs)
        wk_ptrs = W_k_ptr + rd[:, None] * d_i + rdi[None, :]
        Wk_chunk = tl.load(wk_ptrs)

        Cq_dot = tl.dot(Q_chunk, Wdq_chunk, acc=Cq_dot)
        Hk_dot = tl.dot(K_chunk, Wk_chunk, acc=Hk_dot)

    # Quantize Cq and Hk for the inner up-projection.
    if IS_FP8:
        Cq_amax = tl.maximum(tl.max(tl.abs(Cq_dot)), _FP8_AMAX_EPS)
        Cq_inter = Cq_amax / _FP8_E4M3_MAX
        C_q = (Cq_dot / Cq_inter).to(in_dtype)
        Hk_amax = tl.maximum(tl.max(tl.abs(Hk_dot)), _FP8_AMAX_EPS)
        Hk_inter = Hk_amax / _FP8_E4M3_MAX
        Hk_T = tl.trans((Hk_dot / Hk_inter).to(in_dtype))
    else:
        C_q = Cq_dot.to(in_dtype)
        Hk_T = tl.trans(Hk_dot.to(in_dtype))
        Cq_inter = 1.0
        Hk_inter = 1.0

    acc = tl.zeros((BLOCK_T, BLOCK_S), dtype=tl.float32)

    w_base = b * (oH * H * T_t) + h_outer * (H * T_t)
    for h_idx in range(H):
        # W_uq[h_idx, :, :] is a contiguous (d_c, d_i) block of W_uq (H, d_c, d_i).
        wuq_ptrs = W_uq_ptr + h_idx * (d_c * d_i) + rdc[:, None] * d_i + rdi[None, :]
        Wuq_h = tl.load(wuq_ptrs)

        # Hq = C_q @ W_uq[h_idx]: (BLOCK_T, d_i)
        Hq_dot = tl.dot(C_q, Wuq_h)
        if IS_FP8:
            Hq_amax = tl.maximum(tl.max(tl.abs(Hq_dot)), _FP8_AMAX_EPS)
            Hq_inter = Hq_amax / _FP8_E4M3_MAX
            Hq_h = (Hq_dot / Hq_inter).to(in_dtype)
        else:
            Hq_h = Hq_dot.to(in_dtype)
            Hq_inter = 1.0

        # Hi = relu(Hq @ Hk^T): (BLOCK_T, BLOCK_S). FP8 MFMA in FP8 mode.
        Hi_raw = tl.dot(Hq_h, Hk_T)
        Hi = tl.maximum(Hi_raw, 0.0)

        # weights[b, h_outer, h_idx, t]: contiguous BLOCK_T-vector.
        w_ptrs = weights_ptr + w_base + h_idx * T_t + rt
        w_i = tl.load(w_ptrs, mask=rt_mask, other=0.0)

        if IS_FP8:
            acc += Hi * (Hq_inter * w_i)[:, None]
        else:
            acc += Hi * w_i[:, None]

    # Apply combined per-tensor scale + carried-out intermediate scales.
    scale = tl.load(scale_ptr)
    if IS_FP8:
        acc = acc * (scale * Cq_inter * Hk_inter)
    else:
        acc = acc * scale

    # Store O tile: (BLOCK_T, BLOCK_S). O has shape (B, oH, T, S).
    o_base = b * (oH * T_t * T_s) + h_outer * (T_t * T_s)
    o_ptrs = O_ptr + o_base + rt[:, None] * T_s + rs[None, :]
    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty),
             mask=rt_mask[:, None] & rs_mask[None, :])


# --- JAX primitive ---------------------------------------------------------------

_indexer_p = extend_core.Primitive("te_indexer_triton")
_indexer_p.multiple_results = True


_FP8_DTYPES = frozenset([
    jnp.dtype("float8_e4m3fn"),
    jnp.dtype("float8_e5m2"),
    jnp.dtype("float8_e4m3fnuz"),
    jnp.dtype("float8_e5m2fnuz"),
])


def _is_fp8_dtype(dt):
    return jnp.dtype(dt) in _FP8_DTYPES


@_indexer_p.def_abstract_eval
def _indexer_abstract(Q, K, W_uq, W_dq, W_k, weights, scale, *, out_dtype):
    del W_uq, W_dq, W_k, weights, scale
    B, oH, T_t, _ = Q.shape
    _, _, T_s, _ = K.shape
    return [core.ShapedArray((B, oH, T_t, T_s), out_dtype)]


_indexer_p.def_impl(functools.partial(xla.apply_primitive, _indexer_p))


def _indexer_lowering(ctx, Q, K, W_uq, W_dq, W_k, weights, scale, *, out_dtype):
    del out_dtype  # baked into the output aval
    Q_aval = ctx.avals_in[0]
    K_aval = ctx.avals_in[1]
    W_uq_aval = ctx.avals_in[2]
    B, oH, T_t, d = Q_aval.shape
    T_s = K_aval.shape[2]
    H, d_c, d_i = W_uq_aval.shape

    is_fp8 = _is_fp8_dtype(Q_aval.dtype)

    # Per-config grid: BLOCK_T/BLOCK_S come from the autotuned config kwargs
    # (or fall back to a sensible default if autotune is not active).
    def grid_fn(merged_kwargs):
        bt = merged_kwargs.get("BLOCK_T", 128)
        bs = merged_kwargs.get("BLOCK_S", 64)
        return (B * oH, triton.cdiv(T_t, bt), triton.cdiv(T_s, bs))

    return triton_call_lowering(
        ctx,
        _indexer_kernel,
        Q,
        K,
        W_uq,
        W_dq,
        W_k,
        weights,
        scale,
        grid=grid_fn,
        num_warps=4,
        num_stages=1,
        constexprs={
            "B": B,
            "oH": oH,
            "T_t": T_t,
            "T_s": T_s,
            "d": d,
            "d_c": d_c,
            "H": H,
            "d_i": d_i,
            "IS_FP8": is_fp8,
        },
    )


mlir.register_lowering(_indexer_p, _indexer_lowering, platform="rocm")
mlir.register_lowering(_indexer_p, _indexer_lowering, platform="cuda")


def indexer_fused_triton(
    Q,
    K,
    W_uq,
    W_dq,
    W_k,
    weights,
    *,
    scale_q=None,
    scale_k=None,
    scale_wq=None,
    scale_wd=None,
    scale_wk=None,
    out_dtype=None,
):
    """Raw-Triton low-rank indexer (BHSD).

    Args:
        Q:       (B, oH, T, d)          high-precision (bf16/fp32) or FP8 e4m3
        K:       (B, oH, S, d)          must match Q's dtype
        W_uq:    (H, d_c, d_i)          up-projection; must match Q's dtype
        W_dq:    (d, d_c)               down-projection; must match Q's dtype
        W_k:     (d, d_i)               key projection; must match Q's dtype
        weights: (B, oH, H, T)          high-precision regardless of Q dtype
        scale_q, scale_k, scale_wq, scale_wd, scale_wk:
                 per-tensor fp32 dequant scales. All five required when Q is FP8.
        out_dtype: dtype of the output O. Defaults to Q.dtype for non-FP8 and
                 weights.dtype (typically bf16) for FP8.

        BLOCK_T / BLOCK_S / BLOCK_D / num_warps / num_stages are autotuned at
        first invocation per (IS_FP8, d, d_c, H, d_i) key.

    Returns:
        O of shape (B, oH, T, S)
    """
    if Q.ndim != 4 or K.ndim != 4 or weights.ndim != 4:
        raise ValueError(
            "indexer_fused_triton expects rank-4 BHSD Q, K, weights. Got "
            f"Q.shape={Q.shape}, K.shape={K.shape}, weights.shape={weights.shape}."
        )
    B, oH, T_t, d = Q.shape
    Bk, oHk, T_s, dk = K.shape
    H, d_c_uq, d_i = W_uq.shape
    d_dq, d_c_dq = W_dq.shape
    d_wk, d_i_wk = W_k.shape
    Bw, oHw, Hw, T_w = weights.shape
    if (Bk, oHk) != (B, oH):
        raise ValueError(f"(B,oH) mismatch: Q has {(B, oH)}, K has {(Bk, oHk)}")
    if not (d == dk == d_dq == d_wk):
        raise ValueError(f"d mismatch across Q/K/W_dq/W_k: {d}, {dk}, {d_dq}, {d_wk}")
    if d_c_uq != d_c_dq:
        raise ValueError(f"d_c mismatch: W_uq has {d_c_uq}, W_dq has {d_c_dq}")
    if d_i != d_i_wk:
        raise ValueError(f"d_i mismatch: W_uq has {d_i}, W_k has {d_i_wk}")
    if (Bw, oHw, Hw, T_w) != (B, oH, H, T_t):
        raise ValueError(
            f"weights shape {weights.shape} does not match expected "
            f"(B={B}, oH={oH}, H={H}, T={T_t})"
        )

    is_fp8 = _is_fp8_dtype(Q.dtype)
    if is_fp8:
        for nm, t in (("K", K), ("W_uq", W_uq), ("W_dq", W_dq), ("W_k", W_k)):
            if t.dtype != Q.dtype:
                raise ValueError(
                    f"FP8 mode requires Q/K/W_uq/W_dq/W_k all match dtype; "
                    f"Q is {Q.dtype} but {nm} is {t.dtype}."
                )
        scales = (scale_q, scale_k, scale_wq, scale_wd, scale_wk)
        if any(s is None for s in scales):
            raise ValueError(
                "FP8 mode requires scale_q, scale_k, scale_wq, scale_wd, scale_wk."
            )
        scale_combined = jnp.asarray(
            jnp.float32(scale_q) * jnp.float32(scale_k)
            * jnp.float32(scale_wq) * jnp.float32(scale_wd)
            * jnp.float32(scale_wk),
            dtype=jnp.float32,
        )
        if out_dtype is None:
            out_dtype = weights.dtype
    else:
        scale_combined = jnp.asarray(1.0, dtype=jnp.float32)
        if out_dtype is None:
            out_dtype = Q.dtype

    return _indexer_p.bind(
        Q,
        K,
        W_uq,
        W_dq,
        W_k,
        weights,
        scale_combined,
        out_dtype=jnp.dtype(out_dtype),
    )[0]


# --- Score+ReLU+H-reduce fused kernel (hybrid backend) -------------------------
#
# Inputs are *already projected*: Hq, Hk, W_o all come from upstream einsum
# calls (hipBLASLt). This kernel does only the score matmul, the relu, and the
# per-token-per-head weighted sum over H — the pieces that have no efficient
# einsum/HLO equivalent because they'd require materializing the (B, oH, T, H, S)
# pre-relu score tensor in HBM. By fusing them in registers, we eliminate that
# round-trip entirely.

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

    b = pid_bh // oH
    h_outer = pid_bh % oH

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


def score_reduce_triton(Hq, Hk, W_o, *, out_dtype=None):
    """Triton fused score-matmul + relu + per-(t, h) weighted H-reduction.

    Replaces the pattern:

        scores = relu(jnp.einsum("...thi,...si->...ths", Hq, Hk))   # never write
        O      = jnp.einsum("...ths,...th->...ts", scores, W_o)

    with a single kernel that holds the per-head score tile in registers,
    avoiding the (B, oH, T, H, S) HBM round-trip that an einsum+XLA chain
    pays (the dominant cost in profile_indexer's einsum baseline).

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

    return _score_reduce_p.bind(
        Hq, Hk, W_o, out_dtype=jnp.dtype(out_dtype)
    )[0]


# --- Top-K fused variant -------------------------------------------------------
#
# FlashAttention-style: each CTA owns one (b, h, t-tile) and serializes over s
# tiles, maintaining a running per-row top-k of the score matrix. Output is
# (B, H, T_t, k) values + (B, H, T_t, k) int32 indices — never materializes the
# full (T_t, T_s) score tensor.
#
# Top-k merge: pack (val_bits << 32) | idx_u32 into uint64, build (BLOCK_T,
# k+BLOCK_S) via gather+where (tl.cat is 1D-only on this Triton), sort
# descending, take first k. Constraints: k pow2, k+block_s pow2.
#
# Score values are post-ReLU (≥ 0) so the fp32 bit pattern sorts correctly as
# uint32. Init sentinel: (val=0.0, idx=0xFFFFFFFF) — real positive values
# displace it; rows with fewer than k positive scores trail with idx=-1.

_DEFAULT_K = 64


@triton.jit
def _indexer_topk_kernel(
    Q_ptr,
    K_ptr,
    W_q_ptr,
    W_k_ptr,
    weights_ptr,
    O_v_ptr,
    O_i_ptr,
    B: tl.constexpr,
    H: tl.constexpr,
    T_t: tl.constexpr,
    T_s: tl.constexpr,
    d: tl.constexpr,
    I: tl.constexpr,
    d_i: tl.constexpr,
    K_TOPK: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_S: tl.constexpr,
    KS_SUM: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_t = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    rt = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    rd = tl.arange(0, d)
    rdi = tl.arange(0, d_i)
    rt_mask = rt < T_t

    q_base = b * (H * T_t * d) + h * (T_t * d)
    Q = tl.load(Q_ptr + q_base + rt[:, None] * d + rd[None, :],
                mask=rt_mask[:, None], other=0.0)

    Wk = tl.load(W_k_ptr + rd[:, None] * d_i + rdi[None, :])

    running_pack = tl.full((BLOCK_T, K_TOPK), 0xFFFFFFFF, dtype=tl.uint64)

    w_base = b * (H * T_t * I) + h * (T_t * I)
    k_base = b * (H * T_s * d) + h * (T_s * d)

    n_s_tiles = T_s // BLOCK_S
    for s_idx in range(n_s_tiles):
        s_off = s_idx * BLOCK_S
        rs = s_off + tl.arange(0, BLOCK_S)

        Kt = tl.load(K_ptr + k_base + rs[:, None] * d + rd[None, :])
        Hk = tl.dot(Kt, Wk).to(Q.dtype)

        acc = tl.zeros((BLOCK_T, BLOCK_S), dtype=tl.float32)
        for i in range(I):
            Wq_i = tl.load(W_q_ptr + i * (d * d_i) + rd[:, None] * d_i + rdi[None, :])
            Hq_i = tl.dot(Q, Wq_i).to(Q.dtype)
            Hi_raw = tl.dot(Hq_i, tl.trans(Hk))
            Hi = tl.maximum(Hi_raw, 0.0)
            w_i = tl.load(weights_ptr + w_base + rt * I + i, mask=rt_mask, other=0.0)
            acc += Hi * w_i[:, None]

        # Encode fp32 -> monotonic uint32 (radix-sort fp32 trick) so negative
        # acc values sort below positive ones.
        acc_bits = acc.to(tl.uint32, bitcast=True)
        acc_sext = (acc.to(tl.int32, bitcast=True) >> 31).to(tl.uint32)
        enc_mask = acc_sext | tl.cast(0x80000000, tl.uint32)
        acc_key = acc_bits ^ enc_mask                              # (BLOCK_T, BLOCK_S) u32
        tile_idx = rs.to(tl.uint32)
        tile_v_u = acc_key.to(tl.uint64)
        tile_i_u = tile_idx.to(tl.uint64)
        tile_pack = (tile_v_u << 32) | tile_i_u[None, :].broadcast_to((BLOCK_T, BLOCK_S))

        pos = tl.arange(0, KS_SUM)
        r_idx = tl.minimum(pos, K_TOPK - 1)
        t_idx = tl.maximum(pos.to(tl.int32) - K_TOPK, 0).to(tl.int32)
        r_ext = tl.gather(running_pack, r_idx[None, :].broadcast_to((BLOCK_T, KS_SUM)), axis=1)
        t_ext = tl.gather(tile_pack, t_idx[None, :].broadcast_to((BLOCK_T, KS_SUM)), axis=1)
        combined = tl.where((pos < K_TOPK)[None, :].broadcast_to((BLOCK_T, KS_SUM)),
                            r_ext, t_ext)

        running_pack = tl.topk(combined, K_TOPK, dim=1)

    # Decode monotonic uint32 key -> fp32 bits.
    out_key = (running_pack >> 32).to(tl.uint32)
    out_key_sext = (~out_key.to(tl.int32, bitcast=True) >> 31).to(tl.uint32)
    dec_mask = out_key_sext | tl.cast(0x80000000, tl.uint32)
    out_bits = out_key ^ dec_mask
    out_vals_fp32 = out_bits.to(tl.float32, bitcast=True)
    out_idxs = (running_pack & 0xFFFFFFFF).to(tl.uint32).to(tl.int32)

    rk = tl.arange(0, K_TOPK)
    o_base = b * (H * T_t * K_TOPK) + h * (T_t * K_TOPK)
    tl.store(O_v_ptr + o_base + rt[:, None] * K_TOPK + rk[None, :],
             out_vals_fp32.to(O_v_ptr.dtype.element_ty),
             mask=rt_mask[:, None])
    tl.store(O_i_ptr + o_base + rt[:, None] * K_TOPK + rk[None, :],
             out_idxs, mask=rt_mask[:, None])


_indexer_topk_p = extend_core.Primitive("te_indexer_topk_triton")
_indexer_topk_p.multiple_results = True


@_indexer_topk_p.def_abstract_eval
def _indexer_topk_abstract(Q, K, W_q, W_k, weights, *,
                           k, block_t, block_s, num_warps, num_stages):
    del W_q, W_k, weights, block_t, block_s, num_warps, num_stages
    B, H, T_t, _ = Q.shape
    return [
        core.ShapedArray((B, H, T_t, k), Q.dtype),
        core.ShapedArray((B, H, T_t, k), jnp.int32),
    ]


_indexer_topk_p.def_impl(functools.partial(xla.apply_primitive, _indexer_topk_p))


def _indexer_topk_lowering(ctx, Q, K, W_q, W_k, weights, *,
                           k, block_t, block_s, num_warps, num_stages):
    Q_aval = ctx.avals_in[0]
    K_aval = ctx.avals_in[1]
    W_q_aval = ctx.avals_in[2]
    B, H, T_t, d = Q_aval.shape
    T_s = K_aval.shape[2]
    I, _, d_i = W_q_aval.shape

    grid = (B * H, triton.cdiv(T_t, block_t))

    return triton_call_lowering(
        ctx,
        _indexer_topk_kernel,
        Q,
        K,
        W_q,
        W_k,
        weights,
        grid=grid,
        num_warps=num_warps,
        num_stages=num_stages,
        constexprs={
            "B": B,
            "H": H,
            "T_t": T_t,
            "T_s": T_s,
            "d": d,
            "I": I,
            "d_i": d_i,
            "K_TOPK": k,
            "BLOCK_T": block_t,
            "BLOCK_S": block_s,
            "KS_SUM": k + block_s,
        },
    )


mlir.register_lowering(_indexer_topk_p, _indexer_topk_lowering, platform="rocm")
mlir.register_lowering(_indexer_topk_p, _indexer_topk_lowering, platform="cuda")


def _is_pow2(n):
    return n > 0 and (n & (n - 1)) == 0


def indexer_fused_topk_triton(
    Q,
    K,
    W_q,
    W_k,
    weights,
    *,
    k: int = _DEFAULT_K,
    block_t: int = 128,
    block_s: int = 64,
    num_warps: int = 4,
    num_stages: int = 1,
):
    """Fused indexer + per-row top-k along T_s. Returns (vals, idxs).

    vals: (B, H, T_t, k) Q.dtype  — descending top-k post-ReLU scores
    idxs: (B, H, T_t, k) int32    — corresponding s positions in [0, T_s)

    Constraints:
        * Q, K, weights are rank-4 BHSD.
        * T_s % block_s == 0 (no masking inside inner loop).
        * k and (k + block_s) are powers of 2 (tl.sort and tl.arange).
    """
    if Q.ndim != 4 or K.ndim != 4 or weights.ndim != 4:
        raise ValueError(
            "indexer_fused_topk_triton expects rank-4 BHSD Q, K, weights. Got "
            f"Q.shape={Q.shape}, K.shape={K.shape}, weights.shape={weights.shape}."
        )
    B, H, T_t, d = Q.shape
    Bk, Hk, T_s, dk = K.shape
    I, d2, d_i = W_q.shape
    d3, d_i_k = W_k.shape
    Bw, Hw, T_t_w, I_w = weights.shape
    if (Bk, Hk) != (B, H):
        raise ValueError(f"(B,H) mismatch: Q has {(B, H)}, K has {(Bk, Hk)}")
    if not (d == dk == d2 == d3):
        raise ValueError(f"d mismatch across Q/K/W_q/W_k: {d}, {dk}, {d2}, {d3}")
    if d_i != d_i_k:
        raise ValueError(f"d_i mismatch: W_q has {d_i}, W_k has {d_i_k}")
    if (Bw, Hw, T_t_w, I_w) != (B, H, T_t, I):
        raise ValueError(
            f"weights shape {weights.shape} does not match expected "
            f"(B={B}, H={H}, T_t={T_t}, I={I})"
        )
    if k > T_s:
        raise ValueError(f"k={k} exceeds T_s={T_s}")

    block_t = min(block_t, T_t)
    block_s = min(block_s, T_s)

    if T_s % block_s != 0:
        raise ValueError(
            f"T_s={T_s} must be divisible by block_s={block_s} (kernel doesn't "
            "mask invalid s positions in the inner loop)."
        )
    if not _is_pow2(k):
        raise ValueError(f"k={k} must be a power of 2 (tl.arange requirement)")
    if not _is_pow2(k + block_s):
        raise ValueError(
            f"k + block_s = {k + block_s} must be a power of 2 "
            f"(k={k}, block_s={block_s})"
        )

    return _indexer_topk_p.bind(
        Q, K, W_q, W_k, weights,
        k=k,
        block_t=block_t,
        block_s=block_s,
        num_warps=num_warps,
        num_stages=num_stages,
    )
