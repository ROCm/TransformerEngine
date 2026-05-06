# Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Pallas kernel for the fused indexer (BHSD layout).

Reference math (see transformer_engine.jax.indexer._indexer_impl_reference):

    H_q = einsum("bhtd,dij->bhtij",     Q, W_q)            # (B, H, T_t, I, d_i)
    H_k = K @ W_k                                           # (B, H, T_s, d_i)
    H   = relu(einsum("bhtij,bhsj->bhtis", H_q, H_k))       # (B, H, T_t, I, T_s)
    O   = einsum("bhtis,bhti->bhts", H, weights)            # (B, H, T_t, T_s)

``weights`` is precomputed per-(token, indexer-head) weighting with shape
matching Q's leading dims (B, H, T_t, I). It plays the role DeepSeek's
``weights_proj(x)`` plays in their lightning-indexer: a learned, data-
dependent per-head weight, not a static parameter.

The launcher (``indexer_fused`` at the bottom of this file) wires up grid
size, BlockSpecs, and shape inference. The kernel body itself
(``_indexer_pallas_kernel_body``) is the part you fill in.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


# Preferred tile sizes (used when the LDS budget allows). Pallas-Triton stages
# Q and K tiles to LDS via BlockSpec, so the dominant LDS cost is
# (BLOCK_T + BLOCK_S) * d * dtype_bytes. Auto-shrink keeps that under
# _LDS_BUDGET_BYTES per workgroup.
_PREFERRED_BLOCK_T = 128
_PREFERRED_BLOCK_S = 64

# gfx950 reports 160 KB LDS; leave ~10% headroom for compiler-staged
# intermediates (Hk, Hq_i, accumulators).
_LDS_BUDGET_BYTES = 144 * 1024
_MIN_BLOCK = 16  # Triton requires matmul dims >= 16

_FP8_DTYPES = frozenset([
    jnp.dtype("float8_e4m3fn"),
    jnp.dtype("float8_e5m2"),
    jnp.dtype("float8_e4m3fnuz"),
    jnp.dtype("float8_e5m2fnuz"),
])


def _is_fp8_dtype(dt):
    return jnp.dtype(dt) in _FP8_DTYPES


def _pick_tiles(T_t, T_s, d, dtype):
    """Return (BLOCK_T, BLOCK_S) that fit the LDS budget for Q+K staging.

    Halve BLOCK_T first (less critical for inner-loop reuse), then BLOCK_S.
    """
    elem_bytes = jnp.dtype(dtype).itemsize
    bt = min(_PREFERRED_BLOCK_T, T_t)
    bs = min(_PREFERRED_BLOCK_S, T_s)

    def cost(bt, bs):
        return (bt + bs) * d * elem_bytes

    while cost(bt, bs) > _LDS_BUDGET_BYTES and bt > _MIN_BLOCK:
        bt //= 2
    while cost(bt, bs) > _LDS_BUDGET_BYTES and bs > _MIN_BLOCK:
        bs //= 2
    return bt, bs


def _estimate_lds_bytes(BLOCK_T, BLOCK_S, d, d_i, dtype):
    """Worst-case LDS estimate. The dominant cost is usually K_tile + W_q[i]
    slice (Pallas-Triton stages the per-iteration W_q[i] of shape (d, d_i)).
    """
    elem_bytes = jnp.dtype(dtype).itemsize
    k_tile = BLOCK_S * d * elem_bytes
    q_tile = BLOCK_T * d * elem_bytes
    w_q_slice = d * d_i * elem_bytes
    # The two pairs that have actually been observed empirically:
    return max(k_tile + w_q_slice, q_tile + w_q_slice)


class PallasIndexerInfeasible(RuntimeError):
    """Raised when no valid (BLOCK_T, BLOCK_S) fits the LDS budget for the
    given (d, d_i, dtype). The W_q[i] slice (size d*d_i*dtype_bytes) is the
    typical culprit; mitigation requires d-tiling the inner matmul."""


def _dot_fp32(a, b):
    """`jnp.dot` with the fp32 accumulator made explicit.

    Without `preferred_element_type`, JAX promotion picks the input dtype as
    the dot output dtype. For FP8 inputs that means the accumulated dot is
    clamped to fp8 max (~448) BEFORE the fp32 cast — so any real workload
    silently saturates. Force fp32 accumulation everywhere.
    """
    return jax.lax.dot_general(
        a, b, (((a.ndim - 1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )


def _make_kernel_body(BLOCK_T, BLOCK_S, d, I, d_i, is_fp8):
    """Closure that bakes the static shape constants into the kernel body.

    Pallas kernel bodies trace under jit, so values referenced by Python-level
    control flow (``range(I)`` etc.) must be static. The simplest way to make
    them static is to capture them in a closure here.

    For FP8 inputs, the outer two dots (K@W_k, Q@W_q[i]) consume FP8 directly
    via _dot_fp32; their fp32 outputs are downcast to bf16 for the inner
    (Hq_i @ Hk^T) matmul. The combined per-tensor dequant scale is applied
    to the fp32 accumulator at the very end; ReLU commutes with positive
    scaling so this is exact.
    """
    inter_dtype = jnp.bfloat16 if is_fp8 else None  # None = preserve dtype

    def _indexer_pallas_kernel_body(
        Q_ref,        # (1, 1, BLOCK_T, d)        - one (b, h) slice, T_t-tile
        K_ref,        # (1, 1, BLOCK_S, d)        - one (b, h) slice, T_s-tile
        W_q_ref,      # (I, d, d_i)               - whole tensor, replicated
        W_k_ref,      # (d, d_i)                  - whole tensor, replicated
        weights_ref,  # (1, 1, BLOCK_T, I)        - one (b, h) slice, T_t-tile
        scale_ref,    # (1,)                       - combined fp32 scale
        O_ref,        # (1, 1, BLOCK_T, BLOCK_S)  - one tile of the output
    ):
        """
        Compute one (BLOCK_T, BLOCK_S) tile of O for one (b, h).
        """
        Q  = Q_ref[0, 0]                              # (BLOCK_T, d)
        K  = K_ref[0, 0]                              # (BLOCK_S, d)
        Wk = W_k_ref[...]                             # (d, d_i)
        Hk = _dot_fp32(K, Wk)                         # (BLOCK_S, d_i)
        if inter_dtype is not None:
            Hk = Hk.astype(inter_dtype)

        acc = jnp.zeros((BLOCK_T, BLOCK_S), dtype=jnp.float32)
        for i in range(I):
            Wq_i = W_q_ref[i]                         # (d, d_i)
            Hq_i = _dot_fp32(Q, Wq_i)                 # (BLOCK_T, d_i)
            if inter_dtype is not None:
                Hq_i = Hq_i.astype(inter_dtype)
            Hi   = jax.nn.relu(_dot_fp32(Hq_i, Hk.T)) # (BLOCK_T, BLOCK_S)
            w_i  = weights_ref[0, 0, :, i]            # (BLOCK_T,)
            acc  = acc + Hi * w_i[:, None]

        acc = acc * scale_ref[0]
        O_ref[0, 0] = acc.astype(O_ref.dtype)

    return _indexer_pallas_kernel_body


def indexer_fused(
    Q, K, W_q, W_k, weights,
    *,
    scale_q=None, scale_k=None, scale_wq=None, scale_wk=None,
    out_dtype=None,
):
    """Pallas-backed fused indexer. Strict BHSD.

    Args:
        Q:       (B, H, T_t, d)         high-precision (bf16/fp32) or FP8 e4m3
        K:       (B, H, T_s, d)         must match Q's dtype
        W_q:     (I, d, d_i)             must match Q's dtype
        W_k:     (d, d_i)                must match Q's dtype
        weights: (B, H, T_t, I)         high-precision regardless of Q dtype
        scale_q, scale_k, scale_wq, scale_wk:
                 per-tensor fp32 dequant scales. Required when Q is FP8.
        out_dtype: defaults to Q.dtype for non-FP8, weights.dtype for FP8.

    Returns:
        O:   (B, H, T_t, T_s)
    """
    if Q.ndim != 4 or K.ndim != 4 or weights.ndim != 4:
        raise ValueError(
            f"indexer_fused (pallas) expects rank-4 BHSD Q, K and weights. Got "
            f"Q.shape={Q.shape}, K.shape={K.shape}, weights.shape={weights.shape}. "
            "Reshape (or add singleton head/batch axes) before calling the fused path."
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

    is_fp8 = _is_fp8_dtype(Q.dtype)
    if is_fp8:
        for nm, t in (("K", K), ("W_q", W_q), ("W_k", W_k)):
            if t.dtype != Q.dtype:
                raise ValueError(
                    f"FP8 mode requires Q/K/W_q/W_k all match dtype; "
                    f"Q is {Q.dtype} but {nm} is {t.dtype}."
                )
        if any(s is None for s in (scale_q, scale_k, scale_wq, scale_wk)):
            raise ValueError(
                "FP8 mode requires scale_q, scale_k, scale_wq, scale_wk."
            )
        scale_combined = jnp.asarray(
            jnp.float32(scale_q) * jnp.float32(scale_k)
            * jnp.float32(scale_wq) * jnp.float32(scale_wk),
            dtype=jnp.float32,
        ).reshape((1,))
        if out_dtype is None:
            out_dtype = weights.dtype
    else:
        scale_combined = jnp.asarray(1.0, dtype=jnp.float32).reshape((1,))
        if out_dtype is None:
            out_dtype = Q.dtype

    BLOCK_T, BLOCK_S = _pick_tiles(T_t, T_s, d, Q.dtype)
    lds = _estimate_lds_bytes(BLOCK_T, BLOCK_S, d, d_i, Q.dtype)
    if lds > _LDS_BUDGET_BYTES:
        raise PallasIndexerInfeasible(
            f"Pallas indexer infeasible for this config: estimated LDS "
            f"{lds // 1024} KB > budget {_LDS_BUDGET_BYTES // 1024} KB. "
            f"Dominant cost is W_q[i] slice = d*d_i*dtype = "
            f"{d * d_i * jnp.dtype(Q.dtype).itemsize // 1024} KB. "
            f"Mitigation: d-tile the inner matmul (not implemented). "
            f"For this config use the Triton backend instead."
        )

    grid = (B * H, pl.cdiv(T_t, BLOCK_T), pl.cdiv(T_s, BLOCK_S))

    # BlockSpecs: each input/output is sliced based on (program_id_0,
    # program_id_1, program_id_2). index_map returns the *block index* per
    # axis (Pallas multiplies by block_shape internally).
    def q_idx(bh, tt, ts):       return (bh // H, bh % H, tt, 0)
    def k_idx(bh, tt, ts):       return (bh // H, bh % H, ts, 0)
    def wq_idx(bh, tt, ts):      return (0, 0, 0)
    def wk_idx(bh, tt, ts):      return (0, 0)
    def weights_idx(bh, tt, ts): return (bh // H, bh % H, tt, 0)
    def scale_idx(bh, tt, ts):   return (0,)
    def o_idx(bh, tt, ts):       return (bh // H, bh % H, tt, ts)

    in_specs = [
        pl.BlockSpec(block_shape=(1, 1, BLOCK_T, d),  index_map=q_idx),
        pl.BlockSpec(block_shape=(1, 1, BLOCK_S, d),  index_map=k_idx),
        pl.BlockSpec(block_shape=(I, d, d_i),         index_map=wq_idx),
        pl.BlockSpec(block_shape=(d, d_i),            index_map=wk_idx),
        pl.BlockSpec(block_shape=(1, 1, BLOCK_T, I),  index_map=weights_idx),
        pl.BlockSpec(block_shape=(1,),                index_map=scale_idx),
    ]
    out_spec = pl.BlockSpec(
        block_shape=(1, 1, BLOCK_T, BLOCK_S),
        index_map=o_idx,
    )
    out_shape = jax.ShapeDtypeStruct((B, H, T_t, T_s), out_dtype)

    kernel_body = _make_kernel_body(BLOCK_T, BLOCK_S, d, I, d_i, is_fp8)

    return pl.pallas_call(
        kernel_body,
        grid=grid,
        in_specs=in_specs,
        out_specs=out_spec,
        out_shape=out_shape,
    )(Q, K, W_q, W_k, weights, scale_combined)
