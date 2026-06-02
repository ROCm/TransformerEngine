# Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
"""Indexer op (forward only), bf16 inputs.

Two canonical backends:
  * ``"reference"`` — pure ``jnp.einsum``. Materializes the
    (B, oH, T, H, S) pre-relu score tensor in HBM via hipBLASLt.
  * ``"hybrid"`` — same einsum projections (C_q, H_q, H_k, W_o) followed
    by a fused Triton kernel that does score+relu+H-reduction in
    registers. Avoids the score-tensor HBM round-trip that dominates the
    reference path.

Top-level entry point: ``indexer(Q, K, W_uq, W_dq, W_k, W_w, *, backend=...)``.

Math (low-rank form: Q is hidden state; query heads are produced by a
down-projection (d -> d_c) followed by an up-projection (d_c -> H * d_i);
output weights are produced from Q via a learnable d -> H projection):

    C_q = Q @ W_dq                                           # (..., T, d_c)
    H_q = einsum("...tc,hci->...thi", C_q, W_uq)             # (..., T, H, d_i)
    H_k = K @ W_k                                             # (..., S, d_i)
    W_o = Q @ W_w                                             # (..., T, H)
    H   = relu(einsum("...thi,...si->...ths", H_q, H_k))      # (..., T, H, S)
    O   = einsum("...ths,...th->...ts", H, W_o)               # (..., T, S)
"""

import functools

import jax
import jax.numpy as jnp


def _indexer_projections(Q, K, W_uq, W_dq, W_k, W_w):
    """Low-rank indexer projections shared by every backend.

    Returns (H_q, H_k, W_o) with shapes
    (..., T, H, d_i), (..., S, d_i), (..., T, H).
    """
    C_q = jnp.einsum("...td,dc->...tc", Q, W_dq)
    H_q = jnp.einsum("...tc,hci->...thi", C_q, W_uq)
    H_k = jnp.einsum("...sd,di->...si", K, W_k)
    W_o = jnp.einsum("...td,dh->...th", Q, W_w)
    return H_q, H_k, W_o


def _indexer_impl_reference(Q, K, W_uq, W_dq, W_k, W_w, out_dtype=None):
    """
    Q       [..., T, d]
    K       [..., S, d]
    W_dq    [d, d_c]
    W_uq    [H, d_c, d_i]
    W_k     [d, d_i]
    W_w     [..., d, H]    # leading dims must match Q's
    """
    H_q, H_k, W_o = _indexer_projections(Q, K, W_uq, W_dq, W_k, W_w)
    H = jax.nn.relu(jnp.einsum("...thi,...si->...ths", H_q, H_k))  # (..., T, H, S)
    O = jnp.einsum("...ths,...th->...ts", H, W_o)                  # (..., T, S)
    if out_dtype is not None:
        O = O.astype(out_dtype)
    return O


def _indexer_impl_hybrid(Q, K, W_uq, W_dq, W_k, W_w, out_dtype=None):
    """Einsum projections + Triton score-relu-reduce.

    Mirrors ``_indexer_impl_reference`` for the four projections (which
    lower to hipBLASLt bf16 GEMMs), then hands Hq / Hk / W_o to a fused
    Triton kernel that does score+relu+H-reduction in registers —
    eliminating the (B, oH, T, H, S) pre-relu-score HBM round-trip the
    pure-einsum path pays.
    """
    from transformer_engine.jax.triton_extensions.indexer import score_reduce_triton

    H_q, H_k, W_o = _indexer_projections(Q, K, W_uq, W_dq, W_k, W_w)
    return score_reduce_triton(H_q, H_k, W_o,
                               out_dtype=out_dtype if out_dtype else Q.dtype)


@functools.partial(jax.jit, static_argnames=("k",))
def indexer_topk(Q, K, W_uq, W_dq, W_k, weights, *, k):
    """Lightning-indexer + top-k (fused).

    Same projections as ``indexer()`` (reference math), then a single Triton
    kernel that computes the score row, ReLU, weighted H-reduction, and
    streaming top-k all in one pass — the (B, oH, T_t, T_s) score matrix is
    never materialized.

    Args:
        Q, K, W_uq, W_dq, W_k, weights: same as ``indexer()``.
        k: number of top scores to return per (B, oH, T_t) row.
           Must be a power of 2 and <= S.

    Returns:
        Topk_idx: (..., T_t, k) int32 — top-k indices into the S axis,
        in descending score order.
    """
    from transformer_engine.jax.triton_extensions.indexer import score_topk_triton
    H_q, H_k, W_o = _indexer_projections(Q, K, W_uq, W_dq, W_k, weights)
    return score_topk_triton(H_q, H_k, W_o, k=k)


@functools.partial(jax.jit, static_argnames=("backend", "out_dtype"))
def indexer(Q, K, W_uq, W_dq, W_k, weights, *, out_dtype=None, backend="reference"):
    """Low-rank lightning-indexer (bf16).

    Args:
        Q:       (..., T, d)            hidden state (per token)
        K:       (..., S, d)            key hidden state
        W_uq:    (H, d_c, d_i)          up-projection: d_c -> d_i (per head)
        W_dq:    (d, d_c)               down-projection: d -> d_c
        W_k:     (d, d_i)               key projection
        weights: (d, H)                 learnable output-weight projection
                                        (W_o = Q @ weights inside the impl)
        out_dtype: output dtype override (defaults to Q.dtype).
        backend: "reference" (pure einsum) or "hybrid" (einsum projections
                 + Triton score-relu-reduce kernel).

    Returns:
        O of shape (..., T, S).
    """
    if backend == "reference":
        return _indexer_impl_reference(Q, K, W_uq, W_dq, W_k, weights, out_dtype=out_dtype)
    if backend == "hybrid":
        return _indexer_impl_hybrid(Q, K, W_uq, W_dq, W_k, weights, out_dtype=out_dtype)
    raise ValueError(
        f"unknown backend {backend!r}; expected 'reference' or 'hybrid'"
    )
