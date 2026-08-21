# Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
"""Indexer op (forward only), bf16 inputs.

The op runs a hybrid backend: einsum projections (C_q, H_q, H_k, W_o) —
which lower to hipBLASLt bf16 GEMMs — followed by a fused Triton kernel that
does score+relu+H-reduction in registers. This avoids materializing the
(B, oH, T, H, S) pre-relu score tensor in HBM.

Functional entry point: ``indexer(Q, K, W_uq, W_dq, W_k, W_w)``.
User-facing Flax module: :class:`LightningIndexer`, which owns the projection
weights and delegates to ``indexer`` / ``indexer_topk``.

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
from typing import Optional

import jax
import jax.numpy as jnp
from flax import linen as nn


def _indexer_projections(Q, K, W_uq, W_dq, W_k, W_w):
    """Low-rank indexer projections shared by the score and top-k paths.

    Returns (H_q, H_k, W_o) with shapes
    (..., T, H, d_i), (..., S, d_i), (..., T, H).
    """
    C_q = jnp.einsum("...td,dc->...tc", Q, W_dq)
    H_q = jnp.einsum("...tc,hci->...thi", C_q, W_uq)
    H_k = jnp.einsum("...sd,di->...si", K, W_k)
    W_o = jnp.einsum("...td,dh->...th", Q, W_w)
    return H_q, H_k, W_o


def _indexer_impl_hybrid(Q, K, W_uq, W_dq, W_k, W_w, out_dtype=None):
    """Einsum projections + Triton score-relu-reduce.

    Runs the four projections (which lower to hipBLASLt bf16 GEMMs), then
    hands Hq / Hk / W_o to a fused Triton kernel that does
    score+relu+H-reduction in registers — eliminating the
    (B, oH, T, H, S) pre-relu-score HBM round-trip a pure-einsum path pays.
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


@functools.partial(jax.jit, static_argnames=("out_dtype",))
def indexer(Q, K, W_uq, W_dq, W_k, weights, *, out_dtype=None):
    """Low-rank lightning-indexer (bf16), hybrid Triton backend.

    Args:
        Q:       (..., T, d)            hidden state (per token)
        K:       (..., S, d)            key hidden state
        W_uq:    (H, d_c, d_i)          up-projection: d_c -> d_i (per head)
        W_dq:    (d, d_c)               down-projection: d -> d_c
        W_k:     (d, d_i)               key projection
        weights: (d, H)                 learnable output-weight projection
                                        (W_o = Q @ weights inside the impl)
        out_dtype: output dtype override (defaults to Q.dtype).

    Returns:
        O of shape (..., T, S).
    """
    return _indexer_impl_hybrid(Q, K, W_uq, W_dq, W_k, weights, out_dtype=out_dtype)


class LightningIndexer(nn.Module):  # pylint: disable=too-few-public-methods
    """Lightning-indexer Flax module — the user-facing indexer API.

    Owns the low-rank indexer projection weights (``W_dq``, ``W_uq``, ``W_k``,
    ``W_w``) and delegates to the functional :func:`indexer` / :func:`indexer_topk`
    ops. Weight shapes mirror :func:`indexer`'s ``Args`` and are inferred from the
    trailing hidden dimension ``d`` of ``Q`` at call time.

    Parameters
    ----------
    num_heads : int
        Number of indexer-internal heads (``H``).
    d_c : int
        Down-projection rank (``d -> d_c``).
    d_i : int
        Inner head dimension (``d_i``).
    topk : Optional[int], default ``None``
        If set, :meth:`__call__` returns the fused top-``k`` indices
        (``(..., T, k)`` int32) via :func:`indexer_topk`, and ``out_dtype`` is
        ignored (top-k always uses the fused Triton kernel).
        If ``None``, :meth:`__call__` returns the full score tensor
        ``(..., T, S)`` (hybrid Triton backend).
    out_dtype : Optional[jnp.dtype]
        Output dtype override; defaults to ``Q.dtype``. Unused when ``topk`` is set.
    dtype : Optional[jnp.dtype]
        Parameter dtype. Defaults to the input dtype.
    """

    num_heads: int
    d_c: int
    d_i: int
    topk: Optional[int] = None
    out_dtype: Optional[jnp.dtype] = None
    dtype: Optional[jnp.dtype] = None

    @nn.compact
    def __call__(self, Q: jax.Array, K: jax.Array) -> jax.Array:
        """Run the indexer on ``Q`` / ``K``.

        Args:
            Q: ``(..., T, d)`` query-side hidden state.
            K: ``(..., S, d)`` key-side hidden state.

        Returns:
            ``(..., T, S)`` scores if ``topk is None``, else ``(..., T, k)``
            int32 top-k indices (in descending score order).
        """
        d = Q.shape[-1]
        param_dtype = self.dtype if self.dtype is not None else Q.dtype
        init = nn.initializers.variance_scaling(1.0, "fan_in", "truncated_normal")

        W_dq = self.param("W_dq", init, (d, self.d_c), param_dtype)
        W_uq = self.param("W_uq", init, (self.num_heads, self.d_c, self.d_i), param_dtype)
        W_k = self.param("W_k", init, (d, self.d_i), param_dtype)
        W_w = self.param("W_w", init, (d, self.num_heads), param_dtype)

        if self.topk is not None:
            return indexer_topk(Q, K, W_uq, W_dq, W_k, W_w, k=self.topk)
        return indexer(Q, K, W_uq, W_dq, W_k, W_w, out_dtype=self.out_dtype)
