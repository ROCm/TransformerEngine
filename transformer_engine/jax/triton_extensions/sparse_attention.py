# Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
"""Phase-2 scaffold for the fused sparse-attention Triton primitive.

The functional API `fused_sparse_attention_triton(query, key, value,
indexer_query, indexer_key, indexer_weights, *, k, ...)` is declared and
registered as a JAX primitive with abstract evaluation, but the kernel
body and MLIR lowering both raise NotImplementedError.

Purpose: lock the call signature so the DSA Flax module can dispatch to
this primitive via ``backend="fused"`` today, and the real kernel can
land later without any caller-side changes.

The composition path in ``transformer_engine.jax.sparse_attention``
(indexer + sparse mask + DotProductAttention) is the supported phase-1
implementation.
"""

import functools

import jax.numpy as jnp

from jax import core
from jax.extend import core as extend_core
from jax.interpreters import mlir, xla


_fused_sparse_attention_p = extend_core.Primitive("te_fused_sparse_attention_triton")
_fused_sparse_attention_p.multiple_results = False


@_fused_sparse_attention_p.def_abstract_eval
def _fused_sparse_attention_abstract(
    query, key, value, indexer_query, indexer_key, indexer_weights, *, k
):
    """Output has the same shape/dtype as ``query`` (BSHD layout assumed)."""
    del key, value, indexer_query, indexer_key, indexer_weights, k
    return core.ShapedArray(query.shape, query.dtype)


_fused_sparse_attention_p.def_impl(
    functools.partial(xla.apply_primitive, _fused_sparse_attention_p)
)


def _fused_sparse_attention_lowering_unavailable(ctx, *args, **kwargs):
    raise NotImplementedError(
        "fused_sparse_attention_triton is a phase-2 scaffold: the Triton kernel "
        "has not been implemented yet. Use backend='composition' in "
        "transformer_engine.jax.sparse_attention.deep_sparse_attention_core(...) "
        "for the working composition path."
    )


mlir.register_lowering(
    _fused_sparse_attention_p,
    _fused_sparse_attention_lowering_unavailable,
    platform="rocm",
)
mlir.register_lowering(
    _fused_sparse_attention_p,
    _fused_sparse_attention_lowering_unavailable,
    platform="cuda",
)


def fused_sparse_attention_triton(
    query,
    key,
    value,
    indexer_query,
    indexer_key,
    indexer_weights,
    *,
    k: int,
):
    """Fused indexer + sparse attention (phase-2 scaffold — raises NotImplementedError).

    Intended contract for the future fused kernel:

    Args:
        query:           (B, T_t, H, D)         attention queries (BSHD)
        key:             (B, T_s, H_kv, D)      attention keys
        value:           (B, T_s, H_kv, D)      attention values
        indexer_query:   (B, T_t, H_idx, d_i)   post-projection indexer Hq
        indexer_key:     (B, T_s, d_i)          post-projection indexer Hk
        indexer_weights: (B, T_t, H_idx)        post-projection indexer W_o
        k:               number of top-k key positions per query token

    Returns:
        Output of shape (B, T_t, H, D) — sparse attention output where each
        query attends only to its indexer-selected top-k key positions
        (intersected with the causal mask).

    The signature is intentionally minimal so phase-2 has room to grow it
    (e.g. window_size, attn_bias). Add kwargs only via the function
    signature — abstract_eval and lowering both already accept ``**kwargs``.
    """
    return _fused_sparse_attention_p.bind(
        query,
        key,
        value,
        indexer_query,
        indexer_key,
        indexer_weights,
        k=k,
    )
