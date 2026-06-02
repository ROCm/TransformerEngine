# Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
"""Correctness tests for the lightning-indexer JAX ops.

Ported from the in-module ``__main__`` smoke tests of
``transformer_engine.jax.sparse_attention.indexer``. The hybrid and top-k backends require
rank-4 ``(B, oH, T, d)`` inputs, so every leading shape here is length-2.
"""

import jax
import jax.numpy as jnp
import pytest

from transformer_engine.jax.sparse_attention.indexer import indexer, indexer_topk


def _indexer_inputs(B, oH, T_t, T_s, d, d_c, H, d_i, seed):
    keys = jax.random.split(jax.random.PRNGKey(seed), 6)
    Q = jax.random.normal(keys[0], (B, oH, T_t, d), dtype=jnp.bfloat16)
    K = jax.random.normal(keys[1], (B, oH, T_s, d), dtype=jnp.bfloat16)
    W_uq = jax.random.normal(keys[2], (H, d_c, d_i), dtype=jnp.bfloat16)
    W_dq = jax.random.normal(keys[3], (d, d_c), dtype=jnp.bfloat16)
    W_k = jax.random.normal(keys[4], (d, d_i), dtype=jnp.bfloat16)
    W_w = jax.random.normal(keys[5], (d, H), dtype=jnp.bfloat16)
    return Q, K, W_uq, W_dq, W_k, W_w


def _rel_err(actual, ref):
    actual = actual.astype(jnp.float32)
    ref = ref.astype(jnp.float32)
    return float(jnp.linalg.norm(actual - ref) / (jnp.linalg.norm(ref) + 1e-30))


@pytest.mark.parametrize("B,oH", [(2, 3), (1, 1), (1, 4)])
def test_hybrid_matches_reference(B, oH):
    """Hybrid Triton score-reduce matches the pure-einsum reference forward."""
    args = _indexer_inputs(B, oH, T_t=64, T_s=64, d=32, d_c=32, H=8, d_i=32, seed=100)
    o_ref = indexer(*args, backend="reference")
    o_hyb = indexer(*args, backend="hybrid")
    assert o_hyb.shape == o_ref.shape
    assert _rel_err(o_hyb, o_ref) < 5e-3


@pytest.mark.parametrize("k", [32])
def test_topk_matches_reference(k):
    """Fused top-k selects the same scores as reference + ``jax.lax.top_k``.

    Index set-equality is too strict (backends break ties differently), so the
    check is on the *scores* at the fused-selected indices. ``k`` is kept in the
    top quartile of ``T_s``: a cutoff in the dense middle of the distribution
    makes boundary scores closely spaced, so the kernel's fp32 ranking and the
    bf16-rounded reference grid resolve near-ties differently (a test-grid
    sensitivity, not a kernel error).
    """
    args = _indexer_inputs(2, 3, T_t=64, T_s=128, d=32, d_c=32, H=16, d_i=32, seed=200)
    o_ref = indexer(*args, backend="reference").astype(jnp.float32)
    topk_idx = indexer_topk(*args, k=k)
    assert topk_idx.shape == (2, 3, 64, k)

    ref_vals = jax.lax.top_k(o_ref, k=k)[0]
    picked = jnp.take_along_axis(o_ref, topk_idx, axis=-1)
    picked_sorted = jnp.sort(picked, axis=-1)[..., ::-1]
    max_rel = float((jnp.abs(ref_vals - picked_sorted) / (jnp.abs(ref_vals) + 1e-6)).max())
    assert max_rel < 1e-2


@pytest.mark.parametrize("B,oH", [(2, 3), (1, 2)])
def test_hybrid_backward_matches_reference_grad(B, oH):
    """``jax.grad`` through the hybrid backend matches grad through reference.

    Tolerance is 5e-2 (bf16 projections + Triton score recompute) — looser than
    the 5e-3 forward tolerance; tighten once per-grad error is characterized
    on-device.
    """
    args = _indexer_inputs(B, oH, T_t=32, T_s=32, d=32, d_c=32, H=8, d_i=32, seed=300)

    def _loss(backend):
        def inner(*a):
            return jnp.sum(indexer(*a, backend=backend).astype(jnp.float32))
        return inner

    argnums = (0, 1, 2, 3, 4, 5)
    grads_ref = jax.grad(_loss("reference"), argnums=argnums)(*args)
    grads_hyb = jax.grad(_loss("hybrid"), argnums=argnums)(*args)
    for gr, gh in zip(grads_ref, grads_hyb):
        assert _rel_err(gh, gr) < 5e-2
