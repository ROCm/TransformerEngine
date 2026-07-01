# Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
"""Correctness tests for the lightning-indexer JAX ops.

Ported from the in-module ``__main__`` smoke tests of
``transformer_engine.jax.sparse_attention.indexer``. The hybrid and top-k backends require
rank-4 ``(B, oH, T, d)`` inputs, so every leading shape here is length-2.
"""

import functools

import jax
import jax.numpy as jnp
import pytest

from transformer_engine.jax.sparse_attention.indexer import (
    LightningIndexer,
    indexer,
    indexer_topk,
)


@functools.partial(jax.jit, static_argnames=("out_dtype",))
def _indexer_reference(Q, K, W_uq, W_dq, W_k, W_w, out_dtype=None):
    """Pure-einsum lightning-indexer reference (test oracle).

    Materializes the (..., T, H, S) pre-relu score tensor, unlike the hybrid
    Triton op under test. Shapes: Q [..., T, d], K [..., S, d], W_dq [d, d_c],
    W_uq [H, d_c, d_i], W_k [d, d_i], W_w [d, H]. Returns O [..., T, S].

    JIT-compiled so its HLO (reduction order / bf16 rounding) matches the
    standalone-compiled reference op it replaces — the DSA composition test's
    top-k tie-breaking is sensitive to sub-ULP score perturbations.
    """
    C_q = jnp.einsum("...td,dc->...tc", Q, W_dq)
    H_q = jnp.einsum("...tc,hci->...thi", C_q, W_uq)
    H_k = jnp.einsum("...sd,di->...si", K, W_k)
    W_o = jnp.einsum("...td,dh->...th", Q, W_w)
    H = jax.nn.relu(jnp.einsum("...thi,...si->...ths", H_q, H_k))  # (..., T, H, S)
    O = jnp.einsum("...ths,...th->...ts", H, W_o)                  # (..., T, S)
    if out_dtype is not None:
        O = O.astype(out_dtype)
    return O


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
    o_ref = _indexer_reference(*args)
    o_hyb = indexer(*args)
    assert o_hyb.shape == o_ref.shape
    assert _rel_err(o_hyb, o_ref) < 5e-3


@pytest.mark.parametrize("k", [32])
def test_topk_matches_reference(k):
    """Fused top-k selects the same scores as reference + ``jax.lax.top_k``.

    Index set-equality is too strict (backends break ties differently), so the
    check is on the *scores* at the fused-selected indices. ``k`` is kept in the
    top quartile of ``T_s`` so the cutoff lands above the dense band of near-tied
    scores where fp32 and bf16-rounded rankings would disagree.
    """
    args = _indexer_inputs(2, 3, T_t=64, T_s=128, d=32, d_c=32, H=16, d_i=32, seed=200)
    o_ref = _indexer_reference(*args).astype(jnp.float32)
    topk_idx = indexer_topk(*args, k=k)
    assert topk_idx.shape == (2, 3, 64, k)

    ref_vals = jax.lax.top_k(o_ref, k=k)[0]
    assert float(ref_vals.max()) > 0, "degenerate test: all top-k scores are zero"
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

    def _loss(fn):
        def inner(*a):
            return jnp.sum(fn(*a).astype(jnp.float32))
        return inner

    argnums = (0, 1, 2, 3, 4, 5)
    grads_ref = jax.grad(_loss(_indexer_reference), argnums=argnums)(*args)
    grads_hyb = jax.grad(_loss(indexer), argnums=argnums)(*args)
    for gr, gh in zip(grads_ref, grads_hyb):
        assert _rel_err(gh, gr) < 5e-2


def test_lightning_indexer_module_matches_functional():
    """``LightningIndexer`` (Flax module) reproduces the functional ``indexer``
    when fed the module's own initialized weights."""
    B, oH, T_t, T_s, d, d_c, H, d_i = 2, 3, 64, 64, 32, 32, 8, 32
    keys = jax.random.split(jax.random.PRNGKey(7), 3)
    Q = jax.random.normal(keys[0], (B, oH, T_t, d), dtype=jnp.bfloat16)
    K = jax.random.normal(keys[1], (B, oH, T_s, d), dtype=jnp.bfloat16)

    mod = LightningIndexer(num_heads=H, d_c=d_c, d_i=d_i)
    variables = mod.init(keys[2], Q, K)
    o_mod = mod.apply(variables, Q, K)
    assert o_mod.shape == (B, oH, T_t, T_s)

    p = variables["params"]
    o_fn = indexer(Q, K, p["W_uq"], p["W_dq"], p["W_k"], p["W_w"])
    assert _rel_err(o_mod, o_fn) < 1e-5


def test_lightning_indexer_topk_mode():
    """``LightningIndexer(topk=k)`` returns fused top-k indices of shape (..., T, k)."""
    B, oH, T_t, T_s, d, d_c, H, d_i, k = 2, 3, 64, 128, 32, 32, 16, 32, 32
    keys = jax.random.split(jax.random.PRNGKey(9), 2)
    Q = jax.random.normal(keys[0], (B, oH, T_t, d), dtype=jnp.bfloat16)
    K = jax.random.normal(keys[1], (B, oH, T_s, d), dtype=jnp.bfloat16)

    mod = LightningIndexer(num_heads=H, d_c=d_c, d_i=d_i, topk=k)
    variables = mod.init(jax.random.PRNGKey(0), Q, K)
    idx = mod.apply(variables, Q, K)
    assert idx.shape == (B, oH, T_t, k)
    assert idx.dtype == jnp.int32
