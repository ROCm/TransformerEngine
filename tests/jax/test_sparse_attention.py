# Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
"""Tests for Deep Sparse Attention (DSA) composition + HCA / fused scaffold contracts."""

import jax
import jax.numpy as jnp
import pytest
from flax import linen as nn

from transformer_engine.jax.sparse_attention import (
    DeepSparseAttention,
    deep_sparse_attention_core,
    _causal_keep_mask,
    _topk_indices_to_attn_mask,
)
from transformer_engine.jax.compressed_attention import (
    HeavilyCompressedAttention,
    heavily_compressed_attention,
)
from transformer_engine.jax.indexer import indexer
from transformer_engine.jax.triton_extensions import fused_sparse_attention_triton


@pytest.fixture(autouse=True)
def _force_unfused_attn(monkeypatch):
    """Override conftest's enable_fused_attn_after_hopper for this module.

    The DSA composition path uses an arbitrary topk-derived attention mask. The
    fused-attention backends on some platforms restrict mask semantics (padding-
    style only). Force the unfused softmax path so reference comparisons hold.
    Production callers can still set NVTE_FUSED_ATTN=1 — these tests are only
    asserting the composition math, not the fused-path's mask handling.
    """
    monkeypatch.setenv("NVTE_FUSED_ATTN", "0")
    yield


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _make_dsa_module(*, oH=4, D=8, iH=2, idc=16, idi=16, k=4,
                     backend="composition", indexer_backend="hybrid"):
    return DeepSparseAttention(
        head_dim=D,
        num_attention_heads=oH,
        indexer_num_heads=iH,
        indexer_d_c=idc,
        indexer_d_i=idi,
        topk=k,
        backend=backend,
        indexer_backend=indexer_backend,
        dtype=jnp.bfloat16,
    )


def _make_inputs(B=1, oH=4, T=16, hidden=32, dtype=jnp.bfloat16, seed=0):
    """Rank-4 inputs [B, oH, T, hidden]."""
    return jax.random.normal(jax.random.PRNGKey(seed), (B, oH, T, hidden), dtype=dtype)


def _ref_dense_softmax_per_head(query, key, value, mask_out, scale):
    """Per-head dense softmax attention with an arbitrary mask (no DPA).

    query/key/value: [B, oH, T, head_dim]; mask_out: [B, oH, T_t, T_s] uint8
    (1 = mask out). Returns [B, oH, T_t, head_dim].
    """
    logits = jnp.einsum("bhtd,bhsd->bhts", query, key) * scale
    logits = logits.astype(jnp.float32)
    logits = jnp.where(
        mask_out.astype(jnp.bool_),
        jnp.asarray(-jnp.inf, jnp.float32),
        logits,
    )
    weights = jax.nn.softmax(logits, axis=-1)
    return jnp.einsum("bhts,bhsd->bhtd", weights.astype(value.dtype), value)


def _ref_dsa_jax(
    inputs_q, inputs_kv,
    W_q_kernel, W_k_kernel, W_v_kernel,
    W_uq, W_dq, W_k_idx, W_w,
    *,
    head_dim, k, causal,
):
    """Pure-JAX reference matching ``deep_sparse_attention_core``."""
    T_t = inputs_q.shape[2]
    T_s = inputs_kv.shape[2]

    q = jnp.einsum("bhtd,dk->bhtk", inputs_q, W_q_kernel)
    kk = jnp.einsum("bhsd,dk->bhsk", inputs_kv, W_k_kernel)
    v = jnp.einsum("bhsd,dk->bhsk", inputs_kv, W_v_kernel)

    scores = indexer(
        inputs_q, inputs_kv, W_uq, W_dq, W_k_idx, W_w,
        backend="reference", out_dtype=jnp.float32,
    )
    if causal:
        ckeep = _causal_keep_mask(T_t, T_s)[None, None, :, :]
        scores = jnp.where(ckeep, scores, jnp.asarray(-jnp.inf, jnp.float32))
    _, topk_idx = jax.lax.top_k(scores, min(k, T_s))
    mask_out = _topk_indices_to_attn_mask(topk_idx, T_s, causal=causal)
    return _ref_dense_softmax_per_head(
        q, kk, v, mask_out, scale=1.0 / jnp.sqrt(head_dim).astype(q.dtype),
    )


# -----------------------------------------------------------------------------
# Mask helpers
# -----------------------------------------------------------------------------


def test_causal_keep_mask_self_attention():
    """T_t == T_s: standard lower-triangular keep mask."""
    m = _causal_keep_mask(4, 4)
    expected = jnp.tril(jnp.ones((4, 4), dtype=jnp.bool_))
    assert jnp.array_equal(m, expected)


def test_causal_keep_mask_cross_attention_with_prefix():
    """T_t < T_s: causal cutoff aligned to bottom-right (prefix context allowed)."""
    m = _causal_keep_mask(2, 5)                 # T_t=2, T_s=5 → prefix of 3 always visible
    expected = jnp.array(
        [[True, True, True, True, False],
         [True, True, True, True, True]],
        dtype=jnp.bool_,
    )
    assert jnp.array_equal(m, expected)


def test_topk_indices_to_attn_mask_basic():
    # B=1, oH=1, T_t=2, k=2
    indices = jnp.array([[[[0, 2], [1, 3]]]], dtype=jnp.int32)        # [1, 1, 2, 2]
    mask_out = _topk_indices_to_attn_mask(indices, T_s=4, causal=False)
    expected = jnp.array(
        [[[[0, 1, 0, 1],
           [1, 0, 1, 0]]]],
        dtype=jnp.uint8,
    )
    assert mask_out.shape == (1, 1, 2, 4)
    assert mask_out.dtype == jnp.uint8
    assert jnp.array_equal(mask_out, expected)


def test_topk_indices_to_attn_mask_per_head_diverges():
    """Different oH heads pick different topk → different per-head masks."""
    # B=1, oH=2, T_t=1, k=2
    indices = jnp.array([[[[0, 1]], [[2, 3]]]], dtype=jnp.int32)
    mask_out = _topk_indices_to_attn_mask(indices, T_s=4, causal=False)
    # Head 0 keeps {0,1} → mask [0,0,1,1]; head 1 keeps {2,3} → mask [1,1,0,0].
    expected = jnp.array(
        [[[[0, 0, 1, 1]],
          [[1, 1, 0, 0]]]],
        dtype=jnp.uint8,
    )
    assert mask_out.shape == (1, 2, 1, 4)
    assert jnp.array_equal(mask_out, expected)


def test_topk_indices_to_attn_mask_causal_intersect():
    """Causal AND topk in self-attention: query t cannot keep positions > t."""
    # B=1, oH=1, T_t=T_s=4, k=2
    indices = jnp.array([[[[2, 3], [0, 1], [1, 2], [2, 3]]]], dtype=jnp.int32)
    mask_out = _topk_indices_to_attn_mask(indices, T_s=4, causal=True)
    # q=0: picks {2,3}, causal {0} → intersect {} → all-1 row.
    assert bool((mask_out[0, 0, 0, :] == 1).all())
    # q=2: picks {1,2}, causal {0,1,2} → intersect {1,2}.
    assert mask_out[0, 0, 2, 0] == 1
    assert mask_out[0, 0, 2, 1] == 0
    assert mask_out[0, 0, 2, 2] == 0
    assert mask_out[0, 0, 2, 3] == 1


# -----------------------------------------------------------------------------
# DSA composition correctness
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("B,oH,T,hidden,D,iH,idc,idi,k", [
    (1, 4, 16, 32, 8, 2, 16, 16, 4),
    (2, 4, 32, 64, 16, 2, 32, 32, 8),
    (1, 2, 8, 16, 8, 1, 8, 8, 2),
])
def test_dsa_composition_vs_pure_jax_reference(B, oH, T, hidden, D, iH, idc, idi, k):
    """DSA module output (composition + hybrid indexer) matches pure-JAX reference."""
    inputs = _make_inputs(B=B, oH=oH, T=T, hidden=hidden)
    keys = jax.random.split(jax.random.PRNGKey(123), 2)
    module = _make_dsa_module(oH=oH, D=D, iH=iH, idc=idc, idi=idi, k=k)
    params = module.init(keys[0], inputs, inputs, deterministic=True)
    out = module.apply(params, inputs, inputs, deterministic=True)
    assert out.shape == (B, oH, T, D)

    p = nn.meta.unbox(params)["params"]
    out_ref = _ref_dsa_jax(
        inputs, inputs,
        p["query"]["kernel"], p["key"]["kernel"], p["value"]["kernel"],
        p["indexer_W_uq"], p["indexer_W_dq"], p["indexer_W_k"], p["indexer_W_w"],
        head_dim=D, k=k, causal=True,
    )

    diff = (out.astype(jnp.float32) - out_ref.astype(jnp.float32))
    rel = float(
        jnp.linalg.norm(diff)
        / (jnp.linalg.norm(out_ref.astype(jnp.float32)) + 1e-30)
    )
    assert rel < 5e-2, f"DSA output diverges from reference: rel.err={rel:.3e}"


def test_dsa_composition_reference_indexer_matches_hybrid():
    """Same correctness check using indexer_backend='reference' (pure einsum)."""
    B, oH, T, hidden, D, iH, idc, idi, k = 1, 2, 8, 16, 8, 1, 8, 8, 2
    inputs = _make_inputs(B=B, oH=oH, T=T, hidden=hidden)
    keys = jax.random.split(jax.random.PRNGKey(7), 2)
    module = _make_dsa_module(oH=oH, D=D, iH=iH, idc=idc, idi=idi, k=k,
                              indexer_backend="reference")
    params = module.init(keys[0], inputs, inputs, deterministic=True)
    out = module.apply(params, inputs, inputs, deterministic=True)
    assert out.shape == (B, oH, T, D)


@pytest.mark.parametrize("T_t,T_s,k", [(8, 8, 4), (8, 8, 2), (16, 16, 8)])
def test_dsa_topk_count_equals_kept_count_under_causal(T_t, T_s, k):
    """For each query t, the number of unmasked key positions equals min(k, t+1)."""
    B, oH, hidden = 1, 2, 16
    inputs = _make_inputs(B=B, oH=oH, T=T_t, hidden=hidden, seed=7)
    keys = jax.random.split(jax.random.PRNGKey(7), 2)
    module = _make_dsa_module(oH=oH, D=8, iH=1, idc=8, idi=8, k=k)
    params = module.init(keys[0], inputs, inputs, deterministic=True)

    p = nn.meta.unbox(params)["params"]
    scores = indexer(
        inputs, inputs,
        p["indexer_W_uq"], p["indexer_W_dq"], p["indexer_W_k"], p["indexer_W_w"],
        backend="reference", out_dtype=jnp.float32,
    )                                                       # [B, oH, T_t, T_s]
    ckeep = _causal_keep_mask(T_t, T_s)[None, None, :, :]
    scores_masked = jnp.where(ckeep, scores, -jnp.inf)
    _, topk_idx = jax.lax.top_k(scores_masked, min(k, T_s))
    mask_out = _topk_indices_to_attn_mask(topk_idx, T_s, causal=True)
    # Each (b, h, t) row should have exactly min(k, t+1) zeros.
    for h in range(oH):
        kept_per_q = (mask_out[0, h] == 0).sum(axis=-1)     # [T_t]
        for t in range(T_t):
            expected = min(k, t + 1)
            assert int(kept_per_q[t]) == expected, (
                f"oH={h}, t={t}: kept {int(kept_per_q[t])} keys, expected {expected}"
            )


# -----------------------------------------------------------------------------
# Backward shape sanity
# -----------------------------------------------------------------------------


def test_dsa_backward_runs_without_shape_errors():
    inputs = _make_inputs(B=1, oH=2, T=8, hidden=16)
    keys = jax.random.split(jax.random.PRNGKey(5), 2)
    module = _make_dsa_module(oH=2, D=8, iH=1, idc=8, idi=8, k=2)
    params = module.init(keys[0], inputs, inputs, deterministic=True)

    def loss(p, x):
        out = module.apply(p, x, x, deterministic=True)
        return jnp.sum(out.astype(jnp.float32))

    grads = jax.grad(loss)(params, inputs)
    leaves = jax.tree_util.tree_leaves(grads)
    assert all(bool(jnp.isfinite(leaf).all()) for leaf in leaves), \
        "DSA backward produced NaN/Inf gradients"


# -----------------------------------------------------------------------------
# Scaffold contracts
# -----------------------------------------------------------------------------


def test_dsa_fused_backend_raises_not_implemented():
    inputs = _make_inputs(B=1, oH=2, T=8, hidden=16)
    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    module = _make_dsa_module(oH=2, D=8, iH=1, idc=8, idi=8, k=2, backend="fused")
    # Flax materializes the call during init, so NotImplementedError fires there.
    with pytest.raises(NotImplementedError, match="phase-2 scaffold"):
        module.init(keys[0], inputs, inputs, deterministic=True)


def test_fused_sparse_attention_triton_direct_raises():
    """Calling the primitive directly also raises (locked contract)."""
    q = jnp.zeros((1, 2, 4, 8), dtype=jnp.bfloat16)         # [B, T, H, D]
    kk = jnp.zeros((1, 2, 4, 8), dtype=jnp.bfloat16)
    v = jnp.zeros((1, 2, 4, 8), dtype=jnp.bfloat16)
    iq = jnp.zeros((1, 2, 4, 8), dtype=jnp.bfloat16)
    ik = jnp.zeros((1, 2, 8), dtype=jnp.bfloat16)
    iw = jnp.zeros((1, 2, 2), dtype=jnp.bfloat16)
    with pytest.raises(NotImplementedError, match="phase-2 scaffold"):
        jax.jit(
            lambda *args: fused_sparse_attention_triton(*args, k=2)
        )(q, kk, v, iq, ik, iw)


def test_hca_module_raises_not_implemented():
    module = HeavilyCompressedAttention(
        head_dim=8, num_attention_heads=4,
        q_lora_rank=16, kv_lora_rank=16,
        qk_nope_head_dim=4, qk_rope_head_dim=4, v_head_dim=8,
    )
    inputs = jax.random.normal(jax.random.PRNGKey(0), (1, 4, 32), dtype=jnp.bfloat16)
    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    with pytest.raises(NotImplementedError, match="design.*deferred|DESIGN DEFERRED|scaffold"):
        module.init(keys[0], inputs, inputs, deterministic=True)


def test_hca_functional_raises_not_implemented():
    inputs = jax.random.normal(jax.random.PRNGKey(0), (1, 4, 32), dtype=jnp.bfloat16)
    with pytest.raises(NotImplementedError):
        heavily_compressed_attention(
            inputs, inputs,
            head_dim=8, num_attention_heads=4,
            q_lora_rank=16, kv_lora_rank=16,
            qk_nope_head_dim=4, qk_rope_head_dim=4, v_head_dim=8,
        )


# -----------------------------------------------------------------------------
# Functional API surface
# -----------------------------------------------------------------------------


def test_deep_sparse_attention_core_invalid_backend_raises():
    q = jnp.zeros((1, 2, 4, 8))                              # rank-4
    iq = jnp.zeros((1, 2, 4, 16))
    W = jnp.zeros((16, 8))
    Wuq = jnp.zeros((1, 8, 8))
    with pytest.raises(ValueError, match="unknown backend"):
        deep_sparse_attention_core(
            q, q, q, iq, iq, Wuq, W, W[:, :8], W[:, :1],
            k=2, backend="bogus",
        )


def test_deep_sparse_attention_core_unsupported_mask_type_raises():
    q = jnp.zeros((1, 2, 4, 8))
    iq = jnp.zeros((1, 2, 4, 16))
    W = jnp.zeros((16, 8))
    Wuq = jnp.zeros((1, 8, 8))
    with pytest.raises(NotImplementedError, match="attn_mask_type"):
        deep_sparse_attention_core(
            q, q, q, iq, iq, Wuq, W, W[:, :8], W[:, :1],
            k=2, attn_mask_type="padding",
        )


def test_deep_sparse_attention_core_rejects_rank3_inputs():
    """Rank-3 inputs (missing oH) should be rejected with a clear error."""
    q3 = jnp.zeros((1, 4, 8))                                # rank-3
    iq4 = jnp.zeros((1, 2, 4, 16))
    W = jnp.zeros((16, 8))
    Wuq = jnp.zeros((1, 8, 8))
    with pytest.raises(ValueError, match="rank-4"):
        deep_sparse_attention_core(
            q3, q3, q3, iq4, iq4, Wuq, W, W[:, :8], W[:, :1],
            k=2,
        )


def test_dsa_module_rejects_oh_mismatch():
    """Module asserts num_attention_heads matches inputs.shape[1]."""
    inputs = _make_inputs(B=1, oH=3, T=8, hidden=16)         # oH=3 in input
    module = _make_dsa_module(oH=4, D=8, iH=1, idc=8, idi=8, k=2)  # oH=4 in module
    with pytest.raises(ValueError, match="must equal num_attention_heads"):
        module.init(jax.random.PRNGKey(0), inputs, inputs, deterministic=True)
