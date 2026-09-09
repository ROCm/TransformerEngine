# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
"""Correctness tests for the lightning-indexer JAX ops.
"""

import functools

import jax
import jax.numpy as jnp
import pytest

from transformer_engine.jax.sparse_attention.indexer import (
    LightningIndexer,
    _indexer_projections,
    indexer,
    indexer_topk,
)
from transformer_engine.jax.triton_extensions.indexer import (
    fp8_dtype,
    quantize_e4m3,
    score_reduce_triton,
    score_topk_triton,
)


@pytest.fixture(autouse=True)
def _disable_indexer_autotune(monkeypatch):
    """Pin each indexer Triton kernel to a single (prune-valid) config so the
    suite skips the multi-minute autotune sweep."""
    monkeypatch.setenv("NVTE_INDEXER_DISABLE_AUTOTUNE", "1")


@functools.partial(jax.jit, static_argnames=("out_dtype",))
def _indexer_reference(Q, K, W_uq, W_dq, W_k, W_w, out_dtype=None):
    """Pure-einsum lightning-indexer reference (test oracle).

    Materializes the (..., T, H, S) pre-relu score tensor, unlike the hybrid
    Triton op under test. Shapes: Q [..., T, d], K [..., S, d], W_dq [d, d_c],
    W_uq [H, d_c, d_i], W_k [d, d_i], W_w [d, H]. Returns O [..., T, S].

    JIT-compiled so its HLO (reduction order / bf16 rounding) is stable — the
    top-k test feeds these reference scores to ``jax.lax.top_k``, whose
    tie-breaking is sensitive to sub-ULP score perturbations.
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


# Tolerances against the bf16 einsum reference. The fp8 path quantizes both
# score-matmul operands to e4m3 (4-bit mantissa) with per-row scales, so it is
# held to looser bounds than the bf16 path.
#
# The fp8 backward bound is looser than the forward one by more than the forward
# error alone: cotangents follow the operand dtype, so a gradient flowing out of
# the score op through e4m3 H_q / H_k picks up a second e4m3 rounding on the way
# back (~6% per element) on top of the forward quantization error. That is the
# cost of the ops consuming pre-quantized operands rather than quantizing behind
# their own custom_vjp.
_FWD_TOL = {False: 5e-3, True: 5e-2}
_BWD_TOL = {False: 5e-2, True: 2.5e-1}


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("B,oH", [(2, 3), (1, 1), (1, 4)])
def test_hybrid_matches_reference(B, oH, fp8):
    """Hybrid Triton score-reduce matches the pure-einsum reference forward."""
    args = _indexer_inputs(B, oH, T_t=64, T_s=64, d=32, d_c=32, H=8, d_i=32, seed=100)
    o_ref = _indexer_reference(*args)
    o_hyb = indexer(*args, fp8=fp8)
    assert o_hyb.shape == o_ref.shape
    assert _rel_err(o_hyb, o_ref) < _FWD_TOL[fp8]


def test_fp8_is_the_default():
    """``indexer`` runs the fp8 score matmul unless asked not to.

    Guards against the default silently reverting: the fp8 and bf16 paths must
    agree to within fp8 tolerance but must not be bit-identical, and the default
    call must match the explicit ``fp8=True`` one exactly.
    """
    args = _indexer_inputs(1, 2, T_t=64, T_s=64, d=32, d_c=32, H=8, d_i=32, seed=101)
    o_default = indexer(*args)
    o_fp8 = indexer(*args, fp8=True)
    o_bf16 = indexer(*args, fp8=False)

    assert jnp.array_equal(o_default, o_fp8), "default is not the fp8 path"
    assert not jnp.array_equal(o_fp8, o_bf16), "fp8 and bf16 paths are identical"
    assert _rel_err(o_fp8, o_bf16) < _FWD_TOL[True]


def test_fp8_falls_back_below_mfma_k():
    """``d_i`` under one e4m3 MFMA K-tile (32) silently uses the bf16 matmul."""
    args = _indexer_inputs(1, 2, T_t=64, T_s=64, d=32, d_c=32, H=8, d_i=16, seed=102)
    assert jnp.array_equal(indexer(*args, fp8=True), indexer(*args, fp8=False))


def _score_reference(H_q, H_k, W_o):
    """Pure-einsum score+relu+H-reduce over already-materialized operands."""
    scores = jax.nn.relu(jnp.einsum("...thi,...si->...ths", H_q, H_k))
    return jnp.einsum("...ths,...th->...ts", scores, W_o)


def test_score_ops_accept_prequantized_operands():
    """Pre-quantized e4m3 operands go straight into the score ops.

    This is the externally-quantized path: a caller holding fp8 index-q /
    index-k plus their scales hands them in directly, with no requantization.

    The oracle is the reference score computed from the *dequantized* operands,
    so the only slack is the score matmul's accumulation order — an op that
    requantized internally, or that ignored the supplied scales, would miss by
    far more than this tolerance.
    """
    args = _indexer_inputs(1, 2, T_t=64, T_s=128, d=32, d_c=32, H=8, d_i=32, seed=400)
    H_q, H_k, W_o = _indexer_projections(*args)

    H_q_q, Sq = quantize_e4m3(H_q)
    H_k_q, Ks = quantize_e4m3(H_k)
    assert H_q_q.dtype == fp8_dtype() and H_k_q.dtype == fp8_dtype()

    o_ext = score_reduce_triton(H_q_q, H_k_q, W_o, Sq=Sq, Ks=Ks)
    assert o_ext.dtype == W_o.dtype

    o_ref = _score_reference(
        (H_q_q.astype(jnp.float32) * Sq[..., None]).astype(jnp.float32),
        (H_k_q.astype(jnp.float32) * Ks[..., None]).astype(jnp.float32),
        W_o.astype(jnp.float32),
    )
    assert _rel_err(o_ext, o_ref) < 5e-3

    idx_ext = score_topk_triton(H_q_q, H_k_q, W_o, k=32, Sq=Sq, Ks=Ks)
    ref_vals = jax.lax.top_k(o_ref, k=32)[0]
    picked = jnp.sort(jnp.take_along_axis(o_ref, idx_ext, axis=-1), axis=-1)[..., ::-1]
    assert float(jnp.abs(ref_vals - picked).max()) / float(ref_vals.max()) < 1e-2


def test_prequantized_path_agrees_with_indexer():
    """Quantizing externally matches letting ``indexer`` do it one layer up.

    Not bitwise: the projections here are traced outside ``indexer``'s ``jit``,
    so XLA fuses them differently and the last bf16 ulp of H_q / H_k can shift,
    which e4m3 quantization then amplifies.
    """
    args = _indexer_inputs(1, 2, T_t=64, T_s=128, d=32, d_c=32, H=8, d_i=32, seed=400)
    H_q, H_k, W_o = _indexer_projections(*args)
    H_q_q, Sq = quantize_e4m3(H_q)
    H_k_q, Ks = quantize_e4m3(H_k)

    o_ext = score_reduce_triton(H_q_q, H_k_q, W_o, Sq=Sq, Ks=Ks)
    assert _rel_err(o_ext, indexer(*args, fp8=True)) < 1e-2


def test_omitted_scales_mean_unit_scales():
    """Leaving ``Sq`` / ``Ks`` unset is the same as passing all-ones."""
    args = _indexer_inputs(1, 2, T_t=64, T_s=64, d=32, d_c=32, H=8, d_i=32, seed=401)
    H_q, H_k, W_o = _indexer_projections(*args)

    ones_q = jnp.ones(H_q.shape[:-1], jnp.float32)
    ones_k = jnp.ones(H_k.shape[:-1], jnp.float32)
    assert jnp.array_equal(
        score_reduce_triton(H_q, H_k, W_o),
        score_reduce_triton(H_q, H_k, W_o, Sq=ones_q, Ks=ones_k),
    )


def test_prequantized_grads_follow_operand_dtype():
    """Gradients w.r.t. e4m3 operands come back as e4m3, and track the bf16 ones.

    The cotangent is w.r.t. the *quantized* operand (``dHq_true * sq``), so
    rescaling it by the row scale must recover the bf16-path gradient to within
    e4m3 resolution.
    """
    args = _indexer_inputs(1, 2, T_t=32, T_s=32, d=32, d_c=32, H=8, d_i=32, seed=402)
    H_q, H_k, W_o = _indexer_projections(*args)
    H_q_q, Sq = quantize_e4m3(H_q)
    H_k_q, Ks = quantize_e4m3(H_k)

    def _loss(hq, hk, sq, ks):
        return jnp.sum(score_reduce_triton(hq, hk, W_o, Sq=sq, Ks=ks).astype(jnp.float32))

    dHq_q, dHk_q, dSq, dKs = jax.grad(_loss, argnums=(0, 1, 2, 3))(H_q_q, H_k_q, Sq, Ks)
    assert dHq_q.dtype == H_q_q.dtype and dHk_q.dtype == H_k_q.dtype
    # Scales are quantization metadata, not learnables.
    assert not jnp.any(dSq) and not jnp.any(dKs)

    dHq_bf16, dHk_bf16 = jax.grad(
        lambda hq, hk: jnp.sum(score_reduce_triton(hq, hk, W_o).astype(jnp.float32)),
        argnums=(0, 1),
    )(H_q, H_k)
    # Undo the operand-space rescaling to compare against the bf16 gradient.
    assert _rel_err(dHq_q.astype(jnp.float32) / Sq[..., None], dHq_bf16) < 2e-1
    assert _rel_err(dHk_q.astype(jnp.float32) / Ks[..., None], dHk_bf16) < 2e-1


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("k", [32, 64, 128, 256, 512, 1024])
def test_topk_matches_reference(k, fp8):
    """Fused top-k selects the same scores as reference + ``jax.lax.top_k``.

    Index set-equality is too strict (backends break ties differently), so the
    check is on the *scores* at the fused-selected indices, compared rank-by-rank
    against the reference top-k. ``T_s`` is ``4 * max(k)`` so the largest ``k``
    (1024) sits at the top quartile — deep enough to exercise the streaming
    top-k path (2K candidate buffer) that small ``k`` / ``T_s`` never reaches.

    The gap is normalized by the overall score *scale* (max reference score), not
    per element: as ``k`` grows into the near-zero ReLU tail, per-element relative
    error is dominated by ties the fp32/bf16 paths break differently (denominators
    ~0 blow it up), while the absolute gap stays ~0.1% of the max score.

    Leading dims (B, oH, T_t) are kept small so the reference — which materializes
    the (B, oH, T_t, H, T_s) pre-relu score tensor — stays a few MB even at
    T_s=4096; a larger footprint tips shared-GPU GEMMs into resource errors.
    """
    B, oH, T_t = 1, 2, 32
    args = _indexer_inputs(B, oH, T_t, T_s=4096, d=32, d_c=32, H=16, d_i=32, seed=200)
    o_ref = _indexer_reference(*args).astype(jnp.float32)
    topk_idx = indexer_topk(*args, k=k, fp8=fp8)
    assert topk_idx.shape == (B, oH, T_t, k)

    ref_vals = jax.lax.top_k(o_ref, k=k)[0]
    scale = float(ref_vals.max())
    assert scale > 0, "degenerate test: all top-k scores are zero"
    picked = jnp.take_along_axis(o_ref, topk_idx, axis=-1)
    picked_sorted = jnp.sort(picked, axis=-1)[..., ::-1]
    max_gap = float(jnp.abs(ref_vals - picked_sorted).max()) / scale
    # e4m3 perturbs scores enough to reorder near-equal neighbours, so the fp8
    # bound is looser -- but still measured against the *reference* score at each
    # rank, so a genuinely wrong selection (not just a swapped near-tie) fails.
    tol = 1e-1 if fp8 else 1e-2
    assert max_gap < tol, f"fused top-k scores diverge: max_gap={max_gap:.3e} (k={k})"


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("B,oH", [(2, 3), (1, 2)])
def test_hybrid_backward_matches_reference_grad(B, oH, fp8):
    """``jax.grad`` through the hybrid backend matches grad through reference.

    The reference is the bf16 einsum chain in both cases; the fp8 path is held
    to a looser bound because its backward recomputes the score matmul in e4m3,
    matching its forward.
    """
    args = _indexer_inputs(B, oH, T_t=32, T_s=32, d=32, d_c=32, H=8, d_i=32, seed=300)

    def _loss(fn, **kwargs):
        def inner(*a):
            return jnp.sum(fn(*a, **kwargs).astype(jnp.float32))
        return inner

    argnums = (0, 1, 2, 3, 4, 5)
    grads_ref = jax.grad(_loss(_indexer_reference), argnums=argnums)(*args)
    grads_hyb = jax.grad(_loss(indexer, fp8=fp8), argnums=argnums)(*args)
    for gr, gh in zip(grads_ref, grads_hyb):
        assert _rel_err(gh, gr) < _BWD_TOL[fp8]


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
