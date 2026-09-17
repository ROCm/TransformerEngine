# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
"""Correctness tests for the lightning-indexer JAX ops.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from transformer_engine.jax.sparse_attention.indexer import (
    LightningIndexer,
    _mask_keys,
    _indexer_projections,
    indexer,
    indexer_topk,
)
from transformer_engine.jax.triton_extensions.indexer import (
    _COMPACT_TILE_S,
    _COMPACT_TILE_T,
    _compact_override,
    _tile_launch_order,
    _tile_validity_map,
    fp8_dtype,
    quantize_e4m3,
    score_reduce_triton,
)


@pytest.fixture(autouse=True)
def _disable_indexer_autotune(monkeypatch):
    """Pin each indexer Triton kernel to a single (prune-valid) config so the
    suite skips the multi-minute autotune sweep."""
    monkeypatch.setenv("NVTE_INDEXER_DISABLE_AUTOTUNE", "1")


@functools.partial(jax.jit, static_argnames=("out_dtype",))
def _indexer_reference(Q, K, W_uq, W_dq, W_k, W_w, out_dtype=None, mask=None):
    """Pure-einsum lightning-indexer reference (test oracle).

    Materializes the (..., T, H, S) pre-relu score tensor, unlike the hybrid
    Triton op under test. Shapes: Q [..., T, d], K [..., S, d], W_dq [d, d_c],
    W_uq [H, d_c, d_i], W_k [d, d_i], W_w [d, H]. Returns O [..., T, S].

    ``mask``, when given, is a (B, T, S) boolean of *valid* pairs; invalid slots
    are filled the same way the kernel fills them.

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
    if mask is not None:
        # mask is (B, T, S); O is (B, oH, T, S) -- broadcast over the head axis,
        # matching the kernel, whose segment metadata has no oH axis.
        O = jnp.where(mask[:, None], O, _mask_fill(O.dtype))
    return O


def _mask_fill(dtype):
    """The value the kernel writes at a masked slot."""
    return jnp.finfo(dtype).min


def _segments(B, T, lengths, seed=0):
    """Build TE-JAX (segment_ids, segment_pos) for a packed batch.

    ``lengths`` is the per-segment token count, applied identically to every
    batch row; tokens past ``sum(lengths)`` are padding (segment id 0). Returns
    int32 arrays of shape (B, T).
    """
    del seed
    assert sum(lengths) <= T, f"segments {lengths} overflow T={T}"
    ids, pos = [], []
    for seg, n in enumerate(lengths, start=1):
        ids.extend([seg] * n)
        pos.extend(range(n))
    pad = T - len(ids)
    ids.extend([0] * pad)
    pos.extend([0] * pad)
    ids = jnp.broadcast_to(jnp.asarray(ids, jnp.int32), (B, T))
    pos = jnp.broadcast_to(jnp.asarray(pos, jnp.int32), (B, T))
    return ids, pos


def _reference_mask(attn_mask_type, B, T_t, T_s, seg_q=None, pos_q=None):
    """Independent (B, T_t, T_s) bool oracle for the kernel's mask.

    Written from the mask *definitions* (attention.py's AttnMaskType docstrings)
    rather than from the (seg, key) reduction under test, so a bug in that
    reduction cannot cancel out against the oracle.
    """
    if attn_mask_type == "no_mask":
        return None
    causal = "causal" in attn_mask_type
    padding = "padding" in attn_mask_type
    if padding:
        same = seg_q[:, :, None] == seg_q[:, None, :]
        real = (seg_q != 0)
        valid = same & real[:, :, None] & real[:, None, :]
        if causal:
            valid = valid & (pos_q[:, :, None] >= pos_q[:, None, :])
        return valid
    order = jnp.arange(T_t)[:, None] >= jnp.arange(T_s)[None, :]
    return jnp.broadcast_to(order, (B, T_t, T_s))


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
@pytest.mark.parametrize("k", [32, 100, 128, 256, 512, 1024])
def test_topk_matches_reference(k, fp8):
    """``indexer_topk`` selects the same scores as reference + ``jax.lax.top_k``.

    Index set-equality is too strict (the op's logits and the fp32 reference
    break ties differently), so the check is on the *scores* at the selected
    indices, compared rank-by-rank against the reference top-k. ``T_s`` is
    ``4 * max(k)`` so the largest ``k`` (1024) still sits at the top quartile
    rather than sweeping nearly the whole row. ``k=100`` covers a non-power-of-2
    ``k``, which the previous fused kernel could not accept.

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
    assert max_gap < tol, f"top-k scores diverge: max_gap={max_gap:.3e} (k={k})"


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


# --- Masking (variable-length / THD support) ---------------------------------
#
# The score kernel applies one predicate, `seg_q == seg_k and key_q >= key_k`,
# and fills every other slot with finfo(dtype).min. These tests check that
# predicate against an oracle written from the mask definitions, that gradients
# are zero at masked pairs, and -- the reason the mask exists -- that top-k can
# no longer reach across a segment boundary.

_MASK_TYPES = ["padding", "causal", "padding_causal"]


def _masked_case(mask_type, B, T, lengths=(40, 16)):
    """Segment metadata + reference mask for a packed self-attention batch."""
    seg, pos = _segments(B, T, list(lengths))
    mask = _reference_mask(mask_type, B, T, T, seg, pos)
    kwargs = {"attn_mask_type": mask_type}
    if "padding" in mask_type:
        kwargs.update(segment_ids_q=seg, segment_pos_q=pos)
    return seg, pos, mask, kwargs


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("mask_type", _MASK_TYPES)
def test_mask_matches_reference(mask_type, fp8):
    """Masked forward matches the einsum reference with the same mask applied."""
    B, oH, T = 2, 2, 64
    args = _indexer_inputs(B, oH, T, T, d=32, d_c=32, H=8, d_i=32, seed=500)
    _, _, mask, kwargs = _masked_case(mask_type, B, T)

    o_ref = _indexer_reference(*args, mask=mask)
    o_out = indexer(*args, fp8=fp8, **kwargs)
    assert o_out.shape == o_ref.shape

    fill = _mask_fill(o_out.dtype)
    # Boolean indexing does not broadcast, so widen the mask over the head axis.
    valid = jnp.broadcast_to(mask[:, None], o_out.shape)
    # Masked slots must be exactly the fill, not merely close to it.
    assert jnp.all(o_out[~valid] == fill)
    # Valid slots are compared on their own, so the fill (which is ~1e38 and
    # would swamp any norm) cannot hide a real error.
    assert _rel_err(o_out[valid], o_ref[valid]) < _FWD_TOL[fp8]


def test_causal_mask_is_lower_triangular():
    """``causal`` keeps s <= t and drops s > t — catches a >= / > slip."""
    B, oH, T_t, T_s = 1, 2, 32, 48
    args = _indexer_inputs(B, oH, T_t, T_s, d=32, d_c=32, H=8, d_i=32, seed=501)
    o = indexer(*args, attn_mask_type="causal")
    fill = _mask_fill(o.dtype)

    lower = jnp.arange(T_t)[:, None] >= jnp.arange(T_s)[None, :]
    assert jnp.all(o[:, :, ~lower] == fill), "kept a strictly-future key"
    assert not jnp.any(o[:, :, lower] == fill), "dropped a valid past key"


def test_padding_rows_are_entirely_fill():
    """A padding query matches nothing — including other padding.

    Padding is segment id 0 on both sides; the two sides are mapped to *distinct*
    sentinels so pad-vs-pad fails too. These rows are also the fully-masked-tile
    case, so this exercises the kernel's tile-skip path.
    """
    B, oH, T = 2, 2, 64
    lengths = [40, 16]  # 8 trailing padding tokens
    args = _indexer_inputs(B, oH, T, T, d=32, d_c=32, H=8, d_i=32, seed=502)
    seg, pos = _segments(B, T, lengths)
    o = indexer(*args, attn_mask_type="padding_causal",
                segment_ids_q=seg, segment_pos_q=pos)

    pad_rows = o[:, :, sum(lengths):, :]
    assert pad_rows.size > 0, "degenerate test: no padding tokens"
    assert jnp.all(pad_rows == _mask_fill(o.dtype))


def test_topk_never_selects_masked():
    """The payoff: top-k cannot reach across a segment or into the future.

    This is what fails without an in-kernel mask — the logits are signed (ReLU
    sits inside the H-reduction, ahead of the signed W_o weighting), so nothing
    outside the kernel can keep a cross-segment key from outranking a valid
    negative one.

    Only rows with at least ``k`` valid keys are checked: a query earlier than
    ``k`` positions into its segment genuinely has fewer candidates than asked
    for, and its surplus slots are masked ones by construction.
    """
    B, oH, T, k = 2, 2, 64, 4
    lengths = [40, 16]
    args = _indexer_inputs(B, oH, T, T, d=32, d_c=32, H=16, d_i=32, seed=503)
    seg, pos = _segments(B, T, lengths)

    idx = indexer_topk(*args, k=k, attn_mask_type="padding_causal",
                       segment_ids_q=seg, segment_pos_q=pos)
    assert idx.shape == (B, oH, T, k)

    picked_seg = jnp.take_along_axis(seg[:, None, None, :].repeat(oH, 1).repeat(T, 2),
                                     idx, axis=-1)          # (B, oH, T, k)
    picked_pos = jnp.take_along_axis(pos[:, None, None, :].repeat(oH, 1).repeat(T, 2),
                                     idx, axis=-1)
    # Rows with >= k valid keys: real token, at least k-1 positions into its segment.
    checkable = ((seg != 0) & (pos >= k - 1))[:, None, :, None]

    same_seg = picked_seg == seg[:, None, :, None]
    causal = picked_pos <= pos[:, None, :, None]
    assert jnp.all(jnp.where(checkable, same_seg, True)), "top-k crossed a segment"
    assert jnp.all(jnp.where(checkable, causal, True)), "top-k selected a future key"


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("mask_type", _MASK_TYPES)
def test_mask_backward_matches_reference_grad(mask_type, fp8):
    """Masked pairs contribute no gradient.

    The cotangent is all-ones *including at masked slots*, so the op has to zero
    them itself; a kernel that only masked the forward would leak gradient here.
    Using ``jax.vjp`` rather than ``grad(sum(...))`` keeps the fill value out of
    the loss, where it would otherwise overflow to -inf.
    """
    B, oH, T = 2, 2, 32
    args = _indexer_inputs(B, oH, T, T, d=32, d_c=32, H=8, d_i=32, seed=504)
    _, _, mask, kwargs = _masked_case(mask_type, B, T, lengths=(20, 8))

    def _grads(fn):
        out, vjp_fn = jax.vjp(fn, *args)
        return vjp_fn(jnp.ones_like(out))

    grads_ref = _grads(lambda *a: _indexer_reference(*a, mask=mask))
    grads_out = _grads(lambda *a: indexer(*a, fp8=fp8, **kwargs))
    for gr, go in zip(grads_ref, grads_out):
        assert _rel_err(go, gr) < _BWD_TOL[fp8]


def test_permissive_mask_is_bit_identical_to_no_mask():
    """A mask that permits everything changes nothing, bit for bit.

    ``HAS_MASK`` is a ``tl.constexpr``, so ``no_mask`` compiles the mask away
    entirely and the two paths take different code. Feeding the masked path a
    predicate that is true everywhere (one segment, constant key) isolates the
    masking machinery itself: any difference here is the mask perturbing the
    accumulation, not a genuine masking decision.
    """
    B, oH, T = 2, 2, 64
    args = _indexer_inputs(B, oH, T, T, d=32, d_c=32, H=8, d_i=32, seed=505)
    H_q, H_k, W_o = _indexer_projections(*args)

    permissive = (
        jnp.zeros((B, T), jnp.int32),  # seg_q — one segment
        jnp.zeros((B, T), jnp.int32),  # key_q — constant, so key_q >= key_k holds
        jnp.zeros((B, T), jnp.int32),  # seg_k
        jnp.zeros((B, T), jnp.int32),  # key_k
    )
    assert jnp.array_equal(
        score_reduce_triton(H_q, H_k, W_o),
        score_reduce_triton(H_q, H_k, W_o, mask=permissive),
    )


# --- Compacted tile list ------------------------------------------------------
#
# Instead of launching the full rectangle and skipping fully-masked tiles
# in-kernel, the masked path precomputes which tiles survive, compacts them into
# a list, and launches over that. It is a launch-shape change and nothing else,
# so the bar is bit-identity with the rectangular path -- not a tolerance.


def _causal_tiles_bruteforce(T_t, T_s, BLOCK_T, BLOCK_S):
    """Tiles containing at least one causal pair, counted directly."""
    n_t = -(-T_t // BLOCK_T)
    n_s = -(-T_s // BLOCK_S)
    return sum(
        1
        for i in range(n_t)
        for j in range(n_s)
        if j * BLOCK_S <= min(i * BLOCK_T + BLOCK_T - 1, T_t - 1)
    )


def _tile_map_bruteforce(seg_q, key_q, seg_k, key_k):
    """Per tile, does any (t, s) in it actually satisfy the predicate?"""
    v = (seg_q[:, :, None] == seg_k[:, None, :]) & (key_q[:, :, None] >= key_k[:, None, :])
    B, T_t, T_s = v.shape
    nt, ns = -(-T_t // _COMPACT_TILE_T), -(-T_s // _COMPACT_TILE_S)
    out = np.zeros((B, nt, ns), bool)
    v = np.asarray(v)
    for i in range(nt):
        for j in range(ns):
            sl = v[:, i * _COMPACT_TILE_T:(i + 1) * _COMPACT_TILE_T,
                   j * _COMPACT_TILE_S:(j + 1) * _COMPACT_TILE_S]
            out[:, i, j] = sl.any(axis=(1, 2))
    return out


@pytest.mark.parametrize("mask_type", _MASK_TYPES)
def test_tile_map_never_drops_a_live_tile(mask_type):
    """The tile map may over-keep, but must never lose a tile that has work.

    A dropped tile is never dispatched and silently reads back as MASK_FILL, so
    false negatives are the one failure mode that breaks correctness. False
    positives only cost a wasted tile and are allowed.
    """
    B, T = 2, 1024
    seg, pos = _segments(B, T, [300, 150, 400])
    kwargs = {"segment_ids_q": seg, "segment_pos_q": pos} if "padding" in mask_type else {}
    mask = _mask_keys(B, T, T, mask_type,
                      kwargs.get("segment_ids_q"), kwargs.get("segment_pos_q"),
                      kwargs.get("segment_ids_q"), kwargs.get("segment_pos_q"))

    got = np.asarray(_tile_validity_map(*mask))
    want = _tile_map_bruteforce(*[np.asarray(x) for x in mask])
    assert not (want & ~got).any(), f"{int((want & ~got).sum())} live tiles dropped"

    # The launch order must be a *permutation*: every tile exactly once, live
    # ones (non-negative payload) first, ruled-out ones encoded as -(id + 1).
    # Anything less and some output slot is written by no CTA at all.
    lst = np.asarray(_tile_launch_order(jnp.asarray(got)))
    n = got.shape[1] * got.shape[2]
    for b in range(B):
        kept = sorted(np.flatnonzero(got[b].reshape(-1)).tolist())
        live, dead = lst[b][lst[b] >= 0], lst[b][lst[b] < 0]
        assert sorted(live.tolist()) == kept
        assert sorted((-dead - 1).tolist()) == sorted(set(range(n)) - set(kept))
        assert np.array_equal(lst[b][:len(kept)], live), "live tiles must come first"


def test_tile_map_is_exact_under_causal():
    """With one segment the map must keep exactly the tiles a causal mask needs.

    The conservative range test admits false positives in general, but under a
    plain causal mask it should reduce to the triangular enumeration with none
    at all -- so this pins the tight case, not just the safe one. The sizes
    below include one that divides neither block size.
    """
    B = 1
    for T in (512, 1000, 1024, 4096):
        mask = _mask_keys(B, T, T, "causal")
        n = int(np.asarray(_tile_validity_map(*mask))[0].sum())
        assert n == _causal_tiles_bruteforce(T, T, _COMPACT_TILE_T, _COMPACT_TILE_S), T


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("mask_type", _MASK_TYPES)
def test_launch_paths_are_bit_identical(monkeypatch, mask_type, fp8):
    """rectangular vs COMPACT is a launch shape, nothing more.

    The two must agree bit for bit -- this is not a tolerance question. T is far
    below _COMPACT_MIN_T, so the env knob is what forces compaction on for the
    single-segment causal case. The segment lengths are unequal and do not tile
    evenly, so partially-live and cross-segment tiles both appear.
    """
    B, oH, T = 2, 2, 512
    args = _indexer_inputs(B, oH, T, T, d=32, d_c=32, H=8, d_i=32, seed=507)
    _, _, _, kwargs = _masked_case(mask_type, B, T, lengths=(300, 150))

    def run(compact):
        monkeypatch.setenv("NVTE_INDEXER_COMPACT", compact)
        _compact_override.cache_clear()
        # The env var is read during lowering, so a cached trace would silently
        # reuse the other path and make this comparison vacuous.
        jax.clear_caches()
        return indexer(*args, fp8=fp8, **kwargs)

    assert jnp.array_equal(run("1"), run("0"))

    _compact_override.cache_clear()
    jax.clear_caches()


def test_unsupported_mask_type_is_rejected():
    """Bottom-right causal is not implemented and must say so, not mis-mask."""
    args = _indexer_inputs(1, 1, T_t=32, T_s=32, d=32, d_c=32, H=8, d_i=32, seed=506)
    with pytest.raises(NotImplementedError, match="not supported by the indexer"):
        indexer(*args, attn_mask_type="causal_bottom_right")


def test_lightning_indexer_mask_passthrough():
    """``LightningIndexer(attn_mask_type=...)`` matches the functional call."""
    B, oH, T, d, d_c, H, d_i = 2, 2, 64, 32, 32, 8, 32
    keys = jax.random.split(jax.random.PRNGKey(11), 3)
    Q = jax.random.normal(keys[0], (B, oH, T, d), dtype=jnp.bfloat16)
    K = jax.random.normal(keys[1], (B, oH, T, d), dtype=jnp.bfloat16)
    seg, pos = _segments(B, T, [40, 16])

    mod = LightningIndexer(num_heads=H, d_c=d_c, d_i=d_i,
                           attn_mask_type="padding_causal")
    variables = mod.init(keys[2], Q, K, seg, pos)
    o_mod = mod.apply(variables, Q, K, seg, pos)

    p = variables["params"]
    o_fn = indexer(Q, K, p["W_uq"], p["W_dq"], p["W_k"], p["W_w"],
                   attn_mask_type="padding_causal",
                   segment_ids_q=seg, segment_pos_q=pos)
    assert jnp.array_equal(o_mod, o_fn)


def test_lightning_indexer_topk_mode():
    """``LightningIndexer(topk=k)`` returns top-k indices of shape (..., T, k)."""
    B, oH, T_t, T_s, d, d_c, H, d_i, k = 2, 3, 64, 128, 32, 32, 16, 32, 32
    keys = jax.random.split(jax.random.PRNGKey(9), 2)
    Q = jax.random.normal(keys[0], (B, oH, T_t, d), dtype=jnp.bfloat16)
    K = jax.random.normal(keys[1], (B, oH, T_s, d), dtype=jnp.bfloat16)

    mod = LightningIndexer(num_heads=H, d_c=d_c, d_i=d_i, topk=k)
    variables = mod.init(jax.random.PRNGKey(0), Q, K)
    idx = mod.apply(variables, Q, K)
    assert idx.shape == (B, oH, T_t, k)
    assert idx.dtype == jnp.int32
