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


def _indexer_impl_reference(Q, K, W_uq, W_dq, W_k, W_w, out_dtype=None):
    """
    Q       [..., T, d]
    K       [..., S, d]
    W_dq    [d, d_c]
    W_uq    [H, d_c, d_i]
    W_k     [d, d_i]
    W_w     [..., d, H]    # leading dims must match Q's
    """
    C_q = jnp.einsum("...td,dc->...tc", Q, W_dq)                # (..., T, d_c)
    H_q = jnp.einsum("...tc,hci->...thi", C_q, W_uq)            # (..., T, H, d_i)
    H_k = jnp.einsum("...sd,di->...si", K, W_k)                 # (..., S, d_i)
    H = jax.nn.relu(jnp.einsum("...thi,...si->...ths", H_q, H_k))  # (..., T, H, S)
    W_o = jnp.einsum("...td,dh->...th", Q, W_w)
    O = jnp.einsum("...ths,...th->...ts", H, W_o)               # (..., T, S)
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

    C_q = jnp.einsum("...td,dc->...tc", Q, W_dq)         # (..., T, d_c)
    H_q = jnp.einsum("...tc,hci->...thi", C_q, W_uq)     # (..., T, H, d_i)
    H_k = jnp.einsum("...sd,di->...si", K, W_k)          # (..., S, d_i)
    W_o = jnp.einsum("...td,dh->...th", Q, W_w)          # (..., T, H)

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
    C_q = jnp.einsum("...td,dc->...tc", Q, W_dq)         # (..., T, d_c)
    H_q = jnp.einsum("...tc,hci->...thi", C_q, W_uq)     # (..., T, H, d_i)
    H_k = jnp.einsum("...sd,di->...si", K, W_k)          # (..., S, d_i)
    W_o = jnp.einsum("...td,dh->...th", Q, weights)      # (..., T, H)
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


# --- Tests ----------------------------------------------------------------------

def _run_test(leading_shape, seed, backend):
    # The hybrid backend's Triton kernel requires rank-4 BHSD inputs.
    T_t, T_s, d, d_c, H, d_i = 64, 64, 32, 32, 8, 32
    keys = jax.random.split(jax.random.PRNGKey(seed), 6)
    Q   = jax.random.normal(keys[0], (*leading_shape, T_t, d), dtype=jnp.bfloat16)
    K   = jax.random.normal(keys[1], (*leading_shape, T_s, d), dtype=jnp.bfloat16)
    W_uq = jax.random.normal(keys[2], (H, d_c, d_i),           dtype=jnp.bfloat16)
    W_dq = jax.random.normal(keys[3], (d, d_c),                dtype=jnp.bfloat16)
    W_k  = jax.random.normal(keys[4], (d, d_i),                dtype=jnp.bfloat16)
    W_w  = jax.random.normal(keys[5], (d, H),                  dtype=jnp.bfloat16)

    O_ref = indexer(Q, K, W_uq, W_dq, W_k, W_w, backend="reference")
    O_b   = indexer(Q, K, W_uq, W_dq, W_k, W_w, backend=backend)

    diff = (O_ref.astype(jnp.float32) - O_b.astype(jnp.float32))
    rel_err = float(jnp.linalg.norm(diff) /
                    (jnp.linalg.norm(O_ref.astype(jnp.float32)) + 1e-30))
    tag = "OK" if rel_err < 5e-3 else "FAIL"
    print(f"  backend={backend:<10s} leading={str(leading_shape):10s} "
          f"O.shape={O_b.shape}  rel.err={rel_err:.2e}  [{tag}]")


def _run_topk_test(leading_shape, seed, k):
    # H=16 to keep the matmul in [BLOCK_S, H] friendly to MFMA tile sizes.
    T_t, T_s, d, d_c, H, d_i = 64, 128, 32, 32, 16, 32
    keys = jax.random.split(jax.random.PRNGKey(seed), 6)
    Q   = jax.random.normal(keys[0], (*leading_shape, T_t, d), dtype=jnp.bfloat16)
    K   = jax.random.normal(keys[1], (*leading_shape, T_s, d), dtype=jnp.bfloat16)
    W_uq = jax.random.normal(keys[2], (H, d_c, d_i),           dtype=jnp.bfloat16)
    W_dq = jax.random.normal(keys[3], (d, d_c),                dtype=jnp.bfloat16)
    W_k  = jax.random.normal(keys[4], (d, d_i),                dtype=jnp.bfloat16)
    W_w  = jax.random.normal(keys[5], (d, H),                  dtype=jnp.bfloat16)

    O_ref = indexer(Q, K, W_uq, W_dq, W_k, W_w, backend="reference")
    topk_fused = indexer_topk(Q, K, W_uq, W_dq, W_k, W_w, k=k)

    # Correctness check: the scores at the fused-picked indices should equal the
    # top-k scores from the reference (within bf16 noise). Set-equality of indices
    # is too strict — different backends break ties differently.
    O_ref32 = O_ref.astype(jnp.float32)
    ref_topk_vals = jax.lax.top_k(O_ref32, k=k)[0]               # [..., T_t, k] sorted desc
    fused_picked_vals = jnp.take_along_axis(O_ref32, topk_fused, axis=-1)
    fused_picked_sorted = jnp.sort(fused_picked_vals, axis=-1)[..., ::-1]
    rel_diff = jnp.abs(ref_topk_vals - fused_picked_sorted) / (jnp.abs(ref_topk_vals) + 1e-6)
    max_rel = float(rel_diff.max())
    tag = "OK" if max_rel < 1e-2 else f"FAIL (max_rel={max_rel:.2e})"
    print(f"  topk     leading={str(leading_shape):10s} k={k:<4d} "
          f"out.shape={topk_fused.shape}  max_rel={max_rel:.2e}  [{tag}]")


def _run_bwd_test(leading_shape, seed):
    """Compare hybrid backward against jax.grad on the reference impl."""
    T_t, T_s, d, d_c, H, d_i = 32, 32, 32, 32, 8, 32
    keys = jax.random.split(jax.random.PRNGKey(seed), 6)
    Q   = jax.random.normal(keys[0], (*leading_shape, T_t, d), dtype=jnp.bfloat16)
    K   = jax.random.normal(keys[1], (*leading_shape, T_s, d), dtype=jnp.bfloat16)
    W_uq = jax.random.normal(keys[2], (H, d_c, d_i),           dtype=jnp.bfloat16)
    W_dq = jax.random.normal(keys[3], (d, d_c),                dtype=jnp.bfloat16)
    W_k  = jax.random.normal(keys[4], (d, d_i),                dtype=jnp.bfloat16)
    W_w  = jax.random.normal(keys[5], (d, H),                  dtype=jnp.bfloat16)

    def loss_ref(Q, K, W_uq, W_dq, W_k, W_w):
        O = indexer(Q, K, W_uq, W_dq, W_k, W_w, backend="reference")
        return jnp.sum(O.astype(jnp.float32))

    def loss_hyb(Q, K, W_uq, W_dq, W_k, W_w):
        O = indexer(Q, K, W_uq, W_dq, W_k, W_w, backend="hybrid")
        return jnp.sum(O.astype(jnp.float32))

    grads_ref = jax.grad(loss_ref, argnums=(0, 1, 2, 3, 4, 5))(Q, K, W_uq, W_dq, W_k, W_w)
    grads_hyb = jax.grad(loss_hyb, argnums=(0, 1, 2, 3, 4, 5))(Q, K, W_uq, W_dq, W_k, W_w)

    names = ("dQ", "dK", "dW_uq", "dW_dq", "dW_k", "dW_w")
    all_ok = True
    for name, gr, gh in zip(names, grads_ref, grads_hyb):
        diff = (gr.astype(jnp.float32) - gh.astype(jnp.float32))
        rel = float(jnp.linalg.norm(diff) /
                    (jnp.linalg.norm(gr.astype(jnp.float32)) + 1e-30))
        ok = rel < 5e-2
        all_ok = all_ok and ok
        tag = "OK" if ok else "FAIL"
        print(f"    {name:<6} shape={str(gh.shape):<22s} rel.err={rel:.2e}  [{tag}]")
    overall = "OK" if all_ok else "FAIL"
    print(f"  bwd      leading={str(leading_shape):10s}  overall=[{overall}]")


if __name__ == "__main__":
    print("=== reference vs reference (sanity) ===")
    _run_test((2, 3), seed=0, backend="reference")

    print("\n=== hybrid vs reference ===")
    _run_test((2, 3), seed=100, backend="hybrid")

    print("\n=== indexer_topk vs reference + jax.lax.top_k ===")
    _run_topk_test((2, 3), seed=200, k=32)

    print("\n=== backward: hybrid vs jax.grad(reference) ===")
    _run_bwd_test((2, 3), seed=300)
