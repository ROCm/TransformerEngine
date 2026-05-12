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


if __name__ == "__main__":
    print("=== reference vs reference (sanity) ===")
    _run_test((2, 3), seed=0, backend="reference")

    print("\n=== hybrid vs reference ===")
    _run_test((2, 3), seed=100, backend="hybrid")
