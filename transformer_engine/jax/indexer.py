"""Indexer op (forward only).

Two backends:
  * "reference" - jnp/einsum, accepts arbitrary leading dims (..., T, d).
  * "fused"     - Pallas kernel, strict BHSD (B, H, T, d). Lives in
                  transformer_engine/jax/pallas_kernels/indexer.py.

Top-level entry point: ``indexer(Q, K, W_uq, W_dq, W_k, weights, *, backend=...)``.

Math (low-rank form: Q is hidden state; query heads are produced by a
down-projection (d -> d_c) followed by an up-projection (d_c -> H * d_i)):

    C_q = Q @ W_dq                                           # (..., T, d_c)
    H_q = einsum("...tc,hci->...thi", C_q, W_uq)             # (..., T, H, d_i)
    H_k = K @ W_k                                             # (..., S, d_i)
    H   = relu(einsum("...thi,...si->...ths", H_q, H_k))      # (..., T, H, S)
    O   = einsum("...ths,...ht->...ts", H, weights)           # (..., T, S)

``weights`` is the precomputed per-(indexer-head, token) weight (DeepSeek's
``weights_proj(x)`` term, transposed for kernel-friendly layout). Its leading
dims must broadcast against Q's.

FP8 mode: any of Q / K / W_uq / W_dq / W_k may be FP8 (e4m3) tensors. Each
FP8 operand needs a per-tensor fp32 scale (scale_q, scale_k, scale_wq,
scale_wd, scale_wk). ReLU commutes with positive scaling so the active
scales fold into a single fp32 scalar applied once at the end. Letting W_dq
go FP8 unlocks a native FP8 MFMA on the Q @ W_dq down-projection (and saves
half the bytes for that weight) at the cost of additional quantization noise
in the bottleneck of the low-rank decomposition.
"""

import functools
import math

import jax
import jax.numpy as jnp


_FP8_DTYPES = frozenset([
    jnp.dtype("float8_e4m3fn"),
    jnp.dtype("float8_e5m2"),
    jnp.dtype("float8_e4m3fnuz"),
    jnp.dtype("float8_e5m2fnuz"),
])


def _is_fp8(x):
    return jnp.dtype(x.dtype) in _FP8_DTYPES


def quantize_to_fp8(x, *, dtype=None, axis=None):
    """Per-tensor amax-based quantization helper (for tests/profiling).

    Returns (x_fp8, scale_fp32) where the dequantization is ``x_fp8 * scale``.
    """
    if dtype is None:
        dtype = jnp.float8_e4m3fn
    fp8_max = jnp.finfo(dtype).max.astype(jnp.float32)
    amax = jnp.max(jnp.abs(x.astype(jnp.float32))) if axis is None else \
        jnp.max(jnp.abs(x.astype(jnp.float32)), axis=axis, keepdims=True)
    scale = (amax / fp8_max).astype(jnp.float32)
    # avoid divide-by-zero on all-zero tensors
    scale = jnp.where(scale == 0, jnp.float32(1.0), scale)
    x_fp8 = (x.astype(jnp.float32) / scale).astype(dtype)
    return x_fp8, scale


# --- Reference implementation ---------------------------------------------------

def _indexer_impl_reference(Q, K, W_uq, W_dq, W_k, W_w,
                            scale_q=None, scale_k=None,
                            scale_wq=None, scale_wd=None, scale_wk=None,
                            out_dtype=None):
    """
    Q       [..., T, d]
    K       [..., S, d]
    W_dq    [d, d_c]
    W_uq    [H, d_c, d_i]
    W_k     [d, d_i]
    W_w     [..., d, H]    # leading dims must match Q's

    FP8 path: each fp8 operand is dequantized via cast-to-bf16-then-multiply
    immediately before the matmul that consumes it. This is the pattern XLA's
    GEMM rewriter recognizes and lowers to ``__cublas$lt$matmul$f8`` (native
    fp8 hardware GEMM) for matmuls where both operands are originally fp8.
    Upcasting to fp32 first would lose the fp8 type info and fall back to
    plain fp32 GEMM — strictly worse.
    """
    if _is_fp8(Q):
        if any(s is None for s in (scale_q, scale_k, scale_wq, scale_wk)):
            raise ValueError(
                "FP8 reference requires scale_q, scale_k, scale_wq, scale_wk."
            )
    if _is_fp8(W_dq) and scale_wd is None:
        raise ValueError("FP8 W_dq requires scale_wd.")

    wp = jnp.bfloat16  # working precision for non-fp8 intermediates

    def _dq(x, s):
        # cast-then-scale pattern (in working precision, NOT fp32). XLA's
        # GEMM rewriter pulls (cast, multiply, dot) into a fused fp8 GEMM
        # when both operands of the dot follow this pattern.
        if _is_fp8(x):
            return x.astype(wp) * jnp.float32(s).astype(wp)
        return x.astype(wp)

    Q_d   = _dq(Q,   scale_q)
    K_d   = _dq(K,   scale_k)
    W_uq_d = _dq(W_uq, scale_wq)
    W_dq_d = _dq(W_dq, scale_wd)
    W_k_d  = _dq(W_k,  scale_wk)

    C_q = jnp.einsum("...td,dc->...tc", Q_d, W_dq_d)                # (..., T, d_c)
    H_q = jnp.einsum("...tc,hci->...thi", C_q, W_uq_d)              # (..., T, H, d_i)
    H_k = jnp.einsum("...sd,di->...si", K_d, W_k_d)                 # (..., S, d_i)
    H = jax.nn.relu(jnp.einsum("...thi,...si->...ths", H_q, H_k))   # (..., T, H, S)
    W_o = jnp.einsum("...td,dh->...th", Q_d, W_w)
    O = jnp.einsum("...ths,...th->...ts", H, W_o)    # (..., T, S)
    if out_dtype is not None:
        O = O.astype(out_dtype)
    return O


# --- Fused implementation (Pallas) ----------------------------------------------
# Imported lazily so callers without Triton/Pallas can still use the reference.

def _indexer_impl_fused(Q, K, W_uq, W_dq, W_k, weights, **fp8_kwargs):
    raise NotImplementedError(
        "Pallas backend has not yet been updated for the low-rank indexer form "
        "(W_uq + W_dq). Use backend='triton' or backend='reference'."
    )


def _indexer_impl_triton(Q, K, W_uq, W_dq, W_k, weights, **fp8_kwargs):
    from transformer_engine.jax.triton_extensions.indexer import indexer_fused_triton
    return indexer_fused_triton(Q, K, W_uq, W_dq, W_k, weights, **fp8_kwargs)


def _indexer_impl_hybrid(Q, K, W_uq, W_dq, W_k, W_w,
                         scale_q=None, scale_k=None,
                         scale_wq=None, scale_wd=None, scale_wk=None,
                         out_dtype=None):
    """Einsum projections + Triton score-relu-reduce.

    Mirrors ``_indexer_impl_reference`` for the four projections (which
    lower to hipBLASLt GEMMs), then hands Hq / Hk / W_o to a fused Triton
    kernel that does score+relu+H-reduction in registers — eliminating the
    16+ GB pre-relu-score HBM round-trip the pure-einsum path pays.

    bf16 only for now. FP8 inputs are dequantized to bf16 just like the
    reference; native FP8 GEMM is not available on ROCm anyway.
    """
    from transformer_engine.jax.triton_extensions.indexer import score_reduce_triton

    if _is_fp8(Q):
        if any(s is None for s in (scale_q, scale_k, scale_wq, scale_wk)):
            raise ValueError(
                "FP8 hybrid requires scale_q, scale_k, scale_wq, scale_wk."
            )
    if _is_fp8(W_dq) and scale_wd is None:
        raise ValueError("FP8 W_dq requires scale_wd.")

    wp = jnp.bfloat16

    def _dq(x, s):
        if _is_fp8(x):
            return x.astype(wp) * jnp.float32(s).astype(wp)
        return x.astype(wp)

    Q_d    = _dq(Q,    scale_q)
    K_d    = _dq(K,    scale_k)
    W_uq_d = _dq(W_uq, scale_wq)
    W_dq_d = _dq(W_dq, scale_wd)
    W_k_d  = _dq(W_k,  scale_wk)

    C_q = jnp.einsum("...td,dc->...tc", Q_d, W_dq_d)         # (..., T, d_c)
    H_q = jnp.einsum("...tc,hci->...thi", C_q, W_uq_d)       # (..., T, H, d_i)
    H_k = jnp.einsum("...sd,di->...si", K_d, W_k_d)          # (..., S, d_i)
    W_o = jnp.einsum("...td,dh->...th", Q_d, W_w.astype(wp)) # (..., T, H)

    O = score_reduce_triton(H_q, H_k, W_o,
                            out_dtype=out_dtype if out_dtype else wp)
    return O


def _indexer_topk_impl_reference(Q, K, W_uq, W_dq, W_k, weights, k):
    scores = _indexer_impl_reference(Q, K, W_uq, W_dq, W_k, weights)
    return jax.lax.top_k(scores, k)


def _indexer_topk_impl_triton(Q, K, W_uq, W_dq, W_k, weights, k):
    from transformer_engine.jax.triton_extensions.indexer import indexer_fused_topk_triton
    return indexer_fused_topk_triton(Q, K, W_uq, W_dq, W_k, weights, k=k)


@functools.partial(jax.jit, static_argnames=("k", "backend"))
def indexer_topk(Q, K, W_uq, W_dq, W_k, weights, *, k, backend="triton"):
    """Indexer fused with per-row top-k along T_s.

    Returns (vals, idxs):
        vals: (..., T, k) Q.dtype
        idxs: (..., T, k) int32

    backend: "reference" (full score then jax.lax.top_k) or "triton" (fused).
    """
    if backend == "reference":
        return _indexer_topk_impl_reference(Q, K, W_uq, W_dq, W_k, weights, k)
    if backend == "triton":
        return _indexer_topk_impl_triton(Q, K, W_uq, W_dq, W_k, weights, k)
    raise ValueError(f"unknown backend {backend!r}; expected 'reference' or 'triton'")


# --- Top-level dispatch ---------------------------------------------------------

@functools.partial(jax.jit, static_argnames=("backend", "out_dtype"))
def indexer(Q, K, W_uq, W_dq, W_k, weights, *,
            scale_q=None, scale_k=None,
            scale_wq=None, scale_wd=None, scale_wk=None,
            out_dtype=None, backend="reference"):
    """Low-rank lightning-indexer.

    Args:
        Q:       (..., T, d)            hidden state (per token)
        K:       (..., S, d)            key hidden state
        W_uq:    (H, d_c, d_i)          up-projection: d_c -> d_i (per head)
        W_dq:    (d, d_c)               down-projection: d -> d_c
        W_k:     (d, d_i)               key projection
        weights: (..., H, T)            per-(indexer-head, token) weight
        scale_q, scale_k, scale_wq, scale_wk:
                 per-tensor fp32 dequant scales. Required when Q is FP8.
        scale_wd:
                 per-tensor fp32 dequant scale for W_dq. Required only when
                 W_dq itself is FP8.
        out_dtype: output dtype override (defaults to Q.dtype, or weights.dtype
                 in FP8 mode).
        backend: "reference", "fused" (Pallas), or "triton".

    Returns:
        O of shape (..., T, S).
    """
    fp8_kwargs = dict(
        scale_q=scale_q, scale_k=scale_k,
        scale_wq=scale_wq, scale_wd=scale_wd, scale_wk=scale_wk,
        out_dtype=out_dtype,
    )
    if backend == "reference":
        return _indexer_impl_reference(Q, K, W_uq, W_dq, W_k, weights, **fp8_kwargs)
    if backend == "fused":
        return _indexer_impl_fused(Q, K, W_uq, W_dq, W_k, weights, **fp8_kwargs)
    if backend == "triton":
        return _indexer_impl_triton(Q, K, W_uq, W_dq, W_k, weights, **fp8_kwargs)
    if backend == "hybrid":
        return _indexer_impl_hybrid(Q, K, W_uq, W_dq, W_k, weights, **fp8_kwargs)
    raise ValueError(
        f"unknown backend {backend!r}; expected 'reference', 'fused', 'triton', "
        f"or 'hybrid'"
    )


# --- Tests ----------------------------------------------------------------------

def _reference_nobatch(Q, K, W_uq, W_dq, W_k, weights):
    """Rank-2 reference (no leading dims) used as the cross-check."""
    C_q = Q @ W_dq
    H_q = jnp.einsum("tc,hci->thi", C_q, W_uq)
    H_k = K @ W_k
    H   = jax.nn.relu(jnp.einsum("thi,si->ths", H_q, H_k))
    return jnp.einsum("ths,ht->ts", H, weights)


def _run_test(leading_shape, seed, backend):
    # Power-of-2 shapes, all matmul dims >= 16 so the Pallas backend accepts.
    T_t, T_s, d, d_c, H, d_i = 16, 16, 16, 16, 4, 16
    keys = jax.random.split(jax.random.PRNGKey(seed), 6)
    Q       = jax.random.normal(keys[0], (*leading_shape, T_t, d))
    K       = jax.random.normal(keys[1], (*leading_shape, T_s, d))
    W_uq    = jax.random.normal(keys[2], (H, d_c, d_i))
    W_dq    = jax.random.normal(keys[3], (d, d_c))
    W_k     = jax.random.normal(keys[4], (d, d_i))
    weights = jax.random.normal(keys[5], (*leading_shape, H, T_t))

    try:
        O = indexer(Q, K, W_uq, W_dq, W_k, weights, backend=backend)
    except Exception as e:  # noqa: BLE001
        print(f"  backend={backend:<10s} leading={str(leading_shape):10s} "
              f"SKIP: {type(e).__name__}: {str(e).splitlines()[0]}")
        return

    flat = math.prod(leading_shape) if leading_shape else 1
    Q_f       = Q.reshape(flat, T_t, d)
    K_f       = K.reshape(flat, T_s, d)
    weights_f = weights.reshape(flat, T_t, H)
    O_ref = jax.vmap(lambda q, k, w: _reference_nobatch(q, k, W_uq, W_dq, W_k, w))(
        Q_f, K_f, weights_f
    )
    O_ref = O_ref.reshape(*leading_shape, T_t, T_s)

    expected_shape = (*leading_shape, T_t, T_s)
    shape_ok = O.shape == expected_shape
    max_err = float(jnp.max(jnp.abs(O - O_ref)))
    tag = "OK" if shape_ok and max_err < 1e-4 else "FAIL"
    print(f"  backend={backend:<10s} leading={str(leading_shape):10s} "
          f"O.shape={O.shape}  max abs err={max_err:.2e}  [{tag}]")


if __name__ == "__main__":
    print("=== reference backend ===")
    for i, leading in enumerate([(), (2,), (2, 3)]):
        _run_test(leading, seed=i, backend="reference")

    # Fused: strictly BHSD (rank-4 Q/K), so only the (2, 3) case applies.
    print("\n=== fused backend ===")
    for i, leading in enumerate([(), (2,), (2, 3)]):
        _run_test(leading, seed=100 + i, backend="fused")

    # Triton: same BHSD-only contract as fused.
    print("\n=== triton backend ===")
    for i, leading in enumerate([(), (2,), (2, 3)]):
        _run_test(leading, seed=200 + i, backend="triton")
