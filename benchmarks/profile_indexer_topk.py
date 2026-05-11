"""Profile indexer + per-row top-k along T_s.

Same canonical backends as ``profile_indexer.py`` (reference einsum vs
hybrid einsum+Triton score-reduce), with ``jax.lax.top_k`` applied to the
score matrix. Reports wall time and effective TFLOPS for the indexer
compute (top-k is comparison-only and counted as 0 FLOP).

Run inside the container:
  docker exec zain-w2 sh -c 'cd /workspace && python benchmarks/profile_indexer_topk.py'
"""

import time

import jax
import jax.numpy as jnp

from transformer_engine.jax.indexer import indexer, quantize_to_fp8

# Triton hybrid backend: einsum projections + Triton score-relu-reduce.
try:
    from transformer_engine.jax.triton_extensions.indexer import score_reduce_triton  # noqa: F401
    _HAVE_HYBRID = True
except Exception as _e:  # noqa: BLE001
    _HAVE_HYBRID = False
    _HYBRID_IMPORT_ERROR = _e


# --- Inputs / FLOP accounting ---------------------------------------------------
# Mirrors profile_indexer.py — keeping the two profilers in lockstep.

def make_inputs(B, oH, T, S, d, d_c, H, d_i, dtype, seed=0):
    keys = jax.random.split(jax.random.PRNGKey(seed), 6)
    Q    = jax.random.normal(keys[0], (B, oH, T, d), dtype=dtype)
    K    = jax.random.normal(keys[1], (B, oH, S, d), dtype=dtype)
    W_uq = jax.random.normal(keys[2], (H, d_c, d_i), dtype=dtype)
    W_dq = jax.random.normal(keys[3], (d, d_c),      dtype=dtype)
    W_k  = jax.random.normal(keys[4], (d, d_i),      dtype=dtype)
    W_w  = jax.random.normal(keys[5], (d, H),        dtype=dtype)
    return Q, K, W_uq, W_dq, W_k, W_w


def make_fp8_inputs(B, oH, T, S, d, d_c, H, d_i, *,
                    fp8_dtype=jnp.float8_e4m3fn, weights_dtype=jnp.bfloat16,
                    seed=0):
    Q, K, W_uq, W_dq, W_k, W_w = make_inputs(
        B, oH, T, S, d, d_c, H, d_i, jnp.bfloat16, seed=seed
    )
    Q_q,  sq   = quantize_to_fp8(Q,   dtype=fp8_dtype)
    K_q,  sk   = quantize_to_fp8(K,   dtype=fp8_dtype)
    Wuq_q, swq = quantize_to_fp8(W_uq, dtype=fp8_dtype)
    Wdq_q, swd = quantize_to_fp8(W_dq, dtype=fp8_dtype)
    Wk_q,  swk = quantize_to_fp8(W_k,  dtype=fp8_dtype)
    W_w = W_w.astype(weights_dtype)
    scales = dict(scale_q=sq, scale_k=sk,
                  scale_wq=swq, scale_wd=swd, scale_wk=swk)
    return Q_q, K_q, Wuq_q, Wdq_q, Wk_q, W_w, scales


def theoretical_flops(B, oH, T, S, d, d_c, H, d_i):
    # 2 flops per multiply-add. top-k is comparison-only, counted as 0 FLOP.
    n = B * oH
    return 2 * (
        n * T * d_c * d
        + n * T * H * d_i * d_c
        + n * S * d_i * d
        + n * T * H * S * d_i
        + n * T * d * H
        + n * T * S * H
    )


def time_fn(fn, args, n_warmup=15, n_iter=50):
    for _ in range(n_warmup):
        out = fn(*args)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), out)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        out = fn(*args)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), out)
    return (time.perf_counter() - t0) / n_iter


# --- Driver ---------------------------------------------------------------------

CONFIGS = [
    #(B, oH, T,    S,    d,   d_c,  H,  d_i, dtype)
    ( 2, 64, 1024, 1024, 512, 1024, 64, 128, jnp.bfloat16),
]

K_TOPK = 64


def _is_fp8(dt):
    return jnp.dtype(dt) in (
        jnp.dtype("float8_e4m3fn"), jnp.dtype("float8_e5m2"),
        jnp.dtype("float8_e4m3fnuz"), jnp.dtype("float8_e5m2fnuz"),
    )


def _bind_topk(scales, *, backend, k):
    """Build a jit'd indexer-then-topk closure for the given backend + scales."""
    extra = {"backend": backend}
    if scales is not None:
        merged = dict(extra, **scales)
    else:
        merged = extra

    @jax.jit
    def fn(Q, K, W_uq, W_dq, W_k, W_w):
        scores = indexer(Q, K, W_uq, W_dq, W_k, W_w, **merged)
        return jax.lax.top_k(scores, k)

    return fn


def _build_impls(scales, k):
    impls = [
        ("baseline+topk", _bind_topk(scales, backend="reference", k=k)),
    ]
    if _HAVE_HYBRID:
        impls.append(
            ("hybrid+topk",  _bind_topk(scales, backend="hybrid", k=k))
        )
    return impls


@jax.jit
def _topk_only(scores):
    return jax.lax.top_k(scores, K_TOPK)


if not _HAVE_HYBRID:
    print(f"[profile_indexer_topk] Hybrid backend unavailable: {_HYBRID_IMPORT_ERROR}")


def main():
    print(f"jax devices: {jax.devices()}\nk = {K_TOPK}\n")
    for cfg in CONFIGS:
        B, oH, T, S, d, d_c, H, d_i, dtype = cfg
        is_fp8 = _is_fp8(dtype)
        if is_fp8:
            Q, K, W_uq, W_dq, W_k, W_w, scales = make_fp8_inputs(
                B, oH, T, S, d, d_c, H, d_i, fp8_dtype=dtype
            )
        else:
            Q, K, W_uq, W_dq, W_k, W_w = make_inputs(
                B, oH, T, S, d, d_c, H, d_i, dtype
            )
            scales = None
        args = (Q, K, W_uq, W_dq, W_k, W_w)
        impls = _build_impls(scales, K_TOPK)
        flops = theoretical_flops(B, oH, T, S, d, d_c, H, d_i)

        print(f"--- B={B} oH={oH} T={T} S={S} d={d} d_c={d_c} H={H} d_i={d_i} "
              f"{dtype.dtype.name} ---")
        print(f"    theoretical work = {flops/1e9:.2f} GFLOPs/call (top-k = 0 FLOP)")
        baseline_ms = None
        for name, fn in impls:
            try:
                sec = time_fn(fn, args)
                ms = sec * 1e3
                tflops = flops / sec / 1e12
                if name == "baseline+topk":
                    baseline_ms = ms
                    speed = ""
                else:
                    speed = f" ({baseline_ms/ms:.2f}x baseline)"
                print(f"    {name:<14} {ms:8.3f} ms   {tflops:6.2f} TFLOP/s{speed}")
            except Exception as e:  # noqa: BLE001
                print(f"    {name:<14} FAILED: {type(e).__name__}: {str(e).splitlines()[0]}")

        # Time top_k alone on a precomputed (reference) score matrix to
        # isolate the top-k cost from the indexer compute.
        try:
            kw = {"backend": "reference", **(scales or {})}
            scores_mat = indexer(*args, **kw)
            sec = time_fn(_topk_only, (scores_mat,))
            print(f"    {'(top_k alone)':<14} {sec*1e3:8.3f} ms")
        except Exception as e:  # noqa: BLE001
            print(f"    (top_k alone)  FAILED: {type(e).__name__}")
        print()


if __name__ == "__main__":
    main()
