"""Profile indexer + per-row top-k along T_s (bf16).

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

from transformer_engine.jax.sparse_attention.indexer import indexer, indexer_topk

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
    #(B, oH, T,    S,    d,   d_c,  H,  d_i)
    ( 2, 64, 1024, 1024, 512, 1024, 64, 128),
]

K_TOPK = 512


def _build_topk(backend, k):
    @jax.jit
    def fn(Q, K, W_uq, W_dq, W_k, W_w):
        scores = indexer(Q, K, W_uq, W_dq, W_k, W_w, backend=backend)
        return jax.lax.top_k(scores, k)
    return fn


def _build_fused_topk(k):
    @jax.jit
    def fn(Q, K, W_uq, W_dq, W_k, W_w):
        return indexer_topk(Q, K, W_uq, W_dq, W_k, W_w, k=k)
    return fn


@jax.jit
def _topk_only(scores):
    return jax.lax.top_k(scores, K_TOPK)


if not _HAVE_HYBRID:
    print(f"[profile_indexer_topk] Hybrid backend unavailable: {_HYBRID_IMPORT_ERROR}")


def main():
    print(f"jax devices: {jax.devices()}\nk = {K_TOPK}\n")
    for B, oH, T, S, d, d_c, H, d_i in CONFIGS:
        Q, K, W_uq, W_dq, W_k, W_w = make_inputs(
            B, oH, T, S, d, d_c, H, d_i, jnp.bfloat16
        )
        args = (Q, K, W_uq, W_dq, W_k, W_w)
        flops = theoretical_flops(B, oH, T, S, d, d_c, H, d_i)

        print(f"--- B={B} oH={oH} T={T} S={S} d={d} d_c={d_c} H={H} d_i={d_i} bfloat16 ---")
        print(f"    theoretical work = {flops/1e9:.2f} GFLOPs/call (top-k = 0 FLOP)")

        # impls = [("baseline+topk", _build_topk("reference", K_TOPK))]
        impls = []
        if _HAVE_HYBRID:
            impls.append(("hybrid+topk", _build_topk("hybrid", K_TOPK)))
            impls.append(("hybrid_fused_topk", _build_fused_topk(K_TOPK)))

        baseline_ms = None
        for name, fn in impls:
            try:
                sec = time_fn(fn, args)
                ms = sec * 1e3
                tflops = flops / sec / 1e12
                if name == "baseline+topk":
                    baseline_ms = ms
                    speed = ""
                elif baseline_ms is not None:
                    speed = f" ({baseline_ms/ms:.2f}x baseline)"
                else:
                    speed = ""
                print(f"    {name:<18} {ms:8.3f} ms   {tflops:6.2f} TFLOP/s{speed}")
            except Exception as e:  # noqa: BLE001
                print(f"    {name:<18} FAILED: {type(e).__name__}: {str(e).splitlines()[0]}")

        # Time top_k alone on a precomputed (reference) score matrix to
        # isolate the top-k cost from the indexer compute.
        try:
            scores_mat = indexer(*args, backend="reference")
            sec = time_fn(_topk_only, (scores_mat,))
            print(f"    {'(top_k alone)':<18} {sec*1e3:8.3f} ms")
        except Exception as e:  # noqa: BLE001
            print(f"    {'(top_k alone) FAILED':<18} {type(e).__name__}")
        print()


if __name__ == "__main__":
    main()
