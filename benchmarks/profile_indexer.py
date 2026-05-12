"""Profile the low-rank lightning-indexer at realistic shapes (bf16).

Measures wall time and effective TFLOPS for the einsum baseline vs the
fused Triton kernel.

Run inside the container:
  docker exec zain-w2 sh -c 'cd /workspace && python benchmarks/profile_indexer.py'
"""

import time

import jax
import jax.numpy as jnp

from transformer_engine.jax.indexer import indexer

try:
    from transformer_engine.jax.triton_extensions.indexer import score_reduce_triton  # noqa: F401
    _HAVE_HYBRID = True
except Exception as _e:  # noqa: BLE001
    _HAVE_HYBRID = False
    _HYBRID_IMPORT_ERROR = _e


# --- Inputs / FLOP accounting ----------------------------------------------------

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
    # 2 flops per multiply-add. Counts the contractions in the low-rank
    # indexer with learnable output-weight projection:
    #   C_q = Q @ W_dq                  : 2 * B*oH * T * d_c * d
    #   H_q = einsum(C_q, W_uq)         : 2 * B*oH * T * H * d_i * d_c
    #   H_k = K @ W_k                    : 2 * B*oH * S * d_i * d
    #   scores = relu(H_q @ H_k^T)       : 2 * B*oH * T * H * S * d_i
    #   W_o = Q @ W_w                    : 2 * B*oH * T * d * H
    #   O   = sum_h scores * W_o         : 2 * B*oH * T * S * H
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
    jax.block_until_ready(out)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        out = fn(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - t0) / n_iter


# --- Driver ---------------------------------------------------------------------

CONFIGS = [
    #(B, oH, T,    S,    d,   d_c,  H,  d_i)
    ( 2, 64, 1024, 1024, 512, 1024, 64, 128),
]


def _build_impl(backend):
    @jax.jit
    def fn(Q, K, W_uq, W_dq, W_k, W_w):
        return indexer(Q, K, W_uq, W_dq, W_k, W_w, backend=backend)
    return fn


def _dump_autotuner_winner():
    """Print the autotuner-selected config(s) for _score_reduce_kernel."""
    if not _HAVE_HYBRID:
        return
    try:
        from transformer_engine.jax.triton_extensions.indexer import (
            _score_reduce_kernel,
        )
    except ImportError:
        return
    cache = getattr(_score_reduce_kernel, "cache", None)
    if not cache:
        print("    [autotune] no cache entries")
        return
    for key, cfg in cache.items():
        print(f"    [autotune] key={key} -> {cfg}")


if not _HAVE_HYBRID:
    print(f"[profile_indexer] Hybrid backend unavailable: {_HYBRID_IMPORT_ERROR}")


def main():
    print(f"jax devices: {jax.devices()}\n")
    for B, oH, T, S, d, d_c, H, d_i in CONFIGS:
        Q, K, W_uq, W_dq, W_k, W_w = make_inputs(
            B, oH, T, S, d, d_c, H, d_i, jnp.bfloat16
        )
        args = (Q, K, W_uq, W_dq, W_k, W_w)
        flops = theoretical_flops(B, oH, T, S, d, d_c, H, d_i)

        print(f"--- B={B} oH={oH} T={T} S={S} d={d} d_c={d_c} H={H} d_i={d_i} bfloat16 ---")
        print(f"    theoretical work = {flops/1e9:.2f} GFLOPs/call")

        impls = [("baseline", _build_impl("reference"))]
        if _HAVE_HYBRID:
            impls.append(("hybrid", _build_impl("hybrid")))

        baseline_ms = None
        for name, fn in impls:
            try:
                sec = time_fn(fn, args)
                tflops = flops / sec / 1e12
                ms = sec * 1e3
                if name == "baseline":
                    baseline_ms = ms
                    speed = ""
                else:
                    speed = f" ({baseline_ms/ms:.2f}x baseline)"
                print(f"    {name:<10} {ms:8.3f} ms   {tflops:6.2f} TFLOP/s{speed}")
            except Exception as e:  # noqa: BLE001
                print(f"    {name:<10} FAILED: {type(e).__name__}: {str(e).splitlines()[0]}")
        _dump_autotuner_winner()
        print()


if __name__ == "__main__":
    main()
