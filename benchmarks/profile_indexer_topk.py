"""Benchmark fused indexer+topk vs reference (full score then jax.lax.top_k).

Production config: B=4, H=16, T_t=T_s=4096, d=128, I=4, d_i=64, k=64, bf16.

Sweeps (block_t, block_s, num_warps, num_stages) for the triton kernel and
reports TFLOP/s, ms, and vs-reference speedup. FLOPs counted as the underlying
indexer compute (top-k itself is comparison-only, treated as 0 FLOP).

Usage:
  docker exec zain-w2 sh -c 'cd /workspace && python benchmarks/profile_indexer_topk.py'
"""

import time
import functools

import jax
import jax.numpy as jnp

from transformer_engine.jax.indexer import _indexer_impl_reference, quantize_to_fp8
from transformer_engine.jax.triton_extensions.indexer import (
    indexer_fused_topk_triton,
    indexer_fused_triton,
)
try:
    from transformer_engine.jax.pallas_kernels.indexer import indexer_fused as _pallas_indexer
    _HAVE_PALLAS = True
except Exception:
    _pallas_indexer = None
    _HAVE_PALLAS = False


_FP8_DTYPES = (
    jnp.dtype("float8_e4m3fn"), jnp.dtype("float8_e5m2"),
    jnp.dtype("float8_e4m3fnuz"), jnp.dtype("float8_e5m2fnuz"),
)


def _is_fp8(dt):
    return jnp.dtype(dt) in _FP8_DTYPES


def make_inputs(B, H, T_t, T_s, d, I, d_i, dtype, seed=0):
    keys = jax.random.split(jax.random.PRNGKey(seed), 5)
    Q       = jax.random.normal(keys[0], (B, H, T_t, d),  dtype=dtype)
    K       = jax.random.normal(keys[1], (B, H, T_s, d),  dtype=dtype)
    W_q     = jax.random.normal(keys[2], (I, d, d_i),     dtype=dtype)
    W_k     = jax.random.normal(keys[3], (d, d_i),        dtype=dtype)
    weights = jax.random.normal(keys[4], (B, H, T_t, I),  dtype=dtype)
    return Q, K, W_q, W_k, weights


def make_fp8_inputs(B, H, T_t, T_s, d, I, d_i, *, fp8_dtype, seed=0):
    Q, K, W_q, W_k, weights = make_inputs(
        B, H, T_t, T_s, d, I, d_i, jnp.bfloat16, seed=seed
    )
    Q_q,  sq  = quantize_to_fp8(Q,   dtype=fp8_dtype)
    K_q,  sk  = quantize_to_fp8(K,   dtype=fp8_dtype)
    Wq_q, swq = quantize_to_fp8(W_q, dtype=fp8_dtype)
    Wk_q, swk = quantize_to_fp8(W_k, dtype=fp8_dtype)
    return Q_q, K_q, Wq_q, Wk_q, weights, dict(
        scale_q=sq, scale_k=sk, scale_wq=swq, scale_wk=swk,
    )


def theoretical_flops(B, H, T_t, T_s, d, I, d_i):
    n = B * H
    return 2 * (
        n * T_t * I * d_i * d
        + n * T_s * d_i * d
        + n * T_t * I * T_s * d_i
        + n * T_t * T_s * I
    )


def time_fn(fn, args, n_warmup=5, n_iter=50):
    for _ in range(n_warmup):
        out = fn(*args)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), out)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        out = fn(*args)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), out)
    return (time.perf_counter() - t0) / n_iter


# Reference, pallas+topk, triton+topk: each accepts an optional `scales` dict
# (None for high-precision). Built fresh per-config since the scales are baked
# into the closure.
def _make_reference_topk(scales):
    if scales is None:
        @jax.jit
        def fn(Q, K, W_q, W_k, weights):
            scores = _indexer_impl_reference(Q, K, W_q, W_k, weights)
            return jax.lax.top_k(scores, K_TOPK_GLOBAL)
    else:
        @jax.jit
        def fn(Q, K, W_q, W_k, weights):
            scores = _indexer_impl_reference(Q, K, W_q, W_k, weights, **scales)
            return jax.lax.top_k(scores, K_TOPK_GLOBAL)
    return fn


def _make_pallas_then_topk(scales):
    if not _HAVE_PALLAS:
        return None
    if scales is None:
        @jax.jit
        def fn(Q, K, W_q, W_k, weights):
            scores = _pallas_indexer(Q, K, W_q, W_k, weights)
            return jax.lax.top_k(scores, K_TOPK_GLOBAL)
    else:
        @jax.jit
        def fn(Q, K, W_q, W_k, weights):
            scores = _pallas_indexer(Q, K, W_q, W_k, weights, **scales)
            return jax.lax.top_k(scores, K_TOPK_GLOBAL)
    return fn


def _make_triton_then_topk(scales):
    if scales is None:
        @jax.jit
        def fn(Q, K, W_q, W_k, weights):
            scores = indexer_fused_triton(Q, K, W_q, W_k, weights)
            return jax.lax.top_k(scores, K_TOPK_GLOBAL)
    else:
        @jax.jit
        def fn(Q, K, W_q, W_k, weights):
            scores = indexer_fused_triton(Q, K, W_q, W_k, weights, **scales)
            return jax.lax.top_k(scores, K_TOPK_GLOBAL)
    return fn


# Standalone: just time jax.lax.top_k on a precomputed score matrix.
@jax.jit
def topk_only(scores):
    return jax.lax.top_k(scores, K_TOPK_GLOBAL)


def _make_triton(k, bt, bs, nw, ns):
    fn = jax.jit(functools.partial(
        indexer_fused_topk_triton,
        k=k, block_t=bt, block_s=bs, num_warps=nw, num_stages=ns,
    ))
    return fn


CONFIGS = [
    # (B, H, T_t, T_s, d, I, d_i, dtype)
    ( 4, 16, 2048, 2048, 128, 4,  64, jnp.bfloat16),
    ( 4, 16, 4096, 4096, 128, 4,  64, jnp.bfloat16),
    # FP8 e4m3 — fused-topk Triton kernel doesn't accept FP8 yet; the row will
    # report "(skipped: fp8 not supported)" for that impl. The other three
    # paths (reference, pallas+topk, triton+topk) all run end-to-end in FP8.
    ( 4, 16, 2048, 2048, 128, 4,  64, jnp.float8_e4m3fn),
    ( 4, 16, 4096, 4096, 128, 4,  64, jnp.float8_e4m3fn),
]

K_TOPK_GLOBAL = 64

SWEEP = [
    # (block_t, block_s, num_warps, num_stages)
    ( 64,  64, 4, 1),
    ( 64,  64, 8, 1),
    (128,  64, 4, 1),
    (128,  64, 8, 1),
    ( 32,  32, 4, 1),
    ( 32,  64, 4, 1),
    ( 32, 128, 4, 1),  # k=64+128=192 not pow2; will be skipped
    ( 64,  32, 4, 1),
    (256,  64, 4, 1),
    (256,  64, 8, 1),
]


def main():
    print(f"jax devices: {jax.devices()}\nk = {K_TOPK_GLOBAL}\n")
    for cfg in CONFIGS:
        B, H, T_t, T_s, d, I, d_i, dtype = cfg
        is_fp8 = _is_fp8(dtype)
        if is_fp8:
            Q, K, W_q, W_k, weights, scales = make_fp8_inputs(
                B, H, T_t, T_s, d, I, d_i, fp8_dtype=dtype
            )
            args = (Q, K, W_q, W_k, weights)
        else:
            args = make_inputs(B, H, T_t, T_s, d, I, d_i, dtype)
            scales = None
        flops = theoretical_flops(B, H, T_t, T_s, d, I, d_i)
        print(f"--- B={B} H={H} T_t={T_t} T_s={T_s} d={d} I={I} d_i={d_i} {dtype.dtype.name} ---")
        print(f"    theoretical work = {flops/1e9:.2f} GFLOPs/call")

        impls = [
            ("ref(einsum+topk)", _make_reference_topk(scales)),
            ("pallas+topk",      _make_pallas_then_topk(scales)),
            ("triton+topk",      _make_triton_then_topk(scales)),
        ]

        ref_ms = None
        for name, fn in impls:
            if fn is None:
                continue
            try:
                sec = time_fn(fn, args)
                ms = sec * 1e3
                tflops = flops / sec / 1e12
                if name == "pallas+topk":
                    ref_ms = ms
                print(f"    {name:<22} {ms:8.3f} ms   {tflops:6.2f} TFLOP/s")
            except Exception as e:  # noqa: BLE001
                print(f"    {name:<22} FAILED: {type(e).__name__}: {str(e).splitlines()[0]}")

        # Time top_k alone (on pre-materialized scores). For FP8 inputs the
        # reference dequantizes internally and returns a high-precision matrix.
        try:
            if scales is None:
                scores_mat = _indexer_impl_reference(*args)
            else:
                scores_mat = _indexer_impl_reference(*args, **scales)
            sec = time_fn(topk_only, (scores_mat,))
            print(f"    {'(top_k alone)':<22} {sec*1e3:8.3f} ms")
        except Exception as e:  # noqa: BLE001
            print(f"    (top_k alone)         FAILED: {type(e).__name__}")

        # Fused-topk Triton kernel does not accept FP8 yet — skip the sweep
        # for FP8 configs.
        if is_fp8:
            print(f"    {'fused-topk triton':<22} (skipped: fp8 not supported by topk kernel)")
            print()
            continue

        # Triton fused-topk sweep (high-precision only)
        for bt, bs, nw, ns in SWEEP:
            if (K_TOPK_GLOBAL + bs) & (K_TOPK_GLOBAL + bs - 1) != 0:
                continue  # k+block_s must be pow2
            label = f"triton bt={bt} bs={bs} W={nw} S={ns}"
            try:
                fn = _make_triton(K_TOPK_GLOBAL, bt, bs, nw, ns)
                sec = time_fn(fn, args)
                ms = sec * 1e3
                tflops = flops / sec / 1e12
                speed = f" ({ref_ms/ms:.2f}x ref)" if ref_ms else ""
                print(f"    {label:<22} {ms:8.3f} ms   {tflops:6.2f} TFLOP/s{speed}")
            except Exception as e:  # noqa: BLE001
                print(f"    {label:<22} FAILED: {type(e).__name__}: {str(e).splitlines()[0]}")
        print()


if __name__ == "__main__":
    main()
