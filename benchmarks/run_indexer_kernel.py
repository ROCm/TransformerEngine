"""Minimal direct invocation of the low-rank indexer kernel for profiling.

No baselines, no comparisons. Just: build inputs once, jit the kernel,
warm it up, then run a fixed number of iterations under whatever
profiler is wrapping this process.

Run inside the container:
  docker exec zain-w2 sh -c 'cd /workspace && python benchmarks/run_indexer_kernel.py'
"""

import argparse
import time

import jax
import jax.numpy as jnp

from transformer_engine.jax.indexer import quantize_to_fp8
from transformer_engine.jax.triton_extensions.indexer import indexer_fused_triton as _triton_indexer

_BACKENDS = {
    "triton": _triton_indexer,
}

_DTYPE_MAP = {
    "bf16": jnp.bfloat16,
    "fp32": jnp.float32,
    "fp8":  jnp.float8_e4m3fn,
}


def make_inputs(B, oH, T, S, d, d_c, H, d_i, dtype, seed=0):
    keys = jax.random.split(jax.random.PRNGKey(seed), 6)
    Q       = jax.random.normal(keys[0], (B, oH, T, d),    dtype=dtype)
    K       = jax.random.normal(keys[1], (B, oH, S, d),    dtype=dtype)
    W_uq    = jax.random.normal(keys[2], (H, d_c, d_i),    dtype=dtype)
    W_dq    = jax.random.normal(keys[3], (d, d_c),         dtype=dtype)
    W_k     = jax.random.normal(keys[4], (d, d_i),         dtype=dtype)
    weights = jax.random.normal(keys[5], (B, oH, H, T),    dtype=dtype)
    return Q, K, W_uq, W_dq, W_k, weights


def make_fp8_inputs(B, oH, T, S, d, d_c, H, d_i, *, fp8_dtype, seed=0):
    """Quantize all five matrices to FP8; weights stay bf16."""
    Q, K, W_uq, W_dq, W_k, weights = make_inputs(
        B, oH, T, S, d, d_c, H, d_i, jnp.bfloat16, seed=seed
    )
    Q_q,  sq   = quantize_to_fp8(Q,   dtype=fp8_dtype)
    K_q,  sk   = quantize_to_fp8(K,   dtype=fp8_dtype)
    Wuq_q, swq = quantize_to_fp8(W_uq, dtype=fp8_dtype)
    Wdq_q, swd = quantize_to_fp8(W_dq, dtype=fp8_dtype)
    Wk_q,  swk = quantize_to_fp8(W_k,  dtype=fp8_dtype)
    return Q_q, K_q, Wuq_q, Wdq_q, Wk_q, weights, dict(
        scale_q=sq, scale_k=sk, scale_wq=swq, scale_wd=swd, scale_wk=swk,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--B",   type=int, default=4)
    p.add_argument("--oH",  type=int, default=16, help="outer (multi-attn) heads")
    p.add_argument("--T",   type=int, default=2048)
    p.add_argument("--S",   type=int, default=2048)
    p.add_argument("--d",   type=int, default=512, help="hidden dim")
    p.add_argument("--d_c", type=int, default=128, help="down-projection rank")
    p.add_argument("--H",   type=int, default=64,  help="indexer-head count")
    p.add_argument("--d_i", type=int, default=128, help="per-indexer-head dim")
    p.add_argument("--dtype", choices=list(_DTYPE_MAP), default="bf16")
    p.add_argument("--warmup",  type=int, default=5)
    p.add_argument("--iters",   type=int, default=50)
    p.add_argument("--backend", choices=list(_BACKENDS), default="triton")
    args = p.parse_args()

    dtype = _DTYPE_MAP[args.dtype]
    is_fp8 = args.dtype == "fp8"
    print(f"jax devices: {jax.devices()}")
    print(f"shape: B={args.B} oH={args.oH} T={args.T} S={args.S} "
          f"d={args.d} d_c={args.d_c} H={args.H} d_i={args.d_i} "
          f"dtype={args.dtype} backend={args.backend}")

    if is_fp8:
        Q, K, W_uq, W_dq, W_k, weights, scales = make_fp8_inputs(
            args.B, args.oH, args.T, args.S,
            args.d, args.d_c, args.H, args.d_i, fp8_dtype=dtype,
        )
        inputs = (Q, K, W_uq, W_dq, W_k, weights)
    else:
        inputs = make_inputs(args.B, args.oH, args.T, args.S,
                             args.d, args.d_c, args.H, args.d_i, dtype)
        scales = None

    raw_fn = _BACKENDS[args.backend]

    if scales is None:
        @jax.jit
        def fn(Q, K, W_uq, W_dq, W_k, weights):
            return raw_fn(Q, K, W_uq, W_dq, W_k, weights)
    else:
        @jax.jit
        def fn(Q, K, W_uq, W_dq, W_k, weights):
            return raw_fn(Q, K, W_uq, W_dq, W_k, weights, **scales)

    # Warmup: triggers JIT compile + first-launch overhead.
    for _ in range(args.warmup):
        out = fn(*inputs)
    jax.block_until_ready(out)

    # Timed region: this is what the profiler should focus on.
    t0 = time.perf_counter()
    for _ in range(args.iters):
        out = fn(*inputs)
    jax.block_until_ready(out)
    sec = (time.perf_counter() - t0) / args.iters
    print(f"avg per call: {sec*1e3:.3f} ms ({args.iters} iters)")


if __name__ == "__main__":
    main()
