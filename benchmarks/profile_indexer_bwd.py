"""Profile lightning-indexer backward pass throughput (bf16).

Measures wall time and effective TFLOPS for forward, backward, and
value_and_grad. Uses the standard "backward = 2x forward FLOPs" convention,
so value_and_grad total work = 3x forward FLOPs.

Run inside the container:
  docker exec zain-w2 sh -c 'cd /workspace && python benchmarks/profile_indexer_bwd.py'
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


def make_inputs(B, oH, T, S, d, d_c, H, d_i, dtype, seed=0):
    keys = jax.random.split(jax.random.PRNGKey(seed), 6)
    Q    = jax.random.normal(keys[0], (B, oH, T, d), dtype=dtype)
    K    = jax.random.normal(keys[1], (B, oH, S, d), dtype=dtype)
    W_uq = jax.random.normal(keys[2], (H, d_c, d_i), dtype=dtype)
    W_dq = jax.random.normal(keys[3], (d, d_c),      dtype=dtype)
    W_k  = jax.random.normal(keys[4], (d, d_i),      dtype=dtype)
    W_w  = jax.random.normal(keys[5], (d, H),        dtype=dtype)
    return Q, K, W_uq, W_dq, W_k, W_w


def theoretical_fwd_flops(B, oH, T, S, d, d_c, H, d_i):
    n = B * oH
    return 2 * (
        n * T * d_c * d
        + n * T * H * d_i * d_c
        + n * S * d_i * d
        + n * T * H * S * d_i
        + n * T * d * H
        + n * T * S * H
    )


def time_fn(fn, args, n_warmup=10, n_iter=30):
    for _ in range(n_warmup):
        out = fn(*args)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), out)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        out = fn(*args)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), out)
    return (time.perf_counter() - t0) / n_iter


CONFIGS = [
    #(B, oH, T,    S,    d,   d_c,  H,  d_i)
    ( 2, 64, 1024, 1024, 512, 1024, 64, 128),
]


def _build_fwd(backend):
    @jax.jit
    def fn(Q, K, W_uq, W_dq, W_k, W_w):
        O = indexer(Q, K, W_uq, W_dq, W_k, W_w, backend=backend)
        return jnp.sum(O.astype(jnp.float32))
    return fn


def _build_bwd(backend):
    """Backward only: returns gradients."""
    fwd = _build_fwd(backend)
    return jax.jit(jax.grad(fwd, argnums=(0, 1, 2, 3, 4, 5)))


def _build_value_and_grad(backend):
    fwd = _build_fwd(backend)
    return jax.jit(jax.value_and_grad(fwd, argnums=(0, 1, 2, 3, 4, 5)))


def main():
    print(f"jax devices: {jax.devices()}\n")
    for B, oH, T, S, d, d_c, H, d_i in CONFIGS:
        Q, K, W_uq, W_dq, W_k, W_w = make_inputs(
            B, oH, T, S, d, d_c, H, d_i, jnp.bfloat16
        )
        args = (Q, K, W_uq, W_dq, W_k, W_w)
        fwd_flops = theoretical_fwd_flops(B, oH, T, S, d, d_c, H, d_i)

        print(f"--- B={B} oH={oH} T={T} S={S} d={d} d_c={d_c} H={H} d_i={d_i} bfloat16 ---")
        print(f"    forward GFLOPs/call:   {fwd_flops/1e9:.2f}")
        print(f"    bwd GFLOPs/call (~2x): {2*fwd_flops/1e9:.2f}")
        print(f"    f+b GFLOPs/call (~3x): {3*fwd_flops/1e9:.2f}")
        print()

        backends = ["reference"]
        if _HAVE_HYBRID:
            backends.append("hybrid")

        # Headers
        print(f"    {'backend':<10s} {'pass':<14s}   {'ms':>8s}   {'TFLOP/s':>8s}")

        for backend in backends:
            try:
                # Forward (loss only)
                fwd = _build_fwd(backend)
                sec = time_fn(fwd, args)
                ms = sec * 1e3
                tflops = fwd_flops / sec / 1e12
                print(f"    {backend:<10s} {'forward':<14s}   {ms:8.3f}   {tflops:8.2f}")

                # Backward only (jax.grad — XLA may re-trace forward inside)
                bwd = _build_bwd(backend)
                sec = time_fn(bwd, args)
                ms = sec * 1e3
                tflops = 2 * fwd_flops / sec / 1e12   # bwd ~= 2x fwd
                print(f"    {backend:<10s} {'backward':<14s}   {ms:8.3f}   {tflops:8.2f}")

                # value_and_grad (forward + backward, single pass)
                vag = _build_value_and_grad(backend)
                sec = time_fn(vag, args)
                ms = sec * 1e3
                tflops = 3 * fwd_flops / sec / 1e12   # f+b ~= 3x fwd
                print(f"    {backend:<10s} {'value_and_grad':<14s}   {ms:8.3f}   {tflops:8.2f}")
            except Exception as e:  # noqa: BLE001
                print(f"    {backend:<10s} FAILED: {type(e).__name__}: {str(e).splitlines()[0]}")
            print()


if __name__ == "__main__":
    main()
