"""Profile the low-rank lightning-indexer at realistic shapes.

Measures wall time and effective TFLOPS for the einsum baseline vs the
fused Triton kernel.

Run inside the container:
  docker exec zain-w2 sh -c 'cd /workspace && python benchmarks/profile_indexer.py'
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


# --- Inputs / FLOP accounting ----------------------------------------------------

def make_inputs(B, oH, T, S, d, d_c, H, d_i, dtype, seed=0):
    keys = jax.random.split(jax.random.PRNGKey(seed), 6)
    Q    = jax.random.normal(keys[0], (B, oH, T, d), dtype=dtype)
    K    = jax.random.normal(keys[1], (B, oH, S, d), dtype=dtype)
    W_uq = jax.random.normal(keys[2], (H, d_c, d_i), dtype=dtype)
    W_dq = jax.random.normal(keys[3], (d, d_c),      dtype=dtype)
    W_k  = jax.random.normal(keys[4], (d, d_i),      dtype=dtype)
    # Learnable per-(token, indexer-head) weight projection: W_o = Q @ W_w.
    W_w  = jax.random.normal(keys[5], (d, H),        dtype=dtype)
    return Q, K, W_uq, W_dq, W_k, W_w


def make_fp8_inputs(B, oH, T, S, d, d_c, H, d_i, *,
                    fp8_dtype=jnp.float8_e4m3fn, weights_dtype=jnp.bfloat16,
                    seed=0):
    """Sample bf16 tensors then quantize Q/K/W_uq/W_dq/W_k to FP8.

    W_w stays in ``weights_dtype`` (bf16) — the reference impl does not
    dequantize it.

    Returns (Q, K, W_uq, W_dq, W_k, W_w, scales_dict).
    """
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
    #(B, oH, T,    S,    d,   d_c,  H,  d_i, dtype)
    ( 2, 64, 1024, 1024, 512, 1024, 64, 128, jnp.bfloat16),
]


def _is_fp8(dt):
    return jnp.dtype(dt) in (
        jnp.dtype("float8_e4m3fn"), jnp.dtype("float8_e5m2"),
        jnp.dtype("float8_e4m3fnuz"), jnp.dtype("float8_e5m2fnuz"),
    )


def _bind_scales(fn, scales, *, backend=None):
    """Return a 6-arg jit-able function that internally adds scale kwargs.

    If ``backend`` is given, it is forwarded as a kwarg to ``fn`` (used to
    select between einsum / hybrid / pure-triton via the same ``indexer``
    entry point).
    """
    extra = {}
    if backend is not None:
        extra["backend"] = backend
    if scales is None and not extra:
        return jax.jit(fn)
    @jax.jit
    def wrapped(Q, K, W_uq, W_dq, W_k, W_w):
        kwargs = dict(extra)
        if scales is not None:
            kwargs.update(scales)
        return fn(Q, K, W_uq, W_dq, W_k, W_w, **kwargs)
    return wrapped


def _build_impls(scales):
    impls = [
        ("baseline", _bind_scales(indexer, scales, backend="reference")),
    ]
    if _HAVE_HYBRID:
        impls.append(("hybrid", _bind_scales(indexer, scales, backend="hybrid")))
    return impls


if not _HAVE_HYBRID:
    print(f"[profile_indexer] Hybrid backend unavailable: {_HYBRID_IMPORT_ERROR}")


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


def main():
    print(f"jax devices: {jax.devices()}\n")
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
        impls = _build_impls(scales)
        flops = theoretical_flops(B, oH, T, S, d, d_c, H, d_i)

        print(f"--- B={B} oH={oH} T={T} S={S} d={d} d_c={d_c} H={H} d_i={d_i} "
              f"{dtype.dtype.name} ---")
        print(f"    theoretical work = {flops/1e9:.2f} GFLOPs/call")
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
