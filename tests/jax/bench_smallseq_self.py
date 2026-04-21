"""
Benchmark: small-seq self-attention MFMA kernels vs standard CK path vs JAX unfused.

Measures Forward, Backward (only), and Fwd+Bwd separately using jax.vjp.

Reference JAX unfused-attn adapted from:
  https://github.com/ROCm/frameworks-internal/issues/16088

Usage (inside docker):
  HIP_VISIBLE_DEVICES=0 XLA_FLAGS="--xla_gpu_enable_command_buffer=" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python tests/jax/bench_smallseq_self.py
"""

import os
import time
import statistics

import jax
import jax.numpy as jnp

from transformer_engine.jax.attention import (
    fused_attn,
    AttnBiasType,
    AttnMaskType,
    QKVLayout,
)
from transformer_engine.jax.sharding import MeshResource, global_shard_guard


# ---------------------------------------------------------------------------
# JAX unfused attention (reference from ROCm/frameworks-internal#16088)
# ---------------------------------------------------------------------------
def jax_unfused_attn(query, key, value, softmax_scale):
    """
    Pure JAX unfused self-attention (no masking, no dropout).
    Input layout: BSHD [batch, seq, heads, dim]
    """
    query = jnp.einsum("b n h d -> b h n d", query)
    key = jnp.einsum("b n h d -> b h n d", key)

    scores = jnp.einsum("b h n d, b h s d -> b h n s", query, key)
    scores = (scores * softmax_scale).astype(query.dtype)

    attention_weights = jax.nn.softmax(
        jnp.asarray(scores, dtype=jnp.float32), axis=-1
    )
    attention_weights = attention_weights.astype(value.dtype)

    out = jnp.einsum("b h s S, b S h d -> b h s d", attention_weights, value)
    out = jnp.einsum("b h n d -> b n h d", out)
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_inputs(b, s, h, d, dtype=jnp.bfloat16):
    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    q = jax.random.normal(k1, (b, s, h, d), dtype=dtype)
    k = jax.random.normal(k2, (b, s, h, d), dtype=dtype)
    v = jax.random.normal(k3, (b, s, h, d), dtype=dtype)
    mask = jnp.ones((b, 1, s, s), dtype=jnp.bool_)
    dropout_rng = k4
    return q, k, v, mask, dropout_rng


# ---------------------------------------------------------------------------
# TE (fused_attn) builders — fwd, bwd-only, fwd+bwd
# ---------------------------------------------------------------------------
def _te_kwargs(scale):
    return dict(
        attn_bias_type=AttnBiasType.NO_BIAS,
        attn_mask_type=AttnMaskType.NO_MASK,
        scaling_factor=scale,
        dropout_probability=0.0,
        is_training=True,
        qkv_layout=QKVLayout.BSHD_BSHD_BSHD,
        max_segments_per_seq=1,
        window_size=None,
    )


def build_te_fwd_fn(scale):
    kwargs = _te_kwargs(scale)

    @jax.jit
    def fwd(q, k, v, mask, dropout_rng):
        return fused_attn((q, k, v), None, mask, dropout_rng, **kwargs).astype(q.dtype)

    return fwd


def build_te_vjp_fwd_fn(scale):
    """Returns a JIT'd function that runs the forward pass via vjp and returns
    (output, vjp_fn).  The vjp_fn can later be called to time bwd separately."""
    kwargs = _te_kwargs(scale)

    def attn_fn(q, k, v, mask, dropout_rng):
        out = fused_attn((q, k, v), None, mask, dropout_rng, **kwargs)
        return jnp.mean(out.astype(jnp.float32)).astype(q.dtype)

    @jax.jit
    def vjp_fwd(q, k, v, mask, dropout_rng):
        return jax.vjp(attn_fn, q, k, v, mask, dropout_rng)

    return vjp_fwd


def build_te_fwdbwd_fn(scale):
    kwargs = _te_kwargs(scale)

    def loss_fn(q, k, v, mask, dropout_rng):
        out = fused_attn((q, k, v), None, mask, dropout_rng, **kwargs)
        return jnp.mean(out.astype(jnp.float32)).astype(q.dtype)

    return jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1, 2)))


# ---------------------------------------------------------------------------
# JAX unfused builders — fwd, bwd-only, fwd+bwd
# ---------------------------------------------------------------------------
def build_jax_fwd_fn(scale):
    @jax.jit
    def fwd(q, k, v):
        return jax_unfused_attn(q, k, v, scale)

    return fwd


def build_jax_vjp_fwd_fn(scale):
    def attn_fn(q, k, v):
        out = jax_unfused_attn(q, k, v, scale)
        return jnp.mean(out.astype(jnp.float32)).astype(q.dtype)

    @jax.jit
    def vjp_fwd(q, k, v):
        return jax.vjp(attn_fn, q, k, v)

    return vjp_fwd


def build_jax_fwdbwd_fn(scale):
    def loss_fn(q, k, v):
        out = jax_unfused_attn(q, k, v, scale)
        return jnp.mean(out.astype(jnp.float32)).astype(q.dtype)

    return jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1, 2)))


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------
def _block(x):
    """Call block_until_ready on JAX arrays, skip numpy arrays."""
    if hasattr(x, 'block_until_ready'):
        x.block_until_ready()
    return x


def bench(fn, args, n_warmup=5, n_iter=20):
    for _ in range(n_warmup):
        result = fn(*args)
        jax.tree.map(_block, result)

    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        result = fn(*args)
        jax.tree.map(_block, result)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times


def bench_bwd_only(vjp_fwd_fn, fwd_args, n_warmup=5, n_iter=20):
    """Time only the backward (VJP) pass.  Runs vjp_fwd once to get vjp_fn,
    then times repeated calls to vjp_fn with a ones-like cotangent."""
    # Run vjp_fwd to get primals + vjp_fn (this includes the forward)
    primals, vjp_fn = vjp_fwd_fn(*fwd_args)
    jax.tree.map(_block, primals)
    cotangent = jnp.ones_like(primals)

    # Warmup bwd
    for _ in range(n_warmup):
        # Re-run fwd to get a fresh vjp_fn each time (residuals may be consumed)
        primals, vjp_fn = vjp_fwd_fn(*fwd_args)
        jax.tree.map(_block, primals)
        grads = vjp_fn(cotangent)
        jax.tree.map(_block, grads)

    times = []
    for _ in range(n_iter):
        # Get fresh vjp_fn (forward), wait for it
        primals, vjp_fn = vjp_fwd_fn(*fwd_args)
        jax.tree.map(_block, primals)
        # Time only the backward
        t0 = time.perf_counter()
        grads = vjp_fn(cotangent)
        jax.tree.map(_block, grads)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times


def fmt(times):
    return (f"mean={statistics.mean(times)*1000:.3f}ms  "
            f"median={statistics.median(times)*1000:.3f}ms  "
            f"min={min(times)*1000:.3f}ms")


def main():
    b, h, d = 2048, 16, 128
    seq_lens = [16, 17]
    scale = 1.0 / (d ** 0.5)

    print(f"{'='*70}")
    print(f"Benchmark: small-seq self-attn MFMA vs CK vs JAX unfused")
    print(f"b={b}, h={h}, d={d}, dtype=BF16")
    print(f"{'='*70}")

    with global_shard_guard(MeshResource()):
        for s in seq_lens:
            print(f"\n{'─'*70}")
            print(f"  s_q = s_kv = {s}")
            print(f"{'─'*70}")

            q, k, v, mask, dropout_rng = make_inputs(b, s, h, d)
            te_args = (q, k, v, mask, dropout_rng)
            jax_args = (q, k, v)

            results = {}  # {label: {fwd, bwd, fwdbwd}}

            # --- JAX unfused ---
            label = "JAX unfused"
            print(f"\n  [{label}] Forward:")
            fwd_times = bench(build_jax_fwd_fn(scale), jax_args)
            print(f"    {fmt(fwd_times)}")

            print(f"  [{label}] Backward only:")
            bwd_times = bench_bwd_only(build_jax_vjp_fwd_fn(scale), jax_args)
            print(f"    {fmt(bwd_times)}")

            print(f"  [{label}] Fwd+Bwd:")
            fwdbwd_times = bench(build_jax_fwdbwd_fn(scale), jax_args)
            print(f"    {fmt(fwdbwd_times)}")

            results[label] = {
                "fwd": statistics.median(fwd_times),
                "bwd": statistics.median(bwd_times),
                "fwdbwd": statistics.median(fwdbwd_times),
            }
            jax.clear_caches()

            # --- MFMA path ---
            os.environ["NVTE_FUSED_ATTN_CK_SMALLSEQ"] = "1"
            label = "MFMA smallseq"
            print(f"\n  [{label}] Forward:")
            fwd_times = bench(build_te_fwd_fn(scale), te_args)
            print(f"    {fmt(fwd_times)}")

            print(f"  [{label}] Backward only:")
            bwd_times = bench_bwd_only(build_te_vjp_fwd_fn(scale), te_args)
            print(f"    {fmt(bwd_times)}")

            print(f"  [{label}] Fwd+Bwd:")
            fwdbwd_times = bench(build_te_fwdbwd_fn(scale), te_args)
            print(f"    {fmt(fwdbwd_times)}")

            results[label] = {
                "fwd": statistics.median(fwd_times),
                "bwd": statistics.median(bwd_times),
                "fwdbwd": statistics.median(fwdbwd_times),
            }

            # --- Standard CK path ---
            os.environ["NVTE_FUSED_ATTN_CK_SMALLSEQ"] = "0"
            jax.clear_caches()
            label = "Standard CK"
            print(f"\n  [{label}] Forward:")
            fwd_times = bench(build_te_fwd_fn(scale), te_args)
            print(f"    {fmt(fwd_times)}")

            print(f"  [{label}] Backward only:")
            bwd_times = bench_bwd_only(build_te_vjp_fwd_fn(scale), te_args)
            print(f"    {fmt(bwd_times)}")

            print(f"  [{label}] Fwd+Bwd:")
            fwdbwd_times = bench(build_te_fwdbwd_fn(scale), te_args)
            print(f"    {fmt(fwdbwd_times)}")

            results[label] = {
                "fwd": statistics.median(fwd_times),
                "bwd": statistics.median(bwd_times),
                "fwdbwd": statistics.median(fwdbwd_times),
            }

            # --- Summary table ---
            print(f"\n  Summary (median ms):")
            print(f"    {'':20s} {'Forward':>10s}  {'Backward':>10s}  {'Fwd+Bwd':>10s}")
            for lbl in ["JAX unfused", "MFMA smallseq", "Standard CK"]:
                r = results[lbl]
                print(f"    {lbl:20s} {r['fwd']*1000:10.3f}  {r['bwd']*1000:10.3f}  {r['fwdbwd']*1000:10.3f}")

            jax_r = results["JAX unfused"]
            mfma_r = results["MFMA smallseq"]
            ck_r = results["Standard CK"]

            print(f"\n  Speedup vs JAX unfused (JAX / kernel):")
            print(f"    MFMA  fwd: {jax_r['fwd']/mfma_r['fwd']:.2f}x  "
                  f"bwd: {jax_r['bwd']/mfma_r['bwd']:.2f}x  "
                  f"fwd+bwd: {jax_r['fwdbwd']/mfma_r['fwdbwd']:.2f}x")
            print(f"    CK    fwd: {jax_r['fwd']/ck_r['fwd']:.2f}x  "
                  f"bwd: {jax_r['bwd']/ck_r['bwd']:.2f}x  "
                  f"fwd+bwd: {jax_r['fwdbwd']/ck_r['fwdbwd']:.2f}x")
            print(f"  Speedup MFMA vs CK:")
            print(f"    fwd: {ck_r['fwd']/mfma_r['fwd']:.2f}x  "
                  f"bwd: {ck_r['bwd']/mfma_r['bwd']:.2f}x  "
                  f"fwd+bwd: {ck_r['fwdbwd']/mfma_r['fwdbwd']:.2f}x")

            jax.clear_caches()
            os.environ.pop("NVTE_FUSED_ATTN_CK_SMALLSEQ", None)


if __name__ == "__main__":
    main()
