# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information
"""Numeric tests for the Kimi Delta Attention (KDA) Triton kernels, JAX driver."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from utils import require_triton_or_skip_test_file

require_triton_or_skip_test_file()
pytest.importorskip("triton")

from transformer_engine.common.triton.kda import kda_device_arch
from transformer_engine.jax.kda import kimi_delta_attn

# RMS error relative to the reference's RMS; bf16 inputs land at ~3-5e-3.
_ERR_RATIO = 1e-2

_FLASH = dict(
    use_qk_l2norm_in_kernel=True,
    use_gate_in_kernel=True,
    use_beta_sigmoid_in_kernel=True,
    safe_gate=True,
    lower_bound=-5.0,
)
_SOFTPLUS = dict(
    use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True
)


@pytest.fixture(autouse=True, scope="module")
def init():
    """WAR for CUDA uninitialize error"""
    _ = jnp.zeros(0)
    yield


def _softplus(x):
    return np.where(x < 20.0, np.log1p(np.exp(np.minimum(x, 20.0))), x)


def _ref_kda(x, cu_seqlens, opts):
    """Naive fp32 KDA in NumPy: per token, decay, delta-rule write, read out."""
    f = lambda a: np.asarray(a, dtype=np.float32)  # noqa: E731
    q, k, v, g, beta = (f(x[n]) for n in ("q", "k", "v", "g", "beta"))
    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[-1]
    if opts.get("use_qk_l2norm_in_kernel"):
        q = q / np.sqrt((q * q).sum(-1, keepdims=True) + 1e-6)
        k = k / np.sqrt((k * k).sum(-1, keepdims=True) + 1e-6)
    if opts.get("use_gate_in_kernel"):
        gg = g + f(x["dt_bias"]).reshape(HV, K)
        a = np.exp(f(x["A_log"])).reshape(HV, 1)
        lb = opts.get("lower_bound")
        gate = lb / (1.0 + np.exp(-a * gg)) if lb is not None else -a * _softplus(gg)
    else:
        gate = g
    if opts.get("use_beta_sigmoid_in_kernel"):
        beta = 1.0 / (1.0 + np.exp(-beta))
    v_first = opts.get("state_v_first", False)
    q = np.repeat(q, HV // H, axis=2).reshape(-1, HV, K) * (K**-0.5)
    k = np.repeat(k, HV // H, axis=2).reshape(-1, HV, K)
    v, gate, beta = v.reshape(-1, HV, V), gate.reshape(-1, HV, K), beta.reshape(-1, HV)
    if cu_seqlens is None:
        bounds = [(b * T, (b + 1) * T) for b in range(B)]
    else:
        cu = np.asarray(cu_seqlens)
        bounds = list(zip(cu[:-1].tolist(), cu[1:].tolist()))
    h0 = x.get("initial_state")
    o = np.zeros_like(v)
    finals = []
    for n, (bos, eos) in enumerate(bounds):
        if h0 is None:
            S = np.zeros((HV, K, V), np.float32)
        else:
            S = f(h0[n]).swapaxes(-1, -2).copy() if v_first else f(h0[n]).copy()
        for t in range(bos, eos):
            S = S * np.exp(gate[t])[..., None]
            u = v[t] - np.einsum("hk,hkv->hv", k[t], S)
            S = S + np.einsum("hk,hv->hkv", k[t] * beta[t][:, None], u)
            o[t] = np.einsum("hk,hkv->hv", q[t], S)
        finals.append(S.swapaxes(-1, -2) if v_first else S)
    return o.reshape(B, T, HV, V), np.stack(finals)


def _ref_kda_jnp(x, cu_seqlens, opts):
    """``_ref_kda`` in fp32 jnp with a ``lax.scan`` per sequence, for autodiff."""
    q, k, v, g, beta = (x[n] for n in ("q", "k", "v", "g", "beta"))
    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[-1]
    if opts.get("use_qk_l2norm_in_kernel"):
        q = q * jax.lax.rsqrt((q * q).sum(-1, keepdims=True) + 1e-6)
        k = k * jax.lax.rsqrt((k * k).sum(-1, keepdims=True) + 1e-6)
    if opts.get("use_gate_in_kernel"):
        gg = g + x["dt_bias"].reshape(HV, K)
        a = jnp.exp(x["A_log"]).reshape(HV, 1)
        lb = opts.get("lower_bound")
        gate = lb * jax.nn.sigmoid(a * gg) if lb is not None else -a * jax.nn.softplus(gg)
    else:
        gate = g
    if opts.get("use_beta_sigmoid_in_kernel"):
        beta = jax.nn.sigmoid(beta)
    v_first = opts.get("state_v_first", False)
    q = jnp.repeat(q, HV // H, axis=2).reshape(-1, HV, K) * (K**-0.5)
    k = jnp.repeat(k, HV // H, axis=2).reshape(-1, HV, K)
    v, gate, beta = v.reshape(-1, HV, V), gate.reshape(-1, HV, K), beta.reshape(-1, HV)
    if cu_seqlens is None:
        bounds = [(b * T, (b + 1) * T) for b in range(B)]
    else:
        cu = np.asarray(cu_seqlens)
        bounds = list(zip(cu[:-1].tolist(), cu[1:].tolist()))
    h0 = x.get("initial_state")

    def step(S, t):
        qt, kt, vt, gt, bt = t
        S = S * jnp.exp(gt)[..., None]
        u = vt - jnp.einsum("hk,hkv->hv", kt, S)
        S = S + jnp.einsum("hk,hv->hkv", kt * bt[:, None], u)
        return S, jnp.einsum("hk,hkv->hv", qt, S)

    outs, finals = [], []
    for n, (bos, eos) in enumerate(bounds):
        if h0 is None:
            S = jnp.zeros((HV, K, V), jnp.float32)
        else:
            S = h0[n].swapaxes(-1, -2) if v_first else h0[n]
        S, o = jax.lax.scan(
            step, S, (q[bos:eos], k[bos:eos], v[bos:eos], gate[bos:eos], beta[bos:eos])
        )
        outs.append(o)
        finals.append(S.swapaxes(-1, -2) if v_first else S)
    return jnp.concatenate(outs).reshape(B, T, HV, V), jnp.stack(finals)


def _err_ratio(ref, out):
    d = np.asarray(out, np.float32) - np.asarray(ref, np.float32)
    return float(
        np.sqrt((d**2).mean()) / (np.sqrt((np.asarray(ref, np.float32) ** 2).mean()) + 1e-8)
    )


def _inputs(B, T, H, HV, K, V, varlen, h0=False, v_first=False, precomputed_gate=False, seed=0):
    keys = jax.random.split(jax.random.PRNGKey(seed), 8)
    Bx = 1 if varlen else B
    x = dict(
        q=jax.random.normal(keys[0], (Bx, T, H, K), jnp.bfloat16),
        k=jax.random.normal(keys[1], (Bx, T, H, K), jnp.bfloat16),
        v=jax.random.normal(keys[2], (Bx, T, HV, V), jnp.bfloat16),
        g=jax.random.normal(keys[3], (Bx, T, HV, K), jnp.bfloat16),
        beta=jax.random.normal(keys[4], (Bx, T, HV), jnp.float32),
        A_log=jax.random.normal(keys[5], (HV,), jnp.float32),
        dt_bias=jax.random.normal(keys[6], (HV * K,), jnp.float32),
    )
    if precomputed_gate:
        x["g"] = jax.nn.log_sigmoid(x["g"].astype(jnp.float32) + 3).astype(jnp.bfloat16)
    cu = None
    if varlen:
        cuts = np.sort(np.random.default_rng(seed).integers(1, T, B - 1))
        cu = jnp.asarray(np.concatenate([[0], cuts, [T]]).astype(np.int32))
    if h0:
        shape = (B, HV, V, K) if v_first else (B, HV, K, V)
        x["initial_state"] = jax.random.normal(keys[7], shape, jnp.float32) * 0.1
    return x, cu


def _check(x, cu, opts, jit=True):
    call = lambda x, cu: kimi_delta_attn(  # noqa: E731
        **x, cu_seqlens=cu, output_final_state=True, **opts
    )
    o, s = (jax.jit(call) if jit else call)(x, cu)
    ro, rs = _ref_kda(x, cu, opts)
    assert o.dtype == x["q"].dtype and o.shape == x["v"].shape
    assert np.isfinite(np.asarray(o, np.float32)).all() and np.isfinite(np.asarray(s)).all()
    assert _err_ratio(ro, o) < _ERR_RATIO
    assert _err_ratio(rs, s) < _ERR_RATIO


@pytest.mark.triton
class TestKDA:
    """KDA forward through TE's JAX Triton bridge."""

    @pytest.mark.parametrize("varlen", [False, True])
    @pytest.mark.parametrize("chunks_per_seg", [None, 4], ids=["auto", "seg4"])
    @pytest.mark.parametrize("gluon", ["1", "0"], ids=["gluon", "triton"])
    def test_flash_kda(self, monkeypatch, varlen, chunks_per_seg, gluon):
        """FlashKDA (K = V = 128, bf16, fused Kimi gate), segmented and not."""
        if gluon == "1" and kda_device_arch(0) != "gfx950":
            pytest.skip("Gluon KDA kernels are gfx950-only.")
        monkeypatch.setenv("NVTE_KDA_USE_GLUON", gluon)
        if chunks_per_seg is not None:
            monkeypatch.setenv("NVTE_KDA_FLASH_CHUNKS_PER_SEG", str(chunks_per_seg))
        x, cu = _inputs(3, 1024, 4, 4, 128, 128, varlen, h0=True, v_first=varlen)
        _check(x, cu, dict(_FLASH, state_v_first=varlen))

    @pytest.mark.parametrize("varlen", [False, True])
    @pytest.mark.parametrize(
        "H,HV,K,V,opts",
        [
            (4, 4, 128, 128, dict(_FLASH, chunk_size=64)),
            (
                2,
                4,
                64,
                64,
                dict(
                    use_qk_l2norm_in_kernel=True,
                    use_gate_in_kernel=True,
                    use_beta_sigmoid_in_kernel=True,
                ),
            ),
            (2, 4, 128, 64, dict(_FLASH, state_v_first=True)),
            (4, 4, 128, 128, dict(use_qk_l2norm_in_kernel=True, use_beta_sigmoid_in_kernel=True)),
        ],
        ids=["safe_chunk64", "softplus_gva_k64", "safe_gva_vfirst", "precomputed_gate"],
    )
    def test_general_pipeline(self, varlen, H, HV, K, V, opts):
        """General pipeline: GVA, other head dims, softplus / precomputed gates."""
        x, cu = _inputs(
            3,
            700,
            H,
            HV,
            K,
            V,
            varlen,
            h0=True,
            v_first=opts.get("state_v_first", False),
            precomputed_gate="use_gate_in_kernel" not in opts,
        )
        _check(x, cu, opts)

    def test_eager(self):
        x, cu = _inputs(2, 500, 4, 4, 128, 128, True)
        _check(x, cu, _FLASH, jit=False)

    def test_varlen_jit_retrace_free(self):
        """New sequence lengths with the same shapes reuse the compiled executable."""
        x, cu = _inputs(4, 900, 4, 4, 128, 128, True)
        fn = jax.jit(lambda x, cu: kimi_delta_attn(**x, cu_seqlens=cu, **_FLASH)[0])
        fn(x, cu)
        cu2 = jnp.asarray(np.array([0, 100, 350, 351, 900], np.int32))
        o2 = fn(x, cu2)
        assert fn._cache_size() == 1  # pylint: disable=protected-access
        ro, _ = _ref_kda(x, cu2, _FLASH)
        assert _err_ratio(ro, o2) < _ERR_RATIO

    @pytest.mark.parametrize(
        "B,T,H,HV,K,V,varlen,state,opts",
        [
            (2, 256, 2, 2, 128, 128, False, "h0", _FLASH),
            (3, 300, 2, 2, 128, 128, True, "h0_vfirst", _FLASH),
            (3, 250, 2, 4, 64, 64, True, "h0", _SOFTPLUS),
            (1, 130, 2, 2, 256, 128, False, "none", dict(_SOFTPLUS, chunk_size=32)),
            (
                2,
                150,
                2,
                2,
                128,
                128,
                False,
                "h0",
                dict(use_qk_l2norm_in_kernel=True, use_beta_sigmoid_in_kernel=True),
            ),
        ],
        ids=["flash", "flash_varlen_vfirst", "softplus_gva_varlen", "softplus_k256", "precomputed"],
    )
    def test_backward(self, B, T, H, HV, K, V, varlen, state, opts):
        """Gradients of every input match autodiff through the fp32 recurrence."""
        opts = dict(opts, state_v_first=state == "h0_vfirst")
        x, cu = _inputs(
            B,
            T,
            H,
            HV,
            K,
            V,
            varlen,
            h0=state != "none",
            v_first=state == "h0_vfirst",
            precomputed_gate="use_gate_in_kernel" not in opts,
            seed=1,
        )
        if "use_gate_in_kernel" not in opts:
            del x["A_log"], x["dt_bias"]
        keys = jax.random.split(jax.random.PRNGKey(7))
        do = jax.random.normal(keys[0], x["v"].shape, jnp.float32)
        ds_shape = (B, HV, V, K) if opts["state_v_first"] else (B, HV, K, V)
        ds = jax.random.normal(keys[1], ds_shape, jnp.float32) * 0.1

        def loss(x, fn):
            o, s = fn(x)
            return (o.astype(jnp.float32) * do).sum() + (s.astype(jnp.float32) * ds).sum()

        kernel = lambda x: kimi_delta_attn(  # noqa: E731
            **x, cu_seqlens=cu, output_final_state=True, **opts
        )
        grads = jax.jit(jax.grad(lambda x: loss(x, kernel)))(x)
        x32 = {n: t.astype(jnp.float32) for n, t in x.items()}
        ref = jax.grad(lambda x: loss(x, lambda x: _ref_kda_jnp(x, cu, opts)))(x32)
        for n in x:
            grad = np.asarray(grads[n], np.float32)
            assert grads[n].dtype == x[n].dtype and np.isfinite(grad).all(), n
            if n == "A_log":
                # One heavily cancelling sum per head: bound by the size of its terms.
                gate_in = x32["g"] + x32["dt_bias"].reshape(HV, K)
                terms = np.abs(np.asarray(ref["g"] * gate_in)).sum((0, 1, 3))
                assert (np.abs(grad - np.asarray(ref[n])) <= 1e-3 * terms).all(), n
            else:
                assert _err_ratio(ref[n], grad) < _ERR_RATIO, n
