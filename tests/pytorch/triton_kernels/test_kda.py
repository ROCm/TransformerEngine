# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information
"""Numeric tests for the Kimi Delta Attention (KDA) Triton kernels, PyTorch driver."""

import pytest
import torch
import torch.nn.functional as F

from transformer_engine.common.triton.kda import kda_device_arch
from transformer_engine.pytorch.triton.kda import kimi_delta_attn

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="KDA needs a GPU.")

# RMS error of the output / final state relative to their RMS, against an fp32
# per-token recurrence. bf16 inputs land at ~3-5e-3 on every path.
_ERR_RATIO = 1e-2

_FLASH = dict(
    use_qk_l2norm_in_kernel=True,
    use_gate_in_kernel=True,
    use_beta_sigmoid_in_kernel=True,
    safe_gate=True,
    lower_bound=-5.0,
)


def _ref_kda(q, k, v, g, beta, A_log, dt_bias, scale, h0, cu_seqlens, opts):
    """Naive fp32 KDA: per token, decay the state, apply the delta-rule write, read out."""
    q, k, v, g, beta = (t.float() for t in (q, k, v, g, beta))
    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[-1]
    if opts.get("use_qk_l2norm_in_kernel"):
        q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    if opts.get("use_gate_in_kernel"):
        gg = g + (dt_bias.float().view(HV, K) if dt_bias is not None else 0.0)
        a = torch.exp(A_log.float()).view(HV, 1)
        lb = opts.get("lower_bound")
        gate = lb * torch.sigmoid(a * gg) if lb is not None else -a * F.softplus(gg, threshold=20)
    else:
        gate = g
    if opts.get("use_beta_sigmoid_in_kernel"):
        beta = torch.sigmoid(beta)
    v_first = opts.get("state_v_first", False)
    q = q.repeat_interleave(HV // H, dim=2).reshape(-1, HV, K)
    k = k.repeat_interleave(HV // H, dim=2).reshape(-1, HV, K)
    v, gate, beta = v.reshape(-1, HV, V), gate.reshape(-1, HV, K), beta.reshape(-1, HV)
    if cu_seqlens is None:
        bounds = [(b * T, (b + 1) * T) for b in range(B)]
    else:
        bounds = list(zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()))
    o = torch.zeros_like(v)
    finals = []
    for n, (bos, eos) in enumerate(bounds):
        if h0 is None:
            S = torch.zeros(HV, K, V, device=q.device)
        else:
            S = (h0[n].float().transpose(-1, -2) if v_first else h0[n].float()).clone()
        for t in range(bos, eos):
            S = S * torch.exp(gate[t])[..., None]
            u = v[t] - torch.einsum("hk,hkv->hv", k[t], S)
            S = S + torch.einsum("hk,hv->hkv", k[t] * beta[t][:, None], u)
            o[t] = torch.einsum("hk,hkv->hv", q[t] * scale, S)
        finals.append(S.transpose(-1, -2) if v_first else S)
    return o.view(B, T, HV, V), torch.stack(finals)


def _err_ratio(ref, x):
    d = (x.float() - ref.float()).flatten()
    return (d.square().mean().sqrt() / (ref.float().flatten().square().mean().sqrt() + 1e-8)).item()


def _inputs(B, T, H, HV, K, V, varlen, h0=False, v_first=False, precomputed_gate=False, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def r(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, device="cuda", dtype=dtype, generator=gen)

    Bx = 1 if varlen else B
    x = dict(
        q=r(Bx, T, H, K),
        k=r(Bx, T, H, K),
        v=r(Bx, T, HV, V),
        g=r(Bx, T, HV, K),
        beta=r(Bx, T, HV, dtype=torch.float32),
        A_log=r(HV, dtype=torch.float32),
        dt_bias=r(HV * K, dtype=torch.float32),
    )
    if precomputed_gate:
        # A precomputed gate is a log-space decay, so it must be negative.
        x["g"] = F.logsigmoid(x["g"].float() + 3).to(torch.bfloat16)
    cu = None
    if varlen:
        cuts = torch.randint(1, T, (B - 1,), generator=torch.Generator().manual_seed(seed))
        cu = torch.tensor([0] + sorted(cuts.tolist()) + [T], dtype=torch.int32, device="cuda")
    if h0:
        shape = (B, HV, V, K) if v_first else (B, HV, K, V)
        x["initial_state"] = r(*shape, dtype=torch.float32) * 0.1
    return x, cu


def _check(x, cu, opts, **extra):
    o, s = kimi_delta_attn(**x, cu_seqlens=cu, output_final_state=True, **opts, **extra)
    K = x["q"].shape[-1]
    ro, rs = _ref_kda(
        x["q"],
        x["k"],
        x["v"],
        x["g"],
        x["beta"],
        x.get("A_log"),
        x.get("dt_bias"),
        K**-0.5,
        x.get("initial_state"),
        cu,
        opts,
    )
    assert o.dtype == x["q"].dtype and o.shape == x["v"].shape
    assert torch.isfinite(o).all() and torch.isfinite(s).all()
    assert _err_ratio(ro, o) < _ERR_RATIO
    assert _err_ratio(rs, s) < _ERR_RATIO
    return o, s


@pytest.mark.parametrize("varlen", [False, True])
@pytest.mark.parametrize("chunks_per_seg", [None, 4], ids=["auto", "seg4"])
@pytest.mark.parametrize("gluon", ["1", "0"], ids=["gluon", "triton"])
@pytest.mark.parametrize("state", ["none", "h0", "h0_vfirst"])
def test_flash_kda(monkeypatch, varlen, chunks_per_seg, gluon, state):
    """FlashKDA (K = V = 128, bf16, fused Kimi gate), segmented and not."""
    if gluon == "1" and kda_device_arch(0) != "gfx950":
        pytest.skip("Gluon KDA kernels are gfx950-only.")
    monkeypatch.setenv("NVTE_KDA_USE_GLUON", gluon)
    if chunks_per_seg is not None:
        monkeypatch.setenv("NVTE_KDA_FLASH_CHUNKS_PER_SEG", str(chunks_per_seg))
    x, cu = _inputs(
        3, 1024, 4, 4, 128, 128, varlen, h0=state != "none", v_first=state == "h0_vfirst"
    )
    _check(x, cu, dict(_FLASH, state_v_first=state == "h0_vfirst"))


@pytest.mark.parametrize("lower_bound", [-5.0, -1.0, -0.1])
def test_flash_kda_lower_bound(lower_bound):
    """A weaker gate stops damping L, which is where the inverse's sign conventions show."""
    x, cu = _inputs(2, 512, 4, 4, 128, 128, False)
    _check(x, cu, dict(_FLASH, lower_bound=lower_bound))


_GENERAL_CASES = {
    # name: (H, HV, K, V, opts)
    "safe_chunk64": (4, 4, 128, 128, dict(_FLASH, chunk_size=64)),
    "softplus_gva_k64": (
        2,
        4,
        64,
        64,
        dict(
            use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True
        ),
    ),
    "safe_gva_v64": (2, 4, 128, 64, dict(_FLASH)),
    "softplus_chunk32_k256": (
        2,
        2,
        256,
        128,
        dict(
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            chunk_size=32,
        ),
    ),
    "precomputed_gate": (
        4,
        4,
        128,
        128,
        dict(use_qk_l2norm_in_kernel=True, use_beta_sigmoid_in_kernel=True),
    ),
}


@pytest.mark.parametrize("varlen", [False, True])
@pytest.mark.parametrize("state", ["none", "h0", "h0_vfirst"])
@pytest.mark.parametrize("case", list(_GENERAL_CASES))
def test_general_pipeline(varlen, state, case):
    """The general pipeline: GVA, other head dims, softplus / precomputed gates, chunk 32/64."""
    H, HV, K, V, opts = _GENERAL_CASES[case]
    x, cu = _inputs(
        3,
        700,
        H,
        HV,
        K,
        V,
        varlen,
        h0=state != "none",
        v_first=state == "h0_vfirst",
        precomputed_gate="use_gate_in_kernel" not in opts,
    )
    _check(x, cu, dict(opts, state_v_first=state == "h0_vfirst"))


def test_flash_matches_general(monkeypatch):
    """FlashKDA and the general pipeline agree on a call both can serve."""
    x, cu = _inputs(2, 700, 4, 4, 128, 128, True, h0=True)
    o_flash, s_flash = kimi_delta_attn(**x, cu_seqlens=cu, output_final_state=True, **_FLASH)
    monkeypatch.setenv("NVTE_KDA_FLASH", "0")
    o_gen, s_gen = kimi_delta_attn(**x, cu_seqlens=cu, output_final_state=True, **_FLASH)
    assert _err_ratio(o_gen, o_flash) < _ERR_RATIO
    assert _err_ratio(s_gen, s_flash) < _ERR_RATIO


@pytest.mark.parametrize("flash", ["1", "0"])
def test_varlen_cuda_graph(monkeypatch, flash):
    """Varlen planning stays on the device, so the whole op can be graph-captured."""
    monkeypatch.setenv("NVTE_KDA_FLASH", flash)
    x, cu = _inputs(4, 900, 4, 4, 128, 128, True, h0=True)
    kw = dict(_FLASH, cu_seqlens=cu, output_final_state=True)
    o_ref, s_ref = kimi_delta_attn(**x, **kw)  # also compiles every kernel
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        kimi_delta_attn(**x, **kw)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        o, s = kimi_delta_attn(**x, **kw)
    # New lengths, same shapes: the graph re-plans from the new cu_seqlens.
    cu.copy_(torch.tensor([0, 100, 350, 351, 900], dtype=cu.dtype, device=cu.device))
    graph.replay()
    o_new, s_new = kimi_delta_attn(**x, **kw)
    assert torch.equal(o, o_new) and torch.equal(s, s_new)
    assert not torch.equal(o_new, o_ref)


def test_max_seqlen_bound():
    """``max_seqlen`` only tightens scheduling; results stay within tolerance."""
    x, cu = _inputs(5, 3000, 4, 4, 128, 128, True)
    o1, s1 = kimi_delta_attn(**x, cu_seqlens=cu, output_final_state=True, **_FLASH)
    max_len = int((cu[1:] - cu[:-1]).max())
    o2, s2 = kimi_delta_attn(
        **x, cu_seqlens=cu, max_seqlen=max_len, output_final_state=True, **_FLASH
    )
    assert _err_ratio(o1, o2) < _ERR_RATIO and _err_ratio(s1, s2) < _ERR_RATIO


def test_backward_not_implemented():
    x, cu = _inputs(1, 128, 2, 2, 128, 128, False)
    x["q"].requires_grad_(True)
    o, _ = kimi_delta_attn(**x, **_FLASH)
    with pytest.raises(NotImplementedError):
        o.float().sum().backward()


def test_aiter_parity():
    """Bitwise identical to AITER's chunk_kimi_delta_attn when the schedules match."""
    aiter_kda = pytest.importorskip("aiter.ops.triton.kimi_delta_attn")
    for varlen, opts in ((False, _FLASH), (True, dict(_FLASH, chunk_size=64))):
        x, cu = _inputs(3, 800, 4, 4, 128, 128, varlen, h0=True)
        max_len = int((cu[1:] - cu[:-1]).max()) if varlen else None
        o_ref, s_ref = aiter_kda.chunk_kimi_delta_attn(
            **x, cu_seqlens=cu, output_final_state=True, **opts
        )
        o, s = kimi_delta_attn(
            **x, cu_seqlens=cu, max_seqlen=max_len, output_final_state=True, **opts
        )
        assert torch.equal(o, o_ref) and torch.equal(s, s_ref)
