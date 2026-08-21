# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# License for AMD contributions = MIT. See LICENSE for more information

"""FlyDSL sparse-MLA attention forward tests.

Exercises ``sparse_mla_attn_fwd`` under ``NVTE_USE_FLYDSL=1``. Unlike the GEMM
tests there is no second backend to cross-check against, so every case compares
against a vectorized torch reference.

Two references are needed because the kernel has two *different* semantics:

- **gather** modes consume ``topk_indices``.
- the **banded** mode ignores ``topk_indices`` entirely and derives KV positions
  in closed form (``kv = token - 127 + rank``). Checking it against the gather
  reference would fail for the wrong reason, so it gets a window reference and a
  deliberately-garbage index tensor proving the indices are unread.

The reference scores over **512** dims, not 576 -- see the contract warning on
``sparse_mla_attn_fwd``. ``test_rope_dims_are_not_read`` pins that behavior so a
future 576-dim extension cannot land silently.
"""

import os

import pytest
import torch

from transformer_engine.pytorch.flydsl_kernels import FlyDSLUnsupportedError
from transformer_engine.pytorch.flydsl_kernels.attention import sparse_mla_attn_fwd


# --- Feature detection --------------------------------------------------------

major, minor = torch.cuda.get_device_capability()

# ds_read_tr16_b64, permlane16/32_swap and the K=32 bf16 MFMAs are CDNA4-only.
has_sparse_mla_support = (major, minor) == (9, 5)

requires_sparse_mla_support = pytest.mark.skipif(
    not has_sparse_mla_support,
    reason="FlyDSL sparse-MLA attention requires gfx950",
)


# --- Test parameters ----------------------------------------------------------

D_QK = 576
D_LATENT = 512
WINDOW = 128

# (tokens, heads, num_kv, topk) chosen to cover every dispatch path:
#   bh=64 + pstore (4-buffer)   -- heads=64,  non-banded
#   bh=128 + 2-buffer           -- heads=128, non-banded
#   many-tile (topk>256)        -- exercises the pf_pv/pf_qk many-tile knobs
GATHER_SHAPES = [
    pytest.param(64, 64, 256, 64, id="bh64_pstore"),
    pytest.param(64, 128, 256, 64, id="bh128_2buf"),
    pytest.param(32, 64, 1024, 512, id="many_tile"),
]


# --- Fixtures -----------------------------------------------------------------

@pytest.fixture(autouse=True)
def cleanup_env():
    """Enable FlyDSL for the test, then restore the prior environment."""
    saved = {
        key: os.environ.get(key)
        for key in ("NVTE_USE_FLYDSL", "NVTE_FLYDSL_SPARSE_MLA_FAST_PATH")
    }
    os.environ["NVTE_USE_FLYDSL"] = "1"

    yield

    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


# --- Helpers ------------------------------------------------------------------

def make_inputs(tokens, heads, num_kv, topk, seed=0, pad=0):
    """Random bf16 q/kv plus distinct int32 topk indices, optionally -1 padded."""
    torch.manual_seed(seed)
    q = torch.randn(tokens, heads, D_QK, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(num_kv, D_QK, device="cuda", dtype=torch.bfloat16)
    idx = torch.stack(
        [torch.randperm(num_kv, device="cuda")[:topk] for _ in range(tokens)]
    ).to(torch.int32)
    if pad:
        # -1 marks unused topk slots; the kernel masks them with an additive -inf.
        idx[:, -pad:] = -1
    return q, kv, idx


def attention_reference(q, kv_rows, valid, scale, attn_sink=None, qk_dims=D_LATENT):
    """Softmax attention over pre-gathered KV rows.

    Args:
        kv_rows: ``[tokens, n, D_QK]`` float32 gathered/windowed KV.
        valid: ``[tokens, n]`` bool mask.
    """
    scores = torch.einsum(
        "thd,tkd->thk", q.float()[:, :, :qk_dims], kv_rows[:, :, :qk_dims]
    ) * scale
    scores = scores.masked_fill(~valid[:, None, :], float("-inf"))

    row_max = scores.max(dim=-1, keepdim=True).values
    if attn_sink is not None:
        row_max = torch.maximum(row_max, attn_sink[None, :, None])

    weights = torch.exp(scores - row_max)
    denom = weights.sum(dim=-1, keepdim=True)
    if attn_sink is not None:
        denom = denom + torch.exp(attn_sink[None, :, None] - row_max)

    out = torch.einsum("thk,tkd->thd", weights / denom, kv_rows[:, :, :D_LATENT])
    lse = row_max.squeeze(-1) + torch.log(denom.squeeze(-1))
    return out, lse


def gather_reference(q, kv, idx, scale, attn_sink=None, qk_dims=D_LATENT):
    """Reference for the topk gather modes."""
    valid = idx >= 0
    rows = kv.float()[idx.clamp(min=0).long()]
    return attention_reference(q, rows, valid, scale, attn_sink, qk_dims)


def window_reference(q, kv, scale, qk_dims=D_LATENT):
    """Reference for the banded mode: token t attends [t-127 .. t]."""
    tokens = q.shape[0]
    rank = torch.arange(WINDOW, device=q.device)
    kv_id = torch.arange(tokens, device=q.device)[:, None] - (WINDOW - 1) + rank[None, :]
    valid = kv_id >= 0
    rows = kv.float()[kv_id.clamp(min=0)]
    return attention_reference(q, rows, valid, scale, None, qk_dims)


def assert_attention_close(out, lse, out_ref, lse_ref, *, rtol=5e-3, lse_atol=2e-3):
    """Compare against the reference, scaled by the reference's own magnitude."""
    scale = max(out_ref.abs().max().item(), 1e-6)
    rel = (out.float() - out_ref).abs().max().item() / scale
    assert rel < rtol, f"output rel err {rel:.3e} >= {rtol:.3e}"

    finite = torch.isfinite(lse_ref)
    lse_err = (lse.float()[finite] - lse_ref[finite]).abs().max().item()
    assert lse_err < lse_atol, f"lse abs err {lse_err:.3e} >= {lse_atol:.3e}"


# --- Tests --------------------------------------------------------------------

@requires_sparse_mla_support
@pytest.mark.parametrize("tokens, heads, num_kv, topk", GATHER_SHAPES)
def test_gather_vs_reference(tokens, heads, num_kv, topk):
    """Each non-banded dispatch path matches the gather reference."""
    q, kv, idx = make_inputs(tokens, heads, num_kv, topk)
    scale = 1.0 / (D_QK**0.5)

    out, lse = sparse_mla_attn_fwd(q, kv, idx, scale=scale)
    out_ref, lse_ref = gather_reference(q, kv, idx, scale)

    assert out.shape == (tokens, heads, D_LATENT)
    assert out.dtype == q.dtype
    assert lse.shape == (tokens, heads)
    assert lse.dtype == torch.float32
    assert_attention_close(out, lse, out_ref, lse_ref)


@requires_sparse_mla_support
def test_topk_padding_is_masked():
    """-1 padded topk slots contribute nothing."""
    q, kv, idx = make_inputs(64, 64, 256, 64, pad=16)
    scale = 1.0 / (D_QK**0.5)

    out, lse = sparse_mla_attn_fwd(q, kv, idx, scale=scale)
    out_ref, lse_ref = gather_reference(q, kv, idx, scale)

    assert (idx == -1).any(), "test should actually exercise padding"
    assert_attention_close(out, lse, out_ref, lse_ref)


@requires_sparse_mla_support
def test_banded_ignores_topk_indices():
    """Banded mode is closed-form: it must not read topk_indices at all."""
    tokens, heads, num_kv, topk = 256, 64, 256, WINDOW
    q, kv, _ = make_inputs(tokens, heads, num_kv, topk)
    scale = 1.0 / (D_QK**0.5)

    # Deliberately invalid indices: a gather path would fault or mask everything.
    garbage = torch.full((tokens, topk), -7, device="cuda", dtype=torch.int32)

    out, lse = sparse_mla_attn_fwd(q, kv, garbage, scale=scale)
    out_ref, lse_ref = window_reference(q, kv, scale)

    assert_attention_close(out, lse, out_ref, lse_ref)


@requires_sparse_mla_support
def test_attn_sink():
    """The per-head attention sink participates in the softmax denominator."""
    q, kv, idx = make_inputs(64, 64, 256, 64)
    scale = 1.0 / (D_QK**0.5)
    sink = torch.randn(64, device="cuda", dtype=torch.float32)

    out, lse = sparse_mla_attn_fwd(q, kv, idx, attn_sink=sink, scale=scale)
    out_ref, lse_ref = gather_reference(q, kv, idx, scale, attn_sink=sink)

    assert_attention_close(out, lse, out_ref, lse_ref)


@requires_sparse_mla_support
def test_fast_path_matches_exact_softmax():
    """Fixed-max (fast_path) and running-max softmax agree.

    ``fast_path`` bounds every tile by the first pair's row max instead of a
    running max, which lets the compiler fold away the per-tile rescale. It is
    exact under shift-invariance, so the two must agree beyond bf16 noise.
    """
    q, kv, idx = make_inputs(32, 64, 1024, 512)
    scale = 1.0 / (D_QK**0.5)

    os.environ["NVTE_FLYDSL_SPARSE_MLA_FAST_PATH"] = "1"
    fast_out, fast_lse = sparse_mla_attn_fwd(q, kv, idx, scale=scale)

    os.environ["NVTE_FLYDSL_SPARSE_MLA_FAST_PATH"] = "0"
    exact_out, exact_lse = sparse_mla_attn_fwd(q, kv, idx, scale=scale)

    out_ref, lse_ref = gather_reference(q, kv, idx, scale)
    assert_attention_close(fast_out, fast_lse, out_ref, lse_ref)
    assert_attention_close(exact_out, exact_lse, out_ref, lse_ref)

    scale_o = max(exact_out.float().abs().max().item(), 1e-6)
    rel = (fast_out.float() - exact_out.float()).abs().max().item() / scale_o
    assert rel < 5e-3, f"fast_path vs exact rel err {rel:.3e}"


@requires_sparse_mla_support
def test_rope_dims_are_not_read():
    """Pin the 512-dim contract: dims [512,576) must not affect the output.

    This is a *limitation*, not a feature -- the decoupled RoPE sub-head is
    ignored. The test exists so a 576-dim extension cannot land silently.
    """
    q, kv, idx = make_inputs(64, 64, 256, 64)
    scale = 1.0 / (D_QK**0.5)
    baseline, _ = sparse_mla_attn_fwd(q, kv, idx, scale=scale)

    # Perturb only the RoPE sub-head of both operands.
    q_perturbed = q.clone()
    kv_perturbed = kv.clone()
    q_perturbed[:, :, D_LATENT:] += 10.0
    kv_perturbed[:, D_LATENT:] += 10.0

    perturbed, _ = sparse_mla_attn_fwd(q_perturbed, kv_perturbed, idx, scale=scale)
    torch.testing.assert_close(baseline, perturbed, rtol=0, atol=0)


# --- Validation ---------------------------------------------------------------

@requires_sparse_mla_support
def test_rejects_head_count_below_block():
    """num_heads=32 would launch a zero-size grid; it must be rejected.

    Upstream asserts only ``num_heads % 32 == 0``, so this shape reached the
    launch and failed with hipErrorInvalidValue.
    """
    q, kv, idx = make_inputs(64, 32, 256, 64)
    with pytest.raises(FlyDSLUnsupportedError, match="multiple of 64"):
        sparse_mla_attn_fwd(q, kv, idx)


@requires_sparse_mla_support
@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda q, kv, i: (q.float(), kv, i), "bf16"),
        (lambda q, kv, i: (q, kv, i.to(torch.int64)), "int32"),
        (lambda q, kv, i: (q, kv, i[:, :48].contiguous()), "multiple of 32"),
        (lambda q, kv, i: (q[:, :, :512].contiguous(), kv, i), "d_qk=512"),
        (lambda q, kv, i: (q.transpose(0, 1), kv, i), "contiguous"),
    ],
    ids=["dtype", "index_dtype", "topk_granularity", "d_qk", "contiguity"],
)
def test_rejects_unsupported_requests(mutate, match):
    """Unsupported requests raise rather than falling back or launching."""
    q, kv, idx = make_inputs(64, 64, 256, 64)
    q, kv, idx = mutate(q, kv, idx)
    with pytest.raises(FlyDSLUnsupportedError, match=match):
        sparse_mla_attn_fwd(q, kv, idx)


@requires_sparse_mla_support
def test_disabled_without_env_flag():
    """Without NVTE_USE_FLYDSL the backend refuses rather than silently running."""
    os.environ["NVTE_USE_FLYDSL"] = "0"
    q = torch.empty(8, 64, D_QK, device="cuda", dtype=torch.bfloat16)
    kv = torch.empty(64, D_QK, device="cuda", dtype=torch.bfloat16)
    idx = torch.zeros(8, 32, device="cuda", dtype=torch.int32)
    with pytest.raises(FlyDSLUnsupportedError, match="NVTE_USE_FLYDSL"):
        sparse_mla_attn_fwd(q, kv, idx)
