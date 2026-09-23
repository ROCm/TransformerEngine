# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# License for AMD contributions = MIT. See LICENSE for more information

"""FlyDSL sparse-MLA attention forward tests.

Exercises ``sparse_mla_attn_fwd`` under ``NVTE_USE_FLYDSL=1``. Unlike the GEMM
tests there is no second backend to cross-check against, so every case compares
against a vectorized torch reference.

Separate references are needed because the kernel has *different* semantics:

- the **gather** mode consumes ``topk_indices``.
- the closed-form **banded** and **pool** modes ignore ``topk_indices`` entirely
  and derive KV positions in closed form (``kv = token - 127 + rank``, plus the
  causal pool rows). Checking them against the gather reference would fail for
  the wrong reason, so they get their own references and a deliberately-garbage
  index tensor proving the indices are unread.
- **varlen** (packed) batches run the gather mode on indices built by
  ``indexer_topk_to_sparse_mla``; they get a dense block-diagonal reference.

The reference scores over **512** dims, not 576 -- see the contract warning on
``sparse_mla_attn_fwd``. ``test_rope_dims_are_not_read`` pins that behavior so a
future 576-dim extension cannot land silently.
"""

import os

import pytest
import torch

from transformer_engine.pytorch.flydsl_kernels import FlyDSLUnsupportedError
from transformer_engine.pytorch.flydsl_kernels.attention import (
    indexer_topk_to_sparse_mla,
    sparse_mla_attn_fwd,
)


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


def pool_reference(q, kv, scale, pool):
    """Reference for the pool mode: the banded window plus causal pool rows.

    Pool row ``p`` (kv row ``tokens + p``) is visible to token ``t`` iff
    ``t >= cr * p + cr - 1``, with ``cr = tokens // pool``.
    """
    tokens = q.shape[0]
    cr = tokens // pool
    t = torch.arange(tokens, device=q.device)[:, None]
    rank = torch.arange(WINDOW, device=q.device)[None, :]
    p = torch.arange(pool, device=q.device)[None, :]
    swa_id = t - (WINDOW - 1) + rank
    kv_id = torch.cat([swa_id, (tokens + p).expand(tokens, -1)], dim=1)
    valid = torch.cat([swa_id >= 0, t >= cr * p + cr - 1], dim=1)
    rows = kv.float()[kv_id.clamp(min=0)]
    return attention_reference(q, rows, valid, scale)


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

    out, lse = sparse_mla_attn_fwd(q, kv, garbage, scale=scale, mode="banded")
    out_ref, lse_ref = window_reference(q, kv, scale)

    assert_attention_close(out, lse, out_ref, lse_ref)


@requires_sparse_mla_support
def test_pool_ignores_topk_indices():
    """Pool mode is closed-form too: SWA window plus causal pool, indices unread."""
    tokens, heads, pool = 256, 64, 4
    topk = 160  # >= WINDOW + pool, rounded up to the 32 granularity
    q, kv, _ = make_inputs(tokens, heads, tokens + pool, topk)
    scale = 1.0 / (D_QK**0.5)
    garbage = torch.full((tokens, topk), -7, device="cuda", dtype=torch.int32)

    out, lse = sparse_mla_attn_fwd(q, kv, garbage, scale=scale, mode="pool")
    out_ref, lse_ref = pool_reference(q, kv, scale, pool)

    assert_attention_close(out, lse, out_ref, lse_ref)


@requires_sparse_mla_support
@pytest.mark.parametrize("heads", [64, 128])
def test_gather_on_window_shape_reads_indices(heads):
    """Gather is the default even when the shapes fit the banded mode.

    ``num_kv == total_tokens and topk == 128`` is what any packed self-attention
    batch with 128 indexer picks looks like. Inferring the banded mode from that
    shape used to discard the indices and attend a sliding window instead.
    """
    tokens = 256
    q, kv, idx = make_inputs(tokens, heads, tokens, WINDOW)
    scale = 1.0 / (D_QK**0.5)

    out, lse = sparse_mla_attn_fwd(q, kv, idx, scale=scale)
    out_ref, lse_ref = gather_reference(q, kv, idx, scale)

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


# --- Varlen -------------------------------------------------------------------

# Segment lengths of each packed row of VARLEN_T tokens: row 0 ends in 18 tokens
# of padding (segment id 0), row 1 is fully packed.
VARLEN_T = 128
VARLEN_SEGMENTS = [[50, 60], [30, 98]]


def make_segments(row_lengths, T=VARLEN_T, device="cuda"):
    """``[B, T]`` segment ids (1-based, 0 = trailing padding) and positions."""
    ids, pos = [], []
    for lengths in row_lengths:
        row_ids = torch.zeros(T, dtype=torch.int32)
        row_pos = torch.zeros(T, dtype=torch.int32)
        start = 0
        for seg, n in enumerate(lengths, start=1):
            row_ids[start : start + n] = seg
            row_pos[start : start + n] = torch.arange(n, dtype=torch.int32)
            start += n
        ids.append(row_ids)
        pos.append(row_pos)
    return torch.stack(ids).to(device), torch.stack(pos).to(device)


def dense_allowed(attn_mask_type, seg_ids, seg_pos):
    """``[B, T, T]`` self-attention mask, written densely (not per top-k entry)."""
    T = seg_ids.shape[1]
    allowed = torch.ones(seg_ids.shape[0], T, T, dtype=torch.bool, device=seg_ids.device)
    if attn_mask_type.startswith("padding"):
        allowed &= (seg_ids[:, :, None] == seg_ids[:, None, :]) & (seg_ids[:, :, None] != 0)
    if attn_mask_type == "causal":
        allowed &= torch.ones(T, T, dtype=torch.bool, device=seg_ids.device).tril()
    elif attn_mask_type == "padding_causal":
        allowed &= seg_pos[:, None, :] <= seg_pos[:, :, None]
    return allowed


def indexer_like_topk(allowed, k, seed=0):
    """Top-k of random scores with disallowed keys at ``finfo.min``, as the indexer does.

    Short rows therefore end in picks of masked keys, which the adapter must drop.
    """
    torch.manual_seed(seed)
    scores = torch.randn(allowed.shape, device=allowed.device)
    scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
    return scores.topk(k, dim=-1).indices.to(torch.int32)


@requires_sparse_mla_support
@pytest.mark.parametrize("attn_mask_type", ["no_mask", "padding", "causal", "padding_causal"])
def test_varlen_indexer_topk(attn_mask_type):
    """Packed batch: indexer top-k -> adapter -> gather mode matches a dense reference."""
    B, T, heads, k = len(VARLEN_SEGMENTS), VARLEN_T, 64, 40
    seg_ids, seg_pos = make_segments(VARLEN_SEGMENTS)
    allowed = dense_allowed(attn_mask_type, seg_ids, seg_pos)
    topk = indexer_like_topk(allowed, k)

    idx = indexer_topk_to_sparse_mla(
        topk,
        kv_seqlen=T,
        attn_mask_type=attn_mask_type,
        segment_ids_q=seg_ids,
        segment_pos_q=seg_pos,
    )
    assert idx.shape == (B * T, 64) and idx.dtype == torch.int32

    q, kv, _ = make_inputs(B * T, heads, B * T, 32)
    scale = 1.0 / (D_QK**0.5)
    out, lse = sparse_mla_attn_fwd(q, kv, idx, scale=scale)

    # Dense reference over the flat pool: a key counts iff the indexer picked it
    # and the mask allows it; rows never see another batch row's keys.
    picked = torch.zeros(B, T, T, dtype=torch.bool, device="cuda")
    picked.scatter_(2, topk.long(), True)
    keep = torch.zeros(B * T, B * T, dtype=torch.bool, device="cuda")
    for b in range(B):
        keep[b * T : (b + 1) * T, b * T : (b + 1) * T] = picked[b] & allowed[b]

    scores = torch.einsum("thd,sd->ths", q.float()[..., :D_LATENT], kv.float()[:, :D_LATENT])
    scores = (scores * scale).masked_fill(~keep[:, None, :], float("-inf"))
    live = keep.any(dim=-1)
    out_ref = torch.einsum("ths,sd->thd", scores[live].softmax(-1), kv.float()[:, :D_LATENT])
    lse_ref = scores[live].logsumexp(-1)

    # Looser than the default: causal rows start with 1-3 keys, where the bf16
    # rounding of P no longer averages out. fast_path makes it worse, since its
    # max is floored at 0 so even a single key gets p = exp(s) != 1.
    assert_attention_close(out[live], lse[live], out_ref, lse_ref, rtol=1e-2)
    # Tokens with no allowed key (padding) produce zeros and -inf LSE.
    assert (~live).any() == attn_mask_type.startswith("padding")
    assert (out[~live] == 0).all()
    assert torch.isneginf(lse[~live]).all()


@requires_sparse_mla_support
def test_indexer_topk_adapter_cross_attention():
    """Cross-attention offsets use kv_seqlen, and out-of-row entries become -1."""
    B, T, S = 2, 4, 6
    seg_q = torch.tensor([[1, 1, 2, 0], [1, 1, 1, 1]], dtype=torch.int32, device="cuda")
    seg_kv = torch.tensor(
        [[1, 2, 2, 0, 1, 2], [1, 1, 1, 0, 0, 0]], dtype=torch.int32, device="cuda"
    )
    topk = torch.tensor(
        [
            [[0, 1], [4, 6], [2, 5], [0, 1]],  # 6 is past S, the last token is padding
            [[2, 3], [0, -1], [5, 1], [0, 2]],  # key 3/5 is padding, -1 stays -1
        ],
        dtype=torch.int32,
        device="cuda",
    )

    idx = indexer_topk_to_sparse_mla(
        topk, kv_seqlen=S, attn_mask_type="padding", segment_ids_q=seg_q, segment_ids_kv=seg_kv
    )

    expected = torch.tensor(
        [[0, -1], [4, -1], [2, 5], [-1, -1], [8, -1], [6, -1], [-1, 7], [6, 8]],
        dtype=torch.int32,
        device="cuda",
    )
    assert idx.shape == (B * T, 32)
    torch.testing.assert_close(idx[:, :2], expected, rtol=0, atol=0)
    assert (idx[:, 2:] == -1).all()


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
@pytest.mark.parametrize(
    "mode, match",
    [("banded", "topk == 128"), ("pool", "num_kv = total_tokens \\+ P"), ("window", "one of")],
)
def test_rejects_mode_shape_mismatch(mode, match):
    """A closed-form mode refuses shapes it cannot represent, instead of guessing."""
    q, kv, idx = make_inputs(64, 64, 256, 64)
    with pytest.raises(FlyDSLUnsupportedError, match=match):
        sparse_mla_attn_fwd(q, kv, idx, mode=mode)


@requires_sparse_mla_support
def test_disabled_without_env_flag():
    """Without NVTE_USE_FLYDSL the backend refuses rather than silently running."""
    os.environ["NVTE_USE_FLYDSL"] = "0"
    q = torch.empty(8, 64, D_QK, device="cuda", dtype=torch.bfloat16)
    kv = torch.empty(64, D_QK, device="cuda", dtype=torch.bfloat16)
    idx = torch.zeros(8, 32, device="cuda", dtype=torch.int32)
    with pytest.raises(FlyDSLUnsupportedError, match="NVTE_USE_FLYDSL"):
        sparse_mla_attn_fwd(q, kv, idx)


@requires_sparse_mla_support
@pytest.mark.parametrize("installed", ["0.2.4", "0.4.0"])
def test_rejects_unsupported_flydsl_version(monkeypatch, installed):
    """An out-of-range flydsl is refused before the kernel module is imported."""
    import importlib.metadata

    monkeypatch.setattr(importlib.metadata, "version", lambda name: installed)
    q, kv, idx = make_inputs(64, 64, 256, 64)
    with pytest.raises(FlyDSLUnsupportedError, match=f"flydsl {installed} is installed"):
        sparse_mla_attn_fwd(q, kv, idx)
