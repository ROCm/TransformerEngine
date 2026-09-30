# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

import pytest
import torch

from transformer_engine.pytorch.triton_kernels.max_logit import max_logit_fwd


def _reference_max_logit(q, k, qkv_format, attn_mask_type, scale):
    """fp32 reference: full score matrix, masked keys to -inf, amax over (b, i, j)."""
    if qkv_format == "sbhd":
        q, k = q.transpose(0, 1), k.transpose(0, 1)
    q, k = q.float(), k.float()
    num_heads, num_heads_kv = q.shape[2], k.shape[2]
    k = k.repeat_interleave(num_heads // num_heads_kv, dim=2)
    scores = torch.einsum("bihd,bjhd->bhij", q, k) * scale
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    if attn_mask_type != "no_mask":
        shift = seqlen_k - seqlen_q if attn_mask_type == "causal_bottom_right" else 0
        i = torch.arange(seqlen_q, device=q.device)[:, None]
        j = torch.arange(seqlen_k, device=q.device)[None, :]
        scores = scores.masked_fill(j > i + shift, float("-inf"))
    return torch.amax(scores, dim=(0, 2, 3))


def _make_qk(batch, seqlen_q, seqlen_k, num_heads, num_heads_kv, head_dim, qkv_format, dtype):
    def make(seqlen, heads):
        shape = (batch, seqlen, heads, head_dim) if qkv_format == "bshd" else (seqlen, batch, heads, head_dim)
        return torch.randn(shape, dtype=dtype, device="cuda")

    return make(seqlen_q, num_heads), make(seqlen_k, num_heads_kv)


def _check(q, k, qkv_format, attn_mask_type, scale=None):
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    out = max_logit_fwd(q, k, qkv_format=qkv_format, attn_mask_type=attn_mask_type, softmax_scale=scale)
    ref = _reference_max_logit(q, k, qkv_format, attn_mask_type, scale)
    assert out.dtype == q.dtype and out.shape == (q.shape[2],)
    torch.testing.assert_close(out.float(), ref, atol=0.02, rtol=1e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("qkv_format", ["bshd", "sbhd"])
@pytest.mark.parametrize("attn_mask_type", ["no_mask", "causal", "causal_bottom_right"])
@pytest.mark.parametrize(
    "batch, seqlen_q, seqlen_k, num_heads, num_heads_kv, head_dim",
    [
        (2, 128, 128, 4, 4, 64),
        (2, 512, 512, 8, 2, 128),  # GQA
        (1, 1000, 1000, 4, 1, 128),  # MQA, S not a multiple of the block size
        (2, 200, 700, 4, 4, 128),  # S_q < S_k
        (2, 700, 200, 4, 2, 128),  # S_q > S_k: bottom-right rows above the diagonal are fully masked
        (1, 1, 333, 2, 2, 128),  # single query (decode-like)
        (2, 256, 256, 4, 4, 96),  # head_dim not a power of 2
        (1, 384, 384, 2, 2, 256),
    ],
)
def test_max_logit(
    batch, seqlen_q, seqlen_k, num_heads, num_heads_kv, head_dim, attn_mask_type, qkv_format, dtype
):
    torch.manual_seed(1234)
    q, k = _make_qk(batch, seqlen_q, seqlen_k, num_heads, num_heads_kv, head_dim, qkv_format, dtype)
    _check(q, k, qkv_format, attn_mask_type)


@pytest.mark.parametrize("attn_mask_type", ["no_mask", "causal", "causal_bottom_right"])
def test_max_logit_large_spread(attn_mask_type):
    """One large logit per head at a different position must be found exactly."""
    torch.manual_seed(1234)
    q, k = _make_qk(2, 1024, 1024, 4, 4, 128, "bshd", torch.bfloat16)
    q.mul_(0.1)
    k.mul_(0.1)
    for h, (b, i, j) in enumerate([(0, 1023, 0), (1, 500, 499), (0, 64, 64), (1, 1000, 3)]):
        q[b, i, h] = 1.0
        k[b, j, h] = 1.0 + h
    _check(q, k, "bshd", attn_mask_type)


def test_max_logit_masked_out_spike():
    """A large logit above the causal diagonal must be ignored."""
    torch.manual_seed(1234)
    q, k = _make_qk(1, 256, 256, 2, 2, 128, "bshd", torch.bfloat16)
    q[0, 10, 0] = 5.0
    k[0, 200, 0] = 5.0
    _check(q, k, "bshd", "causal")


def test_max_logit_non_contiguous():
    """q and k sliced out of a packed [B, S, 3, H, D] qkv tensor."""
    torch.manual_seed(1234)
    qkv = torch.randn(2, 300, 3, 8, 128, dtype=torch.bfloat16, device="cuda")
    q, k = qkv[:, :, 0], qkv[:, :, 1]
    _check(q, k, "bshd", "causal")


def test_max_logit_unsupported():
    q = torch.randn(2, 64, 2, 64, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(NotImplementedError):
        max_logit_fwd(q, q, qkv_format="thd")
    with pytest.raises(NotImplementedError):
        max_logit_fwd(q, q, attn_mask_type="padding_causal")
