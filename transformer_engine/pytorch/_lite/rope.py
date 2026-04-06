# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Rotary Position Embedding (RoPE) -- AITER Triton or PyTorch-native fallback.

When AITER is available, uses its optimized Triton RoPE kernel.
Otherwise, falls back to PyTorch-native implementation.
"""

import torch

# Try to import AITER RoPE
_aiter_rope_available = False
try:
    from aiter import rope as aiter_rope
    _aiter_rope_available = True
except ImportError:
    pass


def _apply_rope_pytorch(t, freqs, transpose_output=False):
    """Apply RoPE using PyTorch operations.

    t: (..., seq_len, num_heads, head_dim)
    freqs: (seq_len, 1, head_dim) -- cos and sin interleaved or separate
    """
    # Split into pairs for rotation
    d = t.shape[-1]
    t1, t2 = t[..., :d // 2], t[..., d // 2:]

    # freqs should contain cos and sin values
    cos_freqs = freqs[..., :d // 2]
    sin_freqs = freqs[..., d // 2:]

    out1 = t1 * cos_freqs - t2 * sin_freqs
    out2 = t1 * sin_freqs + t2 * cos_freqs

    return torch.cat([out1, out2], dim=-1)


def fused_rope_forward(t, freqs, transpose_output=False):
    """Fused RoPE forward."""
    if _aiter_rope_available:
        return aiter_rope.fused_rope_forward(t, freqs, transpose_output)
    return _apply_rope_pytorch(t, freqs, transpose_output)


def fused_rope_backward(grad_output, freqs, transpose_output=False):
    """Fused RoPE backward.

    RoPE backward is the same as forward but with negated sin component.
    """
    if _aiter_rope_available:
        return aiter_rope.fused_rope_backward(grad_output, freqs, transpose_output)

    d = grad_output.shape[-1]
    g1, g2 = grad_output[..., :d // 2], grad_output[..., d // 2:]
    cos_freqs = freqs[..., :d // 2]
    sin_freqs = freqs[..., d // 2:]

    # Inverse rotation
    out1 = g1 * cos_freqs + g2 * sin_freqs
    out2 = -g1 * sin_freqs + g2 * cos_freqs

    return torch.cat([out1, out2], dim=-1)


def fused_qkv_rope_forward(qkv, freqs_q, freqs_k=None, transpose_output=False):
    """Fused QKV RoPE forward -- apply RoPE to Q and K within a packed QKV tensor."""
    if _aiter_rope_available:
        return aiter_rope.fused_qkv_rope_forward(qkv, freqs_q, freqs_k, transpose_output)

    # QKV is packed: split into Q, K, V
    # Assume last dim is 3 * head_dim or there are 3 heads
    q, k, v = qkv.chunk(3, dim=-1)
    q_rot = _apply_rope_pytorch(q, freqs_q)
    k_freqs = freqs_k if freqs_k is not None else freqs_q
    k_rot = _apply_rope_pytorch(k, k_freqs)
    return torch.cat([q_rot, k_rot, v], dim=-1)


def fused_qkv_rope_backward(grad_output, freqs_q, freqs_k=None, transpose_output=False):
    """Fused QKV RoPE backward."""
    if _aiter_rope_available:
        return aiter_rope.fused_qkv_rope_backward(grad_output, freqs_q, freqs_k, transpose_output)

    gq, gk, gv = grad_output.chunk(3, dim=-1)
    gq_rot = fused_rope_backward(gq, freqs_q)
    k_freqs = freqs_k if freqs_k is not None else freqs_q
    gk_rot = fused_rope_backward(gk, k_freqs)
    return torch.cat([gq_rot, gk_rot, gv], dim=-1)
