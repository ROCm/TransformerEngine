# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Rotary Position Embedding (RoPE) -- AITER CK-JIT or PyTorch-native fallback.

When AITER is available, uses its fused CK-JIT RoPE kernel (single kernel launch).
Otherwise, falls back to PyTorch-native implementation (~8 kernel launches).

Supports context parallelism (cp_size, cp_rank) for DualChunkSwap
sequence partitioning used by TE's CP implementation.
"""

import torch
from typing import Optional, Union

from .aiter_utils import get_aiter_rope


# ---------------------------------------------------------------------------
# Context parallelism helpers
# ---------------------------------------------------------------------------

def _get_freqs_on_this_cp_rank(
    freqs: torch.Tensor, seqlen: int, cp_size: int, cp_rank: int
) -> torch.Tensor:
    """Slice positional embedding frequencies for this CP rank.

    Implements the DualChunkSwap position mapping: each rank gets two
    non-contiguous segments of the full frequency table.

    Args:
        freqs: Full frequency tensor, shape ``[s_full, ...]``.
        seqlen: Local sequence length on this rank (= s_full / cp_size).
        cp_size: Context parallel world size.
        cp_rank: Context parallel rank.

    Returns:
        Frequency tensor of shape ``[seqlen, ...]`` with the two
        DualChunkSwap chunks concatenated.
    """
    if cp_size > 1:
        cp_seg = seqlen // 2
        full_seqlen = cp_size * seqlen
        return torch.cat(
            [
                freqs[cp_rank * cp_seg : (cp_rank + 1) * cp_seg],
                freqs[full_seqlen - (cp_rank + 1) * cp_seg : full_seqlen - cp_rank * cp_seg],
            ]
        )
    return freqs[:seqlen]


# ---------------------------------------------------------------------------
# QKV format enum values (mirrors NVTE_QKV_Format)
# ---------------------------------------------------------------------------

_BSHD = 0
_SBHD = 1
_THD = 2

# AITER rotate_style enum: 0 = NEOX (TE non-interleaved), 1 = GPTJ (TE interleaved)
_NEOX = 0
_GPTJ = 1


def _seqlen_from_tensor(t, qkv_format_int):
    """Return the sequence length from a tensor given its QKV format."""
    if qkv_format_int == _BSHD:
        return t.shape[1]
    return t.shape[0]


def _te_interleaved_to_aiter_style(interleaved):
    """Map TE interleaved flag to AITER rotate_style int."""
    return _GPTJ if interleaved else _NEOX


# ---------------------------------------------------------------------------
# AITER adapter helpers
# ---------------------------------------------------------------------------

def _aiter_fwd(aiter_rope, t, freqs, interleaved, qkv_format):
    """Call AITER rope_fwd with TE parameter conventions.

    AITER expects SBHD [s, b, h, d]. For BSHD input, transpose around the call.
    TE freqs are [s, 1, 1, rot_dim] (already doubled); AITER with
    reuse_freqs_front_part=False expects the same shape.
    """
    style = _te_interleaved_to_aiter_style(interleaved)
    if qkv_format == _BSHD:
        t_sbhd = t.transpose(0, 1).contiguous()
        out = aiter_rope.rope_fwd(t_sbhd, freqs, style, False, False)
        return out.transpose(0, 1).contiguous()
    return aiter_rope.rope_fwd(t, freqs, style, False, False)


def _aiter_bwd(aiter_rope, grad, freqs, interleaved, qkv_format):
    """Call AITER rope_bwd with TE parameter conventions."""
    style = _te_interleaved_to_aiter_style(interleaved)
    if qkv_format == _BSHD:
        g_sbhd = grad.transpose(0, 1).contiguous()
        out = aiter_rope.rope_bwd(g_sbhd, freqs, style, False, False)
        return out.transpose(0, 1).contiguous()
    return aiter_rope.rope_bwd(grad, freqs, style, False, False)


def _aiter_thd_fwd(aiter_rope, t, cu_seqlens, freqs, interleaved):
    """Call AITER rope_thd_fwd with TE parameter conventions."""
    style = _te_interleaved_to_aiter_style(interleaved)
    return aiter_rope.rope_thd_fwd(t, cu_seqlens, freqs, style, False, False)


def _aiter_thd_bwd(aiter_rope, grad, cu_seqlens, freqs, interleaved):
    """Call AITER rope_thd_bwd with TE parameter conventions."""
    style = _te_interleaved_to_aiter_style(interleaved)
    return aiter_rope.rope_thd_bwd(grad, cu_seqlens, freqs, style, False, False)


# ---------------------------------------------------------------------------
# Core PyTorch RoPE (fallback when AITER is unavailable)
# ---------------------------------------------------------------------------

def _rotate_half(x):
    """Rotate the last dimension: [-x2, x1] from [x1, x2]."""
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _rotate_half_interleaved(x):
    """Rotate with interleaved layout: pairs are (even, odd) indices."""
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x_new = torch.stack((-x2, x1), dim=-1)
    return x_new.view(*x.shape)


def _apply_rope_pytorch(t, freqs, interleaved=False):
    """Apply RoPE using PyTorch operations.

    Freqs are raw angle values (not pre-computed cos/sin).
    The rotation is: t * cos(freqs) + rotate_half(t) * sin(freqs).
    Computation is done in float32 for precision (matching the C++ fused kernel).

    Args:
        t: Input tensor, last dim is head_dim.
        freqs: Angle tensor, shape broadcastable to t, last dim is rot_dim.
            ``rot_dim <= head_dim``; unrotated dims are passed through.
        interleaved: If True, use interleaved rotation pattern.
    """
    orig_dtype = t.dtype
    cos_ = torch.cos(freqs)
    sin_ = torch.sin(freqs)

    rot_dim = freqs.shape[-1]
    t_rot, t_pass = t[..., :rot_dim].float(), t[..., rot_dim:]

    rotate_fn = _rotate_half_interleaved if interleaved else _rotate_half
    t_rot = t_rot * cos_ + rotate_fn(t_rot) * sin_
    return torch.cat((t_rot.to(orig_dtype), t_pass), dim=-1)


def _inverse_rope_pytorch(grad_output, freqs, interleaved=False):
    """Inverse RoPE rotation for backward pass.

    The inverse of ``t * cos + rotate_half(t) * sin`` is
    ``g * cos + rotate_half(g) * (-sin)``, i.e. negate sin.
    Computation is done in float32 for precision (matching the C++ fused kernel).
    """
    orig_dtype = grad_output.dtype
    cos_ = torch.cos(freqs)
    sin_ = torch.sin(freqs)

    rot_dim = freqs.shape[-1]
    g_rot, g_pass = grad_output[..., :rot_dim].float(), grad_output[..., rot_dim:]

    rotate_fn = _rotate_half_interleaved if interleaved else _rotate_half
    g_rot = g_rot * cos_ + rotate_fn(g_rot) * (-sin_)
    return torch.cat((g_rot.to(orig_dtype), g_pass), dim=-1)


# ---------------------------------------------------------------------------
# Public API: fused_rope_forward / backward
# ---------------------------------------------------------------------------

def fused_rope_forward(
    t: torch.Tensor,
    freqs: torch.Tensor,
    start_positions: Optional[torch.Tensor] = None,
    qkv_format: int = _SBHD,
    interleaved: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    cp_size: int = 1,
    cp_rank: int = 0,
) -> torch.Tensor:
    """Fused RoPE forward -- lite replacement for ``tex.fused_rope_forward``.

    Signature matches the C++ binding so that ``FusedRoPEFunc`` in
    ``transformer_engine.pytorch.attention.rope`` can call through
    ``tex.fused_rope_forward`` transparently.
    """
    _aiter_rope = get_aiter_rope()

    # Determine local sequence length from tensor + format
    seqlen = _seqlen_from_tensor(t, qkv_format)

    # Handle start_positions: stack per-batch offset freqs before CP slicing
    # start_positions offsets each batch element's freqs independently.
    if start_positions is not None:
        freqs = torch.cat(
            [freqs[int(p) : int(p) + seqlen * cp_size] for p in start_positions], dim=1
        )
        # freqs now has shape [seqlen*cp_size, batch, 1, d]

    # Slice frequencies for this CP rank
    if cp_size > 1:
        freqs = _get_freqs_on_this_cp_rank(freqs, seqlen, cp_size, cp_rank)
    else:
        freqs = freqs[:seqlen]

    # THD format with cu_seqlens
    if qkv_format == _THD and cu_seqlens is not None:
        if cp_size > 1:
            cu_seqlens = cu_seqlens // cp_size
        if _aiter_rope is not None and start_positions is None:
            return _aiter_thd_fwd(_aiter_rope, t, cu_seqlens, freqs, interleaved)
        # PyTorch fallback: split by sequence, apply per-sequence
        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        results = []
        for idx, x in enumerate(torch.split(t, seqlens)):
            seq_freqs = freqs[:x.size(0)]
            results.append(_apply_rope_pytorch(x.unsqueeze(1), seq_freqs, interleaved).squeeze(1))
        return torch.cat(results)

    # BSHD/SBHD path -- use AITER fused kernel when available
    if _aiter_rope is not None and start_positions is None:
        return _aiter_fwd(_aiter_rope, t, freqs, interleaved, qkv_format)

    # PyTorch fallback
    if qkv_format == _BSHD:
        freqs = freqs.transpose(0, 1) if freqs.dim() == 4 else freqs
    return _apply_rope_pytorch(t, freqs, interleaved)


def fused_rope_backward(
    grad_output: torch.Tensor,
    freqs: torch.Tensor,
    start_positions: Optional[torch.Tensor] = None,
    qkv_format: int = _SBHD,
    interleaved: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    cp_size: int = 1,
    cp_rank: int = 0,
) -> torch.Tensor:
    """Fused RoPE backward -- lite replacement for ``tex.fused_rope_backward``.

    RoPE backward is the inverse rotation (negate sin component).
    """
    _aiter_rope = get_aiter_rope()

    seqlen = _seqlen_from_tensor(grad_output, qkv_format)

    # Handle start_positions: stack per-batch offset freqs
    if start_positions is not None:
        freqs = torch.cat(
            [freqs[int(p) : int(p) + seqlen * cp_size] for p in start_positions], dim=1
        )

    if cp_size > 1:
        freqs = _get_freqs_on_this_cp_rank(freqs, seqlen, cp_size, cp_rank)
    else:
        freqs = freqs[:seqlen]

    # THD
    if qkv_format == _THD and cu_seqlens is not None:
        if cp_size > 1:
            cu_seqlens = cu_seqlens // cp_size
        if _aiter_rope is not None and start_positions is None:
            return _aiter_thd_bwd(_aiter_rope, grad_output, cu_seqlens, freqs, interleaved)
        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        results = []
        for idx, g in enumerate(torch.split(grad_output, seqlens)):
            seq_freqs = freqs[:g.size(0)]
            results.append(_inverse_rope_pytorch(g.unsqueeze(1), seq_freqs, interleaved).squeeze(1))
        return torch.cat(results)

    # BSHD/SBHD -- use AITER fused kernel when available
    if _aiter_rope is not None and start_positions is None:
        return _aiter_bwd(_aiter_rope, grad_output, freqs, interleaved, qkv_format)

    # PyTorch fallback
    if qkv_format == _BSHD:
        freqs = freqs.transpose(0, 1) if freqs.dim() == 4 else freqs
    return _inverse_rope_pytorch(grad_output, freqs, interleaved)


# ---------------------------------------------------------------------------
# Public API: fused_qkv_rope_forward / backward
# ---------------------------------------------------------------------------

def fused_qkv_rope_forward(
    qkv: torch.Tensor,
    q_freqs: torch.Tensor,
    k_freqs: torch.Tensor,
    start_positions: Optional[torch.Tensor] = None,
    qkv_split_arg_list=None,
    qkv_format: int = _SBHD,
    interleaved: bool = False,
    cp_size: int = 1,
    cp_rank: int = 0,
) -> torch.Tensor:
    """Fused QKV RoPE forward -- lite replacement for ``tex.fused_qkv_rope_forward``.

    Apply RoPE to Q and K within a packed QKV tensor.
    Returns tuple (Q_rotated, K_rotated, V_unchanged).
    """
    _aiter_rope = get_aiter_rope()

    seqlen = _seqlen_from_tensor(qkv, qkv_format)

    # Slice frequencies for CP
    if cp_size > 1:
        q_freqs = _get_freqs_on_this_cp_rank(q_freqs, seqlen, cp_size, cp_rank)
        k_freqs = _get_freqs_on_this_cp_rank(k_freqs, seqlen, cp_size, cp_rank)
    else:
        q_freqs = q_freqs[:seqlen]
        k_freqs = k_freqs[:seqlen]

    # Split QKV along the last (head_dim) dimension
    if qkv_split_arg_list is not None:
        q, k, v = torch.split(qkv, qkv_split_arg_list, dim=-1)
    else:
        q, k, v = qkv.chunk(3, dim=-1)

    # The C++ kernel reshapes Q/K so each split is expressed as (num_heads, head_dim)
    # where head_dim is derived from the K split (which is always 1 head_dim per head).
    # e.g. Q [s, b, 64, 512] with K head_dim=128 -> [s, b, 256, 128]
    # This allows partial rotation (rot_dim < head_dim) to work correctly.
    head_dim = k.shape[-1]

    if q.shape[-1] != head_dim:
        new_q_heads = q.shape[-2] * q.shape[-1] // head_dim
        q = q.reshape(*q.shape[:-2], new_q_heads, head_dim)

    # Use AITER fused kernel when available
    if _aiter_rope is not None and start_positions is None:
        q_rot = _aiter_fwd(_aiter_rope, q, q_freqs, interleaved, qkv_format)
        k_rot = _aiter_fwd(_aiter_rope, k, k_freqs, interleaved, qkv_format)
        return q_rot, k_rot, v

    # PyTorch fallback
    if qkv_format == _BSHD:
        q_freqs = q_freqs.transpose(0, 1) if q_freqs.dim() == 4 else q_freqs
        k_freqs = k_freqs.transpose(0, 1) if k_freqs.dim() == 4 else k_freqs

    q_rot = _apply_rope_pytorch(q, q_freqs, interleaved)
    k_rot = _apply_rope_pytorch(k, k_freqs, interleaved)

    return q_rot, k_rot, v


def fused_qkv_rope_backward(
    grad_output_q: torch.Tensor,
    grad_output_k: torch.Tensor,
    grad_output_v: torch.Tensor,
    q_freqs: torch.Tensor,
    k_freqs: torch.Tensor,
    qkv_split_arg_list=None,
    qkv_format: int = _SBHD,
    interleaved: bool = False,
    cp_size: int = 1,
    cp_rank: int = 0,
) -> torch.Tensor:
    """Fused QKV RoPE backward -- lite replacement for ``tex.fused_qkv_rope_backward``."""
    _aiter_rope = get_aiter_rope()

    seqlen = _seqlen_from_tensor(grad_output_q, qkv_format)

    if cp_size > 1:
        q_freqs = _get_freqs_on_this_cp_rank(q_freqs, seqlen, cp_size, cp_rank)
        k_freqs = _get_freqs_on_this_cp_rank(k_freqs, seqlen, cp_size, cp_rank)
    else:
        q_freqs = q_freqs[:seqlen]
        k_freqs = k_freqs[:seqlen]

    # Use AITER fused kernel when available
    if _aiter_rope is not None:
        gq_rot = _aiter_bwd(_aiter_rope, grad_output_q, q_freqs, interleaved, qkv_format)
        gk_rot = _aiter_bwd(_aiter_rope, grad_output_k, k_freqs, interleaved, qkv_format)
    else:
        if qkv_format == _BSHD:
            q_freqs = q_freqs.transpose(0, 1) if q_freqs.dim() == 4 else q_freqs
            k_freqs = k_freqs.transpose(0, 1) if k_freqs.dim() == 4 else k_freqs
        gq_rot = _inverse_rope_pytorch(grad_output_q, q_freqs, interleaved)
        gk_rot = _inverse_rope_pytorch(grad_output_k, k_freqs, interleaved)

    # Reshape Q/K grads back to original split dims before concatenation.
    # The forward reshaped e.g. [s, b, 64, 512] -> [s, b, 256, 128];
    # backward receives [s, b, 256, 128] and must produce [s, b, 64, 512].
    if qkv_split_arg_list is not None:
        q_split_dim = qkv_split_arg_list[0]
        v_head_dim = grad_output_v.shape[-2]  # original num_heads
        if gq_rot.shape[-1] != q_split_dim:
            gq_rot = gq_rot.reshape(*gq_rot.shape[:-2], v_head_dim, q_split_dim)

    return torch.cat([gq_rot, gk_rot, grad_output_v], dim=-1)
