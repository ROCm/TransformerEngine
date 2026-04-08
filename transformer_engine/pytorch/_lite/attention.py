# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Attention operations -- multi-backend: AITER, flash-attn (stub), PyTorch SDPA.

Backend priority: AITER CK kernels > flash-attn (stubbed) > PyTorch SDPA fallback.
"""

import math
import os
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from .aiter_utils import is_aiter_available, get_aiter
from .enums import (
    NVTE_Fused_Attn_Backend, NVTE_Mask_Type, NVTE_Bias_Type,
    NVTE_QKV_Layout, NVTE_QKV_Format,
)

# ---------------------------------------------------------------------------
# AITER raw kernel imports (lazy)
# ---------------------------------------------------------------------------
_aiter_fwd = None
_aiter_bwd = None
_aiter_varlen_fwd = None
_aiter_varlen_bwd = None
_aiter_import_attempted = False


def _try_load_aiter_attn():
    """Lazy-import AITER raw MHA kernels. Called once, result cached."""
    global _aiter_fwd, _aiter_bwd, _aiter_varlen_fwd, _aiter_varlen_bwd
    global _aiter_import_attempted
    if _aiter_import_attempted:
        return
    _aiter_import_attempted = True
    if not is_aiter_available():
        return
    try:
        from aiter.ops.mha import (
            _flash_attn_forward,
            _flash_attn_backward,
            _flash_attn_varlen_forward,
            _flash_attn_varlen_backward,
        )
        _aiter_fwd = _flash_attn_forward
        _aiter_bwd = _flash_attn_backward
        _aiter_varlen_fwd = _flash_attn_varlen_forward
        _aiter_varlen_bwd = _flash_attn_varlen_backward
    except (ImportError, AttributeError):
        pass


# ---------------------------------------------------------------------------
# Flash-attention (stubbed -- placeholder for future integration)
# ---------------------------------------------------------------------------
_flash_attn_available = False
# Uncomment when ready to integrate:
# try:
#     from flash_attn.flash_attn_interface import (
#         _flash_attn_forward as _fa_fwd,
#         _flash_attn_backward as _fa_bwd,
#         _flash_attn_varlen_forward as _fa_varlen_fwd,
#         _flash_attn_varlen_backward as _fa_varlen_bwd,
#     )
#     _flash_attn_available = True
# except ImportError:
#     pass


# ---------------------------------------------------------------------------
# QKV layout helpers
# ---------------------------------------------------------------------------

# Map NVTE_QKV_Layout enum values -> (q_format, kv_format)
_LAYOUT_TO_FMT = {
    NVTE_QKV_Layout.NVTE_SB3HD: ("sbhd", "sbhd"),
    NVTE_QKV_Layout.NVTE_SBH3D: ("sbhd", "sbhd"),
    NVTE_QKV_Layout.NVTE_SBHD_SB2HD: ("sbhd", "sbhd"),
    NVTE_QKV_Layout.NVTE_SBHD_SBH2D: ("sbhd", "sbhd"),
    NVTE_QKV_Layout.NVTE_SBHD_SBHD_SBHD: ("sbhd", "sbhd"),
    NVTE_QKV_Layout.NVTE_BS3HD: ("bshd", "bshd"),
    NVTE_QKV_Layout.NVTE_BSH3D: ("bshd", "bshd"),
    NVTE_QKV_Layout.NVTE_BSHD_BS2HD: ("bshd", "bshd"),
    NVTE_QKV_Layout.NVTE_BSHD_BSH2D: ("bshd", "bshd"),
    NVTE_QKV_Layout.NVTE_BSHD_BSHD_BSHD: ("bshd", "bshd"),
    NVTE_QKV_Layout.NVTE_T3HD: ("thd", "thd"),
    NVTE_QKV_Layout.NVTE_TH3D: ("thd", "thd"),
    NVTE_QKV_Layout.NVTE_THD_T2HD: ("thd", "thd"),
    NVTE_QKV_Layout.NVTE_THD_TH2D: ("thd", "thd"),
    NVTE_QKV_Layout.NVTE_THD_THD_THD: ("thd", "thd"),
    NVTE_QKV_Layout.NVTE_SBHD_BSHD_BSHD: ("sbhd", "bshd"),
    NVTE_QKV_Layout.NVTE_BSHD_SBHD_SBHD: ("bshd", "sbhd"),
    NVTE_QKV_Layout.NVTE_THD_BSHD_BSHD: ("thd", "bshd"),
    NVTE_QKV_Layout.NVTE_THD_SBHD_SBHD: ("thd", "sbhd"),
}

# Mask types that imply causal attention
_CAUSAL_MASKS = {
    NVTE_Mask_Type.NVTE_CAUSAL_MASK,
    NVTE_Mask_Type.NVTE_PADDING_CAUSAL_MASK,
    NVTE_Mask_Type.NVTE_CAUSAL_BOTTOM_RIGHT_MASK,
    NVTE_Mask_Type.NVTE_PADDING_CAUSAL_BOTTOM_RIGHT_MASK,
}

# Bias types that don't pass a bias tensor to the kernel
_NO_BIAS_TENSOR = {NVTE_Bias_Type.NVTE_NO_BIAS, NVTE_Bias_Type.NVTE_ALIBI}


def _get_qkv_format(qkv_layout) -> Tuple[str, str]:
    """Extract per-tensor format from a TE qkv_layout (enum int or string).

    Returns (q_format, kv_format) where each is one of 'bshd', 'sbhd', 'thd'.
    """
    if isinstance(qkv_layout, int):
        return _LAYOUT_TO_FMT[qkv_layout]
    # String fallback (used by direct tests)
    canon = qkv_layout.replace("3", "").replace("2", "")
    parts = canon.split("_")
    # Filter out "paged", "kv" prefixes
    parts = [p for p in parts if p not in ("paged", "kv")]
    q_fmt = parts[0]
    kv_fmt = parts[-1] if len(parts) > 1 else q_fmt
    return q_fmt, kv_fmt


def _to_bshd(t: torch.Tensor, fmt: str) -> torch.Tensor:
    """Convert tensor from *fmt* to BSHD layout. Returns a contiguous tensor."""
    if fmt == "bshd":
        return t
    if fmt == "sbhd":
        return t.transpose(0, 1).contiguous()
    raise ValueError(f"_to_bshd does not handle format '{fmt}' (use varlen path for thd)")


def _from_bshd(t: torch.Tensor, fmt: str) -> torch.Tensor:
    """Convert tensor from BSHD back to *fmt*."""
    if fmt == "bshd":
        return t
    if fmt == "sbhd":
        return t.transpose(0, 1).contiguous()
    raise ValueError(f"_from_bshd does not handle format '{fmt}'")


def _is_causal(attn_mask_type) -> bool:
    """Check if mask type implies causal attention. Accepts enum int or string."""
    if isinstance(attn_mask_type, int):
        return attn_mask_type in _CAUSAL_MASKS
    return "causal" in attn_mask_type


def _has_bias_tensor(bias_type) -> bool:
    """Check if bias type carries an actual bias tensor."""
    if isinstance(bias_type, int):
        return bias_type not in _NO_BIAS_TENSOR
    return bias_type not in ("no_bias", "alibi")


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

def get_fused_attn_backend(
    is_training,
    q_type,
    kv_type,
    qkv_layout,
    bias_type,
    mask_type,
    softmax_type,
    dropout,
    num_heads,
    num_gqa_groups,
    max_seqlen_q,
    max_seqlen_kv,
    head_dim_qk,
    head_dim_v,
    window_size_left=-1,
    window_size_right=-1,
    return_max_logit=False,
    cuda_graph=False,
):
    """Select the best available attention backend for lite mode.

    Priority: AITER CK > (flash-attn, stubbed) > PyTorch SDPA.
    """
    _try_load_aiter_attn()

    # AITER available -- covers causal, padding, sliding window, GQA, bias
    if _aiter_varlen_fwd is not None:
        return NVTE_Fused_Attn_Backend.NVTE_CK

    # Flash-attention (currently stubbed)
    if _flash_attn_available:
        return NVTE_Fused_Attn_Backend.NVTE_Flash

    # PyTorch SDPA fallback -- always available (PyTorch >= 2.0)
    return NVTE_Fused_Attn_Backend.NVTE_SDPA


# ---------------------------------------------------------------------------
# AITER forward / backward
# ---------------------------------------------------------------------------

def _aiter_attn_fwd(
    q, k, v,
    cu_seqlens_q, cu_seqlens_kv,
    max_seqlen_q, max_seqlen_kv,
    attn_scale, dropout, is_training, causal,
    window_size, attn_bias, qkv_layout,
    cu_seqlens_q_padded=None, cu_seqlens_kv_padded=None,
):
    """AITER CK attention forward via raw _flash_attn_*_forward."""
    q_fmt, kv_fmt = _get_qkv_format(qkv_layout)
    wl, wr = window_size

    if q_fmt == "thd":
        # Q/K/V already in (total, heads, dim) -- use varlen API
        out, softmax_lse, _, rng_state = _aiter_varlen_fwd(
            q, k, v,
            cu_seqlens_q, cu_seqlens_kv,
            cu_seqlens_q_padded, cu_seqlens_kv_padded,
            max_seqlen_q, max_seqlen_kv,
            0,  # min_seqlen_q
            dropout if is_training else 0.0,
            attn_scale,
            causal=causal,
            window_size_left=wl,
            window_size_right=wr,
            bias=attn_bias,
            return_lse=True,
        )
    else:
        # bshd or sbhd -- convert to bshd, use non-varlen API
        q_bshd = _to_bshd(q, q_fmt)
        k_bshd = _to_bshd(k, kv_fmt)
        v_bshd = _to_bshd(v, kv_fmt)
        out, softmax_lse, _, rng_state = _aiter_fwd(
            q_bshd, k_bshd, v_bshd,
            dropout if is_training else 0.0,
            attn_scale,
            causal,
            wl, wr,
            attn_bias,     # bias
            None,          # alibi_slopes
            True,          # return_lse
            False,         # return_softmax
            1,             # how_v3_bf16_cvt
            cu_seqlens_q,  # cu_seqlens_q (optional for padding support)
            cu_seqlens_kv, # cu_seqlens_kv
        )
        out = _from_bshd(out, q_fmt)

    aux_ctx_tensors = [softmax_lse, rng_state]
    return out, aux_ctx_tensors


def _aiter_attn_bwd(
    d_o, q, k, v, o, softmax_lse, rng_state,
    cu_seqlens_q, cu_seqlens_kv,
    max_seqlen_q, max_seqlen_kv,
    attn_scale, dropout, causal,
    window_size, qkv_layout, deterministic,
    cu_seqlens_q_padded=None, cu_seqlens_kv_padded=None,
):
    """AITER CK attention backward via raw _flash_attn_*_backward."""
    q_fmt, kv_fmt = _get_qkv_format(qkv_layout)
    wl, wr = window_size

    if q_fmt == "thd":
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        _aiter_varlen_bwd(
            d_o, q, k, v, o, softmax_lse,
            dq, dk, dv,
            cu_seqlens_q, cu_seqlens_kv,
            max_seqlen_q, max_seqlen_kv,
            dropout, attn_scale, causal,
            wl, wr,
            None,           # alibi_slopes
            deterministic,
            rng_state=rng_state,
            cu_seqlens_q_padded=cu_seqlens_q_padded,
            cu_seqlens_k_padded=cu_seqlens_kv_padded,
        )
    else:
        q_bshd = _to_bshd(q, q_fmt)
        k_bshd = _to_bshd(k, kv_fmt)
        v_bshd = _to_bshd(v, kv_fmt)
        o_bshd = _to_bshd(o, q_fmt)
        d_o_bshd = _to_bshd(d_o, q_fmt)
        dq = torch.empty_like(q_bshd)
        dk = torch.empty_like(k_bshd)
        dv = torch.empty_like(v_bshd)
        _aiter_bwd(
            d_o_bshd, q_bshd, k_bshd, v_bshd, o_bshd, softmax_lse,
            dq, dk, dv,
            None,           # dbias
            dropout, attn_scale, causal,
            wl, wr,
            None,           # bias (not needed for grad computation)
            None,           # alibi_slopes
            deterministic,
            rng_state,
            True,           # is_v3_atomic_fp32
            1,              # how_v3_bf16_cvt
        )
        dq = _from_bshd(dq, q_fmt)
        dk = _from_bshd(dk, kv_fmt)
        dv = _from_bshd(dv, kv_fmt)

    return dq, dk, dv


# ---------------------------------------------------------------------------
# PyTorch SDPA forward / backward
# ---------------------------------------------------------------------------

def _sdpa_attn_fwd(
    q, k, v,
    cu_seqlens_q, cu_seqlens_kv,
    max_seqlen_q, max_seqlen_kv,
    attn_scale, dropout, is_training, causal,
    window_size, attn_bias, qkv_layout,
):
    """PyTorch SDPA attention forward.

    SDPA expects (batch, heads, seq, dim). We convert from TE formats.
    For thd (variable-length packed), we unpack to bshd first.
    """
    q_fmt, kv_fmt = _get_qkv_format(qkv_layout)

    if q_fmt == "thd":
        # Unpack thd -> bshd for SDPA
        q_bshd = convert_thd_to_bshd(q, cu_seqlens_q, None, max_seqlen_q)
        k_bshd = convert_thd_to_bshd(k, cu_seqlens_kv, None, max_seqlen_kv)
        v_bshd = convert_thd_to_bshd(v, cu_seqlens_kv, None, max_seqlen_kv)
    else:
        q_bshd = _to_bshd(q, q_fmt)
        k_bshd = _to_bshd(k, kv_fmt)
        v_bshd = _to_bshd(v, kv_fmt)

    # SDPA expects (B, H, S, D)
    q_sdpa = q_bshd.transpose(1, 2)
    k_sdpa = k_bshd.transpose(1, 2)
    v_sdpa = v_bshd.transpose(1, 2)

    # GQA: expand K/V heads to match Q heads
    num_heads_q = q_sdpa.shape[1]
    num_heads_kv = k_sdpa.shape[1]
    if num_heads_kv < num_heads_q:
        repeat = num_heads_q // num_heads_kv
        k_sdpa = k_sdpa.repeat_interleave(repeat, dim=1)
        v_sdpa = v_sdpa.repeat_interleave(repeat, dim=1)

    # Build attention mask from attn_bias if provided
    sdpa_attn_mask = None
    if attn_bias is not None:
        sdpa_attn_mask = attn_bias

    # SDPA handles dropout and causal natively
    with torch.nn.attention.sdpa_kernel(
        [torch.nn.attention.SDPBackend.FLASH_ATTENTION,
         torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
         torch.nn.attention.SDPBackend.MATH]
    ):
        out_sdpa = F.scaled_dot_product_attention(
            q_sdpa, k_sdpa, v_sdpa,
            attn_mask=sdpa_attn_mask,
            dropout_p=dropout if is_training else 0.0,
            is_causal=causal and sdpa_attn_mask is None,
            scale=attn_scale,
        )

    # Convert back: (B, H, S, D) -> bshd -> original format
    out_bshd = out_sdpa.transpose(1, 2).contiguous()

    if q_fmt == "thd":
        batch_size = cu_seqlens_q.shape[0] - 1
        out = convert_bshd_to_thd(out_bshd, cu_seqlens_q, q.shape[0])
    else:
        out = _from_bshd(out_bshd, q_fmt)

    # SDPA doesn't expose softmax stats, but backends.py always accesses
    # aux_ctx_tensors[0] (contiguity check) and saves them for backward.
    # Provide dummy tensors that pass through safely.
    batch_size = cu_seqlens_q.shape[0] - 1
    num_heads = q_bshd.shape[2]
    dummy_lse = torch.zeros(
        batch_size, num_heads, max_seqlen_q,
        dtype=torch.float32, device=q_bshd.device,
    )
    dummy_rng = torch.zeros(2, dtype=torch.int64, device=q_bshd.device)
    aux_ctx_tensors = [dummy_lse, dummy_rng]
    return out, aux_ctx_tensors


def _sdpa_attn_bwd(
    d_o, q, k, v, o,
    cu_seqlens_q, cu_seqlens_kv,
    max_seqlen_q, max_seqlen_kv,
    attn_scale, dropout, causal,
    window_size, attn_bias, qkv_layout,
):
    """PyTorch SDPA backward via autograd recomputation."""
    q_fmt, kv_fmt = _get_qkv_format(qkv_layout)

    if q_fmt == "thd":
        q_in = convert_thd_to_bshd(q, cu_seqlens_q, None, max_seqlen_q)
        k_in = convert_thd_to_bshd(k, cu_seqlens_kv, None, max_seqlen_kv)
        v_in = convert_thd_to_bshd(v, cu_seqlens_kv, None, max_seqlen_kv)
        d_o_in = convert_thd_to_bshd(d_o, cu_seqlens_q, None, max_seqlen_q)
    else:
        q_in = _to_bshd(q, q_fmt)
        k_in = _to_bshd(k, kv_fmt)
        v_in = _to_bshd(v, kv_fmt)
        d_o_in = _to_bshd(d_o, q_fmt)

    # Re-run forward with autograd to compute gradients
    q_g = q_in.detach().requires_grad_(True)
    k_g = k_in.detach().requires_grad_(True)
    v_g = v_in.detach().requires_grad_(True)

    # (B, S, H, D) -> (B, H, S, D)
    q_sdpa = q_g.transpose(1, 2)
    k_sdpa = k_g.transpose(1, 2)
    v_sdpa = v_g.transpose(1, 2)

    num_heads_q = q_sdpa.shape[1]
    num_heads_kv = k_sdpa.shape[1]
    if num_heads_kv < num_heads_q:
        repeat = num_heads_q // num_heads_kv
        k_sdpa = k_sdpa.repeat_interleave(repeat, dim=1)
        v_sdpa = v_sdpa.repeat_interleave(repeat, dim=1)

    sdpa_attn_mask = attn_bias if attn_bias is not None else None

    out_sdpa = F.scaled_dot_product_attention(
        q_sdpa, k_sdpa, v_sdpa,
        attn_mask=sdpa_attn_mask,
        dropout_p=0.0,  # no dropout in backward recompute for determinism
        is_causal=causal and sdpa_attn_mask is None,
        scale=attn_scale,
    )

    out_bshd = out_sdpa.transpose(1, 2).contiguous()
    d_o_bshd = d_o_in

    out_bshd.backward(d_o_bshd)

    dq_bshd = q_g.grad
    dk_bshd = k_g.grad
    dv_bshd = v_g.grad

    if q_fmt == "thd":
        batch_size = cu_seqlens_q.shape[0] - 1
        dq = convert_bshd_to_thd(dq_bshd, cu_seqlens_q, q.shape[0])
        dk = convert_bshd_to_thd(dk_bshd, cu_seqlens_kv, k.shape[0])
        dv = convert_bshd_to_thd(dv_bshd, cu_seqlens_kv, v.shape[0])
    else:
        dq = _from_bshd(dq_bshd, q_fmt)
        dk = _from_bshd(dk_bshd, kv_fmt)
        dv = _from_bshd(dv_bshd, kv_fmt)

    return dq, dk, dv


# ---------------------------------------------------------------------------
# Public API: fused_attn_fwd / fused_attn_bwd
# ---------------------------------------------------------------------------

def fused_attn_fwd(
    max_seqlen_q,
    max_seqlen_kv,
    is_training,
    attn_scale,
    p_dropout,
    set_zero,
    qkv_layout,
    bias_type,
    attn_mask_type,
    softmax_type,
    window_size,
    cu_seqlens_q,
    cu_seqlens_kv,
    q,
    k,
    v,
    fake_dtype,
    cu_seqlens_q_padded=None,
    cu_seqlens_kv_padded=None,
    page_table_k=None,
    page_table_v=None,
    s_quantizer=None,
    o_quantizer=None,
    attn_bias=None,
    softmax_offset=None,
    rng_gen=None,
    rng_elts_per_thread=0,
    return_max_logit=False,
    cuda_graph=False,
):
    """Fused attention forward -- lite multi-backend dispatcher.

    Signature matches the C++ tex.fused_attn_fwd binding (positional arg order).
    Called from transformer_engine.pytorch.cpp_extensions.fused_attn.fused_attn_fwd.

    Returns a list of tensors: [output, *aux_ctx_tensors].
    """
    _try_load_aiter_attn()

    causal = _is_causal(attn_mask_type)
    bias_tensor = attn_bias if _has_bias_tensor(bias_type) else None

    # Select backend if not already determined
    if _aiter_varlen_fwd is not None:
        backend = NVTE_Fused_Attn_Backend.NVTE_CK
    elif _flash_attn_available:
        backend = NVTE_Fused_Attn_Backend.NVTE_Flash
    else:
        backend = NVTE_Fused_Attn_Backend.NVTE_SDPA

    if backend == NVTE_Fused_Attn_Backend.NVTE_CK:
        out, aux_ctx_tensors = _aiter_attn_fwd(
            q, k, v, cu_seqlens_q, cu_seqlens_kv,
            max_seqlen_q, max_seqlen_kv,
            attn_scale, p_dropout, is_training, causal,
            window_size, bias_tensor, qkv_layout,
            cu_seqlens_q_padded, cu_seqlens_kv_padded,
        )
    elif backend == NVTE_Fused_Attn_Backend.NVTE_Flash:
        raise NotImplementedError(
            "Flash-attention backend is stubbed in lite mode. "
            "Install AITER or use the SDPA fallback."
        )
    else:
        out, aux_ctx_tensors = _sdpa_attn_fwd(
            q, k, v, cu_seqlens_q, cu_seqlens_kv,
            max_seqlen_q, max_seqlen_kv,
            attn_scale, p_dropout, is_training, causal,
            window_size, bias_tensor, qkv_layout,
        )

    # Return format must match C++ extension: list of [output, *aux]
    # The Python wrapper (cpp_extensions/fused_attn.py) does:
    #   return output_tensors[0], output_tensors[1:]
    result = [out] + aux_ctx_tensors
    return result


def fused_attn_bwd(
    max_seqlen_q,
    max_seqlen_kv,
    attn_scale,
    p_dropout,
    set_zero,
    qkv_layout,
    bias_type,
    attn_mask_type,
    softmax_type,
    window_size,
    deterministic,
    cu_seqlens_q,
    cu_seqlens_kv,
    q,
    k,
    v,
    o,
    d_o,
    fake_dtype,
    dqkv_dtype,
    aux_ctx_tensors,
    cu_seqlens_q_padded=None,
    cu_seqlens_kv_padded=None,
    s_quantizer=None,
    dp_quantizer=None,
    dqkv_quantizer=None,
    cuda_graph=False,
):
    """Fused attention backward -- lite multi-backend dispatcher.

    Signature matches the C++ tex.fused_attn_bwd binding (positional arg order).
    Called from transformer_engine.pytorch.cpp_extensions.fused_attn.fused_attn_bwd.

    Returns [dQ, dK, dV, dBias, dSoftmaxOffset].
    """
    _try_load_aiter_attn()

    causal = _is_causal(attn_mask_type)

    if _aiter_varlen_fwd is not None:
        backend = NVTE_Fused_Attn_Backend.NVTE_CK
    elif _flash_attn_available:
        backend = NVTE_Fused_Attn_Backend.NVTE_Flash
    else:
        backend = NVTE_Fused_Attn_Backend.NVTE_SDPA

    if backend == NVTE_Fused_Attn_Backend.NVTE_CK:
        softmax_lse = aux_ctx_tensors[0] if aux_ctx_tensors else None
        rng_state = aux_ctx_tensors[1] if len(aux_ctx_tensors) > 1 else None
        dq, dk, dv = _aiter_attn_bwd(
            d_o, q, k, v, o, softmax_lse, rng_state,
            cu_seqlens_q, cu_seqlens_kv,
            max_seqlen_q, max_seqlen_kv,
            attn_scale, p_dropout, causal,
            window_size, qkv_layout, deterministic,
            cu_seqlens_q_padded, cu_seqlens_kv_padded,
        )
    elif backend == NVTE_Fused_Attn_Backend.NVTE_Flash:
        raise NotImplementedError(
            "Flash-attention backward is stubbed in lite mode."
        )
    else:
        dq, dk, dv = _sdpa_attn_bwd(
            d_o, q, k, v, o,
            cu_seqlens_q, cu_seqlens_kv,
            max_seqlen_q, max_seqlen_kv,
            attn_scale, p_dropout, causal,
            window_size, None, qkv_layout,
        )

    # Return format matches C++ extension: [dQ, dK, dV, dBias, dSoftmaxOffset]
    return [dq, dk, dv, None, None]


# ---------------------------------------------------------------------------
# QKV preparation (flash-attn interleaved format conversions)
# ---------------------------------------------------------------------------

def fa_prepare_fwd(qkvi: torch.Tensor) -> torch.Tensor:
    """Convert interleaved QKV from [s, b, n, 3*h] to [3, b, s, n, h].

    Pure PyTorch replacement for the C++ nvte_prepare_flash_attn_fwd kernel.
    """
    s, b, n, three_h = qkvi.shape
    h = three_h // 3
    # Reshape to [s, b, n, 3, h] then permute to [3, b, s, n, h]
    return qkvi.view(s, b, n, 3, h).permute(3, 1, 0, 2, 4).contiguous()


def fa_prepare_bwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Convert 3 x [s, b, n, h] to [b, s, n, 3*h].

    Pure PyTorch replacement for the C++ nvte_prepare_flash_attn_bwd kernel.
    """
    s, b, n, h = q.shape
    # Stack on new dim -> [3, s, b, n, h], then permute to [b, s, n, 3, h], reshape
    stacked = torch.stack([q, k, v], dim=0)          # [3, s, b, n, h]
    transposed = stacked.permute(2, 1, 3, 0, 4)      # [b, s, n, 3, h]
    return transposed.reshape(b, s, n, 3 * h).contiguous()


# ---------------------------------------------------------------------------
# KV cache operations
# ---------------------------------------------------------------------------

def copy_to_kv_cache(
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cu_new_lens: torch.Tensor,
    cu_cached_lens: torch.Tensor,
    qkv_format: int,
    batch_size: int,
    max_ctx_len: int,
    max_seq_len: int,
    max_pages_per_seq: int,
    is_non_paged: bool,
) -> None:
    """Copy new KV tokens into a KV cache.

    Pure PyTorch replacement for the C++ nvte_copy_to_kv_cache kernel.
    Supports non-paged caches in BSHD and SBHD formats.
    """
    new_lens = cu_new_lens[1:] - cu_new_lens[:-1]
    cached_lens = cu_cached_lens[1:] - cu_cached_lens[:-1]

    # Determine format from enum value
    is_sbhd = (qkv_format == 1)  # NVTE_QKV_Format.NVTE_SBHD

    for b in range(batch_size):
        nl = int(new_lens[b].item())
        cl = int(cached_lens[b].item())
        if nl == 0:
            continue

        if is_sbhd:
            # new_k/v: [seq, batch, heads, dim], cache: [seq, batch, heads, dim]
            k_cache[cl:cl + nl, b] = new_k[:nl, b]
            v_cache[cl:cl + nl, b] = new_v[:nl, b]
        else:
            # BSHD: new_k/v: [batch, seq, heads, dim], cache: same
            k_cache[b, cl:cl + nl] = new_k[b, :nl]
            v_cache[b, cl:cl + nl] = new_v[b, :nl]


# ---------------------------------------------------------------------------
# THD <-> BSHD format conversion
# ---------------------------------------------------------------------------

def convert_thd_to_bshd(
    tensor: torch.Tensor,
    cu_seqlens: torch.Tensor,
    batch_size: Optional[int],
    max_seq_len: int,
) -> torch.Tensor:
    """Convert tensor from THD [total, heads, dim] to BSHD [batch, seq, heads, dim].

    Pure PyTorch replacement for the C++ nvte_convert_thd_to_bshd kernel.
    Sequences shorter than max_seq_len are zero-padded.
    """
    if batch_size is None:
        batch_size = cu_seqlens.shape[0] - 1
    h, d = tensor.shape[1], tensor.shape[2]
    out = tensor.new_zeros(batch_size, max_seq_len, h, d)
    for b in range(batch_size):
        start = int(cu_seqlens[b].item())
        end = int(cu_seqlens[b + 1].item())
        length = end - start
        out[b, :length] = tensor[start:end]
    return out


def convert_bshd_to_thd(
    tensor: torch.Tensor,
    cu_seqlens: torch.Tensor,
    total: int,
) -> torch.Tensor:
    """Convert tensor from BSHD [batch, seq, heads, dim] to THD [total, heads, dim].

    Pure PyTorch replacement for the C++ nvte_convert_bshd_to_thd kernel.
    Strips padding based on cu_seqlens.
    """
    batch_size = tensor.shape[0]
    h, d = tensor.shape[2], tensor.shape[3]
    out = tensor.new_empty(total, h, d)
    for b in range(batch_size):
        start = int(cu_seqlens[b].item())
        end = int(cu_seqlens[b + 1].item())
        length = end - start
        out[start:end] = tensor[b, :length]
    return out
