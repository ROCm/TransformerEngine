# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""FlyDSL fused attention adapter for TransformerEngine.

Wraps FlyDSL flash attention forward kernels for use as a TE fused attention backend.
Backward pass uses PyTorch SDPA recompute (FlyDSL backward kernels are not yet available).

Usage: set NVTE_FUSED_ATTN_FLYDSL=1 to enable this backend.
"""

import math
import os
import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger("FlyDSL")

_fwd_call_count = 0
_bwd_call_count = 0


# ---------------------------------------------------------------------------
# FlyDSL import
# ---------------------------------------------------------------------------

_flydsl_available = None
_flydsl_flash_attn_func = None


def _try_import_flydsl():
    """Lazy-import FlyDSL flash attention from TE-bundled kernels."""
    global _flydsl_available, _flydsl_flash_attn_func

    if _flydsl_available is not None:
        return _flydsl_available

    try:
        from transformer_engine.pytorch.attention.flydsl_kernels.flash_attn_interface import (
            flydsl_flash_attn_func,
        )
        _flydsl_flash_attn_func = flydsl_flash_attn_func
        _flydsl_available = True
        print("[FlyDSL] Successfully loaded from TE-bundled flydsl_kernels")
    except Exception as e:
        _flydsl_available = False
        print(f"[FlyDSL] Failed to load: {e}")
        import traceback; traceback.print_exc()

    return _flydsl_available


def is_flydsl_available():
    """Return True if FlyDSL flash attention kernels can be imported."""
    return _try_import_flydsl()


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def _qkv_layout_to_format(qkv_layout: str) -> str:
    """Extract the Q-tensor format from a TE qkv_layout string.

    Examples:
        "bshd_bshd_bshd" -> "bshd"
        "sbhd_sbhd_sbhd" -> "sbhd"
        "bs3hd"          -> "bshd"
        "bsh3d"          -> "bshd"
    """
    # For separate layouts like "bshd_bshd_bshd", take the first segment
    first = qkv_layout.split("_")[0]
    # Strip digits (e.g. "bs3hd" -> "bshd")
    return "".join(c for c in first if c.isalpha())


def _to_bshd(tensor, fmt):
    """Convert a tensor from TE format to BSHD [B, S, H, D]."""
    if fmt == "bshd":
        return tensor
    if fmt == "sbhd":
        return tensor.transpose(0, 1).contiguous()
    raise ValueError(f"Unsupported QKV format for FlyDSL: {fmt}")


def _from_bshd(tensor, fmt):
    """Convert a tensor from BSHD [B, S, H, D] back to TE format."""
    if fmt == "bshd":
        return tensor
    if fmt == "sbhd":
        return tensor.transpose(0, 1).contiguous()
    raise ValueError(f"Unsupported QKV format for FlyDSL: {fmt}")


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------

def flydsl_fused_attn_fwd(
    is_training,
    max_seqlen_q,
    max_seqlen_kv,
    cu_seqlens_q,
    cu_seqlens_kv,
    q,
    k,
    v,
    qkv_layout="bshd_bshd_bshd",
    attn_mask_type="causal",
    attn_scale=None,
):
    """Run FlyDSL flash attention forward.

    Returns (output, aux_ctx_tensors) matching the interface of fused_attn_fwd.
    aux_ctx_tensors = [softmax_lse, rng_state] — lse is a placeholder since
    FlyDSL doesn't output it; backward will recompute via SDPA.
    """
    global _fwd_call_count
    _fwd_call_count += 1

    fmt = _qkv_layout_to_format(qkv_layout)
    q_bshd = _to_bshd(q, fmt)
    k_bshd = _to_bshd(k, fmt)
    v_bshd = _to_bshd(v, fmt)

    B, S_q, H_q, D = q_bshd.shape
    H_kv = k_bshd.shape[2]

    if _fwd_call_count <= 2 or _fwd_call_count % 100 == 0:
        print(
            f"[FlyDSL] fwd #{_fwd_call_count}: "
            f"B={B} S_q={S_q} H_q={H_q} H_kv={H_kv} D={D} "
            f"causal={'causal' in attn_mask_type} layout={qkv_layout} "
            f"dtype={q.dtype}"
        )

    causal = "causal" in attn_mask_type

    # Call the unified FlyDSL API — handles GQA, flattening, kernel caching
    output_bshd = _flydsl_flash_attn_func(
        q_bshd, k_bshd, v_bshd,
        causal=causal,
        num_kv_heads=H_kv,
    )

    output = _from_bshd(output_bshd, fmt)

    # Create placeholder aux_ctx_tensors for backward compatibility.
    # softmax_lse: [B, H_q, max_seqlen_q, 1] (float32) — placeholder, backward recomputes
    softmax_lse = torch.zeros(B, H_q, max_seqlen_q, 1, dtype=torch.float32, device=q.device)
    # rng_state: [2] (int64) — no dropout
    rng_state = torch.zeros(2, dtype=torch.int64, device=q.device)
    aux_ctx_tensors = [softmax_lse, rng_state]

    return output, aux_ctx_tensors


# ---------------------------------------------------------------------------
# Backward (via PyTorch SDPA recompute)
# ---------------------------------------------------------------------------

def flydsl_fused_attn_bwd(
    max_seqlen_q,
    max_seqlen_kv,
    cu_seqlens_q,
    cu_seqlens_kv,
    q,
    k,
    v,
    o,
    d_o,
    qkv_layout="bshd_bshd_bshd",
    attn_mask_type="causal",
    attn_bias_type="no_bias",
    attn_scale=None,
    dropout=0.0,
    window_size=(-1, -1),
):
    """Compute attention backward via PyTorch SDPA recomputation.

    FlyDSL doesn't have backward kernels yet, so we recompute the forward
    via F.scaled_dot_product_attention and let PyTorch autograd handle gradients.

    Returns (dq, dk, dv) in the original TE layout format.
    """
    global _bwd_call_count
    _bwd_call_count += 1

    fmt = _qkv_layout_to_format(qkv_layout)

    # Convert to BSHD
    q_bshd = _to_bshd(q, fmt).detach()
    k_bshd = _to_bshd(k, fmt).detach()
    v_bshd = _to_bshd(v, fmt).detach()
    d_out_bshd = _to_bshd(d_o, fmt)

    B, S_q, H_q, D = q_bshd.shape
    H_kv = k_bshd.shape[2]

    if _bwd_call_count <= 2 or _bwd_call_count % 100 == 0:
        print(
            f"[FlyDSL] bwd #{_bwd_call_count} (SDPA recompute): "
            f"B={B} S_q={S_q} H_q={H_q} H_kv={H_kv} D={D}"
        )

    # GQA/MQA: expand KV heads to match Q heads for SDPA
    if H_kv != H_q:
        repeats = H_q // H_kv
        k_bshd = k_bshd.repeat_interleave(repeats, dim=2)
        v_bshd = v_bshd.repeat_interleave(repeats, dim=2)

    if attn_scale is None:
        attn_scale = 1.0 / math.sqrt(D)

    causal = "causal" in attn_mask_type

    # SDPA expects [B, H, S, D]
    q_bhsd = q_bshd.transpose(1, 2).requires_grad_(True)
    k_bhsd = k_bshd.transpose(1, 2).requires_grad_(True)
    v_bhsd = v_bshd.transpose(1, 2).requires_grad_(True)
    d_out_bhsd = d_out_bshd.transpose(1, 2).contiguous()

    with torch.enable_grad():
        out_bhsd = F.scaled_dot_product_attention(
            q_bhsd, k_bhsd, v_bhsd,
            is_causal=causal,
            scale=attn_scale,
        )
        out_bhsd.backward(d_out_bhsd)

    # Gradients: [B, H_q, S, D] -> [B, S, H_q, D] -> TE layout
    dq = _from_bshd(q_bhsd.grad.transpose(1, 2).contiguous(), fmt)

    # GQA/MQA: sum expanded KV gradients back to original head count
    dk_expanded = k_bhsd.grad.transpose(1, 2)  # [B, S, H_q, D]
    dv_expanded = v_bhsd.grad.transpose(1, 2)  # [B, S, H_q, D]
    if H_kv != H_q:
        repeats = H_q // H_kv
        dk_expanded = dk_expanded.view(B, S_q, H_kv, repeats, D).sum(dim=3)
        dv_expanded = dv_expanded.view(B, S_q, H_kv, repeats, D).sum(dim=3)
    dk = _from_bshd(dk_expanded.contiguous(), fmt)
    dv = _from_bshd(dv_expanded.contiguous(), fmt)

    return dq, dk, dv
