# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Normalization -- AITER Triton, TE Triton, or PyTorch-native fallback.

Backend priority:
  1. AITER Triton kernels (aiter.ops.triton.rmsnorm / norm) -- tuned for MI300X
  2. TE Triton kernels (triton_kernels/norms_common.py)
  3. Pure PyTorch fallback

The fused norm+quantize path (AITER's dynamicquant/smoothquant) uses per-row
scaling which is incompatible with TE's per-tensor FP8 scaling. Quantization
is therefore applied as a separate step via the quantizer interface.
"""

import torch

from .aiter_utils import is_aiter_available

# ---------------------------------------------------------------------------
# Lazy-loaded backends. None = not yet attempted.
# ---------------------------------------------------------------------------

# AITER Triton norm functions
_aiter_rms_fwd = None
_aiter_rms_bwd = None
_aiter_ln_fwd = None
_aiter_ln_bwd = None
_aiter_import_attempted = False

# TE Triton norm functions (fallback)
_triton_ln_fwd = None
_triton_ln_bwd = None
_triton_rms_fwd = None
_triton_rms_bwd = None
_triton_import_attempted = False


def _try_load_aiter_norms():
    """Lazy-import AITER Triton norm kernels. Called once, result cached."""
    global _aiter_rms_fwd, _aiter_rms_bwd, _aiter_ln_fwd, _aiter_ln_bwd
    global _aiter_import_attempted

    if _aiter_import_attempted:
        return
    _aiter_import_attempted = True

    if not is_aiter_available():
        return
    try:
        from aiter.ops.triton.rmsnorm import (
            _rmsnorm_forward,
            _rmsnorm_backward,
        )
        from aiter.ops.triton.norm import (
            _layernorm_forward,
            _layernorm_backward,
        )
        _aiter_rms_fwd = _rmsnorm_forward
        _aiter_rms_bwd = _rmsnorm_backward
        _aiter_ln_fwd = _layernorm_forward
        _aiter_ln_bwd = _layernorm_backward
    except (ImportError, AttributeError):
        pass


def _try_load_triton_norms():
    """Lazy-import TE Triton norm kernels. Called once, result cached."""
    global _triton_ln_fwd, _triton_ln_bwd
    global _triton_rms_fwd, _triton_rms_bwd
    global _triton_import_attempted

    if _triton_import_attempted:
        return

    _triton_import_attempted = True
    try:
        from transformer_engine.pytorch.triton_kernels.norms_common import (
            te_layernorm_fwd_triton,
            te_layernorm_bwd_triton,
            te_rmsnorm_fwd_triton,
            te_rmsnorm_bwd_triton,
        )
        _triton_ln_fwd = te_layernorm_fwd_triton
        _triton_ln_bwd = te_layernorm_bwd_triton
        _triton_rms_fwd = te_rmsnorm_fwd_triton
        _triton_rms_bwd = te_rmsnorm_bwd_triton
    except (ImportError, ModuleNotFoundError):
        pass


# ---------------------------------------------------------------------------
# PyTorch fallback implementations
# ---------------------------------------------------------------------------

def _layernorm_fwd_pytorch(input, weight, bias, eps, zero_centered_gamma):
    """LayerNorm forward -- pure PyTorch."""
    if zero_centered_gamma:
        weight = weight + 1.0
    mean = input.mean(dim=-1, keepdim=True)
    var = input.var(dim=-1, keepdim=True, unbiased=False)
    rstdev = torch.rsqrt(var + eps)
    output = (input - mean) * rstdev * weight
    if bias is not None:
        output = output + bias
    return output, mean.squeeze(-1), rstdev.squeeze(-1)


def _layernorm_bwd_pytorch(grad_output, input, mean, rstdev, weight,
                           zero_centered_gamma):
    """LayerNorm backward -- pure PyTorch."""
    if zero_centered_gamma:
        weight = weight + 1.0
    hidden_size = input.shape[-1]
    x_hat = (input - mean.unsqueeze(-1)) * rstdev.unsqueeze(-1)
    grad_weight = (grad_output * x_hat).sum(dim=tuple(range(grad_output.ndim - 1)))
    grad_bias = grad_output.sum(dim=tuple(range(grad_output.ndim - 1)))
    dx_hat = grad_output * weight
    dvar = (dx_hat * (input - mean.unsqueeze(-1)) * (-0.5) *
            (rstdev.unsqueeze(-1) ** 3)).sum(dim=-1, keepdim=True)
    dmean = (-dx_hat * rstdev.unsqueeze(-1)).sum(dim=-1, keepdim=True) + \
            dvar * (-2.0 / hidden_size) * (input - mean.unsqueeze(-1)).sum(dim=-1, keepdim=True)
    grad_input = dx_hat * rstdev.unsqueeze(-1) + \
                 dvar * 2.0 / hidden_size * (input - mean.unsqueeze(-1)) + \
                 dmean / hidden_size
    return grad_input, grad_weight, grad_bias


def _rmsnorm_fwd_pytorch(input, weight, eps, zero_centered_gamma):
    """RMSNorm forward -- pure PyTorch."""
    if zero_centered_gamma:
        weight = weight + 1.0
    rms = input.float().square().mean(dim=-1, keepdim=True).add_(eps).rsqrt()
    output = (input * rms).to(input.dtype) * weight
    return output, rms.squeeze(-1)


def _rmsnorm_bwd_pytorch(grad_output, input, rstdev, weight, zero_centered_gamma):
    """RMSNorm backward -- pure PyTorch."""
    if zero_centered_gamma:
        weight = weight + 1.0
    hidden_size = input.shape[-1]
    x_hat = input * rstdev.unsqueeze(-1)
    grad_weight = (grad_output * x_hat).sum(dim=tuple(range(grad_output.ndim - 1)))
    dx_hat = grad_output * weight
    grad_input = dx_hat * rstdev.unsqueeze(-1) - \
                 (dx_hat * input).sum(dim=-1, keepdim=True) * input * \
                 (rstdev.unsqueeze(-1) ** 3) / hidden_size
    return grad_input, grad_weight


# ---------------------------------------------------------------------------
# AITER adapter functions
# ---------------------------------------------------------------------------

def _aiter_layernorm_fwd(input_2d, weight, bias, eps, zero_centered_gamma):
    """LayerNorm forward via AITER Triton kernel.

    AITER's _layernorm_forward writes into pre-allocated tensors.
    """
    if zero_centered_gamma:
        weight = weight + 1.0
    M, N = input_2d.shape
    y = torch.empty_like(input_2d)
    mean = torch.empty(M, dtype=torch.float32, device=input_2d.device)
    rstd = torch.empty(M, dtype=torch.float32, device=input_2d.device)
    _aiter_ln_fwd(y, input_2d, weight, bias, mean, rstd, eps)
    return y, mean, rstd


def _aiter_layernorm_bwd(grad_output_2d, input_2d, mean, rstdev, weight,
                         zero_centered_gamma):
    """LayerNorm backward via AITER Triton kernel.

    AITER's _layernorm_backward writes into pre-allocated tensors.
    """
    if zero_centered_gamma:
        weight = weight + 1.0
    dx = torch.empty_like(input_2d)
    dw = torch.empty_like(weight)
    db = torch.empty_like(weight)
    _aiter_ln_bwd(grad_output_2d, dx, dw, db, input_2d, weight, mean, rstdev)
    return dx, dw, db


def _aiter_rmsnorm_fwd(input_2d, weight, eps, zero_centered_gamma):
    """RMSNorm forward via AITER Triton kernel.

    AITER's _rmsnorm_forward allocates output internally.
    """
    if zero_centered_gamma:
        weight = weight + 1.0
    y, rsigma = _aiter_rms_fwd(input_2d, weight, eps)
    return y, rsigma


def _aiter_rmsnorm_bwd(grad_output_2d, input_2d, rstdev, weight,
                       zero_centered_gamma):
    """RMSNorm backward via AITER Triton kernel."""
    if zero_centered_gamma:
        weight = weight + 1.0
    dx, dgamma = _aiter_rms_bwd(grad_output_2d, input_2d, weight, rstdev)
    return dx, dgamma


# ---------------------------------------------------------------------------
# Reshape helpers for N-D input
# ---------------------------------------------------------------------------

def _ensure_2d(t):
    """Reshape to 2D (M, N) if needed. Returns (tensor_2d, original_shape)."""
    if t.ndim <= 2:
        return t, t.shape
    orig = t.shape
    return t.reshape(-1, orig[-1]), orig


def _restore_nd(t, orig_shape):
    """Restore from 2D back to original N-D shape."""
    if len(orig_shape) <= 2:
        return t
    return t.reshape(orig_shape)


def _restore_nd_quantized(out, orig_shape):
    """Restore N-D shape for possibly-quantized output."""
    if len(orig_shape) <= 2:
        return out
    batch_shape = orig_shape[:-1]
    if hasattr(out, '_data'):
        out._data = out._data.reshape(*batch_shape, -1)
    elif isinstance(out, torch.Tensor):
        out = out.reshape(*batch_shape, -1)
    return out


def _restore_stats(stats, orig_shape):
    """Restore stats (mean, rstdev) to batch shape."""
    if len(orig_shape) <= 2 or stats is None:
        return stats
    if stats.numel() == 0:
        return stats
    return stats.reshape(orig_shape[:-1])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def layernorm_fwd(input, weight, bias, eps, ln_out, quantizer, otype, sm_margin,
                  zero_centered_gamma):
    """LayerNorm forward.

    Backend priority: AITER Triton > TE Triton > PyTorch.
    """
    _try_load_aiter_norms()
    _try_load_triton_norms()

    input_2d, orig_shape = _ensure_2d(input)

    # Try AITER Triton
    if _aiter_ln_fwd is not None:
        out, mu, rsigma = _aiter_layernorm_fwd(input_2d, weight, bias, eps,
                                                zero_centered_gamma)
    # Try TE Triton
    elif _triton_ln_fwd is not None:
        if otype is None:
            otype = input.dtype
        out, mu, rsigma = _triton_ln_fwd(
            input_2d, weight, bias, eps, ln_out, quantizer, otype,
            sm_margin, zero_centered_gamma,
        )
        # TE Triton handles quantizer internally
        out = _restore_nd_quantized(out, orig_shape)
        mu = _restore_stats(mu, orig_shape)
        rsigma = _restore_stats(rsigma, orig_shape)
        return out, mu, rsigma
    # PyTorch fallback
    else:
        out, mu, rsigma = _layernorm_fwd_pytorch(input_2d, weight, bias, eps,
                                                  zero_centered_gamma)

    # Apply quantizer (separate step -- AITER and PyTorch paths)
    if quantizer is not None and hasattr(quantizer, 'quantize'):
        out = quantizer.quantize(out)

    if ln_out is not None and ln_out is not out:
        ln_out.copy_(out)
    else:
        ln_out = out

    ln_out = _restore_nd_quantized(ln_out, orig_shape)
    mu = _restore_stats(mu, orig_shape)
    rsigma = _restore_stats(rsigma, orig_shape)
    return ln_out, mu, rsigma


def layernorm_bwd(grad_output, input, mean, rstdev, weight, sm_margin,
                  zero_centered_gamma):
    """LayerNorm backward.

    Backend priority: AITER Triton > TE Triton > PyTorch.
    """
    _try_load_aiter_norms()
    _try_load_triton_norms()

    orig_shape = input.shape
    input_2d, _ = _ensure_2d(input)
    grad_2d, _ = _ensure_2d(grad_output)
    if mean is not None and mean.ndim > 1:
        mean = mean.reshape(-1)
    if rstdev.ndim > 1:
        rstdev = rstdev.reshape(-1)

    if _aiter_ln_bwd is not None:
        dx, dgamma, dbeta = _aiter_layernorm_bwd(grad_2d, input_2d, mean, rstdev,
                                                   weight, zero_centered_gamma)
    elif _triton_ln_bwd is not None:
        dx, dgamma, dbeta = _triton_ln_bwd(
            grad_2d, input_2d, mean, rstdev, weight, sm_margin,
            zero_centered_gamma,
        )
    else:
        dx, dgamma, dbeta = _layernorm_bwd_pytorch(grad_2d, input_2d, mean, rstdev,
                                                     weight, zero_centered_gamma)

    dx = _restore_nd(dx, orig_shape)
    return dx, dgamma, dbeta


def rmsnorm_fwd(input, weight, eps, ln_out, quantizer, otype, sm_margin,
                zero_centered_gamma):
    """RMSNorm forward.

    Backend priority: AITER Triton > TE Triton > PyTorch.
    """
    _try_load_aiter_norms()
    _try_load_triton_norms()

    input_2d, orig_shape = _ensure_2d(input)

    # Try AITER Triton
    if _aiter_rms_fwd is not None:
        out, rsigma = _aiter_rmsnorm_fwd(input_2d, weight, eps, zero_centered_gamma)
        mu = torch.Tensor()  # empty, matches C++ signature
    # Try TE Triton
    elif _triton_rms_fwd is not None:
        if otype is None:
            otype = input.dtype
        out, mu, rsigma = _triton_rms_fwd(
            input_2d, weight, eps, ln_out, quantizer, otype,
            sm_margin, zero_centered_gamma,
        )
        out = _restore_nd_quantized(out, orig_shape)
        rsigma = _restore_stats(rsigma, orig_shape)
        return out, mu, rsigma
    # PyTorch fallback
    else:
        out, rsigma = _rmsnorm_fwd_pytorch(input_2d, weight, eps, zero_centered_gamma)
        mu = torch.Tensor()

    # Apply quantizer (separate step -- AITER and PyTorch paths)
    if quantizer is not None and hasattr(quantizer, 'quantize'):
        out = quantizer.quantize(out)

    if ln_out is not None and ln_out is not out:
        ln_out.copy_(out)
    else:
        ln_out = out

    ln_out = _restore_nd_quantized(ln_out, orig_shape)
    rsigma = _restore_stats(rsigma, orig_shape)
    return ln_out, mu, rsigma


def rmsnorm_bwd(grad_output, input, rstdev, weight, sm_margin, zero_centered_gamma):
    """RMSNorm backward.

    Backend priority: AITER Triton > TE Triton > PyTorch.
    """
    _try_load_aiter_norms()
    _try_load_triton_norms()

    orig_shape = input.shape
    input_2d, _ = _ensure_2d(input)
    grad_2d, _ = _ensure_2d(grad_output)
    if rstdev.ndim > 1:
        rstdev = rstdev.reshape(-1)

    if _aiter_rms_bwd is not None:
        dx, dgamma = _aiter_rmsnorm_bwd(grad_2d, input_2d, rstdev, weight,
                                         zero_centered_gamma)
    elif _triton_rms_bwd is not None:
        dx, dgamma = _triton_rms_bwd(
            grad_2d, input_2d, rstdev, weight, sm_margin,
            zero_centered_gamma,
        )
    else:
        dx, dgamma = _rmsnorm_bwd_pytorch(grad_2d, input_2d, rstdev, weight,
                                           zero_centered_gamma)

    dx = _restore_nd(dx, orig_shape)
    return dx, dgamma


def rmsnorm_bwd_add(grad_output, input, rstdev, weight, zero_centered_gamma):
    """Fused RMSNorm backward + add. Returns (grad_input, grad_weight)."""
    return rmsnorm_bwd(grad_output, input, rstdev, weight, 0, zero_centered_gamma)
