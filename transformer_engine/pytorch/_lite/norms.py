# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Normalization -- Triton kernels with PyTorch-native fallback.

Uses Triton kernels from triton_kernels/norms_common.py when available,
falls back to pure PyTorch implementations otherwise.
"""

import torch

# Lazy-loaded Triton norm functions. None = not yet attempted.
_triton_ln_fwd = None
_triton_ln_bwd = None
_triton_rms_fwd = None
_triton_rms_bwd = None
_triton_import_attempted = False


def _try_load_triton_norms():
    """Lazy-import Triton norm kernels. Called once, result cached."""
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
        pass  # Triton not available, will use PyTorch fallback


# ---------------------------------------------------------------------------
# PyTorch fallback implementations
# ---------------------------------------------------------------------------

def _layernorm_fwd_pytorch(input, weight, bias, eps, ln_out, quantizer, otype,
                           sm_margin, zero_centered_gamma):
    """LayerNorm forward -- PyTorch fallback."""
    if zero_centered_gamma:
        weight = weight + 1.0

    mean = input.mean(dim=-1, keepdim=True)
    var = input.var(dim=-1, keepdim=True, unbiased=False)
    rstdev = torch.rsqrt(var + eps)

    output = (input - mean) * rstdev * weight
    if bias is not None:
        output = output + bias

    if quantizer is not None and hasattr(quantizer, 'quantize'):
        output = quantizer.quantize(output)

    if ln_out is not None:
        ln_out.copy_(output)
    else:
        ln_out = output

    return ln_out, mean.squeeze(-1), rstdev.squeeze(-1)


def _layernorm_bwd_pytorch(grad_output, input, mean, rstdev, weight,
                           sm_margin, zero_centered_gamma):
    """LayerNorm backward -- PyTorch fallback."""
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


def _rmsnorm_fwd_pytorch(input, weight, eps, ln_out, quantizer, otype,
                         sm_margin, zero_centered_gamma):
    """RMSNorm forward -- PyTorch fallback."""
    if zero_centered_gamma:
        weight = weight + 1.0

    rms = input.float().square().mean(dim=-1, keepdim=True).add_(eps).rsqrt()
    output = (input * rms).to(input.dtype) * weight

    if quantizer is not None and hasattr(quantizer, 'quantize'):
        output = quantizer.quantize(output)

    if ln_out is not None:
        ln_out.copy_(output)
    else:
        ln_out = output

    # Return 3 values to match C++ signature: (output, dummy_mean, rstdev)
    return ln_out, torch.Tensor(), rms.squeeze(-1)


def _rmsnorm_bwd_pytorch(grad_output, input, rstdev, weight, sm_margin,
                         zero_centered_gamma):
    """RMSNorm backward -- PyTorch fallback."""
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
# Public API -- Triton with PyTorch fallback
# ---------------------------------------------------------------------------

def layernorm_fwd(input, weight, bias, eps, ln_out, quantizer, otype, sm_margin,
                  zero_centered_gamma):
    """LayerNorm forward. Uses Triton kernel when available."""
    _try_load_triton_norms()

    if _triton_ln_fwd is None:
        return _layernorm_fwd_pytorch(
            input, weight, bias, eps, ln_out, quantizer, otype,
            sm_margin, zero_centered_gamma,
        )

    # Triton kernels require 2D input (M, N)
    orig_shape = input.shape
    if input.ndim > 2:
        input = input.reshape(-1, orig_shape[-1])

    # Triton kernel needs a concrete otype for output allocation
    if otype is None:
        otype = input.dtype

    out, mu, rsigma = _triton_ln_fwd(
        input, weight, bias, eps, ln_out, quantizer, otype,
        sm_margin, zero_centered_gamma,
    )

    # Reshape output back if we flattened
    if len(orig_shape) > 2:
        batch_shape = orig_shape[:-1]
        if hasattr(out, '_data'):
            # QuantizedTensor: reshape the underlying data
            out._data = out._data.reshape(*batch_shape, -1)
        elif isinstance(out, torch.Tensor):
            out = out.reshape(*batch_shape, -1)
        if mu is not None:
            mu = mu.reshape(batch_shape)
        rsigma = rsigma.reshape(batch_shape)

    return out, mu, rsigma


def layernorm_bwd(grad_output, input, mean, rstdev, weight, sm_margin,
                  zero_centered_gamma):
    """LayerNorm backward. Uses Triton kernel when available."""
    _try_load_triton_norms()

    if _triton_ln_bwd is None:
        return _layernorm_bwd_pytorch(
            grad_output, input, mean, rstdev, weight, sm_margin,
            zero_centered_gamma,
        )

    # Triton kernels require 2D input (M, N)
    orig_shape = input.shape
    if input.ndim > 2:
        input = input.reshape(-1, orig_shape[-1])
        grad_output = grad_output.reshape(-1, orig_shape[-1])
        mean = mean.reshape(-1)
        rstdev = rstdev.reshape(-1)

    dx, dgamma, dbeta = _triton_ln_bwd(
        grad_output, input, mean, rstdev, weight, sm_margin,
        zero_centered_gamma,
    )

    if len(orig_shape) > 2:
        dx = dx.reshape(orig_shape)

    return dx, dgamma, dbeta


def rmsnorm_fwd(input, weight, eps, ln_out, quantizer, otype, sm_margin,
                zero_centered_gamma):
    """RMSNorm forward. Uses Triton kernel when available."""
    _try_load_triton_norms()

    if _triton_rms_fwd is None:
        return _rmsnorm_fwd_pytorch(
            input, weight, eps, ln_out, quantizer, otype,
            sm_margin, zero_centered_gamma,
        )

    # Triton kernels require 2D input (M, N)
    orig_shape = input.shape
    if input.ndim > 2:
        input = input.reshape(-1, orig_shape[-1])

    # Triton kernel needs a concrete otype for output allocation
    if otype is None:
        otype = input.dtype

    out, mu, rsigma = _triton_rms_fwd(
        input, weight, eps, ln_out, quantizer, otype,
        sm_margin, zero_centered_gamma,
    )

    if len(orig_shape) > 2:
        batch_shape = orig_shape[:-1]
        if hasattr(out, '_data'):
            out._data = out._data.reshape(*batch_shape, -1)
        elif isinstance(out, torch.Tensor):
            out = out.reshape(*batch_shape, -1)
        rsigma = rsigma.reshape(batch_shape)

    return out, mu, rsigma


def rmsnorm_bwd(grad_output, input, rstdev, weight, sm_margin, zero_centered_gamma):
    """RMSNorm backward. Uses Triton kernel when available."""
    _try_load_triton_norms()

    if _triton_rms_bwd is None:
        return _rmsnorm_bwd_pytorch(
            grad_output, input, rstdev, weight, sm_margin,
            zero_centered_gamma,
        )

    # Triton kernels require 2D input (M, N)
    orig_shape = input.shape
    if input.ndim > 2:
        input = input.reshape(-1, orig_shape[-1])
        grad_output = grad_output.reshape(-1, orig_shape[-1])
        rstdev = rstdev.reshape(-1)

    dx, dgamma = _triton_rms_bwd(
        grad_output, input, rstdev, weight, sm_margin,
        zero_centered_gamma,
    )

    if len(orig_shape) > 2:
        dx = dx.reshape(orig_shape)

    return dx, dgamma


def rmsnorm_bwd_add(grad_output, input, rstdev, weight, zero_centered_gamma):
    """Fused RMSNorm backward + add. Returns (grad_input, grad_weight)."""
    return rmsnorm_bwd(grad_output, input, rstdev, weight, 0, zero_centered_gamma)
