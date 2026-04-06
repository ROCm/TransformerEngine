# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Normalization -- wrappers around existing Triton kernels.

TODO Phase 1: Wire up to triton_kernels/layernorm.py and rmsnorm.py.
For now, uses PyTorch-native implementations as placeholder.
"""

import torch


def layernorm_fwd(input, weight, bias, eps, ln_out, quantizer, otype, sm_margin,
                  zero_centered_gamma):
    """LayerNorm forward."""
    if zero_centered_gamma:
        weight = weight + 1.0

    # Compute mean and rstdev
    mean = input.mean(dim=-1, keepdim=True)
    var = input.var(dim=-1, keepdim=True, unbiased=False)
    rstdev = torch.rsqrt(var + eps)

    # Normalize
    output = (input - mean) * rstdev * weight
    if bias is not None:
        output = output + bias

    # Quantize if needed
    if quantizer is not None and hasattr(quantizer, 'quantize'):
        output = quantizer.quantize(output)

    if ln_out is not None:
        ln_out.copy_(output)
    else:
        ln_out = output

    return ln_out, mean.squeeze(-1), rstdev.squeeze(-1)


def layernorm_bwd(grad_output, input, mean, rstdev, weight, sm_margin, zero_centered_gamma):
    """LayerNorm backward."""
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

    if zero_centered_gamma:
        # Adjust grad_weight for zero_centered_gamma
        pass  # grad_weight is already correct for (weight + 1)

    return grad_input, grad_weight, grad_bias


def rmsnorm_fwd(input, weight, eps, ln_out, quantizer, otype, sm_margin, zero_centered_gamma):
    """RMSNorm forward."""
    if zero_centered_gamma:
        weight = weight + 1.0

    # Compute RMS
    rms = input.float().square().mean(dim=-1, keepdim=True).add_(eps).rsqrt()
    output = (input * rms).to(input.dtype) * weight

    # Quantize if needed
    if quantizer is not None and hasattr(quantizer, 'quantize'):
        output = quantizer.quantize(output)

    if ln_out is not None:
        ln_out.copy_(output)
    else:
        ln_out = output

    # Return 3 values to match C++ signature: (output, dummy_mean, rstdev)
    return ln_out, torch.Tensor(), rms.squeeze(-1)


def rmsnorm_bwd(grad_output, input, rstdev, weight, sm_margin, zero_centered_gamma):
    """RMSNorm backward."""
    if zero_centered_gamma:
        weight = weight + 1.0

    hidden_size = input.shape[-1]
    x_hat = input * rstdev.unsqueeze(-1)

    grad_weight = (grad_output * x_hat).sum(dim=tuple(range(grad_output.ndim - 1)))

    dx_hat = grad_output * weight
    # d(x * rsqrt(mean(x^2) + eps)) / dx
    grad_input = dx_hat * rstdev.unsqueeze(-1) - \
                 (dx_hat * input).sum(dim=-1, keepdim=True) * input * \
                 (rstdev.unsqueeze(-1) ** 3) / hidden_size

    return grad_input, grad_weight


def rmsnorm_bwd_add(grad_output, input, rstdev, weight, zero_centered_gamma):
    """Fused RMSNorm backward + add. Returns (grad_input, grad_weight)."""
    return rmsnorm_bwd(grad_output, input, rstdev, weight, zero_centered_gamma)
