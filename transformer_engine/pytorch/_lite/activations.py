# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Activation functions -- PyTorch-native implementations.

TODO Phase 2: Replace with AITER fused_fp8_quant or standalone Triton fused act+quantize.
"""

import torch
import torch.nn.functional as F
import math


def _apply_quantizer(output, quantizer):
    """Apply quantizer if provided, otherwise return as-is."""
    if quantizer is not None and hasattr(quantizer, 'quantize'):
        return quantizer.quantize(output)
    return output


# --------------------------------------------------------------------------- #
# Forward activations
# --------------------------------------------------------------------------- #

def gelu(input, quantizer):
    """GeLU activation (tanh approximation)."""
    out = F.gelu(input, approximate='tanh')
    return _apply_quantizer(out, quantizer)


def geglu(input, quantizer):
    """GeGLU: split input in half, apply GELU to first, multiply by second."""
    chunks = input.chunk(2, dim=-1)
    out = F.gelu(chunks[0], approximate='tanh') * chunks[1]
    return _apply_quantizer(out, quantizer)


def qgelu(input, quantizer):
    """QuickGELU: x * sigmoid(1.702 * x)."""
    out = input * torch.sigmoid(1.702 * input)
    return _apply_quantizer(out, quantizer)


def qgeglu(input, quantizer):
    """Quick GeGLU: gated variant of QuickGELU."""
    chunks = input.chunk(2, dim=-1)
    out = (chunks[0] * torch.sigmoid(1.702 * chunks[0])) * chunks[1]
    return _apply_quantizer(out, quantizer)


def relu(input, quantizer):
    """ReLU activation."""
    out = F.relu(input)
    return _apply_quantizer(out, quantizer)


def reglu(input, quantizer):
    """ReGLU: gated variant of ReLU."""
    chunks = input.chunk(2, dim=-1)
    out = F.relu(chunks[0]) * chunks[1]
    return _apply_quantizer(out, quantizer)


def srelu(input, quantizer):
    """Squared ReLU: relu(x)^2."""
    out = F.relu(input).square()
    return _apply_quantizer(out, quantizer)


def sreglu(input, quantizer):
    """Squared ReGLU: gated variant of squared ReLU."""
    chunks = input.chunk(2, dim=-1)
    out = F.relu(chunks[0]).square() * chunks[1]
    return _apply_quantizer(out, quantizer)


def silu(input, quantizer):
    """SiLU (Swish) activation."""
    out = F.silu(input)
    return _apply_quantizer(out, quantizer)


def swiglu(input, quantizer):
    """SwiGLU: gated variant of SiLU."""
    chunks = input.chunk(2, dim=-1)
    out = F.silu(chunks[0]) * chunks[1]
    return _apply_quantizer(out, quantizer)


def clamped_swiglu(input, quantizer, limit=7.0, alpha=1.702):
    """SwiGLU with clamping (GPT OSS variant)."""
    chunks = input.chunk(2, dim=-1)
    out = F.silu(chunks[0]) * chunks[1]
    out = out.clamp(min=-limit, max=limit)
    return _apply_quantizer(out, quantizer)


# --------------------------------------------------------------------------- #
# Backward activations
# --------------------------------------------------------------------------- #

def _gelu_backward(grad, x):
    """Backward of tanh-approximated GELU."""
    kBeta = math.sqrt(2.0 / math.pi)
    kKappa = 0.044715
    x_cube = x * x * x
    inner = kBeta * (x + kKappa * x_cube)
    tanh_inner = torch.tanh(inner)
    dtanh = 1.0 - tanh_inner * tanh_inner
    d_inner = kBeta * (1.0 + 3.0 * kKappa * x * x)
    return grad * 0.5 * (1.0 + tanh_inner + x * dtanh * d_inner)


def dgelu(grad, fwd_input, quantizer):
    """Backward of GeLU."""
    out = _gelu_backward(grad, fwd_input)
    return _apply_quantizer(out, quantizer)


def dgeglu(grad, fwd_input, quantizer):
    """Backward of GeGLU."""
    chunks = fwd_input.chunk(2, dim=-1)
    x, gate = chunks[0], chunks[1]
    gelu_x = F.gelu(x, approximate='tanh')
    dgelu_x = _gelu_backward(grad * gate, x)
    dgate = grad * gelu_x
    out = torch.cat([dgelu_x, dgate], dim=-1)
    return _apply_quantizer(out, quantizer)


def dqgelu(grad, fwd_input, quantizer):
    """Backward of QuickGELU."""
    sig = torch.sigmoid(1.702 * fwd_input)
    out = grad * sig * (1.0 + 1.702 * fwd_input * (1.0 - sig))
    return _apply_quantizer(out, quantizer)


def dqgeglu(grad, fwd_input, quantizer):
    """Backward of Quick GeGLU."""
    chunks = fwd_input.chunk(2, dim=-1)
    x, gate = chunks[0], chunks[1]
    sig = torch.sigmoid(1.702 * x)
    qgelu_x = x * sig
    dqgelu_x = grad * gate * sig * (1.0 + 1.702 * x * (1.0 - sig))
    dgate = grad * qgelu_x
    out = torch.cat([dqgelu_x, dgate], dim=-1)
    return _apply_quantizer(out, quantizer)


def drelu(grad, fwd_input, quantizer):
    """Backward of ReLU."""
    out = grad * (fwd_input > 0).to(grad.dtype)
    return _apply_quantizer(out, quantizer)


def dreglu(grad, fwd_input, quantizer):
    """Backward of ReGLU."""
    chunks = fwd_input.chunk(2, dim=-1)
    x, gate = chunks[0], chunks[1]
    dx = grad * gate * (x > 0).to(grad.dtype)
    dgate = grad * F.relu(x)
    out = torch.cat([dx, dgate], dim=-1)
    return _apply_quantizer(out, quantizer)


def dsrelu(grad, fwd_input, quantizer):
    """Backward of Squared ReLU."""
    out = grad * 2.0 * F.relu(fwd_input)
    return _apply_quantizer(out, quantizer)


def dsreglu(grad, fwd_input, quantizer):
    """Backward of Squared ReGLU."""
    chunks = fwd_input.chunk(2, dim=-1)
    x, gate = chunks[0], chunks[1]
    dx = grad * gate * 2.0 * F.relu(x)
    dgate = grad * F.relu(x).square()
    out = torch.cat([dx, dgate], dim=-1)
    return _apply_quantizer(out, quantizer)


def dsilu(grad, fwd_input, quantizer):
    """Backward of SiLU."""
    sig = torch.sigmoid(fwd_input)
    out = grad * sig * (1.0 + fwd_input * (1.0 - sig))
    return _apply_quantizer(out, quantizer)


def dswiglu(grad, fwd_input, quantizer):
    """Backward of SwiGLU."""
    chunks = fwd_input.chunk(2, dim=-1)
    x, gate = chunks[0], chunks[1]
    sig = torch.sigmoid(x)
    silu_x = x * sig
    dx = grad * gate * sig * (1.0 + x * (1.0 - sig))
    dgate = grad * silu_x
    out = torch.cat([dx, dgate], dim=-1)
    return _apply_quantizer(out, quantizer)


def clamped_dswiglu(grad, fwd_input, quantizer, limit=7.0, alpha=1.702):
    """Backward of clamped SwiGLU."""
    chunks = fwd_input.chunk(2, dim=-1)
    x, gate = chunks[0], chunks[1]
    sig = torch.sigmoid(x)
    silu_x = x * sig
    fwd_out = silu_x * gate
    # Zero out gradient where clamped
    mask = (fwd_out >= -limit) & (fwd_out <= limit)
    grad = grad * mask.to(grad.dtype)
    dx = grad * gate * sig * (1.0 + x * (1.0 - sig))
    dgate = grad * silu_x
    out = torch.cat([dx, dgate], dim=-1)
    return _apply_quantizer(out, quantizer)


# --------------------------------------------------------------------------- #
# DBias + DAct fusions
# --------------------------------------------------------------------------- #

def dbias_dgelu(grad, fwd_input, quantizer):
    """Fused DGeLU + DBias: returns (dact, dbias)."""
    dact = _gelu_backward(grad, fwd_input)
    dbias = dact.sum(dim=tuple(range(dact.ndim - 1)))
    return _apply_quantizer(dact, quantizer), dbias


def dbias_dsilu(grad, fwd_input, quantizer):
    """Fused DSiLU + DBias: returns (dact, dbias)."""
    sig = torch.sigmoid(fwd_input)
    dact = grad * sig * (1.0 + fwd_input * (1.0 - sig))
    dbias = dact.sum(dim=tuple(range(dact.ndim - 1)))
    return _apply_quantizer(dact, quantizer), dbias


def dbias_drelu(grad, fwd_input, quantizer):
    """Fused DReLU + DBias: returns (dact, dbias)."""
    dact = grad * (fwd_input > 0).to(grad.dtype)
    dbias = dact.sum(dim=tuple(range(dact.ndim - 1)))
    return _apply_quantizer(dact, quantizer), dbias


def dbias_dqgelu(grad, fwd_input, quantizer):
    """Fused DQGeLU + DBias: returns (dact, dbias)."""
    sig = torch.sigmoid(1.702 * fwd_input)
    dact = grad * sig * (1.0 + 1.702 * fwd_input * (1.0 - sig))
    dbias = dact.sum(dim=tuple(range(dact.ndim - 1)))
    return _apply_quantizer(dact, quantizer), dbias


def dbias_dsrelu(grad, fwd_input, quantizer):
    """Fused DSquaredReLU + DBias: returns (dact, dbias)."""
    dact = grad * 2.0 * F.relu(fwd_input)
    dbias = dact.sum(dim=tuple(range(dact.ndim - 1)))
    return _apply_quantizer(dact, quantizer), dbias
