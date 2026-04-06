# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fused softmax variants -- PyTorch-native implementation.

Note: These are rarely hit in lite mode since SDPA/AITER handle softmax internally.
"""

import torch
import torch.nn.functional as F


def _softmax_backward(grad_output, output):
    """Common softmax backward: output * (grad - (output * grad).sum(dim=-1, keepdim=True))."""
    dot = (output * grad_output).sum(dim=-1, keepdim=True)
    return output * (grad_output - dot)


def scaled_softmax_forward(input, scale_factor):
    """Scaled softmax forward: softmax(input * scale)."""
    return torch.softmax(input * scale_factor, dim=-1)


def scaled_softmax_backward(grad_output, output, scale_factor):
    """Scaled softmax backward."""
    grad_input = _softmax_backward(grad_output, output)
    return grad_input * scale_factor


def scaled_masked_softmax_forward(input, mask, scale_factor):
    """Scaled masked softmax forward."""
    scaled = input * scale_factor
    if mask is not None:
        scaled = scaled.masked_fill(mask, float('-inf'))
    return torch.softmax(scaled, dim=-1)


def scaled_masked_softmax_backward(grad_output, output, scale_factor):
    """Scaled masked softmax backward."""
    grad_input = _softmax_backward(grad_output, output)
    return grad_input * scale_factor


def scaled_upper_triang_masked_softmax_forward(input, scale_factor):
    """Scaled upper-triangular masked softmax forward (causal mask)."""
    seq_len = input.size(-1)
    mask = torch.triu(torch.ones(seq_len, seq_len, device=input.device, dtype=torch.bool), diagonal=1)
    scaled = input * scale_factor
    scaled = scaled.masked_fill(mask, float('-inf'))
    return torch.softmax(scaled, dim=-1)


def scaled_upper_triang_masked_softmax_backward(grad_output, output, scale_factor):
    """Scaled upper-triangular masked softmax backward."""
    grad_input = _softmax_backward(grad_output, output)
    return grad_input * scale_factor


def scaled_aligned_causal_masked_softmax_forward(input, scale_factor):
    """Scaled bottom-right corner aligned causal masked softmax forward."""
    q_len = input.size(-2)
    k_len = input.size(-1)
    # Bottom-right aligned causal mask: position i can attend to positions <= i + (k_len - q_len)
    row_idx = torch.arange(q_len, device=input.device).unsqueeze(1)
    col_idx = torch.arange(k_len, device=input.device).unsqueeze(0)
    offset = k_len - q_len
    mask = col_idx > (row_idx + offset)
    scaled = input * scale_factor
    scaled = scaled.masked_fill(mask, float('-inf'))
    return torch.softmax(scaled, dim=-1)


def scaled_aligned_causal_masked_softmax_backward(grad_output, output, scale_factor):
    """Scaled aligned causal masked softmax backward."""
    grad_input = _softmax_backward(grad_output, output)
    return grad_input * scale_factor
