# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Dropout -- PyTorch-native implementation."""

import torch
import torch.nn.functional as F


def dropout_fwd(input, dropout_probability, out=None):
    """Dropout forward.

    Returns (output, mask) tuple. The C++ version uses 8-bit RNG masks;
    here we use standard PyTorch boolean masks.
    """
    if dropout_probability == 0.0:
        mask = torch.ones_like(input, dtype=torch.uint8)
        if out is not None:
            out.copy_(input)
            return out, mask
        return input.clone(), mask

    keep_prob = 1.0 - dropout_probability
    mask = (torch.rand_like(input) < keep_prob).to(torch.uint8)
    output = input * mask.to(input.dtype) / keep_prob

    if out is not None:
        out.copy_(output)
        return out, mask
    return output, mask


def dropout_bwd(grad_output, mask, dropout_probability, grad_input=None):
    """Dropout backward."""
    if dropout_probability == 0.0:
        if grad_input is not None:
            grad_input.copy_(grad_output)
            return grad_input
        return grad_output.clone()

    keep_prob = 1.0 - dropout_probability
    output = grad_output * mask.to(grad_output.dtype) / keep_prob

    if grad_input is not None:
        grad_input.copy_(output)
        return grad_input
    return output
