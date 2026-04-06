# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""MOE permutation operations.

TODO Phase 1: Wire up to existing triton/permutation.py.
For now, uses PyTorch-native implementations.
"""

import torch


def moe_permute_fwd(input, indices, num_out_tokens=None, padded_mode=False):
    """MOE permute forward: gather rows according to indices."""
    if indices.ndim == 2:
        # Flatten indices for gathering
        flat_indices = indices.view(-1)
    else:
        flat_indices = indices

    if num_out_tokens is not None:
        flat_indices = flat_indices[:num_out_tokens]

    output = input[flat_indices]
    return output


def moe_permute_bwd(grad_output, indices, num_tokens, padded_mode=False):
    """MOE permute backward: scatter-add gradients back."""
    if indices.ndim == 2:
        flat_indices = indices.view(-1)
    else:
        flat_indices = indices

    grad_input = torch.zeros(num_tokens, grad_output.shape[-1],
                             device=grad_output.device, dtype=grad_output.dtype)
    flat_indices = flat_indices[:grad_output.shape[0]]
    grad_input.index_add_(0, flat_indices, grad_output)
    return grad_input


def moe_unpermute_fwd(input, indices, probs=None, padded_mode=False):
    """MOE unpermute forward: reverse the permutation."""
    if indices.ndim == 2:
        flat_indices = indices.view(-1)
    else:
        flat_indices = indices

    num_tokens = flat_indices.max().item() + 1
    output = torch.zeros(num_tokens, input.shape[-1],
                         device=input.device, dtype=input.dtype)

    if probs is not None:
        # Weight by routing probabilities
        weighted = input * probs.view(-1, 1)[:input.shape[0]]
        output.index_add_(0, flat_indices[:input.shape[0]], weighted)
    else:
        output.index_add_(0, flat_indices[:input.shape[0]], input)

    return output


def moe_unpermute_bwd(grad_output, indices, probs=None, padded_mode=False):
    """MOE unpermute backward."""
    if indices.ndim == 2:
        flat_indices = indices.view(-1)
    else:
        flat_indices = indices

    grad_input = grad_output[flat_indices]

    if probs is not None:
        grad_input = grad_input * probs.view(-1, 1)[:grad_input.shape[0]]

    return grad_input
