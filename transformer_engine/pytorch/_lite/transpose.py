# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Transpose operations -- PyTorch-native implementation."""

import torch


def fp8_transpose(input, dtype, *, out):
    """Transpose a 2D tensor. dtype is ignored since we work with PyTorch tensors directly."""
    result = input.t().contiguous()
    out.copy_(result)
    return out


def swap_first_dims(tensor, *, out):
    """Swap first two dimensions of a tensor."""
    result = tensor.transpose(0, 1).contiguous()
    out.copy_(result)
    return out
