# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Transpose operations -- PyTorch-native implementation."""

import torch


def fp8_transpose(input, dtype, *, out=None):
    """FP8 transpose: move the last dim to the front.

    For a 2D tensor [M, K], this is equivalent to .t() → [K, M].
    For an N-D tensor [d0, d1, ..., K], produces [K, d0, d1, ...] — matching
    the transpose_shape convention in the quantizer's make_empty().
    dtype is ignored since we work with PyTorch tensors directly.
    """
    if input.ndim == 2:
        result = input.t().contiguous()
    else:
        # Permute last axis to front: [..., K] -> [K, ...]
        perm = [input.ndim - 1] + list(range(input.ndim - 1))
        result = input.permute(*perm).contiguous()
    if out is None:
        return result
    out.copy_(result.reshape(out.shape) if result.shape != out.shape else result)
    return out


def swap_first_dims(tensor, *, out):
    """Swap first two dimensions of a tensor."""
    result = tensor.transpose(0, 1).contiguous()
    out.copy_(result)
    return out
