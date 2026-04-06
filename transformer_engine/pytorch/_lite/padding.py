# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Padding operations -- PyTorch-native implementation.

TODO Phase 1: Wire up to existing triton/pad.py.
"""

import torch
import torch.nn.functional as F


def fused_multi_row_padding(input, padded_sizes, padded_output):
    """Pad multiple rows to specified sizes.

    input: concatenated rows
    padded_sizes: target size for each row
    padded_output: pre-allocated output tensor
    """
    # Simple implementation: pad each row to target size
    offset = 0
    out_offset = 0
    for size in padded_sizes:
        row = input[offset:offset + size]
        padded_output[out_offset:out_offset + size].copy_(row)
        offset += size
        out_offset += size


def fused_multi_row_unpadding(padded_input, original_sizes, output):
    """Remove padding from multiple rows.

    padded_input: padded concatenated rows
    original_sizes: original size for each row
    output: pre-allocated output tensor
    """
    offset = 0
    out_offset = 0
    for size in original_sizes:
        output[out_offset:out_offset + size].copy_(padded_input[offset:offset + size])
        offset += size
        out_offset += size
