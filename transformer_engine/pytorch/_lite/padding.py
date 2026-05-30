# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Multi-row padding / unpadding -- tex-compatible interface.

Uses PyTorch-native operations.  The existing Triton ``zero_pad_kernel``
in ``common/triton/pad.py`` is purpose-built for 2-D columnwise-scale
alignment padding and does not apply to the multi-row copy-with-padding
pattern needed here.
"""

import torch


def fused_multi_row_padding(input, output, input_row_list, padded_input_row_list):
    """Copy rows from *input* into *output*, zero-padding the extra rows.

    Matches ``tex.fused_multi_row_padding(input, output, src_splits, dst_splits)``.

    Parameters
    ----------
    input : torch.Tensor
        Source tensor of shape ``[sum(input_row_list), features]``.
    output : torch.Tensor
        Pre-allocated destination of shape ``[sum(padded_input_row_list), features]``.
    input_row_list : list[int]
        Number of rows per group in the source tensor.
    padded_input_row_list : list[int]
        Number of rows per group in the destination tensor (≥ corresponding
        entry in *input_row_list*).
    """
    in_offset = 0
    out_offset = 0
    for src_rows, dst_rows in zip(input_row_list, padded_input_row_list):
        if src_rows > 0:
            output[out_offset:out_offset + src_rows].copy_(
                input[in_offset:in_offset + src_rows],
            )
        if dst_rows > src_rows:
            output[out_offset + src_rows:out_offset + dst_rows].zero_()
        in_offset += src_rows
        out_offset += dst_rows


def fused_multi_row_unpadding(input, output, input_row_list, unpadded_input_row_list):
    """Extract unpadded rows from a padded tensor.

    Matches ``tex.fused_multi_row_unpadding(input, output, src_splits, dst_splits)``.

    Parameters
    ----------
    input : torch.Tensor
        Padded source tensor of shape ``[sum(input_row_list), features]``.
    output : torch.Tensor
        Pre-allocated destination of shape ``[sum(unpadded_input_row_list), features]``.
    input_row_list : list[int]
        Number of rows per group in the padded source tensor.
    unpadded_input_row_list : list[int]
        Number of rows per group to extract (≤ corresponding entry in
        *input_row_list*).
    """
    in_offset = 0
    out_offset = 0
    for src_rows, dst_rows in zip(input_row_list, unpadded_input_row_list):
        if dst_rows > 0:
            output[out_offset:out_offset + dst_rows].copy_(
                input[in_offset:in_offset + dst_rows],
            )
        in_offset += src_rows
        out_offset += dst_rows
