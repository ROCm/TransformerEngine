# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Miscellaneous utility functions."""


def get_num_cublas_streams():
    """Get number of compute streams. Returns default 1 in lite mode."""
    return 1
