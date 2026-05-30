# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Context parallel THD format helpers -- PyTorch-native tensor ops.

TODO Phase 3: Implement as PyTorch tensor slicing operations.
"""


def thd_read_half_tensor(*args, **kwargs):
    """Read first or second half of each sequence in a THD tensor."""
    raise NotImplementedError("thd_read_half_tensor not yet implemented in lite mode.")


def thd_second_half_lse_correction(*args, **kwargs):
    """Correct the second half of softmax_lse."""
    raise NotImplementedError("thd_second_half_lse_correction not yet implemented in lite mode.")


def thd_read_second_half_lse(*args, **kwargs):
    """Read the second half of softmax_lse."""
    raise NotImplementedError("thd_read_second_half_lse not yet implemented in lite mode.")


def thd_out_correction(*args, **kwargs):
    """Correct THD format output of context parallelism in forward pass."""
    raise NotImplementedError("thd_out_correction not yet implemented in lite mode.")


def thd_grad_correction(*args, **kwargs):
    """Correct THD format gradients of context parallelism in backward pass."""
    raise NotImplementedError("thd_grad_correction not yet implemented in lite mode.")


def thd_get_partitioned_indices(*args, **kwargs):
    """Generate partitioned indices for inputs in THD format."""
    raise NotImplementedError("thd_get_partitioned_indices not yet implemented in lite mode.")
