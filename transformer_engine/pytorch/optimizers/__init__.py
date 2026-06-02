# This file was modified for portability to AMDGPU
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fused optimizers and multi-tensor kernels."""
import torch

from transformer_engine_torch import (
    multi_tensor_scale as _multi_tensor_scale,
    multi_tensor_scale_tensor,
    multi_tensor_l2norm as _multi_tensor_l2norm,
    multi_tensor_unscale_l2norm as _multi_tensor_unscale_l2norm,
    multi_tensor_adam,
    multi_tensor_adam_fp8,
    multi_tensor_adam_capturable,
    multi_tensor_adam_capturable_master,
    multi_tensor_sgd,
)
from .fused_adam import FusedAdam
from .fused_sgd import FusedSGD
from .multi_tensor_apply import MultiTensorApply, multi_tensor_applier


def _is_single_tensor_list(tensor_lists: list[list[torch.Tensor]], expected_lists: int) -> bool:
    return len(tensor_lists) == expected_lists and all(len(tensor_list) == 1 for tensor_list in tensor_lists)


def _update_noop_flag(noop_flag_buffer: torch.Tensor | None, condition: torch.Tensor) -> None:
    if noop_flag_buffer is None:
        return
    noop_flag_buffer.add_(condition.to(dtype=noop_flag_buffer.dtype, device=noop_flag_buffer.device))


def multi_tensor_scale(chunk_size, noop_flag_buffer, tensor_lists, scale):
    if _is_single_tensor_list(tensor_lists, 2) and (
        noop_flag_buffer is None or noop_flag_buffer.numel() == 0
    ):
        input_tensor = tensor_lists[0][0]
        output_tensor = tensor_lists[1][0]
        torch.mul(input_tensor, scale, out=output_tensor)
        return None
    return _multi_tensor_scale(chunk_size, noop_flag_buffer, tensor_lists, scale)


def multi_tensor_l2norm(chunk_size, noop_flag_buffer, tensor_lists, per_tensor):
    if _is_single_tensor_list(tensor_lists, 1):
        input_tensor = tensor_lists[0][0]
        norm = torch.empty((), device=input_tensor.device, dtype=torch.float32)
        torch.linalg.vector_norm(input_tensor, ord=2, dtype=torch.float32, out=norm)
        _update_noop_flag(noop_flag_buffer, ~torch.isfinite(norm))
        norm = norm.reshape(1)
        per_tensor_norm = norm if per_tensor else torch.empty(0, device=input_tensor.device, dtype=torch.float32)
        return norm, per_tensor_norm
    return _multi_tensor_l2norm(chunk_size, noop_flag_buffer, tensor_lists, per_tensor)


def multi_tensor_unscale_l2norm(chunk_size, noop_flag_buffer, tensor_lists, inv_scale, per_tensor):
    if _is_single_tensor_list(tensor_lists, 1):
        input_tensor = tensor_lists[0][0]
        scaled_norm = torch.empty((), device=input_tensor.device, dtype=torch.float32)
        torch.linalg.vector_norm(input_tensor, ord=2, dtype=torch.float32, out=scaled_norm)
        scaled_norm.mul_(torch.abs(inv_scale.reshape(())).to(dtype=torch.float32))
        _update_noop_flag(noop_flag_buffer, ~torch.isfinite(scaled_norm))
        scaled_norm = scaled_norm.reshape(1)
        per_tensor_norm = (
            scaled_norm
            if per_tensor
            else torch.empty(0, device=input_tensor.device, dtype=torch.float32)
        )
        return scaled_norm, per_tensor_norm
    return _multi_tensor_unscale_l2norm(chunk_size, noop_flag_buffer, tensor_lists, inv_scale, per_tensor)
