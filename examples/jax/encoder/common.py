# This file was modified for portability to AMDGPU
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# See LICENSE for license information.
"""Shared functions for the encoder tests"""
from functools import lru_cache

from transformer_engine_jax import get_device_compute_capability
from transformer_engine.jax import is_hip_extension
if is_hip_extension():
    from transformer_engine.jax.util import is_mi200

@lru_cache
def is_bf16_supported():
    """Return if BF16 has hardware supported"""
    gpu_arch = get_device_compute_capability(0)
    if is_hip_extension():
        # only GFX9.4 and MI200 machines support bf16
        return gpu_arch == 94 or is_mi200()
    return gpu_arch >= 80


@lru_cache
def is_fp8_supported():
    """Return if FP8 has hardware supported"""
    gpu_arch = get_device_compute_capability(0)
    if is_hip_extension():
        # only GFX9.4 machines support fp8
        return gpu_arch == 94
    return gpu_arch >= 90
