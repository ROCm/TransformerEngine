# This file was modified for portability to AMDGPU
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Enums for e2e transformer"""
import torch
import torch.distributed
import transformer_engine_torch as tex
from .utils import get_torch_float8_e4m3_type, get_torch_float8_e5m2_type

"""
This is a map: torch.dtype -> int
Used for passing dtypes into cuda
extension. Has one to one mapping
with enum in transformer_engine.h
"""
TE_DType = {
    torch.uint8: tex.DType.kByte,
    torch.float8_e4m3fn: tex.DType.kFloat8E4M3,
    torch.float8_e5m2: tex.DType.kFloat8E5M2,
    torch.int32: tex.DType.kInt32,
    torch.float32: tex.DType.kFloat32,
    torch.half: tex.DType.kFloat16,
    torch.bfloat16: tex.DType.kBFloat16,
}
from torch.utils.cpp_extension import IS_HIP_EXTENSION
if IS_HIP_EXTENSION:
    TE_DType.update({torch.float8_e4m3fnuz: tex.DType.kFloat8E4M3, 
                     torch.float8_e5m2fnuz: tex.DType.kFloat8E5M2})

_FP8_KEYS = (tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2)

class Custom_DType_Dict(dict):
    def __missing__(self, key):
        if key in _FP8_KEYS:
            value = (
                get_torch_float8_e4m3_type() if key is tex.DType.kFloat8E4M3
                else get_torch_float8_e5m2_type()
            )
            self[key] = value 
            return value
        raise KeyError(key)
    
TE_DType_To_Torch = Custom_DType_Dict({
    tex.DType.kByte: torch.uint8,
    tex.DType.kInt32: torch.int32,
    tex.DType.kFloat32: torch.float32,
    tex.DType.kFloat16: torch.half,
    tex.DType.kBFloat16: torch.bfloat16,
})

AttnMaskTypes = (
    "no_mask",
    "padding",
    "causal",
    "padding_causal",
    "causal_bottom_right",
    "padding_causal_bottom_right",
    "arbitrary",
)

AttnTypes = ("self", "cross")

AttnBiasTypes = ("pre_scale_bias", "post_scale_bias", "no_bias", "alibi")

QKVLayouts = (
    "sb3hd",
    "sbh3d",
    "sbhd_sb2hd",
    "sbhd_sbh2d",
    "sbhd_sbhd_sbhd",
    "bs3hd",
    "bsh3d",
    "bshd_bs2hd",
    "bshd_bsh2d",
    "bshd_bshd_bshd",
    "t3hd",
    "th3d",
    "thd_t2hd",
    "thd_th2d",
    "thd_thd_thd",
    "sbhd_bshd_bshd",
    "bshd_sbhd_sbhd",
    "thd_bshd_bshd",
    "thd_sbhd_sbhd",
    "paged_kv_bshd_bshd_bshd",
    "paged_kv_bshd_sbhd_sbhd",
    "paged_kv_sbhd_bshd_bshd",
    "paged_kv_sbhd_sbhd_sbhd",
    "paged_kv_thd_bshd_bshd",
    "paged_kv_thd_sbhd_sbhd",
)

LayerTypes = ("encoder", "decoder")

GemmParallelModes = ("row", "column", None)

dist_group_type = torch.distributed.ProcessGroup

MXFP8_BLOCK_SCALING_SIZE = 32
