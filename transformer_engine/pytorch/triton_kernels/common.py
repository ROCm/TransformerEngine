# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

import os
import torch
import triton
import triton.language as tl
from transformer_engine import pytorch as te
import transformer_engine_torch as tex
from ..utils import is_fp8_fnuz

def is_cdna4():
    return triton.runtime.driver.active.get_current_target().arch == "gfx950"

torch_e4m3_type = torch.float8_e4m3fn if is_cdna4() else torch.float8_e4m3fnuz
torch_e5m2_type = torch.float8_e5m2 if is_cdna4() else torch.float8_e5m2fnuz

# Convert te dtype to torch type.
def te_dtype_to_torch_dtype(te_dtype):
    return {
        tex.DType.kByte: torch.uint8, 
        tex.DType.kInt32: torch.int32,
        tex.DType.kFloat32: torch.float32, 
        tex.DType.kFloat16: torch.float16, 
        tex.DType.kBFloat16: torch.bfloat16, 
        tex.DType.kFloat8E4M3: torch_e4m3_type, 
        tex.DType.kFloat8E5M2: torch_e5m2_type, 
    }[te_dtype]

# Convert PyTorch type to TE type.
def torch_dtype_to_te_dtype(dtype):
    return {
        torch.float32: tex.DType.kFloat32,
        torch.float16: tex.DType.kFloat16,
        torch.bfloat16: tex.DType.kBFloat16,
        torch.float8_e4m3fnuz: tex.DType.kFloat8E4M3,
        torch.float8_e4m3fn: tex.DType.kFloat8E4M3,
        torch.float8_e5m2fnuz: tex.DType.kFloat8E5M2,
        torch.float8_e5m2: tex.DType.kFloat8E5M2,
    }[dtype]

# Convert te dtype to pytorch aten type.
# used in ln_out allocation
def te_dtype_to_aten_dtype(te_dtype):
    return {
        tex.DType.kInt32: torch.int32,
        tex.DType.kFloat32: torch.float32, 
        tex.DType.kFloat16: torch.float16, 
        tex.DType.kBFloat16: torch.bfloat16, 
        tex.DType.kByte: torch.uint8, 
        tex.DType.kFloat8E4M3: torch.uint8, 
        tex.DType.kFloat8E5M2: torch.uint8, 
    }[te_dtype]

# convert te_dtype to their enum value in class DType
def te_dtype_to_enum_value(te_dtype):
    return {
      tex.DType.kByte: 0, 
      tex.DType.kInt32: 1,
      #kInt64 not exported to transformer_engine_pytorch
      #tex.DType.kInt64: 2,
      tex.DType.kFloat32: 3,
      tex.DType.kFloat16: 4,
      tex.DType.kBFloat16: 5,
      tex.DType.kFloat8E4M3: 6,
      tex.DType.kFloat8E5M2: 7,
    }[te_dtype]

# convert te_dtype to their enum value in class DType
def enum_value_to_te_dtype(te_dtype_enum):
    return {
      0: tex.DType.kByte, 
      1: tex.DType.kInt32,
      #kInt64 not exported to transformer_engine_pytorch
      #2: tex.DType.kInt64,
      3: tex.DType.kFloat32,
      4: tex.DType.kFloat16,
      5: tex.DType.kBFloat16,
      6: tex.DType.kFloat8E4M3,
      7: tex.DType.kFloat8E5M2,
    }[te_dtype_enum]

def is_fp8_torch_dtype(dtype):
    return (dtype == torch_e4m3_type) or (dtype == torch_e5m2_type)

def te_dtype_to_triton_dtype(dtype: tex.DType):
    if dtype == tex.DType.kFloat8E4M3:
        return tl.float8e4b8 if is_fp8_fnuz() else tl.float8e4nv
    if dtype == tex.DType.kFloat8E5M2:
        return tl.float8e5b16 if is_fp8_fnuz() else tl.float8e5

def get_fp8_max(dtype: tex.DType):
    if dtype == tex.DType.kFloat8E4M3:
        return 240.0 if is_fp8_fnuz() else 448.0
    if dtype == tex.DType.kFloat8E5M2:
        return 57344.0

