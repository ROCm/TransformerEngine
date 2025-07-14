# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.# 
#
# License for AMD contributions = MIT. See LICENSE for more information

import torch
import transformer_engine_torch as tex

from .tensor._internal.float8_tensor_base import Float8TensorBase
from .utils import clear_tensor_data


__all__ = [
    "create_fp8_weight_transpose_cache",
    "clear_fp8_weight_transpose_cache",
]


def create_fp8_weight_transpose_cache(weight: Float8TensorBase):
    assert isinstance(weight, Float8TensorBase), "weight should be `Float8TensorBase`."
    if not weight._transpose_invalid:
        # transpose buffer is existance.
        return

    data = weight._data
    if not data.is_contiguous():
        data = data.contiguous()
    weight._transpose = torch.empty_like(data, dtype=torch.uint8)
    tex.fp8_transpose(data, weight._fp8_dtype, out=weight._transpose)
    weight._transpose_invalid = False


def clear_fp8_weight_transpose_cache(weight: Float8TensorBase):
    assert isinstance(weight, Float8TensorBase), "weight should be `Float8TensorBase`."
    if weight._transpose_invalid:
        # transpose buffer is not existance.
        return

    clear_tensor_data(weight._transpose) 
    weight._transpose_invalid = True