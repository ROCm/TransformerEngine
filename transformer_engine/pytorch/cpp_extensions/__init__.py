# This file was modified for portability to AMDGPU
# Copyright (c) 2022-2024, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Python interface for c++ extensions"""
from transformer_engine_torch import *

from torch.utils.cpp_extension import IS_HIP_EXTENSION
if not IS_HIP_EXTENSION:
  from .fused_attn import *
from .gemm import *
from .transpose import *
from .activation import *
from .normalization import *
from .cast import *
from .padding import *
