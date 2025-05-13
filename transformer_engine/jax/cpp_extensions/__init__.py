# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Python interface for c++ extensions"""
from .activation import *
from .attention import *
from .normalization import *
from .quantization import *
from .softmax import *
from .transpose import *
from .gemm import *
from .misc import *
from .custom_call import *


for _name, _value in transformer_engine_jax.registrations().items():
    if _name.endswith("_ffi"):
        if is_ffi_enabled():
            jax.ffi.register_ffi_target(
                _name, _value, platform="ROCM" if is_hip_extension() else "CUDA", api_version=CustomCallAPIVersion.FFI.value
            )
    else:
        jax.ffi.register_ffi_target(
            _name, _value, platform="ROCM" if is_hip_extension() else "CUDA", api_version=CustomCallAPIVersion.OPAQUE.value
        )