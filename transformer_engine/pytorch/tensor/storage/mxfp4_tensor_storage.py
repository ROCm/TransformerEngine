# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Shim (plugin plan S5.2): module moved to transformer_engine.te_rocm.tensors.storage.mxfp4_tensor_storage.

Kept for import compatibility and pickle GLOBAL resolution - checkpoints written on
either side of the move resolve through this path (same class objects, __module__
pinned here). The MXFP8 rename shim is the precedent.
"""
from transformer_engine.te_rocm.tensors.storage.mxfp4_tensor_storage import *  # noqa: F401,F403
from transformer_engine.te_rocm.tensors.storage.mxfp4_tensor_storage import (  # noqa: F401
    MXFP4TensorStorage,
    _FromMXFP4Func,
)
