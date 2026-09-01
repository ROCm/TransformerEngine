# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Shim package (plugin plan S5.2): triton_kernels moved to transformer_engine.te_rocm.triton_kernels.

Path redirect: this package keeps the old importable name but serves submodules from the new
directory, so `from ..triton_kernels.cast import ...` keeps working everywhere (these kernels
contribute no pickle GLOBALs - module identity is import-only).
"""
from transformer_engine.te_rocm import triton_kernels as _new

__path__ = _new.__path__  # submodule resolution -> new directory
from transformer_engine.te_rocm.triton_kernels import *  # noqa: F401,F403
