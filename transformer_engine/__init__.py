# This file was modified for portability to AMDGPU
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Top level package"""

# pylint: disable=unused-import

from importlib import metadata
import os
import transformer_engine.common

_use_pytorch = True
_use_jax = True

if os.getenv("NVTE_FRAMEWORK"):
    _frameworks=os.getenv("NVTE_FRAMEWORK").split(",")

    # Special framework names
    if "none" in _frameworks:
        _use_pytorch = False
        _use_jax = False
    elif "all" in _frameworks:
        pass
    else:
        _use_pytorch = "pytorch" in _frameworks
        _use_jax = "jax" in _frameworks

try:
    if _use_pytorch: from . import pytorch
except (ImportError, FileNotFoundError):
    pass

try:
    if _use_jax: from . import jax
except (ImportError, FileNotFoundError):
    pass

try:
    if _use_jax: import transformer_engine_jax
except ImportError:
    pass

__version__ = str(metadata.version("transformer_engine"))
