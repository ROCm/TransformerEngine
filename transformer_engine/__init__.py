# This file was modified for portability to AMDGPU
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Top level package"""

# pylint: disable=unused-import

import os
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
except ImportError:
    pass
except FileNotFoundError as e:
    if "Could not find shared object file" not in str(e):
        raise e  # Unexpected error
    else:
        if os.getenv("NVTE_FRAMEWORK"):
            frameworks = os.getenv("NVTE_FRAMEWORK").split(",")
            if "pytorch" in frameworks or "all" in frameworks:
                raise e
        else:
            # If we got here, we could import `torch` but could not load the framework extension.
            # This can happen when a user wants to work only with `transformer_engine.jax` on a system that
            # also has a PyTorch installation. In order to enable that use case, we issue a warning here
            # about the missing PyTorch extension in case the user hasn't set NVTE_FRAMEWORK.
            import warnings

            warnings.warn(
                "Detected a PyTorch installation but could not find the shared object file for the "
                "Transformer Engine PyTorch extension library. If this is not intentional, please "
                "reinstall Transformer Engine with `pip install transformer_engine[pytorch]` or "
                "build from source with `NVTE_FRAMEWORK=pytorch`.",
                category=RuntimeWarning,
            )

try:
    if _use_jax: from . import jax
except ImportError:
    pass
except FileNotFoundError as e:
    if "Could not find shared object file" not in str(e):
        raise e  # Unexpected error
    else:
        if os.getenv("NVTE_FRAMEWORK"):
            frameworks = os.getenv("NVTE_FRAMEWORK").split(",")
            if "jax" in frameworks or "all" in frameworks:
                raise e
        else:
            # If we got here, we could import `jax` but could not load the framework extension.
            # This can happen when a user wants to work only with `transformer_engine.pytorch` on a system
            # that also has a Jax installation. In order to enable that use case, we issue a warning here
            # about the missing Jax extension in case the user hasn't set NVTE_FRAMEWORK.
            import warnings

            warnings.warn(
                "Detected a Jax installation but could not find the shared object file for the "
                "Transformer Engine Jax extension library. If this is not intentional, please "
                "reinstall Transformer Engine with `pip install transformer_engine[jax]` or "
                "build from source with `NVTE_FRAMEWORK=jax`.",
                category=RuntimeWarning,
            )

# Set AITER_ASM_DIR to point to the asm dir in the installed package
local_asm_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "3rdparty", "aiter", "hsa",
)
if "AITER_ASM_DIR" not in os.environ:
    os.environ["AITER_ASM_DIR"] = local_asm_dir
__version__ = str(metadata.version("transformer_engine"))
