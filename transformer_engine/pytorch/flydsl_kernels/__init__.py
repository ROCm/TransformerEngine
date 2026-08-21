# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""FlyDSL kernel backends for Transformer Engine (ROCm, gfx950).

Importing this package pulls in FlyDSL via ``gemm``, so it must only be imported
lazily from a call site that has already confirmed FlyDSL availability -- see
``transformer_engine/pytorch/cpp_extensions/gemm.py``.
"""

from . import attention
from . import gemm
from .exceptions import FlyDSLUnsupportedError

__all__ = ["FlyDSLUnsupportedError", "attention", "gemm"]
