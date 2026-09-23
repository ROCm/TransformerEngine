# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""FlyDSL kernel backends for Transformer Engine (ROCm, gfx950).

Subpackages are not imported here: ``gemm`` enforces the supported FlyDSL
version at import time, and ``attention`` defers its FlyDSL import to first use.
Import them explicitly from call sites that have confirmed FlyDSL availability.
"""

from .exceptions import FlyDSLUnsupportedError

__all__ = ["FlyDSLUnsupportedError"]
