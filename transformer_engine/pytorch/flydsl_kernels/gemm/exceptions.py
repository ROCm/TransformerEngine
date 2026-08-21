# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Re-export of the shared FlyDSL exception type.

The type was promoted to ``flydsl_kernels.exceptions`` when a second backend
(attention) needed it; this module is kept so existing GEMM imports still work.
"""

from ..exceptions import FlyDSLUnsupportedError

__all__ = ["FlyDSLUnsupportedError"]
