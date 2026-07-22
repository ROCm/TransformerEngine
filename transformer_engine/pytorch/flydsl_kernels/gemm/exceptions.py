# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

class FlyDSLUnsupportedError(RuntimeError):
    """The GEMM request is valid but unsupported by the available FlyDSL kernels."""