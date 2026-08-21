# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Exception types shared by the FlyDSL kernel backends."""


class FlyDSLUnsupportedError(RuntimeError):
    """The request is valid but unsupported by the available FlyDSL kernels."""
