# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""FlyDSL GEMM kernels (dense, non-grouped) for BF16/FP16/FP32/FP8/MXFP8."""

# A missing, too-old, or too-new flydsl raises ImportError here, which
# general_gemm catches to warn once and fall back to the default backend.
from ..version import _check_flydsl_version

_check_flydsl_version("GEMM")

from .exceptions import FlyDSLUnsupportedError
from .gemm_wrappers import te_generic_gemm_flydsl

__all__ = [
    "FlyDSLUnsupportedError",
    "te_generic_gemm_flydsl",
]
