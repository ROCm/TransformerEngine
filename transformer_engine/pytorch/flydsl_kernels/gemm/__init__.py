# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""FlyDSL GEMM kernels (dense, non-grouped) for BF16/FP16/FP32/FP8/MXFP8."""

from .gemm_wrappers import (
    te_generic_gemm_flydsl,
)

__all__ = [
    "te_generic_gemm_flydsl",
]