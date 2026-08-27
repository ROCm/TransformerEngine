# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Triton GEMM kernels (dense, non-grouped) for BF16/FP16/FP32/FP8/MXFP8."""

from .gemm_wrapper import te_gemm_triton, te_generic_gemm_triton, matmul, mxfp8_matmul
from .gemm_common import (
    is_fp8_dtype,
    reinterpret_as_fp8_tensor,
    getGemmOutputShape,
    product,
)

# Dtype conversions (torch <-> tex.DType, architecture-native FP8 dtypes) are
# NOT re-exported here -- import them directly from
# ``transformer_engine.pytorch.triton_kernels.common``, which is the
# authoritative source shared across all Triton kernel backends.

__all__ = [
    "te_gemm_triton",
    "te_generic_gemm_triton",
    "matmul",
    "mxfp8_matmul",
    "is_fp8_dtype",
    "reinterpret_as_fp8_tensor",
    "getGemmOutputShape",
    "product",
]
