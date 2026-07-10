# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Triton GEMM kernels (dense, non-grouped) for BF16/FP16/FP32/FP8/MXFP8."""

from .gemm_wrapper import te_gemm_triton, te_generic_gemm_triton, matmul, mxfp8_matmul
from .gemm_common import (
    Float8TensorWrapper,
    MXFP8TensorWrapper,
    torch_to_te_dtype,
    te_to_torch_dtype,
    is_fp8_dtype,
    _get_fp8_dtypes,
    reinterpret_as_fp8_tensor,
    getGemmOutputShape,
    product,
)

__all__ = [
    "te_gemm_triton",
    "te_generic_gemm_triton",
    "matmul",
    "mxfp8_matmul",
    "Float8TensorWrapper",
    "MXFP8TensorWrapper",
    "torch_to_te_dtype",
    "te_to_torch_dtype",
    "is_fp8_dtype",
    "_get_fp8_dtypes",
    "reinterpret_as_fp8_tensor",
    "getGemmOutputShape",
    "product",
]
