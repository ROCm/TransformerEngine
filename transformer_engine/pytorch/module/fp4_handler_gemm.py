# Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""FP4 GEMM handler using AITER ASM a4w4 kernels.

Kernel selection and split-K tuning are handled by AITER internally
via CSV-based GEMM config lookup (see aiter.ops.gemm_op_a4w4).
"""

import torch
import aiter
from aiter.ops.shuffle import shuffle_weight
from ..utils import cast_if_needed


def _fp4_gemm_core(A_fp4, A_scales, B_fp4, B_scales, out_dtype=torch.bfloat16,
                    out_buffer=None, b_pre_shuffled=True):
    """Core FP4 GEMM via AITER ASM a4w4 kernel."""
    _fp4_dtype = torch.float4_e2m1fn_x2
    A_fp4 = A_fp4.view(_fp4_dtype) if A_fp4.dtype != _fp4_dtype else A_fp4
    B_fp4 = B_fp4.view(_fp4_dtype) if B_fp4.dtype != _fp4_dtype else B_fp4
    A_scales_uint8 = A_scales.view(torch.uint8)
    B_scales_uint8 = B_scales.view(torch.uint8)

    B_shuffled = B_fp4 if b_pre_shuffled else shuffle_weight(B_fp4, layout=(16, 16))

    M = A_fp4.shape[0]
    N = B_fp4.shape[0]

    if out_buffer is not None:
        out_hp = out_buffer
    else:
        padded_M = (M + 31) // 32 * 32
        out_hp = torch.empty((padded_M, N), dtype=out_dtype, device=A_fp4.device)

    result = aiter.gemm_a4w4_asm(
        A_fp4, B_shuffled, A_scales_uint8, B_scales_uint8,
        out_hp, "", None,
        bpreshuffle=True,
    )

    return result[:M, :] if result.shape[0] > M else result


def fp4_gemm_layout(
    A,
    B,
    layout: str = "TN",
    out_dtype: torch.dtype = torch.bfloat16,
    bias=None,
    out=None,
    grad: bool = False,
    accumulate: bool = False,
):
    """FP4 GEMM using layout notation (TN/NN/NT).

    Layout mapping:
        TN: A=weight, B=input       -> fprop: input @ weight^T
        NN: A=weight, B=grad_output -> dgrad: grad_output @ weight
        NT: A=input, B=grad_output  -> wgrad: grad_output^T @ input
    """
    with torch._C._DisableTorchDispatch():
        if layout == "TN":
            A_fp4 = B._rowwise_data
            A_scales = B._rowwise_scale_inv
            B_fp4 = A._rowwise_data
            B_scales = A._rowwise_scale_inv
            b_pre_shuffled = True
            gemm_M = B._rowwise_data.shape[0]
            gemm_N = A._rowwise_data.shape[0]
            gemm_K = A._rowwise_data.shape[1] * 2

        elif layout == "NN":
            A_fp4 = B._rowwise_data
            A_scales = B._rowwise_scale_inv
            B_fp4 = A._columnwise_data
            B_scales = A._columnwise_scale_inv
            b_pre_shuffled = True
            gemm_M = B._rowwise_data.shape[0]
            gemm_N = A._columnwise_data.shape[0]
            gemm_K = A._columnwise_data.shape[1] * 2

        elif layout == "NT":
            A_fp4 = B._columnwise_data
            A_scales = B._columnwise_scale_inv
            B_fp4 = A._columnwise_data
            B_scales = A._columnwise_scale_inv
            b_pre_shuffled = False
            gemm_M = B._columnwise_data.shape[0]
            gemm_N = A._columnwise_data.shape[0]
            gemm_K = A._columnwise_data.shape[1] * 2

        else:
            raise ValueError(f"Unsupported layout for FP4 GEMM: {layout}")

        if accumulate and out is not None:
            result = _fp4_gemm_core(
                A_fp4, A_scales, B_fp4, B_scales,
                out_dtype=out.dtype, out_buffer=None,
                b_pre_shuffled=b_pre_shuffled,
            )
            out.add_(result)
            result = None
        else:
            result = _fp4_gemm_core(
                A_fp4, A_scales, B_fp4, B_scales,
                out_dtype=out_dtype, out_buffer=out,
                b_pre_shuffled=b_pre_shuffled,
            )

        if bias is not None and layout == "TN" and not grad:
            bias_casted = cast_if_needed(bias, out_dtype)
            if result is not None:
                result = result + bias_casted

        return result
