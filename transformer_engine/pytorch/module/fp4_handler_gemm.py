# Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""FP4 GEMM handler with shape-aware kernel tuning via AITER CSV lookup.

When NVTE_FP4_GEMM_TUNING=1 (default), selects the optimal AITER a4w4
kernel per (M, N, K) shape from the tuned CSV pointed to by
AITER_CONFIG_GEMM_A4W4. Falls back to layout-based defaults otherwise.
"""

import os
import torch
import aiter
from aiter.ops.shuffle import shuffle_weight
from aiter.ops.gemm_op_a4w4 import get_GEMM_config
from ..utils import cast_if_needed

_FP4_GEMM_TUNING = int(os.environ.get("NVTE_FP4_GEMM_TUNING", "1"))
_FP4_LOG_SHAPES = int(os.environ.get("NVTE_FP4_LOG_GEMM_SHAPES", "0"))

_DEFAULT_FPROP_DGRAD = "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_128x512E"
_DEFAULT_WGRAD = "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256E"


def _select_kernel(layout: str, grad: bool, M: int, N: int, K: int):
    """Select kernel via tuned CSV lookup, falling back to layout-based default."""
    kernel_name = _DEFAULT_WGRAD if (layout == "NT" and grad) else _DEFAULT_FPROP_DGRAD
    split_k = 0

    if _FP4_GEMM_TUNING:
        cfg = get_GEMM_config(M, N, K)
        if cfg is not None:
            kernel_name = cfg["kernelName"]
            split_k = int(cfg.get("splitK", 0))

    if _FP4_LOG_SHAPES:
        tag = "256x256" if "256x256" in kernel_name else "128x512"
        print(f"[FP4-GEMM] {layout} grad={grad} M={M} N={N} K={K} "
              f"kernel={tag} splitK={split_k}", flush=True)

    return kernel_name, split_k


def _fp4_gemm_core(A_fp4, A_scales, B_fp4, B_scales, out_dtype=torch.bfloat16,
                    out_buffer=None, kernel_name="", b_pre_shuffled=True, log2_k_split=0):
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
        out_hp, kernel_name, None,
        bpreshuffle=True, log2_k_split=log2_k_split,
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

        kernel_name, split_k = _select_kernel(layout, grad, gemm_M, gemm_N, gemm_K)

        if accumulate and out is not None:
            result = _fp4_gemm_core(
                A_fp4, A_scales, B_fp4, B_scales,
                out_dtype=out.dtype, out_buffer=None,
                kernel_name=kernel_name, b_pre_shuffled=b_pre_shuffled,
                log2_k_split=split_k,
            )
            out.add_(result)
            result = None
        else:
            result = _fp4_gemm_core(
                A_fp4, A_scales, B_fp4, B_scales,
                out_dtype=out_dtype, out_buffer=out,
                kernel_name=kernel_name, b_pre_shuffled=b_pre_shuffled,
                log2_k_split=split_k,
            )

        if bias is not None and layout == "TN" and not grad:
            bias_casted = cast_if_needed(bias, out_dtype)
            if result is not None:
                result = result + bias_casted

        return result
