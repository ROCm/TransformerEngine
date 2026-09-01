# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""MXFP4 GEMM dispatch over AITER a4w4 kernels (plugin plan S5.4 sidecar burn-down).

Extracted whole from pytorch/cpp_extensions/gemm.py - the shared file keeps a one-line
re-import. AITER imports stay lazy (inside the functions), exactly as before.
"""
from typing import Optional, Tuple

import os

import torch
from torch.utils.cpp_extension import IS_HIP_EXTENSION

import transformer_engine_torch as tex
from transformer_engine.pytorch.utils import cast_if_needed

_FP4_USE_TUNED_GEMM = int(os.environ.get("NVTE_FP4_USE_TUNED_GEMM", "1"))

_FP4_LOG_SHAPES = int(os.environ.get("NVTE_FP4_LOG_GEMM_SHAPES", "0"))


def get_tensor_device(tensor: torch.Tensor) -> int:
    """
    Returns tensor device as an integer.

    This method is used because checking instances of
    QuantizedTensor or Storage incurs more CPU overhead.
    The order of attributes checked is important to also
    minimize overhead.
    """
    if hasattr(tensor, "device"):
        return tensor.device.index
    if hasattr(tensor, "_rowwise_data") and tensor._rowwise_data is not None:
        return tensor._rowwise_data.device.index
    if hasattr(tensor, "_columnwise_data") and tensor._columnwise_data is not None:
        return tensor._columnwise_data.device.index
    if hasattr(tensor, "_data") and tensor._data is not None:
        return tensor._data.device.index
    if hasattr(tensor, "_transpose") and tensor._transpose is not None:
        return tensor._transpose.device.index
    return torch.cuda.current_device()


if IS_HIP_EXTENSION:
    def _should_use_bf16_output_for_nvfp4_tn(
        A,
        B,
        layout: str,
        out_dtype: Optional[torch.dtype],
        out,
        bias,
        quantization_params,
        debug_quantizer,
        grad: bool,
        accumulate: bool,
        ub,
        extra_output,
        gelu: bool,
    ) -> bool:
        """Work around ROCm NVFP4 TN GEMM corruption when requesting FP32 output.

        FIXME: hipBLASLt BF16xBF16->FP32 GEMM algos with ALPHA_DEVICE_VECTOR
        produce incorrect results intermittently on AMDGPU. Return True for the
        narrow path where we force BF16 output, which empirically covers the
        corruption cases.
        """
        return (
            layout == "TN"
            and out_dtype == torch.float32
            and out is None
            and bias is not None
            and quantization_params is None
            and debug_quantizer is None
            and not grad
            and not accumulate
            and ub is None
            and extra_output is None
            and not gelu
            and (isinstance(A, NVFP4TensorStorage) or isinstance(B, NVFP4TensorStorage))
        )


def _select_kernel_fp4(layout: str, grad: bool, M: int, N: int, K: int):
    """Select kernel via tuned CSV lookup, falling back to AITER heuristic."""
    from aiter.ops.gemm_op_a4w4 import get_GEMM_config

    kernel_name = ""
    split_k = 0

    if _FP4_USE_TUNED_GEMM:
        cfg = get_GEMM_config(M, N, K)
        if cfg is not None:
            kernel_name = cfg["kernelName"]
            split_k = int(cfg.get("splitK", 0))

    if _FP4_LOG_SHAPES:
        print(f"[FP4-GEMM] {layout} grad={grad} M={M} N={N} K={K} "
              f"kernel={'heuristic' if not kernel_name else kernel_name} "
              f"splitK={split_k}", flush=True)

    return kernel_name, split_k


def _fp4_gemm_core(A_fp4, A_scales, B_fp4, B_scales, out_dtype=torch.bfloat16,
                    out_buffer=None, kernel_name="", b_pre_shuffled=True, log2_k_split=0):
    """Core FP4 GEMM via AITER a4w4 kernels.

    Routes to the ASM backend when ``kernel_name`` is an ASM-mangled symbol
    (starts with ``_ZN``) or empty (heuristic). Otherwise routes to the CK
    blockscale backend, matching AITER's own ``gemm_a4w4`` dispatcher.
    """
    import aiter
    from aiter.ops.shuffle import shuffle_weight
    from aiter.ops.gemm_op_a4w4 import gemm_a4w4_blockscale

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

    use_ck = bool(kernel_name) and kernel_name.find("_ZN") == -1
    if use_ck:
        result = gemm_a4w4_blockscale(
            A_fp4, B_shuffled, A_scales_uint8, B_scales_uint8, out_hp,
            splitK=log2_k_split,
        )
    else:
        result = aiter.gemm_a4w4_asm(
            A_fp4, B_shuffled, A_scales_uint8, B_scales_uint8,
            out_hp, kernel_name, None,
            bpreshuffle=True, log2_k_split=log2_k_split,
        )

    return result[:M, :] if result.shape[0] > M else result


def mxfp4_gemm(
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
    # Capture the logical batch shape from the wrapper tensor (B) before
    # extracting raw _rowwise_data/_columnwise_data buffers. The wrapper's
    # .size() reflects the original N-D logical shape, which we need to
    # restore after the 2D GEMM kernel. Reading from _rowwise_data.shape
    # alone would lose leading dims if storage was flattened to 2D.
    a_logical_shape = B.size()
    a_batch_shape = a_logical_shape[:-1]

    if layout == "TN":
        A_fp4 = B._rowwise_data
        A_scales = B._rowwise_scale_inv
        B_fp4 = A._rowwise_data
        B_scales = A._rowwise_scale_inv
        b_pre_shuffled = A._shuffle_rowwise_data
    elif layout == "NN":
        A_fp4 = B._rowwise_data
        A_scales = B._rowwise_scale_inv
        B_fp4 = A._columnwise_data
        B_scales = A._columnwise_scale_inv
        b_pre_shuffled = A._shuffle_columnwise_data
    elif layout == "NT":
        A_fp4 = B._columnwise_data
        A_scales = B._columnwise_scale_inv
        B_fp4 = A._columnwise_data
        B_scales = A._columnwise_scale_inv
        b_pre_shuffled = A._shuffle_columnwise_data

    else:
        raise ValueError(f"Unsupported layout for FP4 GEMM: {layout}")

    # AITER a4w4 kernels require 2D inputs (M, K/2). Flatten to
    # (M_total, K/2) and restore the batch shape afterward.
    if A_fp4.ndim > 2:
        A_fp4 = A_fp4.reshape(-1, A_fp4.shape[-1])

    out_flat = out
    if out is not None and out.ndim > 2:
        out_flat = out.reshape(-1, out.shape[-1])

    gemm_M = A_fp4.shape[0]
    gemm_N = B_fp4.shape[0]
    gemm_K = B_fp4.shape[-1] * 2

    kernel_name, split_k = _select_kernel_fp4(layout, grad, gemm_M, gemm_N, gemm_K)

    if accumulate and out_flat is not None:
        result = _fp4_gemm_core(
            A_fp4, A_scales, B_fp4, B_scales,
            out_dtype=out_flat.dtype, out_buffer=None,
            kernel_name=kernel_name, b_pre_shuffled=b_pre_shuffled,
            log2_k_split=split_k,
        )
        out_flat.add_(result)
        result = out_flat
    else:
        result = _fp4_gemm_core(
            A_fp4, A_scales, B_fp4, B_scales,
            out_dtype=out_dtype, out_buffer=out_flat,
            kernel_name=kernel_name, b_pre_shuffled=b_pre_shuffled,
            log2_k_split=split_k,
        )

    if bias is not None and layout == "TN" and not grad:
        bias_casted = cast_if_needed(bias, out_dtype)
        if result is not None:
            result = result + bias_casted

    if len(a_batch_shape) > 1 and result is not None:
        result = result.reshape(*a_batch_shape, result.shape[-1])

    return result
