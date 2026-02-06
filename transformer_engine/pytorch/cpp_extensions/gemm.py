# This file was modified for portability to AMDGPU
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Python interface for GEMM extensions"""

from typing import Iterable, Optional, Tuple, Union, List
import math
import os
import torch
import transformer_engine_torch as tex
from ..constants import TE_DType
from ..utils import get_sm_count, _empty_tensor

from torch.utils.cpp_extension import IS_HIP_EXTENSION

from ..tensor.quantized_tensor import Quantizer
from ..tensor._internal.mxfp8_tensor_base import MXFP8TensorBase
from ..tensor._internal.float8_blockwise_tensor_base import Float8BlockwiseQTensorBase
from ...debug.pytorch.debug_quantization import DebugQuantizer

__all__ = [
    "general_gemm",
    "general_grouped_gemm",
]


def validate_gemm_scale(scale: Optional[float], required: bool) -> float:
    """Validate whether a GEMM scaling factor is consistent with its usage"""
    if required:
        return scale if scale is not None else 1.0
    if scale not in (0.0, None):
        raise ValueError("scale must be zero")
    return 0.0


def unpad_scales(tensor: torch.Tensor, transpose: bool) -> torch.Tensor:
    """Removes padding from scales in MXFP8 Tensors if present"""
    if isinstance(tensor, MXFP8TensorBase):
        block_size = 32
    elif isinstance(tensor, Float8BlockwiseQTensorBase):
        block_size = 128
    else:
        raise ValueError("Only MXFP8 and FP8 Block scaling can be unpadded")

    if tensor._rowwise_scale_inv is not None:
        if transpose:
            rows, cols = tensor._rowwise_data.shape[1], tensor._rowwise_data.shape[0]
        else:
            rows, cols = tensor._rowwise_data.shape[0], tensor._rowwise_data.shape[1]

        actual_scale_shape   = tensor._rowwise_scale_inv.shape
        expected_scale_shape = (rows, math.ceil(cols / block_size))

        if actual_scale_shape != expected_scale_shape:
            tensor._rowwise_scale_inv = tensor._rowwise_scale_inv[:expected_scale_shape[0], :expected_scale_shape[1]]

    if tensor._columnwise_scale_inv is not None:
        if transpose:
            rows, cols = tensor._columnwise_data.shape[1], tensor._columnwise_data.shape[0]
        else:
            rows, cols = tensor._columnwise_data.shape[0], tensor._columnwise_data.shape[1]

        actual_scale_shape   = tensor._columnwise_scale_inv.shape
        expected_scale_shape = (math.ceil(rows / block_size), cols)

        if actual_scale_shape != expected_scale_shape:
            tensor._columnwise_scale_inv = tensor._columnwise_scale_inv[:expected_scale_shape[0], :expected_scale_shape[1]]

    return tensor


def general_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    workspace: torch.Tensor,
    out_dtype: Optional[torch.dtype] = None,
    quantization_params: Optional[Quantizer] = None,
    gelu: bool = False,
    gelu_in: torch.Tensor = None,
    alpha: float = 1.0,
    beta: Optional[float] = None,
    accumulate: bool = False,
    layout: str = "TN",
    out: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    use_split_accumulator: bool = False,
    grad: bool = False,
    ub: Union[tex.CommOverlap, tex.CommOverlapP2P] = None,
    ub_type: tex.CommOverlapType = None,
    extra_output: Optional[torch.Tensor] = None,
    bulk_overlap: bool = False,
) -> Iterable[Optional[torch.Tensor]]:
    """GEMM supporting fp8 inputs."""

    assert layout in ("TN", "NN", "NT"), f"GEMM layout {layout} not supported."
    transa = layout[0] == "T"
    transb = layout[1] == "T"
    # assert quantization_params is None, "FP8 output not supported yet"

    alpha = validate_gemm_scale(alpha, True)
    beta = validate_gemm_scale(beta, accumulate)

    if IS_HIP_EXTENSION:
        if isinstance(A, (MXFP8TensorBase, Float8BlockwiseQTensorBase)):
            A = unpad_scales(A, transa)
        if isinstance(B, (MXFP8TensorBase, Float8BlockwiseQTensorBase)):
            A = unpad_scales(B, transb)

    if ub_type is not None:
        assert ub is not None, (
            f"{'AG+GEMM' if ub_type == tex.CommOverlapType.AG else 'GEMM+RS'} overlap requires"
            + "a valid `ub` communicator object."
        )

    if ub is not None:
        assert ub_type is not None, "Comm+GEMM overlap requires a valid `comm_type` argument."
        if ub_type == tex.CommOverlapType.RS:
            if not (bulk_overlap and not ub.is_fp8_ubuf()):
                assert extra_output is not None, "GEMM+RS overlap requires extra output tensor."

    if out is not None:
        if not out.is_contiguous():
            raise ValueError("Output tensor is not contiguous.")

    debug_quantizer = None
    if isinstance(quantization_params, DebugQuantizer):
        debug_quantizer = quantization_params
        quantization_params = quantization_params.parent_quantizer
        A = A.get_tensor(not transa)
        B = B.get_tensor(transb)

    # Use bfloat16 as default bias_dtype
    bias_dtype = TE_DType[torch.bfloat16 if bias is None else bias.dtype]

    if isinstance(A, Float8BlockwiseQTensorBase) or isinstance(B, Float8BlockwiseQTensorBase):
        # There is not use_split_accumulator == False
        # implementation for Float8BlockwiseQTensorBase GEMM
        use_split_accumulator = True

        # Check that data format is supported
        if (
            A._data_format != tex.Float8BlockScaleTensorFormat.GEMM_READY
            or B._data_format != tex.Float8BlockScaleTensorFormat.GEMM_READY
        ):
            raise RuntimeError("GEMM with Float8BlockwiseQTensor requires GEMM_READY format")

    args = (
        A,
        transa,  # transa
        B,
        transb,  # transb
        out,
        quantization_params,
        TE_DType[out_dtype] if out_dtype is not None else None,
        bias,
        bias_dtype,
        gelu,
        gelu_in,
        grad,  # grad
        workspace,
        workspace.shape[0],
        accumulate,
        use_split_accumulator,
    )
    kwargs = {
        "comm_overlap": ub,
        "comm_type": ub_type,
        "extra_output": extra_output,
        "bulk_overlap": bulk_overlap,
        "alpha": alpha,
        "beta": beta,
    }

    out, bias_grad, gelu_input, extra_output = tex.generic_gemm(*args, **kwargs)

    if debug_quantizer is not None:
        out = debug_quantizer.process_gemm_output(out)

    return out, bias_grad, gelu_input, extra_output


def general_grouped_gemm(
    A: List[torch.Tensor],
    B: List[torch.Tensor],
    out: List[torch.Tensor],
    out_dtype: torch.dtype,
    workspaces: List[torch.Tensor],
    layout: str = "TN",
    m_splits: Optional[List[int]] = None,
    gelu: bool = False,
    grad=False,
    accumulate: bool = False,
    bias: Optional[List[torch.Tensor]] = None,
    use_bias: bool = False,
    use_split_accumulator: bool = False,
    D_dtype: Optional[tex.DType] = None,
    single_output=False,
) -> Tuple[List[torch.Tensor], ...]:
    """
    TN layout Grouped GEMM with fp8 inputs.
    """
    num_gemms = len(A)

    transa = layout[0] == "T"
    transb = layout[1] == "T"

    empty_tensor = _empty_tensor()
    empty_tensors = [empty_tensor] * num_gemms

    # Use bfloat16 as default bias_dtype
    gelu_input = empty_tensors
    out_dtype = TE_DType[out[0].dtype] if D_dtype is None else D_dtype

    sm_count = get_sm_count()
    if grad and use_bias:
        grad_bias = [
            torch.empty(B[i].shape[1], dtype=out[0].dtype, device="cuda") for i in range(num_gemms)
        ]
    else:
        grad_bias = empty_tensors
    bias = bias if use_bias else empty_tensors
    if use_bias:
        bias_dtype = TE_DType[grad_bias[0].dtype] if grad else TE_DType[bias[0].dtype]
    else:
        bias_dtype = TE_DType[torch.bfloat16]

    if gelu:
        gelu_input = [
            torch.empty_like(o, dtype=bias_dtype, memory_format=torch.contiguous_format)
            for o in out
        ]  # this should differ with respect to single output

    bias = tex.te_general_grouped_gemm(
        A,
        transa,
        B,
        transb,
        out,
        out_dtype,
        m_splits,
        grad_bias if grad else bias,
        bias_dtype,
        single_output,
        gelu_input,  # this is pre_gelu_out
        grad,  # grad
        workspaces,
        workspaces[0].shape[0],
        accumulate,
        use_split_accumulator,
        sm_count - int(os.getenv("NVTE_EXT_MARGIN_SM", str(sm_count))),
    )

    return out, bias, gelu_input
