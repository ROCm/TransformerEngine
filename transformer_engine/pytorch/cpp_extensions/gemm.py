# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# This file was modified for portability to AMDGPU
# See LICENSE for license information.

"""Python interface for GEMM extensions"""

from typing import Iterable, Optional, Tuple, Union, List
import os
import torch
import transformer_engine_torch as tex
from ..constants import TE_DType
from ..utils import get_sm_count, _empty_tensor

from ..tensor.quantized_tensor import Quantizer
from ..tensor._internal.mxfp4_tensor_base import MXFP4TensorBase
import os

__all__ = [
    "general_gemm",
    "general_grouped_gemm",
]


def print_rank_0(*args, **kwargs):
    """Print only from rank 0 to avoid duplicate logs in distributed training."""
    import torch.distributed as dist
    if (dist.get_rank() if dist.is_initialized() else 0) == 0:
        print(*args, **kwargs)


def general_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    workspace: torch.Tensor,
    out_dtype: Optional[torch.dtype] = None,
    quantization_params: Optional[Quantizer] = None,
    gelu: bool = False,
    gelu_in: torch.Tensor = None,
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

    # ================= FP4 Path =============================================
    if isinstance(A, MXFP4TensorBase) and isinstance(B, MXFP4TensorBase):
        from ..module.fp4_handler_gemm import fp4_gemm_layout
        
        # For FP4 tensors, we don't call update_usage here because:
        # 1. They should already have both orientations from quantization
        # 2. Unlike FP8, FP4 can't add orientations dynamically
        # The usage flags will be set appropriately during quantization in linear.py
        
        # Call FP4 GEMM with layout
        result = fp4_gemm_layout(
            A, B,
            layout=layout,
            out_dtype=out_dtype if out_dtype is not None else torch.bfloat16,
            bias=bias,
            out=out,
            grad=grad,
            accumulate=accumulate,
        )
        
        # Return in same format as general_gemm
        return result, None, None, None
    
    # ================= FP8 Path =============================================
    assert layout in ("TN", "NN", "NT"), f"GEMM layout {layout} not supported."
    transa = layout[0] == "T"
    transb = layout[1] == "T"

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

    # Use bfloat16 as default bias_dtype
    bias_dtype = TE_DType[torch.bfloat16 if bias is None else bias.dtype]


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
