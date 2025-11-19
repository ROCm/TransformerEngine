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
from ..tensor._internal.float8_blockwise_tensor_base import Float8BlockwiseQTensorBase
from ..tensor._internal.mxfp4_tensor_base import MXFP4TensorBase
from ...debug.pytorch.debug_quantization import DebugQuantizer

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

    # MXFP4 forward pass dispatch to AITER
    if isinstance(A, MXFP4TensorBase) and isinstance(B, MXFP4TensorBase):
        try:
            import aiter
            from aiter.ops.shuffle import shuffle_weight
        except ImportError:
            raise ImportError(
                "AITER library not found. Please install AITER to use MXFP4 GEMM. "
                "Install via: pip install -e /path/to/aiter"
            )

        # use_A_columnwise = layout == "NN"
        # use_B_columnwise = False  # Currently only A (weight) benefits from columnwise in dgrad

        use_A_columnwise = layout[0] == "T"
        use_B_columnwise = layout[1] == "T"
        
        if use_A_columnwise:
            if A._columnwise_data is None or A._columnwise_scale is None:
                raise RuntimeError(
                    f"layout={layout} requested columnwise data from A, but A._columnwise_data is None. "
                    "Ensure quantizer was configured with columnwise=True during forward pass."
                )
            weight_data = A._columnwise_data    # [K, N/2] - this is Quantize(A.T)
            weight_scale = A._columnwise_scale  # [K, N/32]
        else:
            weight_data = A._rowwise_data    # [N, K/2] where N = output_features
            weight_scale = A._rowwise_scale  # [N, K/32]
        
        if use_B_columnwise:
            if B._columnwise_data is None or B._columnwise_scale is None:
                raise RuntimeError(
                    f"layout={layout} requested columnwise data from B, but B._columnwise_data is None."
                )
            input_data = B._columnwise_data
            input_scale = B._columnwise_scale
        else:
            input_data = B._rowwise_data    # [M, K/2] where M = batch_size
            input_scale = B._rowwise_scale  # [M, K/32]

        M = input_data.shape[0]   
        N = weight_data.shape[0]

        # Pad M to multiple of 32 for AITER kernel requirements
        padded_M = (M + 31) // 32 * 32
        
        if out is None:
            out = torch.empty(
                padded_M, N,
                dtype=out_dtype if out_dtype is not None else torch.bfloat16,
                device=input_data.device
            )

        # Shuffle weight for FP4 layout (16x16) and call gemm_a4w4_asm
        # AITER expects: gemm_a4w4_asm(input, weight_shuffled, input_scale, weight_scale, ...)
        # Wrap in DisableTorchDispatch to prevent recursive dequantization, TODO revisit DisableTorchDispatch
        with torch._C._DisableTorchDispatch():
            weight_layout = (16, 16)
            weight_data_shuffled = shuffle_weight(weight_data, layout=weight_layout)
            
            result = aiter.gemm_a4w4_asm(
                input_data,              
                weight_data_shuffled,    
                input_scale,             
                weight_scale,            
                out,
                "" if bias is None else bias,  
                None,
                bpreshuffle=True,
                log2_k_split=0,
            )
            
            # Trim padding if necessary
            if result.shape[0] > M:
                result = result[:M, :]
            
            # Reshape output back to original shape 
            original_input_shape = getattr(B, '_original_shape', None)  # Changed from A to B (input)
            if original_input_shape is not None and len(original_input_shape) > 2:
                # Reshape [M, N] -> [..., N] where ... matches the original leading dims
                output_shape = list(original_input_shape[:-1]) + [N]
                result = result.view(output_shape)

        if int(os.getenv("NVTE_MXFP4_DEBUG", "0")) == 1:
            print_rank_0(
                f"[{__file__}] [MXFP4 DEBUG] Dispatching to AITER gemm_a4w4_asm:\t"
                f"  Weight (A) shape={weight_data.shape}, dtype={weight_data.dtype}; "
                f"scales shape={weight_scale.shape}, dtype={weight_scale.dtype}\t"
                f"  Input (B) shape={input_data.shape}, dtype={input_data.dtype}; "
                f"scales shape={input_scale.shape}, dtype={input_scale.dtype}\t"
                f"  Weight shuffled shape={weight_data_shuffled.shape}\t"
                f"  Calling aiter.gemm_a4w4_asm: M={M} (padded to {padded_M}), N={N}, "
                f"bias={'None' if bias is None else 'provided'}, "
                f"out_shape={out.shape}, result_shape (before reshape)={result.shape}\t"
                f"  Original input shape: {original_input_shape}, final result shape: {result.shape}\t"
                "  AITER gemm_a4w4_asm returned successfully"
            )

        # MXFP4 does not support GELU fusion yet
        if gelu:
            raise NotImplementedError("GELU fusion not supported with MXFP4")

        # Return in the same format as generic_gemm (use reshaped result)
        return result, None, None, extra_output

    assert layout in ("TN", "NN", "NT"), f"GEMM layout {layout} not supported."
    transa = layout[0] == "T"
    transb = layout[1] == "T"
    # assert quantization_params is None, "FP8 output not supported yet"

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