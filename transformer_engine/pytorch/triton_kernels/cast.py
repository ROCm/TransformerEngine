# Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

"""Python interface for cast extensions"""
import os
from typing import Iterable, List, Optional, Tuple, Union
import functools
import torch
import warnings

from ..utils import non_tn_fp8_gemm_supported

from ..tensor._internal.float8_tensor_base import Float8TensorBase
from .cast_transpose import te_cast_transpose_mxfp8_triton, te_cast_transpose_noop_triton, te_dequantize_mxfp8_triton
import transformer_engine_torch as tex
from ..tensor.quantized_tensor import QuantizedTensor, Quantizer
from ..tensor._internal.mxfp8_tensor_base import MXFP8TensorBase

@functools.lru_cache(maxsize=None)
def _empty_tensor() -> torch.Tensor:
    """Get tensor with no entries and no data"""
    return torch.Tensor().cuda()

def _setup_conditional_transpose_storage(
        tensor: QuantizedTensor,
    ) -> QuantizedTensor:
        shape = tensor.shape
        quantizer = tensor._get_quantizer()

        # Allocate FP8 data transpose if needed
        data_transpose = None
        create_transpose = quantizer.columnwise_usage and not non_tn_fp8_gemm_supported(); 
        if quantizer.columnwise_usage and create_transpose:
            if tensor.ndim == 0:
                # If the original tensor is a scalar, its transpose is also a scalar.
                data_transpose = torch.empty((), dtype=torch.uint8, device=tensor.device)
            else:
                transposed_shape = (shape[-1],) + shape[:-1]
                data_transpose = torch.empty(
                    transposed_shape,
                    dtype=torch.uint8,
                    device=tensor.device,
                )

        # Construct FP8 tensor
        tensor._transpose = data_transpose
        tensor._transpose_invalid = tensor._transpose is None

def te_quantize_triton(
    tensor: torch.Tensor,
    quantizer: Quantizer,
    output: Optional[torch.Tensor] = None,
    noop_flag: torch.Tensor = None 
) -> torch.Tensor:
    """
    Quantizes the input tensor using a specified quantizer,
    with an option to utilize Triton-based `cast_transpose` for performance.
    """
    input_tensor = tensor.contiguous()
    fake_tensor_type = input_tensor.dtype
    if not fake_tensor_type.is_floating_point:
        fake_tensor_type = torch.float32
    
    out: QuantizedTensor = None
    if output is None:
        assert quantizer is not None, "Quantizer object cannot be None. Please provide a valid quantizer."
        # Create an empty QuantizedTensor if no output tensor is provided
        if input_tensor.ndim == 0:
            out = quantizer.make_empty((1, ), dtype=fake_tensor_type)
            out._data = out._data.squeeze(0)
            out._transpose = out._transpose.squeeze(0)
            _setup_conditional_transpose_storage(out)
        else:
            out = quantizer.make_empty(input_tensor.shape, dtype=fake_tensor_type)
            if isinstance(out, Float8TensorBase):
                _setup_conditional_transpose_storage(out)
    else:
        # Create a QuantizedTensor from the provided output tensor
        out = output
    
    # Construct no-op flag if needed
    if noop_flag is None:
        noop_flag = _empty_tensor()

    if out.size().numel() == 0:
        # Return empty output if the quantized tensor has no elements
        return out
    
    if isinstance(out, Float8TensorBase):
        if input_tensor.nelement() > 0:
            if out.get_metadata()["data_transpose"] is not None:
                quantizer = out._get_quantizer()
                input_scale = quantizer.scale
                amax_out = quantizer.amax
                otype = quantizer.dtype
                cast_out = out._data
                trans_out = out._transpose
                scale_inv_out = out._scale_inv
                te_cast_transpose_noop_triton(
                    input_tensor,
                    noop_flag,
                    input_scale=input_scale,
                    cast_out=cast_out,
                    trans_out=trans_out,
                    amax_out=amax_out,
                    scale_inv_out=scale_inv_out,
                    otype=otype
                )
                
            else:
                out = tex.quantize(input_tensor, quantizer, out, noop_flag)
    elif isinstance(out, MXFP8TensorBase):
        te_cast_transpose_mxfp8_triton(input_tensor, out)
        # out = tex.quantize(input_tensor, quantizer, out, noop_flag)
    else:
        raise NotImplementedError(f"Not implemented for tensor type: '{type(out).__name__}'")

    return out

def te_dequantize_triton(input, dtype=torch.float32):
    return te_dequantize_mxfp8_triton(input, dtype)
