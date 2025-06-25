# Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

"""Python interface for cast extensions"""
import os
from typing import List, Optional, Tuple, Union
import functools
import torch

from transformer_engine.pytorch.triton_kernels.cast_transpose import te_cast_transpose_noop_triton
import transformer_engine_torch as tex
from ..tensor.quantized_tensor import QuantizedTensor, Quantizer

@functools.lru_cache(maxsize=None)
def _empty_tensor() -> torch.Tensor:
    """Get tensor with no entries and no data"""
    return torch.Tensor().cuda()

def quantize_triton(
    tensor: torch.Tensor,
    quantizer: Quantizer,
    output: Optional[torch.Tensor] = None,
    noop_flag: torch.Tensor = None 
) -> torch.Tensor:
    """quantize"""

    input_tensor = tensor.contiguous()
    input_shape = list(input_tensor.shape)
    fake_tensor_type = input_tensor.dtype
    if not fake_tensor_type.is_floating_point:
        fake_tensor_type = torch.float32
    
    out: QuantizedTensor = None
    if output is None:
        out = quantizer.make_empty(input_shape, dtype=fake_tensor_type)
    else:
        out = quantizer.create_tensor_from_data(output, fake_dtype=fake_tensor_type)
    
    if noop_flag is None:
        noop_flag = _empty_tensor()

    if out.nelement() == 0:
        return out
        
    if input_tensor.nelement() > 0:
        
        if out.get_metadata()["data_transpose"] is not None:
            quantizer = out._get_quantizer()
            input_scale = quantizer.scale
            amax_out = quantizer.amax
            otype = quantizer.dtype
            cast_out = out._data
            trans_out = out._transpose
            scale_inv_out = out._scale_inv
            use_cast_transpose_triton =  bool( int(os.environ.get('NVTE_USE_CAST_TRANSPOSE_TRITON', '0')) )
            if use_cast_transpose_triton:
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
                out = quantizer.quantize(input_tensor)
        else:
            out = quantizer.quantize(input_tensor)
    return out