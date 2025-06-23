# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Python interface for cast extensions"""
from typing import List, Optional, Tuple, Union
import functools
import torch

from transformer_engine.pytorch.triton_kernels.cast_transpose import te_cast_transpose_noop_triton_new
import transformer_engine_torch as tex
from ..tensor.quantized_tensor import Quantizer

@functools.lru_cache(maxsize=None)
def _empty_tensor() -> torch.Tensor:
    """Get tensor with no entries and no data"""
    return torch.Tensor().cuda()

def quantize(
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
    
    out = None
    if output is None:
        out = quantizer.make_empty(input_shape)
    else:
        out = quantizer.create_tensor_from_data(output, fake_dtype=fake_tensor_type)
    
    if noop_flag is None:
        noop_flag = _empty_tensor()

    if out.nelement() == 0:
        return out
        
    if input_tensor.nelement() > 0:
        te_cast_transpose_noop_triton_new(
            input_tensor,
            noop_flag,
            out
        )
        # use_cast_transpose_triton = bool( int(os.environ.get('NVTE_USE_CAST_TRANSPOSE_TRITON', '0')) )
        # if use_cast_transpose_triton:
        #     te_cast_transpose_noop_triton_new(
        #         input_tensor,
        #         noop_flag,
        #         out
        #     )
        # else:
        #     tex.fused_cast_transpose_noop(
        #         inp,
        #         noop_flag,
        #         fp8_scales["scale"],
        #         fp8_scales["amax"],
        #         fp8_scales["scale_inv"],
        #         cast_out,
        #         transpose_out,
        #         otype,
        #         **fp8_scales_offsets,
        #     )

    return out