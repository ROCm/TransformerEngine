# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

import os
import torch
import triton
from transformer_engine.pytorch.tensor.float8_tensor import Float8Tensor, Float8CurrentScalingQuantizer
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor, MXFP8Quantizer
from .common import te_dtype_to_torch_dtype

def get_ln_sm_margin(sm_margin_type):
    assert sm_margin_type in {"FWD", "BWD", "INF"}
    try:
        sm_margin = max(
            int(os.getenv(f"NVTE_{sm_margin_type}_LAYERNORM_SM_MARGIN", "0")), 0
        )
    except ValueError:
        sm_margin = 0
    assert sm_margin >= 0
    return sm_margin


def get_fwd_ln_sm_margin():
    return get_ln_sm_margin("FWD")


def get_bwd_ln_sm_margin():
    return get_ln_sm_margin("BWD")


def get_inf_ln_sm_margin():
    return get_ln_sm_margin("INF")


def get_num_sms(sm_margin=None):
    num_sms = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    if sm_margin is not None and sm_margin > 0:
        num_sms = max(num_sms - int(sm_margin), 1)
    return num_sms


def num_programs(x, sm_margin=None):
    return min(x.shape[0], get_num_sms(sm_margin))


def block_size(x, norm="layer"):
    max_fused_size = (65536 if norm=="rms" else 16384) // x.element_size()
    block_size = min(max_fused_size, triton.next_power_of_2(x.shape[1]), 1024)
    return block_size


def use_blocked(x):
    return x.shape[1] > block_size(x)


def make_ln_out(ln_out, quantizer=None, input_shape=None, out_dtype=torch.float32):

    if ln_out is None:
        # TODO(micky774): Remove corresponding FP8Quantizer check when kernels properly support MXFP8/float8_current_scaling as a fused operation
        if quantizer is None or isinstance(quantizer, MXFP8Quantizer) or isinstance(quantizer, Float8CurrentScalingQuantizer):
            return torch.empty(input_shape, dtype=out_dtype, device='cuda')
        return quantizer.make_empty(input_shape, dtype=out_dtype)

    # TODO: revisit the logic here, whether we should create dequantized/higher precision based on quantizer or quantized tensor type

    # TODO(micky774): Remove when kernels properly support MXFP8 as a fused operation
    if isinstance(ln_out, MXFP8Tensor):
        return ln_out.dequantize(dtype=out_dtype).to("cuda")
    # TODO(micky774): Remove when kernels properly support MXFP8 as a fused operation
    if isinstance(quantizer, MXFP8Quantizer):
        return torch.empty(input_shape, dtype=out_dtype, device='cuda')

    # TODO: remove when triton kernels support fp8 current scaling
    if isinstance(quantizer, Float8CurrentScalingQuantizer):
        return torch.empty(input_shape, dtype=out_dtype, device='cuda')

    if isinstance(ln_out, Float8Tensor):
        if ln_out.dtype == out_dtype:
            return ln_out
        return quantizer.make_empty(input_shape, dtype=out_dtype)

    if quantizer is not None:
        return quantizer.create_tensor_from_data(
                ln_out.view(te_dtype_to_torch_dtype(quantizer.dtype)),
                fake_dtype=out_dtype
            )

    return ln_out
