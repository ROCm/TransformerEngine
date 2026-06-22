# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

import os
import torch
import triton
from transformer_engine.pytorch.tensor.float8_tensor import Float8Tensor, Float8CurrentScalingQuantizer
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor, MXFP8Quantizer
from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer
from transformer_engine.pytorch.utils import get_sm_count
from .common import te_dtype_to_torch_dtype


def use_cuda_graph_autotune():
    """Whether Triton autotuning should benchmark configs via CUDA/HIP graphs.

    Triton's ``do_bench_cudagraph`` path (``use_cuda_graph=True``) captures each
    candidate launch into a graph. On ROCm/HIP this is fragile: if any single
    candidate errors mid-capture it poisons the whole capture and every later op
    fails with "HIP error: operation failed due to a previous error during
    capture" (Code 901), aborting the first training step. The plain ``do_bench``
    path (``use_cuda_graph=False``) times each config independently, so a bad
    config is simply pruned. Default to graphs on CUDA only; allow an explicit
    override via ``NVTE_TRITON_AUTOTUNE_WITH_CUDA_GRAPH`` (set to 1 or 0).
    """
    override = os.getenv("NVTE_TRITON_AUTOTUNE_WITH_CUDA_GRAPH")
    if override is not None:
        return override == "1"
    return torch.version.hip is None

    
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
    n = get_sm_count()
    if sm_margin is not None and sm_margin > 0:
        n = max(n - int(sm_margin), 1)
    return n


def num_programs(x, sm_margin=None):
    return min(x.shape[0], get_num_sms(sm_margin))


def block_size(x, norm="layer"):
    max_fused_size = (65536 if norm=="rms" else 16384) // x.element_size()
    block_size = min(max_fused_size, triton.next_power_of_2(x.shape[1]))
    return block_size


def use_blocked(x):
    return x.shape[1] > block_size(x)


def make_ln_out(ln_out, quantizer=None, input_shape=None, out_dtype=torch.float32):

    if ln_out is None:
        # TODO(micky774): Remove corresponding FP8Quantizer check when kernels properly support MXFP8/float8_current_scaling as a fused operation
        if quantizer is None or isinstance(quantizer, (MXFP8Quantizer, Float8CurrentScalingQuantizer, NVFP4Quantizer)):
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
