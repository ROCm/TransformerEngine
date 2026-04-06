# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Quantization operations -- PyTorch-native with hooks for Triton kernels.

TODO Phase 1: Wire up to triton_kernels/cast.py and cast_transpose.py.
"""

import torch


def quantize(tensor, quantizer, output=None, noop=None):
    """Quantize tensor using the provided quantizer."""
    if quantizer is not None and hasattr(quantizer, 'quantize'):
        return quantizer.quantize(tensor)
    if output is not None:
        output.copy_(tensor)
        return output
    return tensor


def dequantize(input, otype):
    """Dequantize tensor to the specified output type."""
    if hasattr(input, 'dequantize'):
        return input.dequantize()
    # Convert otype enum to torch dtype
    dtype_map = {0: torch.uint8, 2: torch.float32, 3: torch.float16, 4: torch.bfloat16}
    target_dtype = dtype_map.get(int(otype), torch.float32) if not isinstance(otype, torch.dtype) else otype
    return input.to(target_dtype)


def bgrad_quantize(input, quantizer):
    """Compute bias gradient and quantize."""
    bgrad = input.sum(dim=tuple(range(input.ndim - 1)))
    quantized = quantize(input, quantizer)
    return quantized, bgrad


def multi_tensor_quantize(tensor_list, quantizer_list):
    """Quantize multiple tensors with corresponding quantizers."""
    results = []
    for tensor, quant in zip(tensor_list, quantizer_list):
        results.append(quantize(tensor, quant))
    return results


def split_quantize(tensor, split_sections, quantizer_list):
    """Split tensor and quantize each section."""
    splits = torch.split(tensor, split_sections, dim=0)
    results = []
    for split, quant in zip(splits, quantizer_list):
        results.append(quantize(split, quant))
    return results


def compute_amax(input, amax):
    """Compute absolute max value in tensor."""
    amax.fill_(input.abs().max().item())


def fused_amax_and_scale_update_after_reduction(
    amax_history, scale, scale_inv, scale_inv_mask, fp8_max, recipe_type,
    amax_compute_algo, is_mxfp8
):
    """Update amax history and FP8 scale/scale_inv after reduction."""
    # Simple implementation: use most recent amax to compute scale
    current_amax = amax_history[0].clone()
    # Avoid zero amax
    current_amax = torch.clamp(current_amax, min=1e-12)
    # scale = fp8_max / amax
    new_scale = fp8_max / current_amax
    scale.copy_(new_scale)
    scale_inv.copy_(1.0 / new_scale)


def fp8_block_scaling_compute_partial_amax(tensor, amax, h, w, start_offset, block_len):
    """Compute partial amax from master weights for fp8 block scaling."""
    # Reshape into blocks and compute per-block amax
    partial = tensor.view(-1)[start_offset:start_offset + h * w].view(h, w)
    num_blocks_h = (h + block_len - 1) // block_len
    num_blocks_w = (w + block_len - 1) // block_len

    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            h_start = i * block_len
            h_end = min(h_start + block_len, h)
            w_start = j * block_len
            w_end = min(w_start + block_len, w)
            block = partial[h_start:h_end, w_start:w_end]
            block_amax = block.abs().max()
            amax[i * num_blocks_w + j] = block_amax


def fp8_block_scaling_partial_cast(inp, out, scale, h, w, start_offset, block_len, out_dtype):
    """Partial cast from master weights for fp8 block scaling."""
    partial = inp.view(-1)[start_offset:start_offset + h * w].view(h, w)
    # Apply per-block scaling and cast
    num_blocks_h = (h + block_len - 1) // block_len
    num_blocks_w = (w + block_len - 1) // block_len

    result = torch.empty_like(partial)
    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            h_start = i * block_len
            h_end = min(h_start + block_len, h)
            w_start = j * block_len
            w_end = min(w_start + block_len, w)
            block = partial[h_start:h_end, w_start:w_end]
            s = scale[i * num_blocks_w + j]
            result[h_start:h_end, w_start:w_end] = block * s

    out.copy_(result)
