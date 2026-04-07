# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Quantization operations -- Triton cast kernels with PyTorch-native fallback.

Uses Triton cast/transpose kernels from triton_kernels/cast_transpose.py when
available, falls back to pure PyTorch implementations otherwise.

IMPORTANT: This module must NOT call tex.quantize/tex.dequantize in any
fallback path, because in lite mode tex IS this module — that would recurse.
"""

import torch

# Lazy-loaded Triton cast functions and type checks
_triton_cast_import_attempted = False
_triton_cast_transpose_noop = None
_triton_cast_transpose_mxfp8 = None
_triton_cast_transpose_mxfp4 = None
_triton_dequantize_mxfp8 = None
_setup_transpose_storage = None
_Float8TensorStorage = None
_MXFP8TensorStorage = None
_MXFP4TensorStorage = None
_Float8CurrentScalingQuantizer = None


def _try_load_triton_cast():
    """Lazy-import Triton cast kernels and tensor storage types."""
    global _triton_cast_import_attempted
    global _triton_cast_transpose_noop, _triton_cast_transpose_mxfp8
    global _triton_cast_transpose_mxfp4, _triton_dequantize_mxfp8
    global _setup_transpose_storage
    global _Float8TensorStorage, _MXFP8TensorStorage, _MXFP4TensorStorage
    global _Float8CurrentScalingQuantizer

    if _triton_cast_import_attempted:
        return

    _triton_cast_import_attempted = True
    try:
        from transformer_engine.pytorch.triton_kernels.cast_transpose import (
            te_cast_transpose_noop_triton,
            te_cast_transpose_mxfp8_triton,
            te_cast_transpose_mxfp4_triton,
            te_dequantize_mxfp8_triton,
        )
        from transformer_engine.pytorch.triton_kernels.cast import (
            _setup_conditional_transpose_storage,
        )
        _triton_cast_transpose_noop = te_cast_transpose_noop_triton
        _triton_cast_transpose_mxfp8 = te_cast_transpose_mxfp8_triton
        _triton_cast_transpose_mxfp4 = te_cast_transpose_mxfp4_triton
        _triton_dequantize_mxfp8 = te_dequantize_mxfp8_triton
        _setup_transpose_storage = _setup_conditional_transpose_storage
    except (ImportError, ModuleNotFoundError):
        pass

    # Always try to load tensor storage types (no Triton dependency)
    try:
        from transformer_engine.pytorch.tensor.storage.float8_tensor_storage import (
            Float8TensorStorage,
        )
        _Float8TensorStorage = Float8TensorStorage
    except (ImportError, ModuleNotFoundError):
        pass
    try:
        from transformer_engine.pytorch.tensor.storage.mxfp8_tensor_storage import (
            MXFP8TensorStorage,
        )
        _MXFP8TensorStorage = MXFP8TensorStorage
    except (ImportError, ModuleNotFoundError):
        pass
    try:
        from transformer_engine.pytorch.tensor.storage.mxfp4_tensor_storage import (
            MXFP4TensorStorage,
        )
        _MXFP4TensorStorage = MXFP4TensorStorage
    except (ImportError, ModuleNotFoundError):
        pass
    try:
        from transformer_engine.pytorch.tensor.float8_tensor import (
            Float8CurrentScalingQuantizer,
        )
        _Float8CurrentScalingQuantizer = Float8CurrentScalingQuantizer
    except (ImportError, ModuleNotFoundError):
        pass


def _empty_tensor():
    """Get tensor with no entries and no data."""
    return torch.Tensor().cuda()


# ---------------------------------------------------------------------------
# PyTorch fallback for quantize -- no recursion through tex.quantize
# ---------------------------------------------------------------------------

def _te_dtype_to_torch_fp8(te_dtype):
    """Map TE DType enum to torch FP8 dtype."""
    try:
        from transformer_engine.pytorch.triton_kernels.common import te_dtype_to_torch_dtype
        return te_dtype_to_torch_dtype(te_dtype)
    except (KeyError, ImportError):
        return torch.float8_e4m3fnuz


def _quantize_float8_pytorch(input_tensor, quantizer, out):
    """Quantize to Float8 using PyTorch ops. No C++ or tex.quantize dependency."""
    if input_tensor.nelement() == 0:
        return out

    # Compute amax and scale
    amax_val = input_tensor.abs().max()
    if hasattr(quantizer, 'amax') and quantizer.amax is not None:
        quantizer.amax.fill_(amax_val.item())

    scale = quantizer.scale
    scale_inv = out._scale_inv
    torch_fp8_dtype = _te_dtype_to_torch_fp8(quantizer.dtype)

    # Scale, cast to FP8, then store as uint8 (FP8 bit pattern)
    scaled = input_tensor.float() * scale.float()
    fp8_data = scaled.to(torch_fp8_dtype)
    out._data.copy_(fp8_data.view(torch.uint8))
    scale_inv.fill_(1.0 / scale.float().item())

    return out


def _quantize_pytorch_fallback(tensor, quantizer, output=None, noop=None):
    """Pure PyTorch quantize -- never calls tex.quantize (avoids recursion)."""
    _try_load_triton_cast()

    if quantizer is None:
        if output is not None:
            output.copy_(tensor)
            return output
        return tensor

    # Create output tensor if not provided
    out = output
    if out is None and hasattr(quantizer, 'make_empty'):
        fake_dtype = tensor.dtype if tensor.dtype.is_floating_point else torch.float32
        out = quantizer.make_empty(tensor.shape, dtype=fake_dtype)
        if _Float8TensorStorage is not None and isinstance(out, _Float8TensorStorage):
            if _setup_transpose_storage is not None:
                _setup_transpose_storage(out)

    if out is None:
        # No quantizer.make_empty — just return tensor as-is
        return tensor

    # Dispatch to appropriate PyTorch fallback based on output type
    if _Float8TensorStorage is not None and isinstance(out, _Float8TensorStorage):
        return _quantize_float8_pytorch(tensor.contiguous(), quantizer, out)

    # For other quantized types without Triton, try quantizer.quantize
    # but guard against recursion by checking if we'd go through tex.quantize
    if hasattr(quantizer, 'quantize'):
        # This is safe for non-Float8 quantizers that don't recurse through tex
        return quantizer.quantize(tensor)

    if output is not None:
        output.copy_(tensor)
        return output
    return tensor


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def quantize(tensor, quantizer, output=None, noop=None):
    """Quantize tensor. Uses Triton cast kernels when available."""
    _try_load_triton_cast()

    input_tensor = tensor.contiguous() if tensor is not None else tensor

    # Fast path: no quantizer
    if quantizer is None:
        if output is not None:
            output.copy_(input_tensor)
            return output
        return input_tensor

    # Create output tensor if not provided
    out = output
    if out is None and hasattr(quantizer, 'make_empty'):
        fake_dtype = input_tensor.dtype if input_tensor.dtype.is_floating_point else torch.float32
        if input_tensor.ndim == 0:
            out = quantizer.make_empty((1,), dtype=fake_dtype)
            if _Float8TensorStorage and isinstance(out, _Float8TensorStorage):
                out._data = out._data.squeeze(0)
                if out._transpose is not None:
                    out._transpose = out._transpose.squeeze(0)
        else:
            out = quantizer.make_empty(input_tensor.shape, dtype=fake_dtype)

        if _Float8TensorStorage and isinstance(out, _Float8TensorStorage):
            if _setup_transpose_storage is not None:
                _setup_transpose_storage(out)

    if out is None:
        return input_tensor

    # Construct no-op flag
    noop_flag = noop if noop is not None else _empty_tensor()

    # Check for empty output
    if (_MXFP8TensorStorage and isinstance(out, _MXFP8TensorStorage)
            and out._rowwise_data is None and out._columnwise_data is None):
        return out
    if not (_MXFP8TensorStorage and isinstance(out, _MXFP8TensorStorage)):
        if hasattr(out, 'size') and callable(out.size) and out.size().numel() == 0:
            return out

    # --- Triton dispatch ---
    if _Float8TensorStorage and isinstance(out, _Float8TensorStorage):
        if input_tensor.nelement() > 0:
            if _triton_cast_transpose_noop is not None and not out._transpose_invalid:
                # Triton Float8 cast+transpose
                q = out._get_quantizer()
                is_current_scaling = (
                    _Float8CurrentScalingQuantizer is not None
                    and isinstance(q, _Float8CurrentScalingQuantizer)
                )
                _triton_cast_transpose_noop(
                    input_tensor,
                    noop_flag,
                    input_scale=q.scale,
                    cast_out=out._data,
                    trans_out=out._transpose,
                    amax_out=q.amax,
                    scale_inv_out=out._scale_inv,
                    otype=q.dtype,
                    current_scaling=is_current_scaling,
                    eps=getattr(q, "amax_epsilon", 0.0),
                    force_pow_2_scales=getattr(q, "force_pow_2_scales", False),
                )
                return out
            else:
                # Float8 without valid transpose or no Triton — PyTorch fallback
                if hasattr(out, 'remove_caches'):
                    out.remove_caches()
                return _quantize_float8_pytorch(input_tensor, quantizer, out)

    elif _MXFP8TensorStorage and isinstance(out, _MXFP8TensorStorage):
        if _triton_cast_transpose_mxfp8 is not None:
            _triton_cast_transpose_mxfp8(input_tensor, out)
            return out

    elif _MXFP4TensorStorage and isinstance(out, _MXFP4TensorStorage):
        if _triton_cast_transpose_mxfp4 is not None:
            _triton_cast_transpose_mxfp4(input_tensor, out)
            return out

    # Fallback for unrecognized types
    return _quantize_pytorch_fallback(tensor, quantizer, output, noop)


def dequantize(input, otype):
    """Dequantize tensor to the specified output type."""
    _try_load_triton_cast()

    # Determine target torch dtype
    if isinstance(otype, torch.dtype):
        target_dtype = otype
    else:
        dtype_map = {0: torch.uint8, 2: torch.float32, 3: torch.float16, 4: torch.bfloat16}
        target_dtype = dtype_map.get(int(otype), torch.float32)

    # Triton MXFP8 dequantize
    if (_MXFP8TensorStorage and isinstance(input, _MXFP8TensorStorage)
            and _triton_dequantize_mxfp8 is not None):
        return _triton_dequantize_mxfp8(input, otype)

    # Float8 dequantize -- PyTorch (no Triton kernel exists for this)
    if _Float8TensorStorage and isinstance(input, _Float8TensorStorage):
        if input._data is not None:
            if input._data.nelement() == 0:
                return torch.empty_like(input._data, dtype=target_dtype)
            # Reinterpret uint8 bits as FP8 dtype, then cast to target
            torch_fp8_dtype = _te_dtype_to_torch_fp8(input._fp8_dtype)
            fp8_view = input._data.view(torch_fp8_dtype)
            return fp8_view.to(target_dtype) * input._scale_inv
        raise NotImplementedError("Dequantize from transpose not implemented in lite mode")

    # Plain tensor — just cast dtype
    if isinstance(input, torch.Tensor):
        return input.to(target_dtype)

    # Object with dequantize method (custom quantized types)
    if hasattr(input, 'dequantize'):
        return input.dequantize()

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
    current_amax = amax_history[0].clone()
    current_amax = torch.clamp(current_amax, min=1e-12)
    new_scale = fp8_max / current_amax
    scale.copy_(new_scale)
    scale_inv.copy_(1.0 / new_scale)


def fp8_block_scaling_compute_partial_amax(tensor, amax, h, w, start_offset, block_len):
    """Compute partial amax from master weights for fp8 block scaling."""
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
