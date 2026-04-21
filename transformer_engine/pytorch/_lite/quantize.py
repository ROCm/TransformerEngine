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
from collections import Counter as _QuantCounter
_QUANT_CALLS = _QuantCounter()

def _quant_bump(tag):
    _QUANT_CALLS[tag] += 1
    if sum(_QUANT_CALLS.values()) % 500 == 0:
        print(f"[LITE-QUANT] {dict(_QUANT_CALLS)}", flush=True)

_FP8_FALLBACK_DIAG_PRINTS = 0
_FP8_FALLBACK_DIAG_MAX = 5

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

# AITER per-row dynamic quantize (lazy-loaded)
_aiter_dynamic_per_token_quant = None
_aiter_quant_import_attempted = False


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


def _try_load_aiter_quant():
    """Lazy-import AITER per-row dynamic quantize kernel."""
    global _aiter_dynamic_per_token_quant, _aiter_quant_import_attempted

    if _aiter_quant_import_attempted:
        return
    _aiter_quant_import_attempted = True

    try:
        from .aiter_utils import is_aiter_available
        if not is_aiter_available():
            return
        from aiter.ops.triton.quant import dynamic_per_token_quant_fp8_i8
        _aiter_dynamic_per_token_quant = dynamic_per_token_quant_fp8_i8
    except (ImportError, AttributeError):
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


def _linear_scale_to_e8m0(scale_float32):
    """Convert linear float32 scales to E8M0 biased exponent (uint8).

    E8M0 format: value = 2^(exponent - 127), stored as uint8.
    Conversion: e8m0 = floor(log2(scale)) + 127, clamped to [0, 254].

    Args:
        scale_float32: float32 tensor of per-group linear dequant scales
    Returns:
        uint8 tensor of E8M0 biased exponents
    """
    scale_clamped = scale_float32.float().clamp(min=2**-127)
    exponent = torch.floor(torch.log2(scale_clamped)) + 127
    return exponent.clamp(0, 254).to(torch.uint8)


def _quantize_float8_pytorch(input_tensor, quantizer, out):
    """Quantize to Float8 using PyTorch ops. No C++ or tex.quantize dependency."""
    if input_tensor.nelement() == 0:
        return out

    # Compute amax and scale. Keep both on-device: .item() would force a
    # CPU<->GPU sync on every quantize call.
    amax_val = input_tensor.abs().amax()
    if hasattr(quantizer, 'amax') and quantizer.amax is not None:
        quantizer.amax.copy_(amax_val)

    scale = quantizer.scale
    scale_inv = out._scale_inv
    torch_fp8_dtype = _te_dtype_to_torch_fp8(quantizer.dtype)

    # Scale, cast to FP8, then store as uint8 (FP8 bit pattern)
    scaled = input_tensor.float() * scale.float()
    fp8_data = scaled.to(torch_fp8_dtype)
    out._data.copy_(fp8_data.view(torch.uint8))
    scale_inv.copy_(scale.float().reciprocal())

    return out


def _quantize_per_row_dynamic(input_tensor, quantizer, out):
    """Per-row dynamic FP8 quantize via AITER dynamic_per_token_quant_fp8_i8.

    Each row gets its own scale computed in-kernel (no global amax pass).
    Output Float8Tensor has _scale_inv shape (M,) instead of scalar.
    Used for CurrentScaling in backward (dY quantization) and standalone
    quantize calls.
    """
    if input_tensor.nelement() == 0:
        return out

    input_2d = input_tensor.reshape(-1, input_tensor.shape[-1])
    M, N = input_2d.shape
    torch_fp8_dtype = _te_dtype_to_torch_fp8(quantizer.dtype)

    # Pre-allocate output tensors for the AITER kernel
    qx = torch.empty(M, N, dtype=torch_fp8_dtype, device=input_2d.device)
    scale_out = torch.empty(M, dtype=torch.float32, device=input_2d.device)

    _aiter_dynamic_per_token_quant(qx, input_2d, scale_out)

    # Write FP8 data into the output container
    fp8_bytes = qx.view(torch.uint8)
    out._data.copy_(fp8_bytes.reshape(out._data.shape))
    # Store per-row dequant scales — downstream GEMM detects numel() > 1
    out._scale_inv = scale_out
    # Mark transpose cache stale so update_usage(columnwise=True) will
    # regenerate it from the freshly-written _data instead of using the
    # uninitialized buffer allocated by make_empty().
    if hasattr(out, '_transpose_invalid'):
        out._transpose_invalid = True

    return out


def _quantize_mxfp8_pytorch(input_tensor, quantizer, out):
    """Quantize to MXFP8 using pure PyTorch ops — no Triton dependency.

    Implements group_size=32 block scaling with E8M0 scale format:
    1. Reshape input into groups of 32
    2. Compute per-group amax → E8M0 biased exponent
    3. Scale groups, cast to FP8, store as uint8
    """
    if input_tensor.nelement() == 0:
        return out

    try:
        from transformer_engine.pytorch.constants import MXFP8_BLOCK_SCALING_SIZE
        group_size = MXFP8_BLOCK_SCALING_SIZE  # 32
    except ImportError:
        group_size = 32

    input_2d = input_tensor.reshape(-1, input_tensor.shape[-1])
    M, K = input_2d.shape
    torch_fp8_dtype = _te_dtype_to_torch_fp8(quantizer.dtype)

    # Pad K to multiple of group_size if needed
    K_padded = ((K + group_size - 1) // group_size) * group_size
    if K_padded != K:
        input_padded = torch.nn.functional.pad(input_2d, (0, K_padded - K))
    else:
        input_padded = input_2d

    # Reshape into groups: (M, K/32, 32)
    num_groups = K_padded // group_size
    grouped = input_padded.float().reshape(M, num_groups, group_size)

    # Per-group amax
    group_amax = grouped.abs().amax(dim=-1)  # (M, num_groups)
    group_amax = group_amax.clamp(min=2**-127)

    # E8M0 biased exponent: floor(log2(amax)) + 127
    biased_exp = torch.floor(torch.log2(group_amax)) + 127
    biased_exp = biased_exp.clamp(0, 254)

    # Dequant: output = fp8_data * 2^(biased_exp - 127)
    # Quantize: fp8_data = input / 2^(biased_exp - 127)
    dequant_scale = torch.exp2(biased_exp - 127)  # (M, num_groups)
    inv_scale = 1.0 / dequant_scale  # (M, num_groups)

    # Scale each group and cast to FP8
    scaled = grouped * inv_scale.unsqueeze(-1)  # (M, num_groups, 32)
    fp8_data = scaled.reshape(M, K_padded)[:, :K].contiguous().to(torch_fp8_dtype)
    fp8_bytes = fp8_data.view(torch.uint8)

    # Write into output container
    if hasattr(out, '_rowwise_data') and out._rowwise_data is not None:
        out._rowwise_data.copy_(fp8_bytes.reshape(out._rowwise_data.shape))
    if hasattr(out, '_rowwise_scale_inv') and out._rowwise_scale_inv is not None:
        e8m0 = biased_exp[:, :((K + group_size - 1) // group_size)].to(torch.uint8)
        out._rowwise_scale_inv.copy_(e8m0.reshape(out._rowwise_scale_inv.shape))

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
    if _MXFP8TensorStorage is not None and isinstance(out, _MXFP8TensorStorage):
        return _quantize_mxfp8_pytorch(tensor.contiguous(), quantizer, out)
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

    # --- Per-row dynamic quantize (CurrentScaling + AITER) ---
    # Must come before per-tensor paths: per-row is strictly better for
    # CurrentScaling (fused single kernel, no global amax pass).
    _try_load_aiter_quant()
    if (_Float8TensorStorage and isinstance(out, _Float8TensorStorage)
            and _Float8CurrentScalingQuantizer is not None
            and isinstance(quantizer, _Float8CurrentScalingQuantizer)
            and _aiter_dynamic_per_token_quant is not None
            and input_tensor.nelement() > 0):
        _quant_bump("per_row_dynamic_aiter")
        return _quantize_per_row_dynamic(input_tensor, quantizer, out)

    # --- Triton dispatch ---
    if _Float8TensorStorage and isinstance(out, _Float8TensorStorage):
        if input_tensor.nelement() > 0:
            global _FP8_FALLBACK_DIAG_PRINTS
            if _FP8_FALLBACK_DIAG_PRINTS < _FP8_FALLBACK_DIAG_MAX:
                _FP8_FALLBACK_DIAG_PRINTS += 1
                # Also read usage from the quantizer copy the tensor holds —
                # that's what _setup_conditional_transpose_storage looked at.
                stored_q = getattr(out, "_quantizer", None)
                stored_rw = getattr(stored_q, "rowwise_usage", "MISSING")
                stored_cw = getattr(stored_q, "columnwise_usage", "MISSING")
                print(
                    f"[LITE-QUANT-DIAG #{_FP8_FALLBACK_DIAG_PRINTS}] Float8 path: "
                    f"qt={type(quantizer).__name__}, "
                    f"live_q.rw={getattr(quantizer, 'rowwise_usage', '?')}, "
                    f"live_q.cw={getattr(quantizer, 'columnwise_usage', '?')}, "
                    f"stored_q.rw={stored_rw}, stored_q.cw={stored_cw}, "
                    f"transpose_none={out._transpose is None}, "
                    f"transpose_invalid={out._transpose_invalid}, "
                    f"shape={tuple(input_tensor.shape)}",
                    flush=True,
                )
            if _triton_cast_transpose_noop is not None and not out._transpose_invalid:
                _quant_bump("float8_triton_cast_transpose")
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
                _quant_bump("float8_pytorch_fallback")
                # Float8 without valid transpose or no Triton — PyTorch fallback
                if hasattr(out, 'remove_caches'):
                    out.remove_caches()
                return _quantize_float8_pytorch(input_tensor, quantizer, out)

    elif _MXFP8TensorStorage and isinstance(out, _MXFP8TensorStorage):
        if _triton_cast_transpose_mxfp8 is not None:
            _quant_bump("mxfp8_triton")
            _triton_cast_transpose_mxfp8(input_tensor, out)
            return out
        else:
            _quant_bump("mxfp8_pytorch_fallback")
            return _quantize_mxfp8_pytorch(input_tensor, quantizer, out)

    elif _MXFP4TensorStorage and isinstance(out, _MXFP4TensorStorage):
        if _triton_cast_transpose_mxfp4 is not None:
            _quant_bump("mxfp4_triton")
            _triton_cast_transpose_mxfp4(input_tensor, out)
            return out

    # Fallback for unrecognized types
    _quant_bump("unrecognized_pytorch_fallback")
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
            hp = fp8_view.to(target_dtype)
            scale_inv = input._scale_inv
            if scale_inv.numel() == 1:
                return hp * scale_inv
            # Per-row scale: quantize produced (M_flat,) scale from a 2D view,
            # but _data may be stored in N-D shape. Reshape scale to match
            # hp's leading dims so broadcast against the last dim works.
            leading_numel = 1
            for d in hp.shape[:-1]:
                leading_numel *= d
            if scale_inv.numel() == leading_numel:
                scale_inv = scale_inv.reshape(*hp.shape[:-1], 1)
            else:
                scale_inv = scale_inv.reshape(
                    *scale_inv.shape, *([1] * (hp.ndim - scale_inv.ndim))
                )
            return hp * scale_inv
        raise NotImplementedError("Dequantize from transpose not implemented in lite mode")

    # Plain tensor — just cast dtype
    if isinstance(input, torch.Tensor):
        return input.to(target_dtype)

    # Object with dequantize method (custom quantized types)
    if hasattr(input, 'dequantize'):
        return input.dequantize()

    return input.to(target_dtype)


def bgrad_quantize(input, quantizer):
    """Compute bias gradient and quantize.

    Uses separate sum + quantize. Both ops dispatch to optimized CUDA/Triton
    kernels individually. A true single-pass fusion would require merging
    bgrad accumulation into the cast kernel (te_cast_transpose_noop_triton).
    """
    bgrad = input.sum(dim=tuple(range(input.ndim - 1)))
    quantized = quantize(input, quantizer)
    return bgrad, quantized


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
    amax.copy_(input.abs().amax())


def fused_amax_and_scale_update_after_reduction(
    contiguous_amax, amax_histories, scales,
    amax_compute_algo, fp8_dtype, margin,
):
    """Update amax history and FP8 scale after amax reduction (delayed scaling).

    Called by FP8GlobalStateManager.reduce_and_update_fp8_tensors during
    every training step. Mirrors the fused C++ kernel: writes the current
    step's reduced amax into the history buffer, rolls the window, and
    recomputes the scale from the max (or most_recent) of the history.

    Args:
        contiguous_amax: flat tensor of reduced amax values for all tensors
        amax_histories: list of [history_len, N_i] tensors (per-module group)
        scales: list of [N_i] scale buffers (per-module group)
        amax_compute_algo: "max" or "most_recent" (callable handled upstream)
        fp8_dtype: TE_DType (kFloat8E4M3 or kFloat8E5M2)
        margin: int, scale = fp8_max / amax / 2**margin
    """
    from transformer_engine.common.recipe import _FormatMaxVals

    # Map FP8 dtype → max representable value (matches get_fp8_max). On ROCm
    # (fnuz dtypes) E4M3 is clamped to 240 instead of 448.
    try:
        is_fnuz = torch.float8_e4m3fnuz is not None and torch.cuda.is_available()
    except AttributeError:
        is_fnuz = False
    dtype_name = str(fp8_dtype).rsplit('.', 1)[-1]  # "kFloat8E4M3" or "kFloat8E5M2"
    if "E4M3" in dtype_name:
        fp8_max = _FormatMaxVals.E4M3.value[1 if is_fnuz else 0]
    else:
        fp8_max = _FormatMaxVals.E5M2.value[1 if is_fnuz else 0]

    # Split the flat contiguous_amax by each group's per-tensor count (last dim
    # of history). E.g. history [1024, 3] → chunk of size 3 in contiguous_amax.
    chunk_sizes = [h.shape[-1] for h in amax_histories]
    splits = contiguous_amax.split(chunk_sizes)
    for amax_history, scale, amax_chunk in zip(amax_histories, scales, splits):
        # Write current step's reduced amax into slot 0 of history
        amax_history[0].copy_(amax_chunk)

        # Compute effective amax from history
        if amax_compute_algo == "most_recent":
            amax = amax_history[0].clone()
        else:  # "max"
            amax = amax_history.max(dim=0).values

        # Roll history window: slot 0 gets zeroed for next step's write
        if amax_history.shape[0] > 1:
            amax_history.copy_(torch.roll(amax_history, -1, 0))
        amax_history[0].fill_(0.0)

        # Compute scale: fp8_max / amax / 2**margin, with safe fallbacks
        sf = (fp8_max / amax) / (2 ** margin)
        fp32_max = torch.finfo(torch.float32).max
        sf = torch.where(amax > 0.0, sf, scale)
        sf = torch.where(torch.isfinite(amax), sf, scale)
        sf = torch.where(torch.isinf(sf), torch.full_like(sf, fp32_max), sf)
        scale.copy_(sf)


# ---------------------------------------------------------------------------
# Triton kernels for FP8 block scaling
# ---------------------------------------------------------------------------

_triton_block_scaling_loaded = False
_triton_block_amax_kernel = None
_triton_block_cast_kernel = None


def _try_load_triton_block_scaling():
    """Define Triton kernels for block scaling on first call."""
    global _triton_block_scaling_loaded, _triton_block_amax_kernel, _triton_block_cast_kernel

    if _triton_block_scaling_loaded:
        return
    _triton_block_scaling_loaded = True

    try:
        import triton
        import triton.language as tl

        @triton.autotune(
            configs=[
                triton.Config({"TILE_ROWS": 4}, num_warps=4),
                triton.Config({"TILE_ROWS": 8}, num_warps=4),
                triton.Config({"TILE_ROWS": 16}, num_warps=8),
                triton.Config({"TILE_ROWS": 32}, num_warps=8),
            ],
            key=["BLOCK_LEN"],
        )
        @triton.jit
        def _block_amax_kernel(
            input_ptr, amax_ptr,
            h, w,
            input_row_stride,
            num_blocks_w,
            BLOCK_LEN: tl.constexpr,
            TILE_ROWS: tl.constexpr,
        ):
            """2D-tiled per-block amax reduction.

            Each program handles one (BLOCK_LEN x BLOCK_LEN) block.
            Loads TILE_ROWS rows x BLOCK_LEN cols per iteration,
            processing all rows in ceil(BLOCK_LEN / TILE_ROWS) steps.
            """
            block_idx = tl.program_id(0)
            block_i = block_idx // num_blocks_w
            block_j = block_idx % num_blocks_w

            row_start = block_i * BLOCK_LEN
            col_start = block_j * BLOCK_LEN

            # 2D offsets for one tile: (TILE_ROWS, BLOCK_LEN)
            row_offsets = tl.arange(0, TILE_ROWS)    # [TILE_ROWS]
            col_offsets = tl.arange(0, BLOCK_LEN)    # [BLOCK_LEN]

            max_val = 0.0
            for tile_start in tl.static_range(0, BLOCK_LEN, TILE_ROWS):
                rows = row_start + tile_start + row_offsets  # [TILE_ROWS]
                cols = col_start + col_offsets                # [BLOCK_LEN]

                # 2D mask: valid rows AND valid cols
                row_mask = rows < h                           # [TILE_ROWS]
                col_mask = cols < w                           # [BLOCK_LEN]
                mask = row_mask[:, None] & col_mask[None, :]  # [TILE_ROWS, BLOCK_LEN]

                # 2D load
                ptrs = input_ptr + rows[:, None] * input_row_stride + cols[None, :]
                vals = tl.load(ptrs, mask=mask, other=0.0)    # [TILE_ROWS, BLOCK_LEN]

                max_val = tl.maximum(max_val, tl.max(tl.abs(vals)))

            tl.store(amax_ptr + block_idx, max_val)

        @triton.autotune(
            configs=[
                triton.Config({"TILE_ROWS": 4}, num_warps=4),
                triton.Config({"TILE_ROWS": 8}, num_warps=4),
                triton.Config({"TILE_ROWS": 16}, num_warps=8),
                triton.Config({"TILE_ROWS": 32}, num_warps=8),
            ],
            key=["BLOCK_LEN"],
        )
        @triton.jit
        def _block_cast_kernel(
            input_ptr, output_ptr, scale_ptr,
            h, w,
            input_row_stride, output_row_stride,
            num_blocks_w,
            BLOCK_LEN: tl.constexpr,
            TILE_ROWS: tl.constexpr,
        ):
            """2D-tiled per-block scale and copy.

            Each program handles one (BLOCK_LEN x BLOCK_LEN) block.
            Loads TILE_ROWS rows x BLOCK_LEN cols per iteration.
            """
            block_idx = tl.program_id(0)
            block_i = block_idx // num_blocks_w
            block_j = block_idx % num_blocks_w

            row_start = block_i * BLOCK_LEN
            col_start = block_j * BLOCK_LEN

            s = tl.load(scale_ptr + block_idx)

            row_offsets = tl.arange(0, TILE_ROWS)
            col_offsets = tl.arange(0, BLOCK_LEN)

            for tile_start in tl.static_range(0, BLOCK_LEN, TILE_ROWS):
                rows = row_start + tile_start + row_offsets
                cols = col_start + col_offsets

                row_mask = rows < h
                col_mask = cols < w
                mask = row_mask[:, None] & col_mask[None, :]

                in_ptrs = input_ptr + rows[:, None] * input_row_stride + cols[None, :]
                vals = tl.load(in_ptrs, mask=mask, other=0.0)

                out_ptrs = output_ptr + rows[:, None] * output_row_stride + cols[None, :]
                tl.store(out_ptrs, vals * s, mask=mask)

        _triton_block_amax_kernel = _block_amax_kernel
        _triton_block_cast_kernel = _block_cast_kernel

    except (ImportError, ModuleNotFoundError):
        pass


# ---------------------------------------------------------------------------
# PyTorch fallbacks for block scaling (used when Triton unavailable)
# ---------------------------------------------------------------------------

def _fp8_block_scaling_compute_partial_amax_pytorch(partial, amax, h, w, block_len):
    """Vectorized PyTorch fallback for block amax."""
    num_blocks_h = (h + block_len - 1) // block_len
    num_blocks_w = (w + block_len - 1) // block_len

    pad_h = num_blocks_h * block_len - h
    pad_w = num_blocks_w * block_len - w
    if pad_h > 0 or pad_w > 0:
        partial = torch.nn.functional.pad(partial, (0, pad_w, 0, pad_h), value=0.0)

    blocked = partial.reshape(num_blocks_h, block_len, num_blocks_w, block_len)
    block_amaxes = blocked.abs().amax(dim=(1, 3))
    amax.copy_(block_amaxes.reshape(-1))


def _fp8_block_scaling_partial_cast_pytorch(partial, out, scale, h, w, block_len):
    """Vectorized PyTorch fallback for block cast."""
    num_blocks_h = (h + block_len - 1) // block_len
    num_blocks_w = (w + block_len - 1) // block_len

    pad_h = num_blocks_h * block_len - h
    pad_w = num_blocks_w * block_len - w
    if pad_h > 0 or pad_w > 0:
        partial = torch.nn.functional.pad(partial, (0, pad_w, 0, pad_h), value=0.0)

    blocked = partial.reshape(num_blocks_h, block_len, num_blocks_w, block_len)
    scale_2d = scale.reshape(num_blocks_h, num_blocks_w)[:, None, :, None]
    scaled = blocked * scale_2d
    result = scaled.reshape(num_blocks_h * block_len, num_blocks_w * block_len)
    out.copy_(result[:h, :w])


# ---------------------------------------------------------------------------
# Public API for block scaling
# ---------------------------------------------------------------------------

def fp8_block_scaling_compute_partial_amax(tensor, amax, h, w, start_offset, block_len):
    """Compute per-block amax. Uses Triton kernel when available."""
    partial = tensor.view(-1)[start_offset:start_offset + h * w].view(h, w)
    num_blocks_h = (h + block_len - 1) // block_len
    num_blocks_w = (w + block_len - 1) // block_len

    _try_load_triton_block_scaling()
    if _triton_block_amax_kernel is not None:
        grid = (num_blocks_h * num_blocks_w,)
        _triton_block_amax_kernel[grid](
            partial, amax,
            h, w,
            partial.stride(0),
            num_blocks_w,
            BLOCK_LEN=block_len,
        )
        return

    _fp8_block_scaling_compute_partial_amax_pytorch(partial, amax, h, w, block_len)


def fp8_block_scaling_partial_cast(inp, out, scale, h, w, start_offset, block_len, out_dtype):
    """Partial cast with per-block scaling. Uses Triton kernel when available."""
    partial = inp.view(-1)[start_offset:start_offset + h * w].view(h, w)
    num_blocks_h = (h + block_len - 1) // block_len
    num_blocks_w = (w + block_len - 1) // block_len

    _try_load_triton_block_scaling()
    if _triton_block_cast_kernel is not None:
        grid = (num_blocks_h * num_blocks_w,)
        _triton_block_cast_kernel[grid](
            partial, out, scale,
            h, w,
            partial.stride(0), out.stride(0),
            num_blocks_w,
            BLOCK_LEN=block_len,
        )
        return

    _fp8_block_scaling_partial_cast_pytorch(partial, out, scale, h, w, block_len)
