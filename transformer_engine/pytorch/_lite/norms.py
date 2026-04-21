# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Normalization -- AITER Triton, TE Triton, or PyTorch-native fallback.

Backend priority:
  1. AITER fused norm+quantize (single kernel: RMSNorm/LayerNorm -> FP8 cast)
     - Current scaling:   rmsnorm2d_fwd_with_dynamicquant (Float8CurrentScalingQuantizer)
       Per-row dynamic: computes per-row scale in-kernel, no global amax pass.
       Output: FP8 data + yscale(M,) per-row dequant scales.
     - Per-tensor static: fused_rms_fp8_per_tensor_static_quant (Float8Quantizer)
     - Block scaling:     fused_rms_fp8_group_quant (MXFP8Quantizer)
  2. AITER Triton norm kernels (no quantize fusion)
  3. TE Triton kernels (triton_kernels/norms_common.py)
  4. Pure PyTorch fallback

Fused norm+quantize is used when a compatible quantizer is provided in the
forward pass. Otherwise falls back to norm -> quantizer.quantize() separately.
"""

import torch

from .aiter_utils import is_aiter_available

from collections import Counter as _NormCounter
_NORM_CALLS = _NormCounter()

def _norm_bump(tag):
    _NORM_CALLS[tag] += 1
    if sum(_NORM_CALLS.values()) % 500 == 0:
        print(f"[LITE-NORM] {dict(_NORM_CALLS)}", flush=True)

# ---------------------------------------------------------------------------
# Lazy-loaded backends. None = not yet attempted.
# ---------------------------------------------------------------------------

# AITER Triton norm functions
_aiter_rms_fwd = None
_aiter_rms_bwd = None
_aiter_ln_fwd = None
_aiter_ln_bwd = None
# AITER fused norm+quantize kernels
_aiter_fused_rms_fp8_static = None
_aiter_fused_rms_fp8_group = None
_aiter_fused_rms_dynamic_quant = None  # Per-row dynamic: rmsnorm2d_fwd_with_dynamicquant
_aiter_fused_ln_fp8_static = None  # LayerNorm variant (if available)
_aiter_import_attempted = False

# TE Triton norm functions (fallback)
_triton_ln_fwd = None
_triton_ln_bwd = None
_triton_rms_fwd = None
_triton_rms_bwd = None
_triton_import_attempted = False


def _try_load_aiter_norms():
    """Lazy-import AITER Triton norm kernels. Called once, result cached."""
    global _aiter_rms_fwd, _aiter_rms_bwd, _aiter_ln_fwd, _aiter_ln_bwd
    global _aiter_fused_rms_fp8_static, _aiter_fused_rms_fp8_group
    global _aiter_fused_rms_dynamic_quant
    global _aiter_fused_ln_fp8_static
    global _aiter_import_attempted

    if _aiter_import_attempted:
        return
    _aiter_import_attempted = True

    if not is_aiter_available():
        return
    try:
        from aiter.ops.triton.rmsnorm import (
            _rmsnorm_forward,
            _rmsnorm_backward,
        )
        from aiter.ops.triton.norm import (
            _layernorm_forward,
            _layernorm_backward,
        )
        _aiter_rms_fwd = _rmsnorm_forward
        _aiter_rms_bwd = _rmsnorm_backward
        _aiter_ln_fwd = _layernorm_forward
        _aiter_ln_bwd = _layernorm_backward
    except (ImportError, AttributeError):
        pass

    # Fused norm+quantize kernels. AITER reorganized these into a `quant/`
    # subpackage in newer versions; try the new path first, then the legacy
    # top-level path for older installs.
    _fused_static = None
    _fused_group = None
    for _mod_path in (
        "aiter.ops.triton.quant.fused_fp8_quant",
        "aiter.ops.triton.fused_fp8_quant",
    ):
        try:
            _mod = __import__(_mod_path, fromlist=[
                "fused_rms_fp8_per_tensor_static_quant",
                "fused_rms_fp8_group_quant",
            ])
            _fused_static = getattr(_mod, "fused_rms_fp8_per_tensor_static_quant", None)
            _fused_group = getattr(_mod, "fused_rms_fp8_group_quant", None)
            if _fused_static is not None or _fused_group is not None:
                break
        except BaseException as _e:
            print(
                f"[LITE-NORM-DIAG] {_mod_path} import failed: "
                f"{type(_e).__name__}: {_e}",
                flush=True,
            )
    if _fused_static is not None:
        _aiter_fused_rms_fp8_static = _fused_static
    if _fused_group is not None:
        _aiter_fused_rms_fp8_group = _fused_group

    # Fused RMSNorm + per-row dynamic FP8 quantize (current scaling)
    try:
        from aiter.ops.triton.rmsnorm import (
            rmsnorm2d_fwd_with_dynamicquant,
        )
        _aiter_fused_rms_dynamic_quant = rmsnorm2d_fwd_with_dynamicquant
    except (ImportError, AttributeError):
        pass


def _try_load_triton_norms():
    """Lazy-import TE Triton norm kernels. Called once, result cached."""
    global _triton_ln_fwd, _triton_ln_bwd
    global _triton_rms_fwd, _triton_rms_bwd
    global _triton_import_attempted

    if _triton_import_attempted:
        return

    _triton_import_attempted = True
    try:
        from transformer_engine.pytorch.triton_kernels.norms_common import (
            te_layernorm_fwd_triton,
            te_layernorm_bwd_triton,
            te_rmsnorm_fwd_triton,
            te_rmsnorm_bwd_triton,
        )
        _triton_ln_fwd = te_layernorm_fwd_triton
        _triton_ln_bwd = te_layernorm_bwd_triton
        _triton_rms_fwd = te_rmsnorm_fwd_triton
        _triton_rms_bwd = te_rmsnorm_bwd_triton
    except (ImportError, ModuleNotFoundError):
        pass


# ---------------------------------------------------------------------------
# PyTorch fallback implementations
# ---------------------------------------------------------------------------

def _layernorm_fwd_pytorch(input, weight, bias, eps, zero_centered_gamma):
    """LayerNorm forward -- pure PyTorch."""
    if zero_centered_gamma:
        weight = weight + 1.0
    mean = input.mean(dim=-1, keepdim=True)
    var = input.var(dim=-1, keepdim=True, unbiased=False)
    rstdev = torch.rsqrt(var + eps)
    output = (input - mean) * rstdev * weight
    if bias is not None:
        output = output + bias
    return output, mean.squeeze(-1), rstdev.squeeze(-1)


def _layernorm_bwd_pytorch(grad_output, input, mean, rstdev, weight,
                           zero_centered_gamma):
    """LayerNorm backward -- pure PyTorch."""
    if zero_centered_gamma:
        weight = weight + 1.0
    hidden_size = input.shape[-1]
    x_hat = (input - mean.unsqueeze(-1)) * rstdev.unsqueeze(-1)
    grad_weight = (grad_output * x_hat).sum(dim=tuple(range(grad_output.ndim - 1)))
    grad_bias = grad_output.sum(dim=tuple(range(grad_output.ndim - 1)))
    dx_hat = grad_output * weight
    dvar = (dx_hat * (input - mean.unsqueeze(-1)) * (-0.5) *
            (rstdev.unsqueeze(-1) ** 3)).sum(dim=-1, keepdim=True)
    dmean = (-dx_hat * rstdev.unsqueeze(-1)).sum(dim=-1, keepdim=True) + \
            dvar * (-2.0 / hidden_size) * (input - mean.unsqueeze(-1)).sum(dim=-1, keepdim=True)
    grad_input = dx_hat * rstdev.unsqueeze(-1) + \
                 dvar * 2.0 / hidden_size * (input - mean.unsqueeze(-1)) + \
                 dmean / hidden_size
    return grad_input, grad_weight, grad_bias


def _rmsnorm_fwd_pytorch(input, weight, eps, zero_centered_gamma):
    """RMSNorm forward -- pure PyTorch."""
    if zero_centered_gamma:
        weight = weight + 1.0
    rms = input.float().square().mean(dim=-1, keepdim=True).add_(eps).rsqrt()
    output = (input * rms).to(input.dtype) * weight
    return output, rms.squeeze(-1)


def _rmsnorm_bwd_pytorch(grad_output, input, rstdev, weight, zero_centered_gamma):
    """RMSNorm backward -- pure PyTorch."""
    if zero_centered_gamma:
        weight = weight + 1.0
    hidden_size = input.shape[-1]
    x_hat = input * rstdev.unsqueeze(-1)
    grad_weight = (grad_output * x_hat).sum(dim=tuple(range(grad_output.ndim - 1)))
    dx_hat = grad_output * weight
    grad_input = dx_hat * rstdev.unsqueeze(-1) - \
                 (dx_hat * input).sum(dim=-1, keepdim=True) * input * \
                 (rstdev.unsqueeze(-1) ** 3) / hidden_size
    return grad_input, grad_weight


# ---------------------------------------------------------------------------
# AITER adapter functions
# ---------------------------------------------------------------------------

def _aiter_layernorm_fwd(input_2d, weight, bias, eps, zero_centered_gamma):
    """LayerNorm forward via AITER Triton kernel.

    AITER's _layernorm_forward writes into pre-allocated tensors.
    """
    if zero_centered_gamma:
        weight = weight + 1.0
    M, N = input_2d.shape
    y = torch.empty_like(input_2d)
    mean = torch.empty(M, dtype=torch.float32, device=input_2d.device)
    rstd = torch.empty(M, dtype=torch.float32, device=input_2d.device)
    _aiter_ln_fwd(y, input_2d, weight, bias, mean, rstd, eps)
    return y, mean, rstd


def _aiter_layernorm_bwd(grad_output_2d, input_2d, mean, rstdev, weight,
                         zero_centered_gamma):
    """LayerNorm backward via AITER Triton kernel.

    AITER's _layernorm_backward writes into pre-allocated tensors.
    """
    if zero_centered_gamma:
        weight = weight + 1.0
    dx = torch.empty_like(input_2d)
    dw = torch.empty_like(weight)
    db = torch.empty_like(weight)
    _aiter_ln_bwd(grad_output_2d, dx, dw, db, input_2d, weight, mean, rstdev)
    return dx, dw, db


def _aiter_rmsnorm_fwd(input_2d, weight, eps, zero_centered_gamma):
    """RMSNorm forward via AITER Triton kernel.

    AITER's _rmsnorm_forward allocates output internally.
    """
    if zero_centered_gamma:
        weight = weight + 1.0
    y, rsigma = _aiter_rms_fwd(input_2d, weight, eps)
    return y, rsigma


def _aiter_rmsnorm_bwd(grad_output_2d, input_2d, rstdev, weight,
                       zero_centered_gamma):
    """RMSNorm backward via AITER Triton kernel."""
    if zero_centered_gamma:
        weight = weight + 1.0
    dx, dgamma = _aiter_rms_bwd(grad_output_2d, input_2d, weight, rstdev)
    return dx, dgamma


# ---------------------------------------------------------------------------
# Fused norm+quantize adapters
# ---------------------------------------------------------------------------

def _is_delayed_scaling_quantizer(quantizer):
    """Check if quantizer is Float8Quantizer (delayed per-tensor scaling)."""
    # Avoid importing Float8Quantizer at module level (circular import risk).
    # Use duck typing: has scale (pre-computed) and amax (to be updated).
    return (
        quantizer is not None
        and type(quantizer).__name__ == "Float8Quantizer"
        and hasattr(quantizer, "scale")
        and hasattr(quantizer, "amax")
    )


def _is_current_scaling_quantizer(quantizer):
    """Check if quantizer is Float8CurrentScalingQuantizer (per-tensor current scaling).

    This quantizer computes amax from the current tensor (no history window).
    With per-row fusion, we bypass the per-tensor amax entirely — each row
    gets its own dynamic scale computed inside the fused kernel.
    """
    return (
        quantizer is not None
        and type(quantizer).__name__ == "Float8CurrentScalingQuantizer"
    )


def _is_mxfp8_quantizer(quantizer):
    """Check if quantizer is MXFP8Quantizer (block scaling)."""
    return (
        quantizer is not None
        and type(quantizer).__name__ == "MXFP8Quantizer"
    )


def _get_fp8_torch_dtype(quantizer):
    """Get the torch FP8 dtype from a quantizer's TE dtype."""
    try:
        from transformer_engine.pytorch._lite.quantize import _te_dtype_to_torch_fp8
        return _te_dtype_to_torch_fp8(quantizer.dtype)
    except (ImportError, AttributeError):
        return torch.float8_e4m3fnuz


_FUSED_RMS_DIAG_PRINTED = False

def _try_fused_rmsnorm_quant(input_2d, weight, eps, quantizer, zero_centered_gamma,
                             orig_shape=None):
    """Attempt fused RMSNorm+FP8 quantize via AITER.

    Returns (output, rsigma) on success, or None if fusion not possible.
    The output is a QuantizedTensor (Float8Tensor or MXFP8Tensor).
    """
    global _FUSED_RMS_DIAG_PRINTED
    if not _FUSED_RMS_DIAG_PRINTED:
        _FUSED_RMS_DIAG_PRINTED = True
        qtype = type(quantizer).__name__ if quantizer is not None else "None"
        print(
            f"[LITE-NORM-DIAG] first fused-rms attempt: "
            f"quantizer_type={qtype}, "
            f"fused_dynamic={_aiter_fused_rms_dynamic_quant is not None}, "
            f"fused_static={_aiter_fused_rms_fp8_static is not None}, "
            f"fused_group={_aiter_fused_rms_fp8_group is not None}",
            flush=True,
        )

    if orig_shape is None:
        orig_shape = input_2d.shape

    if zero_centered_gamma:
        weight = weight + 1.0

    # Float8CurrentScalingQuantizer: rmsnorm2d_fwd_with_dynamicquant
    # Fused RMSNorm + per-row dynamic FP8 quantize in a single kernel.
    # Each row computes its own scale in registers — no global amax pass,
    # no BF16 intermediate written to HBM.
    if _is_current_scaling_quantizer(quantizer) and _aiter_fused_rms_dynamic_quant is not None:
        M, N = input_2d.shape
        fp8_dtype = _get_fp8_torch_dtype(quantizer)

        # Pre-allocate output tensors for the AITER kernel
        out_fp8 = torch.empty(M, N, dtype=fp8_dtype, device=input_2d.device)
        yscale = torch.empty(M, dtype=torch.float32, device=input_2d.device)

        _aiter_fused_rms_dynamic_quant(out_fp8, input_2d, yscale, weight, eps)

        # yscale is the per-row dequant scale (multiply FP8 data by yscale to
        # recover high-precision values). Wrap in Float8Tensor with vector
        # _scale_inv of shape (M,) instead of the usual scalar.
        out = quantizer.make_empty(
            orig_shape, dtype=input_2d.dtype, device=input_2d.device,
        )
        fp8_bytes = out_fp8.view(torch.uint8)
        if hasattr(out, '_data'):
            out._data.copy_(fp8_bytes.reshape(out._data.shape))
        # Store per-row dequant scales — downstream GEMM dispatch will detect
        # scale_inv.numel() > 1 and route to gemm_a8w8_per_token_scale.
        if hasattr(out, '_scale_inv'):
            out._scale_inv = yscale
        # make_empty allocated the transpose buffer (columnwise_usage was set
        # on the quantizer) but the fused kernel only writes _data. Mark the
        # buffer stale so update_usage/_create_transpose regenerates it from
        # _data on demand — otherwise downstream wgrad reads uninitialized
        # memory.
        if hasattr(out, '_transpose') and out._transpose is not None:
            out._transpose_invalid = True

        # Compute rsigma for backward pass. The fused kernel doesn't return it,
        # so cheaply recompute from input (one reduction, no FP8 cast).
        rsigma = input_2d.float().square().mean(dim=-1).add_(eps).rsqrt()
        return out, rsigma

    # Float8Quantizer: fused_rms_fp8_per_tensor_static_quant
    if _is_delayed_scaling_quantizer(quantizer) and _aiter_fused_rms_fp8_static is not None:
        # AITER kernel expects dequant scale = 1/quant_scale
        dequant_scale = (1.0 / quantizer.scale).to(torch.float32)

        out_fp8, _, _, _ = _aiter_fused_rms_fp8_static(
            input_2d, weight, eps, dequant_scale,
        )

        # Update amax for next iteration's delayed scaling.
        # copy_() keeps the reduction on-device; .item() would force a
        # CPU<->GPU sync on every RMSNorm forward.
        quantizer.amax.copy_(input_2d.abs().amax())

        # Wrap raw FP8 data in Float8Tensor via the quantizer.
        # Create empty container with the ORIGINAL (possibly N-D) shape,
        # then copy in the 2D FP8 data from the fused kernel.
        out = quantizer.make_empty(
            orig_shape, dtype=input_2d.dtype, device=input_2d.device,
        )
        fp8_bytes = out_fp8.view(torch.uint8)
        if hasattr(out, '_data'):
            out._data.copy_(fp8_bytes.reshape(out._data.shape))
        if hasattr(out, '_scale_inv'):
            out._scale_inv.copy_(dequant_scale)
        # make_empty allocated the transpose buffer (columnwise_usage was set
        # on the quantizer) but the fused kernel only writes _data. Mark the
        # buffer stale so update_usage/_create_transpose regenerates it from
        # _data on demand — otherwise downstream wgrad reads uninitialized
        # memory.
        if hasattr(out, '_transpose') and out._transpose is not None:
            out._transpose_invalid = True

        # Compute rsigma for backward pass (we need it, but the fused kernel
        # doesn't return it). Cheaply recompute from input.
        rsigma = input_2d.float().square().mean(dim=-1).add_(eps).rsqrt()
        return out, rsigma

    # MXFP8Quantizer: fused_rms_fp8_group_quant
    # Single kernel: RMSNorm → per-block FP8 quantize (group_size=32).
    if _is_mxfp8_quantizer(quantizer) and _aiter_fused_rms_fp8_group is not None:
        try:
            from transformer_engine.pytorch._lite.quantize import _linear_scale_to_e8m0

            (out_fp8, out_scales), _, _, _ = _aiter_fused_rms_fp8_group(
                input_2d, weight, eps, group_size=32,
            )

            # Create empty MXFP8 container via quantizer
            out = quantizer.make_empty(
                orig_shape, dtype=input_2d.dtype, device=input_2d.device,
            )

            # Copy FP8 data (uint8 bit pattern)
            fp8_bytes = out_fp8.view(torch.uint8)
            if hasattr(out, '_rowwise_data') and out._rowwise_data is not None:
                out._rowwise_data.copy_(fp8_bytes.reshape(out._rowwise_data.shape))

            # Convert AITER linear float32 scales → E8M0 uint8 and store
            e8m0_scales = _linear_scale_to_e8m0(out_scales)
            if hasattr(out, '_rowwise_scale_inv') and out._rowwise_scale_inv is not None:
                out._rowwise_scale_inv.copy_(
                    e8m0_scales.reshape(out._rowwise_scale_inv.shape)
                )

            # Compute rsigma for backward pass
            rsigma = input_2d.float().square().mean(dim=-1).add_(eps).rsqrt()
            return out, rsigma
        except (RuntimeError, ValueError):
            # Scale shape mismatch or other issue — fall back to separate path
            pass

    return None


# ---------------------------------------------------------------------------
# Reshape helpers for N-D input
# ---------------------------------------------------------------------------

def _ensure_2d(t):
    """Reshape to 2D (M, N) if needed. Returns (tensor_2d, original_shape)."""
    if t.ndim <= 2:
        return t, t.shape
    orig = t.shape
    return t.reshape(-1, orig[-1]), orig


def _restore_nd(t, orig_shape):
    """Restore from 2D back to original N-D shape."""
    if len(orig_shape) <= 2:
        return t
    return t.reshape(orig_shape)


def _restore_nd_quantized(out, orig_shape):
    """Restore N-D shape for possibly-quantized output."""
    if len(orig_shape) <= 2:
        return out
    batch_shape = orig_shape[:-1]
    if hasattr(out, '_data'):
        out._data = out._data.reshape(*batch_shape, -1)
    elif isinstance(out, torch.Tensor):
        out = out.reshape(*batch_shape, -1)
    return out


def _restore_stats(stats, orig_shape):
    """Restore stats (mean, rstdev) to batch shape."""
    if len(orig_shape) <= 2 or stats is None:
        return stats
    if stats.numel() == 0:
        return stats
    return stats.reshape(orig_shape[:-1])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def layernorm_fwd(input, weight, bias, eps, ln_out, quantizer, otype, sm_margin,
                  zero_centered_gamma):
    """LayerNorm forward.

    Backend priority: AITER Triton > TE Triton > PyTorch.
    """
    _try_load_aiter_norms()
    _try_load_triton_norms()

    input_2d, orig_shape = _ensure_2d(input)

    # Try AITER Triton
    if _aiter_ln_fwd is not None:
        _norm_bump("ln_fwd_aiter_triton")
        out, mu, rsigma = _aiter_layernorm_fwd(input_2d, weight, bias, eps,
                                                zero_centered_gamma)
    # Try TE Triton
    elif _triton_ln_fwd is not None:
        _norm_bump("ln_fwd_te_triton")
        if otype is None:
            otype = input.dtype
        out, mu, rsigma = _triton_ln_fwd(
            input_2d, weight, bias, eps, ln_out, quantizer, otype,
            sm_margin, zero_centered_gamma,
        )
        # TE Triton handles quantizer internally
        out = _restore_nd_quantized(out, orig_shape)
        mu = _restore_stats(mu, orig_shape)
        rsigma = _restore_stats(rsigma, orig_shape)
        return out, mu, rsigma
    # PyTorch fallback
    else:
        _norm_bump("ln_fwd_pytorch")
        out, mu, rsigma = _layernorm_fwd_pytorch(input_2d, weight, bias, eps,
                                                  zero_centered_gamma)

    # Apply quantizer (separate step -- AITER and PyTorch paths)
    if quantizer is not None and hasattr(quantizer, 'quantize'):
        out = quantizer.quantize(out)

    if ln_out is not None and ln_out is not out:
        ln_out.copy_(out)
    else:
        ln_out = out

    ln_out = _restore_nd_quantized(ln_out, orig_shape)
    mu = _restore_stats(mu, orig_shape)
    rsigma = _restore_stats(rsigma, orig_shape)
    return ln_out, mu, rsigma


def layernorm_bwd(grad_output, input, mean, rstdev, weight, sm_margin,
                  zero_centered_gamma):
    """LayerNorm backward.

    Backend priority: AITER Triton > TE Triton > PyTorch.
    """
    _try_load_aiter_norms()
    _try_load_triton_norms()

    # Dequantize grad_output if it arrived as a QuantizedTensor (e.g., from
    # the dgrad GEMM of a LayerNormLinear under FP8 CurrentScaling).
    if hasattr(grad_output, 'dequantize') and hasattr(grad_output, '_fp8_dtype'):
        grad_output = grad_output.dequantize(dtype=input.dtype)

    orig_shape = input.shape
    input_2d, _ = _ensure_2d(input)
    grad_2d, _ = _ensure_2d(grad_output)
    if mean is not None and mean.ndim > 1:
        mean = mean.reshape(-1)
    if rstdev.ndim > 1:
        rstdev = rstdev.reshape(-1)

    if _aiter_ln_bwd is not None:
        _norm_bump("ln_bwd_aiter_triton")
        dx, dgamma, dbeta = _aiter_layernorm_bwd(grad_2d, input_2d, mean, rstdev,
                                                   weight, zero_centered_gamma)
    elif _triton_ln_bwd is not None:
        _norm_bump("ln_bwd_te_triton")
        dx, dgamma, dbeta = _triton_ln_bwd(
            grad_2d, input_2d, mean, rstdev, weight, sm_margin,
            zero_centered_gamma,
        )
    else:
        _norm_bump("ln_bwd_pytorch")
        dx, dgamma, dbeta = _layernorm_bwd_pytorch(grad_2d, input_2d, mean, rstdev,
                                                     weight, zero_centered_gamma)

    dx = _restore_nd(dx, orig_shape)
    return dx, dgamma, dbeta


def rmsnorm_fwd(input, weight, eps, ln_out, quantizer, otype, sm_margin,
                zero_centered_gamma):
    """RMSNorm forward.

    Backend priority:
      1. AITER fused norm+quantize (single kernel, Float8Quantizer only)
      2. AITER Triton norm + separate quantize
      3. TE Triton norm (handles quantizer internally)
      4. PyTorch fallback norm + separate quantize
    """
    _try_load_aiter_norms()
    _try_load_triton_norms()

    input_2d, orig_shape = _ensure_2d(input)

    # Try AITER fused norm+quantize (single kernel launch)
    fused_result = _try_fused_rmsnorm_quant(
        input_2d, weight, eps, quantizer, zero_centered_gamma,
        orig_shape=orig_shape,
    )
    if fused_result is not None:
        _norm_bump("rms_fwd_aiter_fused_norm_quant")
        out, rsigma = fused_result
        rsigma = _restore_stats(rsigma, orig_shape)
        return out, torch.Tensor(), rsigma

    # Try AITER Triton (norm only, quantize separate)
    if _aiter_rms_fwd is not None:
        _norm_bump("rms_fwd_aiter_triton_unfused")
        out, rsigma = _aiter_rmsnorm_fwd(input_2d, weight, eps, zero_centered_gamma)
        mu = torch.Tensor()
    # Try TE Triton (handles quantizer internally)
    elif _triton_rms_fwd is not None:
        _norm_bump("rms_fwd_te_triton")
        if otype is None:
            otype = input.dtype
        out, mu, rsigma = _triton_rms_fwd(
            input_2d, weight, eps, ln_out, quantizer, otype,
            sm_margin, zero_centered_gamma,
        )
        out = _restore_nd_quantized(out, orig_shape)
        rsigma = _restore_stats(rsigma, orig_shape)
        return out, mu, rsigma
    # PyTorch fallback
    else:
        _norm_bump("rms_fwd_pytorch")
        out, rsigma = _rmsnorm_fwd_pytorch(input_2d, weight, eps, zero_centered_gamma)
        mu = torch.Tensor()

    # Apply quantizer (separate step -- AITER norm and PyTorch paths)
    if quantizer is not None and hasattr(quantizer, 'quantize'):
        out = quantizer.quantize(out)

    if ln_out is not None and ln_out is not out:
        ln_out.copy_(out)
    else:
        ln_out = out

    ln_out = _restore_nd_quantized(ln_out, orig_shape)
    rsigma = _restore_stats(rsigma, orig_shape)
    return ln_out, mu, rsigma


def rmsnorm_bwd(grad_output, input, rstdev, weight, sm_margin, zero_centered_gamma):
    """RMSNorm backward.

    Backend priority: AITER Triton > TE Triton > PyTorch.
    """
    _try_load_aiter_norms()
    _try_load_triton_norms()

    # Dequantize grad_output if it arrived as a QuantizedTensor (e.g., from
    # the dgrad GEMM of a LayerNormLinear under FP8 CurrentScaling).
    if hasattr(grad_output, 'dequantize') and hasattr(grad_output, '_fp8_dtype'):
        grad_output = grad_output.dequantize(dtype=input.dtype)

    orig_shape = input.shape
    input_2d, _ = _ensure_2d(input)
    grad_2d, _ = _ensure_2d(grad_output)
    if rstdev.ndim > 1:
        rstdev = rstdev.reshape(-1)

    if _aiter_rms_bwd is not None:
        _norm_bump("rms_bwd_aiter_triton")
        dx, dgamma = _aiter_rmsnorm_bwd(grad_2d, input_2d, rstdev, weight,
                                         zero_centered_gamma)
    elif _triton_rms_bwd is not None:
        _norm_bump("rms_bwd_te_triton")
        dx, dgamma = _triton_rms_bwd(
            grad_2d, input_2d, rstdev, weight, sm_margin,
            zero_centered_gamma,
        )
    else:
        _norm_bump("rms_bwd_pytorch")
        dx, dgamma = _rmsnorm_bwd_pytorch(grad_2d, input_2d, rstdev, weight,
                                           zero_centered_gamma)

    dx = _restore_nd(dx, orig_shape)
    return dx, dgamma


def rmsnorm_bwd_add(grad_output, input, rstdev, weight, zero_centered_gamma):
    """Fused RMSNorm backward + add. Returns (grad_input, grad_weight)."""
    return rmsnorm_bwd(grad_output, input, rstdev, weight, 0, zero_centered_gamma)
