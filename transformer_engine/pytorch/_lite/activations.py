# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Activation functions -- AITER fused gated activations with PyTorch fallback.

When AITER is available, gated activations (swiglu, geglu) use AITER's
fused kernels (silu_and_mul, gelu_tanh_and_mul) which combine
chunk + activation + gate multiply in a single kernel. For block-scaled
FP8 quantization, AITER's act_mul_and_fp8_group_quant fuses activation +
gate multiply + FP8 cast in a single kernel, eliminating the intermediate
bf16 round-trip.
"""

import torch
import torch.nn.functional as F
import math

from .aiter_utils import is_aiter_available, get_aiter


# Lazy-loaded references to avoid circular imports
_Float8BlockQuantizer = None
_Float8BlockwiseQTensorStorage = None
_Float8BlockScaleTensorFormat = None
_aiter_act_mul_fp8_group_quant = None
_fused_act_quant_loaded = False


def _try_load_fused_act_quant():
    """Lazy-load Float8Block types and AITER fused act+quant kernel."""
    global _Float8BlockQuantizer, _Float8BlockwiseQTensorStorage
    global _Float8BlockScaleTensorFormat, _aiter_act_mul_fp8_group_quant
    global _fused_act_quant_loaded

    if _fused_act_quant_loaded:
        return
    _fused_act_quant_loaded = True

    try:
        from transformer_engine.pytorch.tensor.float8_blockwise_tensor import (
            Float8BlockQuantizer,
        )
        from transformer_engine.pytorch.tensor.storage.float8_blockwise_tensor_storage import (
            Float8BlockwiseQTensorStorage,
        )
        from .enums import Float8BlockScaleTensorFormat
        _Float8BlockQuantizer = Float8BlockQuantizer
        _Float8BlockwiseQTensorStorage = Float8BlockwiseQTensorStorage
        _Float8BlockScaleTensorFormat = Float8BlockScaleTensorFormat
    except ImportError:
        pass

    if is_aiter_available():
        try:
            from aiter.ops.triton.activation import act_mul_and_fp8_group_quant
            _aiter_act_mul_fp8_group_quant = act_mul_and_fp8_group_quant
        except ImportError:
            pass


# Map TE activation names → AITER activation strings for fused act+quant
_AITER_ACT_QUANT_MAP = {
    "swiglu": "silu",
    "geglu": "gelu_tanh",
    "reglu": "relu",
}


def _aiter_fused_gated_act_quant(input, activation, quantizer):
    """Try AITER fused gated activation + block FP8 quantize.

    Returns the quantized tensor, or None if the fused path isn't available
    (wrong quantizer type, AITER not installed, etc.).
    """
    _try_load_fused_act_quant()

    if _aiter_act_mul_fp8_group_quant is None or _Float8BlockQuantizer is None:
        return None
    if not isinstance(quantizer, _Float8BlockQuantizer):
        return None

    aiter_act = _AITER_ACT_QUANT_MAP.get(activation)
    if aiter_act is None:
        return None

    # Flatten to 2D for AITER kernel
    orig_shape = input.shape
    input_2d = input.reshape(-1, input.shape[-1])

    try:
        fp8_data, scale_inv = _aiter_act_mul_fp8_group_quant(
            input_2d, aiter_act, group_size=quantizer.block_len,
        )
    except (RuntimeError, TypeError):
        return None

    # Reshape fp8_data back to original leading dims
    half_size = input.shape[-1] // 2
    out_shape = orig_shape[:-1] + (half_size,)
    fp8_data = fp8_data.reshape(out_shape) if fp8_data.shape != out_shape else fp8_data

    # Wrap in Float8BlockwiseQTensorStorage
    result = _Float8BlockwiseQTensorStorage(
        rowwise_data=fp8_data.view(torch.uint8),
        rowwise_scale_inv=scale_inv,
        columnwise_data=None,
        columnwise_scale_inv=None,
        fp8_dtype=quantizer.dtype,
        quantizer=quantizer,
        is_2D_scaled=False,
        data_format=_Float8BlockScaleTensorFormat.COMPACT,
    )
    return result


def _apply_quantizer(output, quantizer):
    """Apply quantizer if provided, otherwise return as-is."""
    if quantizer is not None and hasattr(quantizer, 'quantize'):
        return quantizer.quantize(output)
    return output


def _aiter_gated_act(input, aiter_fn_name):
    """Try AITER fused gated activation. Returns None if unsupported.

    AITER gated activation API: fn(out, input) -> None
    Input is (*, 2*H), output is (*, H). Fuses chunk + act + gate multiply.
    """
    aiter = get_aiter()
    if aiter is None:
        return None
    fn = getattr(aiter, aiter_fn_name, None)
    if fn is None:
        return None
    try:
        half_size = input.shape[-1] // 2
        out_shape = input.shape[:-1] + (half_size,)
        out = torch.empty(out_shape, dtype=input.dtype, device=input.device)
        fn(out, input)
        return out
    except (RuntimeError, TypeError):
        return None


# --------------------------------------------------------------------------- #
# Forward activations
# --------------------------------------------------------------------------- #

def gelu(input, quantizer):
    """GeLU activation (tanh approximation)."""
    out = F.gelu(input, approximate='tanh')
    return _apply_quantizer(out, quantizer)


def geglu(input, quantizer):
    """GeGLU: split input in half, apply GELU to first, multiply by second."""
    # Try fused gated act + block FP8 quantize (single kernel)
    fused = _aiter_fused_gated_act_quant(input, "geglu", quantizer)
    if fused is not None:
        return fused
    if is_aiter_available():
        result = _aiter_gated_act(input, 'gelu_tanh_and_mul')
        if result is not None:
            return _apply_quantizer(result, quantizer)
    chunks = input.chunk(2, dim=-1)
    out = F.gelu(chunks[0], approximate='tanh') * chunks[1]
    return _apply_quantizer(out, quantizer)


def qgelu(input, quantizer):
    """QuickGELU: x * sigmoid(1.702 * x)."""
    out = input * torch.sigmoid(1.702 * input)
    return _apply_quantizer(out, quantizer)


def qgeglu(input, quantizer):
    """Quick GeGLU: gated variant of QuickGELU."""
    chunks = input.chunk(2, dim=-1)
    out = (chunks[0] * torch.sigmoid(1.702 * chunks[0])) * chunks[1]
    return _apply_quantizer(out, quantizer)


def relu(input, quantizer):
    """ReLU activation."""
    out = F.relu(input)
    return _apply_quantizer(out, quantizer)


def reglu(input, quantizer):
    """ReGLU: gated variant of ReLU."""
    fused = _aiter_fused_gated_act_quant(input, "reglu", quantizer)
    if fused is not None:
        return fused
    chunks = input.chunk(2, dim=-1)
    out = F.relu(chunks[0]) * chunks[1]
    return _apply_quantizer(out, quantizer)


def srelu(input, quantizer):
    """Squared ReLU: relu(x)^2."""
    out = F.relu(input).square()
    return _apply_quantizer(out, quantizer)


def sreglu(input, quantizer):
    """Squared ReGLU: gated variant of squared ReLU."""
    chunks = input.chunk(2, dim=-1)
    out = F.relu(chunks[0]).square() * chunks[1]
    return _apply_quantizer(out, quantizer)


def silu(input, quantizer):
    """SiLU (Swish) activation."""
    out = F.silu(input)
    return _apply_quantizer(out, quantizer)


def swiglu(input, quantizer):
    """SwiGLU: gated variant of SiLU."""
    fused = _aiter_fused_gated_act_quant(input, "swiglu", quantizer)
    if fused is not None:
        return fused
    if is_aiter_available():
        result = _aiter_gated_act(input, 'silu_and_mul')
        if result is not None:
            return _apply_quantizer(result, quantizer)
    chunks = input.chunk(2, dim=-1)
    out = F.silu(chunks[0]) * chunks[1]
    return _apply_quantizer(out, quantizer)


def clamped_swiglu(input, quantizer, limit=7.0, alpha=1.702):
    """SwiGLU with clamping (GPT OSS variant)."""
    chunks = input.chunk(2, dim=-1)
    out = F.silu(chunks[0]) * chunks[1]
    out = out.clamp(min=-limit, max=limit)
    return _apply_quantizer(out, quantizer)


# --------------------------------------------------------------------------- #
# Backward activations
# --------------------------------------------------------------------------- #

def _gelu_backward(grad, x):
    """Backward of tanh-approximated GELU."""
    kBeta = math.sqrt(2.0 / math.pi)
    kKappa = 0.044715
    x_cube = x * x * x
    inner = kBeta * (x + kKappa * x_cube)
    tanh_inner = torch.tanh(inner)
    dtanh = 1.0 - tanh_inner * tanh_inner
    d_inner = kBeta * (1.0 + 3.0 * kKappa * x * x)
    return grad * 0.5 * (1.0 + tanh_inner + x * dtanh * d_inner)


def dgelu(grad, fwd_input, quantizer):
    """Backward of GeLU."""
    out = _gelu_backward(grad, fwd_input)
    return _apply_quantizer(out, quantizer)


def dgeglu(grad, fwd_input, quantizer):
    """Backward of GeGLU."""
    chunks = fwd_input.chunk(2, dim=-1)
    x, gate = chunks[0], chunks[1]
    gelu_x = F.gelu(x, approximate='tanh')
    dgelu_x = _gelu_backward(grad * gate, x)
    dgate = grad * gelu_x
    out = torch.cat([dgelu_x, dgate], dim=-1)
    return _apply_quantizer(out, quantizer)


def dqgelu(grad, fwd_input, quantizer):
    """Backward of QuickGELU."""
    sig = torch.sigmoid(1.702 * fwd_input)
    out = grad * sig * (1.0 + 1.702 * fwd_input * (1.0 - sig))
    return _apply_quantizer(out, quantizer)


def dqgeglu(grad, fwd_input, quantizer):
    """Backward of Quick GeGLU."""
    chunks = fwd_input.chunk(2, dim=-1)
    x, gate = chunks[0], chunks[1]
    sig = torch.sigmoid(1.702 * x)
    qgelu_x = x * sig
    dqgelu_x = grad * gate * sig * (1.0 + 1.702 * x * (1.0 - sig))
    dgate = grad * qgelu_x
    out = torch.cat([dqgelu_x, dgate], dim=-1)
    return _apply_quantizer(out, quantizer)


def drelu(grad, fwd_input, quantizer):
    """Backward of ReLU."""
    out = grad * (fwd_input > 0).to(grad.dtype)
    return _apply_quantizer(out, quantizer)


def dreglu(grad, fwd_input, quantizer):
    """Backward of ReGLU."""
    chunks = fwd_input.chunk(2, dim=-1)
    x, gate = chunks[0], chunks[1]
    dx = grad * gate * (x > 0).to(grad.dtype)
    dgate = grad * F.relu(x)
    out = torch.cat([dx, dgate], dim=-1)
    return _apply_quantizer(out, quantizer)


def dsrelu(grad, fwd_input, quantizer):
    """Backward of Squared ReLU."""
    out = grad * 2.0 * F.relu(fwd_input)
    return _apply_quantizer(out, quantizer)


def dsreglu(grad, fwd_input, quantizer):
    """Backward of Squared ReGLU."""
    chunks = fwd_input.chunk(2, dim=-1)
    x, gate = chunks[0], chunks[1]
    dx = grad * gate * 2.0 * F.relu(x)
    dgate = grad * F.relu(x).square()
    out = torch.cat([dx, dgate], dim=-1)
    return _apply_quantizer(out, quantizer)


def dsilu(grad, fwd_input, quantizer):
    """Backward of SiLU."""
    sig = torch.sigmoid(fwd_input)
    out = grad * sig * (1.0 + fwd_input * (1.0 - sig))
    return _apply_quantizer(out, quantizer)


def dswiglu(grad, fwd_input, quantizer):
    """Backward of SwiGLU."""
    chunks = fwd_input.chunk(2, dim=-1)
    x, gate = chunks[0], chunks[1]
    sig = torch.sigmoid(x)
    silu_x = x * sig
    dx = grad * gate * sig * (1.0 + x * (1.0 - sig))
    dgate = grad * silu_x
    out = torch.cat([dx, dgate], dim=-1)
    return _apply_quantizer(out, quantizer)


def clamped_dswiglu(grad, fwd_input, quantizer, limit=7.0, alpha=1.702):
    """Backward of clamped SwiGLU."""
    chunks = fwd_input.chunk(2, dim=-1)
    x, gate = chunks[0], chunks[1]
    sig = torch.sigmoid(x)
    silu_x = x * sig
    fwd_out = silu_x * gate
    # Zero out gradient where clamped
    mask = (fwd_out >= -limit) & (fwd_out <= limit)
    grad = grad * mask.to(grad.dtype)
    dx = grad * gate * sig * (1.0 + x * (1.0 - sig))
    dgate = grad * silu_x
    out = torch.cat([dx, dgate], dim=-1)
    return _apply_quantizer(out, quantizer)


# --------------------------------------------------------------------------- #
# DBias + DAct fusions
# --------------------------------------------------------------------------- #

def dbias_dgelu(grad, fwd_input, quantizer):
    """Fused DGeLU + DBias: returns (dact, dbias)."""
    dact = _gelu_backward(grad, fwd_input)
    dbias = dact.sum(dim=tuple(range(dact.ndim - 1)))
    return _apply_quantizer(dact, quantizer), dbias


def dbias_dsilu(grad, fwd_input, quantizer):
    """Fused DSiLU + DBias: returns (dact, dbias)."""
    sig = torch.sigmoid(fwd_input)
    dact = grad * sig * (1.0 + fwd_input * (1.0 - sig))
    dbias = dact.sum(dim=tuple(range(dact.ndim - 1)))
    return _apply_quantizer(dact, quantizer), dbias


def dbias_drelu(grad, fwd_input, quantizer):
    """Fused DReLU + DBias: returns (dact, dbias)."""
    dact = grad * (fwd_input > 0).to(grad.dtype)
    dbias = dact.sum(dim=tuple(range(dact.ndim - 1)))
    return _apply_quantizer(dact, quantizer), dbias


def dbias_dqgelu(grad, fwd_input, quantizer):
    """Fused DQGeLU + DBias: returns (dact, dbias)."""
    sig = torch.sigmoid(1.702 * fwd_input)
    dact = grad * sig * (1.0 + 1.702 * fwd_input * (1.0 - sig))
    dbias = dact.sum(dim=tuple(range(dact.ndim - 1)))
    return _apply_quantizer(dact, quantizer), dbias


def dbias_dsrelu(grad, fwd_input, quantizer):
    """Fused DSquaredReLU + DBias: returns (dact, dbias)."""
    dact = grad * 2.0 * F.relu(fwd_input)
    dbias = dact.sum(dim=tuple(range(dact.ndim - 1)))
    return _apply_quantizer(dact, quantizer), dbias
