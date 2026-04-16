# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""GEMM operations -- multi-backend with AITER, Triton, and PyTorch fallback.

Backend priority (configurable via NVTE_LITE_GEMM_BACKEND env var):
1. AITER CK GEMM (default) -- CK/ASM kernels for FP8 precisions
2. AITER Triton GEMM -- dedicated Triton kernels for FP8 and BF16/FP16
3. torch.matmul -- PyTorch fallback (always available)

Set NVTE_LITE_GEMM_BACKEND to override:
  "ck"      -- prefer AITER CK kernels (default)
  "triton"  -- prefer AITER Triton GEMM kernels
  "pytorch" -- skip AITER, use torch.matmul directly
"""

import os
import torch

from .aiter_utils import is_aiter_available, get_aiter

# FP8 dtypes for detection
_FP8_DTYPES = (
    torch.float8_e4m3fn, torch.float8_e5m2,
    torch.float8_e4m3fnuz, torch.float8_e5m2fnuz,
)

_GEMM_BACKEND = os.environ.get("NVTE_LITE_GEMM_BACKEND", "ck").lower()


def _dequantize_if_needed(tensor):
    """Dequantize FP8/quantized tensor to BF16 for matmul."""
    if _is_mxfp8(tensor):
        return tensor.dequantize(dtype=torch.bfloat16)
    if _is_blockwise_fp8(tensor):
        return tensor.dequantize(dtype=torch.bfloat16)
    if hasattr(tensor, 'dequantize'):
        return tensor.dequantize()
    if isinstance(tensor, torch.Tensor) and tensor.dtype in _FP8_DTYPES:
        return tensor.to(torch.bfloat16)
    return tensor


def _is_quantized(tensor):
    """Check if tensor is a quantized type with FP8 data."""
    if hasattr(tensor, '_data') and hasattr(tensor, '_scale_inv'):
        return True
    if _is_mxfp8(tensor):
        return True
    return False


def _get_raw_data(tensor):
    """Extract raw data and scale from a quantized tensor, or return tensor as-is."""
    if _is_blockwise_fp8(tensor):
        data, scale = _get_blockwise_data(tensor, need_rowwise=True)
        return data, scale
    if _is_mxfp8(tensor):
        # MXFP8 scales are E8M0 uint8 — not directly usable as float scales
        # for AITER GEMM dispatch. Return data only; GEMM will dequantize.
        return tensor._rowwise_data, None
    if hasattr(tensor, '_data') and hasattr(tensor, '_scale_inv'):
        data = tensor._data
        if data is None:
            # Columnwise-only tensor: _data was deleted by update_usage.
            # Use transpose if available, otherwise dequantize.
            if hasattr(tensor, '_transpose') and tensor._transpose is not None:
                data = tensor._transpose
            else:
                return tensor, None
        # Float8Tensor stores FP8 bit patterns as uint8 — reinterpret as the
        # actual FP8 dtype so downstream Triton kernels see the correct type.
        if data.dtype == torch.uint8 and hasattr(tensor, '_fp8_dtype'):
            from transformer_engine.pytorch._lite.quantize import _te_dtype_to_torch_fp8
            data = data.view(_te_dtype_to_torch_fp8(tensor._fp8_dtype))
        return data, tensor._scale_inv
    return tensor, None


# ---------------------------------------------------------------------------
# AITER CK GEMM dispatch
# ---------------------------------------------------------------------------

def _is_per_row_scaled(scale):
    """Check if scale tensor is per-row (one scale per token/row).

    Per-row scales have shape (M,) or (M, 1) — 1D with numel > 1.
    Block scales are 2D with shape (ceil(M/block), ceil(N/block)).
    """
    return (scale is not None
            and scale.numel() > 1
            and scale.ndim == 1)


def _is_block_scaled(scale):
    """Check if scale tensor indicates block scaling (2D multi-element scale).

    Excludes per-row scales (1D) — those use gemm_a8w8_per_token_scale.
    """
    return (scale is not None
            and scale.numel() > 1
            and not _is_per_row_scaled(scale))


def _is_fp4(tensor):
    """Check if tensor is MXFP4 quantized.

    Discriminates from MXFP8 via _fp4_dtype (MXFP4) vs _fp8_dtype (MXFP8).
    """
    return (hasattr(tensor, '_rowwise_data') and
            hasattr(tensor, '_fp4_dtype') and
            not hasattr(tensor, '_is_2D_scaled') and  # exclude Float8Blockwise
            tensor._rowwise_data is not None)


def _is_mxfp8(tensor):
    """Check if tensor is MXFP8 quantized (block-scaled FP8, group_size=32).

    MXFP8 uses _rowwise_data/_rowwise_scale_inv (shared attribute names with
    MXFP4), distinguished by _fp8_dtype. No AITER GEMM kernel exists on MI300X;
    future MI350 kernel hook is in _aiter_ck_gemm/_aiter_triton_gemm.
    """
    return (hasattr(tensor, '_rowwise_data') and
            hasattr(tensor, '_fp8_dtype') and
            not hasattr(tensor, '_is_2D_scaled') and  # exclude Float8Blockwise
            not hasattr(tensor, '_data') and           # exclude Float8Tensor
            tensor._rowwise_data is not None)


def _get_fp4_data(tensor):
    """Extract FP4 data and scale from MXFP4 tensor."""
    return tensor._rowwise_data, tensor._rowwise_scale_inv


def _is_blockwise_fp8(tensor):
    """Check if tensor is Float8BlockwiseQTensorStorage (2D block-scaled FP8)."""
    return hasattr(tensor, '_is_2D_scaled') and hasattr(tensor, '_data_format')


def _get_blockwise_data(tensor, need_rowwise=True):
    """Extract data and scale from Float8BlockwiseQTensorStorage.

    Returns (data, scale_inv) for the requested orientation.
    For GEMM: A (weight) typically needs columnwise, B (activation) needs rowwise.
    """
    if need_rowwise and tensor._rowwise_data is not None:
        return tensor._rowwise_data, tensor._rowwise_scale_inv
    if not need_rowwise and tensor._columnwise_data is not None:
        return tensor._columnwise_data, tensor._columnwise_scale_inv
    # Fall back to whatever is available
    if tensor._rowwise_data is not None:
        return tensor._rowwise_data, tensor._rowwise_scale_inv
    return tensor._columnwise_data, tensor._columnwise_scale_inv


def _aiter_ck_gemm(aiter, a_data, a_scale, b_data, b_scale,
                   a_is_fp8, b_is_fp8, transA, transB,
                   A, B):
    """Dispatch to AITER CK/ASM kernels. Returns result tensor or None."""
    try:
        # MXFP8: No hardware GEMM on MI300X. Fall through to dequant path.
        # TODO(MI350): Add aiter.gemm_mxfp8() dispatch when available.
        if _is_mxfp8(A) or _is_mxfp8(B):
            return None

        # FP4 × FP4
        if _is_fp4(A) and _is_fp4(B):
            if hasattr(aiter, 'gemm_a4w4'):
                a4_data, a4_scale = _get_fp4_data(A)
                b4_data, b4_scale = _get_fp4_data(B)
                M, _ = a4_data.shape
                N, _ = b4_data.shape
                out = torch.empty(M, N, dtype=torch.bfloat16, device=a4_data.device)
                return aiter.gemm_a4w4(a4_data, b4_data, a4_scale, b4_scale, out)

        # Float8Blockwise (2D block-scaled, 128×128 blocks) — always block-scaled
        a_is_blockwise = _is_blockwise_fp8(A)
        b_is_blockwise = _is_blockwise_fp8(B)

        # FP8 × FP8
        if a_is_fp8 and b_is_fp8:
            # Determine layout: Y = X @ W^T
            # TE: result = B @ A. transA=True means A is (N,K) weight layout.
            if b_is_blockwise:
                x, x_scale = _get_blockwise_data(B, need_rowwise=not transB)
            else:
                x = b_data if not transB else b_data.t().contiguous()
                x_scale = b_scale

            if a_is_blockwise:
                w, w_scale = _get_blockwise_data(A, need_rowwise=transA)
            else:
                w = a_data if transA else a_data.t().contiguous()
                w_scale = a_scale

            if _is_per_row_scaled(x_scale) or _is_per_row_scaled(w_scale):
                # Per-row (per-token) FP8 — from CurrentScaling fused norm+quant.
                # Triton-only kernel; no CK variant exists. Fall through to None
                # so the caller tries the Triton backend next.
                pass
            elif (_is_block_scaled(x_scale) or _is_block_scaled(w_scale)
                    or a_is_blockwise or b_is_blockwise):
                # Block-scale FP8 (includes Float8Blockwise)
                if hasattr(aiter, 'gemm_a8w8_blockscale'):
                    return aiter.gemm_a8w8_blockscale(x, w, x_scale, w_scale)
            else:
                # Per-tensor FP8
                if hasattr(aiter, 'gemm_a8w8_CK'):
                    return aiter.gemm_a8w8_CK(x, w, x_scale, w_scale)

        elif not a_is_fp8 and b_is_fp8:
            if hasattr(aiter, 'gemm_a16w8'):
                a_mat = _dequantize_if_needed(A)
                if transA:
                    a_mat = a_mat.t()
                b_mat = b_data.t() if transB else b_data
                return aiter.gemm_a16w8(a_mat, b_mat, b_scale)

        elif not a_is_fp8 and not b_is_fp8:
            pass  # No CK kernel for non-FP8/FP4 individual GEMM

    except (RuntimeError, TypeError, AttributeError):
        pass
    return None


# ---------------------------------------------------------------------------
# AITER Triton GEMM dispatch (dedicated Triton kernels per precision)
# ---------------------------------------------------------------------------

def _aiter_triton_gemm(A, transA, B, transB, a_data, a_scale, b_data, b_scale,
                       a_is_fp8, b_is_fp8):
    """Dispatch to AITER's dedicated Triton GEMM kernels.

    Per precision:
      FP4×FP4: aiter.ops.triton.gemm_afp4wfp4
      FP8×FP8 per-row:     aiter.ops.triton.gemm_a8w8_per_token_scale
      FP8×FP8 block-scale: aiter.ops.triton.gemm_a8w8_blockscale
      FP8×FP8 per-tensor:  aiter.ops.triton.gemm_a8w8
      BF16/FP16: aiter.ops.triton.gemm_a16w16
    All kernels compute Y = X @ W^T (weight is internally transposed).
    Returns result tensor or None.
    """
    try:
        # MXFP8: No Triton MXFP8 GEMM on MI300X. Fall through to dequant path.
        # TODO(MI350): Add Triton MXFP8 GEMM kernel dispatch when available.
        if _is_mxfp8(A) or _is_mxfp8(B):
            return None

        # FP4 × FP4
        if _is_fp4(A) and _is_fp4(B):
            from aiter.ops.triton.gemm_afp4wfp4 import (
                gemm_afp4wfp4 as triton_gemm_fp4,
            )
            a4_data, a4_scale = _get_fp4_data(A)
            b4_data, b4_scale = _get_fp4_data(B)
            return triton_gemm_fp4(a4_data, b4_data, a4_scale, b4_scale)

        # Float8Blockwise and standard FP8 layout mapping for Y = X @ W^T
        a_is_blockwise = _is_blockwise_fp8(A)
        b_is_blockwise = _is_blockwise_fp8(B)

        if b_is_blockwise:
            x, x_scale = _get_blockwise_data(B, need_rowwise=not transB)
        else:
            x = b_data if not transB else b_data.t().contiguous()
            x_scale = b_scale

        if a_is_blockwise:
            w, w_scale = _get_blockwise_data(A, need_rowwise=transA)
        else:
            w = a_data if transA else a_data.t().contiguous()
            w_scale = a_scale

        if a_is_fp8 and b_is_fp8:
            if _is_per_row_scaled(x_scale) or _is_per_row_scaled(w_scale):
                # Per-row (per-token) FP8 — from CurrentScaling fused norm+quant.
                # x_scale (M,) = per-token activation scale
                # w_scale may be scalar (per-tensor weight) or (N,) per-channel.
                from aiter.ops.triton.gemm_a8w8_per_token_scale import (
                    gemm_a8w8_per_token_scale as triton_a8w8_pt,
                )
                # Kernel expects (M, 1) and (N, 1) shaped scales
                if x_scale is not None and x_scale.ndim == 1:
                    x_scale = x_scale.unsqueeze(1)
                if w_scale is not None and w_scale.numel() == 1:
                    w_scale = w_scale.expand(w.shape[0]).unsqueeze(1)
                elif w_scale is not None and w_scale.ndim == 1:
                    w_scale = w_scale.unsqueeze(1)
                return triton_a8w8_pt(x, w, x_scale, w_scale)
            elif (_is_block_scaled(x_scale) or _is_block_scaled(w_scale)
                    or a_is_blockwise or b_is_blockwise):
                from aiter.ops.triton.gemm_a8w8_blockscale import (
                    gemm_a8w8_blockscale as triton_a8w8_bs,
                )
                return triton_a8w8_bs(x, w, x_scale, w_scale)
            else:
                from aiter.ops.triton.gemm_a8w8 import (
                    gemm_a8w8 as triton_a8w8,
                )
                return triton_a8w8(x, w, x_scale, w_scale)

        elif not a_is_fp8 and b_is_fp8:
            try:
                from aiter.ops.triton.gemm_a16w8_blockscale import (
                    gemm_a16w8_blockscale as triton_a16w8,
                )
                x_hp = _dequantize_if_needed(B)
                if transB:
                    x_hp = x_hp.t().contiguous()
                return triton_a16w8(x_hp, w, w_scale)
            except ImportError:
                pass

        elif not a_is_fp8 and not b_is_fp8:
            # Skip FP32 — Triton GEMM only supports BF16/FP16
            a_mat = _dequantize_if_needed(A)
            if a_mat.dtype == torch.float32:
                return None

            from aiter.ops.triton.gemm_a16w16 import (
                gemm_a16w16 as triton_a16w16,
            )
            b_mat = _dequantize_if_needed(B)
            x = b_mat if not transB else b_mat.t().contiguous()
            w = a_mat if transA else a_mat.t().contiguous()
            return triton_a16w16(x, w)

    except (RuntimeError, TypeError, AttributeError, ImportError):
        pass
    return None


# ---------------------------------------------------------------------------
# Unified AITER dispatch with backend selection
# ---------------------------------------------------------------------------

def _aiter_gemm(A, transA, B, transB, D, quantizer, output_dtype,
                bias, bias_type, gelu, gelu_in, grad,
                accumulate, alpha):
    """Dispatch GEMM to AITER backend selected by NVTE_LITE_GEMM_BACKEND.

    Falls back through: preferred backend -> other backends -> None (PyTorch).
    """
    aiter = get_aiter()
    if aiter is None:
        return None

    a_data, a_scale = _get_raw_data(A)
    b_data, b_scale = _get_raw_data(B)

    a_is_fp8 = _is_quantized(A) or (isinstance(a_data, torch.Tensor) and a_data.dtype in _FP8_DTYPES)
    b_is_fp8 = _is_quantized(B) or (isinstance(b_data, torch.Tensor) and b_data.dtype in _FP8_DTYPES)

    result = None

    triton_args = (A, transA, B, transB, a_data, a_scale, b_data, b_scale,
                   a_is_fp8, b_is_fp8)

    if _GEMM_BACKEND == "triton":
        result = _aiter_triton_gemm(*triton_args)
        if result is None:
            result = _aiter_ck_gemm(
                aiter, a_data, a_scale, b_data, b_scale,
                a_is_fp8, b_is_fp8, transA, transB, A, B,
            )

    else:
        # Default "ck" path
        result = _aiter_ck_gemm(
            aiter, a_data, a_scale, b_data, b_scale,
            a_is_fp8, b_is_fp8, transA, transB, A, B,
        )
        if result is None:
            result = _aiter_triton_gemm(*triton_args)

    if result is None:
        return None  # Signal caller to use PyTorch fallback

    # --- Post-GEMM epilogues ---
    if alpha != 1.0:
        result = result * alpha

    bias_grad = torch.Tensor()
    if bias is not None and bias.numel() > 0:
        if grad:
            grad_out = _dequantize_if_needed(B)
            bias_grad = grad_out.reshape(-1, grad_out.shape[-1]).sum(dim=0)
        else:
            result = result + bias

    gelu_input = torch.Tensor()
    if gelu and gelu_in is not None:
        gelu_in.copy_(result)
        gelu_input = gelu_in
        result = torch.nn.functional.gelu(result, approximate='tanh')

    if accumulate and D is not None:
        D.add_(result)
    elif D is not None:
        D.copy_(result)
    else:
        D = result

    if quantizer is not None and hasattr(quantizer, 'quantize'):
        D = quantizer.quantize(D)

    return D, bias_grad, gelu_input, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generic_gemm(A, transA, B, transB, D, quantizer, output_dtype,
                 bias, bias_type, gelu, gelu_in, grad, workspace, workspace_size,
                 accumulate, use_split_accumulator,
                 comm_overlap=None, comm_type=None, extra_output=None,
                 bulk_overlap=False, alpha=1.0, beta=None):
    """General matrix-matrix multiply with optional bias, GELU, and accumulation.

    This is the primary GEMM entry point, replacing tex.generic_gemm.
    Dispatches to AITER CK/Triton kernels when available, falls back to torch.matmul.

    Backend selection via NVTE_LITE_GEMM_BACKEND env var:
      "ck" (default), "triton", "pytorch"
    """
    # --- AITER dispatch (all precisions) ---
    if _GEMM_BACKEND != "pytorch" and is_aiter_available():
        result = _aiter_gemm(
            A, transA, B, transB, D, quantizer, output_dtype,
            bias, bias_type, gelu, gelu_in, grad, accumulate, alpha,
        )
        if result is not None:
            return result[0], result[1], result[2], extra_output

    # --- PyTorch fallback ---
    a = _dequantize_if_needed(A)
    b = _dequantize_if_needed(B)

    # cuBLAS column-major: C = op(A) @ op(B)
    # In row-major (PyTorch): C_row = B_row @ A_row (reversed operand order)
    # Typical "TN" layout: transA=True, transB=False
    #   A=[out,in] weight -> a.t()=[in,out], B=[batch,in] -> b as-is
    #   result = b @ a.t() = [batch,in] @ [in,out] = [batch,out]

    # cuBLAS GEMM treats N-D tensors as batched 2D: leading dims of B are
    # preserved in the output.  torch.matmul with 2D operands doesn't do
    # this, so we flatten to 2D, matmul, then restore B's leading dims.
    b_leading = b.shape[:-1]  # leading dims of B (before transpose)
    if a.dim() > 2:
        a = a.reshape(-1, a.shape[-1])
    if b.dim() > 2:
        b = b.reshape(-1, b.shape[-1])

    if transA:
        a = a.t()
    if transB:
        b = b.t()

    compute_dtype = torch.bfloat16
    if a.dtype == torch.float32 or b.dtype == torch.float32:
        compute_dtype = torch.float32
    elif a.dtype == torch.float16 or b.dtype == torch.float16:
        compute_dtype = torch.float16

    a = a.to(compute_dtype)
    b = b.to(compute_dtype)

    result = torch.matmul(b, a)

    # Restore B's leading dimensions in the output (cuBLAS convention)
    if len(b_leading) > 1:
        result = result.view(*b_leading, result.shape[-1])

    if alpha != 1.0:
        result = result * alpha

    bias_grad = torch.Tensor()
    if bias is not None and bias.numel() > 0:
        if grad:
            grad_out = _dequantize_if_needed(B)
            bias_grad = grad_out.reshape(-1, grad_out.shape[-1]).sum(dim=0)
        else:
            result = result + bias

    gelu_input = torch.Tensor()
    if gelu and gelu_in is not None:
        gelu_in.copy_(result)
        gelu_input = gelu_in
        result = torch.nn.functional.gelu(result, approximate='tanh')

    if accumulate and D is not None:
        D.add_(result)
    elif D is not None:
        D.copy_(result)
    else:
        D = result

    if quantizer is not None and hasattr(quantizer, 'quantize'):
        D = quantizer.quantize(D)

    return D, bias_grad, gelu_input, extra_output


def te_general_grouped_gemm(*args, **kwargs):
    """Grouped GEMM for MoE-style expert parallelism.

    Dispatches to general_grouped_gemm_triton which wraps AITER's
    gmm/ptgmm/nptgmm Triton kernels. Falls back to NotImplementedError
    if neither AITER nor the Triton GMM kernels are available.
    """
    try:
        from transformer_engine.pytorch.triton_kernels.grouped_gemm import (
            general_grouped_gemm_triton,
        )
        return general_grouped_gemm_triton(*args, **kwargs)
    except (ImportError, ModuleNotFoundError):
        raise NotImplementedError(
            "Grouped GEMM in lite mode requires AITER or Triton GMM. "
            "Install AITER (pip install amd-aiter) or use the standard GEMM path."
        )
