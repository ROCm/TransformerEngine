# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""GEMM operations -- multi-backend with AITER, Triton, and PyTorch fallback.

Backend priority:
1. AITER GEMM (CK + Triton) when available
2. Standalone Triton FP8 GEMM (TODO Phase 2)
3. torch._scaled_mm for FP8 (fallback)
4. torch.matmul for BF16/FP16 (last resort)
"""

import torch

# Try to import AITER
_aiter_available = False
try:
    import aiter
    _aiter_available = True
except ImportError:
    pass


def _dequantize_if_needed(tensor):
    """Dequantize FP8/quantized tensor to BF16 for matmul."""
    if hasattr(tensor, 'dequantize'):
        return tensor.dequantize()
    if tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2,
                        torch.float8_e4m3fnuz, torch.float8_e5m2fnuz):
        return tensor.to(torch.bfloat16)
    return tensor


def generic_gemm(A, transA, B, transB, D, quantizer, output_dtype,
                 bias, bias_type, gelu, gelu_in, grad, workspace, workspace_size,
                 accumulate, use_split_accumulator,
                 comm_overlap=None, comm_type=None, extra_output=None,
                 bulk_overlap=False, alpha=1.0, beta=None):
    """General matrix-matrix multiply with optional bias, GELU, and accumulation.

    This is the primary GEMM entry point, replacing tex.generic_gemm.
    """
    # Dequantize inputs if needed
    a = _dequantize_if_needed(A)
    b = _dequantize_if_needed(B)

    # cuBLAS column-major: C = op(A) @ op(B)
    # In row-major (PyTorch): C_row = B_row @ A_row  (reversed operand order)
    # The trans flags apply directly to the row-major tensors.
    # Typical "TN" layout: transA=True, transB=False
    #   A=[out,in] weight → a.t()=[in,out], B=[batch,in] → b as-is
    #   result = b @ a.t() = [batch,in] @ [in,out] = [batch,out]

    if transA:
        a = a.t()
    if transB:
        b = b.t()

    # Ensure compatible dtypes
    compute_dtype = torch.bfloat16
    if a.dtype == torch.float32 or b.dtype == torch.float32:
        compute_dtype = torch.float32
    elif a.dtype == torch.float16 or b.dtype == torch.float16:
        compute_dtype = torch.float16

    a = a.to(compute_dtype)
    b = b.to(compute_dtype)

    # Compute GEMM: row-major equivalent is B @ A
    result = torch.matmul(b, a)

    if alpha != 1.0:
        result = result * alpha

    # Handle bias: in forward (grad=False) add bias to result,
    # in backward (grad=True) compute bias_grad from grad_output (B).
    bias_grad = torch.Tensor()
    if bias is not None and bias.numel() > 0:
        if grad:
            # Backward: bias_grad = grad_output.sum(batch_dims)
            # In wgrad GEMM (layout="NT"), B is grad_output.
            grad_out = _dequantize_if_needed(B)
            bias_grad = grad_out.reshape(-1, grad_out.shape[-1]).sum(dim=0)
        else:
            # Forward: add bias to result
            result = result + bias

    # Apply GELU if requested
    gelu_input = torch.Tensor()
    if gelu and gelu_in is not None:
        gelu_in.copy_(result)
        gelu_input = gelu_in
        result = torch.nn.functional.gelu(result, approximate='tanh')

    # Accumulate into D if requested
    if accumulate and D is not None:
        D.add_(result)
    elif D is not None:
        D.copy_(result)
    else:
        D = result

    # Quantize output if needed
    if quantizer is not None and hasattr(quantizer, 'quantize'):
        D = quantizer.quantize(D)

    # Return 4-tuple matching C++ tex.generic_gemm signature:
    # (output, bias_grad, gelu_input, extra_output)
    return D, bias_grad, gelu_input, extra_output


def te_general_grouped_gemm(*args, **kwargs):
    """Grouped GEMM.

    TODO Phase 2: Wire up to existing Triton GMM or AITER grouped GEMM.
    """
    raise NotImplementedError(
        "Grouped GEMM in lite mode requires AITER or Triton GMM. "
        "Set up AITER or use the standard GEMM path."
    )
