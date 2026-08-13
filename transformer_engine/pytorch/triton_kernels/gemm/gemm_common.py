# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Shared helpers for the Triton GEMM backend.

Contains dtype conversion utilities, output-shape computation, and small
helpers used by the Python-side Triton GEMM path to reconstruct rowwise
data from a columnwise-only ``Float8TensorStorage`` and to pick the
correct pre-quantized copy from an ``MXFP8TensorStorage``.
"""

import torch

import transformer_engine_torch as tex

# Reuse the shared architecture-native FP8 dtype helpers from the
# triton_kernels package to stay in sync with the norms / cast kernels.
from ..common import get_torch_e4m3_type, get_torch_e5m2_type


def is_fp8_dtype(dtype: tex.DType) -> bool:
    """Whether a TE ``DType`` is one of the FP8 variants."""
    return dtype in (tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2)


def reinterpret_as_fp8_tensor(a: torch.Tensor, dtype: tex.DType) -> torch.Tensor:
    """View a uint8 tensor as the architecture-native FP8 torch dtype.

    gfx942 (MI300/MI325) uses NANOO (``fnuz``) FP8 variants; gfx950 (MI350)
    uses OCP-standard variants. Delegates dtype selection to
    ``triton_kernels.common.get_torch_e4m3_type`` / ``_e5m2_type``.
    """
    if dtype == tex.DType.kFloat8E4M3:
        return a.view(dtype=get_torch_e4m3_type())
    if dtype == tex.DType.kFloat8E5M2:
        return a.view(dtype=get_torch_e5m2_type())

def getGemmOutputShape(A, transa, B, transb):
    """
    Compute output shape for GEMM following the C++ backend logic.

    Matches getGemmOutputShape in transformer_engine/pytorch/csrc/extensions/gemm.cpp

    Why Does This Preserve B's Batch Dimensions?
    =============================================

    This is a deliberate API design choice that makes the interface consistent and
    predictable for neural network operations.

    Usage Patterns in Linear Layer:

    1. Forward Pass (fprop) - Layout: TN (default)
       output = general_gemm(weight, input)
       - A = weight: [out_features, in_features] - no batch dims
       - B = input: [batch, seq_len, in_features] - HAS batch dims
       - Output: [batch, seq_len, out_features] - preserves B's batch

    2. Input Gradient (dgrad) - Layout: NN
       grad_input = general_gemm(weight, grad_output)
       - A = weight: [out_features, in_features] - no batch dims
       - B = grad_output: [batch, seq_len, out_features] - HAS batch dims
       - Output: [batch, seq_len, in_features] - preserves B's batch

    3. Weight Gradient (wgrad) - Layout: NT
       grad_weight = general_gemm(input, grad_output)
       - A = input: [batch, seq_len, in_features] - batch dims
       - B = grad_output: [batch, seq_len, out_features] - batch dims
       - Output: [out_features, in_features] - NO batch (transb=True)

    Key Insight:
    The calling code consistently places the tensor with desired output batch
    structure as the B operand.
    - For fprop/dgrad: B has batch dimensions -> output keeps them
    - For wgrad: Both have batch, use transb=True -> output flattens them (reduction over batch)

    Why This Convention?
    - Consistency: Always put batched activations as B
    - Predictability: Output shape always relates to B's structure
    - Simplicity: Caller controls output shape by choosing B and transb
    - Efficiency: Avoids extra reshapes in common cases

    Could It Be Different?
    Yes! The API could preserve A's batch instead, but then all calling code would
    need to swap operands. The math would work the same, just with reversed convention.
    """
    # Handle both tensors and torch.Size objects
    A_shape = A if isinstance(A, torch.Size) else A.shape
    B_shape = B if isinstance(B, torch.Size) else B.shape

    # Calculate flattened dimensions (product of all leading dims)
    A0 = product(A_shape[:-1])  # Product of all leading dims
    A1 = A_shape[-1]
    B0 = product(B_shape[:-1])
    B1 = B_shape[-1]

    # Construct output shape following C++ logic:
    # if (transb) { ret = [B1] }
    # else { ret = [B_shape[0], B_shape[1], ..., B_shape[-2]] }  // Unflatten B0
    # if (transa) { ret.append(A0) }
    # else { ret.append(A1) }

    ret = []

    # First part: from B
    if transb:
        ret.append(B1)
    else:
        # Preserve B's batch structure (all dims except last)
        for i in range(len(B_shape) - 1):
            ret.append(B_shape[i])

    # Second part: from A
    if transa:
        ret.append(A0)  # Flattened A
    else:
        ret.append(A1)  # A's last dim

    return torch.Size(ret)

def product(shape):
    ret = 1
    for i in shape:
        ret *= i
    return ret


def materialize_rowwise_from_columnwise(storage) -> torch.Tensor:
    """Reconstruct the rowwise Float8 data buffer from a columnwise-only storage.

    Inverts ``tex.fp8_transpose``'s layout: fp8_transpose treats an n-D rowwise
    tensor with shape (D0, ..., D_{n-2}, K) as 2-D (M, K) with
    M = prod(D0..D_{n-2}), transposes to (K, M), and re-shapes the result to
    [K, D0, ..., D_{n-2}]. To recover the original rowwise layout we rotate
    the leading K dim back to the tail: [K, D0, ..., D_{n-2}] -> [D0, ..., D_{n-2}, K].

    Args:
        storage: A ``Float8TensorStorage`` whose ``_transpose`` (columnwise
            data) is present and valid.

    Returns:
        A contiguous rowwise buffer with the shape the tensor would have had
        before it was columnwise-only.
    """
    cw = storage._transpose
    ndim = cw.dim()
    if ndim == 2:
        return cw.transpose(0, 1).contiguous()
    perm = list(range(1, ndim)) + [0]
    return cw.permute(*perm).contiguous()


def data_and_scale_for_transpose(storage, will_transpose: bool):
    """Pick the pre-quantized MXFP8 copy that matches a BLAS transpose flag.

    MXFP8 cannot be safely re-transposed after quantization (the block-scale
    layout is orientation-locked), so the storage keeps both rowwise and
    columnwise pre-quantized copies. This picks the right one to hand to the
    kernel.

    Args:
        storage: An ``MXFP8TensorStorage`` with both rowwise and columnwise
            data + scale populated.
        will_transpose: If True, the kernel will transpose this operand ->
            return the columnwise copy; else return the rowwise copy.

    Returns:
        ``(data, scale_inv)`` -- both ``torch.Tensor``.
    """
    if will_transpose:
        return storage._columnwise_data, storage._columnwise_scale_inv
    return storage._rowwise_data, storage._rowwise_scale_inv
