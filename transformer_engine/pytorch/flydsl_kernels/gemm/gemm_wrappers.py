# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""TE entry points for the FlyDSL GEMM backend."""

import torch
import transformer_engine_torch as tex

from .bf16_gemm import bf16_matmul
from .mxfp8_gemm import mxfp8_matmul


def _validate_common_epilogue(
    *,
    quantizer,
    bias,
    gelu,
    grad,
    accumulate,
    alpha,
    beta,
):
    """Validate features not yet implemented by the FlyDSL GEMM backend."""
    if quantizer is not None:
        raise NotImplementedError(
            "FlyDSL GEMM output quantization is not implemented"
        )

    if float(alpha) != 1.0 or float(beta) != 0.0:
        raise NotImplementedError(
            "FlyDSL GEMM currently supports only alpha=1 and beta=0"
        )

    if accumulate:
        raise NotImplementedError(
            "FlyDSL GEMM accumulation is not implemented"
        )

    if bias is not None and bias.numel() != 0:
        raise NotImplementedError(
            "FlyDSL GEMM bias is not implemented"
        )

    if gelu or grad:
        raise NotImplementedError(
            "FlyDSL GEMM GELU/gradient epilogues are not implemented"
        )


def _is_mxfp8_operand(t):
    """Return whether ``t`` exposes TE MXFP8 rowwise storage."""
    return hasattr(t, "_rowwise_data") and hasattr(t, "_rowwise_scale_inv")


def _run_mxfp8_tn(A, B, D):
    """Run the existing FlyDSL MXFP8 TN path."""
    A_data = A._rowwise_data
    A_scale = A._rowwise_scale_inv
    B_data = B._rowwise_data
    B_scale = B._rowwise_scale_inv

    if A_data is None or A_scale is None:
        raise RuntimeError("A does not contain rowwise MXFP8 data and scales")

    if B_data is None or B_scale is None:
        raise RuntimeError("B does not contain rowwise MXFP8 data and scales")

    n, k = A_data.shape
    B_flat = B_data.reshape(-1, B_data.shape[-1])
    m, kb = B_flat.shape

    if kb != k:
        raise ValueError(f"MXFP8 inner dimensions do not match: {k} and {kb}")

    A_scale = A_scale.reshape(n, -1)
    B_scale = B_scale.reshape(m, -1)
    output_shape = (*B_data.shape[:-1], n)

    if D is None:
        D = torch.empty(
            output_shape,
            dtype=torch.float16,
            device=B_data.device,
        )
    else:
        if tuple(D.shape) != output_shape:
            raise ValueError(
                f"D shape {tuple(D.shape)} does not match expected {output_shape}"
            )
        if D.dtype != torch.float16:
            raise TypeError(
                f"FlyDSL MXFP8 requires FP16 output, got {D.dtype}"
            )
        if not D.is_contiguous():
            raise ValueError(
                "FlyDSL MXFP8 requires contiguous output storage"
            )

    # Public mxfp8_matmul contract:
    #   a:       [M, K]
    #   a_scale: [M, K/32]
    #   b:       [K, N]
    #   b_scale: [N, K/32]
    #   c:       [M, N] FP16
    mxfp8_matmul(
        B_flat,
        B_scale,
        A_data.transpose(0, 1),
        A_scale,
        D.view(m, n),
    )

    return D


def _run_bf16_tn(A, B, D):
    """Run FlyDSL BF16 for TE's TN operand convention.

    TE supplies:
        A: weight     [N, K]
        B: activation [..., K]

    ``bf16_matmul`` consumes:
        a: activation [M, K]
        b: weight.T   [K, N]
        c: output     [M, N]
    """
    if not isinstance(A, torch.Tensor) or not isinstance(B, torch.Tensor):
        raise TypeError(
            "FlyDSL BF16 GEMM expects plain torch.Tensor operands"
        )

    if A.dtype != torch.bfloat16 or B.dtype != torch.bfloat16:
        raise TypeError(
            "FlyDSL BF16 GEMM requires BF16 inputs, "
            f"got A={A.dtype} and B={B.dtype}"
        )

    if A.ndim != 2:
        raise ValueError(
            f"FlyDSL BF16 TN expects weight A to be rank 2, got {tuple(A.shape)}"
        )
    if B.ndim < 2:
        raise ValueError(
            f"FlyDSL BF16 TN expects activation B to have rank >= 2, got {tuple(B.shape)}"
        )

    n, k = A.shape
    B_flat = B.reshape(-1, B.shape[-1])
    m, kb = B_flat.shape

    if kb != k:
        raise ValueError(
            f"BF16 inner dimensions do not match: A{tuple(A.shape)} and "
            f"B{tuple(B.shape)}"
        )

    output_shape = (*B.shape[:-1], n)

    if D is None:
        D = torch.empty(
            output_shape,
            dtype=torch.bfloat16,
            device=B.device,
        )
    else:
        if tuple(D.shape) != output_shape:
            raise ValueError(
                f"D shape {tuple(D.shape)} does not match expected {output_shape}"
            )
        if D.dtype != torch.bfloat16:
            raise TypeError(
                f"FlyDSL BF16 requires BF16 output, got {D.dtype}"
            )
        if D.device != B.device:
            raise ValueError(
                f"D must be on {B.device}, got {D.device}"
            )
        if not D.is_contiguous():
            raise ValueError(
                "FlyDSL BF16 requires contiguous output storage"
            )

    if A.device != B.device:
        raise ValueError(
            f"A and B must be on the same device, got {A.device} and {B.device}"
        )

    bf16_matmul(
        B_flat,
        A.transpose(0, 1),
        D.view(m, n),
    )

    return D


def te_generic_gemm_flydsl(
    A,
    transa,
    B,
    transb,
    D,
    quantizer,
    output_dtype,
    bias=None,
    bias_type=None,
    gelu=False,
    gelu_in=None,
    grad=False,
    workspace=None,
    workspaceSize=0,
    accumulate=False,
    use_split_accumulator=False,
    comm_overlap=None,
    comm_type=None,
    extra_output=None,
    bulk_overlap=False,
    alpha=1.0,
    beta=0.0,
):
    """Run a supported FlyDSL GEMM through TE's generic GEMM interface.

    Currently supported:
      - MXFP8 TN input with FP16 output
      - BF16 TN input with BF16 output
    """
    del bias_type
    del gelu_in
    del workspace
    del workspaceSize
    del use_split_accumulator
    del comm_overlap
    del comm_type
    del extra_output
    del bulk_overlap

    if not transa or transb:
        raise NotImplementedError(
            "FlyDSL GEMM currently supports only transa=True, transb=False"
        )

    _validate_common_epilogue(
        quantizer=quantizer,
        bias=bias,
        gelu=gelu,
        grad=grad,
        accumulate=accumulate,
        alpha=alpha,
        beta=beta,
    )

    a_is_mxfp8 = _is_mxfp8_operand(A)
    b_is_mxfp8 = _is_mxfp8_operand(B)

    if a_is_mxfp8 or b_is_mxfp8:
        if not (a_is_mxfp8 and b_is_mxfp8):
            raise ValueError(
                "Mixed MXFP8 and non-MXFP8 FlyDSL GEMM inputs are not supported"
            )

        if output_dtype not in (None, tex.DType.kFloat16):
            raise NotImplementedError(
                "FlyDSL MXFP8 currently supports only FP16 output, "
                f"got {output_dtype}"
            )

        D = _run_mxfp8_tn(A, B, D)
        return D, None, None, None

    if not isinstance(A, torch.Tensor) or not isinstance(B, torch.Tensor):
        raise TypeError(
            "Unsupported FlyDSL GEMM operand types: "
            f"{type(A).__name__} and {type(B).__name__}"
        )

    if A.dtype == torch.bfloat16 and B.dtype == torch.bfloat16:
        if output_dtype not in (None, tex.DType.kBFloat16):
            raise NotImplementedError(
                "FlyDSL BF16 currently supports only BF16 output, "
                f"got {output_dtype}"
            )

        D = _run_bf16_tn(A, B, D)
        return D, None, None, None

    raise NotImplementedError(
        "FlyDSL GEMM currently supports only MXFP8 or BF16 inputs; "
        f"got A={A.dtype} and B={B.dtype}"
    )
