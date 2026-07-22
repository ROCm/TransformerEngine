# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Minimal TE entry point for the FlyDSL MXFP8 TN backend."""

import torch
import transformer_engine_torch as tex

from .mxfp8_gemm import mxfp8_matmul


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
    """Run the FlyDSL MXFP8 kernel for TE's TN path."""
    if not transa or transb:
        raise NotImplementedError(
            "FlyDSL MXFP8 currently supports only transa=True, transb=False"
        )

    if output_dtype not in (None, tex.DType.kFloat16):
        raise NotImplementedError(
            f"FlyDSL MXFP8 currently supports only FP16 output, got {output_dtype}"
        )

    if quantizer is not None:
        raise NotImplementedError("FlyDSL MXFP8 output quantization is not implemented")

    if float(alpha) != 1.0 or float(beta) != 0.0:
        raise NotImplementedError("FlyDSL MXFP8 supports only alpha=1 and beta=0")

    if accumulate:
        raise NotImplementedError("FlyDSL MXFP8 accumulation is not implemented")

    if bias is not None and bias.numel() != 0:
        raise NotImplementedError("FlyDSL MXFP8 bias is not implemented")

    if gelu or grad:
        raise NotImplementedError("FlyDSL MXFP8 GELU/gradient epilogues are not implemented")

    # TE TN path:
    #   A rowwise payload: weight     [N, K]
    #   B rowwise payload: activation [..., K]
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
            raise ValueError("FlyDSL MXFP8 requires contiguous output storage")

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

    return D, None, None, None
