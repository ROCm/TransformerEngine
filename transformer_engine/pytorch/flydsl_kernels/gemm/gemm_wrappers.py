# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""TE entry points for the FlyDSL GEMM backend."""

import torch
import transformer_engine_torch as tex

from transformer_engine.pytorch.utils import get_device_compute_capability

from .bf16_gemm import bf16_matmul
from .fp16_gemm import fp16_matmul
from .fp32_gemm import fp32_matmul
from .fp8_gemm import fp8_matmul
from .mxfp8_gemm import mxfp8_matmul


def reinterpret_as_fp8_tensor(
    a: torch.Tensor,
    dtype: tex.DType,
) -> torch.Tensor:
    """View TE's uint8 payload as the native torch FP8 dtype for this GPU."""
    capability = get_device_compute_capability()

    # gfx950 uses OCP FP8. gfx942 and earlier ROCm architectures use FNUZ.
    use_ocp_fp8 = capability == (9, 5)

    if dtype == tex.DType.kFloat8E4M3:
        torch_dtype = (
            torch.float8_e4m3fn
            if use_ocp_fp8
            else torch.float8_e4m3fnuz
        )
    elif dtype == tex.DType.kFloat8E5M2:
        torch_dtype = (
            torch.float8_e5m2
            if use_ocp_fp8
            else torch.float8_e5m2fnuz
        )
    else:
        raise TypeError(f"Unsupported TE FP8 dtype: {dtype}")

    return a.view(torch_dtype)

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


def _is_fp8_operand(t):
    """Return whether ``t`` is a regular TE tensor-wise FP8 operand."""
    try:
        from transformer_engine.pytorch import Float8Tensor
        from transformer_engine.pytorch.tensor.storage.float8_tensor_storage import (
            Float8TensorStorage,
        )
    except ImportError:
        return False

    return isinstance(t, (Float8Tensor, Float8TensorStorage))


def _reinterpret_fp8_payload(data, fp8_dtype, name):
    """Reinterpret TE's uint8 payload using its ``tex.DType`` metadata."""
    if data is None:
        raise RuntimeError(f"{name} does not contain the required FP8 payload")

    if fp8_dtype not in (
        tex.DType.kFloat8E4M3,
        tex.DType.kFloat8E5M2,
    ):
        raise TypeError(
            f"{name} has unsupported TE FP8 dtype metadata: {fp8_dtype}"
        )

    # TE stores Float8Tensor payloads as uint8. Use TE's shared conversion
    # helper so ROCm's correct native torch FP8 type is selected from tex.DType.
    if data.dtype == torch.uint8:
        return reinterpret_as_fp8_tensor(data, fp8_dtype)

    # A materialized payload may already have been reinterpreted. Accept it
    # only when its TE metadata is one of the recognized FP8 enum values.
    if data.element_size() == 1 and data.dtype.is_floating_point:
        return data

    raise TypeError(
        f"{name} FP8 storage must be uint8 or an already reinterpreted "
        f"one-byte floating-point tensor, got {data.dtype}"
    )


def _valid_fp8_transpose(t):
    """Return whether a TE Float8 operand has usable columnwise storage."""
    return (
        hasattr(t, "_transpose")
        and t._transpose is not None
        and not getattr(t, "_transpose_invalid", False)
    )


def _run_fp8_tn(A, B, D):
    """Run tensor-wise E4M3 x E4M3 FlyDSL FP8 for TE's TN convention.

    TE supplies:
        A: weight     [N, K], transa=True
        B: activation [..., K], transb=False

    ``fp8_matmul`` consumes:
        a: activation [M, K]
        b: weight.T   [K, N]
        c: output     [M, N]
    """
    if not (_is_fp8_operand(A) and _is_fp8_operand(B)):
        raise TypeError(
            "FlyDSL FP8 GEMM expects Float8Tensor or Float8TensorStorage operands"
        )

    a_fp8_dtype = getattr(A, "_fp8_dtype", None)
    b_fp8_dtype = getattr(B, "_fp8_dtype", None)
    if (
        a_fp8_dtype != tex.DType.kFloat8E4M3
        or b_fp8_dtype != tex.DType.kFloat8E4M3
    ):
        raise NotImplementedError(
            "The current FlyDSL FP8 kernel supports only "
            "tex.DType.kFloat8E4M3 x tex.DType.kFloat8E4M3; "
            f"got A={a_fp8_dtype} and B={b_fp8_dtype}"
        )

    # A is transposed by the TE TN call. Prefer its already-materialized
    # columnwise payload, which has the exact [K, N] layout consumed by
    # fp8_matmul. Fall back to a transpose view of rowwise [N, K] storage.
    if _valid_fp8_transpose(A):
        A_t = _reinterpret_fp8_payload(A._transpose, a_fp8_dtype, "A._transpose")
        if A_t.ndim != 2:
            raise ValueError(
                f"FlyDSL FP8 TN expects transposed weight storage to be rank 2, "
                f"got {tuple(A_t.shape)}"
            )
        k, n = A_t.shape
    else:
        A_data = _reinterpret_fp8_payload(getattr(A, "_data", None), a_fp8_dtype, "A._data")
        if A_data.ndim != 2:
            raise ValueError(
                f"FlyDSL FP8 TN expects weight A to be rank 2, "
                f"got {tuple(A_data.shape)}"
            )
        n, k = A_data.shape
        A_t = A_data.transpose(0, 1)

    # B is not transposed by TE, so rowwise storage is required. Flatten any
    # leading activation dimensions into M while retaining the K dimension.
    B_data = _reinterpret_fp8_payload(getattr(B, "_data", None), b_fp8_dtype, "B._data")
    if B_data.ndim < 2:
        raise ValueError(
            f"FlyDSL FP8 TN expects activation B to have rank >= 2, "
            f"got {tuple(B_data.shape)}"
        )

    B_flat = B_data.reshape(-1, B_data.shape[-1])
    m, kb = B_flat.shape
    if kb != k:
        raise ValueError(
            f"FP8 inner dimensions do not match: weight K={k} and "
            f"activation K={kb}"
        )

    A_scale_inv = getattr(A, "_scale_inv", None)
    B_scale_inv = getattr(B, "_scale_inv", None)
    for name, scale in (
        ("A._scale_inv", A_scale_inv),
        ("B._scale_inv", B_scale_inv),
    ):
        if not isinstance(scale, torch.Tensor):
            raise RuntimeError(f"{name} is not populated")
        if scale.dtype != torch.float32 or scale.numel() != 1:
            raise ValueError(
                f"{name} must contain exactly one FP32 tensor-wise inverse "
                f"scale, got dtype={scale.dtype}, shape={tuple(scale.shape)}"
            )

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
                f"FlyDSL FP8 requires FP16 output, got {D.dtype}"
            )
        if D.device != B_data.device:
            raise ValueError(
                f"D must be on {B_data.device}, got {D.device}"
            )
        if not D.is_contiguous():
            raise ValueError(
                "FlyDSL FP8 requires contiguous output storage"
            )

    if A_t.device != B_data.device:
        raise ValueError(
            f"A and B must be on the same device, got {A_t.device} "
            f"and {B_data.device}"
        )

    fp8_matmul(
        B_flat,
        B_scale_inv,
        A_t,
        A_scale_inv,
        D.view(m, n),
    )

    return D


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


def _run_fp16_tn(A, B, D):
    """Run FlyDSL FP16 for TE's TN operand convention."""
    if not isinstance(A, torch.Tensor) or not isinstance(B, torch.Tensor):
        raise TypeError(
            "FlyDSL FP16 GEMM expects plain torch.Tensor operands"
        )

    if A.dtype != torch.float16 or B.dtype != torch.float16:
        raise TypeError(
            "FlyDSL FP16 GEMM requires FP16 inputs, "
            f"got A={A.dtype} and B={B.dtype}"
        )

    if A.ndim != 2:
        raise ValueError(
            f"FlyDSL FP16 TN expects weight A to be rank 2, got {tuple(A.shape)}"
        )
    if B.ndim < 2:
        raise ValueError(
            f"FlyDSL FP16 TN expects activation B to have rank >= 2, got {tuple(B.shape)}"
        )

    n, k = A.shape
    B_flat = B.reshape(-1, B.shape[-1])
    m, kb = B_flat.shape

    if kb != k:
        raise ValueError(
            f"FP16 inner dimensions do not match: A{tuple(A.shape)} and "
            f"B{tuple(B.shape)}"
        )

    output_shape = (*B.shape[:-1], n)

    if D is None:
        D = torch.empty(
            output_shape,
            dtype=torch.float16,
            device=B.device,
        )
    else:
        if tuple(D.shape) != output_shape:
            raise ValueError(
                f"D shape {tuple(D.shape)} does not match expected {output_shape}"
            )
        if D.dtype != torch.float16:
            raise TypeError(
                f"FlyDSL FP16 requires FP16 output, got {D.dtype}"
            )
        if D.device != B.device:
            raise ValueError(
                f"D must be on {B.device}, got {D.device}"
            )
        if not D.is_contiguous():
            raise ValueError(
                "FlyDSL FP16 requires contiguous output storage"
            )

    if A.device != B.device:
        raise ValueError(
            f"A and B must be on the same device, got {A.device} and {B.device}"
        )
    
    fp16_matmul(
        B_flat,
        A.transpose(0, 1),
        D.view(m, n),
    )

    return D



def _run_fp32_tn(A, B, D):
    """Run FlyDSL FP32 for TE's TN operand convention."""
    if not isinstance(A, torch.Tensor) or not isinstance(B, torch.Tensor):
        raise TypeError(
            "FlyDSL FP32 GEMM expects plain torch.Tensor operands"
        )

    if A.dtype != torch.float32 or B.dtype != torch.float32:
        raise TypeError(
            "FlyDSL FP32 GEMM requires FP32 inputs, "
            f"got A={A.dtype} and B={B.dtype}"
        )

    if A.ndim != 2:
        raise ValueError(
            f"FlyDSL FP32 TN expects weight A to be rank 2, got {tuple(A.shape)}"
        )
    if B.ndim < 2:
        raise ValueError(
            f"FlyDSL FP32 TN expects activation B to have rank >= 2, got {tuple(B.shape)}"
        )

    n, k = A.shape
    B_flat = B.reshape(-1, B.shape[-1])
    m, kb = B_flat.shape

    if kb != k:
        raise ValueError(
            f"FP32 inner dimensions do not match: A{tuple(A.shape)} and "
            f"B{tuple(B.shape)}"
        )

    output_shape = (*B.shape[:-1], n)

    if D is None:
        D = torch.empty(
            output_shape,
            dtype=torch.float32,
            device=B.device,
        )
    else:
        if tuple(D.shape) != output_shape:
            raise ValueError(
                f"D shape {tuple(D.shape)} does not match expected {output_shape}"
            )
        if D.dtype != torch.float32:
            raise TypeError(
                f"FlyDSL FP32 requires FP32 output, got {D.dtype}"
            )
        if D.device != B.device:
            raise ValueError(
                f"D must be on {B.device}, got {D.device}"
            )
        if not D.is_contiguous():
            raise ValueError(
                "FlyDSL FP32 requires contiguous output storage"
            )

    if A.device != B.device:
        raise ValueError(
            f"A and B must be on the same device, got {A.device} and {B.device}"
        )

    fp32_matmul(
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
      - tensor-wise E4M3 x E4M3 FP8 TN input with FP16 output
      - BF16 TN input with BF16 output
      - FP16 TN input with FP16 output
      - FP32 TN input with FP32 output
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

    a_is_fp8 = _is_fp8_operand(A)
    b_is_fp8 = _is_fp8_operand(B)

    if a_is_fp8 or b_is_fp8:
        if not (a_is_fp8 and b_is_fp8):
            raise ValueError(
                "Mixed regular FP8 and non-FP8 FlyDSL GEMM inputs are not supported"
            )

        if output_dtype not in (None, tex.DType.kFloat16):
            raise NotImplementedError(
                "FlyDSL tensor-wise FP8 currently supports only FP16 output, "
                f"got {output_dtype}"
            )

        D = _run_fp8_tn(A, B, D)
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

    if A.dtype == torch.float16 and B.dtype == torch.float16:
        if output_dtype not in (None, tex.DType.kFloat16):
            raise NotImplementedError(
                "FlyDSL FP16 currently supports only FP16 output, "
                f"got {output_dtype}"
            )

        D = _run_fp16_tn(A, B, D)
        return D, None, None, None

    if A.dtype == torch.float32 and B.dtype == torch.float32:
        if output_dtype not in (None, tex.DType.kFloat32):
            raise NotImplementedError(
                "FlyDSL FP32 currently supports only FP32 output, "
                f"got {output_dtype}"
            )

        D = _run_fp32_tn(A, B, D)
        return D, None, None, None

    raise NotImplementedError(
        "FlyDSL GEMM currently supports only MXFP8, tensor-wise E4M3 FP8, BF16, FP16, or FP32 inputs; "
        f"got A={A.dtype} and B={B.dtype}"
    )
