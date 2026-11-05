# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""TE entry points for the FlyDSL GEMM backend."""

import os
import warnings
from dataclasses import dataclass
from typing import Callable

import torch
import transformer_engine_torch as tex

from . import grouped_gemm_utils as grouped_utils
from .exceptions import FlyDSLUnsupportedError

from .half_prec_gemm import _BLOCK_M as HALF_PREC_BLOCK_M, _BLOCK_N as HALF_PREC_BLOCK_N
from .half_prec_gemm import bf16_matmul, doGemm as half_prec_doGemm, fp16_matmul
from .fp32_gemm import fp32_matmul
from .fp8_gemm import _BLOCK_M as FP8_BLOCK_M, _BLOCK_N as FP8_BLOCK_N
from .fp8_gemm import doGemm as fp8_doGemm, fp8_matmul
from .mxfp8_gemm import BLOCK_M as MXFP8_BLOCK_M, BLOCK_N as MXFP8_BLOCK_N
from .mxfp8_gemm import SCALE_GROUP_SIZE as MXFP8_SCALE_GROUP_SIZE, mxfp8_matmul

# TODO: Some backend-independent GEMM wrapper utilities overlap with the
# Triton GEMM backend (PR#667), including operand classification, logical
# output-shape derivation, and quantized-storage inspection. Once both
# integrations stabilize, factor the genuinely common pieces into a shared
# GEMM wrapper utility module while preserving backend-specific layout and
# storage canonicalization.


def _product(shape):
    """Return the product of dimensions in ``shape``."""
    result = 1
    for dim in shape:
        result *= dim
    return result


def _get_gemm_output_shape(A, transa, B, transb) -> torch.Size:
    """Compute TE's logical GEMM output shape.

    This matches TE's generic GEMM output-shape convention: the
    physical GEMM is flattened to ``[M, N]``, while the returned tensor keeps
    B's leading dimensions when ``transb`` is false.
    """
    A_shape = A if isinstance(A, torch.Size) else A.shape
    B_shape = B if isinstance(B, torch.Size) else B.shape

    if len(A_shape) < 2 or len(B_shape) < 2:
        raise ValueError(
            "FlyDSL GEMM expects both logical operands to have rank >= 2, "
            f"got A={tuple(A_shape)} and B={tuple(B_shape)}"
        )

    A0 = _product(A_shape[:-1])
    A1 = A_shape[-1]
    B1 = B_shape[-1]

    output_shape = [B1] if transb else list(B_shape[:-1])
    output_shape.append(A0 if transa else A1)
    return torch.Size(output_shape)


def reinterpret_as_fp8_tensor(
    a: torch.Tensor,
    dtype: tex.DType,
) -> torch.Tensor:
    """View TE's uint8 payload as the native torch FP8 dtype.

    FlyDSL dispatch is gated to gfx950 (CDNA4), which supports only OCP FP8
    (e4m3fn / e5m2); the hardware has no FNUZ format. FNUZ is gfx94x-only, so
    the fnuz torch dtypes cannot occur on this path. TODO: add the FNUZ mapping
    when gfx942 support is added to the FlyDSL GEMM backends in a future PR.
    """
    if dtype == tex.DType.kFloat8E4M3:
        torch_dtype = torch.float8_e4m3fn
    elif dtype == tex.DType.kFloat8E5M2:
        torch_dtype = torch.float8_e5m2
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
    """Validate features not yet implemented by the FlyDSL GEMM backend.

    Fused forward BIAS and forward GELU_AUX are supported by all FlyDSL GEMM
    backends (mxfp8, fp8, fp16, bf16, fp32). BGRADB (fused bias gradient) and
    DGELU (fused GELU gradient) are not implemented anywhere yet.

    Unsupported configurations raise ``FlyDSLUnsupportedError`` so
    ``general_gemm`` transparently falls back to the C++ backend rather than
    crashing -- these are valid GEMM requests, just not ones FlyDSL can serve.
    """
    if quantizer is not None:
        raise FlyDSLUnsupportedError("FlyDSL GEMM output quantization is not implemented")

    if float(alpha) != 1.0 or float(beta) != 0.0:
        raise FlyDSLUnsupportedError("FlyDSL GEMM currently supports only alpha=1 and beta=0")

    # TODO: Add accumulate option
    if accumulate:
        raise FlyDSLUnsupportedError("FlyDSL GEMM accumulation is not implemented")

    # Forward BIAS is implemented across all backends; the fused bias-gradient
    # (BGRADB) path is not. TODO: add BGRADB to the FlyDSL GEMM backends.
    if grad and bias is not None and bias.numel() != 0:
        raise FlyDSLUnsupportedError("FlyDSL GEMM fused bias gradient (BGRADB) is not implemented")

    # Fused forward GELU (GELU_AUX) is supported; the backward fused GELU
    # gradient (DGELU) is not. TODO: add DGELU to the FlyDSL GEMM backends.
    if gelu and grad:
        raise FlyDSLUnsupportedError("FlyDSL GEMM fused GELU gradient (DGELU) is not implemented")


def _resolve_bias(bias, n):
    """Normalize a TE bias into the kernel's ``(epilogue, bias_arg)`` contract.

    FlyDSL kernels index bias by the output-feature (N) axis and expect a
    contiguous length-N fp32 vector. TE may hand bias in the output dtype.
    Returns ``("DEFAULT", None)`` when no bias is present.
    """
    if bias is None or bias.numel() == 0:
        return "DEFAULT", None
    if bias.numel() != n:
        raise ValueError(f"FlyDSL GEMM bias length {bias.numel()} != N (out_features) {n}")
    return "BIAS", bias.reshape(-1).to(torch.float32).contiguous()


def _resolve_epilogue(bias, n, gelu=False):
    """Resolve the ``(epilogue, bias_arg)`` contract including fused GELU.

    Combines the bias vector (see :func:`_resolve_bias`) with an optional
    forward GELU into one of DEFAULT / BIAS / GELU_AUX / GELU_AUX_BIAS. The
    GELU epilogues additionally produce a pre-activation aux output, which the
    caller allocates and passes to the kernel.
    """
    bias_epilogue, bias_arg = _resolve_bias(bias, n)
    has_bias = bias_epilogue == "BIAS"
    if gelu:
        return ("GELU_AUX_BIAS" if has_bias else "GELU_AUX"), bias_arg
    return bias_epilogue, bias_arg


def _classify_input(t):
    """Classify a GEMM operand for the FlyDSL backend."""
    try:
        from transformer_engine.pytorch.float8_tensor import Float8Tensor
        from transformer_engine.pytorch.tensor.storage.float8_tensor_storage import (
            Float8TensorStorage,
        )

        if isinstance(t, (Float8Tensor, Float8TensorStorage)):
            return "fp8", t
    except ImportError:
        pass

    try:
        from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor
        from transformer_engine.pytorch.tensor.storage.mxfp8_tensor_storage import (
            MXFP8TensorStorage,
        )

        if isinstance(t, (MXFP8Tensor, MXFP8TensorStorage)):
            return "mxfp8", t
    except ImportError:
        pass

    try:
        from transformer_engine.pytorch.quantized_tensor import (
            QuantizedTensorStorage,
        )

        if isinstance(t, QuantizedTensorStorage):
            raise ValueError(
                "The FlyDSL GEMM backend does not support "
                f"{type(t).__name__}. Only Float8Tensor / "
                "Float8TensorStorage and MXFP8Tensor / "
                "MXFP8TensorStorage are implemented."
            )
    except ImportError:
        pass

    return "regular", None


def _reinterpret_fp8_payload(data, fp8_dtype, name):
    """Reinterpret TE's uint8 payload using its ``tex.DType`` metadata."""
    if data is None:
        raise FlyDSLUnsupportedError(f"{name} does not contain the required FP8 payload")

    if fp8_dtype not in (
        tex.DType.kFloat8E4M3,
        tex.DType.kFloat8E5M2,
    ):
        raise TypeError(f"{name} has unsupported TE FP8 dtype metadata: {fp8_dtype}")

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


def _mxfp8_debug_enabled() -> bool:
    value = os.getenv("DEBUG_FLYDSL_MXFP8_GEMM", "")
    return value.lower() not in ("", "0", "false", "no", "off")


def _mxfp8_debug(message: str) -> None:
    if _mxfp8_debug_enabled():
        print(f"[DEBUG_FLYDSL_MXFP8_GEMM] {message}")


def _fp8_debug_enabled() -> bool:
    value = os.getenv("DEBUG_FLYDSL_FP8_GEMM", "")
    return value.lower() not in ("", "0", "false", "no", "off")


def _fp8_debug(message: str) -> None:
    if _fp8_debug_enabled():
        print(f"[DEBUG_FLYDSL_FP8_GEMM] {message}")


def _fp8_tensor_debug(name: str, tensor: torch.Tensor) -> None:
    if not _fp8_debug_enabled():
        return
    _fp8_debug(
        f"{name}: shape={tuple(tensor.shape)}, stride={tuple(tensor.stride())}, "
        f"dtype={tensor.dtype}, device={tensor.device}, "
        f"contiguous={tensor.is_contiguous()}, data_ptr=0x{tensor.data_ptr():x}"
    )


def _fp8_scale_debug(name: str, scale: torch.Tensor) -> None:
    if not _fp8_debug_enabled():
        return
    value = scale.detach().float().reshape(-1).cpu().tolist()
    _fp8_debug(
        f"{name}: shape={tuple(scale.shape)}, dtype={scale.dtype}, "
        f"device={scale.device}, data_ptr=0x{scale.data_ptr():x}, value={value}"
    )


def _flatten_rowwise(t: torch.Tensor, name: str) -> torch.Tensor:
    """Flatten all leading dimensions while preserving the final dimension."""
    if t.ndim < 2:
        raise ValueError(f"FlyDSL GEMM expects {name} to have rank >= 2, got {tuple(t.shape)}")
    return t.reshape(-1, t.shape[-1])


def _flatten_columnwise(t: torch.Tensor, name: str) -> torch.Tensor:
    """Flatten TE columnwise storage while preserving its leading dimension."""
    if t.ndim < 2:
        raise ValueError(f"FlyDSL GEMM expects {name} to have rank >= 2, got {tuple(t.shape)}")
    return t.reshape(t.shape[0], -1)


def _validate_or_allocate_output(
    D,
    *,
    shape,
    dtype,
    device,
    backend_name,
):
    if D is None:
        return torch.empty(shape, dtype=dtype, device=device)

    if tuple(D.shape) != tuple(shape):
        raise ValueError(f"D shape {tuple(D.shape)} does not match expected {tuple(shape)}")
    if D.dtype != dtype:
        raise FlyDSLUnsupportedError(
            f"FlyDSL {backend_name} requires {dtype} output, got {D.dtype}"
        )
    if D.device != device:
        raise ValueError(f"D must be on {device}, got {D.device}")
    if not D.is_contiguous():
        raise FlyDSLUnsupportedError(f"FlyDSL {backend_name} requires contiguous output storage")
    return D


def _run_bf16_gemm(
    A,
    transa,
    B,
    transb,
    D,
    *,
    output_dtype: torch.dtype,
    bias=None,
    gelu=False,
):
    """Dispatch BF16 using the original row-major operand allocations.

    No operand transpose view is created:

        TN: kernel A = TE B [M,K], kernel B = TE A [N,K]
        NN: kernel A = TE B [M,K], kernel B = TE A [K,N]
        NT: kernel A = TE B [K,M], kernel B = TE A [K,N]
    """
    if not isinstance(A, torch.Tensor) or not isinstance(B, torch.Tensor):
        raise TypeError("FlyDSL BF16 GEMM expects plain torch.Tensor operands")
    if A.dtype != torch.bfloat16 or B.dtype != torch.bfloat16:
        raise TypeError(
            f"FlyDSL BF16 GEMM requires torch.bfloat16 inputs, got A={A.dtype} and B={B.dtype}"
        )
    if A.device != B.device:
        raise ValueError(f"A and B must be on the same device, got {A.device} and {B.device}")

    dispatch = {
        (True, False): "TN",
        (False, False): "NN",
        (False, True): "NT",
    }
    try:
        layout = dispatch[(bool(transa), bool(transb))]
    except KeyError as exc:
        raise FlyDSLUnsupportedError(
            "FlyDSL GEMM does not support transa=True, transb=True (TT)"
        ) from exc

    output_shape = _get_gemm_output_shape(A, transa, B, transb)

    # Preserve the original row-major storage. This only collapses leading
    # batch dimensions, matching the wrapper's existing regular-GEMM contract.
    A_data = _flatten_rowwise(A, "A")
    B_data = _flatten_rowwise(B, "B")

    # Kernel ownership is always swapped relative to TE's BLAS arguments.
    a_flydsl = B_data
    b_flydsl = A_data

    if layout == "TN":
        m, k = a_flydsl.shape
        n, kb = b_flydsl.shape
        expected_a = (m, k)
        expected_b = (n, k)
    elif layout == "NN":
        m, k = a_flydsl.shape
        kb, n = b_flydsl.shape
        expected_a = (m, k)
        expected_b = (k, n)
    else:
        k, m = a_flydsl.shape
        kb, n = b_flydsl.shape
        expected_a = (k, m)
        expected_b = (k, n)

    if kb != k:
        raise FlyDSLUnsupportedError(
            f"FlyDSL BF16 {layout} received incompatible row-major operands: "
            f"a={tuple(a_flydsl.shape)}, b={tuple(b_flydsl.shape)}"
        )
    if tuple(a_flydsl.shape) != expected_a or tuple(b_flydsl.shape) != expected_b:
        raise FlyDSLUnsupportedError(
            f"FlyDSL BF16 {layout} physical contract mismatch: "
            f"a={tuple(a_flydsl.shape)} expected={expected_a}; "
            f"b={tuple(b_flydsl.shape)} expected={expected_b}"
        )

    if _product(output_shape) != m * n:
        raise RuntimeError(
            f"FlyDSL BF16 logical output shape {tuple(output_shape)} "
            f"does not match kernel output {(m, n)}"
        )

    D = _validate_or_allocate_output(
        D,
        shape=output_shape,
        dtype=output_dtype,
        device=A.device,
        backend_name=f"BF16 {layout}",
    )

    epilogue, bias_arg = _resolve_epilogue(bias, n, gelu=gelu)
    aux = torch.empty_like(D) if gelu else None
    bf16_matmul(
        a_flydsl,
        b_flydsl,
        D.view(m, n),
        layout=layout,
        m=m,
        n=n,
        k=k,
        epilogue=epilogue,
        bias=bias_arg,
        aux=aux.view(m, n) if aux is not None else None,
    )
    return D, aux


def _run_fp16_gemm(
    A,
    transa,
    B,
    transb,
    D,
    *,
    output_dtype: torch.dtype,
    bias=None,
    gelu=False,
):
    """Dispatch FP16 using the original row-major operand allocations.

    No operand transpose view is created:

        TN: kernel A = TE B [M,K], kernel B = TE A [N,K]
        NN: kernel A = TE B [M,K], kernel B = TE A [K,N]
        NT: kernel A = TE B [K,M], kernel B = TE A [K,N]
    """
    if not isinstance(A, torch.Tensor) or not isinstance(B, torch.Tensor):
        raise TypeError("FlyDSL FP16 GEMM expects plain torch.Tensor operands")
    if A.dtype != torch.float16 or B.dtype != torch.float16:
        raise TypeError(
            f"FlyDSL FP16 GEMM requires torch.float16 inputs, got A={A.dtype} and B={B.dtype}"
        )
    if A.device != B.device:
        raise ValueError(f"A and B must be on the same device, got {A.device} and {B.device}")

    dispatch = {
        (True, False): "TN",
        (False, False): "NN",
        (False, True): "NT",
    }
    try:
        layout = dispatch[(bool(transa), bool(transb))]
    except KeyError as exc:
        raise FlyDSLUnsupportedError(
            "FlyDSL GEMM does not support transa=True, transb=True (TT)"
        ) from exc

    output_shape = _get_gemm_output_shape(A, transa, B, transb)

    # Preserve the original row-major storage. This only collapses leading
    # batch dimensions, matching the wrapper's existing regular-GEMM contract.
    A_data = _flatten_rowwise(A, "A")
    B_data = _flatten_rowwise(B, "B")

    # Kernel ownership is always swapped relative to TE's BLAS arguments.
    a_flydsl = B_data
    b_flydsl = A_data

    if layout == "TN":
        m, k = a_flydsl.shape
        n, kb = b_flydsl.shape
        expected_a = (m, k)
        expected_b = (n, k)
    elif layout == "NN":
        m, k = a_flydsl.shape
        kb, n = b_flydsl.shape
        expected_a = (m, k)
        expected_b = (k, n)
    else:
        k, m = a_flydsl.shape
        kb, n = b_flydsl.shape
        expected_a = (k, m)
        expected_b = (k, n)

    if kb != k:
        raise FlyDSLUnsupportedError(
            f"FlyDSL FP16 {layout} received incompatible row-major operands: "
            f"a={tuple(a_flydsl.shape)}, b={tuple(b_flydsl.shape)}"
        )
    if tuple(a_flydsl.shape) != expected_a or tuple(b_flydsl.shape) != expected_b:
        raise FlyDSLUnsupportedError(
            f"FlyDSL FP16 {layout} physical contract mismatch: "
            f"a={tuple(a_flydsl.shape)} expected={expected_a}; "
            f"b={tuple(b_flydsl.shape)} expected={expected_b}"
        )

    if _product(output_shape) != m * n:
        raise RuntimeError(
            f"FlyDSL FP16 logical output shape {tuple(output_shape)} "
            f"does not match kernel output {(m, n)}"
        )

    D = _validate_or_allocate_output(
        D,
        shape=output_shape,
        dtype=output_dtype,
        device=A.device,
        backend_name=f"FP16 {layout}",
    )

    epilogue, bias_arg = _resolve_epilogue(bias, n, gelu=gelu)
    aux = torch.empty_like(D) if gelu else None
    fp16_matmul(
        a_flydsl,
        b_flydsl,
        D.view(m, n),
        layout=layout,
        m=m,
        n=n,
        k=k,
        epilogue=epilogue,
        bias=bias_arg,
        aux=aux.view(m, n) if aux is not None else None,
    )
    return D, aux


def _run_fp32_gemm(
    A,
    transa,
    B,
    transb,
    D,
    *,
    bias=None,
    gelu=False,
):
    """Normalize FP32 TN/NN/NT inputs to the current kernel's TN interface.

    The private FP32 core is TN-only and, like every other dtype backend,
    consumes its operands as:

        a_tn: [M, K]
        b_tn: [N, K]

    TE provides BLAS-shaped operands, so ownership is swapped and only the
    operands whose BLAS transpose flags require it are materialized:

        TN: a_tn = B
            b_tn = A

        NN: a_tn = B
            b_tn = A.T

        NT: a_tn = B.T
            b_tn = A.T

    ``transpose(...).contiguous()`` is therefore used only for the FP32
    operands that are not already in the TN kernel orientation. The core
    performs no internal transpose. BF16/FP16/FP8/MXFP8 dispatch is unchanged.
    """
    if not isinstance(A, torch.Tensor) or not isinstance(B, torch.Tensor):
        raise TypeError("FlyDSL FP32 GEMM expects plain torch.Tensor operands")
    if A.dtype != torch.float32 or B.dtype != torch.float32:
        raise TypeError(
            f"FlyDSL FP32 GEMM requires torch.float32 inputs, got A={A.dtype} and B={B.dtype}"
        )
    if A.device != B.device:
        raise ValueError(f"A and B must be on the same device, got {A.device} and {B.device}")
    if bool(transa) and bool(transb):
        raise FlyDSLUnsupportedError("FlyDSL GEMM does not support transa=True, transb=True (TT)")

    output_shape = _get_gemm_output_shape(A, transa, B, transb)

    A_flat = _flatten_rowwise(A, "A")
    B_flat = _flatten_rowwise(B, "B")

    # Swap TE's BLAS operands into the core's TN ownership and materialize the
    # NN/NT layouts into the [M, K] x [N, K] kernel contract with explicit
    # transpose copies here in the wrapper. The core (like every other dtype
    # backend) consumes b as [N, K] directly and performs no internal transpose.
    # TODO: Optimize FP32 NN/NT execution.
    if bool(transb):
        a_tn = B_flat.transpose(0, 1).contiguous()
    else:
        a_tn = B_flat

    if bool(transa):
        b_tn = A_flat
    else:
        b_tn = A_flat.transpose(0, 1).contiguous()

    if not a_tn.is_contiguous():
        a_tn = a_tn.contiguous()
    if not b_tn.is_contiguous():
        b_tn = b_tn.contiguous()

    if a_tn.ndim != 2 or b_tn.ndim != 2:
        raise RuntimeError(
            "FlyDSL FP32 TN normalization produced rank mismatch: "
            f"a={tuple(a_tn.shape)}, b={tuple(b_tn.shape)}"
        )

    m, k = a_tn.shape
    n, kb = b_tn.shape
    if kb != k:
        layout = f"{'T' if transa else 'N'}{'T' if transb else 'N'}"
        raise FlyDSLUnsupportedError(
            f"FlyDSL FP32 {layout} could not normalize to TN: "
            f"a_tn={tuple(a_tn.shape)}, b_tn={tuple(b_tn.shape)}"
        )

    if _product(output_shape) != m * n:
        raise RuntimeError(
            f"FlyDSL FP32 logical output shape {tuple(output_shape)} "
            f"does not match normalized TN output {(m, n)}"
        )

    D = _validate_or_allocate_output(
        D,
        shape=output_shape,
        dtype=torch.float32,
        device=A.device,
        backend_name="FP32 via TN core",
    )

    epilogue, bias_arg = _resolve_epilogue(bias, n, gelu=gelu)
    aux = torch.empty_like(D) if gelu else None
    fp32_matmul(
        a_tn,
        b_tn,
        D.view(m, n),
        epilogue=epilogue,
        bias=bias_arg,
        aux=aux.view(m, n) if aux is not None else None,
    )
    return D, aux


def _get_fp8_rowwise_payload(t, name):
    """Return TE's existing rowwise ``_data`` payload without copying."""
    data = getattr(t, "_data", None)
    if data is None:
        raise FlyDSLUnsupportedError(f"FlyDSL FP8 requires existing {name} rowwise (_data) storage")
    return _reinterpret_fp8_payload(
        data,
        getattr(t, "_fp8_dtype", None),
        f"{name}._data",
    )


def _get_fp8_columnwise_payload(t, name):
    """Return TE's existing columnwise ``_transpose`` payload without copying."""
    if not _valid_fp8_transpose(t):
        raise FlyDSLUnsupportedError(
            f"FlyDSL FP8 requires valid {name} columnwise (_transpose) storage"
        )
    return _reinterpret_fp8_payload(
        t._transpose,
        getattr(t, "_fp8_dtype", None),
        f"{name}._transpose",
    )


def _validate_fp8_kernel_operands(
    kernel_a,
    kernel_b,
    *,
    layout,
    a_storage,
    b_storage,
):
    """Validate zero-copy physical operands before launching an FP8 kernel."""
    if kernel_a.ndim != 2 or kernel_b.ndim != 2:
        raise ValueError(
            f"FlyDSL FP8 {layout} expects rank-2 kernel operands, got "
            f"{a_storage}={tuple(kernel_a.shape)} and "
            f"{b_storage}={tuple(kernel_b.shape)}"
        )
    if not kernel_a.is_contiguous() or not kernel_b.is_contiguous():
        raise ValueError(
            f"FlyDSL FP8 {layout} requires contiguous {a_storage} and "
            f"{b_storage}; refusing to materialize replacement operands"
        )
    if kernel_a.device != kernel_b.device:
        raise ValueError(
            f"FlyDSL FP8 {layout} operands must be on the same device, got "
            f"{kernel_a.device} and {kernel_b.device}"
        )


def _fp8_output_shape(D, m, n):
    """Preserve TE's logical output shape when D is preallocated."""
    output_shape = D.shape if D is not None else torch.Size((m, n))
    if _product(output_shape) != m * n:
        raise RuntimeError(
            f"FlyDSL FP8 logical output shape {tuple(output_shape)} does not "
            f"match flattened kernel shape {(m, n)}"
        )
    return output_shape


def _select_mxfp8_data_and_scale(
    t,
    *,
    will_transpose: bool,
    name: str,
):
    """Select the TE MXFP8 representation required by BLAS semantics."""
    if will_transpose:
        data = getattr(t, "_columnwise_data", None)
        scale = getattr(t, "_columnwise_scale_inv", None)
        orientation = "columnwise"
    else:
        data = getattr(t, "_rowwise_data", None)
        scale = getattr(t, "_rowwise_scale_inv", None)
        orientation = "rowwise"

    _mxfp8_debug(
        f"{name}: will_transpose={will_transpose}, "
        f"selected={orientation}, data_present={data is not None}, "
        f"scale_present={scale is not None}"
    )

    if data is None or scale is None:
        raise RuntimeError(f"{name} does not contain required {orientation} MXFP8 data and scales")

    _mxfp8_debug(
        f"{name} selected data shape={tuple(data.shape)}, "
        f"dtype={data.dtype}, stride={tuple(data.stride())}; "
        f"scale shape={tuple(scale.shape)}, dtype={scale.dtype}, "
        f"stride={tuple(scale.stride())}"
    )
    return data, scale


def _mxfp8_logical_shape(t, name: str) -> torch.Size:
    """Return MXFP8 logical shape from a populated backing tensor.

    MXFP8TensorStorage is not a torch.Tensor and does not expose ``.shape``.
    Rowwise and columnwise MXFP8 payloads retain the same logical row-major
    shape, so either populated backing is sufficient for shape derivation.
    """
    data = getattr(t, "_rowwise_data", None)
    if data is None:
        data = getattr(t, "_columnwise_data", None)
    if data is None:
        raise FlyDSLUnsupportedError(f"{name} has neither rowwise nor columnwise MXFP8 data")
    return torch.Size(data.shape)


def _flatten_mxfp8_scale(
    t: torch.Tensor,
    name: str,
    *,
    source_colwise: bool,
) -> torch.Tensor:
    """Flatten a raw TE MXFP8 scale tensor without changing orientation.

    Rowwise source:
        [..., K/32] -> [outer, K/32]

    Columnwise source:
        [K/32, ...] -> [K/32, outer]
    """
    if t.ndim < 2:
        raise ValueError(f"FlyDSL MXFP8 expects {name} scale rank >= 2, got {tuple(t.shape)}")

    original_shape = tuple(t.shape)
    if source_colwise:
        t = t.reshape(t.shape[0], -1)
        orientation = "columnwise"
    else:
        t = t.reshape(-1, t.shape[-1])
        orientation = "rowwise"

    _mxfp8_debug(
        f"{name} {orientation} scale flatten: "
        f"{original_shape} -> {tuple(t.shape)}, "
        f"contiguous={t.is_contiguous()}"
    )
    return t


def _run_mxfp8(
    A,
    transa,
    B,
    transb,
    D,
    *,
    output_dtype: torch.dtype,
    bias=None,
    gelu=False,
):
    """Dispatch MXFP8 through exact TN/NN/NT physical contracts.

    TE owns BLAS-shaped operands. After the usual ownership swap, FlyDSL
    kernels consume the selected backing directly:

        TN: a = B.rowwise      [M, K]
            b = A.rowwise      [N, K]

        NN: a = B.rowwise      [M, K]
            b = A.columnwise   [K, N]

        NT: a = B.columnwise   [K, M]
            b = A.columnwise   [K, N]

    MXFP8 rowwise and columnwise payloads retain the same logical row-major
    shape. Columnwise selection changes the quantization axis; the specialized
    NN/NT kernels provide the required transpose-read semantics.
    """
    a_fp8_dtype = getattr(A, "_fp8_dtype", None)
    b_fp8_dtype = getattr(B, "_fp8_dtype", None)
    supported_fp8_dtypes = (
        tex.DType.kFloat8E4M3,
        tex.DType.kFloat8E5M2,
    )
    if a_fp8_dtype not in supported_fp8_dtypes or b_fp8_dtype not in supported_fp8_dtypes:
        raise FlyDSLUnsupportedError(
            "FlyDSL MXFP8 supports E4M3 and E5M2 independently for A/B; "
            f"got A={a_fp8_dtype} and B={b_fp8_dtype}"
        )

    layout = f"{'T' if transa else 'N'}{'T' if transb else 'N'}"
    dispatch = {
        (True, False): "TN",
        (False, False): "NN",
        (False, True): "NT",
    }
    try:
        kernel_layout = dispatch[(bool(transa), bool(transb))]
    except KeyError as exc:
        raise FlyDSLUnsupportedError(
            "FlyDSL GEMM does not support transa=True, transb=True (TT)"
        ) from exc

    _mxfp8_debug(
        f"entry: layout={layout}, common_kernel="
        f"{mxfp8_matmul.__module__}.{mxfp8_matmul.__name__}, "
        f"A_type={type(A).__name__}, B_type={type(B).__name__}, "
        f"D_provided={D is not None}"
    )

    # Resolve public shapes from actual payload tensors. Never access
    # MXFP8TensorStorage.shape: the storage wrapper has no such attribute.
    A_logical_shape = _mxfp8_logical_shape(A, "A")
    B_logical_shape = _mxfp8_logical_shape(B, "B")

    # Match TE/C++ MXFP8 representation selection exactly:
    #   A: transa=True  -> rowwise;  transa=False -> columnwise
    #   B: transb=False -> rowwise;  transb=True  -> columnwise
    A_source_colwise = not bool(transa)
    B_source_colwise = bool(transb)

    A_data, A_scale = _select_mxfp8_data_and_scale(
        A,
        will_transpose=A_source_colwise,
        name="A",
    )
    B_data, B_scale = _select_mxfp8_data_and_scale(
        B,
        will_transpose=B_source_colwise,
        name="B",
    )

    # Both MXFP8 payload orientations are stored row-major with the original
    # logical shape. Flatten leading dimensions only; do not transpose or
    # materialize selected columnwise payloads.
    A_data = _flatten_rowwise(A_data, "A MXFP8 payload")
    B_data = _flatten_rowwise(B_data, "B MXFP8 payload")

    if A_data.dtype == torch.uint8:
        A_data = reinterpret_as_fp8_tensor(A_data, a_fp8_dtype)
    if B_data.dtype == torch.uint8:
        B_data = reinterpret_as_fp8_tensor(B_data, b_fp8_dtype)

    A_scale = _flatten_mxfp8_scale(
        A_scale,
        "A",
        source_colwise=A_source_colwise,
    )
    B_scale = _flatten_mxfp8_scale(
        B_scale,
        "B",
        source_colwise=B_source_colwise,
    )

    # Kernel operand ownership is always swapped relative to TE:
    #   kernel a <- TE B
    #   kernel b <- TE A
    if kernel_layout == "TN":
        # Selected rowwise backings already match the TN normal-read contract:
        #   a [M,K], b [N,K]
        a_flydsl = B_data
        b_flydsl = A_data
        a_scale = B_scale
        b_scale = A_scale

        m, k = a_flydsl.shape
        n, kb = b_flydsl.shape
        expected_a_scale = (m, k // 32)
        expected_b_scale = (n, k // 32)

    elif kernel_layout == "NN":
        # A's columnwise MXFP8 payload is still row-major in its original
        # shape, which is exactly the NN kernel's K-major [K,N] source.
        a_flydsl = B_data
        b_flydsl = A_data
        a_scale = B_scale
        b_scale = A_scale

        m, k = a_flydsl.shape
        kb, n = b_flydsl.shape
        expected_a_scale = (m, k // 32)
        expected_b_scale = (k // 32, n)

    else:
        # Both selected columnwise payloads directly satisfy the NT kernel's
        # K-major contracts without tensor transposes or copies.
        a_flydsl = B_data
        b_flydsl = A_data
        a_scale = B_scale
        b_scale = A_scale

        k, m = a_flydsl.shape
        kb, n = b_flydsl.shape
        expected_a_scale = (k // 32, m)
        expected_b_scale = (k // 32, n)

    if kb != k:
        raise FlyDSLUnsupportedError(
            f"FlyDSL MXFP8 {layout} selected incompatible payloads: "
            f"a={tuple(a_flydsl.shape)} and b={tuple(b_flydsl.shape)}"
        )

    if a_flydsl.device != b_flydsl.device:
        raise ValueError(
            f"FlyDSL MXFP8 {layout} operands must be on the same device, got "
            f"{a_flydsl.device} and {b_flydsl.device}"
        )

    if k % 32 != 0:
        raise FlyDSLUnsupportedError(f"K={k} must be divisible by MXFP8 scale group size 32")

    if tuple(a_scale.shape) != expected_a_scale:
        raise ValueError(
            f"FlyDSL MXFP8 {layout} a_scale shape "
            f"{tuple(a_scale.shape)} != expected {expected_a_scale}"
        )
    if tuple(b_scale.shape) != expected_b_scale:
        raise ValueError(
            f"FlyDSL MXFP8 {layout} b_scale shape "
            f"{tuple(b_scale.shape)} != expected {expected_b_scale}"
        )
    if a_scale.dtype != torch.uint8 or b_scale.dtype != torch.uint8:
        raise TypeError("FlyDSL MXFP8 expects raw E8M0 scales as torch.uint8")

    # Derive the public result shape from payload shapes, not storage wrappers.
    if D is not None:
        output_shape = torch.Size(D.shape)
    else:
        output_shape = _get_gemm_output_shape(
            A_logical_shape,
            transa,
            B_logical_shape,
            transb,
        )
    if _product(output_shape) != m * n:
        raise FlyDSLUnsupportedError(
            f"FlyDSL MXFP8 {layout} logical output shape "
            f"{tuple(output_shape)} does not match kernel output {(m, n)}"
        )

    D = _validate_or_allocate_output(
        D,
        shape=output_shape,
        dtype=output_dtype,
        device=a_flydsl.device,
        backend_name=f"MXFP8 {kernel_layout}",
    )

    _mxfp8_debug(
        f"dispatch layout={layout}: "
        f"a={tuple(a_flydsl.shape)}, stride={tuple(a_flydsl.stride())}; "
        f"b={tuple(b_flydsl.shape)}, stride={tuple(b_flydsl.stride())}; "
        f"a_scale={tuple(a_scale.shape)}; "
        f"b_scale={tuple(b_scale.shape)}; "
        f"M={m}, N={n}, K={k}"
    )

    epilogue, bias_arg = _resolve_epilogue(bias, n, gelu=gelu)
    # GELU_AUX modes emit a pre-activation aux output (M x N, output dtype)
    # for the backward pass; the wrapper allocates it and the kernel fills it
    # in place. Return it so the dispatcher can hand it back as gelu_input.
    aux = torch.empty_like(D) if gelu else None
    mxfp8_matmul(
        a_flydsl,
        a_scale,
        b_flydsl,
        b_scale,
        D.view(m, n),
        layout=kernel_layout,
        epilogue=epilogue,
        bias=bias_arg,
        aux=aux.view(m, n) if aux is not None else None,
    )
    return D, aux


# ---------------------------------------------------------------------------------------
# Grouped (MoE) GEMM: one persistent launch spans every expert.
#
# general_grouped_gemm_flydsl is a drop-in for cpp_extensions.general_grouped_gemm. The
# layout selects the phase (TN forward, NN dgrad, NT wgrad) and the operands' storage
# selects the kernel family. A _GroupedKernel describes the only parts that differ per
# family -- how an operand's payload and scale are read and how the kernel is launched --
# while output validation, the descriptor queue, schedule choice, and the wgrad
# K-bucketing and scatter are shared by every family.
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _GroupedKernel:
    """How one FlyDSL kernel family reads grouped operands and launches."""

    name: str
    block_m: int
    block_n: int
    # tensor -> (payload, scale) whose contraction runs along the payload's last dim.
    rows: Callable
    # tensor -> (payload, scale) read the other way: dgrad's B and both wgrad operands.
    cols: Callable
    # (per-expert scales, device, k_pad) -> the kernel's scale argument. ``k_pad`` is
    # the wgrad bucket's padded token count when the operands were padded, else None.
    scale_arg: Callable
    # (a, a_scale, b, b_scale, D, *, layout, m, n, k, **grouped) -> None
    launch: Callable
    # Kernel layout for dgrad: NN, or TN when cols() yields a materialized transpose.
    dgrad_layout: str
    # Axis of the wgrad operands (as returned by cols) that holds the tokens.
    wgrad_token_axis: int


def _mxfp8_scale_arg(scales, device, k_pad):
    # Per-expert E8M0 scales, packed by the kernel wrapper; a padded wgrad bucket pads
    # them to the bucket's token count as well.
    if k_pad is None:
        return list(scales)
    return grouped_utils.uniform_pad_concat(list(scales), k_pad // MXFP8_SCALE_GROUP_SIZE)


def _launch_mxfp8(a, a_scale, b, b_scale, D, *, layout, m, n, k, **grouped):
    mxfp8_matmul(a, a_scale, b, b_scale, D, layout=layout, **grouped)


def _fp8_transpose(t):
    """The materialized transpose of FP8 operand ``t`` (the TN-only kernel reads it)."""
    if t._transpose is None or t._transpose_invalid:
        raise FlyDSLUnsupportedError("FP8 operand has no valid transpose")
    return t._transpose


def _launch_fp8(a, a_scale, b, b_scale, D, *, layout, m, n, k, **grouped):
    # Tensor-wise FP8 is TN-only; dgrad and wgrad reach it through transposed operands.
    fp8_doGemm(a, b, D, a_scale, b_scale, m=m, n=n, k=k, **grouped)


def _as_rows(t):
    """``t`` as 2-D rows; GroupedLinear's per-expert tensors already are, so skip the view."""
    return t if t.dim() == 2 else t.reshape(-1, t.shape[-1])


def _launch_half(a, a_scale, b, b_scale, D, *, layout, m, n, k, **grouped):
    # BF16 and FP16 share this kernel; the MFMA opcode is selected per operand dtype.
    half_prec_doGemm(
        a,
        b,
        D,
        layout=layout,
        m=m,
        n=n,
        k=k,
        input_dtype=a.dtype,
        label="half-precision grouped",
        **grouped,
    )


_MXFP8_GROUPED = _GroupedKernel(
    name="MXFP8",
    block_m=MXFP8_BLOCK_M,
    block_n=MXFP8_BLOCK_N,
    rows=lambda t: (t._rowwise_data, t._rowwise_scale_inv),
    cols=lambda t: (t._columnwise_data, t._columnwise_scale_inv),
    scale_arg=_mxfp8_scale_arg,
    launch=_launch_mxfp8,
    dgrad_layout="NN",
    wgrad_token_axis=0,
)

# One scalar inverse scale per expert, read in place through a pointer array.
_FP8_GROUPED = _GroupedKernel(
    name="FP8",
    block_m=FP8_BLOCK_M,
    block_n=FP8_BLOCK_N,
    rows=lambda t: (t._data, t._scale_inv),
    cols=lambda t: (_fp8_transpose(t), t._scale_inv),
    scale_arg=lambda scales, device, k_pad: grouped_utils.expert_ptrs(list(scales), device),
    launch=_launch_fp8,
    dgrad_layout="TN",
    wgrad_token_axis=1,
)

_HALF_GROUPED = _GroupedKernel(
    name="half-precision",
    block_m=HALF_PREC_BLOCK_M,
    block_n=HALF_PREC_BLOCK_N,
    rows=lambda t: (_as_rows(t), None),
    cols=lambda t: (_as_rows(t), None),
    scale_arg=lambda scales, device, k_pad: None,
    launch=_launch_half,
    dgrad_layout="NN",
    wgrad_token_axis=0,
)


def _grouped_kernel_for(tensors):
    """The kernel family for ``tensors``, which must all share one storage type."""
    from ...quantized_tensor import QuantizedTensorStorage
    from ...tensor.storage import Float8TensorStorage, MXFP8TensorStorage

    if all(isinstance(t, MXFP8TensorStorage) for t in tensors):
        return _MXFP8_GROUPED
    if all(isinstance(t, Float8TensorStorage) for t in tensors):
        return _FP8_GROUPED
    if any(isinstance(t, QuantizedTensorStorage) for t in tensors):
        raise FlyDSLUnsupportedError(
            "FlyDSL grouped GEMM supports only MXFP8, tensor-wise FP8, and BF16/FP16 "
            f"operands, got {sorted({type(t).__name__ for t in tensors})}"
        )
    dtypes = {t.dtype for t in tensors}
    if len(dtypes) == 1 and dtypes <= {torch.bfloat16, torch.float16}:
        return _HALF_GROUPED
    raise FlyDSLUnsupportedError(
        f"FlyDSL half-precision grouped GEMM requires bf16/fp16 operands, got {dtypes}"
    )


def _as_kernel_dtype(data, t):
    """Reinterpret TE's uint8 FP8 payload ``data`` of operand ``t`` as its FP8 dtype."""
    if data.dtype == torch.uint8:
        return reinterpret_as_fp8_tensor(data, t._fp8_dtype)
    return data


def _row_operand(datas, m_splits):
    """Per-expert ``[m_e, K]`` row payloads as one grouped-kernel operand.

    Returns ``(data, data_ptrs)``: each expert's rows are read in place through base
    pointers shifted to their packed row (``grouped_gemm_utils.stacked_operand``), so
    the kernel sees the ``[M_total, K]`` layout without a concat. Falls back to
    ``(concat_or_view_rows(datas), None)`` when the payloads are not row-major
    ``[m_e, K]`` blocks.
    """
    if all(d.dim() == 2 and d.size(0) == int(m) for d, m in zip(datas, m_splits)):
        stacked = grouped_utils.stacked_operand(datas, datas[0].device)
        if stacked is not None:
            return stacked
    return grouped_utils.concat_or_view_rows(datas), None


def _run_grouped(kernel, a_tensors, b_tensors, D, *, m_splits, layout):
    """One packed forward or dgrad launch: ``D[rows of e] = A_e @ op(B_e)`` per expert.

    ``layout`` is the GEMM's TE layout. TN (forward): A = activations ``[M_total, K]``,
    B = per-expert weights ``[N, K]``. NN (dgrad): A = grad_output ``[M_total, N_out]``,
    B = per-expert weights ``[N_out, K_in]``, read through ``kernel.cols`` and launched
    with ``kernel.dgrad_layout``. A is read per expert in place; the weights are never
    concatenated -- the kernel reads expert ``e``'s from ``b_data_ptrs[e]`` (``b_data``
    is one representative, for dtype and shape). The per-tile ``(pid_m, pid_n, expert)``
    descriptor queue is built on the host from ``m_splits``, so there is no
    device-to-host sync. Operands arrive already quantized.
    """
    if layout == "NN":
        b_read, layout = kernel.cols, kernel.dgrad_layout
    else:
        b_read = kernel.rows
    a_datas, a_scales = zip(*(kernel.rows(t) for t in a_tensors))
    b_datas, b_scales = zip(*(b_read(t) for t in b_tensors))
    a_data, a_data_ptrs = _row_operand(list(a_datas), m_splits)
    device = a_data.device
    b_data_ptrs = grouped_utils.expert_ptrs(list(b_datas), device)
    a_data = _as_kernel_dtype(a_data, a_tensors[0])
    b_data = _as_kernel_dtype(b_datas[0], b_tensors[0])

    # TN: B is [N, K]; NN: B is [K, N].
    n = b_data.size(0) if layout == "TN" else b_data.size(-1)
    k = a_data.size(-1)
    m_total = sum(int(m) for m in m_splits)
    D = _validate_or_allocate_output(
        D,
        shape=torch.Size((m_total, n)),
        dtype=D.dtype,
        device=device,
        backend_name=f"{kernel.name} grouped {layout}",
    )

    # Tiles cover each expert's rows of the packed output; expert token counts must be
    # BLOCK_M multiples (the queue builder raises otherwise), so no tile spans two
    # experts. "static" walks the queue with a grid stride; "worksteal" claims tiles
    # atomically to balance uneven experts (NVTE_FLYDSL_GROUPED_SCHED overrides).
    work_queue, num_tiles = grouped_utils.build_fwd_dgrad_work_queue(
        m_splits, n // kernel.block_n, kernel.block_m, device
    )
    sched = grouped_utils.select_grouped_sched(num_tiles, device)
    kernel.launch(
        a_data,
        kernel.scale_arg(a_scales, device, None),
        b_data,
        kernel.scale_arg(b_scales, device, None),
        D.view(m_total, n),
        layout=layout,
        m=m_total,
        n=n,
        k=k,
        group=True,
        num_groups=len(m_splits),
        work_queue=work_queue,
        work_counter=grouped_utils.work_counter(sched, device),
        sched=sched,
        num_work=num_tiles,
        b_data_ptrs=b_data_ptrs,
        a_data_ptrs=a_data_ptrs,
        a_shape=(m_total, a_data.size(-1)) if a_data_ptrs is not None else None,
    )


def _wgrad_operands(dy_b, x_b, unpadded, pad, device):
    """One wgrad bucket's (dy, dy_ptrs, x, x_ptrs), shared by every dtype.

    When no expert in the bucket needs padding (``unpadded``), both operands are read
    in place from their per-expert tensors (``grouped_utils.stacked_operand``), so the kernel
    sees the packed layout without a host-side concat. Otherwise ``pad`` builds the
    padded packed operand and the pointers are ``None``.
    """
    if unpadded:
        dy = grouped_utils.stacked_operand(dy_b, device)
        x = grouped_utils.stacked_operand(x_b, device)
        if dy is not None and x is not None:
            return dy[0], dy[1], x[0], x[1]
    return pad(dy_b), None, pad(x_b), None


def _wgrad_max_buckets():
    """Launch-count cap for the K-bucketed wgrad (env override, default uncapped)."""
    v = os.getenv("NVTE_FLYDSL_WGRAD_MAX_BUCKETS", "")
    if v.strip() in ("", "0"):
        return None
    return int(v)


def _pad_tokens(tensors, k_pad, token_axis):
    """Zero-pad each expert's tokens to ``k_pad`` and stack the bucket's experts."""
    if token_axis == 0:
        return grouped_utils.uniform_pad_concat(tensors, k_pad)
    padded = []
    for t in tensors:
        # [feature, m_e]: tokens are the trailing axis.
        if t.size(1) == k_pad:
            padded.append(t.contiguous())
        else:
            out = t.new_zeros((t.size(0), k_pad))
            out[:, : t.size(1)].copy_(t)
            padded.append(out)
    return torch.cat(padded, dim=0)


def _run_grouped_wgrad(kernel, dy_tensors, x_tensors, dW, *, m_splits):
    """Grouped wgrad: ``dW_e = dY_e^T @ X_e`` per expert, via K-bucketed launches.

    Both operands are read through ``kernel.cols``: ``[m_e, feature]`` for MXFP8 and
    half precision (an NT launch), or the materialized ``[feature, m_e]`` transposes for
    tensor-wise FP8 (a TN launch). ``dW`` is the packed ``[G*N_out, K_in]`` output.

    The kernel's contraction K (the per-expert token count) is baked in at compile
    time, so experts are grouped into **K-buckets** by their own padded token count
    (``bucketize_wgrad_tokens``): each expert rounds up to its own BLOCK_M multiple
    (floor 512), and experts sharing a rounded K share one launch. A single large
    expert therefore no longer forces every small expert to contract over its token
    count (which wasted work and overflowed the int32 launch-arg limit under skew).
    Balanced token counts collapse to a single bucket.

    Per bucket, the experts' operands are read in place when none needs padding,
    otherwise zero-padded on the token axis to the bucket's ``k_pad`` and stacked
    (bucket-local, so the kernel's ``expert*K`` addressing uses the bucket-local
    index). Output goes straight to ``dW`` when the bucket is every expert in order,
    else to a bucket-local buffer copied into each expert's global rows. The
    hand-scheduled 4-wave MFMA pipeline is untouched -- K stays compile-time per launch.
    """
    dy_datas, dy_scales = zip(*(kernel.cols(t) for t in dy_tensors))
    x_datas, x_scales = zip(*(kernel.cols(t) for t in x_tensors))
    axis = kernel.wgrad_token_axis
    n_out = dy_datas[0].size(1 - axis)
    k_in = x_datas[0].size(1 - axis)
    num_groups = len(m_splits)
    device = dW.device
    dW = _validate_or_allocate_output(
        dW,
        shape=torch.Size((num_groups * n_out, k_in)),
        dtype=dW.dtype,
        device=device,
        backend_name=f"{kernel.name} grouped wgrad",
    )

    # One (pid_m, pid_n, expert) descriptor per tile of each expert's [N_out, K_in]
    # weight gradient; the kernel offsets the output row by expert*N_out.
    num_pid_m = n_out // kernel.block_m
    num_pid_n = k_in // kernel.block_n
    buckets = grouped_utils.bucketize_wgrad_tokens(
        m_splits, kernel.block_m, max_buckets=_wgrad_max_buckets()
    )
    buckets = grouped_utils.drop_empty_experts(buckets, m_splits, dW, n_out)

    for k_pad, ids in buckets:
        bsize = len(ids)
        dy_b = [dy_datas[e] for e in ids]
        x_b = [x_datas[e] for e in ids]
        dy_data, dy_ptrs, x_data, x_ptrs = _wgrad_operands(
            dy_b,
            x_b,
            all(t.size(axis) == k_pad for t in dy_b + x_b),
            lambda ts: _pad_tokens(ts, k_pad, axis),
            device,
        )
        dy_data = _as_kernel_dtype(dy_data, dy_tensors[0])
        x_data = _as_kernel_dtype(x_data, x_tensors[0])
        # Scales are read in place when the operands are; otherwise padded alongside them.
        scale_k_pad = None if dy_ptrs is not None else k_pad
        dy_scale = kernel.scale_arg([dy_scales[e] for e in ids], device, scale_k_pad)
        x_scale = kernel.scale_arg([x_scales[e] for e in ids], device, scale_k_pad)

        # When the bucket holds all experts in order (the balanced case), bucket-local
        # and global expert indices match, so the kernel writes dW directly.
        direct = ids == list(range(num_groups))
        dW_b = dW if direct else torch.empty(bsize * n_out, k_in, dtype=dW.dtype, device=device)

        work_queue, num_tiles = grouped_utils.build_wgrad_work_queue(
            bsize, num_pid_m, num_pid_n, device
        )
        sched = grouped_utils.select_grouped_sched(num_tiles, device)
        if axis == 0:
            a_shape, b_shape = (bsize * k_pad, n_out), (bsize * k_pad, k_in)
        else:  # transposed operands stack the bucket's experts on the feature axis
            a_shape, b_shape = (bsize * n_out, k_pad), (bsize * k_in, k_pad)
        kernel.launch(
            dy_data,
            dy_scale,
            x_data,
            x_scale,
            dW_b,
            layout="NT",
            m=n_out,
            n=k_in,
            k=k_pad,
            group=False,
            num_groups=bsize,
            work_queue=work_queue,
            work_counter=grouped_utils.work_counter(sched, device),
            sched=sched,
            wgrad=True,
            num_work=num_tiles,
            a_data_ptrs=dy_ptrs,
            b_data_ptrs=x_ptrs,
            a_shape=a_shape if dy_ptrs is not None else None,
            b_shape=b_shape if x_ptrs is not None else None,
        )

        if not direct:
            # Scatter each bucket-local expert block to its global dW rows.
            for j, e in enumerate(ids):
                dW[e * n_out : (e + 1) * n_out].copy_(dW_b[j * n_out : (j + 1) * n_out])


def _packed_wgrad_view(wgrad_list, n_out, k_in):
    """Return a ``[G*n_out, k_in]`` view over ``wgrad_list`` if it is one packed run.

    The non-fused grouped-wgrad destinations are adjacent ``[n_out, k_in]`` slices of
    one contiguous ``[G, n_out, k_in]`` buffer, so the kernel can write its packed
    output directly into that storage, eliding the per-expert copy-out. Return ``None``
    (the caller allocates a scratch buffer and scatters) whenever the destinations are
    not a single contiguous, row-contiguous, correctly-shaped run.
    """
    first = wgrad_list[0]
    elem = first.element_size()
    expected_ptr = first.data_ptr()
    for w in wgrad_list:
        if (
            w.dtype != first.dtype
            or tuple(w.shape) != (n_out, k_in)
            or not w.is_contiguous()
            or w.data_ptr() != expected_ptr
        ):
            return None
        expected_ptr += w.numel() * elem
    # One contiguous run anchored at first.data_ptr(); span all rows over its storage.
    return torch.as_strided(first, (len(wgrad_list) * n_out, k_in), (k_in, 1))


def _grouped_gemm_flydsl(A, B, out, *, layout, m_splits, single_output):
    """Run ``general_grouped_gemm``'s plain GEMM on FlyDSL, or raise if unsupported.

    TE's grouped layouts map onto the three phases: TN forward ``out = B_e @ A_e^T``
    (A = weights, B = activations), NN dgrad ``out = B_e @ A_e`` (A = weights,
    B = grad_output), NT wgrad ``out[e] = B_e^T @ A_e`` (A = activations,
    B = grad_output). Raises ``FlyDSLUnsupportedError`` for any operand or config the
    kernels cannot serve.
    """
    if layout not in ("TN", "NN", "NT") or m_splits is None:
        raise FlyDSLUnsupportedError(f"FlyDSL grouped GEMM does not support layout {layout}")
    kernel = _grouped_kernel_for(list(A) + list(B))
    if layout == "NT":
        if single_output:
            raise FlyDSLUnsupportedError("FlyDSL grouped wgrad writes per-expert outputs")
        k_in = kernel.cols(A[0])[0].size(1 - kernel.wgrad_token_axis)
        n_out = out[0].size(0)
        # Write straight into the packed wgrad storage when ``out`` is one contiguous
        # run; otherwise launch into scratch and scatter.
        dW = _packed_wgrad_view(out, n_out, k_in)
        scatter = dW is None
        if scatter:
            dW = torch.empty(len(out) * n_out, k_in, dtype=out[0].dtype, device=out[0].device)
        _run_grouped_wgrad(kernel, B, A, dW, m_splits=m_splits)
        if scatter:
            for g, w in enumerate(out):
                w.copy_(dW[g * n_out : (g + 1) * n_out])
        return
    if not single_output:
        raise FlyDSLUnsupportedError("FlyDSL grouped fwd/dgrad writes one packed output")
    _run_grouped(kernel, B, A, out[0], m_splits=m_splits, layout=layout)


def _warn_fallback_enabled() -> bool:
    return os.environ.get("NVTE_FLYDSL_GEMM_WARN_FALLBACK", "0").lower() not in (
        "",
        "0",
        "false",
        "no",
        "off",
    )


def general_grouped_gemm_flydsl(
    A,
    B,
    out,
    quantization_params,
    out_dtype,
    layout="TN",
    m_splits=None,
    gelu=False,
    grad=False,
    accumulate=False,
    bias=None,
    use_bias=False,
    use_split_accumulator=False,
    D_dtype=None,
    single_output=False,
    **kwargs,
):
    """Drop-in replacement for ``general_grouped_gemm`` using the FlyDSL grouped kernels.

    Serves the plain grouped GEMM over BF16/FP16, tensor-wise FP8, and MXFP8 operands
    (gfx950) in all three layouts. Anything else -- bias, GELU, accumulation, an
    explicit output dtype, another quantized storage, or a shape the kernel tiling
    cannot cover -- falls back to ``general_grouped_gemm``, mirroring the dense FlyDSL
    fallback in ``cpp_extensions.gemm``. ``NVTE_FLYDSL_GEMM_WARN_FALLBACK=1`` warns on
    fallback. Returns ``general_grouped_gemm``'s ``(out, bias, gelu_input)`` triple.
    """
    from ...cpp_extensions.gemm import _empty_tensor, general_grouped_gemm

    try:
        if gelu or accumulate or use_bias or D_dtype is not None:
            raise FlyDSLUnsupportedError(
                "FlyDSL grouped GEMM does not support gelu, accumulate, bias, or D_dtype"
            )
        _grouped_gemm_flydsl(
            A, B, out, layout=layout, m_splits=m_splits, single_output=single_output
        )
    except FlyDSLUnsupportedError as exc:
        if _warn_fallback_enabled():
            warnings.warn(
                "[FLYDSL WARNING]: FlyDSL grouped GEMM does not support this configuration "
                f"(layout={layout}); falling back to the default backend. Reason: {exc}",
                UserWarning,
                stacklevel=2,
            )
        return general_grouped_gemm(
            A,
            B,
            out,
            quantization_params,
            out_dtype,
            layout=layout,
            m_splits=m_splits,
            gelu=gelu,
            grad=grad,
            accumulate=accumulate,
            bias=bias,
            use_bias=use_bias,
            use_split_accumulator=use_split_accumulator,
            D_dtype=D_dtype,
            single_output=single_output,
        )
    empty_tensors = [_empty_tensor()] * len(A)
    return out, empty_tensors, empty_tensors


def _select_fp8_storage_for_layout(A, transa, B, transb):
    """Select existing TE FP8 backings and normalize to one core contract.

    Tensor-wise FP8 columnwise storage is a materialized transpose, unlike
    MXFP8 columnwise storage, which denotes a different quantization direction.

    After selecting the required TE backing and swapping BLAS operand ownership,
    every supported layout produces the same kernel-visible operands:

        TN: B._data      [M,K], A._data      [N,K]
        NN: B._data      [M,K], A._transpose [N,K]
        NT: B._transpose [M,K], A._transpose [N,K]

    No kernel-side transpose, transpose staging, or transpose-read is needed.
    """
    layout = (bool(transa), bool(transb))

    if layout == (True, False):  # TN
        A_payload = _get_fp8_rowwise_payload(A, "A")
        A_storage = "A._data"
        A_data = _flatten_rowwise(A_payload, A_storage)

        B_payload = _get_fp8_rowwise_payload(B, "B")
        B_storage = "B._data"
        B_data = _flatten_rowwise(B_payload, B_storage)

    elif layout == (False, False):  # NN
        A_payload = _get_fp8_columnwise_payload(A, "A")
        A_storage = "A._transpose"
        A_data = _flatten_columnwise(A_payload, A_storage)

        B_payload = _get_fp8_rowwise_payload(B, "B")
        B_storage = "B._data"
        B_data = _flatten_rowwise(B_payload, B_storage)

    elif layout == (False, True):  # NT / dW
        # Both selected transpose allocations are already materialized
        # row-major [outer,K] backings for the normalized common core.
        A_payload = _get_fp8_columnwise_payload(A, "A")
        A_storage = "A._transpose"
        A_data = _flatten_columnwise(A_payload, A_storage)

        B_payload = _get_fp8_columnwise_payload(B, "B")
        B_storage = "B._transpose"
        B_data = _flatten_columnwise(B_payload, B_storage)

    else:
        raise FlyDSLUnsupportedError("FlyDSL GEMM does not support transa=True, transb=True (TT)")

    return (
        A_data,
        A_storage,
        torch.Size(A_payload.shape),
        B_data,
        B_storage,
        torch.Size(B_payload.shape),
    )


def _run_fp8(
    A,
    transa,
    B,
    transb,
    D,
    *,
    output_dtype: torch.dtype,
    bias=None,
    gelu=False,
):
    """Normalize tensor-wise FP8 storage and invoke the common FP8 core."""
    supported_fp8_dtypes = (
        tex.DType.kFloat8E4M3,
        tex.DType.kFloat8E5M2,
    )
    a_fp8_dtype = getattr(A, "_fp8_dtype", None)
    b_fp8_dtype = getattr(B, "_fp8_dtype", None)
    if a_fp8_dtype not in supported_fp8_dtypes or b_fp8_dtype not in supported_fp8_dtypes:
        raise FlyDSLUnsupportedError(
            "FlyDSL FP8 supports E4M3 and E5M2 independently for A/B; "
            f"got A={a_fp8_dtype} and B={b_fp8_dtype}"
        )

    A_scale_inv = getattr(A, "_scale_inv", None)
    B_scale_inv = getattr(B, "_scale_inv", None)
    for name, scale in (
        ("A._scale_inv", A_scale_inv),
        ("B._scale_inv", B_scale_inv),
    ):
        if not isinstance(scale, torch.Tensor):
            raise FlyDSLUnsupportedError(f"{name} is not populated")
        if scale.dtype != torch.float32 or scale.numel() != 1:
            raise FlyDSLUnsupportedError(
                f"{name} must contain exactly one FP32 tensor-wise inverse "
                f"scale, got dtype={scale.dtype}, shape={tuple(scale.shape)}"
            )

    layout = f"{'T' if transa else 'N'}{'T' if transb else 'N'}"

    (
        A_data,
        A_storage,
        A_payload_shape,
        B_data,
        B_storage,
        B_payload_shape,
    ) = _select_fp8_storage_for_layout(
        A,
        bool(transa),
        B,
        bool(transb),
    )

    _validate_fp8_kernel_operands(
        A_data,
        B_data,
        layout=layout,
        a_storage=A_storage,
        b_storage=B_storage,
    )

    a_scale = B_scale_inv
    b_scale = A_scale_inv

    # Storage selection is layout-specific; execution is not. Tensor-wise FP8
    # transpose backing is already a materialized row-major transpose, so all
    # supported layouts normalize to the common [M,K] x [N,K] core contract.
    matmul = fp8_matmul
    kernel_layout = "common"

    a_flydsl = B_data
    b_flydsl = A_data

    m, k = a_flydsl.shape
    n, kb = b_flydsl.shape

    if not a_flydsl.is_contiguous() or not b_flydsl.is_contiguous():
        raise FlyDSLUnsupportedError(
            f"FlyDSL FP8 {layout} kernel contract requires contiguous final "
            f"operands, got a={tuple(a_flydsl.shape)} "
            f"stride={tuple(a_flydsl.stride())} and "
            f"b={tuple(b_flydsl.shape)} stride={tuple(b_flydsl.stride())}"
        )

    if kb != k:
        raise FlyDSLUnsupportedError(
            f"FlyDSL FP8 {layout} selected incompatible physical backings: "
            f"{B_storage}={tuple(B_data.shape)} and "
            f"{A_storage}={tuple(A_data.shape)}; "
            f"kernel operands are {tuple(a_flydsl.shape)} and "
            f"{tuple(b_flydsl.shape)}"
        )

    if D is not None:
        logical_output_shape = torch.Size(D.shape)
    elif layout in ("TN", "NN"):
        logical_output_shape = torch.Size((*B_payload_shape[:-1], n))
    else:
        logical_output_shape = torch.Size((m, n))
    if _product(logical_output_shape) != m * n:
        raise FlyDSLUnsupportedError(
            f"FlyDSL FP8 {layout} logical output shape "
            f"{tuple(logical_output_shape)} does not match kernel output "
            f"shape {(m, n)}"
        )

    D = _validate_or_allocate_output(
        D,
        shape=logical_output_shape,
        dtype=output_dtype,
        device=a_flydsl.device,
        backend_name=f"FP8 {layout} via {kernel_layout} core",
    )

    _fp8_debug(
        f"dispatch entry: transa={bool(transa)}, transb={bool(transb)}, "
        f"layout={layout}, normalized_core={matmul.__module__}.{matmul.__name__}"
    )
    _fp8_debug(f"selected TE storage: A={A_storage}, B={B_storage}")
    _fp8_tensor_debug(f"selected/{A_storage}", A_data)
    _fp8_tensor_debug(f"selected/{B_storage}", B_data)
    _fp8_tensor_debug("a_flydsl", a_flydsl)
    _fp8_tensor_debug("b_flydsl", b_flydsl)
    _fp8_scale_debug("a_scale", a_scale)
    _fp8_scale_debug("b_scale", b_scale)
    _fp8_debug(f"derived M={m}, N={n}, K={k}")
    _fp8_tensor_debug("output/D", D)

    epilogue, bias_arg = _resolve_epilogue(bias, n, gelu=gelu)
    aux = torch.empty_like(D) if gelu else None
    matmul(
        a_flydsl,
        a_scale,
        b_flydsl,
        b_scale,
        D.view(m, n),
        epilogue=epilogue,
        bias=bias_arg,
        aux=aux.view(m, n) if aux is not None else None,
    )
    return D, aux


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

    Supported layouts:
      - TN: transa=True,  transb=False
      - NN: transa=False, transb=False
      - NT: transa=False, transb=True

    TT is intentionally rejected.

    Supported dtypes:
      - MXFP8 E4M3/E5M2 A/B combinations with FP16, BF16, or FP32 output
      - tensor-wise E4M3/E5M2 FP8 A/B combinations with FP16, BF16, or FP32 output
      - BF16 input with FP16, BF16, or FP32 output
      - FP16 input with FP16, BF16, or FP32 output
      - FP32 input with FP32 output
    """
    # These are hints or unused given the epilogue/dtype guards below: bias_type
    # is inferred, gelu_in is only for DGELU (rejected), and FlyDSL has no
    # separate workspace or split-accumulator path.
    del bias_type
    del gelu_in
    del workspace
    del workspaceSize
    del use_split_accumulator

    # Comm+GEMM overlap (tensor-parallel) is not implemented. These carry real
    # side effects -- extra_output is the reduce-scatter destination -- so they
    # must be rejected, not dropped: silently discarding them would no-op the
    # collective and return wrong results. Raise FlyDSLUnsupportedError so
    # general_gemm falls back to the C++ backend.
    if (
        comm_overlap is not None
        or comm_type is not None
        or extra_output is not None
        or bulk_overlap
    ):
        raise FlyDSLUnsupportedError("FlyDSL GEMM does not support comm+GEMM overlap")

    if transa and transb:
        raise FlyDSLUnsupportedError("FlyDSL GEMM does not support transa=True, transb=True (TT)")

    a_kind, _ = _classify_input(A)
    b_kind, _ = _classify_input(B)

    _validate_common_epilogue(
        quantizer=quantizer,
        bias=bias,
        gelu=gelu,
        grad=grad,
        accumulate=accumulate,
        alpha=alpha,
        beta=beta,
    )

    if a_kind == "mxfp8" or b_kind == "mxfp8":
        # Validate both are MXFP8
        if a_kind != b_kind:
            raise ValueError("Mixed MXFP8 and non-MXFP8 FlyDSL GEMM inputs are not supported")

        # Sanity: both operands must have at least one pre-quantized copy.
        if (
            getattr(A, "_rowwise_data", None) is None
            and getattr(A, "_columnwise_data", None) is None
        ):
            raise RuntimeError("MXFP8Tensor has neither rowwise nor columnwise data")
        if (
            getattr(B, "_rowwise_data", None) is None
            and getattr(B, "_columnwise_data", None) is None
        ):
            raise RuntimeError("MXFP8Tensor has neither rowwise nor columnwise data")

        mxfp8_output_dtypes = {
            None: torch.float16,
            tex.DType.kFloat16: torch.float16,
            tex.DType.kBFloat16: torch.bfloat16,
            tex.DType.kFloat32: torch.float32,
        }
        if output_dtype not in mxfp8_output_dtypes:
            raise FlyDSLUnsupportedError(
                f"FlyDSL MXFP8 supports FP16, BF16, or FP32 output, got {output_dtype}"
            )

        D, gelu_input = _run_mxfp8(
            A,
            transa,
            B,
            transb,
            D,
            output_dtype=mxfp8_output_dtypes[output_dtype],
            bias=bias,
            gelu=gelu,
        )
        return D, None, gelu_input, None

    if a_kind == "fp8" or b_kind == "fp8":
        if a_kind != b_kind:
            raise ValueError("Mixed regular FP8 and non-FP8 FlyDSL GEMM inputs are not supported")

        fp8_output_dtypes = {
            None: torch.float16,
            tex.DType.kFloat16: torch.float16,
            tex.DType.kBFloat16: torch.bfloat16,
            tex.DType.kFloat32: torch.float32,
        }
        if output_dtype not in fp8_output_dtypes:
            raise FlyDSLUnsupportedError(
                f"FlyDSL tensor-wise FP8 supports FP16, BF16, or FP32 output, got {output_dtype}"
            )

        D, gelu_input = _run_fp8(
            A,
            transa,
            B,
            transb,
            D,
            output_dtype=fp8_output_dtypes[output_dtype],
            bias=bias,
            gelu=gelu,
        )
        return D, None, gelu_input, None

    if a_kind != "regular" or b_kind != "regular":
        raise TypeError(
            f"Unsupported FlyDSL GEMM operand types: {type(A).__name__} and {type(B).__name__}"
        )
    if not isinstance(A, torch.Tensor) or not isinstance(B, torch.Tensor):
        raise TypeError("FlyDSL regular GEMM expects plain torch.Tensor operands")

    if A.dtype == torch.bfloat16 and B.dtype == torch.bfloat16:
        bf16_output_dtypes = {
            None: torch.bfloat16,
            tex.DType.kFloat16: torch.float16,
            tex.DType.kBFloat16: torch.bfloat16,
            tex.DType.kFloat32: torch.float32,
        }
        if output_dtype not in bf16_output_dtypes:
            raise FlyDSLUnsupportedError(
                f"FlyDSL BF16 supports FP16, BF16, or FP32 output, got {output_dtype}"
            )
        D, gelu_input = _run_bf16_gemm(
            A,
            transa,
            B,
            transb,
            D,
            output_dtype=bf16_output_dtypes[output_dtype],
            bias=bias,
            gelu=gelu,
        )
        return D, None, gelu_input, None

    if A.dtype == torch.float16 and B.dtype == torch.float16:
        fp16_output_dtypes = {
            None: torch.float16,
            tex.DType.kFloat16: torch.float16,
            tex.DType.kBFloat16: torch.bfloat16,
            tex.DType.kFloat32: torch.float32,
        }
        if output_dtype not in fp16_output_dtypes:
            raise FlyDSLUnsupportedError(
                f"FlyDSL FP16 supports FP16, BF16, or FP32 output, got {output_dtype}"
            )
        D, gelu_input = _run_fp16_gemm(
            A,
            transa,
            B,
            transb,
            D,
            output_dtype=fp16_output_dtypes[output_dtype],
            bias=bias,
            gelu=gelu,
        )
        return D, None, gelu_input, None

    if A.dtype == torch.float32 and B.dtype == torch.float32:
        if output_dtype not in (None, tex.DType.kFloat32):
            raise FlyDSLUnsupportedError(
                f"FlyDSL FP32 currently supports only FP32 output, got {output_dtype}"
            )
        D, gelu_input = _run_fp32_gemm(
            A,
            transa,
            B,
            transb,
            D,
            bias=bias,
            gelu=gelu,
        )
        return D, None, gelu_input, None

    raise FlyDSLUnsupportedError(
        "FlyDSL GEMM currently supports only MXFP8, tensor-wise E4M3 FP8, "
        "BF16, FP16, or FP32 inputs; "
        f"got A={A.dtype} and B={B.dtype}"
    )
