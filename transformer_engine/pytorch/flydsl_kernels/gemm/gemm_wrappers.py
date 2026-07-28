# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""TE entry points for the FlyDSL GEMM backend."""

import os

import torch
import transformer_engine_torch as tex

from transformer_engine.pytorch.utils import get_device_compute_capability

from .exceptions import FlyDSLUnsupportedError

from .bf16_gemm import bf16_matmul
from .fp16_gemm import fp16_matmul
from .fp32_gemm import fp32_matmul
from .fp8_gemm import fp8_matmul
from .fp8_gemm_nn import fp8_matmul as fp8_matmul_nn
from .fp8_gemm_nt import fp8_matmul as fp8_matmul_nt
from .mxfp8_gemm import mxfp8_matmul
from .mxfp8_gemm_nn import mxfp8_matmul as mxfp8_matmul_nn
from .mxfp8_gemm_nt import mxfp8_matmul as mxfp8_matmul_nt


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
                f"The FlyDSL GEMM backend does not support "
                f"{type(t).__name__}. Only Float8Tensor / "
                f"Float8TensorStorage and MXFP8Tensor / "
                f"MXFP8TensorStorage are implemented."
            )
    except ImportError:
        pass

    return "regular", None


def _reinterpret_fp8_payload(data, fp8_dtype, name):
    """Reinterpret TE's uint8 payload using its ``tex.DType`` metadata."""
    if data is None:
        raise FlyDSLUnsupportedError(
            f"{name} does not contain the required FP8 payload"
        )

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


def _canonicalize_blas_pair(
    A_data: torch.Tensor,
    transa: bool,
    B_data: torch.Tensor,
    transb: bool,
):
    """Swap TE BLAS operands and apply their original transpose flags."""
    a_flydsl = B_data.transpose(0, 1) if transb else B_data
    b_flydsl = A_data.transpose(0, 1) if transa else A_data
    return a_flydsl, b_flydsl


def _flatten_rowwise(t: torch.Tensor, name: str) -> torch.Tensor:
    """Flatten all leading dimensions while preserving the final dimension."""
    if t.ndim < 2:
        raise ValueError(
            f"FlyDSL GEMM expects {name} to have rank >= 2, got {tuple(t.shape)}"
        )
    return t.reshape(-1, t.shape[-1])


def _flatten_columnwise(t: torch.Tensor, name: str) -> torch.Tensor:
    """Flatten TE columnwise storage while preserving its leading dimension."""
    if t.ndim < 2:
        raise ValueError(
            f"FlyDSL GEMM expects {name} to have rank >= 2, got {tuple(t.shape)}"
        )
    return t.reshape(t.shape[0], -1)


def _canonicalize_blas_operands(
    A_data: torch.Tensor,
    transa: bool,
    B_data: torch.Tensor,
    transb: bool,
):
    """Convert TE's BLAS-shaped operands to FlyDSL row-major operands.

    TE's generic GEMM interface follows BLAS column-major interpretation.
    FlyDSL kernels consume ordinary row-major matrices:

        a_flydsl: [M, K]
        b_flydsl: [K, N]

    The standard conversion is to swap A/B and apply the original transpose
    flags to the swapped operands:

        a_flydsl = op(B)
        b_flydsl = op(A)
    """
    if transa and transb:
        raise NotImplementedError(
            "FlyDSL GEMM does not support transa=True, transb=True (TT)"
        )

    A_flat = _flatten_rowwise(A_data, "A")
    B_flat = _flatten_rowwise(B_data, "B")

    a_flydsl, b_flydsl = _canonicalize_blas_pair(
        A_flat,
        transa,
        B_flat,
        transb,
    )

    m, k = a_flydsl.shape
    kb, n = b_flydsl.shape
    if kb != k:
        layout = f"{'T' if transa else 'N'}{'T' if transb else 'N'}"
        raise ValueError(
            f"FlyDSL {layout} canonicalization produced incompatible operands: "
            f"{tuple(a_flydsl.shape)} @ {tuple(b_flydsl.shape)}"
        )

    return a_flydsl, b_flydsl, m, n, k


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
        raise ValueError(
            f"D shape {tuple(D.shape)} does not match expected {tuple(shape)}"
        )
    if D.dtype != dtype:
        raise TypeError(
            f"FlyDSL {backend_name} requires {dtype} output, got {D.dtype}"
        )
    if D.device != device:
        raise ValueError(
            f"D must be on {device}, got {D.device}"
        )
    if not D.is_contiguous():
        raise ValueError(
            f"FlyDSL {backend_name} requires contiguous output storage"
        )
    return D


def _run_regular_gemm(
    A,
    transa,
    B,
    transb,
    D,
    *,
    dtype,
    matmul,
    backend_name,
    output_dtype=None,
):
    """Run FP16/BF16/FP32 through shared TN/NN/NT shape handling."""
    if not isinstance(A, torch.Tensor) or not isinstance(B, torch.Tensor):
        raise TypeError(
            f"FlyDSL {backend_name} GEMM expects plain torch.Tensor operands"
        )
    if A.dtype != dtype or B.dtype != dtype:
        raise TypeError(
            f"FlyDSL {backend_name} GEMM requires {dtype} inputs, "
            f"got A={A.dtype} and B={B.dtype}"
        )
    if A.device != B.device:
        raise ValueError(
            f"A and B must be on the same device, got {A.device} and {B.device}"
        )

    output_shape = _get_gemm_output_shape(A, transa, B, transb)

    a_flydsl, b_flydsl, m, n, _ = _canonicalize_blas_operands(
        A, transa, B, transb
    )
    if _product(output_shape) != m * n:
        raise RuntimeError(
            f"FlyDSL {backend_name} logical output shape {tuple(output_shape)} "
            f"does not match flattened GEMM shape {(m, n)}"
        )

    if output_dtype is None:
        output_dtype = dtype

    D = _validate_or_allocate_output(
        D,
        shape=output_shape,
        dtype=output_dtype,
        device=A.device,
        backend_name=backend_name,
    )

    matmul(
        a_flydsl,
        b_flydsl,
        D.view(m, n),
    )
    return D



def _get_fp8_rowwise_payload(t, name):
    """Return TE's existing rowwise ``_data`` payload without copying."""
    data = getattr(t, "_data", None)
    if data is None:
        raise FlyDSLUnsupportedError(
            f"FlyDSL FP8 requires existing {name} rowwise (_data) storage"
        )
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
        raise RuntimeError(
            f"{name} does not contain required {orientation} MXFP8 data and scales"
        )

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
        raise FlyDSLUnsupportedError(
            f"{name} has neither rowwise nor columnwise MXFP8 data"
        )
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
        raise ValueError(
            f"FlyDSL MXFP8 expects {name} scale rank >= 2, "
            f"got {tuple(t.shape)}"
        )

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
):
    """Dispatch MXFP8 through exact TN/NN/NT physical contracts.

    TE owns BLAS-shaped operands. After the usual ownership swap, FlyDSL
    kernels consume:

        TN: a = B.rowwise      [M, K]
            b = A.rowwise.T    [K, N]  (validated TN adapter contract)

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
    if (
        a_fp8_dtype not in supported_fp8_dtypes
        or b_fp8_dtype not in supported_fp8_dtypes
    ):
        raise FlyDSLUnsupportedError(
            "FlyDSL MXFP8 supports E4M3 and E5M2 independently for A/B; "
            f"got A={a_fp8_dtype} and B={b_fp8_dtype}"
        )

    layout = f"{'T' if transa else 'N'}{'T' if transb else 'N'}"
    dispatch = {
        (True, False): ("TN", mxfp8_matmul),
        (False, False): ("NN", mxfp8_matmul_nn),
        (False, True): ("NT", mxfp8_matmul_nt),
    }
    try:
        kernel_layout, matmul = dispatch[(bool(transa), bool(transb))]
    except KeyError as exc:
        raise FlyDSLUnsupportedError(
            "FlyDSL GEMM does not support transa=True, transb=True (TT)"
        ) from exc

    _mxfp8_debug(
        f"entry: layout={layout}, selected_kernel="
        f"{matmul.__module__}.{matmul.__name__}, "
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
        # Preserve the validated TN adapter contract:
        #   a [M,K], b [K,N]
        a_flydsl = B_data
        b_flydsl = A_data.transpose(0, 1)
        a_scale = B_scale
        b_scale = A_scale.transpose(0, 1)

        m, k = a_flydsl.shape
        kb, n = b_flydsl.shape
        expected_a_scale = (m, k // 32)
        expected_b_scale = (k // 32, n)

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
        raise ValueError(
            f"K={k} must be divisible by MXFP8 scale group size 32"
        )

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

    matmul(
        a_flydsl,
        a_scale,
        b_flydsl,
        b_scale,
        D.view(m, n),
    )
    return D


def _select_fp8_storage_for_layout(A, transa, B, transb):
    """Select the exact existing TE FP8 backing required by each layout.

    Fixed zero-copy routes selected for the final kernel contracts:

        TN: wrapper swaps B._data/A._data -> [M,K], [N,K]
        NN: wrapper swaps B._data/A._transpose -> [M,K], [N,K]
        NT: wrapper swaps B._transpose/A._transpose -> [M,K], [N,K]

    NT is the NN transpose-storage path applied to both operands. Both
    transpose allocations stay contiguous in their native [outer,K] shapes;
    no tensor transpose, reshape reinterpretation, or materialization occurs.
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
        # NT extends NN's transpose-storage handling to both operands.
        # After ownership swap, B._transpose is kernel A [M,K] and
        # A._transpose is kernel B [N,K].
        A_payload = _get_fp8_columnwise_payload(A, "A")
        A_storage = "A._transpose"
        A_data = _flatten_columnwise(A_payload, A_storage)

        B_payload = _get_fp8_columnwise_payload(B, "B")
        B_storage = "B._transpose"
        B_data = _flatten_columnwise(B_payload, B_storage)

    else:
        raise FlyDSLUnsupportedError(
            "FlyDSL GEMM does not support transa=True, transb=True (TT)"
        )

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
):
    """Dispatch tensor-wise FP8 using exact per-kernel storage contracts."""
    supported_fp8_dtypes = (
        tex.DType.kFloat8E4M3,
        tex.DType.kFloat8E5M2,
    )
    a_fp8_dtype = getattr(A, "_fp8_dtype", None)
    b_fp8_dtype = getattr(B, "_fp8_dtype", None)
    if (
        a_fp8_dtype not in supported_fp8_dtypes
        or b_fp8_dtype not in supported_fp8_dtypes
    ):
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

    if layout == "TN":
        matmul = fp8_matmul
        kernel_layout = "TN"

        a_flydsl = B_data
        b_flydsl = A_data

        m, k = a_flydsl.shape
        n, kb = b_flydsl.shape

    elif layout == "NN":
        matmul = fp8_matmul_nn
        kernel_layout = "NN"

        a_flydsl = B_data
        b_flydsl = A_data

        m, k = a_flydsl.shape
        n, kb = b_flydsl.shape

    elif layout == "NT":
        matmul = fp8_matmul_nt
        kernel_layout = "NT"

        # Correct fp8_gemm_nt contract: NN's [outer,K] transpose-storage
        # path applied to both operands.
        a_flydsl = B_data
        b_flydsl = A_data

        m, k = a_flydsl.shape
        n, kb = b_flydsl.shape

    else:
        raise FlyDSLUnsupportedError(
            "FlyDSL GEMM does not support transa=True, transb=True (TT)"
        )

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
        backend_name=f"FP8 {kernel_layout}",
    )

    _fp8_debug(
        f"dispatch entry: transa={bool(transa)}, transb={bool(transb)}, "
        f"layout={layout}, selected_kernel={matmul.__module__}.{matmul.__name__}"
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

    matmul(
        a_flydsl,
        a_scale,
        b_flydsl,
        b_scale,
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
    del bias_type
    del gelu_in
    del workspace
    del workspaceSize
    del use_split_accumulator
    del comm_overlap
    del comm_type
    del extra_output
    del bulk_overlap

    if transa and transb:
        raise NotImplementedError(
            "FlyDSL GEMM does not support transa=True, transb=True (TT)"
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

    a_kind, _ = _classify_input(A)
    b_kind, _ = _classify_input(B)

    if a_kind == "mxfp8" or b_kind == "mxfp8":
         # Validate both are MXFP8
        if a_kind != b_kind:
            raise ValueError(
                "Mixed MXFP8 and non-MXFP8 FlyDSL GEMM inputs are not supported"
            )

        # Sanity: both operands must have at least one pre-quantized copy.
        if getattr(A, '_rowwise_data', None) is None and getattr(A, '_columnwise_data', None) is None:
            raise RuntimeError("MXFP8Tensor has neither rowwise nor columnwise data")
        if getattr(B, '_rowwise_data', None) is None and getattr(B, '_columnwise_data', None) is None:
            raise RuntimeError("MXFP8Tensor has neither rowwise nor columnwise data")

        mxfp8_output_dtypes = {
            None: torch.float16,
            tex.DType.kFloat16: torch.float16,
            tex.DType.kBFloat16: torch.bfloat16,
            tex.DType.kFloat32: torch.float32,
        }
        if output_dtype not in mxfp8_output_dtypes:
            raise NotImplementedError(
                "FlyDSL MXFP8 supports FP16, BF16, or FP32 output, "
                f"got {output_dtype}"
            )

        D = _run_mxfp8(
            A,
            transa,
            B,
            transb,
            D,
            output_dtype=mxfp8_output_dtypes[output_dtype],
        )
        return D, None, None, None

    if a_kind == "fp8" or b_kind == "fp8":
        if a_kind != b_kind:
            raise ValueError(
                "Mixed regular FP8 and non-FP8 FlyDSL GEMM inputs are not supported"
            )

        fp8_output_dtypes = {
            None: torch.float16,
            tex.DType.kFloat16: torch.float16,
            tex.DType.kBFloat16: torch.bfloat16,
            tex.DType.kFloat32: torch.float32,
        }
        if output_dtype not in fp8_output_dtypes:
            raise NotImplementedError(
                "FlyDSL tensor-wise FP8 supports FP16, BF16, or FP32 output, "
                f"got {output_dtype}"
            )

        D = _run_fp8(
            A,
            transa,
            B,
            transb,
            D,
            output_dtype=fp8_output_dtypes[output_dtype],
        )
        return D, None, None, None

    if a_kind != "regular" or b_kind != "regular":
        raise TypeError(
            "Unsupported FlyDSL GEMM operand types: "
            f"{type(A).__name__} and {type(B).__name__}"
        )
    if not isinstance(A, torch.Tensor) or not isinstance(B, torch.Tensor):
        raise TypeError(
            "FlyDSL regular GEMM expects plain torch.Tensor operands"
        )

    if A.dtype == torch.bfloat16 and B.dtype == torch.bfloat16:
        bf16_output_dtypes = {
            None: torch.bfloat16,
            tex.DType.kFloat16: torch.float16,
            tex.DType.kBFloat16: torch.bfloat16,
            tex.DType.kFloat32: torch.float32,
        }
        if output_dtype not in bf16_output_dtypes:
            raise NotImplementedError(
                "FlyDSL BF16 supports FP16, BF16, or FP32 output, "
                f"got {output_dtype}"
            )
        D = _run_regular_gemm(
            A,
            transa,
            B,
            transb,
            D,
            dtype=torch.bfloat16,
            matmul=bf16_matmul,
            backend_name="BF16",
            output_dtype=bf16_output_dtypes[output_dtype],
        )
        return D, None, None, None

    if A.dtype == torch.float16 and B.dtype == torch.float16:
        fp16_output_dtypes = {
            None: torch.float16,
            tex.DType.kFloat16: torch.float16,
            tex.DType.kBFloat16: torch.bfloat16,
            tex.DType.kFloat32: torch.float32,
        }
        if output_dtype not in fp16_output_dtypes:
            raise NotImplementedError(
                "FlyDSL FP16 supports FP16, BF16, or FP32 output, "
                f"got {output_dtype}"
            )
        D = _run_regular_gemm(
            A,
            transa,
            B,
            transb,
            D,
            dtype=torch.float16,
            matmul=fp16_matmul,
            backend_name="FP16",
            output_dtype=fp16_output_dtypes[output_dtype],
        )
        return D, None, None, None

    if A.dtype == torch.float32 and B.dtype == torch.float32:
        if output_dtype not in (None, tex.DType.kFloat32):
            raise NotImplementedError(
                "FlyDSL FP32 currently supports only FP32 output, "
                f"got {output_dtype}"
            )
        D = _run_regular_gemm(
            A,
            transa,
            B,
            transb,
            D,
            dtype=torch.float32,
            matmul=fp32_matmul,
            backend_name="FP32",
        )
        return D, None, None, None

    raise NotImplementedError(
        "FlyDSL GEMM currently supports only MXFP8, tensor-wise E4M3 FP8, "
        "BF16, FP16, or FP32 inputs; "
        f"got A={A.dtype} and B={B.dtype}"
    )