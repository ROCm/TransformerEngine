# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""TE entry points for the FlyDSL GEMM backend."""

import os

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



def _mxfp8_debug_enabled() -> bool:
    value = os.getenv("DEBUG_FLYDSL_MXFP8_GEMM", "")
    return value.lower() not in ("", "0", "false", "no", "off")


def _mxfp8_debug(message: str) -> None:
    if _mxfp8_debug_enabled():
        print(f"[DEBUG_FLYDSL_MXFP8_GEMM] {message}")


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

    a_flydsl, b_flydsl, m, n, _ = _canonicalize_blas_operands(
        A, transa, B, transb
    )

    D = _validate_or_allocate_output(
        D,
        shape=(m, n),
        dtype=dtype,
        device=A.device,
        backend_name=backend_name,
    )

    matmul(
        a_flydsl,
        b_flydsl,
        D.view(m, n),
    )
    return D


def _get_fp8_logical_rowwise_payload(t, name):
    """Return logical rowwise FP8 data, matching the Triton wrapper.

    Prefer TE's rowwise ``_data``. If only valid columnwise ``_transpose``
    storage exists, materialize a rowwise copy once for canonicalization.
    """
    fp8_dtype = getattr(t, "_fp8_dtype", None)
    data = getattr(t, "_data", None)

    if data is not None:
        return _reinterpret_fp8_payload(
            data,
            fp8_dtype,
            f"{name}._data",
        )

    if not _valid_fp8_transpose(t):
        raise RuntimeError(
            f"{name} has neither valid rowwise (_data) nor "
            f"columnwise (_transpose) FP8 storage"
        )

    transpose_data = _reinterpret_fp8_payload(
        t._transpose,
        fp8_dtype,
        f"{name}._transpose",
    )

    if transpose_data.ndim < 2:
        raise ValueError(
            f"{name}._transpose must have rank >= 2, "
            f"got {tuple(transpose_data.shape)}"
        )

    # TE's columnwise payload represents the transpose of the logical rowwise
    # tensor. Materialize rowwise storage before applying BLAS transpose flags,
    # exactly as the Triton wrapper's materialize_rowwise_from_columnwise path.
    return transpose_data.transpose(-2, -1).contiguous()


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


def _flatten_mxfp8_scale(t: torch.Tensor, name: str) -> torch.Tensor:
    if t.ndim < 2:
        raise ValueError(
            f"FlyDSL MXFP8 expects {name} scale rank >= 2, "
            f"got {tuple(t.shape)}"
        )
    original_shape = tuple(t.shape)
    if t.ndim > 2:
        t = t.reshape(-1, t.shape[-1])
    _mxfp8_debug(
        f"{name} scale flatten: {original_shape} -> {tuple(t.shape)}, "
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
    """Canonicalize TE MXFP8 operands, then launch the fused backend."""
    layout = f"{'T' if transa else 'N'}{'T' if transb else 'N'}"
    _mxfp8_debug(
        f"entry: layout={layout}, A_type={type(A).__name__}, "
        f"B_type={type(B).__name__}, D_provided={D is not None}"
    )

    # Match TE CanonicalizeGemmInput / Triton data_and_scale_for_transpose:
    # A: transa=True -> rowwise,  transa=False -> columnwise
    # B: transb=True -> columnwise, transb=False -> rowwise
    A_data, A_scale = _select_mxfp8_data_and_scale(
        A,
        will_transpose=not transa,
        name="A",
    )
    B_data, B_scale = _select_mxfp8_data_and_scale(
        B,
        will_transpose=transb,
        name="B",
    )

    a_flydsl, b_flydsl, m, n, k = _canonicalize_blas_operands(
        A_data,
        transa,
        B_data,
        transb,
    )

    A_scale = _flatten_mxfp8_scale(A_scale, "A")
    B_scale = _flatten_mxfp8_scale(B_scale, "B")
    a_scale, b_scale = _canonicalize_blas_pair(
        A_scale,
        transa,
        B_scale,
        transb,
    )

    _mxfp8_debug(
        f"canonicalized layout={layout}: "
        f"a={tuple(a_flydsl.shape)}, stride={tuple(a_flydsl.stride())}; "
        f"b={tuple(b_flydsl.shape)}, stride={tuple(b_flydsl.stride())}"
    )
    _mxfp8_debug(
        f"canonicalized scales: "
        f"a_scale={tuple(a_scale.shape)}, stride={tuple(a_scale.stride())}; "
        f"b_scale={tuple(b_scale.shape)}, stride={tuple(b_scale.stride())}"
    )
    _mxfp8_debug(f"derived GEMM dimensions: M={m}, N={n}, K={k}")

    if a_flydsl.device != b_flydsl.device:
        raise ValueError(
            f"A and B must be on the same device, got "
            f"{a_flydsl.device} and {b_flydsl.device}"
        )

    scale_group_size = 32
    if k % scale_group_size != 0:
        raise ValueError(
            f"K={k} must be divisible by MXFP8 scale group size "
            f"{scale_group_size}"
        )

    # Shared BLAS canonicalization yields:
    #   a_scale [M, K/32]
    #   b_scale [K/32, N]
    expected_a_scale = (m, k // scale_group_size)
    expected_b_scale = (k // scale_group_size, n)
    if tuple(a_scale.shape) != expected_a_scale:
        raise ValueError(
            f"A scale shape {tuple(a_scale.shape)} != expected "
            f"{expected_a_scale}"
        )
    if tuple(b_scale.shape) != expected_b_scale:
        raise ValueError(
            f"B scale shape {tuple(b_scale.shape)} != expected "
            f"{expected_b_scale}"
        )
    if a_scale.dtype != torch.uint8 or b_scale.dtype != torch.uint8:
        raise TypeError("FlyDSL MXFP8 expects raw E8M0 scales as torch.uint8")

    D = _validate_or_allocate_output(
        D,
        shape=(m, n),
        dtype=output_dtype,
        device=a_flydsl.device,
        backend_name="MXFP8",
    )

    return mxfp8_matmul(
        a_flydsl,
        a_scale,
        b_flydsl,
        b_scale,
        D.view(m, n),
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
    """Run tensor-wise E4M3 x E4M3 FP8 for TN/NN/NT."""
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

    if transa and transb:
        raise NotImplementedError(
            "FlyDSL GEMM does not support transa=True, transb=True (TT)"
        )

    # Match Triton's regular-FP8 handling: establish logical rowwise
    # payloads first, then apply the same shared BLAS-to-row-major
    # canonicalization used for FP16/BF16/FP32.
    A_data = _get_fp8_logical_rowwise_payload(A, "A")
    B_data = _get_fp8_logical_rowwise_payload(B, "B")

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

    a_flydsl, b_flydsl, m, n, _ = _canonicalize_blas_operands(
        A_data, transa, B_data, transb
    )

    if a_flydsl.device != b_flydsl.device:
        raise ValueError(
            f"A and B must be on the same device, got "
            f"{a_flydsl.device} and {b_flydsl.device}"
        )

    D = _validate_or_allocate_output(
        D,
        shape=(m, n),
        dtype=output_dtype,
        device=a_flydsl.device,
        backend_name="FP8",
    )

    # Operand swap means B's tensor-wise scale belongs to a_flydsl and A's
    # tensor-wise scale belongs to b_flydsl.
    fp8_matmul(
        a_flydsl,
        B_scale_inv,
        b_flydsl,
        A_scale_inv,
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
      - MXFP8 input with FP16, BF16, or FP32 output
      - tensor-wise E4M3 x E4M3 FP8 input with FP16, BF16, or FP32 output
      - BF16 input with BF16 output
      - FP16 input with FP16 output
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
        if output_dtype not in (None, tex.DType.kBFloat16):
            raise NotImplementedError(
                "FlyDSL BF16 currently supports only BF16 output, "
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
        )
        return D, None, None, None

    if A.dtype == torch.float16 and B.dtype == torch.float16:
        if output_dtype not in (None, tex.DType.kFloat16):
            raise NotImplementedError(
                "FlyDSL FP16 currently supports only FP16 output, "
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
