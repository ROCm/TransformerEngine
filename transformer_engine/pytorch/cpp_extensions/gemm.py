# This file was modified for portability to AMDGPU
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Python interface for GEMM extensions"""

from typing import Iterable, Optional, Tuple, Union, List
import os
import functools
import warnings
import torch
from torch.utils.cpp_extension import IS_HIP_EXTENSION
import transformer_engine_torch as tex
from ..constants import TE_DType, DType
from ..utils import get_sm_count, _empty_tensor, get_gemm_backend
if IS_HIP_EXTENSION:
    from ..utils import get_device_compute_capability
    from ..utils import cast_if_needed

from ..quantized_tensor import Quantizer
from ..tensor.storage.float8_blockwise_tensor_storage import Float8BlockwiseQTensorStorage
from ..tensor.storage.mxfp8_tensor_storage import MXFP8TensorStorage
from ..tensor.storage.grouped_tensor_storage import GroupedTensorStorage
from ..tensor.storage.nvfp4_tensor_storage import NVFP4TensorStorage
from ..tensor.utils import is_custom
from ..custom_recipes.gemm import custom_gemm
from ...debug.pytorch.debug_quantization import DebugQuantizer

_FP4_USE_TUNED_GEMM = int(os.environ.get("NVTE_FP4_USE_TUNED_GEMM", "1"))
_FP4_LOG_SHAPES = int(os.environ.get("NVTE_FP4_LOG_GEMM_SHAPES", "0"))

__all__ = [
    "general_gemm",
    "general_grouped_gemm",
    "general_grouped_gemm_for_grouped_tensor",
]


_NUM_MAX_UB_STREAMS = 3


def get_cublas_workspace_size_bytes() -> None:
    """Return workspace size needed for current architecture."""
    if IS_HIP_EXTENSION:
        """Return 64 MiB for gfx50x, 32 MiB for all other architectures."""
        if get_device_compute_capability() == (9, 5):
            return 67_108_864
        return 33_554_432
    """Return 32 MiB if using hopper, 4 MiB for all other architectures."""
    if torch.cuda.get_device_properties(torch.cuda.current_device()).major >= 9:
        # 32 MiB for NVFP4 GEMM, plus additional 1024 B for alignment and misc scales
        return 32 * 1024 * 1024 + 1024
    return 4_194_304


@functools.lru_cache(maxsize=None)
def get_cublas_workspace(device: int, ub: bool, grouped_gemm: bool) -> torch.Tensor:
    """Returns workspace for cublas GEMM."""
    assert not (ub and grouped_gemm), "UB is unsupported for grouped GEMM."

    if ub:
        return torch.empty(
            get_cublas_workspace_size_bytes() * _NUM_MAX_UB_STREAMS,
            dtype=torch.uint8,
            device=device,
        )
    if grouped_gemm:
        _multi_stream_cublas_workspace = []
        for _ in range(tex.get_num_cublas_streams()):
            _multi_stream_cublas_workspace.append(
                torch.empty(get_cublas_workspace_size_bytes(), dtype=torch.uint8, device=device)
            )
        return _multi_stream_cublas_workspace

    return torch.empty(get_cublas_workspace_size_bytes(), dtype=torch.uint8, device=device)


def validate_gemm_scale(scale: Optional[float], required: bool) -> float:
    """Validate whether a GEMM scaling factor is consistent with its usage"""
    if required:
        return scale if scale is not None else 1.0
    if scale not in (0.0, None):
        raise ValueError("scale must be zero")
    return 0.0


def get_tensor_device(tensor: torch.Tensor) -> int:
    """
    Returns tensor device as an integer.

    This method is used because checking instances of
    QuantizedTensor or Storage incurs more CPU overhead.
    The order of attributes checked is important to also
    minimize overhead.
    """
    if hasattr(tensor, "device"):
        return tensor.device.index
    if hasattr(tensor, "_rowwise_data") and tensor._rowwise_data is not None:
        return tensor._rowwise_data.device.index
    if hasattr(tensor, "_columnwise_data") and tensor._columnwise_data is not None:
        return tensor._columnwise_data.device.index
    if hasattr(tensor, "_data") and tensor._data is not None:
        return tensor._data.device.index
    if hasattr(tensor, "_transpose") and tensor._transpose is not None:
        return tensor._transpose.device.index
    return torch.cuda.current_device()


if IS_HIP_EXTENSION:
    def _should_use_bf16_output_for_nvfp4_tn(
        A,
        B,
        layout: str,
        out_dtype: Optional[torch.dtype],
        out,
        bias,
        quantization_params,
        debug_quantizer,
        grad: bool,
        accumulate: bool,
        ub,
        extra_output,
        gelu: bool,
    ) -> bool:
        """Work around ROCm NVFP4 TN GEMM corruption when requesting FP32 output.

        FIXME: hipBLASLt BF16xBF16->FP32 GEMM algos with ALPHA_DEVICE_VECTOR
        produce incorrect results intermittently on AMDGPU. Return True for the
        narrow path where we force BF16 output, which empirically covers the
        corruption cases.
        """
        return (
            layout == "TN"
            and out_dtype == torch.float32
            and out is None
            and bias is not None
            and quantization_params is None
            and debug_quantizer is None
            and not grad
            and not accumulate
            and ub is None
            and extra_output is None
            and not gelu
            and (isinstance(A, NVFP4TensorStorage) or isinstance(B, NVFP4TensorStorage))
        )


def _select_kernel_fp4(layout: str, grad: bool, M: int, N: int, K: int):
    """Select kernel via tuned CSV lookup, falling back to AITER heuristic."""
    from aiter.ops.gemm_op_a4w4 import get_GEMM_config

    kernel_name = ""
    split_k = 0

    if _FP4_USE_TUNED_GEMM:
        cfg = get_GEMM_config(M, N, K)
        if cfg is not None:
            kernel_name = cfg["kernelName"]
            split_k = int(cfg.get("splitK", 0))

    if _FP4_LOG_SHAPES:
        print(f"[FP4-GEMM] {layout} grad={grad} M={M} N={N} K={K} "
              f"kernel={'heuristic' if not kernel_name else kernel_name} "
              f"splitK={split_k}", flush=True)

    return kernel_name, split_k


def _fp4_gemm_core(A_fp4, A_scales, B_fp4, B_scales, out_dtype=torch.bfloat16,
                    out_buffer=None, kernel_name="", b_pre_shuffled=True, log2_k_split=0):
    """Core FP4 GEMM via AITER a4w4 kernels.

    Routes to the ASM backend when ``kernel_name`` is an ASM-mangled symbol
    (starts with ``_ZN``) or empty (heuristic). Otherwise routes to the CK
    blockscale backend, matching AITER's own ``gemm_a4w4`` dispatcher.
    """
    import aiter
    from aiter.ops.shuffle import shuffle_weight
    from aiter.ops.gemm_op_a4w4 import gemm_a4w4_blockscale

    _fp4_dtype = torch.float4_e2m1fn_x2
    A_fp4 = A_fp4.view(_fp4_dtype) if A_fp4.dtype != _fp4_dtype else A_fp4
    B_fp4 = B_fp4.view(_fp4_dtype) if B_fp4.dtype != _fp4_dtype else B_fp4
    A_scales_uint8 = A_scales.view(torch.uint8)
    B_scales_uint8 = B_scales.view(torch.uint8)

    B_shuffled = B_fp4 if b_pre_shuffled else shuffle_weight(B_fp4, layout=(16, 16))

    M = A_fp4.shape[0]
    N = B_fp4.shape[0]

    if out_buffer is not None:
        out_hp = out_buffer
    else:
        padded_M = (M + 31) // 32 * 32
        out_hp = torch.empty((padded_M, N), dtype=out_dtype, device=A_fp4.device)

    use_ck = bool(kernel_name) and kernel_name.find("_ZN") == -1
    if use_ck:
        result = gemm_a4w4_blockscale(
            A_fp4, B_shuffled, A_scales_uint8, B_scales_uint8, out_hp,
            splitK=log2_k_split,
        )
    else:
        result = aiter.gemm_a4w4_asm(
            A_fp4, B_shuffled, A_scales_uint8, B_scales_uint8,
            out_hp, kernel_name, None,
            bpreshuffle=True, log2_k_split=log2_k_split,
        )

    return result[:M, :] if result.shape[0] > M else result


def mxfp4_gemm(
    A,
    B,
    layout: str = "TN",
    out_dtype: torch.dtype = torch.bfloat16,
    bias=None,
    out=None,
    grad: bool = False,
    accumulate: bool = False,
):
    """FP4 GEMM using layout notation (TN/NN/NT).

    Layout mapping:
        TN: A=weight, B=input       -> fprop: input @ weight^T
        NN: A=weight, B=grad_output -> dgrad: grad_output @ weight
        NT: A=input, B=grad_output  -> wgrad: grad_output^T @ input
    """
    # Capture the logical batch shape from the wrapper tensor (B) before
    # extracting raw _rowwise_data/_columnwise_data buffers. The wrapper's
    # .size() reflects the original N-D logical shape, which we need to
    # restore after the 2D GEMM kernel. Reading from _rowwise_data.shape
    # alone would lose leading dims if storage was flattened to 2D.
    a_logical_shape = B.size()
    a_batch_shape = a_logical_shape[:-1]

    if layout == "TN":
        A_fp4 = B._rowwise_data
        A_scales = B._rowwise_scale_inv
        B_fp4 = A._rowwise_data
        B_scales = A._rowwise_scale_inv
        b_pre_shuffled = A._shuffle_rowwise_data
    elif layout == "NN":
        A_fp4 = B._rowwise_data
        A_scales = B._rowwise_scale_inv
        B_fp4 = A._columnwise_data
        B_scales = A._columnwise_scale_inv
        b_pre_shuffled = A._shuffle_columnwise_data
    elif layout == "NT":
        A_fp4 = B._columnwise_data
        A_scales = B._columnwise_scale_inv
        B_fp4 = A._columnwise_data
        B_scales = A._columnwise_scale_inv
        b_pre_shuffled = A._shuffle_columnwise_data

    else:
        raise ValueError(f"Unsupported layout for FP4 GEMM: {layout}")

    # AITER a4w4 kernels require 2D inputs (M, K/2). Flatten to
    # (M_total, K/2) and restore the batch shape afterward.
    if A_fp4.ndim > 2:
        A_fp4 = A_fp4.reshape(-1, A_fp4.shape[-1])

    out_flat = out
    if out is not None and out.ndim > 2:
        out_flat = out.reshape(-1, out.shape[-1])

    gemm_M = A_fp4.shape[0]
    gemm_N = B_fp4.shape[0]
    gemm_K = B_fp4.shape[-1] * 2

    kernel_name, split_k = _select_kernel_fp4(layout, grad, gemm_M, gemm_N, gemm_K)

    if accumulate and out_flat is not None:
        result = _fp4_gemm_core(
            A_fp4, A_scales, B_fp4, B_scales,
            out_dtype=out_flat.dtype, out_buffer=None,
            kernel_name=kernel_name, b_pre_shuffled=b_pre_shuffled,
            log2_k_split=split_k,
        )
        out_flat.add_(result)
        result = out_flat
    else:
        result = _fp4_gemm_core(
            A_fp4, A_scales, B_fp4, B_scales,
            out_dtype=out_dtype, out_buffer=out_flat,
            kernel_name=kernel_name, b_pre_shuffled=b_pre_shuffled,
            log2_k_split=split_k,
        )

    if bias is not None and layout == "TN" and not grad:
        bias_casted = cast_if_needed(bias, out_dtype)
        if result is not None:
            result = result + bias_casted

    if len(a_batch_shape) > 1 and result is not None:
        result = result.reshape(*a_batch_shape, result.shape[-1])

    return result


def _is_nvfp4_row_scaled_tensor(tensor: torch.Tensor) -> bool:
    """Whether tensor carries row-scaled NVFP4 global amax metadata."""
    return isinstance(tensor, NVFP4TensorStorage) and tensor._row_scaled_nvfp4


def _nvfp4_row_scaled_gemm_inputs(
    A: NVFP4TensorStorage,
    B: NVFP4TensorStorage,
    *,
    transa: bool,
) -> Tuple[NVFP4TensorStorage, NVFP4TensorStorage, torch.Tensor]:
    """Return GEMM aliases and FP32 output scales for row-scaled NVFP4."""
    A_metadata = A.get_metadata()
    weight_amax = A._amax_rowwise if transa else A._amax_columnwise
    assert weight_amax is not None and weight_amax.numel() == 1
    A_metadata["amax_rowwise" if transa else "amax_columnwise"] = weight_amax.new_ones(1)
    A_metadata["row_scaled_nvfp4"] = False

    B_metadata = B.get_metadata()
    rhs_rowwise_amax = B._amax_rowwise
    assert rhs_rowwise_amax is not None
    B_metadata["amax_rowwise"] = rhs_rowwise_amax.new_ones(1)
    B_metadata["row_scaled_nvfp4"] = False

    assert rhs_rowwise_amax.dtype == torch.float32 and weight_amax.dtype == torch.float32
    return (
        NVFP4TensorStorage(**A_metadata),
        NVFP4TensorStorage(**B_metadata),
        (rhs_rowwise_amax * weight_amax).view(-1, 1),
    )


# Warn only once when NVTE_GEMM_BACKEND=FLYDSL but the flydsl package is missing,
# so a misconfigured run is surfaced without spamming the per-GEMM hot path.
_flydsl_import_warned = False


def general_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    out_dtype: Optional[torch.dtype] = None,
    quantization_params: Optional[Quantizer] = None,
    gelu: bool = False,
    gelu_in: torch.Tensor = None,
    alpha: float = 1.0,
    beta: Optional[float] = None,
    accumulate: bool = False,
    layout: str = "TN",
    out: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    use_split_accumulator: bool = False,
    grad: bool = False,
    ub: Union[tex.CommOverlap, tex.CommOverlapP2P] = None,
    ub_type: tex.CommOverlapType = None,
    extra_output: Optional[torch.Tensor] = None,
    bulk_overlap: bool = False,
) -> Iterable[Optional[torch.Tensor]]:
    """GEMM supporting fp8 inputs."""

    assert layout in ("TN", "NN", "NT"), f"GEMM layout {layout} not supported."
    transa = layout[0] == "T"
    transb = layout[1] == "T"

    alpha = validate_gemm_scale(alpha, True)
    beta = validate_gemm_scale(beta, accumulate)

    workspace = get_cublas_workspace(A.device.index, ub is not None, False)

    # On ROCm, FP4 is dequantized to BF16 in the workspace before GEMM.
    # Compute the required extra space and extend the workspace if needed.
    if IS_HIP_EXTENSION and (
        isinstance(A, NVFP4TensorStorage) or isinstance(B, NVFP4TensorStorage)
    ):
        assert ub is None, "User buffers (comm overlap) are not supported with NVFP4"
        import math
        bf16_size = torch.bfloat16.itemsize
        fp4_extra = 0
        if isinstance(A, NVFP4TensorStorage):
            fp4_extra += math.prod(A.size()) * bf16_size
            fp4_extra += A.size(0) * 4  # alpha vector (m floats)
        if isinstance(B, NVFP4TensorStorage):
            fp4_extra += math.prod(B.size()) * bf16_size
        total_needed = fp4_extra + get_cublas_workspace_size_bytes()
        if workspace.numel() < total_needed:
            workspace = torch.empty(total_needed, dtype=torch.uint8, device=workspace.device)

    if ub_type is not None:
        assert ub is not None, (
            f"{'AG+GEMM' if ub_type == tex.CommOverlapType.AG else 'GEMM+RS'} overlap requires"
            + "a valid `ub` communicator object."
        )

    if ub is not None:
        assert ub_type is not None, "Comm+GEMM overlap requires a valid `comm_type` argument."
        if ub_type == tex.CommOverlapType.RS:
            if not (bulk_overlap and not ub.is_fp8_ubuf()):
                assert extra_output is not None, "GEMM+RS overlap requires extra output tensor."

    if out is not None:
        if not out.is_contiguous():
            raise ValueError("Output tensor is not contiguous.")

    # If A or B are custom tensors -> dispatch to quantizers's qgemm implementation
    if is_custom(A) or is_custom(B):
        return custom_gemm(
            A,
            B,
            workspace,
            out_dtype,
            quantization_params,
            gelu,
            gelu_in,
            accumulate,
            layout,
            out,
            bias,
            use_split_accumulator,
            grad,
        )

    debug_quantizer = None
    if isinstance(quantization_params, DebugQuantizer):
        debug_quantizer = quantization_params
        quantization_params = quantization_params.parent_quantizer
        A = A.get_tensor(not transa)
        B = B.get_tensor(transb)

    # Use bfloat16 as default bias_dtype
    bias_dtype = TE_DType[torch.bfloat16 if bias is None else bias.dtype]

    # MXFP4 GEMM: route to AITER a4w4 ASM kernels
    from ..tensor.storage.mxfp4_tensor_storage import MXFP4TensorStorage

    if isinstance(A, MXFP4TensorStorage) or isinstance(B, MXFP4TensorStorage):
        result = mxfp4_gemm(
            A,
            B,
            layout=layout,
            out_dtype=out_dtype if out_dtype is not None else torch.bfloat16,
            bias=bias,
            out=out,
            grad=grad,
            accumulate=accumulate,
        )
        return result, None, None, None

    if isinstance(A, Float8BlockwiseQTensorStorage) or isinstance(B, Float8BlockwiseQTensorStorage):
        # FP8 block-scaling requires split accumulator
        use_split_accumulator = True

    if IS_HIP_EXTENSION:
        use_bf16_tn_output_workaround = _should_use_bf16_output_for_nvfp4_tn(
            A,
            B,
            layout,
            out_dtype,
            out,
            bias,
            quantization_params,
            debug_quantizer,
            grad,
            accumulate,
            ub,
            extra_output,
            gelu,
        )
        out_dtype = torch.bfloat16 if use_bf16_tn_output_workaround else out_dtype

    args = (
        A,
        transa,  # transa
        B,
        transb,  # transb
        out,
        quantization_params,
        TE_DType[out_dtype] if out_dtype is not None else None,
        bias,
        bias_dtype,
        gelu,
        gelu_in,
        grad,  # grad
        workspace,
        workspace.shape[0],
        accumulate,
        use_split_accumulator,
    )
    kwargs = {
        "comm_overlap": ub,
        "comm_type": ub_type,
        "extra_output": extra_output,
        "bulk_overlap": bulk_overlap,
        "alpha": alpha,
        "beta": beta,
    }

    # ROCm-only backend: the Triton kernels use gfx942/gfx950-specific MFMA
    # instructions and autotune configs, so refuse to enable on non-HIP builds.
    # NVFP4 is not supported by the Triton path; when the Triton backend is
    # opted into, te_generic_gemm_triton raises ValueError for NVFP4 inputs
    # (surfaced as a pytest.skip via tests/pytorch/conftest.py).
    gemm_backend = get_gemm_backend()
    use_gemm_triton = IS_HIP_EXTENSION and gemm_backend == "triton"
    if use_gemm_triton:
        # Lazy: only pull in Triton when the backend is opted into. Keeps
        # `triton` off the module-import path when NVTE_GEMM_BACKEND is not
        # TRITON (the default), so stacks without pytorch-triton-rocm can
        # still use the C++ hipBLASLt path.
        from ..triton_kernels.gemm import te_generic_gemm_triton
        out, bias_grad, gelu_input, extra_output = te_generic_gemm_triton(*args, **kwargs)
    elif not _is_nvfp4_row_scaled_tensor(A) and not _is_nvfp4_row_scaled_tensor(B):
        use_gemm_flydsl = (
            IS_HIP_EXTENSION
            and get_device_compute_capability() == (9, 5)
            and gemm_backend == "flydsl"
        )
        if use_gemm_flydsl:
            try:
                # Lazy import keeps FlyDSL off the normal Transformer Engine
                # import path. It is done inside the try so a wheel built without
                # flydsl (NVTE_GEMM_BACKEND!=FLYDSL at build time) degrades to the
                # default backend instead of raising a bare ImportError.
                from ..flydsl_kernels.gemm import (
                    FlyDSLUnsupportedError,
                    te_generic_gemm_flydsl,
                )

                out, bias_grad, gelu_input, extra_output = te_generic_gemm_flydsl(
                    *args,
                    **kwargs,
                )
            except ImportError as exc:
                # NVTE_GEMM_BACKEND=FLYDSL was requested but the flydsl package is
                # missing or too old (see flydsl_kernels.gemm._MIN_FLYDSL). This
                # is a misconfiguration, not an unsupported GEMM config, so always
                # warn (once) regardless of the opt-in fallback flag before
                # degrading to the default backend.
                global _flydsl_import_warned
                if not _flydsl_import_warned:
                    _flydsl_import_warned = True
                    warnings.warn(
                        "[FLYDSL WARNING]: NVTE_GEMM_BACKEND=FLYDSL but the flydsl "
                        "package is unavailable; falling back to the default backend. "
                        f"Install a supported version (e.g. `pip install flydsl`) to "
                        f"enable it. Reason: {exc}",
                        UserWarning,
                        stacklevel=2,
                    )

                out, bias_grad, gelu_input, extra_output = tex.generic_gemm(*args, **kwargs)
            except FlyDSLUnsupportedError as exc:
                warn_fallback = os.environ.get(
                    "NVTE_FLYDSL_GEMM_WARN_FALLBACK",
                    "0",
                ).lower() not in ("", "0", "false", "no", "off")

                if warn_fallback:
                    warnings.warn(
                        "[FLYDSL WARNING]: FlyDSL GEMM does not support this configuration; "
                        f"falling back to the default backend. Reason: {exc}",
                        UserWarning,
                        stacklevel=2,
                    )

                out, bias_grad, gelu_input, extra_output = tex.generic_gemm(*args, **kwargs)
        else:
            out, bias_grad, gelu_input, extra_output = tex.generic_gemm(*args, **kwargs)
    else:
        if _is_nvfp4_row_scaled_tensor(A):
            raise NotImplementedError("Row-scaled NVFP4 GEMM does not support row-scaled A.")
        assert layout[1] == "N", "Row-scaled NVFP4 GEMM currently supports N-layout B only."
        if grad:
            raise RuntimeError(
                "Row-scaled NVFP4 GEMM currently supports fprop only. "
                "Backward NVFP4 gradient quantizers should use scalar global amax."
            )
        assert not gelu, "Row-scaled NVFP4 GEMM currently does not support fused GELU."
        assert not accumulate, "Row-scaled NVFP4 GEMM currently does not support accumulation."
        assert (
            quantization_params is None
        ), "Row-scaled NVFP4 GEMM currently does not support output quantization."
        assert ub is None, "Row-scaled NVFP4 GEMM currently does not support CommOverlap."
        assert (
            extra_output is None
        ), "Row-scaled NVFP4 GEMM currently does not support extra output."
        assert not bulk_overlap, "Row-scaled NVFP4 GEMM currently does not support bulk overlap."
        assert out is None or (
            isinstance(out, torch.Tensor) and not is_custom(out)
        ), "Row-scaled NVFP4 GEMM currently supports only plain torch.Tensor outputs."
        assert isinstance(
            A, NVFP4TensorStorage
        ), "Row-scaled NVFP4 GEMM currently requires NVFP4 A."
        # cuBLAS folds NVFP4 global amax values into GEMM alpha. Keep the row-scaled
        # recipe's global scales out of alpha and apply them in FP32 below.
        gemm_A, gemm_B, rowwise_global_scales = _nvfp4_row_scaled_gemm_inputs(A, B, transa=transa)

        requested_out, requested_out_dtype = out, out_dtype
        fp32_out = (
            torch.empty_like(requested_out, dtype=torch.float32)
            if requested_out is not None
            else None
        )
        gemm_args = list(args)
        gemm_args[0] = gemm_A  # A
        gemm_args[2] = gemm_B  # B
        gemm_args[4] = fp32_out  # out
        gemm_args[5] = None  # quantization_params
        gemm_args[6] = TE_DType[torch.float32]  # out_dtype
        gemm_args[7] = None  # bias
        out, bias_grad, gelu_input, extra_output = tex.generic_gemm(*gemm_args, **kwargs)
        out_2d = out.reshape(-1, out.shape[-1])

        assert rowwise_global_scales.dtype == torch.float32 and out.dtype == torch.float32
        assert rowwise_global_scales.numel() == out_2d.shape[0]

        out_2d.mul_(rowwise_global_scales)
        if bias is not None:
            out_2d.add_(bias.to(dtype=torch.float32))

        if requested_out is not None:
            requested_out.copy_(out.to(dtype=requested_out.dtype))
            out = requested_out
        elif requested_out_dtype is not None and requested_out_dtype != torch.float32:
            out = out.to(dtype=requested_out_dtype)

    if IS_HIP_EXTENSION and use_bf16_tn_output_workaround:
        out = cast_if_needed(out, torch.float32)

    if debug_quantizer is not None:
        out = debug_quantizer.process_gemm_output(out)

    return out, bias_grad, gelu_input, extra_output


def general_grouped_gemm(
    A: List[torch.Tensor],
    B: List[torch.Tensor],
    out: List[torch.Tensor],
    quantization_params: List[Optional[Quantizer]],
    out_dtype: torch.dtype,
    layout: str = "TN",
    m_splits: Optional[List[int]] = None,
    gelu: bool = False,
    grad=False,
    accumulate: bool = False,
    bias: Optional[List[torch.Tensor]] = None,
    use_bias: bool = False,
    use_split_accumulator: bool = False,
    D_dtype: Optional[DType] = None,
    single_output=False,
) -> Tuple[List[torch.Tensor], ...]:
    """
    TN layout Grouped GEMM with fp8 inputs.
    """
    num_gemms = len(A)

    transa = layout[0] == "T"
    transb = layout[1] == "T"

    empty_tensor = _empty_tensor()
    empty_tensors = [empty_tensor] * num_gemms

    # Use bfloat16 as default bias_dtype
    gelu_input = empty_tensors
    out_dtype = TE_DType[out[0].dtype] if D_dtype is None else D_dtype

    sm_count = get_sm_count()
    workspaces = get_cublas_workspace(A[0].device.index, False, True)

    if grad and use_bias:
        grad_bias = [
            torch.empty(B[i].size(1), dtype=out[0].dtype, device="cuda") for i in range(num_gemms)
        ]
    else:
        grad_bias = empty_tensors
    bias = bias if use_bias else empty_tensors
    if use_bias:
        bias_dtype = TE_DType[grad_bias[0].dtype] if grad else TE_DType[bias[0].dtype]
    else:
        bias_dtype = TE_DType[torch.bfloat16]

    if any(_is_nvfp4_row_scaled_tensor(tensor) for tensor in A):
        raise NotImplementedError("Row-scaled NVFP4 grouped GEMM does not support row-scaled A.")
    if any(_is_nvfp4_row_scaled_tensor(tensor) for tensor in B):
        assert D_dtype is None, "Row-scaled NVFP4 grouped GEMM currently does not support D_dtype."
        if single_output:
            assert (
                m_splits is not None
            ), "Row-scaled NVFP4 grouped GEMM requires m_splits with single output."
        out_init = out[0] if single_output else None
        if single_output:
            start_idx = 0
            out_views = []
            for i in range(num_gemms):
                size = m_splits[i]
                out_views.append(out_init[start_idx : start_idx + size])
                start_idx += size
        else:
            out_views = out
        for i in range(num_gemms):
            if out_views[i].numel() == 0:
                continue
            general_gemm(
                A[i],
                B[i],
                quantization_params=quantization_params[i],
                out_dtype=out_views[i].dtype,
                out=out_views[i],
                gelu=gelu,
                accumulate=accumulate,
                layout=layout,
                bias=bias[i] if use_bias else None,
                use_split_accumulator=use_split_accumulator,
                grad=grad,
            )
        if single_output:
            out = out_init
        return out, grad_bias, gelu_input

    if isinstance(quantization_params[0], DebugQuantizer):
        assert not gelu, "GELU not supported in debug mode"
        if single_output:
            out_init = out[0]
            start_idx = 0
            out = [None] * num_gemms
            for i in range(num_gemms):
                size = m_splits[i]
                out[i] = out_init[start_idx : start_idx + size]
                start_idx += size
        for i in range(num_gemms):
            _, bias_or_grad, _, _ = general_gemm(
                A[i],
                B[i],
                quantization_params=quantization_params[i],
                out_dtype=out[0].dtype,
                layout=layout,
                accumulate=accumulate,
                out=out[i],
                bias=bias[i] if use_bias else None,
                use_split_accumulator=use_split_accumulator,
                grad=grad,
            )
            if grad and use_bias:
                grad_bias[i] = bias_or_grad
        if single_output:
            out = out_init

        return out, grad_bias if grad else bias, None

    if gelu:
        gelu_input = [
            torch.empty_like(o, dtype=bias_dtype, memory_format=torch.contiguous_format)
            for o in out
        ]  # this should differ with respect to single output

    bias = tex.te_general_grouped_gemm(
        A,
        transa,
        B,
        transb,
        out,
        out_dtype,
        m_splits,
        grad_bias if grad else bias,
        bias_dtype,
        single_output,
        gelu_input,  # this is pre_gelu_out
        grad,  # grad
        workspaces,
        workspaces[0].shape[0],
        accumulate,
        use_split_accumulator,
        sm_count - int(os.getenv("NVTE_EXT_MARGIN_SM", str(sm_count))),
    )

    return out, bias, gelu_input


@functools.lru_cache(maxsize=None)
def get_grouped_gemm_setup_workspace_size(num_tensors: int) -> int:
    """Return workspace size for grouped GEMM pointer setup."""
    return tex.get_grouped_gemm_setup_workspace_size(num_tensors)


@functools.lru_cache(maxsize=None)
def _get_fp32_ones_tensor(num_tensors: int, device: torch.device) -> torch.Tensor:
    """Cached ones tensor."""
    return torch.ones(num_tensors, dtype=torch.float32, device=device)


@functools.lru_cache(maxsize=None)
def _get_fp32_zeros_tensor(num_tensors: int, device: torch.device) -> torch.Tensor:
    """Cached zeros tensor."""
    return torch.zeros(num_tensors, dtype=torch.float32, device=device)


def general_grouped_gemm_for_grouped_tensor(
    A,
    B,
    out,
    *,
    layout: str = "TN",
    accumulate: bool = False,
    use_split_accumulator: bool = False,
    bias=None,
    bias_scale: Optional[torch.Tensor] = None,
    grad: bool = False,
    alpha: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """
    Grouped GEMM using GroupedTensor inputs.

    This uses nvte_grouped_gemm and supports different per-matrix shapes.

    The caller must ensure that GroupedTensor metadata is already compatible with the
    underlying GEMM implementation (e.g., aligned offsets and output metadata layout).
    """
    assert layout in ("TN", "NN", "NT"), f"GEMM layout {layout} not supported."
    if grad:
        raise NotImplementedError("grad is not supported for grouped_tensor GEMM yet.")
    transa = layout[0] == "T"
    transb = layout[1] == "T"
    is_discrete_out = isinstance(out, list)
    is_discrete_in = isinstance(A, list)
    if is_discrete_in and is_discrete_out:
        raise ValueError("Both A and out are discrete. This is not supported yet.")

    if isinstance(A, GroupedTensorStorage) and A.row_scaled_nvfp4:
        raise NotImplementedError("Row-scaled NVFP4 GroupedTensor GEMM is not supported yet.")
    if isinstance(B, GroupedTensorStorage) and B.row_scaled_nvfp4:
        raise NotImplementedError("Row-scaled NVFP4 GroupedTensor GEMM is not supported yet.")
    if isinstance(out, GroupedTensorStorage) and out.row_scaled_nvfp4:
        raise NotImplementedError("Row-scaled NVFP4 GroupedTensor GEMM is not supported yet.")

    if is_discrete_out:
        # wgrad case.
        grouped_gemm_impl = tex.te_general_grouped_gemm_for_discrete_out
    elif is_discrete_in:
        # Use-case: forward pass with list of weights.
        grouped_gemm_impl = tex.te_general_grouped_gemm_for_discrete_in
    else:
        # Use-case: Single Grouped Parameter for Weight/ Weight Grads.
        grouped_gemm_impl = tex.te_general_grouped_gemm_for_grouped_tensor

    if is_discrete_out and bias is not None:
        raise ValueError(
            "Bias is not supported when out is a list (discrete_out mode) yet. "
            "Apply bias manually after the GEMM."
        )

    if bias_scale is not None and bias is None:
        raise ValueError("bias_scale requires bias to be provided.")

    num_tensors = B.num_tensors
    rowwise = B.rowwise_data
    device = rowwise.device if rowwise is not None else B.columnwise_data.device

    # Hopper (SM90) uses a single shared alpha/beta scalar;
    # Blackwell+ (SM100) supports per-group alpha/beta arrays.
    per_group = torch.cuda.get_device_capability() >= (10, 0)
    num_alphabeta = num_tensors if per_group else 1

    if alpha is None:
        alpha = _get_fp32_ones_tensor(num_alphabeta, device)
    if beta is None:
        if accumulate:
            beta = _get_fp32_ones_tensor(num_alphabeta, device)
        else:
            beta = _get_fp32_zeros_tensor(num_alphabeta, device)

    if not alpha.is_cuda or not beta.is_cuda:
        raise ValueError("alpha and beta must be CUDA tensors.")

    workspace_setup = torch.empty(
        get_grouped_gemm_setup_workspace_size(num_tensors),
        dtype=torch.uint8,
        device=device,
    )
    workspace_cublas = torch.empty(
        get_cublas_workspace_size_bytes(),
        dtype=torch.uint8,
        device=device,
    )

    sm_count = get_sm_count()
    sm_count = sm_count - int(os.getenv("NVTE_EXT_MARGIN_SM", str(sm_count)))

    return grouped_gemm_impl(
        A,
        transa,
        B,
        transb,
        out,
        bias,
        bias_scale,
        alpha,
        beta,
        workspace_setup,
        workspace_cublas,
        use_split_accumulator,
        sm_count,
    )
