# This file was modified for portability to AMDGPU
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Python interface for GEMM extensions"""

from typing import Iterable, Optional, Tuple, Union, List
import os
import functools
import torch
from torch.utils.cpp_extension import IS_HIP_EXTENSION
import transformer_engine_torch as tex
from ..constants import TE_DType
from ..utils import get_sm_count, _empty_tensor
if IS_HIP_EXTENSION:
    from ..utils import get_device_compute_capability
    from ..utils import cast_if_needed

from ..quantized_tensor import Quantizer
from ..tensor.storage.float8_blockwise_tensor_storage import Float8BlockwiseQTensorStorage
from ..tensor.storage.mxfp8_tensor_storage import MXFP8TensorStorage
from ..tensor.storage.nvfp4_tensor_storage import NVFP4TensorStorage
from ..tensor.utils import is_custom
from ..custom_recipes.gemm import custom_gemm
from ...debug.pytorch.debug_quantization import DebugQuantizer

_FP4_USE_TUNED_GEMM = int(os.environ.get("NVTE_FP4_USE_TUNED_GEMM", "1"))
_FP4_LOG_SHAPES = int(os.environ.get("NVTE_FP4_LOG_GEMM_SHAPES", "0"))

__all__ = [
    "general_gemm",
    "general_grouped_gemm",
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


def _hipkittens_workspace_bytes(m: int, n: int, k: int, layout: str) -> int:
    """Compute workspace bytes needed for HipKittens MXFP8 GEMM."""
    def _align(x: int) -> int:
        return (x + 255) & ~255

    transa = layout[0] == "T"
    transb = layout[1] == "T"
    k_iters = k // 128
    scale_k = k // 32
    sa_pk = _align(k_iters * m * 4)
    sb_pk = k_iters * n * 4
    needed = _align(sa_pk) + sb_pk
    if not transa:
        needed += _align(m * k) + _align(m * scale_k)
    if transb:
        needed += _align(n * k) + _align(n * scale_k) + _align(sb_pk)
    return needed


_workspace_cache: dict[int, torch.Tensor] = {}


def _get_or_grow_workspace(device: int, needed: int) -> torch.Tensor:
    """Return a cached workspace tensor, growing it if needed."""
    needed = max(needed, get_cublas_workspace_size_bytes())
    ws = _workspace_cache.get(device)
    if ws is None or ws.shape[0] < needed:
        ws = torch.empty(needed, dtype=torch.uint8, device=device)
        _workspace_cache[device] = ws
    return ws


def _use_hipkittens() -> bool:
    """Check if HipKittens MXFP8 backend is active."""
    if not IS_HIP_EXTENSION:
        return False
    if get_device_compute_capability() != (9, 5):
        return False
    return os.environ.get("NVTE_ROCM_USE_HIPBLASLT_MXFP8", "0") != "1"


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

    is_mxfp8 = isinstance(A, MXFP8TensorStorage)
    if is_mxfp8 and _use_hipkittens():
        a_size = A.size()
        b_size = B.size()
        m  = a_size[0] if transa else a_size[-1]
        n  = b_size[-1] if transb else b_size[0]
        k  = a_size[-1] if transa else a_size[0]
        needed = _hipkittens_workspace_bytes(m, n, k, layout)
        workspace = _get_or_grow_workspace(get_tensor_device(A), needed)
    else:
        workspace = get_cublas_workspace(get_tensor_device(A), ub is not None, False)

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

    out, bias_grad, gelu_input, extra_output = tex.generic_gemm(*args, **kwargs)

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
    D_dtype: Optional[tex.DType] = None,
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
    workspaces = get_cublas_workspace(get_tensor_device(A[0]), False, True)

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
