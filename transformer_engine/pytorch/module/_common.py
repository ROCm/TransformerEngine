# This file was modified for portability to AMDGPU
# Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Internal function used by multiple modules."""

from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import os

import torch
import queue
from torch.utils.cpp_extension import IS_HIP_EXTENSION

from .. import cpp_extensions as tex
from ..export import is_in_onnx_export_mode
from ..fp8 import get_fp8_te_dtype
from ..utils import get_default_init_method

if IS_HIP_EXTENSION:
    from ..triton_kernels.rmsnorm_triton import te_rmsnorm_fwd_noalloc_triton, te_rmsnorm_fwd_inf_triton, te_rmsnorm_bwd_triton

def _get_normalization_func(
    normalization: str, fp8_output: bool, is_grad_enabled: bool, forward: bool
):
    use_rmsnorm_triton = bool( int(os.environ.get('NVTE_USE_RMSNORM_TRITON', '0')) ) and IS_HIP_EXTENSION
    fwd_normalization_funcs = {
        ("LayerNorm", True, True): tex.layernorm_fwd_fp8,
        ("LayerNorm", True, False): tex.layernorm_fwd_fp8_inf,
        ("LayerNorm", False, True): tex.layernorm_fwd_noalloc,
        ("LayerNorm", False, False): tex.layernorm_fwd_inf,
        ("RMSNorm", True, True): tex.rmsnorm_fwd_fp8,
        ("RMSNorm", True, False): tex.rmsnorm_fwd_fp8_inf,
        ("RMSNorm", False, True): te_rmsnorm_fwd_noalloc_triton if use_rmsnorm_triton else tex.rmsnorm_fwd_noalloc,
        ("RMSNorm", False, False): te_rmsnorm_fwd_inf_triton if use_rmsnorm_triton else tex.rmsnorm_fwd_inf,
    }
    bwd_normalization_funcs = {
        "LayerNorm": tex.layernorm_bwd,
        "RMSNorm": te_rmsnorm_bwd_triton if use_rmsnorm_triton else tex.rmsnorm_bwd,
    }

    if forward:
        return fwd_normalization_funcs[(normalization, fp8_output, is_grad_enabled)]
    assert not fp8_output, "FP8 output is not supported in backward normalization!"
    assert is_grad_enabled, "Gradient has to be enabled to call backward normalization!"
    return bwd_normalization_funcs[normalization]


def _apply_normalization(
    inputmat: torch.Tensor,
    ln_out: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: Union[torch.Tensor, None],
    eps: float,
    fp8_out: bool,
    fp8_meta: Dict[str, Any],
    normalization: str,
    fwd_ln_sm_margin: int,
    zero_centered_gamma: bool,
    is_grad_enabled: bool,
    fp8_scale: Optional[torch.Tensor] = None,
    fp8_amax: Optional[torch.Tensor] = None,
    fp8_scale_inv: Optional[torch.Tensor] = None,
):
    normalization_func = _get_normalization_func(normalization, fp8_out, is_grad_enabled, True)

    inputs = (inputmat, ln_weight) if ln_bias is None else (inputmat, ln_weight, ln_bias)
    if fp8_out:
        fp8_dtype_forward = get_fp8_te_dtype(fp8_meta["recipe"], fprop_tensor=True)

        if is_grad_enabled:
            output_key = "ln_out" if normalization == "LayerNorm" else "rmsnorm_out"
            output_kwarg = {output_key: ln_out}
            output = normalization_func(
                *inputs,
                eps,
                fp8_meta["scaling_fwd"],
                tex.FP8FwdTensors.GEMM1_INPUT,
                fp8_dtype_forward,
                fwd_ln_sm_margin,
                zero_centered_gamma,
                scale=fp8_scale,
                amax=fp8_amax,
                scale_inv=fp8_scale_inv,
                **output_kwarg,
            )
        else:
            return (
                normalization_func(
                    *inputs,
                    eps,
                    fp8_meta["scaling_fwd"],
                    tex.FP8FwdTensors.GEMM1_INPUT,
                    fp8_dtype_forward,
                    fwd_ln_sm_margin,
                    zero_centered_gamma,
                    scale=fp8_scale,
                    amax=fp8_amax,
                    scale_inv=fp8_scale_inv,
                ),
                None,
                None,
            )
    else:
        if is_grad_enabled:
            output = normalization_func(*inputs, ln_out, eps, fwd_ln_sm_margin, zero_centered_gamma)
        else:
            return (
                normalization_func(*inputs, eps, fwd_ln_sm_margin, zero_centered_gamma),
                None,
                None,
            )
    if normalization == "RMSNorm":
        output = (ln_out, None, output[1])
    elif normalization == "LayerNorm":
        output = (ln_out, output[1], output[2])
    return output


class _NoopCatFunc(torch.autograd.Function):
    """Concatenate tensors, doing a no-op if possible

    See _noop_cat.

    """

    @staticmethod
    def forward(
        ctx: Any,
        dim: int,
        *tensors: Tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        # pylint: disable=missing-function-docstring

        # Check first tensor
        if not tensors:
            raise ValueError("Attempted to concatenate 0 tensors")
        num_dims = tensors[0].dim()
        if not -num_dims <= dim < num_dims:
            raise ValueError(
                "Attempted to concatenate tensor "
                f"with shape {list(tensors[0].size())} along dim {dim}"
            )
        dim %= num_dims

        # Check remaining tensors
        out_shape = list(tensors[0].size())
        split_ranges = [(0, tensors[0].size(dim))]
        for tensor in tensors[1:]:
            in_shape = list(tensor.size())
            if (
                len(in_shape) != num_dims
                or in_shape[:dim] != out_shape[:dim]
                or in_shape[dim + 1 :] != out_shape[dim + 1 :]
            ):
                raise ValueError(
                    "Attempted to concatenate tensors with shapes "
                    f"{[list(tensor.size()) for tensor in tensors]} "
                    f"along dim {dim}"
                )
            split_start = out_shape[dim]
            split_end = split_start + in_shape[dim]
            out_shape[dim] = split_end
            split_ranges.append((split_start, split_end))

        # Save state for backward
        ctx.dim = dim
        ctx.split_ranges = split_ranges

        # Out-of-place concatenation if needed
        dtype = tensors[0].dtype
        device = tensors[0].device
        strides = tensors[0].stride()
        data_ptr_stride = strides[dim] * tensors[0].element_size()
        data_ptr = tensors[0].data_ptr() + tensors[0].size(dim) * data_ptr_stride
        numel = tensors[0].numel()
        for tensor in tensors[1:]:
            if (
                tensor.dtype != dtype
                or tensor.device != device
                or tensor.stride() != strides
                or tensor.data_ptr() != data_ptr
            ):
                return torch.cat(tensors, dim=dim)
            data_ptr += tensor.size(dim) * data_ptr_stride
            numel += tensor.numel()

        # Out-of-place concatenation and reallocation if storage size is not sufficient
        if tensors[0].untyped_storage().size() < numel*tensors[0].element_size():
            out = torch.cat(tensors, dim=dim)
            for tensor, (split_start, split_end) in zip(tensors, split_ranges):
                tensor.data = out[split_start:split_end]
            return out

        # No-op concatenation
        out = tensors[0].new()
        out.set_(
            tensors[0].untyped_storage(),
            tensors[0].storage_offset(),
            out_shape,
            strides,
        )
        out.requires_grad = any(tensor.requires_grad for tensor in tensors)
        return out

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        # pylint: disable=missing-function-docstring
        grad_inputs = []
        for split_start, split_end in ctx.split_ranges:
            slices = [slice(None)] * grad_output.dim()
            slices[ctx.dim] = slice(split_start, split_end)
            grad_inputs.append(grad_output[tuple(slices)])
        return None, *grad_inputs


def _noop_cat(
    tensors: List[torch.Tensor],
    dim: int = 0,
) -> torch.Tensor:
    """Concatenate tensors, doing a no-op if possible

    If tensors are already concatenated in memory, a tensor view of
    that memory region will be returned. Otherwise the tensors will be
    concatenated out-of-place, as usual.

    """
    if not tensors:
        raise ValueError("Attempted to concatenate 0 tensors")
    if len(tensors) == 1:
        return tensors[0]
    if is_in_onnx_export_mode():
        return torch.cat(tensors, dim=dim)
    return _NoopCatFunc.apply(dim, *tensors)


@dataclass
class _ParameterInitMeta:
    """
    Stores essential metadata needed to support deferred parameter initialization.
    """

    init_fn: Optional[Callable] = get_default_init_method()
    get_rng_state_tracker: Optional[Callable] = None
    fp8_meta_index: Optional[int] = None

    def __post_init__(self):
        """Safeguard reference to the parameter's parent module and initialization function."""
        if self.init_fn is None:
            self.init_fn = get_default_init_method()


class WeightGradStore:
    """
    A class to manage weight gradient storage and computation in Transformer modules.
    This class enables split backward propagation for better memory efficiency.
    """

    def __init__(
        self, split_bw=False, use_bias=False, fuse_wgrad_accumulation=True, ub_bulk_wgrad=False
    ):
        """
        Initialize the WeightGradStore.
        Args:
            split_bw (bool): Whether to enable split backward propagation
        """
        if split_bw:
            self.context = queue.Queue()
            assert ub_bulk_wgrad == False, "ub_bulk_wgrad is not supported when enabling split_bw"
            self.enabled = split_bw
        else:
            self.context = None
            self.enabled = False

    def split_bw(self):
        """
        Get the current split backward propagation status.
        Returns:
            bool: True if split backward is enabled, False otherwise
        """
        return self.enabled

    def enable_split_bw(self):
        """Enable split backward propagation."""
        self.enabled = True

    def disable_split_bw(self):
        """Disable split backward propagation."""
        self.enabled = False

    def put(self, tensor_list, func):
        """
        Store tensors and computation function for later execution.
        Args:
            tensor_list (list): List of tensors needed for computation
            func (callable): Function to be executed with the tensors
        """
        assert self.enabled == True, "split_bw is not enabled"
        self.context.put([tensor_list, func])
        return

    def pop(self):
        """
        Execute the stored computation with the stored tensors.
        Raises an exception if the queue is empty.
        """
        assert self.enabled == True, "split_bw is not enabled"
        if self.context.qsize() > 0:
            tensor_list, func = self.context.get()
            return func(*tensor_list), tensor_list
        else:
            rank = torch.distributed.get_rank()
            raise Exception(f"Pop empty queue. rank {rank}")

    def assert_empty(self):
        """
        Assert that the queue is empty.
        Used for debugging and ensuring proper cleanup.
        """
        assert self.enabled == True, "split_bw is not enabled"
        rank = torch.distributed.get_rank()
        assert self.context.empty(), f"Queue is not empty. rank {rank}"
