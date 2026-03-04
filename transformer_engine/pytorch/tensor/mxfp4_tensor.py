# Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""Tensor class with MXFP4 data"""

from __future__ import annotations
from collections.abc import Iterable
import math
from typing import Optional, Tuple, Union

import torch
from ..triton_kernels.cast import te_quantize_triton

import transformer_engine_torch as tex
from transformer_engine_torch import DType as TE_DType

from transformer_engine.common.recipe import MXFP4BlockScaling, Recipe
from ..constants import MXFP8_BLOCK_SCALING_SIZE  # MXFP4 uses same block size
from ..utils import devices_match, round_up_to_nearest_multiple

from ._internal.mxfp4_tensor_base import MXFP4TensorBase, _FromMXFP4Func
from .quantized_tensor import QuantizedTensor, Quantizer, _IdentityFunc

MXFP4_BLOCK_SCALING_SIZE = MXFP8_BLOCK_SCALING_SIZE

aten = torch.ops.aten


def _logical_to_rowwise_data_shape(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Convert logical tensor shape to rowwise packed FP4 data shape [..., K/2]."""
    return shape[:-1] + (shape[-1] // 2,)


def _logical_to_columnwise_data_shape(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Convert logical tensor shape to columnwise packed FP4 data shape (K, M/2)."""
    return (shape[-1], math.prod(shape[:-1]) // 2)


class MXFP4Quantizer(Quantizer):
    """Builder class for FP4 tensors with MX block scaling

    High-precision tensors (e.g. in FP32 or BF16) are quantized to FP4 by
    dividing them into groups of 32 elements, each scaled and cast
    separately using AITER's per_1x32_f4_quant_hip kernel.

    The quantization produces:
    - FP4 data: [M, K/2] uint8 (2 FP4 values packed per byte)
    - E8M0 scales: [M, K/32] uint8 (one scale per 32-element block)

    """

    dtype: TE_DType

    def __init__(
        self,
        fp4_dtype: TE_DType = tex.DType.kFloat4E2M1,
        *,
        rowwise: bool = True,
        columnwise: bool = True,
    ) -> None:
        super().__init__(rowwise=rowwise, columnwise=columnwise)
        self.dtype = fp4_dtype
        assert self.dtype == tex.DType.kFloat4E2M1, "Only E2M1 format supported for MXFP4"

    def update_quantized(
        self,
        src: torch.Tensor,
        dst: QuantizedTensor,
        *,
        noop_flag: Optional[torch.Tensor] = None,
    ) -> QuantizedTensor:

        assert isinstance(dst, MXFP4Tensor), f"Cannot store quantized MXFP4 in {type(dst)} type."

        # Make sure input is in expected format
        if not devices_match(src.device, dst.device):
            src = src.to(device=dst.device)
        if not src.is_contiguous():
            src = src.contiguous()

        te_quantize_triton(src, self, dst, noop_flag)

        # Update FP4 dtype
        dst._fp4_dtype = self.dtype

        return dst

    def is_quantizable(self, inp: torch.Tensor) -> bool:
        """Returns whether or not given inp can be quantized"""
        if inp.ndim < 2:
            return False
        if inp.shape[-1] % MXFP4_BLOCK_SCALING_SIZE != 0:
            return False
        if math.prod(inp.shape[:-1]) % MXFP4_BLOCK_SCALING_SIZE != 0:
            return False
        return True

    def make_empty(
        self,
        shape: Iterable[int],
        *,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        requires_grad: bool = False,
    ) -> MXFP4Tensor:

        # Canonicalize tensor attributes
        if device is None:
            device = torch.device("cuda")

        assert (
            shape[-1] % MXFP4_BLOCK_SCALING_SIZE == 0
            and math.prod(shape[:-1]) % MXFP4_BLOCK_SCALING_SIZE == 0
        ), (
            f"Incorrect shape {shape} for MXFP4. Tensor dims must be divisible by"
            f" {MXFP4_BLOCK_SCALING_SIZE}"
        )

        M = math.prod(shape[:-1])
        K = shape[-1]

        # Allocate FP4 data: [M, K/2]
        rowwise_data = torch.empty(M, K // 2, dtype=torch.uint8, device=device)
        
        # Allocate PADDED scale tensors for shuffle compatibility
        rowwise_scale_K = math.ceil(K / MXFP4_BLOCK_SCALING_SIZE)
        rowwise_scale_inv = torch.zeros(
                round_up_to_nearest_multiple(M, 256),
                round_up_to_nearest_multiple(rowwise_scale_K, 8),
                dtype=torch.uint8,
                device=device,
            )

        # Allocate FP4 data transpose if needed
        columnwise_data = None
        columnwise_scale_inv = None
        if self.columnwise_usage:
            columnwise_data = torch.empty(K, M // 2, dtype=torch.uint8, device=device)
            colwise_scale_M = math.ceil(M / MXFP4_BLOCK_SCALING_SIZE)
            columnwise_scale_inv = torch.zeros(
                    round_up_to_nearest_multiple(K, 256),
                    round_up_to_nearest_multiple(colwise_scale_M, 8),
                    dtype=torch.uint8,
                    device=device,
                )

        # Construct FP4 tensor
        return MXFP4Tensor(
            shape=shape,
            dtype=dtype,
            fp4_dtype=self.dtype,
            rowwise_data=rowwise_data,
            rowwise_scale_inv=rowwise_scale_inv,
            columnwise_data=columnwise_data,
            columnwise_scale_inv=columnwise_scale_inv,
            quantizer=self,
            requires_grad=requires_grad,
        )

    def calibrate(self, tensor: torch.Tensor) -> None:
        # No calibration needed for MXFP4 (uses per-block current scaling)
        pass

    def create_tensor_from_data(
        self,
        data: torch.Tensor,
        scale_inv: torch.Tensor,
        fake_dtype: torch.dtype,
        fp4_dtype: TE_DType = tex.DType.kFloat4E2M1,
    ) -> MXFP4Tensor:
        """Create a new MXFP4Tensor from data and scale_inv."""
        # data is packed [M, K/2]; logical shape is [M, K]
        logical_shape = data.shape[:-1] + (data.shape[-1] * 2,)
        return MXFP4Tensor(
            shape=logical_shape,
            dtype=fake_dtype,
            rowwise_data=data,
            rowwise_scale_inv=scale_inv,
            columnwise_data=None,
            columnwise_scale_inv=None,
            fp4_dtype=fp4_dtype,
            quantizer=self,
        )

    def _get_compatible_recipe(self) -> Union[type[Recipe], None]:
        return MXFP4BlockScaling


class MXFP4Tensor(MXFP4TensorBase, QuantizedTensor):
    """Experimental tensor class with FP4 data

    The tensor presents as having a standard, higher-precision dtype,
    but the data itself is (scaled) FP4. The FP4 data is packed with
    2 FP4 values per byte.

    For MXFP4 forward-only training:
    - Forward pass: Uses FP4 quantized data with AITER gemm_a4w4
    - Backward pass: Uses high-precision (BF16) gradients

    Parameters
    ----------
    data: torch.Tensor
          Raw FP4 data in a uint8 tensor [M, K/2]
    fp4_dtype: transformer_engine_torch.DType, default = kFloat4E2M1
               FP4 format (E2M1: 2 bits exponent, 1 bit mantissa)
    fp4_scale: torch.Tensor
               E8M0 scaling factors [M, K/32], one per 32-element block
    dtype: torch.dtype, default = torch.float32
           Nominal tensor datatype

    """

    # NOTE: We reorder the *args so that we can instantiate a MXFP4TensorBase with positional args,
    # which significantly reduces the Pybind11 overhead when calling the constructor from C++.
    def __new__(
        cls,
        *args,
        rowwise_data: Optional[torch.Tensor],
        rowwise_scale_inv: Optional[torch.Tensor],
        columnwise_data: Optional[torch.Tensor],
        columnwise_scale_inv: Optional[torch.Tensor],
        fp4_dtype: TE_DType,
        quantizer: Optional[Quantizer],
        **kwargs,
    ):
        instance = super().__new__(
            cls,
            rowwise_data,
            rowwise_scale_inv,
            columnwise_data,
            columnwise_scale_inv,
            fp4_dtype,
            quantizer,
            *args,
            **kwargs,
        )
        return instance

    def __repr__(self, *, tensor_contents=None):
        return (
            f"MXFP4Tensor(fp4_dtype={self._fp4_dtype}, "
            f"shape={self.shape}, "
            f"rowwise_data_shape={self._rowwise_data.shape if self._rowwise_data is not None else None})"
        )

    def dequantize(self, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """
        Construct plain PyTorch tensor from MXFP4Tensor

        By default the resulting tensor's dtype is the MXFP4Tensor's nominal dtype.
        
        Note: For MXFP4 forward-only training, this is typically not needed as
        backward pass uses high-precision activations.
        """
        # Convert PyTorch dtype to TE dtype
        if dtype is None:
            dtype = self.dtype

        if torch.is_grad_enabled():
            return _FromMXFP4Func.apply(self, dtype)
        return _FromMXFP4Func.forward(None, self, dtype)

    def _get_quantizer(self) -> Quantizer:
        """Get builder for quantized tensor

        Quantizer can be used for in-place operations.

        """
        if self._quantizer is not None:
            return self._quantizer
        return MXFP4Quantizer(
            fp4_dtype=self._fp4_dtype,
        )

    def quantize_(
        self,
        tensor: torch.Tensor,
        *,
        noop_flag: Optional[torch.Tensor] = None,
    ) -> MXFP4Tensor:
        """Quantize a tensor and store result in this tensor

        This updates the FP4 data and scales in-place.

        """
        if isinstance(tensor, QuantizedTensor):
            return self.quantize_(tensor.dequantize())
        self._get_quantizer().update_quantized(tensor, self, noop_flag=noop_flag)
        return self

    def detach(self) -> MXFP4Tensor:
        # pylint: disable=missing-function-docstring
        # TODO(ksivamani): Fix the detach bug
        return MXFP4Tensor.make_like(self)

    def clone(self) -> MXFP4Tensor:
        # pylint: disable=missing-function-docstring
        assert self._rowwise_data is not None
        rowwise_data = self._rowwise_data.detach().clone()
        columnwise_data = None
        if self._columnwise_data is not None:
            columnwise_data = self._columnwise_data.detach().clone()
        return _IdentityFunc.apply(
            self,
            {
                "rowwise_data": rowwise_data,
                "columnwise_data": columnwise_data,
            },
        )

    def view(self, *shape: Tuple[int]) -> MXFP4Tensor:
        # pylint: disable=missing-function-docstring
        return _ViewFunc.apply(self, shape)

    def reshape(self, *shape: Tuple[int]) -> MXFP4Tensor:
        # pylint: disable=missing-function-docstring
        return _ReshapeFunc.apply(self, shape)


    def contiguous(
        self,
        memory_format: torch.memory_format = torch.contiguous_format,
    ) -> MXFP4Tensor:
        """Returns tensor with data in provided memory format

        Returns `self` if data is already in correct memory format.

        """
        if self._rowwise_data is not None and self._rowwise_data.is_contiguous(
            memory_format=memory_format
        ):
            return self
        if self._columnwise_data is not None and self._columnwise_data.is_contiguous(
            memory_format=memory_format
        ):
            return self
        raise ValueError("MXFP4Tensor does not support different memory formats!")
    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):

        # View op
        if func == aten.view.default:
            tensor = args[0]
            data = tensor._rowwise_data
            # User shape is logical (..., K); packed data has shape (..., K/2)
            user_shape = args[1] if len(args) == 2 and isinstance(args[1], (tuple, list)) else tuple(args[1:])
            user_shape = tuple(user_shape)
            if -1 in user_shape:
                user_shape = list(user_shape)
                logical_numel = math.prod(tensor.shape)
                inferred = logical_numel // math.prod(d for d in user_shape if d != -1)
                for i, d in enumerate(user_shape):
                    if d == -1:
                        user_shape[i] = inferred
                        break
                user_shape = tuple(user_shape)
            data_shape = _logical_to_rowwise_data_shape(user_shape)
            out_data = data.__torch_dispatch__(
                func,
                types,
                [data] + list(data_shape),
                kwargs,
            )
            return MXFP4Tensor(
                shape=user_shape,
                dtype=tensor.dtype,
                rowwise_data=out_data,
                rowwise_scale_inv=tensor._rowwise_scale_inv,
                columnwise_data=tensor._columnwise_data,
                columnwise_scale_inv=tensor._columnwise_scale_inv,
                quantizer=tensor._quantizer,
                requires_grad=False,
                fp4_dtype=tensor._fp4_dtype,
            )

        # Default case
        return super().__torch_dispatch__(func, types, args, kwargs)

    @classmethod
    def _make_in_reduce_ex(
        cls,
        rowwise_data: torch.Tensor,
        rowwise_scale_inv: torch.Tensor,
        columnwise_data: torch.Tensor,
        columnwise_scale_inv: torch.Tensor,
        fp4_dtype: TE_DType,
        dtype: torch.dtype,
        shape: torch.shape,
        quantizer: Optional[Quantizer] = None,
    ) -> MXFP4Tensor:
        """Build MXFP4Tensor, for use in __reduce__

        __reduce_ex__ assumes object constructor has positional
        arguments.

        """
        return MXFP4Tensor(
            rowwise_data=rowwise_data,
            rowwise_scale_inv=rowwise_scale_inv,
            fp4_dtype=fp4_dtype,
            columnwise_data=columnwise_data,
            columnwise_scale_inv=columnwise_scale_inv,
            dtype=dtype,
            shape=shape,
            quantizer=quantizer,
        )

    def __reduce_ex__(self, protocol: int) -> tuple:
        """Custom pickling"""
        return (
            MXFP4Tensor._make_in_reduce_ex,
            (
                self._rowwise_data,
                self._rowwise_scale_inv,
                self._columnwise_data,
                self._columnwise_scale_inv,
                self._fp4_dtype,
                self.dtype,
                self.shape,
                self._quantizer,
            ),
        )

    def _get_data(self) -> MXFP4Tensor:
        """Get tensor data property"""
        return super().data

    @torch.no_grad()
    def _set_data(self, tensor: torch.Tensor) -> None:
        """Set tensor data property

        Just takes FP4 data if setting from a MXFP4Tensor. Otherwise
        casts to FP4.

        """

        # Tensor device
        new_device = tensor.device if tensor.is_cuda else self.device
        if not devices_match(new_device, tensor.device):
            tensor = tensor.to(device=new_device)

        # Just copy FP4 data if other tensor is MXFP4Tensor
        if isinstance(tensor, MXFP4Tensor):
            if (  # pylint: disable=too-many-boolean-expressions
                self.size() != tensor.size()
                or self.stride() != tensor.stride()
                or self.storage_offset() != tensor.storage_offset()
                or self.dtype != tensor.dtype
                or self.layout != tensor.layout
                or not devices_match(self.device, new_device)
            ):
                dummy_tensor = torch.Tensor._make_wrapper_subclass(
                    MXFP4Tensor,
                    tensor.size(),
                    strides=tensor.stride(),
                    storage_offset=tensor.storage_offset(),
                    dtype=tensor.dtype,
                    layout=tensor.layout,
                    requires_grad=tensor.requires_grad,
                    device=new_device,
                )
                # pylint: disable=unnecessary-dunder-call
                super(MXFP4Tensor, type(self)).data.__set__(self, dummy_tensor)
            self._rowwise_data = tensor._rowwise_data
            self._columnwise_data = tensor._columnwise_data
            self._quantizer = tensor._quantizer.copy()
            self._fp4_dtype = tensor._fp4_dtype
            self._rowwise_scale_inv = tensor._rowwise_scale_inv
            self._columnwise_scale_inv = tensor._columnwise_scale_inv
            return

        # Quantize to FP4
        assert self._quantizer is not None, "Can't quantize without a quantizer"
        self._quantizer.internal = False
        self.data = self._quantizer.quantize(tensor)
        if self.requires_grad != tensor.requires_grad:
            self.requires_grad_(requires_grad=tensor.requires_grad)

    # Cast to FP4 when setting MXFP4Tensor.data
    data = property(_get_data, _set_data)


class _ViewFunc(torch.autograd.Function):
    """View function

    View the MXFP4Tensor using the provided shape.

    """

    @staticmethod
    def forward(
        ctx,
        tensor: MXFP4Tensor,
        shape: Optional[list[int]] = None,
    ) -> MXFP4Tensor:
        # pylint: disable=missing-function-docstring

        # Return input tensor if shape is not provided
        ctx.shape = tensor.shape
        if shape is None:
            return tensor

        # Canonicalize shape
        if not isinstance(shape, Iterable):
            shape = [shape]
        elif len(shape) == 1 and isinstance(shape[0], Iterable):
            shape = shape[0]
        if -1 in shape:
            shape = list(shape)
            d_inferred = -math.prod(ctx.shape) // math.prod(shape)
            for i, d in enumerate(shape):
                if d == -1:
                    shape[i] = d_inferred
                    break
        if shape[-1] != ctx.shape[-1]:
            raise RuntimeError(
                "MXFP4Tensor does not support reshaping inner dimension "
                f"(attempted to reshape dims={tuple(tensor.shape)} to {tuple(shape)})"
            )

        # Packed data shapes: rowwise [..., K/2], columnwise (K, M/2)
        shape_t = tuple(shape)
        rowwise_data_shape = _logical_to_rowwise_data_shape(shape_t)
        colwise_data_shape = _logical_to_columnwise_data_shape(shape_t)
        new_rowwise_data = None
        new_columnwise_data = None
        if tensor._rowwise_data is not None:
            new_rowwise_data = tensor._rowwise_data.view(*rowwise_data_shape)
        if tensor._columnwise_data is not None:
            new_columnwise_data = tensor._columnwise_data.view(*colwise_data_shape)
        return MXFP4Tensor(
            shape,
            tensor.dtype,
            rowwise_data=new_rowwise_data,
            rowwise_scale_inv=tensor._rowwise_scale_inv,
            columnwise_data=new_columnwise_data,
            columnwise_scale_inv=tensor._columnwise_scale_inv,
            fp4_dtype=tensor._fp4_dtype,
            quantizer=tensor._quantizer,
        )

    @staticmethod
    def backward(
        ctx,
        grad: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        # pylint: disable=missing-function-docstring

        if isinstance(grad, MXFP4Tensor):
            shape_t = tuple(ctx.shape)
            rowwise_data_shape = _logical_to_rowwise_data_shape(shape_t)
            colwise_data_shape = _logical_to_columnwise_data_shape(shape_t)
            new_data = (
                grad._rowwise_data.view(*rowwise_data_shape) if grad._rowwise_data is not None else None
            )
            if grad._columnwise_data is not None:
                new_columnwise_data = grad._columnwise_data.view(*colwise_data_shape)
            else:
                new_columnwise_data = None
            dgrad = MXFP4Tensor(
                ctx.shape,
                grad.dtype,
                rowwise_data=new_data,
                rowwise_scale_inv=grad._rowwise_scale_inv,
                columnwise_data=new_columnwise_data,
                columnwise_scale_inv=grad._columnwise_scale_inv,
                fp4_dtype=grad._fp4_dtype,
                quantizer=grad._quantizer,
            )
            return dgrad, None
        return grad.view(ctx.shape), None


class _ReshapeFunc(torch.autograd.Function):
    """Reshape function

    Reshape the MXFP4Tensor using the provided shape.

    """

    @staticmethod
    def forward(
        ctx,
        tensor: MXFP4Tensor,
        shape: Optional[list[int]] = None,
    ) -> MXFP4Tensor:
        # pylint: disable=missing-function-docstring

        # Return input tensor if shape is not provided
        ctx.shape = tensor.shape
        if shape is None:
            return tensor

        # Canonicalize shape
        if not isinstance(shape, Iterable):
            shape = [shape]
        elif len(shape) == 1 and isinstance(shape[0], Iterable):
            shape = shape[0]
        if -1 in shape:
            shape = list(shape)
            d_inferred = -math.prod(ctx.shape) // math.prod(shape)
            for i, d in enumerate(shape):
                if d == -1:
                    shape[i] = d_inferred
                    break
        if shape[-1] != ctx.shape[-1]:
            raise RuntimeError(
                "MXFP4Tensor does not support reshaping inner dimension "
                f"(attempted to reshape dims={tuple(tensor.shape)} to {tuple(shape)})"
            )

        # Packed data shapes: rowwise [..., K/2], columnwise (K, M/2)
        shape_t = tuple(shape)
        rowwise_data_shape = _logical_to_rowwise_data_shape(shape_t)
        colwise_data_shape = _logical_to_columnwise_data_shape(shape_t)
        new_rowwise_data = None
        new_columnwise_data = None
        if tensor._rowwise_data is not None:
            new_rowwise_data = tensor._rowwise_data.reshape(*rowwise_data_shape)
        if tensor._columnwise_data is not None:
            new_columnwise_data = tensor._columnwise_data.view(*colwise_data_shape)

        return MXFP4Tensor(
            shape,
            tensor.dtype,
            rowwise_data=new_rowwise_data,
            rowwise_scale_inv=tensor._rowwise_scale_inv,
            columnwise_data=new_columnwise_data,
            columnwise_scale_inv=tensor._columnwise_scale_inv,
            fp4_dtype=tensor._fp4_dtype,
            quantizer=tensor._quantizer,
        )

    @staticmethod
    def backward(
        ctx,
        grad: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        # pylint: disable=missing-function-docstring

        if isinstance(grad, MXFP4Tensor):
            shape_t = tuple(ctx.shape)
            rowwise_data_shape = _logical_to_rowwise_data_shape(shape_t)
            colwise_data_shape = _logical_to_columnwise_data_shape(shape_t)
            new_rowwise_data = None
            new_columnwise_data = None
            if grad._rowwise_data is not None:
                new_rowwise_data = grad._rowwise_data.view(*rowwise_data_shape)
            if grad._columnwise_data is not None:
                new_columnwise_data = grad._columnwise_data.view(*colwise_data_shape)
            dgrad = MXFP4Tensor(
                ctx.shape,
                grad.dtype,
                rowwise_data=new_rowwise_data,
                rowwise_scale_inv=grad._rowwise_scale_inv,
                columnwise_data=new_columnwise_data,
                columnwise_scale_inv=grad._columnwise_scale_inv,
                fp4_dtype=grad._fp4_dtype,
                quantizer=grad._quantizer,
            )
            return dgrad, None
        return grad.view(ctx.shape), None
