# Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""Mixin class holding data specific for MXFP4Tensor"""

from __future__ import annotations
from typing import Optional, Dict, Any, Tuple
import torch

from transformer_engine_torch import DType as TE_DType

from ..quantized_tensor import QuantizedTensorBase
from ..quantized_tensor import Quantizer
from ...utils import _empty_tensor


class _FromMXFP4Func(torch.autograd.Function):
    """Cast from MXFP4 to other dtype"""

    @staticmethod
    def forward(
        _ctx: Optional[torch.autograd.function.FunctionCtx],  # unused
        tensor: MXFP4TensorBase,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        # pylint: disable=missing-function-docstring
        # For MXFP4, return cached high-precision data if available
        # Full dequantization from FP4 will be implemented later with AITER support
        if hasattr(tensor, '_data') and tensor._data is not None:
            # Return cached high-precision data (used during model initialization/teardown)
            return tensor._data.to(dtype) if tensor._data.dtype != dtype else tensor._data
        
        # If no cached data, we would need to dequantize from rowwise FP4 data
        # This path should not be hit in forward-only MXFP4 training

        # TODO: Implement MXFP4 dequantization from packed FP4 using AITER kernels
        raise NotImplementedError(
            "MXFP4 dequantization from packed FP4 not yet implemented. "
            "This should only be called during model teardown with cached high-precision data."
        )

    @staticmethod
    def backward(
        _ctx: torch.autograd.function.FunctionCtx,  # unused
        grad: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        # pylint: disable=missing-function-docstring
        # Assume that we want gradients in full precision
        return grad, None


class MXFP4TensorBase(QuantizedTensorBase):
    """Mixin class that holds data attributes of MXFP4Tensor.

    MXFP4Tensor inherits from the PyTorch tensor class and this mixin
    class. If this class is instantiated directly, it has the same
    data, lower CPU overhead, and less functionality. It should only
    be instantiated directly for performance-critical internal usage.

    FP4 data format:
    - Data: [M, K/2] uint8 tensor (2 FP4 values packed per byte)
    - Scale: [M, K/32] uint8 tensor (E8M0 format, one scale per 32-element block)

    """

    _rowwise_data: Optional[torch.Tensor]  # [M, K/2] uint8
    _columnwise_data: Optional[torch.Tensor]  # [K, M/2] uint8 (transposed)
    _quantizer: Optional[Quantizer]
    _fp4_dtype: TE_DType
    _rowwise_scale: torch.Tensor  # [M, K/32] uint8 E8M0
    _columnwise_scale: torch.Tensor  # [K, M/32] uint8 E8M0
    _original_shape: Optional[Tuple[int, ...]]  # Original shape before reshape (for 3D inputs)

    def __new__(
        cls,
        *args,
        rowwise_data: Optional[torch.Tensor],
        rowwise_scale: torch.Tensor,
        columnwise_data: Optional[torch.Tensor],
        columnwise_scale: torch.Tensor,
        fp4_dtype: TE_DType,
        quantizer: Optional[Quantizer] = None,
        original_shape: Optional[Tuple[int, ...]] = None,
        **kwargs,
    ):
        instance = super().__new__(cls, *args, **kwargs)
        instance._rowwise_data = rowwise_data
        instance._columnwise_data = columnwise_data
        instance._quantizer = quantizer
        instance._fp4_dtype = fp4_dtype
        instance._rowwise_scale = rowwise_scale
        instance._columnwise_scale = columnwise_scale
        instance._original_shape = original_shape

        return instance

    def clear(self):
        """Deallocate this tensor's memory. Typically not needed and must be used carefully."""
        for t in (
            self._rowwise_data,
            self._columnwise_data,
            self._rowwise_scale,
            self._columnwise_scale,
        ):
            if t is not None:
                t.data = _empty_tensor()

    def get_metadata(self) -> Dict[str, Any]:
        """Get this tensor's metadata."""
        return {
            "rowwise_data": self._rowwise_data,
            "rowwise_scale": self._rowwise_scale,
            "columnwise_data": self._columnwise_data,
            "columnwise_scale": self._columnwise_scale,
            "fp4_dtype": self._fp4_dtype,
            "quantizer": self._quantizer,
        }

    def prepare_for_saving(self) -> Tuple[list[Optional[torch.Tensor]], MXFP4TensorBase]:
        """Prepare the tensor base for saving for backward"""
        tensors = [
            self._rowwise_data,
            self._columnwise_data,
            self._rowwise_scale,
            self._columnwise_scale,
        ]
        self._rowwise_data = None
        self._columnwise_data = None
        self._rowwise_scale = None
        self._columnwise_scale = None
        return tensors, self

    def restore_from_saved(
        self, tensors: list[Optional[torch.Tensor]]
    ) -> list[Optional[torch.Tensor]]:
        """Restore the tensor base data from the saved tensors list."""
        self._rowwise_data = tensors[0]
        self._columnwise_data = tensors[1]
        self._rowwise_scale = tensors[2]
        self._columnwise_scale = tensors[3]
        return tensors[4:]

    def get_data_tensors(self):
        """Get this Tensor's data."""
        return self._rowwise_data, self._columnwise_data

    def dequantize(self, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Dequantize to a higher precision."""
        return _FromMXFP4Func.forward(None, self, dtype)

    def size(self, *args, **kwargs):
        # pylint: disable=missing-function-docstring
        if self._rowwise_data is not None:
            # Note: Rowwise data is [M, K/2] but we report logical size [M, K]
            shape = list(self._rowwise_data.size(*args, **kwargs))
            if len(shape) > 0:
                shape[-1] = shape[-1] * 2  # Unpacked size
            return torch.Size(shape) if not args and not kwargs else shape
        # Similar logic for columnwise data
        shape = list(self._columnwise_data.size(*args, **kwargs))
        if len(shape) > 0:
            shape[-1] = shape[-1] * 2  # Unpacked size
        return torch.Size(shape) if not args and not kwargs else shape

    def __repr__(self):
        return (
            "MXFP4TensorBase("
            f"fp4_dtype={self._fp4_dtype}, "
            f"rowwise_data_shape={self._rowwise_data.shape if self._rowwise_data is not None else None}, "
            f"rowwise_scale_shape={self._rowwise_scale.shape if self._rowwise_scale is not None else None}"
            ")"
        )

    def update_usage(
        self,
        rowwise_usage: Optional[bool] = None,
        columnwise_usage: Optional[bool] = None,
    ):
        """
        Update the usage of the MXFP4TensorBase.
    
        """

        # Default usage is based on available data
        if rowwise_usage is None:
            rowwise_usage = self._rowwise_data is not None
        if columnwise_usage is None:
            columnwise_usage = self._columnwise_data is not None

        # Update row-scaled data
        if rowwise_usage:
            if self._rowwise_data is None:
                raise RuntimeError(
                    "Requested row-wise usage, but MXFP4Tensor is missing row-scaled FP4 data"
                )
            if self._rowwise_scale is None:
                raise RuntimeError(
                    "Requested row-wise usage, but MXFP4Tensor is missing row-scaled scales"
                )
        else:
            self._rowwise_data = None
            self._rowwise_scale = None

        # Update column-scaled data
        if columnwise_usage:
            if self._columnwise_data is None:
                raise RuntimeError(
                    "Requested column-wise usage, but MXFP4Tensor is missing column-scaled FP4 data"
                )
            if self._columnwise_scale is None:
                raise RuntimeError(
                    "Requested column-wise usage, but MXFP4Tensor is missing column-scaled scales"
                )
        else:
            self._columnwise_data = None
            self._columnwise_scale = None
