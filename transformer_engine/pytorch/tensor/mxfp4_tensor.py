# Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""Tensor class with MXFP4 data"""

from __future__ import annotations
from collections.abc import Iterable
import math
from typing import Optional

import torch
import transformer_engine_torch as tex
from transformer_engine_torch import DType as TE_DType

from ..constants import MXFP8_BLOCK_SCALING_SIZE  # MXFP4 uses same block size
from ..utils import devices_match

from ._internal.mxfp4_tensor_base import MXFP4TensorBase, _FromMXFP4Func
from .quantized_tensor import QuantizedTensor, Quantizer

MXFP4_BLOCK_SCALING_SIZE = MXFP8_BLOCK_SCALING_SIZE

aten = torch.ops.aten


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
        """
        A bulider class that returns a MXFP4Tensor object.

        Parameters
        ----------
        fp4_dtype: transformer_engine_torch.DType, default = kFloat4E2M1
            FP4 format (E2M1: 2 bits exponent, 1 bit mantissa)
        rowwise: bool, default = True
            Whether to quantize the tensor rowwise
        columnwise: bool, default = True
            Whether to quantize the tensor columnwise
        """
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
        """Quantize a tensor and store result in this tensor

        This updates the FP4 data and scales in-place.

        Parameters
        ----------
        src: torch.Tensor
            The tensor to quantize
        dst: QuantizedTensor
            The tensor to store the quantized result
        noop_flag: Optional[torch.Tensor], default = None
            A flag to indicate if the quantization should be skipped
        Returns
        -------
        QuantizedTensor
            The quantized tensor
        """

        assert isinstance(dst, MXFP4Tensor), f"Cannot store quantized MXFP4 in {type(dst)} type."

        # Make sure input is in expected format
        if not devices_match(src.device, dst.device):
            src = src.to(device=dst.device)
        if not src.is_contiguous():
            src = src.contiguous()

        # MXFP4 only supports 2D tensors (matrices) for AITER gemm_a4w4
        # Reshape if needed (e.g. [N, 1, K] -> [N, K])
        original_shape = src.shape
        if src.dim() > 2:
            src = src.view(-1, src.shape[-1])
        if src.dim() != 2:
            raise ValueError(
                f"MXFP4 quantization requires 2D tensors for AITER gemm_a4w4, "
                f"but got tensor with shape {original_shape} (dim={len(original_shape)}). "
                f"Biases and other 1D tensors should not be quantized with MXFP4."
            )

        try:
            import aiter
        except ImportError:
            raise ImportError(
                "AITER library not found. Please install AITER to use MXFP4 quantization. "
                "Install via: pip install -e /path/to/aiter"
            )

        # Quantize using AITER's triton quantizer
        # Wrap in DisableTorchDispatch to prevent recursive dequantization, TODO revisit DisableTorchDispatch
        with torch._C._DisableTorchDispatch():
            quantizer = aiter.get_triton_quant(aiter.QuantType.per_1x32)
            fp4_data, e8m0_scale = quantizer(
                src, shuffle=True
            )

            # Store rowwise quantized data
            if dst._rowwise_data is not None:
                dst._rowwise_data.copy_(fp4_data)
                dst._rowwise_scale.copy_(e8m0_scale)

            # Store columnwise quantized data (if needed)
            if dst._columnwise_data is not None:
                # For columnwise, we need to transpose first, then quantize
                # Columnwise quant == Quantize(Transpose(src), rowwise=True)
                src_t = src.t().contiguous()
                fp4_data_t, e8m0_scale_t = quantizer(
                    src_t, shuffle=True
                )
                dst._columnwise_data.copy_(fp4_data_t)
                dst._columnwise_scale.copy_(e8m0_scale_t)

        # Update FP4 dtype
        dst._fp4_dtype = self.dtype
        
        # # Store original shape if it was 3D (for reshaping output after GEMM)
        # if len(original_shape) > 2:
        #     dst._original_shape = original_shape
        
        # # Cache high-precision tensor for potential dequantization
        # dst._data = src

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

        shape = tuple(shape)
        assert (
            shape[-1] % MXFP4_BLOCK_SCALING_SIZE == 0
            and math.prod(shape[:-1]) % MXFP4_BLOCK_SCALING_SIZE == 0
        ), (
            f"Incorrect shape {shape} for MXFP4. Tensor dims must be divisible by"
            f" {MXFP4_BLOCK_SCALING_SIZE}"
        )

        M = math.prod(shape[:-1])
        K = shape[-1]

        # Allocate FP4 data: [M, K/2] with proper FP4 dtype (not uint8!)
        # AITER returns torch.float4_e2m1fn_x2, so we must use that dtype
        rowwise_data = torch.empty(M, K // 2, dtype=torch.float4_e2m1fn_x2, device=device)
        rowwise_scale = torch.empty(M, K // MXFP4_BLOCK_SCALING_SIZE, dtype=torch.float8_e8m0fnu, device=device)

        # Allocate FP4 data transpose if needed
        columnwise_data = None
        columnwise_scale = None
        if self.columnwise_usage:
            # For columnwise: [K, M/2] and [K, M/32]
            columnwise_data = torch.empty(K, M // 2, dtype=torch.float4_e2m1fn_x2, device=device)
            columnwise_scale = torch.empty(
                K, M // MXFP4_BLOCK_SCALING_SIZE, dtype=torch.float8_e8m0fnu, device=device
            )

        # Construct FP4 tensor
        return MXFP4Tensor(
            shape=shape,
            dtype=dtype,
            fp4_dtype=self.dtype,
            rowwise_data=rowwise_data,
            rowwise_scale=rowwise_scale,
            columnwise_data=columnwise_data,
            columnwise_scale=columnwise_scale,
            quantizer=self,
            original_shape=None,  # Will be set during update_quantized if needed
            requires_grad=requires_grad,
        )

    def calibrate(self, tensor: torch.Tensor) -> None:
        # No calibration needed for MXFP4 (uses per-block current scaling)
        pass


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

    def detach(self) -> MXFP4Tensor:
        # pylint: disable=missing-function-docstring
        return MXFP4Tensor.make_like(self)

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
        quantizer = self._get_quantizer()
        return quantizer.update_quantized(tensor, self, noop_flag=noop_flag)

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
