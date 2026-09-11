# This file was modified for portability to AMDGPU
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Mixin class holding data specific for Float8BlockwiseQTensor"""

from __future__ import annotations
import math
import os
from typing import Optional, Dict, Any, Tuple, Union
import torch

import transformer_engine_torch as tex

from ...quantized_tensor import QuantizedTensorStorage, Quantizer
from .._quantization_helpers import safe_quantized_repr

from ...constants import TE_DType_To_Torch, DType

from ...utils import _empty_tensor


class _FromFloat8BlockwiseFunc(torch.autograd.Function):
    """Cast from Float8 blockwise to other dtype"""

    @staticmethod
    def forward(
        _ctx: Optional[torch.autograd.function.FunctionCtx],  # unused
        tensor: Float8BlockwiseQTensorStorage,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        # pylint: disable=missing-function-docstring
        if dtype is None:
            dtype = tensor._dtype

        if tensor._rowwise_data is not None and tensor._rowwise_data.numel() == 0:
            return torch.empty(tensor.size(), dtype=dtype, device=tensor.device)

        block_len = 128
        if not tensor._is_2D_scaled:
            return tensor._dequantize_vectorwise(dtype=dtype)

        def format_scale_as_logical_shape(q_K, scales, block_len):
            # The GEMM for 2D blocks required padding in the scales.
            derived_scale_k_shape = math.ceil(q_K / block_len)
            _, scale_K = scales.shape
            if derived_scale_k_shape == scale_K:
                return scales
            return scales[:, :derived_scale_k_shape].contiguous()

        q_M, q_K = 1, 1
        if tensor._rowwise_data is not None:
            q = tensor._rowwise_data
            scale_inv = tensor._rowwise_scale_inv
            transpose_output = False
            if len(q.shape) >= 1:
                q_K = q.shape[-1]
            for i in range(len(q.shape) - 1):
                q_M *= q.shape[i]
        else:
            assert tensor._columnwise_data is not None, "No data to dequantize"
            q = tensor._columnwise_data
            scale_inv = tensor._columnwise_scale_inv
            transpose_output = True
            if len(q.shape) >= 1:
                q_M = q.shape[0]
            for i in range(1, len(q.shape)):
                q_K *= q.shape[i]

        orig_shape = q.shape
        q = q.reshape(q_M, q_K)
        formatted_scales = format_scale_as_logical_shape(q_K, scale_inv, block_len)
        assert len(formatted_scales.shape) == 2
        m_tiles, k_tiles = formatted_scales.shape
        unpadded_m, unpadded_k = q_M, q_K
        m_block_len = block_len
        k_block_len = block_len
        if q_M % m_block_len != 0 or q_K % k_block_len != 0:
            m_pad_amount = (m_block_len - (q_M % m_block_len)) % m_block_len
            k_pad_amount = (k_block_len - (q_K % k_block_len)) % k_block_len
            q = torch.nn.functional.pad(
                q, (0, k_pad_amount, 0, m_pad_amount), mode="constant", value=0
            ).contiguous()
        padded_M, padded_K = q.shape
        q_tiled = q.reshape(m_tiles, m_block_len, k_tiles, k_block_len)

        torch_q_dtype = TE_DType_To_Torch[tensor._fp8_dtype]

        result = q_tiled.view(torch_q_dtype).to(torch.float32) * formatted_scales.view(
            m_tiles, 1, k_tiles, 1
        )
        result = result.view(padded_M, padded_K).to(dtype)
        if padded_M != unpadded_m or padded_K != unpadded_k:
            result = result[:unpadded_m, :unpadded_k]
        if len(orig_shape) == 0:
            result = result.reshape([])
        else:
            result = result.reshape(*orig_shape).contiguous()
        if transpose_output:
            return tensor._transpose_dq_columnwise_output(result)
        return result

    @staticmethod
    def backward(
        _ctx: torch.autograd.function.FunctionCtx,  # unused
        grad: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        # pylint: disable=missing-function-docstring
        # Assume that we want gradients in full precision
        return grad, None


class Float8BlockwiseQTensorStorage(QuantizedTensorStorage):
    """Mixin class that holds data attributes of Float8BlockwiseQTensor.

    Float8BlockwiseQTensor inherits from the PyTorch tensor class and this
    mixin class. If this class is instantiated directly, it has the same
    data, lower CPU overhead, and less functionality. It should only
    be instantiated directly for performance-critical internal usage.
    """

    _rowwise_data: Optional[torch.Tensor]
    _columnwise_data: Optional[torch.Tensor]
    _quantizer: Quantizer
    _fp8_dtype: DType
    _rowwise_scale_inv: Optional[torch.Tensor]
    _columnwise_scale_inv: Optional[torch.Tensor]
    _is_2D_scaled: bool
    # ROCm-only: padded per-segment (MoE group) offsets for the variable-K grouped
    # wgrad. Set only when the columnwise operand is segment-padded (columnwise data
    # is [M_pad, N] rather than the standard transposed [N, M]); otherwise None.
    _vk_group_offs: Optional[torch.Tensor]

    def __new__(
        cls,
        rowwise_data: Optional[torch.Tensor],
        rowwise_scale_inv: Optional[torch.Tensor],
        columnwise_data: Optional[torch.Tensor],
        columnwise_scale_inv: Optional[torch.Tensor],
        fp8_dtype: Union[DType, tex.DType],
        quantizer: Quantizer,
        is_2D_scaled: bool,
        *args,
        fake_dtype: Optional[torch.dtype] = None,
        vk_group_offs: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if cls is Float8BlockwiseQTensorStorage:
            instance = object.__new__(cls)
            instance._dtype = fake_dtype if fake_dtype is not None else torch.float32
        else:
            instance = super().__new__(cls, *args, fake_dtype=fake_dtype, **kwargs)
        instance._rowwise_data = rowwise_data
        instance._columnwise_data = columnwise_data
        instance._quantizer = quantizer.copy() if quantizer is not None else None
        instance._fp8_dtype = DType.cast(fp8_dtype)
        instance._rowwise_scale_inv = rowwise_scale_inv
        instance._columnwise_scale_inv = columnwise_scale_inv
        instance._is_2D_scaled = is_2D_scaled
        instance._vk_group_offs = vk_group_offs

        return instance

    def clear(self):
        """Deallocate this tensor's memory. Typically not needed and must be used carefully."""
        for t in (
            self._rowwise_data,
            self._columnwise_data,
            self._rowwise_scale_inv,
            self._columnwise_scale_inv,
        ):
            if t is not None:
                t.data = _empty_tensor()

    def copy_from_storage(self, src: QuantizedTensorStorage) -> None:
        """Copy data buffers from another Float8BlockwiseQTensorStorage."""
        if not isinstance(src, Float8BlockwiseQTensorStorage):
            raise TypeError("copy_from_storage expects Float8BlockwiseQTensorStorage")
        if self._fp8_dtype != src._fp8_dtype:
            raise RuntimeError("FP8 dtype mismatch in copy_from_storage")
        if self._is_2D_scaled != src._is_2D_scaled:
            raise RuntimeError("Scale layout mismatch in copy_from_storage")

        def _copy_optional(dst: Optional[torch.Tensor], src_tensor: Optional[torch.Tensor]):
            if dst is not None and src_tensor is not None:
                dst.copy_(src_tensor)

        _copy_optional(self._rowwise_data, src._rowwise_data)
        _copy_optional(self._columnwise_data, src._columnwise_data)
        _copy_optional(self._rowwise_scale_inv, src._rowwise_scale_inv)
        _copy_optional(self._columnwise_scale_inv, src._columnwise_scale_inv)

    def get_metadata(self) -> Dict[str, Any]:
        """Get this tensor's metadata."""
        return {
            "rowwise_data": self._rowwise_data,
            "rowwise_scale_inv": self._rowwise_scale_inv,
            "columnwise_data": self._columnwise_data,
            "columnwise_scale_inv": self._columnwise_scale_inv,
            "fp8_dtype": self._fp8_dtype,
            "quantizer": self._quantizer,
            "is_2D_scaled": self._is_2D_scaled,
            "fake_dtype": self._dtype,
        }

    def prepare_for_saving(
        self,
    ) -> Tuple[list[Optional[torch.Tensor]], Float8BlockwiseQTensorStorage]:
        """
        Prepare the tensor base for saving for backward
        """
        tensors = [
            self._rowwise_data,
            self._columnwise_data,
            self._rowwise_scale_inv,
            self._columnwise_scale_inv,
        ]
        self._rowwise_data = None
        self._columnwise_data = None
        self._rowwise_scale_inv = None
        self._columnwise_scale_inv = None
        return tensors, self

    def restore_from_saved(
        self, tensors: list[Optional[torch.Tensor]]
    ) -> list[Optional[torch.Tensor]]:
        """Restore the tensor base data from the saved tensors list."""
        self._rowwise_data = tensors[0]
        self._columnwise_data = tensors[1]
        self._rowwise_scale_inv = tensors[2]
        self._columnwise_scale_inv = tensors[3]
        return tensors[4:]

    def get_data_tensors(self, rowwise_data: bool = True, columnwise_data: bool = True):
        """Get this Tensor's data."""
        if rowwise_data and columnwise_data:
            return self._rowwise_data, self._columnwise_data
        if rowwise_data:
            return self._rowwise_data
        if columnwise_data:
            return self._columnwise_data
        raise ValueError("No data to get, both rowwise_data and columnwise_data are False")

    def _transpose_dq_columnwise_output(self, columnwise_dq: torch.Tensor) -> torch.Tensor:
        """Takes dequantized columnwise data and permutes to a rowwise shape"""
        if columnwise_dq.dim() < 2:
            return columnwise_dq
        permute_dims = list(range(1, columnwise_dq.dim()))
        permute_dims.append(0)
        return torch.permute(columnwise_dq, tuple(permute_dims)).contiguous()

    def _dequantize_vectorwise(self, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if dtype is None:
            dtype = self._dtype
        block_len = 128

        q_M, q_K = 1, 1
        if self._rowwise_data is not None:
            q = self._rowwise_data
            scale_inv = self._rowwise_scale_inv
            transpose_output = False
            if len(q.shape) >= 1:
                q_K = q.shape[-1]
            for i in range(len(q.shape) - 1):
                q_M *= q.shape[i]
            inner_q_dimension_tiled = True
            scales_tiled_dim, scales_untiled_dim = scale_inv.shape
        else:
            assert self._columnwise_data is not None, "No data to dequantize"
            q = self._columnwise_data
            scale_inv = self._columnwise_scale_inv
            scales_tiled_dim, scales_untiled_dim = scale_inv.shape
            inner_q_dimension_tiled = True
            transpose_output = True
            if len(q.shape) >= 1:
                q_M = q.shape[0]
            for i in range(1, len(q.shape)):
                q_K *= q.shape[i]

        orig_shape = q.shape
        q = q.reshape(q_M, q_K)
        if inner_q_dimension_tiled:
            if q_K % block_len != 0:
                k_pad_amount = (block_len - (q_K % block_len)) % block_len
                q = torch.nn.functional.pad(
                    q, (0, k_pad_amount, 0, 0), mode="constant", value=0
                ).contiguous()
            padded_M, padded_K = q.shape
            q_tiled = q.reshape(q_M, scales_tiled_dim, block_len)
        else:
            if q_M % block_len != 0:
                m_pad_amount = (block_len - (q_M % block_len)) % block_len
                q = torch.nn.functional.pad(
                    q, (0, 0, 0, m_pad_amount), mode="constant", value=0
                ).contiguous()
            padded_M, padded_K = q.shape
            q_tiled = q.reshape(scales_tiled_dim, block_len, q_K)
        if scales_untiled_dim > q_M:
            # untiled scale dimension is 4 element aligned.
            scale_inv = scale_inv[:, :q_M].contiguous()
        dq_scale = scale_inv.transpose(-2, -1).contiguous().reshape(q_M, scales_tiled_dim, 1)
        torch_q_dtype = TE_DType_To_Torch[self._fp8_dtype]
        result = q_tiled.view(torch_q_dtype).to(torch.float32) * dq_scale
        if padded_M != q_M or padded_K != q_K:
            result = result.reshape(padded_M, padded_K)[:q_M, :q_K]
        result = result.to(dtype)
        if len(orig_shape) == 0:
            result = result.reshape([])
        else:
            result = result.reshape(*orig_shape).contiguous()

        if transpose_output:
            return self._transpose_dq_columnwise_output(result)
        return result

    def dequantize(self, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """
        Construct plain PyTorch tensor from Float8BlockwiseQTensor
        """
        return _FromFloat8BlockwiseFunc.forward(None, self, dtype)

    def size(self, *args, **kwargs):
        # pylint: disable=missing-function-docstring
        if self._rowwise_data is not None:
            return self._rowwise_data.size(*args, **kwargs)
        dims = list(self._columnwise_data.size(*args, **kwargs))
        reordered = []
        for i in range(1, len(dims)):
            reordered.append(dims[i])
        reordered.append(dims[0])
        return torch.Size(reordered)

    @property
    def device(self):
        """Return the device of the tensor. Define this to avoid expensive PyObject lookups."""
        if self._rowwise_data is not None:
            return self._rowwise_data.device
        if self._columnwise_data is not None:
            return self._columnwise_data.device
        raise RuntimeError("Float8BlockwiseQTensorStorage has no data!")

    def _allocate_columnwise_buffers(self) -> None:
        """Allocate transposed columnwise payload + 1D scale_inv if missing."""
        from ..float8_blockwise_tensor import Float8BlockQuantizer

        if self._quantizer is not None:
            quantizer = self._quantizer.copy()
            quantizer.set_usage(rowwise=False, columnwise=True)
        else:
            quantizer = Float8BlockQuantizer(
                self._fp8_dtype, rowwise=False, columnwise=True, block_scaling_dim=1
            )

        empty = quantizer.make_empty(
            tuple(self.size()), dtype=self._dtype, device=self.device
        )
        if self._columnwise_data is None:
            self._columnwise_data = empty._columnwise_data
        if self._columnwise_scale_inv is None:
            self._columnwise_scale_inv = empty._columnwise_scale_inv

    def _create_columnwise_1d_from_high_precision(self) -> None:
        """Fallback: dequant to HBM then columnwise-only 1x128 quantize."""
        from ..float8_blockwise_tensor import Float8BlockQuantizer, Float8BlockwiseQTensor

        hp_dtype = (
            self._dtype if getattr(self._dtype, "is_floating_point", False) else torch.bfloat16
        )
        hp = self.dequantize(dtype=hp_dtype)
        if self._quantizer is not None:
            quantizer = self._quantizer.copy()
            quantizer.set_usage(rowwise=False, columnwise=True)
        else:
            quantizer = Float8BlockQuantizer(
                self._fp8_dtype,
                rowwise=False,
                columnwise=True,
                block_scaling_dim=1,
                force_pow_2_scales=True,
            )
        dst = Float8BlockwiseQTensor(
            shape=tuple(self.size()),
            dtype=hp_dtype,
            rowwise_data=None,
            rowwise_scale_inv=None,
            columnwise_data=self._columnwise_data,
            columnwise_scale_inv=self._columnwise_scale_inv,
            fp8_dtype=self._fp8_dtype,
            quantizer=quantizer,
            is_2D_scaled=False,
        )
        quantizer.update_quantized(hp, dst)
        self._columnwise_data = dst._columnwise_data
        self._columnwise_scale_inv = dst._columnwise_scale_inv

    def _create_columnwise(self):
        """Create columnwise data from rowwise.

        2D 128x128: byte transpose + scale transpose.
        1D 1x128: fused dequant + columnwise requant (pow-2 scales).
        """
        if not self._is_2D_scaled:
            if (
                self._columnwise_data is not None
                and self._columnwise_scale_inv is not None
            ):
                return
            if self._rowwise_data is None or self._rowwise_scale_inv is None:
                raise RuntimeError(
                    "Cannot create 1D columnwise FP8 without rowwise payload and scales"
                )
            self._allocate_columnwise_buffers()
            use_triton = int(os.environ.get("NVTE_FP8_BLOCKWISE_1D_ROWWISE_TO_COLWISE", "1"))
            if use_triton:
                from ...triton_kernels.cast_transpose import (
                    te_fp8_blockwise_1d_rowwise_to_columnwise_triton,
                )

                te_fp8_blockwise_1d_rowwise_to_columnwise_triton(self)
            else:
                self._create_columnwise_1d_from_high_precision()
            return

        rowwise_data = self._rowwise_data
        if not rowwise_data.is_contiguous():
            rowwise_data = rowwise_data.contiguous()
        self._columnwise_data = tex.fp8_transpose(
            rowwise_data, self._fp8_dtype, out=self._columnwise_data
        )

        if self._columnwise_scale_inv is None:
            assert self._quantizer is not None, (
                "._quantizer of Float8BlockwiseQTensor cannot be None because all the blockwise "
                "quantized tensors are supposed to be generated from the quantizer."
            )
            columnwise_scale_inv_shape = self._quantizer.get_scale_shape(rowwise_data.shape, True)
            self._columnwise_scale_inv = torch.empty(
                columnwise_scale_inv_shape,
                dtype=self._rowwise_scale_inv.dtype,
                device=self._rowwise_scale_inv.device,
            )
        assert len(self._rowwise_scale_inv.shape) == 2
        assert len(self._columnwise_scale_inv.shape) == 2
        rowwise_scale_inv = self._rowwise_scale_inv
        columnwise_scale_inv = rowwise_scale_inv.transpose(-2, -1)
        h = min(self._columnwise_scale_inv.shape[0], columnwise_scale_inv.shape[0])
        w = min(self._columnwise_scale_inv.shape[1], columnwise_scale_inv.shape[1])
        self._columnwise_scale_inv[0:h, 0:w].copy_(columnwise_scale_inv[0:h, 0:w])

    def _transpose_columnwise_data(self):
        """Plainly transpose the columnwise data and scale inv."""
        if self._columnwise_data is not None:
            # TODO(yuzhongw, tmoon): Figure out why _old_data is not automatically
            # deallocated by GC. Manually deallocating is a temporary hack.
            _old_data = self._columnwise_data
            self._columnwise_data = tex.fp8_transpose(
                self._columnwise_data, self._fp8_dtype, out=None
            )
            _old_data.data = _empty_tensor()
            del _old_data

    def __repr__(self):
        try:
            if self._rowwise_data is not None:
                data = self.dequantize()
                descriptor = "rowwise"
            else:
                data = self.dequantize()
                descriptor = "columnwise"
            return (
                "Float8BlockwiseQTensorStorage("
                f"fp8_dtype={self._fp8_dtype}, "
                f"{descriptor}_scaled_data={data})"
            )
        except Exception as exc:  # pylint: disable=broad-except
            return safe_quantized_repr(
                self,
                "Float8BlockwiseQTensorStorage",
                extras={"is_2D_scaled": self._is_2D_scaled},
                error=exc,
            )

    def update_usage(
        self, rowwise_usage: Optional[bool] = None, columnwise_usage: Optional[bool] = None
    ):
        """Keep, drop, or synthesize blockwise layouts.

        2D: columnwise is a transpose of rowwise. 1D (1x128): columnwise is
        built by fused dequant + requant when pow-2 scales are used.
        """

        if rowwise_usage is None:
            rowwise_usage = self._rowwise_data is not None
        if columnwise_usage is None:
            columnwise_usage = self._columnwise_data is not None
        assert (
            columnwise_usage or rowwise_usage
        ), "Must retain some data either columnwise or rowwise"

        if columnwise_usage:
            if self._columnwise_data is None or self._columnwise_scale_inv is None:
                assert (
                    self._rowwise_data is not None and self._rowwise_scale_inv is not None
                ), "Cannot create columnwise usage because rowwise data is None."
                self._create_columnwise()
            if self._columnwise_data is None or self._columnwise_scale_inv is None:
                raise RuntimeError("Requested column-wise usage but columnwise buffers are missing")

        if rowwise_usage:
            assert (
                self._rowwise_data is not None and self._rowwise_scale_inv is not None
            ), "Cannot update to rowwise usage."
        else:
            self._rowwise_data = None
            self._rowwise_scale_inv = None

        if not columnwise_usage:
            self._columnwise_data = None
            self._columnwise_scale_inv = None

    def get_usages(self) -> Dict[str, bool]:
        """Get the usage of the tensor"""
        return {
            "rowwise": self._rowwise_data is not None,
            "columnwise": self._columnwise_data is not None,
        }
