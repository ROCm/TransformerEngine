# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

"""Reference MXFP4-family recipes.

Extensible ``CustomRecipe`` quantizer factories built on the existing
``MXFP4QuantizerRef``.

  * ``mxfp4_grouped_quantizer_factory`` -- a4w4: E2M1 weights and activations.
  * ``a8w4_quantizer_factory``          -- Kimi-K3 a8w4: MXFP8 E4M3 activation x
                                           MXFP4 E2M1 weight.
"""

from __future__ import annotations

from typing import Optional

import torch

from transformer_engine.pytorch.custom_recipes import quantization
from transformer_engine.pytorch.quantization import QuantizerRole
from transformer_engine.pytorch.quantized_tensor import QuantizedTensorStorage, Quantizer

from .quantization_mxfp4 import (
    MXFP4_BLOCK_SIZE,
    MXFP4QuantizerRef,
    MXFP4TensorRef,
    e8m0_to_f32,
    mxfp4_to_f32,
)


def _deq_ref_operand(data, scale, storage, *, default_e4m3: bool) -> torch.Tensor:
    """Dequantize one reference operand to f32 by its own stored format."""
    quantizer = getattr(storage, "_quantizer", None)
    if quantizer is not None:
        is_e4m3 = isinstance(quantizer, MXFP8E4M3QuantizerRef)
    else:
        is_e4m3 = default_e4m3
    hp = data.float() if is_e4m3 else mxfp4_to_f32(data)
    rows, k = hp.shape
    grid_k = k // MXFP4_BLOCK_SIZE
    s = e8m0_to_f32(scale[:rows, :grid_k]).repeat_interleave(MXFP4_BLOCK_SIZE, dim=1)
    return hp * s


class MXFP8E4M3QuantizerRef(Quantizer):
    """Reference MXFP8 (E4M3) block-scaled quantizer for the a8w4 activation."""

    def __init__(self, rowwise: bool = True, columnwise: bool = True):
        super().__init__(rowwise=rowwise, columnwise=columnwise)
        self.internal = True

    @property
    def custom(self) -> bool:
        return True

    def _quantize(self, tensor: torch.Tensor):
        """``tensor`` [M, K] -> (E4M3 data [M, K], E8M0 scale u8 [M, K/32])."""
        fmax = torch.finfo(torch.float8_e4m3fn).max  # 448
        m, k = tensor.shape
        assert k % MXFP4_BLOCK_SIZE == 0, "K must be a multiple of the 32-elem block"
        xb = tensor.float().reshape(m, k // MXFP4_BLOCK_SIZE, MXFP4_BLOCK_SIZE)
        amax = xb.abs().amax(-1, keepdim=True).clamp_min(1e-30)
        exp = torch.ceil(torch.log2(amax / fmax)).clamp(-127, 127)
        q = (xb / torch.exp2(exp)).clamp(-fmax, fmax).to(torch.float8_e4m3fn)
        data = q.reshape(m, k).contiguous()
        scale = (exp.squeeze(-1) + 127).to(torch.uint8)  # E8M0, biased by 127
        return data, scale

    def quantize(self, tensor: torch.Tensor, **kwargs) -> MXFP4TensorRef:  # noqa: ARG002
        """Quantize a high-precision tensor, returning an MXFP4TensorRef holder."""
        original_shape = tensor.shape
        if tensor.ndim > 2:
            tensor = tensor.view(-1, tensor.shape[-1])
        qx, sx = self._quantize(tensor) if self.rowwise_usage else (None, None)
        if self.columnwise_usage:
            qx_t, sx_t = self._quantize(tensor.t().contiguous())
        else:
            qx_t = sx_t = None
        return MXFP4TensorRef(
            data=qx,
            scale=sx,
            data_t=qx_t,
            scale_t=sx_t,
            dtype=tensor.dtype,
            device=tensor.device,
            original_shape=original_shape,
            _quantizer=self,
        )

    # ------------------------------------------------------------------
    # Reference GEMM
    # ------------------------------------------------------------------
    def qgemm(
        self,
        qx: torch.Tensor,
        qw: torch.Tensor,
        m_params: quantization.MMParams,
        out_dtype: torch.dtype,
        sx: torch.Tensor,
        sw: torch.Tensor,
        bias: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
        accumulate: bool = False,
        gemm_type: quantization.GEMMType = quantization.GEMMType.FPROP,
        qresult_x: QuantizedTensorStorage | None = None,
        qresult_w: QuantizedTensorStorage | None = None,
        layout: str = "TN",
    ) -> torch.Tensor:
        """``Y = dequant(qx) @ dequant(qw)^T``; each operand by its own format."""
        assert bias is None, "Bias not yet supported in the a8w4 reference GEMM."
        hp_x = _deq_ref_operand(qx, sx, qresult_x, default_e4m3=True)  # A: E4M3 here
        hp_w = _deq_ref_operand(qw, sw, qresult_w, default_e4m3=False)  # B: E2M1 weight
        assert hp_x.shape[1] == hp_w.shape[1], "K mismatch between operands"
        y = torch.mm(hp_x, hp_w.t()).to(out_dtype)
        if accumulate:
            assert out is not None
            y = y + out.to(out_dtype)
        return y

    @staticmethod
    def _dequantize(data_e4m3, scale_u8, dtype=torch.float32) -> torch.Tensor:
        m, k = data_e4m3.shape
        vals = data_e4m3.float().reshape(m, k // MXFP4_BLOCK_SIZE, MXFP4_BLOCK_SIZE)
        scale = e8m0_to_f32(scale_u8[:m, : k // MXFP4_BLOCK_SIZE]).unsqueeze(-1)
        return (vals * scale).reshape(m, k).to(dtype)

    def dequantize_rowwise(self, ref: MXFP4TensorRef, dtype=torch.float32) -> torch.Tensor:
        assert ref.data is not None and ref.scale is not None
        return self._dequantize(ref.data, ref.scale, dtype=dtype)

    def dequantize_columnwise(self, ref: MXFP4TensorRef, dtype=torch.float32) -> torch.Tensor:
        assert ref.data_t is not None and ref.scale_t is not None
        return self._dequantize(ref.data_t, ref.scale_t, dtype=dtype).t().contiguous()

    @property
    def supports_allgather_fp8(self) -> bool:
        return False

    @property
    def supports_dequantize(self) -> bool:
        return True

    def transpose_qresult(self, qresult):  # noqa: ARG002
        raise NotImplementedError

    @property
    def is_data_t_transposed_in_memory(self) -> bool:
        raise NotImplementedError


def _e2m1_ref(*, rowwise: bool, columnwise: bool) -> MXFP4QuantizerRef:
    """Plain (un-swizzled, no-Hadamard) E2M1 reference quantizer."""
    return MXFP4QuantizerRef(
        rowwise=rowwise,
        columnwise=columnwise,
        shuffle_rowwise_data=False,
        shuffle_columnwise_data=False,
        with_gemm_swizzled_scales=False,
        use_hadamard=False,
    )


def _make_mxfp4_family_factory(*, activation: str):
    """Build a role-based MXFP4-family ``qfactory``.

    ``activation`` selects the non-weight format: ``"e2m1"`` -> a4w4 (E2M1
    everywhere); ``"e4m3"`` -> a8w4 (E4M3 activation and grad, E2M1 weight).
    Weights are always E2M1 and need both layouts (fprop row + dgrad col).
    """

    def factory(role: Optional[QuantizerRole]) -> Quantizer:
        is_linear = role is not None and role.module_type in ("linear", "grouped_linear")
        tensor_type = role.tensor_type if role is not None else ""
        if is_linear and tensor_type == "weight":
            return _e2m1_ref(rowwise=True, columnwise=True)
        if activation == "e2m1":
            return _e2m1_ref(rowwise=True, columnwise=True)
        # a8w4: E4M3 for activation and grad_output (uniform E4M3 keeps every
        # backward GEMM E4M3xE2M1 / E4M3xE4M3, all handled by the ref qgemm).
        return MXFP8E4M3QuantizerRef(rowwise=True, columnwise=True)

    return factory


# a4w4: pure E2M1
mxfp4_grouped_quantizer_factory = _make_mxfp4_family_factory(activation="e2m1")

# Kimi-K3 a8w4: E4M3 activation x E2M1 weight
a8w4_quantizer_factory = _make_mxfp4_family_factory(activation="e4m3")
