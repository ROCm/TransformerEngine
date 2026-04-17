# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""MXFP4 recipe reference implementation.

Pure-Python/NumPy reference for MXFP4 quantization with 32-element blocks
and E8M0 (power-of-two) scales.  Used by tests to validate the native
MXFP4Quantizer kernels (Triton and C++).
"""

import dataclasses
from typing import Optional, Tuple

import numpy as np
import torch

import transformer_engine_torch as tex

from transformer_engine.pytorch.custom_recipes import quantization
from transformer_engine.pytorch.tensor.mxfp4_tensor import MXFP4Quantizer
from transformer_engine.pytorch.quantized_tensor import QuantizedTensorStorage, Quantizer


MXFP4_BLOCK_SIZE = 32

# E2M1: nibble 0..15 -> float
_E2M1_TABLE_NP = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)

_E2M1_TABLE_TORCH = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def unpack_fp4x2(packed: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """Unpack packed FP4 uint8 ``(M, N/2)`` into float ``(M, N)``."""
    table = _E2M1_TABLE_TORCH.to(device=packed.device, dtype=out_dtype)
    lo = (packed & 0x0F).long()
    hi = (packed >> 4).long()
    out = torch.empty(
        packed.shape[0], packed.shape[1] * 2, device=packed.device, dtype=out_dtype,
    )
    out[:, 0::2] = table[lo]
    out[:, 1::2] = table[hi]
    return out


def e8m0_to_float(scales_u8: torch.Tensor) -> torch.Tensor:
    """Convert uint8 E8M0 exponents to float32 decode scales: ``2**(e - 127)``."""
    return torch.pow(2.0, scales_u8.to(torch.float32) - 127.0)


def _round_up(x: int, multiple: int) -> int:
    """Round *x* up to the nearest multiple of *multiple*."""
    return ((x + multiple - 1) // multiple) * multiple


def _shuffle_scales(scales: torch.Tensor) -> torch.Tensor:
    """Shuffle E8M0 scales into the layout expected by AITER GEMM kernels."""
    sm, sn = scales.shape
    scales = scales.view(sm // 32, 2, 16, sn // 8, 2, 4, 1)
    scales = scales.permute(0, 3, 5, 2, 4, 1, 6).contiguous()
    return scales.view(sm // 32, sn * 32)


def _shuffle_fp4_data(x: torch.Tensor, layout=(16, 16)) -> torch.Tensor:
    """Shuffle packed FP4 data into the layout expected by AITER GEMM kernels.

    Mirrors ``aiter.ops.shuffle.shuffle_weight`` for uint8 packed FP4 tensors.
    """
    IN, IK = layout
    BN = IN
    BK = IK * 2
    K = 16 // x.element_size()
    assert x.shape[-2] % BN == 0, f"{x.shape[-2]} % {BN} != 0"
    assert x.shape[-1] % BK == 0, f"{x.shape[-1]} % {BK} != 0"
    x_ = x.view(-1, x.shape[-2] // BN, BN, x.shape[-1] // BK, BK // K, K)
    x_ = x_.permute(0, 1, 3, 4, 2, 5).contiguous()
    return x_.view(*x.shape)

def mxfp4_to_f32(x):
    if x.dtype == torch.float4_e2m1fn_x2:
        x = x.view(torch.uint8)

    # 2 because we pack fp4 in uint8.
    x = x.repeat_interleave(2, dim=-1)
    x[..., ::2] = x[..., ::2] & 0xF
    x[..., 1::2] = x[..., 1::2] >> 4
    mxfp4_list = [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ]
    mxfp4_in_f32 = torch.tensor(mxfp4_list, dtype=torch.float32, device=x.device)
    return mxfp4_in_f32[x.long()]

def e8m0_to_f32(scale_e8m0_biased):
    scale_e8m0_biased = scale_e8m0_biased.view(torch.uint8)
    zero_case = scale_e8m0_biased == 0
    nan_case = scale_e8m0_biased == 0xFF
    scale_f32 = scale_e8m0_biased.to(torch.int32) << 23
    scale_f32[zero_case] = 0x00400000
    scale_f32[nan_case] = 0x7F800001
    scale_f32 = scale_f32.view(torch.float32)
    return scale_f32

# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class MXFP4TensorRef(QuantizedTensorStorage):
    """Container for MXFP4 quantization results.

    Attributes
    ----------
    data : torch.Tensor or None
        Packed FP4 rowwise data, shape ``(M, K/2)`` uint8.
    scale : torch.Tensor or None
        E8M0 rowwise scales, shape ``(M, K/32)`` uint8.
    data_t : torch.Tensor or None
        Packed FP4 columnwise data, shape ``(K, M/2)`` uint8.
    scale_t : torch.Tensor or None
        E8M0 columnwise scales, shape ``(K, M/32)`` uint8.
    """

    data: Optional[torch.Tensor] = None
    scale: Optional[torch.Tensor] = None
    data_t: Optional[torch.Tensor] = None
    scale_t: Optional[torch.Tensor] = None

    dtype: Optional[torch.dtype] = None
    device: Optional[torch.device] = None
    original_shape: Optional[Tuple[int, ...]] = None
    _quantizer: Optional[Quantizer] = None

    @property
    def custom(self) -> bool:
        return True

    def prepare_for_saving(self):
        tensors = [self.data, self.data_t, self.scale, self.scale_t]
        self.data = self.data_t = self.scale = self.scale_t = None
        return tensors, self

    def restore_from_saved(self, tensors):
        self.data, self.data_t, self.scale, self.scale_t = tensors[:4]
        return tensors[4:]

    @property
    def _data(self):
        return self.data

    @_data.setter
    def _data(self, value):
        self.data = value

    @property
    def _scale_inv(self):
        return self.scale

    @_scale_inv.setter
    def _scale_inv(self, value):
        self.scale = value

    def update_usage(self, rowwise_usage=None, columnwise_usage=None):
        pass
    
    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"dtype={self.dtype}, "
            f"device={self.device}, "
            f"original_shape={self.original_shape}"
            ")"
        )

    def size(self, *args, **kwargs):
        assert self.original_shape is not None
        return torch.Size(self.original_shape)


# ---------------------------------------------------------------------------
# Reference quantizer
# ---------------------------------------------------------------------------

class MXFP4QuantizerRef(Quantizer):
    """Pure-Python reference implementation of MXFP4 quantization.

    Uses 32-element blocks with E8M0 (power-of-two) scales matching the
    HIP / Triton kernel behaviour:

    * E8M0 scale: ``floor(log2(amax_rounded)) - 2 + 127``
    * FP4 encoding: threshold-based nearest-neighbour to E2M1 values
    * Packing: ``(odd_nibble << 4) | even_nibble``
    """

    def __init__(
        self,
        rowwise: bool = True,
        columnwise: bool = True,
        shuffle_B_matrix_for_aiter: bool = False,
        shuffle_scales: bool = False,
        use_hadamard: bool = False,
        use_te_quantizer: bool = False,
    ):
        super().__init__(rowwise=rowwise, columnwise=columnwise)
        self.internal = True
        self.shuffle_B_matrix_for_aiter = shuffle_B_matrix_for_aiter
        self.shuffle_scales = shuffle_scales
        self.use_hadamard = use_hadamard
        self.use_te_quantizer = use_te_quantizer
    @property
    def custom(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Hadamard helpers
    # ------------------------------------------------------------------

    # The kernel applies a 16-point Hadamard to each half of a 32-element
    # block (elements 0-15 and 16-31 independently).
    HADAMARD_DIM = 16

    @staticmethod
    def _build_hadamard_matrix(size: int) -> np.ndarray:
        """Sylvester-constructed Hadamard matrix (+-1 entries)."""
        assert (size & (size - 1)) == 0, "Hadamard size must be a power of two"
        h = np.array([[1.0]], dtype=np.float32)
        while h.shape[0] < size:
            h = np.block([[h, h], [h, -h]])
        return h

    def _apply_hadamard(self, data: np.ndarray) -> np.ndarray:
        """Apply fixed 16-point Hadamard to each 32-element block.

        Each block of 32 contiguous values is split into two halves of 16.
        Each half is independently transformed by H16 and scaled by
        ``1/sqrt(16)``.  This matches the kernel's ``hadamard16_inplace``.
        """
        if not self.use_hadamard:
            return data

        dim = self.HADAMARD_DIM
        H = self._build_hadamard_matrix(dim)
        scale = 1.0 / np.sqrt(dim)
        transform = H * scale  # (16, 16)

        M, N = data.shape
        # (M, num_blocks, 2, 16) — two independent 16-element halves per block
        reshaped = data.reshape(M, N // MXFP4_BLOCK_SIZE, 2, dim)
        transformed = np.einsum("...j,jk->...k", reshaped, transform)
        return transformed.reshape(M, N)

    # ------------------------------------------------------------------
    # Core quantization (operates on a 2-D float tensor)
    # ------------------------------------------------------------------

    def _quantize(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize a 2-D tensor to packed FP4 + E8M0 scales.

        Parameters
        ----------
        tensor : torch.Tensor
            Shape ``(M, N)`` with ``N`` divisible by 32.

        Returns
        -------
        fp4_packed : torch.Tensor
            uint8 ``(M, N/2)``
        scales : torch.Tensor
            uint8 ``(M, N/32)``
        """
        M, N = tensor.shape
        assert N % MXFP4_BLOCK_SIZE == 0

        data = tensor.cpu().float().numpy()

        # Optional Hadamard transform (applied before scale computation)
        data = self._apply_hadamard(data)

        num_blocks = N // MXFP4_BLOCK_SIZE
        data_blocks = data.reshape(M, num_blocks, MXFP4_BLOCK_SIZE)
        amax_blocks = np.max(np.abs(data_blocks), axis=2)

        # --- E8M0 scale computation (matches HIP compute_e8m0_scale) ---
        amax_int = amax_blocks.astype(np.float32).view(np.uint32)
        amax_int = ((amax_int + 0x200000) & 0xFF800000).astype(np.uint32)
        amax_rounded = amax_int.view(np.float32)

        with np.errstate(divide="ignore", invalid="ignore"):
            scale_unbiased = np.floor(np.log2(np.maximum(amax_rounded, 1e-45))) - 2
        scale_unbiased = np.clip(scale_unbiased, -127, 127)
        scales = (scale_unbiased + 127).astype(np.uint8)
        scales = np.where(amax_blocks == 0, 127, scales)

        scale_vals = np.where(
            amax_blocks[:, :, None] > 0,
            2.0 ** (-(scales[:, :, None].astype(np.float32) - 127)),
            1.0,
        )
        scaled_blocks = data_blocks * scale_vals

        # --- FP4 encoding (threshold-based) ---
        signs = (scaled_blocks < 0).astype(np.uint8)
        abs_vals = np.abs(scaled_blocks)
        indices = np.zeros_like(abs_vals, dtype=np.uint8)
        for threshold, code in [(0.25, 1), (0.75, 2), (1.25, 3),
                                (1.75, 4), (2.5, 5), (3.5, 6), (5.0, 7)]:
            indices = np.where(abs_vals >= threshold, code, indices)
        fp4_flat = ((signs << 3) | indices).reshape(M, N)

        # --- Pack two nibbles per byte ---
        fp4_packed = ((fp4_flat[:, 1::2] << 4) | fp4_flat[:, 0::2]).astype(np.uint8)

        fp4_packed_torch = torch.from_numpy(fp4_packed).to(tensor.device)

        if self.shuffle_B_matrix_for_aiter:
            fp4_packed_torch = _shuffle_fp4_data(fp4_packed_torch)

        scales_valid = torch.from_numpy(scales).to(tensor.device)

        # Pad scales to match native allocator layout (multiples of 256 x 8)
        num_scale_rows = M
        num_scale_cols = N // MXFP4_BLOCK_SIZE
        padded_rows = _round_up(num_scale_rows, 256)
        padded_cols = _round_up(num_scale_cols, 8)
        scales_torch = torch.zeros(
            padded_rows, padded_cols, dtype=torch.uint8, device=tensor.device,
        )
        scales_torch[:num_scale_rows, :num_scale_cols] = scales_valid

        if self.shuffle_scales:
            scales_torch = _shuffle_scales(scales_torch)
            scales_torch = scales_torch.view(scales_torch.shape[0] * 32, -1)

        return fp4_packed_torch, scales_torch

    @staticmethod
    def _dequantize(
        fp4_packed: torch.Tensor,
        scales: torch.Tensor,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Dequantize packed FP4 + E8M0 scales back to float.

        Parameters
        ----------
        fp4_packed : torch.Tensor  uint8 (M, N/2)
        scales     : torch.Tensor  uint8 — may be padded; only ``[:M, :N//32]`` is used.
        """
        packed = fp4_packed.cpu().numpy().astype(np.uint8)

        M, halfN = packed.shape
        N = halfN * 2
        num_blocks = N // MXFP4_BLOCK_SIZE

        scales_np = scales[:M, :num_blocks].cpu().numpy().astype(np.uint8)

        fp4_even = packed & 0x0F
        fp4_odd = (packed >> 4) & 0x0F
        fp4_flat = np.empty((M, N), dtype=np.uint8)
        fp4_flat[:, 0::2] = fp4_even
        fp4_flat[:, 1::2] = fp4_odd

        decoded = _E2M1_TABLE_NP[fp4_flat]
        scale_vals = 2.0 ** (scales_np[:, :, None].astype(np.float32) - 127)
        out = (decoded.reshape(M, num_blocks, MXFP4_BLOCK_SIZE) * scale_vals).reshape(M, N)

        return torch.from_numpy(out.astype(np.float32)).to(
            device=fp4_packed.device, dtype=dtype,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def quantize(
        self, 
        tensor: torch.Tensor,
        **kwargs,  # pylint: disable=unused-argument
    ) -> MXFP4TensorRef:
        """Quantize a high-precision tensor, returning an MXFP4TensorRef."""
        original_shape = tensor.shape
        if tensor.ndim > 2:
            tensor = tensor.view(-1, tensor.shape[-1])
        if self.use_te_quantizer:
            te_quantizer = MXFP4Quantizer(rowwise=self.rowwise_usage, 
                                        columnwise=self.columnwise_usage, 
                                        shuffle_B_matrix_for_aiter=self.shuffle_B_matrix_for_aiter, 
                                        shuffle_scales=self.shuffle_scales, 
                                        use_hadamard=self.use_hadamard)
            q_tensor = tex.quantize(tensor, te_quantizer)
            return MXFP4TensorRef(
                data=q_tensor._rowwise_data,
                scale=q_tensor._rowwise_scale_inv,
                data_t=q_tensor._columnwise_data,
                scale_t=q_tensor._columnwise_scale_inv,
                dtype=tensor.dtype,
                device=tensor.device,
                original_shape=original_shape,
                _quantizer=self,
            )

        if self.rowwise_usage:
            qx, sx = self._quantize(tensor)
        else:
            qx = sx = None

        if self.columnwise_usage:
            t_input = tensor.t().contiguous()
            qx_t, sx_t = self._quantize(t_input)
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

    def dequantize_rowwise(self, ref: MXFP4TensorRef, dtype=torch.float32) -> torch.Tensor:
        """Dequantize rowwise data back to high-precision tensor."""
        assert ref.data is not None and ref.scale is not None
        return self._dequantize(ref.data, ref.scale, dtype=dtype)

    def dequantize_columnwise(self, ref: MXFP4TensorRef, dtype=torch.float32) -> torch.Tensor:
        """Dequantize columnwise data and transpose back."""
        assert ref.data_t is not None and ref.scale_t is not None
        deq = self._dequantize(ref.data_t, ref.scale_t, dtype=dtype)
        return deq.t().contiguous()

    # ------------------------------------------------------------------
    # Reference GEMM
    # ------------------------------------------------------------------

    def qgemm(
        self,
        qx: torch.Tensor,
        qw: torch.Tensor,
        m_params: quantization.MMParams,  # pylint: disable=unused-argument
        out_dtype: torch.dtype,
        sx: torch.Tensor,
        sw: torch.Tensor,
        bias: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
        accumulate: bool = False,
        gemm_type: quantization.GEMMType = quantization.GEMMType.FPROP,  # pylint: disable=unused-argument
        qresult_x: QuantizedTensorStorage | None = None,  # pylint: disable=unused-argument
        qresult_w: QuantizedTensorStorage | None = None,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        """Reference MXFP4 GEMM: ``Y = dequant(qx, sx) @ dequant(qw, sw)^T``.

        Block-scaled inner products are accumulated in float32 to match the
        hardware tensor-core accumulation model, then cast to ``out_dtype``.

        Parameters
        ----------
        qx : torch.Tensor
            uint8 ``(M, K/2)`` packed FP4 activations.
        qw : torch.Tensor
            uint8 ``(N, K/2)`` packed FP4 weights.
        m_params : quantization.MMParams
            Matmul parameters (unused by the reference; kept for API parity
            with the NVFP4 reference and ``custom_gemm`` dispatcher).
        out_dtype : torch.dtype
            Desired output dtype.
        sx : torch.Tensor
            uint8 ``(M, K/32)`` E8M0 activation scales (may be padded).
        sw : torch.Tensor
            uint8 ``(N, K/32)`` E8M0 weight scales (may be padded).
        bias : torch.Tensor, optional
            Not yet supported.
        out : torch.Tensor, optional
            Pre-existing output tensor; required when ``accumulate=True``.
        accumulate : bool
            If True, add the GEMM result to ``out`` (in float32) before
            casting to ``out_dtype``.
        gemm_type : quantization.GEMMType
            Which GEMM this call represents (FPROP/DGRAD/WGRAD). Unused by
            the reference but accepted for API parity.
        qresult_x, qresult_w : QuantizedTensorStorage, optional
            Full quantized tensor storages. Unused by the reference but
            accepted for API parity with NVFP4.
        """
        assert bias is None, "Bias not yet supported in MXFP4 reference GEMM."

        hp_x = mxfp4_to_f32(qx)
        hp_w = mxfp4_to_f32(qw)

        M, K = hp_x.shape
        N, K_w = hp_w.shape
        assert K == K_w, "K dimension mismatch between qx and qw"
        assert K % MXFP4_BLOCK_SIZE == 0, "K must be divisible by MXFP4 block size (32)"

        grid_k = K // MXFP4_BLOCK_SIZE

        # Scales may be padded; slice to valid region, then broadcast to K
        sx_f = e8m0_to_f32(sx[:M, :grid_k]).repeat_interleave(MXFP4_BLOCK_SIZE, dim=1)
        sw_f = e8m0_to_f32(sw[:N, :grid_k]).repeat_interleave(MXFP4_BLOCK_SIZE, dim=1)

        hp_x = hp_x * sx_f
        hp_w = hp_w * sw_f

        y = torch.mm(hp_x, hp_w.T).to(out_dtype)

        if accumulate:
            assert out is not None
            y = y + out.to(out_dtype)

        return y

    # ------------------------------------------------------------------
    # Stubs required by Quantizer base class
    # ------------------------------------------------------------------

    @property
    def supports_allgather_fp8(self) -> bool:
        return False

    @property
    def supports_dequantize(self) -> bool:
        return True

    def transpose_qresult(self, qresult):
        raise NotImplementedError

    @property
    def is_data_t_transposed_in_memory(self) -> bool:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Factory for CustomRecipe (module-level tests)
# ---------------------------------------------------------------------------

def mxfp4_ref_quantizer_factory(role: str) -> Optional[MXFP4QuantizerRef]:
    """Quantizer factory for use with ``recipe.CustomRecipe``.

    Maps TE module roles to appropriately configured MXFP4QuantizerRef
    instances.

    Usage::

        custom_recipe = recipe.CustomRecipe(
            qfactory=mxfp4_ref_quantizer_factory,
        )
        with te.autocast(enabled=True, recipe=custom_recipe):
            out = model(inp)
    """
    if role == "linear_input":
        return MXFP4QuantizerRef(rowwise=True, columnwise=False)
    if role == "linear_weight":
        return MXFP4QuantizerRef(rowwise=True, columnwise=True)
    if role == "linear_grad_output":
        return MXFP4QuantizerRef(rowwise=True, columnwise=False)
    return None
