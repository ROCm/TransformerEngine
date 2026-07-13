# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Shared helpers for the Triton GEMM backend.

Contains dtype conversion utilities, output-shape computation, and the
``Float8TensorWrapper`` / ``MXFP8TensorWrapper`` classes that adapt TE's
quantized tensors for use in the Python-side Triton GEMM path.
"""

import torch

import transformer_engine_torch as tex

# Reuse the shared dtype-conversion utilities that already live in the
# triton_kernels package. Keeping the GEMM backend on the same helpers as
# the norms / cast kernels avoids drift when new dtypes land.
from ..common import (
    get_torch_e4m3_type,
    get_torch_e5m2_type,
    torch_dtype_to_te_dtype,
    te_dtype_to_torch_dtype,
)


def is_fp8_dtype(dtype: tex.DType) -> bool:
    """Whether a TE ``DType`` is one of the FP8 variants."""
    return dtype in (tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2)


def reinterpret_as_fp8_tensor(a: torch.Tensor, dtype: tex.DType) -> torch.Tensor:
    """View a uint8 tensor as the architecture-native FP8 torch dtype.

    gfx942 (MI300/MI325) uses NANOO (``fnuz``) FP8 variants; gfx950 (MI350)
    uses OCP-standard variants. Delegates dtype selection to
    ``triton_kernels.common.get_torch_e4m3_type`` / ``_e5m2_type``.
    """
    if dtype == tex.DType.kFloat8E4M3:
        return a.view(dtype=get_torch_e4m3_type())
    if dtype == tex.DType.kFloat8E5M2:
        return a.view(dtype=get_torch_e5m2_type())

def getGemmOutputShape(A, transa, B, transb):
    """
    Compute output shape for GEMM following the C++ backend logic.

    Matches getGemmOutputShape in transformer_engine/pytorch/csrc/extensions/gemm.cpp

    Why Does This Preserve B's Batch Dimensions?
    =============================================

    This is a deliberate API design choice that makes the interface consistent and
    predictable for neural network operations.

    Usage Patterns in Linear Layer:

    1. Forward Pass (fprop) - Layout: TN (default)
       output = general_gemm(weight, input)
       - A = weight: [out_features, in_features] - no batch dims
       - B = input: [batch, seq_len, in_features] - HAS batch dims
       - Output: [batch, seq_len, out_features] - preserves B's batch ✓

    2. Input Gradient (dgrad) - Layout: NN
       grad_input = general_gemm(weight, grad_output)
       - A = weight: [out_features, in_features] - no batch dims
       - B = grad_output: [batch, seq_len, out_features] - HAS batch dims
       - Output: [batch, seq_len, in_features] - preserves B's batch ✓

    3. Weight Gradient (wgrad) - Layout: NT
       grad_weight = general_gemm(input, grad_output)
       - A = input: [batch, seq_len, in_features] - batch dims
       - B = grad_output: [batch, seq_len, out_features] - batch dims
       - Output: [out_features, in_features] - NO batch (transb=True) ✓

    Key Insight:
    The calling code consistently places the tensor with desired output batch
    structure as the B operand.
    - For fprop/dgrad: B has batch dimensions → output keeps them
    - For wgrad: Both have batch, use transb=True → output flattens them (reduction over batch)

    Why This Convention?
    - Consistency: Always put batched activations as B
    - Predictability: Output shape always relates to B's structure
    - Simplicity: Caller controls output shape by choosing B and transb
    - Efficiency: Avoids extra reshapes in common cases

    Could It Be Different?
    Yes! The API could preserve A's batch instead, but then all calling code would
    need to swap operands. The math would work the same, just with reversed convention.
    """
    # Handle both tensors and torch.Size objects
    A_shape = A if isinstance(A, torch.Size) else A.shape
    B_shape = B if isinstance(B, torch.Size) else B.shape

    # Calculate flattened dimensions (product of all leading dims)
    A0 = product(A_shape[:-1])  # Product of all leading dims
    A1 = A_shape[-1]
    B0 = product(B_shape[:-1])
    B1 = B_shape[-1]

    # Construct output shape following C++ logic:
    # if (transb) { ret = [B1] }
    # else { ret = [B_shape[0], B_shape[1], ..., B_shape[-2]] }  // Unflatten B0
    # if (transa) { ret.append(A0) }
    # else { ret.append(A1) }

    ret = []

    # First part: from B
    if transb:
        ret.append(B1)
    else:
        # Preserve B's batch structure (all dims except last)
        for i in range(len(B_shape) - 1):
            ret.append(B_shape[i])

    # Second part: from A
    if transa:
        ret.append(A0)  # Flattened A
    else:
        ret.append(A1)  # A's last dim

    return torch.Size(ret)

def product(shape):
    ret = 1
    for i in shape:
        ret *= i
    return ret


class Float8TensorWrapper:
    """
    Python equivalent of C++ TensorWrapper for Float8Tensor.

    Mimics the behavior of NVTETensorFromFloat8Tensor in type_converters_hip.cpp,
    which stores pointers to both rowwise (_data) and columnwise (_transpose) data
    without modifying them, similar to how the C++ TensorWrapper holds both formats.
    """

    def __init__(self, tensor):
        """
        Create wrapper from Float8Tensor, Float8TensorStorage, or regular tensor.

        Args:
            tensor: Input tensor (Float8Tensor, Float8TensorStorage, or torch.Tensor)
        """
        # Import here to avoid circular dependency
        try:
            from transformer_engine.pytorch.float8_tensor import Float8Tensor
            from transformer_engine.pytorch.tensor.storage.float8_tensor_storage import Float8TensorStorage
            is_fp8_tensor = isinstance(tensor, (Float8Tensor, Float8TensorStorage))
        except ImportError:
            is_fp8_tensor = False

        # Refuse other QuantizedTensorStorage subclasses (NVFP4, ...) rather
        # than falling through to the "regular tensor" branch, which crashes
        # on `tensor.dtype` (QuantizedTensorStorage exposes `_dtype`).
        if not is_fp8_tensor:
            try:
                from transformer_engine.pytorch.quantized_tensor import QuantizedTensorStorage
                if isinstance(tensor, QuantizedTensorStorage):
                    raise ValueError(
                        f"The Triton GEMM backend (NVTE_USE_GEMM_TRITON=1) does not "
                        f"support {type(tensor).__name__}. Only Float8Tensor / "
                        f"Float8TensorStorage (regular FP8) and MXFP8TensorStorage "
                        f"(via MXFP8TensorWrapper) are implemented. Disable the "
                        f"Triton backend for this recipe (unset NVTE_USE_GEMM_TRITON)."
                    )
            except ImportError:
                pass

        if is_fp8_tensor:
            # Extract FP8 components (similar to NVTETensorFromFloat8Tensor in C++)
            self._is_fp8 = True

            # Rowwise data (_data) - may be None
            self._rowwise_data = tensor._data if tensor._data is not None else None

            # Columnwise data (_transpose) - may be None
            self._columnwise_data = None
            transpose_valid = (
                hasattr(tensor, '_transpose') and
                tensor._transpose is not None and
                not getattr(tensor, '_transpose_invalid', False)
            )
            if transpose_valid:
                self._columnwise_data = tensor._transpose

            # Check that we have at least one data format
            if self._rowwise_data is None and self._columnwise_data is None:
                raise RuntimeError(
                    "Float8Tensor has neither valid rowwise (_data) nor columnwise (_transpose) data."
                )

            # FP8 metadata
            self._fp8_dtype = tensor._fp8_dtype
            self._scale_inv = tensor._scale_inv

            # Nominal dtype (may not exist for Float8TensorStorage)
            self._nominal_dtype = getattr(tensor, 'dtype', None)

            # Compute logical size (in rowwise format)
            if self._rowwise_data is not None:
                self._size = self._rowwise_data.size()
            else:
                # Only columnwise available
                # Columnwise format: [K, M, *batch_dims] (matrix dims first, batch dims at end)
                # Rowwise format: [*batch_dims, M, K] (batch dims first, matrix dims at end)
                self._original_columnwise_shape = self._columnwise_data.size()
                ndim = self._columnwise_data.dim()

                if ndim == 2:
                    # Simple 2D case: just transpose
                    rowwise_data = self._columnwise_data.transpose(0, 1).contiguous()
                else:
                    # fp8_transpose (see transpose_hip.cpp) treats an n-D rowwise
                    # tensor with shape (D0, ..., D_{n-2}, K) as 2-D (M, K) with
                    # M = prod(D0..D_{n-2}), transposes to (K, M), and re-shapes
                    # the result to [K, D0, ..., D_{n-2}]. To recover the original
                    # rowwise layout we must rotate the leading K dim back to the
                    # tail: [K, D0, ..., D_{n-2}] -> [D0, ..., D_{n-2}, K].
                    # The previous formula (batch_dims + [1, 0]) assumed columnwise
                    # was [K, M, b1, b2, ...] with M kept as a separate dim, which
                    # does not match fp8_transpose's output and silently scrambled
                    # the batch dimensions for ndim >= 3.
                    perm = list(range(1, ndim)) + [0]
                    rowwise_data = self._columnwise_data.permute(*perm).contiguous()

                # Store the rowwise data for use in get_data_for_gemm()
                self._rowwise_data = rowwise_data
                self._size = rowwise_data.size()
        else:
            # Regular tensor - simple wrapper
            self._is_fp8 = False
            self._rowwise_data = tensor
            self._columnwise_data = None
            self._fp8_dtype = None
            self._scale_inv = torch.Tensor()  # Empty tensor (data_ptr() == 0)
            self._nominal_dtype = tensor.dtype
            self._size = tensor.size()

    def size(self):
        """Get logical tensor size (in rowwise format)."""
        return self._size

    @property
    def is_fp8(self):
        """Check if this is an FP8 tensor."""
        return self._is_fp8

    @property
    def fp8_dtype(self):
        """Get FP8 dtype (tex.DType)."""
        return self._fp8_dtype

    @property
    def scale_inv(self):
        """Get scale inverse tensor."""
        return self._scale_inv

    @property
    def nominal_dtype(self):
        """Get nominal dtype (what the FP8 tensor represents, e.g., bfloat16)."""
        return self._nominal_dtype

    def get_data_for_gemm(self, will_transpose):
        """
        Get appropriate data tensor for GEMM operation.

        Always returns data in rowwise orientation to match self._size.

        Args:
            will_transpose: Whether the GEMM operation will transpose this operand
                           (currently unused - kept for future optimization)

        Returns:
            torch.Tensor: Data tensor in rowwise orientation (uint8 for FP8, regular dtype otherwise)
        """
        if not self._is_fp8:
            return self._rowwise_data

        # For FP8 tensors, always return rowwise orientation to match self._size
        if self._rowwise_data is not None:
            return self._rowwise_data
        else:
            # Only columnwise available - transpose back to rowwise
            # Columnwise has matrix dims (first 2) transposed, so transpose(0,1) gives rowwise
            return self._columnwise_data.transpose(0, 1).contiguous()


class MXFP8TensorWrapper:
    """
    Python equivalent of C++ TensorWrapper for MXFP8Tensor.

    Mimics NVTETensorFromMXFP8Tensor in type_converters.cpp, extracting
    both rowwise and columnwise data/scales.
    """

    def __init__(self, tensor):
        """
        Create wrapper from MXFP8Tensor or MXFP8TensorStorage.

        Args:
            tensor: Input tensor (MXFP8Tensor, MXFP8TensorStorage, or regular tensor)
        """
        # Import here to avoid circular dependency
        try:
            from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor
            from transformer_engine.pytorch.tensor.storage.mxfp8_tensor_storage import MXFP8TensorStorage
            is_mxfp8_tensor = isinstance(tensor, (MXFP8Tensor, MXFP8TensorStorage))
        except ImportError:
            is_mxfp8_tensor = False

        if is_mxfp8_tensor:
            # Extract MXFP8 components (matching NVTETensorFromMXFP8Tensor)
            self._is_mxfp8 = True

            # Rowwise data and scales
            self._rowwise_data = tensor._rowwise_data if hasattr(tensor, '_rowwise_data') and tensor._rowwise_data is not None else None
            self._rowwise_scale_inv = tensor._rowwise_scale_inv if hasattr(tensor, '_rowwise_scale_inv') and tensor._rowwise_scale_inv is not None else None

            # Columnwise data and scales
            self._columnwise_data = tensor._columnwise_data if hasattr(tensor, '_columnwise_data') and tensor._columnwise_data is not None else None
            self._columnwise_scale_inv = tensor._columnwise_scale_inv if hasattr(tensor, '_columnwise_scale_inv') and tensor._columnwise_scale_inv is not None else None

            # Verify we have at least one format
            if self._rowwise_data is None and self._columnwise_data is None:
                raise RuntimeError(
                    "MXFP8Tensor has neither rowwise nor columnwise data"
                )

            # FP8 metadata
            self._fp8_dtype = tensor._fp8_dtype
            self._nominal_dtype = tensor.dtype if hasattr(tensor, 'dtype') else torch.float32

            # Determine logical size from available data
            if self._rowwise_data is not None:
                self._size = self._rowwise_data.size()
            else:
                # IMPORTANT: For MXFP8, columnwise has the SAME shape as rowwise
                # (unlike Float8Tensor where columnwise is transposed)
                # Both rowwise and columnwise have shape [*batch, M, K]
                self._size = self._columnwise_data.size()
        else:
            # Not MXFP8 - wrap as regular tensor
            self._is_mxfp8 = False
            self._rowwise_data = tensor
            self._columnwise_data = None
            self._rowwise_scale_inv = None
            self._columnwise_scale_inv = None
            self._fp8_dtype = None
            self._nominal_dtype = tensor.dtype
            self._size = tensor.size()

    def size(self):
        """Get logical tensor size (in rowwise format)."""
        return self._size

    @property
    def is_mxfp8(self):
        """Check if this is an MXFP8 tensor."""
        return self._is_mxfp8

    @property
    def fp8_dtype(self):
        """Get FP8 dtype."""
        return self._fp8_dtype

    @property
    def nominal_dtype(self):
        """Get nominal dtype (what the MXFP8 tensor represents)."""
        return self._nominal_dtype

    def get_data_and_scale_for_gemm(self, will_transpose):
        """
        Get appropriate data and scale tensors for GEMM based on transpose flag.

        For MXFP8, scales are tied to the data layout due to block quantization.
        We must select the pre-quantized copy that matches our needs:
        - will_transpose=True: use columnwise (already transposed, avoids requantization)
        - will_transpose=False: use rowwise (normal orientation)

        Args:
            will_transpose: Whether this operand will be transposed in GEMM

        Returns:
            tuple: (data_tensor, scale_inv_tensor)
        """
        if not self._is_mxfp8:
            # Regular tensor - no scales
            return self._rowwise_data, None

        # Select appropriate pre-quantized copy based on transpose
        if will_transpose:
            # Will be transposed: use columnwise copy (already in transposed orientation)
            if self._columnwise_data is not None:
                return self._columnwise_data, self._columnwise_scale_inv
            else:
                # Fallback: use rowwise (will have scale mismatch issues)
                import warnings
                warnings.warn("MXFP8: transpose requested but no columnwise copy available")
                return self._rowwise_data, self._rowwise_scale_inv
        else:
            # Not transposed: use rowwise copy
            if self._rowwise_data is not None:
                return self._rowwise_data, self._rowwise_scale_inv
            else:
                raise RuntimeError("MXFP8Tensor missing rowwise data")
