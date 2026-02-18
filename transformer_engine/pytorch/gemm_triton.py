# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#
# License for AMD contributions = MIT. See LICENSE for more information

from enum import IntEnum
import functools
import torch

import transformer_engine_torch as tex
from transformer_engine.pytorch.constants import MXFP8_BLOCK_SCALING_SIZE

import triton
import triton.language as tl

@functools.lru_cache(maxsize=1)
def _get_fp8_dtypes():
    """
    Get the appropriate FP8 dtypes based on GPU architecture.

    AMD GPU FP8 format support:
    - gfx942 (MI300/MI325): Uses NANOO FP8 formats
        - torch.float8_e4m3fnuz (e4m3)
        - torch.float8_e5m2fnuz (e5m2)
    - gfx950 (MI350): Uses OCP standard FP8 formats (same as NVIDIA)
        - torch.float8_e4m3fn (e4m3)
        - torch.float8_e5m2 (e5m2)

    Returns:
        tuple: (e4m3_dtype, e5m2_dtype) - PyTorch FP8 dtypes for current architecture
    """
    major, minor = torch.cuda.get_device_capability()

    # gfx950 (compute capability 9.5) uses OCP standard FP8 formats
    if major == 9 and minor >= 5:
        return (torch.float8_e4m3fn, torch.float8_e5m2)

    # gfx942 (compute capability 9.4) and earlier use NANOO FP8 formats
    # This includes MI300/MI325 (gfx942) and MI200 (gfx90a)
    return (torch.float8_e4m3fnuz, torch.float8_e5m2fnuz)

def torch_to_te_dtype(dtype):
    torch_to_TE_dtypes = {
        torch.int8: tex.DType.kByte,
        torch.int32: tex.DType.kInt32,
        torch.float32: tex.DType.kFloat32,
        torch.float16: tex.DType.kFloat16,
        torch.bfloat16: tex.DType.kBFloat16,
        # Both NANOO and OCP FP8 formats map to the same TE dtypes
        torch.float8_e4m3fnuz: tex.DType.kFloat8E4M3,  # NANOO format (gfx942)
        torch.float8_e5m2fnuz: tex.DType.kFloat8E5M2,  # NANOO format (gfx942)
        torch.float8_e4m3fn: tex.DType.kFloat8E4M3,    # OCP format (gfx950)
        torch.float8_e5m2: tex.DType.kFloat8E5M2,      # OCP format (gfx950)
    }
    return torch_to_TE_dtypes[dtype]

def te_to_torch_dtype(dtype):
    te_dtype_to_torch_dtype = {
            tex.DType.kByte : torch.int8,
            tex.DType.kInt32 : torch.int32,
            tex.DType.kFloat32 : torch.float32,
            tex.DType.kFloat16 : torch.float16,
            tex.DType.kBFloat16 : torch.bfloat16,
            #tex.DType.kFloat8E4M3: torch.float8_e4m3fnuz,
            #tex.DType.kFloat8E5M2: torch.float8_e5m2fnuz,
            # Currently, TE does not use Pytorch's fp8 data types
            # Instead it has its own Float8Tensor, which uses
            # torch.uint8 as its data type
            tex.DType.kFloat8E4M3: torch.uint8,
            tex.DType.kFloat8E5M2: torch.uint8,
            }
    return te_dtype_to_torch_dtype[dtype]

def is_fp8_dtype(dtype):
    return dtype in (tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2)

def reinterpret_as_fp8_tensor(a: torch.Tensor, dtype: tex.DType):
    """
    Reinterpret a uint8 tensor as an FP8 tensor using the appropriate format for the GPU architecture.

    Uses architecture-specific FP8 formats:
    - gfx942 (MI300/MI325): NANOO FP8 (fnuz variants)
    - gfx950 (MI350): OCP FP8 (fn/standard variants)
    """
    fp8_e4m3_dtype, fp8_e5m2_dtype = _get_fp8_dtypes()

    if dtype == tex.DType.kFloat8E4M3:
        return a.view(dtype=fp8_e4m3_dtype)
    if dtype == tex.DType.kFloat8E5M2:
        return a.view(dtype=fp8_e5m2_dtype)

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
        Create wrapper from Float8Tensor, Float8TensorBase, or regular tensor.

        Args:
            tensor: Input tensor (Float8Tensor, Float8TensorBase, or torch.Tensor)
        """
        # Import here to avoid circular dependency
        try:
            from transformer_engine.pytorch.float8_tensor import Float8Tensor
            from transformer_engine.pytorch.tensor._internal.float8_tensor_base import Float8TensorBase
            is_fp8_tensor = isinstance(tensor, (Float8Tensor, Float8TensorBase))
        except ImportError:
            is_fp8_tensor = False

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

            # Nominal dtype (may not exist for Float8TensorBase)
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
                    # Batch dimensions exist (at the end of columnwise)
                    # Move batch dims to front and swap matrix dims: [K,M,b1,b2,...] -> [b1,b2,...,M,K]
                    # Create permutation: (2, 3, ..., ndim-1, 1, 0)
                    batch_dims = list(range(2, ndim))  # [2, 3, ..., ndim-1]
                    perm = batch_dims + [1, 0]  # [2,3,...,ndim-1, 1, 0]
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
        Create wrapper from MXFP8Tensor or MXFP8TensorBase.

        Args:
            tensor: Input tensor (MXFP8Tensor, MXFP8TensorBase, or regular tensor)
        """
        # Import here to avoid circular dependency
        try:
            from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor
            from transformer_engine.pytorch.tensor._internal.mxfp8_tensor_base import MXFP8TensorBase
            is_mxfp8_tensor = isinstance(tensor, (MXFP8Tensor, MXFP8TensorBase))
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


def te_generic_gemm_triton(A,
                            transa,
                            B,
                            transb,
                            D,
                            quantizer,
                            output_dtype,
                            bias,
                            bias_type,
                            gelu,
                            gelu_in,
                            grad,
                            workspace,
                            workspaceSize,
                            accumulate,
                            use_split_accumulator,
                            comm_overlap,
                            comm_type,
                            extra_output,
                            bulk_overlap):

    # Wrap inputs to handle Float8Tensor and MXFP8Tensor uniformly
    # Try MXFP8 first, then Float8, then regular
    try:
        from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor
        from transformer_engine.pytorch.tensor._internal.mxfp8_tensor_base import MXFP8TensorBase
        is_mxfp8_a = isinstance(A, (MXFP8Tensor, MXFP8TensorBase))
        is_mxfp8_b = isinstance(B, (MXFP8Tensor, MXFP8TensorBase))
    except ImportError:
        is_mxfp8_a = False
        is_mxfp8_b = False

    if is_mxfp8_a or is_mxfp8_b:
        # Use MXFP8TensorWrapper
        A_wrapper = MXFP8TensorWrapper(A)
        B_wrapper = MXFP8TensorWrapper(B)

        # Validate both are MXFP8
        if A_wrapper.is_mxfp8 != B_wrapper.is_mxfp8:
            raise ValueError("Mixed MXFP8 and non-MXFP8 inputs not supported")

        # IMPORTANT: Match the C++ CanonicalizeGemmInput logic for MXFP8
        # The C++ code selects data/scales based on the BLAS transpose flags:
        #
        # For A:
        #   - transa=True:  Use rowwise data and scales
        #   - transa=False: Use columnwise data and scales
        # For B:
        #   - transb=True:  Use columnwise data and scales
        #   - transb=False: Use rowwise data and scales
        #
        # The wrapper's get_data_and_scale_for_gemm takes will_transpose parameter:
        #   - will_transpose=True → returns columnwise
        #   - will_transpose=False → returns rowwise

        # Extract data and scales for Triton (row-major) requirements
        # Triton needs different selection than C++ because it works in row-major
        # For A (first operand): needs [M, K] with scales [M, K//32] (rowwise pattern)
        # For B (second operand): needs [K, N] with scales [K//32, N] (columnwise pattern)

        # Debug: print available data
        import os
        if os.getenv("DEBUG_MXFP8_SELECT"):
            print(f"[DEBUG] MXFP8 data selection:")
            print(f"  A shape: {A_wrapper.size()}, transA={transa}")
            if hasattr(A_wrapper, '_rowwise_data') and A_wrapper._rowwise_data is not None:
                print(f"    A rowwise: data {A_wrapper._rowwise_data.shape}, scale {A_wrapper._rowwise_scale_inv.shape}")
            if hasattr(A_wrapper, '_columnwise_data') and A_wrapper._columnwise_data is not None:
                print(f"    A columnwise: data {A_wrapper._columnwise_data.shape}, scale {A_wrapper._columnwise_scale_inv.shape}")
            print(f"  B shape: {B_wrapper.size()}, transB={transb}")
            if hasattr(B_wrapper, '_rowwise_data') and B_wrapper._rowwise_data is not None:
                print(f"    B rowwise: data {B_wrapper._rowwise_data.shape}, scale {B_wrapper._rowwise_scale_inv.shape}")
            if hasattr(B_wrapper, '_columnwise_data') and B_wrapper._columnwise_data is not None:
                print(f"    B columnwise: data {B_wrapper._columnwise_data.shape}, scale {B_wrapper._columnwise_scale_inv.shape}")

        # MXFP8 Selection for BLAS API compatibility
        #
        # The API uses BLAS convention with column-major interpretation
        # We need to select the right MXFP8 format based on BLAS transpose flags
        # Following the C++ logic from CanonicalizeGemmInput:
        #   - When transA=True: use rowwise
        #   - When transA=False: use columnwise
        #   - When transB=True: use columnwise
        #   - When transB=False: use rowwise

        if transa:
            # BLAS transA=True → use rowwise
            A_data = A_wrapper._rowwise_data
            a_scale_inv = A_wrapper._rowwise_scale_inv
        else:
            # BLAS transA=False → use columnwise
            A_data = A_wrapper._columnwise_data
            a_scale_inv = A_wrapper._columnwise_scale_inv

        if transb:
            # BLAS transB=True → use columnwise
            B_data = B_wrapper._columnwise_data
            b_scale_inv = B_wrapper._columnwise_scale_inv
        else:
            # BLAS transB=False → use rowwise
            B_data = B_wrapper._rowwise_data
            b_scale_inv = B_wrapper._rowwise_scale_inv

        # Debug output
        if os.getenv("DEBUG_MXFP8_SELECT"):
            print(f"[DEBUG] MXFP8 selection with logical transpose:")
            print(f"  transA={transa}, transB={transb}")
            print(f"  A selected: data {A_data.shape}, scale {a_scale_inv.shape}")
            print(f"  B selected: data {B_data.shape}, scale {b_scale_inv.shape}")

        a_fp8_dtype = A_wrapper.fp8_dtype
        b_fp8_dtype = B_wrapper.fp8_dtype

        input_mxfp8 = True
    else:
        # Use Float8TensorWrapper (existing code)
        A_wrapper = Float8TensorWrapper(A)
        B_wrapper = Float8TensorWrapper(B)

        A_data = A_wrapper.get_data_for_gemm(will_transpose=transa)
        B_data = B_wrapper.get_data_for_gemm(will_transpose=transb)

        a_fp8_dtype = A_wrapper.fp8_dtype
        b_fp8_dtype = B_wrapper.fp8_dtype
        a_scale_inv = A_wrapper.scale_inv
        b_scale_inv = B_wrapper.scale_inv

        input_mxfp8 = False

    # Reinterpret uint8 as native FP8 types for Triton
    # The FP8 tensor data is stored as torch.uint8 but Triton needs torch.float8_e4m3fnuz
    if a_fp8_dtype is not None:
        A_data = reinterpret_as_fp8_tensor(A_data, a_fp8_dtype)
    if b_fp8_dtype is not None:
        B_data = reinterpret_as_fp8_tensor(B_data, b_fp8_dtype)

    # Compute dimensions using wrapper sizes
    # Wrapper handles Float8TensorBase which doesn't have .shape attribute
    #
    # BLAS column-major interpretation:
    # PyTorch tensors are row-major in memory, but BLAS interprets them as column-major.
    # A PyTorch tensor with shape [X, Y] is seen by BLAS as column-major [Y, X].
    #
    # For A with PyTorch shape [A0, A1]:
    #   - BLAS sees it as column-major with A0 columns and A1 rows
    #   - If transa=False: we use A as-is in column-major = [A1, A0] in BLAS = shape (A1 rows, A0 cols)
    #                      So M = A1, K = A0
    #   - If transa=True:  we transpose in column-major = [A0, A1] in BLAS = shape (A0 rows, A1 cols)
    #                      So M = A0, K = A1
    #
    # For B with PyTorch shape [B0, B1]:
    #   - If transb=False: BLAS sees [B1, B0], so K = B1, N = B0... wait that's backwards
    #
    # Actually, let me use the test as reference. From test_gemm_triton.py line 126-127:
    #   a is (K, M) in PyTorch → BLAS column-major (M, K)
    #   b is (N, K) in PyTorch → BLAS column-major (K, N)
    # So PyTorch (row_dim, col_dim) → BLAS column-major (col_dim, row_dim)
    #
    # For matrix A with PyTorch shape (A_shape[0], A_shape[1]):
    #   - BLAS column-major interpretation: (A_shape[1], A_shape[0])
    #   - If transa=False: use as column-major (A_shape[1], A_shape[0]), so m=A_shape[1], k=A_shape[0]
    #   - If transa=True: transpose in column-major gives (A_shape[0], A_shape[1]), so m=A_shape[0], k=A_shape[1]
    #
    # Using product notation where A0 = product(A_shape[:-1]), A1 = A_shape[-1]:
    A0 = product(A_wrapper.size()[:-1])  # First dim(s)
    A1 = product(A_wrapper.size()[-1:])  # Last dim
    B0 = product(B_wrapper.size()[:-1])
    B1 = product(B_wrapper.size()[-1:])

    m = A0 if transa else A1  # Original code
    k = A1 if transa else A0  # Original code
    n = B1 if transb else B0  # Original code

    assert not (transa and transb), 'TT layout not allowed'

    ## general_gemm() follows BLAS convention: tensors are interpreted as column-major
    ## PyTorch tensors are stored row-major in memory, but BLAS APIs treat them as column-major
    ## Triton matmul kernel expects row-major layout
    ## Convert using the standard trick: swap operands and transpose as needed
    ##
    ## For column-major interpretation:
    ##   TN: compute B @ A.T (transpose A, not B)
    ##   NN: compute B @ A (no transposes)
    ##   NT: compute B.T @ A (transpose B, not A)

    # For multi-dimensional tensors: flatten leading dims first, then transpose
    # This implements "flattened multi-dimensional matmul" semantics
    A_flat = A_data.reshape(-1, A_data.shape[-1])  # [prod(batch dims), last_dim]
    B_flat = B_data.reshape(-1, B_data.shape[-1])

    # For MXFP8, also flatten scale tensors to match data flattening
    if input_mxfp8:
        # MXFP8 scales must be flattened to match data
        if a_scale_inv is not None and a_scale_inv.dim() > 2:
            a_scale_inv = a_scale_inv.reshape(-1, a_scale_inv.shape[-1])
        if b_scale_inv is not None and b_scale_inv.dim() > 2:
            b_scale_inv = b_scale_inv.reshape(-1, b_scale_inv.shape[-1])

        # Skip padding handling for now - will be fixed in a later update
        # from the ROCm Transformer Engine repo

    # Convert from BLAS column-major to Triton row-major by swapping operands
    if input_mxfp8:
        # MXFP8 follows the same pattern as regular FP8:
        # 1. Swap operands (B becomes first, A becomes second)
        # 2. Apply logical transpose based on the swapped flags
        # 3. For MXFP8, transpose BOTH data and scales

        # First operand for Triton (originally B from BLAS)
        # Apply transpose if transb is True
        a_row_major = B_flat.T if transb else B_flat
        a_scale_triton = b_scale_inv.T if (transb and b_scale_inv is not None) else b_scale_inv

        # Second operand for Triton (originally A from BLAS)
        # Apply transpose if transa is True
        b_row_major = A_flat.T if transa else A_flat
        b_scale_triton = a_scale_inv.T if (transa and a_scale_inv is not None) else a_scale_inv
    else:
        # For regular FP8 and standard types, apply BLAS column-major to row-major conversion
        # This swaps A and B
        a_row_major = B_flat.T if transb else B_flat
        b_row_major = A_flat.T if transa else A_flat
        # Scales are swapped to match operand swap (B→a, A→b in row-major)
        a_scale_triton = b_scale_inv
        b_scale_triton = a_scale_inv

    epilogue = 'DEFAULT'
    #if bias.data_ptr() != 0:
        #if grad:
            #epilogue = 'BGRADB'
        #else:
            #epilogue = 'BIAS'

    # Compute output shape
    if input_mxfp8:
        # For MXFP8, we swapped operands, so use BLAS logic for output shape
        # BLAS computes in column-major, and we want the row-major result
        D_shape = getGemmOutputShape(A_wrapper.size(), transa, B_wrapper.size(), transb)
    else:
        # For regular path, use BLAS column-major logic
        D_shape = getGemmOutputShape(A_wrapper.size(), transa, B_wrapper.size(), transb)

    if D is None:
        # Determine output dtype
        if output_dtype is not None:
            # Use explicitly provided output dtype (from TE_DType)
            out_dtype = te_to_torch_dtype(output_dtype)
        elif hasattr(A_wrapper, 'is_mxfp8') and A_wrapper.is_mxfp8:
            # MXFP8 input: use nominal dtype
            out_dtype = A_wrapper.nominal_dtype
        elif hasattr(A_wrapper, 'is_fp8') and A_wrapper.is_fp8:
            # Regular FP8 input: use nominal dtype if available
            if A_wrapper.nominal_dtype is None:
                raise RuntimeError(
                    "FP8 input detected (Float8TensorBase without nominal dtype) but output_dtype "
                    "parameter is not provided. Please explicitly provide the output_dtype parameter "
                    "to general_gemm()."
                )
            out_dtype = A_wrapper.nominal_dtype
        else:
            # Regular input: use A's dtype
            out_dtype = A_data.dtype

        D = torch.empty(D_shape, dtype=out_dtype, device=A_data.device)

    d_row_major = D.view(-1, D.shape[-1])

    # Set FP8 flags
    is_fp8_wrapper = hasattr(A_wrapper, 'is_fp8') and A_wrapper.is_fp8 and B_wrapper.is_fp8
    is_mxfp8_wrapper = hasattr(A_wrapper, 'is_mxfp8') and A_wrapper.is_mxfp8 and B_wrapper.is_mxfp8
    input_fp8 = is_fp8_wrapper or is_mxfp8_wrapper
    output_fp8 = False  # Not supporting FP8 output yet

    # Empty tensors for unused parameters (matching C++ empty tensor pattern)
    D_scale = torch.Tensor()
    bias_tensor = torch.Tensor()
    D_amax = torch.Tensor()

    # Dispatch to appropriate kernel based on input type
    if input_mxfp8:
        # MXFP8 path: compute directly in row-major without BLAS column-major conversion
        #
        # The BLAS column-major conversion (swapping A and B) doesn't work well for MXFP8
        # because the scales are tied to specific data orientations.
        # Instead, we compute the matmul directly based on what the user requested.
        #
        # User requested (in BLAS column-major): C = op(A) @ op(B)
        # We need to figure out what that means in row-major and call the kernel.
        #
        # The kernel computes: C = A_kernel @ B_kernel in row-major
        #
        # Mapping:
        # - NN layout (transa=False, transb=False): C = A @ B
        #   Row-major: C[M,N] = A[M,K] @ B[K,N]
        #   Use: A_data (no transpose), B_data (no transpose)
        #
        # - TN layout (transa=True, transb=False): C = A^T @ B
        #   Row-major: C[M,N] = A^T[M,K] @ B[K,N] where A is originally [K,M]
        #   Use: A_data^T or columnwise, B_data
        #
        # - NT layout (transa=False, transb=True): C = A @ B^T
        #   Row-major: C[M,N] = A[M,K] @ B^T[K,N] where B is originally [N,K]
        #   Use: A_data, B_data^T or columnwise

        # Use output dimensions to get correct M, N
        # After operand swap and transpose handling:
        # - a_row_major: [M, K] (first operand for Triton)
        # - b_row_major: [K, N] (second operand for Triton)
        # - d_row_major: [M, N] (output)
        actual_m = d_row_major.shape[0]
        actual_n = d_row_major.shape[1]

        # Verify operands are compatible for matmul
        if a_row_major.shape[1] != b_row_major.shape[0]:
            print(f"[ERROR] Dimension mismatch after swap/transpose:")
            print(f"  a_row_major: {a_row_major.shape}")
            print(f"  b_row_major: {b_row_major.shape}")
            print(f"  Cannot multiply: {a_row_major.shape} @ {b_row_major.shape}")
            print(f"  Original: A{A_wrapper.size()}, B{B_wrapper.size()}, trans={'T' if transa else 'N'}{'T' if transb else 'N'}")
            assert False, f"Dimension mismatch: {a_row_major.shape} @ {b_row_major.shape}"
        actual_k = a_row_major.shape[1]

        # Debug output
        import os
        if os.getenv("DEBUG_MXFP8_GEMM"):
            print(f"\n[DEBUG] MXFP8 GEMM call:")
            print(f"  BLAS API: A{A_wrapper.size()}, B{B_wrapper.size()}, trans={'T' if transa else 'N'}{'T' if transb else 'N'}")

            # Identify the operation type based on shapes and transpose flags
            op_type = "unknown"
            if transa and not transb:
                op_type = "fprop (TN)"
            elif not transa and not transb:
                op_type = "dgrad (NN)"
            elif not transa and transb:
                op_type = "wgrad (NT)"

            # For wgrad, provide more details about the tensor interpretation
            if op_type == "wgrad (NT)":
                print(f"  Interpreting wgrad tensors:")
                print(f"    A (grad_output): original shape {A_wrapper.size()}")
                print(f"      After flatten: {A_flat.shape}")
                print(f"    B (input): original shape {B_wrapper.size()}")
                print(f"      After flatten: {B_flat.shape}")
                print(f"    Expected: dY^T @ X where dY=[batch*seq, out_feat], X=[batch*seq, in_feat]")

            print(f"  Operation type: {op_type}")
            print(f"  Expected output shape: {D_shape}")
            print(f"  After operand swap and transpose for Triton:")
            print(f"    First operand: {a_row_major.shape} (from B, transb={transb})")
            print(f"    Second operand: {b_row_major.shape} (from A, transa={transa})")
            print(f"    Output: {d_row_major.shape}")
            print(f"  Actual dimensions: M={actual_m}, N={actual_n}, K={actual_k}")
            if a_scale_triton is not None and b_scale_triton is not None:
                print(f"  Scale shapes: {a_scale_triton.shape}, {b_scale_triton.shape}")
                # Check scale compatibility
                expected_a_scale = (actual_m, actual_k // 32)
                expected_b_scale = (actual_k // 32, actual_n)
                if a_scale_triton.shape[:2] != expected_a_scale:
                    print(f"    ⚠ First scale mismatch: {a_scale_triton.shape} vs expected {expected_a_scale}")
                if b_scale_triton.shape[:2] != expected_b_scale:
                    print(f"    ⚠ Second scale mismatch: {b_scale_triton.shape} vs expected {expected_b_scale}")

        # Call kernel with correct dimensions
        # Use the actual output dimensions and inner dimension
        # After swapping and transpose, we have:
        # a_row_major: [M, K] (first operand)
        # b_row_major: [K, N] (second operand)
        # d_row_major: [M, N] (output)

        # Verify dimensions match
        assert a_row_major.shape[1] == b_row_major.shape[0], \
            f"Inner dimensions don't match: {a_row_major.shape} @ {b_row_major.shape}"

        mxfp8_matmul(
            a_row_major, a_scale_triton,  # First operand (from B)
            b_row_major, b_scale_triton,  # Second operand (from A)
            d_row_major,                  # Output
            actual_m, actual_n, actual_k,  # Use pre-computed dimensions
            b_fp8_dtype, a_fp8_dtype      # Swap FP8 formats to match swapped operands
        )
    else:
        # Call regular FP8 or standard matmul kernel
        matmul(a_row_major, b_row_major, d_row_major, a_scale_triton, b_scale_triton,
               D_scale, bias_tensor, D_amax, epilogue, input_fp8, output_fp8)

    return D, bias, None, None
        
    
                            
def te_gemm_triton(A,
                   A_scale_inverse,
                   A_fp8_tensor,
                   A_type,
                   transa,
                   B,
                   B_scale_inverse,
                   B_fp8_tensor,
                   B_type,
                   transb,
                   D,
                   D_scale,
                   D_type,
                   D_amax,
                   bias,
                   bias_type,
                   pre_gelu_out,
                   grad,
                   # Below are dummy inputs for now
                   workspace, 
                   workspaceSize, 
                   accumulate, 
                   use_split_accumulator 
                   ):
    '''
    Returns:
        None

    Currently support epilogues: DEFAULT, BIAS, BIAS_BGRADB
    TODO: To support GELU_AUX, DGELU, GELU_AUX_BIAS, DGELU_BGRAD

    epilogue               bias         gelu       grad
    DEFAULT:               False        False      False 
    BIAS:                  True         False      False
    BIAS_BGRADB:           True         False      True
    GELU_AUX:              False        True       False 
    DGELU:                 False        True       True 
    GELU_AUX_BIAS:         True         True       False
    DGELU_BGRAD:           True         True       True

    When bias or pre_gelu_out is not used, they are passed in as torch.Tensor()
    which is an empty tensor, which has data_ptr() == 0

    Trans(A) = A.T if transa else A
    Trans(B) = B.T if transb else B
    Trans(A) is (blas_n, blas_k) in column major - (blas_k, blas_n) in row major
    Trans(B) is (blas_k, blas_n) in column major - (blas_n, blas_k) in row major
    blas_m, blas_n, blas_k here is consistent with the notation in BLAS
    For epilogue BIAS, bias vector length is blas_m
    for epilogue BGRADB, bias gradient vector length is blas_n
    '''
    assert te_to_torch_dtype(A_type) == A.dtype, 'A dtype does not match.'
    assert te_to_torch_dtype(B_type) == B.dtype, 'B dtype does not match.'
    assert te_to_torch_dtype(D_type) == D.dtype, 'D dtype does not match.'
    assert (bias.data_ptr() == 0) or (te_to_torch_dtype(bias_type) == bias.dtype), 'bias dtype does not match.'
    

    assert not is_fp8_dtype(A_type) or A_scale_inverse.data_ptr() != 0, 'fp8 input to GEMM requires inverse of scale!'
    assert not is_fp8_dtype(B_type) or B_scale_inverse.data_ptr() != 0, 'fp8 input to GEMM requires inverse of scale!'

    ## The fp8 tensor passed from TE is in torch.uint8
    ## Need to reinterpret as the float8 type in torch
    if is_fp8_dtype(A_type):
        A = reinterpret_as_fp8_tensor(A, A_type)

    if is_fp8_dtype(B_type):
        B = reinterpret_as_fp8_tensor(B, B_type)

    if is_fp8_dtype(D_type):
        D = reinterpret_as_fp8_tensor(D, D_type)

    if A_scale_inverse.numel():
        A_scale_inverse = A_scale_inverse[A_fp8_tensor]

    if B_scale_inverse.numel():
        B_scale_inverse = B_scale_inverse[B_fp8_tensor]

    m = A.shape[0] if transa else A.shape[1]
    k = A.shape[1] if transa else A.shape[0]
    n = B.shape[1] if transb else B.shape[0]

    assert not (transa and transb), 'TT layout not allowed'

    assert pre_gelu_out.data_ptr() == 0, 'GEMM+Gelu is not supported yet.'

    ## A and B are column major following BLAS convention
    ## Triton matmul function assumes row major layouts
    ## Therefore, use the trick of swapping operands again 
    a_row_major = B.T if transb else B
    b_row_major = A.T if transa else A
    a_scale_triton = B_scale_inverse
    b_scale_triton = A_scale_inverse

    epilogue = 'DEFAULT'
    if bias.data_ptr() != 0:
        if grad:
            epilogue = 'BGRADB'
        else:
            epilogue = 'BIAS'

    input_fp8 = is_fp8_dtype(A_type) and is_fp8_dtype(B_type)
    output_fp8 = is_fp8_dtype(D_type)
    matmul(a_row_major, b_row_major, D, a_scale_triton, b_scale_triton, D_scale, bias, D_amax, epilogue, input_fp8, output_fp8)


# MXFP8 (Microscaling FP8) Matmul Kernel and Wrapper
# Uses Triton's tl.dot_scaled() for native block-scaled FP8 matmul

@triton.autotune(
    configs=[
        # Simpler configs for MXFP8 - BLOCK_K must be multiple of 32 (VEC_SIZE)
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 4}),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 4}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4}),
    ],
    key=['M', 'N', 'K'],
)
@triton.heuristics({
    'EVEN_K': lambda args: args['K'] % args['BLOCK_SIZE_K'] == 0,
})
@triton.jit
def mxfp8_matmul_kernel(
    # Data pointers
    a_ptr, b_ptr, c_ptr,
    # Scale pointers (E8M0 format, uint8)
    a_scale_ptr, b_scale_ptr,
    # Matrix dimensions
    M, N, K,
    # Data strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Scale strides
    stride_a_scale_m, stride_a_scale_k,
    stride_b_scale_k, stride_b_scale_n,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    VEC_SIZE: tl.constexpr,  # MXFP8_BLOCK_SCALING_SIZE (always 32)
    FP8_FORMAT_A: tl.constexpr,  # "e4m3" or "e5m2"
    FP8_FORMAT_B: tl.constexpr,  # "e4m3" or "e5m2"
):
    """
    MXFP8 matmul kernel using tl.dot_scaled() for block-scaled FP8 computation.

    Scales are stored in E8M0 format (uint8 biased exponents) and converted to FP32.
    """

    # Program ID
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # Swizzled block mapping for better L2 cache utilization
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Compute block offsets
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Data pointers
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # K-loop
    num_k_blocks = tl.cdiv(K, BLOCK_SIZE_K)
    for k in range(num_k_blocks):
        # Load FP8 data
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            k_remaining = K - k * BLOCK_SIZE_K
            mask_k = offs_k < k_remaining
            a = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)

        # Load E8M0 scales for this K-block
        # We have:
        # - A columnwise scales: [M//VEC_SIZE, K] - one scale per 32 rows for each column
        # - B rowwise scales: [K, N//VEC_SIZE] - one scale per 32 columns for each row
        #
        # tl.dot_scaled expects:
        # - A scales: [BLOCK_SIZE_M, BLOCK_SIZE_K // VEC_SIZE] - one scale per 32 elements along K
        # - B scales: [BLOCK_SIZE_K // VEC_SIZE, BLOCK_SIZE_N] - one scale per 32 elements along K
        #
        # For columnwise A: each column has M//32 scales
        # We need to gather the right scales for our block

        # A scales: we need scales for rows [pid_m*BLOCK_SIZE_M : (pid_m+1)*BLOCK_SIZE_M]
        #           and columns [k*BLOCK_SIZE_K : (k+1)*BLOCK_SIZE_K]
        # From columnwise layout [M//VEC_SIZE, K], we need to:
        # - Select row indices: [pid_m*BLOCK_SIZE_M//VEC_SIZE : (pid_m+1)*BLOCK_SIZE_M//VEC_SIZE]
        # - Select column indices: [k*BLOCK_SIZE_K : (k+1)*BLOCK_SIZE_K]
        # This gives us [BLOCK_SIZE_M//VEC_SIZE, BLOCK_SIZE_K] scales
        # But we need [BLOCK_SIZE_M, BLOCK_SIZE_K//VEC_SIZE] for tl.dot_scaled

        # Actually, let's think about what the scales mean:
        # Columnwise: scale[i,j] applies to data[i*32:(i+1)*32, j]
        # So for block starting at row pid_m*BLOCK_SIZE_M, col k*BLOCK_SIZE_K:
        # We need scales from row indices pid_m*BLOCK_SIZE_M//32 to (pid_m+1)*BLOCK_SIZE_M//32

        # With reversed selection, A now has rowwise scales [M, K//32]
        # For tl.dot_scaled we need [BLOCK_SIZE_M, BLOCK_SIZE_K//32]
        k_block_start = k * (BLOCK_SIZE_K // VEC_SIZE)
        offs_a_scale_k = k_block_start + tl.arange(0, BLOCK_SIZE_K // VEC_SIZE)

        a_scale_ptrs = a_scale_ptr + (offs_am[:, None] * stride_a_scale_m +
                                       offs_a_scale_k[None, :] * stride_a_scale_k)

        # Check bounds for scale loading
        # Use other=127 for out-of-bounds E8M0 scales (127 = scale of 1.0)
        mask_a_scale_m = offs_am < M
        mask_a_scale_k = offs_a_scale_k < (K // VEC_SIZE)
        a_scale_mask = mask_a_scale_m[:, None] & mask_a_scale_k[None, :]
        a_scale_e8m0 = tl.load(a_scale_ptrs, mask=a_scale_mask, other=127)

        # With reversed selection, B now has columnwise scales [K//32, N]
        # For tl.dot_scaled we need [BLOCK_SIZE_K//32, BLOCK_SIZE_N]
        offs_b_scale_k = k_block_start + tl.arange(0, BLOCK_SIZE_K // VEC_SIZE)
        b_scale_ptrs = b_scale_ptr + (offs_b_scale_k[:, None] * stride_b_scale_k +
                                       offs_bn[None, :] * stride_b_scale_n)

        mask_b_scale_k = offs_b_scale_k < (K // VEC_SIZE)
        mask_b_scale_n = offs_bn < N
        b_scale_mask = mask_b_scale_k[:, None] & mask_b_scale_n[None, :]
        b_scale_e8m0 = tl.load(b_scale_ptrs, mask=b_scale_mask, other=127)

        # Block-scaled matmul using tl.dot_scaled
        # tl.dot_scaled expects E8M0 scales (uint8) and handles conversion internally
        # a: [BLOCK_SIZE_M, BLOCK_SIZE_K] FP8
        # a_scale_e8m0: [BLOCK_SIZE_M, BLOCK_SIZE_K // VEC_SIZE] uint8 (E8M0)
        # b: [BLOCK_SIZE_K, BLOCK_SIZE_N] FP8 (already in correct layout, no transpose needed)
        # b_scale_e8m0: [BLOCK_SIZE_K // VEC_SIZE, BLOCK_SIZE_N] uint8 (E8M0)
        accumulator = tl.dot_scaled(
            a,                # [BLOCK_SIZE_M, BLOCK_SIZE_K]
            a_scale_e8m0,     # [BLOCK_SIZE_M, BLOCK_SIZE_K // VEC_SIZE] E8M0
            FP8_FORMAT_A,     # "e4m3" or "e5m2"
            b,                # [BLOCK_SIZE_K, BLOCK_SIZE_N] - NO transpose
            b_scale_e8m0,     # [BLOCK_SIZE_K // VEC_SIZE, BLOCK_SIZE_N] E8M0
            FP8_FORMAT_B,     # "e4m3" or "e5m2"
            accumulator       # [BLOCK_SIZE_M, BLOCK_SIZE_N]
        )

        # Advance data pointers
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Store output (convert to target dtype)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    c = accumulator.to(c_ptr.type.element_ty)
    tl.store(c_ptrs, c, mask=c_mask)


def mxfp8_matmul(a, a_scale, b, b_scale, c, M, N, K, a_fp8_dtype, b_fp8_dtype):
    """
    MXFP8 matmul wrapper using tl.dot_scaled()

    Args:
        a: FP8 data tensor [M, K] (uint8)
        a_scale: E8M0 scale tensor [M, K//32] (uint8)
        b: FP8 data tensor [K, N] (uint8)
        b_scale: E8M0 scale tensor [K//32, N] (uint8)
        c: Output tensor [M, N] (fp32/bf16/fp16)
        M, N, K: Matrix dimensions
        a_fp8_dtype: FP8 dtype for A (tex.DType.kFloat8E4M3 or kFloat8E5M2)
        b_fp8_dtype: FP8 dtype for B
    """
    # Validate that a_scale and b_scale exist
    if a_scale is None or b_scale is None:
        raise RuntimeError("MXFP8 matmul requires both a_scale and b_scale to be provided")

    # Validate BLOCK_SIZE_K will be multiple of VEC_SIZE (32)
    # This is enforced by the autotune configs

    # Convert TE DType to Triton format string
    def te_dtype_to_triton_format(dtype):
        if dtype == tex.DType.kFloat8E4M3:
            return "e4m3"
        elif dtype == tex.DType.kFloat8E5M2:
            return "e5m2"
        else:
            raise ValueError(f"Unsupported FP8 dtype for MXFP8: {dtype}")

    fp8_format_a = te_dtype_to_triton_format(a_fp8_dtype)
    fp8_format_b = te_dtype_to_triton_format(b_fp8_dtype)

    # Launch kernel
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    mxfp8_matmul_kernel[grid](
        a, b, c,
        a_scale, b_scale,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        a_scale.stride(0), a_scale.stride(1),
        b_scale.stride(0), b_scale.stride(1),
        VEC_SIZE=MXFP8_BLOCK_SCALING_SIZE,  # 32
        FP8_FORMAT_A=fp8_format_a,
        FP8_FORMAT_B=fp8_format_b,
    )


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 4, 'waves_per_eu': 0}, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'waves_per_eu': 0}, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 4, 'waves_per_eu': 2}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2}, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 32, 'waves_per_eu': 2}, num_warps=4),
    ],
    # TODO: do we need to use different data types as key?
    key=['M', 'N', 'K'],
    # Ran into stream capture error when using cuda_graph, thus disabled.
    #use_cuda_graph=True,
    
)
@triton.heuristics({
    'EVEN_K': lambda args: args['K'] % args['BLOCK_SIZE_K'] == 0,
})
@triton.jit
def matmul_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Pointers to scales
        a_scale_ptr, b_scale_ptr, c_scale_ptr,
        # Pointer to bias
        bias_ptr,
        # Pointer to amax
        c_amax_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        EVEN_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        EPILOGUE: tl.constexpr,
        # Whether multiplied by scale_a * scale_b
        INPUT_FP8: tl.constexpr,
        # Whether to output fp8 or not, if so, also calculate amax.
        OUTPUT_FP8: tl.constexpr
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    M = blas_n, K = blas_k, N = blas_m
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetics` section for details
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    acc_dtype = tl.float32 if c_ptr.type.element_ty != tl.int8 else tl.int32
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

    if INPUT_FP8:
        a_scale = tl.load(a_scale_ptr)
        b_scale = tl.load(b_scale_ptr)
        scale = a_scale * b_scale 

    if OUTPUT_FP8:
        c_scale = tl.load(c_scale_ptr)

    if EPILOGUE == 'BGRADB' and not INPUT_FP8:
        bias_gradient = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)

        if EPILOGUE == 'BGRADB' and not INPUT_FP8:
            if pid_n == 0:
                ## It is necessary to upcast to fp32 for reduction to ensure accuracy.
                bias_gradient_partial = tl.sum(a.to(tl.float32), axis=1)
                bias_gradient += bias_gradient_partial

        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk


    if EPILOGUE == 'BGRADB' and not INPUT_FP8:
        if pid_n == 0:
            offs_bias_gradient = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            bias_gradient_ptrs = bias_ptr + offs_bias_gradient
            ## Though bias_gradient is fp32, type conversion will occur before store
            tl.store(bias_gradient_ptrs, bias_gradient, mask=(offs_bias_gradient<M))

    if INPUT_FP8:
        accumulator *= scale
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    if EPILOGUE == 'BIAS':
        offs_bias = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N) 
        bias_ptrs = bias_ptr + offs_bias
        bias = tl.load(bias_ptrs, mask=(offs_bias < N), other=0.0).to(tl.float32)
        accumulator = accumulator + bias[None, :]


    # Get amax first and then scale c before conversion to fp8
    if OUTPUT_FP8:
        tile_c_amax = tl.max(tl.abs(accumulator))
        tl.atomic_max(c_amax_ptr, tile_c_amax)
        c = (accumulator * c_scale).to(c_ptr.type.element_ty)
    else:
        c = accumulator.to(c_ptr.type.element_ty)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# %%
# We can now create a convenience wrapper function that only takes two input tensors,
# and (1) checks any shape constraint; (2) allocates the output; (3) launches the above kernel.
def matmul(a, b, c, a_scale, b_scale, c_scale, bias, c_amax, epilogue='DEFAULT', input_fp8=False, output_fp8=False):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    M, K = a.shape
    K, N = b.shape

    if c_amax is not None:
        c_amax.zero_()

    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel[grid](
        a, b, c,
        a_scale, b_scale, c_scale,
        bias,
        c_amax,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        EPILOGUE=epilogue,
        INPUT_FP8=input_fp8,
        OUTPUT_FP8=output_fp8
    )


