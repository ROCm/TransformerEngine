# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Python wrappers and TE-shaped entry points for the Triton GEMM backend.

- ``matmul``: thin wrapper around the FP32/FP16/BF16/FP8 kernel
- ``mxfp8_matmul``: thin wrapper around the MXFP8 (tl.dot_scaled) kernel
- ``te_gemm_triton``: low-level TE-style entry point (BLAS-shaped args)
- ``te_generic_gemm_triton``: high-level entry point that detects
  Float8Tensor / MXFP8Tensor inputs and dispatches to the right kernel;
  used by ``cpp_extensions.gemm.general_gemm`` when NVTE_USE_GEMM_TRITON=1
"""

import torch

import transformer_engine_torch as tex
from transformer_engine.pytorch.constants import MXFP8_BLOCK_SCALING_SIZE

import triton

from ..common import torch_dtype_to_te_dtype, te_dtype_to_torch_dtype
from .gemm_kernels import matmul_kernel, mxfp8_matmul_kernel
from .gemm_common import (
    Float8TensorWrapper,
    MXFP8TensorWrapper,
    is_fp8_dtype,
    reinterpret_as_fp8_tensor,
    getGemmOutputShape,
    product,
)


# %%
# We can now create a convenience wrapper function that only takes two input tensors,
# and (1) checks any shape constraint; (2) allocates the output; (3) launches the above kernel.
def matmul(a, b, c, a_scale, b_scale, c_scale, bias, c_amax, epilogue='DEFAULT', input_fp8=False, output_fp8=False, accumulate=False, alpha=1.0, beta=0.0):
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
        float(alpha), float(beta),
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        EPILOGUE=epilogue,
        INPUT_FP8=input_fp8,
        OUTPUT_FP8=output_fp8,
        ACCUMULATE=accumulate,
        ALPHA_IS_ONE=(float(alpha) == 1.0),
    )


def mxfp8_matmul(a, a_scale, b, b_scale, c, M, N, K, a_fp8_dtype, b_fp8_dtype):
    """
    MXFP8 matmul wrapper using tl.dot_scaled()

    Args:
        a: FP8 data tensor [M, K] (uint8)
        a_scale: E8M0 scale tensor [M, K//32] (uint8)
        b: FP8 data tensor [K, N] (uint8)
        b_scale: E8M0 scale tensor [K//32, N] (uint8) -- will be transposed
                 to [N, K//32] internally for the new dot_scaled API
        c: Output tensor [M, N] (fp32/bf16/fp16)
        M, N, K: Matrix dimensions
        a_fp8_dtype: FP8 dtype for A (tex.DType.kFloat8E4M3 or kFloat8E5M2)
        b_fp8_dtype: FP8 dtype for B
    """
    # Validate that a_scale and b_scale exist
    if a_scale is None or b_scale is None:
        raise RuntimeError("MXFP8 matmul requires both a_scale and b_scale to be provided")

    # Transpose b_scale from [K//32, N] to [N, K//32] for new dot_scaled API
    # The new API expects rhs_scale in [N, K//32] layout (NOT transposed)
    b_scale = b_scale.T.contiguous()

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
    assert te_dtype_to_torch_dtype(A_type) == A.dtype, 'A dtype does not match.'
    assert te_dtype_to_torch_dtype(B_type) == B.dtype, 'B dtype does not match.'
    assert te_dtype_to_torch_dtype(D_type) == D.dtype, 'D dtype does not match.'
    assert (bias.data_ptr() == 0) or (te_dtype_to_torch_dtype(bias_type) == bias.dtype), 'bias dtype does not match.'


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
    matmul(a_row_major, b_row_major, D, a_scale_triton, b_scale_triton, D_scale, bias, D_amax, epilogue, input_fp8, output_fp8, accumulate=accumulate)
    # (te_gemm_triton low-level path has no alpha/beta in its signature; callers
    # wanting fused α/β should use te_generic_gemm_triton via general_gemm().)


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
                            bulk_overlap,
                            alpha=1.0,
                            beta=0.0):

    # Wrap inputs to handle Float8Tensor and MXFP8Tensor uniformly
    # Try MXFP8 first, then Float8, then regular
    try:
        from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor
        from transformer_engine.pytorch.tensor.storage.mxfp8_tensor_storage import MXFP8TensorStorage
        is_mxfp8_a = isinstance(A, (MXFP8Tensor, MXFP8TensorStorage))
        is_mxfp8_b = isinstance(B, (MXFP8Tensor, MXFP8TensorStorage))
    except ImportError:
        is_mxfp8_a = False
        is_mxfp8_b = False

    if is_mxfp8_a or is_mxfp8_b:
        # MXFP8 Triton GEMM requires PyTorch >= 2.10 which ships a Triton version
        # with the tl.dot_scaled() RHS scale layout bug fix.
        # Earlier versions silently produce wrong results for non-uniform B scales.
        from transformer_engine.pytorch import torch_version
        if torch_version() < (2, 10):
            raise RuntimeError(
                f"Triton MXFP8 GEMM requires PyTorch >= 2.10 (found {torch_version()}). "
                "Earlier versions contain a Triton compiler bug in tl.dot_scaled() that "
                "produces incorrect results for the RHS scale operand. "
                "Set NVTE_USE_GEMM_TRITON=0 to use the C++ GEMM backend instead."
            )

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

    # Mixed FP8 types (e.g. A=e4m3, B=e5m2) are not supported due to a Triton
    # compiler bug: when the MFMA layout is transposed, operand B is packed using
    # A's element type, and the instruction's format encoding doesn't account for
    # the operand swap. This produces silently wrong results for all MFMA variants.
    # Fixed upstream in triton-lang/triton PR #9567 (commit eaaa75cf5, 2026-02-27).
    # Not yet included in any pytorch-triton-rocm release as of PyTorch 2.11.
    # Expected in PyTorch 2.12+ once the Triton pin is bumped.
    # TODO: Remove this guard once pytorch-triton-rocm includes the fix.
    if (a_fp8_dtype is not None and b_fp8_dtype is not None
            and a_fp8_dtype != b_fp8_dtype):
        raise ValueError(
            f"Mixed FP8 types (A={a_fp8_dtype}, B={b_fp8_dtype}) are not supported "
            f"in the Triton GEMM backend due to a Triton compiler bug "
            f"(triton-lang/triton#9567). Use the same FP8 format for both operands, "
            f"or disable the Triton backend (unset NVTE_USE_GEMM_TRITON)."
        )

    # Reinterpret uint8 as native FP8 types for Triton
    # The FP8 tensor data is stored as torch.uint8 but Triton needs torch.float8_e4m3fnuz
    if a_fp8_dtype is not None:
        A_data = reinterpret_as_fp8_tensor(A_data, a_fp8_dtype)
    if b_fp8_dtype is not None:
        B_data = reinterpret_as_fp8_tensor(B_data, b_fp8_dtype)

    # Compute dimensions using wrapper sizes
    # Wrapper handles Float8TensorStorage which doesn't have .shape attribute
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

    # Bias / bias-gradient wiring for the regular/FP8 matmul kernel.
    # MXFP8 takes a separate path below and does not yet support bias.
    has_bias = bias is not None and bias.numel() > 0
    if has_bias and grad:
        epilogue = 'BGRADB'
        bias_grad = torch.empty_like(bias)
    elif has_bias:
        epilogue = 'BIAS'
        bias_grad = None
    else:
        epilogue = 'DEFAULT'
        bias_grad = None

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
            out_dtype = te_dtype_to_torch_dtype(output_dtype)
        elif hasattr(A_wrapper, 'is_mxfp8') and A_wrapper.is_mxfp8:
            # MXFP8 input: use nominal dtype
            out_dtype = A_wrapper.nominal_dtype
        elif hasattr(A_wrapper, 'is_fp8') and A_wrapper.is_fp8:
            # Regular FP8 input: use nominal dtype if available
            if A_wrapper.nominal_dtype is None:
                raise RuntimeError(
                    "FP8 input detected (Float8TensorStorage without nominal dtype) but output_dtype "
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
    D_amax = torch.Tensor()
    # Bias tensor passed to the kernel:
    #   - BIAS:   the actual bias, read and added to output
    #   - BGRADB: output buffer that receives the bias gradient
    #   - DEFAULT: empty (kernel ignores it)
    if epilogue == 'BIAS':
        bias_tensor = bias
    elif epilogue == 'BGRADB':
        bias_tensor = bias_grad
    else:
        bias_tensor = torch.Tensor()

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
               D_scale, bias_tensor, D_amax, epilogue, input_fp8, output_fp8,
               accumulate=accumulate, alpha=alpha, beta=beta)

    # Fused FP8 output quantization is not wired through this wrapper (see the
    # `output_fp8 = False` above): the kernel produces D in `output_dtype`
    # (typically fp32/bf16) rather than writing FP8 with scale + amax directly.
    # When the caller passed a quantizer, apply it here so the returned tensor
    # is the Float8Tensor / MXFP8Tensor the caller expects. This matches what
    # the hipBLASLt fused path produces bit-for-bit: same fp32 accumulator ->
    # same quantizer -> same FP8 payload. Skip DebugQuantizer because
    # general_gemm strips it out before dispatching to us.
    if quantizer is not None:
        D = quantizer(D)

    return D, bias_grad, None, None
