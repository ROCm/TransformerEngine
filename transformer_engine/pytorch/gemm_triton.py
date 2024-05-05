# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#
# License for AMD contributions = MIT. See LICENSE for more information

from enum import IntEnum
import torch

import transformer_engine_extensions as tex

import triton
import triton.language as tl

def torch_to_te_dtype(dtype):
    torch_to_TE_dtypes = {
        torch.int8: tex.DType.kByte,
        torch.int32: tex.DType.kInt32,
        torch.float32: tex.DType.kFloat32,
        torch.float16: tex.DType.kFloat16,
        torch.bfloat16: tex.DType.kBFloat16,
        torch.float8_e4m3fnuz: tex.DType.kFloat8E4M3,
        torch.float8_e5m2fnuz: tex.DType.kFloat8E5M2,
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
    if dtype == tex.DType.kFloat8E4M3:
        return a.view(dtype=torch.float8_e4m3fnuz)
    if dtype == tex.DType.kFloat8E5M2:
        return a.view(dtype=torch.float8_e5m2fnuz)

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

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 4, 'waves_per_eu': 0}, num_warps=8, num_stages=0),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'waves_per_eu': 0}, num_warps=8, num_stages=0),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 4, 'waves_per_eu': 2}, num_warps=4, num_stages=0),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2}, num_warps=8, num_stages=0),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 32, 'waves_per_eu': 2}, num_warps=4, num_stages=0),
    ],
    # TODO: do we need to use different data types as key?
    key=['M', 'N', 'K'],
    use_cuda_graph=True,
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


