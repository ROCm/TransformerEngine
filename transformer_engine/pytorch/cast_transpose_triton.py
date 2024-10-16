import torch

import transformer_engine_extensions as tex
import triton
import triton.language as tl

def is_fp8_dtype(dtype):
    return dtype in (tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2)

def get_triton_dtype(dtype: tex.DType):
    if dtype == tex.DType.kFloat8E4M3:
        return tl.float8e4b8
    if dtype == tex.DType.kFloat8E5M2:
        return tl.float8e5b16

def get_fp8_max(dtype):
    if dtype == tex.DType.kFloat8E4M3:
        return 240.0
    if dtype == tex.DType.kFloat8E5M2:
        return 57344.0

'''
@triton.autotune(
        configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'GROUP_M': 1}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'GROUP_M': 8}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'GROUP_M': 8}, num_warps=8),
        ],
        key=['M', 'N']
)
'''
@triton.jit
def _transpose_triton(A, C, T, stride_am, stride_an, stride_bn, stride_bm, M, N, scale_ptr, amax_ptr, max_fp8: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    scale = tl.load(scale_ptr)

    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size
    
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    A = A + (rm[:, None] * stride_am + rn[None, :] * stride_an)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    a = tl.load(A, mask=mask)
    a = a.to(tl.float32)

    scaled_a = a * scale
    scaled_a = tl.clamp(scaled_a, -max_fp8, max_fp8)
    fp8_a = scaled_a.to(C.type.element_ty)
    C = C + (rm[:, None] * stride_am + rn[None, :] * stride_an)
    tl.store(C, fp8_a, mask=mask)
    
    # rematerialize to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    T = T + (rm[:, None] * stride_bm + rn[None, :] * stride_bn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    tl.store(T, fp8_a, mask=mask)

    amax = tl.max(tl.abs(a))
    tl.atomic_max(amax_ptr, amax, sem='relaxed')

def te_cast_transpose_triton(input, input_scale, cast_out, trans_out, amax_out, otype):

    M, N = input.shape
    
    tl_dtype = get_triton_dtype(otype)
    
    assert trans_out.size(0) == N and trans_out.size(1) == M
    
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    _transpose_triton[grid](input, triton.reinterpret(cast_out, tl_dtype), triton.reinterpret(trans_out, tl_dtype), input.stride(0), input.stride(1), trans_out.stride(0), trans_out.stride(1), M, N, input_scale, amax_out, get_fp8_max(otype), BLOCK_M=128, BLOCK_N=128, GROUP_M=8, num_warps=8)

