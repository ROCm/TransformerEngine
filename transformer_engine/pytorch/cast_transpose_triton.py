import torch

import transformer_engine_torch as tex
import triton
import triton.language as tl

def is_fp8_dtype(dtype):
    return dtype in (tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2)

def get_triton_dtype(dtype: tex.DType):
    if dtype == tex.DType.kFloat8E4M3:
        return tl.float8e4b8
    if dtype == tex.DType.kFloat8E5M2:
        return tl.float8e5b16

def get_te_dtype(dtype):
    if dtype == torch.float8_e4m3fnuz:
        return tex.DType.kFloat8E4M3
    if dtype == torch.float8_e5m2fnuz:
        return tex.DType.kFloat8E5M2

def get_fp8_max(dtype: tex.DType):
    if dtype == tex.DType.kFloat8E4M3:
        return 240.0
    if dtype == tex.DType.kFloat8E5M2:
        return 57344.0

##########################################
#### cast_transpose
##########################################
def get_hip_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'waves_per_eu': 2}, num_warps=4,num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'waves_per_eu': 2}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'waves_per_eu': 2}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'waves_per_eu': 3}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'waves_per_eu': 8}, num_warps=4, num_stages=2),
        
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'waves_per_eu': 2}, num_warps=8, num_stages=3),            
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'waves_per_eu': 2}, num_warps=8, num_stages=3),         
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'waves_per_eu': 4}, num_warps=4, num_stages=4),          
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'waves_per_eu': 2}, num_warps=4, num_stages=4),            
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'waves_per_eu': 2}, num_warps=4, num_stages=4),           
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'waves_per_eu': 3}, num_warps=4, num_stages=4),          
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'waves_per_eu': 3}, num_warps=4, num_stages=4),    
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'waves_per_eu': 8}, num_warps=2, num_stages=2),       
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'waves_per_eu': 3}, num_warps=1, num_stages=8),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16, 'waves_per_eu': 8}, num_warps=1, num_stages=8),


        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'waves_per_eu': 8}, num_warps=8, num_stages=8),                 
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'waves_per_eu': 8}, num_warps=8, num_stages=8),           
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'waves_per_eu': 8}, num_warps=8, num_stages=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'waves_per_eu': 3}, num_warps=8, num_stages=8), 
        triton.Config({'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 8,  'waves_per_eu': 8}, num_warps=8, num_stages=8),
        triton.Config({'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 8,  'waves_per_eu': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 8,  'waves_per_eu': 8}, num_warps=1, num_stages=4),
    ]

def get_autotune_config():
    #if is_cuda():
    #    return get_cuda_autotune_config()
    #else:
    return get_hip_autotune_config()


# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
#   - A list of `triton.Config` objects that define different configurations of
#       meta-parameters (e.g., `BLOCK_SIZE_M`) and compilation options (e.g., `num_warps`) to try
#   - An auto-tuning *key* whose change in values will trigger evaluation of all the
#       provided configs
@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N'],
)  
 


@triton.jit
def cast_transpose_kernel(
        noop_flag,
        input_ptr,
        scale_ptr,
        output_c_ptr,
        output_t_ptr,
        amax_ptr,
        M, N,
        stride_input,
        stride_output_c,
        stride_output_t,
        fp8_max: tl.constexpr, 
        use_noop: tl.constexpr, 
        BLOCK_SIZE_M: tl.constexpr, 
        BLOCK_SIZE_N: tl.constexpr
):
    """Kernel for casting and transposing a tensor."""
    if use_noop:
        noop = tl.load(noop_ptr)
        if noop == 1.0:
            return

    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    
    offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    mask_m = offsets_m[:, None] < M
    mask_n = offsets_n[None, :] < N

    mask_tm = offsets_m[None, :] < M
    mask_tn = offsets_n[:, None] < N

    input_ptrs = input_ptr + offsets_m[:, None] * stride_input + offsets_n[None, :] 
    
    input_ = tl.load(input_ptrs, mask=mask_m & mask_n)
    scale = tl.load(scale_ptr)
    input_ = input_.to(tl.float32)
    output_c = input_ * scale
   
    output_c = tl.clamp(output_c, -fp8_max, fp8_max)
    output_c = tl.cast(output_c, output_c_ptr.type.element_ty) 
    tl.store(output_c_ptr + offsets_m[:, None] * stride_output_c + offsets_n[None, :], output_c, mask=mask_m & mask_n)

    output_t = output_c.trans()
   
    tl.store(output_t_ptr + offsets_n[:, None] * stride_output_t + offsets_m[None, :] , output_t, mask=mask_tn & mask_tm)
    amax = tl.max(tl.abs(input_))
    tl.atomic_max(amax_ptr, amax, sem='relaxed')
    #tl.store(output_t_ptr + offsets_n[None, :] * stride_output_t + offsets_m[:, None] , output_c, mask=mask_n & mask_m)
    
dtype_map = {
        torch.float32: tl.float32,
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.float8_e5m2fnuz: tl.float8e5b16,
        torch.float8_e4m3fnuz: tl.float8e4b8
    }

def te_cast_transpose_noop_triton(input, noop_flag, input_scale, cast_out, trans_out, amax_out, otype):

    M, N = input.shape
    
    tl_dtype = get_triton_dtype(otype)
    
    assert trans_out.size(0) == N and trans_out.size(1) == M

    if noop_flag.nelement() > 0:
        use_noop = True
    else:
        use_noop = False
    
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']))
    cast_transpose_kernel[grid](noop_flag, input, input_scale, triton.reinterpret(cast_out, tl_dtype), triton.reinterpret(trans_out, tl_dtype), amax_out, M, N, input.stride(0), cast_out.stride(0), trans_out.stride(0), get_fp8_max(otype), use_noop)
    #grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    #_cast_transpose_triton[grid](input, noop_flag, triton.reinterpret(cast_out, tl_dtype), triton.reinterpret(trans_out, tl_dtype), input.stride(0), input.stride(1), trans_out.stride(0), trans_out.stride(1), M, N, input_scale, amax_out, get_fp8_max(otype), use_noop)
  
'''
@triton.autotune(
        configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'GROUP_M': 1}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'GROUP_M': 8}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'GROUP_M': 8}, num_warps=8),
        ],
        key=['M', 'N']
)
@triton.jit
def _cast_transpose_triton(A, noop_ptr, C, T, stride_am, stride_an, stride_bn, stride_bm, M, N, scale_ptr, amax_ptr, max_fp8: tl.constexpr, use_noop: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, GROUP_M: tl.constexpr):
    if use_noop:
        noop = tl.load(noop_ptr)
        if noop == 1.0:
            return

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

def te_cast_transpose_noop_triton(input, noop_flag, input_scale, cast_out, trans_out, amax_out, otype):

    M, N = input.shape
    
    tl_dtype = get_triton_dtype(otype)
    
    assert trans_out.size(0) == N and trans_out.size(1) == M

    if noop_flag.nelement() > 0:
        use_noop = True
    else:
        use_noop = False
    
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    _cast_transpose_triton[grid](input, noop_flag, triton.reinterpret(cast_out, tl_dtype), triton.reinterpret(trans_out, tl_dtype), input.stride(0), input.stride(1), trans_out.stride(0), trans_out.stride(1), M, N, input_scale, amax_out, get_fp8_max(otype), use_noop)
'''

##########################################
#### cast_transpose_dbias
##########################################
@triton.autotune(
        configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'GROUP_M': 1}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'GROUP_M': 8}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'GROUP_M': 8}, num_warps=8),
        ],
        key=['M', 'N']
)
@triton.jit
def _transpose_triton_dbias(A, C, T, stride_am, stride_an, stride_bn, stride_bm, M, N, scale_ptr, amax_ptr, partial_dbias, fp8_max: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, GROUP_M: tl.constexpr):
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
    a = tl.load(A, mask=mask, other=0.)
    a = a.to(tl.float32)

    partial_sum_a = tl.sum(a, axis=0)
    partial_dbias = partial_dbias + pid_m * stride_am + rn * stride_an
    tl.store(partial_dbias, partial_sum_a, mask=(rn<N))

    scaled_a = a * scale
    scaled_a = tl.clamp(scaled_a, -fp8_max, fp8_max)
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

# There is a Triton bug that makes this kernel produce incorrect result
# Not in use for now
@triton.autotune(
        configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64 }, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=1),
        ],
        key=['M', 'N']
)
@triton.jit
def _reduce_bias_triton(A, out, stride_am, stride_an, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    iters_m = (M + BLOCK_M - 1) // BLOCK_M
    rn = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    dbias_reg = tl.zeros((BLOCK_N,), tl.float32)
    rm = tl.arange(0, BLOCK_M)
    for i in range(iters_m):
        #rm = i * BLOCK_M + tl.arange(0, BLOCK_M)
        A_ptr = A + (rm[:, None] * stride_am + rn[None, :] * stride_an)
        mask = (rm < M)[:, None] & (rn < N)[None, :]
        a = tl.load(A_ptr, mask=mask, other=0.)
        dbias_reg += tl.sum(a, axis=0)
        rm += BLOCK_M

    dbias_reg = dbias_reg.to(out.type.element_ty)
    out = out + rn * stride_an
    tl.store(out, dbias_reg, mask=(rn<N))

@torch.compile
def reduce_dbias_kernel(partial_dbias, dtype):
    return partial_dbias.to(torch.float32).sum(axis=0).to(dtype)

def te_cast_transpose_dbias_triton(input, input_scale, amax_out, otype):
    M, N = input.shape
    cast_out = torch.empty(M, N, dtype=torch.uint8, device='cuda')
    trans_out = torch.empty(N, M, dtype=torch.uint8, device='cuda')
    dbias_out = torch.empty(N, dtype=input.dtype, device='cuda')

    if M == 0 or N == 0:
        return dbias_out, cast_out, trans_out

    MIN_BLOCK_M = 64 ## This needs to be changed  
    partial_dbias = torch.empty(triton.cdiv(M, MIN_BLOCK_M), N, dtype=torch.float32, device='cuda')

    assert trans_out.size(0) == N and trans_out.size(1) == M
    assert input.stride(0) == 1 or input.stride(1) == 1
    assert trans_out.stride(0) == 1 or trans_out.stride(1) == 1

    tl_dtype = get_triton_dtype(otype)
    
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    _transpose_triton_dbias[grid](input, triton.reinterpret(cast_out, tl_dtype), triton.reinterpret(trans_out, tl_dtype), input.stride(0), input.stride(1), trans_out.stride(0), trans_out.stride(1), M, N, input_scale, amax_out, partial_dbias, get_fp8_max(otype))
    best_config = _transpose_triton_dbias.best_config
    #print('best_config=', best_config)
    block_m_1 = int(best_config.kwargs['BLOCK_M'])

    grid2 = lambda META: (triton.cdiv(N, META['BLOCK_N']),)
    _reduce_bias_triton[grid2](partial_dbias, dbias_out, partial_dbias.stride(0), partial_dbias.stride(1), triton.cdiv(M, block_m_1), N)
    dbias_out = reduce_dbias_kernel(partial_dbias[0:triton.cdiv(M, block_m_1)], input.dtype)
    #dbias_out = partial_dbias[0:triton.cdiv(M, block_m_1)].to(torch.float32).sum(axis=0).to(torch.float16)
    return dbias_out, cast_out, trans_out

    