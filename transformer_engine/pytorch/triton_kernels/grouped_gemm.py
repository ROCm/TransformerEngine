from transformer_engine.pytorch.triton_kernels.common import torch_dtype_to_triton_dtype
import triton
import triton.language as tl
import torch
from typing import Optional
from triton.runtime import driver

def num_sms():
    """Get the number of streaming multiprocessors/compute units on current device"""
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    return 304  # Default for MI300X

# NVIDIA-optimized configurations for grouped GEMM
_NV_CONFIGS = [
    triton.Config(
        {
            "BLOCK_SIZE_M": block_size_m,
            "BLOCK_SIZE_N": block_size_n,
            "BLOCK_SIZE_K": block_size_k,
            "NUM_SM": num_sm,
        },
        num_stages=num_stages,
        num_warps=num_warps,
        num_ctas=num_ctas,
    )
    for block_size_m in [64, 128]
    for block_size_n in [64, 128, 256]
    for block_size_k in [64, 128, 256]
    for num_stages in [3, 4]
    for num_warps in [4, 8]
    for num_sm in [84, 128]
    for num_ctas in [1]
]

# AMD-optimized configurations for grouped GEMM
_AMD_CONFIGS = [
    triton.Config(
        {
            "BLOCK_SIZE_M": block_size_m,
            "BLOCK_SIZE_N": block_size_n,
            "BLOCK_SIZE_K": block_size_k,
            "NUM_SM": num_sm,
            "waves_per_eu": waves_per_eu,
            "matrix_instr_nonkdim": matrix_instr_nonkdim,
        },
        num_stages=num_stages,
        num_warps=num_warps,
    )
    for block_size_m in [32, 64, 128]
    for block_size_n in [32, 64, 128, 256]
    for block_size_k in [128, 256]
    for num_stages in [1, 2]
    for num_warps, waves_per_eu in [(4, 1), (8, 2), (16, 4)]
    for num_sm in [304]  # MI300X
    for matrix_instr_nonkdim in [16]
]

def early_config_prune(configs, named_args, dtsize=None, dtype=None, **kwargs):
    """Prune configurations that are invalid or inefficient"""
    device = torch.cuda.current_device()
    
    # Infer dtsize if not provided
    if dtsize is None:
        dtsize = 2  # float16/bfloat16 default
    
    pruned_configs = []
    for config in configs:
        kw = config.kwargs
        BLOCK_M = kw["BLOCK_SIZE_M"]
        BLOCK_N = kw["BLOCK_SIZE_N"]
        BLOCK_K = kw["BLOCK_SIZE_K"]
        num_stages = config.num_stages
        
        # Get group size (number of experts)
        G = named_args.get("group_size", 64)
        
        # Get device properties
        props = driver.active.utils.get_device_properties(device)
        max_shared_memory = props["max_shared_mem"]
        num_sm = props["multiprocessor_count"]
        
        # 1. Make sure we have enough shared memory
        if torch.version.hip:
            required_shared_memory = BLOCK_N * BLOCK_K * num_stages * dtsize
        else:
            required_shared_memory = (BLOCK_M + BLOCK_N) * BLOCK_K * num_stages * dtsize
        
        if required_shared_memory > max_shared_memory:
            continue
        
        # 2. Estimate average M per group (tokens per expert)
        # Assume roughly uniform distribution
        M_PER_GROUP = 24576 // G  # Conservative estimate based on typical workload
        
        MIN_M_TILES = 32 if torch.version.hip else 64
        
        # Don't load M tiles that are too big
        if BLOCK_M > MIN_M_TILES and BLOCK_M > (M_PER_GROUP * 2):
            continue
        
        # Don't load M tiles that are too small
        if BLOCK_M < 128 and BLOCK_M < (M_PER_GROUP // 2):
            continue
        
        # 3. Estimate N (output features) - typically 2816 for this workload
        N = 2816
        N_TILES = N // BLOCK_N if BLOCK_N > 0 else 1
        
        MIN_N_TILES = 32 if torch.version.hip else 64
        
        # Don't load N tiles that are too big
        if BLOCK_N > MIN_N_TILES and M_PER_GROUP * N_TILES < num_sm:
            continue
        
        # Don't load N tiles that are too small
        if BLOCK_N < 128 and M_PER_GROUP * N_TILES > 2 * num_sm:
            continue
        
        # 4. Make sure K can be evenly divided (typical K is 2048)
        # This is less strict - we can handle misalignment but prefer even division
        K = 2048
        if K % BLOCK_K != 0 and BLOCK_K > 64:
            continue
        
        pruned_configs.append(config)
    
    return pruned_configs

@triton.autotune(
    configs=_AMD_CONFIGS if torch.version.hip else _NV_CONFIGS,
    key=['group_size'],
    prune_configs_by={'early_config_prune': early_config_prune},
)
@triton.jit
def _kernel_grouped_gemm_forward(
    # device tensor of matrices pointers
    group_a_ptrs,
    group_b_ptrs,
    group_c_ptrs,
    # device tensor of gemm sizes. its shape is [group_size, 3]
    # dim 0 is group_size, dim 1 is the values of <M, N, K> of each gemm
    group_gemm_sizes,
    # device tensor of leading dimension sizes. its shape is [group_size, 3]
    # dim 0 is group_size, dim 1 is the values of <lda, ldb, ldc> of each gemm
    g_lds,
    # number of gemms
    group_size,
    # number of virtual SM
    NUM_SM: tl.constexpr,
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    DTYPE: tl.constexpr,
):
    tile_idx = tl.program_id(0)
    last_problem_end = 0
    for g in range(group_size):
        # get the gemm size of the current problem (use element-based indexing)
        gm = tl.load(group_gemm_sizes + g * 3)
        gn = tl.load(group_gemm_sizes + g * 3 + 1)
        gk = tl.load(group_gemm_sizes + g * 3 + 2)
        num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
        num_tiles = num_m_tiles * num_n_tiles
        # iterate through the tiles in the current gemm problem
        while (tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles):
            # pick up a tile from the current gemm problem
            k = gk
            # Load leading dimensions (element-based indexing)
            lda = tl.load(g_lds + g * 3)
            ldb = tl.load(g_lds + g * 3 + 1)
            ldc = tl.load(g_lds + g * 3 + 2)
            # Load pointers and cast to pointer type (matching tutorial)
            a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(DTYPE))
            b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(DTYPE))
            c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(DTYPE))
            # figure out tile coordinates
            tile_idx_in_gemm = tile_idx - last_problem_end
            tile_m_idx = tile_idx_in_gemm // num_n_tiles
            tile_n_idx = tile_idx_in_gemm % num_n_tiles

            # do regular gemm here
            offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            offs_k = tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + offs_am[:, None] * lda + offs_k[None, :]
            # Weight is [N, K], so to get W^T for gemm, load with transposed indices
            b_ptrs = b_ptr + offs_bn[:, None] * ldb + offs_k[None, :]
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for kk in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
                # hint to Triton compiler to do proper loop pipelining
                tl.multiple_of(a_ptrs, [16, 16])
                tl.multiple_of(b_ptrs, [16, 16])
                # Add masking for partial tiles
                k_remaining = k - kk * BLOCK_SIZE_K
                mask_a = (offs_am[:, None] < gm) & (offs_k[None, :] < k_remaining)
                # B is [N, K] layout, so mask is [N, K]
                mask_b = (offs_bn[:, None] < gn) & (offs_k[None, :] < k_remaining)
                a = tl.load(a_ptrs, mask=mask_a, other=0.0)
                b = tl.load(b_ptrs, mask=mask_b, other=0.0)
                # B is loaded as [N, K], transpose it to [K, N] for dot product
                accumulator += tl.dot(a, b.T)
                a_ptrs += BLOCK_SIZE_K
                # B is [N, K], so increment along K dimension (columns)
                b_ptrs += BLOCK_SIZE_K
            c = accumulator

            offs_cm = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_ptr + ldc * offs_cm[:, None] + offs_cn[None, :]

            # Add masking for output stores
            mask_c = (offs_cm[:, None] < gm) & (offs_cn[None, :] < gn)
            tl.store(c_ptrs, c, mask=mask_c)

            # go to the next tile by advancing NUM_SM
            tile_idx += NUM_SM

        # get ready to go to the next gemm problem
        last_problem_end = last_problem_end + num_tiles

@triton.autotune(
    configs=_AMD_CONFIGS if torch.version.hip else _NV_CONFIGS,
    key=['group_size'],
    prune_configs_by={'early_config_prune': early_config_prune},
)
@triton.jit
def _kernel_grouped_gemm_backward(
    # device tensor of matrices pointers
    group_a_ptrs,
    group_b_ptrs,
    group_c_ptrs,
    # device tensor of gemm sizes. its shape is [group_size, 3]
    # dim 0 is group_size, dim 1 is the values of <M, K, N> of each gemm
    # Note: For backward, sizes are [M, K, N] where N is reduction dimension
    group_gemm_sizes,
    # device tensor of leading dimension sizes. its shape is [group_size, 3]
    # dim 0 is group_size, dim 1 is the values of <lda, ldb, ldc> of each gemm
    g_lds,
    # number of gemms
    group_size,
    # number of virtual SM
    NUM_SM: tl.constexpr,
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """
    Backward pass kernel for dgrad: dgrad = grad_output @ weight
    - grad_output: [M, N] where N = out_features
    - weight: [N, K] where K = in_features
    - dgrad: [M, K]
    Reduction is over N dimension.
    """
    tile_idx = tl.program_id(0)
    last_problem_end = 0
    for g in range(group_size):
        # get the gemm size of the current problem (use element-based indexing)
        gm = tl.load(group_gemm_sizes + g * 3)  # M
        gk = tl.load(group_gemm_sizes + g * 3 + 1)  # K (out dimension)
        gn = tl.load(group_gemm_sizes + g * 3 + 2)  # N (reduction dimension)
        num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
        num_k_tiles = tl.cdiv(gk, BLOCK_SIZE_K)
        num_tiles = num_m_tiles * num_k_tiles
        # iterate through the tiles in the current gemm problem
        while (tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles):
            # pick up a tile from the current gemm problem
            n = gn  # reduction dimension
            # Load leading dimensions (element-based indexing)
            lda = tl.load(g_lds + g * 3)      # grad_output stride
            ldb = tl.load(g_lds + g * 3 + 1)  # weight stride
            ldc = tl.load(g_lds + g * 3 + 2)  # dgrad stride
            # Load pointers and cast to pointer type (matching tutorial)
            a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(DTYPE))  # grad_output
            b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(DTYPE))  # weight
            c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(DTYPE))  # dgrad
            # figure out tile coordinates
            tile_idx_in_gemm = tile_idx - last_problem_end
            tile_m_idx = tile_idx_in_gemm // num_k_tiles
            tile_k_idx = tile_idx_in_gemm % num_k_tiles

            # do regular gemm here (same structure as forward, but reduce over N)
            offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_ck = tile_k_idx * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            offs_n = tl.arange(0, BLOCK_SIZE_N)
            a_ptrs = a_ptr + offs_am[:, None] * lda + offs_n[None, :]
            # Weight is [N, K], load with N and K indices
            b_ptrs = b_ptr + offs_n[:, None] * ldb + offs_ck[None, :]
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
            for n_idx in range(0, tl.cdiv(n, BLOCK_SIZE_N)):
                # hint to Triton compiler to do proper loop pipelining
                tl.multiple_of(a_ptrs, [16, 16])
                tl.multiple_of(b_ptrs, [16, 16])
                # Add masking for partial tiles
                n_remaining = n - n_idx * BLOCK_SIZE_N
                mask_a = (offs_am[:, None] < gm) & (offs_n[None, :] < n_remaining)
                # B is [N, K] layout, so mask is [N, K]
                mask_b = (offs_n[:, None] < gn) & (offs_ck[None, :] < gk)
                a = tl.load(a_ptrs, mask=mask_a, other=0.0)
                b = tl.load(b_ptrs, mask=mask_b, other=0.0)
                # Only difference from forward: no transpose (a @ b instead of a @ b.T)
                accumulator += tl.dot(a, b)
                a_ptrs += BLOCK_SIZE_N
                # B is [N, K], so increment along N dimension (rows)
                b_ptrs += BLOCK_SIZE_N
            c = accumulator

            offs_cm = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_ck = tile_k_idx * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            c_ptrs = c_ptr + ldc * offs_cm[:, None] + offs_ck[None, :]

            # Add masking for output stores
            mask_c = (offs_cm[:, None] < gm) & (offs_ck[None, :] < gk)
            tl.store(c_ptrs, c, mask=mask_c)

            # go to the next tile by advancing NUM_SM
            tile_idx += NUM_SM

        # get ready to go to the next gemm problem
        last_problem_end = last_problem_end + num_tiles

@triton.autotune(
    configs=_AMD_CONFIGS if torch.version.hip else _NV_CONFIGS,
    key=['group_size'],
    prune_configs_by={'early_config_prune': early_config_prune},
)
@triton.jit
def _kernel_grouped_gemm_wgrad(
    # device tensor of matrices pointers
    group_a_ptrs,
    group_b_ptrs,
    group_c_ptrs,
    # device tensor of gemm sizes. its shape is [group_size, 3]
    # dim 0 is group_size, dim 1 is the values of <N, K, M> of each gemm
    # Note: For wgrad, sizes are [N, K, M] where M is reduction dimension
    group_gemm_sizes,
    # device tensor of leading dimension sizes. its shape is [group_size, 3]
    # dim 0 is group_size, dim 1 is the values of <lda, ldb, ldc> of each gemm
    g_lds,
    # number of gemms
    group_size,
    # number of virtual SM
    NUM_SM: tl.constexpr,
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """
    Weight gradient kernel: wgrad = grad_output^T @ input
    - grad_output: [M, N] where M = tokens, N = out_features
    - input: [M, K] where K = in_features
    - wgrad: [N, K]
    Reduction is over M dimension.
    """
    tile_idx = tl.program_id(0)
    last_problem_end = 0
    for g in range(group_size):
        # get the gemm size of the current problem (use element-based indexing)
        gn = tl.load(group_gemm_sizes + g * 3)  # N
        gk = tl.load(group_gemm_sizes + g * 3 + 1)  # K
        gm = tl.load(group_gemm_sizes + g * 3 + 2)  # M (reduction dimension)
        num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
        num_k_tiles = tl.cdiv(gk, BLOCK_SIZE_K)
        num_tiles = num_n_tiles * num_k_tiles
        # iterate through the tiles in the current gemm problem
        while (tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles):
            # pick up a tile from the current gemm problem
            m = gm  # reduction dimension
            # Load leading dimensions (element-based indexing)
            lda = tl.load(g_lds + g * 3)      # grad_output stride
            ldb = tl.load(g_lds + g * 3 + 1)  # input stride
            ldc = tl.load(g_lds + g * 3 + 2)  # wgrad stride
            # Load pointers and cast to pointer type (matching tutorial)
            a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(DTYPE))  # grad_output
            b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(DTYPE))  # input
            c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(DTYPE))  # wgrad
            # figure out tile coordinates
            tile_idx_in_gemm = tile_idx - last_problem_end
            tile_n_idx = tile_idx_in_gemm // num_k_tiles
            tile_k_idx = tile_idx_in_gemm % num_k_tiles

            # do regular gemm here (same structure as forward/dgrad, but reduce over M)
            offs_cn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            offs_ck = tile_k_idx * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            offs_m = tl.arange(0, BLOCK_SIZE_M)
            # grad_output is [M, N], load transposed: [N, M]
            a_ptrs = a_ptr + offs_m[:, None] * lda + offs_cn[None, :]
            # input is [M, K], load with M and K indices
            b_ptrs = b_ptr + offs_m[:, None] * ldb + offs_ck[None, :]
            accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K), dtype=tl.float32)
            for m_idx in range(0, tl.cdiv(m, BLOCK_SIZE_M)):
                # hint to Triton compiler to do proper loop pipelining
                tl.multiple_of(a_ptrs, [16, 16])
                tl.multiple_of(b_ptrs, [16, 16])
                # Add masking for partial tiles
                m_remaining = m - m_idx * BLOCK_SIZE_M
                # grad_output is [M, N], mask is [M, N]
                mask_a = (offs_m[:, None] < m_remaining) & (offs_cn[None, :] < gn)
                # input is [M, K], mask is [M, K]
                mask_b = (offs_m[:, None] < m_remaining) & (offs_ck[None, :] < gk)
                a = tl.load(a_ptrs, mask=mask_a, other=0.0)
                b = tl.load(b_ptrs, mask=mask_b, other=0.0)
                # grad_output^T @ input: transpose a (from [M, N] to [N, M]) then dot with b [M, K]
                accumulator += tl.dot(a.T, b)
                a_ptrs += BLOCK_SIZE_M * lda
                # input is [M, K], so increment along M dimension (rows)
                b_ptrs += BLOCK_SIZE_M * ldb
            c = accumulator

            offs_cn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            offs_ck = tile_k_idx * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            c_ptrs = c_ptr + ldc * offs_cn[:, None] + offs_ck[None, :]

            # Add masking for output stores
            mask_c = (offs_cn[:, None] < gn) & (offs_ck[None, :] < gk)
            tl.store(c_ptrs, c, mask=mask_c)

            # go to the next tile by advancing NUM_SM
            tile_idx += NUM_SM

        # get ready to go to the next gemm problem
        last_problem_end = last_problem_end + num_tiles

# =============== Wrappers for Forward GGEMM =================
def _grouped_gemm_forward(
    inputmats: list,  # List of input tensors, each [m_splits[i], K]
    weights: list,  # List of weight tensors, each [N, K]
    output: torch.Tensor,  # Pre-allocated output tensor [M_total, N]
    m_splits: list,  # Token counts per expert
) -> None:
    """
    Forward pass grouped GEMM: Y = X @ W^T
    Uses persistent kernel from Triton tutorial
    
    Args:
        inputmats: List of input tensors for each expert
        weights: List of weight tensors for each expert
        output: Pre-allocated output tensor to write results
        m_splits: Token counts per expert
    """
    num_experts = len(weights)
    device = output.device
    dtype = output.dtype
    tl_dtype = torch_dtype_to_triton_dtype(dtype)
    N = output.shape[1]  # out_features
    K = inputmats[0].shape[1]  # in_features
    
    # Collect all metadata in one loop (matching Triton tutorial pattern)
    # Skip zero-length splits to avoid garbage pointers
    a_addrs = []
    b_addrs = []
    c_addrs = []
    g_sizes = []
    g_lds = []
    
    offset = 0
    for i in range(num_experts):
        # Skip zero-length splits (empty tensors have garbage data_ptr)
        if m_splits[i] == 0:
            offset += m_splits[i]
            continue
            
        # Collect pointers
        a_addrs.append(inputmats[i].data_ptr())
        b_addrs.append(weights[i].data_ptr())
        c_addrs.append(output[offset:offset+m_splits[i]].data_ptr())
        
        # Collect sizes: [M, N, K] for this expert
        g_sizes += [m_splits[i], N, K]
        
        # Collect leading dimensions: [lda, ldb, ldc] for this expert
        g_lds += [K, K, N]  # K for input stride, K for weight stride, N for output stride
        
        offset += m_splits[i]
    
    # Update num_experts to reflect actual non-zero experts
    num_experts = len(a_addrs)
    
    # Early return if all splits were zero
    if num_experts == 0:
        return  # Output tensor is already zero-initialized
    
    # Create device tensors - single memcpy per tensor
    a_ptrs = torch.tensor(a_addrs, dtype=torch.int64, device=device)
    b_ptrs = torch.tensor(b_addrs, dtype=torch.int64, device=device)
    c_ptrs = torch.tensor(c_addrs, dtype=torch.int64, device=device)
    gemm_sizes = torch.tensor(g_sizes, dtype=torch.int32, device=device)
    lds = torch.tensor(g_lds, dtype=torch.int32, device=device)
    
    # Launch kernel with 1D grid of NUM_SM
    grid = lambda meta: (meta['NUM_SM'],)
    
    # Pass tensors to kernel (Triton extracts data_ptr automatically)
    _kernel_grouped_gemm_forward[grid](
        a_ptrs,
        b_ptrs,
        c_ptrs,
        gemm_sizes,
        lds,
        num_experts,
        DTYPE=tl_dtype,
    )
    # Output tensor is modified in-place, no need to return


def _grouped_gemm_backward(
    grad_outputs: list,  # List of grad_output tensors, each [m_splits[i], N]
    weights: list,  # List of weight tensors, each [N, K]
    dgrad: torch.Tensor,  # Pre-allocated dgrad tensor [M_total, K]
    m_splits: list,  # Token counts per expert
) -> None:
    """
    Backward pass grouped GEMM for dgrad: dgrad = grad_output @ weight
    
    Args:
        grad_outputs: List of grad_output tensors for each expert [M_i, N]
        weights: List of weight tensors for each expert [N, K]
        dgrad: Pre-allocated dgrad tensor to write results [M_total, K]
        m_splits: Token counts per expert
    """
    num_experts = len(weights)
    device = dgrad.device
    dtype = dgrad.dtype
    tl_dtype = torch_dtype_to_triton_dtype(dtype)
    K = dgrad.shape[1]  # in_features (output dimension)
    N = grad_outputs[0].shape[1]  # out_features (reduction dimension)
    
    # Collect all metadata in one loop
    # Skip zero-length splits to avoid garbage pointers
    a_addrs = []
    b_addrs = []
    c_addrs = []
    g_sizes = []
    g_lds = []
    
    offset = 0
    for i in range(num_experts):
        # Skip zero-length splits (empty tensors have garbage data_ptr)
        if m_splits[i] == 0:
            offset += m_splits[i]
            continue
            
        # Collect pointers
        a_addrs.append(grad_outputs[i].data_ptr())  # grad_output [M, N]
        b_addrs.append(weights[i].data_ptr())        # weight [N, K]
        c_addrs.append(dgrad[offset:offset+m_splits[i]].data_ptr())  # dgrad [M, K]
        
        # Collect sizes: [M, K, N] for this expert
        # M = rows of grad_output/dgrad
        # K = cols of dgrad/weight (in_features)
        # N = cols of grad_output/rows of weight (out_features, reduction dimension)
        g_sizes += [m_splits[i], K, N]
        
        # Collect leading dimensions: [lda, ldb, ldc] for this expert
        # lda: grad_output stride (N for row-major [M, N])
        # ldb: weight stride (K for row-major [N, K])
        # ldc: dgrad stride (K for row-major [M, K])
        g_lds += [N, K, K]
        
        offset += m_splits[i]
    
    # Update num_experts to reflect actual non-zero experts
    num_experts = len(a_addrs)
    
    # Early return if all splits were zero
    if num_experts == 0:
        return  # Output tensor is already zero-initialized
    
    # Create device tensors - single memcpy per tensor
    a_ptrs = torch.tensor(a_addrs, dtype=torch.int64, device=device)
    b_ptrs = torch.tensor(b_addrs, dtype=torch.int64, device=device)
    c_ptrs = torch.tensor(c_addrs, dtype=torch.int64, device=device)
    gemm_sizes = torch.tensor(g_sizes, dtype=torch.int32, device=device)
    lds = torch.tensor(g_lds, dtype=torch.int32, device=device)
    
    # Launch kernel with 1D grid of NUM_SM
    grid = lambda meta: (meta['NUM_SM'],)
    
    # Pass tensors to kernel (Triton extracts data_ptr automatically)
    _kernel_grouped_gemm_backward[grid](
        a_ptrs,
        b_ptrs,
        c_ptrs,
        gemm_sizes,
        lds,
        num_experts,
        DTYPE=tl_dtype,
    )
    # Output tensor is modified in-place, no need to return


def _grouped_gemm_wgrad(
    grad_outputs: list,  # List of grad_output tensors, each [m_splits[i], N]
    inputs: list,  # List of input tensors, each [m_splits[i], K]
    wgrad: list,  # List of pre-allocated wgrad tensors, each [N, K]
    m_splits: list,  # Token counts per expert
) -> None:
    """
    Weight gradient grouped GEMM: wgrad = grad_output^T @ input
    
    Args:
        grad_outputs: List of grad_output tensors for each expert [M_i, N]
        inputs: List of input tensors for each expert [M_i, K]
        wgrad: List of pre-allocated wgrad tensors to write results [N, K]
        m_splits: Token counts per expert
    """
    num_experts = len(inputs)
    device = wgrad[0].device
    dtype = wgrad[0].dtype
    tl_dtype = torch_dtype_to_triton_dtype(dtype)
    N = wgrad[0].shape[0]  # out_features
    K = wgrad[0].shape[1]  # in_features
    
    # Collect all metadata in one loop
    # Skip zero-length splits to avoid garbage pointers
    a_addrs = []
    b_addrs = []
    c_addrs = []
    g_sizes = []
    g_lds = []
    
    for i in range(num_experts):
        # Skip zero-length splits (empty tensors have garbage data_ptr)
        if m_splits[i] == 0:
            continue
            
        # Collect pointers
        a_addrs.append(grad_outputs[i].data_ptr())  # grad_output [M, N]
        b_addrs.append(inputs[i].data_ptr())         # input [M, K]
        c_addrs.append(wgrad[i].data_ptr())         # wgrad [N, K]
        
        # Collect sizes: [N, K, M] for this expert
        # N = rows of wgrad/cols of grad_output (out_features)
        # K = cols of wgrad/cols of input (in_features)
        # M = rows of grad_output/input (tokens, reduction dimension)
        g_sizes += [N, K, m_splits[i]]
        
        # Collect leading dimensions: [lda, ldb, ldc] for this expert
        # lda: grad_output stride (N for row-major [M, N])
        # ldb: input stride (K for row-major [M, K])
        # ldc: wgrad stride (K for row-major [N, K])
        g_lds += [N, K, K]
    
    # Update num_experts to reflect actual non-zero experts
    num_experts = len(a_addrs)
    
    # Early return if all splits were zero
    if num_experts == 0:
        return  # Output tensors are already zero-initialized
    
    # Create device tensors - single memcpy per tensor
    a_ptrs = torch.tensor(a_addrs, dtype=torch.int64, device=device)
    b_ptrs = torch.tensor(b_addrs, dtype=torch.int64, device=device)
    c_ptrs = torch.tensor(c_addrs, dtype=torch.int64, device=device)
    gemm_sizes = torch.tensor(g_sizes, dtype=torch.int32, device=device)
    lds = torch.tensor(g_lds, dtype=torch.int32, device=device)
    
    # Launch kernel with 1D grid of NUM_SM
    grid = lambda meta: (meta['NUM_SM'],)
    
    # Pass tensors to kernel (Triton extracts data_ptr automatically)
    _kernel_grouped_gemm_wgrad[grid](
        a_ptrs,
        b_ptrs,
        c_ptrs,
        gemm_sizes,
        lds,
        num_experts,
        DTYPE=tl_dtype,
    )
    # Output tensors are modified in-place, no need to return


def general_grouped_gemm_triton(
    weights: list,  # List of weight tensors [out_features, in_features]
    inputmats: list,  # List of input tensors or grad_outputs
    outputs: list,  # List to store output tensors
    out_dtype: torch.dtype = None,
    workspace=None,  # Unused, for compatibility
    single_output: bool = True,
    m_splits: list = None,
    bias: list = None,
    use_bias: bool = False,
    use_split_accumulator: bool = False,  # Unused, for compatibility
    layout: str = "TN",  # "TN" for forward, "NN" for dgrad
    grad: bool = False,  # True for backward pass
    **kwargs,
) -> list:
    """
    Drop-in replacement for general_grouped_gemm using Triton persistent grouped GEMM kernel.
    
    Supports:
    - Forward pass (layout="TN"): output = input @ weight^T
    - Backward pass dgrad (layout="NN", grad=True): dgrad = grad_output @ weight
    - Backward pass wgrad (layout="NT", grad=True): wgrad = grad_output^T @ input
    
    Args:
        weights: List of weight tensors, each [out_features, in_features]
        inputmats: List of input tensors (forward) or grad_output tensors (backward)
        outputs: List to populate with output tensor(s)
        out_dtype: Output dtype
        workspace: Workspace tensor (unused, for compatibility)
        single_output: Whether to produce single concatenated output
        m_splits: List of token counts per expert
        bias: List of bias tensors (optional)
        use_bias: Whether to apply bias
        use_split_accumulator: Unused, for compatibility
        layout: "TN" for forward pass, "NN" for dgrad backward pass, "NT" for wgrad backward pass
        grad: True for backward pass
        
    Returns:
        List of output tensors (matches outputs parameter)
    """
    assert m_splits is not None, "m_splits required for Triton kernel"
    assert len(outputs) > 0, "Output tensor(s) must be pre-allocated and passed in outputs list"
    
    # Determine operation type
    is_dgrad = (layout == "NN" and grad)
    is_wgrad = (layout == "NT" and grad)
    
    if is_wgrad:
        # Backward pass: wgrad = grad_output^T @ input
        _grouped_gemm_wgrad(
            inputmats,  # grad_outputs [M, N]
            weights,    # inputs [M, K] - note: weights parameter is repurposed as inputs for wgrad
            outputs,    # wgrad list [N, K] per expert
            m_splits,
        )
    elif is_dgrad:
        # Use pre-allocated output tensor
        out = outputs[0]
        # Backward pass: dgrad = grad_output @ weight
        _grouped_gemm_backward(
            inputmats,  # grad_outputs [M, N]
            weights,    # weights [N, K]
            out,        # dgrad [M, K]
            m_splits,
        )
    else:
        # Use pre-allocated output tensor
        out = outputs[0]
        # Forward pass: output = input @ weight^T
        _grouped_gemm_forward(
            inputmats,  # inputs [M, K]
            weights,    # weights [N, K]
            out,        # output [M, N]
            m_splits,
        )
    
    # Apply bias if requested (only for forward/dgrad, not wgrad)
    if use_bias and bias is not None and not is_wgrad:
        offset = 0
        for i, count in enumerate(m_splits):
            if bias[i] is not None:
                bias_tensor = bias[i]
                if bias_tensor.dtype != out_dtype:
                    bias_tensor = bias_tensor.to(out_dtype)
                out[offset:offset+count] += bias_tensor
            offset += count
    
    return outputs, bias, None