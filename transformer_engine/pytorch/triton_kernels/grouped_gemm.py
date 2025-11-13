import triton
import triton.language as tl
import torch
from typing import Optional

def num_sms():
    """Get the number of streaming multiprocessors/compute units on current device"""
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    return 304  # Default for MI300X

@triton.autotune(
    configs=[
        triton.Config({
            'BLOCK_SIZE_M': 128,
            'BLOCK_SIZE_N': 128,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': 84,
        }),
        triton.Config({
            'BLOCK_SIZE_M': 128,
            'BLOCK_SIZE_N': 128,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': 128,
        }),
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 64,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': 84,
        }),
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 64,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': 128,
        }),
        triton.Config({
            'BLOCK_SIZE_M': 128,
            'BLOCK_SIZE_N': 128,
            'BLOCK_SIZE_K': 64,
            'NUM_SM': 304,  # MI300X default
        }),
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 128,
            'BLOCK_SIZE_K': 64,
            'NUM_SM': 304,  # MI300X default
        }),
    ],
    key=['group_size'],
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
):
    tile_idx = tl.program_id(0)
    last_problem_end = 0
    for g in range(group_size):
        # get the gemm size of the current problem
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
            lda = tl.load(g_lds + g * 3)
            ldb = tl.load(g_lds + g * 3 + 1)
            ldc = tl.load(g_lds + g * 3 + 2)
            # Load pointers and cast to pointer type (matching tutorial)
            a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(tl.float16))
            b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(tl.float16))
            c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(tl.float16))
            # figure out tile coordinates
            tile_idx_in_gemm = tile_idx - last_problem_end
            tile_m_idx = tile_idx_in_gemm // num_n_tiles
            tile_n_idx = tile_idx_in_gemm % num_n_tiles

            # do regular gemm here
            offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            offs_k = tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + offs_am[:, None] * lda + offs_k[None, :]
            b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_bn[None, :]
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for kk in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
                # hint to Triton compiler to do proper loop pipelining
                tl.multiple_of(a_ptrs, [16, 16])
                tl.multiple_of(b_ptrs, [16, 16])
                # Add masking for partial tiles
                k_remaining = k - kk * BLOCK_SIZE_K
                mask_a = (offs_am[:, None] < gm) & (offs_k[None, :] < k_remaining)
                mask_b = (offs_k[:, None] < k_remaining) & (offs_bn[None, :] < gn)
                a = tl.load(a_ptrs, mask=mask_a, other=0.0)
                b = tl.load(b_ptrs, mask=mask_b, other=0.0)
                accumulator += tl.dot(a, b)
                a_ptrs += BLOCK_SIZE_K
                b_ptrs += BLOCK_SIZE_K * ldb
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
    N = output.shape[1]  # out_features
    K = inputmats[0].shape[1]  # in_features
    
    # Create pointer arrays for inputs, weights, and outputs
    a_ptrs = torch.zeros(num_experts, dtype=torch.int64, device=device)
    b_ptrs = torch.zeros(num_experts, dtype=torch.int64, device=device)
    c_ptrs = torch.zeros(num_experts, dtype=torch.int64, device=device)
    
    # Point to slices of the pre-allocated output tensor
    offset = 0
    for i in range(num_experts):
        a_ptrs[i] = inputmats[i].data_ptr()
        b_ptrs[i] = weights[i].data_ptr()
        c_ptrs[i] = output[offset:offset+m_splits[i]].data_ptr()
        offset += m_splits[i]
    
    # Create flattened gemm sizes array: [M0, N0, K0, M1, N1, K1, ...]
    # This matches the Triton tutorial format
    g_sizes = []
    for i in range(num_experts):
        g_sizes += [m_splits[i], N, K]  # Flatten into 1D list
    gemm_sizes = torch.tensor(g_sizes, dtype=torch.int32, device=device)
    
    # Create flattened leading dimensions array: [lda0, ldb0, ldc0, lda1, ldb1, ldc1, ...]
    # For row-major: lda = K, ldb = K (weight is transposed), ldc = N
    g_lds = []
    for i in range(num_experts):
        g_lds += [K, K, N]  # lda, ldb, ldc
    lds = torch.tensor(g_lds, dtype=torch.int32, device=device)
    
    # Launch kernel with 1D grid of NUM_SM
    grid = lambda meta: (meta['NUM_SM'],)
    
    _kernel_grouped_gemm_forward[grid](
        a_ptrs,
        b_ptrs,
        c_ptrs,
        gemm_sizes,
        lds,
        num_experts,
    )
    # Output tensor is modified in-place, no need to return


def general_grouped_gemm_triton(
    weights: list,  # List of weight tensors [out_features, in_features]
    inputmats: list,  # List of input tensors
    outputs: list,  # List to store output tensors
    out_dtype: torch.dtype = None,
    workspace=None,  # Unused, for compatibility
    single_output: bool = True,
    m_splits: list = None,
    bias: list = None,
    use_bias: bool = False,
    use_split_accumulator: bool = False,  # Unused, for compatibility
    **kwargs,
) -> list:
    """
    Drop-in replacement for general_grouped_gemm using Triton persistent grouped GEMM kernel.
    
    Args:
        weights: List of weight tensors, each [out_features, in_features]
        inputmats: List of input tensors, each [m_splits[i], in_features]
        outputs: List to populate with output tensor(s)
        out_dtype: Output dtype
        workspace: Workspace tensor (unused, for compatibility)
        single_output: Whether to produce single concatenated output
        m_splits: List of token counts per expert
        bias: List of bias tensors (optional)
        use_bias: Whether to apply bias
        use_split_accumulator: Unused, for compatibility
        
    Returns:
        List of output tensors (matches outputs parameter)
    """
    assert single_output, "Triton kernel only supports single_output=True"
    assert m_splits is not None, "m_splits required for Triton kernel"
    assert len(outputs) > 0, "Output tensor must be pre-allocated and passed in outputs list"
    
    # Use pre-allocated output tensor
    out = outputs[0]
    
    # Ensure inputs are contiguous and correct dtype
    inputmats_processed = []
    for inp in inputmats:
        if inp.dtype != out_dtype:
            inp = inp.to(out_dtype)
        if not inp.is_contiguous():
            inp = inp.contiguous()
        inputmats_processed.append(inp)
    
    # Ensure weights are contiguous
    weights_processed = []
    for w in weights:
        if not w.is_contiguous():
            w = w.contiguous()
        weights_processed.append(w)
    
    # Call Triton grouped GEMM kernel (modifies out in-place)
    _grouped_gemm_forward(
        inputmats_processed,
        weights_processed,
        out,
        m_splits,
    )
    
    # Apply bias if requested
    if use_bias and bias is not None:
        offset = 0
        for i, count in enumerate(m_splits):
            if bias[i] is not None:
                bias_tensor = bias[i]
                if bias_tensor.dtype != out_dtype:
                    bias_tensor = bias_tensor.to(out_dtype)
                out[offset:offset+count] += bias_tensor
            offset += count
    
    return outputs