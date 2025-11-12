import triton
import triton.language as tl
import torch


def get_num_sms():
    """Get the number of streaming multiprocessors/compute units on current device"""
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    return 304  # Default for MI300X


STANDARD_CONFIGS = [
    # Balanced configs for typical MoE workloads (MI300X optimized)
    triton.Config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64},
        num_warps=4, num_stages=2
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64},
        num_warps=8, num_stages=2
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64},
        num_warps=8, num_stages=2
    ),
    
    # High K dimension configs (for K=2048 like your workload)
    triton.Config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128},
        num_warps=8, num_stages=2
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128},
        num_warps=4, num_stages=2
    ),
    
    # Cache-optimized configs (smaller tiles for better L2 reuse on AMD)
    triton.Config(
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64},
        num_warps=4, num_stages=2
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64},
        num_warps=4, num_stages=2
    ),
    
    # High throughput configs (larger tiles when well-balanced)
    triton.Config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
        num_warps=4, num_stages=3
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
        num_warps=4, num_stages=3
    ),
    
    # Wide N configs (for N=2816 like your workload)
    triton.Config(
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64},
        num_warps=8, num_stages=2
    ),
]


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, super_group_m):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * super_group_m
    group_size_m = min(num_pid_m - first_pid_m, super_group_m)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.autotune(
    configs=STANDARD_CONFIGS,
    key=["M_TOTAL", "N", "K"],
)
@triton.jit
def _kernel_grouped_gemm_forward(
    # Pointers to matrices
    a_ptr,  # inputs [M_TOTAL, K]
    b_ptr,  # weights [num_experts, N, K]
    c_ptr,  # outputs [M_TOTAL, N]
    # Pointer to indices array
    indices_ptr,
    # Matrix dimensions
    M_TOTAL: tl.constexpr,  # Total M dimension (sum of all groups)
    N: tl.constexpr,  # Output features
    K: tl.constexpr,  # Input features
    # Number of experts
    NUM_EXPERTS: tl.constexpr,
    # Tiling parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    # Group size (for aligned loads)
    GROUP_SIZE_M: tl.constexpr = 128,
):
    """
    Forward pass: Y = X @ W^T
    X: [M_TOTAL, K], W: [num_experts, N, K], Y: [M_TOTAL, N]
    """
    # Standard 2D grid over output tiles
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    
    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N
    
    # Early exit if out of bounds
    if m_start >= M_TOTAL:
        return
    
    offs_m = m_start + tl.arange(0, BLOCK_SIZE_M)
    offs_n = n_start + tl.arange(0, BLOCK_SIZE_N)
    
    # Determine expert for this block
    group_idx = m_start // GROUP_SIZE_M
    expert_idx = tl.load(indices_ptr + group_idx * GROUP_SIZE_M)
    
    # Accumulator for the output
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    for ki in range(k_tiles):
        offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        
        # Create masks
        mask_m = offs_m < M_TOTAL
        mask_n = offs_n < N
        mask_k = offs_k < K
        mask_a = mask_m[:, None] & mask_k[None, :]
        mask_b = mask_n[:, None] & mask_k[None, :]
        
        # Load inputs: X[m, k]
        a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        
        # Load weights: W[expert, n, k]
        b_ptrs = b_ptr + expert_idx * N * K + offs_n[:, None] * K + offs_k[None, :]
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        
        # Y[m,n] = sum_k(X[m,k] * W[expert,n,k])
        accumulator += tl.dot(a, b.T)
    
    # Store output
    mask_m = offs_m < M_TOTAL
    mask_n = offs_n < N
    mask_c = mask_m[:, None] & mask_n[None, :]
    
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, accumulator.to(c_ptr.dtype.element_ty), mask=mask_c)


@triton.autotune(
    configs=STANDARD_CONFIGS,
    key=["M_TOTAL", "K", "N"],
)
@triton.jit
def _kernel_grouped_gemm_backward(
    # Pointers to matrices
    grad_output_ptr,  # grad_output [M_TOTAL, N]
    weights_ptr,      # weights [num_experts, N, K]
    grad_input_ptr,   # grad_input [M_TOTAL, K]
    # Pointer to indices array
    indices_ptr,
    # Matrix dimensions
    M_TOTAL: tl.constexpr,  # Total M dimension
    K: tl.constexpr,  # Input features (output of this kernel)
    N: tl.constexpr,  # Output features (input to this kernel)
    # Number of experts
    NUM_EXPERTS: tl.constexpr,
    # Tiling parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  # Note: K and N swapped vs forward
    BLOCK_SIZE_N: tl.constexpr,
    # Group size
    GROUP_SIZE_M: tl.constexpr = 128,
):
    """
    Backward pass (dgrad): dX = dY @ W
    dY: [M_TOTAL, N], W: [num_experts, N, K], dX: [M_TOTAL, K]
    """
    # Standard 2D grid over grad_input tiles
    pid_m = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)
    
    m_start = pid_m * BLOCK_SIZE_M
    k_start = pid_k * BLOCK_SIZE_K
    
    # Early exit if out of bounds
    if m_start >= M_TOTAL:
        return
    
    offs_m = m_start + tl.arange(0, BLOCK_SIZE_M)
    offs_k = k_start + tl.arange(0, BLOCK_SIZE_K)
    
    # Determine expert for this block
    group_idx = m_start // GROUP_SIZE_M
    expert_idx = tl.load(indices_ptr + group_idx * GROUP_SIZE_M)
    
    # Accumulator for the grad_input
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    
    # Loop over N dimension
    n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    for ni in range(n_tiles):
        offs_n = ni * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        
        # Create masks
        mask_m = offs_m < M_TOTAL
        mask_k = offs_k < K
        mask_n = offs_n < N
        mask_grad = mask_m[:, None] & mask_n[None, :]
        mask_w = mask_n[:, None] & mask_k[None, :]
        
        # Load grad_output: dY[m, n]
        grad_ptrs = grad_output_ptr + offs_m[:, None] * N + offs_n[None, :]
        grad = tl.load(grad_ptrs, mask=mask_grad, other=0.0)
        
        # Load weights: W[expert, n, k]
        w_ptrs = weights_ptr + expert_idx * N * K + offs_n[:, None] * K + offs_k[None, :]
        w = tl.load(w_ptrs, mask=mask_w, other=0.0)
        
        # dX[m,k] = sum_n(dY[m,n] * W[expert,n,k])
        accumulator += tl.dot(grad, w)
    
    # Store grad_input
    mask_m = offs_m < M_TOTAL
    mask_k = offs_k < K
    mask_c = mask_m[:, None] & mask_k[None, :]
    
    c_ptrs = grad_input_ptr + offs_m[:, None] * K + offs_k[None, :]
    tl.store(c_ptrs, accumulator.to(grad_input_ptr.dtype.element_ty), mask=mask_c)

# =============== Wrappers for Forward and Backward GGEMM =================
def _grouped_gemm_forward(
    inputs: torch.Tensor,  # [M_total, K]
    expert_weights: torch.Tensor,  # [num_experts, N, K]
    expert_indices: torch.Tensor,  # [M_total]
    group_size_m: int = 128,
) -> torch.Tensor:
    """
    Forward pass grouped GEMM: Y = X @ W^T
    
    Args:
        inputs: Input tensor of shape [M_total, K]
        expert_weights: Expert weight tensor of shape [num_experts, N, K]
        expert_indices: Indices tensor of shape [M_total] mapping each token to its expert
        group_size_m: Size of contiguous token blocks for each expert (default: 128)
    Returns:
        Output tensor of shape [M_total, N]
    """
    # Validate inputs
    assert inputs.is_contiguous(), "Input tensor must be contiguous"
    assert expert_weights.is_contiguous(), "Expert weights tensor must be contiguous"
    assert expert_indices.is_contiguous(), "Expert indices tensor must be contiguous"

    M_total, K = inputs.shape
    assert M_total % group_size_m == 0, f"M_total ({M_total}) must be a multiple of group_size_m ({group_size_m})"

    # Convert expert_indices to int32 if needed
    if expert_indices.dtype != torch.int32:
        expert_indices = expert_indices.to(torch.int32)

    # Get dimensions
    num_experts, N, K_weights = expert_weights.shape

    # Validate dimensions
    assert K == K_weights, f"Input K ({K}) must match weight K ({K_weights})"
    assert expert_indices.shape[0] == M_total, f"Expert indices length must match M_total ({M_total}) but got {expert_indices.shape[0]}"

    # Create output tensor
    output = torch.empty((M_total, N), device=inputs.device, dtype=inputs.dtype)

    # Calculate grid size for standard 2D grid
    # Grid is organized as (num_tiles_m, num_tiles_n)
    # BLOCK_SIZE_M and BLOCK_SIZE_N will be provided by the autotuner
    # We use a default of 128 for the calculation (overridden by autotune)
    BLOCK_SIZE_M_DEFAULT = 128
    BLOCK_SIZE_N_DEFAULT = 128
    num_tiles_m = triton.cdiv(M_total, BLOCK_SIZE_M_DEFAULT)
    num_tiles_n = triton.cdiv(N, BLOCK_SIZE_N_DEFAULT)
    grid = (num_tiles_m, num_tiles_n)
    
    # Launch forward kernel
    _kernel_grouped_gemm_forward[grid](
        inputs,
        expert_weights,
        output,
        expert_indices,
        M_TOTAL=M_total,
        N=N,
        K=K,
        NUM_EXPERTS=num_experts,
        GROUP_SIZE_M=group_size_m,
    )
    return output


def _grouped_gemm_backward(
    grad_output: torch.Tensor,  # [M_total, N]
    expert_weights: torch.Tensor,  # [num_experts, N, K]
    expert_indices: torch.Tensor,  # [M_total]
    group_size_m: int = 128,
) -> torch.Tensor:
    """
    Backward pass grouped GEMM (dgrad): dX = dY @ W
    
    Args:
        grad_output: Gradient of output tensor of shape [M_total, N]
        expert_weights: Expert weight tensor of shape [num_experts, N, K]
        expert_indices: Indices tensor of shape [M_total] mapping each token to its expert
        group_size_m: Size of contiguous token blocks for each expert (default: 128)
    Returns:
        Gradient of input tensor of shape [M_total, K]
    """
    # Validate inputs
    assert grad_output.is_contiguous(), "Grad output tensor must be contiguous"
    assert expert_weights.is_contiguous(), "Expert weights tensor must be contiguous"
    assert expert_indices.is_contiguous(), "Expert indices tensor must be contiguous"

    M_total, N = grad_output.shape
    assert M_total % group_size_m == 0, f"M_total ({M_total}) must be a multiple of group_size_m ({group_size_m})"

    # Convert expert_indices to int32 if needed
    if expert_indices.dtype != torch.int32:
        expert_indices = expert_indices.to(torch.int32)

    # Get dimensions
    num_experts, N_weights, K = expert_weights.shape

    # Validate dimensions
    assert N == N_weights, f"Grad output N ({N}) must match weight N ({N_weights})"
    assert expert_indices.shape[0] == M_total, f"Expert indices length must match M_total ({M_total}) but got {expert_indices.shape[0]}"

    # Create grad input tensor
    grad_input = torch.empty((M_total, K), device=grad_output.device, dtype=grad_output.dtype)

    # Calculate grid size for standard 2D grid
    # Grid is organized as (num_tiles_m, num_tiles_k)
    # BLOCK_SIZE_M and BLOCK_SIZE_K will be provided by the autotuner
    # We use a default of 128 for the calculation (overridden by autotune)
    BLOCK_SIZE_M_DEFAULT = 128
    BLOCK_SIZE_K_DEFAULT = 128
    num_tiles_m = triton.cdiv(M_total, BLOCK_SIZE_M_DEFAULT)
    num_tiles_k = triton.cdiv(K, BLOCK_SIZE_K_DEFAULT)
    grid = (num_tiles_m, num_tiles_k)
    
    # Launch backward kernel
    _kernel_grouped_gemm_backward[grid](
        grad_output,
        expert_weights,
        grad_input,
        expert_indices,
        M_TOTAL=M_total,
        K=K,
        N=N,
        NUM_EXPERTS=num_experts,
        GROUP_SIZE_M=group_size_m,
    )
    return grad_input

def general_grouped_gemm_triton(
    weights: list,  # List of weight tensors [out_features, in_features]
    inputmats: list,  # List of input tensors (or single tensor in list)
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
    Drop-in replacement for general_grouped_gemm using Triton kernel.
    
    This wrapper converts the TE grouped GEMM interface to Triton kernel format:
    - Stacks individual weight tensors into single [num_experts, N, K] tensor
    - Concatenates input tensors (if needed) into single [M_total, K] tensor
    - Creates expert indices from m_splits
    - Applies bias if requested
    
    Args:
        weights: List of weight tensors, each [out_features, in_features]
        inputmats: List of input tensors (or single input in a list)
        outputs: List to populate with output tensor(s)
        out_dtype: Output dtype (should be bfloat16)
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
    
    device = weights[0].device
    num_experts = len(weights)
    out_features, in_features = weights[0].shape
    M_total = sum(m_splits)
    
    # Detect if this is a backward pass by checking input dimensions
    # Forward: inputmats have shape [m_splits[i], in_features]
    # Backward (dgrad): inputmats (grad_output) have shape [m_splits[i], out_features]
    actual_input_dim = inputmats[0].shape[1]
    is_backward_pass = (actual_input_dim == out_features)
    
    # Stack weights: [out_features, in_features] x num_experts -> [num_experts, out_features, in_features]
    weights_stacked = torch.stack(weights, dim=0).contiguous()
    
    # Concatenate input tensors
    inputs = torch.cat(inputmats, dim=0).contiguous()
    
    # Ensure input dtype matches
    if inputs.dtype != out_dtype:
        inputs = inputs.to(out_dtype)
    
    # Verify the concatenated input shape matches m_splits
    assert inputs.shape[0] == M_total, f"Input tensor has {inputs.shape[0]} rows but m_splits sum to {M_total}"
    
    # Create expert indices from m_splits
    expert_indices = []
    for expert_id, count in enumerate(m_splits):
        expert_indices.extend([expert_id] * count)
    expert_indices = torch.tensor(expert_indices, dtype=torch.int32, device=device)
    
    # Call appropriate Triton kernel (forward or backward)
    if is_backward_pass:
        # Backward: inputs is actually grad_output [M, N], output is grad_input [M, K]
        out = _grouped_gemm_backward(inputs, weights_stacked, expert_indices)
    else:
        # Forward: inputs is X [M, K], output is Y [M, N]
        out = _grouped_gemm_forward(inputs, weights_stacked, expert_indices)
    
    # Apply bias if requested (only for forward pass)
    if use_bias and bias is not None and not is_backward_pass:
        offset = 0
        for i, count in enumerate(m_splits):
            if bias[i] is not None:
                bias_tensor = bias[i]
                if bias_tensor.dtype != out_dtype:
                    bias_tensor = bias_tensor.to(out_dtype)
                out[offset:offset+count] += bias_tensor
            offset += count
    
    # Populate output list (for interface compatibility)
    if len(outputs) > 0:
        outputs[0] = out
    else:
        outputs.append(out)
    
    return outputs