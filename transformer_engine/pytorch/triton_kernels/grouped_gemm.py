import triton
import triton.language as tl
import torch
from typing import Any, Optional
import functools
import json
import os.path

from torch import Tensor

def is_power_of_2(x: int) -> bool:
    return (x > 0) and (x & (x - 1) == 0)

def _gmm_grid(
    N: int,
    block_size_m: int,
    block_size_n: int,
    group_sizes: Tensor,
    grid_dim: int,
) -> tuple[int]:
    assert N > 0, f"N must be positive, it's {N}."
    assert is_power_of_2(
        block_size_m
    ), f"M-dimension tile size must be a power of 2 (it's {block_size_m})."
    assert is_power_of_2(
        block_size_n
    ), f"N-dimension tile size must be a power of 2 (it's {block_size_n})."
    assert torch.all(group_sizes >= 0).item(), "All group_sizes must be non-negative."
    assert grid_dim > 0, f"Grid dimension must be positive (it's {grid_dim})."
    num_m_tiles = (group_sizes + block_size_m - 1) // block_size_m
    assert torch.all(num_m_tiles >= 0).item(), "All num_m_tiles must be non-negative."
    num_n_tiles = triton.cdiv(N, block_size_n)
    assert num_n_tiles > 0, f"num_n_tiles must be positive, it's {num_n_tiles}."
    num_tiles = torch.sum(num_m_tiles * num_n_tiles).item()
    assert num_tiles > 0, f"num_tiles must be positive, it's {num_tiles}."
    num_programs = int(min(grid_dim, num_tiles))
    assert num_programs > 0, f"num_programs must be positive, it's {num_programs}."
    return (num_programs,)

@triton.jit
def pid_grid(pid: int, num_pid_m: int, num_pid_n: int, GROUP_SIZE_M: tl.constexpr = 1):
    """
    Maps 1D pid to 2D grid coords (pid_m, pid_n).

    Args:
        - pid: 1D pid
        - num_pid_m: grid m size
        - num_pid_n: grid n size
        - GROUP_SIZE_M: tl.constexpr: default is 1
    """
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

    return pid_m, pid_n

@triton.jit
def remap_xcd(pid, GRID_MN, NUM_XCDS: tl.constexpr = 8):
    ## pid remapping on xcds
    # Number of pids per XCD in the new arrangement
    pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
    # When GRID_MN cannot divide NUM_XCDS, some xcds will have
    # pids_per_xcd pids, the other will have pids_per_xcd - 1 pids.
    # We calculate the number of xcds that have pids_per_xcd pids as
    # tall_xcds
    tall_xcds = GRID_MN % NUM_XCDS
    tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
    # Compute current XCD and local pid within the XCD
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    # Calculate new pid based on the new grouping
    # Note that we need to consider the following two cases:
    # 1. the current pid is on a tall xcd
    # 2. the current pid is on a short xcd
    if xcd < tall_xcds:
        pid = xcd * pids_per_xcd + local_pid
    else:
        pid = (
            tall_xcds * pids_per_xcd
            + (xcd - tall_xcds) * (pids_per_xcd - 1)
            + local_pid
        )

    return pid

# XCD remapping followed by 1D PID to 2D grid mapping.
@triton.jit
def _remap_xcd_tile_grid(
    tile_in_mm,
    num_row_tiles,
    num_col_tiles,
    GROUP_SIZE: tl.constexpr = 1,
    NUM_XCDS: tl.constexpr = 8,
):
    return pid_grid(
        remap_xcd(tile_in_mm, num_row_tiles * num_col_tiles, NUM_XCDS=NUM_XCDS),
        num_row_tiles,
        num_col_tiles,
        GROUP_SIZE_M=GROUP_SIZE,
    )

@triton.heuristics(
    {
        "K_DIVISIBLE_BY_BLOCK_SIZE_K": lambda META: META["K"] % META["BLOCK_SIZE_K"]
        == 0,
    }
)
@triton.jit
def gmm_kernel(
    # Tensor pointers:
    lhs_ptr,
    rhs_ptr,
    group_sizes_ptr,
    out_ptr,
    # Tensor shapes:
    M: int,
    K: int,
    N: int,
    G: int,
    # Meta-parameters:
    TRANS_RHS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    K_DIVISIBLE_BY_BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    GRID_DIM: tl.constexpr,
):
    tl.assume(M > 0)
    tl.assume(K > 0)
    tl.assume(N > 0)
    tl.assume(G > 0)

    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    tl.device_assert(num_n_tiles > 0, "num_n_tiles <= 0")

    # Current tile. Each program computes multiple tiles of each group.
    tile = tl.program_id(0)
    tl.device_assert(tile >= 0, "tile < 0 (at initialization)")

    # Tile limit of last MM problem (inclusive).
    last_mm_tile = 0

    # Last input row of lhs and output row of out. Each group reads some rows of
    # lhs and writes some rows to out.
    last_m = 0

    # Loop through all (m, K, N) MM problems:
    #   (m, K) x (K, N) = (m, N)
    #   sum(m) = M
    for g in range(G):
        # Get m dimension of current MM problem.
        m = tl.load(group_sizes_ptr + g)
        # m can be zero if group is empty
        tl.device_assert(m >= 0, "m < 0")

        num_m_tiles = tl.cdiv(m, BLOCK_SIZE_M)
        # num_m_tiles can be zero if group is empty
        tl.device_assert(num_m_tiles >= 0, "num_m_tiles < 0")

        num_tiles = num_m_tiles * num_n_tiles
        # num_tiles can be zero if group is empty
        tl.device_assert(num_tiles >= 0, "num_tiles < 0")

        # Loop through tiles of current MM problem.
        while tile >= last_mm_tile and tile < last_mm_tile + num_tiles:
            # Figure out tile coordinates in current MM problem.
            tile_in_mm = tile - last_mm_tile
            tl.device_assert(tile_in_mm >= 0, "tile_in_mm < 0")

            tile_m, tile_n = _remap_xcd_tile_grid(
                tile_in_mm, num_m_tiles, num_n_tiles, GROUP_SIZE=GROUP_SIZE
            )

            # Do regular MM:

            tl.device_assert(tile_m * BLOCK_SIZE_M >= 0, "tile_m * BLOCK_SIZE_M < 0")
            tl.device_assert(tile_n * BLOCK_SIZE_N >= 0, "tile_n * BLOCK_SIZE_N < 0")

            offs_lhs_m = (
                tile_m.to(tl.int64) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            ) % m
            offs_rhs_n = (
                tile_n.to(tl.int64) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            ) % N
            offs_k = tl.arange(0, BLOCK_SIZE_K).to(tl.int64)

            lhs_ptrs = lhs_ptr + (last_m + offs_lhs_m[:, None]) * K + offs_k[None, :]

            if TRANS_RHS:
                rhs_ptrs = (
                    rhs_ptr
                    + g.to(tl.int64) * K * N
                    + offs_k[:, None]
                    + offs_rhs_n[None, :] * K
                )
            else:
                rhs_ptrs = (
                    rhs_ptr
                    + g.to(tl.int64) * K * N
                    + offs_k[:, None] * N
                    + offs_rhs_n[None, :]
                )

            acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

            for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
                if K_DIVISIBLE_BY_BLOCK_SIZE_K:
                    lhs = tl.load(lhs_ptrs)
                    rhs = tl.load(rhs_ptrs)
                else:
                    k_mask_limit = K - k * BLOCK_SIZE_K
                    lhs = tl.load(
                        lhs_ptrs, mask=offs_k[None, :] < k_mask_limit, other=0
                    )
                    rhs = tl.load(
                        rhs_ptrs, mask=offs_k[:, None] < k_mask_limit, other=0
                    )

                acc += tl.dot(lhs, rhs, input_precision="ieee")

                lhs_ptrs += BLOCK_SIZE_K

                if TRANS_RHS:
                    rhs_ptrs += BLOCK_SIZE_K
                else:
                    rhs_ptrs += BLOCK_SIZE_K * N

            acc = acc.to(out_ptr.type.element_ty)

            offs_out_m = tile_m.to(tl.int64) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_out_n = tile_n.to(tl.int64) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

            out_ptrs = (
                out_ptr + (last_m + offs_out_m[:, None]) * N + offs_out_n[None, :]
            )

            tl.store(
                out_ptrs,
                acc,
                mask=(offs_out_m[:, None] < m) & (offs_out_n[None, :] < N),
            )

            # Go to the next tile by advancing number of programs.
            tile += GRID_DIM
            tl.device_assert(tile > 0, "tile <= 0 (at update)")

        # Get ready to go to the next MM problem.

        last_mm_tile += num_tiles
        # last_mm_tile can be zero if group 0 is skipped
        tl.device_assert(last_mm_tile >= 0, "last_mm_tile < 0 (at update)")

        last_m += m
        # last_m can be zero if group 0 is skipped
        tl.device_assert(last_m >= 0, "last_m < 0 (at update)")
        tl.device_assert(last_m <= M, "last_m > M (at update)")

def _ptgmm_grid(
    K: int,
    N: int,
    G: int,
    block_size_k: int,
    block_size_n: int,
    grid_dim: int,
) -> tuple[int]:
    assert K > 0, f"K must be positive, it's {K}."
    assert N > 0, f"N must be positive, it's {N}."
    assert G > 0, f"G must be positive, it's {G}."
    assert is_power_of_2(
        block_size_k
    ), f"K-dimension tile size must be a power of 2 (it's {block_size_k})."
    assert is_power_of_2(
        block_size_n
    ), f"N-dimension tile size must be a power of 2 (it's {block_size_n})."
    assert grid_dim > 0, f"Grid dimension must be positive (it's {grid_dim})."
    num_k_tiles = triton.cdiv(K, block_size_k)
    assert num_k_tiles > 0, f"num_k_tiles must be positive, it's {num_k_tiles}."
    num_n_tiles = triton.cdiv(N, block_size_n)
    assert num_n_tiles > 0, f"num_n_tiles must be positive, it's {num_n_tiles}."
    num_tiles = G * num_k_tiles * num_n_tiles
    assert num_tiles > 0, f"num_tiles must be positive, it's {num_tiles}."
    num_programs = min(grid_dim, num_tiles)
    assert num_programs > 0, f"num_programs must be positive, it's {num_programs}."
    return (num_programs,)

def _nptgmm_grid(
    K: int,
    N: int,
    G: int,
    block_size_k: int,
    block_size_n: int,
) -> tuple[int, int]:
    assert K > 0, f"K must be positive, it's {K}."
    assert N > 0, f"N must be positive, it's {N}."
    assert G > 0, f"G must be positive, it's {G}."
    assert is_power_of_2(
        block_size_k
    ), f"K-dimension tile size must be a power of 2 (it's {block_size_k})."
    assert is_power_of_2(
        block_size_n
    ), f"N-dimension tile size must be a power of 2 (it's {block_size_n})."
    num_k_tiles = triton.cdiv(K, block_size_k)
    assert num_k_tiles > 0, f"num_k_tiles must be positive, it's {num_k_tiles}."
    num_n_tiles = triton.cdiv(N, block_size_n)
    assert num_n_tiles > 0, f"num_n_tiles must be positive, it's {num_n_tiles}."
    num_tiles_per_mm = num_k_tiles * num_n_tiles
    assert (
        num_tiles_per_mm > 0
    ), f"num_tiles_per_mm must be positive, it's {num_tiles_per_mm}."
    return (G, num_tiles_per_mm)



@triton.jit
def tgmm_persistent_kernel(
    # Tensor pointers:
    lhs_ptr,
    rhs_ptr,
    group_sizes_ptr,
    out_ptr,
    # Tensor shapes:
    M: int,
    K: int,
    N: int,
    G: int,
    # Meta-parameters:
    TRANS_LHS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    GRID_DIM: tl.constexpr,
):
    tl.assume(M > 0)
    tl.assume(K > 0)
    tl.assume(N > 0)
    tl.assume(G > 0)

    num_k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    tl.device_assert(num_k_tiles > 0, "num_k_tiles <= 0")

    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    tl.device_assert(num_n_tiles > 0, "num_n_tiles <= 0")

    num_tiles = num_k_tiles * num_n_tiles
    tl.device_assert(num_tiles > 0, "num_tiles <= 0")

    # Current tile. Each program computes multiple tiles of each group.
    tile = tl.program_id(0)
    tl.device_assert(tile >= 0, "tile < 0 (at initialization)")

    # Tile limit of last MM problem (inclusive).
    last_mm_tile = 0

    # Last input column of lhs and input row of rhs. Each group reads some
    # columns of lhs and some rows of rhs.
    last_m = 0

    # Loop through all (K, m, N) MM problems:
    #   (K, m) x (m, N) = (K, N)
    #   sum(m) = M
    for g in range(G):
        # Get m dimension of current MM problem.
        m = tl.load(group_sizes_ptr + g)
        # m can be zero if group is empty
        tl.device_assert(m >= 0, "m < 0")

        # Loop through tiles of current MM problem.
        while tile >= last_mm_tile and tile < last_mm_tile + num_tiles:
            # Figure out tile coordinates in current MM problem.
            tile_in_mm = tile - last_mm_tile
            tl.device_assert(tile_in_mm >= 0, "tile_in_mm < 0")

            tile_k, tile_n = _remap_xcd_tile_grid(
                tile_in_mm, num_k_tiles, num_n_tiles, GROUP_SIZE=GROUP_SIZE
            )

            # Do regular MM:

            tl.device_assert(tile_k * BLOCK_SIZE_K >= 0, "tile_k * BLOCK_SIZE_K < 0")
            tl.device_assert(tile_n * BLOCK_SIZE_N >= 0, "tile_n * BLOCK_SIZE_N < 0")

            offs_lhs_k = (
                tile_k.to(tl.int64) * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            ) % K
            offs_rhs_n = (
                tile_n.to(tl.int64) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            ) % N
            offs_m = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)

            if TRANS_LHS:
                lhs_ptrs = (
                    lhs_ptr + offs_lhs_k[:, None] + (last_m + offs_m[None, :]) * K
                )
            else:
                lhs_ptrs = (
                    lhs_ptr + offs_lhs_k[:, None] * M + (last_m + offs_m[None, :])
                )

            rhs_ptrs = rhs_ptr + (last_m + offs_m[:, None]) * N + offs_rhs_n[None, :]

            loop_m = tl.cdiv(m, BLOCK_SIZE_M)
            m_divisible_by_block_m = m % BLOCK_SIZE_M == 0
            if not m_divisible_by_block_m:
                loop_m -= 1

            acc = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float32)

            for _ in range(0, loop_m):
                lhs = tl.load(lhs_ptrs)
                rhs = tl.load(rhs_ptrs)

                acc += tl.dot(lhs, rhs, input_precision="ieee")

                if TRANS_LHS:
                    lhs_ptrs += BLOCK_SIZE_M * K
                else:
                    lhs_ptrs += BLOCK_SIZE_M

                rhs_ptrs += BLOCK_SIZE_M * N

            if not m_divisible_by_block_m:
                offs_lhs_k = (
                    tile_k.to(tl.int64) * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
                ) % K
                offs_rhs_n = (
                    tile_n.to(tl.int64) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                ) % N
                offs_m = loop_m.to(tl.int64) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                lhs = tl.load(lhs_ptrs, mask=offs_m[None, :] < m, other=0)
                rhs = tl.load(rhs_ptrs, mask=offs_m[:, None] < m, other=0)
                acc += tl.dot(lhs, rhs, input_precision="ieee")

            acc = acc.to(out_ptr.type.element_ty)

            offs_out_k = tile_k.to(tl.int64) * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            offs_out_n = tile_n.to(tl.int64) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

            out_ptrs = (
                out_ptr
                + g.to(tl.int64) * K * N
                + offs_out_k[:, None] * N
                + offs_out_n[None, :]
            )

            tl.store(
                out_ptrs,
                acc,
                mask=(offs_out_k[:, None] < K) & (offs_out_n[None, :] < N),
            )

            # Go to the next tile by advancing number of programs.
            tile += GRID_DIM
            tl.device_assert(tile > 0, "tile <= 0 (at update)")

        # Get ready to go to the next MM problem.

        last_mm_tile += num_tiles
        # last_mm_tile can be zero if group 0 is skipped
        tl.device_assert(last_mm_tile >= 0, "last_mm_tile < 0 (at update)")

        last_m += m
        # last_m can be zero if group 0 is skipped
        tl.device_assert(last_m >= 0, "last_m < 0 (at update)")
        tl.device_assert(last_m <= M, "last_m > M (at update)")



def num_sms():
    """Get the number of streaming multiprocessors/compute units on current device"""
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    return 304  # Default for MI300X


# Configuration loader for AITER kernels
@functools.lru_cache()
def get_gmm_config():
    """
    Load or provide default configuration for GMM kernels (forward and dgrad).
    Returns configuration dict with BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, etc.
    """
    # Configuration optimized for MI300X GMM
    return {
        "BLOCK_SIZE_M": 256,
        "BLOCK_SIZE_K": 64,
        "BLOCK_SIZE_N": 256,
        "GROUP_SIZE": 1,
        "num_warps": 8,
        "num_stages": 1,
        "GRID_DIM": num_sms(),
    }

@functools.lru_cache()
def get_ptgmm_config():
    """
    Load or provide default configuration for PTGMM kernels (wgrad).
    Returns configuration dict with BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, etc.
    """
    # Configuration optimized for MI300X PTGMM (persistent TGMM)
    return {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_K": 256,
        "BLOCK_SIZE_N": 256,
        "GROUP_SIZE": 1,
        "num_warps": 8,
        "num_stages": 1,
        "GRID_DIM": num_sms(),
    }

def _grouped_gemm_forward(
    inputmats: list,  # List with single concatenated input tensor [M_total, K]
    weights: list,  # List of weight tensors, each [N, K]
    output: torch.Tensor,  # Pre-allocated output tensor [M_total, N]
    m_splits: list,  # Token counts per expert
) -> None:
    """
    Forward pass grouped GEMM using AITER's kernel: Y = X @ W^T
    
    Args:
        inputmats: List with single concatenated input tensor for all experts
        weights: List of weight tensors for each expert
        output: Pre-allocated output tensor to write results
        m_splits: Token counts per expert
    """
    # Get configuration
    config = get_gmm_config()
    
    # Extract dimensions
    num_experts = len(weights)
    device = output.device
    dtype = output.dtype
    N = output.shape[1]  # out_features
    
    # Handle input - either single concatenated tensor or list
    if len(inputmats) == 1:
        # Single concatenated tensor from Triton-optimized path
        input_tensor = inputmats[0]
    else:
        # List of tensors - concatenate them
        input_tensor = torch.cat(inputmats, dim=0)
    
    K = input_tensor.shape[1]  # in_features
    M = input_tensor.shape[0]  # total tokens
    
    # Create group_sizes tensor
    group_sizes = torch.tensor(m_splits, dtype=torch.int32, device=device)
    
    # Stack weights into single tensor [G, N, K] for TRANS_RHS=True access
    weights_stacked = torch.stack(weights, dim=0)  # [G, N, K]
    
    
    # Launch kernel
    grid = _gmm_grid(
        N,
        config["BLOCK_SIZE_M"],
        config["BLOCK_SIZE_N"],
        group_sizes,
        config["GRID_DIM"],
    )

    
    gmm_kernel[grid](
        input_tensor,
        weights_stacked,
        group_sizes,
        output,
        M, K, N, num_experts,
        TRANS_RHS=True,  # Weights are [N, K], need transpose
        **config,
    )


def _grouped_gemm_backward(
    grad_outputs: list,  # List of grad_output tensors, each [m_splits[i], N]
    weights: list,  # List of weight tensors, each [N, K]
    dgrad: torch.Tensor,  # Pre-allocated dgrad tensor [M_total, K]
    m_splits: list,  # Token counts per expert
) -> None:
    """
    Backward pass grouped GEMM for dgrad using AITER's GMM kernel.
    Computes: dgrad = grad_output @ weight
    
    This is: (m, N) @ (N, K) = (m, K) per group
    which fits GMM pattern: (m, K_in) x (K_in, N_out) = (m, N_out)
    with K_in=N, N_out=K
    
    Args:
        grad_outputs: List of grad_output tensors for each expert [M_i, N]
        weights: List of weight tensors for each expert [N, K]
        dgrad: Pre-allocated dgrad tensor to write results [M_total, K]
        m_splits: Token counts per expert
    """
    # Get configuration
    config = get_gmm_config()
    
    # Extract dimensions
    num_experts = len(weights)
    device = dgrad.device
    dtype = dgrad.dtype
    K = dgrad.shape[1]  # in_features (output of this op)
    N = grad_outputs[0].shape[1]  # out_features (input to this op)
    
    # Handle grad_outputs - either single concatenated tensor or list
    if len(grad_outputs) == 1:
        grad_output_tensor = grad_outputs[0]
    else:
        grad_output_tensor = torch.cat(grad_outputs, dim=0)
    
    M = grad_output_tensor.shape[0]  # total tokens
    
    # Create group_sizes tensor
    group_sizes = torch.tensor(m_splits, dtype=torch.int32, device=device)
    
    # For gmm_kernel: (m, K_in) x (K_in, N_out) = (m, N_out)
    # We have: (m, N) x (N, K) = (m, K)
    # So K_in=N, N_out=K
    # Weights are [N, K], which is already in the right layout (no transpose needed)
    
    # Stack weights into single tensor [G, N, K]
    weights_stacked = torch.stack(weights, dim=0)  # [G, N, K]
    
    # Launch kernel with proper grid calculation
    grid = _gmm_grid(
        K,  # N_out dimension (output columns)
        config["BLOCK_SIZE_M"],
        config["BLOCK_SIZE_N"],
        group_sizes,
        config["GRID_DIM"],
    )
    
    gmm_kernel[grid](
        grad_output_tensor,
        weights_stacked,
        group_sizes,
        dgrad,
        M, N, K, num_experts,
        TRANS_RHS=True,  # Weights are [N, K]
        **config,
    )

def _grouped_gemm_wgrad(
    grad_outputs: list,  # List of grad_output tensors, each [m_splits[i], N]
    inputs: list,  # List of input tensors, each [m_splits[i], K]
    wgrad: list,  # List of pre-allocated wgrad tensors, each [N, K]
    m_splits: list,  # Token counts per expert
) -> None:
    """
    Weight gradient grouped GEMM using AITER's TGMM kernel.
    Computes: wgrad = grad_output^T @ input
    
    This is: (N, m) @ (m, K) = (N, K) per group
    which fits TGMM pattern: (K_out, m) x (m, N_out) = (K_out, N_out)
    with K_out=N, N_out=K, and TRANS_LHS=True
    
    Args:
        grad_outputs: List of grad_output tensors for each expert [M_i, N]
        inputs: List of input tensors for each expert [M_i, K]
        wgrad: List of pre-allocated wgrad tensors to write results [N, K]
        m_splits: Token counts per expert
    """
    # Get configuration for PTGMM (wgrad uses persistent TGMM kernel)
    config = get_ptgmm_config()
    
    # Extract dimensions
    num_experts = len(inputs)
    device = wgrad[0].device
    dtype = wgrad[0].dtype
    
    # Handle grad_outputs - either single concatenated tensor or list
    if len(grad_outputs) == 1:
        grad_output_tensor = grad_outputs[0]
    else:
        grad_output_tensor = torch.cat(grad_outputs, dim=0)
        
    # Handle inputs - either single concatenated tensor or list
    if len(inputs) == 1:
        input_tensor = inputs[0]
    else:
        input_tensor = torch.cat(inputs, dim=0)
        
    M = grad_output_tensor.shape[0]  # total tokens
    K = grad_output_tensor.shape[1]  # out_features
    N = input_tensor.shape[1]  # in_features
    
    # Create group_sizes tensor
    group_sizes = torch.tensor(m_splits, dtype=torch.int32, device=device)
    
    # Stack wgrad tensors to write results [G, K, N]
    wgrad_stacked = torch.stack(wgrad, dim=0).contiguous()  # [G, K, N]
    
    # Verify expected strides for the kernel: should be (K*N, N, 1)
    expected_strides = (K * N, N, 1)
    if wgrad_stacked.stride() != expected_strides:
        print(f"WARNING: wgrad_stacked has unexpected strides {wgrad_stacked.stride()}, expected {expected_strides}")
        print(f"Creating new contiguous tensor with correct layout")
        # Create a properly strided tensor
        wgrad_stacked = wgrad_stacked.contiguous()
    
    # For TGMM: lhs[K_out, m] @ rhs[m, N_out] = out[G, K_out, N_out]
    # We need: grad_output^T[K, m] @ input[m, N] = wgrad[G, K, N]
    # With TRANS_LHS=True: grad_output[M, K] -> [K, M]
    # So: lhs[K, M], rhs[M, N], out[G, K, N]
    
    # Launch kernel
    grid = _ptgmm_grid(
        K,
        N,
        num_experts,
        config["BLOCK_SIZE_K"],
        config["BLOCK_SIZE_N"],
        config["GRID_DIM"],
    )
    
    # Launch kernel with all config parameters (including num_warps, num_stages)
    tgmm_persistent_kernel[grid](
        grad_output_tensor,
        input_tensor,
        group_sizes,
        wgrad_stacked,
        M, K, N, num_experts,
        TRANS_LHS=True,  # grad_output [M, K] accessed as transposed [K, M]
        **config,  # Pass all config params: BLOCK_SIZE_M/K/N, GROUP_SIZE, GRID_DIM, num_warps, num_stages
    )

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
    Drop-in replacement for general_grouped_gemm using AITER's Triton kernels.
    
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
