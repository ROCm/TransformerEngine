# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#
# Blockwise FP8 quantization Triton kernels (1x128 activation, 128x128 weight),
# adapted from AMD Primus-Turbo (primus_turbo/triton/quantization/quant_blockwise.py
# and primus_turbo/pytorch/kernels/quantization/quantization_impl.py).

import torch
import triton
import triton.language as tl

__all__ = [
    "quantize_fp8_blockwise",
    "quantize_fp8_blockwise_dual",
    "quantize_fp8_blockwise_weight",
    "quantize_fp8_blockwise_segment_m",
]


@triton.jit
def _floor_to_pow2(scale):
    scale_bits = scale.to(tl.uint32, bitcast=True) & 0xFF800000
    return scale_bits.to(tl.float32, bitcast=True)


@triton.jit
def compute_scale_and_quant(x_tile, x_tile_abs, axis, FP8_MAX, ROUND_POW2: tl.constexpr):
    x_tile_max = tl.max(x_tile_abs, axis=axis, keep_dims=True)
    x_tile_max = tl.maximum(x_tile_max, 1e-4)
    x_scales_tile = FP8_MAX / x_tile_max
    if ROUND_POW2:
        x_scales_tile = _floor_to_pow2(x_scales_tile)
    x_fp8_tile = x_tile * x_scales_tile
    x_fp8_tile = tl.clamp(x_fp8_tile, min=-FP8_MAX, max=FP8_MAX)
    return x_fp8_tile, x_scales_tile


@triton.jit
def quant_fp8_blockwise_kernel(
    x_ptr,
    x_fp8_ptr,
    x_scales_ptr,
    M,
    N,
    BLOCK_SIZE: tl.constexpr,
    FP8_MAX: tl.constexpr,
    AXIS: tl.constexpr,
    ROUND_POW2: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_m = tl.cast(pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE), tl.int64)
    offs_n = tl.cast(pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE), tl.int64)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    x_ptrs = x_ptr + offs_m[:, None] * N + offs_n[None, :]
    x_tile = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    x_tile_abs = tl.abs(x_tile)

    x_fp8_tile, x_scales_tile = compute_scale_and_quant(
        x_tile, x_tile_abs, AXIS, FP8_MAX, ROUND_POW2
    )

    x_fp8_ptrs = x_fp8_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(x_fp8_ptrs, x_fp8_tile.to(x_fp8_ptr.dtype.element_ty), mask=mask)

    if AXIS == 1:
        scale_offs = offs_m * tl.cdiv(N, BLOCK_SIZE) + pid_n
        scale_mask = offs_m < M
    else:
        scale_offs = pid_m * N + offs_n
        scale_mask = offs_n < N
    x_scales_tile_inv = tl.reshape(1.0 / x_scales_tile, BLOCK_SIZE)
    tl.store(x_scales_ptr + scale_offs, x_scales_tile_inv, mask=scale_mask)


@triton.jit
def quant_fp8_blockwise_dual_kernel(
    x_ptr,
    x_fp8_row_ptr,
    x_scales_row_ptr,
    x_fp8_col_ptr,
    x_scales_col_ptr,
    M,
    N,
    BLOCK_SIZE: tl.constexpr,
    FP8_MAX: tl.constexpr,
    ROUND_POW2: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_m = tl.cast(pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE), tl.int64)
    offs_n = tl.cast(pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE), tl.int64)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    x_ptrs = x_ptr + offs_m[:, None] * N + offs_n[None, :]
    x_tile = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    x_tile_abs = tl.abs(x_tile)

    x_fp8_row_tile, x_scales_row_tile = compute_scale_and_quant(
        x_tile, x_tile_abs, 1, FP8_MAX, ROUND_POW2
    )
    x_fp8_col_tile, x_scales_col_tile = compute_scale_and_quant(
        x_tile, x_tile_abs, 0, FP8_MAX, ROUND_POW2
    )

    x_fp8_row_ptrs = x_fp8_row_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(x_fp8_row_ptrs, x_fp8_row_tile.to(x_fp8_row_ptr.dtype.element_ty), mask=mask)

    x_fp8_col_ptrs = x_fp8_col_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(x_fp8_col_ptrs, x_fp8_col_tile.to(x_fp8_col_ptr.dtype.element_ty), mask=mask)

    row_scale_offs = offs_m * tl.cdiv(N, BLOCK_SIZE) + pid_n
    row_scale_mask = offs_m < M
    x_scales_row_tile_inv = tl.reshape(1.0 / x_scales_row_tile, BLOCK_SIZE)
    tl.store(x_scales_row_ptr + row_scale_offs, x_scales_row_tile_inv, mask=row_scale_mask)

    col_scale_offs = pid_m * N + offs_n
    col_scale_mask = offs_n < N
    x_scales_col_tile_inv = tl.reshape(1.0 / x_scales_col_tile, BLOCK_SIZE)
    tl.store(x_scales_col_ptr + col_scale_offs, x_scales_col_tile_inv, mask=col_scale_mask)


@triton.jit
def quant_fp8_blockwise_for_weight_kernel(
    w_ptr,
    w_fp8_ptr,
    w_scales_ptr,
    M,
    N,
    BLOCK_SIZE: tl.constexpr,
    FP8_MAX: tl.constexpr,
    ROUND_POW2: tl.constexpr,
):
    bid = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)

    batch_offset_w = bid * M * N
    batch_offset_scales = bid * tl.cdiv(M, BLOCK_SIZE) * tl.cdiv(N, BLOCK_SIZE)

    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    w_ptrs = w_ptr + batch_offset_w + offs_m[:, None] * N + offs_n[None, :]
    w_tile = tl.load(w_ptrs, mask=mask, other=0.0).to(tl.float32)

    w_tile_abs = tl.abs(w_tile)
    w_tile_max = tl.max(w_tile_abs)
    w_tile_max = tl.maximum(w_tile_max, 1e-4)
    w_scales = FP8_MAX / w_tile_max
    if ROUND_POW2:
        w_scales = _floor_to_pow2(w_scales)
    w_fp8_tile = w_tile * w_scales
    w_fp8_tile = tl.clamp(w_fp8_tile, min=-FP8_MAX, max=FP8_MAX)

    w_fp8_ptrs = w_fp8_ptr + batch_offset_w + offs_m[:, None] * N + offs_n[None, :]
    tl.store(w_fp8_ptrs, w_fp8_tile.to(w_fp8_ptr.dtype.element_ty), mask=mask)
    scale_offs = batch_offset_scales + pid_m * tl.cdiv(N, BLOCK_SIZE) + pid_n
    w_scales_inv = 1.0 / w_scales
    tl.store(w_scales_ptr + scale_offs, w_scales_inv)


@triton.jit
def quant_fp8_blockwise_segment_m_kernel(
    x_ptr,  # Input tensor [M_in, N]
    x_fp8_ptr,  # Output tensor [M_out, N] (padded)
    x_scales_ptr,  # Output scales [M_out // BLOCK_SIZE, N]
    group_offs_ptr,  # Original group offsets [B+1]
    padded_group_offs_ptr,  # Padded group offsets [B+1]
    N,
    num_groups,
    BLOCK_SIZE: tl.constexpr,
    FP8_MAX: tl.constexpr,
    ROUND_POW2: tl.constexpr,
):
    """Colwise (axis=0) blockwise quantize with per-segment M padding.

    Reads from the original (unpadded) tensor and writes to a segment-aligned
    padded output, so each MoE group starts on a BLOCK_SIZE boundary. Used to
    produce the column-wise FP8 operand for the grouped backward (wgrad), always
    quantizing from the original high-precision input (no double-quantization).
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    M_padded = tl.load(padded_group_offs_ptr + num_groups)
    block_start = pid_m * BLOCK_SIZE
    if block_start >= M_padded:
        return

    group_id = 0
    for g in range(num_groups):
        padded_start = tl.load(padded_group_offs_ptr + g)
        padded_end = tl.load(padded_group_offs_ptr + g + 1)
        if block_start >= padded_start and block_start < padded_end:
            group_id = g

    orig_group_start = tl.load(group_offs_ptr + group_id)
    orig_group_end = tl.load(group_offs_ptr + group_id + 1)
    padded_group_start = tl.load(padded_group_offs_ptr + group_id)

    offs_m_out = tl.cast(pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE), tl.int64)
    offs_n = tl.cast(pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE), tl.int64)
    offs_m_in = orig_group_start + (offs_m_out - padded_group_start)

    mask = (
        (offs_m_in[:, None] >= orig_group_start)
        & (offs_m_in[:, None] < orig_group_end)
        & (offs_n[None, :] < N)
    )

    x_ptrs = x_ptr + offs_m_in[:, None] * N + offs_n[None, :]
    x_tile = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    x_tile_abs = tl.abs(x_tile)

    x_fp8_tile, x_scales_tile = compute_scale_and_quant(x_tile, x_tile_abs, 0, FP8_MAX, ROUND_POW2)

    x_fp8_ptrs = x_fp8_ptr + offs_m_out[:, None] * N + offs_n[None, :]
    out_mask = (offs_m_out[:, None] < M_padded) & (offs_n[None, :] < N)
    tl.store(x_fp8_ptrs, x_fp8_tile.to(x_fp8_ptr.dtype.element_ty), mask=out_mask)

    scale_offs = pid_m * N + offs_n
    scale_mask = (pid_m < tl.cdiv(M_padded, BLOCK_SIZE)) & (offs_n < N)
    x_scales_tile_inv = tl.reshape(1.0 / x_scales_tile, BLOCK_SIZE)
    tl.store(x_scales_ptr + scale_offs, x_scales_tile_inv, mask=scale_mask)


# -----------------------------------------------------------------------------
# Host launchers
#
# Plain functions, not ``torch.library.custom_op``. A custom_op wrapper may be
# needed if torch.compile hits the inductor ``identify_mutated_tensors`` bug
# seen on gfx942 + Triton 3.7.
# -----------------------------------------------------------------------------


def quantize_fp8_blockwise_dual(
    x: torch.Tensor, dtype: torch.dtype, block_size: int = 128, pow2: bool = False
):
    """Blockwise-quantize a 2D tensor in BOTH row (1xB) and column (Bx1) modes in one pass.

    Returns (x_fp8_row, x_scales_row, x_fp8_col, x_scales_col); scales hold the
    dequant scale (amax/FP8_MAX) as fp32.
    """
    assert x.is_contiguous() and x.dim() == 2, "Input must be 2D and contiguous"
    M, N = x.shape
    fp8_max = torch.finfo(dtype).max

    x_fp8_row = torch.empty((M, N), dtype=dtype, device=x.device)
    x_scales_row = torch.empty(
        (M, triton.cdiv(N, block_size)), dtype=torch.float32, device=x.device
    )
    x_fp8_col = torch.empty((M, N), dtype=dtype, device=x.device)
    x_scales_col = torch.empty(
        (triton.cdiv(M, block_size), N), dtype=torch.float32, device=x.device
    )

    grid = (triton.cdiv(M, block_size), triton.cdiv(N, block_size))
    quant_fp8_blockwise_dual_kernel[grid](
        x,
        x_fp8_row,
        x_scales_row,
        x_fp8_col,
        x_scales_col,
        M,
        N,
        BLOCK_SIZE=block_size,
        FP8_MAX=fp8_max,
        ROUND_POW2=pow2,
    )
    return x_fp8_row, x_scales_row, x_fp8_col, x_scales_col


def quantize_fp8_blockwise(
    x: torch.Tensor, dtype: torch.dtype, axis: int, block_size: int = 128, pow2: bool = False
):
    """Single-direction blockwise quantize. axis=1 -> rowwise (1xB), axis=0 -> colwise (Bx1)."""
    assert x.is_contiguous() and x.dim() == 2, "Input must be 2D and contiguous"
    M, N = x.shape
    fp8_max = torch.finfo(dtype).max
    x_fp8 = torch.empty((M, N), dtype=dtype, device=x.device)
    if axis == 1:
        scales = torch.empty((M, triton.cdiv(N, block_size)), dtype=torch.float32, device=x.device)
    else:
        scales = torch.empty((triton.cdiv(M, block_size), N), dtype=torch.float32, device=x.device)

    grid = (triton.cdiv(M, block_size), triton.cdiv(N, block_size))
    quant_fp8_blockwise_kernel[grid](
        x,
        x_fp8,
        scales,
        M,
        N,
        BLOCK_SIZE=block_size,
        FP8_MAX=fp8_max,
        AXIS=axis,
        ROUND_POW2=pow2,
    )
    return x_fp8, scales


def quantize_fp8_blockwise_weight(
    w: torch.Tensor, dtype: torch.dtype, block_size: int = 128, pow2: bool = False
):
    """128x128 weight blockwise quantize. w is [B, M, N] (or [M, N], promoted to B=1)."""
    squeeze = False
    if w.dim() == 2:
        w = w.unsqueeze(0)
        squeeze = True
    assert w.is_contiguous() and w.dim() == 3, "Weight must be 3D [B,M,N] and contiguous"
    B, M, N = w.shape
    fp8_max = torch.finfo(dtype).max

    w_fp8 = torch.empty((B, M, N), dtype=dtype, device=w.device)
    w_scales = torch.empty(
        (B, triton.cdiv(M, block_size), triton.cdiv(N, block_size)),
        dtype=torch.float32,
        device=w.device,
    )
    grid = (B, triton.cdiv(M, block_size), triton.cdiv(N, block_size))
    quant_fp8_blockwise_for_weight_kernel[grid](
        w,
        w_fp8,
        w_scales,
        M,
        N,
        BLOCK_SIZE=block_size,
        FP8_MAX=fp8_max,
        ROUND_POW2=pow2,
    )
    if squeeze:
        return w_fp8.squeeze(0), w_scales.squeeze(0)
    return w_fp8, w_scales


def quantize_fp8_blockwise_segment_m(
    x, dtype, block_size, group_lens, group_offs, pow2: bool = False
):
    """Colwise blockwise quantize with per-segment (MoE group) M padding.

    Returns (x_fp8 [M_pad, N], x_scales, var_k_group_lens [B], var_k_group_offs [B+1]).
    Each group is padded up to a multiple of block_size. Allocates with an upper
    bound (M + B*block_size) to avoid a device->host sync (graph-capture safe).
    """
    assert x.is_contiguous() and x.dim() == 2, "Input must be 2D and contiguous"
    M, N = x.shape
    num_groups = group_lens.size(0)
    fp8_max = torch.finfo(dtype).max

    var_k_group_lens = ((group_lens + block_size - 1) // block_size) * block_size
    var_k_group_offs = torch.zeros(num_groups + 1, dtype=torch.int64, device=x.device)
    var_k_group_offs[1:] = torch.cumsum(var_k_group_lens, dim=0)

    m_padded_max = M + num_groups * block_size
    x_fp8 = torch.zeros((m_padded_max, N), dtype=dtype, device=x.device)
    x_scales = torch.zeros(
        (triton.cdiv(m_padded_max, block_size), N), dtype=torch.float32, device=x.device
    )

    grid = (triton.cdiv(m_padded_max, block_size), triton.cdiv(N, block_size))
    quant_fp8_blockwise_segment_m_kernel[grid](
        x,
        x_fp8,
        x_scales,
        group_offs,
        var_k_group_offs,
        N,
        num_groups,
        BLOCK_SIZE=block_size,
        FP8_MAX=fp8_max,
        ROUND_POW2=pow2,
    )
    return x_fp8, x_scales, var_k_group_lens, var_k_group_offs
