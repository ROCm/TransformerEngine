# Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

import torch
from typing import Optional

try:
    from ..constants import MXFP8_BLOCK_SCALING_SIZE
    from .common import (
        te_dtype_to_triton_dtype,
        te_dtype_to_torch_dtype,
        get_fp8_max,
    )
except Exception:  # pragma: no cover - fallback for standalone benchmarking
    MXFP8_BLOCK_SCALING_SIZE = 32

    def _missing(*args, **kwargs):
        raise ImportError(
            "transformer_engine dependencies not available. "
            "Ensure transformer_engine_torch is installed."
        )

    te_dtype_to_triton_dtype = _missing  # type: ignore
    te_dtype_to_torch_dtype = _missing  # type: ignore
    get_fp8_max = _missing  # type: ignore

import triton
import triton.language as tl


def _e8m0_shuffle_torch(scale: torch.Tensor, pad_rows: int, pad_cols: int) -> torch.Tensor:
    """
    Software fallback for the MX block-scale shuffle used by AITER.

    Args:
        scale: [rows, cols] tensor (uint8) without padding/shuffle.
        pad_rows: padded row count (multiple of 256).
        pad_cols: padded col count (multiple of 8).
    """
    m, n = scale.shape
    out = torch.full(
        (pad_rows, pad_cols),
        0x7F,
        dtype=scale.dtype,
        device=scale.device,
    )
    out[:m, :n] = scale
    sm, sn = out.shape
    out = out.view(sm // 32, 2, 16, sn // 8, 2, 4)
    out = out.permute(0, 3, 5, 2, 4, 1).contiguous()
    return out.view(sm, sn)
##########################################
#### cast_transpose
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
def _cast_transpose_triton(A, noop_ptr, C, T, stride_am, stride_an, stride_bn, stride_bm, M, N, scale_ptr, amax_ptr, scale_inv_ptr, max_fp8: tl.constexpr, use_noop: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, GROUP_M: tl.constexpr):
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
    
    rm = pid_m.to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n.to(tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N)
    A = A + rm[:, None] * stride_am + rn[None, :] * stride_an
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    a = tl.load(A, mask=mask)
    a = a.to(tl.float32)

    scaled_a = a * scale
    scaled_a = tl.clamp(scaled_a, -max_fp8, max_fp8)
    fp8_a = scaled_a.to(C.type.element_ty)
    C = C + rm[:, None] * stride_am + rn[None, :] * stride_an
    tl.store(C, fp8_a, mask=mask)
    
    # rematerialize to save registers
    rm = pid_m.to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n.to(tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N)
    T = T + rm[:, None] * stride_bm + rn[None, :] * stride_bn
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    tl.store(T, fp8_a, mask=mask)

    amax = tl.max(tl.abs(a))
    tl.atomic_max(amax_ptr, amax, sem='relaxed')
    if pid == 0:
        scale_inv_out = tl.fdiv(1.0, scale)
        tl.store(scale_inv_ptr, scale_inv_out)

FP32_EXPONENT_BIAS = tl.constexpr(127)
FP32_MANTISSA_BITS = tl.constexpr(23)
@triton.jit
def exp2f_rcp_triton(biased_exp: tl.uint8) -> tl.float32:
    biased_exp_f32 = biased_exp.to(tl.float32)
    exp_val = FP32_EXPONENT_BIAS - biased_exp_f32
    result = tl.exp2(exp_val)
    final_result = tl.where(biased_exp == 0, 1.0, result)
    return final_result

@triton.jit
def float_to_e8m0_triton(val: tl.float32) -> tl.uint8:
    is_nan = (val != val)
    is_inf = (tl.abs(val) == float('inf'))
    is_zero = val == 0.0

    result_e8m0 = tl.zeros(val.shape, dtype=tl.uint8) # Placeholder
    val_u32 = tl.cast(val, tl.uint32, bitcast=True)

    # Extract exponent and mantissa
    exponent_raw = (val_u32 >> FP32_MANTISSA_BITS) & 0xFF 
    mantissa = val_u32 & 0x7FFFFF

    # Round up exponent and deal with satfinite.
    # (mantissa > 0 && exponent != 0xFE) && !(exponent == 0 && mantissa <= 0x400000)
    cond1 = mantissa > 0
    cond2 = exponent_raw != 0xFE 
    cond3_part1 = exponent_raw == 0
    cond3_part2 = mantissa <= 0x400000
    cond3 = cond3_part1 & cond3_part2 
    
    round_up_condition = (cond1 & cond2) & ~cond3

    # Increment exponent if the condition is true
    calculated_exponent = tl.where(round_up_condition, exponent_raw + 1, exponent_raw)
    
    # Priority: NaN -> Inf -> Zero -> Calculated Exponent
    result_e8m0 = tl.where(is_nan, tl.full(val.shape, 0xFF, dtype=tl.uint8), result_e8m0)
    result_e8m0 = tl.where(~is_nan & is_inf, tl.full(val.shape, 0xFE, dtype=tl.uint8), result_e8m0)
    result_e8m0 = tl.where(~is_nan & ~is_inf & is_zero, tl.full(val.shape, 0x00, dtype=tl.uint8), result_e8m0)
    result_e8m0 = tl.where(~is_nan & ~is_inf & ~is_zero, calculated_exponent, result_e8m0)

    return result_e8m0

@triton.jit
def _cast_transpose_triton_mxfp8(
    x_ptr, rowwise_y_ptr, colwise_y_ptr, 
    stride_rowwise_row, stride_rowwise_col, 
    n_rows, n_cols, 
    rowwise_scale_inv_ptr, stride_rowwise_scale_inv_row, stride_rowwise_scale_inv_col,
    rowwise_scale_M, rowwise_scale_N,
    colwise_scale_inv_ptr, stride_colwise_scale_inv_row, stride_colwise_scale_inv_col,
    colwise_scale_M, colwise_scale_N,
    max_fp8: tl.constexpr, BLOCK_X: tl.constexpr, BLOCK_Y: tl.constexpr, GROUP_Y: tl.constexpr, MXFP8_BLOCK_SCALING_SIZE: tl.constexpr,  USE_ROWWISE_SCALING: tl.constexpr, USE_COLWISE_SCALING: tl.constexpr):
   
    pid = tl.program_id(0)

    num_pid_along_Y = tl.cdiv(n_rows, BLOCK_Y)
    num_pid_along_X = tl.cdiv(n_cols, BLOCK_X)
    num_pid_in_group = GROUP_Y * num_pid_along_X

    group_id = pid // num_pid_in_group
    group_size = min(num_pid_along_Y - group_id * GROUP_Y, GROUP_Y)
    pid_m = group_id * GROUP_Y + ((pid % num_pid_in_group) % group_size)
    pid_n = (pid % num_pid_in_group) // group_size

    global_offset_Y_base = pid_m.to(tl.int64) * BLOCK_Y
    global_offset_X_base = pid_n.to(tl.int64) * BLOCK_X
    
    num_chunks_in_block_Y = BLOCK_Y // MXFP8_BLOCK_SCALING_SIZE
    num_chunks_in_block_X = BLOCK_X // MXFP8_BLOCK_SCALING_SIZE
    max_norm_rcp = 1.0 / max_fp8

    for chunk_id_y in range(0, num_chunks_in_block_Y):
        offsets_Y = global_offset_Y_base + chunk_id_y * MXFP8_BLOCK_SCALING_SIZE + tl.arange(0, MXFP8_BLOCK_SCALING_SIZE)
        for chunk_id_x in range(0, num_chunks_in_block_X):
            offsets_X = global_offset_X_base  + chunk_id_x * MXFP8_BLOCK_SCALING_SIZE + tl.arange(0, MXFP8_BLOCK_SCALING_SIZE)
            x_ptr_current_chunk = x_ptr + offsets_Y[:, None] * stride_rowwise_row + offsets_X[None, :] * stride_rowwise_col
            mask = (offsets_Y < n_rows)[:, None] & (offsets_X < n_cols)[None, :]
            # (MXFP8_BLOCK_SCALING_SIZE, MXFP8_BLOCK_SCALING_SIZE)
            x_chunk = tl.load(x_ptr_current_chunk, mask=mask, other=0.0).to(tl.float32)

            # Rowwise
            if USE_ROWWISE_SCALING:
                subwarp_amax_rowwise = tl.max(tl.abs(x_chunk), axis=-1, keep_dims=True)
                biased_exponent_rowwise = float_to_e8m0_triton(subwarp_amax_rowwise * max_norm_rcp)
                
                scale_offset_X = (pid_n * num_chunks_in_block_X) + chunk_id_x
                rowwise_scale_inv_store_offsets = (offsets_Y[:, None] * stride_rowwise_scale_inv_row) + scale_offset_X * stride_rowwise_scale_inv_col 
                rowwise_scale_inv_store_mask = (offsets_Y < rowwise_scale_M)[:, None] & (scale_offset_X < rowwise_scale_N)
                tl.store(rowwise_scale_inv_ptr + rowwise_scale_inv_store_offsets, biased_exponent_rowwise, mask = rowwise_scale_inv_store_mask)
                
                block_inverse_scale_rowwise = exp2f_rcp_triton(biased_exponent_rowwise)
                y_chunk_rowwise_scaled = x_chunk * block_inverse_scale_rowwise
                rowwise_y_ptr_current_chunk = rowwise_y_ptr + offsets_Y[:, None] * stride_rowwise_row + offsets_X[None, :] * stride_rowwise_col
                tl.store(rowwise_y_ptr_current_chunk, y_chunk_rowwise_scaled.to(rowwise_y_ptr.type.element_ty), mask=mask)

            # Colwise
            if USE_COLWISE_SCALING:
                subwarp_amax_colwise = tl.max(tl.abs(x_chunk), axis=0, keep_dims=True)
                biased_exponent_colwise = float_to_e8m0_triton(subwarp_amax_colwise * max_norm_rcp)

                scale_offset_Y = (pid_m * num_chunks_in_block_Y) + chunk_id_y
                colwise_scale_inv_store_offsets = scale_offset_Y * stride_colwise_scale_inv_row + (offsets_X[None, :] * stride_colwise_scale_inv_col) 
                colwise_scale_inv_store_mask = (scale_offset_Y < colwise_scale_M) & (offsets_X < colwise_scale_N)[None, :]
                tl.store(colwise_scale_inv_ptr + colwise_scale_inv_store_offsets, biased_exponent_colwise, mask = colwise_scale_inv_store_mask)
                
                block_inverse_scale_colwise = exp2f_rcp_triton(biased_exponent_colwise)
                y_chunk_colwise_scaled = x_chunk * block_inverse_scale_colwise
                colwise_y_ptr_current_chunk = colwise_y_ptr + offsets_Y[:, None] * stride_rowwise_row + offsets_X[None, :] * stride_rowwise_col
                tl.store(colwise_y_ptr_current_chunk, y_chunk_colwise_scaled.to(colwise_y_ptr.type.element_ty), mask=mask)

##########################################
#### cast_transpose_mxfp4
##########################################

@triton.jit
def _cast_transpose_triton_mxfp4(
    x_ptr,
    rowwise_fp4_ptr,
    rowwise_scale_ptr,
    colwise_fp4_ptr,
    colwise_scale_ptr,
    stride_x_m,
    stride_x_n,
    stride_rowwise_fp4_m,
    stride_rowwise_fp4_n,
    stride_rowwise_scale_m,
    stride_rowwise_scale_n,
    stride_colwise_fp4_m,
    stride_colwise_fp4_n,
    stride_colwise_scale_m,
    stride_colwise_scale_n,
    M: tl.constexpr,
    N: tl.constexpr,
    rowwise_scale_N: tl.constexpr,
    rowwise_scale_M_pad: tl.constexpr,
    rowwise_scale_N_pad: tl.constexpr,
    colwise_scale_M: tl.constexpr,
    colwise_scale_N: tl.constexpr,
    colwise_scale_M_pad: tl.constexpr,
    colwise_scale_N_pad: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MXFP4_BLOCK_SIZE: tl.constexpr,
    USE_ROWWISE: tl.constexpr,
    USE_COLWISE: tl.constexpr,
    SHUFFLE_ROWWISE: tl.constexpr,
    SHUFFLE_COLWISE: tl.constexpr,
):
    """
    MXFP4 cast + transpose (rowwise + columnwise) following the MXFP8 fused pattern.

    Example to keep in mind:
        Input  (M, N)    = (4096, 6144) bf16
        Rowwise output   = (M, N/2) uint8  (two FP4 packed per byte)
        Colwise output   = (N, M/2) uint8

    Grid layout:
        BLOCK_M x BLOCK_N tile (default 128 x 128).
        Inside the tile we iterate over 32 x 32 MXFP4 blocks.

    Strides:
        - stride_x_m / stride_x_n point into the source matrix.
        - stride_rowwise_fp4_* index rowwise packed bytes.
        - stride_colwise_fp4_* index columnwise packed bytes.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    stride_x_m = tl.cast(stride_x_m, tl.int64)
    stride_x_n = tl.cast(stride_x_n, tl.int64)
    stride_rowwise_fp4_m = tl.cast(stride_rowwise_fp4_m, tl.int64)
    stride_rowwise_fp4_n = tl.cast(stride_rowwise_fp4_n, tl.int64)
    stride_colwise_fp4_m = tl.cast(stride_colwise_fp4_m, tl.int64)
    stride_colwise_fp4_n = tl.cast(stride_colwise_fp4_n, tl.int64)

    num_chunks_m = BLOCK_M // MXFP4_BLOCK_SIZE
    num_chunks_n = BLOCK_N // MXFP4_BLOCK_SIZE

    base_m = pid_m * BLOCK_M
    base_n = pid_n * BLOCK_N

    # Each BLOCK_M covers BLOCK_M / 32 MXFP4 row blocks.
    row_block_base = (base_m // MXFP4_BLOCK_SIZE)

    E8_BIAS = tl.constexpr(127)
    E2_BIAS = tl.constexpr(1)

    for chunk_m in range(num_chunks_m):
        offs_m = base_m + chunk_m * MXFP4_BLOCK_SIZE + tl.arange(0, MXFP4_BLOCK_SIZE)
        row_mask = offs_m < M

        for chunk_n in range(num_chunks_n):
            offs_n = base_n + chunk_n * MXFP4_BLOCK_SIZE + tl.arange(0, MXFP4_BLOCK_SIZE)
            col_mask = offs_n < N

            mask = row_mask[:, None] & col_mask[None, :]

            # Load a 32x32 bf16 tile (promoted to fp32) so both row/col passes reuse the same data. TODO @saraora to double check if this is necessary.
            #   offs_m = 128*k + [0..31]
            #   offs_n = 128*l + [0..31]
            # This chunk is reused for both rowwise and columnwise passes.
            x_chunk = tl.load(
                x_ptr + offs_m[:, None] * stride_x_m + offs_n[None, :] * stride_x_n,
                mask=mask,
                other=0.0,
            ).to(tl.float32)

            # ---------- Rowwise path ----------
            if USE_ROWWISE:
                # For each row in the current tile (base_m + row0), process the elements in [base_n : base_n + 31].
                # Compute one E8M0 scale per row (32 elements).
                amax_row = tl.max(tl.abs(x_chunk), axis=1, keep_dims=True)
                amax_row = amax_row.to(tl.int32, bitcast=True)
                amax_row = (amax_row + 0x200000).to(tl.uint32, bitcast=True) & 0xFF800000
                amax_row = amax_row.to(tl.float32, bitcast=True)
                scale_unbiased_row = tl.log2(amax_row).floor() - 2
                scale_unbiased_row = tl.clamp(scale_unbiased_row, min=-127, max=127)
                quant_scale_row = tl.exp2(-scale_unbiased_row)

                qx_row = x_chunk * quant_scale_row
                bs_row = scale_unbiased_row.to(tl.uint8) + 127

                qx_row_u32 = qx_row.to(tl.uint32, bitcast=True)
                s_row = qx_row_u32 & 0x80000000
                e_row = (qx_row_u32 >> 23) & 0xFF
                m_row = qx_row_u32 & 0x7FFFFF

                adjusted_row = tl.core.sub(E8_BIAS, e_row + 1, sanitize_overflow=False)
                m_row = tl.where(e_row < E8_BIAS, (0x400000 | (m_row >> 1)) >> adjusted_row, m_row)
                e_row = tl.maximum(e_row, E8_BIAS - E2_BIAS) - (E8_BIAS - E2_BIAS)

                e2m1_row = tl.minimum((((e_row << 2) | (m_row >> 21)) + 1) >> 1, 0x7)
                e2m1_row = ((s_row >> 28) | e2m1_row).to(tl.uint8)

                # Pack columns (C0,C1) -> byte0, (C2,C3) -> byte1, etc.
                row_pairs = tl.reshape(
                    e2m1_row, [MXFP4_BLOCK_SIZE, MXFP4_BLOCK_SIZE // 2, 2]
                )
                vals_even, vals_odd = tl.split(row_pairs)
                packed_row = vals_even | (vals_odd << 4)

                row_out_rows = offs_m
                row_out_cols = (
                    (pid_n * BLOCK_N) // 2
                    + chunk_n * (MXFP4_BLOCK_SIZE // 2)
                    + tl.arange(0, MXFP4_BLOCK_SIZE // 2)
                )
                row_store_mask = (
                    (row_out_rows < M)[:, None]
                    & (row_out_cols < (N // 2))[None, :]
                )

                tl.store(
                    rowwise_fp4_ptr
                    + row_out_rows[:, None] * stride_rowwise_fp4_m
                    + row_out_cols[None, :] * stride_rowwise_fp4_n,
                    packed_row,
                    mask=row_store_mask,
                )

                scale_offset_x = (pid_n * num_chunks_n) + chunk_n
                scale_rows = offs_m

                if SHUFFLE_ROWWISE:
                    # Rowwise shuffle matches AITER's e8m0_shuffle:
                    # view(sm//32, 2, 16, sn//8, 2, 4) -> permute(0, 3, 5, 2, 4, 1) -> view(sm, sn)
                    # where sm = M (rows), sn = N/32 (scale columns)
                    #
                    # For input (row=scale_rows, col=scale_offset_x):
                    #   i0 = row // 32
                    #   i1 = (row % 32) // 16
                    #   i2 = row % 16
                    #   i3 = col // 8
                    #   i4 = (col % 8) // 4
                    #   i5 = col % 4
                    # Output linear = i0*(sn//8*256) + i3*256 + i5*64 + i2*4 + i4*2 + i1
                    i0 = scale_rows[:, None] // 32
                    i1 = (scale_rows[:, None] % 32) // 16
                    i2 = scale_rows[:, None] % 16
                    i3 = scale_offset_x // 8
                    i4 = (scale_offset_x % 8) // 4
                    i5 = scale_offset_x % 4
                    
                    # rowwise_scale_N_pad is already (N/32) rounded up to multiple of 8
                    bs_offs = (
                        i0 * (rowwise_scale_N_pad // 8 * 256) +
                        i3 * 256 +
                        i5 * 64 +
                        i2 * 4 +
                        i4 * 2 +
                        i1
                    )
                    mask_valid = (scale_rows < M)[:, None] & (
                        scale_offset_x < rowwise_scale_N
                    )
                    mask_pad = (scale_rows < rowwise_scale_M_pad)[:, None] & (
                        scale_offset_x < rowwise_scale_N_pad
                    )
                    vals = tl.where(mask_valid, bs_row, 127)
                    tl.store(rowwise_scale_ptr + bs_offs, vals, mask=mask_pad)
                else:
                    scale_mask = (scale_rows < M)[:, None] & (
                        scale_offset_x < rowwise_scale_N
                    )
                    tl.store(
                        rowwise_scale_ptr
                        + scale_rows[:, None] * stride_rowwise_scale_m
                        + scale_offset_x * stride_rowwise_scale_n,
                        bs_row,
                        mask=scale_mask,
                    )

            # ---------- Columnwise path ----------
            if USE_COLWISE:
                # Treat columns as rows by transposing to reuse the same per-row logic.
                # Instead of manually transposing indices, view the tile transposed.
                x_col = tl.trans(x_chunk)
                amax_col = tl.max(tl.abs(x_col), axis=1, keep_dims=True)
                amax_col = amax_col.to(tl.int32, bitcast=True)
                amax_col = (amax_col + 0x200000).to(tl.uint32, bitcast=True) & 0xFF800000
                amax_col = amax_col.to(tl.float32, bitcast=True)
                scale_unbiased_col = tl.log2(amax_col).floor() - 2
                scale_unbiased_col = tl.clamp(scale_unbiased_col, min=-127, max=127)
                quant_scale_col = tl.exp2(-scale_unbiased_col)

                qx_col = x_col * quant_scale_col
                bs_col = scale_unbiased_col.to(tl.uint8) + 127

                qx_col_u32 = qx_col.to(tl.uint32, bitcast=True)
                s_col = qx_col_u32 & 0x80000000
                e_col = (qx_col_u32 >> 23) & 0xFF
                m_col = qx_col_u32 & 0x7FFFFF

                adjusted_col = tl.core.sub(E8_BIAS, e_col + 1, sanitize_overflow=False)
                m_col = tl.where(e_col < E8_BIAS, (0x400000 | (m_col >> 1)) >> adjusted_col, m_col)
                e_col = tl.maximum(e_col, E8_BIAS - E2_BIAS) - (E8_BIAS - E2_BIAS)

                e2m1_col = tl.minimum((((e_col << 2) | (m_col >> 21)) + 1) >> 1, 0x7)
                e2m1_col = ((s_col >> 28) | e2m1_col).to(tl.uint8)

                # After transpose, each row in x_col is one column from the original tile.
                col_pairs = tl.reshape(
                    e2m1_col, [MXFP4_BLOCK_SIZE, MXFP4_BLOCK_SIZE // 2, 2]
                )
                vals_even, vals_odd = tl.split(col_pairs)
                packed_col = vals_even | (vals_odd << 4)  # [cols, row_pairs]

                col_indices = (
                    base_n + chunk_n * MXFP4_BLOCK_SIZE + tl.arange(0, MXFP4_BLOCK_SIZE)
                )
                row_pairs = tl.arange(0, MXFP4_BLOCK_SIZE // 2)
                rowpair_base = (base_m // 2) + chunk_m * (MXFP4_BLOCK_SIZE // 2)
                rowpair_indices = rowpair_base + row_pairs

                # col_indices: [base_n + chunk_n*MXFP4_BLOCK_SIZE + i for i in range(MXFP4_BLOCK_SIZE)]
                # rowpair_indices: [base_m // 2 + chunk_m*(MXFP4_BLOCK_SIZE//2) + j for j in range(MXFP4_BLOCK_SIZE//2)]
                col_fp4_mask = (col_indices < N)[:, None] & (
                    rowpair_indices < (M // 2)
                )[None, :]

                # Store directly into the [N, M/2] layout expected by columnwise tensors.
                tl.store(
                    colwise_fp4_ptr
                    + col_indices[:, None] * stride_colwise_fp4_m
                    + rowpair_indices[None, :] * stride_colwise_fp4_n,
                    packed_col,
                    mask=col_fp4_mask,
                )

                scale_chunk = (pid_m * num_chunks_m) + chunk_m

                if SHUFFLE_COLWISE:
                    # Columnwise shuffle matches AITER's e8m0_shuffle:
                    # view(sm//32, 2, 16, sn//8, 2, 4) -> permute(0, 3, 5, 2, 4, 1) -> view(sm, sn)
                    # where sm = colwise_scale_M (N), sn = colwise_scale_N (M/32)
                    #
                    # For input (row=col_indices, col=scale_chunk):
                    #   i0 = row // 32
                    #   i1 = (row % 32) // 16
                    #   i2 = row % 16
                    #   i3 = col // 8
                    #   i4 = (col % 8) // 4
                    #   i5 = col % 4
                    # Output linear = i0*(sn//8*256) + i3*256 + i5*64 + i2*4 + i4*2 + i1
                    bs_col_1d = tl.reshape(bs_col, [MXFP4_BLOCK_SIZE])
                    i0 = col_indices // 32
                    i1 = (col_indices % 32) // 16
                    i2 = col_indices % 16
                    i3 = scale_chunk // 8
                    i4 = (scale_chunk % 8) // 4
                    i5 = scale_chunk % 4
                    
                    # colwise_scale_N_pad is already (M/32) rounded up to multiple of 8
                    bs_offs = (
                        i0 * (colwise_scale_N_pad // 8 * 256) +
                        i3 * 256 +
                        i5 * 64 +
                        i2 * 4 +
                        i4 * 2 +
                        i1
                    )
                    mask_valid = (col_indices < colwise_scale_M) & (
                        scale_chunk < colwise_scale_N
                    )
                    mask_pad = (col_indices < colwise_scale_M_pad) & (
                        scale_chunk < colwise_scale_N_pad
                    )
                    vals = tl.where(mask_valid, bs_col_1d, 127)
                    tl.store(colwise_scale_ptr + bs_offs, vals, mask=mask_pad)
                else:
                    # Simple row-major layout: each column has scale_chunk entries along the N-dimension.
                    scale_mask = (col_indices < colwise_scale_M)[:, None] & (
                        scale_chunk < colwise_scale_N
                    )
                    tl.store(
                        colwise_scale_ptr
                        + col_indices[:, None] * stride_colwise_scale_m
                        + scale_chunk * stride_colwise_scale_n,
                        bs_col,
                        mask=scale_mask,
                    )

@triton.jit
def _dequantize_mxfp8_triton(
    x_ptr, y_ptr,
    stride_row, stride_col, 
    n_rows, n_cols, 
    scale_inv_ptr, stride_scale_inv_row, stride_scale_inv_col,
    scale_n_rows, scale_n_cols,
    BLOCK_X: tl.constexpr, BLOCK_Y: tl.constexpr, GROUP_Y: tl.constexpr, USE_ROWWISE_SCALING: tl.constexpr, MXFP8_BLOCK_SCALING_SIZE: tl.constexpr):
   
    pid = tl.program_id(0)

    num_pid_along_Y = tl.cdiv(n_rows, BLOCK_Y)
    num_pid_along_X = tl.cdiv(n_cols, BLOCK_X)
    num_pid_in_group = GROUP_Y * num_pid_along_X

    group_id = pid // num_pid_in_group
    group_size = min(num_pid_along_Y - group_id * GROUP_Y, GROUP_Y)
    pid_m = group_id * GROUP_Y + ((pid % num_pid_in_group) % group_size)
    pid_n = (pid % num_pid_in_group) // group_size

    global_offset_Y_base = pid_m.to(tl.int64) * BLOCK_Y
    global_offset_X_base = pid_n.to(tl.int64) * BLOCK_X
    
    num_chunks_in_block_Y = BLOCK_Y // MXFP8_BLOCK_SCALING_SIZE
    num_chunks_in_block_X = BLOCK_X // MXFP8_BLOCK_SCALING_SIZE

    for chunk_id_y in range(0, num_chunks_in_block_Y):
        offsets_Y = global_offset_Y_base + chunk_id_y * MXFP8_BLOCK_SCALING_SIZE + tl.arange(0, MXFP8_BLOCK_SCALING_SIZE)
        for chunk_id_x in range(0, num_chunks_in_block_X):
            offsets_X = global_offset_X_base  + chunk_id_x * MXFP8_BLOCK_SCALING_SIZE + tl.arange(0, MXFP8_BLOCK_SCALING_SIZE)
            x_ptr_current_chunk = x_ptr + offsets_Y[:, None] * stride_row + offsets_X[None, :] * stride_col
            mask = (offsets_Y < n_rows)[:, None] & (offsets_X < n_cols)[None, :]
            x_chunk = tl.load(x_ptr_current_chunk, mask=mask)

            if USE_ROWWISE_SCALING:
                scale_offset_X = (pid_n * num_chunks_in_block_X) + chunk_id_x
                scale_inv_store_offsets = (offsets_Y[:, None] * stride_scale_inv_row) + scale_offset_X * stride_scale_inv_col 
                scale_inv_store_mask = (offsets_Y < scale_n_rows)[:, None] & (scale_offset_X < scale_n_cols)
            else:
                scale_offset_Y = (pid_m * num_chunks_in_block_Y) + chunk_id_y
                scale_inv_store_offsets = scale_offset_Y * stride_scale_inv_row + (offsets_X[None, :] * stride_scale_inv_col) 
                scale_inv_store_mask = (scale_offset_Y < scale_n_rows) & (offsets_X < scale_n_cols)[None, :]
                
            biased_exponent = tl.load(scale_inv_ptr + scale_inv_store_offsets, mask=scale_inv_store_mask, other=127)
            block_scale = tl.exp2(biased_exponent.to(tl.float32) - 127)
            y_chunk_scaled = x_chunk.to(tl.float32) * block_scale
            y_ptr_current_chunk = y_ptr + offsets_Y[:, None] * stride_row + offsets_X[None, :] * stride_col
            tl.store(y_ptr_current_chunk, y_chunk_scaled.to(y_ptr.type.element_ty), mask=mask)

# Reshapes input of any given shape to 2D for processing, 
# then uses the Triton kernel to perform casting and transposition efficiently.
def te_cast_transpose_noop_triton(input, noop_flag, input_scale, cast_out, trans_out, amax_out, scale_inv_out, otype):

    row_length = input.shape[-1] if len(input.shape) > 0 else 1
    num_rows = input.numel() // row_length
    input_2d_view = input.reshape(num_rows, row_length)
    cast_out_2d_view = cast_out.reshape(num_rows, row_length)
    trans_out_2d_view =  trans_out.reshape(row_length, num_rows)

    input_stride_M = input_2d_view.stride(0)
    input_stride_N = input_2d_view.stride(1)

    trans_out_stride_M = trans_out_2d_view.stride(0)
    trans_out_stride_N = trans_out_2d_view.stride(1)
    
    tl_dtype = te_dtype_to_triton_dtype(otype)
    
    if noop_flag.nelement() > 0:
        use_noop = True
    else:
        use_noop = False
    
    grid = lambda META: (triton.cdiv(num_rows, META['BLOCK_M']) * triton.cdiv(row_length, META['BLOCK_N']),)
    _cast_transpose_triton[grid](input_2d_view, noop_flag, triton.reinterpret(cast_out_2d_view, tl_dtype), triton.reinterpret(trans_out_2d_view, tl_dtype), input_stride_M, input_stride_N, trans_out_stride_M, trans_out_stride_N, num_rows, row_length, input_scale, amax_out, scale_inv_out, get_fp8_max(otype), use_noop)

def te_cast_transpose_mxfp8_triton(input, out, noop_flag=None):
    row_length = input.shape[-1] if len(input.shape) > 0 else 1
    num_rows = input.numel() // row_length
    input_2d_view = input.reshape(num_rows, row_length)
    out_metadata = out.get_metadata()

    USE_ROWWISE_SCALING = out_metadata["rowwise_data"] is not None
    USE_COLWISE_SCALING = out_metadata["columnwise_data"] is not None
    
    fp8_dtype = out_metadata["fp8_dtype"]
    tl_dtype = te_dtype_to_triton_dtype(fp8_dtype)
    
    rowwise_y_ptr, rowwise_scale_inv_ptr = None, None
    rowwise_scale_M, rowwise_scale_N = 1, 1
    rowwise_scale_stride_M, rowwise_scale_stride_N = 1, 1
    if USE_ROWWISE_SCALING:
        rowwise_y_ptr = out_metadata["rowwise_data"].reshape(num_rows, row_length)
        rowwise_y_ptr = triton.reinterpret(rowwise_y_ptr, tl_dtype)
        rowwise_scale_inv_ptr = out_metadata["rowwise_scale_inv"]
        rowwise_scale_M, rowwise_scale_N = rowwise_scale_inv_ptr.shape
        rowwise_scale_stride_M, rowwise_scale_stride_N = rowwise_scale_inv_ptr.stride(0), rowwise_scale_inv_ptr.stride(1)
    
    colwise_y_ptr, colwise_scale_inv_ptr = None, None
    colwise_scale_M, colwise_scale_N = 1, 1
    colwise_scale_stride_M, colwise_scale_stride_N = 1, 1
    if USE_COLWISE_SCALING:
        colwise_y_ptr = out_metadata["columnwise_data"].reshape(num_rows, row_length)
        colwise_y_ptr = triton.reinterpret(colwise_y_ptr, tl_dtype)
        colwise_scale_inv_ptr = out_metadata["columnwise_scale_inv"]
        colwise_scale_M, colwise_scale_N = colwise_scale_inv_ptr.shape
        colwise_scale_stride_M, colwise_scale_stride_N = colwise_scale_inv_ptr.stride(0), colwise_scale_inv_ptr.stride(1)
    
    
    BLOCK_X = 64
    BLOCK_Y = 64
    GROUP_Y = MXFP8_BLOCK_SCALING_SIZE
    max_fp8 = get_fp8_max(fp8_dtype)
    grid = lambda META: (triton.cdiv(num_rows, META['BLOCK_Y']) * triton.cdiv(row_length, META['BLOCK_X']),)
    _cast_transpose_triton_mxfp8[grid](
        input_2d_view, rowwise_y_ptr, colwise_y_ptr, 
        input_2d_view.stride(0), input_2d_view.stride(1), 
        num_rows, row_length, 
        rowwise_scale_inv_ptr, rowwise_scale_stride_M, rowwise_scale_stride_N,
        rowwise_scale_M, rowwise_scale_N,
        colwise_scale_inv_ptr, colwise_scale_stride_M, colwise_scale_stride_N,
        colwise_scale_M, colwise_scale_N,
        max_fp8, BLOCK_X, BLOCK_Y, GROUP_Y, MXFP8_BLOCK_SCALING_SIZE, USE_ROWWISE_SCALING, USE_COLWISE_SCALING)

def te_cast_transpose_mxfp4_triton(
    input: torch.Tensor,
    rowwise_fp4_out: Optional[torch.Tensor] = None,
    rowwise_scale_out: Optional[torch.Tensor] = None,
    colwise_fp4_out: Optional[torch.Tensor] = None,
    colwise_scale_out: Optional[torch.Tensor] = None,
    shuffle_rowwise: bool = True,
    shuffle_colwise: bool = True,
) -> tuple:
    """
    Fused MXFP4 quantization with optional transpose
    
    Performs quantization for both rowwise and columnwise layouts
    
    Args:
        input: Input tensor [M, N] in BF16/FP16
        rowwise_fp4_out: Optional pre-allocated rowwise FP4 output [M, N/2]
        rowwise_scale_out: Optional pre-allocated rowwise E8M0 scales
        colwise_fp4_out: Optional pre-allocated colwise FP4 output [N, M/2]
        colwise_scale_out: Optional pre-allocated colwise E8M0 scales
        shuffle_rowwise: Whether to apply shuffle permutation to rowwise scales
        shuffle_colwise: Whether to apply shuffle permutation to colwise scales
    
    Returns:
        (rowwise_fp4, rowwise_scale, colwise_fp4, colwise_scale)
    """
    # Reshape input to 2D
    original_shape = input.shape
    if input.dim() > 2:
        input = input.view(-1, input.shape[-1])
    if input.dim() != 2:
        raise ValueError(f"Input must be 2D or reshapeable to 2D, got shape {original_shape}")
    
    M, N = input.shape
    MXFP4_BLOCK_SIZE = 32
    BLOCK_M = 128
    BLOCK_N = 128
    
    # Validate dimensions
    assert N % MXFP4_BLOCK_SIZE == 0, f"N={N} must be divisible by {MXFP4_BLOCK_SIZE}"
    
    device = input.device
    USE_ROWWISE = rowwise_fp4_out is not None or colwise_fp4_out is None
    USE_COLWISE = colwise_fp4_out is not None
    
    # Allocate rowwise outputs (matching AITER layout)
    if USE_ROWWISE:
        if rowwise_fp4_out is None:
            rowwise_fp4_out = torch.empty(M, N // 2, dtype=torch.uint8, device=device)
        
        scaleN_row = triton.cdiv(N, MXFP4_BLOCK_SIZE)
        if rowwise_scale_out is None:
            if shuffle_rowwise:
                # AITER shuffled layout
                scaleM = triton.cdiv(M, 32) * 32
                scaleN = triton.cdiv(scaleN_row, 8) * 8
                rowwise_scale_out = torch.empty(
                    triton.cdiv(M, 256) * 256, scaleN,
                    dtype=torch.uint8, device=device
                )
            else:
                # Non-shuffled layout
                rowwise_scale_out = torch.empty(M, scaleN_row, dtype=torch.uint8, device=device)
        
        scaleM_pad = triton.cdiv(M, 32) * 32
        scaleN_pad = triton.cdiv(scaleN_row, 8) * 8
    else:
        scaleN_row = 1
        scaleM_pad = scaleN_pad = 1
    
    colwise_scale_tmp = None
    kernel_colwise_scale = None
    kernel_colwise_scale_M = kernel_colwise_scale_N = 1
    kernel_colwise_scale_M_pad = kernel_colwise_scale_N_pad = 1

    # Allocate columnwise outputs (transposed)
    if USE_COLWISE:
        if colwise_fp4_out is None:
            colwise_fp4_out = torch.empty(N, M // 2, dtype=torch.uint8, device=device)
        
        scaleN_colwise_valid = triton.cdiv(M, MXFP4_BLOCK_SIZE)
        if colwise_scale_out is None:
            if shuffle_colwise:
                # AITER shuffled layout for colwise
                scaleM_colwise_pad = triton.cdiv(N, 32) * 32
                scaleN_colwise_pad = triton.cdiv(scaleN_colwise_valid, 8) * 8
                colwise_scale_out = torch.empty(
                    triton.cdiv(N, 256) * 256, scaleN_colwise_pad,
                    dtype=torch.uint8, device=device
                )
            else:
                # Non-shuffled layout
                colwise_scale_out = torch.empty(N, scaleN_colwise_valid, dtype=torch.uint8, device=device)
        
        if shuffle_colwise:
            scaleM_colwise_pad = triton.cdiv(N, 256) * 256
            scaleN_colwise_pad = triton.cdiv(scaleN_colwise_valid, 8) * 8
        else:
            scaleM_colwise_pad = N
            scaleN_colwise_pad = scaleN_colwise_valid

        if shuffle_colwise:
            # Allocate padded temporary tensor for shuffled output
            colwise_scale_tmp = torch.empty(
                scaleM_colwise_pad,
                scaleN_colwise_pad,
                dtype=torch.uint8,
                device=device,
            )
            kernel_colwise_scale = colwise_scale_tmp
            kernel_colwise_scale_M = N  # Valid (non-padded) dimension
            kernel_colwise_scale_N = scaleN_colwise_valid  # Valid (non-padded) dimension
            kernel_colwise_scale_M_pad = scaleM_colwise_pad
            kernel_colwise_scale_N_pad = scaleN_colwise_pad
        else:
            kernel_colwise_scale = colwise_scale_out
            kernel_colwise_scale_M = colwise_scale_out.shape[0]
            kernel_colwise_scale_N = colwise_scale_out.shape[1]
            kernel_colwise_scale_M_pad = scaleM_colwise_pad
            kernel_colwise_scale_N_pad = scaleN_colwise_pad
    else:
        scaleM_colwise_pad = scaleN_colwise_pad = 1
        kernel_colwise_scale = colwise_scale_out
    
    # Ensure input is contiguous
    if not input.is_contiguous():
        input = input.contiguous()
    
    # Launch kernel with (M_blocks, N_blocks)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    _cast_transpose_triton_mxfp4[grid](
        input,
        rowwise_fp4_out if USE_ROWWISE else None,
        rowwise_scale_out if USE_ROWWISE else None,
        colwise_fp4_out if USE_COLWISE else None,
        kernel_colwise_scale if USE_COLWISE else None,
        input.stride(0), input.stride(1),
        rowwise_fp4_out.stride(0) if USE_ROWWISE else 1,
        rowwise_fp4_out.stride(1) if USE_ROWWISE else 1,
        rowwise_scale_out.stride(0) if USE_ROWWISE else 1,
        rowwise_scale_out.stride(1) if USE_ROWWISE else 1,
        colwise_fp4_out.stride(0) if USE_COLWISE else 1,
        colwise_fp4_out.stride(1) if USE_COLWISE else 1,
        kernel_colwise_scale.stride(0) if USE_COLWISE else 1,
        kernel_colwise_scale.stride(1) if USE_COLWISE else 1,
        M=M,
        N=N,
        rowwise_scale_N=scaleN_row,
        rowwise_scale_M_pad=scaleM_pad,
        rowwise_scale_N_pad=scaleN_pad,
        colwise_scale_M=kernel_colwise_scale_M,
        colwise_scale_N=kernel_colwise_scale_N,
        colwise_scale_M_pad=kernel_colwise_scale_M_pad,
        colwise_scale_N_pad=kernel_colwise_scale_N_pad,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        MXFP4_BLOCK_SIZE=MXFP4_BLOCK_SIZE,
        USE_ROWWISE=USE_ROWWISE,
        USE_COLWISE=USE_COLWISE,
        SHUFFLE_ROWWISE=shuffle_rowwise,
        SHUFFLE_COLWISE=shuffle_colwise,
    )
    
    # Copy shuffled columnwise scales to output tensor (trim padding)
    if USE_COLWISE and shuffle_colwise:
        colwise_scale_out[:N, :scaleN_colwise_valid] = kernel_colwise_scale[:N, :scaleN_colwise_valid]
    
    return rowwise_fp4_out, rowwise_scale_out, colwise_fp4_out, colwise_scale_out

def te_dequantize_mxfp8_triton(input, dtype):
    input_metadata = input.get_metadata()
    use_rowwise_scaling = input_metadata["rowwise_data"] is not None
    x_ptr = None
    scale_inv_ptr = None
    
    if use_rowwise_scaling:
        x_ptr = input_metadata["rowwise_data"]
        row_length = x_ptr.shape[-1] if len(x_ptr.shape) > 0 else 1
        num_rows = x_ptr.numel() // row_length
        x_ptr = x_ptr.reshape(num_rows, row_length)
        scale_inv_ptr = input_metadata["rowwise_scale_inv"]
    else:
        x_ptr = input_metadata["columnwise_data"]
        row_length = x_ptr.shape[-1] if len(x_ptr.shape) > 0 else 1
        num_rows = x_ptr.numel() // row_length
        x_ptr = x_ptr.reshape(num_rows, row_length)
        scale_inv_ptr = input_metadata["columnwise_scale_inv"]
    
    fp8_dtype = input_metadata["fp8_dtype"]
    scale_M, scale_N = scale_inv_ptr.shape
    dtype = te_dtype_to_torch_dtype(dtype)
    out = torch.zeros(input.shape, dtype=dtype, device=x_ptr.device)

    BLOCK_X = 64
    BLOCK_Y = 64
    GROUP_Y = 4
    tl_dtype = te_dtype_to_triton_dtype(fp8_dtype)

    grid = lambda META: (triton.cdiv(num_rows, META['BLOCK_Y']) * triton.cdiv(row_length, META['BLOCK_X']),)
    _dequantize_mxfp8_triton[grid](
    triton.reinterpret(x_ptr, tl_dtype), out,
    x_ptr.stride(0), x_ptr.stride(1), 
    num_rows, row_length, 
    scale_inv_ptr, scale_inv_ptr.stride(0), scale_inv_ptr.stride(1),
    scale_M, scale_N,
    BLOCK_X, BLOCK_Y, GROUP_Y, use_rowwise_scaling, MXFP8_BLOCK_SCALING_SIZE)

    return out

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
def _transpose_triton_dbias(A, C, T, stride_am, stride_an, stride_bn, stride_bm, M, N, scale_ptr, amax_ptr, scale_inv_ptr, partial_dbias, fp8_max: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    scale = tl.load(scale_ptr)

    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size
    
    rm = pid_m.to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n.to(tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N)
    A = A + rm[:, None] * stride_am + rn[None, :] * stride_an
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    a = tl.load(A, mask=mask, other=0.)
    a = a.to(tl.float32)

    partial_sum_a = tl.sum(a, axis=0)
    partial_dbias = partial_dbias + pid_m.to(tl.int64) * stride_am + rn * stride_an
    tl.store(partial_dbias, partial_sum_a, mask=(rn<N))

    scaled_a = a * scale
    scaled_a = tl.clamp(scaled_a, -fp8_max, fp8_max)
    fp8_a = scaled_a.to(C.type.element_ty)
    C = C + rm[:, None] * stride_am + rn[None, :] * stride_an
    tl.store(C, fp8_a, mask=mask)
    
    # rematerialize to save registers
    rm = pid_m.to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n.to(tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N)
    T = T + rm[:, None] * stride_bm + rn[None, :] * stride_bn
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    tl.store(T, fp8_a, mask=mask)
    amax = tl.max(tl.abs(a))
    tl.atomic_max(amax_ptr, amax, sem='relaxed')
    if pid == 0:
        scale_inv_out = tl.fdiv(1.0, scale)
        tl.store(scale_inv_ptr, scale_inv_out)

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
    rn = pid.to(tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N)
    dbias_reg = tl.zeros((BLOCK_N,), tl.float32)
    rm = tl.arange(0, BLOCK_M)
    for i in range(iters_m):
        #rm = i * BLOCK_M + tl.arange(0, BLOCK_M)
        A_ptr = A + rm[:, None] * stride_am + rn[None, :] * stride_an
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

def te_cast_transpose_dbias_triton(input, input_scale, amax_out, scale_inv_out, otype):
    M, N = input.shape
    cast_out = torch.empty(M, N, dtype=torch.uint8, device='cuda')
    trans_out = torch.empty(N, M, dtype=torch.uint8, device='cuda')
    dbias_out = torch.empty(N, dtype=input.dtype, device='cuda')

    if M == 0 or N == 0:
        return dbias_out.zero_(), cast_out, trans_out

    MIN_BLOCK_M = 64 ## This needs to be changed if minimum block size m changed
    partial_dbias = torch.empty(triton.cdiv(M, MIN_BLOCK_M), N, dtype=torch.float32, device='cuda')

    assert trans_out.size(0) == N and trans_out.size(1) == M
    assert input.stride(0) == 1 or input.stride(1) == 1
    assert trans_out.stride(0) == 1 or trans_out.stride(1) == 1

    tl_dtype = te_dtype_to_triton_dtype(otype)
    
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    _transpose_triton_dbias[grid](input, triton.reinterpret(cast_out, tl_dtype), triton.reinterpret(trans_out, tl_dtype), input.stride(0), input.stride(1), trans_out.stride(0), trans_out.stride(1), M, N, input_scale, amax_out, scale_inv_out, partial_dbias, get_fp8_max(otype))
    best_config = _transpose_triton_dbias.best_config
    block_m_1 = int(best_config.kwargs['BLOCK_M'])

    dbias_out = reduce_dbias_kernel(partial_dbias[0:triton.cdiv(M, block_m_1)], input.dtype)
    return dbias_out, cast_out, trans_out

    
