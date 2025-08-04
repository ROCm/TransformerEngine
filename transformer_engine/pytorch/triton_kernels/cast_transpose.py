# Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

import torch

from ..constants import MXFP8_BLOCK_SCALING_SIZE
import transformer_engine_torch as tex
import triton
import triton.language as tl
from .common import (
    te_dtype_to_triton_dtype,
    get_fp8_max,
)
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

    
