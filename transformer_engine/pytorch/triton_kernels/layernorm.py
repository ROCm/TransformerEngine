# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information


from itertools import product

import triton
import triton.language as tl

def get_autotune_config(safe_tuning=False, full_tuning_space=False):
    if full_tuning_space:
        tuning_space = product([1, 2, 4], [4, 8, 16])
    else:
        tuning_space = [(1, 8), (1, 16), (2, 16), (4, 4), (4, 8), (4, 16)]
    if safe_tuning:
        tuning_space = [(w, nw) for (w, nw) in tuning_space if w <= 2 and nw <= 8]
    return [
        triton.Config({"waves_per_eu": waves_per_eu}, num_warps=num_warps, num_stages=1)
        for waves_per_eu, num_warps in tuning_space
    ]


@triton.jit
def _layernorm_fwd_triton_impl(
    input_ptr,
    output_ptr,
    g_ptr,
    b_ptr,
    mean_ptr,
    rsigma_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    epsilon,
    q_amax_ptr,
    q_scale_ptr,
    scale_inv_ptr,
    out_transpose_ptr,
    out_transpose_stride,
    ZERO_CENTERED_GAMMA: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    IS_FP8: tl.constexpr,
    APPLY_ATOMIC: tl.constexpr,
    PERSISTENT: tl.constexpr,
    FP8_MAX: tl.constexpr,
    MAKE_TRANSPOSE: tl.constexpr
):

    # Enable the transpose cache only in FP8 mode.
    tl.static_assert(not MAKE_TRANSPOSE or IS_FP8, "Transpose cache requires fp8 data type.")

    # program id
    pid = tl.program_id(0)
    num_tiles = tl.num_programs(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    if PERSISTENT:
        rows_per_tile = n_rows // num_tiles
        if pid < n_rows % num_tiles:
            rows_per_tile += 1
            start_row = rows_per_tile * pid
        else:
            start_row = (rows_per_tile * pid) + (n_rows % num_tiles)
    else:
        rows_per_tile = 1
        start_row = pid

    if IS_FP8:
        scale = tl.load(q_scale_ptr)
        amax = 0.0

    for row_idx in range(start_row, start_row + rows_per_tile):
        x_ptr_start = input_ptr + (row_idx * input_row_stride)
        y_ptr_start = output_ptr + (row_idx * output_row_stride)

        n_cols_blks = tl.cdiv(n_cols, BLOCK_SIZE) - 1

        # calculate mean
        _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for blk_idx in range(0, n_cols_blks):
            cols = blk_idx * BLOCK_SIZE + col_offsets
            x_block = tl.load(x_ptr_start + cols).to(tl.float32)  # Unmasked loads
            _mean += x_block

        # For last iteration, do masked load
        cols = n_cols_blks * BLOCK_SIZE + col_offsets
        x_block = tl.load(x_ptr_start + cols, mask=cols < n_cols, other=0.0).to(tl.float32)
        _mean += x_block
        mean = tl.sum(_mean, axis=0) / n_cols

        # variance
        _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for blk_idx in range(0, n_cols_blks):
            cols = blk_idx * BLOCK_SIZE + col_offsets
            x_block = tl.load(x_ptr_start + cols).to(tl.float32)  # Unmasked loads
            x_block = x_block - mean
            _var += x_block * x_block

        # For last iteration, do masked load
        cols = n_cols_blks * BLOCK_SIZE + col_offsets
        x_block = tl.load(x_ptr_start + cols, mask=cols < n_cols, other=0.0).to(tl.float32)
        x_block = tl.where(cols < n_cols, x_block - mean, 0.0)
        _var += x_block * x_block

        var = tl.sum(_var, axis=0) / n_cols
        rstd = tl.rsqrt(var + epsilon)

        # Write mean / rstd
        tl.store(mean_ptr + row_idx, mean)
        tl.store(rsigma_ptr + row_idx, rstd)

        # Normalize and store
        for blk_idx in range(0, n_cols_blks):
            cols = blk_idx * BLOCK_SIZE + col_offsets
            w_block = tl.load(g_ptr + cols).to(tl.float32)
            b_block = tl.load(b_ptr + cols).to(tl.float32)
            x_block = tl.load(x_ptr_start + cols).to(tl.float32)
            if ZERO_CENTERED_GAMMA:
                w_block += 1
            y_block = (x_block - mean) * rstd
            y_block = y_block * w_block + b_block
            if IS_FP8:
                amax_temp = tl.max(tl.abs(y_block), axis=-1)
                amax = amax_temp if amax_temp > amax else amax
                y_block = y_block * scale
                y_block = tl.clamp(y_block, -FP8_MAX, FP8_MAX)
            y_block = y_block.to(output_ptr.type.element_ty)
            tl.store(y_ptr_start + cols, y_block)
            if MAKE_TRANSPOSE:
                output_t_ptrs = out_transpose_ptr + cols * out_transpose_stride + row_idx
                tl.store(output_t_ptrs, y_block)

        # For last iteration, do masked load and store
        cols = n_cols_blks * BLOCK_SIZE + col_offsets
        mask = cols < n_cols
        w_block = tl.load(g_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        b_block = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        x_block = tl.load(x_ptr_start + cols, mask=mask, other=0.0).to(tl.float32)
        if ZERO_CENTERED_GAMMA:
            w_block += 1
        y_block = (x_block - mean) * rstd
        y_block = y_block * w_block + b_block
        if IS_FP8:
            amax_temp = tl.max(tl.abs(y_block), axis=-1)
            amax = amax_temp if amax_temp > amax else amax
            y_block = y_block * scale
            y_block = tl.clamp(y_block, -FP8_MAX, FP8_MAX)
        y_block = y_block.to(output_ptr.type.element_ty)
        tl.store(y_ptr_start + cols, y_block, mask=mask)
        if MAKE_TRANSPOSE:
            output_t_ptrs = out_transpose_ptr + cols * out_transpose_stride + row_idx
            tl.store(output_t_ptrs, y_block, mask=mask)

    if IS_FP8:
        if pid == 0:
            scale_inv = tl.fdiv(1.0, scale)
            tl.store(scale_inv_ptr, scale_inv)
        if APPLY_ATOMIC:
            tl.atomic_max(q_amax_ptr, amax, sem="relaxed")
        else:
            tl.store(q_amax_ptr + pid, amax)

autotune_dec = triton.autotune(configs=get_autotune_config(safe_tuning=True), key=["n_rows", "n_cols"], use_cuda_graph=True)
_layernorm_fwd_triton = autotune_dec(_layernorm_fwd_triton_impl)

@triton.jit
def _layernorm_fwd_reduce_triton(
    amax_input_ptr,
    amax_output_ptr,
    n_rows,
    BLOCK_SIZE: tl.constexpr,
):

    # program id
    pid = tl.program_id(0)

    amax_offs = tl.arange(0, BLOCK_SIZE) + (pid * BLOCK_SIZE)
    amax_input_ptrs = amax_input_ptr + amax_offs
    amax_mask = amax_offs < n_rows
    _amax = tl.load(amax_input_ptrs, mask=amax_mask, other=0.0)

    amax = tl.max(_amax, axis=-1)

    tl.atomic_max(amax_output_ptr, amax, sem="relaxed")

@triton.jit
def _layernorm_bwd_dx_fused_triton(
    DX,  # pointer to the input gradient
    DY,  # pointer to the output gradient
    DW,  # pointer to the partial sum of weights gradient
    DB,  # pointer to the partial sum of biases gradient
    X,  # pointer to the input
    W,  # pointer to the weights
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    ZERO_CENTERED_GAMMA: tl.constexpr,
    NUM_ROWS: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    USE_BLOCKED: tl.constexpr,
    IGNORE_DW_DB: tl.constexpr = False,
):
    # Map the program id to the elements of X, DX, and DY it should compute.
    pid = tl.program_id(0)
    tile_num = tl.num_programs(0)
    rows_per_tile = NUM_ROWS // tile_num
    if pid < NUM_ROWS % tile_num:
        rows_per_tile += 1

    if USE_BLOCKED:
        # Blocked approach:

        col_offsets = tl.arange(0, BLOCK_SIZE_N)
        num_col_blocks = tl.cdiv(N, BLOCK_SIZE_N) - 1
        row = pid

        for _ in range(0, rows_per_tile):
            # Load row statistics:
            mean = tl.load(Mean + row)
            rstd = tl.load(Rstd + row)

            # Accumulate c1 and c2 sums:

            x_row_ptr = X + row * stride
            dy_row_ptr = DY + row * stride

            c1 = 0.0
            c2 = 0.0

            for block_idx in tl.range(0, num_col_blocks):
                cols = block_idx * BLOCK_SIZE_N + col_offsets

                x = tl.load(x_row_ptr + cols).to(tl.float32)
                dy = tl.load(dy_row_ptr + cols).to(tl.float32)
                w = tl.load(W + cols).to(tl.float32)

                xhat = (x - mean) * rstd
                if ZERO_CENTERED_GAMMA:
                    w += 1
                wdy = w * dy
                c1 += tl.sum(xhat * wdy, axis=0)
                c2 += tl.sum(wdy, axis=0)

            cols = num_col_blocks * BLOCK_SIZE_N + col_offsets
            mask = cols < N

            x = tl.load(x_row_ptr + cols, mask=mask, other=0).to(tl.float32)
            dy = tl.load(dy_row_ptr + cols, mask=mask, other=0).to(tl.float32)
            w = tl.load(W + cols, mask=mask, other=0).to(tl.float32)

            xhat = (x - mean) * rstd
            if ZERO_CENTERED_GAMMA:
                w += 1
            wdy = w * dy
            wdy = tl.where(mask, wdy, 0)
            c1 += tl.sum(xhat * wdy, axis=0)
            c2 += tl.sum(wdy, axis=0)

            c1 /= N
            c2 /= N

            # Compute dx and partial sums for dw and db:

            dx_row_ptr = DX + row * stride
            if not IGNORE_DW_DB:
                dw_row_ptr = DW + pid * N
                db_row_ptr = DB + pid * N

            for block_idx in tl.range(0, num_col_blocks):
                cols = block_idx * BLOCK_SIZE_N + col_offsets

                x = tl.load(x_row_ptr + cols).to(tl.float32)
                dy = tl.load(dy_row_ptr + cols).to(tl.float32)
                w = tl.load(W + cols).to(tl.float32)

                xhat = (x - mean) * rstd
                if ZERO_CENTERED_GAMMA:
                    w += 1
                wdy = w * dy

                dx = (wdy - (xhat * c1 + c2)) * rstd
                tl.store(dx_row_ptr + cols, dx.to(DX.type.element_ty))
                if not IGNORE_DW_DB:
                    partial_dw = dy * xhat
                    dw_ptrs = dw_row_ptr + cols
                    partial_dw += tl.load(dw_ptrs).to(tl.float32)
                    tl.store(dw_ptrs, partial_dw.to(DW.type.element_ty))

                    partial_db = dy
                    db_ptrs = db_row_ptr + cols
                    partial_db += tl.load(db_ptrs).to(tl.float32)
                    tl.store(db_ptrs, partial_db.to(DB.type.element_ty))

            cols = num_col_blocks * BLOCK_SIZE_N + col_offsets
            mask = cols < N

            x = tl.load(x_row_ptr + cols, mask=mask, other=0).to(tl.float32)
            dy = tl.load(dy_row_ptr + cols, mask=mask, other=0).to(tl.float32)
            w = tl.load(W + cols, mask=mask, other=0).to(tl.float32)

            xhat = (x - mean) * rstd
            if ZERO_CENTERED_GAMMA:
                w += 1
            wdy = w * dy

            dx = (wdy - (xhat * c1 + c2)) * rstd
            tl.store(dx_row_ptr + cols, dx.to(DX.type.element_ty), mask=mask)
            if not IGNORE_DW_DB:
                partial_dw = dy * xhat
                dw_ptrs = dw_row_ptr + cols
                partial_dw += tl.load(dw_ptrs, mask=mask).to(tl.float32)
                tl.store(dw_ptrs, partial_dw.to(DW.type.element_ty), mask=mask)

                partial_db = dy
                db_ptrs = db_row_ptr + cols
                partial_db += tl.load(db_ptrs, mask=mask).to(tl.float32)
                tl.store(db_ptrs, partial_db.to(DB.type.element_ty), mask=mask)

            # Advance to next row.
            row += tile_num

    else:
        # Unblocked approach:

        cols = tl.arange(0, BLOCK_SIZE_N)
        mask = cols < N
        row = pid
        if not IGNORE_DW_DB:
            dw_row = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
            db_row = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)

        for _ in range(0, rows_per_tile):
            # Compute pointers:
            x_ptrs = X + row * stride
            dy_ptrs = DY + row * stride
            dx_ptrs = DX + row * stride

            # Load data to SRAM:
            x = tl.load(x_ptrs + cols, mask=mask, other=0).to(tl.float32)
            dy = tl.load(dy_ptrs + cols, mask=mask, other=0).to(tl.float32)
            w = tl.load(W + cols, mask=mask, other=0).to(tl.float32)
            mean = tl.load(Mean + row)
            rstd = tl.load(Rstd + row)

            # Compute dx:
            xhat = (x - mean) * rstd
            if ZERO_CENTERED_GAMMA:
                w += 1
            wdy = w * dy
            wdy = tl.where(mask, wdy, 0)
            c1 = tl.sum(xhat * wdy, axis=0) / N
            c2 = tl.sum(wdy, axis=0) / N
            dx = (wdy - (xhat * c1 + c2)) * rstd

            # Write dx:
            tl.store(dx_ptrs + cols, dx.to(DX.type.element_ty), mask=mask)
            if not IGNORE_DW_DB:
                # Accumulate partial sums for dw and db:
                dw_row += dy * xhat
                db_row += dy

            # Advance to next row:
            row += tile_num
        if not IGNORE_DW_DB:
            tl.store(DW + pid * N + cols, dw_row.to(DW.type.element_ty), mask=mask)
            tl.store(DB + pid * N + cols, db_row.to(DB.type.element_ty), mask=mask)


@triton.jit
def _layernorm_bwd_dwdb_triton(
    DW,  # pointer to the partial sum of weights gradient
    DB,  # pointer to the partial sum of biases gradient
    FINAL_DW,  # pointer to the weights gradient
    FINAL_DB,  # pointer to the biases gradient
    M,  # GROUP_SIZE_M
    N,  # number of columns
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Map the program id to the elements of DW and DB it should compute.
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    db = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # Iterate through the rows of DW and DB to sum the partial sums.
    for i in range(0, M, BLOCK_SIZE_M):
        rows = i + tl.arange(0, BLOCK_SIZE_M)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        offs = rows[:, None] * N + cols[None, :]
        dw += tl.load(DW + offs, mask=mask, other=0.0)
        db += tl.load(DB + offs, mask=mask, other=0.0)
    # Write the final sum to the output.
    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)
    tl.store(FINAL_DW + cols, sum_dw.to(FINAL_DW.type.element_ty), mask=cols < N)
    tl.store(FINAL_DB + cols, sum_db.to(FINAL_DB.type.element_ty), mask=cols < N)


@triton.jit
def _layernorm_bwd_dwdb_triton_v2(
    X,  # pointer to the input
    DY,  # pointer to the output gradient
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride,
    FINAL_DW,  # pointer to the weights gradient
    FINAL_DB,  # pointer to the biases gradient
    M,  # GROUP_SIZE_M
    N,  # number of columns
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    db = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # Iterate through the rows of x and dy to compute dw and db
    for i in range(0, M, BLOCK_SIZE_M):
        rows = i + tl.arange(0, BLOCK_SIZE_M)
        means = tl.load(Mean + rows, mask=rows < M, other=0.0).to(tl.float32)
        rstds = tl.load(Rstd + rows, mask=rows < M, other=0.0).to(tl.float32)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        offs = rows[:, None] * stride + cols[None, :]
        x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(DY + offs, mask=mask, other=0.0).to(tl.float32)
        xhat = (x - means[:, None]) * rstds[:, None]
        dw += dy * xhat
        db += dy
    # Write the final sum to the output.
    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)
    tl.store(FINAL_DW + cols, sum_dw.to(FINAL_DW.type.element_ty), mask=cols < N)
    tl.store(FINAL_DB + cols, sum_db.to(FINAL_DB.type.element_ty), mask=cols < N)
