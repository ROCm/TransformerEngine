# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information


from itertools import product

import torch
import triton
import triton.language as tl

from .norm_common_triton import block_size


def get_autotune_config(full_tuning_space=False):
    if full_tuning_space:
        tuning_space = product([1, 2, 4], [4, 8, 16])
    else:
        tuning_space = [(1, 8), (1, 16), (2, 16), (4, 4), (4, 8), (4, 16)]
    return [
        triton.Config({"waves_per_eu": waves_per_eu}, num_warps=num_warps, num_stages=1)
        for waves_per_eu, num_warps in tuning_space
    ]


@triton.autotune(
    configs=get_autotune_config(), key=["n_rows", "n_cols"], use_cuda_graph=True
)
@triton.jit
def _layernorm_fwd_triton(
    x_ptr,
    y_ptr,
    w_ptr,
    b_ptr,
    mean_ptr,
    rstd_ptr,
    x_row_stride,
    y_row_stride,
    n_rows,
    n_cols,
    eps,
    ZERO_CENTERED_GAMMA: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):

    # program id
    row = tl.program_id(0)
    x_ptr_start = x_ptr + (row * x_row_stride)
    y_ptr_start = y_ptr + (row * y_row_stride)

    loop_num = tl.cdiv(n_cols, BLOCK_SIZE) - 1

    # calculate mean
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    loop_num_l = loop_num
    for b in range(0, loop_num_l):
        col_offsets = b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x_block = tl.load(x_ptr_start + col_offsets).to(tl.float32)  # Unmasked loads
        _mean += x_block

    # For last iteration, do masked load
    col_offsets = loop_num_l * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x_block = tl.load(
        x_ptr_start + col_offsets, mask=col_offsets < n_cols, other=0.0
    ).to(tl.float32)
    _mean += x_block
    mean = tl.sum(_mean, axis=0) / n_cols

    # variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    loop_num_l = loop_num
    for b in range(0, loop_num_l):
        col_offsets = b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x_block = tl.load(x_ptr_start + col_offsets).to(tl.float32)  # Unmasked loads
        x_block = x_block - mean
        _var += x_block * x_block

    # For last iteration, do masked load
    col_offsets = loop_num_l * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x_block = tl.load(
        x_ptr_start + col_offsets, mask=col_offsets < n_cols, other=0.0
    ).to(tl.float32)
    x_block = tl.where(col_offsets < n_cols, x_block - mean, 0.0)
    _var += x_block * x_block

    var = tl.sum(_var, axis=0) / n_cols
    rstd = tl.rsqrt(var + eps)

    # Write mean / rstd
    tl.store(mean_ptr + row, mean)
    tl.store(rstd_ptr + row, rstd)

    # Normalize and store
    loop_num_l = loop_num
    for b in range(0, loop_num_l):
        col_offsets = b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        w_block = tl.load(w_ptr + col_offsets).to(tl.float32)
        b_block = tl.load(b_ptr + col_offsets).to(tl.float32)
        x_block = tl.load(x_ptr_start + col_offsets).to(tl.float32)
        if ZERO_CENTERED_GAMMA:
            w_block += 1
        y_block = (x_block - mean) * rstd
        y_block = y_block * w_block + b_block
        tl.store(y_ptr_start + col_offsets, y_block.to(y_ptr.type.element_ty))

    # For last iteration, do masked load and store
    col_offsets = loop_num_l * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    w_block = tl.load(w_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    b_block = tl.load(b_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    x_block = tl.load(x_ptr_start + col_offsets, mask=mask, other=0.0).to(tl.float32)
    if ZERO_CENTERED_GAMMA:
        w_block += 1
    y_block = (x_block - mean) * rstd
    y_block = y_block * w_block + b_block
    tl.store(y_ptr_start + col_offsets, y_block.to(y_ptr.type.element_ty), mask=mask)


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
            xhat = tl.where(mask, xhat, 0)
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
            xhat = tl.where(mask, xhat, 0)
            wdy = tl.where(mask, wdy, 0)

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
            xhat = tl.where(mask, xhat, 0)
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


# TODO: Implement persistent kernel in forward and add `sm_margin` to the interface.
def te_layernorm_fwd_fp8_noalloc_triton(
    x, gamma, beta, eps, y, out_dtype, zero_centered_gamma
):
    M, N = x.shape
    y = y.view(out_dtype)
    mu = torch.empty((M,), dtype=torch.float32, device=x.device)
    rsigma = torch.empty((M,), dtype=torch.float32, device=x.device)

    BLOCK_SIZE = block_size(x)
    _layernorm_fwd_triton[(M,)](
        x,
        y,
        gamma,
        beta,
        mu,
        rsigma,
        x.stride(0),
        y.stride(0),
        M,
        N,
        eps,
        ZERO_CENTERED_GAMMA=zero_centered_gamma,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return y, mu, rsigma


# TODO: Add `sm_margin` to the interface.
def te_layernorm_bwd_triton(dz, x, mu, rsigma, gamma, zero_centered_gamma):
    M, N = x.shape
    # calculate dw and db separately when M is small
    IGNORE_DW_DB_IN_FUSED = M <= 512
    tile_num = max(min(256, M // 4), 1)
    if M <= 512 and M * N < 64 * 1024 * 1024:
        tile_num = M
    elif M > 16384:
        tile_num = 2048
        if IGNORE_DW_DB_IN_FUSED:
            tile_num = 4096

    max_fused_size = 32768 // x.element_size()
    next_power = triton.next_power_of_2(N)
    BLOCK_SIZE = min(max_fused_size, next_power)
    # For cases with small M and large N, decrease block size to help with occupancy and register spill
    if tile_num == M:
        if tile_num > 256:
            BLOCK_SIZE = min(BLOCK_SIZE, 2048)
        else:
            BLOCK_SIZE = min(BLOCK_SIZE, 4096)
    USE_BLOCKED = N > BLOCK_SIZE
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)

    dx = torch.empty_like(x)
    if not IGNORE_DW_DB_IN_FUSED:
        _dgamma = torch.zeros((tile_num, N), dtype=torch.float32, device=gamma.device)
        _dbeta = torch.zeros((tile_num, N), dtype=torch.float32, device=gamma.device)
    else:
        _dgamma = None
        _dbeta = None
    dgamma = torch.zeros((N,), dtype=gamma.dtype, device=gamma.device)
    dbeta = torch.zeros((N,), dtype=gamma.dtype, device=gamma.device)
    grid_bwd = (tile_num,)
    _layernorm_bwd_dx_fused_triton[grid_bwd](
        dx,
        dz,
        _dgamma,
        _dbeta,
        x,
        gamma,
        mu,
        rsigma,
        x.stride(0),
        N,
        ZERO_CENTERED_GAMMA=zero_centered_gamma,
        NUM_ROWS=M,
        BLOCK_SIZE_N=BLOCK_SIZE,
        USE_BLOCKED=USE_BLOCKED,
        num_warps=num_warps,
        IGNORE_DW_DB=IGNORE_DW_DB_IN_FUSED,
    )
    grid_reduce = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE_N"]),)
    if not IGNORE_DW_DB_IN_FUSED:
        dwdb_block_n = max(16, N // 256)
        dwdb_block_n = triton.next_power_of_2(dwdb_block_n)
        dwdb_block_m = (64 * 128) // dwdb_block_n
        dwdb_block_m = min(triton.next_power_of_2(tile_num), dwdb_block_m)
        _layernorm_bwd_dwdb_triton[grid_reduce](
            _dgamma,
            _dbeta,
            dgamma,
            dbeta,
            min(tile_num, M),
            N,
            BLOCK_SIZE_M=dwdb_block_m,
            BLOCK_SIZE_N=dwdb_block_n,
        )
    else:
        dwdb_block_n = max(16, N // 256)
        dwdb_block_n = triton.next_power_of_2(dwdb_block_n)
        dwdb_block_m = (64 * 128) // dwdb_block_n
        dwdb_block_m = min(triton.next_power_of_2(M), dwdb_block_m)
        _layernorm_bwd_dwdb_triton_v2[grid_reduce](
            x,
            dz,
            mu,
            rsigma,
            x.stride(0),
            dgamma,
            dbeta,
            M,
            N,
            BLOCK_SIZE_M=dwdb_block_m,
            BLOCK_SIZE_N=dwdb_block_n,
        )

    return dx, dgamma, dbeta
