# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

import torch
import triton
import triton.language as tl

def get_autotune_config():
    return [triton.Config({'waves_per_eu': 0}, num_warps=nw) for nw in (1, 4, 8, 16)]


def _prune_rms_configs(configs, named_args, **kwargs):
    """Prune autotune configs whose ``num_warps`` is a poor fit for ``n_cols``.

    Bounds threads-per-row to ``[n_cols/16, n_cols*2]`` so the autotuner never
    wastes a measurement on a config that is obviously too serial (too few
    lanes) or mostly idle (too many lanes). Shared by the fwd and bwd kernels.
    """
    n_cols = named_args.get('n_cols')
    if n_cols is None:
        n_cols = kwargs.get('n_cols')
    if n_cols is None:
        return configs
    out = []
    for cfg in configs:
        threads = cfg.num_warps * 64  # AMD wavefront = 64
        if threads < n_cols / 16:  # too serial
            continue
        if threads > n_cols * 2:  # too many idle lanes
            continue
        out.append(cfg)
    return out if out else configs


# TODO(micky774) Implement fused MXFP8 quantization within the kernel
@triton.jit
def _rmsnorm_fwd_triton_impl(
    input_ptr,
    output_ptr,
    g_ptr,
    rsigma_ptr,
    input_row_stride,
    output_row_stride,
    n_rows, n_cols,
    epsilon,
    q_amax_ptr,
    q_scale_ptr,
    scale_inv_ptr,
    ZERO_CENTERED_GAMMA: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    USE_BLOCKED: tl.constexpr,
    NUM_PRGMS: tl.constexpr,
    IS_FP8: tl.constexpr,
    FP8_MAX: tl.constexpr,
    INPUT_ALIGNED_16: tl.constexpr,
    OUTPUT_ALIGNED_16: tl.constexpr,
    ROWS_PER_PID: tl.constexpr = 1,
    NEEDS_I64_OFFSETS: tl.constexpr = False,
    HOIST_GAMMA: tl.constexpr = True,
):

    row_start = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    # as older version Triton doesn't support tl.assume and BUFF OPS, comment out for now
    # tl.assume(input_row_stride >= 0)
    # tl.assume(output_row_stride >= 0)
    # tl.assume(row_start >= 0)
    output_type = output_ptr.type.element_ty
    if IS_FP8:
        scale = tl.load(q_scale_ptr)
        amax = 0.0

    if USE_BLOCKED:

        # Persistent loop for rows
        for row_idx in tl.range(row_start, n_rows, NUM_PRGMS, num_stages=1):
            row_input_ptr = input_ptr + row_idx * input_row_stride
            row_output_ptr = output_ptr + row_idx * output_row_stride

            # Accumulate sum of squares
            n_cols_blks = tl.cdiv(n_cols, BLOCK_SIZE) - 1
            # older version of triton doesn't accept below init
            # sum_squares: tl.float32 = 0.
            # however, with type promoting rule in triton, sum_squares should be always fp32 with below init
            sum_squares = 0.
            for blk_idx in tl.range(0, n_cols_blks, num_stages=2):
                cols = blk_idx * BLOCK_SIZE + col_offsets
                input_ptrs = row_input_ptr + cols
                if INPUT_ALIGNED_16:
                    input_ptrs = tl.multiple_of(input_ptrs, (16, ))
                x = tl.load(input_ptrs).to(tl.float32)
                sum_squares += tl.sum(x * x, axis=0)

            # Handle remainder
            cols = n_cols_blks * BLOCK_SIZE + col_offsets
            mask = cols < n_cols
            input_ptrs = row_input_ptr + cols
            if INPUT_ALIGNED_16:
                input_ptrs = tl.multiple_of(input_ptrs, (16, ))
            x = tl.load(input_ptrs, mask=mask, other=0.0, cache_modifier=".cg").to(tl.float32)
            sum_squares += tl.sum(x * x, axis=0)

            # Compute normalization factor
            mean_square = sum_squares / n_cols
            norm_factor = tl.rsqrt(mean_square + epsilon)

            # Store rsigma (norm_factor)
            tl.store(rsigma_ptr + row_idx, norm_factor)

            # Normalize and write output
            for blk_idx in tl.range(0, n_cols_blks, num_stages=2):
                cols = blk_idx * BLOCK_SIZE + col_offsets
                input_ptrs = row_input_ptr + cols
                if INPUT_ALIGNED_16:
                    input_ptrs = tl.multiple_of(input_ptrs, (16, ))
                x = tl.load(input_ptrs).to(tl.float32)
                g_ptrs = g_ptr + cols
                g = tl.load(g_ptrs).to(tl.float32)
                if (ZERO_CENTERED_GAMMA):
                    g += 1
                rms_norm = x * norm_factor * g
                output_ptrs = row_output_ptr + cols
                if OUTPUT_ALIGNED_16:
                    output_ptrs = tl.multiple_of(output_ptrs, (16, ))
                if IS_FP8:
                    amax_temp = tl.max(tl.abs(rms_norm), axis=-1)
                    amax = tl.maximum(amax, amax_temp)
                    rms_norm = rms_norm * scale
                    rms_norm = tl.clamp(rms_norm, -FP8_MAX, FP8_MAX)
                tl.store(output_ptrs, rms_norm.to(output_type))

            # Handle remainder
            cols = n_cols_blks * BLOCK_SIZE + col_offsets
            mask = cols < n_cols
            input_ptrs = row_input_ptr + cols
            if INPUT_ALIGNED_16:
                input_ptrs = tl.multiple_of(input_ptrs, (16, ))
            x = tl.load(input_ptrs, mask=mask, other=0.0, cache_modifier=".cg").to(tl.float32)
            g_ptrs = g_ptr + cols
            g = tl.load(g_ptrs, mask=mask, other=0.0).to(tl.float32)
            if (ZERO_CENTERED_GAMMA):
                g += 1
            rms_norm = x * norm_factor * g
            output_ptrs = row_output_ptr + cols
            if OUTPUT_ALIGNED_16:
                output_ptrs = tl.multiple_of(output_ptrs, (16, ))
            if IS_FP8:
                amax_temp = tl.max(tl.abs(rms_norm), axis=-1)
                amax = tl.maximum(amax, amax_temp)
                rms_norm = rms_norm * scale
                rms_norm = tl.clamp(rms_norm, -FP8_MAX, FP8_MAX)
            tl.store(output_ptrs, rms_norm.to(output_type), mask=mask)

    else:
        mask = col_offsets < n_cols
        inv_n_cols = 1.0 / n_cols
        if HOIST_GAMMA:
            g = tl.load(g_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
            if (ZERO_CENTERED_GAMMA):
                g += 1
        n_chunks = tl.cdiv(n_rows, ROWS_PER_PID)
        for chunk_idx in tl.range(row_start, n_chunks, NUM_PRGMS, num_stages=2):
            base_row = chunk_idx * ROWS_PER_PID
            for i in tl.static_range(ROWS_PER_PID):
                row_idx = base_row + i
                # Mask rows past n_rows: ROWS_PER_PID need not divide n_rows.
                row_valid = row_idx < n_rows
                row_mask = mask & row_valid
                if NEEDS_I64_OFFSETS:
                    row_off = row_idx.to(tl.int64)
                else:
                    row_off = row_idx
                input_ptrs = input_ptr + row_off * input_row_stride + col_offsets
                if INPUT_ALIGNED_16:
                    input_ptrs = tl.multiple_of(input_ptrs, (16, ))
                if HOIST_GAMMA and ROWS_PER_PID == 1:
                    row = tl.load(input_ptrs, mask=row_mask, other=0.0, cache_modifier=".cg").to(tl.float32)
                else:
                    row = tl.load(input_ptrs, mask=row_mask, other=0.0).to(tl.float32)
                row_norm = tl.sum(row * row, axis=-1)
                norm_factor = tl.math.rsqrt(row_norm * inv_n_cols + epsilon)

                # Store rsigma (norm_factor)
                tl.store(rsigma_ptr + row_idx, norm_factor, mask=row_valid)

                if not HOIST_GAMMA:
                    g = tl.load(g_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
                    if (ZERO_CENTERED_GAMMA):
                        g += 1
                rms_norm = row * norm_factor * g

                output_ptrs = output_ptr + row_off * output_row_stride + col_offsets
                if OUTPUT_ALIGNED_16:
                    output_ptrs = tl.multiple_of(output_ptrs, (16, ))
                if IS_FP8:
                    amax_temp = tl.max(tl.abs(rms_norm), axis=-1)
                    amax = tl.maximum(amax, amax_temp)
                    rms_norm = rms_norm * scale
                    rms_norm = tl.clamp(rms_norm, -FP8_MAX, FP8_MAX)
                tl.store(output_ptrs, rms_norm.to(output_type), mask=row_mask)
    if IS_FP8:
        tl.atomic_max(q_amax_ptr, amax, sem="relaxed")
        if row_start == 0:
            scale = tl.load(q_scale_ptr)
            scale_inv = tl.fdiv(1.0, scale)
            tl.store(scale_inv_ptr, scale_inv)

autotune_dec = triton.autotune(
    configs=get_autotune_config(),
    key=['n_rows', 'n_cols'],
    prune_configs_by={'early_config_prune': _prune_rms_configs},
    use_cuda_graph=True,
)
_rmsnorm_fwd_triton = autotune_dec(_rmsnorm_fwd_triton_impl)

@triton.jit
def _rmsnorm_bwd_triton_impl(grad_output_ptr, input_ptr, g_ptr, rsigma_ptr, dx_ptr, dg_ptr, input_row_stride, output_row_stride,
                        n_rows, n_cols, ZERO_CENTERED_GAMMA: tl.constexpr, BLOCK_SIZE: tl.constexpr,
                        USE_BLOCKED: tl.constexpr, NUM_PRGMS: tl.constexpr,
                        INPUT_ALIGNED_16: tl.constexpr, GRAD_OUTPUT_ALIGNED_16: tl.constexpr,
                        DX_ALIGNED_16: tl.constexpr, DG_ALIGNED_16: tl.constexpr,
                        ROWS_PER_PID: tl.constexpr = 1,
                        NEEDS_I64_OFFSETS: tl.constexpr = False):
    row_start = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    inv_n_cols = 1.0 / n_cols
    #   tl.assume(input_row_stride >= 0)
    #   tl.assume(output_row_stride >= 0)
    #   tl.assume(row_start >= 0)

    if USE_BLOCKED:
        for row_idx in tl.range(row_start, n_rows, NUM_PRGMS, num_stages=2):
            row_input_ptr = input_ptr + row_idx * input_row_stride
            row_grad_output_ptr = grad_output_ptr + row_idx * output_row_stride
            row_dx_ptr = dx_ptr + row_idx * input_row_stride
            row_dg_ptr = dg_ptr + row_idx * n_cols

            # Compute gradients sum of all colums for each row
            n_cols_blks = tl.cdiv(n_cols, BLOCK_SIZE) - 1
            grad_sum = 0.0
            for blk_idx in tl.range(0, n_cols_blks, num_stages=2):
                cols = blk_idx * BLOCK_SIZE + col_offsets
                input_ptrs = row_input_ptr + cols
                grad_output_ptrs = row_grad_output_ptr + cols

                if INPUT_ALIGNED_16:
                    input_ptrs = tl.multiple_of(input_ptrs, (16, ))
                if GRAD_OUTPUT_ALIGNED_16:
                    grad_output_ptrs = tl.multiple_of(grad_output_ptrs, (16, ))

                x = tl.load(input_ptrs).to(tl.float32)
                grad_output = tl.load(grad_output_ptrs).to(tl.float32)
                g_ptrs = g_ptr + cols
                g = tl.load(g_ptrs).to(tl.float32)
                if (ZERO_CENTERED_GAMMA):
                    g += 1.
                grad_sum += tl.sum(grad_output * x * g, axis=0)

            # remainder for grad_sum:
            cols = n_cols_blks * BLOCK_SIZE + col_offsets
            mask = cols < n_cols
            input_ptrs = row_input_ptr + cols
            x = tl.load(input_ptrs, mask=mask, other=0.0).to(tl.float32)
            grad_output_ptrs = row_grad_output_ptr + cols
            grad_output = tl.load(grad_output_ptrs, mask=mask, other=0.0).to(tl.float32)
            g_ptrs = g_ptr + cols
            g = tl.load(g_ptrs, mask=mask, other=0.0).to(tl.float32)
            if (ZERO_CENTERED_GAMMA):
                g += 1.
            grad_sum += tl.sum(grad_output * x * g, axis=0)

            # Load r_sigma
            norm_factor = tl.load(rsigma_ptr + row_idx).to(tl.float32)
            # Precomputed per-row invariant: c = nf*nf * grad_sum / n_cols
            # used in calculating dx = nf * (dz*g - c*x)
            c_scalar = norm_factor * norm_factor * grad_sum * inv_n_cols

            for blk_idx in tl.range(0, n_cols_blks, num_stages=2):
                cols = blk_idx * BLOCK_SIZE + col_offsets
                input_ptrs = row_input_ptr + cols
                grad_output_ptrs = row_grad_output_ptr + cols

                if INPUT_ALIGNED_16:
                    input_ptrs = tl.multiple_of(input_ptrs, (16, ))
                if GRAD_OUTPUT_ALIGNED_16:
                    grad_output_ptrs = tl.multiple_of(grad_output_ptrs, (16, ))

                x = tl.load(input_ptrs).to(tl.float32)
                grad_output = tl.load(grad_output_ptrs).to(tl.float32)

                g_ptrs = g_ptr + cols
                g = tl.load(g_ptrs).to(tl.float32)
                if (ZERO_CENTERED_GAMMA):
                    g += 1.
                grad_input = norm_factor * (grad_output * g - c_scalar * x)

                dx_ptrs = row_dx_ptr + cols
                if DX_ALIGNED_16:
                    dx_ptrs = tl.multiple_of(dx_ptrs, (16, ))
                tl.store(dx_ptrs, grad_input.to(dx_ptr.type.element_ty))

                dg = grad_output * x * norm_factor
                dg_ptrs = row_dg_ptr + cols
                if DG_ALIGNED_16:
                    dg_ptrs = tl.multiple_of(dg_ptrs, (16, ))
                tl.store(dg_ptrs, dg)

            # Handle remainder
            cols = n_cols_blks * BLOCK_SIZE + col_offsets
            mask = cols < n_cols

            input_ptrs = row_input_ptr + cols
            x = tl.load(input_ptrs, mask=mask, other=0.0).to(tl.float32)
            grad_output_ptrs = row_grad_output_ptr + cols
            grad_output = tl.load(grad_output_ptrs, mask=mask, other=0.0).to(tl.float32)
            g_ptrs = g_ptr + cols
            g = tl.load(g_ptrs, mask=mask, other=0.0).to(tl.float32)
            if (ZERO_CENTERED_GAMMA):
                g += 1.
            grad_input = norm_factor * (grad_output * g - c_scalar * x)

            dx_ptrs = row_dx_ptr + cols
            if DX_ALIGNED_16:
                dx_ptrs = tl.multiple_of(dx_ptrs, (16, ))
            tl.store(dx_ptrs, grad_input.to(dx_ptr.type.element_ty), mask=mask)

            dg = grad_output * x * norm_factor
            dg_ptrs = row_dg_ptr + cols
            if DG_ALIGNED_16:
                dg_ptrs = tl.multiple_of(dg_ptrs, (16, ))
            tl.store(dg_ptrs, dg, mask=mask)

    else:
        mask = col_offsets < n_cols
        dg_col_redux = tl.zeros((BLOCK_SIZE, ), dtype=tl.float32)

        g = tl.load(g_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
        if (ZERO_CENTERED_GAMMA):
            g += 1.

        # Persistent outer loop with multi-row-per-chunk.
        n_chunks = tl.cdiv(n_rows, ROWS_PER_PID)
        for chunk_idx in tl.range(row_start, n_chunks, NUM_PRGMS, num_stages=2):
            base_row = chunk_idx * ROWS_PER_PID
            for i in tl.static_range(ROWS_PER_PID):
                row_idx = base_row + i
                # Mask rows past n_rows: ROWS_PER_PID need not divide n_rows.
                row_valid = row_idx < n_rows
                row_mask = mask & row_valid
                if NEEDS_I64_OFFSETS:
                    row_off = row_idx.to(tl.int64)
                else:
                    row_off = row_idx
                input_ptrs = input_ptr + row_off * input_row_stride + col_offsets
                grad_output_ptrs = grad_output_ptr + row_off * output_row_stride + col_offsets
                dx_ptrs = dx_ptr + row_off * input_row_stride + col_offsets

                if INPUT_ALIGNED_16:
                    input_ptrs = tl.multiple_of(input_ptrs, (16, ))
                if GRAD_OUTPUT_ALIGNED_16:
                    grad_output_ptrs = tl.multiple_of(grad_output_ptrs, (16, ))
                if DX_ALIGNED_16:
                    dx_ptrs = tl.multiple_of(dx_ptrs, (16, ))

                x = tl.load(input_ptrs, mask=row_mask, other=0.0).to(tl.float32)
                grad_output = tl.load(grad_output_ptrs, mask=row_mask, other=0.0).to(tl.float32)

                norm_factor = tl.load(rsigma_ptr + row_idx, mask=row_valid, other=1.0).to(tl.float32)
                grad_sum = tl.sum(grad_output * x * g, axis=0)
                c_scalar = norm_factor * norm_factor * grad_sum * inv_n_cols

                grad_input = norm_factor * (grad_output * g - c_scalar * x)
                tl.store(dx_ptrs, grad_input.to(dx_ptr.type.element_ty), mask=row_mask)

                dg = grad_output * x * norm_factor
                dg_col_redux += dg.to(tl.float32)

        # Each program owns exactly one dg_tmp partial row (index == row_start).
        tl.store(dg_ptr + row_start * n_cols + col_offsets, dg_col_redux, mask=mask)


# Autotune wrapper. Mirrors the fwd autotune layout (same config set + prune)
# so callers can toggle autotune via the same flag.
_rmsnorm_bwd_triton = triton.autotune(
    configs=get_autotune_config(),
    key=['n_rows', 'n_cols'],
    prune_configs_by={'early_config_prune': _prune_rms_configs},
    use_cuda_graph=True,
)(_rmsnorm_bwd_triton_impl)


@triton.jit
def _rmsnorm_bwd_dg_reduce_triton_impl(dg_in_ptr, dg_out_ptr, dg_in_stride, n_rows, n_cols, BLOCK_SIZE_M: tl.constexpr,
                                  BLOCK_SIZE_N: tl.constexpr):
    # we want parallelism in N direction
    # if N is small, we will just use one CU,
    # otherwise, it can be split by N/BLOCK_SIZE
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(0, n_rows, BLOCK_SIZE_M):
        rows = i + tl.arange(0, BLOCK_SIZE_M)
        mask = (rows[:, None] < n_rows) & (cols[None, :] < n_cols)
        offs = rows[:, None] * n_cols + cols[None, :]
        acc += tl.load(dg_in_ptr + offs, mask=mask, other=0., cache_modifier=".cg").to(tl.float32)

    sum_dg = tl.sum(acc, axis=0)
    tl.store(dg_out_ptr + cols, sum_dg.to(dg_out_ptr.type.element_ty), mask=cols < n_cols)


def _get_dg_reduce_configs():
    # n_rows is NUM_PRGMS so the M dimension is small.
    # The reduce kernel is <1% of bwd cost, so a tight 6-config sweep is plenty;
    # bigger sweeps just pay first-call compile tax for marginal gain.
    return [
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 64},  num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64},  num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64},  num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64},  num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128}, num_warps=8),
    ]


_rmsnorm_bwd_dg_reduce_triton = triton.autotune(
    configs=_get_dg_reduce_configs(),
    key=['n_rows', 'n_cols'],
    use_cuda_graph=True,
)(_rmsnorm_bwd_dg_reduce_triton_impl)
