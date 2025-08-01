# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

import torch
import triton
import triton.language as tl
from itertools import product
from .norm_common import num_programs, block_size, use_blocked
from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer, Float8Tensor
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
from transformer_engine.pytorch.triton_kernels.common import (
    te_dtype_to_torch_dtype,
    te_dtype_to_triton_dtype,
)


def dg_tmp_rows(x, sm_margin=None):
    return x.shape[0] if use_blocked(x) else num_programs(x, sm_margin)


def get_autotune_config():
    return [triton.Config({'waves_per_eu': we}, num_warps=nw) for (we, nw) in product([0, 1, 2, 4], [4, 8, 16])]


@triton.autotune(configs=get_autotune_config(), key=['n_rows', 'n_cols'], use_cuda_graph=True)
@triton.jit
def _rmsnorm_fwd_triton(
    output_ptr,
    input_ptr,
    g_ptr, rsigma_ptr,
    input_row_stride,
    output_row_stride,
    n_rows, n_cols,
    epsilon,
    amax_ptr,
    q_amax_ptr,
    q_scale_ptr,
    scale_inv_ptr,
    out_transpose_ptr,
    transpose_row_stride,
    ZERO_CENTERED_GAMMA: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    USE_BLOCKED: tl.constexpr,
    NUM_PRGMS: tl.constexpr,
    IS_FP8: tl.constexpr,
    MAKE_TRANSPOSE: tl.constexpr,
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
                input_ptrs = tl.multiple_of(input_ptrs, (16, ))
                x = tl.load(input_ptrs).to(tl.float32)
                sum_squares += tl.sum(x * x, axis=0)

            # Handle remainder
            cols = n_cols_blks * BLOCK_SIZE + col_offsets
            mask = cols < n_cols
            input_ptrs = row_input_ptr + cols
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
                input_ptrs = tl.multiple_of(input_ptrs, (16, ))
                x = tl.load(input_ptrs).to(tl.float32)
                g_ptrs = g_ptr + cols
                g = tl.load(g_ptrs).to(tl.float32)
                if (ZERO_CENTERED_GAMMA):
                    g += 1
                rms_norm = x * norm_factor * g
                output_ptrs = row_output_ptr + cols
                if IS_FP8:
                    amax_temp = tl.max(tl.abs(rms_norm), axis=-1)
                    amax = tl.maximum(amax, amax_temp)
                    rms_norm = rms_norm * scale
                    if MAKE_TRANSPOSE:
                        output_t_ptrs = out_transpose_ptr + col_offsets * transpose_row_stride + blk_idx * BLOCK_SIZE + row_idx
                        tl.store(output_t_ptrs, rms_norm.to(output_type))
                tl.store(output_ptrs, rms_norm.to(output_type))

            # Handle remainder
            cols = n_cols_blks * BLOCK_SIZE + col_offsets
            mask = cols < n_cols
            input_ptrs = row_input_ptr + cols
            x = tl.load(input_ptrs, mask=mask, other=0.0, cache_modifier=".cg").to(tl.float32)
            g_ptrs = g_ptr + cols
            g = tl.load(g_ptrs, mask=mask, other=0.0).to(tl.float32)
            if (ZERO_CENTERED_GAMMA):
                g += 1
            rms_norm = x * norm_factor * g
            output_ptrs = row_output_ptr + cols
            if IS_FP8:
                amax_temp = tl.max(tl.abs(rms_norm), axis=-1)
                amax = tl.maximum(amax, amax_temp)
                rms_norm = rms_norm * scale
                if MAKE_TRANSPOSE:
                    output_t_ptrs = out_transpose_ptr + col_offsets * transpose_row_stride + n_cols_blks * BLOCK_SIZE + row_idx
                    tl.store(output_t_ptrs, rms_norm.to(output_type), mask=mask)
            tl.store(output_ptrs, rms_norm.to(output_type), mask=mask)

    else:
        mask = col_offsets < n_cols
        for row_idx in tl.range(row_start, n_rows, NUM_PRGMS, num_stages=2):
            input_ptrs = input_ptr + row_idx * input_row_stride + col_offsets
            input_ptrs = tl.multiple_of(input_ptrs, (16, ))
            row = tl.load(input_ptrs, mask=mask, other=0.0, cache_modifier=".cg").to(tl.float32)
            g = tl.load(g_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
            row_norm = row * row
            row_norm = tl.sum(row_norm, axis=-1)
            norm_factor = tl.math.rsqrt((row_norm / n_cols) + epsilon)

            # Store rsigma (norm_factor)
            rsigma_output_ptr = rsigma_ptr + row_idx
            tl.store(rsigma_output_ptr, norm_factor)

            if (ZERO_CENTERED_GAMMA):
                g += 1
            rms_norm = row * norm_factor * g

            output_ptrs = output_ptr + row_idx * output_row_stride + col_offsets
            output_ptrs = tl.multiple_of(output_ptrs, (16, ))
            if IS_FP8:
                amax_temp = tl.max(tl.abs(rms_norm), axis=-1)
                amax = tl.maximum(amax, amax_temp)
                rms_norm = rms_norm * scale
                if MAKE_TRANSPOSE:
                    output_t_ptrs = out_transpose_ptr + col_offsets * transpose_row_stride + row_idx
                    tl.store(output_t_ptrs, rms_norm.to(output_type), mask=mask)
            tl.store(output_ptrs, rms_norm.to(output_type), mask=mask)
    if IS_FP8:
        tl.store(amax_ptr + row_start, amax)
        tl.atomic_max(q_amax_ptr, amax, sem="relaxed")
        if row_start == 0:
            scale = tl.load(q_scale_ptr)
            scale_inv = tl.fdiv(1.0, scale)
            tl.store(scale_inv_ptr, scale_inv)

@triton.jit
def _rmsnorm_bwd_triton(grad_output_ptr, input_ptr, g_ptr, rsigma_ptr, dx_ptr, dg_ptr, input_row_stride, output_row_stride,
                        n_rows, n_cols, ZERO_CENTERED_GAMMA: tl.constexpr, BLOCK_SIZE: tl.constexpr,
                        USE_BLOCKED: tl.constexpr, NUM_PRGMS: tl.constexpr):
    row_start = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    #   tl.assume(input_row_stride >= 0)
    #   tl.assume(output_row_stride >= 0)
    #   tl.assume(row_start >= 0)

    if USE_BLOCKED:
        for row_idx in tl.range(row_start, n_rows, NUM_PRGMS, num_stages=1):
            row_input_ptr = input_ptr + row_idx * input_row_stride
            row_grad_output_ptr = grad_output_ptr + row_idx * output_row_stride
            row_dx_ptr = dx_ptr + row_idx * input_row_stride
            row_dg_ptr = dg_ptr + row_idx * input_row_stride

            # Compute gradients sum of all colums for each row
            n_cols_blks = tl.cdiv(n_cols, BLOCK_SIZE) - 1
            # older version of triton doesn't accept below init
            # comment out for now to make it compatible with triton 3.1
            # grad_sum: tl.float32 = 0.0
            grad_sum = 0.0
            for blk_idx in tl.range(0, n_cols_blks, num_stages=2):
                cols = blk_idx * BLOCK_SIZE + col_offsets
                input_ptrs = row_input_ptr + cols
                grad_output_ptrs = row_grad_output_ptr + cols

                input_ptrs = tl.multiple_of(input_ptrs, (16, ))
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

            for blk_idx in tl.range(0, n_cols_blks, num_stages=2):
                cols = blk_idx * BLOCK_SIZE + col_offsets
                input_ptrs = row_input_ptr + cols
                grad_output_ptrs = row_grad_output_ptr + cols

                input_ptrs = tl.multiple_of(input_ptrs, (16, ))
                grad_output_ptrs = tl.multiple_of(grad_output_ptrs, (16, ))

                x = tl.load(input_ptrs).to(tl.float32)
                grad_output = tl.load(grad_output_ptrs).to(tl.float32)

                g_ptrs = g_ptr + cols
                g = tl.load(g_ptrs).to(tl.float32)
                if (ZERO_CENTERED_GAMMA):
                    g += 1.
                grad_input = grad_output * norm_factor * g - (norm_factor * norm_factor * norm_factor) * x * (grad_sum /
                                                                                                              n_cols)

                dx_ptrs = row_dx_ptr + cols
                tl.store(dx_ptrs, grad_input.to(dx_ptr.type.element_ty))

                dg = grad_output * x * norm_factor
                dg_ptrs = row_dg_ptr + cols
                tl.store(dg_ptrs, dg.to(tl.float32))

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
            grad_input = grad_output * norm_factor * g - (norm_factor * norm_factor * norm_factor) * x * (grad_sum /
                                                                                                          n_cols)

            dx_ptrs = row_dx_ptr + cols
            tl.store(dx_ptrs, grad_input.to(dx_ptr.type.element_ty), mask=mask)

            dg = grad_output * x * norm_factor
            dg_ptrs = row_dg_ptr + cols
            tl.store(dg_ptrs, dg.to(tl.float32), mask=mask)

    else:
        mask = col_offsets < n_cols
        dg_col_redux = tl.zeros((BLOCK_SIZE, ), dtype=tl.float32)

        for row_idx in tl.range(row_start, n_rows, NUM_PRGMS, num_stages=2):
            input_ptrs = input_ptr + row_idx * input_row_stride + col_offsets
            grad_output_ptrs = grad_output_ptr + row_idx * output_row_stride + col_offsets
            dx_ptrs = dx_ptr + row_idx * input_row_stride + col_offsets

            input_ptrs = tl.multiple_of(input_ptrs, (16, ))
            grad_output_ptrs = tl.multiple_of(grad_output_ptrs, (16, ))
            dx_ptrs = tl.multiple_of(dx_ptrs, (16, ))

            x = tl.load(input_ptrs, mask=mask, other=0.0).to(tl.float32)
            grad_output = tl.load(grad_output_ptrs, mask=mask, other=0.0).to(tl.float32)
            g = tl.load(g_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
            if (ZERO_CENTERED_GAMMA):
                g += 1.

            norm_factor = tl.load(rsigma_ptr + row_idx).to(tl.float32)
            grad_sum = tl.sum(grad_output * x * g, axis=0)

            grad_input = grad_output * norm_factor * g - (norm_factor * norm_factor * norm_factor) * x * (grad_sum /
                                                                                                          n_cols)
            tl.store(dx_ptrs, grad_input.to(dx_ptr.type.element_ty), mask=mask)

            dg = grad_output * x * norm_factor
            dg_col_redux += dg.to(tl.float32)

        tl.store(dg_ptr + tl.program_id(0) * input_row_stride + col_offsets, dg_col_redux, mask=mask)


@triton.jit
def _rmsnorm_bwd_dg_reduce_triton(dg_in_ptr, dg_out_ptr, dg_in_stride, n_rows, n_cols, BLOCK_SIZE_M: tl.constexpr,
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

# triton drop-in replacement for transformer_engine::pytorch::rmsnorm_bwd
def te_rmsnorm_bwd_triton(dz, x, rsigma, gamma, sm_margin, zero_centered_gamma):
    # may take non-contiguous inputs
    dz_ = dz.contiguous()
    x_ = x.contiguous()
    rsigma_ = rsigma.contiguous()
    gamma_ = gamma.contiguous()

    dx = torch.empty_like(x_)
    dgamma = torch.empty_like(gamma_)

    M, N = x_.shape
    blk_size = block_size(x_)
    USE_BLOCKED = use_blocked(x_)
    NUM_PRGMS = num_programs(x_, sm_margin)
    need_reduction = N > 1
    dg_tmp = torch.empty(dg_tmp_rows(x_, sm_margin), N, device='cuda', dtype=torch.float32, requires_grad=False) if need_reduction else None

    grid_bwd = lambda meta: (NUM_PRGMS, )
    _rmsnorm_bwd_triton[grid_bwd](dz_, x_, gamma_, rsigma_, dx, dg_tmp if need_reduction else dgamma,
                                  x_.stride(0), dz_.stride(0), M, N, zero_centered_gamma, blk_size,
                                  USE_BLOCKED, NUM_PRGMS, num_warps=8)

    if need_reduction:
        grid_reduce = lambda meta: [triton.cdiv(N, meta['BLOCK_SIZE_N'])]
        _rmsnorm_bwd_dg_reduce_triton[grid_reduce](dg_tmp, dgamma, dg_tmp.stride(0), dg_tmp.shape[0], dg_tmp.shape[1],
                                                   BLOCK_SIZE_M=128, BLOCK_SIZE_N=64)

    return dx, dgamma

# triton drop-in replacement for transformer_engine::pytorch::rmsnorm_fwd
def te_rmsnorm_fwd_triton(
    input,
    weight,
    eps,
    ln_out,
    quantizer,
    otype,
    sm_margin,
    zero_centered_gamma
):
    if eps < 0:
        raise ValueError(f"`eps` must be non-negative, but a value of {eps} was passed")
    if len(input.shape) != 2:
        raise ValueError(
            f"The input must be a 2-dimensional matrix, but an input with {input.ndim} was passed.")

    N, H = input.shape
    if weight.shape[0] != H:
        raise ValueError(
            f"The shape of `weight` must be feature-aligned, "
            f"but {weight.shape[0]=} while {input.shape[1]=}"
        )
    IS_FP8 = isinstance(quantizer, Float8Quantizer)
    IS_MFP8 = isinstance(quantizer, MXFP8Quantizer)
    BLOCK_SIZE = block_size(input)
    USE_BLOCKED = use_blocked(input)
    NUM_PRGMS = num_programs(input, sm_margin)
    MAKE_TRANSPOSE = False

    rsigma = torch.empty((N,), dtype=torch.float32, device="cuda")
    pt_otype = (
        otype if isinstance(otype, torch.dtype)
        else te_dtype_to_torch_dtype(otype)
    )
    if IS_FP8:
        MAKE_TRANSPOSE = quantizer.columnwise_usage
        if ln_out is not None:
            out = (
                ln_out if isinstance(ln_out, Float8Tensor) else
                quantizer.create_tensor_from_data(
                    ln_out.view(te_dtype_to_torch_dtype(quantizer.dtype)),
                    fake_dtype=pt_otype
                )
            )
        else:
            out = quantizer.make_empty(input.shape, dtype=pt_otype)

        amax = torch.empty((NUM_PRGMS,), dtype=torch.float32, device="cuda")
        tl_dtype = te_dtype_to_triton_dtype(quantizer.dtype)
        scale_inv_ptr = out._scale_inv
        q_scale = quantizer.scale
        q_amax = quantizer.amax
        out_ptr = triton.reinterpret(out._data, tl_dtype)
        if MAKE_TRANSPOSE:
            if out._transpose_invalid:
                out._transpose = torch.empty((out._data.shape[1], out._data.shape[0]), dtype=out._data.dtype)
                out._transpose_invalid = False
            out_transpose_ptr = triton.reinterpret(out._transpose, tl_dtype)
            out_transpose_stride = out._transpose.stride(0)

    else:
        out = torch.empty_like(input, dtype=pt_otype) if ln_out is None else ln_out
        amax = None
        tl_dtype = None
        scale_inv_ptr = None
        q_scale = None
        q_amax = None
        out_ptr = out

    # Populate with dummy args to satisfy triton compiler
    if not MAKE_TRANSPOSE:
        out_transpose_ptr = out_ptr
        out_transpose_stride = out_ptr.stride(0)


    grid_fwd = lambda meta: (NUM_PRGMS, )
    # TODO(micky774) Implement fused MXFP8 quantization within the kernel
    _rmsnorm_fwd_triton[grid_fwd](
        out_ptr,
        input,
        weight,
        rsigma,
        input.stride(0),
        out_ptr.stride(0),
        N, H, eps,
        amax,
        q_amax,
        q_scale,
        scale_inv_ptr,
        out_transpose_ptr,
        out_transpose_stride,
        zero_centered_gamma,
        BLOCK_SIZE,
        USE_BLOCKED,
        NUM_PRGMS,
        IS_FP8,
        MAKE_TRANSPOSE,
    )
    if IS_MFP8:
        out = quantizer.quantize(out)

    return out, None, rsigma