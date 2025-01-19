# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

import torch

import transformer_engine_torch as tex
import triton
import triton.language as tl
from itertools import product

def get_num_sms():
    current_device_index = torch.cuda.current_device()
    current_device = torch.cuda.get_device_properties(current_device_index)
    num_sms = current_device.multi_processor_count
    return num_sms

def get_autotune_config():
    return [triton.Config({'waves_per_eu': we}, num_warps=nw) for (we, nw) in product([0, 1, 2, 4], [4, 8, 16])]

@triton.autotune(configs=get_autotune_config(), key=['n_rows', 'n_cols'], use_cuda_graph=True)
@triton.jit
def _rmsnorm_triton(output_ptr, input_ptr, g_ptr, rsigma_ptr, input_row_stride, output_row_stride, n_rows, n_cols, epsilon,
                BLOCK_SIZE: tl.constexpr, USE_BLOCKED: tl.constexpr, NUM_PRGMS: tl.constexpr):
    row_start = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    #tl.assume(input_row_stride >= 0)
    #tl.assume(output_row_stride >= 0)
    #tl.assume(row_start >= 0)

    if USE_BLOCKED:

        # Persistent loop for rows
        for row_idx in tl.range(row_start, n_rows, NUM_PRGMS, num_stages=1):
            row_input_ptr = input_ptr + row_idx * input_row_stride
            row_output_ptr = output_ptr + row_idx * output_row_stride

            # Accumulate sum of squares
            n_cols_blks = tl.cdiv(n_cols, BLOCK_SIZE) - 1
            sum_squares: tl.float32 = 0.
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
                rms_norm = x * norm_factor * g
                output_ptrs = row_output_ptr + cols
                tl.store(output_ptrs, rms_norm.to(output_ptr.type.element_ty))

            # Handle remainder
            cols = n_cols_blks * BLOCK_SIZE + col_offsets
            mask = cols < n_cols
            input_ptrs = row_input_ptr + cols
            x = tl.load(input_ptrs, mask=mask, other=0.0, cache_modifier=".cg").to(tl.float32)
            g_ptrs = g_ptr + cols
            g = tl.load(g_ptrs, mask=mask, other=0.0).to(tl.float32)
            rms_norm = x * norm_factor * g
            output_ptrs = row_output_ptr + cols
            tl.store(output_ptrs, rms_norm.to(output_ptr.type.element_ty), mask=mask)

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

            rms_norm = row * norm_factor * g

            output_ptrs = output_ptr + row_idx * output_row_stride + col_offsets
            output_ptrs = tl.multiple_of(output_ptrs, (16, ))
            tl.store(output_ptrs, rms_norm.to(output_ptr.type.element_ty), mask=mask)

# TODO: currently our triton kernel has not supported fp8 yet
def te_rmsnorm_fwd_noalloc_triton(input, weight, eps, ln_out):
    M, N = input.shape
    rsigma = torch.empty((M, ), device='cuda', dtype=torch.float32)
    MAX_FUSED_SIZE = 65536 // input.element_size()
    blk_size = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    USE_BLOCKED = N > blk_size
    NUM_PRGMS = min(M, get_num_sms())
    grid = lambda meta: (NUM_PRGMS, )
    _rmsnorm_triton[grid](ln_out, input, weight, rsigma, input.stride(0), ln_out.stride(0), M, N, eps, blk_size, USE_BLOCKED, NUM_PRGMS)

    return ln_out, rsigma

# specialized version of rmsnorm fwd, optimized for inference
def te_rmsnorm_fwd_inf_triton(input, weight, eps):
    ln_out = torch.empty_like(input)
    ln_out, _ = te_rmsnorm_fwd_noalloc_triton(input, weight, eps, ln_out)
    return ln_out

