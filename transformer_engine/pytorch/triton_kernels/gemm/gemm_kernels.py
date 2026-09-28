# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Triton kernel definitions for dense GEMM.

Contains the raw ``@triton.jit`` kernels used by ``gemm_wrapper.py``:

- ``mxfp8_matmul_kernel``: block-scaled FP8 matmul using ``tl.dot_scaled``
- ``matmul_kernel``: FP32/FP16/BF16/FP8 matmul with optional bias/BGRADB
  epilogue and fused alpha/beta accumulation
"""

import triton
import triton.language as tl


@triton.jit
def swizzle_pid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M: tl.constexpr):
    """Group linear program ids into ``GROUP_SIZE_M``-tall column strips for L2 reuse.

    With ``GROUP_SIZE_M == 1`` this degenerates to the naive row-major mapping
    (`pid_m = pid // num_pid_n`); with larger groups, consecutive pids sweep
    down a strip of ``GROUP_SIZE_M`` rows before advancing to the next N block,
    which lets adjacent blocks reuse the same A rows in L2.
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


# MXFP8 (Microscaling FP8) Matmul Kernel and Wrapper
# Uses Triton's tl.dot_scaled() for native block-scaled FP8 matmul

@triton.autotune(
    configs=[
        # Simpler configs for MXFP8 - BLOCK_K must be multiple of 32 (VEC_SIZE)
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 4}),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 4}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4}),
    ],
    # Include the FP8 format constexprs so E4M3/E5M2 combos get separately-
    # tuned configs (different formats can lower to different MFMA paths).
    key=['M', 'N', 'K', 'FP8_FORMAT_A', 'FP8_FORMAT_B'],
    # With ACCUMULATE=True each benchmark iteration would add computed_c to
    # the output again, multi-copying the result (and generating NaN/Inf as
    # the value explodes). Snapshot and restore c_ptr around every benchmark
    # run. Harmless when ACCUMULATE=False (kernel overwrites regardless).
    restore_value=['c_ptr'],
)
@triton.heuristics({
    'EVEN_K': lambda args: args['K'] % args['BLOCK_SIZE_K'] == 0,
})
@triton.jit
def mxfp8_matmul_kernel(
    # Data pointers
    a_ptr, b_ptr, c_ptr,
    # Scale pointers (E8M0 format, uint8)
    a_scale_ptr, b_scale_ptr,
    # Bias vector along N (broadcast across M). Consulted only when
    # EPILOGUE == 'BIAS'. Signature stays stable even when bias is unused --
    # the wrapper passes a dummy 1-element tensor.
    bias_ptr,
    # GEMM output scale (α) and accumulate scale (β): D = α·(A·B) + bias + β·C.
    # Passed as scalars (not tensors) so the fast-path constexpr toggle
    # ALPHA_IS_ONE below can compile the α multiply out entirely.
    alpha, beta,
    # Matrix dimensions
    M, N, K,
    # Data strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Scale strides
    stride_a_scale_m, stride_a_scale_k,
    stride_b_scale_n, stride_b_scale_k,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    VEC_SIZE: tl.constexpr,  # MXFP8_BLOCK_SCALING_SIZE (always 32)
    FP8_FORMAT_A: tl.constexpr,  # "e4m3" or "e5m2"
    FP8_FORMAT_B: tl.constexpr,  # "e4m3" or "e5m2"
    # Bias epilogue: 'DEFAULT' (no bias) or 'BIAS' (add per-N vector).
    # BGRADB is not implemented for MXFP8 (backward wgrad path uses a
    # separate op and doesn't request fused bias-grad through here).
    EPILOGUE: tl.constexpr,
    # β accumulate: D := existing_C * β + α·(A·B) (fused wgrad accumulate
    # and the ForwardLinearScaleAdd / BackwardLinearScale epilogue ops).
    ACCUMULATE: tl.constexpr,
    # Fast-path: skip `accumulator *= alpha` when α is known to be 1.0.
    ALPHA_IS_ONE: tl.constexpr,
):
    """
    MXFP8 matmul kernel using tl.dot_scaled() for block-scaled FP8 computation.

    Scales are stored in E8M0 format (uint8 biased exponents) and converted to FP32.
    """

    # Program ID
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # Swizzled block mapping for better L2 cache utilization
    pid_m, pid_n = swizzle_pid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M)

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Compute block offsets
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # int64 promotion (see matmul_kernel A/B).
    a_ptrs = a_ptr + (offs_am[:, None].to(tl.int64) * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :].to(tl.int64) * stride_bn)

    # Bound the A/B data loads to M/N. Block sizes (64/128/256) need not divide M/N
    # (MXFP8 only guarantees multiples of VEC_SIZE=32), so a fringe block would otherwise
    # read rows >= M / cols >= N out of bounds.
    mask_m = offs_am < M
    mask_n = offs_bn < N

    # K-loop
    num_k_blocks = tl.cdiv(K, BLOCK_SIZE_K)
    for k in range(num_k_blocks):
        # Load FP8 data
        if EVEN_K:
            a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
            b = tl.load(b_ptrs, mask=mask_n[None, :], other=0.0)
        else:
            k_remaining = K - k * BLOCK_SIZE_K
            mask_k = offs_k < k_remaining
            a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        # Load E8M0 scales for this K-block
        # We have:
        # - A columnwise scales: [M//VEC_SIZE, K] - one scale per 32 rows for each column
        # - B rowwise scales: [K, N//VEC_SIZE] - one scale per 32 columns for each row
        #
        # tl.dot_scaled expects:
        # - A scales: [BLOCK_SIZE_M, BLOCK_SIZE_K // VEC_SIZE] - one scale per 32 elements along K
        # - B scales: [BLOCK_SIZE_K // VEC_SIZE, BLOCK_SIZE_N] - one scale per 32 elements along K
        #
        # For columnwise A: each column has M//32 scales
        # We need to gather the right scales for our block

        # A scales: we need scales for rows [pid_m*BLOCK_SIZE_M : (pid_m+1)*BLOCK_SIZE_M]
        #           and columns [k*BLOCK_SIZE_K : (k+1)*BLOCK_SIZE_K]
        # From columnwise layout [M//VEC_SIZE, K], we need to:
        # - Select row indices: [pid_m*BLOCK_SIZE_M//VEC_SIZE : (pid_m+1)*BLOCK_SIZE_M//VEC_SIZE]
        # - Select column indices: [k*BLOCK_SIZE_K : (k+1)*BLOCK_SIZE_K]
        # This gives us [BLOCK_SIZE_M//VEC_SIZE, BLOCK_SIZE_K] scales
        # But we need [BLOCK_SIZE_M, BLOCK_SIZE_K//VEC_SIZE] for tl.dot_scaled

        # Actually, let's think about what the scales mean:
        # Columnwise: scale[i,j] applies to data[i*32:(i+1)*32, j]
        # So for block starting at row pid_m*BLOCK_SIZE_M, col k*BLOCK_SIZE_K:
        # We need scales from row indices pid_m*BLOCK_SIZE_M//32 to (pid_m+1)*BLOCK_SIZE_M//32

        # With reversed selection, A now has rowwise scales [M, K//32]
        # For tl.dot_scaled we need [BLOCK_SIZE_M, BLOCK_SIZE_K//32]
        k_block_start = k * (BLOCK_SIZE_K // VEC_SIZE)
        offs_a_scale_k = k_block_start + tl.arange(0, BLOCK_SIZE_K // VEC_SIZE)

        a_scale_ptrs = a_scale_ptr + (offs_am[:, None].to(tl.int64) * stride_a_scale_m +
                                       offs_a_scale_k[None, :] * stride_a_scale_k)

        # Check bounds for scale loading
        # Use other=127 for out-of-bounds E8M0 scales (127 = scale of 1.0)
        mask_a_scale_m = offs_am < M
        mask_a_scale_k = offs_a_scale_k < (K // VEC_SIZE)
        a_scale_mask = mask_a_scale_m[:, None] & mask_a_scale_k[None, :]
        a_scale_e8m0 = tl.load(a_scale_ptrs, mask=a_scale_mask, other=127)

        # B scale layout: [N, K//32] (new dot_scaled API -- "Do NOT transpose rhs_scale")
        # For tl.dot_scaled we need [BLOCK_SIZE_N, BLOCK_SIZE_K//32]
        offs_b_scale_k = k_block_start + tl.arange(0, BLOCK_SIZE_K // VEC_SIZE)
        b_scale_ptrs = b_scale_ptr + (offs_bn[:, None].to(tl.int64) * stride_b_scale_n +
                                       offs_b_scale_k[None, :] * stride_b_scale_k)

        mask_b_scale_n = offs_bn < N
        mask_b_scale_k = offs_b_scale_k < (K // VEC_SIZE)
        b_scale_mask = mask_b_scale_n[:, None] & mask_b_scale_k[None, :]
        b_scale_e8m0 = tl.load(b_scale_ptrs, mask=b_scale_mask, other=127)

        # Block-scaled matmul using tl.dot_scaled
        # tl.dot_scaled expects E8M0 scales (uint8) and handles conversion internally
        # a: [BLOCK_SIZE_M, BLOCK_SIZE_K] FP8
        # a_scale_e8m0: [BLOCK_SIZE_M, BLOCK_SIZE_K // VEC_SIZE] uint8 (E8M0)
        # b: [BLOCK_SIZE_K, BLOCK_SIZE_N] FP8 (already in correct layout, no transpose needed)
        # b_scale_e8m0: [BLOCK_SIZE_N, BLOCK_SIZE_K // VEC_SIZE] uint8 (E8M0)
        #   Note: rhs_scale uses [N, K//32] layout (NOT transposed) per new dot_scaled API
        accumulator = tl.dot_scaled(
            a,                # [BLOCK_SIZE_M, BLOCK_SIZE_K]
            a_scale_e8m0,     # [BLOCK_SIZE_M, BLOCK_SIZE_K // VEC_SIZE] E8M0
            FP8_FORMAT_A,     # "e4m3" or "e5m2"
            b,                # [BLOCK_SIZE_K, BLOCK_SIZE_N] - NO transpose
            b_scale_e8m0,     # [BLOCK_SIZE_K // VEC_SIZE, BLOCK_SIZE_N] E8M0
            FP8_FORMAT_B,     # "e4m3" or "e5m2"
            accumulator       # [BLOCK_SIZE_M, BLOCK_SIZE_N]
        )

        # Advance data pointers
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Apply α (GEMM output scale). Skipped via constexpr fast-path when α=1.
    if not ALPHA_IS_ONE:
        accumulator = accumulator * alpha

    # BIAS epilogue: add per-N vector (broadcast across M). Same ordering as
    # matmul_kernel -- bias is added *after* α, *before* the β-accumulate.
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    if EPILOGUE == 'BIAS':
        bias_ptrs = bias_ptr + offs_cn
        bias = tl.load(bias_ptrs, mask=(offs_cn < N), other=0.0).to(tl.float32)
        accumulator = accumulator + bias[None, :]

    # Store output (convert to target dtype)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    # int64 promotion (see matmul_kernel A/B).
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None].to(tl.int64) + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    # β accumulate: D = α·(A·B) + bias + β·C. Fold existing C in *before* the
    # dtype narrowing to keep accumulation in fp32. Matches matmul_kernel's ordering.
    if ACCUMULATE:
        existing_c = tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float32)
        accumulator = accumulator + beta * existing_c

    c = accumulator.to(c_ptr.type.element_ty)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 4, 'waves_per_eu': 0}, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'waves_per_eu': 0}, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 4, 'waves_per_eu': 2}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2}, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 32, 'waves_per_eu': 2}, num_warps=4),
    ],
    # Include the FP8 code-path flags so INPUT_FP8 / OUTPUT_FP8 variants get
    # separately-tuned configs (they change the loop body enough that the
    # best (BLOCK, GROUP, warps) can differ).
    key=['M', 'N', 'K', 'INPUT_FP8', 'OUTPUT_FP8'],
    # Ran into stream capture error when using cuda_graph, thus disabled.
    #use_cuda_graph=True,
    # With ACCUMULATE=True each benchmark iteration would add computed_c to
    # the output again, multi-copying the result. Snapshot and restore c_ptr
    # around every benchmark run. Harmless when ACCUMULATE=False (kernel
    # overwrites regardless); cost is paid once per shape during warmup.
    restore_value=['c_ptr'],
)
@triton.heuristics({
    'EVEN_K': lambda args: args['K'] % args['BLOCK_SIZE_K'] == 0,
})
@triton.jit
def matmul_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Pointers to scales
        a_scale_ptr, b_scale_ptr, c_scale_ptr,
        # Pointer to bias
        bias_ptr,
        # Pointer to amax
        c_amax_ptr,
        # GEMM output scale (α) and accumulate scale (β): D = α·(A·B) + bias + β·C
        alpha, beta,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        EVEN_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        EPILOGUE: tl.constexpr,
        # Whether multiplied by scale_a * scale_b
        INPUT_FP8: tl.constexpr,
        # Whether to output fp8 or not, if so, also calculate amax.
        OUTPUT_FP8: tl.constexpr,
        # β=1 accumulation: C := existing_C + α*A*B (used for fused wgrad accumulate)
        ACCUMULATE: tl.constexpr,
        # Fast-path toggle: skip `accumulator *= alpha` when α is known to be 1.0
        ALPHA_IS_ONE: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    M = blas_n, K = blas_k, N = blas_m
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m, pid_n = swizzle_pid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M)

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetics` section for details
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    # Promote the M/N offsets to int64 before multiplying by a stride that may
    # be as large as K or N: offs_am*stride_am overflows int32 whenever
    # (M-1)*K > 2^31 (e.g. M=143616, K=15360 → 2.2G). The mask arithmetic
    # (offs_* < M/N) stays int32 for speed; only the address computation
    # widens.
    a_ptrs = a_ptr + (offs_am[:, None].to(tl.int64) * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :].to(tl.int64) * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    acc_dtype = tl.float32 if c_ptr.type.element_ty != tl.int8 else tl.int32
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

    if INPUT_FP8:
        a_scale = tl.load(a_scale_ptr)
        b_scale = tl.load(b_scale_ptr)
        scale = a_scale * b_scale

    if OUTPUT_FP8:
        c_scale = tl.load(c_scale_ptr)

    if EPILOGUE == 'BGRADB' and not INPUT_FP8:
        bias_gradient = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)

        if EPILOGUE == 'BGRADB' and not INPUT_FP8:
            if pid_n == 0:
                ## It is necessary to upcast to fp32 for reduction to ensure accuracy.
                bias_gradient_partial = tl.sum(a.to(tl.float32), axis=1)
                bias_gradient += bias_gradient_partial

        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk


    if EPILOGUE == 'BGRADB' and not INPUT_FP8:
        if pid_n == 0:
            offs_bias_gradient = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            bias_gradient_ptrs = bias_ptr + offs_bias_gradient
            ## Though bias_gradient is fp32, type conversion will occur before store
            tl.store(bias_gradient_ptrs, bias_gradient, mask=(offs_bias_gradient<M))

    if INPUT_FP8:
        accumulator *= scale
    # Apply α (GEMM output scale). Skipped via constexpr fast-path when α=1.
    if not ALPHA_IS_ONE:
        accumulator = accumulator * alpha
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    if EPILOGUE == 'BIAS':
        offs_bias = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        bias_ptrs = bias_ptr + offs_bias
        bias = tl.load(bias_ptrs, mask=(offs_bias < N), other=0.0).to(tl.float32)
        accumulator = accumulator + bias[None, :]


    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    # int64 promotion (see matmul_kernel A/B).
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None].to(tl.int64) + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    # β accumulation: D = α·(A·B) + bias + β·C. Fold existing C in *before*
    # any FP8 output scaling so amax reflects the final value. Matches
    # hipBLASLt's epilogue ordering.
    if ACCUMULATE:
        existing_c = tl.load(c_ptrs, mask=c_mask, other=0.0).to(acc_dtype)
        accumulator = accumulator + beta * existing_c

    # Get amax first and then scale c before conversion to fp8
    if OUTPUT_FP8:
        tile_c_amax = tl.max(tl.abs(accumulator))
        tl.atomic_max(c_amax_ptr, tile_c_amax)
        c = (accumulator * c_scale).to(c_ptr.type.element_ty)
    else:
        c = accumulator.to(c_ptr.type.element_ty)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    tl.store(c_ptrs, c, mask=c_mask)
