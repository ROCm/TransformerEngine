# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#
# Blockwise FP8 grouped GEMM Triton kernels (MoE), adapted from AMD Primus-Turbo
# (primus_turbo/triton/grouped_gemm/grouped_gemm_fp8_kernel.py).

import torch
import triton
import triton.language as tl


def is_gfx950() -> bool:
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return "gfx950" in props.gcnArchName


def get_num_cus() -> int:
    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


# AMD gfx950 compiler knobs.
_KNOBS_SET = False


def set_triton_knobs_gfx950() -> None:
    global _KNOBS_SET
    if _KNOBS_SET:
        return
    _KNOBS_SET = True
    if hasattr(triton, "knobs") and hasattr(triton.knobs, "amd"):
        triton.knobs.amd.use_async_copy = True
        triton.knobs.amd.scalarize_packed_fops = True
        triton.knobs.amd.use_block_pingpong = True
    else:
        import os
        os.environ.setdefault("TRITON_HIP_USE_ASYNC_COPY", "1")
        os.environ.setdefault("AMDGCN_SCALARIZE_PACKED_FOPS", "1")
        os.environ.setdefault("TRITON_HIP_USE_BLOCK_PINGPONG", "1")


def _set_amd_knobs(enable: bool = True):
    """Set AMD-specific Triton knobs (non-gfx950 fallback)."""
    if hasattr(triton, "knobs") and hasattr(triton.knobs, "amd"):
        triton.knobs.amd.use_async_copy = enable
        triton.knobs.amd.scalarize_packed_fops = enable


# XCD/chiplet PID remap.
NUM_XCDS = 8


@triton.jit
def _chiplet_transform_chunked(
    pid,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    if pid > (NUM_SMS // (NUM_XCDS * CHUNK_SIZE)) * (NUM_XCDS * CHUNK_SIZE):
        return pid
    local_pid = pid // NUM_XCDS
    chunk_idx = local_pid // CHUNK_SIZE
    pos_in_chunk = local_pid % CHUNK_SIZE
    xcd = pid % NUM_XCDS
    return chunk_idx * NUM_XCDS * CHUNK_SIZE + xcd * CHUNK_SIZE + pos_in_chunk


# Blockwise grouped FP8 kernel and public entrypoint

@triton.jit()
def _grouped_blockwise_fp8_persistent_gemm_kernel(
    # Pointers
    A,  # [M_total, K] FP8
    B,  # [G, ?, ?] FP8
    C,  # [M_total, N]
    A_scales_ptr,  # [K//128, M_total] float32 (pre-transposed for coalesced access)
    B_scales_ptr,  # [G, ?, ?] float32 (block-wise, layout depends on trans_b)
    group_offs_ptr,  # [G+1] int64
    # Dimensions
    G,  # number of groups (runtime)
    N,
    K,
    # Strides
    stride_am,  # A row stride
    stride_bg,  # B group stride: b.stride(0)
    stride_bn,  # B N-stride (within a group)
    stride_cm,  # C row stride
    stride_cn,  # C col stride
    # A_scales strides (pre-transposed: [K//128, M_total])
    stride_as_k,  # A_scales_t.stride(0)
    stride_as_m,  # A_scales_t.stride(1)
    # B_scales strides
    stride_bs_g,  # B_scales.stride(0) — group stride
    stride_bs_n,  # stride along N-block dimension
    stride_bs_k,  # stride along K-block dimension
    # Constexpr strides
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    # Tile config
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    EVEN_K: tl.constexpr,
    CACHE_MODIFIER: tl.constexpr,
):
    """Persistent grouped block-wise FP8 GEMM kernel (CPU-sync-free)."""
    pid = tl.program_id(0)
    if NUM_XCDS != 1:
        pid = _chiplet_transform_chunked(pid, NUM_SMS, NUM_XCDS, CHUNK_SIZE)

    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # ── Compute total tiles across all groups ──
    total_tiles: tl.int32 = 0
    for _g in range(G):
        m_g = (tl.load(group_offs_ptr + _g + 1) - tl.load(group_offs_ptr + _g)).to(tl.int32)
        total_tiles += tl.cdiv(m_g, BLOCK_SIZE_M) * num_pid_n

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    acc_dtype = tl.float32

    for global_tile_id in range(pid, total_tiles, NUM_SMS):
        # ── Find group via linear scan (O(G)) ──
        group_idx: tl.int32 = 0
        tile_start: tl.int32 = 0
        cumsum: tl.int32 = 0
        for _g in range(G):
            m_g_i = (tl.load(group_offs_ptr + _g + 1) - tl.load(group_offs_ptr + _g)).to(tl.int32)
            tiles_g = tl.cdiv(m_g_i, BLOCK_SIZE_M) * num_pid_n
            new_cumsum = cumsum + tiles_g
            if global_tile_id >= new_cumsum:
                group_idx = _g + 1
                tile_start = new_cumsum
            cumsum = new_cumsum

        # ── Group-local tile → (pid_m, pid_n) ──
        local_tile = global_tile_id - tile_start
        m_start_g = tl.load(group_offs_ptr + group_idx)  # int64
        M_g = (tl.load(group_offs_ptr + group_idx + 1) - m_start_g).to(tl.int32)
        tiles_m_g = tl.cdiv(M_g, BLOCK_SIZE_M)

        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        swizzle_group = local_tile // num_pid_in_group
        first_pid_m = swizzle_group * GROUP_SIZE_M
        group_size_m = min(tiles_m_g - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((local_tile % num_pid_in_group) % group_size_m)
        pid_n = (local_tile % num_pid_in_group) // group_size_m
        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        # ── Address computation ──
        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M_g
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rk = tl.arange(0, BLOCK_SIZE_K)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        group_offset_b = group_idx.to(tl.int64) * stride_bg
        A_BASE = A + m_start_g * stride_am + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B + group_offset_b + rk[:, None] * stride_bk + rn[None, :] * stride_bn

        # A_scales pointer: pre-transposed [K//128, M_total]
        as_ptrs_base = A_scales_ptr + (m_start_g + rm.to(tl.int64)) * stride_as_m

        # B_scales pointer: B_scales[g, pn, ki] (2D block scaling)
        bs_ptr_base = B_scales_ptr + group_idx.to(tl.int64) * stride_bs_g + pid_n * stride_bs_n

        # ── K-loop with block-wise scaling (EVEN_K pattern) ──
        loop_k = tl.cdiv(K, BLOCK_SIZE_K)
        if not EVEN_K:
            loop_k -= 1
        tl.assume(loop_k > 1)

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

        for ki in range(0, loop_k):
            if stride_ak == 1:
                a = tl.load(tl.multiple_of(A_BASE, (1, 16)), cache_modifier=CACHE_MODIFIER)
            else:
                a = tl.load(tl.multiple_of(A_BASE, (16, 1)), cache_modifier=CACHE_MODIFIER)

            if stride_bk == 1:
                b = tl.load(tl.multiple_of(B_BASE, (16, 1)), cache_modifier=CACHE_MODIFIER)
            else:
                b = tl.load(tl.multiple_of(B_BASE, (1, 16)), cache_modifier=CACHE_MODIFIER)

            partial = tl.dot(a, b)

            # Block-wise scales: a_s is [BLOCK_M] vector, b_s is scalar
            a_s = tl.load(as_ptrs_base + ki * stride_as_k)
            b_s = tl.load(bs_ptr_base + ki * stride_bs_k)
            acc += partial * (a_s * b_s)[:, None]

            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        if not EVEN_K:
            # ── Last partial K-block (masked) ──
            rk_last = loop_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            A_LAST = A + m_start_g * stride_am + rm[:, None] * stride_am + rk_last[None, :] * stride_ak
            B_LAST = B + group_offset_b + rk_last[:, None] * stride_bk + rn[None, :] * stride_bn
            if stride_ak == 1:
                A_LAST = tl.multiple_of(A_LAST, (1, 16))
            else:
                A_LAST = tl.multiple_of(A_LAST, (16, 1))
            if stride_bk == 1:
                B_LAST = tl.multiple_of(B_LAST, (16, 1))
            else:
                B_LAST = tl.multiple_of(B_LAST, (1, 16))
            a = tl.load(A_LAST, mask=rk_last[None, :] < K, other=0.0, cache_modifier=CACHE_MODIFIER)
            b = tl.load(B_LAST, mask=rk_last[:, None] < K, other=0.0, cache_modifier=CACHE_MODIFIER)
            partial = tl.dot(a, b)
            a_s = tl.load(as_ptrs_base + loop_k * stride_as_k)
            b_s = tl.load(bs_ptr_base + loop_k * stride_bs_k)
            acc += partial * (a_s * b_s)[:, None]

        # ── Store output ──
        c = acc.to(C.type.element_ty)
        rm_s = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M_g
        rn_s = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rn_s = tl.max_contiguous(tl.multiple_of(rn_s, BLOCK_SIZE_N), BLOCK_SIZE_N)
        c_mask = (rm_s[:, None] < M_g) & (rn_s[None, :] < N)
        C_ = C + m_start_g * stride_cm + rm_s[:, None] * stride_cm + rn_s[None, :] * stride_cn
        tl.store(C_, c, c_mask)


# ═══════════════════════════════════════════════════════════════════════════════
# Blockwise FP8 Variable-K Backward Kernel (persistent, CPU-sync-free)
#
# Computes: C[g] = LHS[g]^T @ RHS[g] with 1D+1D block-wise scales
# ═══════════════════════════════════════════════════════════════════════════════



def grouped_gemm_fp8_blockwise_triton_kernel(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    group_offs: torch.Tensor,
    trans_b: bool = True,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Persistent grouped block-wise FP8 GEMM (CPU-sync-free) using Triton.

    Computes: out[offs[g]:offs[g+1], :] = A[offs[g]:offs[g+1], :] @ B_view[g]
    with block-wise scaling for each K-block.

    Args:
        a: [M_total, K] FP8 input (trans_a=False always).
        b: [G, N, K] (if trans_b=True) or [G, K, N] FP8 weights.
        a_scales: [M_total, K//128] float32, block-wise scale for A.
        b_scales: [G, ceil(N/128), ceil(K/128)] or [G, ceil(K/128), ceil(N/128)] float32.
        group_offs: [G+1] int64 prefix sum of group lengths.
        trans_b: If True, b[g] is [N, K] (transposed).
        out_dtype: Output dtype (default bfloat16).

    Returns:
        [M_total, N] output in out_dtype.
    """
    if is_gfx950():
        set_triton_knobs_gfx950()
    else:
        _set_amd_knobs(enable=True)

    assert a.ndim == 2, f"a must be 2D, got {a.shape}"
    assert b.ndim == 3, f"b must be 3D, got {b.shape}"
    assert b_scales.ndim == 3, f"b_scales must be 3D, got {b_scales.shape}"

    M_total, K = a.shape
    G = b.shape[0]

    if trans_b:
        N = b.shape[1]
        stride_bk = b.stride(2)
        stride_bn = b.stride(1)
        stride_bs_n = b_scales.stride(1)
        stride_bs_k = b_scales.stride(2)
    else:
        N = b.shape[2]
        stride_bk = b.stride(1)
        stride_bn = b.stride(2)
        stride_bs_n = b_scales.stride(2)
        stride_bs_k = b_scales.stride(1)

    stride_bg = b.stride(0)
    stride_ak = a.stride(1)

    out = torch.empty((M_total, N), device=a.device, dtype=out_dtype)
    A_scales_t = a_scales.T.contiguous()
    num_sms = get_num_cus()

    blk_m = 256
    blk_n = 128  # Keep 128 to match B_scale block alignment
    blk_k = 128
    even_k = K % blk_k == 0

    # GROUP_SIZE_M heuristic (match tensorwise)
    tiles_m_per_group = (M_total + G * blk_m - 1) // (G * blk_m)
    tiles_n = (N + blk_n - 1) // blk_n
    group_m = 8 if min(tiles_m_per_group, tiles_n) < 16 else 4

    _grouped_blockwise_fp8_persistent_gemm_kernel[(num_sms,)](
        a,
        b,
        out,
        A_scales_t,
        b_scales,
        group_offs,
        G,
        N,
        K,
        a.stride(0),
        stride_bg,
        stride_bn,
        out.stride(0),
        out.stride(1),
        A_scales_t.stride(0),
        A_scales_t.stride(1),
        b_scales.stride(0),
        stride_bs_n,
        stride_bs_k,
        stride_ak=stride_ak,
        stride_bk=stride_bk,
        BLOCK_SIZE_M=blk_m,
        BLOCK_SIZE_N=blk_n,
        BLOCK_SIZE_K=blk_k,
        GROUP_SIZE_M=group_m,
        NUM_SMS=num_sms,
        NUM_XCDS=NUM_XCDS,
        CHUNK_SIZE=32,
        EVEN_K=even_k,
        CACHE_MODIFIER=".ca",
        num_warps=8,
        num_stages=1,  # 256×128×128 needs 48KB/stage; 2 stages=96KB > 64KB LDS
        waves_per_eu=0,
        matrix_instr_nonkdim=16,
        kpack=1,
    )
    return out


# ── Blockwise FP8 Variable-K Backward Public API ──




@triton.jit()
def _grouped_blockwise_fp8_variable_k_gemm_kernel(
    # C[g] = LHS_g^T @ RHS_g * block_scales
    LHS,  # [M_padded_total, OUT_M] FP8
    RHS,  # [M_padded_total, OUT_N] FP8
    C,  # [G, OUT_M, OUT_N]
    LHS_scales_ptr,  # [ceil(M_padded/128), OUT_M] float32
    RHS_scales_ptr,  # [ceil(M_padded/128), OUT_N] float32
    group_offs_ptr,  # [G+1] int64 (padded segment offsets, each aligned to 128)
    G,  # number of groups
    OUT_M,
    OUT_N,
    # Strides
    stride_lhs_m,
    stride_rhs_m,
    stride_cg,
    stride_cm,
    stride_cn,
    # LHS_scales strides
    stride_ls_0,
    stride_ls_1,
    # RHS_scales strides
    stride_rs_0,
    stride_rs_1,
    # Constexpr strides
    stride_lhs_n: tl.constexpr,
    stride_rhs_n: tl.constexpr,
    # Tile config
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    CACHE_MODIFIER: tl.constexpr,
):
    """Persistent grouped block-wise FP8 variable-K GEMM kernel (backward, CPU-sync-free).

    All groups share the same output dims (OUT_M × OUT_N), only the inner product
    dimension M_g varies per group. 1D+1D scale pattern for TN/CRR layout.

    NOTE: Data is segment-padded to BLOCK_SIZE_K (128) boundaries by
    quant_fp8_blockwise_segment_m_impl, so M_g is always a multiple of
    BLOCK_SIZE_K. No masking is needed in the K-loop.
    """
    pid = tl.program_id(0)
    if NUM_XCDS != 1:
        pid = _chiplet_transform_chunked(pid, NUM_SMS, NUM_XCDS, CHUNK_SIZE)

    tiles_m = tl.cdiv(OUT_M, BLOCK_SIZE_M)
    tiles_n = tl.cdiv(OUT_N, BLOCK_SIZE_N)
    tiles_per_group = tiles_m * tiles_n
    total_tiles = G * tiles_per_group

    tl.assume(stride_lhs_m > 0)
    tl.assume(stride_lhs_n > 0)
    tl.assume(stride_rhs_m > 0)
    tl.assume(stride_rhs_n > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    acc_dtype = tl.float32

    for global_tile in range(pid, total_tiles, NUM_SMS):
        # ── Map to (group, local_tile) ──
        group_idx = global_tile // tiles_per_group
        local_tile = global_tile - group_idx * tiles_per_group

        # ── Swizzle local tile → (pid_m, pid_n) ──
        num_pid_in_group = GROUP_SIZE_M * tiles_n
        swizzle_group = local_tile // num_pid_in_group
        first_pid_m = swizzle_group * GROUP_SIZE_M
        group_size_m = min(tiles_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((local_tile % num_pid_in_group) % group_size_m)
        pid_n = (local_tile % num_pid_in_group) // group_size_m
        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        # ── Group boundaries ──
        m_start = tl.load(group_offs_ptr + group_idx)  # int64
        M_g = (tl.load(group_offs_ptr + group_idx + 1) - m_start).to(tl.int32)

        # ── Output indices ──
        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % OUT_M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % OUT_N
        rk = tl.arange(0, BLOCK_SIZE_K)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        # ── Base pointers ──
        LHS_BASE = LHS + m_start * stride_lhs_m + rm[:, None] * stride_lhs_n + rk[None, :] * stride_lhs_m
        RHS_BASE = RHS + m_start * stride_rhs_m + rk[:, None] * stride_rhs_m + rn[None, :] * stride_rhs_n

        scale_row_start = m_start // BLOCK_SIZE_K

        # ── K-loop over M_g with block-wise 1D+1D scaling ──
        # M_g is always a multiple of BLOCK_SIZE_K (data padded), so no masking needed.
        loop_k = M_g // BLOCK_SIZE_K
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

        for k in range(loop_k):
            if stride_lhs_n == 1:
                a = tl.load(
                    tl.multiple_of(LHS_BASE, (16, 1)),
                    cache_modifier=CACHE_MODIFIER,
                )
            else:
                a = tl.load(
                    tl.multiple_of(LHS_BASE, (1, 16)),
                    cache_modifier=CACHE_MODIFIER,
                )

            if stride_rhs_n == 1:
                b = tl.load(
                    tl.multiple_of(RHS_BASE, (1, 16)),
                    cache_modifier=CACHE_MODIFIER,
                )
            else:
                b = tl.load(
                    tl.multiple_of(RHS_BASE, (16, 1)),
                    cache_modifier=CACHE_MODIFIER,
                )

            partial = tl.dot(a, b)

            # 1D+1D block-wise scales
            scale_row = scale_row_start + k
            a_s = tl.load(LHS_scales_ptr + scale_row * stride_ls_0 + rm * stride_ls_1)
            b_s = tl.load(RHS_scales_ptr + scale_row * stride_rs_0 + rn * stride_rs_1)
            acc += partial * a_s[:, None] * b_s[None, :]

            LHS_BASE += BLOCK_SIZE_K * stride_lhs_m
            RHS_BASE += BLOCK_SIZE_K * stride_rhs_m

        # ── Store output ──
        c = acc.to(C.type.element_ty)
        rm_s = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rn_s = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        rn_s = tl.max_contiguous(tl.multiple_of(rn_s % OUT_N, BLOCK_SIZE_N), BLOCK_SIZE_N)
        c_mask = (rm_s[:, None] < OUT_M) & (rn_s[None, :] < OUT_N)
        C_ = C + group_idx.to(tl.int64) * stride_cg + rm_s[:, None] * stride_cm + rn_s[None, :] * stride_cn
        tl.store(C_, c, c_mask)


# ── Blockwise FP8 Forward Public API ──



def grouped_gemm_fp8_blockwise_variable_k_triton_kernel(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    lhs_scales: torch.Tensor,
    rhs_scales: torch.Tensor,
    group_offs: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Variable-K grouped block-wise FP8 GEMM (backward, 1D+1D scaling) using Triton.

    Computes: C[g] = lhs[offs[g]:offs[g+1]]^T @ rhs[offs[g]:offs[g+1]]
    with 1D+1D block-wise scaling applied in the K-loop.

    Output: [G, OUT_M, OUT_N].

    Args:
        lhs: [M_padded_total, OUT_M] FP8 (segment-padded, each segment aligned to 128).
        rhs: [M_padded_total, OUT_N] FP8.
        lhs_scales: [ceil(M_padded/128), OUT_M] float32.
        rhs_scales: [ceil(M_padded/128), OUT_N] float32.
        group_offs: [G+1] int64 padded segment offsets.
        out_dtype: Output dtype (default bfloat16).

    Returns:
        [G, OUT_M, OUT_N] output.
    """
    if is_gfx950():
        set_triton_knobs_gfx950()
    else:
        _set_amd_knobs(enable=False)

    assert lhs.ndim == 2 and rhs.ndim == 2
    assert lhs.shape[0] == rhs.shape[0]
    OUT_M = lhs.shape[1]
    OUT_N = rhs.shape[1]
    G = group_offs.shape[0] - 1

    out = torch.empty((G, OUT_M, OUT_N), device=lhs.device, dtype=out_dtype)
    num_sms = get_num_cus()

    # Use 128x128 tiles to reduce register pressure from double-accumulator
    # (partial + acc both need full tile VGPRs for blockwise scale application).
    # With 256x256 tiles + 8 warps, 2 accumulator sets need ~290 VGPRs/wave
    # which exceeds the 256 limit at 2 waves/SIMD, causing spilling.
    # 128x128 with 4 warps keeps VGPRs at ~170/wave, fitting 2 waves/SIMD.
    _grouped_blockwise_fp8_variable_k_gemm_kernel[(num_sms,)](
        lhs,
        rhs,
        out,
        lhs_scales,
        rhs_scales,
        group_offs,
        G,
        OUT_M,
        OUT_N,
        lhs.stride(0),
        rhs.stride(0),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        lhs_scales.stride(0),
        lhs_scales.stride(1),
        rhs_scales.stride(0),
        rhs_scales.stride(1),
        stride_lhs_n=lhs.stride(1),
        stride_rhs_n=rhs.stride(1),
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=128,
        GROUP_SIZE_M=4,
        NUM_SMS=num_sms,
        NUM_XCDS=NUM_XCDS,
        CHUNK_SIZE=32,
        CACHE_MODIFIER=".ca",
        num_warps=4,
        num_stages=2,
        waves_per_eu=0,
        matrix_instr_nonkdim=16,
        kpack=1,
    )
    return out
