# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#
# Blockwise FP8 grouped GEMM Triton kernels (MoE), adapted from AMD Primus-Turbo
# (primus_turbo/triton/grouped_gemm/grouped_gemm_fp8_kernel.py).

import contextlib
import os

import torch
import triton
import triton.language as tl

from .common import is_cdna3
from .gmm.pid_preprocessing import remap_xcd_chunked


def get_num_cus() -> int:
    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


# ── AMD compiler knobs (scoped) ──
# The AMD codegen toggles live on ``triton.knobs.amd`` and are process-global: they
# are read when a kernel is compiled, so leaving them set would silently change the
# codegen of every unrelated Triton kernel compiled afterwards (cast, gmm, MXFP8,
# ...). We therefore apply them only around our own launches and restore the prior
# state on exit. Triton's setter keeps the backing env var in sync with the
# attribute, so we snapshot/restore both.
_AMD_KNOB_ENV = {
    "use_async_copy": "TRITON_HIP_USE_ASYNC_COPY",
    "scalarize_packed_fops": "AMDGCN_SCALARIZE_PACKED_FOPS",
    "use_block_pingpong": "TRITON_HIP_USE_BLOCK_PINGPONG",
}


def _amd_knob_overrides(*, is_tn: bool) -> dict:
    """Knob values to apply for the enclosed GEMM launch.

    gfx950 (CDNA4): force async_copy / scalarize / block_pingpong on.
    gfx942 (CDNA3): async_copy / scalarize help NT/NN but regress TN/wgrad
    ~5-8%, so gate them on the GEMM layout (block_pingpong left at its default).
    """
    if is_cdna3():
        enable = not is_tn
        return {"use_async_copy": enable, "scalarize_packed_fops": enable}
    return {
        "use_async_copy": True,
        "scalarize_packed_fops": True,
        "use_block_pingpong": True,
    }


@contextlib.contextmanager
def _amd_compiler_knobs(*, is_tn: bool):
    """Temporarily apply AMD Triton compiler knobs, restoring prior state on exit.

    Scoped rather than set once because the knobs are process-global and read at
    kernel compile time; leaving them set would leak into other kernels' codegen.
    """
    amd = getattr(getattr(triton, "knobs", None), "amd", None)
    if amd is None:
        yield
        return
    overrides = _amd_knob_overrides(is_tn=is_tn)
    saved = {
        name: (name in amd.__dict__, amd.__dict__.get(name), os.environ.get(env_key))
        for name, env_key in _AMD_KNOB_ENV.items()
        if name in overrides
    }
    try:
        for name, value in overrides.items():
            setattr(amd, name, value)
        yield
    finally:
        for name, (had_override, override_val, env_val) in saved.items():
            if had_override:
                amd.__dict__[name] = override_val
            else:
                amd.__dict__.pop(name, None)
            env_key = _AMD_KNOB_ENV[name]
            if env_val is None:
                os.environ.pop(env_key, None)
            else:
                os.environ[env_key] = env_val


# Number of XCDs used for round-robin PID swizzling (see ``remap_xcd_chunked``)
# to improve L2 cache locality. This cannot be queried programmatically, so it is
# hardcoded. NOTE: 8 is only correct for a full-config SPX-mode device (e.g. MI300X/
# MI350X with all XCDs active). In MPX (partitioned) mode the per-partition XCD count
# differs, so this value would need to be adjusted accordingly.
NUM_XCDS = 8

# First call per autotune key uses balanced group_offs so the cached config
# is not locked to one uneven MoE routing (Triton keys on G/N/K, not splits).
_grouped_blockwise_warmed = set()
_grouped_blockwise_vk_warmed = set()


def _get_grouped_blockwise_autotune_configs():
    """Curated fwd/dgrad tiles from Primus-Turbo (32-config sweep → these 8).

    BLOCK_SIZE_N is pinned to 128: the kernel loads one B scale per N-tile
    (BN == SCALE_BLOCK_N). BN=256 applies the wrong scale to half the columns.
    """
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
                "CHUNK_SIZE": 32,
            },
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 4,
                "CHUNK_SIZE": 32,
            },
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
                "CHUNK_SIZE": 64,
            },
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 256,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 4,
                "CHUNK_SIZE": 32,
            },
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 256,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
                "CHUNK_SIZE": 32,
            },
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 256,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 4,
                "CHUNK_SIZE": 32,
            },
            num_warps=8,
            num_stages=1,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 256,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
                "CHUNK_SIZE": 64,
            },
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 256,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 4,
                "CHUNK_SIZE": 64,
            },
            num_warps=8,
            num_stages=1,
        ),
    ]


def _bwd_autotune_configs():
    """Curated variable-K wgrad tiles from Primus-Turbo. BLOCK_SIZE_K=128 always."""
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 4,
                "CHUNK_SIZE": 32,
            },
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 4,
                "CHUNK_SIZE": 32,
            },
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
                "CHUNK_SIZE": 32,
            },
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 256,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 4,
                "CHUNK_SIZE": 32,
            },
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 256,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
                "CHUNK_SIZE": 32,
            },
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 256,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 4,
                "CHUNK_SIZE": 32,
            },
            num_warps=8,
            num_stages=1,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 4,
                "CHUNK_SIZE": 32,
            },
            num_warps=8,
            num_stages=1,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
                "CHUNK_SIZE": 32,
            },
            num_warps=4,
            num_stages=2,
        ),
    ]


# Blockwise grouped FP8 kernel and public entrypoint


@triton.autotune(configs=_get_grouped_blockwise_autotune_configs(), key=["G", "N", "K"])
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
        pid = remap_xcd_chunked(pid, NUM_SMS, NUM_XCDS, CHUNK_SIZE)

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
            # Full K-blocks only; the ragged tail is handled below. This can be 0
            # (K < BLOCK_SIZE_K), so only assume non-negativity.
            loop_k -= 1
            tl.assume(loop_k >= 0)
        else:
            # K is a positive multiple of BLOCK_SIZE_K, so there is >= 1 full block.
            tl.assume(loop_k >= 1)

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
            A_LAST = (
                A + m_start_g * stride_am + rm[:, None] * stride_am + rk_last[None, :] * stride_ak
            )
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


def _grouped_gemm_fp8_blockwise_raw(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    group_offs: torch.Tensor,
    trans_b: bool = True,
    out_dtype: torch.dtype = torch.bfloat16,
    a_scales_pretransposed: bool = False,
) -> torch.Tensor:
    """Persistent grouped block-wise FP8 GEMM (CPU-sync-free) using Triton.

    Computes: out[offs[g]:offs[g+1], :] = A[offs[g]:offs[g+1], :] @ B_view[g]
    with block-wise scaling for each K-block.

    Args:
        a: [M_total, K] FP8 input (trans_a=False always).
        b: [G, N, K] (if trans_b=True) or [G, K, N] FP8 weights.
        a_scales: [M_total, K//128] float32, block-wise scale for A. When
            ``a_scales_pretransposed`` is True this is already ``[K//128, M_total]``
            (the GEMM-ready layout) and the internal transpose is skipped.
        b_scales: [G, ceil(N/128), ceil(K/128)] or [G, ceil(K/128), ceil(N/128)] float32.
        group_offs: [G+1] int64 prefix sum of group lengths.
        trans_b: If True, b[g] is [N, K] (transposed).
        out_dtype: Output dtype (default bfloat16).

    Returns:
        [M_total, N] output in out_dtype.
    """
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
    A_scales_t = a_scales if a_scales_pretransposed else a_scales.T.contiguous()
    num_sms = get_num_cus()
    even_k = K % 128 == 0

    def _launch(c_out, offs):
        _grouped_blockwise_fp8_persistent_gemm_kernel[(num_sms,)](
            a,
            b,
            c_out,
            A_scales_t,
            b_scales,
            offs,
            G,
            N,
            K,
            a.stride(0),
            stride_bg,
            stride_bn,
            c_out.stride(0),
            c_out.stride(1),
            A_scales_t.stride(0),
            A_scales_t.stride(1),
            b_scales.stride(0),
            stride_bs_n,
            stride_bs_k,
            stride_ak=stride_ak,
            stride_bk=stride_bk,
            NUM_SMS=num_sms,
            NUM_XCDS=NUM_XCDS,
            EVEN_K=even_k,
            CACHE_MODIFIER=".ca",
            waves_per_eu=0,
            matrix_instr_nonkdim=16,
            kpack=1,
        )

    # Triton's autotune key is (G, N, K) only. Prime once with balanced offs
    # so the cached config is not locked to the first uneven MoE routing.
    warm_key = (G, N, K, even_k)
    # trans_a is always False here: forward/dgrad are NT/NN (never TN). Scope the
    # AMD knobs to these launches so they don't leak into other Triton kernels.
    with _amd_compiler_knobs(is_tn=False):
        if warm_key not in _grouped_blockwise_warmed:
            _grouped_blockwise_warmed.add(warm_key)
            per = M_total // max(G, 1)
            bal_offs = torch.arange(G + 1, device=group_offs.device, dtype=group_offs.dtype) * per
            bal_offs[-1] = M_total
            _launch(torch.empty_like(out), bal_offs)

        _launch(out, group_offs)
    return out


def _stack_weight_qtensors(weights, torch_fp8_dtype):
    """Stack per-expert weight ``Float8BlockwiseQTensor`` into packed grouped
    buffers: FP8 data ``[G, N, K]`` and scales ``[G, ceil(N/128), ceil(K/128)]``."""
    b_fp8 = torch.stack([w._rowwise_data.view(torch_fp8_dtype) for w in weights], 0)
    b_scales = torch.stack([w._rowwise_scale_inv for w in weights], 0)
    return b_fp8, b_scales


def grouped_gemm_fp8_blockwise_triton_kernel(
    a,
    b,
    group_offs: torch.Tensor,
    trans_b: bool = True,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Grouped block-wise FP8 GEMM over ``Float8BlockwiseQTensor`` operands.

    ``a`` is a 1x128 rowwise-scaled ``Float8BlockwiseQTensor`` (``[M_total, K]``);
    its stored ``rowwise_scale_inv`` is already in the ``[K//128, M_total]`` GEMM
    layout. ``b`` is the weight operand in one of three forms:

    * a packed weight ``Float8BlockwiseQTensor`` -- either 2D ``[G*N, K]`` (scale
      ``[G*(N/128), K/128]``, reshaped to ``[G, N, K]`` here using ``G`` from
      ``group_offs``) or already 3D ``[G, N, K]``,
    * a list of per-expert 128x128 weight ``Float8BlockwiseQTensor`` (stacked here), or
    * a pre-stacked ``(b_fp8, b_scales)`` tuple of raw tensors (used by the backward
      pass to reuse the buffers built during the forward pass).

    Raw FP8 fields are extracted here and handed to
    :func:`_grouped_gemm_fp8_blockwise_raw`.
    """
    import transformer_engine_torch as tex
    from ..tensor.float8_blockwise_tensor import Float8BlockwiseQTensor
    from .common import te_dtype_to_torch_dtype

    a_dt = te_dtype_to_torch_dtype(tex.DType(int(a._fp8_dtype)))
    a_fp8 = a._rowwise_data.view(a_dt)
    a_scales = a._rowwise_scale_inv  # already [K//128, M_total]

    if isinstance(b, Float8BlockwiseQTensor):
        b_dt = te_dtype_to_torch_dtype(tex.DType(int(b._fp8_dtype)))
        b_fp8 = b._rowwise_data.view(b_dt)
        b_scales = b._rowwise_scale_inv
        if b_fp8.dim() == 2:
            # Packed 2D [G*N, K] grouped weight: recover [G, N, K] using G from offs.
            groups = group_offs.shape[0] - 1
            n = b_fp8.shape[0] // groups
            b_fp8 = b_fp8.view(groups, n, b_fp8.shape[1])
            bn = b_scales.shape[0] // groups
            b_scales = b_scales.view(groups, bn, b_scales.shape[1])
    elif isinstance(b, (list, tuple)) and isinstance(b[0], Float8BlockwiseQTensor):
        b_dt = te_dtype_to_torch_dtype(tex.DType(int(b[0]._fp8_dtype)))
        b_fp8, b_scales = _stack_weight_qtensors(b, b_dt)
    else:
        # Pre-stacked raw (b_fp8, b_scales); b_fp8 already an FP8-typed tensor.
        b_fp8, b_scales = b

    return _grouped_gemm_fp8_blockwise_raw(
        a_fp8,
        b_fp8,
        a_scales,
        b_scales,
        group_offs,
        trans_b=trans_b,
        out_dtype=out_dtype,
        a_scales_pretransposed=True,
    )


# ── Blockwise FP8 Variable-K Backward Public API ──


@triton.autotune(
    configs=_bwd_autotune_configs(),
    key=["G", "OUT_M", "OUT_N"],
)
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
    ACCUMULATE: tl.constexpr,
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
        pid = remap_xcd_chunked(pid, NUM_SMS, NUM_XCDS, CHUNK_SIZE)

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
        LHS_BASE = (
            LHS + m_start * stride_lhs_m + rm[:, None] * stride_lhs_n + rk[None, :] * stride_lhs_m
        )
        RHS_BASE = (
            RHS + m_start * stride_rhs_m + rk[:, None] * stride_rhs_m + rn[None, :] * stride_rhs_n
        )

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
        rm_s = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rn_s = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        rn_s = tl.max_contiguous(tl.multiple_of(rn_s, BLOCK_SIZE_N), BLOCK_SIZE_N)
        c_mask = (rm_s[:, None] < OUT_M) & (rn_s[None, :] < OUT_N)
        C_ = (
            C
            + group_idx.to(tl.int64) * stride_cg
            + rm_s[:, None] * stride_cm
            + rn_s[None, :] * stride_cn
        )
        c = acc.to(C.type.element_ty)
        if ACCUMULATE:
            c += tl.load(C_, mask=c_mask, other=0)
        tl.store(C_, c, c_mask)


# ── Blockwise FP8 Forward Public API ──


def grouped_gemm_fp8_blockwise_variable_k_triton_kernel(
    lhs,
    rhs,
    out_dtype: torch.dtype = torch.bfloat16,
    out: torch.Tensor = None,
    accumulate: bool = False,
) -> torch.Tensor:
    """Variable-K grouped block-wise FP8 GEMM (backward, 1D+1D scaling) using Triton.

    ``lhs`` and ``rhs`` are ``Float8BlockwiseQTensor`` operands carrying the
    segment-padded columnwise-quantized grad and activation in their columnwise
    slots (``_columnwise_data`` ``[M_pad, N]``, ``_columnwise_scale_inv``
    ``[ceil(M_pad/128), N]``) plus the shared padded segment offsets in
    ``_vk_group_offs``. Raw FP8 fields are extracted here (uint8 data viewed back to
    the FP8 dtype) and handed to :func:`_grouped_gemm_fp8_blockwise_variable_k_raw`.
    """
    import transformer_engine_torch as tex
    from .common import te_dtype_to_torch_dtype

    lhs_dt = te_dtype_to_torch_dtype(tex.DType(int(lhs._fp8_dtype)))
    rhs_dt = te_dtype_to_torch_dtype(tex.DType(int(rhs._fp8_dtype)))
    return _grouped_gemm_fp8_blockwise_variable_k_raw(
        lhs._columnwise_data.view(lhs_dt),
        rhs._columnwise_data.view(rhs_dt),
        lhs._columnwise_scale_inv,
        rhs._columnwise_scale_inv,
        lhs._vk_group_offs,
        out_dtype=out_dtype,
        out=out,
        accumulate=accumulate,
    )


def _grouped_gemm_fp8_blockwise_variable_k_raw(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    lhs_scales: torch.Tensor,
    rhs_scales: torch.Tensor,
    group_offs: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    out: torch.Tensor = None,
    accumulate: bool = False,
) -> torch.Tensor:
    """Variable-K grouped block-wise FP8 GEMM (backward, 1D+1D scaling) using Triton.

    Computes: C[g] = lhs[offs[g]:offs[g+1]]^T @ rhs[offs[g]:offs[g+1]]
    with 1D+1D block-wise scaling applied in the K-loop.

    Output: [G, OUT_M, OUT_N]. When ``out`` is provided the kernel writes in-place
    (and adds into it if ``accumulate``). ``out_dtype`` is ignored in that case.
    """
    assert lhs.ndim == 2 and rhs.ndim == 2
    assert lhs.shape[0] == rhs.shape[0]
    OUT_M = lhs.shape[1]
    OUT_N = rhs.shape[1]
    G = group_offs.shape[0] - 1

    if out is None:
        assert not accumulate, "accumulate=True requires an existing out tensor"
        out = torch.empty((G, OUT_M, OUT_N), device=lhs.device, dtype=out_dtype)
    else:
        assert out.shape == (
            G,
            OUT_M,
            OUT_N,
        ), f"out must be {(G, OUT_M, OUT_N)}, got {tuple(out.shape)}"
    num_sms = get_num_cus()

    def _launch(c_out, offs, accumulate_flag):
        _grouped_blockwise_fp8_variable_k_gemm_kernel[(num_sms,)](
            lhs,
            rhs,
            c_out,
            lhs_scales,
            rhs_scales,
            offs,
            G,
            OUT_M,
            OUT_N,
            lhs.stride(0),
            rhs.stride(0),
            c_out.stride(0),
            c_out.stride(1),
            c_out.stride(2),
            lhs_scales.stride(0),
            lhs_scales.stride(1),
            rhs_scales.stride(0),
            rhs_scales.stride(1),
            stride_lhs_n=lhs.stride(1),
            stride_rhs_n=rhs.stride(1),
            NUM_SMS=num_sms,
            NUM_XCDS=NUM_XCDS,
            CACHE_MODIFIER=".ca",
            ACCUMULATE=accumulate_flag,
            waves_per_eu=0,
            matrix_instr_nonkdim=16,
            kpack=1,
        )

    # Autotune key is (G, OUT_M, OUT_N). Prime on a scratch buffer with
    # balanced padded offs so accumulate=True never benchmarks into ``out``.
    warm_key = (G, OUT_M, OUT_N, out.dtype, accumulate)
    # Variable-K wgrad is TN (lhs^T @ rhs). Scope the AMD knobs to these launches
    # so they don't leak into other Triton kernels.
    with _amd_compiler_knobs(is_tn=True):
        if warm_key not in _grouped_blockwise_vk_warmed:
            _grouped_blockwise_vk_warmed.add(warm_key)
            m_padded = lhs.shape[0]
            per = max((m_padded // max(G, 1)) // 128 * 128, 128)
            bal_offs = torch.arange(G + 1, device=group_offs.device, dtype=group_offs.dtype) * per
            bal_offs[-1] = m_padded
            _launch(torch.zeros_like(out), bal_offs, False)

        _launch(out, group_offs, accumulate)
    return out
