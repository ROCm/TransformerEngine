# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
#
# Blockwise FP8 GEMM Triton kernel: a unified NT/NN/TN kernel covering the
# forward, dgrad and wgrad layouts. Adapted from AMD Primus-Turbo
# (primus_turbo/triton/gemm/gemm_fp8_kernel.py, blockwise section).

import itertools
import os

import torch
import triton
import triton.language as tl


# Local gfx950 check (TODO: use TE's common.get_arch).
def is_gfx950() -> bool:
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return "gfx950" in props.gcnArchName


# AMD gfx950 compiler knobs (from Primus triton_knobs_helper.py).
_KNOBS_SET = False


def set_triton_knobs_gfx950() -> None:
    """Enable AMD compiler knobs for gfx950 (async_copy, block_pingpong, scalarize)."""
    global _KNOBS_SET
    if _KNOBS_SET:
        return
    _KNOBS_SET = True
    if hasattr(triton, "knobs") and hasattr(triton.knobs, "amd"):
        triton.knobs.amd.use_async_copy = True
        triton.knobs.amd.scalarize_packed_fops = True
        triton.knobs.amd.use_block_pingpong = True
    else:
        os.environ.setdefault("TRITON_HIP_USE_ASYNC_COPY", "1")
        os.environ.setdefault("AMDGCN_SCALARIZE_PACKED_FOPS", "1")
        os.environ.setdefault("TRITON_HIP_USE_BLOCK_PINGPONG", "1")


# Blockwise FP8 GEMM kernel, autotune configs and public entrypoint
# (from Primus gemm_fp8_kernel.py).
def _get_blockwise_autotune_configs(
    allow_num_stages_3: bool = True,
    extra_matrix_instr_nonkdim: list | None = None,
):
    """Generate blockwise FP8 GEMM autotune configs.

    Args:
        allow_num_stages_3: whether to include `num_stages=3` candidates.
            Set False for the dual strided-K wgrad kernel which hits a
            Triton 3.7.0 LLVM-backend assertion at ns=3.
        extra_matrix_instr_nonkdim: extra `matrix_instr_nonkdim` values to
            stack on top of the Triton default (which selects 32x32x64 for
            FP8 on gfx950). For example, passing ``[16]`` doubles the
            candidate space by emitting an extra copy of every (BM, BN,
            ns, gm, chunk) tuple with ``matrix_instr_nonkdim=16`` so the
            autotuner can pick `v_mfma_f32_16x16x128_f8f6f4` when that
            instruction wins (typically the AGPR-bound NN/dgrad path).
            ``None`` keeps the original search space.
    """
    configs = []
    if is_gfx950():
        num_stage_values = [1, 2, 3] if allow_num_stages_3 else [1, 2]
        block_n_values = [64, 128]
    else:
        num_stage_values = [1, 2]
        block_n_values = [128]
    for block_m, block_n, kp, gm, chunk, ns in itertools.product(
        [128, 256],
        block_n_values,
        [1, 2],
        [4, 8],
        [32, 64],
        num_stage_values,
    ):
        nw = 4 if block_m == 128 else 8
        # Default-nonkdim (Triton picks 32x32x64 for FP8 on gfx950).
        configs.append(
            triton.Config(
                {
                    "BLOCK_M": block_m,
                    "BLOCK_N": block_n,
                    "BLOCK_K": 128,
                    "GROUP_M": gm,
                    "NUM_XCDS": 8,
                    "CHUNK": chunk,
                },
                num_warps=nw,
                num_stages=ns,
                pre_hook=None,
            )
        )
        # Extra nonkdim variants are only added for BM=256 + nw=8 configs,
        # which is the only branch where 32x32x64 actually overflows into
        # AGPR. BM=128 + nw=4 already fits 16 AGPR comfortably, so we keep
        # the search space tight and skip the duplicate compile cost there.
        if extra_matrix_instr_nonkdim and block_m == 256:
            for nonkdim in extra_matrix_instr_nonkdim:
                configs.append(
                    triton.Config(
                        {
                            "BLOCK_M": block_m,
                            "BLOCK_N": block_n,
                            "BLOCK_K": 128,
                            "GROUP_M": gm,
                            "NUM_XCDS": 8,
                            "CHUNK": chunk,
                            "matrix_instr_nonkdim": nonkdim,
                        },
                        num_warps=nw,
                        num_stages=ns,
                        pre_hook=None,
                    )
                )
    return configs


# ═══════════════════════════════════════════════════════════════════════════════
# Unified blockwise FP8 GEMM kernel — single jit body shared by NT/NN/TN.
#
# Layout differences are guarded by `tl.constexpr` flags so each (key tuple)
# compiles to an independent binary equivalent to the previous per-layout
# kernels (NT forward, NN dgrad, TN wgrad). The mapping is:
#
#     Layout │ A_K_CONTIGUOUS │ B_K_CONTIGUOUS │ SCALE_2D_B │ EVEN_K │ TRANS_C_STORE
#     ───────┼────────────────┼────────────────┼────────────┼────────┼──────────────
#     NT     │ True           │ True           │ True       │ K%128==0 │ False
#     NN     │ True           │ False          │ True       │ K%128==0 │ False
#     TN     │ False          │ False          │ False      │ False    │ True
#
# Kept invariants (do not break without re-benchmarking):
#   * `EVEN_K` fast path skips the K-tail mask on NT/NN (removes two
#     `v_cmp_*` + one `v_cndmask_*` per K iteration). TN keeps the mask path
#     because its `b_ptrs += BLOCK_K * stride_bk_val` arithmetic combined with
#     `num_stages=3` triggered the historic Triton 3.7 LLVM-backend
#     `Begin <= End` assertion; the TN autotune wrapper below already drops
#     `ns=3`, but EVEN_K=False on TN is the matched safety net.
#   * `TRANS_C_STORE=True` takes the `tl.trans(acc)` epilogue so the BF16
#     write coalesces into `(buffer|global)_store_dwordx4` under
#     `trans_c=True` (stride_cm=1, stride_cn=N). NT/NN (`trans_c=False`) keep
#     the direct store; their output buffer is BN-contiguous, so the direct
#     epilogue is already vectorised.
#
# Three `triton.autotune` wrappers below specialise the kernel along the
# dimensions that historically required separate search spaces:
#   * NT  : 96 configs, key=("M","N","K","EVEN_K"), keeps `num_stages=3`.
#   * NN  : 144 configs (96 + matrix_instr_nonkdim=16 stack on BM=256),
#           key=("M","N","K","EVEN_K"), keeps `num_stages=3` (the nonkdim=16
#           candidate is what avoids the 16-AGPR overflow on
#           BM=256/nw=8/ns=3).
#   * TN  : 64 configs (no `num_stages=3` — dual strided-K + ns=3 hits the
#           Triton 3.7 AMD-backend assertion mentioned above),
#           key=("M","N","K").
# ═══════════════════════════════════════════════════════════════════════════════
@triton.jit
def _blockwise_fp8_unified_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    A_scales_ptr,
    B_scales_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak_val,
    stride_bk_val,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_as_k,
    stride_as_m,
    stride_bs_0,
    stride_bs_1,
    NUM_SMS,
    NUM_K_BLOCKS,
    A_K_CONTIGUOUS: tl.constexpr,
    B_K_CONTIGUOUS: tl.constexpr,
    SCALE_2D_B: tl.constexpr,
    EVEN_K: tl.constexpr,
    TRANS_C_STORE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK: tl.constexpr,
):
    pid = tl.program_id(0)

    # XCD-aware PID transform (inlined from _chiplet_transform_chunked
    # because NUM_SMS is not constexpr in this kernel).
    if NUM_XCDS != 1:
        full_chunk_pids = (NUM_SMS // (NUM_XCDS * CHUNK)) * (NUM_XCDS * CHUNK)
        if pid <= full_chunk_pids:
            local_pid = pid // NUM_XCDS
            chunk_idx = local_pid // CHUNK
            pos_in_chunk = local_pid % CHUNK
            xcd = pid % NUM_XCDS
            pid = chunk_idx * NUM_XCDS * CHUNK + xcd * CHUNK + pos_in_chunk

    num_m = tl.cdiv(M, BLOCK_M)
    num_n = tl.cdiv(N, BLOCK_N)
    total = num_m * num_n
    grp = GROUP_M * num_n

    tl.assume(stride_am > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    for tid in range(pid, total, NUM_SMS):
        gid = tid // grp
        fm = gid * GROUP_M
        gs = min(num_m - fm, GROUP_M)
        pm = fm + (tid % grp) % gs
        pn = (tid % grp) // gs
        tl.assume(pm >= 0)
        tl.assume(pn >= 0)

        rm = tl.max_contiguous(tl.multiple_of((pm * BLOCK_M + tl.arange(0, BLOCK_M)) % M, BLOCK_M), BLOCK_M)
        rn = tl.max_contiguous(tl.multiple_of((pn * BLOCK_N + tl.arange(0, BLOCK_N)) % N, BLOCK_N), BLOCK_N)
        rk = tl.arange(0, BLOCK_K)

        if A_K_CONTIGUOUS:
            a_ptrs = A_ptr + rm[:, None].to(tl.int64) * stride_am + rk[None, :].to(tl.int64)
        else:
            a_ptrs = A_ptr + rm[:, None].to(tl.int64) * stride_am + rk[None, :].to(tl.int64) * stride_ak_val

        if B_K_CONTIGUOUS:
            b_ptrs = B_ptr + rk[:, None].to(tl.int64) + rn[None, :].to(tl.int64) * stride_bn
        else:
            b_ptrs = B_ptr + rk[:, None].to(tl.int64) * stride_bk_val + rn[None, :].to(tl.int64) * stride_bn

        as_ptrs = A_scales_ptr + rm * stride_as_m

        if SCALE_2D_B:
            # Blockwise B scales are still stored on 128-column granularity even
            # when autotune tries a narrower output tile such as BLOCK_N=64.
            scale_block_n = (pn * BLOCK_N) // 128
            bs_ptr_base = B_scales_ptr + scale_block_n * stride_bs_0
        else:
            bs_ptrs = B_scales_ptr + rn * stride_bs_0

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for ki in range(NUM_K_BLOCKS):
            if EVEN_K:
                a = tl.load(a_ptrs, cache_modifier=".ca")
                b = tl.load(b_ptrs, cache_modifier=".ca")
            else:
                k_remaining = K - ki * BLOCK_K
                mask_k_col = rk[None, :] < k_remaining
                mask_k_row = rk[:, None] < k_remaining
                a = tl.load(a_ptrs, mask=mask_k_col, other=0.0, cache_modifier=".ca")
                b = tl.load(b_ptrs, mask=mask_k_row, other=0.0, cache_modifier=".ca")

            partial = tl.dot(a, b)

            a_s = tl.load(as_ptrs + ki * stride_as_k)

            if SCALE_2D_B:
                b_s = tl.load(bs_ptr_base + ki * stride_bs_1)
                acc += partial * (a_s * b_s)[:, None]
            else:
                b_s = tl.load(bs_ptrs + ki * stride_bs_1)
                acc += partial * a_s[:, None] * b_s[None, :]

            if A_K_CONTIGUOUS:
                a_ptrs += BLOCK_K
            else:
                a_ptrs += BLOCK_K * stride_ak_val

            if B_K_CONTIGUOUS:
                b_ptrs += BLOCK_K
            else:
                b_ptrs += BLOCK_K * stride_bk_val

        offs_m = pm * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pn * BLOCK_N + tl.arange(0, BLOCK_N)
        if TRANS_C_STORE:
            # `trans_c=True` flips the output to (N, M) with stride_cm=1 and
            # stride_cn=N. Without `tl.trans`, the BF16 store degrades to
            # 64×buffer_store_short per tile; transposing here lets the
            # compiler emit (buffer|global)_store_dwordx4 ×4-8.
            c_ptrs_t = (
                C_ptr + offs_n[:, None].to(tl.int64) * stride_cn + offs_m[None, :].to(tl.int64) * stride_cm
            )
            mask_t = (offs_n[:, None] < N) & (offs_m[None, :] < M)
            acc_t = tl.trans(acc.to(C_ptr.type.element_ty))
            tl.store(c_ptrs_t, acc_t, mask_t)
        else:
            c_ptrs = (
                C_ptr + offs_m[:, None].to(tl.int64) * stride_cm + offs_n[None, :].to(tl.int64) * stride_cn
            )
            mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
            tl.store(c_ptrs, acc.to(C_ptr.type.element_ty), mask)


# Layout-specialised autotune wrappers around the shared jit body. Each wrapper
# owns an independent `Autotuner` instance with its own config search space and
# best-config cache; the underlying jit kernel is shared, so binaries are still
# emitted per (autotune key + constexpr) tuple.
_blockwise_fp8_nt_kernel = triton.autotune(
    configs=_get_blockwise_autotune_configs(),
    key=["M", "N", "K", "EVEN_K"],
)(_blockwise_fp8_unified_kernel)

_blockwise_fp8_nn_kernel = triton.autotune(
    configs=_get_blockwise_autotune_configs(extra_matrix_instr_nonkdim=[16]),
    key=["M", "N", "K", "EVEN_K"],
)(_blockwise_fp8_unified_kernel)

_blockwise_fp8_tn_kernel = triton.autotune(
    configs=_get_blockwise_autotune_configs(allow_num_stages_3=False),
    key=["M", "N", "K"],
)(_blockwise_fp8_unified_kernel)


_TRITON_AUTOTUNE = bool(int(os.environ.get("NVTE_FP8_BLOCK_SCALING_TRITON_AUTOTUNE", "0")))


def get_blockwise_gemm_config(layout, tokens):
    """Baked gfx950 tile config; ``tokens`` is M for NT/NN, K for TN.

    Winners captured offline over the model dense GEMM shapes, replacing per-shape
    autotune. Returns None off gfx950 so the caller falls back to autotune.
    """
    if not is_gfx950():
        return None
    if layout == "NT":
        return dict(BLOCK_M=256, BLOCK_N=64, BLOCK_K=128, GROUP_M=4, CHUNK=32, NUM_XCDS=8, num_warps=8, num_stages=3)
    if layout == "NN":
        if tokens <= 1024:
            return dict(BLOCK_M=256, BLOCK_N=64, BLOCK_K=128, GROUP_M=4, CHUNK=64, NUM_XCDS=8, num_warps=8, num_stages=3)
        return dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, GROUP_M=8, CHUNK=64, NUM_XCDS=8, num_warps=4, num_stages=2)
    if tokens <= 2048:
        return dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, GROUP_M=4, CHUNK=64, NUM_XCDS=8, num_warps=4, num_stages=2)
    return dict(BLOCK_M=256, BLOCK_N=128, BLOCK_K=128, GROUP_M=4, CHUNK=32, NUM_XCDS=8, num_warps=8, num_stages=2)


# ═══════════════════════════════════════════════════════════════════════════════
# Unified Public API — Block-wise FP8 GEMM
# Interface consistent with CK blockwise backend.
# ═══════════════════════════════════════════════════════════════════════════════


def gemm_fp8_blockwise_triton_kernel(
    a: torch.Tensor,
    a_scale_inv: torch.Tensor,
    b: torch.Tensor,
    b_scale_inv: torch.Tensor,
    trans_a: bool = False,
    trans_b: bool = True,
    out_dtype: torch.dtype = torch.bfloat16,
    trans_c: bool = False,
) -> torch.Tensor:
    """Unified block-wise FP8 GEMM Triton kernel.

    Interface consistent with the CK blockwise backend. Internally normalises
    the operands to a `[M, K] @ [K, N]` view and dispatches to one of three
    layout-specialised autotune wrappers (NT/NN/TN); only the scale-tensor
    layout, autotune wrapper, and a few constexpr flags differ between
    layouts.

    Supported layouts:
      NT/RCR (forward):  trans_a=False, trans_b=True
        A: [M, K], A_scales: [M, K//128]
        B: [N, K], B_scales: [N//128, K//128]  (2D block scaling)

      NN/RRR (grad_X):   trans_a=False, trans_b=False
        A: [M, K], A_scales: [M, K//128]
        B: [K, N], B_scales: [N//128, K//128]  (2D, transposed internally)

      TN/CRR (grad_W):   trans_a=True, trans_b=False
        A: [K, M], A_scales: [K//128, M]
        B: [K, N], B_scales: [K//128, N]

    Args:
        a: FP8 input matrix.
        a_scale_inv: Block-wise scale for A, shape depends on layout.
        b: FP8 input matrix.
        b_scale_inv: Block-wise scale for B, shape depends on layout.
        trans_a: Whether A is transposed.
        trans_b: Whether B is transposed.
        out_dtype: Output dtype (default bfloat16).
        trans_c: If True, return transposed output.

    Returns:
        C of shape (M, N) if trans_c=False, or (N, M) if trans_c=True.
    """
    if (trans_a, trans_b) not in {(False, True), (False, False), (True, False)}:
        raise ValueError(f"Unsupported layout for blockwise FP8 Triton: trans_a={trans_a}, trans_b={trans_b}")

    # ── Operand views: always [M, K] @ [K, N] inside the kernel.
    A_view = a.T if trans_a else a
    B_view = b.T if trans_b else b
    M, K = A_view.shape
    _, N = B_view.shape

    # AMD knobs: gfx950 always sets the gfx950 knobs; on gfx942 NT/NN benefit
    # from use_async_copy/scalarize_packed_fops while TN/wgrad regresses 5-8%.
    is_tn = trans_a and not trans_b
    if is_gfx950():
        set_triton_knobs_gfx950()
    else:
        _set_amd_knobs(enable=not is_tn)

    # ── Layout-specialised settings.
    # The only true per-layout differences are: how A/B scales are laid out
    # for the kernel, which autotune wrapper owns the search space/cache, and
    # the three constexpr flags (SCALE_2D_B / EVEN_K / TRANS_C_STORE).
    #
    # TN safety notes:
    #   * `TRANS_C_STORE=True` lets the BF16 epilogue coalesce into dwordx4
    #     under the `(N, M)` output buffer; without it the wgrad path emits
    #     64×buffer_store_short per tile.
    #   * `EVEN_K=False` keeps the mask-load path for the dual strided-K
    #     loads, the matched-pair safety net for the Triton 3.7
    #     `Begin <= End` LLVM-backend assertion (the TN autotune wrapper
    #     already drops `num_stages=3`).
    if not trans_a and trans_b:
        # NT
        A_scales_arg = a_scale_inv.T.contiguous()  # [K//128, M]
        B_scales_arg = b_scale_inv  # [N//128, K//128]
        autotune_kernel = _blockwise_fp8_nt_kernel
        SCALE_2D_B, EVEN_K, TRANS_C_STORE = True, (K % 128) == 0, False
    elif not trans_a and not trans_b:
        # NN — kernel expects [N_output_blocks, K_inner_blocks] for B_scales,
        # while quantization stores it as [N//128, K//128] over the original
        # weight; the .T.contiguous() rebuilds the indexing the kernel reads.
        A_scales_arg = a_scale_inv.T.contiguous()
        B_scales_arg = b_scale_inv.T.contiguous()
        autotune_kernel = _blockwise_fp8_nn_kernel
        SCALE_2D_B, EVEN_K, TRANS_C_STORE = True, (K % 128) == 0, False
    else:
        # TN
        A_scales_arg = a_scale_inv
        B_scales_arg = b_scale_inv
        autotune_kernel = _blockwise_fp8_tn_kernel
        SCALE_2D_B, EVEN_K, TRANS_C_STORE = False, False, True

    # Scale strides:
    #   * SCALE_2D_B=True  (NT/NN): A_scales/B_scales are stored as [outer, inner],
    #     so stride(0)/stride(1) directly map to (outer, inner) the kernel needs.
    #   * SCALE_2D_B=False (TN):    B_scales is [K//128, N] but the kernel addresses
    #     it as `pn * stride_bs_0 + ki * stride_bs_1` (N-step then K-step), hence
    #     the (1, 0) swap.
    stride_as_k, stride_as_m = A_scales_arg.stride(0), A_scales_arg.stride(1)
    if SCALE_2D_B:
        stride_bs_0, stride_bs_1 = B_scales_arg.stride(0), B_scales_arg.stride(1)
    else:
        stride_bs_0, stride_bs_1 = B_scales_arg.stride(1), B_scales_arg.stride(0)

    # ── Output buffer (handle trans_c by swapping strides on a (N, M) buffer).
    if trans_c:
        out = torch.empty((N, M), device=a.device, dtype=out_dtype)
        stride_cm, stride_cn = out.stride(1), out.stride(0)
    else:
        out = torch.empty((M, N), device=a.device, dtype=out_dtype)
        stride_cm, stride_cn = out.stride(0), out.stride(1)

    num_k = (K + 127) // 128
    NUM_SMS = ((M + 127) // 128) * ((N + 127) // 128)

    args = (
        A_view,
        B_view,
        out,
        A_scales_arg,
        B_scales_arg,
        M,
        N,
        K,
        A_view.stride(0),
        A_view.stride(1),
        B_view.stride(0),
        B_view.stride(1),
        stride_cm,
        stride_cn,
        stride_as_k,
        stride_as_m,
        stride_bs_0,
        stride_bs_1,
        NUM_SMS,
        num_k,
    )
    flags = dict(
        A_K_CONTIGUOUS=not trans_a,
        B_K_CONTIGUOUS=trans_b,
        SCALE_2D_B=SCALE_2D_B,
        EVEN_K=EVEN_K,
        TRANS_C_STORE=TRANS_C_STORE,
    )
    # Default: baked gfx950 config. Opt in to autotune via NVTE_FP8_BLOCK_SCALING_TRITON_AUTOTUNE.
    layout = "NT" if (not trans_a and trans_b) else ("NN" if not trans_a else "TN")
    cfg = None if _TRITON_AUTOTUNE else get_blockwise_gemm_config(layout, M if layout != "TN" else K)
    if cfg is None:
        autotune_kernel[(NUM_SMS,)](*args, **flags)
    else:
        _blockwise_fp8_unified_kernel[(NUM_SMS,)](
            *args,
            BLOCK_M=cfg["BLOCK_M"],
            BLOCK_N=cfg["BLOCK_N"],
            BLOCK_K=cfg["BLOCK_K"],
            GROUP_M=cfg["GROUP_M"],
            NUM_XCDS=cfg["NUM_XCDS"],
            CHUNK=cfg["CHUNK"],
            num_warps=cfg["num_warps"],
            num_stages=cfg["num_stages"],
            **flags,
        )
    return out


def gemm_blockwise(A, B, transa, transb, out_dtype, bias=None, out=None):
    """Blockwise FP8 GEMM for Float8Blockwise operands via the Triton kernel.

    ``general_gemm`` computes ``out = op_b(B) @ op_a(A)``; the Triton kernel
    computes ``op(a) @ op(b)``, so a=B, b=A with the transpose flags swapped.
    """
    from .common import te_dtype_to_torch_dtype

    dt = te_dtype_to_torch_dtype(A._fp8_dtype)
    a_data, a_scale = B.get_gemm_operand(is_left=True, trans=transb)
    b_data, b_scale = A.get_gemm_operand(is_left=False, trans=transa)
    res = gemm_fp8_blockwise_triton_kernel(
        a_data.view(dt), a_scale, b_data.view(dt), b_scale,
        trans_a=transb, trans_b=transa, out_dtype=out_dtype,
    )
    if bias is not None:
        res = res + bias.to(res.dtype)
    if out is not None:
        out.copy_(res)
        return out
    return res
