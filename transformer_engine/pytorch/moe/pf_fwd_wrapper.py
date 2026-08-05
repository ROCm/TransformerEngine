# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL permute-free MoE route-list forward gather-GEMM op wrapper.

Mirrors the route-list forward gather-GEMM contract so the same routing
metadata (``sorted_slot_ids`` / ``expert_ids`` / ``block_start`` / ``route_start``) can be
reused verbatim. Writes the compact ``[em_max, WIDTH_N]`` route output in place.

Forward gather-GEMM is FlyDSL-only. This module retains Triton kernels for routing
metadata construction, gated-activation recompute/bwd, and the token-order gather-combine
pass that follows the compact route-list GEMM outputs.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

__all__ = [
    "flydsl_moe_fwd",
    "flydsl_moe_fwd_autotuned",
    "flydsl_moe_fwd_supported",
    "flydsl_moe_fwd_pick_block_m",
]

# Activation ids for fused-epilogue kernels (v2, not yet in-tree).
ACT_SILU = 0
ACT_GELU = 1
_ACT_IDS = {"silu": ACT_SILU, "gelu": ACT_GELU}
_WMMA = 16
_FILL_V = 8
_WARP = 64
_LDS_PAD = 8
_LDS_LIMIT = 163840  # gfx950 per-workgroup LDS (160 KB)

_WARP_CHOICES = [1, 2, 4, 8, 16]


def _env_flag(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() not in ("0", "false", "no", "off", "")


def _v3_enabled() -> bool:
    """Whether to route plain GEMMs through the in-tree v3 (MegaMOE-ported) kernels."""
    return _env_flag("AITER_MOE_FLYDSL_V3", True)


# MegaMOE's hand-tuned bf16 grouped-GEMM tile geometry (offline-swept in primus-turbo's
# bench_mega_moe: BLOCK_M/BLOCK_N=256, GROUP_M=4 fwd / GROUP_M=8 FC1 NN dgrad, num_xcd=1,
# nt_vmcnt=3). TE's FlyDSL fwd align already bumps non-gated FC1 / dgrad / plain FC2 to
# block_size_m=256; v3 pins the same Mega M-tile rather than trusting the wrapper arg.
_V3_BLOCK_M = 256
_V3_BLOCK_N = 256
_V3_GROUP_M = 4
_V3_DGRAD_FC1_GROUP_M = 8  # bench_mega_moe grouped_gemm_combine L1-dgrad (NN) sweep
_V3_NUM_XCD = 1


def _run_v3_fwd(
    A, B, C, sorted_slot_ids, expert_ids, *, num_recv_tokens, block_m,
    transpose_b, index_a_by_route_pos, gated, gated_a, mul_prob, save_preact,
):
    """Dispatch a plain fwd/dgrad GEMM to the v3 (MegaMOE-ported) kernels.

    Covers the three non-fused paths: FC1 gather (``index_a_by_route_pos=False``), FC2
    route-read (``index_a_by_route_pos=True``) and dgrad (``transpose_b``). Fused epilogues
    (gated activation / route-prob / pre-activation save) have no v3 equivalent and raise.
    v3 uses MegaMOE's fixed tile geometry (32x32x16, ``BLOCK_N=256``, ``GROUP_M=4``,
    ``num_xcd=1``), so the wrapper's ``block_n``/``block_k``/warp args are ignored here.
    """
    if gated or gated_a or mul_prob or save_preact:
        raise RuntimeError(
            "Fused permute-free forward (gated activation / dispatched_probs / preact_out) "
            "requires the v2 FlyDSL kernel (moe_fwd_flydsl_v2), which is not in-tree yet. "
            "Use standalone gated-act + plain v3 GEMM (GroupedLinear FC2 path), or port v2."
        )
    block_m = int(block_m)
    em_max = int(sorted_slot_ids.shape[0])
    if em_max % block_m != 0:
        raise ValueError(
            f"v3 expects em_max ({em_max}) divisible by block_m ({block_m}); "
            "check routing align block_size_m."
        )
    num_tile_blocks = em_max // block_m
    expert_ids_i32 = expert_ids.to(torch.int32)
    if expert_ids_i32.numel() != num_tile_blocks:
        raise ValueError(
            f"v3 expects expert_ids length {num_tile_blocks} (em_max={em_max}, BLOCK_M={block_m}), "
            f"got {expert_ids_i32.numel()}; routing align block_size_m must match block_m={block_m}"
        )

    if transpose_b:
        # dgrad: v3 is a native NN GEMM contracting the incoming grad against the weight over
        # the output-feature axis. The wrapper's B is the transposed-weight *view* [E, out, in]
        # (stride relabel); transpose(1,2) recovers the [E, N=out, K=in] forward weight v3
        # dgrad reads NN. FC1 dgrad (index_a_by_route_pos=True) is compact route-read (grad rows
        # == dx rows); FC2 dgrad (index_a_by_route_pos=False) gathers the token-space grad
        # [num_recv, N] into the compact route output [em_max, K].
        from ..flydsl_kernels.permute_free_grouped_gemm.pf_dgrad import grouped_gemm_dgrad_bf16

        # TE passes a stride-relabelled transpose *view* (``stride(2) != 1``); undo the view to
        # recover the forward ``[E, N, K]`` storage without copying (``transpose`` twice == id).
        weight = B.transpose(1, 2) if B.stride(2) != 1 else B
        # Real M-tile count for the dgrad grid: derive from the padded pool shape (host-known,
        # graph-capture safe). Padding tail blocks (expert_ids=-1) early-exit in-kernel, same as
        # the forward v3 path. Do not count active blocks on-device ((expert_ids>=0).sum().item())
        # -- that syncs the GPU and breaks HIP/CUDA graph capture.
        dgrad_group_m = _V3_DGRAD_FC1_GROUP_M if index_a_by_route_pos else _V3_GROUP_M
        grouped_gemm_dgrad_bf16(
            A, weight, C, expert_ids_i32, num_tile_blocks, sorted_slot_ids,
            gather=not index_a_by_route_pos,
            BLOCK_M=block_m, BLOCK_N=_V3_BLOCK_N, GROUP_M=dgrad_group_m, num_xcd=_V3_NUM_XCD,
        )
        return

    from ..flydsl_kernels.permute_free_grouped_gemm.pf_fwd import grouped_gemm_gather_bf16

    grouped_gemm_gather_bf16(
        A, B, C, expert_ids_i32, num_tile_blocks, sorted_slot_ids,
        gather=not index_a_by_route_pos,
        BLOCK_M=block_m, BLOCK_N=_V3_BLOCK_N, GROUP_M=_V3_GROUP_M, num_xcd=_V3_NUM_XCD,
    )


def _mfma_dim(transpose_b: bool) -> int:
    """MFMA output-tile edge: forward GEMM uses the 32x32x16 atom, dgrad the 16x16x32 atom.

    Kept in sync with the kernel's ``MOE_FWD_MFMA32`` escape hatch so the autotuner sweeps the
    tile-divisibility that the compiled atom actually requires.
    """
    if transpose_b:
        return 16
    return 32 if _env_flag("MOE_FWD_MFMA32", False) else 16


def _warp_valid(block_m, block_n, block_k, wm, wn, transpose_b=False):
    n_threads = wm * wn * _WARP
    wmma = _mfma_dim(transpose_b)
    if block_m % (wm * wmma) or block_n % (wn * wmma):
        return False
    if (block_m * block_k) % (n_threads * _FILL_V):
        return False
    if (block_n * block_k) % (n_threads * _FILL_V):
        return False
    return True


def _pick_warps(block_m: int, block_n: int, block_k: int, transpose_b=False):
    """Pick (warps_m, warps_n) balancing per-warp MFMA tile (M_STEPS x N_STEPS) vs occupancy.

    Prefers keeping the per-warp atom counts moderate (fewer accumulators -> more waves) while
    landing a 256-512 thread workgroup, which measured fastest across the Qwen MoE shapes.
    """
    wmma = _mfma_dim(transpose_b)
    best = None
    for wm in _WARP_CHOICES:
        for wn in _WARP_CHOICES:
            if not _warp_valid(block_m, block_n, block_k, wm, wn, transpose_b):
                continue
            n_threads = wm * wn * _WARP
            if n_threads > 512:
                continue
            m_steps = block_m // (wm * wmma)
            n_steps = block_n // (wn * wmma)
            # Favor a small, balanced per-warp atom footprint (fewer accumulators -> more
            # waves), then a 256-512 thread workgroup. Ties broken toward |m_steps-n_steps|
            # small (balanced reuse of A and B fragments).
            score = (
                m_steps * n_steps,
                abs(m_steps - n_steps),
                0 if 256 <= n_threads <= 512 else 1,
                n_threads,
            )
            if best is None or score < best[0]:
                best = (score, (wm, wn))
    return best[1] if best is not None else None


def _fwd_buffering():
    """(num_buffers, lds_pad) for the production DMA+swizzle fill path.

    Both default on (opt out with ``MOE_FWD_DMA=0`` / ``MOE_FWD_SWZ=0``). The DMA path runs
    a distance-2, 3-buffer ring; the register fallback keeps 2-buffer ping/pong.
    """
    use_dma = _env_flag("MOE_FWD_DMA", True)
    swz = _env_flag("MOE_FWD_SWZ", True)
    pad = 0 if (use_dma or swz) else _LDS_PAD
    return (3 if use_dma else 2), pad


def _lds_bytes(block_m, block_n, block_k, gated, transpose_b=False, gated_a=False):
    n_bt = 2 if gated else 1
    nbuf, pad = _fwd_buffering()
    a_tile = block_m * (block_k + pad)
    # Hybrid gated_a DMA path stages the raw ``up`` half in a parallel A tile (2x A LDS).
    hybrid_a = gated_a and _env_flag("MOE_FWD_DMA", True) and _env_flag("MOE_FWD_GATEDA_DMA", False)
    a_tiles = 2 if hybrid_a else 1
    # dgrad stages B as [k, n] (row stride = block_n+pad); fwd as [n, k] (block_k+pad).
    b_tile = block_k * (block_n + pad) if transpose_b else block_n * (block_k + pad)
    return (a_tiles * a_tile + n_bt * b_tile) * nbuf * 2


def _default_block_n(block_m, block_k, gated, transpose_b=False):
    """Widest N tile (128 then 64) that fits LDS -- wider N raises arithmetic intensity."""
    for bn in (128, 64):
        if _lds_bytes(block_m, bn, block_k, gated, transpose_b) <= _LDS_LIMIT and bn % _mfma_dim(transpose_b) == 0:
            return bn
    return 64


def flydsl_moe_fwd_supported(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    block_m: int,
    block_n: int = 64,
    block_k: int = 64,
) -> bool:
    """Whether the in-tree FlyDSL fwd/dgrad kernels can handle these operands."""
    if A.dtype != torch.bfloat16 or B.dtype != torch.bfloat16:
        return False
    if A.stride(1) != 1:
        return False
    if B.stride(2) != 1 and B.stride(1) != 1:
        return False
    em_max = None  # only known at launch for C; skip em_max % block_m precheck here
    _ = (block_n, block_k, em_max)
    return True


def flydsl_moe_fwd(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    sorted_slot_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    block_start: torch.Tensor,
    route_start: torch.Tensor,
    *,
    num_recv_tokens: int,
    block_m: int,
    block_n: Optional[int] = None,
    block_k: int = 64,
    warps_m: Optional[int] = None,
    warps_n: Optional[int] = None,
    index_a_by_route_pos: bool = False,
    activation: Optional[str] = None,
    dispatched_probs: Optional[torch.Tensor] = None,
    preact_out: Optional[torch.Tensor] = None,
    gated_a: bool = False,
) -> None:
    """Route-list gather-GEMM forward, writing the compact ``C[em_max, WIDTH_N]`` in place.

    ``A`` is ``[*, K]`` (received-token acts, gathered by ``sorted_slot_ids`` when
    ``index_a_by_route_pos=False``, else read at the compact route row). ``B`` is
    ``[num_experts, N_OUT, K]`` (contiguous inner ``K``). With ``activation`` set the fused
    **gated** epilogue (FC1, ``gated_a=False``) applies ``act(gate) * up`` over ``N_OUT = 2F``
    into the ``F``-wide ``C``; ``dispatched_probs`` multiplies the per-route prob after the
    activation, and ``preact_out`` (``[em_max, 2F]``) saves the raw ``[gate | up]``
    pre-activation. With ``gated_a=True`` (FC2) ``A`` is the raw ``[gate | up]`` pre-activation
    (width ``2F``), the prologue applies ``act(gate) * up [* prob]`` into an ``F``-wide tile,
    and the GEMM contracts over ``K = F`` against ``B[e, H, F]``.
    """
    assert A.dtype == B.dtype == C.dtype == torch.bfloat16
    assert A.stride(1) == 1, "A must be contiguous along the contraction (K)"
    # B is [E, N_OUT, K]: contiguous along K (fwd) or along N (dgrad transposed-weight view).
    transpose_b = B.stride(2) != 1
    if transpose_b:
        assert B.stride(1) == 1, "transposed B must be contiguous along N (dgrad view)"
        assert activation is None, "dgrad (transposed B) does not support fused activation"

    gated = activation is not None and not gated_a
    if gated_a:
        if activation is None:
            raise ValueError("gated_a requires activation ('silu' or 'gelu')")
        if not index_a_by_route_pos:
            raise ValueError("gated_a requires index_a_by_route_pos=True")

    # Plain GEMM: in-tree v3 kernels (pf_fwd / pf_dgrad). Fused epilogues need v2 (not ported).
    _run_v3_fwd(
        A, B, C, sorted_slot_ids, expert_ids,
        num_recv_tokens=num_recv_tokens, block_m=block_m,
        transpose_b=transpose_b, index_a_by_route_pos=index_a_by_route_pos,
        gated=gated, gated_a=gated_a,
        mul_prob=dispatched_probs is not None, save_preact=preact_out is not None,
    )


# Tile/warp configs the autotuner sweeps: (block_n, block_k, warps_m, warps_n). Fill path
# (DMA+swizzle, 3-buffer) is fixed at the env defaults above -- only tile geometry is tuned.
#
# Two shapes matter for the wide-M (block_m=256) MoE cases, where every block_k=128 and
# 256x64 entry below is filtered out by the LDS limit, leaving only 128x64 candidates:
#   * a square-ish w4x4 warp grid, which spreads the cooperative A-fill (2 fills) / B-fill
#     (1 fill) DMA and the LDS fragment reads more evenly than the tall w8x2 layout;
#   * block_n=256 with block_k=32, which fits LDS at 3 buffers and raises arithmetic
#     intensity to block_m*block_n/(2*(block_m+block_n)) = 64 MAC/byte vs 42.7 at block_n=128
#     (the ratio is independent of block_k, so widening N is what pays).
# Measured on FC1 no-act (qwen235b, block_m=256), idle machine, alias scopes on:
# 256x32 w4x4 ~1628us, 128x64 w4x4 ~1694us, 128x64 w8x2 ~1726us.
#
# An exhaustive sweep of the valid space (all block_n in 64..512 x block_k in 32..256 x warp
# grids >=4 waves; benchmarks/microbenchmarks/sweep_fc1_configs.py) found nothing better, so
# the list below is not missing a winner. Two directions are dead ends and are deliberately
# absent: block_n>=384 costs 4-22x (per-wave accumulator spill plus 120-144KB LDS pinning
# occupancy to 1 workgroup), and trading block_m down to reach block_k=128 -- 4x fewer
# barriers, which the ATT trace makes look attractive -- costs 59% (2581us at block_m=128
# bk=128) because arithmetic intensity falls faster than barrier count. Note block_k>=128 does
# not fit LDS at all once NUM_BUF>=3, which the kernel enforces.
#
# Do not add 256x384x32 w1x4 or 512x32 w1x4 / w2x2: they abort the backend outright
# ("Bad machine code: Virtual register defs don't dominate all uses"), which would take the
# autotuner down with them rather than being skipped. Pre-existing, unrelated to alias scopes.
_FWD_TUNE_CONFIGS = [
    (64, 64, 2, 2),
    (128, 64, 2, 2),
    (128, 64, 4, 2),
    (128, 64, 2, 4),
    (128, 64, 4, 4),
    (128, 64, 8, 2),
    (256, 32, 4, 4),
    (256, 32, 2, 4),
    (256, 32, 4, 2),
    (256, 64, 4, 2),
    (128, 128, 2, 2),
    (128, 128, 4, 2),
    (64, 128, 2, 2),
    (256, 64, 2, 4),
]

# Winner cache: {shape/mode key -> (block_n, block_k, warps_m, warps_n)}.
_FWD_CACHE: dict = {}


def _valid_config(block_m, bn, bk, wm, wn, K, gated, gated_a=False, transpose_b=False):
    if K % bk != 0 or bn % _mfma_dim(transpose_b) != 0:
        return False
    if not _warp_valid(block_m, bn, bk, wm, wn, transpose_b):
        return False
    return _lds_bytes(block_m, bn, bk, gated, gated_a=gated_a) <= _LDS_LIMIT


def flydsl_moe_fwd_pick_block_m(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    gated: bool = False,
    gated_a: bool = False,
    candidates=(256, 128),
) -> Optional[int]:
    """Largest ``block_m`` in ``candidates`` the FlyDSL fwd can actually run for these operands
    and epilogue mode, or ``None`` if the operands are unsupported at every candidate.

    "Can run" == bf16 operands contiguous along the contraction AND at least one autotuner tile
    in :data:`_FWD_TUNE_CONFIGS` fits the per-workgroup LDS budget for the epilogue. The gated FC1
    epilogue stages a ``2F`` ``[gate|up]`` B-tile, so it only fits ``block_m <= 128``; the non-gated
    FC1 fwd and the ``gated_a`` FC2 prologue fit ``block_m = 256``, which lifts the shared
    fwd/dgrad/FC2 align onto the faster ``256x32`` MegaMOE-like tile. Callers should include their
    token-count default among ``candidates`` (the picker only walks high->low over what is passed),
    so a small-token workload is never padded up beyond what the caller offered.
    """
    if A.dtype != torch.bfloat16 or B.dtype != torch.bfloat16:
        return None
    if A.stride(1) != 1:  # A must be contiguous along the contraction (K)
        return None
    if B.stride(2) != 1 and B.stride(1) != 1:  # B contiguous along K (fwd) or N (dgrad view)
        return None
    K = int(B.shape[2])
    for block_m in sorted({int(c) for c in candidates}, reverse=True):
        if any(
            _valid_config(block_m, bn, bk, wm, wn, K, gated, gated_a)
            for (bn, bk, wm, wn) in _FWD_TUNE_CONFIGS
        ):
            return block_m
    return None


def flydsl_moe_fwd_autotuned(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    sorted_slot_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    block_start: torch.Tensor,
    route_start: torch.Tensor,
    *,
    num_recv_tokens: int,
    block_m: int,
    block_k: int = 64,
    index_a_by_route_pos: bool = False,
    activation: Optional[str] = None,
    dispatched_probs: Optional[torch.Tensor] = None,
    preact_out: Optional[torch.Tensor] = None,
    gated_a: bool = False,
    warmup: int = 3,
    iters: int = 10,
) -> None:
    """Shape-autotuned :func:`flydsl_moe_fwd`.

    On the first call for a given (block_m, GEMM shape, epilogue mode) the valid subset of
    ``_FWD_TUNE_CONFIGS`` is benchmarked and the fastest ``(block_n, block_k, warps_m, warps_n)``
    is cached. The production DMA+swizzle fill path is always used; only tile geometry is swept.
    """
    gated = activation is not None and not gated_a
    N_OUT, K = int(B.shape[1]), int(B.shape[2])
    width_n = int(C.shape[1])
    key = (
        int(block_m), N_OUT, K, width_n, bool(gated), bool(gated_a),
        activation, dispatched_probs is not None, preact_out is not None,
        bool(index_a_by_route_pos),
    )

    def _launch(bn, bk, wm, wn):
        flydsl_moe_fwd(
            A, B, C, sorted_slot_ids, expert_ids, block_start, route_start,
            num_recv_tokens=num_recv_tokens, block_m=block_m, block_n=bn, block_k=bk,
            warps_m=wm, warps_n=wn, index_a_by_route_pos=index_a_by_route_pos,
            activation=activation, dispatched_probs=dispatched_probs, preact_out=preact_out,
            gated_a=gated_a,
        )

    best = _FWD_CACHE.get(key)
    if best is None:
        candidates = [
            (bn, bk, wm, wn)
            for (bn, bk, wm, wn) in _FWD_TUNE_CONFIGS
            if _valid_config(block_m, bn, bk, wm, wn, K, gated, gated_a)
        ]
        if not candidates:
            _launch(None, block_k, None, None)  # heuristic fallback
            return
        best_t = None
        for cfg in candidates:
            try:
                for _ in range(warmup):
                    _launch(*cfg)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                start.record()
                for _ in range(iters):
                    _launch(*cfg)
                end.record()
                torch.cuda.synchronize()
                t = start.elapsed_time(end) / iters
            except Exception:  # noqa: BLE001 -- skip configs that fail to compile/run
                continue
            if best_t is None or t < best_t:
                best_t, best = t, cfg
        if best is None:
            _launch(None, block_k, None, None)
            return
        _FWD_CACHE[key] = best

    _launch(*best)
