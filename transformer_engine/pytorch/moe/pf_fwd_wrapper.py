# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL permute-free MoE route-list forward gather-GEMM op wrapper.

Mirrors the route-list forward gather-GEMM contract so the same routing
metadata (``sorted_slot_ids`` / ``expert_ids`` / ``block_start``) can be reused verbatim.
Writes the block-padded ``[em_max, WIDTH_N]`` slot output in place.

Forward gather-GEMM is FlyDSL-only (v3 MegaMOE-ported plain GEMM). Gated activation and
route-prob apply live in standalone Triton helpers (:mod:`pf_helper_kernels`), not in the
GEMM kernel.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

__all__ = [
    "flydsl_moe_fwd",
    "flydsl_moe_fwd_supported",
    "flydsl_moe_fwd_pick_block_m",
]

_FILL_V = 8
_WARP = 64
_LDS_PAD = 8
_LDS_LIMIT = 163840  # gfx950 per-workgroup LDS (160 KB)

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
# nt_vmcnt=3). TE's FlyDSL fwd align already bumps FC1 / dgrad / FC2 to block_size_m=256;
# v3 pins the same Mega M-tile rather than trusting the wrapper arg.
_V3_BLOCK_M = 256
_V3_BLOCK_N = 256
_V3_GROUP_M = 4
_V3_DGRAD_FC1_GROUP_M = 8  # bench_mega_moe grouped_gemm_combine L1-dgrad (NN) sweep
_V3_NUM_XCD = 1


def _run_v3_fwd(
    A, B, C, sorted_slot_ids, expert_ids, *, num_recv_tokens, block_m,
    transpose_b, index_a_by_route_pos,
):
    """Dispatch a plain fwd/dgrad GEMM to the v3 (MegaMOE-ported) kernels.

    Covers the three paths: FC1 gather (``index_a_by_route_pos=False``), FC2 route-read
    (``index_a_by_route_pos=True``) and dgrad (``transpose_b``). v3 uses MegaMOE's fixed tile
    geometry (32x32x16, ``BLOCK_N=256``, ``GROUP_M=4``, ``num_xcd=1``), so the wrapper's
    ``block_n``/``block_k``/warp args are ignored here.
    """
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


def _fwd_buffering():
    """(num_buffers, lds_pad) for the production DMA+swizzle fill path.

    Both default on (opt out with ``MOE_FWD_DMA=0`` / ``MOE_FWD_SWZ=0``). The DMA path runs
    a distance-2, 3-buffer ring; the register fallback keeps 2-buffer ping/pong.
    """
    use_dma = _env_flag("MOE_FWD_DMA", True)
    swz = _env_flag("MOE_FWD_SWZ", True)
    pad = 0 if (use_dma or swz) else _LDS_PAD
    return (3 if use_dma else 2), pad


def _lds_bytes(block_m, block_n, block_k, transpose_b=False):
    nbuf, pad = _fwd_buffering()
    a_tile = block_m * (block_k + pad)
    # dgrad stages B as [k, n] (row stride = block_n+pad); fwd as [n, k] (block_k+pad).
    b_tile = block_k * (block_n + pad) if transpose_b else block_n * (block_k + pad)
    return (a_tile + b_tile) * nbuf * 2


def flydsl_moe_fwd_supported(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    block_m: int,
) -> bool:
    """Whether the in-tree FlyDSL fwd/dgrad kernels can handle these operands."""
    if A.dtype != torch.bfloat16 or B.dtype != torch.bfloat16:
        return False
    if A.stride(1) != 1:
        return False
    if B.stride(2) != 1 and B.stride(1) != 1:
        return False
    # em_max % block_m is only known at launch (from C), so it is not prechecked here.
    return True


def flydsl_moe_fwd(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    sorted_slot_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    *,
    num_recv_tokens: int,
    block_m: int,
    index_a_by_route_pos: bool = False,
) -> None:
    """Route-list gather-GEMM forward, writing the compact ``C[em_max, WIDTH_N]`` in place.

    ``A`` is ``[*, K]`` (received-token acts, gathered by ``sorted_slot_ids`` when
    ``index_a_by_route_pos=False``, else read at the compact route row). ``B`` is
    ``[num_experts, N_OUT, K]`` (contiguous inner ``K``). Gated activation and route-prob
    apply are **not** fused here; use the standalone helpers in :mod:`pf_helper_kernels`.

    The v3 (MegaMOE-ported) kernels use a fixed tile geometry, so only ``block_m`` is a tunable
    knob (chosen once by the align via :func:`flydsl_moe_fwd_pick_block_m`).
    """
    assert A.dtype == B.dtype == C.dtype == torch.bfloat16
    assert A.stride(1) == 1, "A must be contiguous along the contraction (K)"
    # B is [E, N_OUT, K]: contiguous along K (fwd) or along N (dgrad transposed-weight view).
    transpose_b = B.stride(2) != 1
    if transpose_b:
        assert B.stride(1) == 1, "transposed B must be contiguous along N (dgrad view)"

    # Plain GEMM: in-tree v3 kernels (pf_fwd / pf_dgrad).
    _run_v3_fwd(
        A, B, C, sorted_slot_ids, expert_ids,
        num_recv_tokens=num_recv_tokens, block_m=block_m,
        transpose_b=transpose_b, index_a_by_route_pos=index_a_by_route_pos,
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
# An exhaustive sweep of the valid tile space found nothing better, so the list below is not
# missing a winner. Two directions are dead ends and are deliberately
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

def _valid_config(block_m, bn, bk, wm, wn, K, transpose_b=False):
    if K % bk != 0 or bn % _mfma_dim(transpose_b) != 0:
        return False
    if not _warp_valid(block_m, bn, bk, wm, wn, transpose_b):
        return False
    return _lds_bytes(block_m, bn, bk, transpose_b) <= _LDS_LIMIT


def flydsl_moe_fwd_pick_block_m(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    candidates=(256, 128),
) -> Optional[int]:
    """Largest ``block_m`` in ``candidates`` the FlyDSL fwd can actually run for these operands,
    or ``None`` if the operands are unsupported at every candidate.

    "Can run" == bf16 operands contiguous along the contraction AND at least one candidate tile
    in :data:`_FWD_TUNE_CONFIGS` fits the per-workgroup LDS budget. Callers should include their
    default among ``candidates`` (the picker only walks high->low over what is passed), so a
    small-token workload is never padded up beyond what the caller offered.
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
            _valid_config(block_m, bn, bk, wm, wn, K)
            for (bn, bk, wm, wn) in _FWD_TUNE_CONFIGS
        ):
            return block_m
    return None
