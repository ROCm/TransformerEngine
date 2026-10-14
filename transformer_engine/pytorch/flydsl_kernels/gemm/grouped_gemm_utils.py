# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

"""Shared utilities for the FlyDSL persistent work-stealing grouped GEMM.

The grouped (MoE) GEMM path is dtype-independent: one packed kernel launch spans
all local experts, a fixed num-CU persistent grid claims tiles from a precomputed
descriptor queue, and the per-expert offset is folded into operand/scale/output
addressing. MXFP8, BF16/FP16, and tensor-wise FP8 all share this machinery.

This module holds the reusable pieces that can live outside the ``@flyc.kernel``
body. The persistent loop itself -- the ``if tx==0`` atomic claim and the ``while``
drain loops -- is NOT here: the FlyDSL compiler AST-rewrites only the decorated
kernel's own source, so that runtime-SSA control flow MUST stay inline in each
``kernel_gemm``. What's shared instead:

* **Device-side IR emitters** (``global_i32_ptr``, ``make_claim_broadcast``,
  ``read_desc_field``, ``num_tiles_bound``) -- control-flow-free functions that
  append IR at the current insertion point, so they emit identically whether
  defined inline or imported.
* **Host-side builders** (descriptor-queue construction, schedule selection,
  work-counter allocation, uniform token-axis padding) -- plain Python/torch,
  identical across dtypes; they differ only in whether scales are also padded.
"""

import os

import torch

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly as _fly
from flydsl._mlir.dialects import llvm
from flydsl.expr.typing import T


# ---------------------------------------------------------------------------
# Device-side IR emitters (called while tracing a kernel; pure, no control flow)
# ---------------------------------------------------------------------------


def global_i32_ptr(tensor, index):
    """Build an address-space-1 ``!llvm.ptr`` to ``tensor[index]`` (int32).

    ``llvm.atomicrmw`` requires an ``!llvm.ptr`` (not an ``!fly.ptr``); this
    recovers the raw base pointer and offsets it by ``index`` int32 elements.
    """
    ptr_ty = ir.Type.parse("!llvm.ptr<1>")
    base = _fly.extract_aligned_pointer_as_index(ptr_ty, tensor)
    base = llvm.PtrToIntOp(T.i64, base).result
    byte_off = fx.as_ir_value(fx.Int64(fx.Index(index) * fx.Index(4)))
    addr = llvm.AddOp(base, byte_off, llvm.IntegerOverflowFlags(0)).result
    return llvm.IntToPtrOp(ptr_ty, addr).result


def make_claim_broadcast(claim_slot_array):
    """Recast a one-int32 LDS scratch array into a claim iterator + 1-elem view.

    Thread 0 writes the atomically-claimed slot via ``claim_iter``; after a
    workgroup barrier every thread reads the same value through ``claim_view``.
    """
    claim_iter = fx.recast_iter(fx.Int32, claim_slot_array.ptr)
    claim_view = fx.make_view(claim_iter, fx.make_layout(1, 1))
    return claim_iter, claim_view


def read_desc_field(wq_div, desc_ld_atom, slot, field):
    """Read one int32 of the packed ``(pid_m, pid_n, expert)`` descriptor.

    ``wq_div`` is the ``logical_divide`` view of the work-queue buffer tensor and
    ``desc_ld_atom`` an int32 ``BufferCopy32b`` copy atom.
    """
    reg = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Int32)
    fx.copy(desc_ld_atom, fx.slice(wq_div, (None, slot * fx.Int32(3) + field)), reg)
    return fx.memref_load_vec(reg)[0]


def num_tiles_bound(c_m, c_n, block_m, block_n, num_groups, wgrad):
    """Drain bound for the persistent loop.

    For dense/grouped fwd/dgrad, ``c_m`` already spans all experts, so the tile
    count is complete. For wgrad, ``c_m = N_out`` is the PER-EXPERT output-row
    count (the packed ``[G*N_out, K_in]`` output is addressed via
    ``store_m_expert_off``), so multiply by ``num_groups``.
    """
    num_tiles = (c_m // block_m) * (c_n // block_n)
    if wgrad:
        num_tiles = num_tiles * num_groups
    return num_tiles


# ---------------------------------------------------------------------------
# Host-side builders (plain torch; dtype-independent)
# ---------------------------------------------------------------------------

# Above this many persistent-loop rounds, dynamic atomic work-stealing pays for
# itself; below it a single grid-stride sweep already balances the tiles, so the
# atomic-free static schedule is preferred. Matches the HK reference's SCHED_ROUNDS.
_GROUPED_SCHED_ROUNDS = 4


def select_grouped_sched(num_tiles: int, device) -> str:
    """Pick the grouped-GEMM schedule: 'worksteal' (atomic) or 'static' (grid-stride).

    Honors the ``NVTE_FLYDSL_GROUPED_SCHED`` override ('auto', 'static',
    'worksteal') for testing; otherwise applies the round-count heuristic against
    the CU grid.
    """
    override = os.getenv("NVTE_FLYDSL_GROUPED_SCHED", "auto").lower()
    if override in ("static", "worksteal"):
        return override
    if override not in ("auto", ""):
        raise ValueError(
            f"NVTE_FLYDSL_GROUPED_SCHED must be auto|static|worksteal, got {override!r}"
        )
    num_cus = torch.cuda.get_device_properties(device).multi_processor_count
    return "static" if num_tiles <= _GROUPED_SCHED_ROUNDS * num_cus else "worksteal"


def build_fwd_dgrad_descriptors(group_offs_cpu, num_groups, num_pid_n, block_m):
    """Build the fwd/dgrad ``(pid_m, pid_n, expert)`` descriptor list.

    Each expert ``g`` owns the packed output M-block range
    ``[group_offs[g], group_offs[g+1])`` (BLOCK_M-aligned so tiles never straddle
    a group boundary), tiled over ``num_pid_n`` columns.
    """
    descriptors = []
    for g in range(num_groups):
        m_start_blk = int(group_offs_cpu[g]) // block_m
        m_end_blk = int(group_offs_cpu[g + 1]) // block_m
        for pid_m in range(m_start_blk, m_end_blk):
            for pid_n in range(num_pid_n):
                descriptors.append((pid_m, pid_n, g))
    return descriptors


def build_wgrad_descriptors(num_groups, num_pid_m, num_pid_n):
    """Build the wgrad descriptor list: one tile per expert's local [N_out, K_in]."""
    return [
        (pid_m, pid_n, g)
        for g in range(num_groups)
        for pid_m in range(num_pid_m)
        for pid_n in range(num_pid_n)
    ]


def make_work_queue(descriptors, device):
    """Flatten descriptors into the kernel's queue buffer + zeroed atomic counter.

    Returns ``(work_queue, work_counter, num_tiles)``. ``work_queue`` is a flat
    int32 ``[num_tiles*3]`` tensor; ``work_counter`` a zeroed one-int32 dispenser.
    """
    num_tiles = len(descriptors)
    work_queue = torch.tensor(descriptors, dtype=torch.int32, device=device).reshape(-1)
    work_counter = torch.zeros(1, dtype=torch.int32, device=device)
    return work_queue, work_counter, num_tiles


def uniform_pad_concat(tensors, rows):
    """Zero-pad each tensor's leading dim up to ``rows`` and concat on dim 0.

    Used by the wgrad uniform-pad path to stack per-expert operands (or their
    scales) on the contraction/token axis. Padded rows are zero, so they
    contribute nothing to the token-axis contraction.
    """
    def _pad_rows(t):
        if t.size(0) == rows:
            return t.contiguous()
        out = t.new_zeros((rows, *t.shape[1:]))
        out[: t.size(0)].copy_(t)
        return out

    return torch.cat([_pad_rows(t) for t in tensors], dim=0)
