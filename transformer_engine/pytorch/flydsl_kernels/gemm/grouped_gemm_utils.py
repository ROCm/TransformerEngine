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

import functools
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


@functools.lru_cache(maxsize=512)
def _fwd_dgrad_queue_cached(m_splits, num_pid_n, block_m, device):
    """Cached flat ``(pid_m, pid_n, expert)`` work queue for a fwd/dgrad shape.

    Keyed by the per-expert token counts + tiling, so repeated calls with the same
    ``m_splits`` (every microbatch in training, every bench iteration) reuse one
    already-uploaded queue instead of rebuilding + re-H2D-copying it each forward.
    Built from the Python ``m_splits`` tuple directly -- NO device->host copy of
    ``group_offs`` (that copy forced a ``hipDeviceSynchronize`` that stranded the
    GPU idle for hundreds of microseconds before the kernel).
    """
    descriptors = []
    off = 0
    for g, m in enumerate(m_splits):
        m_start_blk = off // block_m
        m_end_blk = (off + m) // block_m
        for pid_m in range(m_start_blk, m_end_blk):
            for pid_n in range(num_pid_n):
                descriptors.append((pid_m, pid_n, g))
        off += m
    num_tiles = len(descriptors)
    work_queue = torch.tensor(descriptors, dtype=torch.int32, device=device).reshape(-1)
    return work_queue, num_tiles


def build_fwd_dgrad_work_queue(m_splits, num_pid_n, block_m, device):
    """Return ``(work_queue, work_counter, num_tiles)`` for a fwd/dgrad launch.

    The descriptor queue is cached (shape-only, data-independent) so only the first
    call with a given ``m_splits`` pays the build+upload; the zeroed atomic counter
    is cheap and allocated fresh each call. ``m_splits`` is a Python list/tuple of
    per-expert token counts -- no GPU->CPU sync.
    """
    work_queue, num_tiles = _fwd_dgrad_queue_cached(
        tuple(int(m) for m in m_splits), int(num_pid_n), int(block_m), device
    )
    work_counter = torch.zeros(1, dtype=torch.int32, device=device)
    return work_queue, work_counter, num_tiles


@functools.lru_cache(maxsize=512)
def _wgrad_queue_cached(num_groups, num_pid_m, num_pid_n, device):
    descriptors = [
        (pid_m, pid_n, g)
        for g in range(num_groups)
        for pid_m in range(num_pid_m)
        for pid_n in range(num_pid_n)
    ]
    num_tiles = len(descriptors)
    work_queue = torch.tensor(descriptors, dtype=torch.int32, device=device).reshape(-1)
    return work_queue, num_tiles


def build_wgrad_work_queue(num_groups, num_pid_m, num_pid_n, device):
    """Return cached ``(work_queue, work_counter, num_tiles)`` for a wgrad launch.

    The wgrad descriptor queue depends only on ``(num_groups, num_pid_m, num_pid_n)``
    (per-expert tile geometry), so it is shape-cached and reused across calls -- no
    per-call Python rebuild or H2D re-upload. Fresh zeroed atomic counter each call.
    """
    work_queue, num_tiles = _wgrad_queue_cached(
        int(num_groups), int(num_pid_m), int(num_pid_n), device
    )
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


def bucketize_wgrad_tokens(m_splits, block_m, max_buckets=None):
    """Group experts by their per-expert padded contraction length (K-bucketing).

    The grouped wgrad kernel bakes its contraction length ``K`` in at compile time,
    so experts that share a padded token count can share one launch. Instead of
    padding every expert to the GLOBAL max token count (which wastes work and can
    OOM under skew), round each expert's ``m_g`` UP to its own ``block_m`` multiple
    (floor 512 = the pipeline's 4-K128-tile minimum), then group experts by that
    rounded value. Each distinct value is one bucket -> one fixed-K launch.

    When the number of distinct values exceeds ``max_buckets``, adjacent buckets
    are merged UPWARD (experts promoted to the larger K) so the launch count is
    capped; promoting only ever adds zero padding, never truncates, so results are
    unchanged. Returns a list of ``(k_pad, [global_expert_ids])`` tuples, ordered
    by ascending ``k_pad``.
    """
    def _round(m):
        return max(((int(m) + block_m - 1) // block_m) * block_m, 512)

    k_of = [_round(m) for m in m_splits]
    distinct = sorted(set(k_of))

    if max_buckets is not None and len(distinct) > max_buckets:
        # Merge the smallest-K buckets upward onto a coarser ladder until the
        # distinct-K count fits. Keep the largest K (one dominant expert) its own
        # bucket so it is never dragged smaller (it can't be) nor inflates others.
        # Simple policy: collapse the sorted distinct values into `max_buckets`
        # contiguous groups, each group promoted to its group-max.
        import math
        per = math.ceil(len(distinct) / max_buckets)
        ladder = {}
        for i in range(0, len(distinct), per):
            grp = distinct[i : i + per]
            top = grp[-1]
            for v in grp:
                ladder[v] = top
        k_of = [ladder[k] for k in k_of]
        distinct = sorted(set(k_of))

    buckets = []
    for k_pad in distinct:
        ids = [g for g, k in enumerate(k_of) if k == k_pad]
        buckets.append((k_pad, ids))
    return buckets
