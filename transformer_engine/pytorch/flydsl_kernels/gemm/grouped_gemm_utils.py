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
  ``read_desc_field``) -- control-flow-free functions that
  append IR at the current insertion point, so they emit identically whether
  defined inline or imported.
* **Host-side builders** (descriptor-queue construction, schedule selection,
  work-counter allocation, uniform token-axis padding) -- plain Python/torch,
  identical across dtypes; they differ only in whether scales are also padded.
"""

import functools
import os

import numpy as np
import torch

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly as _fly
from flydsl._mlir.dialects import llvm
from flydsl.expr.typing import T

from .exceptions import FlyDSLUnsupportedError


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
    byte_off = fx.as_ir_value(fx.Int64(fx.Int64(index) * fx.Int64(4)))
    addr = llvm.AddOp(base, byte_off, llvm.IntegerOverflowFlags(0)).result
    return llvm.IntToPtrOp(ptr_ty, addr).result


def read_ptr_at(ptr_array, index):
    """Load the int64 base address stored at ``ptr_array[index]``.

    ``ptr_array`` is a device ``int64[G]`` of per-expert operand base addresses
    (each the ``data_ptr()`` of one expert's independently-allocated weight slab).
    Recovers the array base, offsets by ``index`` int64 elements (8 bytes), and
    loads the stored address as an i64 IR value -- the per-expert operand base.
    """
    ptr_ty = ir.Type.parse("!llvm.ptr<1>")
    base = _fly.extract_aligned_pointer_as_index(ptr_ty, ptr_array)
    base = llvm.PtrToIntOp(T.i64, base).result
    byte_off = fx.as_ir_value(fx.Int64(fx.Int64(index) * fx.Int64(8)))
    addr = llvm.AddOp(base, byte_off, llvm.IntegerOverflowFlags(0)).result
    elt_ptr = llvm.IntToPtrOp(ptr_ty, addr).result
    return llvm.LoadOp(T.i64, elt_ptr).result


def load_i32_at(tensor, index):
    """Load ``tensor[index]`` (int32) with a plain per-lane global load."""
    return llvm.LoadOp(T.i32, global_i32_ptr(tensor, index)).result


def make_uniform_i64(value):
    """Broadcast lane 0's i64 to the whole wave (``readfirstlane``).

    A buffer descriptor lives in SGPRs. An address loaded by :func:`read_ptr_at` is a
    per-lane value, so a descriptor rebased on it can make the compiler guard each
    buffer access for divergence. Use this when every lane loaded the same address
    (e.g. indexed by the tile's expert), so lane 0's value is every lane's value.
    """
    return fx.rocdl.readfirstlane(T.i64, value)


def load_f32_at_ptr(ptr_array, index):
    """Load the fp32 scalar that ``ptr_array[index]`` points to.

    A plain global load needs no wave-uniform address, so unlike the buffer-descriptor
    bases this skips :func:`make_uniform_i64`: nothing waits on either load until the
    value is used, which keeps a per-tile scale fetch off the tile's critical path.
    """
    ptr_ty = ir.Type.parse("!llvm.ptr<1>")
    addr = read_ptr_at(ptr_array, index)
    return llvm.LoadOp(T.f32, llvm.IntToPtrOp(ptr_ty, addr).result).result


def rebase_buffer_view(base_i64, elem_ir_type, layout):
    """Rebuild a buffer-tensor logical-divide view at a runtime base address.

    Reuses an EXISTING static ``layout`` (compile-time cosize) but swaps the base
    pointer to the runtime ``base_i64`` address. This is how a per-expert operand
    slice is addressed without constructing a fresh runtime-sized layout
    (``fx.make_layout`` requires a static extent): the packed operand's layout is
    reused and only the base moves. ``max_size=True`` lets the buffer descriptor
    span the per-expert extent; the loader's dynamic ``soffset`` walks within it.
    """
    buf_ptr_ty = fx.PointerType.get(elem_ty=elem_ir_type, address_space=1)
    base_ptr = fx.inttoptr(buf_ptr_ty, fx.Int64(base_i64))
    view = fx.make_view(base_ptr, layout)
    buf = fx.rocdl.make_buffer_tensor(fx.Tensor(view), max_size=True)
    return fx.logical_divide(buf, fx.make_layout(1, 1))


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
    return "static" if num_tiles <= _GROUPED_SCHED_ROUNDS * _num_cus(device) else "worksteal"


@functools.lru_cache(maxsize=None)
def _num_cus(device):
    return torch.cuda.get_device_properties(device).multi_processor_count


# Per-call metadata caches. Keys include the current stream: a cached device tensor is
# filled by a non-blocking copy on the stream that first built it, so only launches on
# that stream are ordered after the copy. Pointer arrays and block tables are tiny
# (<= a few KB each), so their cache is large enough to hold a training step's worth of
# rotating per-expert buffers (each step adds several arrays); the work queues can be
# much larger, so their cache stays small.
_SMALL_CACHE_SIZE = 512
_QUEUE_CACHE_SIZE = 64


def current_stream_handle(device=None):
    """Raw handle of the current stream on ``device`` (default: the current device).

    FlyDSL launchers accept an int stream, and this is ~20x cheaper than building a
    torch ``Stream`` object per launch.
    """
    index = None if device is None else device.index
    if index is None:
        index = torch.cuda.current_device()
    return torch._C._cuda_getCurrentRawStream(index)


def upload_to_device(host, device):
    """Copy a host CPU tensor to ``device`` without blocking the host.

    ``torch.tensor(..., device="cuda")`` waits for all queued GPU work before copying,
    which stalls the host on every call. Staging through pinned memory and copying
    ``non_blocking`` does not; the caching host allocator keeps the pinned block alive
    until the copy has run. The staging block comes from ``torch.empty(pin_memory=True)``
    (the caching host allocator) rather than ``Tensor.pin_memory()``, which is ~3x slower
    per call while GPU work is queued.
    """
    staging = torch.empty(host.shape, dtype=host.dtype, pin_memory=True)
    staging.copy_(host)
    return staging.to(device, non_blocking=True)


@functools.lru_cache(maxsize=_QUEUE_CACHE_SIZE)
def _fwd_dgrad_queue_cached(m_splits, num_pid_n, block_m, device, stream):
    """Flat ``(pid_m, pid_n, expert)`` work queue for a fwd/dgrad shape.

    Every expert's token count is a ``block_m`` multiple, so the expert row blocks tile
    ``[0, M_total)`` in order: the queue is each row block repeated over the output
    columns, tagged with its expert. Built with numpy from the host ``m_splits`` (no
    device->host copy of ``group_offs``) and uploaded without blocking the host.
    """
    blocks = np.asarray(m_splits, dtype=np.int64) // block_m
    num_blocks = int(blocks.sum())
    queue = np.empty((num_blocks, num_pid_n, 3), dtype=np.int32)
    queue[:, :, 0] = np.arange(num_blocks, dtype=np.int32)[:, None]
    queue[:, :, 1] = np.arange(num_pid_n, dtype=np.int32)[None, :]
    queue[:, :, 2] = np.repeat(np.arange(len(m_splits), dtype=np.int32), blocks)[:, None]
    work_queue = upload_to_device(torch.from_numpy(queue.reshape(-1)), device)
    return work_queue, num_blocks * num_pid_n


def build_fwd_dgrad_work_queue(m_splits, num_pid_n, block_m, device):
    """Return ``(work_queue, num_tiles)`` for a fwd/dgrad launch.

    Raises ``FlyDSLUnsupportedError`` when an expert's token count is not a ``block_m``
    multiple: the kernel tiles each expert in whole ``block_m`` row blocks, so a
    misaligned split would silently mix experts within a tile. ``num_tiles`` is the
    exact queue length, which the kernel uses as its drain bound. ``m_splits`` is a
    Python list/tuple of per-expert token counts -- no GPU->CPU sync.
    """
    m_splits = tuple(int(m) for m in m_splits)
    for g, m in enumerate(m_splits):
        if m % block_m != 0:
            raise FlyDSLUnsupportedError(
                f"m_splits[{g}]={m} is not a multiple of BLOCK_M={block_m}"
            )
    return _fwd_dgrad_queue_cached(
        m_splits, int(num_pid_n), int(block_m), device, current_stream_handle(device)
    )


@functools.lru_cache(maxsize=_QUEUE_CACHE_SIZE)
def _wgrad_queue_cached(num_groups, num_pid_m, num_pid_n, device, stream):
    queue = np.empty((num_groups, num_pid_m, num_pid_n, 3), dtype=np.int32)
    queue[..., 0] = np.arange(num_pid_m, dtype=np.int32)[None, :, None]
    queue[..., 1] = np.arange(num_pid_n, dtype=np.int32)[None, None, :]
    queue[..., 2] = np.arange(num_groups, dtype=np.int32)[:, None, None]
    work_queue = upload_to_device(torch.from_numpy(queue.reshape(-1)), device)
    return work_queue, num_groups * num_pid_m * num_pid_n


def build_wgrad_work_queue(num_groups, num_pid_m, num_pid_n, device):
    """Return ``(work_queue, num_tiles)`` for a wgrad launch.

    The wgrad descriptor queue depends only on ``(num_groups, num_pid_m, num_pid_n)``
    (per-expert tile geometry), so it is shape-cached and reused across calls;
    ``num_tiles`` is the kernel's drain bound.
    """
    return _wgrad_queue_cached(
        int(num_groups), int(num_pid_m), int(num_pid_n), device, current_stream_handle(device)
    )


@functools.lru_cache(maxsize=None)
def _unused_counter(device):
    return torch.zeros(1, dtype=torch.int32, device=device)


def work_counter(sched, device):
    """The atomic tile counter a grouped launch needs for schedule ``sched``.

    Work-stealing claims tiles from a freshly zeroed counter. The static schedule never
    touches it, so it gets a shared placeholder instead of a per-launch zero fill.
    """
    if sched == "worksteal":
        return torch.zeros(1, dtype=torch.int32, device=device)
    return _unused_counter(device)


class _SpanCAI:
    """Minimal ``__cuda_array_interface__`` holder for a raw uint8 device span.

    Wraps ``total_bytes`` at ``data_ptr`` as a 1-D uint8 CAI so ``torch.as_tensor``
    can build a zero-copy view over a contiguous multi-tensor run (see
    :func:`concat_or_view_rows`). Hoisted to module scope so the class object is
    built once at import, not reconstructed on every call.
    """

    __slots__ = ("__cuda_array_interface__",)

    def __init__(self, total_bytes, data_ptr):
        self.__cuda_array_interface__ = {
            "shape": (total_bytes,),
            "typestr": "|u1",
            "data": (data_ptr, False),
            "strides": None,
            "version": 3,
        }


def concat_or_view_rows(tensors):
    """Concatenate per-expert tensors on dim 0, skipping the copy when possible.

    ``tex.split_quantize`` bulk-allocates its per-expert outputs into one base
    buffer (data blocks packed back-to-back, then scale blocks back-to-back), so
    the per-expert rowwise payloads are frequently already physically adjacent --
    the ``[M_total, cols]`` concatenation the grouped kernel wants already exists
    in memory. When that holds, return a zero-copy spanning view over the shared
    region instead of materializing a redundant ``torch.cat`` copy (the activation
    and activation-scale cats on the MXFP8 forward/dgrad hot path).

    Adjacency is not guaranteed (independently-allocated operands -- e.g. weights
    from separate ``nn.Parameter`` storages -- are scattered), so fall back to
    ``torch.cat`` whenever the inputs are not a single contiguous run. The view and
    the cat produce bitwise-identical operands; only the copy is elided.
    """
    if len(tensors) == 1:
        return tensors[0]

    first = tensors[0]
    if not first.is_cuda or first.ndim < 1:
        return torch.cat(tensors, dim=0)

    # __cuda_array_interface__ is unavailable on grad-tracking tensors. The grouped
    # GEMM operand is consumed inside a custom autograd Function (grad handled at the
    # module level), so a detached data view is correct; detach once up front so the
    # adjacency walk and span wrapper both see plain tensors.
    if any(t.requires_grad for t in tensors):
        tensors = [t.detach() for t in tensors]
        first = tensors[0]

    trailing = first.shape[1:]
    elem = first.element_size()
    total_rows = 0
    expected_ptr = first.data_ptr()
    for t in tensors:
        # Adjacency + uniformity: same trailing shape/dtype, row-contiguous, and
        # starting exactly where the previous tensor ended.
        if (
            t.dtype != first.dtype
            or t.shape[1:] != trailing
            or not t.is_contiguous()
            or t.data_ptr() != expected_ptr
        ):
            return torch.cat(tensors, dim=0)
        total_rows += t.size(0)
        expected_ptr += t.numel() * elem

    # One contiguous run: wrap [total_rows*row_bytes] over the shared base pointer as
    # raw uint8, then reinterpret back to the operand dtype. Going through bytes keeps
    # this dtype-agnostic -- __cuda_array_interface__ has no typestr for bfloat16, so a
    # native-dtype CAI would reject it; uint8 is always representable.
    # The CAI carries no ownership, so anchor the source tensors on the result to keep
    # their bulk buffer alive for the view's lifetime.
    row_bytes = 1
    for d in trailing:
        row_bytes *= d
    row_bytes *= elem
    total_bytes = total_rows * row_bytes

    byte_view = torch.as_tensor(_SpanCAI(total_bytes, first.data_ptr()), device=first.device)
    view = byte_view.view(first.dtype).view(total_rows, *tuple(trailing))
    view._flydsl_span_anchor = tensors
    return view


@functools.lru_cache(maxsize=_SMALL_CACHE_SIZE)
def _ptr_array_cached(ptrs, device, stream):
    return upload_to_device(torch.tensor(ptrs, dtype=torch.int64), device)


def expert_ptrs(tensors, device):
    """int64[G] device array of each per-expert tensor's ``data_ptr()``.

    Grouped kernels read per-expert operands (weights, FP8 scale_inv) through this
    array instead of a host-side concat. Repeated pointer sets reuse the uploaded array;
    a new set costs one non-blocking upload.
    """
    return _ptr_array_cached(
        tuple(t.data_ptr() for t in tensors), device, current_stream_handle(device)
    )


def stacked_operand(tensors, device):
    """Address per-expert tensors stacked on dim 0 in place, as one packed operand.

    Returns ``(representative, ptrs)``, or ``None`` when the tensors are not
    contiguous with a common trailing shape and dtype. Entry ``g`` of the int64[G]
    ``ptrs`` is ``tensors[g].data_ptr()`` minus the byte offset where expert ``g``
    starts in the ``torch.cat(tensors)`` layout, so a kernel keeps its packed
    addressing and only swaps the buffer base per tile. ``representative`` is a
    non-empty expert tensor standing in for the operand's dtype.
    """
    first = tensors[0]
    dtype = first.dtype
    trailing = first.shape[1:]
    ptrs = []
    offset = 0
    for t in tensors:
        if t.dtype is not dtype or t.shape[1:] != trailing or not t.is_contiguous():
            return None
        ptrs.append(t.data_ptr() - offset)
        offset += t.nbytes
    representative = next((t for t in tensors if t.numel() > 0), first)
    return representative, _ptr_array_cached(tuple(ptrs), device, current_stream_handle(device))


@functools.lru_cache(maxsize=_SMALL_CACHE_SIZE)
def _block_expert_table_cached(blocks_per_expert, device, stream):
    table = np.repeat(np.arange(len(blocks_per_expert), dtype=np.int32), blocks_per_expert)
    return upload_to_device(torch.from_numpy(table), device)


def block_expert_table(extents, block, device):
    """int32 table mapping each ``block``-sized slice of a stacked axis to its expert.

    ``extents`` are the per-expert lengths along the stacked axis; each must be a
    multiple of ``block``, so no block straddles two experts.
    """
    for n in extents:
        if int(n) % block != 0:
            raise ValueError(f"per-expert extent {int(n)} is not a multiple of {block}")
    return _block_expert_table_cached(
        tuple(int(n) // block for n in extents), device, current_stream_handle(device)
    )


def drop_empty_experts(buckets, m_splits, dW, n_out):
    """Zero the ``dW`` rows of zero-token experts and drop them from the wgrad buckets.

    An expert with no tokens has ``dW_e = 0``; launching it anyway contracts a zero
    padding that a non-finite scale (FP8 never writes an empty tensor's scale_inv)
    would turn into NaN.
    """
    for e, m in enumerate(m_splits):
        if int(m) == 0:
            dW[e * n_out : (e + 1) * n_out].zero_()
    buckets = [(k_pad, [e for e in ids if int(m_splits[e]) != 0]) for k_pad, ids in buckets]
    return [(k_pad, ids) for k_pad, ids in buckets if ids]


def uniform_pad_concat(tensors, rows):
    """Zero-pad each tensor's leading dim up to ``rows`` and concat on dim 0.

    Used by the wgrad uniform-pad path to stack per-expert operands (or their
    scales) on the contraction/token axis. Padded rows are zero, so they
    contribute nothing to the token-axis contraction.

    When no tensor actually needs padding (every leading dim already equals
    ``rows`` -- the balanced / no-skew case, where each expert's token count is a
    multiple of the bucket's BLOCK_M), there is nothing to zero-fill, so this
    degenerates to a plain concat. Delegate that case to ``concat_or_view_rows``,
    which returns a zero-copy spanning view when the per-expert payloads are
    physically adjacent (the common bulk-allocated layout), eliminating the
    ``torch.cat`` copy on the wgrad hot path.
    """
    if all(t.size(0) == rows for t in tensors):
        return concat_or_view_rows(tensors)

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
