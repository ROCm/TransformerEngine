# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

"""Concat-free FP8 quantization of several tensors under a single shared scale.

The attention FP8 path historically flattened and concatenated Q/K/V into one
buffer purely so that a single amax (and therefore a single scale) covered all
three tensors. The concatenation itself costs a full read+write of Q/K/V.

The kernel here keeps the tensors where they are and walks them as one virtual
concatenation instead: the launch grid is partitioned so that every program
owns a block belonging to exactly one input, and the per-tensor loads/stores of
the other inputs are fully masked off. A single atomic-max accumulator gives
the joint amax without materializing the joint buffer.

Delayed scaling only. Its scale is fixed ahead of the cast, so the joint amax
is a pure output and can be accumulated in the same pass as the cast -- which
is also why the C++ cast kernels atomic-max into the running amax without
clearing it. Current scaling needs the joint amax *before* it can pick a scale
and is deliberately not handled here.
"""

import functools
from typing import List, Optional, Sequence, Tuple

import torch
import triton
import triton.language as tl

from .common import get_fp8_max, te_dtype_to_triton_dtype

# Largest number of tensors a single launch can cover.
MAX_FUSED_TENSORS = 3


@triton.jit
def _multi_cast_kernel(
    p0, p1, p2,
    o0, o1, o2,
    n0, n1, n2,
    scale_ptr,                # float32[1]
    amax_ptr,                 # float32[1], accumulated (not cleared)
    scale_inv_ptr,            # float32[1]
    noop_ptr,
    max_fp8: tl.constexpr,
    USE_NOOP: tl.constexpr,
    NUM_TENSORS: tl.constexpr,
    UNROLL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    if USE_NOOP:
        if tl.load(noop_ptr) == 1.0:
            return

    pid = tl.program_id(0)
    scale = tl.load(scale_ptr)
    CHUNK: tl.constexpr = BLOCK * UNROLL

    # Map the flat program id onto (tensor index, chunk index within tensor).
    nb0 = tl.cdiv(n0, CHUNK)
    nb1 = tl.cdiv(n1, CHUNK) if NUM_TENSORS > 1 else 0
    tid = 0
    local_pid = pid
    if NUM_TENSORS > 1:
        if pid >= nb0:
            tid = 1
            local_pid = pid - nb0
    if NUM_TENSORS > 2:
        if pid >= nb0 + nb1:
            tid = 2
            local_pid = pid - nb0 - nb1

    # One atomic per program, so each program covers UNROLL blocks: the atomics
    # against the single amax address are what limits this kernel, not the
    # memory traffic. Keep the running max as a scalar -- holding a BLOCK-wide
    # vector accumulator instead costs enough registers to lose occupancy.
    acc = 0.0
    base = local_pid.to(tl.int64) * CHUNK
    for u in tl.static_range(UNROLL):
        off = base + u * BLOCK + tl.arange(0, BLOCK)

        # Only one of these masks is ever true, so the other loads/stores issue
        # no memory traffic.
        m0 = (tid == 0) & (off < n0)
        a = tl.load(p0 + off, mask=m0, other=0)
        if NUM_TENSORS > 1:
            m1 = (tid == 1) & (off < n1)
            # tl.where (not a sum) so that signed zeros survive the merge.
            a = tl.where(tid == 1, tl.load(p1 + off, mask=m1, other=0), a)
        if NUM_TENSORS > 2:
            m2 = (tid == 2) & (off < n2)
            a = tl.where(tid == 2, tl.load(p2 + off, mask=m2, other=0), a)

        a = a.to(tl.float32)
        scaled_a = tl.clamp(a * scale, -max_fp8, max_fp8)
        fp8_a = scaled_a.to(o0.type.element_ty)

        tl.store(o0 + off, fp8_a, mask=m0)
        if NUM_TENSORS > 1:
            tl.store(o1 + off, fp8_a, mask=m1)
        if NUM_TENSORS > 2:
            tl.store(o2 + off, fp8_a, mask=m2)

        # amax is taken on the unscaled input, matching the C++ cast kernels.
        acc = tl.maximum(acc, tl.max(tl.abs(a)))

    tl.atomic_max(amax_ptr, acc, sem='relaxed')

    if pid == 0:
        tl.store(scale_inv_ptr, tl.fdiv(1.0, scale))


# This kernel is limited by atomic_max traffic against the single amax address,
# and that cost tracks the *number* of atomics, i.e. the number of programs.
# So make each program as fat as possible, stopping once the grid is only just
# big enough to fill the GPU; past _MAX_CHUNK elements per program the kernel is
# already at the memory-bandwidth floor and there is nothing left to win.
_MIN_CHUNK = 1024
_MAX_CHUNK = 32768
_MAX_BLOCK = 8192


@functools.lru_cache(maxsize=None)
def _min_programs(device_index: int) -> int:
    return 2 * torch.cuda.get_device_properties(device_index).multi_processor_count


def _launch_config(total_elements: int, device_index: int) -> Tuple[int, int, int]:
    """Pick (BLOCK, UNROLL, num_warps) for this streaming 1D kernel.

    Deliberately a static heuristic rather than triton.autotune: this runs on
    every attention call and autotune would recompile whenever a sequence
    length changes.
    """
    chunk = 1 << max(
        0, (total_elements // _min_programs(device_index)).bit_length() - 1
    )
    chunk = min(max(chunk, _MIN_CHUNK), _MAX_CHUNK)
    block = min(chunk, _MAX_BLOCK)
    return block, chunk // block, 8 if block >= _MAX_BLOCK else 4


def _pad_args(values: Sequence, fill) -> List:
    return list(values) + [fill] * (MAX_FUSED_TENSORS - len(values))


def multi_tensor_quantize_fp8_triton(
    tensors: Sequence[torch.Tensor],
    quantizer,
    noop_flag: Optional[torch.Tensor] = None,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Quantize several tensors to FP8 under one shared amax/scale, no concat.

    Numerically equivalent to flattening the tensors, concatenating them and
    quantizing the result with ``quantizer``.

    Parameters
    ----------
    tensors: contiguous high-precision tensors, all of the same dtype. Up to
             ``MAX_FUSED_TENSORS`` of them.
    quantizer: a ``Float8Quantizer`` (per-tensor delayed scaling).
    noop_flag: optional float32[1]; when it holds 1.0 the cast is skipped.

    Returns
    -------
    (data, scale_inv) where ``data`` holds one uint8 tensor per input, shaped
    like that input, and ``scale_inv`` is the single float32[1] buffer shared
    by all of them.
    """
    from ..tensor.float8_tensor import Float8Quantizer

    num_tensors = len(tensors)
    assert 0 < num_tensors <= MAX_FUSED_TENSORS, (
        f"multi_tensor_quantize_fp8_triton supports 1..{MAX_FUSED_TENSORS} tensors,"
        f" got {num_tensors}"
    )
    assert isinstance(quantizer, Float8Quantizer), (
        "multi_tensor_quantize_fp8_triton only supports per-tensor delayed scaling,"
        f" got {type(quantizer).__name__}"
    )
    ref = tensors[0]
    assert all(t.dtype == ref.dtype for t in tensors), "all tensors must share a dtype"
    assert all(t.device == ref.device for t in tensors), "all tensors must share a device"
    assert all(t.is_contiguous() for t in tensors), "all tensors must be contiguous"

    tl_dtype = te_dtype_to_triton_dtype(quantizer.dtype)

    numels = [t.numel() for t in tensors]
    total = sum(numels)

    out_data = [torch.empty(t.shape, dtype=torch.uint8, device=t.device) for t in tensors]
    scale_inv = torch.empty(1, dtype=torch.float32, device=ref.device)

    if total == 0:
        scale_inv.copy_(1.0 / quantizer.scale)
        return out_data, scale_inv

    block, unroll, num_warps = _launch_config(total, ref.device.index)
    grid = (sum(triton.cdiv(n, block * unroll) for n in numels),)

    in_ptrs = _pad_args([t.reshape(-1) for t in tensors], ref.reshape(-1))
    out_ptrs = _pad_args(
        [triton.reinterpret(o.reshape(-1), tl_dtype) for o in out_data],
        triton.reinterpret(out_data[0].reshape(-1), tl_dtype),
    )
    lens = _pad_args(numels, 0)

    use_noop = noop_flag is not None and noop_flag.numel() > 0
    if not use_noop:
        noop_flag = ref  # unused placeholder; the kernel never dereferences it

    _multi_cast_kernel[grid](
        *in_ptrs,
        *out_ptrs,
        *lens,
        quantizer.scale,
        quantizer.amax,
        scale_inv,
        noop_flag,
        max_fp8=get_fp8_max(quantizer.dtype),
        USE_NOOP=use_noop,
        NUM_TENSORS=num_tensors,
        UNROLL=unroll,
        BLOCK=block,
        num_warps=num_warps,
    )

    return out_data, scale_inv
