# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#
# Blockwise FP8 quantization Triton kernels (1x128 activation, 128x128 weight),
# adapted from AMD Primus-Turbo (primus_turbo/triton/quantization/quant_blockwise.py
# and primus_turbo/pytorch/kernels/quantization/quantization_impl.py).

import torch
import triton
import triton.language as tl
import transformer_engine_torch as tex

from .common import te_dtype_to_torch_dtype

__all__ = [
    "quantize_fp8_blockwise",
    "quantize_fp8_blockwise_weight",
    "quantize_fp8_blockwise_segment_m",
    "quantize_fp8_blockwise_act_qtensor",
    "quantize_fp8_blockwise_grouped_weight_qtensor",
    "wrap_fp8_blockwise_grouped_weight_qtensor",
    "quantize_fp8_blockwise_segment_m_operand",
    "wrap_fp8_blockwise_segment_m_qtensor",
    "quantize_fp8_blockwise_act_and_segment_m",
    "quantize_fp8_blockwise_act_operands",
]


@triton.jit
def _floor_to_pow2(scale):
    scale_bits = scale.to(tl.uint32, bitcast=True) & 0xFF800000
    return scale_bits.to(tl.float32, bitcast=True)


@triton.jit
def compute_scale_and_quant(x_tile, x_tile_abs, axis, FP8_MAX, ROUND_POW2: tl.constexpr):
    x_tile_max = tl.max(x_tile_abs, axis=axis, keep_dims=True)
    x_tile_max = tl.maximum(x_tile_max, 1e-4)
    x_scales_tile = FP8_MAX / x_tile_max
    if ROUND_POW2:
        x_scales_tile = _floor_to_pow2(x_scales_tile)
    x_fp8_tile = x_tile * x_scales_tile
    x_fp8_tile = tl.clamp(x_fp8_tile, min=-FP8_MAX, max=FP8_MAX)
    return x_fp8_tile, x_scales_tile


@triton.jit
def quant_fp8_blockwise_kernel(
    x_ptr,
    x_fp8_ptr,
    x_scales_ptr,
    M,
    N,
    BLOCK_SIZE: tl.constexpr,
    FP8_MAX: tl.constexpr,
    AXIS: tl.constexpr,
    ROUND_POW2: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_m = tl.cast(pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE), tl.int64)
    offs_n = tl.cast(pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE), tl.int64)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    x_ptrs = x_ptr + offs_m[:, None] * N + offs_n[None, :]
    x_tile = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    x_tile_abs = tl.abs(x_tile)

    x_fp8_tile, x_scales_tile = compute_scale_and_quant(
        x_tile, x_tile_abs, AXIS, FP8_MAX, ROUND_POW2
    )

    x_fp8_ptrs = x_fp8_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(x_fp8_ptrs, x_fp8_tile.to(x_fp8_ptr.dtype.element_ty), mask=mask)

    if AXIS == 1:
        scale_offs = offs_m * tl.cdiv(N, BLOCK_SIZE) + pid_n
        scale_mask = offs_m < M
    else:
        scale_offs = pid_m * N + offs_n
        scale_mask = offs_n < N
    x_scales_tile_inv = tl.reshape(1.0 / x_scales_tile, BLOCK_SIZE)
    tl.store(x_scales_ptr + scale_offs, x_scales_tile_inv, mask=scale_mask)


@triton.jit
def quant_fp8_blockwise_for_weight_kernel(
    w_ptr,
    w_fp8_ptr,
    w_scales_ptr,
    M,
    N,
    BLOCK_SIZE: tl.constexpr,
    FP8_MAX: tl.constexpr,
    ROUND_POW2: tl.constexpr,
):
    bid = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)

    batch_offset_w = bid * M * N
    batch_offset_scales = bid * tl.cdiv(M, BLOCK_SIZE) * tl.cdiv(N, BLOCK_SIZE)

    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    w_ptrs = w_ptr + batch_offset_w + offs_m[:, None] * N + offs_n[None, :]
    w_tile = tl.load(w_ptrs, mask=mask, other=0.0).to(tl.float32)

    w_tile_abs = tl.abs(w_tile)
    w_tile_max = tl.max(w_tile_abs)
    w_tile_max = tl.maximum(w_tile_max, 1e-4)
    w_scales = FP8_MAX / w_tile_max
    if ROUND_POW2:
        w_scales = _floor_to_pow2(w_scales)
    w_fp8_tile = w_tile * w_scales
    w_fp8_tile = tl.clamp(w_fp8_tile, min=-FP8_MAX, max=FP8_MAX)

    w_fp8_ptrs = w_fp8_ptr + batch_offset_w + offs_m[:, None] * N + offs_n[None, :]
    tl.store(w_fp8_ptrs, w_fp8_tile.to(w_fp8_ptr.dtype.element_ty), mask=mask)
    scale_offs = batch_offset_scales + pid_m * tl.cdiv(N, BLOCK_SIZE) + pid_n
    w_scales_inv = 1.0 / w_scales
    tl.store(w_scales_ptr + scale_offs, w_scales_inv)


@triton.jit
def quant_fp8_blockwise_grouped_kernel(
    x_ptr,  # Input tensor [M_in, N]
    x_fp8_row_ptr,  # Rowwise output [M_in, N] (unused when ROWWISE is False)
    x_scales_row_ptr,  # Rowwise scales [M_in, ceil(N/BLOCK_SIZE)] (unused when ROWWISE is False)
    x_fp8_col_ptr,  # Colwise padded output [M_pad, N] (unused when COLUMNWISE is False)
    x_scales_col_ptr,  # Colwise scales [ceil(M_pad/BLOCK_SIZE), N] (unused when COLUMNWISE is False)
    group_offs_ptr,  # Original group offsets [B+1]
    padded_group_offs_ptr,  # Padded group offsets [B+1]
    N,
    num_groups,
    BLOCK_SIZE: tl.constexpr,
    FP8_MAX: tl.constexpr,
    ROUND_POW2: tl.constexpr,
    ROWWISE: tl.constexpr,
    COLUMNWISE: tl.constexpr,
):
    """Grouped blockwise quantize: rowwise and/or segment-padded columnwise.

    The grid iterates the *segment-padded* output tiles (each MoE group padded up
    to a ``BLOCK_SIZE`` boundary along M). Each program reads one ``[BLOCK_SIZE,
    BLOCK_SIZE]`` block of the original (unpadded) input once and, from that single
    tile, emits the operand(s) selected by the ``constexpr`` flags:

    * ``COLUMNWISE`` (axis=0, BLOCKx1 along M) -- writes FP8 data + 1D scales to the
      *segment-padded* positions; this is the per-segment M padding used by the
      variable-K wgrad (each group starts on a BLOCK_SIZE boundary).
    * ``ROWWISE`` (axis=1, 1xBLOCK along N) -- writes FP8 data + 1D scales to the
      *original* (unpadded) positions (the forward/dgrad activation operand);
      padding rows are skipped.

    With both flags set this is the fused activation + wgrad quantize (one HBM read
    of the input); with only ``COLUMNWISE`` it is the plain segment-padded colwise
    quantize.
    """
    tl.static_assert(ROWWISE or COLUMNWISE, "at least one of ROWWISE/COLUMNWISE must be set")
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    M_padded = tl.load(padded_group_offs_ptr + num_groups)
    block_start = pid_m * BLOCK_SIZE
    if block_start >= M_padded:
        return

    group_id = 0
    for g in range(num_groups):
        padded_start = tl.load(padded_group_offs_ptr + g)
        padded_end = tl.load(padded_group_offs_ptr + g + 1)
        if block_start >= padded_start and block_start < padded_end:
            group_id = g

    orig_group_start = tl.load(group_offs_ptr + group_id)
    orig_group_end = tl.load(group_offs_ptr + group_id + 1)
    padded_group_start = tl.load(padded_group_offs_ptr + group_id)

    offs_m_out = tl.cast(pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE), tl.int64)
    offs_n = tl.cast(pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE), tl.int64)
    offs_m_in = orig_group_start + (offs_m_out - padded_group_start)

    row_valid = (offs_m_in >= orig_group_start) & (offs_m_in < orig_group_end)
    mask = row_valid[:, None] & (offs_n[None, :] < N)

    x_ptrs = x_ptr + offs_m_in[:, None] * N + offs_n[None, :]
    x_tile = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    x_tile_abs = tl.abs(x_tile)

    if COLUMNWISE:
        # axis=0 -> segment-padded output positions.
        x_fp8_col_tile, x_scales_col_tile = compute_scale_and_quant(
            x_tile, x_tile_abs, 0, FP8_MAX, ROUND_POW2
        )
        x_fp8_col_ptrs = x_fp8_col_ptr + offs_m_out[:, None] * N + offs_n[None, :]
        col_out_mask = (offs_m_out[:, None] < M_padded) & (offs_n[None, :] < N)
        tl.store(
            x_fp8_col_ptrs, x_fp8_col_tile.to(x_fp8_col_ptr.dtype.element_ty), mask=col_out_mask
        )
        col_scale_offs = pid_m * N + offs_n
        col_scale_mask = (pid_m < tl.cdiv(M_padded, BLOCK_SIZE)) & (offs_n < N)
        x_scales_col_tile_inv = tl.reshape(1.0 / x_scales_col_tile, BLOCK_SIZE)
        tl.store(x_scales_col_ptr + col_scale_offs, x_scales_col_tile_inv, mask=col_scale_mask)

    if ROWWISE:
        # axis=1 -> original (unpadded) positions; padding rows are skipped.
        x_fp8_row_tile, x_scales_row_tile = compute_scale_and_quant(
            x_tile, x_tile_abs, 1, FP8_MAX, ROUND_POW2
        )
        x_fp8_row_ptrs = x_fp8_row_ptr + offs_m_in[:, None] * N + offs_n[None, :]
        tl.store(x_fp8_row_ptrs, x_fp8_row_tile.to(x_fp8_row_ptr.dtype.element_ty), mask=mask)
        row_scale_offs = offs_m_in * tl.cdiv(N, BLOCK_SIZE) + pid_n
        x_scales_row_tile_inv = tl.reshape(1.0 / x_scales_row_tile, BLOCK_SIZE)
        tl.store(x_scales_row_ptr + row_scale_offs, x_scales_row_tile_inv, mask=row_valid)


# -----------------------------------------------------------------------------
# Host launchers
#
# Plain functions, not ``torch.library.custom_op``. A custom_op wrapper may be
# needed if torch.compile hits the inductor ``identify_mutated_tensors`` bug
# seen on gfx942 + Triton 3.7.
# -----------------------------------------------------------------------------


def quantize_fp8_blockwise(
    x: torch.Tensor, dtype: torch.dtype, axis: int, block_size: int = 128, pow2: bool = False
):
    """Single-direction blockwise quantize. axis=1 -> rowwise (1xB), axis=0 -> colwise (Bx1)."""
    assert x.is_contiguous() and x.dim() == 2, "Input must be 2D and contiguous"
    M, N = x.shape
    fp8_max = torch.finfo(dtype).max
    x_fp8 = torch.empty((M, N), dtype=dtype, device=x.device)
    if axis == 1:
        scales = torch.empty((M, triton.cdiv(N, block_size)), dtype=torch.float32, device=x.device)
    else:
        scales = torch.empty((triton.cdiv(M, block_size), N), dtype=torch.float32, device=x.device)

    grid = (triton.cdiv(M, block_size), triton.cdiv(N, block_size))
    quant_fp8_blockwise_kernel[grid](
        x,
        x_fp8,
        scales,
        M,
        N,
        BLOCK_SIZE=block_size,
        FP8_MAX=fp8_max,
        AXIS=axis,
        ROUND_POW2=pow2,
    )
    return x_fp8, scales


def quantize_fp8_blockwise_weight(
    w: torch.Tensor, dtype: torch.dtype, block_size: int = 128, pow2: bool = False
):
    """128x128 weight blockwise quantize. w is [B, M, N] (or [M, N], promoted to B=1)."""
    squeeze = False
    if w.dim() == 2:
        w = w.unsqueeze(0)
        squeeze = True
    assert w.is_contiguous() and w.dim() == 3, "Weight must be 3D [B,M,N] and contiguous"
    B, M, N = w.shape
    fp8_max = torch.finfo(dtype).max

    w_fp8 = torch.empty((B, M, N), dtype=dtype, device=w.device)
    w_scales = torch.empty(
        (B, triton.cdiv(M, block_size), triton.cdiv(N, block_size)),
        dtype=torch.float32,
        device=w.device,
    )
    grid = (B, triton.cdiv(M, block_size), triton.cdiv(N, block_size))
    quant_fp8_blockwise_for_weight_kernel[grid](
        w,
        w_fp8,
        w_scales,
        M,
        N,
        BLOCK_SIZE=block_size,
        FP8_MAX=fp8_max,
        ROUND_POW2=pow2,
    )
    if squeeze:
        return w_fp8.squeeze(0), w_scales.squeeze(0)
    return w_fp8, w_scales


def quantize_fp8_blockwise_segment_m(
    x, dtype, block_size, group_lens, group_offs, pow2: bool = False
):
    """Colwise blockwise quantize with per-segment (MoE group) M padding.

    Returns (x_fp8 [M_pad, N], x_scales, var_k_group_lens [B], var_k_group_offs [B+1]).
    Each group is padded up to a multiple of block_size. Allocates with an upper
    bound (M + B*block_size) to avoid a device->host sync (graph-capture safe).
    """
    assert x.is_contiguous() and x.dim() == 2, "Input must be 2D and contiguous"
    M, N = x.shape
    num_groups = group_lens.size(0)
    fp8_max = torch.finfo(dtype).max

    var_k_group_lens = ((group_lens + block_size - 1) // block_size) * block_size
    var_k_group_offs = torch.zeros(num_groups + 1, dtype=torch.int64, device=x.device)
    var_k_group_offs[1:] = torch.cumsum(var_k_group_lens, dim=0)

    m_padded_max = M + num_groups * block_size
    x_fp8 = torch.zeros((m_padded_max, N), dtype=dtype, device=x.device)
    x_scales = torch.zeros(
        (triton.cdiv(m_padded_max, block_size), N), dtype=torch.float32, device=x.device
    )

    grid = (triton.cdiv(m_padded_max, block_size), triton.cdiv(N, block_size))
    # Columnwise-only: the rowwise pointer args are unused (compiled out) and are
    # given the colwise buffers as harmless placeholders.
    quant_fp8_blockwise_grouped_kernel[grid](
        x,
        x_fp8,
        x_scales,
        x_fp8,
        x_scales,
        group_offs,
        var_k_group_offs,
        N,
        num_groups,
        BLOCK_SIZE=block_size,
        FP8_MAX=fp8_max,
        ROUND_POW2=pow2,
        ROWWISE=False,
        COLUMNWISE=True,
    )
    return x_fp8, x_scales, var_k_group_lens, var_k_group_offs


def _torch_fp8_dtype(te_fp8_dtype):
    """Arch-correct torch FP8 dtype for a TE ``DType`` (kernel launch / ``.view``)."""
    return te_dtype_to_torch_dtype(tex.DType(int(te_fp8_dtype)))


def _make_blockwise_qtensor(
    x_fp8,
    scale_inv,
    *,
    orig_shape,
    orig_dtype,
    te_fp8_dtype,
    is_2D_scaled,
    pow2,
    columnwise_data=None,
    columnwise_scale_inv=None,
    vk_group_offs=None,
):
    """Wrap raw Triton blockwise-quant outputs in a ``Float8BlockwiseQTensor``.

    ``scale_inv`` (rowwise) must already be in the TE (ROCm) GEMM layout:
    ``[ceil(M/128), ceil(N/128)]`` for 2D (weight) scaling and ``[ceil(K/128), M]``
    for 1D (rowwise activation) scaling.

    The optional columnwise operand carries the **segment-padded** wgrad data
    ``[M_pad, N]`` and its 1D block scales ``[ceil(M_pad/128), N]`` -- note this is
    *not* the standard transposed ``[N, M]`` columnwise layout; the grouped
    variable-K wgrad kernel consumes it together with ``vk_group_offs`` (the padded
    per-segment offsets). Either operand may be omitted (``None``) when only the
    other is needed.
    """
    from ..tensor.float8_blockwise_tensor import Float8BlockwiseQTensor, Float8BlockQuantizer

    rowwise = x_fp8 is not None
    columnwise = columnwise_data is not None
    quantizer = Float8BlockQuantizer(
        fp8_dtype=te_fp8_dtype,
        rowwise=rowwise,
        columnwise=columnwise,
        block_scaling_dim=2 if is_2D_scaled else 1,
        force_pow_2_scales=pow2,
    )
    ref = x_fp8 if rowwise else columnwise_data
    return Float8BlockwiseQTensor(
        shape=tuple(orig_shape),
        dtype=orig_dtype,
        fp8_dtype=te_fp8_dtype,
        rowwise_data=x_fp8.view(torch.uint8) if rowwise else None,
        rowwise_scale_inv=scale_inv,
        columnwise_data=columnwise_data.view(torch.uint8) if columnwise else None,
        columnwise_scale_inv=columnwise_scale_inv,
        quantizer=quantizer,
        is_2D_scaled=is_2D_scaled,
        device=ref.device,
        vk_group_offs=vk_group_offs,
    )


def quantize_fp8_blockwise_act_qtensor(a, fp8_dtype, pow2: bool = False):
    """1x128 rowwise activation quantize returning a ``Float8BlockwiseQTensor``.

    ``fp8_dtype`` is a TE ``DType`` (the QTensor's native dtype); it is mapped to
    the arch-correct torch FP8 dtype only for the Triton launch.
    """
    x_fp8, scales = quantize_fp8_blockwise(
        a, _torch_fp8_dtype(fp8_dtype), axis=1, block_size=128, pow2=pow2
    )
    # Triton emits scales as [M, ceil(K/128)]; TE stores the 1D rowwise scale
    # transposed as [ceil(K/128), M] (the GEMM-ready layout).
    return _make_blockwise_qtensor(
        x_fp8,
        scales.t().contiguous(),
        orig_shape=a.shape,
        orig_dtype=a.dtype,
        te_fp8_dtype=fp8_dtype,
        is_2D_scaled=False,
        pow2=pow2,
    )


def wrap_fp8_blockwise_segment_m_qtensor(
    data, scale_inv, vk_group_offs, orig_dtype, fp8_dtype, pow2: bool = False
):
    """Wrap segment-padded columnwise FP8 fields as a columnwise-only QTensor.

    The segment-padded wgrad operand lives in the ``columnwise_data`` /
    ``columnwise_scale_inv`` slots (``[M_pad, N]`` data, ``[ceil(M_pad/128), N]``
    scales) with the padded per-segment offsets in ``_vk_group_offs``. The rowwise
    operand is left ``None``; only the variable-K wgrad kernel reads this tensor.
    """
    return _make_blockwise_qtensor(
        None,
        None,
        orig_shape=tuple(data.shape),
        orig_dtype=orig_dtype,
        te_fp8_dtype=fp8_dtype,
        is_2D_scaled=False,
        pow2=pow2,
        columnwise_data=data,
        columnwise_scale_inv=scale_inv,
        vk_group_offs=vk_group_offs,
    )


def quantize_fp8_blockwise_segment_m_operand(
    x, fp8_dtype, block_size, group_lens, group_offs, pow2: bool = False
):
    """Segment-padded columnwise quantize returning a columnwise-only QTensor.

    ``fp8_dtype`` is a TE ``DType``; the torch FP8 dtype is derived only for the
    Triton launch.
    """
    x_fp8, x_scales, _vk_lens, vk_offs = quantize_fp8_blockwise_segment_m(
        x, _torch_fp8_dtype(fp8_dtype), block_size, group_lens, group_offs, pow2=pow2
    )
    return wrap_fp8_blockwise_segment_m_qtensor(
        x_fp8, x_scales, vk_offs, x.dtype, fp8_dtype, pow2=pow2
    )


def quantize_fp8_blockwise_act_and_segment_m(
    a, fp8_dtype, block_size, group_lens, group_offs, pow2: bool = False
):
    """Fused rowwise activation + segment-padded columnwise quantize in one pass.

    Reads ``a`` ``[M, K]`` from HBM once and returns a single
    ``Float8BlockwiseQTensor`` carrying **both** operands the grouped
    forward/backward consume:

    * rowwise (1x128 along K) in ``rowwise_data`` with the scale stored transposed
      ``[ceil(K/128), M]`` (the forward/dgrad GEMM layout), identical to
      :func:`quantize_fp8_blockwise_act_qtensor`.
    * segment-padded columnwise (1x128 along M, each MoE group padded to
      ``block_size``) in ``columnwise_data`` + ``_vk_group_offs`` (the variable-K
      wgrad operand), identical to :func:`quantize_fp8_blockwise_segment_m_operand`.

    Both share the same ``pow2`` rounding and source ``a``; this replaces the two
    separate passes (rowwise + segment-padded colwise) with a single launch. Only
    worthwhile when the wgrad operand is needed (training).
    """
    assert a.is_contiguous() and a.dim() == 2, "Input must be 2D and contiguous"
    M, N = a.shape
    num_groups = group_lens.size(0)
    torch_fp8_dtype = _torch_fp8_dtype(fp8_dtype)
    fp8_max = torch.finfo(torch_fp8_dtype).max

    var_k_group_lens = ((group_lens + block_size - 1) // block_size) * block_size
    var_k_group_offs = torch.zeros(num_groups + 1, dtype=torch.int64, device=a.device)
    var_k_group_offs[1:] = torch.cumsum(var_k_group_lens, dim=0)

    x_fp8_row = torch.empty((M, N), dtype=torch_fp8_dtype, device=a.device)
    x_scales_row = torch.empty(
        (M, triton.cdiv(N, block_size)), dtype=torch.float32, device=a.device
    )

    m_padded_max = M + num_groups * block_size
    x_fp8_col = torch.zeros((m_padded_max, N), dtype=torch_fp8_dtype, device=a.device)
    x_scales_col = torch.zeros(
        (triton.cdiv(m_padded_max, block_size), N), dtype=torch.float32, device=a.device
    )

    grid = (triton.cdiv(m_padded_max, block_size), triton.cdiv(N, block_size))
    quant_fp8_blockwise_grouped_kernel[grid](
        a,
        x_fp8_row,
        x_scales_row,
        x_fp8_col,
        x_scales_col,
        group_offs,
        var_k_group_offs,
        N,
        num_groups,
        BLOCK_SIZE=block_size,
        FP8_MAX=fp8_max,
        ROUND_POW2=pow2,
        ROWWISE=True,
        COLUMNWISE=True,
    )

    # Triton emits rowwise scales as [M, ceil(K/128)]; TE stores the 1D rowwise
    # scale transposed as [ceil(K/128), M] (the GEMM-ready layout). The
    # segment-padded columnwise operand rides along in the columnwise slots.
    return _make_blockwise_qtensor(
        x_fp8_row,
        x_scales_row.t().contiguous(),
        orig_shape=a.shape,
        orig_dtype=a.dtype,
        te_fp8_dtype=fp8_dtype,
        is_2D_scaled=False,
        pow2=pow2,
        columnwise_data=x_fp8_col,
        columnwise_scale_inv=x_scales_col,
        vk_group_offs=var_k_group_offs,
    )


def quantize_fp8_blockwise_act_operands(
    a,
    fp8_dtype,
    block_size,
    group_lens,
    group_offs,
    *,
    rowwise,
    columnwise,
    pow2: bool = False,
):
    """Quantize a grouped activation into the operand(s) the caller needs.

    Dispatches over the (``rowwise``, ``columnwise``) request:

    * both -> :func:`quantize_fp8_blockwise_act_and_segment_m` (one fused HBM read).
    * rowwise only -> :func:`quantize_fp8_blockwise_act_qtensor` (forward/dgrad).
    * columnwise only -> :func:`quantize_fp8_blockwise_segment_m_operand` (wgrad).
    * neither -> ``None``.

    ``block_size`` / ``group_lens`` / ``group_offs`` are only consumed by the
    segment-padded columnwise path.
    """
    if rowwise and columnwise:
        return quantize_fp8_blockwise_act_and_segment_m(
            a, fp8_dtype, block_size, group_lens, group_offs, pow2=pow2
        )
    if rowwise:
        return quantize_fp8_blockwise_act_qtensor(a, fp8_dtype, pow2=pow2)
    if columnwise:
        return quantize_fp8_blockwise_segment_m_operand(
            a, fp8_dtype, block_size, group_lens, group_offs, pow2=pow2
        )
    return None


def wrap_fp8_blockwise_grouped_weight_qtensor(
    rowwise_data, rowwise_scale_inv, orig_dtype, fp8_dtype, pow2: bool = True
):
    """Wrap packed grouped-weight FP8 data + scales as a 2D ``[G*N, K]`` QTensor.

    Because ``N`` (out_features) is a multiple of 128, expert boundaries align
    with the 128x128 blocks, so a packed ``[G, N, K]`` weight is bit-identical to
    a single 2D ``[G*N, K]`` blockwise tensor with the scale flattened to
    ``[G*(N/128), K/128]`` -- and it dequantizes correctly. ``rowwise_data`` may
    be ``[G, N, K]`` or already 2D ``[G*N, K]`` (scales correspondingly 3D or 2D).
    """
    if rowwise_data.dim() == 3:
        gg, nn, kk = rowwise_data.shape
        rowwise_data = rowwise_data.reshape(gg * nn, kk)
        rowwise_scale_inv = rowwise_scale_inv.reshape(
            gg * rowwise_scale_inv.shape[1], rowwise_scale_inv.shape[2]
        )
    return _make_blockwise_qtensor(
        rowwise_data,
        rowwise_scale_inv,
        orig_shape=tuple(rowwise_data.shape),
        orig_dtype=orig_dtype,
        te_fp8_dtype=fp8_dtype,
        is_2D_scaled=True,
        pow2=pow2,
    )


def quantize_fp8_blockwise_grouped_weight_qtensor(w, fp8_dtype, pow2: bool = False):
    """128x128 grouped-weight quantize of ``[G, N, K]`` -> 2D ``[G*N, K]`` QTensor.

    ``N`` must be a multiple of 128 so expert boundaries align with 128-blocks
    (see :func:`wrap_fp8_blockwise_grouped_weight_qtensor`). One packed quantize
    launch over ``[G, N, K]``; no per-expert launches, no stack.
    """
    assert w.dim() == 3, "quantize_fp8_blockwise_grouped_weight_qtensor expects [G, N, K]"
    assert w.shape[1] % 128 == 0, "grouped weight N (out_features) must be a multiple of 128"
    x_fp8, scales = quantize_fp8_blockwise_weight(
        w, _torch_fp8_dtype(fp8_dtype), block_size=128, pow2=pow2
    )
    return wrap_fp8_blockwise_grouped_weight_qtensor(x_fp8, scales, w.dtype, fp8_dtype, pow2=pow2)
