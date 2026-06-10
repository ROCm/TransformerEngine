# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

import torch
import triton
import triton.language as tl
import warnings
import transformer_engine_torch as tex

from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer, Float8CurrentScalingQuantizer
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer
from transformer_engine.pytorch.triton_kernels.common import (
    te_dtype_to_torch_dtype,
    te_dtype_to_triton_dtype,
)
from ..quantized_tensor import Quantizer
from .utils import num_programs, block_size, use_blocked, make_ln_out, get_num_sms
from .common import get_fp8_max
from .rmsnorm import (
    _rmsnorm_fwd_triton,
    _rmsnorm_fwd_triton_impl,
    _rmsnorm_bwd_triton,
    _rmsnorm_bwd_triton_impl,
    _rmsnorm_bwd_dg_reduce_triton,
    _rmsnorm_bwd_dg_reduce_triton_impl,
)

# --------------------------------------------------------------------------- #
# External LDS-tiled byte transpose
#
# Produces the column-major fp8 transpose for the RMSNorm fwd path. The
# alternative -- having the main kernel emit `out_transpose_ptr + cols * stride
# + row_idx` strided stores -- is uncoalesced (1 byte/thread to a different
# cache line each) and bottlenecks every fp8_t shape, so RMSNorm always uses
# this kernel instead.
#
# This kernel reads a (BLOCK_M, BLOCK_N) tile coalesced from the row-major
# fp8 output, transposes it through LDS via `tl.trans`, and writes the
# (BLOCK_N, BLOCK_M) tile coalesced to the column-major transpose buffer.
#
# Operates on the raw uint8 storage so the fp8 dtype is irrelevant to
# correctness.
# --------------------------------------------------------------------------- #
@triton.jit
def _fp8_transpose_2d_impl(
    src_ptr,           # uint8 ptr, (n_rows, n_cols) row-major
    dst_ptr,           # uint8 ptr, (n_cols, n_rows) row-major
    n_rows, n_cols,
    src_stride,        # element stride of src row dim (== n_cols when contig)
    dst_stride,        # element stride of dst row dim (== n_rows when contig)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Coalesced read of (BLOCK_M, BLOCK_N) tile (innermost dim = cols).
    src_offs = rm[:, None] * src_stride + cn[None, :]
    src_mask = (rm[:, None] < n_rows) & (cn[None, :] < n_cols)
    tile = tl.load(src_ptr + src_offs, mask=src_mask, other=0)

    # LDS-staged transpose -> (BLOCK_N, BLOCK_M).
    tile_t = tl.trans(tile)

    # Coalesced write of (BLOCK_N, BLOCK_M) tile (innermost dim = rows).
    dst_offs = cn[:, None] * dst_stride + rm[None, :]
    dst_mask = (cn[:, None] < n_cols) & (rm[None, :] < n_rows)
    tl.store(dst_ptr + dst_offs, tile_t, mask=dst_mask)


def _get_fp8_transpose_configs():
    return [
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64},  num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64},  num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64},  num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64},  num_warps=8),
    ]


_fp8_transpose_2d_triton = triton.autotune(
    configs=_get_fp8_transpose_configs(),
    key=['n_rows', 'n_cols'],
    use_cuda_graph=True,
)(_fp8_transpose_2d_impl)

_fp8_transpose_kernels = {
    True: _fp8_transpose_2d_triton,
    False: _fp8_transpose_2d_impl,
}
from .layernorm import (
    _layernorm_fwd_triton,
    _layernorm_fwd_triton_impl,
    _layernorm_fwd_reduce_triton,
    _layernorm_bwd_dwdb_triton,
    _layernorm_bwd_dwdb_triton_v2,
    _layernorm_bwd_dx_fused_triton,
)

_norm_kernels={
    "rms":{
        True: _rmsnorm_fwd_triton,
        False: _rmsnorm_fwd_triton_impl,
    },
    "layer":{
        True: _layernorm_fwd_triton,
        False: _layernorm_fwd_triton_impl,
    }
}

_rmsnorm_bwd_kernels = {
    True: _rmsnorm_bwd_triton,
    False: _rmsnorm_bwd_triton_impl,
}

_rmsnorm_bwd_dg_reduce_kernels = {
    True: _rmsnorm_bwd_dg_reduce_triton,
    False: _rmsnorm_bwd_dg_reduce_triton_impl,
}


_TARGET_PRGMS_PER_CU = 8 
_MAX_PRGMS_PER_CU = 16 
_I32_OFFSET_LIMIT = 1 << 31 

_FWD_FLAT_GRID_MAX_H = 2048

_FWD_CAP_LADDER = ((128, 16), (256, 4), (512, 2))
_BWD_CAP_LADDER = ((128, 4), (512, 2))


def _rows_per_pid(rows, hidden, num_sms, cap_ladder):
    """Largest power-of-two rows-per-program for a narrow-H RMSNorm launch.

    Picks the biggest ``rpp`` that (a) does not exceed the hidden-size cap,
    (b) evenly divides ``rows`` -- so the kernel needs no row-tail mask -- and
    (c) still leaves at least ``_TARGET_PRGMS_PER_CU * num_sms`` programs. If
    the program-count target cannot be met we still pack a little (4 or 2) to
    amortise the gamma load, provided divisibility holds.
    """
    target = _TARGET_PRGMS_PER_CU * num_sms
    cap = 1
    for threshold, c in cap_ladder:
        if hidden <= threshold:
            cap = c
            break
    rpp = 1
    while rpp * 2 <= cap and rows % (rpp * 2) == 0 and rows // (rpp * 2) >= target:
        rpp *= 2
    if rpp == 1:
        if hidden <= 128 and rows % 4 == 0:
            rpp = 4
        elif hidden <= 512 and rows % 2 == 0:
            rpp = 2
    return rpp


# triton drop-in replacement for transformer_engine::pytorch::rmsnorm_fwd
def te_rmsnorm_fwd_triton(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    ln_out: torch.Tensor,
    quantizer: Quantizer,
    otype: tex.DType,
    sm_margin: int,
    zero_centered_gamma: bool,
    autotune: bool = True,
):
    return _te_norm_fwd_triton(
        kernel='rms',
        input_tensor=input,
        weight=weight,
        bias=None,
        eps=eps,
        ln_out=ln_out,
        quantizer=quantizer,
        otype=otype,
        sm_margin=sm_margin,
        zero_centered_gamma=zero_centered_gamma,
        autotune=autotune,
    )

# triton drop-in replacement for transformer_engine::pytorch::layernorm_fwd
def te_layernorm_fwd_triton(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    ln_out: torch.Tensor,
    quantizer: Quantizer,
    otype: tex.DType,
    sm_margin: int,
    zero_centered_gamma: bool,
    autotune: bool = True,
):
    return _te_norm_fwd_triton(
        kernel='layer',
        input_tensor=input,
        weight=weight,
        bias=bias,
        eps=eps,
        ln_out=ln_out,
        quantizer=quantizer,
        otype=otype,
        sm_margin=sm_margin,
        zero_centered_gamma=zero_centered_gamma,
        autotune=autotune,
    )

def _te_norm_fwd_triton(
    kernel: str,
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    ln_out: torch.Tensor,
    quantizer: Quantizer,
    otype: tex.DType,
    sm_margin: int,
    zero_centered_gamma: bool,
    autotune: bool = True,
):
    if kernel not in {'rms', 'layer'}:
        raise ValueError(f"Expected `kernel` in ('rms', 'layer') but got {kernel=} instead.")
    if eps < 0:
        raise ValueError(f"`eps` must be non-negative, but a value of {eps} was passed")
    if len(input_tensor.shape) != 2:
        raise ValueError(
            f"The input must be a 2-dimensional matrix, but an input with {input_tensor.ndim} was passed.")

    device = input_tensor.device
    N, H = input_tensor.shape
    if weight.shape[0] != H:
        raise ValueError(
            f"The shape of `weight` must be feature-aligned, "
            f"but {weight.shape[0]=} while {input_tensor.shape[1]=}"
        )
    IS_FP8 = isinstance(quantizer, Float8Quantizer)
    IS_MXFP8 = isinstance(quantizer, MXFP8Quantizer)
    IS_FP8_CURRENT_SCALING = isinstance(quantizer, Float8CurrentScalingQuantizer)
    BLOCK_SIZE = block_size(input_tensor, norm=kernel)
    USE_BLOCKED = use_blocked(input_tensor)

    # Row-packing (RMSNorm, non-blocked, non-FP8 only)
    ROWS_PER_PID_FWD = 1
    if kernel == 'rms' and not USE_BLOCKED and not IS_FP8:
        ROWS_PER_PID_FWD = _rows_per_pid(N, H, get_num_sms(sm_margin), _FWD_CAP_LADDER)

    # HOIST_GAMMA: load gamma once per program when the program processes more than one row
    HOIST_GAMMA_FWD = True
    if kernel == 'layer':
        # LayerNorm requires one program per row on both paths.
        NUM_PRGMS = N
    elif kernel == 'rms' and not USE_BLOCKED:
        use_flat = (not IS_FP8) and (H <= _FWD_FLAT_GRID_MAX_H)
        if use_flat:
            NUM_PRGMS = N // ROWS_PER_PID_FWD
            HOIST_GAMMA_FWD = ROWS_PER_PID_FWD > 1
        else:
            NUM_PRGMS = num_programs(input_tensor, sm_margin)
            HOIST_GAMMA_FWD = True  # persistent loop reuses gamma across rows
    else:
        NUM_PRGMS = num_programs(input_tensor, sm_margin)
    MAKE_TRANSPOSE = False
    APPLY_ATOMIC = N < 512 or kernel == 'rms'
    ATOMIC_REDUCTION_BLOCK_SIZE=256

    mu = torch.empty((N,), dtype=torch.float32, device=device) if kernel == 'layer' else None
    rsigma = torch.empty((N,), dtype=torch.float32, device=device)
    torch_out_dtype = (
        otype if isinstance(otype, torch.dtype)
        else te_dtype_to_torch_dtype(otype)
    )
    out = make_ln_out(
        ln_out,
        quantizer=quantizer,
        input_shape=input_tensor.shape,
        out_dtype=torch_out_dtype
    )
    amax = None
    tl_dtype = None
    scale_inv_ptr = None
    q_scale = None
    out_ptr = out
    out_transpose_ptr = None
    out_transpose_stride = None
    FP8_MAX = None
    use_external_transpose = False
    if IS_FP8:
        MAKE_TRANSPOSE = quantizer.columnwise_usage
        amax = (
            quantizer.amax if APPLY_ATOMIC else
            torch.empty((NUM_PRGMS,), dtype=torch.float32, device=device)
        )
        tl_dtype = te_dtype_to_triton_dtype(quantizer.dtype)
        scale_inv_ptr = out._scale_inv
        q_scale = quantizer.scale
        out_ptr = triton.reinterpret(out._data, tl_dtype)
        FP8_MAX = get_fp8_max(quantizer.dtype)
        if MAKE_TRANSPOSE:
            if out._transpose_invalid:
                out._transpose = torch.empty(
                    (out._data.shape[1], out._data.shape[0]),
                    dtype=out._data.dtype, device=device
                )
                out._transpose_invalid = False
            if kernel == 'rms':
                # RMSNorm always uses the external LDS-tiled transpose kernel.
                use_external_transpose = True
            else:
                # LayerNorm emits the transpose via in-kernel strided stores.
                out_transpose_ptr = triton.reinterpret(out._transpose, tl_dtype)
                out_transpose_stride = out._transpose.stride(0)

    grid_fwd = lambda meta: (NUM_PRGMS,)
    kernel_func = _norm_kernels[kernel][autotune]
    input_row_stride = input_tensor.stride(0)
    output_row_stride=out_ptr.stride(0)
    kwargs = dict(
        input_ptr=input_tensor,
        output_ptr=out_ptr,
        g_ptr=weight,
        rsigma_ptr=rsigma,
        input_row_stride=input_row_stride,
        output_row_stride=output_row_stride,
        n_rows=N, n_cols=H,
        epsilon=eps,
        q_amax_ptr=amax,
        q_scale_ptr=q_scale,
        scale_inv_ptr=scale_inv_ptr,
        ZERO_CENTERED_GAMMA=zero_centered_gamma,
        BLOCK_SIZE=BLOCK_SIZE,
        IS_FP8=IS_FP8,
        FP8_MAX=FP8_MAX,
    )
    if kernel == 'layer':
        kwargs["APPLY_ATOMIC"]=APPLY_ATOMIC
        kwargs["PERSISTENT"]=False # TODO: Improve persistent algo performance
        kwargs["b_ptr"]=bias
        kwargs["mean_ptr"]=mu
        # LayerNorm emits the column-major fp8 copy via in-kernel strided stores.
        kwargs["out_transpose_ptr"]=out_transpose_ptr
        kwargs["out_transpose_stride"]=out_transpose_stride
        kwargs["MAKE_TRANSPOSE"]=MAKE_TRANSPOSE
    elif kernel == "rms":
        kwargs["USE_BLOCKED"]=USE_BLOCKED
        kwargs["NUM_PRGMS"]=NUM_PRGMS
        kwargs["INPUT_ALIGNED_16"]=(
            input_tensor.data_ptr() % 16 == 0 and
            input_row_stride * getattr(input_tensor.dtype, 'itemsize', 1) % 16 == 0
        )
        kwargs["OUTPUT_ALIGNED_16"]=(
            out_ptr.data_ptr() % 16 == 0 and
            output_row_stride * getattr(out_ptr.dtype, 'itemsize', 1) % 16 == 0
        )
        kwargs["ROWS_PER_PID"] = ROWS_PER_PID_FWD
        kwargs["HOIST_GAMMA"] = HOIST_GAMMA_FWD
        max_off = max(input_row_stride, output_row_stride) * N + H
        kwargs["NEEDS_I64_OFFSETS"] = max_off >= _I32_OFFSET_LIMIT

    kernel_func[grid_fwd](**kwargs)

    if use_external_transpose:
        # out._data: (N rows, H cols) row-major uint8; out._transpose: (H, N).
        transpose_kernel = _fp8_transpose_kernels[autotune]
        if autotune:
            grid_t = lambda meta: (
                triton.cdiv(N, meta['BLOCK_M']),
                triton.cdiv(H, meta['BLOCK_N']),
            )
            transpose_kernel[grid_t](
                out._data, out._transpose,
                N, H,
                out._data.stride(0), out._transpose.stride(0),
            )
        else:
            BLOCK_M, BLOCK_N = 64, 64
            grid_t = (triton.cdiv(N, BLOCK_M), triton.cdiv(H, BLOCK_N))
            transpose_kernel[grid_t](
                out._data, out._transpose,
                N, H,
                out._data.stride(0), out._transpose.stride(0),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            )

    # Reduce and find amax if "not APPLY_ATOMIC" is True for layernorm.
    if IS_FP8 and not APPLY_ATOMIC:
        _layernorm_fwd_reduce_triton[(triton.cdiv(N, ATOMIC_REDUCTION_BLOCK_SIZE),)](
            amax,
            quantizer.amax,
            N, ATOMIC_REDUCTION_BLOCK_SIZE,
        )
    elif IS_MXFP8 or IS_FP8_CURRENT_SCALING or isinstance(quantizer, NVFP4Quantizer):
        _out = quantizer.make_empty(
            input_tensor.shape,
            dtype=te_dtype_to_torch_dtype(otype),
            device=input_tensor.device
        )
        out = quantizer.quantize(out, out=_out)
    return out, mu, rsigma


# triton drop-in replacement for transformer_engine::pytorch::rmsnorm_bwd
def te_rmsnorm_bwd_triton(dz, x, rsigma, gamma, sm_margin, zero_centered_gamma, autotune: bool = True):
    # may take non-contiguous inputs
    dz_ = dz.contiguous()
    x_ = x.contiguous()
    rsigma_ = rsigma.contiguous()
    gamma_ = gamma.contiguous()

    dx = torch.empty_like(x_)
    dgamma = torch.empty_like(gamma_)

    M, N = x_.shape
    blk_size = block_size(x_)
    USE_BLOCKED = use_blocked(x_)

    # Multi-row-per-program for narrow N
    ROWS_PER_PID_BWD = 1
    if not USE_BLOCKED:
        ROWS_PER_PID_BWD = _rows_per_pid(M, N, get_num_sms(sm_margin), _BWD_CAP_LADDER)

    base_prgms = num_programs(x_, sm_margin)
    if USE_BLOCKED:
        NUM_PRGMS = base_prgms
        dg_tmp_rows = M
    elif ROWS_PER_PID_BWD > 1:
        NUM_PRGMS = min(M // ROWS_PER_PID_BWD, _MAX_PRGMS_PER_CU * base_prgms)
        dg_tmp_rows = NUM_PRGMS
    else:
        NUM_PRGMS = base_prgms
        dg_tmp_rows = NUM_PRGMS
    need_reduction = NUM_PRGMS > 1
    dg_tmp = torch.empty(dg_tmp_rows, N, device=x.device, dtype=torch.float32, requires_grad=False) if need_reduction else None

    input_aligned_16 = (x_.data_ptr() % 16 == 0) and (x_.stride(0) * x_.dtype.itemsize % 16 == 0)
    grad_output_aligned_16 = (dz_.data_ptr() % 16 == 0) and (dz_.stride(0) * dz_.dtype.itemsize % 16 == 0)
    dx_aligned_16 = (dx.data_ptr() % 16 == 0) and (dx.stride(0) * dx.dtype.itemsize % 16 == 0)
    dg_target = dg_tmp if need_reduction else dgamma
    dg_aligned_16 = (dg_target.data_ptr() % 16 == 0) and (dg_target.stride(0) * dg_target.dtype.itemsize % 16 == 0)

    grid_bwd = lambda meta: (NUM_PRGMS, )
    # See forward: gate i64 offsets on actual overflow to stay on the fast i32 path.
    needs_i64_offsets_bwd = max(x_.stride(0), dz_.stride(0), dx.stride(0)) * M + N >= _I32_OFFSET_LIMIT
    bwd_kernel = _rmsnorm_bwd_kernels[autotune]
    bwd_kwargs = dict(
        n_rows=M, n_cols=N,
        ZERO_CENTERED_GAMMA=zero_centered_gamma,
        BLOCK_SIZE=blk_size,
        USE_BLOCKED=USE_BLOCKED, NUM_PRGMS=NUM_PRGMS,
        INPUT_ALIGNED_16=input_aligned_16,
        GRAD_OUTPUT_ALIGNED_16=grad_output_aligned_16,
        DX_ALIGNED_16=dx_aligned_16,
        DG_ALIGNED_16=dg_aligned_16,
        ROWS_PER_PID=ROWS_PER_PID_BWD,
        NEEDS_I64_OFFSETS=needs_i64_offsets_bwd,
    )
    if not autotune:
        bwd_kwargs["num_warps"] = 8
    bwd_kernel[grid_bwd](
        dz_, x_, gamma_, rsigma_, dx, dg_tmp if need_reduction else dgamma,
        x_.stride(0), dz_.stride(0),
        **bwd_kwargs,
    )

    if need_reduction:
        reduce_kernel = _rmsnorm_bwd_dg_reduce_kernels[autotune]
        if autotune:
            grid_reduce = lambda meta: [triton.cdiv(N, meta['BLOCK_SIZE_N'])]
            reduce_kernel[grid_reduce](
                dg_tmp, dgamma, dg_tmp.stride(0), dg_tmp.shape[0], dg_tmp.shape[1],
            )
        else:
            BLOCK_SIZE_M, BLOCK_SIZE_N = 128, 64
            grid_reduce = (triton.cdiv(N, BLOCK_SIZE_N),)
            reduce_kernel[grid_reduce](
                dg_tmp, dgamma, dg_tmp.stride(0), dg_tmp.shape[0], dg_tmp.shape[1],
                BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N,
            )

    return dx, dgamma

# drop in replacement for transformer_engine::pytorch::layernorm_bwd
# TODO: Add support for `sm_margin > 0`.
def te_layernorm_bwd_triton(
    dz: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    rsigma: torch.Tensor,
    gamma: torch.Tensor,
    sm_margin: int,
    zero_centered_gamma: bool
):
    if sm_margin is not None and sm_margin > 0:
        warnings.warn(
            '"sm_margin" is not supported in the Triton based backward layer-norm kernel. '
            + f"sm_margin={sm_margin} will be ignored."
        )
    M, N = x.shape
    # calculate dw and db separately when M is small
    IGNORE_DW_DB_IN_FUSED = M <= 512
    tile_num = max(min(256, M // 4), 1)
    if M <= 512 and M * N < 64 * 1024 * 1024:
        tile_num = M
    elif M >= 8192:
        tile_num = 2048
    max_fused_size = 32768 // x.element_size()
    next_power = triton.next_power_of_2(N)
    BLOCK_SIZE = min(max_fused_size, next_power)
    # For cases with small M and large N, decrease block size to help with occupancy and register spill
    if tile_num == M:
        if tile_num > 256:
            BLOCK_SIZE = min(BLOCK_SIZE, 2048)
        else:
            BLOCK_SIZE = min(BLOCK_SIZE, 4096)
    USE_BLOCKED = N > BLOCK_SIZE
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)

    dx = torch.empty_like(x)
    if not IGNORE_DW_DB_IN_FUSED:
        _dgamma = torch.zeros((tile_num, N), dtype=torch.float32, device=gamma.device)
        _dbeta = torch.zeros((tile_num, N), dtype=torch.float32, device=gamma.device)
    else:
        _dgamma = None
        _dbeta = None
    dgamma = torch.zeros((N,), dtype=gamma.dtype, device=gamma.device)
    dbeta = torch.zeros((N,), dtype=gamma.dtype, device=gamma.device)
    grid_bwd = (tile_num,)
    _layernorm_bwd_dx_fused_triton[grid_bwd](
        dx,
        dz,
        _dgamma,
        _dbeta,
        x,
        gamma,
        mu,
        rsigma,
        x.stride(0),
        N,
        ZERO_CENTERED_GAMMA=zero_centered_gamma,
        NUM_ROWS=M,
        BLOCK_SIZE_N=BLOCK_SIZE,
        USE_BLOCKED=USE_BLOCKED,
        num_warps=num_warps,
        IGNORE_DW_DB=IGNORE_DW_DB_IN_FUSED,
    )
    grid_reduce = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE_N"]),)
    if not IGNORE_DW_DB_IN_FUSED:
        dwdb_block_n = max(16, N // 256)
        dwdb_block_n = triton.next_power_of_2(dwdb_block_n)
        dwdb_block_m = (64 * 128) // dwdb_block_n
        dwdb_block_m = min(triton.next_power_of_2(tile_num), dwdb_block_m)
        _layernorm_bwd_dwdb_triton[grid_reduce](
            _dgamma,
            _dbeta,
            dgamma,
            dbeta,
            min(tile_num, M),
            N,
            BLOCK_SIZE_M=dwdb_block_m,
            BLOCK_SIZE_N=dwdb_block_n,
        )
    else:
        dwdb_block_n = max(16, N // 256)
        dwdb_block_n = triton.next_power_of_2(dwdb_block_n)
        dwdb_block_m = (64 * 128) // dwdb_block_n
        dwdb_block_m = min(triton.next_power_of_2(M), dwdb_block_m)
        _layernorm_bwd_dwdb_triton_v2[grid_reduce](
            x,
            dz,
            mu,
            rsigma,
            x.stride(0),
            dgamma,
            dbeta,
            M,
            N,
            BLOCK_SIZE_M=dwdb_block_m,
            BLOCK_SIZE_N=dwdb_block_n,
        )

    return dx, dgamma, dbeta
