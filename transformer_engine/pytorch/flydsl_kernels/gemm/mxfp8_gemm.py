# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""FlyDSL MXFP8 TN/NN/NT 4-wave GEMM implementation.

All supported MXFP8 layouts share one source-level kernel generator while
remaining separate compile-time specializations:

    TN: A [M,K] normal read,    B [N,K] normal read
    NN: A [M,K] normal read,    B [K,N] transpose read
    NT: A [K,M] transpose read, B [K,N] transpose read

The layout is a Python-only cache key. It is never passed as a runtime kernel
argument. Global addressing, LDS fragment reads, scheduler directives, and
scale-source orientation are selected while building each specialized kernel,
so generated TN/NN/NT kernels contain no runtime layout branches.

All operand payloads use direct ``BufferCopyLDS128b`` global-to-LDS staging.
Transpose variants differ only in K-major global addressing and
``ds_read_b64_tr_b8`` LDS-to-register fragment reconstruction.
"""

import functools
import os

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

# Transformer Engine-local FlyDSL utilities.
from .exceptions import FlyDSLUnsupportedError
from .fp8_gemm_utils import (
    G2SLoader,
    S2RLoader,
    compute_global_swizzle,
    make_fp8_buffer_tensor,
    pack_i32x4_i32x8,
    swizzle_128,
    xcd_swizzle,
    barrier
)


_BLOCK_M = 256
_BLOCK_N = 256
_BLOCK_K = 128

# Public metadata consumed by wrappers — keep.
BLOCK_M = _BLOCK_M
BLOCK_N = _BLOCK_N
BLOCK_K = _BLOCK_K
SCALE_GROUP_SIZE = 32


def _debug_enabled() -> bool:
    value = os.getenv("DEBUG_FLYDSL_MXFP8_GEMM", "")
    return value.lower() not in ("", "0", "false", "no", "off")


def _debug(message: str) -> None:
    if _debug_enabled():
        print(f"[DEBUG_FLYDSL_MXFP8_GEMM] {message}")


_SCALE_PACK_THREADS = 256


def _compile_mx32_scale_pack_kernel(
    dim: int,
    qk: int,
    source_colwise: bool,
    stride0: int,
    stride1: int,
):
    """Build one fused scale packing kernel.

    One GPU thread produces one final ``uint32`` word in the GEMM-consumed
    ``[K/128, dim]`` layout.  There is no intermediate ``scale_iter`` tensor
    and no eager PyTorch shift/index/OR kernels.
    """
    if dim % 64 != 0:
        raise FlyDSLUnsupportedError(
            f"Scale outer dimension={dim} must be a multiple of 64"
        )
    if qk % 4 != 0:
        raise FlyDSLUnsupportedError(
            f"Scale K/32 dimension={qk} must be divisible by 4"
        )

    k128_tiles = qk // 4
    total_words = k128_tiles * dim
    if total_words % _SCALE_PACK_THREADS != 0:
        raise FlyDSLUnsupportedError(
            f"Packed scale words={total_words} must be divisible by "
            f"{_SCALE_PACK_THREADS}"
        )

    # Select source addressing before FlyDSL captures the kernel.  The emitted
    # rowwise and columnwise binaries contain no runtime orientation branch.
    if source_colwise:
        def _source_offset(source_k32, source_row):
            # Logical source is [K/32, dim], but the underlying TE tensor may
            # be a non-contiguous view. Strides are in uint8 elements.
            return (
                source_k32 * fx.Index(stride0)
                + source_row * fx.Index(stride1)
            )
    else:
        def _source_offset(source_k32, source_row):
            # Logical source is [dim, K/32], with arbitrary positive strides.
            return (
                source_row * fx.Index(stride0)
                + source_k32 * fx.Index(stride1)
            )

    @flyc.kernel(known_block_size=[_SCALE_PACK_THREADS, 1, 1])
    def kernel_pack_mx32_scales(src: fx.Tensor, dst: fx.Tensor):
        # Address both buffers as flat 1-D arrays so a linear slice coordinate
        # maps to base+off, matching the legacy buffer_load semantics. Building
        # buffer tensors directly from the 2-D src/dst would make logical_divide
        # walk the layout column-major and corrupt every strided access.
        src_flat = fx.rocdl.make_buffer_tensor(
            fx.Tensor(fx.make_view(fx.get_iter(src), fx.make_layout(dim * qk, 1))),
            max_size=True,
        )
        dst_flat = fx.rocdl.make_buffer_tensor(
            fx.Tensor(fx.make_view(fx.get_iter(dst), fx.make_layout(total_words, 1))),
            max_size=True,
        )
        src_div = fx.logical_divide(src_flat, fx.make_layout(1, 1))
        dst_div = fx.logical_divide(dst_flat, fx.make_layout(1, 1))
        scale_load_atom = fx.make_copy_atom(fx.rocdl.BufferCopy8b(), fx.Uint8)
        scale_store_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)

        linear = (
            fx.Index(fx.block_idx.x) * fx.Index(_SCALE_PACK_THREADS)
            + fx.Index(gpu.thread_id("x"))
        )
        k128 = linear // fx.Index(dim)
        dst_row = linear % fx.Index(dim)

        row_within_16 = dst_row % fx.Index(16)
        k_subgroup = (dst_row // fx.Index(16)) % fx.Index(4)
        tile = dst_row // fx.Index(64)
        source_k32 = k128 * fx.Index(4) + k_subgroup

        def load_scale_byte(group):
            source_row = (
                tile * fx.Index(64)
                + fx.Index(group * 16)
                + row_within_16
            )
            off = _source_offset(source_k32, source_row)
            reg = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Uint8)
            fx.copy(scale_load_atom, fx.slice(src_div, (None, fx.Int32(off))), reg)
            value_u8 = fx.memref_load_vec(reg)[0]
            # The byte load widens via sext, so mask to the low 8 bits to
            # preserve the raw E8M0 byte for scales >= 0x80 before packing.
            return fx.Int32(value_u8) & fx.Int32(0xFF)

        b0 = load_scale_byte(0)
        b1 = load_scale_byte(1)
        b2 = load_scale_byte(2)
        b3 = load_scale_byte(3)
        packed = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
        reg_i32 = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Int32)
        fx.memref_store_vec(Vec.filled(1, packed, fx.Int32), reg_i32)
        fx.copy(scale_store_atom, reg_i32, fx.slice(dst_div, (None, fx.Int32(linear))))


    @flyc.jit
    def launch_pack_mx32_scales(
        src: fx.Tensor,
        dst: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        kernel_pack_mx32_scales(src, dst).launch(
            grid=(total_words // _SCALE_PACK_THREADS, 1, 1),
            block=(_SCALE_PACK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_pack_mx32_scales


@functools.lru_cache(maxsize=None)
def _cached_mx32_scale_pack_launch(
    dim: int,
    qk: int,
    source_colwise: bool,
    stride0: int,
    stride1: int,
):
    """Cache orientation-and-stride-specialized fused pack binaries."""
    return _compile_mx32_scale_pack_kernel(
        dim, qk, source_colwise, stride0, stride1
    )


def pack_mx32_scales_for_hk(
    scales_u8: torch.Tensor,
    *,
    source_colwise: bool = False,
    stream=None,
) -> torch.Tensor:
    """Launch one fused GPU kernel producing HK MFMA-ready scale words.

    Input contracts:
      * rowwise:    ``[dim, K/32]``
      * columnwise: ``[K/32, dim]``

    Output contract:
      * ``[K/128, dim]`` ``torch.int32``
    """
    if scales_u8.dtype != torch.uint8:
        raise TypeError(
            f"MXFP8 scales must be torch.uint8 E8M0 bytes, got "
            f"{scales_u8.dtype}"
        )
    if scales_u8.ndim != 2:
        raise ValueError(
            f"MXFP8 scales must be rank 2, got {tuple(scales_u8.shape)}"
        )
    if not scales_u8.is_cuda:
        raise ValueError("MXFP8 scale packing requires a CUDA/ROCm tensor")
    if any(stride <= 0 for stride in scales_u8.stride()):
        raise ValueError(
            f"MXFP8 scale packing requires positive strides, got "
            f"{scales_u8.stride()}"
        )

    if source_colwise:
        qk, dim = scales_u8.shape
    else:
        dim, qk = scales_u8.shape

    if qk % 4 != 0:
        raise ValueError(
            f"Scale K/32 dimension={qk} must be divisible by 4"
        )
    if dim % 64 != 0:
        raise ValueError(
            f"Scale outer dimension={dim} must be a multiple of 64"
        )

    packed = torch.empty(
        (qk // 4, dim),
        dtype=torch.int32,
        device=scales_u8.device,
    )
    if stream is None:
        stream = torch.cuda.current_stream(scales_u8.device)

    stride0, stride1 = (int(x) for x in scales_u8.stride())
    _cached_mx32_scale_pack_launch(
        dim,
        qk,
        bool(source_colwise),
        stride0,
        stride1,
    )(
        scales_u8,
        packed,
        stream=stream,
    )
    return packed


def _compile_kernel(
    K: int,
    a_fp8_dtype: torch.dtype,
    b_fp8_dtype: torch.dtype,
    output_dtype: torch.dtype,
    layout: str,
):
    """Build one compile-time-specialized TN, NN, or NT kernel.

    ``layout`` is a Python string consumed while constructing the FlyDSL IR.
    It is not a runtime kernel argument. Each cache entry therefore contains
    only the addressing, LDS reads, and scheduler directives for that layout.
    """
    if layout not in ("TN", "NN", "NT"):
        raise ValueError(f"Unsupported MXFP8 kernel layout: {layout}")

    a_transpose_read = layout == "NT"
    b_transpose_read = layout in ("NN", "NT")

    BLOCK_M, BLOCK_N, BLOCK_K = _BLOCK_M, _BLOCK_N, _BLOCK_K

    fp8_input_types = {
        torch.float8_e4m3fn: (fx.Float8E4M3FN, 0),
        torch.float8_e5m2: (fx.Float8E5M2, 1),
    }
    try:
        a_fx_dtype, a_matrix_format = fp8_input_types[a_fp8_dtype]
        b_fx_dtype, b_matrix_format = fp8_input_types[b_fp8_dtype]
    except KeyError as exc:
        raise TypeError(
            "FlyDSL MXFP8 input dtype must be torch.float8_e4m3fn or "
            f"torch.float8_e5m2, got A={a_fp8_dtype}, B={b_fp8_dtype}"
        ) from exc

    if output_dtype == torch.float16:
        output_element_bytes = 2
        output_fx_dtype = fx.Float16
    elif output_dtype == torch.bfloat16:
        output_element_bytes = 2
        output_fx_dtype = fx.BFloat16
    elif output_dtype == torch.float32:
        output_element_bytes = 4
        output_fx_dtype = fx.Float32
    else:
        raise TypeError(
            "FlyDSL MXFP8 supports only float16, bfloat16, and float32 "
            f"outputs, got {output_dtype}"
        )

    NUM_THREADS = 256
    WARP_SIZE = 64

    SUBTILE_M = 64
    SUBTILE_N = 64

    MFMA_M = 16
    MFMA_N = 16

    SUBTILES_PER_WAVE = 4
    MFMA_M_PER_SUBTILE = SUBTILE_M // MFMA_M
    MFMA_N_PER_SUBTILE = SUBTILE_N // MFMA_N
    ACCS_PER_WAVE = SUBTILES_PER_WAVE * MFMA_M_PER_SUBTILE * MFMA_N_PER_SUBTILE

    ELEM_BYTES = 1
    VEC_BYTES = 16

    LDS_ELEMS_A = BLOCK_M * BLOCK_K
    LDS_ELEMS_B = BLOCK_N * BLOCK_K
    LDS_BYTES_A = LDS_ELEMS_A * ELEM_BYTES
    LDS_BYTES_B = LDS_ELEMS_B * ELEM_BYTES

    LOAD_PASSES_A = LDS_BYTES_A // (NUM_THREADS * VEC_BYTES)
    LOAD_PASSES_B = LDS_BYTES_B // (NUM_THREADS * VEC_BYTES)
    LOAD_PASSES_A_SUBTILE = LOAD_PASSES_A // 2
    LOAD_PASSES_B_SUBTILE = LOAD_PASSES_B // 2
    LOAD_PASSES_SCALES = 16

    assert K % BLOCK_K == 0, f"K must be a multiple of {BLOCK_K}, got {K}"
    NUM_K_TILES = K // BLOCK_K
    assert NUM_K_TILES >= 4, f"K={K} gives {NUM_K_TILES} K128 tiles; the two-page pipeline needs at least 4"

    LDS_ELEMS_HALF = (BLOCK_M // 2) * BLOCK_K
    LOAD_PASSES_HALF = LDS_ELEMS_HALF // (NUM_THREADS * VEC_BYTES)
    assert LOAD_PASSES_HALF == LOAD_PASSES_A_SUBTILE == LOAD_PASSES_B_SUBTILE

    # Resolve every layout-dependent choice before FlyDSL captures kernel_gemm.
    # These are ordinary Python callables/constants, so each cached layout emits
    # only its selected addressing, fragment-read, and scheduler path.
    Q0_SCHED_DSRD = 2 if a_transpose_read else 1
    PREFETCH_SCHED_DSRD = 4 if a_transpose_read else 2

    if a_transpose_read:
        def _a_leading_dim(c_m):
            return c_m

        def _a_global_base(k_base, subtile, c_m, bx_m_idx):
            return (
                k_base * fx.Index(c_m)
                + bx_m_idx
                + fx.Index(subtile * (BLOCK_M // 2))
            )

        def _load_a_half(
            load_transposed_frag_half,
            load_frag_half_at_byte_base,
            lds_a,
            sm,
            mi,
            half,
            reg_subtile_m_idx0,
            lane_mod_16,
        ):
            subtile_m_idx = reg_subtile_m_idx0 + fx.Index(sm * 2)
            local_m_tile = (
                subtile_m_idx * fx.Index(SUBTILE_M)
                + fx.Index(mi * MFMA_M)
                - fx.Index(sm * (BLOCK_M // 2))
            )
            return load_transposed_frag_half(
                lds_a[sm],
                local_m_tile,
                half,
            )
    else:
        def _a_leading_dim(c_m):
            del c_m
            return K

        def _a_global_base(k_base, subtile, c_m, bx_m_idx):
            del c_m
            return (
                (bx_m_idx + fx.Index(subtile * (BLOCK_M // 2)))
                * fx.Index(K)
                + k_base
            )

        def _load_a_half(
            load_transposed_frag_half,
            load_frag_half_at_byte_base,
            lds_a,
            sm,
            mi,
            half,
            reg_subtile_m_idx0,
            lane_mod_16,
        ):
            del load_transposed_frag_half
            subtile_m_idx = reg_subtile_m_idx0 + fx.Index(sm * 2)
            a_row_addr = (
                subtile_m_idx * fx.Index(SUBTILE_M)
                + fx.Index(mi * MFMA_M)
                + lane_mod_16
            )
            half_row = a_row_addr - fx.Index(sm * (BLOCK_M // 2))
            row_byte_base = half_row * fx.Index(BLOCK_K)
            return load_frag_half_at_byte_base(
                lds_a[sm],
                row_byte_base,
                half,
            )

    if b_transpose_read:
        def _b_leading_dim(c_n):
            return c_n

        def _b_global_base(k_base, subtile, c_n, by_n_idx):
            return (
                k_base * fx.Index(c_n)
                + by_n_idx
                + fx.Index(subtile * (BLOCK_N // 2))
            )

        def _load_b_ni(
            load_transposed_frag,
            load_normal_b_frag,
            lds_b,
            sn,
            ni,
            reg_subtile_n_idx0,
            lane_mod_16,
        ):
            del load_normal_b_frag, lane_mod_16
            subtile_n_idx = reg_subtile_n_idx0 + fx.Index(sn * 2)
            local_n_tile = (
                subtile_n_idx * fx.Index(SUBTILE_N)
                + fx.Index(ni * MFMA_N)
                - fx.Index(sn * (BLOCK_N // 2))
            )
            return load_transposed_frag(lds_b[sn], local_n_tile)
    else:
        def _b_leading_dim(c_n):
            del c_n
            return K

        def _b_global_base(k_base, subtile, c_n, by_n_idx):
            del c_n
            return (
                (by_n_idx + fx.Index(subtile * (BLOCK_N // 2)))
                * fx.Index(K)
                + k_base
            )

        def _load_b_ni(
            load_transposed_frag,
            load_normal_b_frag,
            lds_b,
            sn,
            ni,
            reg_subtile_n_idx0,
            lane_mod_16,
        ):
            del load_transposed_frag
            subtile_n_idx = reg_subtile_n_idx0 + fx.Index(sn * 2)
            b_row_addr = (
                subtile_n_idx * fx.Index(SUBTILE_N)
                + fx.Index(ni * MFMA_N)
                + lane_mod_16
            )
            return load_normal_b_frag(lds_b, b_row_addr, sn)

    if (not a_transpose_read) or (not b_transpose_read):
        def _normal_read_columns(lane_div_16, lane_mod_16):
            reg_k_col0 = lane_div_16 * 16
            reg_k_col1 = 64 + lane_div_16 * 16
            _, col0 = swizzle_128(lane_mod_16, reg_k_col0)
            _, col1 = swizzle_128(lane_mod_16, reg_k_col1)
            return col0, col1
    else:
        def _normal_read_columns(lane_div_16, lane_mod_16):
            del lane_div_16, lane_mod_16
            return fx.Int32(0), fx.Int32(0)

    @fx.struct
    class SharedStorage:
        # Each logical 256x128 page is two independent 128x128 half-pages.
        # The hot loop refills one 16-byte pass of one half-page at a time.
        a0_0: fx.Array[a_fx_dtype, LDS_ELEMS_HALF, 16]
        a0_1: fx.Array[a_fx_dtype, LDS_ELEMS_HALF, 16]
        a1_0: fx.Array[a_fx_dtype, LDS_ELEMS_HALF, 16]
        a1_1: fx.Array[a_fx_dtype, LDS_ELEMS_HALF, 16]
        b0_0: fx.Array[b_fx_dtype, LDS_ELEMS_HALF, 16]
        b0_1: fx.Array[b_fx_dtype, LDS_ELEMS_HALF, 16]
        b1_0: fx.Array[b_fx_dtype, LDS_ELEMS_HALF, 16]
        b1_1: fx.Array[b_fx_dtype, LDS_ELEMS_HALF, 16]

    @flyc.kernel(known_block_size=[NUM_THREADS, 1, 1])
    def kernel_gemm(
        A: fx.Tensor, As: fx.Tensor, B: fx.Tensor, Bs: fx.Tensor, C: fx.Tensor, c_m: fx.Int32, c_n: fx.Int32
    ):
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        lds_a0 = (lds.a0_0, lds.a0_1)
        lds_a1 = (lds.a1_0, lds.a1_1)
        lds_b0 = (lds.b0_0, lds.b0_1)
        lds_b1 = (lds.b1_0, lds.b1_1)

        a_f8_ir_t = a_fx_dtype.ir_type
        b_f8_ir_t = b_fx_dtype.ir_type
        gA = make_fp8_buffer_tensor(A, a_f8_ir_t)
        gB = make_fp8_buffer_tensor(B, b_f8_ir_t)
        a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
        b_div = fx.logical_divide(gB, fx.make_layout(1, 1))
        as_rsrc = fx.rocdl.make_buffer_tensor(As, max_size=True)
        bs_rsrc = fx.rocdl.make_buffer_tensor(Bs, max_size=True)
        as_div = fx.logical_divide(as_rsrc, fx.make_layout(1, 1))
        bs_div = fx.logical_divide(bs_rsrc, fx.make_layout(1, 1))
        scale_ld_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
        tx = gpu.thread_id("x")

        num_blocks_m = c_m // BLOCK_M
        num_blocks_n = c_n // BLOCK_N

        pid_m, pid_n = xcd_swizzle(num_blocks_m, num_blocks_n)

        bx_m = pid_m * BLOCK_M
        by_n = pid_n * BLOCK_N

        # The flattened/XCD-swizzled block coordinates are i32, while global
        # address arithmetic below is expressed in MLIR index type. Convert
        # once here and use these index-typed tile bases for every address.
        bx_m_idx = fx.Index(bx_m)
        by_n_idx = fx.Index(by_n)

        tx_i32 = fx.Int32(tx)
        wave_id = tx_i32 // fx.Int32(WARP_SIZE)
        lane = tx_i32 % fx.Int32(WARP_SIZE)

        # Compile-time global leading dimensions:
        #   normal source    [X,K] -> leading dimension K
        #   transpose source [K,X] -> leading dimension X
        a_leading_dim = _a_leading_dim(c_m)
        b_leading_dim = _b_leading_dim(c_n)

        # gl_off_a/gl_off_b: per-lane lists of LOAD_PASSES_HALF swizzled flat
        # global offsets, indexed by DMA pass. gl_off_a[step] is this lane's
        # static 16-byte source for pass `step`; G2SLoader adds the dynamic
        # K-tile base as soffset. Offsets are pre-swizzled so bytes land in the
        # bank-conflict-free LDS slots the MFMA read (S2RLoader) expects.
        gl_off_a = compute_global_swizzle(
            lane,
            wave_id,
            a_leading_dim,
            LOAD_PASSES_HALF,
            preshuffled=False,
        )
        gl_off_b = compute_global_swizzle(
            lane,
            wave_id,
            b_leading_dim,
            LOAD_PASSES_HALF,
            preshuffled=False,
        )
        a_g2s = G2SLoader(
            a_div,
            gl_off_a,
            LOAD_PASSES_HALF,
            a_f8_ir_t,
            wave_id,
        )
        b_g2s = G2SLoader(
            b_div,
            gl_off_b,
            LOAD_PASSES_HALF,
            b_f8_ir_t,
            wave_id,
        )
        s2r = S2RLoader(fx.Int32(0), 1)

        layout_lane16 = fx.make_layout((4, 16), (16, 1))
        coord_lane16 = fx.idx2crd(fx.Int32(lane), layout_lane16)
        lane_div_16 = fx.get(coord_lane16, 0)
        lane_mod_16 = fx.get(coord_lane16, 1)

        # Per-CTA tile base in C elements, folded into each store's linear
        # coordinate below (matching the scale-load addressing on this build;
        # add_offset on a dynamic Index is unsupported here).
        c_tile_base_elems = bx_m_idx * fx.Index(c_n) + by_n_idx
        gC = fx.rocdl.make_buffer_tensor(C, max_size=True)
        c_div = fx.logical_divide(gC, fx.make_layout(1, 1))
        c_store_atom = fx.make_copy_atom(
            fx.rocdl.BufferCopy32b() if output_element_bytes == 4 else fx.rocdl.BufferCopy16b(),
            output_fx_dtype,
        )

        PIN_ACC_BASE = 0

        def _reg_list(prefix, start, end):
            return ",".join(f"~{{{prefix}{r}}}" for r in range(start, end + 1))

        def reserve_pinned_accumulators():
            # Reserve a fixed physical AGPR bank for all accumulators. In the
            # SSA-lowered path, the compiler generated heavy AGPR <-> VGPR traffic,
            # including v_accvgpr_mov/read sequences, s_nop stalls, and accumulator
            # spills. Pinning each f32x4 accumulator to a stable AGPR range keeps the
            # scaled MFMA accumulation in place and avoids those transfers and spills.
            #
            # ACCS_PER_WAVE = 64 accumulator objects and each object is f32x4,
            # so the physical bank is exactly 64 * 4 = 256 AGPRs: a[0:255].
            clobbers = _reg_list("a", PIN_ACC_BASE, PIN_ACC_BASE + ACCS_PER_WAVE * 4 - 1)
            llvm.InlineAsmOp(
                None,
                [],
                "",
                clobbers,
                has_side_effects=True,
            )

        def zero_pinned_accumulators():
            for ai in range_constexpr(ACCS_PER_WAVE * 4):
                llvm.InlineAsmOp(
                    None,
                    [],
                    f"v_accvgpr_write_b32 a[{PIN_ACC_BASE + ai}], 0",
                    f"~{{a{PIN_ACC_BASE + ai}}}",
                    has_side_effects=True,
                )

        def _inline_asm_i32(asm_string, constraints, operands=None):
            op = llvm.InlineAsmOp(
                T.i32,
                operands or [],
                asm_string,
                constraints,
                has_side_effects=True,
            )
            return _one_i32_result(op)

        def _one_i32_result(op):
            # Accept the result attribute names exposed by the supported MLIR Python bindings.
            return getattr(op, "result", getattr(op, "res", op.results[0]))

        def _to_raw_inline_asm_operand(value):
            # TODO: Replace arith._to_raw once FlyDSL exposes a supported public
            # API for passing wrapped values to llvm.InlineAsmOp. _to_raw is
            # deprecated, but remains heavily used internally by FlyDSL.
            return arith._to_raw(value)

        def read_physical_accumulator_slot(slot_idx):
            acc_pin = PIN_ACC_BASE + slot_idx * 4
            r0 = _inline_asm_i32(f"v_accvgpr_read_b32 $0, a[{acc_pin + 0}]", "=v")
            r1 = _inline_asm_i32(f"v_accvgpr_read_b32 $0, a[{acc_pin + 1}]", "=v")
            r2 = _inline_asm_i32(f"v_accvgpr_read_b32 $0, a[{acc_pin + 2}]", "=v")
            r3 = _inline_asm_i32(f"v_accvgpr_read_b32 $0, a[{acc_pin + 3}]", "=v")
            return Vec.from_elements([r0, r1, r2, r3], fx.Int32).bitcast(fx.Float32)

        # As/Bs are MFMA-ready packed scale words: [K128, row] uint32.
        # Each loaded dword already contains the four 16-row/16-col MFMA scale
        # bytes for this lane's 64-row A/B half.  The MFMA instruction selects
        # the byte via op_sel/op_sel_hi, so there is intentionally no hot-loop
        # byte extraction and no 0x01010101 broadcast here.
        c_m_idx = fx.Index(c_m)
        c_n_idx = fx.Index(c_n)

        def hot_loop_scheduler_q_refill_2n():
            # Steady-state Q1 schedule: eight chunks of one K+2 VMEM/LDS
            # refill pass followed by two MFMAs.
            for _ in range_constexpr(8):
                rocdl.sched_vmem(1)
                rocdl.sched_mfma(2)

            rocdl.sched_barrier(0)

        def hot_loop_scheduler_q0_refill_a1_2n():
            # TN/NN: one normal A-bottom LDS read per chunk.
            # NT: one transpose-read A half plus the matching transpose-read
            # scheduling pressure retained from the passing NT specialization.
            for _ in range_constexpr(8):
                rocdl.sched_vmem(1)
                rocdl.sched_dsrd(Q0_SCHED_DSRD)
                rocdl.sched_mfma(2)

            rocdl.sched_barrier(0)

        def hot_loop_scheduler_q_prefetch_4n():
            # TN/NN retain two scheduled DS reads per chunk. NT retains four
            # because both carried operands use two DS_READ_TR instructions.
            for _ in range_constexpr(8):
                rocdl.sched_dsrd(PREFETCH_SCHED_DSRD)
                rocdl.sched_mfma(4)

            rocdl.sched_barrier(0)

        def load_a_scale_row(k128, row):
            off = k128 * c_m_idx + bx_m_idx + row
            reg = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Int32)
            fx.copy(scale_ld_atom, fx.slice(as_div, (None, fx.Int32(off))), reg)
            return fx.memref_load_vec(reg)[0]

        def load_b_scale_row(k128, row):
            off = k128 * c_n_idx + by_n_idx + row
            reg = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Int32)
            fx.copy(scale_ld_atom, fx.slice(bs_div, (None, fx.Int32(off))), reg)
            return fx.memref_load_vec(reg)[0]

        def load_a_scale_subtile(k128, sm):
            subtile_m_idx = reg_subtile_m_idx0 + fx.Index(sm * 2)
            a_row = subtile_m_idx * fx.Index(SUBTILE_M) + fx.Index(lane)
            a_scale = load_a_scale_row(k128, a_row)
            return (a_scale, a_scale, a_scale, a_scale)

        def load_b_scale_subtile(k128, sn):
            subtile_n_idx = reg_subtile_n_idx0 + fx.Index(sn * 2)
            b_row = subtile_n_idx * fx.Index(SUBTILE_N) + fx.Index(lane)
            b_scale = load_b_scale_row(k128, b_row)
            return (b_scale, b_scale, b_scale, b_scale)

        def load_scale_tile(k128):
            # Load all scale VGPRs needed by this wave for this K128 tile once.
            # Return order: A-top, A-bottom, B-left, B-right.
            return (
                load_a_scale_subtile(k128, 0),
                load_a_scale_subtile(k128, 1),
                load_b_scale_subtile(k128, 0),
                load_b_scale_subtile(k128, 1),
            )

        def stage_a_subtile_pass(k_base, subtile, pass_in_subtile, lds_a):
            a_g2s.load_one(
                lds_a[subtile],
                fx.Int32(_a_global_base(k_base, subtile, c_m, bx_m_idx)),
                pass_in_subtile,
            )

        def stage_b_subtile_pass(k_base, subtile, pass_in_subtile, lds_b):
            b_g2s.load_one(
                lds_b[subtile],
                fx.Int32(_b_global_base(k_base, subtile, c_n, by_n_idx)),
                pass_in_subtile,
            )

        def stage_a_subtile(k_base, subtile, lds_a):
            for pass_in_subtile in range_constexpr(LOAD_PASSES_HALF):
                stage_a_subtile_pass(k_base, subtile, pass_in_subtile, lds_a)

        def stage_b_subtile(k_base, subtile, lds_b):
            for pass_in_subtile in range_constexpr(LOAD_PASSES_HALF):
                stage_b_subtile_pass(k_base, subtile, pass_in_subtile, lds_b)

        def load_frag_half_at_byte_base(lds_page, row_byte_base, half):
            # Issue exactly one 16-byte LDS read for one K64 half of an MFMA operand.
            # Keeping the halves separate allows steady-state Q0 to schedule one
            # A-bottom ds_read_b128 in each refill/MFMA chunk.
            k_col = reg_lds_k_col0 if half == 0 else reg_lds_k_col1
            return s2r.load_one(lds_page, fx.Int32(row_byte_base + k_col))

        def pack_frag_halves(x0, x1):
            return pack_i32x4_i32x8(x0, x1)

        def load_frag_at_byte_base(lds_page, row_byte_base):
            # Default complete-fragment path used outside the dedicated Q0 schedule.
            x0 = load_frag_half_at_byte_base(lds_page, row_byte_base, 0)
            x1 = load_frag_half_at_byte_base(lds_page, row_byte_base, 1)
            return pack_frag_halves(x0, x1)

        def load_normal_b_frag(lds_b, local_row, half):
            # Physical [N,K] page, ordinary TN-style fixed-row read.
            half_row = local_row - fx.Index(half * (BLOCK_N // 2))
            return load_frag_at_byte_base(
                lds_b[half],
                half_row * fx.Index(BLOCK_K),
            )

        def load_transposed_frag_half(lds_page, local_x_tile, half):
            # Exact inverse mapping validated by the MXFP8 NN fragment probe.
            lane_div16_i32 = fx.Int32(lane_div_16)
            lane_in16_i32 = fx.Int32(lane_mod_16)
            source_k = (
                lane_div16_i32 * fx.Int32(16)
                + lane_in16_i32 // fx.Int32(2)
            )
            source_x = (
                fx.Int32(local_x_tile)
                + (lane_in16_i32 % fx.Int32(2)) * fx.Int32(8)
            )

            physical_k, physical_x = swizzle_128(source_k, source_x)
            base = physical_k * fx.Int32(128) + physical_x
            other = base ^ fx.Int32(0x440)
            immediate_offset = 0 if half == 0 else 0x2000

            return s2r.load_one_transpose(
                lds_page,
                base,
                other,
                immediate_offset=immediate_offset,
            )

        def load_transposed_frag(lds_page, local_x_tile):
            x0 = load_transposed_frag_half(lds_page, local_x_tile, 0)
            x1 = load_transposed_frag_half(lds_page, local_x_tile, 1)
            return pack_frag_halves(x0, x1)

        def _acc_idx(subtile_id, mi, ni):
            return subtile_id * MFMA_M_PER_SUBTILE * MFMA_N_PER_SUBTILE + mi * MFMA_N_PER_SUBTILE + ni

        def pinned_mfma(acc_idx, a_frag, b_frag, a_scale, b_scale, mi, ni):
            # Fixed physical accumulator bank, visible SSA A/B/scale operands.
            # acc_idx maps directly to a[PIN_ACC_BASE + 4*acc_idx : +3].
            # The scale operands are MFMA-ready packed dwords.  mi/ni choose
            # which of the four bytes inside the A/B scale dword the MFMA uses.
            acc_pin = PIN_ACC_BASE + acc_idx * 4
            llvm.InlineAsmOp(
                None,
                [
                    _to_raw_inline_asm_operand(a_frag),
                    _to_raw_inline_asm_operand(b_frag),
                    _to_raw_inline_asm_operand(a_scale),
                    _to_raw_inline_asm_operand(b_scale),
                ],
                (
                    f"v_mfma_scale_f32_16x16x128_f8f6f4 "
                    f"a[{acc_pin}:{acc_pin + 3}], "
                    f"$0, $1, "
                    f"a[{acc_pin}:{acc_pin + 3}], "
                    f"$2, $3 "
                    f"op_sel:[{mi & 1},{ni & 1},0] "
                    f"op_sel_hi:[{mi >> 1},{ni >> 1},0] "
                    f"cbsz:{a_matrix_format} blgp:{b_matrix_format}"
                ),
                (f"v,v,v,v,~{{a{acc_pin}}},~{{a{acc_pin + 1}}},~{{a{acc_pin + 2}}},~{{a{acc_pin + 3}}}"),
                has_side_effects=True,
            )

        def pinned_final_mfma(dst_slot, old_acc_idx, a_frag, b_frag, a_scale, b_scale, mi, ni):
            # Final-page form used by HK: destination and previous partial sum
            # may be different AGPR ranges.  Once old_acc_idx is consumed, its
            # physical slot is dead and can be reused as a later destination.
            dst_pin = PIN_ACC_BASE + dst_slot * 4
            old_pin = PIN_ACC_BASE + old_acc_idx * 4
            llvm.InlineAsmOp(
                None,
                [
                    _to_raw_inline_asm_operand(a_frag),
                    _to_raw_inline_asm_operand(b_frag),
                    _to_raw_inline_asm_operand(a_scale),
                    _to_raw_inline_asm_operand(b_scale),
                ],
                (
                    f"v_mfma_scale_f32_16x16x128_f8f6f4 "
                    f"a[{dst_pin}:{dst_pin + 3}], "
                    f"$0, $1, "
                    f"a[{old_pin}:{old_pin + 3}], "
                    f"$2, $3 "
                    f"op_sel:[{mi & 1},{ni & 1},0] "
                    f"op_sel_hi:[{mi >> 1},{ni >> 1},0] "
                    f"cbsz:{a_matrix_format} blgp:{b_matrix_format}"
                ),
                (f"v,v,v,v,~{{a{dst_pin}}},~{{a{dst_pin + 1}}},~{{a{dst_pin + 2}}},~{{a{dst_pin + 3}}}"),
                has_side_effects=True,
            )

        def mfma_4n(acc_base, a_frag, a_scale, b0, b1, b2, b3, bs0, bs1, bs2, bs3):
            """Emit four N-direction scaled MFMAs into fixed physical AGPR accumulators."""
            mi = (acc_base // MFMA_N_PER_SUBTILE) % MFMA_M_PER_SUBTILE
            pinned_mfma(acc_base + 0, a_frag, b0, a_scale, bs0, mi, 0)
            pinned_mfma(acc_base + 1, a_frag, b1, a_scale, bs1, mi, 1)
            pinned_mfma(acc_base + 2, a_frag, b2, a_scale, bs2, mi, 2)
            pinned_mfma(acc_base + 3, a_frag, b3, a_scale, bs3, mi, 3)

        def mfma_2n(acc_base, a_frag, a_scale, b0, b1, bs0, bs1, ni_base):
            mi = (acc_base // MFMA_N_PER_SUBTILE) % MFMA_M_PER_SUBTILE
            pinned_mfma(acc_base + 0, a_frag, b0, a_scale, bs0, mi, ni_base + 0)
            pinned_mfma(acc_base + 1, a_frag, b1, a_scale, bs1, mi, ni_base + 1)

        def store_acc_vector_for_logical_idx(logical_acc_idx, acc):
            subtile_id = logical_acc_idx // (MFMA_M_PER_SUBTILE * MFMA_N_PER_SUBTILE)
            local_idx = logical_acc_idx % (MFMA_M_PER_SUBTILE * MFMA_N_PER_SUBTILE)
            sm = subtile_id // 2
            sn = subtile_id % 2
            mi = local_idx // MFMA_N_PER_SUBTILE
            ni = local_idx % MFMA_N_PER_SUBTILE

            subtile_m_idx = reg_subtile_m_idx0 + fx.Index(sm * 2)
            subtile_n_idx = reg_subtile_n_idx0 + fx.Index(sn * 2)
            row_base = subtile_m_idx * SUBTILE_M + fx.Index(mi * MFMA_M) + lane_div_16 * 4
            col = subtile_n_idx * SUBTILE_N + fx.Index(ni * MFMA_N) + lane_mod_16
            for ii in range_constexpr(4):
                row = row_base + fx.Index(ii)
                c_idx = c_tile_base_elems + row * fx.Index(c_n) + col
                value = Vec(acc)[ii]
                if output_dtype != torch.float32:
                    value = value.to(output_fx_dtype)
                reg = fx.make_rmem_tensor(fx.make_layout(1, 1), output_fx_dtype)
                fx.memref_store_vec(Vec.filled(1, value, output_fx_dtype), reg)
                fx.copy(c_store_atom, reg, fx.slice(c_div, (None, fx.Int32(c_idx))))


        # Explicit register coordinates for HK-style four-quadrant mapping.
        # BLOCK_M/BLOCK_N are 256x256.  Four waves map to warp positions
        # inside each 128x128 quadrant:
        #   cA: (warp_m,     warp_n)
        #   cB: (warp_m,     warp_n + 2)
        #   cC: (warp_m + 2, warp_n)
        #   cD: (warp_m + 2, warp_n + 2)
        reg_lds_k_col0, reg_lds_k_col1 = _normal_read_columns(
            lane_div_16,
            lane_mod_16,
        )

        reg_subtile_m_idx0 = wave_id // 2
        reg_subtile_n_idx0 = wave_id % 2

        reserve_pinned_accumulators()
        zero_pinned_accumulators()

        def load_b_subtile_ni_regs(lds_b, scale_tile, sn, ni):
            subtile_n_idx = reg_subtile_n_idx0 + fx.Index(sn * 2)
            b_scales = scale_tile[2] if sn == 0 else scale_tile[3]

            b_ni = _load_b_ni(
                load_transposed_frag,
                load_normal_b_frag,
                lds_b,
                sn,
                ni,
                reg_subtile_n_idx0,
                lane_mod_16,
            )
            return b_ni, b_scales[ni]

        def load_b_subtile_regs(lds_b, scale_tile, sn):
            b0, bs0 = load_b_subtile_ni_regs(lds_b, scale_tile, sn, 0)
            b1, bs1 = load_b_subtile_ni_regs(lds_b, scale_tile, sn, 1)
            b2, bs2 = load_b_subtile_ni_regs(lds_b, scale_tile, sn, 2)
            b3, bs3 = load_b_subtile_ni_regs(lds_b, scale_tile, sn, 3)
            return b0, b1, b2, b3, bs0, bs1, bs2, bs3

        def load_a_subtile_mi_half(lds_a, sm, mi, half):
            subtile_m_idx = reg_subtile_m_idx0 + fx.Index(sm * 2)

            return _load_a_half(
                load_transposed_frag_half,
                load_frag_half_at_byte_base,
                lds_a,
                sm,
                mi,
                half,
                reg_subtile_m_idx0,
                lane_mod_16,
            )

        def load_a_subtile_mi_regs(lds_a, scale_tile, sm, mi):
            # Fine-grained A register load for one 16-row M-direction MFMA slice.
            a_scales = scale_tile[0] if sm == 0 else scale_tile[1]
            x0 = load_a_subtile_mi_half(lds_a, sm, mi, 0)
            x1 = load_a_subtile_mi_half(lds_a, sm, mi, 1)
            a_mi = pack_frag_halves(x0, x1)
            a_scale_mi = a_scales[mi]
            return a_mi, a_scale_mi

        def load_a_subtile_regs(lds_a, scale_tile, sm):
            a0, as0 = load_a_subtile_mi_regs(lds_a, scale_tile, sm, 0)
            a1, as1 = load_a_subtile_mi_regs(lds_a, scale_tile, sm, 1)
            a2, as2 = load_a_subtile_mi_regs(lds_a, scale_tile, sm, 2)
            a3, as3 = load_a_subtile_mi_regs(lds_a, scale_tile, sm, 3)
            return a0, a1, a2, a3, as0, as1, as2, as3

        def hk_one_k_with_refill(
            k128,
            cur_a,
            cur_b,
            next_a,
            next_b,
            refill_a,
            refill_b,
            a0_regs,
            b0_regs,
            cur_scales,
            prev_refill_scales,
        ):
            # Scale invariant:
            #   cur_scales is HK MFMA-ready for K.
            #   prev_refill_scales is HK MFMA-ready for K+1.
            #   This iteration issues K+2 scale loads and returns them for the
            #   next steady iteration or final tail.

            # Wait only far enough for the current page; the next-page refill may remain in flight.
            barrier(vmcnt=2 * LOAD_PASSES_A_SUBTILE + 2 * LOAD_PASSES_B_SUBTILE, lgkmcnt=0)
            rocdl.sched_barrier(0)

            # Immediately issue MFMA-ready K+2 scale loads.
            # They are returned for the next iteration without any in-kernel
            # byte extraction or broadcast.
            refill_scales = load_scale_tile(fx.Index(k128 + 2))
            next_scales_ready = prev_refill_scales
            # A-top and B-left are both carried as complete 64-row register tiles,
            # so their LDS half-pages can be refilled immediately.
            a00, a01, a02, a03, as00, as01, as02, as03 = a0_regs
            b00, b01, b02, b03, bs00, bs01, bs02, bs03 = b0_regs

            b10, bs10 = load_b_subtile_ni_regs(cur_b, cur_scales, 1, 0)
            b11, bs11 = load_b_subtile_ni_regs(cur_b, cur_scales, 1, 1)
            b12, bs12 = load_b_subtile_ni_regs(cur_b, cur_scales, 1, 2)
            b13, bs13 = load_b_subtile_ni_regs(cur_b, cur_scales, 1, 3)

            # Refill the current ping-pong page with K+2, alternating A and B passes.
            k_refill = fx.Index((k128 + 2) * BLOCK_K)

            # Q0: interleave the current tile's A-bottom LDS reads with K+2
            # refills and Q0 compute. Each complete A-bottom fragment is assembled
            # from two independently scheduled K64 halves.
            rocdl.sched_barrier(0)
            a10_x0 = load_a_subtile_mi_half(cur_a, 1, 0, 0)
            stage_a_subtile_pass(k_refill, 0, 0, refill_a)
            mfma_2n(_acc_idx(0, 0, 0), a00, as00, b00, b01, bs00, bs01, 0)

            a10_x1 = load_a_subtile_mi_half(cur_a, 1, 0, 1)
            stage_b_subtile_pass(k_refill, 0, 0, refill_b)
            mfma_2n(_acc_idx(0, 0, 2), a00, as00, b02, b03, bs02, bs03, 2)

            a11_x0 = load_a_subtile_mi_half(cur_a, 1, 1, 0)
            stage_a_subtile_pass(k_refill, 0, 1, refill_a)
            mfma_2n(_acc_idx(0, 1, 0), a01, as01, b00, b01, bs00, bs01, 0)

            a11_x1 = load_a_subtile_mi_half(cur_a, 1, 1, 1)
            stage_b_subtile_pass(k_refill, 0, 1, refill_b)
            mfma_2n(_acc_idx(0, 1, 2), a01, as01, b02, b03, bs02, bs03, 2)

            a12_x0 = load_a_subtile_mi_half(cur_a, 1, 2, 0)
            stage_a_subtile_pass(k_refill, 0, 2, refill_a)
            mfma_2n(_acc_idx(0, 2, 0), a02, as02, b00, b01, bs00, bs01, 0)

            a12_x1 = load_a_subtile_mi_half(cur_a, 1, 2, 1)
            stage_b_subtile_pass(k_refill, 0, 2, refill_b)
            mfma_2n(_acc_idx(0, 2, 2), a02, as02, b02, b03, bs02, bs03, 2)

            a13_x0 = load_a_subtile_mi_half(cur_a, 1, 3, 0)
            stage_a_subtile_pass(k_refill, 0, 3, refill_a)
            mfma_2n(_acc_idx(0, 3, 0), a03, as03, b00, b01, bs00, bs01, 0)

            a13_x1 = load_a_subtile_mi_half(cur_a, 1, 3, 1)
            stage_b_subtile_pass(k_refill, 0, 3, refill_b)
            mfma_2n(_acc_idx(0, 3, 2), a03, as03, b02, b03, bs02, bs03, 2)

            hot_loop_scheduler_q0_refill_a1_2n()

            # Retire the eight distributed A-bottom LDS reads before K+2 refills
            # overwrite the current page's A-bottom half-page. Keep this wait as
            # late as possible to maximize read/compute overlap.
            rocdl.sched_barrier(0)
            barrier(lgkmcnt=0)
            rocdl.sched_barrier(0)

            a10 = pack_frag_halves(a10_x0, a10_x1)
            a11 = pack_frag_halves(a11_x0, a11_x1)
            a12 = pack_frag_halves(a12_x0, a12_x1)
            a13 = pack_frag_halves(a13_x0, a13_x1)
            as10 = cur_scales[1][0]
            as11 = cur_scales[1][1]
            as12 = cur_scales[1][2]
            as13 = cur_scales[1][3]

            rocdl.sched_barrier(0)
            stage_b_subtile_pass(k_refill, 1, 0, refill_b)
            mfma_2n(_acc_idx(1, 0, 0), a00, as00, b10, b11, bs10, bs11, 0)

            stage_a_subtile_pass(k_refill, 1, 0, refill_a)
            mfma_2n(_acc_idx(1, 0, 2), a00, as00, b12, b13, bs12, bs13, 2)

            stage_b_subtile_pass(k_refill, 1, 1, refill_b)
            mfma_2n(_acc_idx(1, 1, 0), a01, as01, b10, b11, bs10, bs11, 0)

            stage_a_subtile_pass(k_refill, 1, 1, refill_a)
            mfma_2n(_acc_idx(1, 1, 2), a01, as01, b12, b13, bs12, bs13, 2)

            stage_b_subtile_pass(k_refill, 1, 2, refill_b)
            mfma_2n(_acc_idx(1, 2, 0), a02, as02, b10, b11, bs10, bs11, 0)

            stage_a_subtile_pass(k_refill, 1, 2, refill_a)
            mfma_2n(_acc_idx(1, 2, 2), a02, as02, b12, b13, bs12, bs13, 2)

            stage_b_subtile_pass(k_refill, 1, 3, refill_b)
            mfma_2n(_acc_idx(1, 3, 0), a03, as03, b10, b11, bs10, bs11, 0)

            stage_a_subtile_pass(k_refill, 1, 3, refill_a)
            mfma_2n(_acc_idx(1, 3, 2), a03, as03, b12, b13, bs12, bs13, 2)
            hot_loop_scheduler_q_refill_2n()

            # Leave exactly the K+2 refill and scale loads outstanding. The following
            # LDS reads consume the already-ready next page, not the page being refilled.
            rocdl.sched_barrier(0)
            barrier(vmcnt=2 * LOAD_PASSES_A_SUBTILE + 2 * LOAD_PASSES_B_SUBTILE + LOAD_PASSES_SCALES, lgkmcnt=0)
            rocdl.sched_barrier(0)

            next_a00, next_as00 = load_a_subtile_mi_regs(next_a, next_scales_ready, 0, 0)
            mfma_4n(_acc_idx(2, 0, 0), a10, as10, b00, b01, b02, b03, bs00, bs01, bs02, bs03)

            next_a01, next_as01 = load_a_subtile_mi_regs(next_a, next_scales_ready, 0, 1)
            mfma_4n(_acc_idx(2, 1, 0), a11, as11, b00, b01, b02, b03, bs00, bs01, bs02, bs03)

            next_a02, next_as02 = load_a_subtile_mi_regs(next_a, next_scales_ready, 0, 2)
            mfma_4n(_acc_idx(2, 2, 0), a12, as12, b00, b01, b02, b03, bs00, bs01, bs02, bs03)

            next_a03, next_as03 = load_a_subtile_mi_regs(next_a, next_scales_ready, 0, 3)
            mfma_4n(_acc_idx(2, 3, 0), a13, as13, b00, b01, b02, b03, bs00, bs01, bs02, bs03)

            next_b00, next_bs00 = load_b_subtile_ni_regs(next_b, next_scales_ready, 0, 0)
            mfma_4n(_acc_idx(3, 0, 0), a10, as10, b10, b11, b12, b13, bs10, bs11, bs12, bs13)

            next_b01, next_bs01 = load_b_subtile_ni_regs(next_b, next_scales_ready, 0, 1)
            mfma_4n(_acc_idx(3, 1, 0), a11, as11, b10, b11, b12, b13, bs10, bs11, bs12, bs13)

            next_b02, next_bs02 = load_b_subtile_ni_regs(next_b, next_scales_ready, 0, 2)
            mfma_4n(_acc_idx(3, 2, 0), a12, as12, b10, b11, b12, b13, bs10, bs11, bs12, bs13)

            next_b03, next_bs03 = load_b_subtile_ni_regs(next_b, next_scales_ready, 0, 3)
            mfma_4n(_acc_idx(3, 3, 0), a13, as13, b10, b11, b12, b13, bs10, bs11, bs12, bs13)

            hot_loop_scheduler_q_prefetch_4n()

            next_a0_regs = (
                next_a00,
                next_a01,
                next_a02,
                next_a03,
                next_as00,
                next_as01,
                next_as02,
                next_as03,
            )
            next_b0_regs = (
                next_b00,
                next_b01,
                next_b02,
                next_b03,
                next_bs00,
                next_bs01,
                next_bs02,
                next_bs03,
            )

            return next_a0_regs, next_b0_regs, next_scales_ready, refill_scales

        def hk_one_k_tail_with_next(cur_a, cur_b, next_a, next_b, a0_regs, b0_regs, cur_scales, next_scales):
            barrier(vmcnt=2 * LOAD_PASSES_A_SUBTILE + 2 * LOAD_PASSES_B_SUBTILE, lgkmcnt=0)

            a00, a01, a02, a03, as00, as01, as02, as03 = a0_regs
            b00, b01, b02, b03, bs00, bs01, bs02, bs03 = b0_regs

            b10, bs10 = load_b_subtile_ni_regs(cur_b, cur_scales, 1, 0)
            b11, bs11 = load_b_subtile_ni_regs(cur_b, cur_scales, 1, 1)
            b12, bs12 = load_b_subtile_ni_regs(cur_b, cur_scales, 1, 2)
            b13, bs13 = load_b_subtile_ni_regs(cur_b, cur_scales, 1, 3)

            mfma_4n(_acc_idx(0, 0, 0), a00, as00, b00, b01, b02, b03, bs00, bs01, bs02, bs03)
            mfma_4n(_acc_idx(0, 1, 0), a01, as01, b00, b01, b02, b03, bs00, bs01, bs02, bs03)
            mfma_4n(_acc_idx(0, 2, 0), a02, as02, b00, b01, b02, b03, bs00, bs01, bs02, bs03)
            mfma_4n(_acc_idx(0, 3, 0), a03, as03, b00, b01, b02, b03, bs00, bs01, bs02, bs03)

            rocdl.sched_barrier(0)
            barrier(lgkmcnt=0)
            rocdl.sched_barrier(0)

            a10, as10 = load_a_subtile_mi_regs(cur_a, cur_scales, 1, 0)
            a11, as11 = load_a_subtile_mi_regs(cur_a, cur_scales, 1, 1)
            a12, as12 = load_a_subtile_mi_regs(cur_a, cur_scales, 1, 2)
            a13, as13 = load_a_subtile_mi_regs(cur_a, cur_scales, 1, 3)

            mfma_4n(_acc_idx(1, 0, 0), a00, as00, b10, b11, b12, b13, bs10, bs11, bs12, bs13)
            mfma_4n(_acc_idx(1, 1, 0), a01, as01, b10, b11, b12, b13, bs10, bs11, bs12, bs13)
            mfma_4n(_acc_idx(1, 2, 0), a02, as02, b10, b11, b12, b13, bs10, bs11, bs12, bs13)
            mfma_4n(_acc_idx(1, 3, 0), a03, as03, b10, b11, b12, b13, bs10, bs11, bs12, bs13)

            rocdl.sched_barrier(0)
            barrier(vmcnt=LOAD_PASSES_A_SUBTILE + LOAD_PASSES_B_SUBTILE, lgkmcnt=0)
            rocdl.sched_barrier(0)

            next_a00, next_as00 = load_a_subtile_mi_regs(next_a, next_scales, 0, 0)
            mfma_4n(_acc_idx(2, 0, 0), a10, as10, b00, b01, b02, b03, bs00, bs01, bs02, bs03)

            next_a01, next_as01 = load_a_subtile_mi_regs(next_a, next_scales, 0, 1)
            mfma_4n(_acc_idx(2, 1, 0), a11, as11, b00, b01, b02, b03, bs00, bs01, bs02, bs03)

            next_a02, next_as02 = load_a_subtile_mi_regs(next_a, next_scales, 0, 2)
            mfma_4n(_acc_idx(2, 2, 0), a12, as12, b00, b01, b02, b03, bs00, bs01, bs02, bs03)

            next_a03, next_as03 = load_a_subtile_mi_regs(next_a, next_scales, 0, 3)
            mfma_4n(_acc_idx(2, 3, 0), a13, as13, b00, b01, b02, b03, bs00, bs01, bs02, bs03)

            next_b00, next_bs00 = load_b_subtile_ni_regs(next_b, next_scales, 0, 0)
            mfma_4n(_acc_idx(3, 0, 0), a10, as10, b10, b11, b12, b13, bs10, bs11, bs12, bs13)

            next_b01, next_bs01 = load_b_subtile_ni_regs(next_b, next_scales, 0, 1)
            mfma_4n(_acc_idx(3, 1, 0), a11, as11, b10, b11, b12, b13, bs10, bs11, bs12, bs13)

            next_b02, next_bs02 = load_b_subtile_ni_regs(next_b, next_scales, 0, 2)
            mfma_4n(_acc_idx(3, 2, 0), a12, as12, b10, b11, b12, b13, bs10, bs11, bs12, bs13)

            next_b03, next_bs03 = load_b_subtile_ni_regs(next_b, next_scales, 0, 3)
            mfma_4n(_acc_idx(3, 3, 0), a13, as13, b10, b11, b12, b13, bs10, bs11, bs12, bs13)

            hot_loop_scheduler_q_prefetch_4n()

            next_a0_regs = (
                next_a00,
                next_a01,
                next_a02,
                next_a03,
                next_as00,
                next_as01,
                next_as02,
                next_as03,
            )
            next_b0_regs = (
                next_b00,
                next_b01,
                next_b02,
                next_b03,
                next_bs00,
                next_bs01,
                next_bs02,
                next_bs03,
            )

            return next_a0_regs, next_b0_regs

        def hk_one_k_final(cur_a, cur_b, a0_regs, b0_regs, cur_scales):
            barrier(vmcnt=0, lgkmcnt=0)

            a00, a01, a02, a03, as00, as01, as02, as03 = a0_regs
            b00, b01, b02, b03, bs00, bs01, bs02, bs03 = b0_regs

            # Materialize the remaining final-page A/B fragments once.  The
            # subsequent schedule is entirely register/AGPR traffic.
            b10, bs10 = load_b_subtile_ni_regs(cur_b, cur_scales, 1, 0)
            b11, bs11 = load_b_subtile_ni_regs(cur_b, cur_scales, 1, 1)
            b12, bs12 = load_b_subtile_ni_regs(cur_b, cur_scales, 1, 2)
            b13, bs13 = load_b_subtile_ni_regs(cur_b, cur_scales, 1, 3)

            rocdl.sched_barrier(0)
            barrier(lgkmcnt=0)
            rocdl.sched_barrier(0)

            a10, as10 = load_a_subtile_mi_regs(cur_a, cur_scales, 1, 0)
            a11, as11 = load_a_subtile_mi_regs(cur_a, cur_scales, 1, 1)
            a12, as12 = load_a_subtile_mi_regs(cur_a, cur_scales, 1, 2)
            a13, as13 = load_a_subtile_mi_regs(cur_a, cur_scales, 1, 3)

            rocdl.sched_barrier(0)
            barrier(lgkmcnt=0)
            rocdl.sched_barrier(0)

            a_frags = (a00, a01, a02, a03, a10, a11, a12, a13)
            a_scales = (as00, as01, as02, as03, as10, as11, as12, as13)
            b_frags = (b00, b01, b02, b03, b10, b11, b12, b13)
            b_scales = (bs00, bs01, bs02, bs03, bs10, bs11, bs12, bs13)

            # Rolling final-page epilogue.
            #
            # Finalize accumulators in their own physical AGPR slots, but delay
            # each AGPR read/store until several independent final MFMAs have
            # been issued. 
            #
            #   MFMA 0, MFMA 1, MFMA 2, MFMA 3, drain 0,
            #   MFMA 4, drain 1, MFMA 5, drain 2, ...
            #
            # The buffer stores are only issued here; they may remain in flight
            # while later MFMAs and accumulator drains continue.
            FINAL_EPILOGUE_DEPTH = 4
            pending = []

            for old_acc_idx in range_constexpr(ACCS_PER_WAVE):
                subtile_id = old_acc_idx // (MFMA_M_PER_SUBTILE * MFMA_N_PER_SUBTILE)
                local_idx = old_acc_idx % (MFMA_M_PER_SUBTILE * MFMA_N_PER_SUBTILE)
                sm = subtile_id // 2
                sn = subtile_id % 2
                mi = local_idx // MFMA_N_PER_SUBTILE
                ni = local_idx % MFMA_N_PER_SUBTILE

                a_frag_idx = sm * MFMA_M_PER_SUBTILE + mi
                b_frag_idx = sn * MFMA_N_PER_SUBTILE + ni

                # Final MFMA remains in-place.  The logical accumulator's own
                # AGPR slot is unique and cannot conflict with another pending
                # result, so no ad-hoc physical-slot permutation is needed.
                pinned_final_mfma(
                    old_acc_idx,
                    old_acc_idx,
                    a_frags[a_frag_idx],
                    b_frags[b_frag_idx],
                    a_scales[a_frag_idx],
                    b_scales[b_frag_idx],
                    mi,
                    ni,
                )
                pending.append(old_acc_idx)

                # Drain the oldest completed result only after enough newer
                # independent MFMAs have supplied the MFMA->AGPR-read spacing.
                if len(pending) == FINAL_EPILOGUE_DEPTH:
                    drain_acc_idx = pending.pop(0)
                    acc = read_physical_accumulator_slot(drain_acc_idx)
                    store_acc_vector_for_logical_idx(drain_acc_idx, acc)

            # Flush the final results after all final-page MFMAs have issued.
            for drain_acc_idx in pending:
                acc = read_physical_accumulator_slot(drain_acc_idx)
                store_acc_vector_for_logical_idx(drain_acc_idx, acc)

        # Prologue: stage K0/K1 data into ping-pong LDS pages. Scales are not staged in
        # LDS: As/Bs are already MFMA-ready preshuffled packed uint32 [K128, row],
        # and load_scale_tile returns the current wave's scale operands in VGPRs.

        # Load scales first, so that they become the oldest VMEM ops.
        scales0 = load_scale_tile(fx.Index(0))
        scales1 = load_scale_tile(fx.Index(1))

        stage_a_subtile(fx.Index(0), 0, lds_a0)
        stage_b_subtile(fx.Index(0), 0, lds_b0)
        stage_b_subtile(fx.Index(0), 1, lds_b0)
        stage_a_subtile(fx.Index(0), 1, lds_a0)

        stage_a_subtile(fx.Index(BLOCK_K), 0, lds_a1)
        stage_b_subtile(fx.Index(BLOCK_K), 0, lds_b1)
        stage_b_subtile(fx.Index(BLOCK_K), 1, lds_b1)
        stage_a_subtile(fx.Index(BLOCK_K), 1, lds_a1)

        rocdl.sched_barrier(0)
        barrier(vmcnt=3 * LOAD_PASSES_A_SUBTILE + 4 * LOAD_PASSES_B_SUBTILE)
        rocdl.sched_barrier(0)

        # scales0 is already MFMA-ready; no byte extraction or broadcast is needed.
        # Keep the hot loop consistent for k=0 and k>0:
        # K0 is consumed directly.  K1 MFMA-ready scales are carried as
        # prev_refill_scales and become next_scales_ready at loop entry.

        # Seed the carried-register pipeline with K0 A-top. In later steady-state
        # iterations, Q2/Q3 of the preceding iteration prefetch the next tile's
        # A-top and B-left register tiles before their LDS half-pages are reused.
        a0_regs = load_a_subtile_regs(lds_a0, scales0, 0)

        rocdl.sched_barrier(0)
        barrier(vmcnt=3 * LOAD_PASSES_A_SUBTILE + 3 * LOAD_PASSES_B_SUBTILE)
        rocdl.sched_barrier(0)

        # Complete the K0 carried-register seed with B-left.
        b0_regs = load_b_subtile_regs(lds_b0, scales0, 0)

        # Main HK loop: exactly one logical K128 per iteration.
        # Even k consumes and refills LDS0; odd k does the same for LDS1.
        # Scale tiles follow the same K128 progression but remain in VGPRs.
        refill_scales = scales1  # K1 scales become the next ready scale tile at loop entry
        for k128 in range_constexpr(NUM_K_TILES - 2):
            if (k128 % 2) == 0:
                a0_regs, b0_regs, scales1, refill_scales = hk_one_k_with_refill(
                    k128,
                    lds_a0,
                    lds_b0,
                    lds_a1,
                    lds_b1,
                    lds_a0,
                    lds_b0,
                    a0_regs,
                    b0_regs,
                    scales0,
                    refill_scales,
                )
            else:
                a0_regs, b0_regs, scales0, refill_scales = hk_one_k_with_refill(
                    k128,
                    lds_a1,
                    lds_b1,
                    lds_a0,
                    lds_b0,
                    lds_a1,
                    lds_b1,
                    a0_regs,
                    b0_regs,
                    scales1,
                    refill_scales,
                )

        # Common two-page tail. The penultimate tile still uses the Q2/Q3
        # carry-prefetch scheduler to prepare A-top/B-left for the final tile,
        # but it performs no K+2 data or scale refill. The final tile performs
        # compute only. After the steady loop, a0_regs/b0_regs belong to the
        # next tile to consume, while refill_scales belongs to the page most
        # recently refilled; therefore tail page order depends on parity:
        #   even NUM_K_TILES: consume LDS0 then final LDS1
        #   odd  NUM_K_TILES: consume LDS1 then final LDS0
        if (NUM_K_TILES % 2) == 0:
            scales1 = refill_scales
            a0_regs, b0_regs = hk_one_k_tail_with_next(
                lds_a0,
                lds_b0,
                lds_a1,
                lds_b1,
                a0_regs,
                b0_regs,
                scales0,
                scales1,
            )
            hk_one_k_final(lds_a1, lds_b1, a0_regs, b0_regs, scales1)
        else:
            scales0 = refill_scales
            a0_regs, b0_regs = hk_one_k_tail_with_next(
                lds_a1,
                lds_b1,
                lds_a0,
                lds_b0,
                a0_regs,
                b0_regs,
                scales1,
                scales0,
            )
            hk_one_k_final(lds_a0, lds_b0, a0_regs, b0_regs, scales0)

    @flyc.jit
    def launch_gemm(
        A: fx.Tensor,
        As: fx.Tensor,
        B: fx.Tensor,
        Bs: fx.Tensor,
        C: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        # The integration only dispatches aligned shapes; no partial-tile masking exists.
        grid_x = (c_m // BLOCK_M) * (c_n // BLOCK_N)
        kernel_gemm(
            A,
            As,
            B,
            Bs,
            C,
            c_m,
            c_n,
            value_attrs={"rocdl.waves_per_eu": 1, "rocdl.flat_work_group_size": "256,256"},
        ).launch(grid=(grid_x, 1, 1), block=(NUM_THREADS, 1, 1), stream=stream)

    return launch_gemm

def do_gemm(
    A: torch.Tensor,
    As: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    C: torch.Tensor,
    stream=None,
    *,
    layout: str = "TN",
):
    """Launch one cached compile-time MXFP8 layout specialization."""
    if layout == "TN":
        M_runtime, K_runtime = A.shape
        N_runtime, Kb_runtime = B.shape
    elif layout == "NN":
        M_runtime, K_runtime = A.shape
        Kb_runtime, N_runtime = B.shape
    elif layout == "NT":
        K_runtime, M_runtime = A.shape
        Kb_runtime, N_runtime = B.shape
    else:
        raise ValueError(f"Unsupported MXFP8 kernel layout: {layout}")

    assert K_runtime == Kb_runtime, f"A.K={K_runtime} != B.K={Kb_runtime}"
    supported_fp8_dtypes = (torch.float8_e4m3fn, torch.float8_e5m2)
    assert A.dtype in supported_fp8_dtypes, f"unsupported A FP8 dtype: {A.dtype}"
    assert B.dtype in supported_fp8_dtypes, f"unsupported B FP8 dtype: {B.dtype}"

    if M_runtime % _BLOCK_M != 0:
        raise FlyDSLUnsupportedError(
            f"FlyDSL MXFP8 {layout} GEMM requires M to be a multiple of "
            f"{_BLOCK_M}, got M={M_runtime}"
        )
    if N_runtime % _BLOCK_N != 0:
        raise FlyDSLUnsupportedError(
            f"FlyDSL MXFP8 {layout} GEMM requires N to be a multiple of "
            f"{_BLOCK_N}, got N={N_runtime}"
        )
    if K_runtime % _BLOCK_K != 0:
        raise FlyDSLUnsupportedError(
            f"FlyDSL MXFP8 {layout} GEMM requires K to be a multiple of "
            f"{_BLOCK_K}, got K={K_runtime}"
        )
    num_k_tiles = K_runtime // _BLOCK_K
    if num_k_tiles < 4:
        raise FlyDSLUnsupportedError(
            f"FlyDSL MXFP8 {layout} GEMM requires at least 4 K{_BLOCK_K} "
            f"tiles, got K={K_runtime} ({num_k_tiles} tiles)"
        )

    expected_as = (K_runtime // _BLOCK_K, M_runtime)
    expected_bs = (K_runtime // _BLOCK_K, N_runtime)
    assert As.dtype == torch.int32, f"As dtype {As.dtype} != torch.int32 packed scales"
    assert Bs.dtype == torch.int32, f"Bs dtype {Bs.dtype} != torch.int32 packed scales"
    assert tuple(As.shape) == expected_as, f"As shape {tuple(As.shape)} != {expected_as}"
    assert tuple(Bs.shape) == expected_bs, f"Bs shape {tuple(Bs.shape)} != {expected_bs}"
    assert tuple(C.shape) == (M_runtime, N_runtime), (
        f"C shape {tuple(C.shape)} != {(M_runtime, N_runtime)}"
    )
    if C.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(
            "C dtype must be torch.float16, torch.bfloat16, or torch.float32, "
            f"got {C.dtype}"
        )

    tensors = (A, As, B, Bs, C)
    if any(t.device != A.device for t in tensors[1:]):
        raise ValueError("A, B, packed scales, and C must be on the same device")

    if stream is None:
        stream = torch.cuda.current_stream()

    # Preserve the exact flat descriptor contract used by the passing kernels.
    A_arg = A.view(torch.uint8).contiguous().view(-1)
    B_arg = B.view(torch.uint8).contiguous().view(-1)
    As_arg = As.contiguous().view(-1)
    Bs_arg = Bs.contiguous().view(-1)
    C_arg = C.contiguous().view(-1)

    _cached_launch(
        K_runtime,
        A.dtype,
        B.dtype,
        C.dtype,
        layout,
    )(
        A_arg,
        As_arg,
        B_arg,
        Bs_arg,
        C_arg,
        M_runtime,
        N_runtime,
        stream=stream,
    )


@functools.lru_cache(maxsize=None)
def _cached_launch(
    K: int,
    a_fp8_dtype: torch.dtype,
    b_fp8_dtype: torch.dtype,
    output_dtype: torch.dtype,
    layout: str,
):
    """Cache independent TN/NN/NT binaries with no runtime layout argument."""
    return _compile_kernel(
        K,
        a_fp8_dtype,
        b_fp8_dtype,
        output_dtype,
        layout,
    )


def _validate_common_payloads(
    a: torch.Tensor,
    b: torch.Tensor,
    D: torch.Tensor,
    *,
    layout: str,
):
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(
            f"FlyDSL MXFP8 {layout} expects rank-2 operands, got "
            f"a={tuple(a.shape)} and b={tuple(b.shape)}"
        )
    supported_fp8_dtypes = (torch.float8_e4m3fn, torch.float8_e5m2)
    if a.dtype not in supported_fp8_dtypes or b.dtype not in supported_fp8_dtypes:
        raise TypeError(
            f"FlyDSL MXFP8 {layout} expects E4M3 or E5M2 payloads "
            f"independently, got a={a.dtype} and b={b.dtype}"
        )
    if a.device != b.device or D.device != a.device:
        raise ValueError("A, B, and D must be on the same device")
    if D.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(
            "FlyDSL MXFP8 output must be float16, bfloat16, or float32, "
            f"got {D.dtype}"
        )


def mxfp8_matmul(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    D: torch.Tensor,
    stream=None,
    *,
    layout: str = "TN",
):
    """Normalize scale orientation and launch a compile-time layout binary.

    Wrapper-visible contracts:

        TN: a [M,K], b [N,K], scales [M,K/32] and [N,K/32]
        NN: a [M,K], b [K,N], scales [M,K/32] and [K/32,N]
        NT: a [K,M], b [K,N], scales [K/32,M] and [K/32,N]

    TN consumes the selected rowwise payloads directly. NN and NT preserve
    K-major payloads and use ``ds_read_b64_tr_b8`` inside their
    compile-time-specialized kernels.
    """
    if layout not in ("TN", "NN", "NT"):
        raise ValueError(f"Unsupported MXFP8 kernel layout: {layout}")

    _validate_common_payloads(a, b, D, layout=layout)

    if layout == "TN":
        m, k = a.shape
        n, kb = b.shape
    elif layout == "NN":
        m, k = a.shape
        kb, n = b.shape
    else:
        k, m = a.shape
        kb, n = b.shape

    if kb != k:
        raise ValueError(
            f"Incompatible MXFP8 {layout} operands: "
            f"A{tuple(a.shape)} and B{tuple(b.shape)}"
        )
    if tuple(D.shape) != (m, n):
        raise ValueError(f"D shape {tuple(D.shape)} != expected {(m, n)}")
    if k % SCALE_GROUP_SIZE != 0:
        raise ValueError(
            f"K={k} must be divisible by MXFP8 scale group size "
            f"{SCALE_GROUP_SIZE}"
        )

    if layout == "NT":
        expected_a_scale = (k // SCALE_GROUP_SIZE, m)
    else:
        expected_a_scale = (m, k // SCALE_GROUP_SIZE)

    if layout == "TN":
        expected_b_scale = (n, k // SCALE_GROUP_SIZE)
    else:
        expected_b_scale = (k // SCALE_GROUP_SIZE, n)

    if tuple(a_scale.shape) != expected_a_scale:
        raise ValueError(
            f"a_scale shape {tuple(a_scale.shape)} != expected "
            f"{expected_a_scale} for {layout}"
        )
    if tuple(b_scale.shape) != expected_b_scale:
        raise ValueError(
            f"b_scale shape {tuple(b_scale.shape)} != expected "
            f"{expected_b_scale} for {layout}"
        )
    if a_scale.dtype != torch.uint8 or b_scale.dtype != torch.uint8:
        raise TypeError("FlyDSL MXFP8 expects raw E8M0 scales as torch.uint8")
    if a_scale.device != a.device or b_scale.device != a.device:
        raise ValueError("A, B, scales, and D must be on the same device")

    if layout == "TN":
        # TN selected backings already match the normal-read kernel contract:
        #   a [M,K], b [N,K]
        a_kernel = a
        b_kernel = b
        a_scale_hk = pack_mx32_scales_for_hk(
            a_scale,
            source_colwise=False,
            stream=stream,
        )
        b_scale_hk = pack_mx32_scales_for_hk(
            b_scale,
            source_colwise=False,
            stream=stream,
        )
    elif layout == "NN":
        a_kernel = a
        b_kernel = b
        a_scale_hk = pack_mx32_scales_for_hk(
            a_scale,
            source_colwise=False,
            stream=stream,
        )
        b_scale_hk = pack_mx32_scales_for_hk(
            b_scale,
            source_colwise=True,
            stream=stream,
        )
    else:
        a_kernel = a
        b_kernel = b
        a_scale_hk = pack_mx32_scales_for_hk(
            a_scale,
            source_colwise=True,
            stream=stream,
        )
        b_scale_hk = pack_mx32_scales_for_hk(
            b_scale,
            source_colwise=True,
            stream=stream,
        )

    _debug(
        f"{layout} kernel inputs: a={tuple(a_kernel.shape)}, "
        f"b={tuple(b_kernel.shape)}, "
        f"a_scale_hk={tuple(a_scale_hk.shape)}, "
        f"b_scale_hk={tuple(b_scale_hk.shape)}, D={tuple(D.shape)}"
    )

    do_gemm(
        a_kernel,
        a_scale_hk,
        b_kernel,
        b_scale_hk,
        D.view(m, n),
        layout=layout,
        stream=stream,
    )
    return D


def mxfp8_matmul_nn(*args, **kwargs):
    """Compatibility entry point for the common NN specialization."""
    kwargs["layout"] = "NN"
    return mxfp8_matmul(*args, **kwargs)


def mxfp8_matmul_nt(*args, **kwargs):
    """Compatibility entry point for the common NT specialization."""
    kwargs["layout"] = "NT"
    return mxfp8_matmul(*args, **kwargs)


__all__ = [
    "BLOCK_M",
    "BLOCK_N",
    "BLOCK_K",
    "SCALE_GROUP_SIZE",
    "pack_mx32_scales_for_hk",
    "do_gemm",
    "mxfp8_matmul",
    "mxfp8_matmul_nn",
    "mxfp8_matmul_nt",
]
