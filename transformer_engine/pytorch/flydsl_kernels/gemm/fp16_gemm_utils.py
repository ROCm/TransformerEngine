# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Byte-level staging helpers for the four-wave GEMM kernels.

These loaders and swizzle helpers operate on flat byte views and carry no dtype
label, so the half-precision (FP16/BF16) and FP32 cores all share them; each
core selects its own MFMA opcode. The module keeps its historical
``fp16_gemm_utils`` filename.
"""

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm, vector
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as as_mlir_value

# Dtype-independent primitives live in the shared module; re-export them so the
# GEMM kernels can keep importing them from this per-dtype module unchanged.
from .gemm_common_utils import (
    barrier,
    cdiv,
    ceildiv,
    divmod,
    encode_waitcnt,
    min,
    pack_i32x4_i32x8,
    swizzle_128,
    xcd_swizzle,
)


def make_byte_buffer_tensor(arg):
    """Create a BufferDesc directly from the wrapper-provided tensor.

    Dtype-independent: the operand is consumed as a flat byte view, so FP16,
    BF16, and FP32 all share this maker.
    """
    return fx.rocdl.make_buffer_tensor(arg, max_size=False)


def compute_global_swizzle(
    lane_id,
    wave_id,
    row_stride_bytes,
    n_rounds,
    preshuffled=False,
):
    offsets = []
    n_waves = fx.block_dim.x // 64
    for round in range_constexpr(n_rounds):
        if const_expr(preshuffled):
            raise AssertionError("16-bit first-pass port does not support preshuffled operands")
        row = lane_id // 8 + wave_id * 8 + round * (n_waves * 8)
        col_bytes = (lane_id % 8) * 16
        r, c = swizzle_128(row, col_bytes)
        offsets.append(r * row_stride_bytes + c)
    return offsets


def compute_global_transpose_swizzle(
    lane_id,
    wave_id,
    leading_dim_bytes,
    n_rounds,
):
    """Offsets for a K-major 16-bit source staged for ``ds_read_b64_tr_b16``.

    One 128-row output half-page is represented in LDS as two independent
    swizzled ``[K64, X64]`` slices. Each slice is 64 rows by 128 bytes, so the
    complete half-page remains 16 KiB and preserves the existing four-pass
    16-byte/thread DMA cadence.

    The returned offsets are relative to the source tile base:
      ``source[k, x_base]`` for a contiguous K-major 16-bit matrix.
    """
    offsets = []
    n_waves = fx.block_dim.x // 64
    for round in range_constexpr(n_rounds):
        linear_row = lane_id // 8 + wave_id * 8 + round * (n_waves * 8)
        col_bytes = (lane_id % 8) * 16

        slice_idx = linear_row // 64
        physical_k = linear_row % 64

        # XOR swizzle is self-inverse for this layout. Map the physical LDS
        # chunk back to its logical K/X-byte source coordinate.
        logical_k, logical_x_bytes = swizzle_128(physical_k, col_bytes)
        offsets.append(logical_k * leading_dim_bytes + slice_idx * 64 * 2 + logical_x_bytes)
    return offsets


class G2SLoader:
    """Issue native 16-byte BufferDesc-to-LDS copies."""

    def __init__(self, gl_src, gl_offsets, n_load_steps, lds_dtype, wave_id):
        self.g2lds_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)
        self.LdsPtr_t = fx.PointerType.get(lds_dtype, 2, 512)
        self.gl_src = gl_src
        self.gl_offsets = gl_offsets
        self.n_load_steps = n_load_steps
        self.wave_id = wave_id
        self.n_waves = fx.block_dim.x // 64

    def _lds_dst_at(self, lds_dst, step):
        step_off = self.wave_id * 1024 + step * (self.n_waves * 1024)
        base_i32 = fx.Int32(fx.ptrtoint(lds_dst.ptr))
        lds_ptr = fx.inttoptr(
            self.LdsPtr_t,
            base_i32 + fx.Int32(step_off),
        )
        return fx.make_view(lds_ptr, fx.make_layout(1, 1))

    def load(self, lds_dst, byte_offset):
        for step in range_constexpr(self.n_load_steps):
            src = fx.slice(
                self.gl_src,
                (None, fx.Int32(self.gl_offsets[step])),
            )
            fx.copy(
                self.g2lds_atom,
                src,
                self._lds_dst_at(lds_dst, step),
                soffset=fx.Int32(byte_offset),
            )

    def load_one(self, lds_dst, byte_offset, step):
        src = fx.slice(
            self.gl_src,
            (None, fx.Int32(self.gl_offsets[step])),
        )
        fx.copy(
            self.g2lds_atom,
            src,
            self._lds_dst_at(lds_dst, step),
            soffset=fx.Int32(byte_offset),
        )


class S2RLoader:
    """LDS readers used to assemble 16-bit K64 fragments."""

    def __init__(self, wave_idx, n_tiles):
        self.lane_id = fx.thread_idx.x % 64
        self.wave_idx = wave_idx
        self.n_tiles = n_tiles

    def _vec_load_16bytes(self, lds_src, offset):
        ptr_off = fx.add_offset(lds_src.ptr, fx.make_int_tuple(offset))
        i8_iter = fx.recast_iter(fx.Uint8, ptr_off)
        return fx.make_view(i8_iter, fx.make_layout(16, 1)).load()

    def load_one(self, lds_src, lds_offset):
        return self._vec_load_16bytes(
            lds_src,
            lds_offset,
        ).bitcast(fx.Int32)

    def _ds_read_b64_tr_b16(
        self,
        lds_src,
        byte_offset,
        immediate_offset=0,
    ):
        """Issue one gfx950 ``ds_read_b64_tr_b16`` and return i32x2."""
        if immediate_offset == 0:
            asm = "ds_read_b64_tr_b16 $0, $1 offset:0\n"
        elif immediate_offset == 0x1000:
            asm = "ds_read_b64_tr_b16 $0, $1 offset:4096\n"
        else:
            raise ValueError(
                "ds_read_b64_tr_b16 supports immediate offsets 0 and 0x1000, "
                f"got {immediate_offset:#x}"
            )

        base_i32 = fx.Int32(fx.ptrtoint(lds_src.ptr))
        addr_i32 = base_i32 + fx.Int32(byte_offset)
        raw_type = ir.VectorType.get(
            [2],
            ir.IntegerType.get_signless(32),
        )
        raw = _llvm.inline_asm(
            raw_type,
            [as_mlir_value(addr_i32)],
            asm,
            "=v,v,~{memory}",
            has_side_effects=True,
        )
        return Vec(
            vector.BitCastOp(raw_type, raw).result,
            (2,),
            fx.Int32,
        )

    def load_one_transpose(
        self,
        lds_src,
        first_byte_offset,
        second_byte_offset,
        immediate_offset=0,
    ):
        """Return one i32x4 K32 16-bit fragment from two transpose reads."""
        lo = self._ds_read_b64_tr_b16(
            lds_src,
            first_byte_offset,
            immediate_offset,
        )
        hi = self._ds_read_b64_tr_b16(
            lds_src,
            second_byte_offset,
            immediate_offset,
        )
        return lo.shuffle(hi, [0, 1, 2, 3])
