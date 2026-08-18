# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Adapted by AMD from the FlyDSL project's GEMM utility helpers.

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm, vector
from flydsl._mlir.dialects.fly_rocdl import TargetAddressSpace
from flydsl.expr import const_expr, range_constexpr, rocdl
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


def make_fp8_buffer_tensor(arg_i8, fp8_ir_t):
    # max_size=False with no num_records_bytes: cosize(layout) becomes a
    # runtime expression because TensorAdaptor defaults to layout-dynamic
    # memref (post #554), so the descriptor adapts to the actual tensor
    # extent and no longer bakes the first-call's shape into IR.
    t_i8 = fx.rocdl.make_buffer_tensor(arg_i8, max_size=False)
    iter_i8 = fx.get_iter(t_i8)
    f8_buf_ptr_ty = fx.PointerType.get(
        elem_ty=fp8_ir_t,
        address_space=TargetAddressSpace.BufferDesc,
        alignment=fx.PointerType(iter_i8.type).alignment,
    )
    iter_f8 = fx.recast_iter(f8_buf_ptr_ty, iter_i8)
    return fx.Tensor(fx.make_view(iter_f8, fx.get_layout(t_i8)))


# Returns, for one lane (lane_id, wave_id), a list of n_rounds swizzled flat
# global offsets indexed by DMA pass: offsets[step] = r*K + c where (r,c) =
# swizzle_128(row, col). Each is the static per-thread/per-pass source of one
# 16-byte load; the dynamic K-tile base is added later as soffset. K is the
# global row stride (a_leading_dim / b_leading_dim), not the 128 tile width.
def compute_global_swizzle(lane_id, wave_id, K, n_rounds, preshuffled):
    offsets = []
    n_waves = fx.block_dim.x // 64
    for round in range_constexpr(n_rounds):
        if const_expr(preshuffled):
            row = lane_id % 8 + wave_id * 8 + round * (n_waves * 8)
            col = (lane_id // 8) * 16
            offsets.append(
                (row // 16) * (K * 16)
                + (row % 16) * 16
                + (col // 64) * 1024
                + ((col % 64) // 16) * 256
                + (col % 16)
            )
        else:
            row = lane_id // 8 + wave_id * 8 + round * (n_waves * 8)
            col = (lane_id % 8) * 16
            r, c = swizzle_128(row, col)
            offsets.append(r * K + c)
    return offsets


class G2SLoader:
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
        sum_i32 = base_i32 + fx.Int32(step_off)
        lds_ptr = fx.inttoptr(self.LdsPtr_t, sum_i32)
        return fx.make_view(lds_ptr, fx.make_layout(1, 1))

    def load(self, lds_dst, k_offset):
        for step in range_constexpr(self.n_load_steps):
            src = fx.slice(self.gl_src, (None, fx.Int32(self.gl_offsets[step])))
            dst = self._lds_dst_at(lds_dst, step)
            fx.copy(self.g2lds_atom, src, dst, soffset=fx.Int32(k_offset))

    def load_one(self, lds_dst, k_offset, step):
        src = fx.slice(self.gl_src, (None, fx.Int32(self.gl_offsets[step])))
        dst = self._lds_dst_at(lds_dst, step)
        fx.copy(self.g2lds_atom, src, dst, soffset=fx.Int32(k_offset))


class S2RLoader:
    def __init__(self, wave_idx, n_tiles):
        self.lane_id = fx.thread_idx.x % 64
        self.wave_idx = wave_idx
        self.n_tiles = n_tiles

    def _vec_load_16xf8(self, lds_src, offset):
        off_tup = fx.make_int_tuple(offset)
        ptr_off = fx.add_offset(lds_src.ptr, off_tup)
        i8_iter = fx.recast_iter(fx.Uint8, ptr_off)
        view = fx.make_view(i8_iter, fx.make_layout(16, 1))
        return view.load()

    def _vec_load_1xf8(self, lds_src, offset):
        """Naive one-byte LDS load with direct dynamic byte addressing.

        Avoid ``make_int_tuple`` entirely because this FlyDSL build cannot
        reliably infer tuple types from dynamic Index expressions.
        """
        base_i32 = fx.Int32(fx.ptrtoint(lds_src.ptr))
        addr_i32 = base_i32 + fx.Int32(offset)
        i8_lds_ptr_t = fx.PointerType.get(
            elem_ty=ir.IntegerType.get_signless(8),
            address_space=2,
            alignment=1,
        )
        i8_ptr = fx.inttoptr(i8_lds_ptr_t, addr_i32)
        view = fx.make_view(i8_ptr, fx.make_layout(1, 1))
        return view.load()

    def load(self, lds_src, preshuffled=False):
        frag = []
        for i in range_constexpr(self.n_tiles):
            halves = []
            row = self.wave_idx * (self.n_tiles * 16) + i * 16 + self.lane_id % 16
            for step in range_constexpr(2):
                col = (self.lane_id // 16) * 16 + step * 64
                if const_expr(preshuffled):
                    offset = (row // 8) * 1024 + (row % 8) * 16 + (col // 16) * 128
                else:
                    row_swz, col_swz = swizzle_128(row, col)
                    offset = row_swz * 128 + col_swz
                v = self._vec_load_16xf8(lds_src, offset)
                halves.append(v.bitcast(fx.Int32))
            frag.append(pack_i32x4_i32x8(halves[0], halves[1]))
        return frag

    def load_one(self, lds_src, lds_offset):
        v = self._vec_load_16xf8(lds_src, lds_offset)
        return v.bitcast(fx.Int32)

    def _ds_read_b64_tr_b8(self, lds_src, byte_offset, immediate_offset=0):
        """Issue one gfx950 ``ds_read_b64_tr_b8`` and return i32x2.

        ``immediate_offset`` is encoded in the DS instruction itself. The NN
        K128 path uses 0 and 0x2000, where 0x2000 advances the logical K row by
        64 in a 128-byte-wide physical LDS image.
        """
        if immediate_offset == 0:
            asm = "ds_read_b64_tr_b8 $0, $1 offset:0\n"
        elif immediate_offset == 0x2000:
            asm = "ds_read_b64_tr_b8 $0, $1 offset:8192\n"
        else:
            raise ValueError(
                "ds_read_b64_tr_b8 supports immediate offsets 0 and 0x2000, "
                f"got {immediate_offset:#x}"
            )

        base_i32 = fx.Int32(fx.ptrtoint(lds_src.ptr))
        addr_i32 = base_i32 + fx.Int32(byte_offset)
        raw_type = ir.VectorType.get([2], ir.IntegerType.get_signless(32))
        raw = _llvm.inline_asm(
            raw_type,
            [as_mlir_value(addr_i32)],
            asm,
            "=v,v,~{memory}",
            has_side_effects=True,
        )
        return Vec(vector.BitCastOp(raw_type, raw).result, (2,), fx.Int32)

    def load_one_transpose(
        self,
        lds_src,
        first_byte_offset,
        second_byte_offset,
        immediate_offset=0,
    ):
        """Load one 16-byte portion of a K128 FP8 MFMA operand.

        Two transpose reads return four packed i32 values. Calling this once
        with immediate 0 and once with immediate 0x2000 yields the two i32x4
        portions that concatenate into the production i32x8 MFMA fragment.
        """
        lo = self._ds_read_b64_tr_b8(
            lds_src,
            first_byte_offset,
            immediate_offset,
        )
        hi = self._ds_read_b64_tr_b8(
            lds_src,
            second_byte_offset,
            immediate_offset,
        )
        return lo.shuffle(hi, [0, 1, 2, 3])
