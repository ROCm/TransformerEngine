# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Minimal byte-staging helpers for the first-pass BF16 four-wave GEMM."""

import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr

# ceildiv is the canonical cdiv from the shared layer
def cdiv(numer: int, denom: int) -> int:
    return (numer + denom - 1) // denom


ceildiv = cdiv

def divmod(a, b):
    return (a // b, a % b)


def swizzle_128(row, col_in_bytes):
    """HK 128-byte row XOR swizzle; ``col_in_bytes`` is a byte coordinate."""
    offset = row * 128 + col_in_bytes
    swizzle = ((offset % (16 * 128)) >> 8) << 4
    swizzled_offset = offset ^ swizzle
    return swizzled_offset // 128, swizzled_offset % 128


def make_bf16_byte_buffer_tensor(arg_u8):
    """Create a byte-addressed buffer tensor from a contiguous BF16 uint8 view."""
    return fx.rocdl.make_buffer_tensor(arg_u8, max_size=False)


def compute_global_swizzle(lane_id, wave_id, row_stride_bytes, n_rounds, preshuffled=False):
    offsets = []
    n_waves = fx.block_dim.x // 64
    for round in range_constexpr(n_rounds):
        if const_expr(preshuffled):
            raise AssertionError("BF16 first-pass port does not support preshuffled operands")
        row = lane_id // 8 + wave_id * 8 + round * (n_waves * 8)
        col_bytes = (lane_id % 8) * 16
        r, c = swizzle_128(row, col_bytes)
        offsets.append(r * row_stride_bytes + c)
    return offsets


class G2SLoader:
    """Issue raw 16-byte buffer-to-LDS copies.

    Both the global source and LDS destination must be byte-addressed. Fly's copy lowering does not legalize an i8 buffer source paired with a bf16 LDS
    destination even when the transfer width is the same 128 bits.
    """
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
        lds_ptr = fx.inttoptr(self.LdsPtr_t, base_i32 + fx.Int32(step_off))
        return fx.make_view(lds_ptr, fx.make_layout(1, 1))

    def load(self, lds_dst, byte_offset):
        for step in range_constexpr(self.n_load_steps):
            src = fx.slice(self.gl_src, (None, fx.Int32(self.gl_offsets[step])))
            fx.copy(self.g2lds_atom, src, self._lds_dst_at(lds_dst, step), soffset=fx.Int32(byte_offset))

    def load_one(self, lds_dst, byte_offset, step):
        src = fx.slice(self.gl_src, (None, fx.Int32(self.gl_offsets[step])))
        fx.copy(self.g2lds_atom, src, self._lds_dst_at(lds_dst, step), soffset=fx.Int32(byte_offset))


def pack_i32x4_i32x8(lo, hi):
    return lo.shuffle(hi, list(range(8)))


class S2RLoader:
    """Raw 16-byte LDS reader used to assemble an i32x8 BF16 K64 fragment."""
    def __init__(self, wave_idx, n_tiles):
        self.lane_id = fx.thread_idx.x % 64
        self.wave_idx = wave_idx
        self.n_tiles = n_tiles

    def _vec_load_16bytes(self, lds_src, offset):
        ptr_off = fx.add_offset(lds_src.ptr, fx.make_int_tuple(offset))
        i8_iter = fx.recast_iter(fx.Uint8, ptr_off)
        return fx.make_view(i8_iter, fx.make_layout(16, 1)).load()

    def load_one(self, lds_src, lds_offset):
        return self._vec_load_16bytes(lds_src, lds_offset).bitcast(fx.Int32)
