# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Adapted from Primus-Turbo (https://github.com/AMD-AGI/Primus-Turbo):
#   primus_turbo/flydsl/utils/gemm_helper.py
#     (S2R/MFMA/StoreCBf16, wait_barrier, xcd_remap_pid, compute_global_swizzle_bf16)
#   primus_turbo/flydsl/gemm/gemm_bf16_kernel.py
#     (dense_mma_pipeline_bf16, NN tile logic -> gemm_bf16_nn_gather_tile)
#
# See LICENSE for license information.

"""FlyDSL helpers for permute-free MoE grouped GEMM (Mega 8-wave / 32x32x16).

Shared building blocks for ``pf_fwd.py`` and ``pf_dgrad.py``. Dense 4-wave BF16
utilities live in ``fp16_gemm_utils.py``. The Mega 8-wave pipelined MMA loop
lives here as ``dense_mma_pipeline_bf16`` (independent of the 4-wave
``half_prec_gemm`` core), with the NT/NN gather tiles
(``gemm_bf16_nt_gather_tile``, ``gemm_bf16_nn_gather_tile``). Each gather tile carries a
compile-time ``gather`` flag selecting the A-gather variant (FC1 fwd / FC2 dgrad) or the
contiguous route-read/dense variant (FC2 fwd / FC1 dgrad).
"""

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly as fly_dialect
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import ArithValue
from flydsl.expr import arith, const_expr, range_constexpr, rocdl

from .fp16_gemm_utils import G2SLoader, ceildiv, make_byte_buffer_tensor, swizzle_128

# Mega 8-wave K-tile. Independent of the 4-wave ``half_prec_gemm`` BLOCK_K.
BLOCK_K = 64


def _inttoptr_lds(byte_addr):
    """Integer byte address -> !llvm.ptr<3> (LDS). Parsed per call: the type is
    bound to the current MLIRContext and cannot be cached across compiles."""
    return _llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), _raw(fx.Int64(byte_addr)))


def _lds_ptr_from_i32(addr_i32, byte_offset=0):
    """Build an LDS pointer (ptr<3>) from an i32 byte address + optional static offset."""
    ptr = _inttoptr_lds(ArithValue(addr_i32).extui(T.i64))
    if byte_offset != 0:
        ptr = _llvm.getelementptr(
            ir.Type.parse("!llvm.ptr<3>"), ptr, [], [int(byte_offset)], T.i8, None
        )
    return ptr


def _packed_ds_read_tr16(base_ptr, byte_offsets):
    n = len(byte_offsets)
    v2i32 = ir.VectorType.get([2], ir.IntegerType.get_signless(32))
    struct_t = _llvm.StructType.get_literal([v2i32] * n)
    asm = "\n".join(f"ds_read_b64_tr_b16 ${k}, ${n} offset:{byte_offsets[k]}" for k in range(n))
    constraints = ",".join(["=&v"] * n + ["v"] + ["~{memory}"])
    op = _llvm.InlineAsmOp(
        res=struct_t,
        operands_=[_raw(base_ptr)],
        asm_string=asm,
        constraints=constraints,
        has_side_effects=True,
    )
    return [Vec(_llvm.extractvalue(v2i32, op.result, [k])).bitcast(fx.BFloat16) for k in range(n)]


class _S2RLoaderBase:
    """Shared ctor for LDS->register operand loaders: caches the per-lane id,
    this wave's tile index, and the tile count."""

    def __init__(self, wave_idx, n_tiles):
        self.lane_id = fx.thread_idx.x % 64
        self.wave_idx = wave_idx
        self.n_tiles = n_tiles


class _S2RLoaderBf16(_S2RLoaderBase):
    """Shared skeleton for the bf16 operand loaders (cf. _MfmaBf16): n_tiles output
    tiles, each a list of k-sub fragments. Subclasses supply the per-sub offset table
    and _tile(), which holds the LDS address math -- transposed ds_read_tr_b16 or
    swizzled buffer load (too different to share beyond this loop)."""

    def load(self, lds_src):
        return [self._tile(lds_src, i) for i in range_constexpr(self.n_tiles)]


def _read_tr16_sub(base_i32, sub16, row_off):
    """One tr16 sub-block (512 elems/block): packed double-read, the pair 128 bytes
    apart, at sub16*512 + row_off, then assembled."""
    ptr = _lds_ptr_from_i32(base_i32 + (sub16 * 512 + row_off) * 2)
    r0, r1 = _packed_ds_read_tr16(ptr, [0, 128])
    return r0.shuffle(r1, list(range(8)))


class S2RLoaderTrBf16(_S2RLoaderBf16):
    """mfma_f32_32x32x16 operand via ds_read_tr_b16 transpose. Like S2RLoaderTr's
    _K_BASE, _SUB lists the tr16 sub-block of each inst_k=16 mfma step (consecutive
    here); its length is both the sub count and the per-tile block stride."""

    _SUB = (0, 1, 2, 3)

    def _tile(self, lds_src, i):
        m, kblk = self.lane_id % 32, self.lane_id // 32
        row_off = (m // 16) * 256 + kblk * 128 + (m % 16) * 4
        base_i32 = fx.Int32(fx.ptrtoint(lds_src.ptr))
        sub0 = (self.wave_idx * self.n_tiles + i) * len(self._SUB)
        return [
            _read_tr16_sub(base_i32, sub0 + self._SUB[c], row_off)
            for c in range_constexpr(len(self._SUB))
        ]


def _load8_bf16(lds_src, byte_off):
    i8 = fx.recast_iter(fx.Uint8, lds_src.ptr)
    p = fx.add_offset(i8, fx.make_int_tuple(byte_off))
    v = fx.make_view(p, fx.make_layout(16, 1)).load()
    return v.bitcast(fx.BFloat16)


class S2RLoaderBf16(_S2RLoaderBf16):
    """mfma_f32_32x32x16 operand (swizzled, non-transposed). Mirroring S2RLoaderTr,
    _K_BASE lists the K-column (elems) of each inst_k=16 sub of a 32-row tile; its
    length is the sub count -- no BLOCK_K needed."""

    _K_BASE = (0, 16, 32, 48)

    def _tile(self, lds_src, i):
        m, kblk = self.lane_id % 32, self.lane_id // 32
        row = self.wave_idx * (self.n_tiles * 32) + i * 32 + m
        subs = []
        for c in range_constexpr(len(self._K_BASE)):
            col_byte = (self._K_BASE[c] + kblk * 8) * 2
            _, cs = swizzle_128(row, col_byte)
            subs.append(_load8_bf16(lds_src, row * 128 + cs))
        return subs


class _MfmaBf16:
    """Grouped bf16 mfma: accumulate n_tiles_a x n_tiles_b output tiles. The k-sub
    count is taken from each operand's fragment list (len(a[i])), so the atom's
    (m, n, inst_k) is the only shape this class needs -- no BLOCK_K coupling."""

    def __init__(self, n_tiles_a, n_tiles_b, m, n, inst_k):
        self.atom = fx.make_mma_atom(fx.rocdl.MFMA(m, n, inst_k, fx.BFloat16))
        acc_len = m * n // 64  # f32 accum lanes per wave
        self.accum_type = Vec.make_type(acc_len, fx.Float32)
        self.zero_value = Vec.filled(acc_len, 0.0, fx.Float32)
        self.n_tiles_a = n_tiles_a
        self.n_tiles_b = n_tiles_b

    def idx(self, i, j):
        return i * self.n_tiles_b + j

    def call(self, a, b, c):
        assert len(a) == self.n_tiles_a
        assert len(b) == self.n_tiles_b
        for i in range_constexpr(self.n_tiles_a):
            for j in range_constexpr(self.n_tiles_b):
                acc = c[self.idx(i, j)]
                for ks in range_constexpr(len(a[i])):
                    acc = fly_dialect.mma_atom_call_ssa(
                        [self.accum_type], self.atom, a[i][ks], b[j][ks], acc
                    )
                c[self.idx(i, j)] = acc
        return c


class Mfma32x32x16(_MfmaBf16):
    def __init__(self, n_tiles_a, n_tiles_b):
        super().__init__(n_tiles_a, n_tiles_b, 32, 32, 16)


class StoreCBf16:
    def __init__(self, C, c_rows, c_cols, out_ty, cache_modifier=0):
        if cache_modifier:
            raise NotImplementedError(
                "StoreCBf16 no longer supports a cache_modifier V# store path"
            )
        self.c_rows = c_rows
        self.c_cols = c_cols
        self.lane_id = fx.thread_idx.x % 64
        self.out_ty = out_ty
        c_nbytes = c_rows * c_cols * 2
        gC = fx.rocdl.make_buffer_tensor(C, max_size=False, num_records_bytes=c_nbytes)
        self.c_div = fx.logical_divide(gC, fx.make_layout(1, 1))
        self.out_atom_1 = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), out_ty)
        self.reg_out_1 = fx.make_rmem_tensor(fx.make_layout(1, 1), out_ty)
        self.oob = fx.Int32(c_rows * c_cols)  # out-of-bounds sink index

    def _store_masked(self, value, c_index, valid):
        """Store one element to c_index (masked to the OOB sink when invalid)."""
        idx = arith.select(valid, c_index, self.oob)
        val = value.to(self.out_ty)
        fx.memref_store_vec(Vec.filled(1, val, self.out_ty), self.reg_out_1)
        fx.copy(self.out_atom_1, self.reg_out_1, fx.slice(self.c_div, (None, fx.Int32(idx))))

    def store(self, c_frag, base_row, base_col):
        n = self.lane_id % 32
        m_hi = (self.lane_id // 32) * 4
        col = base_col + n
        col_valid = col < self.c_cols
        for ti in range_constexpr(len(c_frag)):
            acc = Vec(c_frag[ti])
            for r in range_constexpr(16):
                row = base_row + ti * 32 + (r // 4) * 8 + m_hi + (r % 4)
                self._store_masked(acc[r], row * self.c_cols + col, col_valid)


class RouteI32Loader:
    """Copy-atom scalar i32 loader for a 1-D route/index tensor.

    Replaces the removed ``buffer_load(rsrc, off, vec_width=1, dtype=i32)``: a
    ``max_size`` buffer tensor so an out-of-range slot reads OOB (hardware 0),
    and a linear ``logical_divide`` so the slice coordinate is the element index.
    Build once per tensor; the copy atom is shared and a fresh register is
    allocated per load (a shared register would alias across live loads).
    """

    def __init__(self, tensor):
        g = fx.rocdl.make_buffer_tensor(tensor, max_size=True)
        self.div = fx.logical_divide(g, fx.make_layout(1, 1))
        self.atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)

    def load(self, off):
        reg = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Int32)
        fx.copy(self.atom, fx.slice(self.div, (None, fx.Int32(off))), reg)
        return fx.Int32(fx.memref_load_vec(reg)[0])


def buffer_load_i32(loader, off):
    """Load one i32 scalar from a :class:`RouteI32Loader` at element ``off``."""
    return loader.load(off)


def _global_swizzle_bf16(lane_id, wave_id, K, n_rounds, src_row):
    """Per-lane global A element offsets for the flat-buffer bf16 (gather) pipeline.

    ``src_row`` is a callable mapping the contiguous *tile* row to its global source row;
    the public variants below differ only in that mapping. The bank swizzle (``c``) stays
    keyed on the tile row so the LDS destination layout -- and therefore the S2R
    transpose-read -- is unchanged; only the global fetch address is redirected. Any index
    loads happen once (K-invariant) and amortize over the whole K-loop.
    """
    offsets = []
    n_waves = fx.block_dim.x // 64
    for r in range_constexpr(n_rounds):
        row = lane_id // 8 + wave_id * 8 + r * (n_waves * 8)
        col_byte = (lane_id % 8) * 16
        _, c = swizzle_128(row, col_byte)
        offsets.append(src_row(row) * K + c // 2)
    return offsets


def compute_global_swizzle_bf16(lane_id, wave_id, K, n_rounds):
    """Contiguous (non-gathering) grouped-GEMM A offsets: source row == tile row."""
    return _global_swizzle_bf16(lane_id, wave_id, K, n_rounds, lambda row: row)


def compute_global_gather_swizzle_bf16(lane_id, wave_id, K, n_rounds, sorted_res, sorted_row_base):
    """Per-lane global A offsets for a *gathering* grouped GEMM.

    Identical to :func:`compute_global_swizzle_bf16` except the tile row is redirected through a
    gather index: the source row for tile row ``row`` is ``SORTED_IDS[sorted_row_base + row]``
    instead of the contiguous pool row. The bank swizzle (``c``) is still keyed on the *tile* row
    so the LDS destination layout (and therefore the S2R transpose-read) is unchanged -- only the
    global fetch address is redirected. The index loads happen once (K-invariant), so they are
    amortized over the whole K-loop.
    """
    return _global_swizzle_bf16(
        lane_id,
        wave_id,
        K,
        n_rounds,
        lambda row: buffer_load_i32(sorted_res, sorted_row_base + row),
    )


def compute_global_identity_swizzle_bf16(lane_id, wave_id, K, n_rounds, sorted_row_base):
    """Per-lane global A offsets reading the *route row directly* (identity gather).

    Same flat-buffer pipeline as :func:`compute_global_gather_swizzle_bf16` but the source row is
    ``sorted_row_base + row`` computed arithmetically instead of loaded from a gather table. This
    is the FC2 route-read (``index_a_by_route_pos``) path: it needs no ``sorted_slot_ids`` tensor,
    so it stays inside a HIP graph capture (no per-call identity allocation) while reusing the
    proven whole-buffer / base-0 gather tile (avoiding the per-tile A-rebase multi-block hazard).
    """
    return _global_swizzle_bf16(
        lane_id, wave_id, K, n_rounds, lambda row: sorted_row_base + row
    )


def compute_global_swizzle_nn_bf16(lane_id, wave_id, c_n, n_steps):
    offsets = []
    n_waves = fx.block_dim.x // 64
    kk = (lane_id % 32) // 2
    g = lane_id // 32
    n_in = g * 16 + (lane_id % 2) * 8
    for step in range_constexpr(n_steps):
        idx = wave_id + step * n_waves
        n_tile = idx // 4
        ks = idx % 4
        offsets.append((ks * 16 + kk) * c_n + n_tile * 32 + n_in)
    return offsets


def xcd_remap_pid(pid, total_pids, num_xcd):
    """Remap the tile id so same-XCD workgroups gather into one contiguous
    block, keeping each XCD's L2 reuse within that XCD. Bijection over
    [0, total_pids); identity when num_xcd <= 1."""
    if num_xcd <= 1:
        return pid
    per_xcd = total_pids // num_xcd  # floor
    rem = total_pids - per_xcd * num_xcd
    xcd = pid % num_xcd
    local = pid // num_xcd
    offset = xcd * per_xcd + arith.select(xcd < rem, xcd, rem)
    return offset + local


def _make_shared_storage(BLOCK_M, BLOCK_N):
    a_lds_size = (BLOCK_M // 2) * BLOCK_K
    b_lds_size = (BLOCK_N // 2) * BLOCK_K

    @fx.struct
    class SharedStorage:
        A_lds_cur_0: fx.Array[fx.BFloat16, a_lds_size, 16]
        A_lds_cur_1: fx.Array[fx.BFloat16, a_lds_size, 16]
        A_lds_next_0: fx.Array[fx.BFloat16, a_lds_size, 16]
        A_lds_next_1: fx.Array[fx.BFloat16, a_lds_size, 16]
        B_lds_cur_0: fx.Array[fx.BFloat16, b_lds_size, 16]
        B_lds_cur_1: fx.Array[fx.BFloat16, b_lds_size, 16]
        B_lds_next_0: fx.Array[fx.BFloat16, b_lds_size, 16]
        B_lds_next_1: fx.Array[fx.BFloat16, b_lds_size, 16]

    return SharedStorage


def wait_barrier(count):
    _llvm.inline_asm(
        res=None,
        operands_=[],
        asm_string=f"s_waitcnt vmcnt({count})\ns_barrier",
        constraints="",
        has_side_effects=True,
    )


@ASTRewriter.transform
def dense_mma_pipeline_bf16(
    lds,
    a_g2s,
    b_g2s,
    a_s2r,
    b_s2r,
    mfma,
    store_c,
    A0_gl_offset,
    A1_gl_offset,
    B0_gl_offset,
    B1_gl_offset,
    a_k_step,
    b_k_step,
    block_m,
    block_n,
    wave_m,
    wave_n,
    K,
    BLOCK_M,
    BLOCK_N,
    nt_vmcnt,
    a_g2s_hi=None,
):
    """Shared 4-quadrant pipelined MMA loop + store epilogue for the Mega bf16 tile.

    ``a_g2s_hi`` (optional): a second A global->LDS loader used only for the upper
    LDS half-tile. Defaults to ``a_g2s``. A gathering GEMM passes a distinct loader
    whose per-lane offsets are redirected through the gather index for the upper rows.
    """
    a_g2s_hi = a_g2s if a_g2s_hi is None else a_g2s_hi
    K_ITERS = K // BLOCK_K
    assert K_ITERS >= 2, f"K_ITERS={K_ITERS} too small; need K >= {2 * BLOCK_K}"
    N_TILES_A = BLOCK_M // 128
    N_TILES_B = BLOCK_N // 256
    N_ACCUMS = N_TILES_A * N_TILES_B
    LDS_BLOCK_M = BLOCK_M // 2
    LDS_BLOCK_N = BLOCK_N // 2
    N_LDS_STEPS_A = LDS_BLOCK_M // 64
    N_LDS_STEPS_B = LDS_BLOCK_N // 64

    a_cur0 = lds.A_lds_cur_0
    a_cur1 = lds.A_lds_cur_1
    a_next0 = lds.A_lds_next_0
    a_next1 = lds.A_lds_next_1
    b_cur0 = lds.B_lds_cur_0
    b_cur1 = lds.B_lds_cur_1
    b_next0 = lds.B_lds_next_0
    b_next1 = lds.B_lds_next_1

    c00_frag = [mfma.zero_value] * N_ACCUMS
    c01_frag = [mfma.zero_value] * N_ACCUMS
    c10_frag = [mfma.zero_value] * N_ACCUMS
    c11_frag = [mfma.zero_value] * N_ACCUMS

    b_g2s.load(b_cur0, B0_gl_offset + 0 * b_k_step)
    a_g2s.load(a_cur0, A0_gl_offset + 0 * a_k_step)
    b_g2s.load(b_cur1, B1_gl_offset + 0 * b_k_step)
    a_g2s_hi.load(a_cur1, A1_gl_offset + 0 * a_k_step)

    if wave_m == 1:
        rocdl.s_barrier()
    wait_barrier(N_LDS_STEPS_A + N_LDS_STEPS_B)

    b_g2s.load(b_next0, B0_gl_offset + 1 * b_k_step)
    a_g2s.load(a_next0, A0_gl_offset + 1 * a_k_step)
    b_g2s.load(b_next1, B1_gl_offset + 1 * b_k_step)

    wait_barrier(N_LDS_STEPS_A + 2 * N_LDS_STEPS_B)

    for k in range_constexpr(K_ITERS - 2):
        b0_frag = b_s2r.load(b_cur0)
        a0_frag = a_s2r.load(a_cur0)
        a_g2s_hi.load(a_next1, A1_gl_offset + (k + 1) * a_k_step)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        b1_frag = b_s2r.load(b_cur1)
        b_g2s.load(b_cur0, B0_gl_offset + (k + 2) * b_k_step)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        a1_frag = a_s2r.load(a_cur1)
        a_g2s.load(a_cur0, A0_gl_offset + (k + 2) * a_k_step)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        b_g2s.load(b_cur1, B1_gl_offset + (k + 2) * b_k_step)
        wait_barrier(2 * N_LDS_STEPS_A + N_LDS_STEPS_B)

        rocdl.s_setprio(1)
        c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        if const_expr(nt_vmcnt >= 0):
            _llvm.inline_asm(
                res=None,
                operands_=[],
                asm_string=f"s_waitcnt vmcnt({nt_vmcnt})",
                constraints="",
                has_side_effects=True,
            )
        a_cur0, a_next0 = a_next0, a_cur0
        a_cur1, a_next1 = a_next1, a_cur1
        b_cur0, b_next0 = b_next0, b_cur0
        b_cur1, b_next1 = b_next1, b_cur1

    k = K_ITERS - 2
    b0_frag = b_s2r.load(b_cur0)
    a0_frag = a_s2r.load(a_cur0)
    rocdl.s_barrier()
    rocdl.s_setprio(1)
    c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
    rocdl.s_setprio(0)
    rocdl.s_barrier()

    b1_frag = b_s2r.load(b_cur1)
    rocdl.s_barrier()
    rocdl.s_setprio(1)
    c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)
    rocdl.s_setprio(0)
    rocdl.s_barrier()

    a1_frag = a_s2r.load(a_cur1)
    rocdl.s_barrier()
    rocdl.s_setprio(1)
    c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)
    rocdl.s_setprio(0)
    rocdl.s_barrier()

    b0_frag = b_s2r.load(b_next0)
    a_g2s_hi.load(a_next1, A1_gl_offset + (k + 1) * a_k_step)
    rocdl.s_barrier()
    rocdl.s_setprio(1)
    c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)
    rocdl.s_setprio(0)
    rocdl.s_barrier()

    a_cur0, a_next0 = a_next0, a_cur0
    a_cur1, a_next1 = a_next1, a_cur1
    b_cur0, b_next0 = b_next0, b_cur0
    b_cur1, b_next1 = b_next1, b_cur1

    a0_frag = a_s2r.load(a_cur0)
    wait_barrier(0)
    rocdl.s_setprio(1)
    c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
    rocdl.s_setprio(0)
    rocdl.s_barrier()

    b1_frag = b_s2r.load(b_cur1)
    rocdl.s_barrier()
    rocdl.s_setprio(1)
    c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)
    rocdl.s_setprio(0)
    rocdl.s_barrier()

    a1_frag = a_s2r.load(a_cur1)
    rocdl.s_barrier()
    rocdl.s_setprio(1)
    c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)
    c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)
    rocdl.s_setprio(0)
    rocdl.s_barrier()

    wave_n_offset = wave_n * (N_TILES_B * 32)
    wave_m_offset = wave_m * (N_TILES_A * 32)
    base_row = block_m * BLOCK_M + wave_m_offset
    base_col = block_n * BLOCK_N + wave_n_offset
    store_c.store(c00_frag, base_row + 0, base_col + 0)
    store_c.store(c01_frag, base_row + 0, base_col + LDS_BLOCK_N)
    store_c.store(c10_frag, base_row + LDS_BLOCK_M, base_col + 0)
    store_c.store(c11_frag, base_row + LDS_BLOCK_M, base_col + LDS_BLOCK_N)


def gemm_bf16_nt_gather_tile(
    A,
    B_T,
    C,
    c_m,
    c_n,
    lds,
    sorted_res,
    sorted_row_base,
    block_n,
    *,
    K,
    BLOCK_M,
    BLOCK_N,
    out_fp16=False,
    nt_vmcnt=3,
    b_group_base,
    gather=True,
):
    """One NT tile of the grouped GEMM with the A rows *gathered* via ``sorted_slot_ids``.

    Identical to Primus-Turbo / Mega ``gemm_bf16_nt_tile`` except:
      * the two LDS A half-tiles use *gather* swizzles (rows redirected through
        ``sorted_slot_ids[sorted_row_base + tile_row]``), so the A base offset is 0 and only the
        K-step rides the load ``soffset``;
      * ``C`` is already rebased to this tile's row block by the caller, so the store block_m is 0.
    """
    assert BLOCK_M >= 128 and BLOCK_N >= 256 and BLOCK_M % 128 == 0 and BLOCK_N % 256 == 0
    assert K % BLOCK_K == 0, f"bf16 NT gather needs K % {BLOCK_K} == 0 (got K={K})"
    N_TILES_A = BLOCK_M // 128
    N_TILES_B = BLOCK_N // 256
    LDS_BLOCK_M = BLOCK_M // 2
    LDS_BLOCK_N = BLOCK_N // 2
    N_LDS_STEPS_A = LDS_BLOCK_M // 64
    N_LDS_STEPS_B = LDS_BLOCK_N // 64
    N_LDS_ROUNDS = max(N_LDS_STEPS_A, N_LDS_STEPS_B)

    lane_id = fx.thread_idx.x % 64
    wave_id = fx.thread_idx.x // 64
    wave_m = wave_id // 4
    wave_n = wave_id % 4

    # A rows carried by the gather offsets -> global base is 0 (K rides soffset only).
    A0_gl_offset = fx.Int32(0)
    A1_gl_offset = fx.Int32(0)
    B0_gl_offset = (block_n * BLOCK_N) * K
    B1_gl_offset = (block_n * BLOCK_N + LDS_BLOCK_N) * K
    if b_group_base is not None:
        B0_gl_offset = B0_gl_offset + b_group_base
        B1_gl_offset = B1_gl_offset + b_group_base

    # ``A`` MUST be a flat 1D buffer view (built by the caller): a linear gather offset
    # (src_row*K + col) indexes row-major elements. A raw 2D tensor's logical_divide/slice
    # indexes the outer (row) dim, so a flat offset runs off the end -> garbage.
    gA = make_byte_buffer_tensor(A)
    gB = make_byte_buffer_tensor(B_T)
    a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
    b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

    # Two gather swizzles: the lo half covers tile rows [0, LDS_BLOCK_M), the hi half
    # [LDS_BLOCK_M, BLOCK_M); each redirects its tile row through sorted_slot_ids. The two
    # halves gather independent rows, so they need distinct loaders (a_g2s / a_g2s_hi).
    if gather:
        gl_off_a_lo = compute_global_gather_swizzle_bf16(
            lane_id, wave_id, K, N_LDS_ROUNDS, sorted_res, sorted_row_base
        )
        gl_off_a_hi = compute_global_gather_swizzle_bf16(
            lane_id, wave_id, K, N_LDS_ROUNDS, sorted_res, sorted_row_base + fx.Int32(LDS_BLOCK_M)
        )
    else:
        # FC2 route-read: identity (row = route position), no sorted_slot_ids load.
        gl_off_a_lo = compute_global_identity_swizzle_bf16(
            lane_id, wave_id, K, N_LDS_ROUNDS, sorted_row_base
        )
        gl_off_a_hi = compute_global_identity_swizzle_bf16(
            lane_id, wave_id, K, N_LDS_ROUNDS, sorted_row_base + fx.Int32(LDS_BLOCK_M)
        )
    gl_off_b = compute_global_swizzle_bf16(lane_id, wave_id, K, N_LDS_ROUNDS)

    mfma = Mfma32x32x16(N_TILES_A, N_TILES_B)
    a_g2s = G2SLoader(a_div, gl_off_a_lo, N_LDS_STEPS_A, fx.BFloat16.ir_type, wave_id)
    a_g2s_hi = G2SLoader(a_div, gl_off_a_hi, N_LDS_STEPS_A, fx.BFloat16.ir_type, wave_id)
    b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, fx.BFloat16.ir_type, wave_id)
    a_s2r = S2RLoaderBf16(wave_m, N_TILES_A)
    b_s2r = S2RLoaderBf16(wave_n, N_TILES_B)
    _out_ty = fx.Float16 if out_fp16 else fx.BFloat16
    store_c = StoreCBf16(C, c_m, c_n, _out_ty)

    dense_mma_pipeline_bf16(
        lds,
        a_g2s,
        b_g2s,
        a_s2r,
        b_s2r,
        mfma,
        store_c,
        A0_gl_offset,
        A1_gl_offset,
        B0_gl_offset,
        B1_gl_offset,
        BLOCK_K,
        BLOCK_K,
        fx.Int32(0),  # store block_m: C is already rebased to this tile
        block_n,
        wave_m,
        wave_n,
        K,
        BLOCK_M,
        BLOCK_N,
        nt_vmcnt,
        a_g2s_hi=a_g2s_hi,
    )


def gemm_bf16_nn_gather_tile(
    A,
    B,
    C,
    c_n,
    lds,
    sorted_res,
    sorted_row_base,
    block_n,
    *,
    Kc,
    BLOCK_M,
    BLOCK_N,
    out_fp16=False,
    nt_vmcnt=3,
    b_group_base,
    gather=True,
):
    """One NN dgrad tile, contracting over ``Kc`` (= N) on the transpose-read B path.

    Single self-contained tile for both NN dgrad flavours (mirrors :func:`gemm_bf16_nt_gather_tile`),
    selected by the compile-time ``gather`` flag:

    * ``gather=True`` (FC2 dgrad): ``A`` is the flat token-space grad buffer; each tile row is
      redirected through ``sorted_slot_ids[sorted_row_base + row]`` (A base 0, two loaders for the
      independently-gathered lo/hi halves).
    * ``gather=False`` (FC1 dgrad): ``A`` is the route-ordered grad tile, **pre-rebased to this
      tile's row block by the caller in i64** (so the base pointer -- not an i32 offset -- carries
      ``block_m``, avoiding the int32 multi-block overflow). Rows are read contiguously with a
      single loader (``a_g2s_hi=None`` aliases to ``a_g2s`` in the pipeline).

    ``C`` is rebased to this tile's row block by the caller in both cases, so the store block_m is 0.
    """
    assert BLOCK_M >= 128 and BLOCK_N >= 256 and BLOCK_M % 128 == 0 and BLOCK_N % 256 == 0
    assert Kc % BLOCK_K == 0, f"NN gather needs N % {BLOCK_K} == 0 (got N={Kc})"
    N_TILES_A = BLOCK_M // 128
    N_TILES_B = BLOCK_N // 256
    LDS_BLOCK_M = BLOCK_M // 2
    LDS_BLOCK_N = BLOCK_N // 2
    N_LDS_STEPS_A = LDS_BLOCK_M // 64
    N_LDS_STEPS_B = LDS_BLOCK_N // 64
    N_LDS_ROUNDS = max(N_LDS_STEPS_A, N_LDS_STEPS_B)

    lane_id = fx.thread_idx.x % 64
    wave_id = fx.thread_idx.x // 64
    wave_m = wave_id // 4
    wave_n = wave_id % 4

    # Gather carries the row through the offsets (base 0); route-read reads the pre-rebased tile
    # contiguously, so its hi half starts LDS_BLOCK_M rows in.
    A0_gl_offset = fx.Int32(0)
    A1_gl_offset = fx.Int32(0) if gather else fx.Int32(LDS_BLOCK_M) * fx.Int32(Kc)
    B0_gl_offset = block_n * BLOCK_N + b_group_base
    B1_gl_offset = block_n * BLOCK_N + LDS_BLOCK_N + b_group_base

    gA = make_byte_buffer_tensor(A)
    gB = make_byte_buffer_tensor(B)
    a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
    b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

    if gather:
        # Two gather swizzles: the lo half covers tile rows [0, LDS_BLOCK_M), the hi half
        # [LDS_BLOCK_M, BLOCK_M); each redirects its tile row through ``sorted_slot_ids``. The
        # two halves gather independent rows, so they need distinct loaders (a_g2s / a_g2s_hi).
        gl_off_a_lo = compute_global_gather_swizzle_bf16(
            lane_id, wave_id, Kc, N_LDS_ROUNDS, sorted_res, sorted_row_base
        )
        gl_off_a_hi = compute_global_gather_swizzle_bf16(
            lane_id, wave_id, Kc, N_LDS_ROUNDS, sorted_res, sorted_row_base + fx.Int32(LDS_BLOCK_M)
        )
    else:
        gl_off_a = compute_global_swizzle_bf16(lane_id, wave_id, Kc, N_LDS_ROUNDS)
    gl_off_b = compute_global_swizzle_nn_bf16(lane_id, wave_id, c_n, N_LDS_STEPS_B)

    mfma = Mfma32x32x16(N_TILES_A, N_TILES_B)
    if gather:
        a_g2s = G2SLoader(a_div, gl_off_a_lo, N_LDS_STEPS_A, fx.BFloat16.ir_type, wave_id)
        a_g2s_hi = G2SLoader(a_div, gl_off_a_hi, N_LDS_STEPS_A, fx.BFloat16.ir_type, wave_id)
    else:
        # Contiguous halves share one loader; ``a_g2s_hi=None`` aliases to ``a_g2s`` in the pipeline.
        a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, fx.BFloat16.ir_type, wave_id)
        a_g2s_hi = None
    b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, fx.BFloat16.ir_type, wave_id)
    a_s2r = S2RLoaderBf16(wave_m, N_TILES_A)
    b_s2r = S2RLoaderTrBf16(wave_n, N_TILES_B)
    _out_ty = fx.Float16 if out_fp16 else fx.BFloat16
    store_c = StoreCBf16(C, fx.Int32(BLOCK_M), c_n, _out_ty)

    dense_mma_pipeline_bf16(
        lds,
        a_g2s,
        b_g2s,
        a_s2r,
        b_s2r,
        mfma,
        store_c,
        A0_gl_offset,
        A1_gl_offset,
        B0_gl_offset,
        B1_gl_offset,
        BLOCK_K,
        BLOCK_K * c_n,
        fx.Int32(0),
        block_n,
        wave_m,
        wave_n,
        Kc,
        BLOCK_M,
        BLOCK_N,
        nt_vmcnt,
        a_g2s_hi=a_g2s_hi,
    )
