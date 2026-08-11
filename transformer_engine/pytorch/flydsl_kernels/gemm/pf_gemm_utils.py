# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""FlyDSL helpers for permute-free MoE grouped GEMM (Mega 8-wave / 32x32x16).

Shared building blocks for ``pf_fwd.py`` and ``pf_dgrad.py``. Dense 4-wave BF16
utilities live in ``fp16_gemm_utils.py``; the pipelined MMA loop lives in
``half_prec_gemm.py`` as ``dense_mma_pipeline_bf16``.
"""

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly as fly_dialect
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import buffer_ops
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import ArithValue
from flydsl.expr import arith, range_constexpr
from flydsl.expr.buffer_ops import buffer_store, create_buffer_resource

from .half_prec_gemm import BLOCK_K, dense_mma_pipeline_bf16
from .fp16_gemm_utils import G2SLoader, ceildiv, make_byte_buffer_tensor, swizzle_128


def _inttoptr_lds(byte_addr):
    """Integer byte address -> !llvm.ptr<3> (LDS). Parsed per call: the type is
    bound to the current MLIRContext and cannot be cached across compiles."""
    return _llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), _raw(fx.Int64(byte_addr)))


_gep = buffer_ops.get_element_ptr


def _lds_ptr_from_i32(addr_i32, byte_offset=0):
    """Build an LDS pointer (ptr<3>) from an i32 byte address + optional static offset."""
    ptr = _inttoptr_lds(ArithValue(addr_i32).extui(T.i64))
    if byte_offset != 0:
        ptr = _gep(ptr, static_byte_offset=byte_offset)
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
        self.c_rows = c_rows
        self.c_cols = c_cols
        self.lane_id = fx.thread_idx.x % 64
        self.out_ty = out_ty
        self.cache_modifier = cache_modifier
        c_nbytes = c_rows * c_cols * 2
        gC = fx.rocdl.make_buffer_tensor(C, max_size=False, num_records_bytes=c_nbytes)
        self.c_div = fx.logical_divide(gC, fx.make_layout(1, 1))
        self.out_atom_1 = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), out_ty)
        self.reg_out_1 = fx.make_rmem_tensor(fx.make_layout(1, 1), out_ty)
        self.c_rsrc = (
            create_buffer_resource(C, max_size=False, num_records_bytes=c_nbytes)
            if cache_modifier
            else None
        )
        self.oob = fx.Int32(c_rows * c_cols)  # out-of-bounds sink index

    def _store_masked(self, value, c_index, valid):
        """Store one element to c_index (masked to the OOB sink when invalid)."""
        idx = arith.select(valid, c_index, self.oob)
        val = value.to(self.out_ty)
        if self.cache_modifier:
            buffer_store(val, self.c_rsrc, fx.Int32(idx), cache_modifier=self.cache_modifier)
        else:
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

    def store16(self, c_frag, base_row, base_col):
        n = self.lane_id % 16
        m_hi = (self.lane_id // 16) * 4
        col = base_col + n
        col_valid = col < self.c_cols
        for ti in range_constexpr(len(c_frag)):
            acc = Vec(c_frag[ti])
            for r in range_constexpr(4):
                row = base_row + ti * 16 + m_hi + r
                self._store_masked(acc[r], row * self.c_cols + col, col_valid)

    def store_trans16(self, c_frag, group_idx, base_m, base_n, out_m, out_n):
        n = self.lane_id % 16
        m_hi = (self.lane_id // 16) * 4
        glob_n = base_n + n
        n_valid = glob_n < out_n
        row_base = (group_idx * out_n + glob_n) * out_m
        for ti in range_constexpr(len(c_frag)):
            acc = Vec(c_frag[ti])
            for r in range_constexpr(4):
                m = base_m + ti * 16 + m_hi + r
                self._store_masked(acc[r], row_base + m, n_valid)


def buffer_load_i32(rsrc, off):
    """Load one i32 scalar from a buffer resource at ``off`` (i32- or index-typed)."""
    return buffer_ops.buffer_load(rsrc, off, vec_width=1, dtype=T.i32)


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


def _gemm_bf16_nn_tn_tile_impl(
    A,
    B,
    C,
    c_m,
    c_n,
    lds,
    block_m,
    block_n,
    *,
    a_transpose,
    K,
    BLOCK_M,
    BLOCK_N,
    n_blocks=None,
    GROUP_M=1,
    num_xcd=8,
    out_fp16=False,
    nt_vmcnt=3,
    b_group_base=None,
    c_cache_modifier=0,
):
    assert BLOCK_M >= 128 and BLOCK_N >= 256 and BLOCK_M % 128 == 0 and BLOCK_N % 256 == 0
    assert K % BLOCK_K == 0, f"bf16 NN/TN needs K % {BLOCK_K} == 0 (got K={K})"
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

    if block_m is None:
        num_pid_m = ceildiv(c_m, BLOCK_M)
        pid = xcd_remap_pid(fx.block_idx.x, num_pid_m * n_blocks, num_xcd)
        num_pid_in_group = GROUP_M * n_blocks
        group_id = pid // num_pid_in_group
        pid_in_group = pid % num_pid_in_group
        first_pid_m = group_id * GROUP_M
        remaining_m = num_pid_m - first_pid_m
        group_size_m = arith.select(remaining_m < GROUP_M, remaining_m, fx.Int32(GROUP_M))
        block_m = first_pid_m + (pid_in_group % group_size_m)
        block_n = pid_in_group // group_size_m

    if a_transpose:
        A0_gl_offset = block_m * BLOCK_M + 0
        A1_gl_offset = block_m * BLOCK_M + LDS_BLOCK_M
        a_k_step = BLOCK_K * c_m
    else:
        A0_gl_offset = (block_m * BLOCK_M) * K
        A1_gl_offset = (block_m * BLOCK_M + LDS_BLOCK_M) * K
        a_k_step = BLOCK_K
    B0_gl_offset = block_n * BLOCK_N + 0
    B1_gl_offset = block_n * BLOCK_N + LDS_BLOCK_N
    b_k_step = BLOCK_K * c_n
    if b_group_base is not None:
        B0_gl_offset = B0_gl_offset + b_group_base
        B1_gl_offset = B1_gl_offset + b_group_base

    gA = make_byte_buffer_tensor(A)
    gB = make_byte_buffer_tensor(B)
    a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
    b_div = fx.logical_divide(gB, fx.make_layout(1, 1))
    if a_transpose:
        gl_off_a = compute_global_swizzle_nn_bf16(lane_id, wave_id, c_m, N_LDS_STEPS_A)
    else:
        gl_off_a = compute_global_swizzle_bf16(lane_id, wave_id, K, N_LDS_ROUNDS)
    gl_off_b = compute_global_swizzle_nn_bf16(lane_id, wave_id, c_n, N_LDS_STEPS_B)

    mfma = Mfma32x32x16(N_TILES_A, N_TILES_B)
    a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, fx.BFloat16.ir_type, wave_id)
    b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, fx.BFloat16.ir_type, wave_id)
    a_s2r = S2RLoaderTrBf16(wave_m, N_TILES_A) if a_transpose else S2RLoaderBf16(wave_m, N_TILES_A)
    b_s2r = S2RLoaderTrBf16(wave_n, N_TILES_B)
    _out_ty = fx.Float16 if out_fp16 else fx.BFloat16
    store_c = StoreCBf16(C, c_m, c_n, _out_ty, cache_modifier=c_cache_modifier)

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
    )


def gemm_bf16_nn_tile(
    A,
    B,
    C,
    c_m,
    c_n,
    lds,
    block_m=None,
    block_n=None,
    *,
    K,
    BLOCK_M,
    BLOCK_N,
    n_blocks=None,
    GROUP_M=1,
    num_xcd=8,
    out_fp16=False,
    nt_vmcnt=3,
    b_group_base=None,
    c_cache_modifier=0,
):
    _gemm_bf16_nn_tn_tile_impl(
        A,
        B,
        C,
        c_m,
        c_n,
        lds,
        block_m,
        block_n,
        a_transpose=False,
        K=K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        n_blocks=n_blocks,
        GROUP_M=GROUP_M,
        num_xcd=num_xcd,
        out_fp16=out_fp16,
        nt_vmcnt=nt_vmcnt,
        b_group_base=b_group_base,
        c_cache_modifier=c_cache_modifier,
    )
