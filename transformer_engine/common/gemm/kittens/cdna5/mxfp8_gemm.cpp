/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

// MXFP8 (block-scaled FP8) GEMM for gfx1250: TDM-filled LDS double buffer, scaled 16x16x128 WMMA.

#include "hip/hip_runtime.h"
#include <algorithm>
#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>
#include "kittens.cuh"
#include "../kittens_common.h"
#include "gemm_common.cuh"

namespace te_kittens::cdna5::mxfp8 {

constexpr int BLOCK_K   = 256;
constexpr int K_STEP    = 128;
constexpr int K_STEPS   = BLOCK_K / K_STEP;
constexpr int STAGES    = 2;
constexpr int FLIP_FRAG = 3;
constexpr int OP_LD     = BLOCK_K + 16;
constexpr int SC_K      = BLOCK_K / 32;

// LDS stage of a ROWS-wide operand: MN-major [BLOCK_K][ROWS + 16] or K-major [ROWS][OP_LD], whichever is larger.
constexpr int op_stage(int rows) { return BLOCK_K * (rows + 16); }
constexpr int sc_stage(int rows) { return SC_K * (rows + 32); }
constexpr int lds_bytes(int bm, int bn) { return STAGES * (op_stage(bm) + op_stage(bn) + sc_stage(bm) + sc_stage(bn)); }

using kittens::v4u32;
using kittens::v8u32;

using FragT = kittens::rt_base<kittens::fp8e4m3, kittens::ducks::rt_layout::row, kittens::ducks::rt_shape::rt_16x128>;

// One 16x128 operand fragment: MN-major with 8 x ds_load_tr8_b64 (one per 16 K rows), else 4 x ds_load_b128.
template <bool MN, int LD>
__device__ __forceinline__ void load_frag(FragT &dst, uint32_t base, int frag, int ks) {
    if constexpr (MN) {
        const uint32_t addr = base + ks * K_STEP * LD + frag * 16;
        kittens::i32x2 *d   = reinterpret_cast<kittens::i32x2 *>(dst.data);
#pragma unroll
        for (int l = 0; l < 8; l++) {
            d[l] = __builtin_amdgcn_ds_load_tr8_b64_v2i32(lds_ptr<kittens::i32x2>(addr + l * 16 * LD));
        }
    } else {
        const uint32_t addr = base + frag * 16 * LD + ks * K_STEP;
        kittens::u32x4 *d   = reinterpret_cast<kittens::u32x4 *>(dst.data);
#pragma unroll
        for (int l = 0; l < 4; l++) {
            d[l] = *lds_ptr<kittens::u32x4>(addr + l * 32);
        }
    }
}

// SEL_*: lane half holding the e8m0 scales; REUSE_*: only if the previous WMMA has the same A (B) registers.
template <int FMT_A, int FMT_B, int SEL_A, int SEL_B, bool REUSE_A, bool REUSE_B, typename AccT>
__device__ __forceinline__ void mma_ABt_base_mx(AccT &d, const FragT &a, const FragT &b, const AccT &c, int scale_a,
                                                int scale_b) {
    typedef int v16i __attribute__((ext_vector_type(16)));
    *reinterpret_cast<v8f *>(d.data) = __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(FMT_A, *reinterpret_cast<const v16i *>(a.data), FMT_B, *reinterpret_cast<const v16i *>(b.data), 0, *reinterpret_cast<const v8f *>(c.data), SEL_A, 0, scale_a, SEL_B, 0, scale_b, REUSE_A, REUSE_B);
}

// Empty register use: keeps a fragment's VGPRs live so the next load cannot take them (and wait on a recent WMMA).
__device__ __forceinline__ void keep_live(const FragT &t) {
    const kittens::u32x4 *v = reinterpret_cast<const kittens::u32x4 *>(t.data);
    asm volatile("" ::"v"(v[0]), "v"(v[1]), "v"(v[2]), "v"(v[3]));
}

// TDM load from prebuilt descriptor groups 1-3; count 0 is the NULL descriptor.
__device__ __forceinline__ void tdm_load(uint32_t count, uint32_t lds, uint64_t gaddr, const v8u32 &g1,
                                         const v4u32 &g2, const v4u32 &g3) {
    const v4u32 g0 = {count, lds, static_cast<uint32_t>(gaddr),
                      (static_cast<uint32_t>(gaddr >> 32) & 0x01FFFFFFu) | (2u << 30)};
    kittens::emit_tensor_load_to_lds(g0, g1, g2, g3);
}

// TDM descriptor groups 1-3 of one wave's part of an operand or scale tile: [0] head block, [1] full blocks.
struct TdmPart {
    v8u32 g1[2];
    v4u32 g2;
    v4u32 g3[2];
    uint32_t cnt[2];
    uint32_t lds;
    uint32_t stage;
    uint64_t src;
    uint64_t gaddr;
    uint64_t step;
};

template <bool TRANSA, bool TRANSB, int BM, int BN>
__device__ __forceinline__ TdmPart make_tdm_part(const uint8_t *A, const uint8_t *B, const uint8_t *scale_A,
                                                 const uint8_t *scale_B, int M, int N, int K, int m_tile, int n_tile,
                                                 int k0, int k_head, uint32_t lds0, uint32_t a_mask, uint32_t b_mask,
                                                 bool is_b, bool scales, int part, int nparts) {
    constexpr int OFF_B  = STAGES * op_stage(BM);
    constexpr int OFF_SA = OFF_B + STAGES * op_stage(BN);
    TdmPart t;
    const bool mn    = is_b ? TRANSB : !TRANSA;
    const bool perm  = mn && !scales;
    const int tile   = is_b ? n_tile : m_tile;
    const int t_rows = is_b ? BN : BM;
    const uint32_t dsize   = scales && !mn && K % 128 != 0 ? 0 : 2;
    const uint32_t rows    = is_b ? N : M;
    const int rem          = int(rows) - tile * t_rows;
    const uint32_t ld      = scales ? (K / 32 + 3) / 4 * 4 : K;
    const uint32_t k_bytes = scales ? SC_K : BLOCK_K;
    const uint32_t k_shift = (mn || scales ? 5 : 0) + (mn ? 0 : dsize);
    const int q_rows       = (mn ? (scales ? SC_K : BLOCK_K) : t_rows) / nparts;
    const int r0           = part * q_rows;
    const uint32_t lds_row = mn ? t_rows + (scales ? 32 : 16) : (scales ? SC_K : OP_LD);
    const uint64_t gl_row  = mn ? rows : ld;
    const uint32_t pad_int = __builtin_ctz(mn ? t_rows : BLOCK_K) - 3;
    t.step  = mn ? uint64_t(k_bytes) * rows : k_bytes;
    t.stage = scales ? sc_stage(t_rows) : op_stage(t_rows);
    t.lds   = lds0 + (scales ? OFF_SA + is_b * STAGES * sc_stage(BM) : is_b * OFF_B) + r0 * lds_row;
    const size_t k_off = scales ? k0 / 32 : k0;
    const uint8_t *src = (is_b ? (scales ? scale_B : B) : (scales ? scale_A : A)) +
                         (mn ? size_t(tile) * t_rows + k_off * rows : size_t(tile) * t_rows * ld + k_off) +
                         size_t(r0) * gl_row;
    t.src   = reinterpret_cast<uint64_t>(src);
    t.gaddr = t.src - t.step * (BLOCK_K - k_head) / BLOCK_K;
    const uint32_t flags = dsize << 16 | (scales && !mn ? 0 : 1u << 20 | pad_int << 22 | (scales ? 7u : 3u) << 25) |
                           (is_b ? b_mask : a_mask);
    const uint32_t tile0   = (mn ? t_rows : k_bytes) >> dsize;
    const uint32_t tile1   = mn ? (scales ? q_rows : 8) : q_rows;
    const uint32_t tile4   = perm ? q_rows / 32 : 0;
    const uint32_t dim23   = perm ? 2 : 0;
    const uint64_t stride0 = (mn ? rows : ld) >> dsize;
    const uint64_t stride1 = perm ? 4ull * rows : 0;
    const uint64_t stride2 = perm ? 2ull * rows : 0;
    const uint64_t stride3 = perm ? 8ull * rows : 0;
    t.g2 = {dim23, dim23, uint32_t(stride2), uint32_t(stride2 >> 32) | dim23 << 16};
#pragma unroll
    for (int full = 0; full < 2; full++) {
        const int kq      = (full ? BLOCK_K : k_head) >> k_shift;
        // Extent left from this part's origin along the split dimension (MN-major operand K: 32-row blocks).
        const int left    = mn ? kq - (scales ? r0 : r0 / 32) : rem - r0;
        const bool valid  = left > 0;
        const uint32_t lv = valid ? uint32_t(left) : 0u;
        const uint32_t dim0 = mn ? uint32_t(rem) / 4 : uint32_t(kq);
        const uint32_t dim1 = mn ? (scales ? lv : 8u) : lv;
        const uint32_t dim4 = perm ? lv : 0u;
        t.g1[full] = {flags,
                      dim0 << 16,
                      dim0 >> 16 | dim1 << 16,
                      dim1 >> 16 | tile0 << 16,
                      tile1 | dim23 << 16,
                      uint32_t(stride0),
                      uint32_t(stride0 >> 32) | uint32_t(stride1) << 16,
                      uint32_t(stride1 >> 16)};
        t.g3[full]  = {uint32_t(stride3), uint32_t(stride3 >> 32) | dim4 << 16, dim4 >> 16 | tile4 << 16, 0};
        t.cnt[full] = valid ? 1u : 0u;
    }
    return t;
}

// Logical K block blk >= 1 into stage st; !pred is the NULL descriptor (moves nothing).
__device__ __forceinline__ void tdm_block(const TdmPart &op, const TdmPart &sc, int blk, int st, bool pred, int su,
                                          int k_iters) {
    const uint64_t pb = pred ? uint64_t(stagger_k(blk, su, k_iters)) : 0;
    tdm_load(pred ? op.cnt[1] : 0u, op.lds + st * op.stage, op.gaddr + pb * op.step, op.g1[1], op.g2, op.g3[1]);
    tdm_load(pred ? sc.cnt[1] : 0u, sc.lds + st * sc.stage, sc.gaddr + pb * sc.step, sc.g1[1], sc.g2, sc.g3[1]);
}

// Scales of one stage. K-major ones are plain loads, so the compiler may pair them into ds_load_2addr_b64.
template <bool TRANSA, bool TRANSB, int NA, int NB>
__device__ __forceinline__ void load_scales(kittens::i32x2 (&sca)[NA], kittens::i32x2 (&scb)[NB], uint32_t sa,
                                            uint32_t sb) {
#pragma unroll
    for (int p = 0; p < NA; p++) {
        sca[p] = TRANSA ? *lds_ptr<kittens::i32x2>(sa + p * 32 * SC_K) : __builtin_amdgcn_ds_load_tr8_b64_v2i32(lds_ptr<kittens::i32x2>(sa + p * 32));
    }
#pragma unroll
    for (int p = 0; p < NB; p++) {
        scb[p] = TRANSB ? __builtin_amdgcn_ds_load_tr8_b64_v2i32(lds_ptr<kittens::i32x2>(sb + p * 32)) : *lds_ptr<kittens::i32x2>(sb + p * 32 * SC_K);
    }
}

// 256x256: slot of inner fragment c in step t (pairs swap slots every step).
constexpr int fi_slot(int c, int t) { return 2 * ((c / 2) ^ t) + c % 2; }

// 256x256: WMMA P of step T: run q = P / 8, inner fragment c (A for NN, B otherwise, reused by rr = 1..3) with
// outer row o.
template <int FMT_A, int FMT_B, bool NN, int T, int P, typename AccT>
__device__ __forceinline__ void mx_mma(AccT &acc, const FragT *fo, const FragT *fi, const kittens::i32x2 *sca,
                                       const kittens::i32x2 *scb) {
    constexpr int q  = P / 8;
    constexpr int jj = P / 4 % 2;
    constexpr int rr = P % 4;
    constexpr int rb = q / 2;
    constexpr int cp = q == 1 || q == 2;
    constexpr int c  = 2 * cp + jj;
    constexpr int o  = 4 * rb + rr;
    int s_out;
    if constexpr (NN) {
        s_out = scb[o / 2][T];
    } else {
        s_out = sca[o / 2][T];
    }
    const int s_in = NN ? sca[cp][T] : scb[cp][T];
    if constexpr (NN) {
        mma_ABt_base_mx<FMT_A, FMT_B, jj, rr % 2, rr != 0, false>(acc.tiles[c][o], fi[fi_slot(c, T)], fo[o], acc.tiles[c][o], s_in, s_out);
    } else {
        mma_ABt_base_mx<FMT_A, FMT_B, rr % 2, jj, false, rr != 0>(acc.tiles[o][c], fo[o], fi[fi_slot(c, T)], acc.tiles[o][c], s_out, s_in);
    }
    kittens::sched::compiler_fence();
    // Loads follow WMMAs 3, 19, 23, 27 and 31 (the caller fences after them).
    if constexpr (P != 3 && (P < 19 || P % 4 != 3)) {
        kittens::sched::compiler_fence();
    }
}

// WMMAs P0 .. P0 + N - 1 of step T.
template <int FMT_A, int FMT_B, bool NN, int T, int P0, typename AccT, int... I>
__device__ __forceinline__ void mx_mmas(AccT &acc, const FragT *fo, const FragT *fi, const kittens::i32x2 *sca,
                                        const kittens::i32x2 *scb, std::integer_sequence<int, I...>) {
    (mx_mma<FMT_A, FMT_B, NN, T, P0 + I>(acc, fo, fi, sca, scb), ...);
}

// 256x256: step T of K block it from stage PAR. A fragment slot is reloaded >= 4 WMMAs after its last reader, so the
// expert-mode WAR wait (s_wait_alu depctr_va_vdst) is partial.
template <int PAR, int T, bool TRANSA, bool TRANSB, bool CLUSTER, int FMT_A, int FMT_B, typename AccT, int NA, int NB>
__device__ __forceinline__ void big_step(AccT &acc, FragT (&fo)[8], FragT (&fi)[4], kittens::i32x2 (&sca)[2][NA],
                                         kittens::i32x2 (&scb)[2][NB], const uint32_t *a_st, const uint32_t *b_st,
                                         const uint32_t *sa_st, const uint32_t *sb_st, const TdmPart &t_op,
                                         const TdmPart &t_sc, int it, int su, int k_iters, int wid) {
    constexpr bool NN    = !TRANSA && !TRANSB;
    constexpr bool O_MN  = NN ? TRANSB : !TRANSA;
    constexpr bool I_MN  = NN ? !TRANSA : TRANSB;
    constexpr bool last  = T == K_STEPS - 1;
    constexpr int ks     = last ? 0 : T + 1;
    constexpr int nst    = last ? PAR ^ 1 : PAR;
    const uint32_t o_cur = NN ? b_st[PAR] : a_st[PAR];
    const uint32_t i_cur = NN ? a_st[PAR] : b_st[PAR];
    const uint32_t o_nxt = NN ? b_st[nst] : a_st[nst];
    const uint32_t i_nxt = NN ? a_st[nst] : b_st[nst];
    mx_mmas<FMT_A, FMT_B, NN, T, 0>(acc, fo, fi, sca[PAR], scb[PAR], std::make_integer_sequence<int, 4>{});
    load_frag<I_MN, OP_LD>(fi[fi_slot(3, T)], i_cur, 3, T);
#pragma unroll
    for (int r = 4; r < 8; r++) {
        load_frag<O_MN, OP_LD>(fo[r], o_cur, r, T);
    }
    kittens::sched::compiler_fence();
    mx_mmas<FMT_A, FMT_B, NN, T, 4>(acc, fo, fi, sca[PAR], scb[PAR], std::make_integer_sequence<int, 16>{});
    if constexpr (last) {
        kittens::sync::wait_ds<0>();
        kittens::sync::wait_tdm<0>();
        kittens::sched::compiler_fence();
        kittens::sync::sync();
        if constexpr (CLUSTER) {
            if (wid == 0) {
                kittens::cluster::arrive();
            }
            kittens::cluster::wait();
        }
        kittens::sched::compiler_fence();
        tdm_block(t_op, t_sc, it + 2, PAR, it + 2 < k_iters, su, k_iters);
        kittens::sched::compiler_fence();
        load_scales<TRANSA, TRANSB>(sca[PAR ^ 1], scb[PAR ^ 1], sa_st[PAR ^ 1], sb_st[PAR ^ 1]);
    }
#pragma unroll
    for (int r = 0; r < 4; r++) {
        load_frag<O_MN, OP_LD>(fo[r], o_nxt, r, ks);
    }
    keep_live(fi[fi_slot(2, T)]);
    kittens::sched::compiler_fence();
    mx_mmas<FMT_A, FMT_B, NN, T, 20>(acc, fo, fi, sca[PAR], scb[PAR], std::make_integer_sequence<int, 4>{});
    load_frag<I_MN, OP_LD>(fi[fi_slot(2, T)], i_nxt, 0, ks);
    keep_live(fi[fi_slot(3, T)]);
    kittens::sched::compiler_fence();
    mx_mmas<FMT_A, FMT_B, NN, T, 24>(acc, fo, fi, sca[PAR], scb[PAR], std::make_integer_sequence<int, 4>{});
    load_frag<I_MN, OP_LD>(fi[fi_slot(3, T)], i_nxt, 1, ks);
    keep_live(fi[fi_slot(0, T)]);
    kittens::sched::compiler_fence();
    mx_mmas<FMT_A, FMT_B, NN, T, 28>(acc, fo, fi, sca[PAR], scb[PAR], std::make_integer_sequence<int, 4>{});
    load_frag<I_MN, OP_LD>(fi[fi_slot(0, T)], i_nxt, 2, ks);
    keep_live(fi[fi_slot(1, T)]);
#pragma unroll
    for (int r = 4; r < 8; r++) {
        keep_live(fo[r]);
    }
    kittens::sched::compiler_fence();
}

// Smaller tiles: outer fragment pair O, O + 1 of step T of K block it from stage PAR. Each pair is reloaded right
// after its last reader; the stage barrier sits after FLIP_FRAG outer fragments of the last step.
template <int PAR, int T, int O, bool TRANSA, bool TRANSB, int FMT_A, int FMT_B, int A_LD, int B_LD, typename AccT,
          int A_FRAGS, int B_FRAGS, int NA, int NB>
__device__ __forceinline__ void small_pair(AccT &acc, FragT (&fa)[A_FRAGS], FragT (&fb)[B_FRAGS],
                                           kittens::i32x2 (&sca)[2][NA], kittens::i32x2 (&scb)[2][NB],
                                           const uint32_t *a_st, const uint32_t *b_st, const uint32_t *sa_st,
                                           const uint32_t *sb_st, const TdmPart &t_op, const TdmPart &t_sc, int it,
                                           int su, int k_iters) {
    constexpr bool NN    = !TRANSA && !TRANSB;
    constexpr bool last  = T == K_STEPS - 1;
    constexpr bool b_out = NN && T % 2 == 1;
    constexpr int ks     = last ? 0 : T + 1;
    const uint32_t a_src = a_st[last ? PAR ^ 1 : PAR];
    const uint32_t b_src = b_st[last ? PAR ^ 1 : PAR];
    constexpr bool reload_outer = !last || O + 1 >= FLIP_FRAG;
    if constexpr (!b_out) {
        const int sa = sca[PAR][O / 2][T];
        mma_ABt_base_mx<FMT_A, FMT_B, 0, 0, false, false>(acc.tiles[O][0], fa[O], fb[0], acc.tiles[O][0], sa, scb[PAR][0][T]);
        mma_ABt_base_mx<FMT_A, FMT_B, 0, 1, true, false>(acc.tiles[O][1], fa[O], fb[1], acc.tiles[O][1], sa, scb[PAR][0][T]);
#pragma unroll
        for (int j = 2; j < B_FRAGS; j += 2) {
            const int sb = scb[PAR][j / 2][T];
            mma_ABt_base_mx<FMT_A, FMT_B, 0, 0, true, false>(acc.tiles[O][j], fa[O], fb[j], acc.tiles[O][j], sa, sb);
            mma_ABt_base_mx<FMT_A, FMT_B, 0, 1, true, false>(acc.tiles[O][j + 1], fa[O], fb[j + 1], acc.tiles[O][j + 1], sa, sb);
        }
    } else {
        const int sb = scb[PAR][O / 2][T];
        mma_ABt_base_mx<FMT_A, FMT_B, 0, 0, false, false>(acc.tiles[0][O], fa[0], fb[O], acc.tiles[0][O], sca[PAR][0][T], sb);
        mma_ABt_base_mx<FMT_A, FMT_B, 1, 0, false, true>(acc.tiles[1][O], fa[1], fb[O], acc.tiles[1][O], sca[PAR][0][T], sb);
#pragma unroll
        for (int i = 2; i < A_FRAGS; i += 2) {
            const int sa = sca[PAR][i / 2][T];
            mma_ABt_base_mx<FMT_A, FMT_B, 0, 0, false, true>(acc.tiles[i][O], fa[i], fb[O], acc.tiles[i][O], sa, sb);
            mma_ABt_base_mx<FMT_A, FMT_B, 1, 0, false, true>(acc.tiles[i + 1][O], fa[i + 1], fb[O], acc.tiles[i + 1][O], sa, sb);
        }
    }

    if constexpr (last && O + 1 == FLIP_FRAG) {
        kittens::sched::compiler_fence();
        kittens::sync::wait_ds<0>();
        kittens::sync::wait_tdm<0>();
        kittens::sched::compiler_fence();
        kittens::sync::sync();
        kittens::sched::compiler_fence();
        tdm_block(t_op, t_sc, it + 2, PAR, it + 2 < k_iters, su, k_iters);
        kittens::sched::compiler_fence();
        load_scales<TRANSA, TRANSB>(sca[PAR ^ 1], scb[PAR ^ 1], sa_st[PAR ^ 1], sb_st[PAR ^ 1]);
        if constexpr (NN) {
            load_frag<TRANSB, B_LD>(fb[0], b_src, 0, ks);
            load_frag<TRANSB, B_LD>(fb[1], b_src, 1, ks);
        } else {
            load_frag<!TRANSA, A_LD>(fa[0], a_src, 0, ks);
            load_frag<!TRANSA, A_LD>(fa[1], a_src, 1, ks);
        }
        kittens::sched::compiler_fence();
    }

    if constexpr (!b_out) {
        const int sa = sca[PAR][O / 2][T];
        mma_ABt_base_mx<FMT_A, FMT_B, 1, 0, false, false>(acc.tiles[O + 1][0], fa[O + 1], fb[0], acc.tiles[O + 1][0], sa, scb[PAR][0][T]);
        mma_ABt_base_mx<FMT_A, FMT_B, 1, 1, true, false>(acc.tiles[O + 1][1], fa[O + 1], fb[1], acc.tiles[O + 1][1], sa, scb[PAR][0][T]);
        kittens::sched::compiler_fence();
        if constexpr (O + 2 == A_FRAGS) {
            load_frag<TRANSB, B_LD>(fb[0], b_src, 0, ks);
            load_frag<TRANSB, B_LD>(fb[1], b_src, 1, ks);
        }
        kittens::sched::compiler_fence();
#pragma unroll
        for (int j = 2; j < B_FRAGS; j += 2) {
            const int sb = scb[PAR][j / 2][T];
            mma_ABt_base_mx<FMT_A, FMT_B, 1, 0, true, false>(acc.tiles[O + 1][j], fa[O + 1], fb[j], acc.tiles[O + 1][j], sa, sb);
            mma_ABt_base_mx<FMT_A, FMT_B, 1, 1, true, false>(acc.tiles[O + 1][j + 1], fa[O + 1], fb[j + 1], acc.tiles[O + 1][j + 1], sa, sb);
            kittens::sched::compiler_fence();
            if (j + 2 == B_FRAGS && reload_outer) {
                load_frag<!TRANSA, A_LD>(fa[O], a_src, O, ks);
                load_frag<!TRANSA, A_LD>(fa[O + 1], a_src, O + 1, ks);
            }
            if constexpr (O + 2 == A_FRAGS) {
                load_frag<TRANSB, B_LD>(fb[j], b_src, j, ks);
                if constexpr (TRANSA) {
                    kittens::sched::compiler_fence();
                }
                load_frag<TRANSB, B_LD>(fb[j + 1], b_src, j + 1, ks);
            }
            kittens::sched::compiler_fence();
        }
    } else {
        const int sb = scb[PAR][O / 2][T];
        mma_ABt_base_mx<FMT_A, FMT_B, 0, 1, false, false>(acc.tiles[0][O + 1], fa[0], fb[O + 1], acc.tiles[0][O + 1], sca[PAR][0][T], sb);
        mma_ABt_base_mx<FMT_A, FMT_B, 1, 1, false, true>(acc.tiles[1][O + 1], fa[1], fb[O + 1], acc.tiles[1][O + 1], sca[PAR][0][T], sb);
        kittens::sched::compiler_fence();
        if constexpr (O + 2 == B_FRAGS) {
            load_frag<!TRANSA, A_LD>(fa[0], a_src, 0, ks);
            load_frag<!TRANSA, A_LD>(fa[1], a_src, 1, ks);
        }
        kittens::sched::compiler_fence();
#pragma unroll
        for (int i = 2; i < A_FRAGS; i += 2) {
            const int sa = sca[PAR][i / 2][T];
            mma_ABt_base_mx<FMT_A, FMT_B, 0, 1, false, true>(acc.tiles[i][O + 1], fa[i], fb[O + 1], acc.tiles[i][O + 1], sa, sb);
            mma_ABt_base_mx<FMT_A, FMT_B, 1, 1, false, true>(acc.tiles[i + 1][O + 1], fa[i + 1], fb[O + 1], acc.tiles[i + 1][O + 1], sa, sb);
            kittens::sched::compiler_fence();
            if constexpr (O + 2 == B_FRAGS) {
                load_frag<!TRANSA, A_LD>(fa[i], a_src, i, ks);
                load_frag<!TRANSA, A_LD>(fa[i + 1], a_src, i + 1, ks);
            }
            if (i + 2 == A_FRAGS && reload_outer) {
                load_frag<TRANSB, B_LD>(fb[O], b_src, O, ks);
                load_frag<TRANSB, B_LD>(fb[O + 1], b_src, O + 1, ks);
            }
            kittens::sched::compiler_fence();
        }
    }
}

template <int PAR, int T, bool TRANSA, bool TRANSB, int FMT_A, int FMT_B, int A_LD, int B_LD, typename AccT,
          int A_FRAGS, int B_FRAGS, int NA, int NB, int... OO>
__device__ __forceinline__ void small_step(AccT &acc, FragT (&fa)[A_FRAGS], FragT (&fb)[B_FRAGS],
                                           kittens::i32x2 (&sca)[2][NA], kittens::i32x2 (&scb)[2][NB],
                                           const uint32_t *a_st, const uint32_t *b_st, const uint32_t *sa_st,
                                           const uint32_t *sb_st, const TdmPart &t_op, const TdmPart &t_sc, int it,
                                           int su, int k_iters, std::integer_sequence<int, OO...>) {
    (small_pair<PAR, T, 2 * OO, TRANSA, TRANSB, FMT_A, FMT_B, A_LD, B_LD>(acc, fa, fb, sca, scb, a_st, b_st, sa_st,
                                                                         sb_st, t_op, t_sc, it, su, k_iters),
     ...);
}

template <bool TRANSA, bool TRANSB, bool MC, int FMT_A, int FMT_B, GemmEpilogue EPILOGUE, bool ACCUMULATE,
          typename OutT, typename AuxT, int BLOCK_ROW, int BLOCK_COL>
__global__ __cluster_dims__(MC ? CLUSTER_DIM : 0, MC ? CLUSTER_DIM : 0, MC ?: 0)
__launch_bounds__(wg_threads(BLOCK_ROW, BLOCK_COL), 256 / wg_threads(BLOCK_ROW, BLOCK_COL))
void mxfp8_gemm_kernel(const uint8_t *A, const uint8_t *B, const uint8_t *scale_A, const uint8_t *scale_B, void *D,
                       const void *bias, void *aux, float alpha, float beta, int M, int N, int K) {
    constexpr bool NN       = !TRANSA && !TRANSB;
    constexpr bool CLUSTER  = MC && ARCH_MCAST;
    constexpr bool BIG      = BLOCK_ROW == 256 && BLOCK_COL == 256;
    constexpr int NUM_WARPS = wg_threads(BLOCK_ROW, BLOCK_COL) / kittens::WARP_THREADS;
    constexpr int WARPS_ROW = BLOCK_ROW != BLOCK_COL ? (BLOCK_ROW > BLOCK_COL ? 4 : 2) : NN && NUM_WARPS == 8 ? 4 : 2;
    constexpr int WARPS_COL = NUM_WARPS / WARPS_ROW;
    constexpr int A_FRAGS   = BLOCK_ROW / WARPS_ROW / 16;
    constexpr int B_FRAGS   = BLOCK_COL / WARPS_COL / 16;
    constexpr int A_LD      = TRANSA ? OP_LD : BLOCK_ROW + 16;
    constexpr int B_LD      = TRANSB ? BLOCK_COL + 16 : OP_LD;
    constexpr int OFF_B     = STAGES * op_stage(BLOCK_ROW);
    constexpr int OFF_SA    = OFF_B + STAGES * op_stage(BLOCK_COL);
    constexpr int OFF_SB    = OFF_SA + STAGES * sc_stage(BLOCK_ROW);
    static_assert(BIG ? (NN ? B_FRAGS : A_FRAGS) == 8 && (NN ? A_FRAGS : B_FRAGS) == 4
                      : A_FRAGS % 2 == 0 && B_FRAGS % 2 == 0 && A_FRAGS > FLIP_FRAG && B_FRAGS > FLIP_FRAG);
    static_assert(BLOCK_ROW <= BLOCK_K && BLOCK_COL <= BLOCK_K);
    static_assert(BLOCK_COL * (BLOCK_ROW * sizeof(float) + 16) <= lds_bytes(BLOCK_ROW, BLOCK_COL) &&
                  lds_bytes(BLOCK_ROW, BLOCK_COL) * (256 / wg_threads(BLOCK_ROW, BLOCK_COL)) <= 320 * 1024);
    static_assert(!MC || BIG);

    extern __shared__ __attribute__((aligned(16))) uint8_t smem[];
    const uint32_t lds0 = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(smem));

    kittens::sched::lock_simd();

    // MC keeps blockIdx order: the cluster masks and the shared K order of a cluster depend on it.
    TileCoord tc = {int(blockIdx.x), int(blockIdx.y), int(blockIdx.z)};
    if constexpr (!MC) {
        tc = workgroup_tile();
    }
    const int m_tile = tc.m;
    const int n_tile = tc.n;
    int k0 = 0;
    int kz = K;
    if constexpr (EPILOGUE == GemmEpilogue::SPLIT_K) {
        const int splits = gridDim.z;
        const int z      = tc.z;
        const int nb     = K / BLOCK_K;
        const int kh     = K % BLOCK_K;
        const int b0     = nb * z / splits;
        const int b1     = nb * (z + 1) / splits;
        k0 = z == 0 ? 0 : kh + b0 * BLOCK_K;
        kz = (b1 - b0) * BLOCK_K + (z == 0 ? kh : 0);
        D  = static_cast<float *>(D) + size_t(z) * M * N;
    }
    const int tiles_M = kittens::ceil_div(M, BLOCK_ROW);
    const int tiles_N = kittens::ceil_div(N, BLOCK_COL);
    const int k_iters = kittens::ceil_div(kz, BLOCK_K);
    const int k_head  = kz % BLOCK_K ? kz % BLOCK_K : BLOCK_K;
    const int su      = MC || k_iters < 3 ? 0 : m_tile % STAGGER_U * STAGGER_STEP % (k_iters - 1);
    if (k_iters <= 0) {
        return;
    }
    // A workgroup that ends leaves the cluster barrier's member count (ISA 5.6.6).
    if (m_tile >= tiles_M || n_tile >= tiles_N) {
        return;
    }

    const int lane   = kittens::laneid();
    const int wid    = warpid_uniform();
    const int m0     = wid / WARPS_COL * (BLOCK_ROW / WARPS_ROW);
    const int n0     = wid % WARPS_COL * (BLOCK_COL / WARPS_COL);
    const int s      = lane % 16;
    const int h      = lane / 16;
    const int tr_row = (s & 3) + 4 * (s >> 3);
    const int tr_col = 8 * ((s >> 2) & 1);

    const uint32_t a_lane  = lds0 + (TRANSA ? (m0 + s) * A_LD + 16 * h : (8 * h + tr_row) * A_LD + m0 + tr_col);
    const uint32_t b_lane  = lds0 + OFF_B +
                             (TRANSB ? (8 * h + tr_row) * B_LD + n0 + tr_col : (n0 + s) * B_LD + 16 * h);
    const uint32_t sa_lane = lds0 + OFF_SA +
                             (TRANSA ? (m0 + lane) * SC_K : tr_row * (BLOCK_ROW + 32) + m0 + 16 * h + tr_col);
    const uint32_t sb_lane = lds0 + OFF_SB +
                             (TRANSB ? tr_row * (BLOCK_COL + 32) + n0 + 16 * h + tr_col : (n0 + lane) * SC_K);

    uint32_t a_mask = 0;
    uint32_t b_mask = 0;
    if constexpr (CLUSTER) {
        const uint32_t cx   = blockIdx.x % CLUSTER_DIM;
        const uint32_t cy   = blockIdx.y % CLUSTER_DIM;
        const uint32_t in_x = min(tiles_M - int(blockIdx.x - cx), CLUSTER_DIM);
        const uint32_t in_y = min(tiles_N - int(blockIdx.y - cy), CLUSTER_DIM);
        a_mask = (0x1111u << cx) & ((1u << (CLUSTER_DIM * in_y)) - 1u);
        b_mask = ((1u << in_x) - 1u) << (CLUSTER_DIM * cy);
    }

    // Wave w moves part w / 2 of A (even w) or B and its scales. A K-major part beyond the M / N edge is the NULL
    // descriptor (stale rows feed only unstored outputs); the head block is loaded whole: its K rows are all summed.
    const TdmPart t_op = make_tdm_part<TRANSA, TRANSB, BLOCK_ROW, BLOCK_COL>(
        A, B, scale_A, scale_B, M, N, K, m_tile, n_tile, k0, k_head, lds0, a_mask, b_mask, (wid & 1) != 0, false,
        wid >> 1, NUM_WARPS / 2);
    const TdmPart t_sc = make_tdm_part<TRANSA, TRANSB, BLOCK_ROW, BLOCK_COL>(
        A, B, scale_A, scale_B, M, N, K, m_tile, n_tile, k0, k_head, lds0, a_mask, b_mask, (wid & 1) != 0, true,
        wid >> 1, NUM_WARPS / 2);

    kittens::rt_fl<A_FRAGS * 16, B_FRAGS * 16, kittens::col_l, kittens::rt_16x16_s> acc;
    kittens::zero(acc);

    if (wid < 4) {
        const TdmPart t_h = make_tdm_part<TRANSA, TRANSB, BLOCK_ROW, BLOCK_COL>(
            A, B, scale_A, scale_B, M, N, K, m_tile, n_tile, k0, k_head, lds0, a_mask, b_mask, (wid & 1) != 0,
            wid >= 2, 0, 1);
        tdm_load(t_h.cnt[0], t_h.lds, t_h.src, t_h.g1[0], t_h.g2, t_h.g3[0]);
    }
    kittens::sync::wait_tdm<0>();
    kittens::sched::compiler_fence();
    kittens::sync::sync();
    if constexpr (CLUSTER) {
        // Do not change: the workgroup synchronizes then one wave signals the cluster.
        if (wid == 0) {
            kittens::cluster::arrive();
        }
        kittens::cluster::wait();
    }
    kittens::sched::compiler_fence();
    // Block 1 into stage 1 (MC as well: every peer passed the cluster barrier, none has read stage 1).
    tdm_block(t_op, t_sc, 1, 1, k_iters > 1, su, k_iters);
    kittens::sched::compiler_fence();

    uint32_t a_st[STAGES];
    uint32_t b_st[STAGES];
    uint32_t sa_st[STAGES];
    uint32_t sb_st[STAGES];
#pragma unroll
    for (int st = 0; st < STAGES; st++) {
        a_st[st]  = a_lane + st * op_stage(BLOCK_ROW);
        b_st[st]  = b_lane + st * op_stage(BLOCK_COL);
        sa_st[st] = sa_lane + st * sc_stage(BLOCK_ROW);
        sb_st[st] = sb_lane + st * sc_stage(BLOCK_COL);
        // Own VGPRs for the LDS bases: no address VALU may write a WMMA source VGPR (ISA 7.12.1).
        asm volatile("" : "+v"(a_st[st]));
        asm volatile("" : "+v"(b_st[st]));
        asm volatile("" : "+v"(sa_st[st]));
        asm volatile("" : "+v"(sb_st[st]));
    }

    kittens::i32x2 sca[2][A_FRAGS / 2];
    kittens::i32x2 scb[2][B_FRAGS / 2];

    if constexpr (BIG) {
        // Outer operand: the 8-fragment side (A for TN / NT, B for NN); fi[fi_slot(c, t)] holds inner fragment c.
        constexpr bool O_MN = NN ? TRANSB : !TRANSA;
        constexpr bool I_MN = NN ? !TRANSA : TRANSB;
        FragT fo[8];
        FragT fi[4];
        load_scales<TRANSA, TRANSB>(sca[0], scb[0], sa_st[0], sb_st[0]);
#pragma unroll
        for (int r = 0; r < 4; r++) {
            load_frag<O_MN, OP_LD>(fo[r], NN ? b_st[0] : a_st[0], r, 0);
        }
#pragma unroll
        for (int c = 0; c < 3; c++) {
            load_frag<I_MN, OP_LD>(fi[c], NN ? a_st[0] : b_st[0], c, 0);
        }
        kittens::sched::compiler_fence();

        // Block i reads stage i % 2; its barrier follows every read of that stage, then block i + 2 is issued
        // into it (MC: after the cluster barrier, so every peer has drained its reads too).
        static_assert(K_STEPS == 2);
        for (int kb = 0; kb < k_iters; kb += 2) {
            big_step<0, 0, TRANSA, TRANSB, CLUSTER, FMT_A, FMT_B>(acc, fo, fi, sca, scb, a_st, b_st, sa_st, sb_st, t_op,
                                                                  t_sc, kb, su, k_iters, wid);
            big_step<0, 1, TRANSA, TRANSB, CLUSTER, FMT_A, FMT_B>(acc, fo, fi, sca, scb, a_st, b_st, sa_st, sb_st, t_op,
                                                                  t_sc, kb, su, k_iters, wid);
            if (kb + 1 < k_iters) {
                big_step<1, 0, TRANSA, TRANSB, CLUSTER, FMT_A, FMT_B>(acc, fo, fi, sca, scb, a_st, b_st, sa_st, sb_st,
                                                                      t_op, t_sc, kb + 1, su, k_iters, wid);
                big_step<1, 1, TRANSA, TRANSB, CLUSTER, FMT_A, FMT_B>(acc, fo, fi, sca, scb, a_st, b_st, sa_st, sb_st,
                                                                      t_op, t_sc, kb + 1, su, k_iters, wid);
            }
        }
    } else {
        FragT fa[A_FRAGS];
        FragT fb[B_FRAGS];
        load_scales<TRANSA, TRANSB>(sca[0], scb[0], sa_st[0], sb_st[0]);
#pragma unroll
        for (int i = 0; i < A_FRAGS; i++) {
            load_frag<!TRANSA, A_LD>(fa[i], a_st[0], i, 0);
        }
#pragma unroll
        for (int j = 0; j < B_FRAGS; j++) {
            load_frag<TRANSB, B_LD>(fb[j], b_st[0], j, 0);
        }
        kittens::sched::compiler_fence();

        using PAIRS0 = std::make_integer_sequence<int, A_FRAGS / 2>;
        using PAIRS1 = std::make_integer_sequence<int, (NN ? B_FRAGS : A_FRAGS) / 2>;
        for (int kb = 0; kb < k_iters; kb += 2) {
            small_step<0, 0, TRANSA, TRANSB, FMT_A, FMT_B, A_LD, B_LD>(acc, fa, fb, sca, scb, a_st, b_st, sa_st, sb_st,
                                                                       t_op, t_sc, kb, su, k_iters, PAIRS0{});
            small_step<0, 1, TRANSA, TRANSB, FMT_A, FMT_B, A_LD, B_LD>(acc, fa, fb, sca, scb, a_st, b_st, sa_st, sb_st,
                                                                       t_op, t_sc, kb, su, k_iters, PAIRS1{});
            kittens::sched::compiler_fence();
            if (kb + 1 < k_iters) {
                small_step<1, 0, TRANSA, TRANSB, FMT_A, FMT_B, A_LD, B_LD>(acc, fa, fb, sca, scb, a_st, b_st, sa_st,
                                                                           sb_st, t_op, t_sc, kb + 1, su, k_iters,
                                                                           PAIRS0{});
                small_step<1, 1, TRANSA, TRANSB, FMT_A, FMT_B, A_LD, B_LD>(acc, fa, fb, sca, scb, a_st, b_st, sa_st,
                                                                           sb_st, t_op, t_sc, kb + 1, su, k_iters,
                                                                           PAIRS1{});
                kittens::sched::compiler_fence();
            }
        }
    }

    const int row0 = m0 + 8 * h;
    const int col0 = n0 + s;
    const bool edge = (m_tile + 1) * BLOCK_ROW > M || (n_tile + 1) * BLOCK_COL > N || M % 8 != 0;
    if constexpr (EPILOGUE == GemmEpilogue::SPLIT_K) {
        float *d = static_cast<float *>(D) + size_t(n_tile) * BLOCK_COL * M + size_t(m_tile) * BLOCK_ROW;
#pragma unroll
        for (int i = 0; i < A_FRAGS; i++) {
#pragma unroll
            for (int j = 0; j < B_FRAGS; j++) {
                const int row = row0 + i * 16;
                const int col = col0 + j * 16;
                if (!edge || (row < M - m_tile * BLOCK_ROW && col < N - n_tile * BLOCK_COL)) {
                    *reinterpret_cast<v8f *>(d + size_t(col) * M + row) =
                        *reinterpret_cast<const v8f *>(acc.tiles[i][j].data);
                }
            }
        }
        return;
    }

    using StT =
        std::conditional_t<EPILOGUE == GemmEpilogue::DEFAULT && !ACCUMULATE && sizeof(OutT) == 2, OutT, float>;
    kittens::sync::wait_tdm<0>();
    kittens::sync::wait_ds<0>();
    kittens::sync::sync();
#pragma unroll
    for (int i = 0; i < A_FRAGS; i++) {
#pragma unroll
        for (int j = 0; j < B_FRAGS; j++) {
            const uint32_t addr =
                lds0 + (col0 + j * 16) * (BLOCK_ROW * sizeof(StT) + 16) + (row0 + i * 16) * sizeof(StT);
            *lds_ptr<vec8<StT>>(addr) = __builtin_convertvector(*reinterpret_cast<const v8f *>(acc.tiles[i][j].data), vec8<StT>);
        }
    }
    kittens::sync::wait_ds<0>();
    kittens::sync::sync();
    OutT *d            = static_cast<OutT *>(D);
    const AuxT *bias_t = static_cast<const AuxT *>(bias);
    AuxT *aux_t        = static_cast<AuxT *>(aux);
    if (edge) {
        gemm_epilogue<BLOCK_ROW, BLOCK_COL, StT, EPILOGUE, ACCUMULATE, OutT, AuxT, !TRANSA, true>(
            lds0, d, bias_t, aux_t, alpha, beta, M, N, m_tile, n_tile);
    } else {
        gemm_epilogue<BLOCK_ROW, BLOCK_COL, StT, EPILOGUE, ACCUMULATE, OutT, AuxT, !TRANSA, false>(
            lds0, d, bias_t, aux_t, alpha, beta, M, N, m_tile, n_tile);
    }
}

template <bool TRANSA, bool TRANSB, int FMT_A, int FMT_B, GemmEpilogue EPILOGUE, bool ACCUMULATE, typename OutT,
          typename AuxT, int BLOCK_ROW = 256, int BLOCK_COL = 256>
static bool launch_gemm(const MXFP8GemmArgs &args, void *d, int splits, bool mcast) {
    constexpr bool BIG = BLOCK_ROW == 256 && BLOCK_COL == 256;
    constexpr int LDS  = lds_bytes(BLOCK_ROW, BLOCK_COL);
    const bool mc      = BIG && mcast;
    const int cluster  = mc ? CLUSTER_DIM : 1;
    const dim3 grid(kittens::ceil_div(kittens::ceil_div(args.M, BLOCK_ROW), cluster) * cluster,
                    kittens::ceil_div(kittens::ceil_div(args.N, BLOCK_COL), cluster) * cluster, splits);
    const auto kernel = mc ? mxfp8_gemm_kernel<TRANSA, TRANSB, BIG, FMT_A, FMT_B, EPILOGUE, ACCUMULATE, OutT, AuxT,
                                               BLOCK_ROW, BLOCK_COL>
                           : mxfp8_gemm_kernel<TRANSA, TRANSB, false, FMT_A, FMT_B, EPILOGUE, ACCUMULATE, OutT, AuxT,
                                               BLOCK_ROW, BLOCK_COL>;
    hipFuncSetAttribute(reinterpret_cast<const void *>(kernel), hipFuncAttributeMaxDynamicSharedMemorySize, LDS);
    hipLaunchKernelGGL(kernel, grid, dim3(wg_threads(BLOCK_ROW, BLOCK_COL)), LDS, args.stream,
                       static_cast<const uint8_t *>(args.A), static_cast<const uint8_t *>(args.B),
                       static_cast<const uint8_t *>(args.scale_A), static_cast<const uint8_t *>(args.scale_B), d,
                       args.bias, args.aux_gelu, args.alpha, args.beta, args.M, args.N, args.K);
    return true;
}

template <bool TRANSA, bool TRANSB, int FMT_A, int FMT_B, bool ACCUMULATE, int BM, int BN>
static bool dispatch_default(const MXFP8GemmArgs &args, const GemmPlan &plan) {
    if (plan.splits > 1) {
        launch_gemm<TRANSA, TRANSB, FMT_A, FMT_B, GemmEpilogue::SPLIT_K, false, float, float, BM, BN>(
            args, plan.workspace, plan.splits, plan.mcast);
        if (args.out_dtype == KITTENS_FLOAT32) {
            launch_reduce<float>(args, plan.workspace, plan.splits);
        } else if (args.out_dtype == KITTENS_FLOAT16) {
            launch_reduce<_Float16>(args, plan.workspace, plan.splits);
        } else {
            launch_reduce<__bf16>(args, plan.workspace, plan.splits);
        }
        return true;
    }
    switch (args.out_dtype) {
    case KITTENS_FLOAT32:
        return launch_gemm<TRANSA, TRANSB, FMT_A, FMT_B, GemmEpilogue::DEFAULT, ACCUMULATE, float, float, BM, BN>(
            args, args.C, 1, plan.mcast);
    case KITTENS_FLOAT16:
        return launch_gemm<TRANSA, TRANSB, FMT_A, FMT_B, GemmEpilogue::DEFAULT, ACCUMULATE, _Float16, _Float16, BM, BN>(
            args, args.C, 1, plan.mcast);
    case KITTENS_BFLOAT16:
        return launch_gemm<TRANSA, TRANSB, FMT_A, FMT_B, GemmEpilogue::DEFAULT, ACCUMULATE, __bf16, __bf16, BM, BN>(
            args, args.C, 1, plan.mcast);
    }
    return false;
}

template <bool TRANSA, bool TRANSB, int FMT_A, int FMT_B, bool ACCUMULATE>
static bool dispatch_gemm(const MXFP8GemmArgs &args, const GemmPlan &plan) {
    switch (plan.epilogue) {
    case GemmEpilogue::DEFAULT:
        if (plan.bm == 128 && plan.bn == 128) {
            return dispatch_default<TRANSA, TRANSB, FMT_A, FMT_B, ACCUMULATE, 128, 128>(args, plan);
        }
        if (plan.bm == 128) {
            return dispatch_default<TRANSA, TRANSB, FMT_A, FMT_B, ACCUMULATE, 128, 256>(args, plan);
        }
        if (plan.bn == 128) {
            return dispatch_default<TRANSA, TRANSB, FMT_A, FMT_B, ACCUMULATE, 256, 128>(args, plan);
        }
        return dispatch_default<TRANSA, TRANSB, FMT_A, FMT_B, ACCUMULATE, 256, 256>(args, plan);
    case GemmEpilogue::BIAS:
        if constexpr (TRANSA || !ACCUMULATE) {
            if (args.out_dtype == KITTENS_FLOAT32 && args.bias_dtype == KITTENS_FLOAT32) {
                return launch_gemm<TRANSA, TRANSB, FMT_A, FMT_B, GemmEpilogue::BIAS, ACCUMULATE, float, float>(
                    args, args.C, 1, plan.mcast);
            }
            if (args.out_dtype == KITTENS_FLOAT32 && args.bias_dtype == KITTENS_BFLOAT16) {
                return launch_gemm<TRANSA, TRANSB, FMT_A, FMT_B, GemmEpilogue::BIAS, ACCUMULATE, float, __bf16>(
                    args, args.C, 1, plan.mcast);
            }
            if (args.out_dtype == KITTENS_FLOAT16 && args.bias_dtype == KITTENS_FLOAT16) {
                return launch_gemm<TRANSA, TRANSB, FMT_A, FMT_B, GemmEpilogue::BIAS, ACCUMULATE, _Float16, _Float16>(
                    args, args.C, 1, plan.mcast);
            }
            if (args.out_dtype == KITTENS_BFLOAT16 && args.bias_dtype == KITTENS_BFLOAT16) {
                return launch_gemm<TRANSA, TRANSB, FMT_A, FMT_B, GemmEpilogue::BIAS, ACCUMULATE, __bf16, __bf16>(
                    args, args.C, 1, plan.mcast);
            }
        }
        return false;
    case GemmEpilogue::GELU_AUX:
        if constexpr (TRANSA && !ACCUMULATE) {
            if (args.aux_dtype != KITTENS_BFLOAT16) {
                return false;
            }
            switch (args.out_dtype) {
            case KITTENS_FLOAT32:
                return launch_gemm<TRANSA, TRANSB, FMT_A, FMT_B, GemmEpilogue::GELU_AUX, ACCUMULATE, float, __bf16>(
                    args, args.C, 1, plan.mcast);
            case KITTENS_FLOAT16:
                return launch_gemm<TRANSA, TRANSB, FMT_A, FMT_B, GemmEpilogue::GELU_AUX, ACCUMULATE, _Float16, __bf16>(
                    args, args.C, 1, plan.mcast);
            case KITTENS_BFLOAT16:
                return launch_gemm<TRANSA, TRANSB, FMT_A, FMT_B, GemmEpilogue::GELU_AUX, ACCUMULATE, __bf16, __bf16>(
                    args, args.C, 1, plan.mcast);
            }
        }
        return false;
    case GemmEpilogue::GELU_AUX_BIAS:
        if constexpr (TRANSA && !ACCUMULATE) {
            if (args.out_dtype == KITTENS_FLOAT32 && args.bias_dtype == KITTENS_BFLOAT16) {
                return launch_gemm<TRANSA, TRANSB, FMT_A, FMT_B, GemmEpilogue::GELU_AUX_BIAS, ACCUMULATE,
                                   float, __bf16>(args, args.C, 1, plan.mcast);
            }
            if (args.out_dtype == KITTENS_FLOAT16 && args.bias_dtype == KITTENS_FLOAT16) {
                return launch_gemm<TRANSA, TRANSB, FMT_A, FMT_B, GemmEpilogue::GELU_AUX_BIAS, ACCUMULATE,
                                   _Float16, _Float16>(args, args.C, 1, plan.mcast);
            }
            if (args.out_dtype == KITTENS_BFLOAT16 && args.bias_dtype == KITTENS_BFLOAT16) {
                return launch_gemm<TRANSA, TRANSB, FMT_A, FMT_B, GemmEpilogue::GELU_AUX_BIAS, ACCUMULATE,
                                   __bf16, __bf16>(args, args.C, 1, plan.mcast);
            }
        }
        return false;
    default:
        return false;
    }
}

template <bool TRANSA, bool TRANSB>
static bool mxfp8_gemm_impl(const MXFP8GemmArgs &args, const GemmPlan &plan) {
    const bool accumulate = args.beta != 0.f;
    const bool a_fp8      = fp8_code(static_cast<KittensDType>(args.a_dtype)) != 0;
    const bool b_fp8      = fp8_code(static_cast<KittensDType>(args.b_dtype)) != 0;
    bool ok = false;
    KITTENS_BOOL_SWITCH(a_fp8, FMT_A,
        KITTENS_BOOL_SWITCH(b_fp8, FMT_B,
            KITTENS_BOOL_SWITCH(accumulate, ACCUMULATE,
                ok = dispatch_gemm<TRANSA, TRANSB, FMT_A, FMT_B, ACCUMULATE>(args, plan);
    ))) // NOLINT(*)
    return ok;
}

class MXFP8GemmCdna5 final : public MXFP8GemmBackend {
 public:
    bool gemm(const MXFP8GemmArgs &args) override {
        if (!args.mxfp8_compact || (args.transa && args.transb) || args.M <= 0 || args.N <= 0 || args.K <= 0 ||
            args.M % 4 != 0 || args.N % 4 != 0 || args.K % 32 != 0 ||
            (args.grad && (args.bias != nullptr || args.aux_gelu != nullptr))) {
            return false;
        }
        const uintptr_t ptrs = reinterpret_cast<uintptr_t>(args.A) | reinterpret_cast<uintptr_t>(args.B) |
                               reinterpret_cast<uintptr_t>(args.C) | reinterpret_cast<uintptr_t>(args.scale_A) |
                               reinterpret_cast<uintptr_t>(args.scale_B) | reinterpret_cast<uintptr_t>(args.bias) |
                               reinterpret_cast<uintptr_t>(args.aux_gelu);
        if (ptrs % 16 != 0) {
            return false;
        }

        GemmPlan plan;
        if (!plan_gemm(args, TileModel{0.57, 0.3, 10.0, 17.0, 3.5}, BLOCK_K, &plan)) {
            return false;
        }
        if (args.transa && !args.transb) {
            return mxfp8_gemm_impl<true, false>(args, plan);
        }
        if (!args.transa && args.transb) {
            return mxfp8_gemm_impl<false, true>(args, plan);
        }
        return mxfp8_gemm_impl<false, false>(args, plan);
    }

    bool grouped_gemm(const MXFP8GroupedGemmArgs &) override { return false; }
    bool grouped_wgrad(const MXFP8WgradArgs &) override { return false; }
};

}  // namespace te_kittens::cdna5::mxfp8

MXFP8GemmBackend *MXFP8GemmBackend::get_cdna5() {
    static te_kittens::cdna5::mxfp8::MXFP8GemmCdna5 impl;
    return &impl;
}
