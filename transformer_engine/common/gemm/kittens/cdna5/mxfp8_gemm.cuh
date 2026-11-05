/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

// gfx1250 MXFP8 GEMM: e8m0 scales [rows][K/32] or, M/N-contiguous, [K/32][rows], rows padded to 4 B.
// M % 4 == 0, N % 4 == 0, K % 32 == 0.

#pragma once

#include "hip/hip_runtime.h"
#include <algorithm>
#include <cstdint>
#include <type_traits>
#include "common.cuh"

namespace te_kittens::cdna5 {
namespace mxfp8_gemm {

constexpr int BLOCK_K         = MXFP8_BLOCK_K;
constexpr int STAGES          = 2;
constexpr int K_STEP          = 128;
constexpr int K_STEPS         = BLOCK_K / K_STEP;
constexpr int MX_BLOCK        = 32;
constexpr int SCALE_ROW_BYTES = BLOCK_K / MX_BLOCK;
constexpr int NUM_WARPS       = 8;
constexpr int RELOAD_GROUP    = 2;

static_assert(NUM_WARPS * kittens::WARP_THREADS == NUM_THREADS);

typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v8i __attribute__((ext_vector_type(8)));
typedef int v16i __attribute__((ext_vector_type(16)));
typedef __attribute__((address_space(3))) v2i lds_v2i;

// TDM descriptor fields (ISA 10.11.4).
constexpr uint32_t TDM_ADDR_HI_MASK = 0x01FFFFFFu;
constexpr uint32_t TDM_TYPE         = 2u << 30;

__device__ __forceinline__ v4i lds_b128(uint32_t addr) {
    return *lds_ptr<v4i>(addr);
}

__device__ __forceinline__ v2i lds_tr8(uint32_t addr) {
    return __builtin_amdgcn_ds_load_tr8_b64_v2i32(lds_ptr<v2i>(addr));
}

constexpr uint32_t ilog2(uint32_t x) { return x <= 1 ? 0 : 1 + ilog2(x / 2); }

// TDM padding codes: pad_amount DWords after every pad_interval DWords.
constexpr uint32_t tdm_pad_interval(int bytes) { return ilog2(bytes / 4) - 1; }
constexpr uint32_t tdm_pad_amount(int bytes) { return bytes / 4 - 1; }

// One operand of R tile rows: [R][BLOCK_K + PAD], or (TRANS) [BLOCK_K][R + PAD] with K rows permuted by the
// TDM so that one ds_load_tr8_b64 reads 16 consecutive rows (ISA 10.9.2).
template <bool TRANS, int R, int FRAGS>
struct Operand {
    static constexpr int PAD         = 16;
    static constexpr int ROW_BYTES   = (TRANS ? R : BLOCK_K) + PAD;
    static constexpr int STAGE_BYTES = (TRANS ? BLOCK_K : R) * ROW_BYTES;
    static constexpr int LOADS       = TRANS ? 8 : 4;

    __device__ static uint32_t lane(int L, int w0) {
        const int s = L & 15;
        const int H = L >> 4;
        if constexpr (TRANS) {
            const int j  = (s & 3) + 4 * (s >> 3);
            const int ch = (s >> 2) & 1;
            return (8 * H + j) * ROW_BYTES + w0 + 8 * ch;
        } else {
            return (w0 + s) * ROW_BYTES + 16 * H;
        }
    }
    static constexpr uint32_t frag_off(int i, int ks) {
        return TRANS ? ks * K_STEP * ROW_BYTES + i * 16 : i * 16 * ROW_BYTES + ks * K_STEP;
    }
    static constexpr uint32_t load_off(int x) {
        return TRANS ? (32 * (x / 2) + 16 * (x % 2)) * ROW_BYTES : 32 * x;
    }

    __device__ static v16i load(uint32_t addr) {
        if constexpr (TRANS) {
            const v2i x0 = lds_tr8(addr + load_off(0));
            const v2i x1 = lds_tr8(addr + load_off(1));
            const v2i x2 = lds_tr8(addr + load_off(2));
            const v2i x3 = lds_tr8(addr + load_off(3));
            const v2i x4 = lds_tr8(addr + load_off(4));
            const v2i x5 = lds_tr8(addr + load_off(5));
            const v2i x6 = lds_tr8(addr + load_off(6));
            const v2i x7 = lds_tr8(addr + load_off(7));
            const v4i a = __builtin_shufflevector(x0, x1, 0, 1, 2, 3);
            const v4i b = __builtin_shufflevector(x2, x3, 0, 1, 2, 3);
            const v4i c = __builtin_shufflevector(x4, x5, 0, 1, 2, 3);
            const v4i d = __builtin_shufflevector(x6, x7, 0, 1, 2, 3);
            const v8i lo = __builtin_shufflevector(a, b, 0, 1, 2, 3, 4, 5, 6, 7);
            const v8i hi = __builtin_shufflevector(c, d, 0, 1, 2, 3, 4, 5, 6, 7);
            return __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        } else {
            const v4i q0 = lds_b128(addr + load_off(0));
            const v4i q1 = lds_b128(addr + load_off(1));
            const v4i q2 = lds_b128(addr + load_off(2));
            const v4i q3 = lds_b128(addr + load_off(3));
            const v8i lo = __builtin_shufflevector(q0, q1, 0, 1, 2, 3, 4, 5, 6, 7);
            const v8i hi = __builtin_shufflevector(q2, q3, 0, 1, 2, 3, 4, 5, 6, 7);
            return __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        }
    }
    static_assert(frag_off(FRAGS - 1, K_STEPS - 1) + load_off(LOADS - 1) < 65536, "offsets must fit the DS offset");
};

// Scales of R tile rows, [R][SCALE_ROW_BYTES] or (TRANS) [SCALE_ROW_BYTES][R + PAD]: lane L of fragment pair p
// gets both K steps' e8m0 dwords of row 32p + L; SCL_OPSEL picks the lane half.
template <bool TRANS, int R>
struct Scales {
    static constexpr int PAD         = 32;
    static constexpr int ROW_BYTES   = TRANS ? R + PAD : SCALE_ROW_BYTES;
    static constexpr int STAGE_BYTES = (TRANS ? SCALE_ROW_BYTES : R) * ROW_BYTES;

    __device__ static uint32_t lane(int L, int w0) {
        if constexpr (TRANS) {
            const int s = L & 15;
            const int H = L >> 4;
            return ((s & 3) + 4 * (s >> 3)) * ROW_BYTES + w0 + 16 * H + 8 * ((s >> 2) & 1);
        } else {
            return (w0 + L) * ROW_BYTES;
        }
    }
    static constexpr uint32_t pair_off(int p) { return TRANS ? 32 * p : p * 32 * ROW_BYTES; }
    __device__ static v2i load(uint32_t addr) {
        if constexpr (TRANS) {
            return lds_tr8(addr);
        } else {
            return *lds_ptr<v2i>(addr);
        }
    }
};

// WMMA order and reload triggers per K-step parity: a fragment reloads after its group's last read, and in a
// block's last K step not before the stage flip at PRE_BAR - 1.
template <int A_FRAGS, int B_FRAGS>
struct Plan {
    int i[2][A_FRAGS * B_FRAGS];
    int j[2][A_FRAGS * B_FRAGS];
    bool ra[2][A_FRAGS * B_FRAGS];
    bool rb[2][A_FRAGS * B_FRAGS];
    int trig_a[2][2][A_FRAGS];
    int trig_b[2][2][B_FRAGS];
};

// Extends each fragment's last read to the last read of its reload group.
constexpr void group_last(int *last, int frags) {
    for (int f = 0; f < frags; f++) {
        const int g = std::min((f / RELOAD_GROUP + 1) * RELOAD_GROUP - 1, frags - 1);
        last[f] = std::max(last[g], last[f]);
    }
}

template <int A_FRAGS, int B_FRAGS, bool ALT_ORDER, int PRE_BAR>
constexpr Plan<A_FRAGS, B_FRAGS> make_plan() {
    constexpr int WMMAS = A_FRAGS * B_FRAGS;
    Plan<A_FRAGS, B_FRAGS> p{};
    for (int q = 0; q < 2; q++) {
        int t = 0;
        if (ALT_ORDER && q == 1) {
            for (int j = 0; j < B_FRAGS; j++) {
                for (int i = 0; i < A_FRAGS; i++) {
                    p.i[q][t] = i;
                    p.j[q][t] = j;
                    p.ra[q][t] = false;
                    p.rb[q][t] = i > 0;
                    t++;
                }
            }
        } else {
            for (int i = 0; i < A_FRAGS; i++) {
                for (int j = 0; j < B_FRAGS; j++) {
                    p.i[q][t] = i;
                    p.j[q][t] = j;
                    p.ra[q][t] = j > 0;
                    p.rb[q][t] = false;
                    t++;
                }
            }
        }
        int last_a[A_FRAGS] = {};
        int last_b[B_FRAGS] = {};
        for (int x = 0; x < WMMAS; x++) {
            last_a[p.i[q][x]] = x;
            last_b[p.j[q][x]] = x;
        }
        group_last(last_a, A_FRAGS);
        group_last(last_b, B_FRAGS);
        for (int last = 0; last < 2; last++) {
            const int lo = last ? PRE_BAR - 1 : 0;
            for (int i = 0; i < A_FRAGS; i++) {
                p.trig_a[q][last][i] = std::max(last_a[i], lo);
            }
            for (int j = 0; j < B_FRAGS; j++) {
                p.trig_b[q][last][j] = std::max(last_b[j], lo);
            }
        }
    }
    return p;
}

// NN: 4x2 warps of 64x128, WMMA nesting alternating per K step; TN and NT: 2x4 warps of 128x64.
template <Layout L>
struct Cfg {
    static constexpr bool TRANS_A = L != TN;
    static constexpr bool TRANS_B = L == NT;
    static constexpr int WARPS_M   = L == NN ? 4 : 2;
    static constexpr int WARPS_N   = NUM_WARPS / WARPS_M;
    static constexpr int WARP_M    = BLOCK_M / WARPS_M;
    static constexpr int WARP_N    = BLOCK_N / WARPS_N;
    static constexpr int A_FRAGS   = WARP_M / 16;
    static constexpr int B_FRAGS   = WARP_N / 16;
    static constexpr int WMMA_STEP = A_FRAGS * B_FRAGS;
    // WMMAs of a block's last K step issued before the stage flip.
    static constexpr int PRE_BAR   = 3 * WMMA_STEP / 8;

    using OA = Operand<TRANS_A, BLOCK_M, A_FRAGS>;
    using OB = Operand<TRANS_B, BLOCK_N, B_FRAGS>;
    using OSA = Scales<TRANS_A, BLOCK_M>;
    using OSB = Scales<TRANS_B, BLOCK_N>;

    // All A stages, all B stages, then the scale stages; the C staging reuses offset 0.
    static constexpr int OFF_A     = 0;
    static constexpr int OFF_B     = OFF_A + STAGES * OA::STAGE_BYTES;
    static constexpr int OFF_SA    = OFF_B + STAGES * OB::STAGE_BYTES;
    static constexpr int OFF_SB    = OFF_SA + STAGES * OSA::STAGE_BYTES;
    static constexpr int LDS_BYTES = OFF_SB + STAGES * OSB::STAGE_BYTES;

    static constexpr Plan<A_FRAGS, B_FRAGS> PLAN = make_plan<A_FRAGS, B_FRAGS, L == NN, PRE_BAR>();

    static_assert(A_FRAGS % 2 == 0 && B_FRAGS % 2 == 0, "scale VGPRs pair fragments through SCL_OPSEL");
    static_assert(BLOCK_N * staged_stride(sizeof(float)) <= LDS_BYTES, "fp32 C staging fits the operand LDS");
    static_assert(LDS_BYTES <= LDS_MAX_BYTES);
};

struct tdm_desc {
    kittens::v4u32 g0;
    kittens::v8u32 g1;
    kittens::v4u32 g2;
    kittens::v4u32 g3;
};

// Dim d > 0 advances the global address by stride[d - 1] elements; extents count from the tile origin, 0 unused.
struct tdm_shape {
    uint32_t dsize_enc;    // 0: 1 byte, 2: 4 bytes
    uint32_t tdim[5];
    uint32_t tile[5];
    uint64_t stride[4];
    uint32_t pad_en;
    uint32_t pad_int_enc;
    uint32_t pad_amt_enc;
};

__device__ __forceinline__ tdm_desc tdm_make(uint32_t lds_addr, const void *gptr, const tdm_shape &s,
                                             uint32_t mask) {
    const uint64_t ga = reinterpret_cast<uint64_t>(gptr);
    tdm_desc d;
    d.g0[0] = 1u;
    d.g0[1] = lds_addr;
    d.g0[2] = static_cast<uint32_t>(ga);
    d.g0[3] = (static_cast<uint32_t>(ga >> 32) & TDM_ADDR_HI_MASK) | TDM_TYPE;
    d.g1[0] = (s.dsize_enc << 16) | (s.pad_en << 20) | (s.pad_int_enc << 22) | (s.pad_amt_enc << 25) |
              (mask & 0xFFFFu);
    d.g1[1] = s.tdim[0] << 16;
    d.g1[2] = (s.tdim[0] >> 16) | (s.tdim[1] << 16);
    d.g1[3] = (s.tdim[1] >> 16) | (s.tile[0] << 16);
    d.g1[4] = s.tile[1] | (s.tile[2] << 16);
    d.g1[5] = static_cast<uint32_t>(s.stride[0]);
    d.g1[6] = static_cast<uint32_t>(s.stride[0] >> 32) | (static_cast<uint32_t>(s.stride[1]) << 16);
    d.g1[7] = static_cast<uint32_t>(s.stride[1] >> 16);
    d.g2[0] = s.tdim[2];
    d.g2[1] = s.tdim[3];
    d.g2[2] = static_cast<uint32_t>(s.stride[2]);
    d.g2[3] = static_cast<uint32_t>(s.stride[2] >> 32) | (s.tile[3] << 16);
    d.g3[0] = static_cast<uint32_t>(s.stride[3]);
    d.g3[1] = static_cast<uint32_t>(s.stride[3] >> 32) | (s.tdim[4] << 16);
    d.g3[2] = (s.tdim[4] >> 16) | (s.tile[4] << 16);
    d.g3[3] = 0;
    return d;
}

// Issue the descriptor for one (stage, K-block): only the LDS and global address words change.
__device__ __forceinline__ void tdm_issue(const tdm_desc &d, uint32_t lds_off, uint64_t g_off) {
    kittens::v4u32 g0 = d.g0;
    const uint64_t ga = ((uint64_t(d.g0[3] & TDM_ADDR_HI_MASK) << 32) | d.g0[2]) + g_off;
    g0[1] = d.g0[1] + lds_off;
    g0[2] = static_cast<uint32_t>(ga);
    g0[3] = (static_cast<uint32_t>(ga >> 32) & TDM_ADDR_HI_MASK) | TDM_TYPE;
    kittens::emit_tensor_load_to_lds(g0, d.g1, d.g2, d.g3);
}

// One K-block of R of Rtot rows, rem rows and kx K from the tile origin; step: global bytes per K-block.
// TRANS uses dims (row, j, h, p, q) of K = 32q + 16h + 8p + j.
template <bool TRANS, int R>
__device__ __forceinline__ tdm_shape operand_shape(int K, int kx, uint32_t Rtot, uint32_t rem, uint32_t &step) {
    using Op = Operand<TRANS, R, 1>;
    tdm_shape s = {};
    s.dsize_enc   = 2;
    s.pad_en      = 1;
    s.pad_amt_enc = tdm_pad_amount(Op::PAD);
    if constexpr (TRANS) {
        step      = uint32_t(BLOCK_K) * Rtot;
        s.tdim[0] = rem / 4;
        s.tdim[1] = 8;
        s.tdim[2] = 2;
        s.tdim[3] = 2;
        s.tdim[4] = uint32_t(kx / 32);
        s.tile[0] = R / 4;
        s.tile[1] = 8;
        s.tile[2] = 2;
        s.tile[3] = 2;
        s.tile[4] = BLOCK_K / 32;
        s.stride[0] = Rtot / 4;
        s.stride[1] = 16ull * Rtot / 4;
        s.stride[2] = 8ull * Rtot / 4;
        s.stride[3] = 32ull * Rtot / 4;
        s.pad_int_enc = tdm_pad_interval(R);
    } else {
        step      = BLOCK_K;
        s.tdim[0] = uint32_t(kx / 4);
        s.tdim[1] = rem;
        s.tile[0] = BLOCK_K / 4;
        s.tile[1] = R;
        s.stride[0] = uint64_t(K / 4);
        s.pad_int_enc = tdm_pad_interval(BLOCK_K);
    }
    return s;
}

// Bytes per row of row-wise scales: K/32 rounded up to 4, TE's gfx1250 scale layout.
__device__ constexpr int scale_ld(int K) { return (K / MX_BLOCK + 3) / 4 * 4; }

// Scales of one K-block: [R][SCALE_ROW_BYTES] of [Rtot][scale_ld], or (TRANS) [SCALE_ROW_BYTES][R + PAD] of
// [K/32][Rtot]. Row-wise scales move as DWords when K % 128 == 0, else as bytes, so the head ends exactly.
template <bool TRANS, int R>
__device__ __forceinline__ tdm_shape scale_shape(int K, int kx, uint32_t Rtot, uint32_t rem, uint32_t &step) {
    const uint32_t ks  = scale_ld(K);
    const uint32_t ksx = kx / MX_BLOCK;
    tdm_shape s = {};
    if constexpr (TRANS) {
        step        = uint32_t(SCALE_ROW_BYTES) * Rtot;
        s.dsize_enc = 2;
        s.tdim[0]   = rem / 4;
        s.tdim[1]   = ksx;
        s.tile[0]   = R / 4;
        s.tile[1]   = SCALE_ROW_BYTES;
        s.stride[0] = Rtot / 4;
        s.pad_en      = 1;
        s.pad_int_enc = tdm_pad_interval(R);
        s.pad_amt_enc = tdm_pad_amount(Scales<TRANS, R>::PAD);
    } else if (K % 128 != 0) {
        step        = SCALE_ROW_BYTES;
        s.dsize_enc = 0;
        s.tdim[0]   = ksx;
        s.tdim[1]   = rem;
        s.tile[0]   = SCALE_ROW_BYTES;
        s.tile[1]   = R;
        s.stride[0] = ks;
    } else {
        step        = SCALE_ROW_BYTES;
        s.dsize_enc = 2;
        s.tdim[0]   = ksx / 4;
        s.tdim[1]   = rem;
        s.tile[0]   = SCALE_ROW_BYTES / 4;
        s.tile[1]   = R;
        s.stride[0] = ks / 4;
    }
    return s;
}

// The descriptor one filling wave builds once; the K loop rewrites only its LDS and global address words.
struct Fill {
    tdm_desc d;
    uint32_t stage_bytes;
    uint32_t step;
};

// Wave 0 fills A, 1 B, 2 A scales, 3 B scales; tiles[] are their K-block 0 origins. kx: K extent from the tile
// origin; back: K elements the origin moves back.
template <Layout L>
__device__ __forceinline__ void make_fill(Fill &f, int wid, uint32_t lds0, const uint8_t *const *tiles,
                                          uint32_t a_mask, uint32_t b_mask, int M, int N, int K, uint32_t m_rem,
                                          uint32_t n_rem, int kx, int back) {
    using Cf = Cfg<L>;
    if (wid == 0) {
        const tdm_shape s = operand_shape<Cf::TRANS_A, BLOCK_M>(K, kx, M, m_rem, f.step);
        f.d = tdm_make(lds0 + Cf::OFF_A, tiles[0] - size_t(f.step) * back / BLOCK_K, s, a_mask);
        f.stage_bytes = Cf::OA::STAGE_BYTES;
    } else if (wid == 1) {
        const tdm_shape s = operand_shape<Cf::TRANS_B, BLOCK_N>(K, kx, N, n_rem, f.step);
        f.d = tdm_make(lds0 + Cf::OFF_B, tiles[1] - size_t(f.step) * back / BLOCK_K, s, b_mask);
        f.stage_bytes = Cf::OB::STAGE_BYTES;
    } else if (wid == 2) {
        const tdm_shape s = scale_shape<Cf::TRANS_A, BLOCK_M>(K, kx, M, m_rem, f.step);
        f.d = tdm_make(lds0 + Cf::OFF_SA, tiles[2] - size_t(f.step) * back / BLOCK_K, s, a_mask);
        f.stage_bytes = Cf::OSA::STAGE_BYTES;
    } else if (wid == 3) {
        const tdm_shape s = scale_shape<Cf::TRANS_B, BLOCK_N>(K, kx, N, n_rem, f.step);
        f.d = tdm_make(lds0 + Cf::OFF_SB, tiles[3] - size_t(f.step) * back / BLOCK_K, s, b_mask);
        f.stage_bytes = Cf::OSB::STAGE_BYTES;
    }
}

// Without this the compiler keeps the descriptor in VGPRs and re-reads it every K-block.
__device__ __forceinline__ void make_uniform(Fill &f) {
    for (int i = 0; i < 4; i++) {
        f.d.g0[i] = __builtin_amdgcn_readfirstlane(f.d.g0[i]);
    }
    for (int i = 0; i < 8; i++) {
        f.d.g1[i] = __builtin_amdgcn_readfirstlane(f.d.g1[i]);
    }
    for (int i = 0; i < 4; i++) {
        f.d.g2[i] = __builtin_amdgcn_readfirstlane(f.d.g2[i]);
    }
    for (int i = 0; i < 4; i++) {
        f.d.g3[i] = __builtin_amdgcn_readfirstlane(f.d.g3[i]);
    }
    f.stage_bytes = __builtin_amdgcn_readfirstlane(f.stage_bytes);
    f.step        = __builtin_amdgcn_readfirstlane(f.step);
}

// Keeps x in its own VGPR (no rematerialization next to the WMMAs).
__device__ __forceinline__ uint32_t opaque(uint32_t x) {
    asm volatile("" : "+v"(x));
    return x;
}

// Per-lane LDS addresses of one stage.
struct bases {
    uint32_t a;
    uint32_t b;
    uint32_t sa;
    uint32_t sb;
};

template <int FMT_A, int FMT_B, int SEL_A, int SEL_B, bool RA, bool RB>
__device__ __forceinline__ void wmma(v8f &c, const v16i &a, const v16i &b, int sa, int sb) {
    c = __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(FMT_A, a, FMT_B, b, (short)0, c, SEL_A, 0, sa, SEL_B, 0, sb,
                                                         RA, RB);
}

template <Layout L>
struct Regs {
    v8f acc[Cfg<L>::A_FRAGS][Cfg<L>::B_FRAGS];
    v16i fa[Cfg<L>::A_FRAGS];
    v16i fb[Cfg<L>::B_FRAGS];
    v2i sca[2][Cfg<L>::A_FRAGS / 2];
    v2i scb[2][Cfg<L>::B_FRAGS / 2];
};

// The K loop. WMMAs and operand reloads follow Cfg<L>::PLAN; the WMMA format and reload immediates are
// compile-time, so the per-WMMA steps are template recursions.
template <Layout L, int FMT_A, int FMT_B, bool MC>
struct KLoop {
    using Cf = Cfg<L>;
    using OA = typename Cf::OA;
    using OB = typename Cf::OB;
    using OSA = typename Cf::OSA;
    using OSB = typename Cf::OSB;

    template <int SET>
    __device__ __forceinline__ static void load_scales(Regs<L> &r, const bases &b) {
#pragma unroll
        for (int p = 0; p < Cf::A_FRAGS / 2; p++) {
            r.sca[SET][p] = OSA::load(b.sa + OSA::pair_off(p));
        }
#pragma unroll
        for (int p = 0; p < Cf::B_FRAGS / 2; p++) {
            r.scb[SET][p] = OSB::load(b.sb + OSB::pair_off(p));
        }
    }

    // The A fragments from I on that reload after WMMA X of a K step, reading K step T_NEXT at src.
    template <int Q, bool LAST, int X, int T_NEXT, int I>
    __device__ __forceinline__ static void reload_a(Regs<L> &r, uint32_t src) {
        if constexpr (Cf::PLAN.trig_a[Q][LAST][I] == X) {
            r.fa[I] = OA::load(src + OA::frag_off(I, T_NEXT));
        }
        if constexpr (I + 1 < Cf::A_FRAGS) {
            reload_a<Q, LAST, X, T_NEXT, I + 1>(r, src);
        }
    }

    template <int Q, bool LAST, int X, int T_NEXT, int J>
    __device__ __forceinline__ static void reload_b(Regs<L> &r, uint32_t src) {
        if constexpr (Cf::PLAN.trig_b[Q][LAST][J] == X) {
            r.fb[J] = OB::load(src + OB::frag_off(J, T_NEXT));
        }
        if constexpr (J + 1 < Cf::B_FRAGS) {
            reload_b<Q, LAST, X, T_NEXT, J + 1>(r, src);
        }
    }

    // WMMAs X.. of K step T of a block of parity PAR; the stage flip precedes WMMA PRE_BAR of the last K step.
    template <int PAR, int T, int X>
    __device__ __forceinline__ static void wmmas(Regs<L> &r, const bases &cur, const bases &nxt, int wid) {
        constexpr bool LAST = T == K_STEPS - 1;
        constexpr int Q      = (PAR * K_STEPS + T) & 1;
        constexpr int T_NEXT = LAST ? 0 : T + 1;
        constexpr int I      = Cf::PLAN.i[Q][X];
        constexpr int J      = Cf::PLAN.j[Q][X];
        // Source of the next K step: same stage one K step on, or the next stage's first.
        const uint32_t a_src = LAST ? nxt.a : cur.a;
        const uint32_t b_src = LAST ? nxt.b : cur.b;

        if constexpr (LAST && X == Cf::PRE_BAR) {
            // Stage flip: this wave's reads of cur are done and the next block has landed.
            kittens::sched::compiler_fence();
            kittens::sync::wait_ds<0>();
            kittens::sync::wait_tdm<0>();
            kittens::sched::compiler_fence();
            kittens::sync::arrive();
            kittens::sync::wait();
            if constexpr (MC) {
                if (wid == 0) {
                    kittens::cluster::arrive();
                }
            }
            kittens::sched::compiler_fence();
            load_scales<PAR ^ 1>(r, nxt);
            reload_a<Q, LAST, Cf::PRE_BAR - 1, T_NEXT, 0>(r, a_src);
            reload_b<Q, LAST, Cf::PRE_BAR - 1, T_NEXT, 0>(r, b_src);
            kittens::sched::compiler_fence();
        }
        wmma<FMT_A, FMT_B, I & 1, J & 1, Cf::PLAN.ra[Q][X], Cf::PLAN.rb[Q][X]>(
            r.acc[I][J], r.fa[I], r.fb[J], r.sca[PAR][I / 2][T], r.scb[PAR][J / 2][T]);
        if constexpr (!(LAST && X == Cf::PRE_BAR - 1)) {
            kittens::sched::compiler_fence();
            reload_a<Q, LAST, X, T_NEXT, 0>(r, a_src);
            reload_b<Q, LAST, X, T_NEXT, 0>(r, b_src);
            kittens::sched::compiler_fence();
        }
        if constexpr (X + 1 < Cf::WMMA_STEP) {
            wmmas<PAR, T, X + 1>(r, cur, nxt, wid);
        }
    }

    // K-block kb; PAR, its parity, names the scale set and the stage.
    template <int PAR>
    __device__ __forceinline__ static void block(Regs<L> &r, const bases *bs, const Fill &fill, int wid, int kb,
                                                 int k_blocks) {
        const bases cur = bs[PAR];
        const bases nxt = bs[PAR ^ 1];

        // Fill block kb + 1 into the stage block kb - 1 used.
        const int next = kb + STAGES - 1;
        if (next < k_blocks) {
            if (wid < 4) {
                tdm_issue(fill.d, uint32_t(next % STAGES) * fill.stage_bytes, uint64_t(next) * fill.step);
            }
        }
        kittens::sched::compiler_fence();

        static_assert(K_STEPS == 2);
        wmmas<PAR, 0, 0>(r, cur, nxt, wid);
        wmmas<PAR, 1, 0>(r, cur, nxt, wid);

        if constexpr (MC) {
            // Peers are done with the stage refilled next.
            kittens::cluster::wait();
        }
        kittens::sched::compiler_fence();
    }
};

// C is staged in LDS as D itself for a 16-bit D without epilogue (one rounding), else as fp32.
template <Epi E, bool ACC, typename OutT>
using staged_t = std::conditional_t<E == NONE && !ACC && sizeof(OutT) == 2, OutT, float>;

// SPLIT: the fp32 partial straight from the accumulators, 32 contiguous bytes per lane per fragment.
template <bool EDGE, int A_FRAGS, int B_FRAGS>
__device__ __forceinline__ void store_split(float *d, const v8f (&acc)[A_FRAGS][B_FRAGS], int row0, int col0,
                                            int m_rem, int n_rem, int M) {
#pragma unroll
    for (int i = 0; i < A_FRAGS; i++) {
#pragma unroll
        for (int j = 0; j < B_FRAGS; j++) {
            const int row = row0 + i * 16;
            const int col = col0 + j * 16;
            if (!EDGE || (row < m_rem && col < n_rem)) {
                *reinterpret_cast<v8f *>(d + size_t(col) * M + row) = acc[i][j];
            }
        }
    }
}

// SPLIT: workgroup z writes the fp32 partial of K-blocks [nb z / splits, nb (z + 1) / splits), slice 0 also the
// K % BLOCK_K head, to D + z M N. The K % BLOCK_K head is block 0; the TDM zero-fills it and rows past M and N.
template <Layout L, int FMT_A, int FMT_B, int CX, int CY, Epi E, bool ACC, typename OutT, typename SideT>
__device__ __forceinline__ void mxfp8_gemm_body(const uint8_t *A, const uint8_t *B, const uint8_t *SA,
                                                const uint8_t *SB, void *D, const void *bias, void *aux, float alpha,
                                                float beta, int M, int N, int K) {
    using Cf = Cfg<L>;
    using OA = typename Cf::OA;
    using OB = typename Cf::OB;
    using OSA = typename Cf::OSA;
    using OSB = typename Cf::OSB;
    constexpr int A_FRAGS = Cf::A_FRAGS;
    constexpr int B_FRAGS = Cf::B_FRAGS;
    constexpr bool MC = CX * CY > 1 && ARCH_MCAST;

    extern __shared__ __attribute__((aligned(16))) uint8_t smem[];
    const uint32_t lds0 = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&smem[0]));

    kittens::sched::lock_simd();

    int k0 = 0;
    int kz = K;
    if constexpr (E == SPLIT) {
        const int splits = gridDim.z;
        const int z      = blockIdx.z;
        const int nb     = K / BLOCK_K;
        const int kh     = K % BLOCK_K;
        const int b0     = nb * z / splits;
        const int b1     = nb * (z + 1) / splits;
        k0 = z == 0 ? 0 : kh + b0 * BLOCK_K;
        kz = (b1 - b0) * BLOCK_K + (z == 0 ? kh : 0);
        D = static_cast<float *>(D) + size_t(z) * M * N;
    }
    const int tile_m   = blockIdx.x;
    const int tile_n   = blockIdx.y;
    const int k_blocks = ceil_div(kz, BLOCK_K);
    const int k_head   = kz % BLOCK_K ? kz % BLOCK_K : BLOCK_K;
    // Never taken (the host keeps K > 0 and splits <= K / BLOCK_K), but without it the accumulator zeroing grows.
    if (k_blocks <= 0) {
        return;
    }

    const int lane      = threadIdx.x % kittens::WARP_THREADS;
    const int wid       = __builtin_amdgcn_readfirstlane(threadIdx.x / kittens::WARP_THREADS);
    const int warp_r    = wid / Cf::WARPS_N;
    const int warp_c    = wid % Cf::WARPS_N;
    const int lane16    = lane % 16;
    const int lane_half = lane / 16;

    // Per-lane LDS bases, one per stage, in their own VGPRs: no address VALU writes into WMMA source VGPRs
    // (ISA 7.12.1).
    const uint32_t a_lane0  = lds0 + Cf::OFF_A + OA::lane(lane, warp_r * Cf::WARP_M);
    const uint32_t b_lane0  = lds0 + Cf::OFF_B + OB::lane(lane, warp_c * Cf::WARP_N);
    const uint32_t sa_lane0 = lds0 + Cf::OFF_SA + OSA::lane(lane, warp_r * Cf::WARP_M);
    const uint32_t sb_lane0 = lds0 + Cf::OFF_SB + OSB::lane(lane, warp_c * Cf::WARP_N);

    // Multicast sets: WGs sharing tile_m receive A, WGs sharing tile_n receive B (0: set of one).
    constexpr uint32_t A_SET = MC && CY > 1 ? cluster_col_bits(CX, CY) : 0u;
    constexpr uint32_t B_SET = MC && CX > 1 ? (1u << CX) - 1u : 0u;
    const uint32_t a_mask = A_SET << (blockIdx.x % CX);
    const uint32_t b_mask = B_SET << (uint32_t(CX) * (blockIdx.y % CY));

    const size_t scales_ld = scale_ld(K);
    const uint8_t *tiles[4];
    tiles[0] = Cf::TRANS_A ? A + size_t(tile_m) * BLOCK_M + size_t(k0) * M : A + size_t(tile_m) * BLOCK_M * K + k0;
    tiles[1] = Cf::TRANS_B ? B + size_t(tile_n) * BLOCK_N + size_t(k0) * N : B + size_t(tile_n) * BLOCK_N * K + k0;
    tiles[2] = Cf::TRANS_A ? SA + size_t(tile_m) * BLOCK_M + size_t(k0 / MX_BLOCK) * M
                           : SA + size_t(tile_m) * BLOCK_M * scales_ld + k0 / MX_BLOCK;
    tiles[3] = Cf::TRANS_B ? SB + size_t(tile_n) * BLOCK_N + size_t(k0 / MX_BLOCK) * N
                           : SB + size_t(tile_n) * BLOCK_N * scales_ld + k0 / MX_BLOCK;

    const uint32_t m_rem = uint32_t(M - tile_m * BLOCK_M);
    const uint32_t n_rem = uint32_t(N - tile_n * BLOCK_N);
    Fill fill = {};
    make_fill<L>(fill, wid, lds0, tiles, a_mask, b_mask, M, N, K, m_rem, n_rem, k_head, 0);

    Regs<L> r;
#pragma unroll
    for (int i = 0; i < A_FRAGS; i++) {
#pragma unroll
        for (int j = 0; j < B_FRAGS; j++) {
            r.acc[i][j] = v8f{};
        }
    }

    if (wid < 4) {
        tdm_issue(fill.d, 0, 0);
    }
    if (E == SPLIT || k_head != BLOCK_K) {
        make_fill<L>(fill, wid, lds0, tiles, a_mask, b_mask, M, N, K, m_rem, n_rem, kz, BLOCK_K - k_head);
    }
    if constexpr (E == SPLIT) {
        make_uniform(fill);
    }
    kittens::sync::wait_tdm<0>();
    kittens::sched::compiler_fence();
    kittens::sync::arrive();
    kittens::sync::wait();
    if constexpr (MC) {
        // ISA 5.6.6: synchronize the workgroup first, then one wave per workgroup signals.
        if (wid == 0) {
            kittens::cluster::arrive();
        }
        kittens::cluster::wait();
    }
    kittens::sched::compiler_fence();

    bases bs[STAGES];
#pragma unroll
    for (int st = 0; st < STAGES; st++) {
        bs[st].a  = opaque(a_lane0 + st * OA::STAGE_BYTES);
        bs[st].b  = opaque(b_lane0 + st * OB::STAGE_BYTES);
        bs[st].sa = opaque(sa_lane0 + st * OSA::STAGE_BYTES);
        bs[st].sb = opaque(sb_lane0 + st * OSB::STAGE_BYTES);
    }

    using Loop = KLoop<L, FMT_A, FMT_B, MC>;
    Loop::template load_scales<0>(r, bs[0]);
#pragma unroll
    for (int i = 0; i < A_FRAGS; i++) {
        r.fa[i] = OA::load(bs[0].a + OA::frag_off(i, 0));
    }
#pragma unroll
    for (int j = 0; j < B_FRAGS; j++) {
        r.fb[j] = OB::load(bs[0].b + OB::frag_off(j, 0));
    }
    kittens::sched::compiler_fence();

    for (int kb = 0; kb < k_blocks; kb += 2) {
        Loop::template block<0>(r, bs, fill, wid, kb, k_blocks);
        if (kb + 1 < k_blocks) {
            Loop::template block<1>(r, bs, fill, wid, kb + 1, k_blocks);
        }
    }

    // Accumulator lane L holds column L % 16 and rows v + 8 (L / 16), v = 0..7 (ISA 7.12.2).
    const int row0 = warp_r * Cf::WARP_M + 8 * lane_half;
    const int col0 = warp_c * Cf::WARP_N + lane16;
    const bool edge = (tile_m + 1) * BLOCK_M > M || (tile_n + 1) * BLOCK_N > N || M % 8 != 0;
    if constexpr (E == SPLIT) {
        float *d = static_cast<float *>(D) + size_t(tile_n) * BLOCK_N * M + size_t(tile_m) * BLOCK_M;
        if (edge) {
            store_split<true>(d, r.acc, row0, col0, m_rem, n_rem, M);
        } else {
            store_split<false>(d, r.acc, row0, col0, m_rem, n_rem, M);
        }
        return;
    }

    using StT = staged_t<E, ACC, OutT>;
    constexpr int STRIDE = staged_stride(sizeof(StT));
    kittens::sync::wait_tdm<0>();
    kittens::sync::wait_ds<0>();
    kittens::sync::arrive();
    kittens::sync::wait();
#pragma unroll
    for (int i = 0; i < A_FRAGS; i++) {
#pragma unroll
        for (int j = 0; j < B_FRAGS; j++) {
            const uint32_t addr = lds0 + (col0 + j * 16) * STRIDE + (row0 + i * 16) * sizeof(StT);
            *lds_ptr<vec8<StT>>(addr) = __builtin_convertvector(r.acc[i][j], vec8<StT>);
        }
    }
    kittens::sync::wait_ds<0>();
    kittens::sync::arrive();
    kittens::sync::wait();
    OutT *d = static_cast<OutT *>(D);
    const SideT *bias_t = static_cast<const SideT *>(bias);
    SideT *aux_t = static_cast<SideT *>(aux);
    if (edge) {
        copy_out<StT, E, ACC, OutT, SideT, L != TN, true>(lds0, d, bias_t, aux_t, alpha, beta, M, N, tile_m, tile_n);
    } else {
        copy_out<StT, E, ACC, OutT, SideT, L != TN, false>(lds0, d, bias_t, aux_t, alpha, beta, M, N, tile_m, tile_n);
    }
}

template <Layout L, int FMT_A, int FMT_B, int CX, int CY, Epi E, bool ACC, typename OutT, typename SideT>
__global__ __cluster_dims__(CX, CY, 1) __launch_bounds__(NUM_THREADS, 1)
void mxfp8_gemm_kernel(const uint8_t *A, const uint8_t *B, const uint8_t *SA, const uint8_t *SB, void *D,
                       const void *bias, void *aux, float alpha, float beta, int M, int N, int K) {
    mxfp8_gemm_body<L, FMT_A, FMT_B, CX, CY, E, ACC, OutT, SideT>(A, B, SA, SB, D, bias, aux, alpha, beta, M, N, K);
}

template <Layout L, int FMT_A, int FMT_B, Epi E, bool ACC, typename OutT, typename SideT>
__global__ __launch_bounds__(NUM_THREADS, 1)
void mxfp8_gemm_kernel_nomc(const uint8_t *A, const uint8_t *B, const uint8_t *SA, const uint8_t *SB, void *D,
                            const void *bias, void *aux, float alpha, float beta, int M, int N, int K) {
    mxfp8_gemm_body<L, FMT_A, FMT_B, 1, 1, E, ACC, OutT, SideT>(A, B, SA, SB, D, bias, aux, alpha, beta, M, N, K);
}

// Cluster side for t tiles: the largest of 4, 2, 1 that divides t.
inline int cluster_side(int t) { return t % 4 == 0 ? 4 : t % 2 == 0 ? 2 : 1; }

typedef void (*kernel_t)(const uint8_t *, const uint8_t *, const uint8_t *, const uint8_t *, void *, const void *,
                         void *, float, float, int, int, int);

template <Layout L, int FMT_A, int FMT_B, int CX, int CY, Epi E, bool ACC, typename OutT, typename SideT>
void launch_grid(const GemmArgs &args, void *d, int splits) {
    kernel_t kernel;
    if constexpr (CX * CY > 1) {
        kernel = mxfp8_gemm_kernel<L, FMT_A, FMT_B, CX, CY, E, ACC, OutT, SideT>;
    } else {
        kernel = mxfp8_gemm_kernel_nomc<L, FMT_A, FMT_B, E, ACC, OutT, SideT>;
    }
    constexpr int lds = Cfg<L>::LDS_BYTES;
    kittens_check(hipFuncSetAttribute(reinterpret_cast<const void *>(kernel),
                                      hipFuncAttributeMaxDynamicSharedMemorySize, lds),
                  "gfx1250 mxfp8_gemm_kernel LDS");
    hipLaunchKernelGGL(kernel, dim3(ceil_div(args.M, BLOCK_M), ceil_div(args.N, BLOCK_N), splits), dim3(NUM_THREADS),
                       lds, args.stream, static_cast<const uint8_t *>(args.A), static_cast<const uint8_t *>(args.B),
                       static_cast<const uint8_t *>(args.scale_A), static_cast<const uint8_t *>(args.scale_B), d,
                       args.bias, args.aux_gelu, args.alpha, args.beta, args.M, args.N, args.K);
    kittens_check(hipGetLastError(), "gfx1250 mxfp8_gemm_kernel");
}

template <Layout L, int FMT_A, int FMT_B, int CX, Epi E, bool ACC, typename OutT, typename SideT>
void launch_cluster_y(const GemmArgs &args, void *d, int splits, int cy) {
    if (cy == 4) {
        launch_grid<L, FMT_A, FMT_B, CX, 4, E, ACC, OutT, SideT>(args, d, splits);
    } else if (cy == 2) {
        launch_grid<L, FMT_A, FMT_B, CX, 2, E, ACC, OutT, SideT>(args, d, splits);
    } else {
        launch_grid<L, FMT_A, FMT_B, CX, 1, E, ACC, OutT, SideT>(args, d, splits);
    }
}

template <Layout L, int FMT_A, int FMT_B, Epi E, bool ACC, typename OutT, typename SideT>
void launch_kernel(const GemmArgs &args, void *d, int splits, bool mcast) {
    const int cx = mcast ? cluster_side(ceil_div(args.M, BLOCK_M)) : 1;
    const int cy = mcast ? cluster_side(ceil_div(args.N, BLOCK_N)) : 1;
    if (cx == 4) {
        launch_cluster_y<L, FMT_A, FMT_B, 4, E, ACC, OutT, SideT>(args, d, splits, cy);
    } else if (cx == 2) {
        launch_cluster_y<L, FMT_A, FMT_B, 2, E, ACC, OutT, SideT>(args, d, splits, cy);
    } else {
        launch_cluster_y<L, FMT_A, FMT_B, 1, E, ACC, OutT, SideT>(args, d, splits, cy);
    }
}

// The compiled variants; any other combination launches nothing.
constexpr bool supported(Layout L, Epi E, bool acc, int out, int side) {
    const bool f32_out_bf16_side = out == KITTENS_FLOAT32 && side == KITTENS_BFLOAT16;
    switch (E) {
    case NONE:          return side == out;
    case BIAS:          return (!acc || L == TN) && (side == out || f32_out_bf16_side);
    case GELU_AUX:      return L == TN && !acc && side == KITTENS_BFLOAT16;
    case GELU_AUX_BIAS: return L == TN && !acc && (out == KITTENS_FLOAT16 ? side == out : side == KITTENS_BFLOAT16);
    default:            return false;
    }
}

template <Layout L, int FMT_A, int FMT_B, Epi E, bool ACC, typename OutT, typename SideT>
bool launch_variant(const GemmArgs &args, void *workspace, int splits, bool mcast) {
    if constexpr (!supported(L, E, ACC, dtype_of<OutT>, dtype_of<SideT>)) {
        return false;
    } else {
        if constexpr (E == NONE) {
            if (splits > 1) {
                launch_kernel<L, FMT_A, FMT_B, SPLIT, false, float, float>(args, workspace, splits, mcast);
                splitk_reduce(args, workspace, splits);
                return true;
            }
        }
        launch_kernel<L, FMT_A, FMT_B, E, ACC, OutT, SideT>(args, args.C, 1, mcast);
        return true;
    }
}

template <Layout L, int FMT_A, int FMT_B, Epi E, bool ACC, typename OutT>
bool switch_side(const GemmArgs &args, void *workspace, int splits, bool mcast) {
    switch (side_dtype(args)) {
    case KITTENS_FLOAT32:  return launch_variant<L, FMT_A, FMT_B, E, ACC, OutT, float>(args, workspace, splits, mcast);
    case KITTENS_FLOAT16:
        return launch_variant<L, FMT_A, FMT_B, E, ACC, OutT, _Float16>(args, workspace, splits, mcast);
    case KITTENS_BFLOAT16: return launch_variant<L, FMT_A, FMT_B, E, ACC, OutT, __bf16>(args, workspace, splits, mcast);
    default:               return false;
    }
}

template <Layout L, int FMT_A, int FMT_B, Epi E, bool ACC>
bool switch_out(const GemmArgs &args, void *workspace, int splits, bool mcast) {
    switch (args.out_dtype) {
    case KITTENS_FLOAT32:  return switch_side<L, FMT_A, FMT_B, E, ACC, float>(args, workspace, splits, mcast);
    case KITTENS_FLOAT16:  return switch_side<L, FMT_A, FMT_B, E, ACC, _Float16>(args, workspace, splits, mcast);
    case KITTENS_BFLOAT16: return switch_side<L, FMT_A, FMT_B, E, ACC, __bf16>(args, workspace, splits, mcast);
    default:               return false;
    }
}

template <Layout L, int FMT_A, int FMT_B, Epi E>
bool switch_acc(const GemmArgs &args, void *workspace, int splits, bool mcast) {
    if (args.beta != 0.f) {
        return switch_out<L, FMT_A, FMT_B, E, true>(args, workspace, splits, mcast);
    }
    return switch_out<L, FMT_A, FMT_B, E, false>(args, workspace, splits, mcast);
}

template <Layout L, int FMT_A, int FMT_B>
bool switch_epi(const GemmArgs &args, void *workspace, int splits, Epi epi, bool mcast) {
    switch (epi) {
    case NONE:          return switch_acc<L, FMT_A, FMT_B, NONE>(args, workspace, splits, mcast);
    case BIAS:          return switch_acc<L, FMT_A, FMT_B, BIAS>(args, workspace, splits, mcast);
    case GELU_AUX:      return switch_acc<L, FMT_A, FMT_B, GELU_AUX>(args, workspace, splits, mcast);
    case GELU_AUX_BIAS: return switch_acc<L, FMT_A, FMT_B, GELU_AUX_BIAS>(args, workspace, splits, mcast);
    default:            return false;
    }
}

}  // namespace mxfp8_gemm

// Format codes: 0 e4m3, 1 e5m2.
template <Layout L>
bool launch_mxfp8(const GemmArgs &args, void *workspace, int splits, Epi epi, bool mcast) {
    const int fmt_a = fp8_code(static_cast<KittensDType>(args.a_dtype));
    const int fmt_b = fp8_code(static_cast<KittensDType>(args.b_dtype));
    if (fmt_a == 0) {
        return fmt_b == 0 ? mxfp8_gemm::switch_epi<L, 0, 0>(args, workspace, splits, epi, mcast)
                          : mxfp8_gemm::switch_epi<L, 0, 1>(args, workspace, splits, epi, mcast);
    }
    return fmt_b == 0 ? mxfp8_gemm::switch_epi<L, 1, 0>(args, workspace, splits, epi, mcast)
                      : mxfp8_gemm::switch_epi<L, 1, 1>(args, workspace, splits, epi, mcast);
}

}  // namespace te_kittens::cdna5
