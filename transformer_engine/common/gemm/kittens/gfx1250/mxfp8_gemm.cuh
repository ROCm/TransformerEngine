/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

/* gfx1250 MXFP8 GEMM, column-major D (M x N); A, B: TN [M][K] [N][K], NN [K][M] [N][K], NT
 * [K][M] [K][N]; e8m0 scales [rows][K/32], [K/32][rows] if M/N-contiguous, with rows of K/32
 * padded to 4 bytes. M, N % 4, K % 32: the TDM zero-fills past D's edge and edge tiles store
 * masked. */

#pragma once

#include "hip/hip_runtime.h"
#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <utility>
#include "kittens.cuh"
#include "gemm.h"

namespace te_kittens::gfx1250 {
namespace mxfp8_gemm {

constexpr int BLOCK_M      = 256;
constexpr int BLOCK_N      = 256;
constexpr int BLOCK_K      = 256;      // fp8 K elements per LDS stage
constexpr int S            = 2;        // LDS stages
constexpr int K_STEP       = 128;      // WMMA K depth
constexpr int KSUB         = BLOCK_K / K_STEP;
constexpr int SCB          = BLOCK_K / 32;    // scale bytes per row per stage
constexpr int NUM_WARPS    = 8;
constexpr int NUM_THREADS  = NUM_WARPS * 32;
constexpr int RELOAD_GROUP = 2;        // rolling fragments are reloaded in groups of this many

typedef int   v2i  __attribute__((ext_vector_type(2)));
typedef int   v4i  __attribute__((ext_vector_type(4)));
typedef int   v8i  __attribute__((ext_vector_type(8)));
typedef int   v16i __attribute__((ext_vector_type(16)));
typedef float v8f  __attribute__((ext_vector_type(8)));
template <typename T> using vec8 = T __attribute__((ext_vector_type(8)));
typedef __attribute__((address_space(3))) v2i lds_v2i;

template <int N, typename F>
__device__ __forceinline__ void static_for(F &&f) {
    [&]<int... I>(std::integer_sequence<int, I...>) {
        (f(std::integral_constant<int, I>{}), ...);
    }(std::make_integer_sequence<int, N>{});
}

template <typename T>
__device__ __forceinline__ __attribute__((address_space(3))) T *lds_ptr(uint32_t addr) {
    return reinterpret_cast<__attribute__((address_space(3))) T *>(static_cast<uintptr_t>(addr));
}

__device__ __forceinline__ v4i lds_b128(uint32_t addr) {
    return *lds_ptr<v4i>(addr);
}

__device__ __forceinline__ v2i lds_tr8(uint32_t addr) {
    return __builtin_amdgcn_ds_load_tr8_b64_v2i32(const_cast<lds_v2i *>(lds_ptr<v2i>(addr)));
}

// One operand of R tile rows: [R][BLOCK_K + 16 B], or (TRANS) [BLOCK_K][R + 16 B] with K rows
// permuted by the TDM so that one ds_load_tr8_b64 reads 16 consecutive rows (ISA 10.9.2).
template <bool TRANS, int R, int NF>
struct opnd {
    static constexpr int ROWB = BLOCK_K + 16;
    static constexpr int XS   = R + 16;
    static constexpr int ST   = TRANS ? BLOCK_K * XS : R * ROWB;
    static constexpr int NLD  = TRANS ? 8 : 4;

    __host__ __device__ static uint32_t lane(int L, int w0) {
        const int s = L & 15, H = L >> 4;
        if constexpr (TRANS) {
            const int j = (s & 3) + 4 * (s >> 3), ch = (s >> 2) & 1;
            return (8 * H + j) * XS + w0 + 8 * ch;
        } else {
            return (w0 + s) * ROWB + 16 * H;
        }
    }
    static constexpr uint32_t off(int i, int T) {
        return TRANS ? T * K_STEP * XS + i * 16 : i * 16 * ROWB + T * K_STEP;
    }
    static constexpr uint32_t imm(int x) {
        return TRANS ? (32 * (x / 2) + 16 * (x % 2)) * XS : 32 * x;
    }

    __device__ static v16i load(uint32_t addr) {
        if constexpr (TRANS) {
            const v2i x0 = lds_tr8(addr + imm(0)), x1 = lds_tr8(addr + imm(1)),
                      x2 = lds_tr8(addr + imm(2)), x3 = lds_tr8(addr + imm(3)),
                      x4 = lds_tr8(addr + imm(4)), x5 = lds_tr8(addr + imm(5)),
                      x6 = lds_tr8(addr + imm(6)), x7 = lds_tr8(addr + imm(7));
            const v4i a = __builtin_shufflevector(x0, x1, 0, 1, 2, 3);
            const v4i b = __builtin_shufflevector(x2, x3, 0, 1, 2, 3);
            const v4i c = __builtin_shufflevector(x4, x5, 0, 1, 2, 3);
            const v4i d = __builtin_shufflevector(x6, x7, 0, 1, 2, 3);
            const v8i lo = __builtin_shufflevector(a, b, 0, 1, 2, 3, 4, 5, 6, 7);
            const v8i hi = __builtin_shufflevector(c, d, 0, 1, 2, 3, 4, 5, 6, 7);
            return __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                                           15);
        } else {
            const v4i q0 = lds_b128(addr + imm(0)), q1 = lds_b128(addr + imm(1)),
                      q2 = lds_b128(addr + imm(2)), q3 = lds_b128(addr + imm(3));
            const v8i lo = __builtin_shufflevector(q0, q1, 0, 1, 2, 3, 4, 5, 6, 7);
            const v8i hi = __builtin_shufflevector(q2, q3, 0, 1, 2, 3, 4, 5, 6, 7);
            return __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                                           15);
        }
    }
    static_assert(off(NF - 1, KSUB - 1) + imm(NLD - 1) < 65536, "offsets must fit the DS offset");
};

// Scales of R tile rows, [R][SCB] or (TRANS) [SCB][R + 32 B]: lane L of fragment pair p gets
// both sub-steps' e8m0 dwords of row 32p + L; SCL_OPSEL picks the lane half.
template <bool TRANS, int R>
struct scales {
    static constexpr int SS = R + 32;
    static constexpr int ST = TRANS ? SCB * SS : R * SCB;

    __host__ __device__ static uint32_t lane(int L, int w0) {
        if constexpr (TRANS) {
            const int s = L & 15, H = L >> 4;
            return ((s & 3) + 4 * (s >> 3)) * SS + w0 + 16 * H + 8 * ((s >> 2) & 1);
        } else {
            return (w0 + L) * SCB;
        }
    }
    static constexpr uint32_t off(int p) { return TRANS ? 32 * p : p * 32 * SCB; }
    __device__ static v2i load(uint32_t addr) {
        if constexpr (TRANS) {
            return lds_tr8(addr);
        } else {
            return *lds_ptr<v2i>(addr);
        }
    }
};

// WMMA order and reload triggers per sub-step parity: a fragment reloads after its group's last
// read, and in a block's last sub-step not before the stage flip at PRE_BAR - 1.
template <int A_FRAGS, int B_FRAGS>
struct plan_t {
    int i[2][A_FRAGS * B_FRAGS], j[2][A_FRAGS * B_FRAGS];
    bool ra[2][A_FRAGS * B_FRAGS], rb[2][A_FRAGS * B_FRAGS];
    int trig_a[2][2][A_FRAGS], trig_b[2][2][B_FRAGS];
};

template <int A_FRAGS, int B_FRAGS, bool ALT_ORDER, int PRE_BAR>
constexpr plan_t<A_FRAGS, B_FRAGS> make_plan() {
    constexpr int NM = A_FRAGS * B_FRAGS;
    plan_t<A_FRAGS, B_FRAGS> p{};
    for (int q = 0; q < 2; q++) {
        int t = 0;
        if (ALT_ORDER && q == 1) {
            for (int j = 0; j < B_FRAGS; j++) {
                for (int i = 0; i < A_FRAGS; i++) {
                    p.i[q][t] = i; p.j[q][t] = j; p.ra[q][t] = false; p.rb[q][t] = i > 0; t++;
                }
            }
        } else {
            for (int i = 0; i < A_FRAGS; i++) {
                for (int j = 0; j < B_FRAGS; j++) {
                    p.i[q][t] = i; p.j[q][t] = j; p.ra[q][t] = j > 0; p.rb[q][t] = false; t++;
                }
            }
        }
        int last_a[A_FRAGS] = {}, last_b[B_FRAGS] = {};
        for (int x = 0; x < NM; x++) {
            last_a[p.i[q][x]] = x;
            last_b[p.j[q][x]] = x;
        }
        for (int i = 0; i < A_FRAGS; i++) {
            const int g = std::min((i / RELOAD_GROUP + 1) * RELOAD_GROUP - 1, A_FRAGS - 1);
            last_a[i] = std::max(last_a[g], last_a[i]);
        }
        for (int j = 0; j < B_FRAGS; j++) {
            const int g = std::min((j / RELOAD_GROUP + 1) * RELOAD_GROUP - 1, B_FRAGS - 1);
            last_b[j] = std::max(last_b[g], last_b[j]);
        }
        for (int last = 0; last < 2; last++) {
            const int lo = last ? PRE_BAR - 1 : 0;
            for (int i = 0; i < A_FRAGS; i++) {
                p.trig_a[q][last][i] = std::max(std::min(last_a[i], NM - 1), lo);
            }
            for (int j = 0; j < B_FRAGS; j++) {
                p.trig_b[q][last][j] = std::max(std::min(last_b[j], NM - 1), lo);
            }
        }
    }
    return p;
}

// NN: 4x2 warps of 64x128, WMMA nesting alternating per sub-step; TN and NT: 2x4 warps of 128x64.
template <Layout L>
struct Cfg {
    static constexpr bool TRANS_A = L != TN;
    static constexpr bool TRANS_B = L == NT;
    static constexpr int WARPS_M = L == NN ? 4 : 2;
    static constexpr int WARPS_N = NUM_WARPS / WARPS_M;
    static constexpr int WARP_M  = BLOCK_M / WARPS_M;
    static constexpr int WARP_N  = BLOCK_N / WARPS_N;
    static constexpr int A_FRAGS = WARP_M / 16;
    static constexpr int B_FRAGS = WARP_N / 16;
    static constexpr int NM      = A_FRAGS * B_FRAGS;
    static constexpr int PRE_BAR = 3 * NM / 8;    // last sub-step WMMAs issued before the flip

    using OA  = opnd<TRANS_A, BLOCK_M, A_FRAGS>;
    using OB  = opnd<TRANS_B, BLOCK_N, B_FRAGS>;
    using OSA = scales<TRANS_A, BLOCK_M>;
    using OSB = scales<TRANS_B, BLOCK_N>;

    // All A stages, all B stages, then the scale stages; the C staging reuses offset 0.
    static constexpr int OFF_A     = 0;
    static constexpr int OFF_B     = OFF_A + S * OA::ST;
    static constexpr int OFF_SA    = OFF_B + S * OB::ST;
    static constexpr int OFF_SB    = OFF_SA + S * OSA::ST;
    static constexpr int LDS_BYTES = OFF_SB + S * OSB::ST;

    static constexpr plan_t<A_FRAGS, B_FRAGS> PLAN =
        make_plan<A_FRAGS, B_FRAGS, L == NN, PRE_BAR>();

    static_assert(A_FRAGS % 2 == 0 && B_FRAGS % 2 == 0,
                  "scale VGPRs pair fragments through SCL_OPSEL");
    static_assert(BLOCK_N * (BLOCK_M * 4 + 16) <= LDS_BYTES, "fp32 C staging fits the operand LDS");
    static_assert(LDS_BYTES <= 327680, "LDS is 320 KiB per workgroup");
};

// ---- TDM (ISA 10.11.4): dim d > 0 advances the global address by stride[d-1] elements; the
// LDS address is the tile's linear index plus pad_amount after every pad_interval DWords ----
struct tdm_desc {
    kittens::v4u32 g0;
    kittens::v8u32 g1;
    kittens::v4u32 g2, g3;
};

struct tdm_shape {
    uint32_t dsize_enc;             // 0: 1 byte, 2: 4 bytes
    uint32_t tdim[5], tile[5];      // tensor extents from the tile origin, tile extents (0: unused)
    uint64_t stride[4];             // element stride of dims 1..4
    uint32_t pad_en, pad_int_enc, pad_amt_enc;
};

__device__ __forceinline__ tdm_desc tdm_make(uint32_t lds_addr, const void *gptr,
                                             const tdm_shape &s, uint32_t mask) {
    const uint64_t ga = reinterpret_cast<uint64_t>(gptr);
    tdm_desc d;
    d.g0[0] = 1u;
    d.g0[1] = lds_addr;
    d.g0[2] = static_cast<uint32_t>(ga);
    d.g0[3] = (static_cast<uint32_t>(ga >> 32) & 0x01FFFFFFu) | (2u << 30);
    d.g1[0] = (s.dsize_enc << 16) | (s.pad_en << 20) | (s.pad_int_enc << 22) |
              (s.pad_amt_enc << 25) | (mask & 0xFFFFu);
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
    const uint64_t ga = ((uint64_t(d.g0[3] & 0x01FFFFFFu) << 32) | d.g0[2]) + g_off;
    g0[1] = d.g0[1] + lds_off;
    g0[2] = static_cast<uint32_t>(ga);
    g0[3] = (static_cast<uint32_t>(ga >> 32) & 0x01FFFFFFu) | (2u << 30);
    kittens::emit_tensor_load_to_lds(g0, d.g1, d.g2, d.g3);
}

constexpr uint32_t ilog2(uint32_t x) { return x <= 1 ? 0 : 1 + ilog2(x / 2); }

// TDM shape of one K-block of R of Rtot rows, rem rows and kx K from the tile origin; step: global
// bytes per K-block. TRANS uses dims (row, j, h, p, q) of K = 32q + 16h + 8p + j.
template <bool TRANS, int R>
__device__ __forceinline__ tdm_shape operand_shape(int K, int kx, uint32_t Rtot, uint32_t rem,
                                                   uint32_t &step) {
    if constexpr (TRANS) {
        step = uint32_t(BLOCK_K) * Rtot;
        return tdm_shape{2, {rem / 4, 8, 2, 2, uint32_t(kx / 32)}, {R / 4, 8, 2, 2, BLOCK_K / 32},
                         {Rtot / 4, 16ull * Rtot / 4, 8ull * Rtot / 4, 32ull * Rtot / 4},
                         1, ilog2(R / 4) - 1, 3};
    } else {
        step = BLOCK_K;
        return tdm_shape{2, {uint32_t(kx / 4), rem, 0, 0, 0}, {BLOCK_K / 4, R, 0, 0, 0},
                         {uint64_t(K / 4), 0, 0, 0}, 1, ilog2(BLOCK_K / 4) - 1, 3};
    }
}

// Bytes per row of row-wise scales: K/32 rounded up to 4, TE's gfx1250 scale layout.
__host__ __device__ constexpr int scale_ld(int K) { return (K / 32 + 3) / 4 * 4; }

// Scales of one K-block: [R][SCB] of [Rtot][scale_ld], or (TRANS) [SCB][R + 32 B] of [K/32][Rtot].
// Row-wise scales move as DWords when K % 128 == 0, else as bytes, so the head ends exactly.
template <bool TRANS, int R>
__device__ __forceinline__ tdm_shape scale_shape(int K, int kx, uint32_t Rtot, uint32_t rem,
                                                 uint32_t &step) {
    const uint32_t ks = scale_ld(K), ksx = kx / 32;
    if constexpr (TRANS) {
        step = uint32_t(SCB) * Rtot;
        return tdm_shape{2, {rem / 4, ksx, 0, 0, 0}, {R / 4, SCB, 0, 0, 0}, {Rtot / 4, 0, 0, 0},
                         1, ilog2(R / 4) - 1, 7};
    } else {
        step = SCB;
        if (K % 128 != 0) {
            return tdm_shape{0, {ksx, rem, 0, 0, 0}, {SCB, R, 0, 0, 0}, {ks, 0, 0, 0}, 0, 0, 0};
        }
        return tdm_shape{2, {ksx / 4, rem, 0, 0, 0}, {SCB / 4, R, 0, 0, 0}, {ks / 4, 0, 0, 0},
                         0, 0, 0};
    }
}

// (A fmt, A, B fmt, B, C mod, C, A scale opsel, A scale fmt (e8m0), A scale, B ..., reuse A, B)
template <int FMT_A, int FMT_B, int SEL_A, int SEL_B, bool RA, bool RB>
__device__ __forceinline__ void wmma(v8f &c, const v16i &a, const v16i &b, int sa, int sb) {
    c = __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(FMT_A, a, FMT_B, b, (short)0, c,
                                                         SEL_A, 0, sa, SEL_B, 0, sb, RA, RB);
}

// Cluster bits (x fastest) of the WGs in one column of a CX x CY cluster.
constexpr uint32_t cluster_col_bits(int cx, int cy) {
    return cy == 0 ? 0u : 1u | (cluster_col_bits(cx, cy - 1) << cx);
}

// The GEMM a tile belongs to; D holds OutT (SPLIT: fp32 partials), bias and aux SideT.
struct Gemm {
    const uint8_t *A;
    const uint8_t *B;
    const uint8_t *SA;
    const uint8_t *SB;
    void *D;
    const void *bias;
    void *aux;
    float alpha, beta;
    int M, N, K;
};

// C is staged in LDS as D itself for a 16-bit D without epilogue (one rounding), else as fp32.
template <Epi E, bool ACC, typename OutT>
using staged_t = std::conditional_t<E == NONE && !ACC && sizeof(OutT) == 2, OutT, float>;

// The first n of the 8 elements at p, one vector access when vec (then n is 8).
template <typename T>
__device__ __forceinline__ v8f load8(const T *p, int n, bool vec) {
    if (vec) {
        return __builtin_convertvector(*reinterpret_cast<const vec8<T> *>(p), v8f);
    }
    v8f v{};
#pragma unroll
    for (int e = 0; e < 8; e++) {
        if (e < n) {
            v[e] = static_cast<float>(p[e]);
        }
    }
    return v;
}

template <typename T, typename V>
__device__ __forceinline__ void store8(T *p, const V &v, int n, bool vec) {
    if (vec) {
        *reinterpret_cast<vec8<T> *>(p) = __builtin_convertvector(v, vec8<T>);
        return;
    }
#pragma unroll
    for (int e = 0; e < 8; e++) {
        if (e < n) {
            p[e] = static_cast<T>(v[e]);
        }
    }
}

// Copy-out of the C tile staged column-major in LDS: D as staged (times alpha with PLAIN_ALPHA), or
// alpha * C (+ beta * D) (+ bias), with GELU after storing that to aux. EDGE: D's rows and columns
// only, element-wise where a thread's 8 rows are not all in D or M % 8 != 0.
template <Epi E, bool ACC, typename OutT, typename SideT, bool PLAIN_ALPHA, bool EDGE>
__device__ __forceinline__ void copy_out(uint32_t lds0, OutT *__restrict__ d,
                                         const SideT *__restrict__ bias, SideT *__restrict__ aux,
                                         float alpha, float beta, int M, int N, int tile_m,
                                         int tile_n) {
    using StT = staged_t<E, ACC, OutT>;
    constexpr bool PLAIN = !std::is_same_v<StT, float>;
    constexpr int SSTR = BLOCK_M * sizeof(StT) + 16;
    constexpr int CH   = BLOCK_M / 8;
    constexpr bool ADD_BIAS = E == BIAS || E == GELU_AUX_BIAS;
    constexpr bool GELU     = E == GELU_AUX || E == GELU_AUX_BIAS;
    constexpr int UNROLL = GELU || EDGE ? 1 : 4;
    static_assert(NUM_THREADS % CH == 0, "a thread's rows are the same in every column");
    const int m8 = threadIdx.x % CH;
    const size_t tile = size_t(tile_n) * BLOCK_N * M + size_t(tile_m) * BLOCK_M + m8 * 8;
    const int rows = EDGE ? std::clamp(M - tile_m * BLOCK_M - m8 * 8, 0, 8) : 8;
    const int cols = EDGE ? std::min(N - tile_n * BLOCK_N, BLOCK_N) : BLOCK_N;
    const bool vec = !EDGE || (rows == 8 && M % 8 == 0);
    v8f b8{};
    if constexpr (ADD_BIAS) {
        b8 = load8(bias + tile_m * BLOCK_M + m8 * 8, rows, vec);
    }
#pragma unroll UNROLL
    for (int col = threadIdx.x / CH; col < cols; col += NUM_THREADS / CH) {
        const size_t off = tile + size_t(col) * M;
        const uint32_t src = lds0 + col * SSTR + m8 * 8 * sizeof(StT);
        if constexpr (PLAIN) {
            if (vec && (!PLAIN_ALPHA || alpha == 1.f)) {
                *reinterpret_cast<v4i *>(d + off) = *lds_ptr<v4i>(src);
            } else if constexpr (PLAIN_ALPHA) {
                store8(d + off, alpha * __builtin_convertvector(*lds_ptr<vec8<StT>>(src), v8f),
                       rows, vec);
            } else {
                store8(d + off, *lds_ptr<vec8<StT>>(src), rows, false);
            }
        } else {
            v8f v = alpha * *lds_ptr<v8f>(src);
            if constexpr (ACC) {
                v += beta * load8(d + off, rows, vec);
            }
            if constexpr (ADD_BIAS) {
                v += b8;
            }
            if constexpr (GELU) {
                store8(aux + off, v, rows, vec);
#pragma unroll
                for (int e = 0; e < 8; e++) {
                    v[e] = kittens::base_ops::gelu::op<float>(v[e]);
                }
            }
            store8(d + off, v, rows, vec);
        }
    }
}

// C tile (tile_m, tile_n) of g over K [k0, k0 + kz) at cluster position (cx, cy); a K % 256 head
// is block 0. The TDM zero-fills it and rows past M and N. Returns with every LDS read done.
template <Layout L, int FMT_A, int FMT_B, int CX, int CY, Epi E, bool ACC, typename OutT,
          typename SideT = OutT>
__device__ __forceinline__ void mxfp8_tile(const Gemm &g, int tile_m, int tile_n, uint32_t cx,
                                           uint32_t cy, int k0, int kz, uint32_t lds0) {
    using Cf  = Cfg<L>;
    using OA  = typename Cf::OA;
    using OB  = typename Cf::OB;
    using OSA = typename Cf::OSA;
    using OSB = typename Cf::OSB;
    constexpr int A_FRAGS = Cf::A_FRAGS;
    constexpr int B_FRAGS = Cf::B_FRAGS;
    constexpr int NM      = Cf::NM;
    constexpr auto PLAN = Cf::PLAN;
    constexpr bool MC = CX * CY > 1 && ARCH_MCAST;
    const int M = g.M, N = g.N, K = g.K;

    const int k_blocks = (kz + BLOCK_K - 1) / BLOCK_K;
    const int k_head   = kz % BLOCK_K ? kz % BLOCK_K : BLOCK_K;
    if (k_blocks <= 0) {
        return;
    }

    const int L_     = threadIdx.x & 31;
    const int wid    = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
    const int warp_r = wid / Cf::WARPS_N;
    const int warp_c = wid % Cf::WARPS_N;
    const int r16    = L_ & 15;
    const int h      = L_ >> 4;

    /* Per-lane LDS bases, one per stage, in their own VGPRs (no address VALU writes into WMMA
     * source VGPRs; ISA 7.12.1). The empty asm keeps them from being rematerialized. */
    struct bases { uint32_t a, b, sa, sb; };
    auto opaque = [](uint32_t x) { asm volatile("" : "+v"(x)); return x; };
    const uint32_t a_lane0  = lds0 + Cf::OFF_A + OA::lane(L_, warp_r * Cf::WARP_M);
    const uint32_t b_lane0  = lds0 + Cf::OFF_B + OB::lane(L_, warp_c * Cf::WARP_N);
    const uint32_t sa_lane0 = lds0 + Cf::OFF_SA + OSA::lane(L_, warp_r * Cf::WARP_M);
    const uint32_t sb_lane0 = lds0 + Cf::OFF_SB + OSB::lane(L_, warp_c * Cf::WARP_N);
    auto stage_bases = [&](int stage) {
        bases r;
        r.a  = opaque(a_lane0 + stage * OA::ST);
        r.b  = opaque(b_lane0 + stage * OB::ST);
        r.sa = opaque(sa_lane0 + stage * OSA::ST);
        r.sb = opaque(sb_lane0 + stage * OSB::ST);
        return r;
    };

    // Multicast sets: WGs sharing tile_m receive A, WGs sharing tile_n receive B (0: set of one).
    constexpr uint32_t A_SET = MC && CY > 1 ? cluster_col_bits(CX, CY) : 0u;
    constexpr uint32_t B_SET = MC && CX > 1 ? (1u << CX) - 1u : 0u;
    const uint32_t a_mask = A_SET << cx;
    const uint32_t b_mask = B_SET << (uint32_t(CX) * cy);

    // Tile origins (K-block 0) in the stored layouts.
    const uint8_t *a_tile  = Cf::TRANS_A ? g.A + size_t(tile_m) * BLOCK_M + size_t(k0) * M
                                         : g.A + size_t(tile_m) * BLOCK_M * K + k0;
    const uint8_t *b_tile  = Cf::TRANS_B ? g.B + size_t(tile_n) * BLOCK_N + size_t(k0) * N
                                         : g.B + size_t(tile_n) * BLOCK_N * K + k0;
    const uint8_t *sa_tile = Cf::TRANS_A ? g.SA + size_t(tile_m) * BLOCK_M + size_t(k0 / 32) * M
                                         : g.SA + size_t(tile_m) * BLOCK_M * scale_ld(K) + k0 / 32;
    const uint8_t *sb_tile = Cf::TRANS_B ? g.SB + size_t(tile_n) * BLOCK_N + size_t(k0 / 32) * N
                                         : g.SB + size_t(tile_n) * BLOCK_N * scale_ld(K) + k0 / 32;

    /* TDM: each filling wave builds its descriptor once (wave 0: A, 1: B, 2: A scales, 3: B
     * scales); the loop rewrites only the LDS and global address words. */
    tdm_desc fill_d{};
    uint32_t fill_stage_bytes = 0, fill_step = 0;
    const uint32_t m_rem = uint32_t(M - tile_m * BLOCK_M), n_rem = uint32_t(N - tile_n * BLOCK_N);
    // kx: K extent from the tile origin; back: K elements the tile origin moves back.
    auto make_fill = [&](int kx, int back) {
        auto mk = [&](uint32_t lds, const uint8_t *tile, const tdm_shape &sh, uint32_t mask) {
            fill_d = tdm_make(lds, tile - size_t(fill_step) * back / BLOCK_K, sh, mask);
        };
        if (wid == 0) {
            mk(lds0 + Cf::OFF_A, a_tile,
               operand_shape<Cf::TRANS_A, BLOCK_M>(K, kx, M, m_rem, fill_step), a_mask);
            fill_stage_bytes = OA::ST;
        } else if (wid == 1) {
            mk(lds0 + Cf::OFF_B, b_tile,
               operand_shape<Cf::TRANS_B, BLOCK_N>(K, kx, N, n_rem, fill_step), b_mask);
            fill_stage_bytes = OB::ST;
        } else if (wid == 2) {
            mk(lds0 + Cf::OFF_SA, sa_tile,
               scale_shape<Cf::TRANS_A, BLOCK_M>(K, kx, M, m_rem, fill_step), a_mask);
            fill_stage_bytes = OSA::ST;
        } else if (wid == 3) {
            mk(lds0 + Cf::OFF_SB, sb_tile,
               scale_shape<Cf::TRANS_B, BLOCK_N>(K, kx, N, n_rem, fill_step), b_mask);
            fill_stage_bytes = OSB::ST;
        }
    };
    make_fill(k_head, 0);

    auto issue_fill = [&](int stage, int kblock) {
        if (wid < 4) {
            tdm_issue(fill_d, uint32_t(stage) * fill_stage_bytes, uint64_t(kblock) * fill_step);
        }
    };

    // ---- registers ----
    v8f acc[A_FRAGS][B_FRAGS];
    static_for<A_FRAGS>([&](auto i) { static_for<B_FRAGS>([&](auto j) { acc[i][j] = v8f{}; }); });

    v16i fa[A_FRAGS];
    v16i fb[B_FRAGS];
    v2i sca[2][A_FRAGS / 2];
    v2i scb[2][B_FRAGS / 2];

    auto load_scales = [&](auto SET, uint32_t sa_base, uint32_t sb_base) {
        constexpr int ST = decltype(SET)::value;
        static_for<A_FRAGS / 2>([&](auto p) { sca[ST][p] = OSA::load(sa_base + OSA::off(p)); });
        static_for<B_FRAGS / 2>([&](auto p) { scb[ST][p] = OSB::load(sb_base + OSB::off(p)); });
    };

    // ---- prologue: fill block 0, wait for it ----
    issue_fill(0, 0);
    if (E == SPLIT || k_head != BLOCK_K) {
        make_fill(kz, BLOCK_K - k_head);
    }
    if constexpr (E == SPLIT) {
        // Without these the compiler keeps the descriptor in VGPRs and re-reads it every K-block.
        for (int i = 0; i < 4; i++) {
            fill_d.g0[i] = __builtin_amdgcn_readfirstlane(fill_d.g0[i]);
        }
        for (int i = 0; i < 8; i++) {
            fill_d.g1[i] = __builtin_amdgcn_readfirstlane(fill_d.g1[i]);
        }
        for (int i = 0; i < 4; i++) {
            fill_d.g2[i] = __builtin_amdgcn_readfirstlane(fill_d.g2[i]);
        }
        for (int i = 0; i < 4; i++) {
            fill_d.g3[i] = __builtin_amdgcn_readfirstlane(fill_d.g3[i]);
        }
        fill_stage_bytes = __builtin_amdgcn_readfirstlane(fill_stage_bytes);
        fill_step        = __builtin_amdgcn_readfirstlane(fill_step);
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

    const bases bs[2] = {stage_bases(0), stage_bases(1)};

    load_scales(std::integral_constant<int, 0>{}, bs[0].sa, bs[0].sb);
    static_for<A_FRAGS>([&](auto i) { fa[i] = OA::load(bs[0].a + OA::off(i, 0)); });
    static_for<B_FRAGS>([&](auto j) { fb[j] = OB::load(bs[0].b + OB::off(j, 0)); });
    kittens::sched::compiler_fence();

    // ---- one K-block; PAR is the block's parity, which names the scale set and the stage ----
    auto block = [&]<int PAR>(int kb) {
        const bases cur = bs[PAR];
        const bases nxt = bs[PAR ^ 1];

        {   // Fill block kb+1 into the stage block kb-1 used.
            const int fk = kb + S - 1;
            if (fk < k_blocks) {
                issue_fill(fk % S, fk);
            }
        }
        kittens::sched::compiler_fence();

        static_for<KSUB>([&](auto Tc) {
            constexpr int T = decltype(Tc)::value;
            constexpr bool LST = T == KSUB - 1;
            constexpr int Q   = (PAR * KSUB + T) & 1;
            constexpr int TN_ = LST ? 0 : T + 1;    // sub-step the reloads read

            // Source of the next sub-step: same stage one sub-step on, or the next stage's first.
            const uint32_t a_src = LST ? nxt.a : cur.a;
            const uint32_t b_src = LST ? nxt.b : cur.b;

            auto loads_after = [&](auto Xc) {
                constexpr int X = decltype(Xc)::value;
                static_for<A_FRAGS>([&](auto i) {
                    if constexpr (PLAN.trig_a[Q][LST][i] == X) {
                        fa[i] = OA::load(a_src + OA::off(i, TN_));
                    }
                });
                static_for<B_FRAGS>([&](auto j) {
                    if constexpr (PLAN.trig_b[Q][LST][j] == X) {
                        fb[j] = OB::load(b_src + OB::off(j, TN_));
                    }
                });
            };

            static_for<NM>([&](auto Xc) {
                constexpr int X = decltype(Xc)::value;
                if constexpr (LST && X == Cf::PRE_BAR) {
                    // ---- stage flip: this wave's reads of `cur` are done; block kb+1 landed ----
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
                    load_scales(std::integral_constant<int, PAR ^ 1>{}, nxt.sa, nxt.sb);
                    loads_after(std::integral_constant<int, Cf::PRE_BAR - 1>{});
                    kittens::sched::compiler_fence();
                }
                constexpr int I = PLAN.i[Q][X], J = PLAN.j[Q][X];
                wmma<FMT_A, FMT_B, I & 1, J & 1, PLAN.ra[Q][X], PLAN.rb[Q][X]>(
                    acc[I][J], fa[I], fb[J], sca[PAR][I / 2][T], scb[PAR][J / 2][T]);
                if constexpr (!(LST && X == Cf::PRE_BAR - 1)) {
                    kittens::sched::compiler_fence();
                    loads_after(Xc);
                    kittens::sched::compiler_fence();
                }
            });
        });

        if constexpr (MC) {
            kittens::cluster::wait();      // peers are done with the stage refilled next
        }
        kittens::sched::compiler_fence();
    };

    for (int kb = 0; kb < k_blocks; kb += 2) {
        block.template operator()<0>(kb);
        if (kb + 1 < k_blocks) {
            block.template operator()<1>(kb + 1);
        }
    }

    // Accumulator lane L holds column L%16 and rows v + 8(L/16), v = 0..7 (ISA 7.12.2). An edge
    // tile stores only D's rows and columns.
    const bool edge = (tile_m + 1) * BLOCK_M > M || (tile_n + 1) * BLOCK_N > N || M % 8 != 0;
    if constexpr (E == SPLIT) {
        // fp32 partial straight from the accumulators: 32 contiguous bytes per lane per fragment.
        auto store = [&](auto EDGE) {
            static_for<A_FRAGS>([&](auto i) {
                static_for<B_FRAGS>([&](auto j) {
                    const int row = tile_m * BLOCK_M + warp_r * Cf::WARP_M + i * 16 + 8 * h;
                    const int col = tile_n * BLOCK_N + warp_c * Cf::WARP_N + j * 16 + r16;
                    if (!EDGE || (row < M && col < N)) {
                        *reinterpret_cast<v8f *>(static_cast<float *>(g.D) + size_t(col) * M +
                                                 row) = acc[i][j];
                    }
                });
            });
        };
        edge ? store(std::true_type{}) : store(std::false_type{});
        return;
    }

    // ---- epilogue: stage C column-major in LDS (16-bit D when PLAIN, else fp32), copy out ----
    using StT = staged_t<E, ACC, OutT>;
    constexpr int SSTR = BLOCK_M * sizeof(StT) + 16;
    kittens::sync::wait_tdm<0>();
    kittens::sync::wait_async<0>();
    kittens::sync::wait_ds<0>();
    kittens::sync::arrive();
    kittens::sync::wait();
    static_for<A_FRAGS>([&](auto i) {
        static_for<B_FRAGS>([&](auto j) {
            const uint32_t addr = lds0 + (warp_c * Cf::WARP_N + j * 16 + r16) * SSTR +
                                  (warp_r * Cf::WARP_M + i * 16 + 8 * h) * sizeof(StT);
            *lds_ptr<vec8<StT>>(addr) = __builtin_convertvector(acc[i][j], vec8<StT>);
        });
    });
    kittens::sync::wait_ds<0>();
    kittens::sync::arrive();
    kittens::sync::wait();
    auto out = [&](auto EDGE) {
        copy_out<E, ACC, OutT, SideT, L != TN, decltype(EDGE)::value>(
            lds0, static_cast<OutT *>(g.D), static_cast<const SideT *>(g.bias),
            static_cast<SideT *>(g.aux), g.alpha, g.beta, M, N, tile_m, tile_n);
    };
    edge ? out(std::true_type{}) : out(std::false_type{});
}

// SPLIT: workgroup z stores its fp32 partial of K-blocks [nb z / S, nb (z+1) / S) (slice 0 also
// the K % 256 head) to D + z M N. Separate arguments save ~10 instructions over a Gemm.
template <Layout L, int FMT_A, int FMT_B, int CX, int CY, Epi E, bool ACC, typename OutT,
          typename SideT>
__device__ __forceinline__ void mxfp8_gemm_body(const uint8_t *A, const uint8_t *B,
                                                const uint8_t *SA, const uint8_t *SB, void *D,
                                                const void *bias, void *aux, float alpha,
                                                float beta, int M, int N, int K) {
    extern __shared__ __attribute__((aligned(16))) uint8_t smem[];
    const uint32_t lds0 = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&smem[0]));

    kittens::sched::lock_simd();

    int k0 = 0, kz = K;
    if constexpr (E == SPLIT) {
        const int splits = gridDim.z, z = blockIdx.z, nb = K / BLOCK_K, kh = K % BLOCK_K;
        const int b0     = nb * z / splits, b1 = nb * (z + 1) / splits;
        k0 = z == 0 ? 0 : kh + b0 * BLOCK_K;
        kz = (b1 - b0) * BLOCK_K + (z == 0 ? kh : 0);
        D  = static_cast<float *>(D) + size_t(z) * M * N;
    }
    const Gemm g{A, B, SA, SB, D, bias, aux, alpha, beta, M, N, K};
    mxfp8_tile<L, FMT_A, FMT_B, CX, CY, E, ACC, OutT, SideT>(
        g, blockIdx.x, blockIdx.y, blockIdx.x & uint32_t(CX - 1), blockIdx.y & uint32_t(CY - 1), k0,
        kz, lds0);
}

template <Layout L, int FMT_A, int FMT_B, int CX, int CY, Epi E, bool ACC, typename OutT,
          typename SideT>
__global__ __cluster_dims__(CX, CY, 1) __launch_bounds__(NUM_THREADS, 1)
void mxfp8_gemm_kernel(const uint8_t *A, const uint8_t *B, const uint8_t *SA, const uint8_t *SB,
                       void *D, const void *bias, void *aux, float alpha, float beta, int M, int N,
                       int K) {
    mxfp8_gemm_body<L, FMT_A, FMT_B, CX, CY, E, ACC, OutT, SideT>(A, B, SA, SB, D, bias, aux, alpha,
                                                                  beta, M, N, K);
}

// No cluster: every workgroup fills its own tiles.
template <Layout L, int FMT_A, int FMT_B, Epi E, bool ACC, typename OutT, typename SideT>
__global__ __launch_bounds__(NUM_THREADS, 1)
void mxfp8_gemm_kernel_nomc(const uint8_t *A, const uint8_t *B, const uint8_t *SA,
                            const uint8_t *SB, void *D, const void *bias, void *aux, float alpha,
                            float beta, int M, int N, int K) {
    mxfp8_gemm_body<L, FMT_A, FMT_B, 1, 1, E, ACC, OutT, SideT>(A, B, SA, SB, D, bias, aux, alpha,
                                                                beta, M, N, K);
}

// Calls f(std::integral_constant<int, V>) for the first V equal to x, else for the last V.
template <int V, int... Vs, typename F>
void with_constant(int x, F &&f) {
    if constexpr (sizeof...(Vs) > 0) {
        if (x != V) {
            with_constant<Vs...>(x, f);
            return;
        }
    }
    f(std::integral_constant<int, V>{});
}

// Cluster side for t tiles: the largest of 4, 2, 1 that divides t.
inline int cluster_side(int t) { return t % 4 == 0 ? 4 : t % 2 == 0 ? 2 : 1; }

template <Layout L, int FMT_A, int FMT_B, int CX, int CY, Epi E, bool ACC, typename OutT,
          typename SideT>
constexpr auto kernel_for() {
    if constexpr (CX * CY > 1) {
        return mxfp8_gemm_kernel<L, FMT_A, FMT_B, CX, CY, E, ACC, OutT, SideT>;
    } else {
        return mxfp8_gemm_kernel_nomc<L, FMT_A, FMT_B, E, ACC, OutT, SideT>;
    }
}

template <Layout L, int FMT_A, int FMT_B, Epi E, bool ACC, typename OutT, typename SideT>
void launch_kernel(const Launch &l, void *d, int splits) {
    const int tiles_m = (l.M + BLOCK_M - 1) / BLOCK_M;
    const int tiles_n = (l.N + BLOCK_N - 1) / BLOCK_N;
    const int sx      = l.mcast ? cluster_side(tiles_m) : 1;
    const int sy      = l.mcast ? cluster_side(tiles_n) : 1;
    with_constant<4, 2, 1>(sx, [&](auto cx) {
        with_constant<4, 2, 1>(sy, [&](auto cy) {
            const auto kernel = kernel_for<L, FMT_A, FMT_B, cx, cy, E, ACC, OutT, SideT>();
            constexpr int lds = Cfg<L>::LDS_BYTES;
            kittens_check(hipFuncSetAttribute(reinterpret_cast<const void *>(kernel),
                                              hipFuncAttributeMaxDynamicSharedMemorySize, lds),
                          "gfx1250 mxfp8_gemm_kernel LDS");
            hipLaunchKernelGGL(kernel, dim3(tiles_m, tiles_n, splits),
                               dim3(NUM_THREADS), lds, l.stream, static_cast<const uint8_t *>(l.A),
                               static_cast<const uint8_t *>(l.B),
                               static_cast<const uint8_t *>(l.scale_A),
                               static_cast<const uint8_t *>(l.scale_B), d, l.bias, l.aux, l.alpha,
                               l.beta, l.M, l.N, l.K);
            kittens_check(hipGetLastError(), "gfx1250 mxfp8_gemm_kernel");
        });
    });
}

template <int A, int B>
struct Formats {
    static constexpr int FMT_A = A;
    static constexpr int FMT_B = B;
};

// Instantiated formats and epilogue variants of layout L, defined in mxfp8_<layout>.cpp.
template <Layout L>
struct Instances;

}  // namespace mxfp8_gemm

template <Layout L>
bool launch_mxfp8(const Launch &l) {
    using namespace mxfp8_gemm;
    const int fmt_a = fp8_code(static_cast<KittensDType>(l.a_dtype));
    const int fmt_b = fp8_code(static_cast<KittensDType>(l.b_dtype));
    return any_of(typename Instances<L>::formats{}, [&](auto f) {
        using F = decltype(f);
        return F::FMT_A == fmt_a && F::FMT_B == fmt_b &&
               any_of(typename Instances<L>::variants{}, [&](auto v) {
                   using V = decltype(v);
                   if (l.epi != V::E || l.acc != V::ACC ||
                       l.out_dtype != dtype_of<typename V::OutT> ||
                       l.side_dtype != dtype_of<typename V::SideT>) {
                       return false;
                   }
                   if constexpr (V::E == NONE) {
                       if (l.splits > 1) {
                           launch_kernel<L, F::FMT_A, F::FMT_B, SPLIT, false, float, float>(
                               l, l.workspace, l.splits);
                           splitk_reduce(l);
                           return true;
                       }
                   }
                   launch_kernel<L, F::FMT_A, F::FMT_B, V::E, V::ACC, typename V::OutT,
                                 typename V::SideT>(l, l.D, 1);
                   return true;
               });
    });
}

}  // namespace te_kittens::gfx1250
