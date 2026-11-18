/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

#include "hip/hip_runtime.h"
#include <algorithm>
#include <cstdint>
#include <string>
#include <type_traits>
#include "kittens.cuh"
#include "../kittens_common.h"
#include "../kittens_kernel_common.cuh"

namespace te_kittens::cdna5::mxfp8 {

enum struct GemmEpilogue {
    DEFAULT,
    BIAS,
    GELU_AUX,
    GELU_AUX_BIAS,
    DGELU,
    BGRADB,
    SPLIT_K,
};

constexpr int BLOCK_ROW         = 256;
constexpr int BLOCK_COL         = 256;
constexpr int BLOCK_K           = 256;
constexpr int NUM_THREADS       = 256;
constexpr int K_STEP            = 128;
constexpr int K_STEPS           = BLOCK_K / K_STEP;
constexpr int NUM_WARPS         = 8;
constexpr int CLUSTER_DIM       = 4;
constexpr int STAGES            = 2;
constexpr int FLIP_FRAG         = 3;
constexpr int OP_LD             = BLOCK_K + 16;
constexpr int OP_STAGE          = BLOCK_ROW * OP_LD;
constexpr int SC_K              = BLOCK_K / 32;
constexpr int SC_LD             = BLOCK_ROW + 32;
constexpr int SC_STAGE          = SC_K * SC_LD;
constexpr int OFF_B             = STAGES * OP_STAGE;
constexpr int OFF_SA            = 2 * OFF_B;
constexpr int OFF_SB            = OFF_SA + STAGES * SC_STAGE;
constexpr int LDS_BYTES         = OFF_SB + STAGES * SC_STAGE;
constexpr int REDUCE_MAX_BLOCKS = 65536;
constexpr int WS_ALIGN          = 256;

constexpr uint32_t TDM_TYPE = 2u << 30;

static_assert(BLOCK_ROW == BLOCK_K && BLOCK_COL == BLOCK_K);
static_assert(NUM_WARPS * kittens::WARP_THREADS == NUM_THREADS);
static_assert(BLOCK_COL * (BLOCK_ROW * sizeof(float) + 16) <= LDS_BYTES && LDS_BYTES <= 320 * 1024);

typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v16i __attribute__((ext_vector_type(16)));
using kittens::v4u32;
using kittens::v8u32;

#ifdef __gfx1250_strict__
constexpr bool ARCH_MCAST = false;
#else
constexpr bool ARCH_MCAST = true;
#endif

template <typename StT, GemmEpilogue EPILOGUE, bool ACCUMULATE, typename OutT, typename AuxT, bool PLAIN_ALPHA,
          bool EDGE>
__device__ __forceinline__ void gemm_epilogue(uint32_t lds0, OutT *__restrict__ d, const AuxT *__restrict__ bias,
                                              AuxT *__restrict__ aux, float alpha, float beta, int M, int N,
                                              int m_tile, int n_tile) {
    constexpr bool PLAIN    = !std::is_same_v<StT, float>;
    constexpr bool HAS_BIAS = EPILOGUE == GemmEpilogue::BIAS || EPILOGUE == GemmEpilogue::GELU_AUX_BIAS;
    constexpr bool HAS_GELU = EPILOGUE == GemmEpilogue::GELU_AUX || EPILOGUE == GemmEpilogue::GELU_AUX_BIAS;
    constexpr int STRIDE     = BLOCK_ROW * sizeof(StT) + 16;
    constexpr int ROW_CHUNKS = BLOCK_ROW / 8;
    constexpr int UNROLL     = HAS_GELU || EPILOGUE == GemmEpilogue::DGELU || EDGE ? 1 : 4;
    const int m8 = threadIdx.x % ROW_CHUNKS;
    const size_t tile = size_t(n_tile) * BLOCK_COL * M + size_t(m_tile) * BLOCK_ROW + m8 * 8;
    const int rows = EDGE ? std::clamp(M - m_tile * BLOCK_ROW - m8 * 8, 0, 8) : 8;
    const int cols = EDGE ? std::min(N - n_tile * BLOCK_COL, BLOCK_COL) : BLOCK_COL;
    const bool vec = !EDGE || (rows == 8 && M % 8 == 0);
    v8f b8{};
    if constexpr (HAS_BIAS) {
        b8 = load8(bias + m_tile * BLOCK_ROW + m8 * 8, rows, vec);
    }
#pragma unroll UNROLL
    for (int col = threadIdx.x / ROW_CHUNKS; col < cols; col += NUM_THREADS / ROW_CHUNKS) {
        const size_t off = tile + size_t(col) * M;
        const uint32_t src = lds0 + col * STRIDE + m8 * 8 * sizeof(StT);
        if constexpr (PLAIN) {
            if (vec && (!PLAIN_ALPHA || alpha == 1.f)) {
                *reinterpret_cast<v4i *>(d + off) = *lds_ptr<v4i>(src);
            } else if constexpr (PLAIN_ALPHA) {
                store8(d + off, alpha * __builtin_convertvector(*lds_ptr<vec8<StT>>(src), v8f), rows, vec);
            } else {
                store8(d + off, *lds_ptr<vec8<StT>>(src), rows, false);
            }
        } else {
            v8f v = alpha * *lds_ptr<v8f>(src);
            if constexpr (ACCUMULATE) {
                v += beta * load8(d + off, rows, vec);
            }
            if constexpr (HAS_BIAS) {
                v += b8;
            }
            if constexpr (HAS_GELU) {
                store8(aux + off, v, rows, vec);
#pragma unroll
                for (int e = 0; e < 8; e++) {
                    v[e] = kittens::base_ops::gelu::op<float>(v[e]);
                }
            }
            if constexpr (EPILOGUE == GemmEpilogue::DGELU) {
                const v8f g = load8(aux + off, rows, vec);
#pragma unroll
                for (int e = 0; e < 8; e++) {
                    v[e] *= kittens::base_ops::dgelu::op<float>(g[e]);
                }
            }
            store8(d + off, v, rows, vec);
        }
    }
}

template <bool MN>
__device__ __forceinline__ v16i load_frag(uint32_t base, int frag, int ks) {
    v16i v;
    if constexpr (MN) {
        const uint32_t addr = base + ks * K_STEP * OP_LD + frag * 16;
#pragma unroll
        for (int l = 0; l < 8; l++) {
            const v2i x  = __builtin_amdgcn_ds_load_tr8_b64_v2i32(lds_ptr<v2i>(addr + l * 16 * OP_LD));
            v[2 * l]     = x[0];
            v[2 * l + 1] = x[1];
        }
    } else {
        const uint32_t addr = base + frag * 16 * OP_LD + ks * K_STEP;
#pragma unroll
        for (int l = 0; l < 4; l++) {
            const v4i x  = *lds_ptr<v4i>(addr + l * 32);
            v[4 * l]     = x[0];
            v[4 * l + 1] = x[1];
            v[4 * l + 2] = x[2];
            v[4 * l + 3] = x[3];
        }
    }
    return v;
}

template <bool TRANSA, bool TRANSB, bool MC, int FMT_A, int FMT_B, GemmEpilogue EPILOGUE, bool ACCUMULATE,
          typename OutT, typename AuxT>
__global__ __cluster_dims__(MC ? CLUSTER_DIM : 0, MC ? CLUSTER_DIM : 0, MC ?: 0) __launch_bounds__(NUM_THREADS, 1)
void mxfp8_gemm_kernel(const uint8_t *A, const uint8_t *B, const uint8_t *scale_A, const uint8_t *scale_B, void *D,
                       const void *bias, void *aux, float alpha, float beta, int M, int N, int K) {
    constexpr bool NN      = !TRANSA && !TRANSB;
    constexpr bool CLUSTER = MC && ARCH_MCAST;
    constexpr int WARPS_ROW = NN ? 4 : 2;
    constexpr int WARPS_COL = NUM_WARPS / WARPS_ROW;
    constexpr int A_FRAGS   = BLOCK_ROW / WARPS_ROW / 16;
    constexpr int B_FRAGS   = BLOCK_COL / WARPS_COL / 16;

    extern __shared__ __attribute__((aligned(16))) uint8_t smem[];
    const uint32_t lds0 = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(smem));

    kittens::sched::lock_simd();

    int k0 = 0;
    int kz = K;
    if constexpr (EPILOGUE == GemmEpilogue::SPLIT_K) {
        const int splits = gridDim.z;
        const int z      = blockIdx.z;
        const int nb     = K / BLOCK_K;
        const int kh     = K % BLOCK_K;
        const int b0     = nb * z / splits;
        const int b1     = nb * (z + 1) / splits;
        k0 = z == 0 ? 0 : kh + b0 * BLOCK_K;
        kz = (b1 - b0) * BLOCK_K + (z == 0 ? kh : 0);
        D  = static_cast<float *>(D) + size_t(z) * M * N;
    }
    const int m_tile  = blockIdx.x;
    const int n_tile  = blockIdx.y;
    const int tiles_M = kittens::ceil_div(M, BLOCK_ROW);
    const int tiles_N = kittens::ceil_div(N, BLOCK_COL);
    const int k_iters = kittens::ceil_div(kz, BLOCK_K);
    const int k_head  = kz % BLOCK_K ? kz % BLOCK_K : BLOCK_K;
    if (k_iters <= 0) {
        return;
    }
    // A workgroup that ends leaves the cluster barrier's member count (ISA 5.6.6).
    if (m_tile >= tiles_M || n_tile >= tiles_N) {
        return;
    }

    const int lane   = threadIdx.x % kittens::WARP_THREADS;
    const int wid    = __builtin_amdgcn_readfirstlane(threadIdx.x / kittens::WARP_THREADS);
    const int m0     = wid / WARPS_COL * (BLOCK_ROW / WARPS_ROW);
    const int n0     = wid % WARPS_COL * (BLOCK_COL / WARPS_COL);
    const int s      = lane % 16;
    const int h      = lane / 16;
    const int tr_row = (s & 3) + 4 * (s >> 3);
    const int tr_col = 8 * ((s >> 2) & 1);

    const uint32_t a_lane  = lds0 + (TRANSA ? (m0 + s) * OP_LD + 16 * h : (8 * h + tr_row) * OP_LD + m0 + tr_col);
    const uint32_t b_lane  = lds0 + OFF_B +
                             (TRANSB ? (8 * h + tr_row) * OP_LD + n0 + tr_col : (n0 + s) * OP_LD + 16 * h);
    const uint32_t sa_lane = lds0 + OFF_SA + (TRANSA ? (m0 + lane) * SC_K : tr_row * SC_LD + m0 + 16 * h + tr_col);
    const uint32_t sb_lane = lds0 + OFF_SB + (TRANSB ? tr_row * SC_LD + n0 + 16 * h + tr_col : (n0 + lane) * SC_K);

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

    const bool is_b   = wid % 2 != 0;
    const bool scales = wid >= 2;
    const bool mn     = is_b ? TRANSB : !TRANSA;
    const bool perm   = mn && !scales;
    const int tile = is_b ? n_tile : m_tile;
    const uint32_t dsize   = scales && !mn && K % 128 != 0 ? 0 : 2;
    const uint32_t rows    = is_b ? N : M;
    const uint32_t rem     = rows - tile * BLOCK_ROW;
    const uint32_t ld      = scales ? (K / 32 + 3) / 4 * 4 : K;
    const uint32_t k_bytes = scales ? SC_K : BLOCK_K;
    const uint32_t k_shift = (mn || scales ? 5 : 0) + (mn ? 0 : dsize);
    const uint32_t step    = mn ? k_bytes * rows : k_bytes;
    const uint32_t stage   = scales ? SC_STAGE : OP_STAGE;
    const uint32_t lds     = lds0 + (scales ? OFF_SA + is_b * STAGES * SC_STAGE : is_b * OFF_B);
    const uint32_t flags   = dsize << 16 | (scales && !mn ? 0 : 1u << 20 | 5u << 22 | (scales ? 7u : 3u) << 25) |
                           (is_b ? b_mask : a_mask);
    const uint32_t tile0   = (mn ? BLOCK_ROW : k_bytes) >> dsize;
    const uint32_t tile1   = mn ? 8 : BLOCK_ROW;
    const uint32_t tile4   = perm ? BLOCK_K / 32 : 0;
    const uint32_t dim23   = perm ? 2 : 0;
    const uint64_t stride0 = (mn ? rows : ld) >> dsize;
    const uint64_t stride1 = perm ? 4ull * rows : 0;
    const uint64_t stride2 = perm ? 2ull * rows : 0;
    const uint64_t stride3 = perm ? 8ull * rows : 0;
    const size_t k_off = scales ? k0 / 32 : k0;
    const uint8_t *src = (wid == 0 ? A : wid == 1 ? B : wid == 2 ? scale_A : scale_B) +
                         (mn ? size_t(tile) * BLOCK_ROW + k_off * rows : size_t(tile) * BLOCK_ROW * ld + k_off);
    const uint64_t gaddr = reinterpret_cast<uint64_t>(src) - uint64_t(step) * (BLOCK_K - k_head) / BLOCK_K;

    v8f acc[A_FRAGS][B_FRAGS];
#pragma unroll
    for (int i = 0; i < A_FRAGS; i++) {
#pragma unroll
        for (int j = 0; j < B_FRAGS; j++) {
            acc[i][j] = v8f{};
        }
    }

    v8u32 g1;
    v4u32 g3;
    const v4u32 g2 = {dim23, dim23, uint32_t(stride2), uint32_t(stride2 >> 32) | dim23 << 16};
#pragma unroll
    for (int full = 0; full < 2; full++) {
        const uint32_t kq   = uint32_t(full ? BLOCK_K : k_head) >> k_shift;
        const uint32_t dim0 = mn ? rem / 4 : kq;
        const uint32_t dim1 = mn ? (scales ? kq : 8) : rem;
        const uint32_t dim4 = perm ? kq : 0;
        g1 = {flags,
              dim0 << 16,
              dim0 >> 16 | dim1 << 16,
              dim1 >> 16 | tile0 << 16,
              tile1 | dim23 << 16,
              uint32_t(stride0),
              uint32_t(stride0 >> 32) | uint32_t(stride1) << 16,
              uint32_t(stride1 >> 16)};
        g3 = {uint32_t(stride3), uint32_t(stride3 >> 32) | dim4 << 16, dim4 >> 16 | tile4 << 16, 0};
        // wid is wave-uniform: a TDM instruction cannot be EXEC-masked.
        if (!full && wid < 4) {
            const uint64_t ga = reinterpret_cast<uint64_t>(src);
            kittens::emit_tensor_load_to_lds(v4u32{1, lds, uint32_t(ga), uint32_t(ga >> 32) | TDM_TYPE}, g1, g2, g3);
        }
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

    uint32_t a_st[STAGES];
    uint32_t b_st[STAGES];
    uint32_t sa_st[STAGES];
    uint32_t sb_st[STAGES];
#pragma unroll
    for (int st = 0; st < STAGES; st++) {
        a_st[st]  = a_lane + st * OP_STAGE;
        b_st[st]  = b_lane + st * OP_STAGE;
        sa_st[st] = sa_lane + st * SC_STAGE;
        sb_st[st] = sb_lane + st * SC_STAGE;
        // Own VGPRs for the LDS bases: no address VALU may write a WMMA source VGPR (ISA 7.12.1).
        asm volatile("" : "+v"(a_st[st]), "+v"(b_st[st]), "+v"(sa_st[st]), "+v"(sb_st[st]));
    }

    v16i fa[A_FRAGS];
    v16i fb[B_FRAGS];
    v2i sca[2][A_FRAGS / 2];
    v2i scb[2][B_FRAGS / 2];
#pragma unroll
    for (int p = 0; p < A_FRAGS / 2; p++) {
        sca[0][p] = TRANSA ? *lds_ptr<v2i>(sa_st[0] + p * 32 * SC_K)
                           : __builtin_amdgcn_ds_load_tr8_b64_v2i32(lds_ptr<v2i>(sa_st[0] + p * 32));
    }
#pragma unroll
    for (int p = 0; p < B_FRAGS / 2; p++) {
        scb[0][p] = TRANSB ? __builtin_amdgcn_ds_load_tr8_b64_v2i32(lds_ptr<v2i>(sb_st[0] + p * 32))
                           : *lds_ptr<v2i>(sb_st[0] + p * 32 * SC_K);
    }
#pragma unroll
    for (int i = 0; i < A_FRAGS; i++) {
        fa[i] = load_frag<!TRANSA>(a_st[0], i, 0);
    }
#pragma unroll
    for (int j = 0; j < B_FRAGS; j++) {
        fb[j] = load_frag<TRANSB>(b_st[0], j, 0);
    }
    kittens::sched::compiler_fence();

    for (int kb = 0; kb < k_iters; kb += 2) {
#pragma unroll
        for (int par = 0; par < 2; par++) {
            if (par == 1 && kb + 1 >= k_iters) {
                continue;
            }
            if (kb + par + 1 < k_iters && wid < 4) {
                const uint64_t ga = gaddr + uint64_t(kb + par + 1) * step;
                kittens::emit_tensor_load_to_lds(
                    v4u32{1, lds + (par ^ 1) * stage, uint32_t(ga), uint32_t(ga >> 32) | TDM_TYPE}, g1, g2, g3);
            }
            kittens::sched::compiler_fence();
#pragma unroll
            for (int t = 0; t < K_STEPS; t++) {
                const bool last    = t == K_STEPS - 1;
                const bool b_outer = NN && t % 2 == 1;
                const int ks = last ? 0 : t + 1;
                const uint32_t a_src = a_st[last ? par ^ 1 : par];
                const uint32_t b_src = b_st[last ? par ^ 1 : par];
#pragma unroll
                for (int o = 0; o < (b_outer ? B_FRAGS : A_FRAGS); o += 2) {
                    if (!b_outer) {
                        const int sa = sca[par][o / 2][t];
                        acc[o][0] = __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(FMT_A, fa[o], FMT_B, fb[0], 0, acc[o][0], 0, 0, sa, 0, 0, scb[par][0][t], false, false);
                        acc[o][1] = __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(FMT_A, fa[o], FMT_B, fb[1], 0, acc[o][1], 0, 0, sa, 1, 0, scb[par][0][t], true, false);
#pragma unroll
                        for (int j = 2; j < B_FRAGS; j += 2) {
                            const int sb = scb[par][j / 2][t];
                            acc[o][j] = __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(FMT_A, fa[o], FMT_B, fb[j], 0, acc[o][j], 0, 0, sa, 0, 0, sb, true, false);
                            acc[o][j + 1] = __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(FMT_A, fa[o], FMT_B, fb[j + 1], 0, acc[o][j + 1], 0, 0, sa, 1, 0, sb, true, false);
                        }
                    } else {
                        const int sb = scb[par][o / 2][t];
                        acc[0][o] = __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(FMT_A, fa[0], FMT_B, fb[o], 0, acc[0][o], 0, 0, sca[par][0][t], 0, 0, sb, false, false);
                        acc[1][o] = __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(FMT_A, fa[1], FMT_B, fb[o], 0, acc[1][o], 1, 0, sca[par][0][t], 0, 0, sb, false, true);
#pragma unroll
                        for (int i = 2; i < A_FRAGS; i += 2) {
                            const int sa = sca[par][i / 2][t];
                            acc[i][o] = __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(FMT_A, fa[i], FMT_B, fb[o], 0, acc[i][o], 0, 0, sa, 0, 0, sb, false, true);
                            acc[i + 1][o] = __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(FMT_A, fa[i + 1], FMT_B, fb[o], 0, acc[i + 1][o], 1, 0, sa, 0, 0, sb, false, true);
                        }
                    }

                    if (last && o + 1 == FLIP_FRAG) {
                        kittens::sched::compiler_fence();
                        kittens::sync::wait_ds<0>();
                        kittens::sync::wait_tdm<0>();
                        kittens::sched::compiler_fence();
                        kittens::sync::sync();
                        if constexpr (CLUSTER) {
                            if (wid == 0) {
                                kittens::cluster::arrive();
                            }
                        }
                        kittens::sched::compiler_fence();
#pragma unroll
                        for (int p = 0; p < A_FRAGS / 2; p++) {
                            sca[par ^ 1][p] =
                                TRANSA ? *lds_ptr<v2i>(sa_st[par ^ 1] + p * 32 * SC_K)
                                       : __builtin_amdgcn_ds_load_tr8_b64_v2i32(lds_ptr<v2i>(sa_st[par ^ 1] + p * 32));
                        }
#pragma unroll
                        for (int p = 0; p < B_FRAGS / 2; p++) {
                            scb[par ^ 1][p] =
                                TRANSB ? __builtin_amdgcn_ds_load_tr8_b64_v2i32(lds_ptr<v2i>(sb_st[par ^ 1] + p * 32))
                                       : *lds_ptr<v2i>(sb_st[par ^ 1] + p * 32 * SC_K);
                        }
                        if constexpr (NN) {
                            fb[0] = load_frag<TRANSB>(b_src, 0, ks);
                            fb[1] = load_frag<TRANSB>(b_src, 1, ks);
                        } else {
                            fa[0] = load_frag<!TRANSA>(a_src, 0, ks);
                            fa[1] = load_frag<!TRANSA>(a_src, 1, ks);
                        }
                        kittens::sched::compiler_fence();
                    }

                    const bool reload_outer = !last || o + 1 >= FLIP_FRAG;
                    if (!b_outer) {
                        const int sa = sca[par][o / 2][t];
                        acc[o + 1][0] = __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(FMT_A, fa[o + 1], FMT_B, fb[0], 0, acc[o + 1][0], 1, 0, sa, 0, 0, scb[par][0][t], false, false);
                        acc[o + 1][1] = __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(FMT_A, fa[o + 1], FMT_B, fb[1], 0, acc[o + 1][1], 1, 0, sa, 1, 0, scb[par][0][t], true, false);
                        kittens::sched::compiler_fence();
                        if (o + 2 == A_FRAGS) {
                            fb[0] = load_frag<TRANSB>(b_src, 0, ks);
                            fb[1] = load_frag<TRANSB>(b_src, 1, ks);
                        }
                        kittens::sched::compiler_fence();
#pragma unroll
                        for (int j = 2; j < B_FRAGS; j += 2) {
                            const int sb = scb[par][j / 2][t];
                            acc[o + 1][j] = __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(FMT_A, fa[o + 1], FMT_B, fb[j], 0, acc[o + 1][j], 1, 0, sa, 0, 0, sb, true, false);
                            acc[o + 1][j + 1] = __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(FMT_A, fa[o + 1], FMT_B, fb[j + 1], 0, acc[o + 1][j + 1], 1, 0, sa, 1, 0, sb, true, false);
                            kittens::sched::compiler_fence();
                            if (j + 2 == B_FRAGS && reload_outer) {
                                fa[o]     = load_frag<!TRANSA>(a_src, o, ks);
                                fa[o + 1] = load_frag<!TRANSA>(a_src, o + 1, ks);
                            }
                            if (o + 2 == A_FRAGS) {
                                fb[j] = load_frag<TRANSB>(b_src, j, ks);
                                if constexpr (TRANSA) {
                                    kittens::sched::compiler_fence();
                                }
                                fb[j + 1] = load_frag<TRANSB>(b_src, j + 1, ks);
                            }
                            kittens::sched::compiler_fence();
                        }
                    } else {
                        const int sb = scb[par][o / 2][t];
                        acc[0][o + 1] = __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(FMT_A, fa[0], FMT_B, fb[o + 1], 0, acc[0][o + 1], 0, 0, sca[par][0][t], 1, 0, sb, false, false);
                        acc[1][o + 1] = __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(FMT_A, fa[1], FMT_B, fb[o + 1], 0, acc[1][o + 1], 1, 0, sca[par][0][t], 1, 0, sb, false, true);
                        kittens::sched::compiler_fence();
                        if (o + 2 == B_FRAGS) {
                            fa[0] = load_frag<!TRANSA>(a_src, 0, ks);
                            fa[1] = load_frag<!TRANSA>(a_src, 1, ks);
                        }
                        kittens::sched::compiler_fence();
#pragma unroll
                        for (int i = 2; i < A_FRAGS; i += 2) {
                            const int sa = sca[par][i / 2][t];
                            acc[i][o + 1] = __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(FMT_A, fa[i], FMT_B, fb[o + 1], 0, acc[i][o + 1], 0, 0, sa, 1, 0, sb, false, true);
                            acc[i + 1][o + 1] = __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(FMT_A, fa[i + 1], FMT_B, fb[o + 1], 0, acc[i + 1][o + 1], 1, 0, sa, 1, 0, sb, false, true);
                            kittens::sched::compiler_fence();
                            if (o + 2 == B_FRAGS) {
                                fa[i]     = load_frag<!TRANSA>(a_src, i, ks);
                                fa[i + 1] = load_frag<!TRANSA>(a_src, i + 1, ks);
                            }
                            if (i + 2 == A_FRAGS && reload_outer) {
                                fb[o]     = load_frag<TRANSB>(b_src, o, ks);
                                fb[o + 1] = load_frag<TRANSB>(b_src, o + 1, ks);
                            }
                            kittens::sched::compiler_fence();
                        }
                    }
                }
            }
            if constexpr (CLUSTER) {
                kittens::cluster::wait();
            }
            kittens::sched::compiler_fence();
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
                    *reinterpret_cast<v8f *>(d + size_t(col) * M + row) = acc[i][j];
                }
            }
        }
        return;
    }

    using StT =
        std::conditional_t<EPILOGUE == GemmEpilogue::DEFAULT && !ACCUMULATE && sizeof(OutT) == 2, OutT, float>;
    constexpr int STRIDE = BLOCK_ROW * sizeof(StT) + 16;
    kittens::sync::wait_tdm<0>();
    kittens::sync::wait_ds<0>();
    kittens::sync::sync();
#pragma unroll
    for (int i = 0; i < A_FRAGS; i++) {
#pragma unroll
        for (int j = 0; j < B_FRAGS; j++) {
            const uint32_t addr = lds0 + (col0 + j * 16) * STRIDE + (row0 + i * 16) * sizeof(StT);
            *lds_ptr<vec8<StT>>(addr) = __builtin_convertvector(acc[i][j], vec8<StT>);
        }
    }
    kittens::sync::wait_ds<0>();
    kittens::sync::sync();
    OutT *d            = static_cast<OutT *>(D);
    const AuxT *bias_t = static_cast<const AuxT *>(bias);
    AuxT *aux_t        = static_cast<AuxT *>(aux);
    if (edge) {
        gemm_epilogue<StT, EPILOGUE, ACCUMULATE, OutT, AuxT, !TRANSA, true>(lds0, d, bias_t, aux_t, alpha, beta, M, N,
                                                                            m_tile, n_tile);
    } else {
        gemm_epilogue<StT, EPILOGUE, ACCUMULATE, OutT, AuxT, !TRANSA, false>(lds0, d, bias_t, aux_t, alpha, beta, M,
                                                                             N, m_tile, n_tile);
    }
}

template <bool TRANSA, bool TRANSB, int FMT_A, int FMT_B, GemmEpilogue EPILOGUE, bool ACCUMULATE, typename OutT,
          typename AuxT>
static bool launch_gemm(const MXFP8GemmArgs &args, void *d, int splits, bool mcast) {
    const int cluster = mcast ? CLUSTER_DIM : 1;
    const dim3 grid(kittens::ceil_div(kittens::ceil_div(args.M, BLOCK_ROW), cluster) * cluster,
                    kittens::ceil_div(kittens::ceil_div(args.N, BLOCK_COL), cluster) * cluster, splits);
    KITTENS_BOOL_SWITCH(mcast, MC,
        const auto kernel = mxfp8_gemm_kernel<TRANSA, TRANSB, MC, FMT_A, FMT_B, EPILOGUE, ACCUMULATE, OutT, AuxT>;
        hipFuncSetAttribute(reinterpret_cast<const void *>(kernel), hipFuncAttributeMaxDynamicSharedMemorySize,
                            LDS_BYTES);
        hipLaunchKernelGGL(kernel, grid, dim3(NUM_THREADS), LDS_BYTES, args.stream,
                           static_cast<const uint8_t *>(args.A), static_cast<const uint8_t *>(args.B),
                           static_cast<const uint8_t *>(args.scale_A), static_cast<const uint8_t *>(args.scale_B), d,
                           args.bias, args.aux_gelu, args.alpha, args.beta, args.M, args.N, args.K);
    ) // NOLINT(*)
    return true;
}

template <typename OutT>
static void launch_reduce(const MXFP8GemmArgs &args, const void *workspace, int splits) {
    const size_t mn     = size_t(args.M) * args.N;
    const size_t blocks = std::min<size_t>((mn / 8 + REDUCE_THREADS - 1) / REDUCE_THREADS, REDUCE_MAX_BLOCKS);
    const auto kernel = args.beta != 0.f ? splitk_reduce_kernel<OutT, true> : splitk_reduce_kernel<OutT, false>;
    hipLaunchKernelGGL(kernel, dim3(blocks), dim3(REDUCE_THREADS), 0, args.stream,
                       static_cast<const float *>(workspace), static_cast<OutT *>(args.C), args.alpha, args.beta,
                       mn, splits);
}

template <bool TRANSA, bool TRANSB, int FMT_A, int FMT_B, bool ACCUMULATE>
static bool dispatch_gemm(const MXFP8GemmArgs &args, void *workspace, int splits, GemmEpilogue epilogue,
                          bool mcast) {
    if (epilogue == GemmEpilogue::DEFAULT && splits > 1) {
        launch_gemm<TRANSA, TRANSB, FMT_A, FMT_B, GemmEpilogue::SPLIT_K, false, float, float>(args, workspace, splits, mcast);
        if (args.out_dtype == KITTENS_FLOAT32) {
            launch_reduce<float>(args, workspace, splits);
        } else if (args.out_dtype == KITTENS_FLOAT16) {
            launch_reduce<_Float16>(args, workspace, splits);
        } else {
            launch_reduce<__bf16>(args, workspace, splits);
        }
        return true;
    }
    switch (epilogue) {
    case GemmEpilogue::DEFAULT:
        switch (args.out_dtype) {
        case KITTENS_FLOAT32:
            return launch_gemm<TRANSA, TRANSB, FMT_A, FMT_B, GemmEpilogue::DEFAULT, ACCUMULATE, float, float>(args, args.C, 1, mcast);
        case KITTENS_FLOAT16:
            return launch_gemm<TRANSA, TRANSB, FMT_A, FMT_B, GemmEpilogue::DEFAULT, ACCUMULATE, _Float16, _Float16>(args, args.C, 1, mcast);
        case KITTENS_BFLOAT16:
            return launch_gemm<TRANSA, TRANSB, FMT_A, FMT_B, GemmEpilogue::DEFAULT, ACCUMULATE, __bf16, __bf16>(args, args.C, 1, mcast);
        }
        return false;
    case GemmEpilogue::BIAS:
        if constexpr (TRANSA || !ACCUMULATE) {
            if (args.out_dtype == KITTENS_FLOAT32 && args.bias_dtype == KITTENS_FLOAT32) {
                return launch_gemm<TRANSA, TRANSB, FMT_A, FMT_B, GemmEpilogue::BIAS, ACCUMULATE, float, float>(args, args.C, 1, mcast);
            }
            if (args.out_dtype == KITTENS_FLOAT32 && args.bias_dtype == KITTENS_BFLOAT16) {
                return launch_gemm<TRANSA, TRANSB, FMT_A, FMT_B, GemmEpilogue::BIAS, ACCUMULATE, float, __bf16>(args, args.C, 1, mcast);
            }
            if (args.out_dtype == KITTENS_FLOAT16 && args.bias_dtype == KITTENS_FLOAT16) {
                return launch_gemm<TRANSA, TRANSB, FMT_A, FMT_B, GemmEpilogue::BIAS, ACCUMULATE, _Float16, _Float16>(args, args.C, 1, mcast);
            }
            if (args.out_dtype == KITTENS_BFLOAT16 && args.bias_dtype == KITTENS_BFLOAT16) {
                return launch_gemm<TRANSA, TRANSB, FMT_A, FMT_B, GemmEpilogue::BIAS, ACCUMULATE, __bf16, __bf16>(args, args.C, 1, mcast);
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
                return launch_gemm<TRANSA, TRANSB, FMT_A, FMT_B, GemmEpilogue::GELU_AUX, ACCUMULATE, float, __bf16>(args, args.C, 1, mcast);
            case KITTENS_FLOAT16:
                return launch_gemm<TRANSA, TRANSB, FMT_A, FMT_B, GemmEpilogue::GELU_AUX, ACCUMULATE, _Float16, __bf16>(args, args.C, 1, mcast);
            case KITTENS_BFLOAT16:
                return launch_gemm<TRANSA, TRANSB, FMT_A, FMT_B, GemmEpilogue::GELU_AUX, ACCUMULATE, __bf16, __bf16>(args, args.C, 1, mcast);
            }
        }
        return false;
    case GemmEpilogue::GELU_AUX_BIAS:
        if constexpr (TRANSA && !ACCUMULATE) {
            if (args.out_dtype == KITTENS_FLOAT32 && args.bias_dtype == KITTENS_BFLOAT16) {
                return launch_gemm<TRANSA, TRANSB, FMT_A, FMT_B, GemmEpilogue::GELU_AUX_BIAS, ACCUMULATE, float, __bf16>(args, args.C, 1, mcast);
            }
            if (args.out_dtype == KITTENS_FLOAT16 && args.bias_dtype == KITTENS_FLOAT16) {
                return launch_gemm<TRANSA, TRANSB, FMT_A, FMT_B, GemmEpilogue::GELU_AUX_BIAS, ACCUMULATE, _Float16, _Float16>(args, args.C, 1, mcast);
            }
            if (args.out_dtype == KITTENS_BFLOAT16 && args.bias_dtype == KITTENS_BFLOAT16) {
                return launch_gemm<TRANSA, TRANSB, FMT_A, FMT_B, GemmEpilogue::GELU_AUX_BIAS, ACCUMULATE, __bf16, __bf16>(args, args.C, 1, mcast);
            }
        }
        return false;
    default:
        return false;
    }
}

template <bool TRANSA, bool TRANSB>
static bool mxfp8_gemm_impl(const MXFP8GemmArgs &args, void *workspace, int splits, GemmEpilogue epilogue,
                            bool mcast) {
    const bool accumulate = args.beta != 0.f;
    const bool a_fp8      = fp8_code(static_cast<KittensDType>(args.a_dtype)) != 0;
    const bool b_fp8      = fp8_code(static_cast<KittensDType>(args.b_dtype)) != 0;
    bool ok = false;
    KITTENS_BOOL_SWITCH(a_fp8, FMT_A,
        KITTENS_BOOL_SWITCH(b_fp8, FMT_B,
            KITTENS_BOOL_SWITCH(accumulate, ACCUMULATE,
                ok = dispatch_gemm<TRANSA, TRANSB, FMT_A, FMT_B, ACCUMULATE>(args, workspace, splits, epilogue, mcast);
    ))) // NOLINT(*)
    return ok;
}

class MXFP8GemmCdna5 final : public MXFP8GemmBackend {
 public:
    bool gemm(const MXFP8GemmArgs &args) override {
        if (!args.mxfp8_compact || (args.transa && args.transb) || args.M <= 0 || args.N <= 0 || args.K <= 0 ||
            args.M % 4 != 0 || args.N % 4 != 0 || args.K % 32 != 0) {
            return false;
        }
        const uintptr_t ptrs = reinterpret_cast<uintptr_t>(args.A) | reinterpret_cast<uintptr_t>(args.B) |
                               reinterpret_cast<uintptr_t>(args.C) | reinterpret_cast<uintptr_t>(args.scale_A) |
                               reinterpret_cast<uintptr_t>(args.scale_B) | reinterpret_cast<uintptr_t>(args.bias) |
                               reinterpret_cast<uintptr_t>(args.aux_gelu);
        if (ptrs % 16 != 0) {
            return false;
        }

        const bool bias = args.bias != nullptr;
        const bool aux  = args.aux_gelu != nullptr;
        if ((bias && aux && args.bias_dtype != args.aux_dtype) || (args.grad && bias && aux)) {
            return false;
        }
        GemmEpilogue epilogue = GemmEpilogue::DEFAULT;
        if (args.grad) {
            epilogue = bias ? GemmEpilogue::BGRADB : aux ? GemmEpilogue::DGELU : GemmEpilogue::DEFAULT;
        } else if (bias || aux) {
            epilogue = !aux ? GemmEpilogue::BIAS : bias ? GemmEpilogue::GELU_AUX_BIAS : GemmEpilogue::GELU_AUX;
        }
        if (args.transa && epilogue == GemmEpilogue::DEFAULT && args.beta == 0.f &&
            args.out_dtype != KITTENS_FLOAT32 && args.alpha != 1.f) {
            return false;
        }

        const uintptr_t ws_raw = reinterpret_cast<uintptr_t>(args.workspace);
        const uintptr_t ws     = (ws_raw + WS_ALIGN - 1) & ~uintptr_t(WS_ALIGN - 1);
        const size_t pad     = ws - ws_raw;
        const size_t ws_size = args.workspace_size > pad ? args.workspace_size - pad : 0;
        const int64_t tiles = int64_t(kittens::ceil_div(args.M, BLOCK_ROW)) * kittens::ceil_div(args.N, BLOCK_COL);
        const size_t split_bytes = size_t(args.M) * args.N * sizeof(float);
        int splits = 1;
        if (epilogue == GemmEpilogue::DEFAULT && args.M % 8 == 0) {
            if (args.splitk > 1 && args.splitk <= args.K / BLOCK_K && split_bytes * args.splitk <= ws_size) {
                splits = args.splitk;
            }
            double best_t = args.K;
            for (int s = 2; args.splitk == 0 && s <= 16 && tiles * s <= args.cu_count; s *= 2) {
                const double t = double(args.K) / s + 512 + 2048 * double(tiles * s) / args.cu_count;
                if (s <= args.K / BLOCK_K && split_bytes * s <= ws_size && t < best_t) {
                    splits = s;
                    best_t = t;
                }
            }
        }

        void *workspace = reinterpret_cast<void *>(ws);
#ifdef NVTE_HK_GFX1250_MCAST
        const bool mcast = NVTE_HK_GFX1250_MCAST != 0;
#else
        const bool mcast = transformer_engine::cuda::sm_arch_name().find("-strict") == std::string::npos;
#endif
        if (args.transa && !args.transb) {
            return mxfp8_gemm_impl<true, false>(args, workspace, splits, epilogue, mcast);
        }
        if (!args.transa && args.transb) {
            return mxfp8_gemm_impl<false, true>(args, workspace, splits, epilogue, mcast);
        }
        return mxfp8_gemm_impl<false, false>(args, workspace, splits, epilogue, mcast);
    }

    bool grouped_gemm(const MXFP8GroupedGemmArgs &) override { return false; }
    bool grouped_wgrad(const MXFP8WgradArgs &) override { return false; }
};

}  // namespace te_kittens::cdna5::mxfp8

MXFP8GemmBackend *MXFP8GemmBackend::get_cdna5() {
    static te_kittens::cdna5::mxfp8::MXFP8GemmCdna5 impl;
    return &impl;
}
