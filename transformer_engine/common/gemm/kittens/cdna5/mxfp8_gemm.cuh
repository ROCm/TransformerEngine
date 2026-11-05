/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

#pragma once

#include "hip/hip_runtime.h"
#include <cstdint>
#include <type_traits>
#include "common.cuh"

namespace te_kittens::cdna5 {
namespace mxfp8 {

constexpr int BLOCK_K   = MXFP8_BLOCK_K;
constexpr int K_STEP    = 128;
constexpr int K_STEPS   = BLOCK_K / K_STEP;
constexpr int NUM_WARPS = 8;
constexpr int STAGES    = 2;
constexpr int FLIP_FRAG = 3;
constexpr int OP_LD     = BLOCK_K + 16;
constexpr int OP_STAGE  = BLOCK_M * OP_LD;
constexpr int SC_K      = BLOCK_K / 32;
constexpr int SC_LD     = BLOCK_M + 32;
constexpr int SC_STAGE  = SC_K * SC_LD;
constexpr int OFF_B     = STAGES * OP_STAGE;
constexpr int OFF_SA    = 2 * OFF_B;
constexpr int OFF_SB    = OFF_SA + STAGES * SC_STAGE;
constexpr int LDS_BYTES = OFF_SB + STAGES * SC_STAGE;

constexpr uint32_t TDM_TYPE = 2u << 30;

static_assert(BLOCK_M == BLOCK_K && BLOCK_N == BLOCK_K);
static_assert(NUM_WARPS * kittens::WARP_THREADS == NUM_THREADS);
static_assert(BLOCK_N * staged_stride(sizeof(float)) <= LDS_BYTES && LDS_BYTES <= LDS_MAX_BYTES);

typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v16i __attribute__((ext_vector_type(16)));
using kittens::v4u32;
using kittens::v8u32;

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

template <bool TRANSA, bool TRANSB, int FMT_A, int FMT_B, int CX, int CY, Epi E, bool ACC, typename OutT,
          typename SideT>
__device__ __forceinline__ void mxfp8_gemm(const uint8_t *A, const uint8_t *B, const uint8_t *SA, const uint8_t *SB,
                                           void *D, const void *bias, void *aux, float alpha, float beta, int M, int N,
                                           int K) {
    constexpr bool NN     = !TRANSA && !TRANSB;
    constexpr bool MC     = CX * CY > 1 && ARCH_MCAST;
    constexpr int WARPS_M = NN ? 4 : 2;
    constexpr int WARPS_N = NUM_WARPS / WARPS_M;
    constexpr int A_FRAGS = BLOCK_M / WARPS_M / 16;
    constexpr int B_FRAGS = BLOCK_N / WARPS_N / 16;

    extern __shared__ __attribute__((aligned(16))) uint8_t smem[];
    const uint32_t lds0 = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(smem));

    kittens::sched::lock_simd();

    int k0 = 0;
    int kz = K;
    if constexpr (E == Epi::SPLIT) {
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
    const int tile_m   = blockIdx.x;
    const int tile_n   = blockIdx.y;
    const int k_blocks = ceil_div(kz, BLOCK_K);
    const int k_head   = kz % BLOCK_K ? kz % BLOCK_K : BLOCK_K;
    if (k_blocks <= 0) {
        return;
    }

    const int lane   = threadIdx.x % kittens::WARP_THREADS;
    const int wid    = __builtin_amdgcn_readfirstlane(threadIdx.x / kittens::WARP_THREADS);
    const int m0     = wid / WARPS_N * (BLOCK_M / WARPS_M);
    const int n0     = wid % WARPS_N * (BLOCK_N / WARPS_N);
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
    if constexpr (MC && CY > 1) {
#pragma unroll
        for (int y = 0; y < CY; y++) {
            a_mask |= 1u << (y * CX + blockIdx.x % CX);
        }
    }
    if constexpr (MC && CX > 1) {
        b_mask = ((1u << CX) - 1) << (CX * (blockIdx.y % CY));
    }

    const bool is_b   = wid % 2 != 0;
    const bool scales = wid >= 2;
    const bool mn     = is_b ? TRANSB : !TRANSA;
    const bool perm   = mn && !scales;
    const int tile = is_b ? tile_n : tile_m;
    const uint32_t dsize   = scales && !mn && K % 128 != 0 ? 0 : 2;
    const uint32_t rows    = is_b ? N : M;
    const uint32_t rem     = rows - tile * BLOCK_M;
    const uint32_t ld      = scales ? (K / 32 + 3) / 4 * 4 : K;
    const uint32_t k_bytes = scales ? SC_K : BLOCK_K;
    const uint32_t k_shift = (mn || scales ? 5 : 0) + (mn ? 0 : dsize);
    const uint32_t step    = mn ? k_bytes * rows : k_bytes;
    const uint32_t stage   = scales ? SC_STAGE : OP_STAGE;
    const uint32_t lds     = lds0 + (scales ? OFF_SA + is_b * STAGES * SC_STAGE : is_b * OFF_B);
    const uint32_t flags   = dsize << 16 | (scales && !mn ? 0 : 1u << 20 | 5u << 22 | (scales ? 7u : 3u) << 25) |
                           (is_b ? b_mask : a_mask);
    const uint32_t tile0   = (mn ? BLOCK_M : k_bytes) >> dsize;
    const uint32_t tile1   = mn ? 8 : BLOCK_M;
    const uint32_t tile4   = perm ? BLOCK_K / 32 : 0;
    const uint32_t dim23   = perm ? 2 : 0;
    const uint64_t stride0 = (mn ? rows : ld) >> dsize;
    const uint64_t stride1 = perm ? 4ull * rows : 0;
    const uint64_t stride2 = perm ? 2ull * rows : 0;
    const uint64_t stride3 = perm ? 8ull * rows : 0;
    const size_t k_off = scales ? k0 / 32 : k0;
    const uint8_t *src = (wid == 0 ? A : wid == 1 ? B : wid == 2 ? SA : SB) +
                         (mn ? size_t(tile) * BLOCK_M + k_off * rows : size_t(tile) * BLOCK_M * ld + k_off);
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
    if constexpr (MC) {
        // ISA 5.6.6: the workgroup synchronizes first, then one wave per workgroup signals the cluster.
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

    for (int kb = 0; kb < k_blocks; kb += 2) {
#pragma unroll
        for (int par = 0; par < 2; par++) {
            if (par == 1 && kb + 1 >= k_blocks) {
                continue;
            }
            if (kb + par + 1 < k_blocks && wid < 4) {
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
                        acc[o][0] = __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(
                            FMT_A, fa[o], FMT_B, fb[0], 0, acc[o][0], 0, 0, sa, 0, 0, scb[par][0][t], false, false);
                        acc[o][1] = __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(
                            FMT_A, fa[o], FMT_B, fb[1], 0, acc[o][1], 0, 0, sa, 1, 0, scb[par][0][t], true, false);
#pragma unroll
                        for (int j = 2; j < B_FRAGS; j += 2) {
                            const int sb = scb[par][j / 2][t];
                            acc[o][j] = __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(
                                FMT_A, fa[o], FMT_B, fb[j], 0, acc[o][j], 0, 0, sa, 0, 0, sb, true, false);
                            acc[o][j + 1] = __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(
                                FMT_A, fa[o], FMT_B, fb[j + 1], 0, acc[o][j + 1], 0, 0, sa, 1, 0, sb, true, false);
                        }
                    } else {
                        const int sb = scb[par][o / 2][t];
                        acc[0][o] = __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(
                            FMT_A, fa[0], FMT_B, fb[o], 0, acc[0][o], 0, 0, sca[par][0][t], 0, 0, sb, false, false);
                        acc[1][o] = __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(
                            FMT_A, fa[1], FMT_B, fb[o], 0, acc[1][o], 1, 0, sca[par][0][t], 0, 0, sb, false, true);
#pragma unroll
                        for (int i = 2; i < A_FRAGS; i += 2) {
                            const int sa = sca[par][i / 2][t];
                            acc[i][o] = __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(
                                FMT_A, fa[i], FMT_B, fb[o], 0, acc[i][o], 0, 0, sa, 0, 0, sb, false, true);
                            acc[i + 1][o] = __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(
                                FMT_A, fa[i + 1], FMT_B, fb[o], 0, acc[i + 1][o], 1, 0, sa, 0, 0, sb, false, true);
                        }
                    }

                    if (last && o + 1 == FLIP_FRAG) {
                        kittens::sched::compiler_fence();
                        kittens::sync::wait_ds<0>();
                        kittens::sync::wait_tdm<0>();
                        kittens::sched::compiler_fence();
                        kittens::sync::sync();
                        if constexpr (MC) {
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
                        acc[o + 1][0] = __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(
                            FMT_A, fa[o + 1], FMT_B, fb[0], 0, acc[o + 1][0], 1, 0, sa, 0, 0, scb[par][0][t], false,
                            false);
                        acc[o + 1][1] = __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(
                            FMT_A, fa[o + 1], FMT_B, fb[1], 0, acc[o + 1][1], 1, 0, sa, 1, 0, scb[par][0][t], true,
                            false);
                        kittens::sched::compiler_fence();
                        if (o + 2 == A_FRAGS) {
                            fb[0] = load_frag<TRANSB>(b_src, 0, ks);
                            fb[1] = load_frag<TRANSB>(b_src, 1, ks);
                        }
                        kittens::sched::compiler_fence();
#pragma unroll
                        for (int j = 2; j < B_FRAGS; j += 2) {
                            const int sb = scb[par][j / 2][t];
                            acc[o + 1][j] = __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(
                                FMT_A, fa[o + 1], FMT_B, fb[j], 0, acc[o + 1][j], 1, 0, sa, 0, 0, sb, true, false);
                            acc[o + 1][j + 1] = __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(
                                FMT_A, fa[o + 1], FMT_B, fb[j + 1], 0, acc[o + 1][j + 1], 1, 0, sa, 1, 0, sb, true,
                                false);
                            kittens::sched::compiler_fence();
                            if (j + 2 == B_FRAGS && reload_outer) {
                                fa[o]     = load_frag<!TRANSA>(a_src, o, ks);
                                fa[o + 1] = load_frag<!TRANSA>(a_src, o + 1, ks);
                            }
                            if (o + 2 == A_FRAGS) {
                                fb[j]     = load_frag<TRANSB>(b_src, j, ks);
                                fb[j + 1] = load_frag<TRANSB>(b_src, j + 1, ks);
                            }
                            kittens::sched::compiler_fence();
                        }
                    } else {
                        const int sb = scb[par][o / 2][t];
                        acc[0][o + 1] = __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(
                            FMT_A, fa[0], FMT_B, fb[o + 1], 0, acc[0][o + 1], 0, 0, sca[par][0][t], 1, 0, sb, false,
                            false);
                        acc[1][o + 1] = __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(
                            FMT_A, fa[1], FMT_B, fb[o + 1], 0, acc[1][o + 1], 1, 0, sca[par][0][t], 1, 0, sb, false,
                            true);
                        kittens::sched::compiler_fence();
                        if (o + 2 == B_FRAGS) {
                            fa[0] = load_frag<!TRANSA>(a_src, 0, ks);
                            fa[1] = load_frag<!TRANSA>(a_src, 1, ks);
                        }
                        kittens::sched::compiler_fence();
#pragma unroll
                        for (int i = 2; i < A_FRAGS; i += 2) {
                            const int sa = sca[par][i / 2][t];
                            acc[i][o + 1] = __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(
                                FMT_A, fa[i], FMT_B, fb[o + 1], 0, acc[i][o + 1], 0, 0, sa, 1, 0, sb, false, true);
                            acc[i + 1][o + 1] = __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(
                                FMT_A, fa[i + 1], FMT_B, fb[o + 1], 0, acc[i + 1][o + 1], 1, 0, sa, 1, 0, sb, false,
                                true);
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
            if constexpr (MC) {
                kittens::cluster::wait();
            }
            kittens::sched::compiler_fence();
        }
    }

    const int row0 = m0 + 8 * h;
    const int col0 = n0 + s;
    const bool edge = (tile_m + 1) * BLOCK_M > M || (tile_n + 1) * BLOCK_N > N || M % 8 != 0;
    if constexpr (E == Epi::SPLIT) {
        float *d = static_cast<float *>(D) + size_t(tile_n) * BLOCK_N * M + size_t(tile_m) * BLOCK_M;
#pragma unroll
        for (int i = 0; i < A_FRAGS; i++) {
#pragma unroll
            for (int j = 0; j < B_FRAGS; j++) {
                const int row = row0 + i * 16;
                const int col = col0 + j * 16;
                if (!edge || (row < M - tile_m * BLOCK_M && col < N - tile_n * BLOCK_N)) {
                    *reinterpret_cast<v8f *>(d + size_t(col) * M + row) = acc[i][j];
                }
            }
        }
        return;
    }

    using StT = std::conditional_t<E == Epi::NONE && !ACC && sizeof(OutT) == 2, OutT, float>;
    constexpr int STRIDE = staged_stride(sizeof(StT));
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
    OutT *d             = static_cast<OutT *>(D);
    const SideT *bias_t = static_cast<const SideT *>(bias);
    SideT *aux_t        = static_cast<SideT *>(aux);
    if (edge) {
        copy_out<StT, E, ACC, OutT, SideT, !TRANSA, true>(lds0, d, bias_t, aux_t, alpha, beta, M, N, tile_m, tile_n);
    } else {
        copy_out<StT, E, ACC, OutT, SideT, !TRANSA, false>(lds0, d, bias_t, aux_t, alpha, beta, M, N, tile_m, tile_n);
    }
}

template <bool TRANSA, bool TRANSB, int FMT_A, int FMT_B, int CX, int CY, Epi E, bool ACC, typename OutT,
          typename SideT>
__global__ __cluster_dims__(CX, CY, 1) __launch_bounds__(NUM_THREADS, 1)
void mxfp8_gemm_kernel(const uint8_t *A, const uint8_t *B, const uint8_t *SA, const uint8_t *SB, void *D,
                       const void *bias, void *aux, float alpha, float beta, int M, int N, int K) {
    mxfp8_gemm<TRANSA, TRANSB, FMT_A, FMT_B, CX, CY, E, ACC, OutT, SideT>(A, B, SA, SB, D, bias, aux, alpha, beta, M,
                                                                          N, K);
}

template <bool TRANSA, bool TRANSB, int FMT_A, int FMT_B, Epi E, bool ACC, typename OutT, typename SideT>
__global__ __launch_bounds__(NUM_THREADS, 1)
void mxfp8_gemm_kernel_nomc(const uint8_t *A, const uint8_t *B, const uint8_t *SA, const uint8_t *SB, void *D,
                            const void *bias, void *aux, float alpha, float beta, int M, int N, int K) {
    mxfp8_gemm<TRANSA, TRANSB, FMT_A, FMT_B, 1, 1, E, ACC, OutT, SideT>(A, B, SA, SB, D, bias, aux, alpha, beta, M, N,
                                                                        K);
}

template <bool TRANSA, bool TRANSB, int FMT_A, int FMT_B, Epi E, bool ACC, typename OutT, typename SideT>
static bool launch_gemm(const GemmArgs &args, void *d, int splits, bool mcast) {
    const int tiles_m = ceil_div(args.M, BLOCK_M);
    const int tiles_n = ceil_div(args.N, BLOCK_N);
    const int cx      = !mcast ? 1 : tiles_m % 4 == 0 ? 4 : tiles_m % 2 == 0 ? 2 : 1;
    const int cy      = !mcast ? 1 : tiles_n % 4 == 0 ? 4 : tiles_n % 2 == 0 ? 2 : 1;
    auto kernel = mxfp8_gemm_kernel_nomc<TRANSA, TRANSB, FMT_A, FMT_B, E, ACC, OutT, SideT>;
    switch (cx * 10 + cy) {
    case 44: kernel = mxfp8_gemm_kernel<TRANSA, TRANSB, FMT_A, FMT_B, 4, 4, E, ACC, OutT, SideT>; break;
    case 42: kernel = mxfp8_gemm_kernel<TRANSA, TRANSB, FMT_A, FMT_B, 4, 2, E, ACC, OutT, SideT>; break;
    case 41: kernel = mxfp8_gemm_kernel<TRANSA, TRANSB, FMT_A, FMT_B, 4, 1, E, ACC, OutT, SideT>; break;
    case 24: kernel = mxfp8_gemm_kernel<TRANSA, TRANSB, FMT_A, FMT_B, 2, 4, E, ACC, OutT, SideT>; break;
    case 22: kernel = mxfp8_gemm_kernel<TRANSA, TRANSB, FMT_A, FMT_B, 2, 2, E, ACC, OutT, SideT>; break;
    case 21: kernel = mxfp8_gemm_kernel<TRANSA, TRANSB, FMT_A, FMT_B, 2, 1, E, ACC, OutT, SideT>; break;
    case 14: kernel = mxfp8_gemm_kernel<TRANSA, TRANSB, FMT_A, FMT_B, 1, 4, E, ACC, OutT, SideT>; break;
    case 12: kernel = mxfp8_gemm_kernel<TRANSA, TRANSB, FMT_A, FMT_B, 1, 2, E, ACC, OutT, SideT>; break;
    }
    kittens_check(hipFuncSetAttribute(reinterpret_cast<const void *>(kernel),
                                      hipFuncAttributeMaxDynamicSharedMemorySize, LDS_BYTES),
                  "gfx1250 mxfp8_gemm_kernel LDS");
    hipLaunchKernelGGL(kernel, dim3(tiles_m, tiles_n, splits), dim3(NUM_THREADS), LDS_BYTES, args.stream,
                       static_cast<const uint8_t *>(args.A), static_cast<const uint8_t *>(args.B),
                       static_cast<const uint8_t *>(args.scale_A), static_cast<const uint8_t *>(args.scale_B), d,
                       args.bias, args.aux_gelu, args.alpha, args.beta, args.M, args.N, args.K);
    kittens_check(hipGetLastError(), "gfx1250 mxfp8_gemm_kernel");
    return true;
}

template <bool TRANSA, bool TRANSB, int FMT_A, int FMT_B, bool ACC>
static bool dispatch_gemm(const GemmArgs &args, void *workspace, int splits, Epi epi, bool mcast) {
    const int out  = args.out_dtype;
    const int side = side_dtype(args);
    if (epi == Epi::NONE && splits > 1) {
        launch_gemm<TRANSA, TRANSB, FMT_A, FMT_B, Epi::SPLIT, false, float, float>(args, workspace, splits, mcast);
        splitk_reduce(args, workspace, splits);
        return true;
    }
    switch (epi) {
    case Epi::NONE:
        switch (out) {
        case KITTENS_FLOAT32:
            return launch_gemm<TRANSA, TRANSB, FMT_A, FMT_B, Epi::NONE, ACC, float, float>(args, args.C, 1, mcast);
        case KITTENS_FLOAT16:
            return launch_gemm<TRANSA, TRANSB, FMT_A, FMT_B, Epi::NONE, ACC, _Float16, _Float16>(args, args.C, 1,
                                                                                                 mcast);
        case KITTENS_BFLOAT16:
            return launch_gemm<TRANSA, TRANSB, FMT_A, FMT_B, Epi::NONE, ACC, __bf16, __bf16>(args, args.C, 1, mcast);
        }
        return false;
    case Epi::BIAS:
        if constexpr (TRANSA || !ACC) {
            if (out == KITTENS_FLOAT32 && side == KITTENS_FLOAT32) {
                return launch_gemm<TRANSA, TRANSB, FMT_A, FMT_B, Epi::BIAS, ACC, float, float>(args, args.C, 1, mcast);
            }
            if (out == KITTENS_FLOAT32 && side == KITTENS_BFLOAT16) {
                return launch_gemm<TRANSA, TRANSB, FMT_A, FMT_B, Epi::BIAS, ACC, float, __bf16>(args, args.C, 1,
                                                                                                mcast);
            }
            if (out == KITTENS_FLOAT16 && side == KITTENS_FLOAT16) {
                return launch_gemm<TRANSA, TRANSB, FMT_A, FMT_B, Epi::BIAS, ACC, _Float16, _Float16>(args, args.C, 1,
                                                                                                     mcast);
            }
            if (out == KITTENS_BFLOAT16 && side == KITTENS_BFLOAT16) {
                return launch_gemm<TRANSA, TRANSB, FMT_A, FMT_B, Epi::BIAS, ACC, __bf16, __bf16>(args, args.C, 1,
                                                                                                 mcast);
            }
        }
        return false;
    case Epi::GELU_AUX:
        if constexpr (TRANSA && !ACC) {
            if (side != KITTENS_BFLOAT16) {
                return false;
            }
            switch (out) {
            case KITTENS_FLOAT32:
                return launch_gemm<TRANSA, TRANSB, FMT_A, FMT_B, Epi::GELU_AUX, ACC, float, __bf16>(args, args.C, 1,
                                                                                                    mcast);
            case KITTENS_FLOAT16:
                return launch_gemm<TRANSA, TRANSB, FMT_A, FMT_B, Epi::GELU_AUX, ACC, _Float16, __bf16>(args, args.C,
                                                                                                       1, mcast);
            case KITTENS_BFLOAT16:
                return launch_gemm<TRANSA, TRANSB, FMT_A, FMT_B, Epi::GELU_AUX, ACC, __bf16, __bf16>(args, args.C, 1,
                                                                                                     mcast);
            }
        }
        return false;
    case Epi::GELU_AUX_BIAS:
        if constexpr (TRANSA && !ACC) {
            if (out == KITTENS_FLOAT32 && side == KITTENS_BFLOAT16) {
                return launch_gemm<TRANSA, TRANSB, FMT_A, FMT_B, Epi::GELU_AUX_BIAS, ACC, float, __bf16>(
                    args, args.C, 1, mcast);
            }
            if (out == KITTENS_FLOAT16 && side == KITTENS_FLOAT16) {
                return launch_gemm<TRANSA, TRANSB, FMT_A, FMT_B, Epi::GELU_AUX_BIAS, ACC, _Float16, _Float16>(
                    args, args.C, 1, mcast);
            }
            if (out == KITTENS_BFLOAT16 && side == KITTENS_BFLOAT16) {
                return launch_gemm<TRANSA, TRANSB, FMT_A, FMT_B, Epi::GELU_AUX_BIAS, ACC, __bf16, __bf16>(
                    args, args.C, 1, mcast);
            }
        }
        return false;
    default:
        return false;
    }
}

}  // namespace mxfp8

template <bool TRANSA, bool TRANSB>
bool launch_mxfp8(const GemmArgs &args, void *workspace, int splits, Epi epi, bool mcast) {
    const bool acc = args.beta != 0.f;
    const int fmt  = 2 * fp8_code(static_cast<KittensDType>(args.a_dtype)) +
                     fp8_code(static_cast<KittensDType>(args.b_dtype));
    switch (fmt) {
    case 0:
        return acc ? mxfp8::dispatch_gemm<TRANSA, TRANSB, 0, 0, true>(args, workspace, splits, epi, mcast)
                   : mxfp8::dispatch_gemm<TRANSA, TRANSB, 0, 0, false>(args, workspace, splits, epi, mcast);
    case 1:
        return acc ? mxfp8::dispatch_gemm<TRANSA, TRANSB, 0, 1, true>(args, workspace, splits, epi, mcast)
                   : mxfp8::dispatch_gemm<TRANSA, TRANSB, 0, 1, false>(args, workspace, splits, epi, mcast);
    case 2:
        return acc ? mxfp8::dispatch_gemm<TRANSA, TRANSB, 1, 0, true>(args, workspace, splits, epi, mcast)
                   : mxfp8::dispatch_gemm<TRANSA, TRANSB, 1, 0, false>(args, workspace, splits, epi, mcast);
    default:
        return acc ? mxfp8::dispatch_gemm<TRANSA, TRANSB, 1, 1, true>(args, workspace, splits, epi, mcast)
                   : mxfp8::dispatch_gemm<TRANSA, TRANSB, 1, 1, false>(args, workspace, splits, epi, mcast);
    }
}

}  // namespace te_kittens::cdna5
