/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/
#pragma once

#include "hip/hip_runtime.h"
#include "kittens.cuh"
#include "overlap_common.cuh"
#include <hip/hip_bfloat16.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace hk_rs_tn {

using namespace hk_overlap;

struct TileDesc {
    int chunk_id;
    int tile_m;
    int tile_n;
};

using namespace kittens;
using G_group = kittens::group<NUM_WARPS>;

constexpr int RS_MAX_TP = 8;

#ifndef COMM_WG
#define COMM_WG 8
#endif

// Sentinel selectors, declared here so the epilogue clamp below can see them. 
// Poison and clamp are quiet negative NaNs so neither can collide with a hardware NaN.
#define RS_SENT_BF16 0xFFAAu
#define RS_SENT_DW ((unsigned int)RS_SENT_BF16 * 0x00010001u)

struct RsPeers {
    unsigned int *arrive[RS_MAX_TP];
    bf16 *recv[RS_MAX_TP];
    unsigned int *ready[RS_MAX_TP];
    bf16 *stage[RS_MAX_TP];
};


#define RS_PUBLISH(p) __hip_atomic_fetch_add((p), 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM)
#define RS_SPIN(p)    __hip_atomic_load((p), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM)
#define RS_ACQUIRE(p) ((void)__hip_atomic_load((p), __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM))
#define LC_PUBLISH_TILE(p) __hip_atomic_fetch_add((p), 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM)
#define RS_TRACE_ARG

struct CdWalk {
    int q, tn, dq, dr;
};

__host__ __device__ __forceinline__
CdWalk cd_walk_init(int pos, int stride, int tiles_N) {
    CdWalk w;
    w.q  = pos / tiles_N;
    w.tn = pos - w.q * tiles_N;
    w.dq = stride / tiles_N;
    w.dr = stride - w.dq * tiles_N;
    return w;
}

template <int TP>
__host__ __device__ __forceinline__
TileDesc cd_walk_desc(CdWalk w, int my_pe, int bands) {
    const int j  = w.q % TP;
    const int tm = w.q / TP;
    TileDesc td;
    td.chunk_id = (j == 0) ? my_pe : ((j - 1 < my_pe) ? j - 1 : j);
    td.tile_m   = td.chunk_id * bands + tm;
    td.tile_n   = w.tn;
    return td;
}

__host__ __device__ __forceinline__
CdWalk cd_walk_next(CdWalk w, int tiles_N) {
    w.q  += w.dq;
    w.tn += w.dr;
    if (w.tn >= tiles_N) {
        w.tn -= tiles_N;
        w.q  += 1;
    }
    return w;
}

template <typename U, typename RT>
__device__ __forceinline__
void store_c_tile(U *base, const RT &src, int row_unit, int col_unit, int row_stride, int lane) {
    using T               = float;
    constexpr int packing = 2;
    U *dst_ptr            = base + (size_t)(row_unit * RT::rows) * row_stride + col_unit * RT::cols;
    const int row_offset  = RT::base_tile_stride * (lane / RT::base_tile_cols);
    const int col_offset  = lane % RT::base_tile_cols;

#pragma unroll
    for (int i = 0; i < RT::height; i++) {
#pragma unroll
        for (int j = 0; j < RT::width; j++) {
            const int col = j * RT::base_tile_cols + col_offset;
#pragma unroll
            for (int k = 0; k < RT::base_tile_num_strides; k++) {
                const int row = i * RT::base_tile_rows + row_offset +
                                k * RT::base_tile_elements_per_stride_group;
#pragma unroll
                for (int l = 0; l < RT::base_tile_stride / packing; l++) {
                    const int idx = l + k * RT::base_tile_stride / packing;
                    dst_ptr[(row + l * 2)     * row_stride + col] =
                        base_types::convertor<U, T>::convert(src.tiles[i][j].data[idx].x);
                    dst_ptr[(row + l * 2 + 1) * row_stride + col] =
                        base_types::convertor<U, T>::convert(src.tiles[i][j].data[idx].y);
                }
            }
        }
    }
}

typedef int rs_v4i __attribute__((ext_vector_type(4)));
typedef const volatile __attribute__((address_space(1))) rs_v4i *rs_gvol4;

__device__ __forceinline__
rs_v4i sent_load16(const rs_v4i *p) {
    return *(rs_gvol4)(size_t)p;
}

// NOTE: a dword compare can match a half of a real value that happens to equal the poison half.
__device__ __forceinline__
unsigned int sent_slot_pending(const int *__restrict__ v) {
    unsigned int hit = 0u;
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const unsigned int y = (unsigned int)v[j] ^ RS_SENT_DW;
        hit |= (y - 0x00010001u) & ~y & 0x80008000u;
    }
    return hit;
}

template <int TP, bool NT>
__device__ __forceinline__
void pull_reduce_all_sent(int my_pe, int ncomm, int bands, int tiles_N,
                          const bf16 *__restrict__ local_stage, const RsPeers &peers,
                          unsigned int *__restrict__ done, bf16 *__restrict__ out,
                          size_t band_elems) {
    const int w = (int)blockIdx.x;
    const size_t lines = band_elems / 8;
    const int peer = (w < TP - 1) ? (w + (w >= my_pe ? 1 : 0)) : -1;
    typedef int v4i __attribute__((ext_vector_type(4)));

    {
        const int b0 = 0;
        const size_t lines_g = lines * (size_t)bands;
        const size_t per_g   = (lines_g + (size_t)ncomm - 1) / (size_t)ncomm;
        const size_t g0      = (size_t)w * per_g;
        if (g0 >= lines_g) return;
        const size_t g1   = (g0 + per_g < lines_g) ? (g0 + per_g) : lines_g;
        const size_t soff = ((size_t)my_pe * bands + b0) * band_elems;
        v4i *dst          = (v4i *)(out + (size_t)b0 * band_elems);

        for (size_t l = g0 + threadIdx.x; l < g1; ) {
            v4i acc;
            unsigned int pend = 0u;
#pragma unroll 1
            for (int s = 0; s < TP; s++) {
                const bf16 *base = (s == my_pe) ? local_stage : peers.stage[s];
                const v4i v = sent_load16((const rs_v4i *)(base + soff) + l);
                pend |= sent_slot_pending((const int *)&v);
                if (s == 0) {
                    acc = v;
                } else {
                    __hip_bfloat16 *a = reinterpret_cast<__hip_bfloat16 *>(&acc);
                    const __hip_bfloat16 *x = reinterpret_cast<const __hip_bfloat16 *>(&v);
#pragma unroll
                    for (int j = 0; j < 8; j++) a[j] = bf16_add(a[j], x[j]);
                }
            }
            if (pend != 0u) {
                continue;
            }
            if (NT) {
                __builtin_nontemporal_store(acc, &dst[l]);
            } else {
                dst[l] = acc;
            }
            l += blockDim.x;
        }
    }
}

// Grid layout: [0, ncomm) communication workgroups, the rest draining the tile queue.
template <int TP>
__global__ __launch_bounds__(NUM_THREADS, 2)
void persistent_rs_bf16_gemm(const gl<bf16, 1, 1, -1, -1> A, const gl<bf16, 1, 1, -1, -1> B,
                             bf16 *__restrict__ local_stage, bf16 *__restrict__ out,
                             const TileDesc *__restrict__ work_queue, int num_tiles,
                             int *__restrict__ tile_counter, const RsPeers peers,
                             unsigned int *__restrict__ done, int my_pe, int comm_wg, int ncomm,
                               int bands, int tiles_N,
                             int xcd_bucket, const XcdBuckets buckets,
                             int *__restrict__ bucket_ctr RS_TRACE_ARG) {
    const int M       = A.rows();
    const int K       = A.cols();
    const int N_TOTAL = B.rows();
    const int k_tiles = K / K_STEP;
    (void)M;

    const size_t band_elems = (size_t)BLOCK_ROW * N_TOTAL;
    const size_t band_bytes = band_elems * sizeof(bf16);

    using ST_A = st_bf<HALF_ROW, K_STEP, st_16x32_s>;
    using ST_B = st_bf<HALF_COL, K_STEP, st_16x32_s>;
    using RT_A = rt_bf<REG_M, K_STEP, row_l, rt_16x32_s>;
    using RT_B = rt_bf<REG_N, K_STEP, row_l, rt_16x32_s>;
    using RT_C = rt_fl<REG_M, REG_N, col_l, rt_16x16_s>;

    __shared__ ST_A As[2][2];
    __shared__ ST_B Bs[2][2];

    const int warp_m = warpid() / WARPS_COL;
    const int warp_n = warpid() % WARPS_COL;
    const int wid    = warpid() % NUM_WARPS;

    using T = bf16;
    constexpr int bpt      = ST_A::underlying_subtile_bytes_per_thread;
    constexpr int bpm      = bpt * NUM_THREADS;
    constexpr int copies_A = HALF_ROW * K_STEP * sizeof(T) / bpm;
    constexpr int copies_B = HALF_COL * K_STEP * sizeof(T) / bpm;

    constexpr int elem_per_warp   = (16 / sizeof(T)) * kittens::WARP_THREADS;
    const uintptr_t lds_warp_off  = wid * elem_per_warp * sizeof(T);

    uint32_t a_lds_00 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&As[0][0].data[0]) + lds_warp_off));
    uint32_t a_lds_01 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&As[0][1].data[0]) + lds_warp_off));
    uint32_t a_lds_10 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&As[1][0].data[0]) + lds_warp_off));
    uint32_t a_lds_11 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&As[1][1].data[0]) + lds_warp_off));
    uint32_t b_lds_00 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&Bs[0][0].data[0]) + lds_warp_off));
    uint32_t b_lds_01 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&Bs[0][1].data[0]) + lds_warp_off));
    uint32_t b_lds_10 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&Bs[1][0].data[0]) + lds_warp_off));
    uint32_t b_lds_11 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&Bs[1][1].data[0]) + lds_warp_off));

    uint32_t sw_A[copies_A], sw_B[copies_B];
    G_group::prefill_swizzled_offsets(As[0][0], A, sw_A);
    G_group::prefill_swizzled_offsets(Bs[0][0], B, sw_B);

    const T *a_base = (const T *)&A[{0, 0, 0, 0}];
    const int a_row_stride = A.template stride<2>() * sizeof(T);
    const int b_row_stride = B.template stride<2>() * sizeof(T);
    i32x4 a_srd = make_srsrc(a_base, (size_t)M * a_row_stride, a_row_stride);
    const T *b_base = (const T *)&B[{0, 0, 0, 0}];
    i32x4 b_srd = make_srsrc(b_base, (size_t)N_TOTAL * b_row_stride, b_row_stride);

    if ((int)blockIdx.x < ncomm) {
        // The comm workgroups are the reduce-scatter: they read all eight sources and write `out` once.
        (void)comm_wg; (void)band_bytes;
        pull_reduce_all_sent<TP, true>(my_pe, ncomm, bands, tiles_N, local_stage, peers, done, out,
                                       band_elems);
    }

    const int hy_ncw = (int)gridDim.x - ncomm;
    const int hy_vid = (int)blockIdx.x - ncomm;
    const int hy_R   = (hy_ncw > 0) ? (num_tiles / hy_ncw) : 0;
    const int hy_S   = hy_R * hy_ncw;
    int hy_left      = (hy_vid >= 0 && !xcd_bucket) ? hy_R : 0;
    int hy_tick      = hy_vid;
    CdWalk hy_cd     = cd_walk_init((hy_vid >= 0) ? hy_vid : 0, (hy_ncw > 0) ? hy_ncw : 1, tiles_N);
    while (true) {
        __shared__ int s_tile_idx;
        const bool hy_static = (hy_left > 0);
        if (threadIdx.x == 0) {
            if (xcd_bucket) {
                // Steal order: own bucket, then the local chunk, then the other XCDs.
                int found = -1;
                const int b0 = (int)blockIdx.x % NUM_XCDS_AFF;
                for (int s = 0; s <= NUM_XCDS_AFF; s++) {
                    int bb;
                    if (s == 0) {
                        bb = b0;
                    } else if (s == 1) {
                        bb = my_pe;
                    } else {
                        bb = (b0 + s - 1) & (NUM_XCDS_AFF - 1);
                    }
                    if (buckets.cnt[bb] == 0) continue;
                    if (__hip_atomic_load(&bucket_ctr[bb], __ATOMIC_RELAXED,
                                          __HIP_MEMORY_SCOPE_AGENT) >= buckets.cnt[bb]) continue;
                    const int idx = atomicAdd(&bucket_ctr[bb], 1);
                    if (idx < buckets.cnt[bb]) {
                        found = buckets.off[bb] + idx;
                        break;
                    }
                }
                s_tile_idx = (found < 0) ? num_tiles : found;
            } else {
                if (hy_static) {
                    s_tile_idx = hy_tick;
                } else {
                    s_tile_idx = hy_S + atomicAdd(tile_counter, 1);
                }
            }
        }
        __syncthreads();
        int tile_idx = s_tile_idx;
        if (tile_idx >= num_tiles) break;

        TileDesc desc;
        if (hy_static) {
            desc    = cd_walk_desc<TP>(hy_cd, my_pe, bands);
            hy_cd   = cd_walk_next(hy_cd, tiles_N);
            hy_tick += hy_ncw;
            hy_left -= 1;
        } else {
            desc = work_queue[tile_idx];
        }

        const int owner = desc.chunk_id;
        const int lrow  = desc.tile_m - owner * bands;

        int block_row = desc.tile_m;
        int block_col = desc.tile_n;

        RT_A a_tile;
        RT_B b_tile_0, b_tile_1;
        RT_C c00, c01, c10, c11;
        zero(c00); zero(c01); zero(c10); zero(c11);

        int tic = 0, toc = 1;

        G_group::load(Bs[tic][0], B, {0, 0, block_col * 2,     0}, sw_B, b_srd, b_base, b_lds_00);
        G_group::load(As[tic][0], A, {0, 0, block_row * 2,     0}, sw_A, a_srd, a_base, a_lds_00);
        G_group::load(Bs[tic][1], B, {0, 0, block_col * 2 + 1, 0}, sw_B, b_srd, b_base, b_lds_01);
        G_group::load(As[tic][1], A, {0, 0, block_row * 2 + 1, 0}, sw_A, a_srd, a_base, a_lds_01);

        if (warp_m == 1) {
            __builtin_amdgcn_s_barrier();
        }

        asm volatile("s_waitcnt vmcnt(4)");
        __builtin_amdgcn_s_barrier();

        G_group::load(Bs[toc][0], B, {0, 0, block_col * 2,     1}, sw_B, b_srd, b_base, b_lds_10);
        G_group::load(As[toc][0], A, {0, 0, block_row * 2,     1}, sw_A, a_srd, a_base, a_lds_10);
        G_group::load(Bs[toc][1], B, {0, 0, block_col * 2 + 1, 1}, sw_B, b_srd, b_base, b_lds_11);

        asm volatile("s_waitcnt vmcnt(4)");
        __builtin_amdgcn_s_barrier();

#pragma unroll 1
        for (int tile = 0; tile < k_tiles - 2; tile += 2) {

            auto st_b = subtile_inplace<REG_N, K_STEP>(Bs[0][0], {warp_n, 0});
            load(b_tile_0, st_b);
            auto st_a = subtile_inplace<REG_M, K_STEP>(As[0][0], {warp_m, 0});
            load(a_tile, st_a);
            G_group::load(As[1][1], A, {0, 0, block_row * 2 + 1, tile + 1}, sw_A, a_srd, a_base, a_lds_11);
            asm volatile("s_waitcnt lgkmcnt(8)");
            __builtin_amdgcn_s_barrier();

            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(c00, a_tile, b_tile_0, c00);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            st_b = subtile_inplace<REG_N, K_STEP>(Bs[0][1], {warp_n, 0});
            load(b_tile_1, st_b);
            G_group::load(Bs[0][0], B, {0, 0, block_col * 2, tile + 2}, sw_B, b_srd, b_base, b_lds_00);
            __builtin_amdgcn_s_barrier();

            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(c01, a_tile, b_tile_1, c01);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            st_a = subtile_inplace<REG_M, K_STEP>(As[0][1], {warp_m, 0});
            load(a_tile, st_a);
            G_group::load(As[0][0], A, {0, 0, block_row * 2, tile + 2}, sw_A, a_srd, a_base, a_lds_00);
            __builtin_amdgcn_s_barrier();

            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(c10, a_tile, b_tile_0, c10);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            st_b = subtile_inplace<REG_N, K_STEP>(Bs[1][0], {warp_n, 0});
            load(b_tile_0, st_b);
            G_group::load(Bs[0][1], B, {0, 0, block_col * 2 + 1, tile + 2}, sw_B, b_srd, b_base, b_lds_01);
            asm volatile("s_waitcnt vmcnt(6)");
            __builtin_amdgcn_s_barrier();

            __builtin_amdgcn_s_setprio(1);
            mma_ABt(c11, a_tile, b_tile_1, c11);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            st_a = subtile_inplace<REG_M, K_STEP>(As[1][0], {warp_m, 0});
            load(a_tile, st_a);
            G_group::load(As[0][1], A, {0, 0, block_row * 2 + 1, tile + 2}, sw_A, a_srd, a_base, a_lds_01);
            asm volatile("s_waitcnt lgkmcnt(8)");
            __builtin_amdgcn_s_barrier();

            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(c00, a_tile, b_tile_0, c00);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            st_b = subtile_inplace<REG_N, K_STEP>(Bs[1][1], {warp_n, 0});
            load(b_tile_1, st_b);
            G_group::load(Bs[1][0], B, {0, 0, block_col * 2, tile + 3}, sw_B, b_srd, b_base, b_lds_10);
            __builtin_amdgcn_s_barrier();

            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(c01, a_tile, b_tile_1, c01);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            st_a = subtile_inplace<REG_M, K_STEP>(As[1][1], {warp_m, 0});
            load(a_tile, st_a);
            G_group::load(As[1][0], A, {0, 0, block_row * 2, tile + 3}, sw_A, a_srd, a_base, a_lds_10);
            __builtin_amdgcn_s_barrier();

            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(c10, a_tile, b_tile_0, c10);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            G_group::load(Bs[1][1], B, {0, 0, block_col * 2 + 1, tile + 3}, sw_B, b_srd, b_base, b_lds_11);
            asm volatile("s_waitcnt vmcnt(4)");
            __builtin_amdgcn_s_barrier();

            __builtin_amdgcn_s_setprio(1);
            mma_ABt(c11, a_tile, b_tile_1, c11);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }

        {
            int tile = k_tiles - 2;

            auto st_b = subtile_inplace<REG_N, K_STEP>(Bs[tic][0], {warp_n, 0});
            load(b_tile_0, st_b);
            auto st_a = subtile_inplace<REG_M, K_STEP>(As[tic][0], {warp_m, 0});
            load(a_tile, st_a);
            G_group::load(As[toc][1], A, {0, 0, block_row * 2 + 1, tile + 1}, sw_A, a_srd, a_base, a_lds_11);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");

            __builtin_amdgcn_s_setprio(1);
            mma_ABt(c00, a_tile, b_tile_0, c00);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            st_b = subtile_inplace<REG_N, K_STEP>(Bs[tic][1], {warp_n, 0});
            load(b_tile_1, st_b);
            __builtin_amdgcn_s_barrier();

            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(c01, a_tile, b_tile_1, c01);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            st_a = subtile_inplace<REG_M, K_STEP>(As[tic][1], {warp_m, 0});
            load(a_tile, st_a);
            asm volatile("s_waitcnt vmcnt(4)");
            __builtin_amdgcn_s_barrier();

            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(c10, a_tile, b_tile_0, c10);
            mma_ABt(c11, a_tile, b_tile_1, c11);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            tic ^= 1; toc ^= 1;
        }

        {
            auto st_b = subtile_inplace<REG_N, K_STEP>(Bs[tic][0], {warp_n, 0});
            load(b_tile_0, st_b);
            auto st_a = subtile_inplace<REG_M, K_STEP>(As[tic][0], {warp_m, 0});
            load(a_tile, st_a);
            asm volatile("s_waitcnt vmcnt(2)");
            __builtin_amdgcn_s_barrier();

            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(c00, a_tile, b_tile_0, c00);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            st_b = subtile_inplace<REG_N, K_STEP>(Bs[tic][1], {warp_n, 0});
            load(b_tile_1, st_b);
            asm volatile("s_waitcnt vmcnt(0)");
            __builtin_amdgcn_s_barrier();

            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(c01, a_tile, b_tile_1, c01);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            st_a = subtile_inplace<REG_M, K_STEP>(As[tic][1], {warp_m, 0});
            load(a_tile, st_a);
            __builtin_amdgcn_s_barrier();

            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(c10, a_tile, b_tile_0, c10);
            mma_ABt(c11, a_tile, b_tile_1, c11);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }

        if (warp_m == 0) {
            __builtin_amdgcn_s_barrier();
        }

        bf16 *sbase = local_stage + ((size_t)owner * bands + lrow) * band_elems;

        const int rf0 = __builtin_amdgcn_readfirstlane(warp_m);
        const int rf1 = __builtin_amdgcn_readfirstlane(WARPS_ROW + warp_m);
        const int cf0 = __builtin_amdgcn_readfirstlane(block_col * WARPS_COL * 2 + warp_n);
        const int cf1 = __builtin_amdgcn_readfirstlane(block_col * WARPS_COL * 2 + WARPS_COL + warp_n);
        int lane_epi  = kittens::laneid();
        asm volatile("" : "+v"(lane_epi));

        store_c_tile<bf16>(sbase, c00, rf0, cf0, N_TOTAL, lane_epi);
        store_c_tile<bf16>(sbase, c01, rf0, cf1, N_TOTAL, lane_epi);
        store_c_tile<bf16>(sbase, c10, rf1, cf0, N_TOTAL, lane_epi);
        store_c_tile<bf16>(sbase, c11, rf1, cf1, N_TOTAL, lane_epi);

        __syncthreads();
        if (threadIdx.x == 0) LC_PUBLISH_TILE(&done[(size_t)owner * bands + lrow]);
        __syncthreads();
    }   // end persistent loop

}

static std::vector<TileDesc> build_rs_work_queue(int M, int N_total, int K, int tp_size, int my_pe) {
    (void)K;
    const int tiles_N = N_total / BLOCK_COL;
    const int bands   = (M / tp_size) / BLOCK_ROW;

    std::vector<TileDesc> queue;
    queue.reserve((size_t)tiles_N * bands * tp_size);

    auto emit = [&](int chunk, int tm) {
        for (int tn = 0; tn < tiles_N; tn++) {
            TileDesc td{};
            td.chunk_id = chunk;
            td.tile_m   = chunk * bands + tm;
            td.tile_n   = tn;
            queue.push_back(td);
        }
    };

    for (int tm = 0; tm < bands; tm++) {
        emit(my_pe, tm);
        for (int c = 0; c < tp_size; c++) {
            if (c != my_pe) {
                emit(c, tm);
            }
        }
    }
    return queue;
}

struct RsLaunchCfg {
    int comm_wg    = COMM_WG;
    int xcd_bucket = 0;
};

static void launch_persistent_rs(int M, int N_TOTAL, int K, bf16 *d_a, bf16 *d_b,
                                 bf16 *d_local_stage, bf16 *d_out, TileDesc *d_queue, int num_tiles,
                                 int *d_tile_counter, RsPeers peers, unsigned int *d_done,
                                 int my_pe, int tp_size, const RsLaunchCfg &cfg,
                                 XcdBuckets buckets, int *d_bucket_ctr, hipStream_t stream,
                                 unsigned long long *d_trace = nullptr) {
    (void)d_trace;
    const int tiles_M = M / BLOCK_ROW;
    const int tiles_N = N_TOTAL / BLOCK_COL;
    const int bands   = (M / tp_size) / BLOCK_ROW;

    gl<bf16, 1, 1, -1, -1> A_gl(d_a, nullptr, nullptr, (size_t)M,       (size_t)K);
    gl<bf16, 1, 1, -1, -1> B_gl(d_b, nullptr, nullptr, (size_t)N_TOTAL, (size_t)K);

    const int ncomm = (tp_size - 1) * cfg.comm_wg;

    int grid = tiles_M * tiles_N + ncomm;
    if (grid > GRID_CAP) grid = GRID_CAP;
    if (grid < ncomm + 1) grid = ncomm + 1;
#define RS_LAUNCH(TPV)                                                                         \
    persistent_rs_bf16_gemm<TPV><<<grid, NUM_THREADS, 0, stream>>>(                            \
        A_gl, B_gl, d_local_stage, d_out, d_queue, num_tiles, d_tile_counter, peers, d_done,   \
        my_pe, cfg.comm_wg, ncomm, bands, tiles_N, cfg.xcd_bucket, buckets, d_bucket_ctr)

    switch (tp_size) {
        case 8: RS_LAUNCH(8); break;
        case 4: RS_LAUNCH(4); break;
        default: break;
    }
#undef RS_LAUNCH
}

}  // namespace hk_rs_tn
