/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/
#pragma once

#include "hip/hip_runtime.h"
#include "kittens.cuh"
#include "overlap_common.cuh"
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace hk_mxfp8_ag_tn {

using namespace kittens;
using namespace hk_overlap;

// Per-PE pointer to each peer's [M,K] A buffer.
using PeerPtrs = hk_overlap::PeerPtrsT<fp8e4m3>;

struct TileDesc {
    int chunk_id;
    int tile_m;
    int tile_n;
};

constexpr int K_STEP = 128;

// MFMA cbsz/blgp format codes: 0 = e4m3, 1 = e5m2. e4m3 inputs only.
constexpr int CBSZ = 0;
constexpr int BLGP = 0;

// Scale tile shared by mxfp8 kernels; one fp8e8m0_4 per (group, lane)
using ST_Scale = kittens::st<kittens::fp8e8m0, 16, 64, kittens::st_16x64_s>;

// Reads the pre-packed lane-native scale for group lg on this lane
__device__ __forceinline__ kittens::fp8e8m0_4 lane_rd(const ST_Scale &s, int lg) {
    return reinterpret_cast<const uint32_t *>(s.data)[lg * 64 + kittens::laneid()];
}

template <int U, bool NT>
__device__ __forceinline__
void gather_peer_tile(int peer, int tn, int sub, int gath_wg, int tiles_per_chunk, char *gather_dst,
                      const PeerPtrs &peers, size_t chunk_bytes, unsigned int *arrive) {
    const size_t tile_bytes = chunk_bytes / tiles_per_chunk;
    const size_t doff       = (size_t)peer * chunk_bytes + (size_t)tn * tile_bytes;

    size_t sub_bytes = (((tile_bytes + gath_wg - 1) / gath_wg) + 15) & ~size_t(15);
    size_t o         = (size_t)sub * sub_bytes;
    size_t l         = (o >= tile_bytes) ? 0 : ((o + sub_bytes <= tile_bytes) ? sub_bytes : tile_bytes - o);

    if (l) gather_copy_wg<U, NT>(gather_dst + doff + o, (const char *)peers.base[peer] + doff + o, l);

    __syncthreads();
    if (NT) {
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    } else {
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    }
    __syncthreads();
    if (threadIdx.x == 0) AG_PUBLISH(&arrive[peer * tiles_per_chunk + tn]);
    __syncthreads();
}

template <int U, bool NT>
__device__ __forceinline__
void gather_all(int my_pe, int gath_wg, int tiles_per_chunk, char *gb, const PeerPtrs &peers,
                size_t chunk_bytes, unsigned int *arrive) {
    const int pi   = (int)blockIdx.x / gath_wg;
    const int sub  = (int)blockIdx.x % gath_wg;
    const int peer = pi + (pi >= my_pe ? 1 : 0);
    for (int tn = 0; tn < tiles_per_chunk; tn++) {
        gather_peer_tile<U, NT>(peer, tn, sub, gath_wg, tiles_per_chunk, gb, peers, chunk_bytes, arrive);
    }
}

template <typename U, typename RT>
__device__ __forceinline__
void store_c_tile(U *base, const RT &src, int row_unit, int col_unit, int row_stride, int lane) {
    using T               = float;
    constexpr int packing = 2;                      // rt_fl holds float2 per data slot
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
                const int row = i * RT::base_tile_rows + row_offset + k * RT::base_tile_elements_per_stride_group;
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

// Epilogue for the AG path: C is [M, N_TOTAL] (row index = A-operand row), the convention
// fused_ag_gemm_tn.cuh uses. Stores straight from the col_l accumulators via store_c_tile --
// no transpose, unlike the non-AG mxfp8_gemm.cpp epilogue whose C is [N, M].
template<typename RT_C>
__device__ __forceinline__ void gemm_epilogue(
    RT_C &cA, RT_C &cB, RT_C &cC, RT_C &cD,
    bf16 *c_base, int N_TOTAL,
    int block_row, int block_col, int warp_m, int warp_n) {

    const int rf0 = __builtin_amdgcn_readfirstlane(block_row * WARPS_ROW * 2 + warp_m);
    const int rf1 = __builtin_amdgcn_readfirstlane(block_row * WARPS_ROW * 2 + WARPS_ROW + warp_m);
    const int cf0 = __builtin_amdgcn_readfirstlane(block_col * WARPS_COL * 2 + warp_n);
    const int cf1 = __builtin_amdgcn_readfirstlane(block_col * WARPS_COL * 2 + WARPS_COL + warp_n);
    int lane_epi  = kittens::laneid();
    asm volatile("" : "+v"(lane_epi));

    store_c_tile<bf16>(c_base, cA, rf0, cf0, N_TOTAL, lane_epi);
    store_c_tile<bf16>(c_base, cB, rf0, cf1, N_TOTAL, lane_epi);
    store_c_tile<bf16>(c_base, cC, rf1, cf0, N_TOTAL, lane_epi);
    store_c_tile<bf16>(c_base, cD, rf1, cf1, N_TOTAL, lane_epi);
}

__global__ __launch_bounds__(NUM_THREADS, 2)
void persistent_ag_mxfp8_gemm(const gl<fp8e4m3, 1, 1, -1, -1> A, const gl<fp8e4m3, 1, 1, -1, -1> B,
    const gl<bf16, 1, 1, -1, -1> C, const gl<fp8e8m0, -1, 1, 16, 64> scale_A_gl,
    const gl<fp8e8m0, -1, 1, 16, 64> scale_B_gl, const TileDesc *__restrict__ work_queue,
    int num_tiles, int *__restrict__ tile_counter, const PeerPtrs peers, unsigned int *__restrict__ arrive,
    int my_pe, int tp_size, int gath_wg, int tiles_per_chunk, size_t chunk_bytes,
    int xcd_bucket, const XcdBuckets buckets, int *__restrict__ bucket_ctr) {

    const int M       = A.rows();
    const int K       = A.cols();
    const int N_TOTAL = B.rows();
    const int k_tiles = K / K_STEP;
    const int tiles_M = M / BLOCK_ROW;
    const int tiles_N = N_TOTAL / BLOCK_COL;

    using ST_A     = st_fp8e4m3<HALF_ROW, K_STEP, st_16x128_s>;
    using ST_B     = st_fp8e4m3<HALF_COL, K_STEP, st_16x128_s>;
    using RT_A     = rt_fp8e4m3<REG_M, K_STEP>;
    using RT_B     = rt_fp8e4m3<REG_N, K_STEP>;
    using RT_C     = rt_fl<REG_M, REG_N, col_l, rt_16x16_s>;

    __shared__ ST_A As[2][2];
    __shared__ ST_B Bs[2][2];

    // B needs 8 scale groups = 2 tiles
    __shared__ ST_Scale scale_A_smem[2], scale_B_lo[2], scale_B_hi[2];

    const int warp_m = kittens::warpid() / WARPS_COL;
    const int warp_n = kittens::warpid() % WARPS_COL;
    const int wid    = kittens::warpid() % NUM_WARPS;

    using T = kittens::fp8e4m3;

    constexpr int bpt = ST_A::underlying_subtile_bytes_per_thread;
    constexpr int bpm = bpt * NUM_THREADS;
    constexpr int copies_A = HALF_ROW * K_STEP * sizeof(T) / bpm;
    constexpr int copies_B = HALF_COL * K_STEP * sizeof(T) / bpm;
    uint32_t sw_A[copies_A], sw_B[copies_B];
    G_group::prefill_swizzled_offsets(As[0][0], A, sw_A);
    G_group::prefill_swizzled_offsets(Bs[0][0], B, sw_B);

    const T *b_base = (const T *)&B[{0, 0, 0, 0}];
    const int b_row_stride = B.template stride<2>() * sizeof(T);
    kittens::i32x4 b_srd = kittens::make_srsrc(b_base, (size_t)N_TOTAL * b_row_stride, b_row_stride);
    bf16 *c_base = (bf16 *)&C[{0, 0, 0, 0}];

    constexpr int elem_per_warp = (16 / sizeof(T)) * kittens::WARP_THREADS;
    uint32_t b_lds[2][2];
    for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) {
        b_lds[i][j] = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
            reinterpret_cast<uintptr_t>(&Bs[i][j].data[0]) + wid * elem_per_warp * sizeof(T)));
    }

    const int NGATH = (tp_size - 1) * gath_wg;
    if ((int)blockIdx.x < NGATH) {
        char *gb = (char *)&A[{0, 0, 0, 0}];
        gather_all<1, true>(my_pe, gath_wg, tiles_per_chunk, gb, peers, chunk_bytes, arrive);
    }

    const bool static_sched = (num_tiles <= SCHED_ROUNDS * (int)gridDim.x);
    int sched_iter = 0;
    while (true) {
        __shared__ int s_tile_idx;
        if (threadIdx.x == 0) {
            if (xcd_bucket) {
                // Steal order: Own bucket, then the local chunk, then the other XCDs.
                int found = -1;
                const int b0 = (int)blockIdx.x % NUM_XCDS_AFF;
                for (int s = 0; s <= NUM_XCDS_AFF; s++) {
                    int bb;
                    if      (s == 0) bb = b0;
                    else if (s == 1) bb = my_pe;
                    else             bb = (b0 + s - 1) & (NUM_XCDS_AFF - 1);
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
                s_tile_idx = static_sched ? (int)(blockIdx.x + (long)sched_iter * gridDim.x)
                                          : atomicAdd(tile_counter, 1);
            }
        }
        __syncthreads();
        int tile_idx = s_tile_idx;
        sched_iter++;
        if (tile_idx >= num_tiles) break;

        TileDesc desc = work_queue[tile_idx];
        if (desc.chunk_id != my_pe) {
            const int tm                   = desc.tile_m - desc.chunk_id * tiles_per_chunk;
            const unsigned needed_arrivals = (unsigned)gath_wg;
            unsigned int *f                = &arrive[(size_t)desc.chunk_id * tiles_per_chunk + tm];
            if (threadIdx.x == 0) {
                do {
                } while (AG_SPIN(f) < needed_arrivals);
                AG_ACQUIRE(f);
            }
            __syncthreads();
        }

        int block_row = desc.tile_m;
        int block_col = desc.tile_n;

        int a_half0 = block_row * 2;
        int a_half1 = a_half0 + 1;
        int b_half0 = block_col * 2;
        int b_half1 = b_half0 + 1;

        RT_A a;
        RT_B b0, b1;
        RT_C cA, cB, cC, cD;
        kittens::zero(cA); kittens::zero(cB); kittens::zero(cC); kittens::zero(cD);

        int tic = 0, toc = 1;
        int tic_scales = 0, toc_scales = 1;

        G_group::load(Bs[tic][0], B, {0, 0, b_half0, 0}, sw_B, b_srd, b_base, b_lds[tic][0]);
        G_group::load(As[tic][0], A, {0, 0, a_half0, 0}, sw_A);
        G_group::load(Bs[tic][1], B, {0, 0, b_half1, 0}, sw_B, b_srd, b_base, b_lds[tic][1]);
        G_group::load(As[tic][1], A, {0, 0, a_half1, 0}, sw_A);

        if (warp_m == 1) __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt vmcnt(4)"); // wait for tic[0] halves; tic[1] halves still in flight
        __builtin_amdgcn_s_barrier();

        G_group::load(As[toc][0], A, {0, 0, a_half0, 1}, sw_A);
        G_group::load(Bs[toc][0], B, {0, 0, b_half0, 1}, sw_B, b_srd, b_base, b_lds[toc][0]);
        G_group::load(Bs[toc][1], B, {0, 0, b_half1, 1}, sw_B, b_srd, b_base, b_lds[toc][1]);
        asm volatile("s_waitcnt vmcnt(6)"); // wait for tic[1] halves; 3 toc loads + scales in flight
        __builtin_amdgcn_s_barrier();

        G_group::load(scale_A_smem[0], scale_A_gl, {block_row, 0, 0, 0});
        G_group::load(scale_B_lo[0],   scale_B_gl, {2 * block_col,     0, 0, 0});
        G_group::load(scale_B_hi[0],   scale_B_gl, {2 * block_col + 1, 0, 0, 0});
        asm volatile("s_waitcnt vmcnt(0)"); // drain all VMEM before first MMA
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();

#pragma unroll 2
        for (int k = 0; k < k_tiles - 2; k++, tic ^= 1, toc ^= 1, tic_scales ^= 1, toc_scales ^= 1) {
            if (k + 1 < k_tiles) {
                G_group::load(scale_A_smem[toc_scales], scale_A_gl, {(k + 1) * tiles_M + block_row, 0, 0, 0});
                G_group::load(scale_B_lo[toc_scales],   scale_B_gl, {2 * ((k + 1) * tiles_N + block_col),     0, 0, 0});
                G_group::load(scale_B_hi[toc_scales],   scale_B_gl, {2 * ((k + 1) * tiles_N + block_col) + 1, 0, 0, 0});
            }
            auto bs0 = kittens::subtile_inplace<REG_N, K_STEP>(Bs[tic][0], {warp_n, 0});
            kittens::load(b0, bs0);
            auto as0 = kittens::subtile_inplace<REG_M, K_STEP>(As[tic][0], {warp_m, 0});
            kittens::load(a, as0);
            G_group::load(As[toc][1], A, {0, 0, a_half1, k + 1}, sw_A);
            asm volatile("s_waitcnt lgkmcnt(8)"); // wait for scale + data LDS stores; As[toc][1] can overlap
            __builtin_amdgcn_s_barrier();

            kittens::fp8e8m0_4 sa_h0 = lane_rd(scale_A_smem[tic_scales], warp_m);
            kittens::fp8e8m0_4 sb_h0 = lane_rd(scale_B_lo[tic_scales], warp_n);
            kittens::fp8e8m0_4 sb_h1 = lane_rd(scale_B_hi[tic_scales], warp_n);
            kittens::fp8e8m0_4 sa_h1 = lane_rd(scale_A_smem[tic_scales], 2 + warp_m);
            __builtin_amdgcn_s_setprio(2);
            kittens::mma_ABt_scaled<CBSZ, BLGP>(cA, a, b0, cA, &sa_h0, &sb_h0);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            auto bs1 = kittens::subtile_inplace<REG_N, K_STEP>(Bs[tic][1], {warp_n, 0});
            kittens::load(b1, bs1);
            G_group::load(As[tic][0], A, {0, 0, a_half0, k + 2}, sw_A);
            asm volatile("s_waitcnt lgkmcnt(0)"); // drain LDS: need bs1 in registers for mma_B
            __builtin_amdgcn_s_barrier();

            __builtin_amdgcn_s_setprio(2);
            kittens::mma_ABt_scaled<CBSZ, BLGP>(cB, a, b1, cB, &sa_h0, &sb_h1);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            auto as1 = kittens::subtile_inplace<REG_M, K_STEP>(As[tic][1], {warp_m, 0});
            kittens::load(a, as1);
            G_group::load(Bs[tic][0], B, {0, 0, b_half0, k + 2}, sw_B, b_srd, b_base, b_lds[tic][0]);
            asm volatile("s_waitcnt lgkmcnt(0)"); // drain LDS: need as1 in registers for mma_C
            __builtin_amdgcn_s_barrier();

            __builtin_amdgcn_s_setprio(2);
            kittens::mma_ABt_scaled<CBSZ, BLGP>(cC, a, b0, cC, &sa_h1, &sb_h0);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            G_group::load(Bs[tic][1], B, {0, 0, b_half1, k + 2}, sw_B, b_srd, b_base, b_lds[tic][1]);
            asm volatile("s_waitcnt vmcnt(6)"); // wait for toc data loads; next-iter prefetches in flight
            __builtin_amdgcn_s_barrier();

            __builtin_amdgcn_s_setprio(2);
            kittens::mma_ABt_scaled<CBSZ, BLGP>(cD, a, b1, cD, &sa_h1, &sb_h1);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }

        { // Epilogue k = k_tiles - 2
            int k = k_tiles - 2;
            if (k + 1 < k_tiles) {
                G_group::load(scale_A_smem[toc_scales], scale_A_gl, {(k + 1) * tiles_M + block_row, 0, 0, 0});
                G_group::load(scale_B_lo[toc_scales],   scale_B_gl, {2 * ((k + 1) * tiles_N + block_col),     0, 0, 0});
                G_group::load(scale_B_hi[toc_scales],   scale_B_gl, {2 * ((k + 1) * tiles_N + block_col) + 1, 0, 0, 0});
            }
            asm volatile("s_waitcnt vmcnt(0)"); // drain all VMEM: last prefetch iteration
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_barrier();
            kittens::fp8e8m0_4 sa_h0 = lane_rd(scale_A_smem[tic_scales], warp_m);
            kittens::fp8e8m0_4 sa_h1 = lane_rd(scale_A_smem[tic_scales], 2 + warp_m);
            kittens::fp8e8m0_4 sb_h0 = lane_rd(scale_B_lo[tic_scales], warp_n);
            kittens::fp8e8m0_4 sb_h1 = lane_rd(scale_B_hi[tic_scales], warp_n);

            auto bs0 = kittens::subtile_inplace<REG_N, K_STEP>(Bs[tic][0], {warp_n, 0});
            kittens::load(b0, bs0);
            auto as0 = kittens::subtile_inplace<REG_M, K_STEP>(As[tic][0], {warp_m, 0});
            kittens::load(a, as0);
            G_group::load(As[toc][1], A, {0, 0, a_half1, k + 1}, sw_A);
            __builtin_amdgcn_s_barrier();

            asm volatile("s_waitcnt lgkmcnt(0)"); // need as0/bs0 in registers for mma_A
            __builtin_amdgcn_s_setprio(2);
            kittens::mma_ABt_scaled<CBSZ, BLGP>(cA, a, b0, cA, &sa_h0, &sb_h0);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            auto bs1 = kittens::subtile_inplace<REG_N, K_STEP>(Bs[tic][1], {warp_n, 0});
            kittens::load(b1, bs1);
            __builtin_amdgcn_s_barrier();

            asm volatile("s_waitcnt lgkmcnt(0)"); // need bs1 in registers for mma_B
            __builtin_amdgcn_s_setprio(2);
            kittens::mma_ABt_scaled<CBSZ, BLGP>(cB, a, b1, cB, &sa_h0, &sb_h1);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            auto as1 = kittens::subtile_inplace<REG_M, K_STEP>(As[tic][1], {warp_m, 0});
            kittens::load(a, as1);
            __builtin_amdgcn_s_barrier();

            asm volatile("s_waitcnt lgkmcnt(0)"); // need as1 in registers for mma_C
            __builtin_amdgcn_s_setprio(2);
            kittens::mma_ABt_scaled<CBSZ, BLGP>(cC, a, b0, cC, &sa_h1, &sb_h0);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            auto bs0_next = kittens::subtile_inplace<REG_N, K_STEP>(Bs[toc][0], {warp_n, 0});
            kittens::load(b0, bs0_next);
            asm volatile("s_waitcnt vmcnt(4)"); // wait for toc data; As[toc][1] still in flight
            __builtin_amdgcn_s_barrier();

            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(2);
            kittens::mma_ABt_scaled<CBSZ, BLGP>(cD, a, b1, cD, &sa_h1, &sb_h1);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            tic ^= 1; toc ^= 1;
            tic_scales ^= 1; toc_scales ^= 1;
        }

        { // Final epilogue k = k_tiles - 1
            asm volatile("s_waitcnt vmcnt(0)");
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_barrier();
            kittens::fp8e8m0_4 sa_h0 = lane_rd(scale_A_smem[tic_scales], warp_m);
            kittens::fp8e8m0_4 sa_h1 = lane_rd(scale_A_smem[tic_scales], 2 + warp_m);
            kittens::fp8e8m0_4 sb_h0 = lane_rd(scale_B_lo[tic_scales], warp_n);
            kittens::fp8e8m0_4 sb_h1 = lane_rd(scale_B_hi[tic_scales], warp_n);

            auto as0 = kittens::subtile_inplace<REG_M, K_STEP>(As[tic][0], {warp_m, 0});
            kittens::load(a, as0);
            __builtin_amdgcn_s_barrier();

            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(2);
            kittens::mma_ABt_scaled<CBSZ, BLGP>(cA, a, b0, cA, &sa_h0, &sb_h0);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            auto bs1 = kittens::subtile_inplace<REG_N, K_STEP>(Bs[tic][1], {warp_n, 0});
            kittens::load(b1, bs1);
            asm volatile("s_waitcnt vmcnt(0)");
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(2);
            kittens::mma_ABt_scaled<CBSZ, BLGP>(cB, a, b1, cB, &sa_h0, &sb_h1);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            auto as1 = kittens::subtile_inplace<REG_M, K_STEP>(As[tic][1], {warp_m, 0});
            kittens::load(a, as1);
            __builtin_amdgcn_s_barrier();

            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(2);
            kittens::mma_ABt_scaled<CBSZ, BLGP>(cC, a, b0, cC, &sa_h1, &sb_h0);
            kittens::mma_ABt_scaled<CBSZ, BLGP>(cD, a, b1, cD, &sa_h1, &sb_h1);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }

        // Closes the warp_m skew opened by the prologue so the next queue tile starts even.
        if (warp_m == 0) {
            __builtin_amdgcn_s_barrier();
        }

        gemm_epilogue<RT_C>(cA, cB, cC, cD, c_base, N_TOTAL, block_row, block_col, warp_m, warp_n);

    } // end persistent loop
}

static std::vector<TileDesc> build_work_queue(int M, int N_total, int K, int tp_size, int my_pe) {
    (void)K;
    const int tiles_N         = N_total / BLOCK_COL;
    const int m_local         = M / tp_size;
    const int tiles_per_chunk = m_local / BLOCK_ROW;

    std::vector<TileDesc> queue;
    queue.reserve((size_t)tiles_N * tiles_per_chunk * tp_size);

    const int N_GROUP = 16;

    auto emit = [&](int chunk, int tm, int n0, int n1) {
        for (int tn = n0; tn < n1; tn++) {
            TileDesc td{};
            td.chunk_id = chunk;
            td.tile_m   = chunk * tiles_per_chunk + tm;
            td.tile_n   = tn;
            queue.push_back(td);
        }
    };

    const int nstep = (N_GROUP > 0 && N_GROUP < tiles_N) ? N_GROUP : tiles_N;

    // Local chunk first, then remote
    for (int n0 = 0; n0 < tiles_N; n0 += nstep) {
        const int n1 = std::min(n0 + nstep, tiles_N);
        for (int tm = 0; tm < tiles_per_chunk; tm++) emit(my_pe, tm, n0, n1);
    }
    for (int n0 = 0; n0 < tiles_N; n0 += nstep) {
        const int n1 = std::min(n0 + nstep, tiles_N);
        for (int tm = 0; tm < tiles_per_chunk; tm++) {
            for (int c = 0; c < tp_size; c++) {
                if (c != my_pe) emit(c, tm, n0, n1);
            }
        }
    }

    return queue;
}

static void launch_persistent(int M, int N_TOTAL, int K, fp8e4m3 *d_a, fp8e4m3 *d_b, bf16 *d_c, uint32_t* packed_sa, uint32_t* packed_sb,
                              TileDesc *d_queue, int num_tiles, int *d_tile_counter, PeerPtrs peers, unsigned int *d_arrive,
                              int my_pe, int tp_size, int gath_wg, int m_local, size_t chunk_bytes, int xcd_bucket,
                              XcdBuckets buckets, int *d_bucket_ctr, hipStream_t stream) {
    const int tiles_M         = M / BLOCK_ROW;
    const int tiles_N         = N_TOTAL / BLOCK_COL;
    const int tiles_per_chunk = m_local / BLOCK_ROW;
    const int k_iters = K / K_STEP;

    gl<fp8e4m3, 1, 1, -1, -1> A_gl(d_a, nullptr, nullptr, (size_t)M,       (size_t)K);
    gl<fp8e4m3, 1, 1, -1, -1> B_gl(d_b, nullptr, nullptr, (size_t)N_TOTAL, (size_t)K);
    gl<bf16, 1, 1, -1, -1> C_gl(d_c, nullptr, nullptr, (size_t)M,       (size_t)N_TOTAL);

    gl<fp8e8m0, -1, 1, 16, 64> SA_gl(reinterpret_cast<kittens::fp8e8m0 *>(const_cast<uint32_t *>(packed_sa)),
                                     k_iters * tiles_M, nullptr, nullptr, nullptr);

    // B scale buffer is 2 tiles for the hi/lo group split
    gl<fp8e8m0, -1, 1, 16, 64> SB_gl(reinterpret_cast<kittens::fp8e8m0 *>(const_cast<uint32_t *>(packed_sb)),
                                     2 * k_iters * tiles_N, nullptr, nullptr, nullptr);

    // Gatherers are dedicated (blockIdx < NGATH) and only join the compute queue once their peer's chunk has landed.
    const int NGATH = (tp_size - 1) * gath_wg;
    static const int grid_cap = getenv("HK_GRID_CAP") ? atoi(getenv("HK_GRID_CAP")) : 256;
    int grid = tiles_M * tiles_N + NGATH;
    if (grid_cap > 0 && grid > grid_cap) grid = grid_cap;
    if (grid < NGATH) grid = NGATH;

    persistent_ag_mxfp8_gemm<<<grid, NUM_THREADS, 0, stream>>>(
        A_gl, B_gl, C_gl, SA_gl, SB_gl, d_queue, num_tiles, d_tile_counter, peers,
        d_arrive, my_pe, tp_size, gath_wg, tiles_per_chunk, chunk_bytes,
        xcd_bucket, buckets, d_bucket_ctr);
}

using persistent_fn_t = void (*)(int, int, int, fp8e4m3 *, fp8e4m3 *, bf16 *, uint32_t *, uint32_t *, TileDesc *, int, int *, PeerPtrs,
                                 unsigned int *, int, int, int, int, size_t, int,
                                 XcdBuckets, int *, hipStream_t);

static persistent_fn_t get_persistent_fn(int M, int N, int K) {
    (void)M; (void)N; (void)K;
    return launch_persistent;
}

} // namespace hk_mxfp8_ag_tn
