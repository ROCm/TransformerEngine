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

namespace hk_ag_tn {

using namespace hk_overlap;

struct TileDesc {
    int chunk_id;
    int tile_m;
    int tile_n;
};

using namespace kittens;

constexpr int SCHED_ROUNDS = 2;


// Per-PE pointer to each peer's [M,K] A buffer.
struct PeerPtrs {
    bf16 *base[8];
};





// NT selects nontemporal stores for the gathered shard for performance.
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

template <int U, bool NT>
__device__ __forceinline__
void gather_copy_wg(void *__restrict__ dst, const void *__restrict__ src, size_t nbytes) {
    typedef int v4i __attribute__((ext_vector_type(4)));
    v4i       *d4 = (v4i *)dst;
    const v4i *s4 = (const v4i *)src;
    size_t n4 = nbytes / sizeof(v4i);
    const size_t stride = blockDim.x;
    const size_t step   = stride * U;
    size_t i = threadIdx.x;

    if (U > 1) {
        for (; i + (size_t)(U - 1) * stride < n4; i += step) {
            v4i v[U];
#pragma unroll
            for (int u = 0; u < U; u++) v[u] = s4[i + (size_t)u * stride];
#pragma unroll
            for (int u = 0; u < U; u++) {
                if (NT) __builtin_nontemporal_store(v[u], &d4[i + (size_t)u * stride]);
                else    d4[i + (size_t)u * stride] = v[u];
            }
        }
    }
    for (; i < n4; i += stride) {
        if (NT) __builtin_nontemporal_store(s4[i], &d4[i]);
        else    d4[i] = s4[i];
    }

    size_t done = n4 * sizeof(v4i);
    if (threadIdx.x == 0) {
        for (size_t j = done; j < nbytes; j++) ((char *)dst)[j] = ((const char *)src)[j];
    }
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

__global__ __launch_bounds__(NUM_THREADS, 2)
void persistent_ag_bf16_gemm(const gl<bf16, 1, 1, -1, -1> A, const gl<bf16, 1, 1, -1, -1> B,
                             const gl<bf16, 1, 1, -1, -1> C, const TileDesc *__restrict__ work_queue, int num_tiles,
                             int *__restrict__ tile_counter, const PeerPtrs peers, unsigned int *__restrict__ arrive,
                             int my_pe, int tp_size, int gath_wg, int tiles_per_chunk, size_t chunk_bytes,
                             int xcd_bucket, const XcdBuckets buckets, int *__restrict__ bucket_ctr) {
    const int M       = A.rows();
    const int K       = A.cols();
    const int N_TOTAL = B.rows();
    const int k_tiles = K / K_STEP;

#include "tn_prologue.inc"
    bf16 *c_base           = (bf16 *)&C[{0, 0, 0, 0}];

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
#include "xcd_steal.inc"
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

        RT_A a_tile;
        RT_B b_tile_0, b_tile_1;
        RT_C c00, c01, c10, c11;
        zero(c00); zero(c01); zero(c10); zero(c11);

        int tic = 0, toc = 1;

        // Prologue: K-tile 0 into stage 0, K-tile 1 into stage 1 (As[1][1] is filled by the first main-loop body).
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
#include "tn_mainloop.inc"

        const int rf0 = __builtin_amdgcn_readfirstlane(block_row * WARPS_ROW * 2 + warp_m);
        const int rf1 = __builtin_amdgcn_readfirstlane(block_row * WARPS_ROW * 2 + WARPS_ROW + warp_m);
        const int cf0 = __builtin_amdgcn_readfirstlane(block_col * WARPS_COL * 2 + warp_n);
        const int cf1 = __builtin_amdgcn_readfirstlane(block_col * WARPS_COL * 2 + WARPS_COL + warp_n);
        int lane_epi  = kittens::laneid();
        asm volatile("" : "+v"(lane_epi));

        store_c_tile<bf16>(c_base, c00, rf0, cf0, N_TOTAL, lane_epi);
        store_c_tile<bf16>(c_base, c01, rf0, cf1, N_TOTAL, lane_epi);
        store_c_tile<bf16>(c_base, c10, rf1, cf0, N_TOTAL, lane_epi);
        store_c_tile<bf16>(c_base, c11, rf1, cf1, N_TOTAL, lane_epi);

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

static void launch_persistent(int M, int N_TOTAL, int K, bf16 *d_a, bf16 *d_b, bf16 *d_c, TileDesc *d_queue,
                              int num_tiles, int *d_tile_counter, PeerPtrs peers, unsigned int *d_arrive, int my_pe,
                              int tp_size, int gath_wg, int m_local, size_t chunk_bytes, int xcd_bucket,
                              XcdBuckets buckets, int *d_bucket_ctr, hipStream_t stream) {
    const int tiles_M         = M / BLOCK_ROW;
    const int tiles_N         = N_TOTAL / BLOCK_COL;
    const int tiles_per_chunk = m_local / BLOCK_ROW;

    gl<bf16, 1, 1, -1, -1> A_gl(d_a, nullptr, nullptr, (size_t)M,       (size_t)K);
    gl<bf16, 1, 1, -1, -1> B_gl(d_b, nullptr, nullptr, (size_t)N_TOTAL, (size_t)K);
    gl<bf16, 1, 1, -1, -1> C_gl(d_c, nullptr, nullptr, (size_t)M,       (size_t)N_TOTAL);

    // Gatherers are dedicated (blockIdx < NGATH) and only join the compute queue once their peer's chunk has landed.
    const int NGATH = (tp_size - 1) * gath_wg;
    int grid = tiles_M * tiles_N + NGATH;
    if (grid > GRID_CAP) grid = GRID_CAP;
    if (grid < NGATH) grid = NGATH;

    persistent_ag_bf16_gemm<<<grid, NUM_THREADS, 0, stream>>>(
        A_gl, B_gl, C_gl, d_queue, num_tiles, d_tile_counter, peers,
        d_arrive, my_pe, tp_size, gath_wg, tiles_per_chunk, chunk_bytes,
        xcd_bucket, buckets, d_bucket_ctr);
}

using persistent_fn_t = void (*)(int, int, int, bf16 *, bf16 *, bf16 *, TileDesc *, int, int *, PeerPtrs,
                                 unsigned int *, int, int, int, int, size_t, int,
                                 XcdBuckets, int *, hipStream_t);

static persistent_fn_t get_persistent_fn(int M, int N, int K) {
    (void)M; (void)N; (void)K;
    return launch_persistent;
}

}  // namespace hk_ag_tn
