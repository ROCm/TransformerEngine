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

namespace hk_mxfp8_ag_nn {

using namespace kittens;
using namespace hk_overlap;

// Per-PE pointer to each peer's [N,K] B buffer (the gathered activation).
using PeerPtrs = hk_overlap::PeerPtrsT<fp8e4m3>;

using G = kittens::group<NUM_WARPS>;

struct TileDesc {
    int chunk_id;
    int tile_m;
    int tile_n;
};

constexpr int BLOCK_K = 128;

// MFMA cbsz/blgp format codes: 0 = e4m3, 1 = e5m2, chosen per operand. HYBRID recipes quantize
// backward tensors e5m2 while forward ones stay e4m3, so dgrad/wgrad legitimately mix; the kernel
// is templated on them and get_persistent_fn() dispatches, as dispatch_gemm() does in
// mxfp8_gemm.cpp. CBSZ is the A operand (weights), BLGP the B operand (gathered activations).

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

// Epilogue for the AG path: C is [N_TOTAL, M], the same orientation mxfp8_gemm.cpp writes, so the
// col_l accumulators are transposed before the store. No bias/gelu/accumulate on this path.
template<typename RT_C, typename RT_C_T, typename OutGL>
__device__ __forceinline__ void gemm_epilogue(
    RT_C &cA, RT_C &cB, RT_C &cC, RT_C &cD,
    const OutGL &C, int block_row, int block_col, int warp_m, int warp_n) {

    auto oc = [&](int n_off, int m_off) {
        return kittens::coord<RT_C_T>{0, 0, block_col * WARPS_COL * 2 + n_off,
                                            block_row * WARPS_ROW * 2 + m_off};
    };

    RT_C_T oA, oB, oC, oD;
    kittens::transpose(oA, cA); kittens::transpose(oB, cB);
    kittens::transpose(oC, cC); kittens::transpose(oD, cD);

    kittens::store(C, oA, oc(warp_n, warp_m));
    kittens::store(C, oB, oc(WARPS_COL + warp_n, warp_m));
    kittens::store(C, oC, oc(warp_n, WARPS_ROW + warp_m));
    kittens::store(C, oD, oc(WARPS_COL + warp_n, WARPS_ROW + warp_m));
}

// BULK=true detaches the all-gather from the GEMM: the gather lands in `gather_dst` instead of
// the A operand, and the compute never waits on arrivals because it does not read the gathered
// bytes. The gathered tensor is untyped here -- gather_copy_wg moves bytes and nothing in this
// kernel interprets them, so its dtype and scales are the caller's business.
template <int CBSZ, int BLGP, bool BULK>
__global__ __launch_bounds__(NUM_THREADS, 2)
void persistent_ag_mxfp8_gemm(const gl<fp8e4m3, 1, 1, -1, -1> A, const gl<fp8e4m3, 1, 1, -1, -1> B,
    const gl<bf16, 1, 1, -1, -1> C, const gl<fp8e8m0, -1, 1, 16, 64> scale_A_gl,
    const gl<fp8e8m0, -1, 1, 16, 64> scale_B_gl, const TileDesc *__restrict__ work_queue,
    int num_tiles, int *__restrict__ tile_counter, const PeerPtrs peers, unsigned int *__restrict__ arrive,
    int my_pe, int tp_size, int gath_wg, int tiles_per_chunk, size_t chunk_bytes,
    int xcd_bucket, const XcdBuckets buckets, int *__restrict__ bucket_ctr,
    // Used by the interleaved gather+pack below when BULK is false. The attribute is for the
    // BULK=true instantiation, which discards that branch -- it does NOT mean "ignored".
    [[maybe_unused]] uint32_t *__restrict__ packed_sb_raw, [[maybe_unused]] size_t scale_base,
    [[maybe_unused]] size_t scale_chunk_bytes, [[maybe_unused]] int scale_K,
    [[maybe_unused]] int interleave_scales,
    [[maybe_unused]] char *__restrict__ gather_dst) {

    // NN (BLAS convention): A is the weight [K, M], B the gathered activation [N_TOTAL, K].
    const int K       = A.rows();
    const int M       = A.cols();
    const int N_TOTAL = B.rows();
    const int k_iters = K / BLOCK_K;
#include "mxfp8_nn_prologue.inc"

    const int NGATH = (tp_size - 1) * gath_wg;
    if ((int)blockIdx.x < NGATH) {
        // In bulk mode chunk_bytes describes the gathered region's shard, not the B operand's.
        char *gb = (char *)&B[{0, 0, 0, 0}];
        if constexpr (BULK) gb = gather_dst;
        if constexpr (BULK) {
            // The gathered tensor is not a GEMM operand here, so its scales are moved verbatim by
            // gather_scales and there is nothing to pack. run_bulk_mxfp8 never enables interleave;
            // this guard keeps the kernel correct if that ever changes.
            gather_all<1, true>(my_pe, gath_wg, tiles_per_chunk, gb, peers, chunk_bytes, arrive);
        } else if (interleave_scales) {
            // scale_A_smem is untouched until this block joins the compute queue, which it cannot
            // do before the gather below finishes -- so stage the pack there rather than growing
            // the kernel's LDS footprint. Same reasoning, same staging size as the TN path.
            static_assert(sizeof(scale_A_smem) >= 2 * 256 * sizeof(uint32_t),
                          "scale_A_smem too small for pack_tile_scales_from staging");
            gather_all_plus_scales<1, true, 32, 8>(
                my_pe, gath_wg, tiles_per_chunk, gb, peers, chunk_bytes, arrive,
                scale_base, scale_chunk_bytes, scale_K, packed_sb_raw, tiles_N, k_iters,
                reinterpret_cast<uint32_t *>(&scale_A_smem[0]));
        } else {
            gather_all<1, true>(my_pe, gath_wg, tiles_per_chunk, gb, peers, chunk_bytes, arrive);
        }
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
        // In bulk mode this GEMM does not read the gathered tensor, so there is nothing to wait for.
        if constexpr (!BULK) {
            if (desc.chunk_id != my_pe) {
                const int tn                   = desc.tile_n - desc.chunk_id * tiles_per_chunk;
                const unsigned needed_arrivals = (unsigned)gath_wg;
                unsigned int *f                = &arrive[(size_t)desc.chunk_id * tiles_per_chunk + tn];
                if (threadIdx.x == 0) {
                    do {
                    } while (AG_SPIN(f) < needed_arrivals);
                    AG_ACQUIRE(f);
                }
                __syncthreads();
            }
        }

        int block_row = desc.tile_m;
        int block_col = desc.tile_n;

        // The shared main loop uses mxfp8_gemm_nn_kernel's spelling for the tile halves, so provide
        // the names it indexes with rather than renaming the gold kernel, whose ISA is the gate.
        int a_row_tile = block_row;
        int n_tile     = block_col;

        int sa_stride = tiles_M;
        int sb_stride = tiles_N;
        int sa_batch = block_row;
        int sb_batch = block_col;

        RT_A a;
        RT_B b0, b1;
        RT_C cA, cB, cC, cD;
        kittens::zero(cA); kittens::zero(cB); kittens::zero(cC); kittens::zero(cD);

        int tic = 0, toc = 1;
        int tic_scales = 0, toc_scales = 1;

        G::load(Bs[tic][0], B_local, {0, 0, n_tile * 2,     0}, sw_B, b_srd, b_base, b_lds[tic][0]);
        G::load(As[tic][0], A_local, {0, 0, 0, a_row_tile * 2    }, sw_A, a_srd, a_base, a_lds[tic][0]);
        G::load(Bs[tic][1], B_local, {0, 0, n_tile * 2 + 1,  0}, sw_B, b_srd, b_base, b_lds[tic][1]);
        G::load(As[tic][1], A_local, {0, 0, 0, a_row_tile * 2 + 1}, sw_A, a_srd, a_base, a_lds[tic][1]);

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();

        G::load(As[toc][0], A_local, {0, 0, 1, a_row_tile * 2    }, sw_A, a_srd, a_base, a_lds[toc][0]);
        G::load(Bs[toc][0], B_local, {0, 0, n_tile * 2,     1}, sw_B, b_srd, b_base, b_lds[toc][0]);
        G::load(Bs[toc][1], B_local, {0, 0, n_tile * 2 + 1, 1}, sw_B, b_srd, b_base, b_lds[toc][1]);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();

        G::load(scale_A_smem[0], scale_A_gl, {block_row, 0, 0, 0});
        G::load(scale_B_lo[0],   scale_B_gl, {2 * block_col,     0, 0, 0});
        G::load(scale_B_hi[0],   scale_B_gl, {2 * block_col + 1, 0, 0, 0});
        asm volatile("s_waitcnt vmcnt(0)");
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();

        if (warp_m == 1) __builtin_amdgcn_s_barrier();

        // The AG path opens a warp_m skew in its prologue; the shared main loop closes it under
        // this guard. The standalone GEMM includes the same file without defining it.
        #define MXFP8_AG_SKEW_CLOSE 1
        #include "../mxfp8_nn_mainloop.inc"
        #undef MXFP8_AG_SKEW_CLOSE

        gemm_epilogue<RT_C, RT_C_T>(cA, cB, cC, cD, C, block_row, block_col, warp_m, warp_n);

    } // end persistent loop
}

static std::vector<TileDesc> build_work_queue(int M, int N_total, int K, int tp_size, int my_pe) {
    (void)K;
    const int tiles_M         = M / BLOCK_ROW;
    const int n_local         = N_total / tp_size;
    const int tiles_per_chunk = n_local / BLOCK_COL;

    std::vector<TileDesc> queue;
    queue.reserve((size_t)tiles_M * tiles_per_chunk * tp_size);

    // Grouping runs over the free axis, which is M now that the gather chunks N.
    const int M_GROUP = 16;

    auto emit = [&](int chunk, int tn, int m0, int m1) {
        for (int tm = m0; tm < m1; tm++) {
            TileDesc td{};
            td.chunk_id = chunk;
            td.tile_n   = chunk * tiles_per_chunk + tn;
            td.tile_m   = tm;
            queue.push_back(td);
        }
    };

    const int mstep = (M_GROUP > 0 && M_GROUP < tiles_M) ? M_GROUP : tiles_M;

    // Local chunk first, then remote
    for (int m0 = 0; m0 < tiles_M; m0 += mstep) {
        const int m1 = std::min(m0 + mstep, tiles_M);
        for (int tn = 0; tn < tiles_per_chunk; tn++) emit(my_pe, tn, m0, m1);
    }
    for (int m0 = 0; m0 < tiles_M; m0 += mstep) {
        const int m1 = std::min(m0 + mstep, tiles_M);
        for (int tn = 0; tn < tiles_per_chunk; tn++) {
            for (int c = 0; c < tp_size; c++) {
                if (c != my_pe) emit(c, tn, m0, m1);
            }
        }
    }

    return queue;
}

template <int CBSZ, int BLGP, bool BULK>
static void launch_persistent_impl(int M, int N_TOTAL, int K, fp8e4m3 *d_a, fp8e4m3 *d_b, bf16 *d_c, uint32_t* packed_sa, uint32_t* packed_sb,
                              TileDesc *d_queue, int num_tiles, int *d_tile_counter, PeerPtrs peers, unsigned int *d_arrive,
                              int my_pe, int tp_size, int gath_wg, int n_local, size_t chunk_bytes, int xcd_bucket,
                              XcdBuckets buckets, int *d_bucket_ctr, size_t scale_base,
                              size_t scale_chunk_bytes, int interleave_scales, char *d_gather_dst,
                              hipStream_t stream) {
    const int tiles_M         = M / BLOCK_ROW;
    const int tiles_N         = N_TOTAL / BLOCK_COL;
    const int tiles_per_chunk = n_local / BLOCK_COL;
    const int k_iters = K / BLOCK_K;

    // NN: A is the weight [K, M]; B the gathered activation [N_TOTAL, K]; C is [N_TOTAL, M].
    gl<fp8e4m3, 1, 1, -1, -1> A_gl(d_a, nullptr, nullptr, (size_t)K,       (size_t)M);
    gl<fp8e4m3, 1, 1, -1, -1> B_gl(d_b, nullptr, nullptr, (size_t)N_TOTAL, (size_t)K);
    gl<bf16, 1, 1, -1, -1> C_gl(d_c, nullptr, nullptr, (size_t)N_TOTAL, (size_t)M);

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

    persistent_ag_mxfp8_gemm<CBSZ, BLGP, BULK><<<grid, NUM_THREADS, 0, stream>>>(
        A_gl, B_gl, C_gl, SA_gl, SB_gl, d_queue, num_tiles, d_tile_counter, peers,
        d_arrive, my_pe, tp_size, gath_wg, tiles_per_chunk, chunk_bytes,
        xcd_bucket, buckets, d_bucket_ctr,
        packed_sb, scale_base, scale_chunk_bytes, K / 32, interleave_scales, d_gather_dst);
}

// Fused: gather feeds the B operand, so there is no separate destination.
template <int CBSZ, int BLGP>
static void launch_persistent(int M, int N_TOTAL, int K, fp8e4m3 *d_a, fp8e4m3 *d_b, bf16 *d_c,
                              uint32_t *packed_sa, uint32_t *packed_sb, TileDesc *d_queue,
                              int num_tiles, int *d_tile_counter, PeerPtrs peers,
                              unsigned int *d_arrive, int my_pe, int tp_size, int gath_wg,
                              int n_local, size_t chunk_bytes, int xcd_bucket, XcdBuckets buckets,
                              int *d_bucket_ctr, size_t scale_base, size_t scale_chunk_bytes,
                              int interleave_scales, hipStream_t stream) {
    launch_persistent_impl<CBSZ, BLGP, false>(
        M, N_TOTAL, K, d_a, d_b, d_c, packed_sa, packed_sb, d_queue, num_tiles, d_tile_counter,
        peers, d_arrive, my_pe, tp_size, gath_wg, n_local, chunk_bytes, xcd_bucket, buckets,
        d_bucket_ctr, scale_base, scale_chunk_bytes, interleave_scales, nullptr, stream);
}

// Bulk: the all-gather is unrelated to this GEMM and lands in d_gather_dst.
template <int CBSZ, int BLGP>
static void launch_persistent_bulk(int M, int N_TOTAL, int K, fp8e4m3 *d_a, fp8e4m3 *d_b, bf16 *d_c,
                                   uint32_t *packed_sa, uint32_t *packed_sb, TileDesc *d_queue,
                                   int num_tiles, int *d_tile_counter, PeerPtrs peers,
                                   char *d_gather_dst, unsigned int *d_arrive, int my_pe,
                                   int tp_size, int gath_wg, int n_local, size_t chunk_bytes,
                                   int xcd_bucket, XcdBuckets buckets, int *d_bucket_ctr,
                                   hipStream_t stream) {
    launch_persistent_impl<CBSZ, BLGP, true>(
        M, N_TOTAL, K, d_a, d_b, d_c, packed_sa, packed_sb, d_queue, num_tiles, d_tile_counter,
        peers, d_arrive, my_pe, tp_size, gath_wg, n_local, chunk_bytes, xcd_bucket, buckets,
        d_bucket_ctr, 0, 0, 0, d_gather_dst, stream);
}

using persistent_fn_t = void (*)(int, int, int, fp8e4m3 *, fp8e4m3 *, bf16 *, uint32_t *, uint32_t *, TileDesc *, int, int *, PeerPtrs,
                                 unsigned int *, int, int, int, int, size_t, int,
                                 XcdBuckets, int *, size_t, size_t, int, hipStream_t);

using persistent_bulk_fn_t = void (*)(int, int, int, fp8e4m3 *, fp8e4m3 *, bf16 *, uint32_t *,
                                      uint32_t *, TileDesc *, int, int *, PeerPtrs, char *,
                                      unsigned int *, int, int, int, int, size_t, int, XcdBuckets,
                                      int *, hipStream_t);

static persistent_bulk_fn_t get_persistent_bulk_fn(int M, int N, int K, int a_fp8_code,
                                                   int b_fp8_code) {
    (void)M; (void)N; (void)K;
    if (a_fp8_code == 0 && b_fp8_code == 0) return launch_persistent_bulk<0, 0>;
    if (a_fp8_code == 0 && b_fp8_code == 1) return launch_persistent_bulk<0, 1>;
    if (a_fp8_code == 1 && b_fp8_code == 0) return launch_persistent_bulk<1, 0>;
    if (a_fp8_code == 1 && b_fp8_code == 1) return launch_persistent_bulk<1, 1>;
    return nullptr;
}

// 4-way dispatch on the operand formats, mirroring dispatch_gemm() in mxfp8_gemm.cpp.
static persistent_fn_t get_persistent_fn(int M, int N, int K, int a_fp8_code, int b_fp8_code) {
    (void)M; (void)N; (void)K;
    if (a_fp8_code == 0 && b_fp8_code == 0) return launch_persistent<0, 0>;
    if (a_fp8_code == 0 && b_fp8_code == 1) return launch_persistent<0, 1>;
    if (a_fp8_code == 1 && b_fp8_code == 0) return launch_persistent<1, 0>;
    if (a_fp8_code == 1 && b_fp8_code == 1) return launch_persistent<1, 1>;
    return nullptr;
}

} // namespace hk_mxfp8_ag_nn
