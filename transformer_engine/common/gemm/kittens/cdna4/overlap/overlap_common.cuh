/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/
#pragma once

#include "hip/hip_runtime.h"
#include "kittens.cuh"
#include <cstddef>
#include <vector>

// Pieces every fused comm+GEMM overlap kernel shares. Each kernel keeps its own namespace
// and pulls these in with `using namespace hk_overlap;`, so unqualified uses in the kernel
// bodies keep resolving exactly as they did when each file carried its own copy.
namespace hk_overlap {

constexpr int SCHED_ROUNDS = 2;
constexpr int NUM_WARPS    = 8;
constexpr int WARPS_ROW    = 2;
constexpr int WARPS_COL    = 4;
constexpr int BLOCK_ROW    = 256;
constexpr int BLOCK_COL    = 256;
constexpr int HALF_ROW     = BLOCK_ROW / 2;
constexpr int HALF_COL     = BLOCK_COL / 2;
constexpr int REG_M        = BLOCK_ROW / WARPS_ROW / 2;
constexpr int REG_N        = BLOCK_COL / WARPS_COL / 2;
constexpr int NUM_THREADS  = NUM_WARPS * kittens::WARP_THREADS;

using G_group = kittens::group<NUM_WARPS>;

// K_STEP is deliberately absent: the bf16 kernels step 64 and the mxfp8 kernels step 128,
// so it stays with each kernel rather than pretending to be shared.

// TileDesc is deliberately NOT here: the bf16 NN kernel's descriptor carries an extra split-K
// index (`ks`) that the other three kernels have no use for. Each kernel declares its own, which
// is why bucketize_by_xcd below is templated on the descriptor type.

// Per-PE pointer to each peer's operand buffer. The element type is what differs between
// the bf16 and mxfp8 kernels, so it is the template parameter.
template <typename T>
struct PeerPtrsT {
    T *base[8];
};

#ifndef GATH_WG
#define GATH_WG 8
#endif

constexpr int NUM_XCDS_AFF = 8;

struct XcdBuckets {
    int off[NUM_XCDS_AFF];
    int cnt[NUM_XCDS_AFF];
};

#ifndef AG_PUBLISH
#define AG_PUBLISH(p) __hip_atomic_fetch_add((p), 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT)
#define AG_SPIN(p) __hip_atomic_load((p), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)
#define AG_ACQUIRE(p) ((void)__hip_atomic_load((p), __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT))
#endif

__host__ __device__ __forceinline__
int tile_xcd(int chunk_id) { return chunk_id & (NUM_XCDS_AFF - 1); }

// Stable-partition the queue into one contiguous segment per XCD.
template <typename TD>
static std::vector<TD> bucketize_by_xcd(const std::vector<TD> &q, XcdBuckets &bk) {
    std::vector<TD> out;
    out.reserve(q.size());
    for (int b = 0; b < NUM_XCDS_AFF; b++) {
        bk.off[b] = (int)out.size();
        for (size_t i = 0; i < q.size(); i++) {
            if (tile_xcd(q[i].chunk_id) == b) out.push_back(q[i]);
        }
        bk.cnt[b] = (int)out.size() - bk.off[b];
    }
    return out;
}

// NT selects nontemporal stores for the gathered shard for performance.
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

// ---------------------------------------------------------------------------------------
// Interleaved scale gather+pack, shared by the MXFP8 TN and NN fused AG+GEMM kernels.
//
// These live here rather than in one kernel header because the host side (comm_gemm.cpp,
// run_mxfp8) hands BOTH layouts the same contract when interleaving is on: it skips
// gather_scales and packs only the local rank, on the promise that the gatherer blocks pack
// the peers' rows. A layout that has the flag but not the implementation silently produces
// garbage -- which is exactly what NN did before these were shared.
//
// The activation scales are row-wise in both layouts and packed_sa has the same layout in
// both (comm_gemm.cpp uses the same launch_pack_scales<false,...> / pack_local_scales_kernel
// for TN and NN), so this code is genuinely layout-independent.
// ---------------------------------------------------------------------------------------
// Packs one 256-row tile of A scales from a peer's buffer into the lane-native layout the GEMM
// reads. Same transform as pack_scales_kernel (mxfp8_gemm.cpp) for COLWISE=false; called by the
// gatherer block that is already fetching that tile, so the raw scales never reach global memory.
// A gather tile is BLOCK_ROW=256 rows, which is also pack_scales_kernel's TILE_WORDS, so the
// tile <-> cblk mapping is 1:1.
//
// Each block handles the contiguous k-range [sub*kpb, +kpb) and writes tile_id = ki*tiles_per_col
// + cblk, so blocks never touch the same output.
// Packs one 256-row tile of the gathered operand's scales into the lane-native layout the GEMM
// reads, straight from the peer's buffer so the raw scales never reach global memory.
//
// <STEP, NG> selects the operand geometry and must match the pack_scales_kernel<_, STEP, NG>
// instantiation mxfp8_gemm.cpp uses for that operand, because both write the same buffer layout:
//     ln[(tile_id * NG + grp) * 64 + lane] = pack_scales(tile, grp * STEP)
// A side is <64, 4> (256 output words/tile), B side is <32, 8> (512, the hi/lo tile pair).
//
// Per k-iter the tile's scales are 256 rows x 4 bytes (BLOCK_K=128 elements / 32 per scale block),
// i.e. one dword per row. NG waves each emit 64 dwords, one per lane, reading a STEP-strided
// window of the staged tile. KPP = NUM_WARPS / NG k-iters run at once so all 8 waves stay busy:
// two for <64,4>, one for <32,8>. pack_scales reads 64 words past the last window start, so the
// staging stride is PAD = (NG-1)*STEP + 64 words with everything at/after row 256 zero-filled --
// for <64,4> that is exactly 256, i.e. unchanged.
//
// Blocks split the k range: block `sub` takes a contiguous [k0,k1) out of gath_wg. Contiguous
// rather than strided so each block's reads stay in one span per row. Blocks write disjoint
// tile_ids, so the only sync is around the staging buffer.
template <int STEP = 64, int NG = 4>
__device__ __forceinline__
void pack_tile_scales_from(const uint8_t *__restrict__ src_rows, uint32_t *__restrict__ ln,
                           int cblk, int tiles_per_col, int k_iters, int scale_K,
                           uint32_t *__restrict__ smem_tile, int sub, int gath_wg) {
    constexpr int ROWS     = 256;                  // rows per tile == BLOCK_ROW
    constexpr int PER_WAVE = STEP;                 // window stride one wave packs from
    constexpr int WAVES    = NG;                   // waves per k-iter: 4 (A side) or 8 (B side)
    constexpr int KPP      = NUM_WARPS / WAVES;    // k-iters in flight: 8/4 = 2, or 8/8 = 1
    constexpr int PAD      = (NG - 1) * STEP + 64; // staging stride; covers pack_scales' OOB read
    static_assert(NUM_WARPS % WAVES == 0, "NG must divide NUM_WARPS");
    static_assert(KPP >= 1, "NG cannot exceed NUM_WARPS");

    const int lane  = (int)threadIdx.x % 64;
    const int wave  = (int)threadIdx.x / 64;
    const int kslot = wave / WAVES;            // which of the KPP k-iters
    const int wgrp  = wave % WAVES;            // which 64-row chunk

    const int kpb = (k_iters + gath_wg - 1) / gath_wg;  // k-iters per block
    const int k0  = sub * kpb;
    const int k1  = (k0 + kpb > k_iters) ? k_iters : k0 + kpb;
    if (k0 >= k_iters) return;

    // Staging is KPP slots of PAD words. <64,4> lands exactly on KPP*PAD == NUM_THREADS, so it
    // keeps the original straight-line stage -- one dword per thread, no loop, no bounds test.
    // That branch is kept verbatim so templating this function is a provable no-op for the A
    // side; only a geometry that does not tile the block exactly pays for the general path.
    constexpr bool EXACT = (KPP * PAD == NUM_THREADS);
    static_assert(KPP * PAD <= 2 * 256, "staging exceeds the caller's scale_A_smem budget");

    const int slot = (int)threadIdx.x / ROWS;   // used by the EXACT path only
    const int srow = (int)threadIdx.x % ROWS;

    for (int kb = k0; kb < k1; kb += KPP) {
        if constexpr (EXACT) {
            uint32_t p = 0;
            if (kb + slot < k1) {
                __builtin_memcpy(&p, &src_rows[(size_t)srow * scale_K + (kb + slot) * 4], 4);
            }
            smem_tile[threadIdx.x] = p;
        } else {
            for (int i = (int)threadIdx.x; i < KPP * PAD; i += NUM_THREADS) {
                const int wslot = i / PAD;  // which of the KPP k-iters this word belongs to
                const int wrow  = i % PAD;  // which row; >= ROWS is the zero-filled OOB tail
                uint32_t p = 0;
                if (wrow < ROWS && kb + wslot < k1) {
                    __builtin_memcpy(&p, &src_rows[(size_t)wrow * scale_K + (kb + wslot) * 4], 4);
                }
                smem_tile[i] = p;
            }
        }
        __syncthreads();
        if (kslot < KPP && kb + kslot < k1) {
            const size_t tile_id = (size_t)(kb + kslot) * tiles_per_col + cblk;
            ln[(tile_id * WAVES + wgrp) * 64 + lane] = kittens::pack_scales(
                (const kittens::fp8e8m0 *)(smem_tile + kslot * PAD), wgrp * PER_WAVE);
        }
        __syncthreads();
    }
}

// STEP/NG select the packed-scale geometry, forwarded to pack_tile_scales_from: <64,4> for an
// operand with 4 scale groups per tile, <32,8> for one with 8.
template <int U, bool NT, int STEP = 64, int NG = 4>
__device__ __forceinline__
void gather_peer_tile_plus_scales(int peer, int tn, int sub, int gath_wg, int tiles_per_chunk,
                                  char *gather_dst, const PeerPtrsT<kittens::fp8e4m3> &peers, size_t chunk_bytes,
                                  unsigned int *arrive, size_t scale_base, size_t scale_chunk_bytes,
                                  int scale_K, uint32_t *packed_sa, int tiles_per_col,
                                  int k_iters, uint32_t *smem_tile) {
    const size_t tile_bytes = chunk_bytes / tiles_per_chunk;
    const size_t doff       = (size_t)peer * chunk_bytes + (size_t)tn * tile_bytes;

    size_t sub_bytes = (((tile_bytes + gath_wg - 1) / gath_wg) + 15) & ~size_t(15);
    size_t o         = (size_t)sub * sub_bytes;
    size_t l         = (o >= tile_bytes) ? 0 : ((o + sub_bytes <= tile_bytes) ? sub_bytes : tile_bytes - o);

    if (l) gather_copy_wg<U, NT>(gather_dst + doff + o, (const char *)peers.base[peer] + doff + o, l);

    __syncthreads();

    // Scales for this tile, read straight from the peer. Same ordering guarantee as the data copy
    // above: ag_ready_kernel has already established that peers finished writing their chunks.
    //
    // Split by k-iteration, one contiguous range per block. tile_id = ki*tiles_per_col + cblk, so
    // blocks write disjoint packed_sa entries and need no coordination. Trip counts differ across
    // blocks when gath_wg does not divide k_iters; that is fine, __syncthreads() is per block and
    // every thread within a block runs the same count.
    {
        const char *peer_scales = (const char *)peers.base[peer] + scale_base
                                + (size_t)peer * scale_chunk_bytes
                                + (size_t)tn * BLOCK_ROW * (size_t)scale_K;
        pack_tile_scales_from<STEP, NG>((const uint8_t *)peer_scales, packed_sa,
                                        peer * tiles_per_chunk + tn, tiles_per_col,
                                        k_iters, scale_K, smem_tile, sub, gath_wg);
    }
    __syncthreads();

    // Fence AFTER the pack so it covers the packed_sa stores as well as the data copy: a consumer
    // that sees the arrival count reach gath_wg must see both.
    if (NT) {
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    } else {
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    }
    __syncthreads();
    if (threadIdx.x == 0) AG_PUBLISH(&arrive[peer * tiles_per_chunk + tn]);
    __syncthreads();
}

template <int U, bool NT, int STEP = 64, int NG = 4>
__device__ __forceinline__
void gather_all_plus_scales(int my_pe, int gath_wg, int tiles_per_chunk, char *gb,
                            const PeerPtrsT<kittens::fp8e4m3> &peers, size_t chunk_bytes, unsigned int *arrive,
                            size_t scale_base, size_t scale_chunk_bytes, int scale_K,
                            uint32_t *packed_sa, int tiles_per_col, int k_iters,
                            uint32_t *smem_tile) {
    const int pi   = (int)blockIdx.x / gath_wg;
    const int sub  = (int)blockIdx.x % gath_wg;
    const int peer = pi + (pi >= my_pe ? 1 : 0);
    for (int tn = 0; tn < tiles_per_chunk; tn++) {
        gather_peer_tile_plus_scales<U, NT, STEP, NG>(
            peer, tn, sub, gath_wg, tiles_per_chunk, gb, peers, chunk_bytes, arrive,
            scale_base, scale_chunk_bytes, scale_K, packed_sa, tiles_per_col, k_iters, smem_tile);
    }
}

}  // namespace hk_overlap
