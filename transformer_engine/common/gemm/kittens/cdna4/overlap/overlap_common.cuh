/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/
#pragma once

#include "hip/hip_runtime.h"
#include "kittens.cuh"
#include <hip/hip_bfloat16.h>
#include <cstddef>
#include <cstdio>
#include <vector>

// Pieces shared by the fused comm+GEMM overlap kernels; each pulls them into its own namespace
// with `using namespace hk_overlap;`.
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
constexpr int K_STEP       = 64;
constexpr int NUM_THREADS  = NUM_WARPS * kittens::WARP_THREADS;

// bf16 kernels only; the MXFP8 pair reads HK_GRID_CAP at launch instead.
constexpr int GRID_CAP     = 256;

using G_group = kittens::group<NUM_WARPS>;

// K_STEP above is the bf16 step; the mxfp8 kernels shadow it with 128 in their own namespace.

// TileDesc is deliberately not shared: the bf16 NN descriptor carries an extra split-K index that
// the other three have no use for, which is why bucketize_by_xcd is templated on the type.

// Per-PE pointer to each peer's operand buffer; the element type differs per kernel.
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

__device__ __forceinline__
__hip_bfloat16 bf16_add(__hip_bfloat16 x, __hip_bfloat16 y) {
    return __float2bfloat16(__bfloat162float(x) + __bfloat162float(y));
}

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
                if constexpr (NT) __builtin_nontemporal_store(v[u], &d4[i + (size_t)u * stride]);
                else    d4[i + (size_t)u * stride] = v[u];
            }
        }
    }
    for (; i < n4; i += stride) {
        if constexpr (NT) __builtin_nontemporal_store(s4[i], &d4[i]);
        else    d4[i] = s4[i];
    }

    size_t done = n4 * sizeof(v4i);
    if (threadIdx.x == 0) {
        for (size_t j = done; j < nbytes; j++) ((char *)dst)[j] = ((const char *)src)[j];
    }
}

// Packs one 256-row tile of the gathered operand's scales into the lane-native layout the GEMM
// reads, straight from the peer's buffer so the raw scales never reach global memory. Shared by
// the MXFP8 TN and NN kernels: the host (run_mxfp8) skips gather_scales when interleaving is on,
// so a layout declaring SUPPORTS_INTERLEAVE without calling this silently packs garbage.
//
// <STEP, NG> must match the pack_scales_kernel<_, STEP, NG> instantiation mxfp8_gemm.cpp uses for
// that operand -- both write ln[(tile_id * NG + grp) * 64 + lane]. A side <64,4>, B side <32,8>.
// pack_scales reads 64 words past the last window start, hence the PAD staging stride below.
//
// Blocks take a contiguous [k0,k1) out of gath_wg and write disjoint tile_ids, so the only sync
// is around the staging buffer.
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
    // stages straight-line: one dword per thread, no loop, no bounds test.
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

template <int U, bool NT, typename T>
__device__ __forceinline__
void gather_peer_tile(int peer, int tn, int sub, int gath_wg, int tiles_per_chunk, char *gather_dst,
                      const PeerPtrsT<T> &peers, size_t chunk_bytes, unsigned int *arrive) {
    const size_t tile_bytes = chunk_bytes / tiles_per_chunk;
    const size_t doff       = (size_t)peer * chunk_bytes + (size_t)tn * tile_bytes;

    size_t sub_bytes = (((tile_bytes + gath_wg - 1) / gath_wg) + 15) & ~size_t(15);
    size_t o         = (size_t)sub * sub_bytes;
    size_t l         = (o >= tile_bytes) ? 0 : ((o + sub_bytes <= tile_bytes) ? sub_bytes : tile_bytes - o);

    if (l) gather_copy_wg<U, NT>(gather_dst + doff + o, (const char *)peers.base[peer] + doff + o, l);

    __syncthreads();
    if constexpr (NT) {
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

    // Ordering comes from ag_ready_kernel, same as the data copy above. Trip counts differ across
    // blocks when gath_wg does not divide k_iters -- fine, __syncthreads() is per block.
    {
        const char *peer_scales = (const char *)peers.base[peer] + scale_base
                                + (size_t)peer * scale_chunk_bytes
                                + (size_t)tn * BLOCK_ROW * (size_t)scale_K;
        pack_tile_scales_from<STEP, NG>((const uint8_t *)peer_scales, packed_sa,
                                        peer * tiles_per_chunk + tn, tiles_per_col,
                                        k_iters, scale_K, smem_tile, sub, gath_wg);
    }
    __syncthreads();

    // Fence AFTER the pack so it covers the packed_sa stores too, not just the data copy: a
    // consumer that sees the arrival count reach gath_wg must see both.
    if constexpr (NT) {
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    } else {
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    }
    __syncthreads();
    if (threadIdx.x == 0) AG_PUBLISH(&arrive[peer * tiles_per_chunk + tn]);
    __syncthreads();
}

template <int U, bool NT, typename T>
__device__ __forceinline__
void gather_all(int my_pe, int gath_wg, int tiles_per_chunk, char *gb, const PeerPtrsT<T> &peers,
                size_t chunk_bytes, unsigned int *arrive) {
    const int pi   = (int)blockIdx.x / gath_wg;
    const int sub  = (int)blockIdx.x % gath_wg;
    const int peer = pi + (pi >= my_pe ? 1 : 0);
    for (int tn = 0; tn < tiles_per_chunk; tn++) {
        gather_peer_tile<U, NT>(peer, tn, sub, gath_wg, tiles_per_chunk, gb, peers, chunk_bytes, arrive);
    }
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
                        kittens::base_types::convertor<U, T>::convert(src.tiles[i][j].data[idx].x);
                    dst_ptr[(row + l * 2 + 1) * row_stride + col] =
                        kittens::base_types::convertor<U, T>::convert(src.tiles[i][j].data[idx].y);
                }
            }
        }
    }
}

constexpr int RS_MAX_TP = 8;

// Arrival poison, a comm workgroup still reading it knows the producing GEMM has not stored there yet
// NOTE: These are signalling NaNs, so while hardware should never output them, if a user initializes their
// data or pads with this specific value, they will see hangs until the timeout is hit.
#define RS_SENT_BF16 0xFFAAu
#define RS_SENT_DW ((unsigned int)RS_SENT_BF16 * 0x00010001u)

// Per-PE pointer to each peer's reduce-scatter staging buffer. Unlike PeerPtrsT this is not
// templated: the stage always holds the bf16 output, whatever the operand dtype was.
struct RsPeers {
    kittens::bf16 *stage[RS_MAX_TP];
};

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

// The reduce half of a fused GEMM+RS: fold every peer's stage slot reserved for `my_pe` into `out`,
// spinning per 16-byte line until the producing GEMM has replaced the poison. Shared by the bf16 and
// MXFP8 TN kernels -- they differ in which axis the bands run along, but a band is contiguous in the
// stage either way, so the fold only ever sees `bands` and `band_elems`.
template <int TP, bool NT>
__device__ __forceinline__
void pull_reduce_all_sent(int my_pe, int ncomm, int bands,
                          const kittens::bf16 *__restrict__ local_stage, const RsPeers &peers,
                          kittens::bf16 *__restrict__ out, size_t band_elems, uint64_t warn_ticks) {
    const int w        = (int)blockIdx.x;
    const size_t lines = band_elems / 8;
    typedef int v4i __attribute__((ext_vector_type(4)));

    const uint64_t deadline = warn_ticks ? wall_clock64() + warn_ticks : 0;
    bool warned = (warn_ticks == 0);

    const size_t per_g = (lines + (size_t)ncomm - 1) / (size_t)ncomm;
    const size_t g0    = (size_t)w * per_g;
    if (g0 >= lines) return;
    const size_t g1    = (g0 + per_g < lines) ? (g0 + per_g) : lines;

    for (int b0 = 0; b0 < bands; b0++) {
        const size_t soff = ((size_t)my_pe * bands + b0) * band_elems;
        v4i *dst          = (v4i *)(out + (size_t)b0 * band_elems);

        for (size_t l = g0 + threadIdx.x; l < g1; ) {
            float acc[8];
            unsigned int pend = 0u;
#pragma unroll 1
            for (int s = 0; s < TP; s++) {
                const kittens::bf16 *base = (s == my_pe) ? local_stage : peers.stage[s];
                const v4i v = sent_load16((const rs_v4i *)(base + soff) + l);
                pend |= sent_slot_pending((const int *)&v);
                const __hip_bfloat16 *x = reinterpret_cast<const __hip_bfloat16 *>(&v);
                if (s == 0) {
#pragma unroll
                    for (int j = 0; j < 8; j++) acc[j] = __bfloat162float(x[j]);
                } else {
#pragma unroll
                    for (int j = 0; j < 8; j++) acc[j] += __bfloat162float(x[j]);
                }
            }
            if (pend != 0u) {
                if (!warned && wall_clock64() > deadline) {
                    unsigned int srcs = 0u;
#pragma unroll 1
                    for (int s = 0; s < TP; s++) {
                        const kittens::bf16 *b = (s == my_pe) ? local_stage : peers.stage[s];
                        const v4i pv = sent_load16((const rs_v4i *)(b + soff) + l);
                        if (sent_slot_pending((const int *)&pv)) srcs |= (1u << s);
                    }
                    printf("[fused GEMM+RS] fold still waiting: my_pe=%d block=%d band=%d "
                           "line=%llu pending_srcs=0x%02x\n",
                           my_pe, (int)blockIdx.x, b0, (unsigned long long)l, srcs);
                    warned = true;
                }
                continue;
            }
            v4i res;
            __hip_bfloat16 *a = reinterpret_cast<__hip_bfloat16 *>(&res);
#pragma unroll
            for (int j = 0; j < 8; j++) a[j] = __float2bfloat16(acc[j]);
            if constexpr (NT) {
                __builtin_nontemporal_store(res, &dst[l]);
            } else {
                dst[l] = res;
            }
            l += blockDim.x;
        }
    }
}

}  // namespace hk_overlap
