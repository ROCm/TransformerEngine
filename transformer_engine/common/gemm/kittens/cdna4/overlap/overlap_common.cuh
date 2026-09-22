/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/
#pragma once

#include "hip/hip_runtime.h"
#include "kittens.cuh"
#include "mxfp8_direct_transpose.cuh"
#include <cstddef>
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

// The shared MXFP8 scale helpers below walk the gathered axis with one constant, but TN
// gathers on N and NN on M.
static_assert(BLOCK_ROW == BLOCK_COL,
              "pack_tile_scales_from and gather_peer_tile_plus_scales assume one tile extent");

constexpr int GRID_CAP     = 256;

// MXFP8's scaling block: 32 elements share one E8M0 scale, along K row-wise and along M
// column-wise. The column-wise derivation below tiles the gathered tensor into bands of 32 rows.
constexpr int MX_BLOCK = 32;

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
// reads, straight from the peer's buffer. dst_rows mirrors the raw bytes into our own Userbuffers
// scale region, which the host leaves untouched when interleaving skips gather_scales.
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
                           uint32_t *__restrict__ smem_tile, int sub, int gath_wg,
                           uint8_t *__restrict__ dst_rows) {
    constexpr int ROWS     = BLOCK_ROW;            // tile extent on the gathered axis
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
                const size_t off = (size_t)srow * scale_K + (kb + slot) * 4;
                __builtin_memcpy(&p, &src_rows[off], 4);
                if (dst_rows != nullptr) {
                    __builtin_memcpy(&dst_rows[off], &p, 4);
                }
            }
            smem_tile[threadIdx.x] = p;
        } else {
            for (int i = (int)threadIdx.x; i < KPP * PAD; i += NUM_THREADS) {
                const int wslot = i / PAD;  // which of the KPP k-iters this word belongs to
                const int wrow  = i % PAD;  // which row; >= ROWS is the zero-filled OOB tail
                uint32_t p = 0;
                if (wrow < ROWS && kb + wslot < k1) {
                    const size_t off = (size_t)wrow * scale_K + (kb + wslot) * 4;
                    __builtin_memcpy(&p, &src_rows[off], 4);
                    if (dst_rows != nullptr) {
                        __builtin_memcpy(&dst_rows[off], &p, 4);
                    }
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

// ---------------------------------------------------------------------------------------------
// Deriving the column-wise MXFP8 view of the gathered tensor, in place of a second all-gather.
//
// Row-parallel backward needs dY twice: row-scaled for dgrad (this GEMM's operand) and
// column-scaled for wgrad. TE stores both with the SAME [rows, cols] data layout -- MXFP8's
// column-wise form is not a transpose, only a re-blocking of the scales from [rows, cols/32] to
// [rows/32, cols] -- so the second copy is derivable from the first by exponent arithmetic alone,
// without the high-precision tensor and without moving a byte over the links. See
// mxfp8_direct_transpose.cuh for the element rule and arXiv:2511.02302 for its derivation.
// ---------------------------------------------------------------------------------------------

// LDS the band routine stages into: the band's source scales, then their per-block max.
constexpr int MX_COLWISE_WIN        = NUM_THREADS * 16;              // columns per pass
constexpr int MX_COLWISE_SMEM_BYTES = MX_COLWISE_WIN + MX_COLWISE_WIN / MX_BLOCK;

// One 32-row band: read its row-wise elements and scales, write the column-wise ones.
// src_data/dst_data point at the band's first row; dst_scales at its single output scale row.
template <int CODE>
// __noinline__ is load-bearing, not a hint. This runs in the same __global__ function as
// the MFMA main loop, and register allocation is worst-case over the whole function: when
// this is inlined its demand merges into the main loop's and the INNER LOOP spills, which
// costs ~45%% on the GEMM even when the derivation never runs. Measured, see
// MXFP8_DIRECT_TRANSPOSE_SCOPE.md.
__device__ __noinline__
void mx_colwise_band(const uint8_t *__restrict__ src_data, const uint8_t *__restrict__ src_scales,
                     uint8_t *__restrict__ dst_data, uint8_t *__restrict__ dst_scales,
                     int K, int scale_K, uint8_t *__restrict__ smem) {
    constexpr int ROWS = MX_BLOCK;          // rows sharing one column-wise scale
    constexpr int BPT  = 16;                // one uint4 per thread per row
    constexpr int WIN  = MX_COLWISE_WIN;
    static_assert(MX_BLOCK % BPT == 0, "a thread's columns must sit inside one scale block");
    static_assert(WIN % MX_BLOCK == 0, "a pass must cover whole scale blocks");

    uint8_t *s_src = smem;                  // [ROWS][nb]  the band's row scales
    uint8_t *s_max = smem + WIN;            // [nb]        their max == the target scale T'

    const int tid = (int)threadIdx.x;
    for (int c0 = 0; c0 < K; c0 += WIN) {
        const int w  = (K - c0 < WIN) ? (K - c0) : WIN;   // columns this pass
        const int nb = w / MX_BLOCK;                      // scale blocks this pass
        const int b0 = c0 / MX_BLOCK;

        __syncthreads();                                  // previous pass is done with the staging
        for (int i = tid; i < ROWS * nb; i += NUM_THREADS) {
            s_src[i] = src_scales[(size_t)(i / nb) * scale_K + b0 + (i % nb)];
        }
        __syncthreads();

        // Step 2: T' = max of the band's 32 row scales. Every column of a 32x32 tile draws from
        // all 32 of them, so all 32 columns get the same target -- the paper's shortcut, and the
        // reason a tile lands on one shared scale rather than 32 per-column ones. Because it is a
        // max, every shift below is downward and overflow is structurally impossible.
        //
        // NEXT STEP: an exact per-column target, i.e. the smallest scale that still represents
        // that column's largest element -- what TE's own column-wise quantizer would pick. It is
        // never larger than this tile max and usually several binades smaller, so it would turn
        // nearly every remaining underflow into an exact value. It is cheap in this layout too:
        // a thread owns whole columns, so the reduction is over elements it already holds, with
        // no cross-lane traffic. What it costs, and why it is not what shipped first:
        //   - shifts become signed. An element may move UP, so a subnormal source needs
        //     renormalizing (a leading-zero count on a 2-3 bit mantissa); mx_rescale() takes
        //     k >= 0 today and would need that branch, and the packed path below clamps k as an
        //     unsigned value, so it would need a signed lane predicate rather than a clamp.
        //   - overflow stops being structurally impossible and starts resting on an argument --
        //     that the column's own max, by construction, still fits.
        //   - the target has to be picked the way ptx::float_to_e8m0(amax * max_norm_rcp) does,
        //     rounding the exponent UP, not the floor(log2(amax)) - emax the paper writes.
        //   - mxfp8_direct_transpose.py proves the max-of-scales rule only, so the host check in
        //     mxfp8_direct_transpose_check.cpp would have to be extended first.
        // Worth doing when measured E4M3 loss on real gradients justifies it: E5M2 is already
        // lossless on every realistic case, and E4M3 keeps 97.6% of elements exact even at a
        // 12-binade spread of row scales.
        for (int b = tid; b < nb; b += NUM_THREADS) {
            int m = s_src[b];
            for (int r = 1; r < ROWS; r++) m = max(m, (int)s_src[r * nb + b]);
            s_max[b] = (uint8_t)m;
        }
        __syncthreads();

        const int col = tid * BPT;                        // this thread's columns within the pass
        if (col < w) {
            const int blk        = col / MX_BLOCK;
            const int target     = s_max[blk];
            // The column-wise scale array is [rows/32, K]: one row per band, one byte per column.
            // All BPT of this thread's columns sit in one block, so they share `target`.
            const uint32_t rep = (uint32_t)target * 0x01010101u;
            *(uint4 *)(dst_scales + c0 + col) = make_uint4(rep, rep, rep, rep);

            // Step 3: the rescale. k is constant across a thread's columns and across nothing
            // else, so it is recomputed per row -- one LDS byte, not a reload.
            //
            // Four bytes at a time: mx_rescale_dword() takes the whole dword off in one packed
            // subtract whenever all four of its elements stay normal, which is the overwhelming
            // majority of them, and only falls back to the per-byte routine for the rest. k == 0
            // short-circuits the whole uint4 before that, since a shift of nothing is a copy.
            for (int r = 0; r < ROWS; r++) {
                const int k      = target - (int)s_src[r * nb + blk];
                const size_t off = (size_t)r * K + c0 + col;
                uint4 x = *(const uint4 *)(src_data + off);
                if (k != 0) {
                    x.x = hk_mxfp8_transpose::mx_rescale_dword<CODE>(x.x, k);
                    x.y = hk_mxfp8_transpose::mx_rescale_dword<CODE>(x.y, k);
                    x.z = hk_mxfp8_transpose::mx_rescale_dword<CODE>(x.z, k);
                    x.w = hk_mxfp8_transpose::mx_rescale_dword<CODE>(x.w, k);
                }
                *(uint4 *)(dst_data + off) = x;
            }
        }
    }
    __syncthreads();
}

// Every band of every chunk, spread round-robin over the gatherer blocks. Unlike the gather this
// replaces, it must cover our OWN chunk too: nobody ships us those rows, and nothing else writes
// the column-wise copy of them any more.
template <int CODE>
// __noinline__ is load-bearing, not a hint. This runs in the same __global__ function as
// the MFMA main loop, and register allocation is worst-case over the whole function: when
// this is inlined its demand merges into the main loop's and the INNER LOOP spills, which
// costs ~45%% on the GEMM even when the derivation never runs. Measured, see
// MXFP8_DIRECT_TRANSPOSE_SCOPE.md.
__device__ __noinline__
void mx_colwise_all(int my_pe, int tp_size, int gath_wg, int tiles_per_chunk, int ngath,
                    const char *__restrict__ ub, size_t chunk_bytes, size_t scale_base,
                    size_t scale_chunk_bytes, char *__restrict__ dst, size_t dst_scale_base,
                    unsigned int *__restrict__ arrive, int K, int scale_K,
                    uint8_t *__restrict__ smem) {
    constexpr int BANDS_PER_TILE = BLOCK_ROW / MX_BLOCK;
    const int bands_per_chunk    = tiles_per_chunk * BANDS_PER_TILE;
    const int total_bands        = tp_size * bands_per_chunk;

    for (int band = (int)blockIdx.x; band < total_bands; band += ngath) {
        const int c  = band / bands_per_chunk;
        const int bl = band - c * bands_per_chunk;

        // A peer's rows are readable once that peer's 256-row tile is complete -- the same flag
        // the compute blocks wait on. The wait only ever points from a block that has finished
        // its own gather to one that has not, so it adds no cycle. Our own chunk was in place
        // before the kernel started.
        if (c != my_pe) {
            unsigned int *f = &arrive[(size_t)c * tiles_per_chunk + bl / BANDS_PER_TILE];
            if (threadIdx.x == 0) {
                do {
                } while (AG_SPIN(f) < (unsigned)gath_wg);
                AG_ACQUIRE(f);
            }
            __syncthreads();
        }

        const size_t row0 = (size_t)bl * MX_BLOCK;
        mx_colwise_band<CODE>(
            (const uint8_t *)ub + c * chunk_bytes + row0 * K,
            (const uint8_t *)ub + scale_base + c * scale_chunk_bytes + row0 * scale_K,
            (uint8_t *)dst + c * chunk_bytes + row0 * K,
            (uint8_t *)dst + dst_scale_base + c * scale_chunk_bytes + (size_t)bl * K,
            K, scale_K, smem);
    }
}

// ---------------------------------------------------------------------------------------------
// WHERE that derivation runs. All three modes below emit the same bytes; what differs is the
// work's position relative to the gather and the GEMM, and therefore which of the two it takes
// its bandwidth from. NVTE_AG_WGRAD_TRANSPOSE picks one -- ag_wgrad_transpose_mode() in
// comm_gemm.cpp reads it.
//
//   PHASE     mx_colwise_all above, run by the gatherer blocks after the gather round and before
//             they join the compute queue. It is on the critical path twice over: it re-reads the
//             whole gathered tensor from HBM, and the kernel cannot retire until it is done.
//   INFLIGHT  the gather copy emits both encodings out of the bytes it is already holding. The
//             re-read disappears; what it costs is a second store stream inside the gather.
//   DEFERRED  the gather is left exactly as it is and the bands are drained from a shared pool
//             between GEMM tiles, so the re-read is still paid but overlaps with compute.
//
// Which of the last two wins is a bandwidth question on the specific shape, not an argument, so
// both are built and selectable. INFLIGHT still needs DEFERRED's pool for one chunk: nothing
// gathers our own rows, so no copy carries them -- see the pool below.
// ---------------------------------------------------------------------------------------------
constexpr int MX_CW_PHASE    = 0;
constexpr int MX_CW_INFLIGHT = 1;
constexpr int MX_CW_DEFERRED = 2;

// Column-wise bands per 256-row gather tile. This is also the gath_wg at which a gather sub-block
// owns exactly one band, which is what the INFLIGHT form needs.
constexpr int MX_BANDS_PER_TILE = BLOCK_ROW / MX_BLOCK;

// The pool of bands blocks claim between GEMM tiles. Bundled because the two drain sites would
// otherwise repeat sixteen arguments each.
struct MxColwisePool {
    int          *ctr;               // claim counter; zeroed with the other workspace counters
    unsigned int *arrive;            // the gather's per-tile arrival flags
    const char   *ub;                // the gathered (row-wise) tensor, ours
    char         *dst;               // the column-wise copy
    size_t chunk_bytes;
    size_t scale_base;
    size_t scale_chunk_bytes;
    size_t dst_scale_base;
    int bands;                       // pool size: one chunk for INFLIGHT, all of them for DEFERRED
    int bands_per_chunk;
    int tiles_per_chunk;
    int my_pe;
    int tp_size;
    int gath_wg;
    int K;
    int scale_K;
};

// Pool index -> (chunk, band within chunk). Our own chunk is slot 0, so the bands that need no
// arrival wait are handed out first and a block is least likely to park itself on a peer tile.
__device__ __forceinline__
void mx_band_locate(int g, int my_pe, int tp_size, int bands_per_chunk, int &c, int &bl) {
    const int slot = g / bands_per_chunk;
    c = slot + my_pe;
    if (c >= tp_size) {
        c -= tp_size;
    }
    bl = g - slot * bands_per_chunk;
}

// Have the rows behind pool band g landed? Ours were in place before the kernel started.
__device__ __forceinline__
bool mx_band_ready(const MxColwisePool &p, int g) {
    int c, bl;
    mx_band_locate(g, p.my_pe, p.tp_size, p.bands_per_chunk, c, bl);
    if (c == p.my_pe) {
        return true;
    }
    unsigned int *f = &p.arrive[(size_t)c * p.tiles_per_chunk + bl / MX_BANDS_PER_TILE];
    return AG_SPIN(f) >= (unsigned)p.gath_wg;
}

// One band out of the pool, with the same wait and the same addressing mx_colwise_all uses.
template <int CODE>
__device__ __forceinline__
void mx_colwise_pool_band(const MxColwisePool &p, int g, uint8_t *__restrict__ smem) {
    int c, bl;
    mx_band_locate(g, p.my_pe, p.tp_size, p.bands_per_chunk, c, bl);
    if (c != p.my_pe) {
        unsigned int *f = &p.arrive[(size_t)c * p.tiles_per_chunk + bl / MX_BANDS_PER_TILE];
        if (threadIdx.x == 0) {
            do {
            } while (AG_SPIN(f) < (unsigned)p.gath_wg);
            AG_ACQUIRE(f);
        }
        __syncthreads();
    }

    const size_t row0 = (size_t)bl * MX_BLOCK;
    mx_colwise_band<CODE>(
        (const uint8_t *)p.ub + c * p.chunk_bytes + row0 * p.K,
        (const uint8_t *)p.ub + p.scale_base + c * p.scale_chunk_bytes + row0 * p.scale_K,
        (uint8_t *)p.dst + c * p.chunk_bytes + row0 * p.K,
        (uint8_t *)p.dst + p.dst_scale_base + c * p.scale_chunk_bytes + (size_t)bl * p.K,
        p.K, p.scale_K, smem);
}

// At most one band, claimed between two GEMM tiles. It peeks before it claims so that a block
// with compute still to do does not park itself on a peer tile that has not landed; the claim can
// still race to a different band, which is why mx_colwise_pool_band() keeps the blocking wait.
template <int CODE>
// __noinline__ is load-bearing, not a hint. This runs in the same __global__ function as
// the MFMA main loop, and register allocation is worst-case over the whole function: when
// this is inlined its demand merges into the main loop's and the INNER LOOP spills, which
// costs ~45%% on the GEMM even when the derivation never runs. Measured, see
// MXFP8_DIRECT_TRANSPOSE_SCOPE.md.
__device__ __noinline__
void mx_colwise_pool_step(const MxColwisePool &p, uint8_t *__restrict__ smem) {
    __shared__ int s_g;
    if (threadIdx.x == 0) {
        const int peek = __hip_atomic_load(p.ctr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        s_g = (peek >= p.bands || !mx_band_ready(p, peek)) ? -1 : atomicAdd(p.ctr, 1);
    }
    __syncthreads();
    const int g = s_g;
    __syncthreads();                    // s_g is reused on the next scheduling step
    if (g < 0 || g >= p.bands) {
        return;
    }
    mx_colwise_pool_band<CODE>(p, g, smem);
}

// Whatever the interleaved steps did not reach. This block is out of GEMM tiles, so there is
// nothing left for the work to hide behind and it waits rather than skips.
template <int CODE>
// __noinline__ is load-bearing, not a hint. This runs in the same __global__ function as
// the MFMA main loop, and register allocation is worst-case over the whole function: when
// this is inlined its demand merges into the main loop's and the INNER LOOP spills, which
// costs ~45%% on the GEMM even when the derivation never runs. Measured, see
// MXFP8_DIRECT_TRANSPOSE_SCOPE.md.
__device__ __noinline__
void mx_colwise_pool_drain(const MxColwisePool &p, uint8_t *__restrict__ smem) {
    while (true) {
        __shared__ int s_g;
        if (threadIdx.x == 0) {
            const int peek = __hip_atomic_load(p.ctr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            s_g = (peek >= p.bands) ? p.bands : atomicAdd(p.ctr, 1);
        }
        __syncthreads();
        const int g = s_g;
        __syncthreads();
        if (g >= p.bands) {
            return;
        }
        mx_colwise_pool_band<CODE>(p, g, smem);
    }
}

// ---------------------------------------------------------------------------------------------
// INFLIGHT: the same derivation, streamed out of the gather instead of run as a pass of its own.
//
// mx_colwise_band above re-reads from HBM bytes the gather already had in registers. It does not
// have to. A gather sub-block copies align16(BLOCK_ROW * K / gath_wg) bytes of a tile, so at
// gath_wg == MX_BANDS_PER_TILE that is exactly MX_BLOCK * K -- rows [32*sub, 32*sub+32) of the
// tile, one whole column-wise band, for any K, with nothing straddling. run_mxfp8() passes that
// gath_wg as a constant; mx_can_stream_colwise() is the gate for everyone else.
//
// The one thing a band needs before its data can move is its target scale T', and T' is a
// reduction over the SCALE array alone -- 32x smaller than the data it describes. So the band's
// K scale bytes are staged and reduced up front, for ~3% of the band's traffic, and every 16-byte
// vector afterwards leaves as two stores instead of one, having been read once.
// ---------------------------------------------------------------------------------------------

// Does the INFLIGHT form apply to this geometry? Callers that fail this keep the gather helpers
// and the pool; they must not silently mis-stride a sub-block that straddles two bands.
__device__ __forceinline__
bool mx_can_stream_colwise(int gath_wg, size_t tile_bytes, int K, int scale_K, int smem_bytes) {
    return gath_wg == MX_BANDS_PER_TILE            // one sub-block == one band
        && tile_bytes == (size_t)BLOCK_ROW * K     // ... of a tile that is BLOCK_ROW K-byte rows
        && (K % MX_BLOCK) == 0                     // whole scale blocks, 16-byte vectors
        && (MX_BLOCK + 1) * scale_K <= smem_bytes; // the band's scales, plus T', fit the staging
}

// One 32-row band, produced from the bytes the gather is moving. src is the peer's band, dst ours;
// cw_data and cw_scales are the column-wise copy's. The element rule, the max-of-scales target and
// the reason all 32 columns of a tile share it are mx_colwise_band's above -- including the
// NEXT STEP note at its step 2, which describes the same future change for this loop.
template <bool NT, int CODE>
// __noinline__ is load-bearing, not a hint. This runs in the same __global__ function as
// the MFMA main loop, and register allocation is worst-case over the whole function: when
// this is inlined its demand merges into the main loop's and the INNER LOOP spills, which
// costs ~45%% on the GEMM even when the derivation never runs. Measured, see
// MXFP8_DIRECT_TRANSPOSE_SCOPE.md.
__device__ __noinline__
void mx_stream_band(const uint8_t *__restrict__ src, const uint8_t *__restrict__ src_scales,
                    uint8_t *__restrict__ dst, uint8_t *__restrict__ cw_data,
                    uint8_t *__restrict__ cw_scales, int K, int scale_K,
                    uint8_t *__restrict__ smem) {
    typedef int v4i __attribute__((ext_vector_type(4)));
    constexpr int ROWS = MX_BLOCK;

    uint8_t *s_src = smem;                       // [ROWS][scale_K]  the band's row-wise scales
    uint8_t *s_max = smem + ROWS * scale_K;      // [scale_K]        their max == the target T'

    const int tid    = (int)threadIdx.x;
    const int stride = (int)blockDim.x;
    const int nsc    = ROWS * scale_K;           // == K: a band's rows are contiguous in the
                                                 // scale array, so its whole block is one run

    __syncthreads();                             // the staging is reused band to band
    for (int i = tid; i < nsc; i += stride) {
        s_src[i] = src_scales[i];
    }
    __syncthreads();

    // Step 2 of mx_colwise_band, over the whole band at once rather than a column window at a
    // time. Same max, so the same T', and k >= 0 stays structurally true.
    for (int b = tid; b < scale_K; b += stride) {
        int m = s_src[b];
        for (int r = 1; r < ROWS; r++) m = max(m, (int)s_src[r * scale_K + b]);
        s_max[b] = (uint8_t)m;
    }
    __syncthreads();

    // The column-wise scale array is [rows/32, K]: one row per band, one byte per column, so the
    // band's whole row is T' broadcast 32-wide. Written here, before the data, because it is tiny
    // and nothing downstream distinguishes the two.
    const int vpr = K / 16;                      // 16-byte vectors per row
    for (int i = tid; i < vpr; i += stride) {
        const uint32_t rep = (uint32_t)s_max[i / 2] * 0x01010101u;
        const v4i      w   = {(int)rep, (int)rep, (int)rep, (int)rep};
        if constexpr (NT) __builtin_nontemporal_store(w, (v4i *)cw_scales + i);
        else    *((v4i *)cw_scales + i) = w;
    }

    const v4i *s4 = (const v4i *)src;
    v4i       *d4 = (v4i *)dst;
    v4i       *w4 = (v4i *)cw_data;
    const int  n4 = ROWS * vpr;

    // Flat, contiguous and fully occupied -- the sweep gather_copy_wg<1, NT> makes, unchanged, so
    // the row-wise gather's access pattern is what it always was. (row, col) is carried alongside
    // rather than divided out of the index, because K is a runtime value; the fixup below costs
    // ceil(stride / vpr) subtractions, two of them per iteration at K = 4096.
    int c = tid, r = 0;
    while (c >= vpr) { c -= vpr; r++; }
    for (int i = tid; i < n4; i += stride) {
        v4i v = s4[i];

        // ORDERING, deliberate: the row-wise store goes out FIRST, before the re-encode touches
        // the register, because it is the only one the caller's arrival publish covers. Nothing
        // waits on the column-wise bytes until this kernel has retired. Do not "tidy" the two
        // stores together or move the rescale above this line.
        if constexpr (NT) __builtin_nontemporal_store(v, &d4[i]);
        else    d4[i] = v;

        // 16 bytes at a 16-byte-aligned column all sit inside ONE 32-element scale block, so one
        // target and one shift cover the whole vector. k == 0 is a copy, as in mx_colwise_band.
        const int blk = c / 2;
        const int k   = (int)s_max[blk] - (int)s_src[r * scale_K + blk];
        if (k != 0) {
            v[0] = (int)hk_mxfp8_transpose::mx_rescale_dword<CODE>((uint32_t)v[0], k);
            v[1] = (int)hk_mxfp8_transpose::mx_rescale_dword<CODE>((uint32_t)v[1], k);
            v[2] = (int)hk_mxfp8_transpose::mx_rescale_dword<CODE>((uint32_t)v[2], k);
            v[3] = (int)hk_mxfp8_transpose::mx_rescale_dword<CODE>((uint32_t)v[3], k);
        }
        if constexpr (NT) __builtin_nontemporal_store(v, &w4[i]);
        else    w4[i] = v;

        c += stride;
        while (c >= vpr) { c -= vpr; r++; }
    }
}

// PUBLISH=false gathers without touching `arrive`: for a second, independent all-gather nothing
// waits on, and whose increments must not land in the primary's flags -- AG_PUBLISH is a
// fetch_add, so sharing an array lets a fast block satisfy a tile's count before its peers have
// written their slices.
template <int U, bool NT, bool PUBLISH = true, typename T>
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
    if constexpr (PUBLISH) {
        if (threadIdx.x == 0) AG_PUBLISH(&arrive[peer * tiles_per_chunk + tn]);
        __syncthreads();
    }
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
        // The copy above already relies on gather_dst mirroring peers.base[peer].
        const size_t soff = scale_base + (size_t)peer * scale_chunk_bytes
                          + (size_t)tn * BLOCK_ROW * (size_t)scale_K;
        pack_tile_scales_from<STEP, NG>((const uint8_t *)peers.base[peer] + soff, packed_sa,
                                        peer * tiles_per_chunk + tn, tiles_per_col,
                                        k_iters, scale_K, smem_tile, sub, gath_wg,
                                        (uint8_t *)gather_dst + soff);
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

template <int U, bool NT, bool PUBLISH = true, typename T>
__device__ __forceinline__
void gather_all(int my_pe, int gath_wg, int tiles_per_chunk, char *gb, const PeerPtrsT<T> &peers,
                size_t chunk_bytes, unsigned int *arrive) {
    const int pi   = (int)blockIdx.x / gath_wg;
    const int sub  = (int)blockIdx.x % gath_wg;
    const int peer = pi + (pi >= my_pe ? 1 : 0);
    for (int tn = 0; tn < tiles_per_chunk; tn++) {
        gather_peer_tile<U, NT, PUBLISH>(peer, tn, sub, gath_wg, tiles_per_chunk, gb, peers, chunk_bytes, arrive);
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

// The MXFP8 NN gather's own tile routine: gather_peer_tile's copy with the column-wise re-encode
// of the bytes it is already holding folded in. gather_peer_tile itself is shared with the bf16
// AG, TN and RS kernels and is left byte-for-byte as it was.
//
// Requires gath_wg == MX_BANDS_PER_TILE, so that sub_bytes == MX_BLOCK * K exactly and this
// sub-block is one whole column-wise band; mx_can_stream_colwise() is the gate.
//
// ON THE PUBLISH. AG_PUBLISH is what releases the compute blocks waiting on this tile, and the
// arrival flag covers the ROW-WISE bytes only -- nothing reads the column-wise region until the
// kernel has retired. The loop therefore issues each vector's row-wise store before its
// column-wise one, so the row-wise stream is always ahead of the fence. What cannot be moved past
// the fence is the column-wise stores themselves: holding a band's worth of output in registers
// across the publish would take 64-128 VGPRs per thread, and the alternative -- publishing first
// and re-reading gather_dst -- is the second pass this mode exists to delete. So INFLIGHT does
// pay for a second store stream inside the publish's window, and whether the gather's spare write
// bandwidth absorbs it is the measurement NVTE_AG_WGRAD_TRANSPOSE=deferred is the control for.
template <bool NT, int CODE>
// __noinline__ is load-bearing, not a hint. This runs in the same __global__ function as
// the MFMA main loop, and register allocation is worst-case over the whole function: when
// this is inlined its demand merges into the main loop's and the INNER LOOP spills, which
// costs ~45%% on the GEMM even when the derivation never runs. Measured, see
// MXFP8_DIRECT_TRANSPOSE_SCOPE.md.
__device__ __noinline__
void gather_peer_tile_colwise(int peer, int tn, int sub, int tiles_per_chunk,
                              char *__restrict__ gather_dst,
                              const PeerPtrsT<kittens::fp8e4m3> &peers, size_t chunk_bytes,
                              unsigned int *arrive, size_t scale_base, size_t scale_chunk_bytes,
                              char *__restrict__ aux_dst, size_t aux_scale_base, int K,
                              int scale_K, uint8_t *__restrict__ smem) {
    const size_t tile_bytes = chunk_bytes / tiles_per_chunk;
    const size_t band_bytes = (size_t)MX_BLOCK * K;              // == sub_bytes under the gate
    const size_t doff       = (size_t)peer * chunk_bytes + (size_t)tn * tile_bytes;
    const size_t o          = (size_t)sub * band_bytes;
    const int    bl         = tn * MX_BANDS_PER_TILE + sub;      // band index within the chunk

    // Both scale arrays are addressed at bl * K: the row-wise one because the band's 32 rows
    // start at row bl*32 of a scale_K-strided array (32 * scale_K == K), the column-wise one
    // because it holds one K-byte row per band. The row-wise scales are read from our OWN region,
    // where the host's gather_scales kernel staged every peer's rows before this launch.
    mx_stream_band<NT, CODE>(
        (const uint8_t *)peers.base[peer] + doff + o,
        (const uint8_t *)gather_dst + scale_base + (size_t)peer * scale_chunk_bytes + (size_t)bl * K,
        (uint8_t *)gather_dst + doff + o,
        (uint8_t *)aux_dst + doff + o,
        (uint8_t *)aux_dst + aux_scale_base + (size_t)peer * scale_chunk_bytes + (size_t)bl * K,
        K, scale_K, smem);

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

// gather_all's shape; the MXFP8 NN gather runs U == 1, so there is no unrolled variant to mirror.
template <bool NT, int CODE>
// __noinline__ is load-bearing, not a hint. This runs in the same __global__ function as
// the MFMA main loop, and register allocation is worst-case over the whole function: when
// this is inlined its demand merges into the main loop's and the INNER LOOP spills, which
// costs ~45%% on the GEMM even when the derivation never runs. Measured, see
// MXFP8_DIRECT_TRANSPOSE_SCOPE.md.
__device__ __noinline__
void gather_all_colwise(int my_pe, int gath_wg, int tiles_per_chunk, char *gb,
                        const PeerPtrsT<kittens::fp8e4m3> &peers, size_t chunk_bytes,
                        unsigned int *arrive, size_t scale_base, size_t scale_chunk_bytes,
                        char *aux_dst, size_t aux_scale_base, int K, int scale_K,
                        uint8_t *smem) {
    const int pi   = (int)blockIdx.x / gath_wg;
    const int sub  = (int)blockIdx.x % gath_wg;
    const int peer = pi + (pi >= my_pe ? 1 : 0);
    for (int tn = 0; tn < tiles_per_chunk; tn++) {
        gather_peer_tile_colwise<NT, CODE>(peer, tn, sub, tiles_per_chunk, gb, peers, chunk_bytes,
                                           arrive, scale_base, scale_chunk_bytes, aux_dst,
                                           aux_scale_base, K, scale_K, smem);
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

}  // namespace hk_overlap
