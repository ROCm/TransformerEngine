/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

#include "hip/hip_runtime.h"
#include "../../comm_gemm.h"
#include "fused_ag_gemm_tn.cuh"
#include "fused_ag_gemm_nn.cuh"
#include "fused_ag_mxfp8_gemm_tn.cuh"
#include "fused_ag_mxfp8_gemm_nn.cuh"
#include "../../kittens_kernel_common.cuh"
#include "bulk_rs_gemm_nt.cuh"
#include "fused_rs_gemm_tn.cuh"

#include <array>
#include <cstdio>
#include <map>
#include <mutex>
#include <vector>

namespace {

struct FusedRsLayout {
    size_t stage_bytes;
    size_t recv_off;
    size_t total_bytes;
};

inline FusedRsLayout fused_rs_layout(size_t shard_bytes, int tp_size) {
    FusedRsLayout l{};
    l.stage_bytes = shard_bytes * static_cast<size_t>(tp_size);
    l.recv_off    = l.stage_bytes;
    l.total_bytes = l.stage_bytes * 2;
    return l;
}


bool sentinel_pattern_agrees() {
    static_assert(((RS_SENT_BF16 >> 7) & 0xFFu) == 0xFFu && (RS_SENT_BF16 & 0x7Fu) != 0u,
                  "RS_SENT_BF16 must be a bf16 NaN: a finite poison can collide with real output.");
    static bool checked = false;
    static bool agrees  = false;
    if (!checked) {
        unsigned int device_dw = 0;
        agrees = hipMemcpyFromSymbol(&device_dw, HIP_SYMBOL(hk_rs_tn::rs_sent_dw_device),
                                     sizeof(device_dw)) == hipSuccess && device_dw == RS_SENT_DW;
        checked = true;
    }
    return agrees;
}

uint64_t compute_warn_ticks() {
    int dev = 0, khz = 0;
    if (hipGetDevice(&dev) != hipSuccess) return 0;
    if (hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, dev) != hipSuccess) return 0;
    return static_cast<uint64_t>(khz) * 1000ull * 120; // 2 min warning timer
}

uint64_t ag_ready_warn_ticks() {
    static const uint64_t ticks = compute_warn_ticks();
    return ticks;
}

// Peer base pointers for the scale all-gather, passed by value like PeerPtrs.
struct ScalePeers {
    const char *base[8];
};

// All-gather of the MXFP8 scale region: each rank copies every peer's scale chunk out of that
// peer's userbuffer into the matching offset of its own.
//
// scale_base and chunk_bytes are always 16B multiples here -- scale_base is SB*K and chunk_bytes
// is m_local*(K/32) with m_local a multiple of 256 -- and run_mxfp8 refuses the launch
// otherwise, so the vector loop covers the whole copy and there is no scalar tail.
template <int U, bool NT>
__global__
void gather_scales(char *__restrict__ ub, ScalePeers peers, int my_pe, int tp_size,
                   size_t scale_base, size_t chunk_bytes) {
    const int peer = static_cast<int>(blockIdx.y);
    if (peer >= tp_size || peer == my_pe) return;

    typedef int v4i __attribute__((ext_vector_type(4)));

    const size_t off = scale_base + static_cast<size_t>(peer) * chunk_bytes;
    const v4i *src = reinterpret_cast<const v4i *>(peers.base[peer] + off);
    v4i       *dst = reinterpret_cast<v4i *>(ub + off);

    const size_t n4     = chunk_bytes / sizeof(v4i);
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    const size_t step   = stride * U;
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (U > 1) {
        for (; i + static_cast<size_t>(U - 1) * stride < n4; i += step) {
            v4i v[U];
#pragma unroll
            for (int u = 0; u < U; u++) v[u] = src[i + static_cast<size_t>(u) * stride];
#pragma unroll
            for (int u = 0; u < U; u++) {
                if (NT) __builtin_nontemporal_store(v[u], &dst[i + static_cast<size_t>(u) * stride]);
                else    dst[i + static_cast<size_t>(u) * stride] = v[u];
            }
        }
    }
    for (; i < n4; i += stride) {
        if (NT) __builtin_nontemporal_store(src[i], &dst[i]);
        else    dst[i] = src[i];
    }
}

int rs_wb_group(int k_local) {
    return k_local <= 768 ? 1 : 8;
}

__global__ __launch_bounds__(64)
void ag_ready_kernel(void *const *__restrict__ peers, size_t offset,
                     const char *__restrict__ local, size_t stride, uint64_t value,
                     int first, int count, int tp_size, uint64_t warn_ticks) {
    const int c = static_cast<int>(threadIdx.x);
    if (c >= tp_size) return;

    uint64_t *pub = reinterpret_cast<uint64_t *>(
        static_cast<char *>(peers[(first + c) % count]) + offset);
    __hip_atomic_store(pub, value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);

    const uint64_t *f = reinterpret_cast<const uint64_t *>(local + static_cast<size_t>(c) * stride);
    // Report once and keep spinning
    const uint64_t deadline = warn_ticks ? wall_clock64() + warn_ticks : 0;
    bool     warned = (warn_ticks == 0);
    uint64_t seen   = __hip_atomic_load(f, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    while (seen < value) {
        if (!warned && wall_clock64() > deadline) {
            printf("[fused AG+GEMM] ag_ready_kernel still waiting: peer_slot=%d peer_first=%d "
                   "tp_size=%d flag=%p expected=%llu observed=%llu\n", c, first, tp_size, static_cast<const void *>(f),
                   static_cast<unsigned long long>(value), static_cast<unsigned long long>(seen));
            warned = true;
        }
        seen = __hip_atomic_load(f, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    }
    __hip_atomic_load(f, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}

struct AgPlan {
    void *queue = nullptr;
    int num_tiles  = 0;
    int xcd_bucket = 0;
    int off[hk_overlap::NUM_XCDS_AFF] = {};
    int cnt[hk_overlap::NUM_XCDS_AFF] = {};
};

// core, M, N, K, tp_size, rank, S. `core` namespaces the cache by kernel family, TN then NN
// within each precision: 0/1 bf16 TN/NN, 2/3 mxfp8 TN/NN. An AgPlan is a schedule over output
// tiles, so it depends on the output geometry and the TP split only -- build_work_queue takes K
// and discards it. g_rs_plans below is a separate cache, so its `core` values are independent of
// these.
using PlanKey = std::array<int, 7>;

std::map<PlanKey, AgPlan> g_plans;
std::map<const void *, std::vector<void *>> g_peers;
std::mutex g_mu;

// comm->gpu_ptrs is device memory, so the peer bases have to come back over the bus. Once per
// registered region, not per call. The returned pointer is only valid while g_mu is held.
const std::vector<void *> *peer_bases(const void *peer_ub, int count) {
    auto it = g_peers.find(peer_ub);
    if (it != g_peers.end() && it->second.size() >= static_cast<size_t>(count)) return &it->second;
    if (it != g_peers.end()) g_peers.erase(it);
    std::vector<void *> v(count);
    if (hipMemcpy(v.data(), peer_ub, count * sizeof(void *), hipMemcpyDeviceToHost) != hipSuccess) {
        return nullptr;
    }
    return &g_peers.emplace(peer_ub, std::move(v)).first->second;
}

// Packs ONLY the local chunk's scale rows. Used by the interleaved path, where every peer chunk is
// packed inside the fused kernel by the gatherer block that fetches it; the local chunk has no
// gatherer (nothing to fetch -- it is already ours) so it still needs a pack, but over m_local
// rows rather than M. Mirrors pack_scales_kernel (mxfp8_gemm.cpp) for COLWISE=false, STEP/NG as
// given, with the output tile index offset to this rank's slice.
template <int STEP, int NG>
__global__ void pack_local_scales_kernel(const uint8_t *__restrict__ scales,
                                         uint32_t *__restrict__ ln, int cblk_off, int tiles_local,
                                         int tiles_per_col, int scale_K, int k_iters) {
    constexpr int TILE_WORDS = 256;
    constexpr int PAD_WORDS  = (NG - 1) * STEP + 64;
    __shared__ uint32_t tile[PAD_WORDS];

    const int id = blockIdx.x;
    if (id >= k_iters * tiles_local) return;
    const int ki      = id / tiles_local;
    const int lblk    = id % tiles_local;
    const int kb_base = ki * 4;
    const int row0    = lblk * TILE_WORDS;

    for (int i = threadIdx.x; i < PAD_WORDS; i += blockDim.x) {
        uint32_t p = 0;
        if (i < TILE_WORDS) {
            __builtin_memcpy(&p, &scales[(size_t)(row0 + i) * scale_K + kb_base], 4);
        }
        tile[i] = p;
    }
    __syncthreads();
    const int lane = threadIdx.x % 64, grp = threadIdx.x / 64;
    const size_t tile_id = (size_t)ki * tiles_per_col + (cblk_off + lblk);
    ln[(tile_id * NG + grp) * 64 + lane] =
        kittens::pack_scales((const kittens::fp8e8m0 *)tile, grp * STEP);
}

// Gather+pack the activation scales inside the gatherer blocks (interleaved) instead of in the
// up-front gather_scales + full-M launch_pack_scales pair? Measured, 24-shape A/B on MI350X
// (gfx950, TP=8, best-of-2, bench_ag_mxfp8.py): K separates the regimes and out_local does not.
//
//     K=16384   8/8 shapes faster interleaved   -1.1 .. -7.7 %
//     K= 8192   1/8                             -3.1 .. +7.9 %
//     K= 4096   0/8                             +0.9 .. +21.8 %
//
// The gatherer reads one contiguous k-range per scale row, which is worth 8x the cache lines at
// K=16384 but only 2x at K=4096 -- below which the per-tile pack inside the arrival gate costs
// more than the removed prologue saves. Geomean speedup vs AG+HK: 1.2176x off, 1.1818x on,
// 1.2282x gated.
int interleave_scales_enabled(int K) {
    // HK_INTERLEAVE_SCALES=0/1 forces the path off/on, for A/B and for bisecting a numerics
    // failure against the K>=16384 threshold. Unset keeps the tuned heuristic.
    if (const char *e = getenv("HK_INTERLEAVE_SCALES")) return atoi(e) ? 1 : 0;
    return K >= 16384 ? 1 : 0;
}

// Peer-dedicated queues win when the queue is walked at most ~2 times and there are enough M-tiles
// to fill the buckets. The 1024 is fixed, not derived from the grid cap.
int auto_xcd_bucket(int num_tiles, int tiles_m) {
    const int grid_cap = getenv("HK_GRID_CAP") ? atoi(getenv("HK_GRID_CAP")) : 256;
    return (grid_cap > 0 && num_tiles <= 1024 && tiles_m >= 5) ? 1 : 0;
}

// Gatherer width. Tuned based off AG BW / GEMM FLOPS ratio.
int gath_wg_tn(int n_total, int tp_size) {
    if (tp_size != 8) return GATH_WG;
    return (n_total >= 3584) ? 6 : GATH_WG;
}

int gath_wg_nn(int n_total, int tp_size) {
    if (tp_size != 8) return GATH_WG;
    return (n_total >= 4608) ? 6 : GATH_WG;
}

// Tuned s.t. wgrad AG BW ~=~ dgrad GEMM TFLOPS
int gath_wg_bulk(int k, int tp_size) {
    if (tp_size != 8) return GATH_WG;
    if (k >= 7168) return 2;
    if (k >= 3584) return 4;
    if (k >= 2304) return 6;
    return GATH_WG;
}

// MXFP8 sibling of gath_wg_bulk. The thresholds are identical to the bf16 ones, and that is the
// measured result rather than an inheritance: the ~2x MFMA rate does not shift this balance on its
// own, because the gathered tensor is MXFP8 too and the comm side halves with it (chunk_bytes is
// m_local*N_TOTAL*1B against 2B for bf16). Both sides of the balance move together.
//
// k is the dgrad GEMM's reduction depth (out_local), not hidden -- the gather moves m*N_TOTAL
// bytes while the GEMM does 2*m*N_TOTAL*k flops, so flops/byte is proportional to k alone and
// neither m nor N_TOTAL belongs in the gate.
//
// Measured, 288-run sweep on MI350X (gfx950, TP=8, min-of-2, 24 shapes x widths {1,2,4,6,8,12},
// bench_bulk_mxfp8.py). Geomean total time vs the width this function returns, per k-regime,
// where * marks that returned width:
//
//     k              g=1     g=2     g=4     g=6     g=8     g=12
//     768, 1280     0.327   0.574   0.880   0.994   1.000*  0.908
//     2304, 3584    0.405   0.701   0.972   0.990   0.965   0.889
//     7168, 13312   0.825   1.000*  1.002   0.996   0.987   0.934
//
// The best alternative anywhere is +0.2% (k>=7168 at g=4) against a 1.0% median run-to-run spread,
// so nothing here is resolvable. Widths above 8 lose in every regime, so the optimum is interior
// and the search bracketed it. Starving the gather is what actually costs: g=1 gives up 17-67%.
//
// Kept separate from gath_wg_bulk despite the identical values, so that retuning bf16 cannot
// silently move MXFP8, and so the next reader can see this was measured rather than assumed.
int gath_wg_mxfp8_bulk(int k, int tp_size) {
    if (tp_size != 8) return GATH_WG;
    if (k >= 7168) return 2;
    if (k >= 3584) return 4;
    if (k >= 2304) return 6;
    return GATH_WG;
}

template <typename Plan, typename TD>
bool upload_plan(Plan &plan, const std::vector<TD> &queue) {
    const size_t bytes = queue.size() * sizeof(TD);
    if (hipMalloc(&plan.queue, bytes) != hipSuccess) return false;
    if (hipMemcpy(plan.queue, queue.data(), bytes, hipMemcpyHostToDevice) != hipSuccess) {
        static_cast<void>(hipFree(plan.queue));
        plan.queue = nullptr;
        return false;
    }
    return true;
}

struct Carve {
    char *base;
    size_t used = 0;
    size_t cap;
    void *take(size_t bytes) {
        void *p = base + used;
        used += (bytes + 255) & ~static_cast<size_t>(255);
        return p;
    }
    bool fits() const { return used <= cap; }
};

bool run_tn(const KittensAgGemmArgs &args) {
    using namespace hk_ag_tn;

    const int M       = args.n;
    const int N_TOTAL = args.m;
    const int K       = args.k;
    const int tp_size = args.nranks;
    const int m_local = M / tp_size;
    const int tiles_m = M / BLOCK_ROW;

    std::lock_guard<std::mutex> lock(g_mu);

    const PlanKey key{0, M, N_TOTAL, K, tp_size, args.rank, 1};
    auto it = g_plans.find(key);
    if (it == g_plans.end()) {
        AgPlan plan;
        auto queue      = build_work_queue(M, N_TOTAL, K, tp_size, args.rank);
        plan.num_tiles  = static_cast<int>(queue.size());
        plan.xcd_bucket = auto_xcd_bucket(plan.num_tiles, tiles_m);
        if (plan.xcd_bucket) {
            XcdBuckets bk{};
            queue = bucketize_by_xcd(queue, bk);
            for (int b = 0; b < NUM_XCDS_AFF; b++) {
                plan.off[b] = bk.off[b];
                plan.cnt[b] = bk.cnt[b];
            }
        }
        if (!upload_plan(plan, queue)) return false;
        it = g_plans.emplace(key, plan).first;
    }
    const AgPlan &plan = it->second;

    const size_t arrive_bytes = static_cast<size_t>(tiles_m) * sizeof(unsigned int);
    Carve ws{static_cast<char *>(args.workspace), 0, args.workspace_size};
    int *tile_counter = static_cast<int *>(ws.take(sizeof(int)));
    int *bucket_ctr   = static_cast<int *>(ws.take(NUM_XCDS_AFF * sizeof(int)));
    unsigned int *arrive = static_cast<unsigned int *>(ws.take(arrive_bytes));
    const size_t counter_bytes = ws.used;
    if (!ws.fits()) return false;

    const std::vector<void *> *bases = peer_bases(args.peer_ub, args.peer_count);
    if (!bases) return false;
    PeerPtrs peers{};
    for (int c = 0; c < tp_size; c++) {
        peers.base[c] = static_cast<bf16 *>((*bases)[(args.peer_first + c) % args.peer_count]);
    }
    peers.base[args.rank] = static_cast<bf16 *>(args.ub);

    XcdBuckets buckets{};
    for (int b = 0; b < NUM_XCDS_AFF; b++) {
        buckets.off[b] = plan.off[b];
        buckets.cnt[b] = plan.cnt[b];
    }

    if (hipMemsetAsync(args.workspace, 0, counter_bytes, args.stream) != hipSuccess) return false;

    if (args.arrive_peers && args.arrive_local) {
        ag_ready_kernel<<<1, 64, 0, args.stream>>>(
            static_cast<void *const *>(const_cast<void *>(args.arrive_peers)), args.arrive_offset,
            static_cast<const char *>(args.arrive_local), args.arrive_stride, args.arrive_value,
            args.peer_first, args.peer_count, tp_size, ag_ready_warn_ticks());
    }

    get_persistent_fn(M, N_TOTAL, K)(
        M, N_TOTAL, K, static_cast<bf16 *>(args.ub),
        static_cast<bf16 *>(const_cast<void *>(args.A)), static_cast<bf16 *>(args.D),
        static_cast<TileDesc *>(plan.queue), plan.num_tiles, tile_counter, peers, arrive,
        args.rank, tp_size, gath_wg_tn(N_TOTAL, tp_size), m_local, args.chunk_bytes, plan.xcd_bucket,
        buckets, bucket_ctr, args.stream);
    return hipGetLastError() == hipSuccess;
}

// Packs raw scales into the lane-native layout the mxfp8 GEMM reads: one block per 256-row scale
// tile packs its words into shared, then NG*64 lane threads emit one fp8e8m0_4 per (group, lane).
// A: STEP=64,NG=4 (256 words/tile); B: STEP=32,NG=8 (512 words, hi/lo tile pair).
// COLWISE=false: raw uint8 [dim, K/32] row-major; COLWISE=true: [K/32, dim] col-major.
template <bool COLWISE, int STEP, int NG>
__global__ void pack_scales_kernel(const uint8_t *__restrict__ scales, uint32_t *__restrict__ ln,
                                   int dim, int scale_K, int k_iters, int tiles_per_col) {
    constexpr int TILE_WORDS = 256;
    constexpr int PAD_WORDS  = (NG - 1) * STEP + 64;  // covers OOB pack_scales read
    __shared__ uint32_t tile[PAD_WORDS];

    const int tile_id = blockIdx.x;
    if (tile_id >= k_iters * tiles_per_col) return;

    const int k_iter  = tile_id / tiles_per_col;
    const int cblk    = tile_id % tiles_per_col;
    const int kb_base = k_iter * 4;
    const int row0    = cblk * TILE_WORDS;

    for (int i = threadIdx.x; i < PAD_WORDS; i += blockDim.x) {
        uint32_t p = 0;
        if (i < TILE_WORDS) {
            const int row = row0 + i;
            if constexpr (COLWISE) {
                const int base = kb_base * dim + row;
                p  =  (uint32_t)scales[base]                  | ((uint32_t)scales[base +     dim] << 8)
                   | ((uint32_t)scales[base + 2 * dim] << 16) | ((uint32_t)scales[base + 3 * dim] << 24);
            } else {
                __builtin_memcpy(&p, &scales[(size_t)row * scale_K + kb_base], 4);
            }
        }
        tile[i] = p;  // OOB tail (i>=256) zero-filled
    }
    __syncthreads();

    const int tid = threadIdx.x, lane = tid % 64, grp = tid / 64;
    kittens::fp8e8m0_4 out = kittens::pack_scales((const kittens::fp8e8m0 *)tile, grp * STEP);
    ln[((size_t)tile_id * NG + grp) * 64 + lane] = out;
}

template <bool COLWISE, int STEP, int NG>
void launch_pack_scales(const uint8_t *scales, uint32_t *ln, int dim, int scale_K, int k_iters,
                        hipStream_t stream) {
    const int tiles_per_col = dim / 256;
    pack_scales_kernel<COLWISE, STEP, NG><<<k_iters * tiles_per_col, NG * 64, 0, stream>>>(
        scales, ln, dim, scale_K, k_iters, tiles_per_col);
}

// The two mxfp8 layouts run an identical host sequence. Now that both kernel namespaces take
// TileDesc, PeerPtrs and XcdBuckets from hk_overlap, persistent_fn_t is one type across them, so
// the layout only has to name its plan tag, how the weight scales are laid out, and its factories.
struct Mxfp8Tn {
    using TileDesc = hk_mxfp8_ag_tn::TileDesc;
    static constexpr int  PLAN_TAG        = 2;
    static constexpr bool A_SCALE_COLWISE = false;
    // BLAS convention, as mxfp8_gemm.cpp defines it: the kernel's A slot holds the weight on the
    // M axis and its B slot the gathered activation on N.
    static constexpr bool GATHERED_ON_M   = false;
    // The kernel gathers AND packs the peers' scale rows itself (gather_all_plus_scales /
    // pack_tile_scales_from), which is the contract the interleaved host path depends on.
    static constexpr bool SUPPORTS_INTERLEAVE = true;
    static std::vector<TileDesc> work_queue(int M, int N, int K, int tp, int pe) {
        return hk_mxfp8_ag_tn::build_work_queue(M, N, K, tp, pe);
    }
    static hk_mxfp8_ag_tn::persistent_fn_t launch_fn(int M, int N, int K, int a_code, int b_code) {
        return hk_mxfp8_ag_tn::get_persistent_fn(M, N, K, a_code, b_code);
    }
};

// NN consumes TE's A operand (this kernel's B) column-wise -- see rocm_comm_gemm_overlap.cpp:296.
struct Mxfp8Nn {
    using TileDesc = hk_mxfp8_ag_nn::TileDesc;
    static constexpr int  PLAN_TAG        = 3;
    static constexpr bool A_SCALE_COLWISE = true;
    // Slots swapped relative to TN: the kernel's A slot holds the gathered activation, so the
    // gathered tokens live on the M axis and the weight is the B operand on N.
    static constexpr bool GATHERED_ON_M   = true;
    // NN now runs the same shared gather_all_plus_scales as TN (overlap_common.cuh), so it
    // upholds the interleaved host path's contract: the gatherer blocks pack the peers' scale
    // rows. Before that it declared the flag and ignored it, leaving 7/8 of packed_sa garbage
    // (max_err 0.373-0.387 vs 0.0147) on every K>=16384 shape.
    static constexpr bool SUPPORTS_INTERLEAVE = true;
    static std::vector<TileDesc> work_queue(int M, int N, int K, int tp, int pe) {
        return hk_mxfp8_ag_nn::build_work_queue(M, N, K, tp, pe);
    }
    static hk_mxfp8_ag_nn::persistent_fn_t launch_fn(int M, int N, int K, int a_code, int b_code) {
        return hk_mxfp8_ag_nn::get_persistent_fn(M, N, K, a_code, b_code);
    }
};

template <class L>
bool run_mxfp8(const KittensAgGemmArgs &args) {
    using hk_overlap::BLOCK_ROW;
    using hk_overlap::BLOCK_COL;
    using hk_overlap::NUM_XCDS_AFF;
    using TileDesc = typename L::TileDesc;
    using hk_overlap::XcdBuckets;
    using PeerPtrs = hk_overlap::PeerPtrsT<kittens::fp8e4m3>;
    using kittens::bf16;
    using kittens::fp8e4m3;

    // Both mxfp8 kernels step K by 128; the step stays with the kernels because the bf16 pair
    // steps 64. Both spell it BLOCK_K, following mxfp8_gemm.cpp.
    constexpr int BLOCK_K = hk_mxfp8_ag_tn::BLOCK_K;
    static_assert(BLOCK_K == hk_mxfp8_ag_nn::BLOCK_K, "mxfp8 TN/NN must agree on the K step");

    // Which kernel slot the gathered activation lands in is the layout's choice. TN keeps the
    // BLAS convention mxfp8_gemm.cpp defines -- A is the weight on the M axis, B the gathered
    // activation on N -- while NN swaps the slots, so its A operand is the gathered activation
    // and the tokens live on M. Everything that follows the gathered operand (axis extents, xcd
    // bucketing, arrival flags, the 64/4 <-> 32/8 scale-pack split and the operand pointers)
    // moves with it; everything else is shared verbatim.
    const int M       = L::GATHERED_ON_M ? args.n : args.m;
    const int N_TOTAL = L::GATHERED_ON_M ? args.m : args.n;
    const int K       = args.k;
    const int tp_size = args.nranks;
    const int tiles_m = M / BLOCK_ROW;
    const int tiles_n = N_TOTAL / BLOCK_COL;
    // The gathered operand's per-rank shard along the gathered axis, and that axis in tiles:
    // m_local / tiles_m for NN, n_local / tiles_n for TN. The gathered axis is args.n on both
    // sides, so both values are the same either way -- only the axis they name moves.
    const int gath_local = (L::GATHERED_ON_M ? M : N_TOTAL) / tp_size;
    const int tiles_gath = L::GATHERED_ON_M ? tiles_m : tiles_n;

    const int k_iters = K / BLOCK_K;
    int scale_K = K / 32;

    // The scale region is sized in CommOverlapP2PBase::initialize; bail out rather than read
    // garbage if its chunking ever disagrees with this kernel's view of the operand. The 16B
    // alignment holds for every shape the eligibility gate admits and is what lets gather_scales
    // run without a scalar tail, so drop to hipBLASLt rather than silently mis-copying if it ever
    // stops holding.
    if (args.scale_chunk_bytes &&
        (args.scale_chunk_bytes != static_cast<size_t>(gath_local) * static_cast<size_t>(scale_K) ||
         args.scale_chunk_bytes % 16 != 0 || args.scale_base_offset % 16 != 0)) {
        return false;
    }

    std::lock_guard<std::mutex> lock(g_mu);

    const PlanKey key{L::PLAN_TAG, M, N_TOTAL, K, tp_size, args.rank, 1};
    auto it = g_plans.find(key);
    if (it == g_plans.end()) {
        AgPlan plan;
        auto queue      = L::work_queue(M, N_TOTAL, K, tp_size, args.rank);
        plan.num_tiles  = static_cast<int>(queue.size());
        plan.xcd_bucket = auto_xcd_bucket(plan.num_tiles, tiles_gath);
        if (plan.xcd_bucket) {
            XcdBuckets bk{};
            queue = hk_overlap::bucketize_by_xcd(queue, bk);
            for (int b = 0; b < NUM_XCDS_AFF; b++) {
                plan.off[b] = bk.off[b];
                plan.cnt[b] = bk.cnt[b];
            }
        }
        if (!upload_plan(plan, queue)) return false;
        it = g_plans.emplace(key, plan).first;
    }
    const AgPlan &plan = it->second;

    // Lane-native scale buffers: A = 256 words/tile, B = 512 (hi/lo pair). If they overflow the
    // caller's budget we return false and fall back to hipBLASLt.
    size_t sa_bytes = kittens_align_up((size_t)k_iters * tiles_m * 256 * sizeof(uint32_t), 256);
    size_t sb_bytes = kittens_align_up((size_t)k_iters * tiles_n * 512 * sizeof(uint32_t), 256);

    const size_t arrive_bytes = static_cast<size_t>(tiles_gath) * sizeof(unsigned int);
    Carve ws{static_cast<char *>(args.workspace), 0, args.workspace_size};
    int *tile_counter = static_cast<int *>(ws.take(sizeof(int)));
    int *bucket_ctr   = static_cast<int *>(ws.take(NUM_XCDS_AFF * sizeof(int)));
    unsigned int *arrive = static_cast<unsigned int *>(ws.take(arrive_bytes));

    const size_t counter_bytes = ws.used;
    uint32_t* packed_sa = static_cast<uint32_t*>(ws.take(sa_bytes));
    uint32_t* packed_sb = static_cast<uint32_t*>(ws.take(sb_bytes));
    if (!ws.fits()) return false;

    // Weight scales are rank-local: no dependency on ag_ready_kernel or any peer's chunk, so this
    // can be enqueued before the gather rather than after it.
    // scale_A is the weight's, whichever slot the weight sits in; only the <STEP, NG> geometry
    // and the destination buffer follow the slot.
    if constexpr (L::GATHERED_ON_M) {
        launch_pack_scales<L::A_SCALE_COLWISE, 32, 8>((const uint8_t *)args.scale_A, packed_sb, N_TOTAL, scale_K, k_iters, args.stream);
    } else {
        launch_pack_scales<L::A_SCALE_COLWISE, 64, 4>((const uint8_t *)args.scale_A, packed_sa, M, scale_K, k_iters, args.stream);
    }

    const std::vector<void *> *bases = peer_bases(args.peer_ub, args.peer_count);
    if (!bases) return false;
    PeerPtrs peers{};
    for (int c = 0; c < tp_size; c++) {
        peers.base[c] = static_cast<fp8e4m3 *>((*bases)[(args.peer_first + c) % args.peer_count]);
    }
    peers.base[args.rank] = static_cast<fp8e4m3 *>(args.ub);

    XcdBuckets buckets{};
    for (int b = 0; b < NUM_XCDS_AFF; b++) {
        buckets.off[b] = plan.off[b];
        buckets.cnt[b] = plan.cnt[b];
    }

    if (hipMemsetAsync(args.workspace, 0, counter_bytes, args.stream) != hipSuccess) return false;

    if (args.arrive_peers && args.arrive_local) {
        ag_ready_kernel<<<1, 64, 0, args.stream>>>(
            static_cast<void *const *>(const_cast<void *>(args.arrive_peers)), args.arrive_offset,
            static_cast<const char *>(args.arrive_local), args.arrive_stride, args.arrive_value,
            args.peer_first, args.peer_count, tp_size, ag_ready_warn_ticks());
    }

    // Activation scales live in the userbuffer, so they have to be gathered before they can be
    // packed -- and only after ag_ready_kernel, which is what guarantees that the peers finished
    // writing their own chunks. The region layout does not depend on TN/NN: scale_B is row-wise
    // in both, which is why this part is shared verbatim.
    // Interleaving needs the peers' raw scales reachable in their userbuffers, which is exactly
    // what scale_chunk_bytes != 0 means. That is now always true for a fused MXFP8 AG buffer
    // (comm_gemm_overlap.cpp), so the guard is defensive rather than a configuration switch.
    const int interleave =
        (L::SUPPORTS_INTERLEAVE && args.scale_chunk_bytes) ? interleave_scales_enabled(K) : 0;

    if (args.scale_chunk_bytes && !interleave) {
        ScalePeers sp{};
        for (int c = 0; c < tp_size; c++) sp.base[c] = reinterpret_cast<const char *>(peers.base[c]);

        constexpr int SCALE_GATHER_U = 4;
        constexpr int SCALE_GATHER_THREADS = 256;
        // Size the grid to the work so the tail blocks are not launched idle. NT stores are off:
        // launch_pack_scales consumes this region in the very next kernel, so keeping it in cache
        // beats streaming it past.
        const size_t lines = args.scale_chunk_bytes / 16;
        const size_t per_block = static_cast<size_t>(SCALE_GATHER_THREADS) * SCALE_GATHER_U;
        int grid_x = static_cast<int>((lines + per_block - 1) / per_block);
        if (grid_x < 1) grid_x = 1;
        if (grid_x > 256) grid_x = 256;

        gather_scales<SCALE_GATHER_U, false>
            <<<dim3(grid_x, tp_size), SCALE_GATHER_THREADS, 0, args.stream>>>(
                static_cast<char *>(args.ub), sp, args.rank, tp_size, args.scale_base_offset,
                args.scale_chunk_bytes);
    }

    if (interleave) {
        // Peer rows are packed by their gatherer blocks inside the kernel; only ours is left.
        const int tiles_per_chunk = gath_local / (L::GATHERED_ON_M ? BLOCK_ROW : BLOCK_COL);
        if constexpr (L::GATHERED_ON_M) {
            pack_local_scales_kernel<64, 4>
                <<<k_iters * tiles_per_chunk, 4 * 64, 0, args.stream>>>(
                    (const uint8_t *)args.scale_B + (size_t)args.rank * args.scale_chunk_bytes,
                    packed_sa, args.rank * tiles_per_chunk, tiles_per_chunk, tiles_gath, scale_K,
                    k_iters);
        } else {
            pack_local_scales_kernel<32, 8>
                <<<k_iters * tiles_per_chunk, 8 * 64, 0, args.stream>>>(
                    (const uint8_t *)args.scale_B + (size_t)args.rank * args.scale_chunk_bytes,
                    packed_sb, args.rank * tiles_per_chunk, tiles_per_chunk, tiles_gath, scale_K,
                    k_iters);
        }
    } else {
        if constexpr (L::GATHERED_ON_M) {
            launch_pack_scales<false, 64, 4>((const uint8_t *)args.scale_B, packed_sa, M, scale_K, k_iters, args.stream);
        } else {
            launch_pack_scales<false, 32, 8>((const uint8_t *)args.scale_B, packed_sb, N_TOTAL, scale_K, k_iters, args.stream);
        }
    }

    // CBSZ is the format of whatever sits in the kernel's A slot and BLGP of its B slot, so the
    // codes travel with the pointers. args.a_fp8_code is the weight's and args.b_fp8_code the
    // gathered activation's -- rocm_comm_gemm_overlap.cpp stays BLAS-canonical either way.
    const int cbsz_code = L::GATHERED_ON_M ? args.b_fp8_code : args.a_fp8_code;
    const int blgp_code = L::GATHERED_ON_M ? args.a_fp8_code : args.b_fp8_code;
    auto launch = L::launch_fn(M, N_TOTAL, K, cbsz_code, blgp_code);
    if (!launch) return false;   // unexpected operand format pair

    fp8e4m3 *const d_weight = static_cast<fp8e4m3 *>(const_cast<void *>(args.A));
    fp8e4m3 *const d_gath   = static_cast<fp8e4m3 *>(args.ub);
    launch(
        M, N_TOTAL, K, L::GATHERED_ON_M ? d_gath : d_weight,
        L::GATHERED_ON_M ? d_weight : d_gath, static_cast<bf16 *>(args.D),
        packed_sa, packed_sb, static_cast<TileDesc *>(plan.queue), plan.num_tiles,
        tile_counter, peers, arrive, args.rank, tp_size, GATH_WG, gath_local, args.chunk_bytes,
        plan.xcd_bucket, buckets, bucket_ctr,
        args.scale_base_offset, args.scale_chunk_bytes, interleave, args.stream);
    return hipGetLastError() == hipSuccess;
}

struct NnSetup {
    const AgPlan *plan;
    hk_ag_nn::PeerPtrs peers;
    hk_overlap::XcdBuckets buckets;
    int *tile_counter;
    int *bucket_ctr;
    unsigned int *arrive;
    float *cw;
};

bool prepare_nn(const KittensAgGemmArgs &args, int S, void *peer_local, NnSetup &out) {
    using namespace hk_ag_nn;

    const int M       = args.n;
    const int N_TOTAL = args.m;
    const int K       = args.k;
    const int tp_size = args.nranks;
    const int tiles_m = M / BLOCK_ROW;

    const PlanKey key{1, M, N_TOTAL, K, tp_size, args.rank, S};
    auto it = g_plans.find(key);
    if (it == g_plans.end()) {
        AgPlan plan;
        auto queue      = build_work_queue(M, N_TOTAL, K, tp_size, args.rank, S);
        plan.num_tiles  = static_cast<int>(queue.size());
        plan.xcd_bucket = auto_xcd_bucket(plan.num_tiles, tiles_m);
        if (plan.xcd_bucket) {
            XcdBuckets bk{};
            queue = bucketize_by_xcd(queue, bk);
            for (int b = 0; b < NUM_XCDS_AFF; b++) {
                plan.off[b] = bk.off[b];
                plan.cnt[b] = bk.cnt[b];
            }
        }
        if (!upload_plan(plan, queue)) return false;
        it = g_plans.emplace(key, plan).first;
    }
    out.plan = &it->second;

    const size_t mn           = static_cast<size_t>(M) * N_TOTAL;
    const size_t arrive_bytes = static_cast<size_t>(tiles_m) * sizeof(unsigned int);
    Carve ws{static_cast<char *>(args.workspace), 0, args.workspace_size};
    out.tile_counter           = static_cast<int *>(ws.take(sizeof(int)));
    out.bucket_ctr             = static_cast<int *>(ws.take(NUM_XCDS_AFF * sizeof(int)));
    out.arrive                 = static_cast<unsigned int *>(ws.take(arrive_bytes));
    const size_t counter_bytes = ws.used;
    out.cw                     = (S > 1) ? static_cast<float *>(ws.take(S * mn * sizeof(float)))
                                         : nullptr;
    if (!ws.fits()) return false;

    const std::vector<void *> *bases = peer_bases(args.peer_ub, args.peer_count);
    if (!bases) return false;
    out.peers = PeerPtrs{};
    for (int c = 0; c < tp_size; c++) {
        out.peers.base[c] = static_cast<bf16 *>((*bases)[(args.peer_first + c) % args.peer_count]);
    }
    out.peers.base[args.rank] = static_cast<bf16 *>(peer_local);

    out.buckets = XcdBuckets{};
    for (int b = 0; b < NUM_XCDS_AFF; b++) {
        out.buckets.off[b] = out.plan->off[b];
        out.buckets.cnt[b] = out.plan->cnt[b];
    }

    if (hipMemsetAsync(args.workspace, 0, counter_bytes, args.stream) != hipSuccess) return false;

    if (args.arrive_peers && args.arrive_local) {
        ag_ready_kernel<<<1, 64, 0, args.stream>>>(
            static_cast<void *const *>(const_cast<void *>(args.arrive_peers)), args.arrive_offset,
            static_cast<const char *>(args.arrive_local), args.arrive_stride, args.arrive_value,
            args.peer_first, args.peer_count, tp_size, ag_ready_warn_ticks());
    }
    return true;
}

int split_k_nn(const KittensAgGemmArgs &args) {
    using namespace hk_ag_nn;
    const int M       = args.n;
    const int N_TOTAL = args.m;
    const int tiles_m = M / BLOCK_ROW;
    int S = select_split_k(tiles_m * (N_TOTAL / BLOCK_COL), args.workspace_size);
    if ((args.k / K_STEP) % (4 * S) != 0) S = 1;
    return S;
}

bool run_nn(const KittensAgGemmArgs &args) {
    using namespace hk_ag_nn;

    const int M       = args.n;
    const int N_TOTAL = args.m;
    const int K       = args.k;
    const int tp_size = args.nranks;
    const int m_local = M / tp_size;

    const int S = split_k_nn(args);
    persistent_fn_t pfn = get_persistent_fn(M, N_TOTAL, K, S);
    if (!pfn) return false;

    std::lock_guard<std::mutex> lock(g_mu);

    NnSetup s{};
    if (!prepare_nn(args, S, args.ub, s)) return false;

    pfn(M, N_TOTAL, K, static_cast<bf16 *>(args.ub),
        static_cast<bf16 *>(const_cast<void *>(args.A)), static_cast<bf16 *>(args.D), s.cw,
        static_cast<TileDesc *>(s.plan->queue), s.plan->num_tiles, s.tile_counter, s.peers, s.arrive,
        args.rank, tp_size, gath_wg_nn(N_TOTAL, tp_size), m_local, args.chunk_bytes, s.plan->xcd_bucket,
        s.buckets, s.bucket_ctr, args.stream);
    if (S > 1) launch_sk_reduce(s.cw, static_cast<bf16 *>(args.D), static_cast<size_t>(M) * N_TOTAL,
                                S, args.stream);
    return hipGetLastError() == hipSuccess;
}

bool run_bulk_nn(const KittensAgGemmArgs &args) {
    using namespace hk_ag_nn;

    const int M          = args.n;
    const int N_TOTAL    = args.m;
    const int K          = args.k;
    const int tp_size    = args.nranks;
    const int gath_tiles = (M / tp_size) / BLOCK_ROW;

    const int S = split_k_nn(args);
    persistent_bulk_fn_t bfn = get_persistent_bulk_fn(M, N_TOTAL, K, S);
    if (!bfn) return false;

    std::lock_guard<std::mutex> lock(g_mu);

    NnSetup s{};
    if (!prepare_nn(args, S, args.gather_dst, s)) return false;

    bfn(M, N_TOTAL, K, static_cast<bf16 *>(args.ub),
        static_cast<bf16 *>(const_cast<void *>(args.A)), static_cast<bf16 *>(args.D), s.cw,
        static_cast<TileDesc *>(s.plan->queue), s.plan->num_tiles, s.tile_counter, s.peers,
        static_cast<bf16 *>(args.gather_dst), s.arrive, args.rank, tp_size, gath_wg_bulk(K, tp_size), gath_tiles,
        args.chunk_bytes, s.plan->xcd_bucket, s.buckets, s.bucket_ctr, args.stream);
    if (S > 1) launch_sk_reduce(s.cw, static_cast<bf16 *>(args.D), static_cast<size_t>(M) * N_TOTAL,
                                S, args.stream);
    return hipGetLastError() == hipSuccess;
}

// Bulk sibling of run_mxfp8. The all-gather is NOT associated with the GEMM: it lands in
// args.gather_dst (the userbuffer) while the GEMM reads args.ub and args.A, and no compute block
// waits on an arrival because nothing here consumes the gathered bytes.
//
// The gathered tensor is untyped as far as this kernel is concerned -- gather_copy_wg moves bytes.
// Its scales, when present, are moved by the same gather_scales kernel the fused path uses, minus
// the pack: nothing in this kernel reads them, so they only have to arrive.
//
// NN only, matching the bf16 bulk entry.
bool run_bulk_mxfp8(const KittensAgGemmArgs &args) {
    using L = Mxfp8Nn;
    using namespace hk_mxfp8_ag_nn;
    using kittens::bf16;
    using kittens::fp8e4m3;

    constexpr int BLOCK_K = hk_mxfp8_ag_nn::BLOCK_K;

    // Same NN kernel as run_mxfp8<Mxfp8Nn>, so the same slot convention: the kernel's A operand is
    // the gathered activation on the M axis and its B operand the weight on N.
    const int M       = args.n;
    const int N_TOTAL = args.m;
    const int K       = args.k;
    const int tp_size = args.nranks;
    const int m_local = M / tp_size;
    const int tiles_m = M / BLOCK_ROW;
    const int tiles_n = N_TOTAL / BLOCK_COL;
    const int k_iters = K / BLOCK_K;
    const int scale_K = K / 32;

    // Same 16B-alignment requirement as the fused path, for the same reason: it is what lets
    // gather_scales run without a scalar tail. Here the region describes the GATHERED tensor's
    // shard, which is not a GEMM operand at all on this path.
    if (args.scale_chunk_bytes &&
        (args.scale_chunk_bytes % 16 != 0 || args.scale_base_offset % 16 != 0)) {
        return false;
    }

    // CBSZ is the A slot's format and BLGP the B slot's, and this kernel's A slot holds the
    // gathered activation (args.b_fp8_code), so the codes swap with the pointers below.
    auto bfn = get_persistent_bulk_fn(M, N_TOTAL, K, args.b_fp8_code, args.a_fp8_code);
    if (!bfn) return false;   // unexpected operand format pair

    std::lock_guard<std::mutex> lock(g_mu);

    // The work queue depends only on the output geometry and the TP split, so bulk and fused share
    // a plan for the same shape.
    const PlanKey key{L::PLAN_TAG, M, N_TOTAL, K, tp_size, args.rank, 1};
    auto it = g_plans.find(key);
    if (it == g_plans.end()) {
        AgPlan plan;
        auto queue      = L::work_queue(M, N_TOTAL, K, tp_size, args.rank);
        plan.num_tiles  = static_cast<int>(queue.size());
        plan.xcd_bucket = auto_xcd_bucket(plan.num_tiles, tiles_m);
        if (plan.xcd_bucket) {
            XcdBuckets bk{};
            queue = hk_overlap::bucketize_by_xcd(queue, bk);
            for (int b = 0; b < NUM_XCDS_AFF; b++) {
                plan.off[b] = bk.off[b];
                plan.cnt[b] = bk.cnt[b];
            }
        }
        if (!upload_plan(plan, queue)) return false;
        it = g_plans.emplace(key, plan).first;
    }
    const AgPlan &plan = it->second;

    size_t sa_bytes = kittens_align_up((size_t)k_iters * tiles_m * 256 * sizeof(uint32_t), 256);
    size_t sb_bytes = kittens_align_up((size_t)k_iters * tiles_n * 512 * sizeof(uint32_t), 256);

    const size_t arrive_bytes = static_cast<size_t>(tiles_m) * sizeof(unsigned int);
    Carve ws{static_cast<char *>(args.workspace), 0, args.workspace_size};
    int *tile_counter = static_cast<int *>(ws.take(sizeof(int)));
    int *bucket_ctr   = static_cast<int *>(ws.take(NUM_XCDS_AFF * sizeof(int)));
    unsigned int *arrive = static_cast<unsigned int *>(ws.take(arrive_bytes));
    const size_t counter_bytes = ws.used;
    uint32_t *packed_sa = static_cast<uint32_t *>(ws.take(sa_bytes));
    uint32_t *packed_sb = static_cast<uint32_t *>(ws.take(sb_bytes));
    if (!ws.fits()) return false;

    // The GEMM's own operand scales. Unrelated to the gathered tensor, so both can be packed up
    // front with no dependency on the gather.
    launch_pack_scales<L::A_SCALE_COLWISE, 32, 8>((const uint8_t *)args.scale_A, packed_sb, N_TOTAL,
                                                  scale_K, k_iters, args.stream);
    launch_pack_scales<false, 64, 4>((const uint8_t *)args.scale_B, packed_sa, M, scale_K,
                                     k_iters, args.stream);

    const std::vector<void *> *bases = peer_bases(args.peer_ub, args.peer_count);
    if (!bases) return false;
    PeerPtrs peers{};
    for (int c = 0; c < tp_size; c++) {
        peers.base[c] = static_cast<fp8e4m3 *>((*bases)[(args.peer_first + c) % args.peer_count]);
    }
    // Our own slot is the gather DESTINATION, not args.ub -- in bulk those are different buffers.
    peers.base[args.rank] = static_cast<fp8e4m3 *>(args.gather_dst);

    XcdBuckets buckets{};
    for (int b = 0; b < NUM_XCDS_AFF; b++) {
        buckets.off[b] = plan.off[b];
        buckets.cnt[b] = plan.cnt[b];
    }

    if (hipMemsetAsync(args.workspace, 0, counter_bytes, args.stream) != hipSuccess) return false;

    if (args.arrive_peers && args.arrive_local) {
        ag_ready_kernel<<<1, 64, 0, args.stream>>>(
            static_cast<void *const *>(const_cast<void *>(args.arrive_peers)), args.arrive_offset,
            static_cast<const char *>(args.arrive_local), args.arrive_stride, args.arrive_value,
            args.peer_first, args.peer_count, tp_size, ag_ready_warn_ticks());
    }

    // Gathered scales, if the tensor carries any. No pack: nothing in this kernel reads them.
    if (args.scale_chunk_bytes) {
        ScalePeers sp{};
        for (int c = 0; c < tp_size; c++) sp.base[c] = reinterpret_cast<const char *>(peers.base[c]);
        constexpr int SCALE_GATHER_U = 4;
        constexpr int SCALE_GATHER_THREADS = 256;
        const size_t lines = args.scale_chunk_bytes / 16;
        const size_t per_block = static_cast<size_t>(SCALE_GATHER_THREADS) * SCALE_GATHER_U;
        int grid_x = static_cast<int>((lines + per_block - 1) / per_block);
        if (grid_x < 1) grid_x = 1;
        if (grid_x > 256) grid_x = 256;
        gather_scales<SCALE_GATHER_U, false>
            <<<dim3(grid_x, tp_size), SCALE_GATHER_THREADS, 0, args.stream>>>(
                static_cast<char *>(args.gather_dst), sp, args.rank, tp_size,
                args.scale_base_offset, args.scale_chunk_bytes);
    }

    bfn(M, N_TOTAL, K, static_cast<fp8e4m3 *>(args.ub),
        static_cast<fp8e4m3 *>(const_cast<void *>(args.A)), static_cast<bf16 *>(args.D),
        packed_sa, packed_sb, static_cast<TileDesc *>(plan.queue), plan.num_tiles, tile_counter,
        peers, static_cast<char *>(args.gather_dst), arrive, args.rank, tp_size,
        gath_wg_mxfp8_bulk(K, tp_size), m_local, args.chunk_bytes, plan.xcd_bucket, buckets,
        bucket_ctr, args.stream);
    return hipGetLastError() == hipSuccess;
}

int rs_comm_wg_bulk(int out_local, int tp_size) {
    if (tp_size != 8) return hk_rs_nt::RS_COMM_WG_DEFAULT;
    if (out_local >= 8192) return 2;
    if (out_local >= 3584) return 4;
    return hk_rs_nt::RS_COMM_WG_DEFAULT;
}

static_assert(hk_rs_nt::BLOCK_SIZE == 256 && hk_rs_nt::K_STEP == 64 && hk_rs_nt::NUM_XCDS == 8,
              "bulk_rs_shape_ok literals are stale against hk_rs_nt geometry");

bool bulk_rs_shape_ok(int M, int N, int K, int splits, int nred) {
    const int nt = (splits >= 1 && (K / 64) % splits == 0) ? (K / 64) / splits : 0;

    // Order matters here
    return M % 256 == 0 && N % 256 == 0 && K % 64 == 0 &&
           splits >= 1 && (K / 64) % splits == 0 &&
           nt >= 4 && (nt & 1) == 0 &&
           nred >= 0 && nred % 8 == 0;
}

bool run_bulk_rs(const KittensRsGemmArgs &args) {
    using namespace hk_rs_nt;

    const int M       = args.n;               // out_local
    const int N       = args.m;               // hidden
    const int K       = args.k;               // tokens
    const int tp_size = args.nranks;

    int splits = select_split_k_shape(M, N, K);
    const size_t partial_bytes = static_cast<size_t>(splits) * M * N * sizeof(float);
    if (splits > 1 && partial_bytes > args.workspace_size) splits = 1;

    const int wgs  = (tp_size - 1) * rs_comm_wg_bulk(M, tp_size);
    const int nred = (wgs + hk_rs_nt::NUM_XCDS - 1) / hk_rs_nt::NUM_XCDS * hk_rs_nt::NUM_XCDS;

    if (!bulk_rs_shape_ok(M, N, K, splits, nred)) return false;

    const size_t shard_elems = args.shard_bytes / sizeof(bf16);
    const size_t band_elems  = static_cast<size_t>(RS_BAND_ROWS) * N;
    const int    bands       = static_cast<int>(shard_elems / band_elems);
    if (!bands_tile_shard(shard_elems, bands, band_elems)) return false;

    std::lock_guard<std::mutex> lock(g_mu);

    const std::vector<void *> *bases = peer_bases(args.peer_ub, args.peer_count);
    if (!bases) return false;
    PeerPtrs peers{};
    for (int c = 0; c < tp_size; c++) {
        peers.base[c] = static_cast<bf16 *>((*bases)[(args.peer_first + c) % args.peer_count]);
    }
    peers.base[args.rank] = static_cast<bf16 *>(args.ub);

    if (args.arrive_peers && args.arrive_local) {
        ag_ready_kernel<<<1, 64, 0, args.stream>>>(
            static_cast<void *const *>(const_cast<void *>(args.arrive_peers)), args.arrive_offset,
            static_cast<const char *>(args.arrive_local), args.arrive_stride, args.arrive_value,
            args.peer_first, args.peer_count, tp_size, ag_ready_warn_ticks());
    }

    rs_globals g{
        _gl_A(static_cast<bf16 *>(const_cast<void *>(args.B)), 1, 1, K, M),
        _gl_B(static_cast<bf16 *>(const_cast<void *>(args.A)), 1, 1, K, N),
        _gl_C(static_cast<bf16 *>(args.D), 1, 1, M, N),
        gl<float, -1, -1, -1, -1>(static_cast<float *>(args.workspace), 1, 1, splits * M, N),
        splits, nred, bands, bands, tp_size, args.rank, peers, shard_elems, band_elems,
        args.stream};
    dispatch(g);
    return hipGetLastError() == hipSuccess;
}

// Fused TN GEMM + reduce-scatter
struct RsPlan {
    void *queue = nullptr;
    int num_tiles  = 0;
};

std::map<PlanKey, RsPlan> g_rs_plans;

std::map<const void *, uint64_t> g_rs_epoch;

int rs_comm_wg_tn(int tokens, int hidden, int k_local) {
    const int tiles = (tokens / hk_rs_tn::BLOCK_ROW) * (hidden / hk_rs_tn::BLOCK_COL);
    const int waves = tiles / hk_rs_tn::GRID_CAP;
    if (k_local >= 4096) return 4;
    if (k_local >= 3072) return 7;
    if (k_local <= 768) return waves >= 16 ? 14 : (waves >= 8 ? 10 : COMM_WG);
    return COMM_WG;
}

bool run_fused_rs(const KittensRsGemmArgs &args) {
    using namespace hk_rs_tn;

    const int M       = args.n;
    const int N_TOTAL = args.m;
    const int K       = args.k;
    const int tp_size = args.nranks;
    const int bands   = (M / tp_size) / BLOCK_ROW;

    std::lock_guard<std::mutex> lock(g_mu);

    const PlanKey key{2, M, N_TOTAL, K, tp_size, args.rank, 1};
    auto it = g_rs_plans.find(key);
    if (it == g_rs_plans.end()) {
        RsPlan plan;
        auto queue = build_rs_work_queue(M, N_TOTAL, K, tp_size, args.rank);
        plan.num_tiles  = static_cast<int>(queue.size());
        if (!upload_plan(plan, queue)) return false;
        it = g_rs_plans.emplace(key, plan).first;
    }
    const RsPlan &plan = it->second;

    Carve ws{static_cast<char *>(args.workspace), 0, args.workspace_size};
    int *tile_counter  = static_cast<int *>(ws.take(sizeof(int)));
    const size_t counter_bytes = ws.used;
    if (!ws.fits()) return false;

    const FusedRsLayout lay = fused_rs_layout(args.shard_bytes, tp_size);

    const uint64_t epoch = g_rs_epoch[args.ub]++;
    const size_t stage_off = (epoch & 1ull) ? lay.recv_off : 0;

    const std::vector<void *> *bases = peer_bases(args.peer_ub, args.peer_count);
    if (!bases) return false;
    RsPeers peers{};
    for (int c = 0; c < tp_size; c++) {
        char *pb = static_cast<char *>((*bases)[(args.peer_first + c) % args.peer_count]);
        peers.stage[c]  = reinterpret_cast<bf16 *>(pb + stage_off);
    }
    char *lb = static_cast<char *>(args.ub);
    peers.stage[args.rank]  = reinterpret_cast<bf16 *>(lb + stage_off);

    bf16 *local_stage = peers.stage[args.rank];

    if (hipMemsetAsync(args.workspace, 0, counter_bytes, args.stream) != hipSuccess) return false;

    // Arms the sentinel, which is what orders a peer's stage read against its epilogue store.
    if (!sentinel_pattern_agrees()) return false;
    const size_t stage_dw = lay.stage_bytes / sizeof(unsigned int);
    if (hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(local_stage), RS_SENT_DW, stage_dw,
                          args.stream) != hipSuccess) {
        return false;
    }

    if (!args.arrive_peers || !args.arrive_local) return false;
    ag_ready_kernel<<<1, 64, 0, args.stream>>>(
        static_cast<void *const *>(const_cast<void *>(args.arrive_peers)), args.arrive_offset,
        static_cast<const char *>(args.arrive_local), args.arrive_stride, args.arrive_value,
        args.peer_first, args.peer_count, tp_size, ag_ready_warn_ticks());

    RsLaunchCfg cfg;
    cfg.comm_wg    = rs_comm_wg_tn(M, N_TOTAL, K);
    cfg.wb_group   = rs_wb_group(K);
    cfg.warn_ticks = ag_ready_warn_ticks();

    launch_persistent_rs(M, N_TOTAL, K, static_cast<bf16 *>(const_cast<void *>(args.A)),
                         static_cast<bf16 *>(const_cast<void *>(args.B)), local_stage,
                         static_cast<bf16 *>(args.D), static_cast<TileDesc *>(plan.queue),
                         plan.num_tiles, tile_counter, peers, args.rank, tp_size, cfg,
                         args.stream);
    return hipGetLastError() == hipSuccess;
}

// Shape and pointer requirements shared by rs entry points
static_assert(hk_rs_tn::BLOCK_ROW == 256 && hk_rs_tn::BLOCK_COL == 256 && hk_rs_tn::K_STEP == 64,
              "rs_guards_ok literals are stale against hk_rs_tn geometry");

bool rs_guards_ok(const KittensRsGemmArgs &args) {
    const int M       = args.n;
    const int N_TOTAL = args.m;
    const int K       = args.k;
    const int tp_size = args.nranks;
    const int bands   = (M / tp_size) / 256;

    // Order matters here
    const bool ok_tp    = (tp_size == 4 || tp_size == 8) && tp_size <= args.peer_count;
    const bool ok_rank  = args.rank >= 0 && args.rank < tp_size;
    const bool ok_m     = M % (tp_size * 256) == 0;
    const bool ok_n     = N_TOTAL % 256 == 0;
    const bool ok_k     = K % 128 == 0 && K >= 256;
    const bool ok_bands = bands >= 1;
    const bool ok_shard = args.shard_bytes != 0 &&
        args.shard_bytes == static_cast<size_t>(M) / tp_size * N_TOTAL * sizeof(uint16_t);
    const bool ok_ptrs  = args.workspace && args.ub && args.A && args.B && args.D && args.peer_ub;
    const bool ok = ok_tp && ok_rank && ok_m && ok_n && ok_k && ok_bands && ok_shard && ok_ptrs;

    // NVTE_RS_DIAG=1 reports which guard rejected a shape, and the geometry, once per call.
    if (!ok && std::getenv("NVTE_RS_DIAG")) {
        const FusedRsLayout l = fused_rs_layout(args.shard_bytes, tp_size);
        std::fprintf(stderr,
            "[RS_DIAG] M=%d N=%d K=%d tp=%d bands=%d shard=%zu stage=%zu total=%zu "
            "tp:%d rank:%d m:%d n:%d k:%d bands:%d shard:%d ptrs:%d\n",
            M, N_TOTAL, K, tp_size, bands, args.shard_bytes, l.stage_bytes, l.total_bytes,
            ok_tp, ok_rank, ok_m, ok_n, ok_k, ok_bands, ok_shard, ok_ptrs);
    }
    return ok;
}

// Shape and pointer requirements shared by ag entry points
bool ag_guards_ok(const KittensAgGemmArgs &args) {
    // args.n is the gathered token axis, so that is the axis that must split evenly across the TP
    // group. Spelled here in BLAS naming, which is TN's; NN puts that same axis on M. The
    // predicates are value-identical either way, since BLOCK_ROW == BLOCK_COL == 256.
    const int M       = args.m;
    const int N_TOTAL = args.n;
    const int K       = args.k;
    const int tp_size = args.nranks;

    // Order matters here
    return tp_size >= 1 && tp_size <= 8 && tp_size <= args.peer_count &&
           args.rank >= 0 && args.rank < tp_size &&
           N_TOTAL % tp_size == 0 && N_TOTAL % 256 == 0 && M % 256 == 0 &&
           K % 128 == 0 && K >= 256 && (N_TOTAL / tp_size) % 256 == 0 &&
           args.workspace && args.ub && args.A && args.D && args.peer_ub;
}

}  // namespace

void kittens_persistent_plans_reset_cdna4() {
    std::lock_guard<std::mutex> lock(g_mu);
    for (auto &kv : g_plans) {
        if (kv.second.queue) static_cast<void>(hipFree(kv.second.queue));
    }
    for (auto &kv : g_rs_plans) {
        if (kv.second.queue) static_cast<void>(hipFree(kv.second.queue));
    }
    g_plans.clear();
    g_rs_plans.clear();
    g_rs_epoch.clear();
    g_peers.clear();
}

bool kittens_fused_ag_gemm_bf16_cdna4(const KittensAgGemmArgs &args) {
    const int M       = args.n;
    const int K       = args.k;
    const int tp_size = args.nranks;

    if (!ag_guards_ok(args)) return false;
    if (args.chunk_bytes != static_cast<size_t>(M / tp_size) * K * sizeof(uint16_t)) return false;

    return args.transa ? run_tn(args) : run_nn(args);
}

bool kittens_fused_ag_gemm_mxfp8_cdna4(const KittensAgGemmArgs &args) {
    const int M       = args.n;
    const int K       = args.k;
    const int tp_size = args.nranks;

    if (!ag_guards_ok(args)) return false;
    // The gathered region IS the [M,K] A operand, so chunk_bytes is pinned to it exactly -- a
    // mis-sized region declines instead of silently gathering a fraction of itself.
    if (args.chunk_bytes != static_cast<size_t>(M / tp_size) * K * sizeof(uint8_t)) return false;

    return args.transa ? run_mxfp8<Mxfp8Tn>(args) : run_mxfp8<Mxfp8Nn>(args);
}

bool kittens_bulk_ag_gemm_mxfp8_cdna4(const KittensAgGemmArgs &args) {
    const int M       = args.n;
    const int N_TOTAL = args.m;
    const int tp_size = args.nranks;

    // NN only
    if (args.transa) return false;
    if (!args.gather_dst) return false;
    if (!ag_guards_ok(args)) return false;
    // The gathered tensor is MXFP8: one byte per element, plus one E8M0 scale per 32.
    if (args.chunk_bytes != static_cast<size_t>(M / tp_size) * N_TOTAL * sizeof(uint8_t)) {
        return false;
    }
    if (args.scale_chunk_bytes &&
        args.scale_chunk_bytes != static_cast<size_t>(M / tp_size) * (N_TOTAL / 32)) {
        return false;
    }

    return run_bulk_mxfp8(args);
}

bool kittens_bulk_ag_gemm_bf16_cdna4(const KittensAgGemmArgs &args) {
    const int M       = args.n;
    const int N_TOTAL = args.m;
    const int tp_size = args.nranks;

    // NN only
    if (args.transa) return false;
    if (!args.gather_dst) return false;
    if (!ag_guards_ok(args)) return false;
    if (args.chunk_bytes != static_cast<size_t>(M / tp_size) * N_TOTAL * sizeof(uint16_t)) {
        return false;
    }

    return run_bulk_nn(args);
}

bool kittens_bulk_rs_gemm_bf16_cdna4(const KittensRsGemmArgs &args) {
    const int tp_size = args.nranks;
    if ((tp_size != 4 && tp_size != 8) || tp_size > args.peer_count) return false;
    if (args.rank < 0 || args.rank >= tp_size) return false;
    if (!args.workspace || !args.ub || !args.A || !args.B || !args.D || !args.peer_ub) return false;
    if (args.shard_bytes == 0 || args.shard_bytes % sizeof(uint16_t) != 0) return false;
    if (args.shard_bytes != static_cast<size_t>(args.k) / tp_size * args.m * sizeof(uint16_t)) {
        return false;
    }

    return run_bulk_rs(args);
}

size_t kittens_fused_rs_region_bytes_cdna4(size_t chunk_bytes, int tp_size) {
    return fused_rs_layout(chunk_bytes, tp_size).total_bytes;
}

bool kittens_fused_rs_gemm_shape_ok_cdna4(int tokens, int hidden, int k, int tp_size) {
    if (tokens <= 0 || hidden <= 0 || k <= 0 || tp_size <= 0) return false;

    KittensRsGemmArgs args{};
    void *probe = &args;
    args.A          = probe;
    args.B          = probe;
    args.D          = probe;
    args.ub         = probe;
    args.peer_ub    = probe;
    args.workspace  = probe;
    args.peer_count = tp_size;
    args.m          = hidden;
    args.n          = tokens;
    args.k          = k;
    args.rank       = 0;
    args.nranks     = tp_size;
    args.shard_bytes =
        static_cast<size_t>(tokens) / tp_size * static_cast<size_t>(hidden) * sizeof(uint16_t);
    return rs_guards_ok(args);
}

bool kittens_fused_rs_gemm_bf16_cdna4(const KittensRsGemmArgs &args) {
    if (!rs_guards_ok(args)) return false;
    return run_fused_rs(args);
}
