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

#include <array>
#include <cstdio>
#include <map>
#include <mutex>
#include <vector>

namespace {

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
// and discards it.
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

template <typename TD>
bool upload_plan(AgPlan &plan, const std::vector<TD> &queue) {
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

    const int M       = args.n;
    const int N_TOTAL = args.m;
    const int K       = args.k;
    const int tp_size = args.nranks;
    const int m_local = M / tp_size;
    const int tiles_m = M / BLOCK_ROW;
    const int tiles_n = N_TOTAL / BLOCK_COL;

    const int k_iters = K / BLOCK_K;
    int scale_K = K / 32;

    // The scale region is sized in CommOverlapP2PBase::initialize; bail out rather than read
    // garbage if its chunking ever disagrees with this kernel's view of the operand. The 16B
    // alignment holds for every shape the eligibility gate admits and is what lets gather_scales
    // run without a scalar tail, so drop to hipBLASLt rather than silently mis-copying if it ever
    // stops holding.
    if (args.scale_chunk_bytes &&
        (args.scale_chunk_bytes != static_cast<size_t>(m_local) * static_cast<size_t>(scale_K) ||
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

    // Lane-native scale buffers: A = 256 words/tile, B = 512 (hi/lo pair). If they overflow the
    // caller's budget we return false and fall back to hipBLASLt.
    size_t sa_bytes = kittens_align_up((size_t)k_iters * tiles_m * 256 * sizeof(uint32_t), 256);
    size_t sb_bytes = kittens_align_up((size_t)k_iters * tiles_n * 512 * sizeof(uint32_t), 256);

    const size_t arrive_bytes = static_cast<size_t>(tiles_m) * sizeof(unsigned int);
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
    launch_pack_scales<L::A_SCALE_COLWISE, 32, 8>((const uint8_t *)args.scale_A, packed_sb, N_TOTAL, scale_K, k_iters, args.stream);

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
        const int tiles_per_chunk = m_local / BLOCK_ROW;
        pack_local_scales_kernel<64, 4>
            <<<k_iters * tiles_per_chunk, 4 * 64, 0, args.stream>>>(
                (const uint8_t *)args.scale_B + (size_t)args.rank * args.scale_chunk_bytes,
                packed_sa, args.rank * tiles_per_chunk, tiles_per_chunk, tiles_m, scale_K, k_iters);
    } else {
        launch_pack_scales<false, 64, 4>((const uint8_t *)args.scale_B, packed_sa, M, scale_K, k_iters, args.stream);
    }

    auto launch = L::launch_fn(M, N_TOTAL, K, args.a_fp8_code, args.b_fp8_code);
    if (!launch) return false;   // unexpected operand format pair

    launch(
        M, N_TOTAL, K, static_cast<fp8e4m3 *>(args.ub),
        static_cast<fp8e4m3 *>(const_cast<void *>(args.A)), static_cast<bf16 *>(args.D),
        packed_sa, packed_sb, static_cast<TileDesc *>(plan.queue), plan.num_tiles,
        tile_counter, peers, arrive, args.rank, tp_size, GATH_WG, m_local, args.chunk_bytes,
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
    // shard rather than the A operand's.
    if (args.scale_chunk_bytes &&
        (args.scale_chunk_bytes % 16 != 0 || args.scale_base_offset % 16 != 0)) {
        return false;
    }

    auto bfn = get_persistent_bulk_fn(M, N_TOTAL, K, args.a_fp8_code, args.b_fp8_code);
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
    launch_pack_scales<L::A_SCALE_COLWISE, 32, 8>((const uint8_t *)args.scale_A, packed_sb,
                                                  N_TOTAL, scale_K, k_iters, args.stream);
    launch_pack_scales<false, 64, 4>((const uint8_t *)args.scale_B, packed_sa, M, scale_K, k_iters,
                                     args.stream);

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

// Shape and pointer requirements shared by all entry points
bool guards_ok(const KittensAgGemmArgs &args) {
    const int M       = args.n;
    const int N_TOTAL = args.m;
    const int K       = args.k;
    const int tp_size = args.nranks;

    // Order matters here
    return tp_size >= 1 && tp_size <= 8 && tp_size <= args.peer_count &&
           args.rank >= 0 && args.rank < tp_size &&
           M % tp_size == 0 && M % 256 == 0 && N_TOTAL % 256 == 0 &&
           K % 128 == 0 && K >= 256 && (M / tp_size) % 256 == 0 &&
           args.workspace && args.ub && args.A && args.D && args.peer_ub;
}

}  // namespace

void kittens_persistent_plans_reset_cdna4() {
    std::lock_guard<std::mutex> lock(g_mu);
    for (auto &kv : g_plans) {
        if (kv.second.queue) static_cast<void>(hipFree(kv.second.queue));
    }
    g_plans.clear();
    g_peers.clear();
}

bool kittens_fused_ag_gemm_bf16_cdna4(const KittensAgGemmArgs &args) {
    const int M       = args.n;
    const int K       = args.k;
    const int tp_size = args.nranks;

    if (!guards_ok(args)) return false;
    if (args.chunk_bytes != static_cast<size_t>(M / tp_size) * K * sizeof(uint16_t)) return false;

    return args.transa ? run_tn(args) : run_nn(args);
}

bool kittens_fused_ag_gemm_mxfp8_cdna4(const KittensAgGemmArgs &args) {
    const int M       = args.n;
    const int K       = args.k;
    const int tp_size = args.nranks;

    if (!guards_ok(args)) return false;
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
    if (!guards_ok(args)) return false;
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
    if (!guards_ok(args)) return false;
    if (args.chunk_bytes != static_cast<size_t>(M / tp_size) * N_TOTAL * sizeof(uint16_t)) {
        return false;
    }

    return run_bulk_nn(args);
}
