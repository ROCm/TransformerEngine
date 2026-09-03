/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

#include "hip/hip_runtime.h"
#include "../fused_ag_gemm.h"
#include "fused_ag_gemm_tn.cuh"
#include "fused_ag_gemm_nn.cuh"
#include "fused_ag_mxfp8_gemm_tn.cuh"
#include "fused_ag_mxfp8_gemm_nn.cuh"
#include "../kittens_kernel_common.cuh"

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
// peer's userbuffer into the matching offset of its own. Offset-preserving, so the result is
// rank-major and identical to what all_gather_into_tensor produced.
//
// scale_base and chunk_bytes are always 16B multiples here -- scale_base is SB*K and chunk_bytes
// is m_local*(K/32) with m_local a multiple of 256 -- and run_mxfp8_tn refuses the launch
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
    int off[hk_ag_tn::NUM_XCDS_AFF] = {};
    int cnt[hk_ag_tn::NUM_XCDS_AFF] = {};
};

using PlanKey = std::array<int, 7>;   // core, M, N, K, tp_size, rank, S

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

bool run_tn(const KittensFusedAgGemmArgs &args) {
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

bool run_mxfp8_tn(const KittensFusedAgGemmArgs &args) {
    using namespace hk_mxfp8_ag_tn;
    
    const int M       = args.n;
    const int N_TOTAL = args.m;
    const int K       = args.k;
    const int tp_size = args.nranks;
    const int m_local = M / tp_size;
    const int tiles_m = M / BLOCK_ROW;
    const int tiles_n = N_TOTAL / BLOCK_COL;

    const int k_iters = K / K_STEP;
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

    // Weight scales are rank-local, so pack them now and let them overlap the gather below.
    launch_pack_scales<false, 32, 8>((const uint8_t *)args.scale_A, packed_sb, N_TOTAL, scale_K, k_iters, args.stream);

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
    // writing their own chunks.
    if (args.scale_chunk_bytes) {
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

    launch_pack_scales<false, 64, 4>((const uint8_t *)args.scale_B, packed_sa, M, scale_K, k_iters, args.stream);

    get_persistent_fn(M, N_TOTAL, K)(
        M, N_TOTAL, K, static_cast<fp8e4m3 *>(args.ub),
        static_cast<fp8e4m3 *>(const_cast<void *>(args.A)), static_cast<bf16 *>(args.D),
        packed_sa, packed_sb, static_cast<TileDesc *>(plan.queue), plan.num_tiles,
        tile_counter, peers, arrive, args.rank, tp_size, GATH_WG, m_local, args.chunk_bytes,
        plan.xcd_bucket, buckets, bucket_ctr, args.stream);
    return hipGetLastError() == hipSuccess;
}

struct NnSetup {
    const AgPlan *plan;
    hk_ag_nn::PeerPtrs peers;
    hk_ag_nn::XcdBuckets buckets;
    int *tile_counter;
    int *bucket_ctr;
    unsigned int *arrive;
    float *cw;
};

bool prepare_nn(const KittensFusedAgGemmArgs &args, int S, void *peer_local, NnSetup &out) {
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

int split_k_nn(const KittensFusedAgGemmArgs &args) {
    using namespace hk_ag_nn;
    const int M       = args.n;
    const int N_TOTAL = args.m;
    const int tiles_m = M / BLOCK_ROW;
    int S = select_split_k(tiles_m * (N_TOTAL / BLOCK_COL), args.workspace_size);
    if ((args.k / K_STEP) % (4 * S) != 0) S = 1;
    return S;
}

bool run_nn(const KittensFusedAgGemmArgs &args) {
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

bool run_bulk_nn(const KittensFusedAgGemmArgs &args) {
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

// Shape and pointer requirements shared by all entry points
bool guards_ok(const KittensFusedAgGemmArgs &args) {
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

void kittens_fused_ag_gemm_reset_cdna4() {
    std::lock_guard<std::mutex> lock(g_mu);
    for (auto &kv : g_plans) {
        if (kv.second.queue) static_cast<void>(hipFree(kv.second.queue));
    }
    g_plans.clear();
    g_peers.clear();
}

bool kittens_fused_ag_gemm_bf16_cdna4(const KittensFusedAgGemmArgs &args) {
    const int M       = args.n;
    const int K       = args.k;
    const int tp_size = args.nranks;

    if (!guards_ok(args)) return false;
    if (args.chunk_bytes != static_cast<size_t>(M / tp_size) * K * sizeof(uint16_t)) return false;

    return args.transa ? run_tn(args) : run_nn(args);
}

bool run_mxfp8_nn(const KittensFusedAgGemmArgs &args) {
    using namespace hk_mxfp8_ag_nn;
    
    const int M       = args.n;
    const int N_TOTAL = args.m;
    const int K       = args.k;
    const int tp_size = args.nranks;
    const int m_local = M / tp_size;
    const int tiles_m = M / BLOCK_ROW;
    const int tiles_n = N_TOTAL / BLOCK_COL;

    const int k_iters = K / K_STEP;
    int scale_K = K / 32;

    // The scale region is sized in CommOverlapP2PBase::initialize; bail out rather than read
    // garbage if its chunking ever disagrees with this kernel's view of the operand. The 16B
    // alignment holds for every shape the eligibility gate admits, so drop to hipBLASLt rather
    // than silently mis-copying if it ever stops holding.
    if (args.scale_chunk_bytes &&
        (args.scale_chunk_bytes != static_cast<size_t>(m_local) * static_cast<size_t>(scale_K) ||
         args.scale_chunk_bytes % 16 != 0 || args.scale_base_offset % 16 != 0)) {
        return false;
    }

    std::lock_guard<std::mutex> lock(g_mu);

    const PlanKey key{2, M, N_TOTAL, K, tp_size, args.rank, 1};
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

    // Weight scales are rank-local, so pack them now and let them overlap the gather below.
    // NN consumes TE's A operand (this kernel's B) column-wise -- see rocm_comm_gemm_overlap.cpp:296.
    launch_pack_scales<true, 32, 8>((const uint8_t *)args.scale_A, packed_sb, N_TOTAL, scale_K, k_iters, args.stream);

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
    // in both, so this mirrors run_mxfp8_tn exactly.
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
                static_cast<char *>(args.ub), sp, args.rank, tp_size, args.scale_base_offset,
                args.scale_chunk_bytes);
    }

    launch_pack_scales<false, 64, 4>((const uint8_t *)args.scale_B, packed_sa, M, scale_K, k_iters, args.stream);

    get_persistent_fn(M, N_TOTAL, K)(
        M, N_TOTAL, K, static_cast<fp8e4m3 *>(args.ub),
        static_cast<fp8e4m3 *>(const_cast<void *>(args.A)), static_cast<bf16 *>(args.D),
        packed_sa, packed_sb, static_cast<TileDesc *>(plan.queue), plan.num_tiles,
        tile_counter, peers, arrive, args.rank, tp_size, GATH_WG, m_local, args.chunk_bytes,
        plan.xcd_bucket, buckets, bucket_ctr, args.stream);
    return hipGetLastError() == hipSuccess;
}

bool kittens_fused_ag_gemm_mxfp8_cdna4(const KittensFusedAgGemmArgs &args) {
    const int M       = args.n;
    const int K       = args.k;
    const int tp_size = args.nranks;

    if (!guards_ok(args)) return false;
    // The gathered region IS the [M,K] A operand, so chunk_bytes is pinned to it exactly -- a
    // mis-sized region declines instead of silently gathering a fraction of itself.
    if (args.chunk_bytes != static_cast<size_t>(M / tp_size) * K * sizeof(uint8_t)) return false;

    return args.transa ? run_mxfp8_tn(args) : run_mxfp8_nn(args);
}

bool kittens_bulk_ag_gemm_bf16_cdna4(const KittensFusedAgGemmArgs &args) {
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
