/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

#include "hip/hip_runtime.h"
#include "../fused_ag_gemm.h"
#include "fused_ag_gemm_tn.cuh"
#include "fused_ag_gemm_nn.cuh"

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
        std::vector<StepInfo> steps;
        auto queue      = build_work_queue(M, N_TOTAL, K, tp_size, args.rank, steps);
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
        peers.b[c] = static_cast<bf16 *>((*bases)[(args.peer_first + c) % args.peer_count]);
    }
    peers.b[args.rank] = static_cast<bf16 *>(args.ub);

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
        args.rank, tp_size, GATH_WG, m_local, args.chunk_bytes, 1u, 0, plan.xcd_bucket, 1, 1,
        buckets, bucket_ctr, 0, args.stream);
    return hipGetLastError() == hipSuccess;
}

bool run_nn(const KittensFusedAgGemmArgs &args) {
    using namespace hk_ag_nn;

    const int M       = args.n;
    const int N_TOTAL = args.m;
    const int K       = args.k;
    const int tp_size = args.nranks;
    const int m_local = M / tp_size;
    const int tiles_m = M / BLOCK_ROW;

    // The workspace is handed over whole and untouched and select_split_k clamps it.
    int S = select_split_k(tiles_m * (N_TOTAL / BLOCK_COL), args.workspace_size);
    if ((K / K_STEP) % (4 * S) != 0) S = 1;

    persistent_fn_t pfn = get_persistent_fn(M, N_TOTAL, K, S);
    if (!pfn) return false;

    std::lock_guard<std::mutex> lock(g_mu);

    const PlanKey key{1, M, N_TOTAL, K, tp_size, args.rank, S};
    auto it = g_plans.find(key);
    if (it == g_plans.end()) {
        AgPlan plan;
        std::vector<StepInfo> steps;
        auto queue      = build_work_queue(M, N_TOTAL, K, tp_size, args.rank, steps, S);
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

    const size_t mn           = static_cast<size_t>(M) * N_TOTAL;
    const size_t arrive_bytes = static_cast<size_t>(tiles_m) * sizeof(unsigned int);
    Carve ws{static_cast<char *>(args.workspace), 0, args.workspace_size};
    int *tile_counter          = static_cast<int *>(ws.take(sizeof(int)));
    int *bucket_ctr            = static_cast<int *>(ws.take(NUM_XCDS_AFF * sizeof(int)));
    unsigned int *arrive       = static_cast<unsigned int *>(ws.take(arrive_bytes));
    const size_t counter_bytes = ws.used;
    float *cw                  = (S > 1) ? static_cast<float *>(ws.take(S * mn * sizeof(float)))
                                         : nullptr;
    if (!ws.fits()) return false;

    const std::vector<void *> *bases = peer_bases(args.peer_ub, args.peer_count);
    if (!bases) return false;
    PeerPtrs peers{};
    for (int c = 0; c < tp_size; c++) {
        peers.b[c] = static_cast<bf16 *>((*bases)[(args.peer_first + c) % args.peer_count]);
    }
    peers.b[args.rank] = static_cast<bf16 *>(args.ub);

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

    pfn(M, N_TOTAL, K, static_cast<bf16 *>(args.ub),
        static_cast<bf16 *>(const_cast<void *>(args.A)), static_cast<bf16 *>(args.D), cw,
        static_cast<TileDesc *>(plan.queue), plan.num_tiles, tile_counter, peers, arrive,
        args.rank, tp_size, GATH_WG, m_local, args.chunk_bytes, 1u, 0, plan.xcd_bucket, 1, 1,
        buckets, bucket_ctr, 0, args.stream);
    if (S > 1) launch_sk_reduce(cw, static_cast<bf16 *>(args.D), mn, S, args.stream);
    return hipGetLastError() == hipSuccess;
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
    const int N_TOTAL = args.m;
    const int K       = args.k;
    const int tp_size = args.nranks;

    if (tp_size < 1 || tp_size > 8 || tp_size > args.peer_count) return false;
    if (args.rank < 0 || args.rank >= tp_size) return false;
    if (M % tp_size != 0 || M % 256 != 0 || N_TOTAL % 256 != 0) return false;
    if (K % 128 != 0 || K < 256 || (M / tp_size) % 256 != 0) return false;
    if (args.chunk_bytes != static_cast<size_t>(M / tp_size) * K * sizeof(uint16_t)) return false;
    if (!args.workspace || !args.ub || !args.A || !args.D || !args.peer_ub) return false;

    return args.transa ? run_tn(args) : run_nn(args);
}
