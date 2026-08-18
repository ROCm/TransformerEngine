/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

#include "hip/hip_runtime.h"
#include "../fused_ag_gemm.h"
#include "fused_ag_gemm_tn.cuh"
#include "fused_ag_gemm_nn.cuh"

#include <array>
#include <map>
#include <mutex>
#include <vector>

namespace {

__global__ __launch_bounds__(64)
void ag_ready_kernel(void *const *__restrict__ peers, size_t offset,
                     const char *__restrict__ local, size_t stride, uint64_t value,
                     int first, int count, int tp_size) {
    const int c = static_cast<int>(threadIdx.x);
    if (c >= tp_size) return;

    uint64_t *pub = reinterpret_cast<uint64_t *>(
        static_cast<char *>(peers[(first + c) % count]) + offset);
    __hip_atomic_store(pub, value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);

    const uint64_t *f = reinterpret_cast<const uint64_t *>(local + static_cast<size_t>(c) * stride);
    while (__hip_atomic_load(f, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) < value) {
    }
    __hip_atomic_load(f, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}

struct AgPlan {
    void *queue   = nullptr;
    int num_tiles = 0;
    int xcd_aff   = 0;
    int off[hk_ag_tn::NUM_XCDS_AFF] = {};
    int cnt[hk_ag_tn::NUM_XCDS_AFF] = {};
};

using PlanKey = std::array<int, 7>;   // core, M, N, K, tp_size, rank, S

std::map<PlanKey, AgPlan> g_plans;
std::map<const void *, std::vector<void *>> g_peers;
std::mutex g_mu;

// comm->gpu_ptrs is device memory, so the peer bases have to come back over the bus. Once per
// registered region, not per call.
const std::vector<void *> *peer_bases(const void *peer_ub, int count) {
    auto it = g_peers.find(peer_ub);
    if (it != g_peers.end()) return &it->second;
    std::vector<void *> v(count);
    if (hipMemcpy(v.data(), peer_ub, count * sizeof(void *), hipMemcpyDeviceToHost) != hipSuccess) {
        return nullptr;
    }
    return &g_peers.emplace(peer_ub, std::move(v)).first->second;
}

// Peer-dedicated queues win when the queue is walked at most ~2 times and there are enough M-tiles
// to fill the buckets. Same gate as the harnesses, on the same GRID_CAP default.
int auto_xcd_aff(int num_tiles, int tiles_m) {
    const int grid_cap = getenv("GRID_CAP") ? atoi(getenv("GRID_CAP")) : 512;
    return (grid_cap > 0 && num_tiles <= 2 * grid_cap && tiles_m >= 5) ? 2 : 0;
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

bool run_c1(const KittensFusedAgGemmArgs &args) {
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
        auto queue     = build_work_queue(M, N_TOTAL, K, tp_size, args.rank, steps);
        plan.num_tiles = static_cast<int>(queue.size());
        plan.xcd_aff   = auto_xcd_aff(plan.num_tiles, tiles_m);
        if (plan.xcd_aff) {
            XcdBuckets bk{};
            queue = bucketize_by_xcd(queue, m_local / BLOCK_ROW, plan.xcd_aff, bk);
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
            args.peer_first, args.peer_count, tp_size);
    }

    get_persistent_fn(M, N_TOTAL, K)(
        M, N_TOTAL, K, static_cast<bf16 *>(args.ub),
        static_cast<bf16 *>(const_cast<void *>(args.A)), static_cast<bf16 *>(args.D),
        static_cast<TileDesc *>(plan.queue), plan.num_tiles, tile_counter, peers, arrive,
        args.rank, tp_size, GATH_WG, m_local, args.chunk_bytes, 1u, 0, plan.xcd_aff, 1, 1,
        buckets, bucket_ctr, 0, args.stream);
    return hipGetLastError() == hipSuccess;
}

bool run_c3(const KittensFusedAgGemmArgs &args) {
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
        auto queue     = build_work_queue(M, N_TOTAL, K, tp_size, args.rank, steps, S);
        plan.num_tiles = static_cast<int>(queue.size());
        plan.xcd_aff   = auto_xcd_aff(plan.num_tiles, tiles_m);
        if (plan.xcd_aff) {
            XcdBuckets bk{};
            queue = bucketize_by_xcd(queue, m_local / BLOCK_ROW, plan.xcd_aff, bk);
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
            args.peer_first, args.peer_count, tp_size);
    }

    pfn(M, N_TOTAL, K, static_cast<bf16 *>(args.ub),
        static_cast<bf16 *>(const_cast<void *>(args.A)), static_cast<bf16 *>(args.D), cw,
        static_cast<TileDesc *>(plan.queue), plan.num_tiles, tile_counter, peers, arrive,
        args.rank, tp_size, GATH_WG, m_local, args.chunk_bytes, 1u, 0, plan.xcd_aff, 1, 1,
        buckets, bucket_ctr, 0, args.stream);
    if (S > 1) launch_sk_reduce(cw, static_cast<bf16 *>(args.D), mn, S, args.stream);
    return hipGetLastError() == hipSuccess;
}

}  // namespace

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

    return args.transa ? run_c1(args) : run_c3(args);
}
