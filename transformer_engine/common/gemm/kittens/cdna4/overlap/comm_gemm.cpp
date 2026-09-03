/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

#include "hip/hip_runtime.h"
#include "../../comm_gemm.h"
#include "fused_ag_gemm_tn.cuh"
#include "fused_ag_gemm_nn.cuh"
#include "bulk_rs_gemm_nt.cuh"
#include "fused_rs_gemm_tn.cuh"

#include <array>
#include <cstdio>
#include <map>
#include <mutex>
#include <vector>

namespace {

constexpr size_t kFusedRsCtrlHalf  = 32u << 10;
constexpr size_t kFusedRsCtrlBytes = 2 * kFusedRsCtrlHalf;

struct FusedRsLayout {
    size_t stage_bytes;
    size_t recv_off;
    size_t arrive_off;
    size_t ready_off;
    size_t total_bytes;
};

inline FusedRsLayout fused_rs_layout(size_t shard_bytes, int tp_size) {
    FusedRsLayout l{};
    l.stage_bytes = shard_bytes * static_cast<size_t>(tp_size);
    l.recv_off    = l.stage_bytes;
    const size_t data = l.stage_bytes * 2;
    l.arrive_off  = (data + 4095) & ~static_cast<size_t>(4095);
    l.ready_off   = l.arrive_off + kFusedRsCtrlHalf;
    l.total_bytes = l.arrive_off + kFusedRsCtrlBytes;
    return l;
}


// 90 D3. The fill pattern here and the compare constant in the device object must not drift apart;
// a stale build fails silently, with every slot reading as already-arrived.
bool sentinel_pattern_agrees() {
    static_assert(static_cast<unsigned int>(RS_SENT_BF16) * 0x00010001u == RS_SENT_DW,
                  "90 D3: the 16-bit fill pattern and the 32-bit compare pattern disagree.");
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
    return (num_tiles <= 1024 && tiles_m >= 5) ? 1 : 0;
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

struct NnSetup {
    const AgPlan *plan;
    hk_ag_nn::PeerPtrs peers;
    hk_ag_nn::XcdBuckets buckets;
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
    int xcd_bucket = 0;
    int off[hk_rs_tn::NUM_XCDS_AFF] = {};
    int cnt[hk_rs_tn::NUM_XCDS_AFF] = {};
    int sched[2] = {};
};

std::map<PlanKey, RsPlan> g_rs_plans;

int rs_comm_wg_tn(int tokens, int hidden, int k_local, int tp_size) {
    if (tp_size != 8) return COMM_WG;
    if (hidden == 16384 && k_local == 6656 &&
        (tokens == 16384 || tokens == 32768 || tokens == 65536)) {
        return 4;
    }
    if (hidden == 8192 && k_local == 1024 && (tokens == 32768 || tokens == 65536)) {
        return 12;
    }
    if (hidden == 4096 && k_local == 512) {
        if (tokens == 65536) return 16;
        if (tokens == 32768) return 12;
    }
    return COMM_WG;
}

bool run_fused_rs(const KittensRsGemmArgs &args) {
    using namespace hk_rs_tn;

    const int M       = args.n;
    const int N_TOTAL = args.m;
    const int K       = args.k;
    const int tp_size = args.nranks;
    const int tiles_m = M / BLOCK_ROW;
    const int bands   = (M / tp_size) / BLOCK_ROW;

    std::lock_guard<std::mutex> lock(g_mu);

    const PlanKey key{2, M, N_TOTAL, K, tp_size, args.rank, 1};
    auto it = g_rs_plans.find(key);
    if (it == g_rs_plans.end()) {
        RsPlan plan;
        auto queue = build_rs_work_queue(M, N_TOTAL, K, tp_size, args.rank);
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
        plan.sched[0] = 0;
        plan.sched[1] = bands;
        if (!upload_plan(plan, queue)) return false;
        it = g_rs_plans.emplace(key, plan).first;
    }
    const RsPlan &plan = it->second;

    const size_t done_bytes = static_cast<size_t>(tp_size) * bands * sizeof(unsigned int);
    Carve ws{static_cast<char *>(args.workspace), 0, args.workspace_size};
    int *tile_counter  = static_cast<int *>(ws.take(sizeof(int)));
    int *bucket_ctr    = static_cast<int *>(ws.take(NUM_XCDS_AFF * sizeof(int)));
    unsigned int *done = static_cast<unsigned int *>(ws.take(done_bytes));
    const size_t counter_bytes = ws.used;
    if (!ws.fits()) return false;

    const FusedRsLayout lay = fused_rs_layout(args.shard_bytes, tp_size);

    const std::vector<void *> *bases = peer_bases(args.peer_ub, args.peer_count);
    if (!bases) return false;
    RsPeers peers{};
    for (int c = 0; c < tp_size; c++) {
        char *pb = static_cast<char *>((*bases)[(args.peer_first + c) % args.peer_count]);
        peers.stage[c]  = reinterpret_cast<bf16 *>(pb);
        peers.recv[c]   = reinterpret_cast<bf16 *>(pb + lay.recv_off);
        peers.arrive[c] = reinterpret_cast<unsigned int *>(pb + lay.arrive_off);
        peers.ready[c]  = reinterpret_cast<unsigned int *>(pb + lay.ready_off);
    }
    char *lb = static_cast<char *>(args.ub);
    peers.stage[args.rank]  = reinterpret_cast<bf16 *>(lb);
    peers.recv[args.rank]   = reinterpret_cast<bf16 *>(lb + lay.recv_off);
    peers.arrive[args.rank] = reinterpret_cast<unsigned int *>(lb + lay.arrive_off);
    peers.ready[args.rank]  = reinterpret_cast<unsigned int *>(lb + lay.ready_off);

    bf16 *local_stage = peers.stage[args.rank];

    XcdBuckets buckets{};
    for (int b = 0; b < NUM_XCDS_AFF; b++) {
        buckets.off[b] = plan.off[b];
        buckets.cnt[b] = plan.cnt[b];
    }

    if (hipMemsetAsync(args.workspace, 0, counter_bytes, args.stream) != hipSuccess) return false;
    if (hipMemsetAsync(lb + lay.arrive_off, 0, kFusedRsCtrlBytes, args.stream) != hipSuccess) {
        return false;
    }

    // Arms the arrival sentinel: the only edge ordering a comm workgroup's read of a peer's stage
    // against that peer's epilogue store. Must precede the rendezvous below.
    if (!sentinel_pattern_agrees()) return false;
    if (hipMemsetD16Async(reinterpret_cast<hipDeviceptr_t>(local_stage), RS_SENT_BF16,
                          lay.stage_bytes / sizeof(bf16), args.stream) != hipSuccess) {
        return false;
    }

    // Mandatory here, unlike the ag paths: it is what quiesces every rank's fill before any rank
    // reads, so without it the fill above orders nothing.
    if (!args.arrive_peers || !args.arrive_local) return false;
    ag_ready_kernel<<<1, 64, 0, args.stream>>>(
        static_cast<void *const *>(const_cast<void *>(args.arrive_peers)), args.arrive_offset,
        static_cast<const char *>(args.arrive_local), args.arrive_stride, args.arrive_value,
        args.peer_first, args.peer_count, tp_size, ag_ready_warn_ticks());

    RsLaunchCfg cfg;
    cfg.comm_wg   = rs_comm_wg_tn(M, N_TOTAL, K, tp_size);
    cfg.xcd_bucket = plan.xcd_bucket;

    launch_persistent_rs(M, N_TOTAL, K, static_cast<bf16 *>(const_cast<void *>(args.A)),
                         static_cast<bf16 *>(const_cast<void *>(args.B)), local_stage,
                         static_cast<bf16 *>(args.D), static_cast<TileDesc *>(plan.queue),
                         plan.num_tiles, tile_counter, peers, done, args.rank, tp_size, cfg,
                         buckets, bucket_ctr, args.stream);
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
    return (tp_size == 4 || tp_size == 8) && tp_size <= args.peer_count &&
           args.rank >= 0 && args.rank < tp_size &&
           M % (tp_size * 256) == 0 && N_TOTAL % 256 == 0 &&
           K % 128 == 0 && K >= 256 && bands >= 1 &&
           static_cast<size_t>(tp_size) * bands * sizeof(unsigned int) <= kFusedRsCtrlHalf &&
           args.shard_bytes != 0 &&
           args.shard_bytes == static_cast<size_t>(M) / tp_size * N_TOTAL * sizeof(uint16_t) &&
           args.workspace && args.ub && args.A && args.B && args.D && args.peer_ub;
}

// Shape and pointer requirements shared by ag entry points
bool ag_guards_ok(const KittensAgGemmArgs &args) {
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
    for (auto &kv : g_rs_plans) {
        if (kv.second.queue) static_cast<void>(hipFree(kv.second.queue));
    }
    g_plans.clear();
    g_rs_plans.clear();
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

bool kittens_fused_rs_gemm_eligible_cdna4(const KittensRsGemmArgs &args) {
    return rs_guards_ok(args);
}

bool kittens_fused_rs_gemm_bf16_cdna4(const KittensRsGemmArgs &args) {
    if (!kittens_fused_rs_gemm_eligible_cdna4(args)) return false;
    return run_fused_rs(args);
}
