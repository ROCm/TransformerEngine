/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

#include "hip/hip_runtime.h"
#include "../../comm_gemm.h"
#include "../../kittens_kernel_common.cuh"
#include "../mxfp8_gemm_helper.cuh"
#include "fused_ag_gemm_tn.cuh"
#include "fused_ag_gemm_nn.cuh"
#include "bulk_rs_gemm_nt.cuh"
#include "fused_rs_gemm_tn.cuh"

#include <array>
#include <type_traits>
#include <cstdio>
#include <map>
#include <mutex>
#include <vector>

namespace {

using te_kittens::cdna4::mxfp8::launch_pack_scales;

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

// All-gather of the MXFP8 scale region: each rank copies every peer's scale chunk into the
// matching offset of its own userbuffer. run_mxfp8 refuses the launch unless base and chunk are
// 16B multiples, so the vector loop covers the whole copy and there is no scalar tail.
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

    if constexpr (U > 1) {
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
        if constexpr (NT) __builtin_nontemporal_store(src[i], &dst[i]);
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

// Packs ONLY the local chunk's scale rows, for the interleaved path: every peer chunk is packed
// in-kernel by the gatherer block that fetches it, but the local chunk has no gatherer.
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

// Gather+pack the activation scales inside the gatherer blocks instead of up front? Only pays at
// large K: the gatherer reads one contiguous k-range per scale row, worth 8x the cache lines at
// K=16384 but only 2x at K=4096, below which the per-tile pack inside the arrival gate costs more
// than the removed prologue saves.
int interleave_scales_enabled(int K) {
    return K >= 16384 ? 1 : 0;
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


// Per-layout bindings for the shared run_mxfp8 host sequence. The layout is one bit: TN puts the
// gathered activation in the B slot on N, NN in the A slot on M, per mxfp8_gemm.cpp's BLAS
// convention. A_SCALE_COLWISE is a separate question that happens to share the answer for both.
template <bool GATHERED>
struct Mxfp8Layout {
    using TileDesc = std::conditional_t<GATHERED, hk_mxfp8_ag_nn::TileDesc,
                                                  hk_mxfp8_ag_tn::TileDesc>;
    static constexpr int PLAN_TAG = GATHERED ? 3 : 2;
    static constexpr bool A_SCALE_COLWISE = GATHERED;
    static constexpr bool GATHERED_ON_M   = GATHERED;

    static std::vector<TileDesc> work_queue(int M, int N, int K, int tp, int pe) {
        if constexpr (GATHERED) {
            return hk_mxfp8_ag_nn::build_work_queue(M, N, K, tp, pe);
        } else {
            return hk_mxfp8_ag_tn::build_work_queue(M, N, K, tp, pe);
        }
    }
    static auto launch_fn(int M, int N, int K, KittensDType a, KittensDType b) {
        if constexpr (GATHERED) {
            return hk_mxfp8_ag_nn::get_persistent_fn(M, N, K, a, b);
        } else {
            return hk_mxfp8_ag_tn::get_persistent_fn(M, N, K, a, b);
        }
    }
};

using Mxfp8Tn = Mxfp8Layout<false>;
using Mxfp8Nn = Mxfp8Layout<true>;

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

    // The bf16 pair steps 64, so the mxfp8 step of 128 stays with the mxfp8 kernels.
    constexpr int BLOCK_K = hk_mxfp8_ag_tn::BLOCK_K;
    static_assert(BLOCK_K == hk_mxfp8_ag_nn::BLOCK_K, "mxfp8 TN/NN must agree on the K step");

    // TN keeps mxfp8_gemm.cpp's BLAS convention (A = weight on M, B = gathered activation on N);
    // NN swaps the slots. Axis extents, xcd bucketing, arrival flags, the 64/4 <-> 32/8 pack split
    // and the operand pointers all follow the gathered operand; everything else is shared.
    const int M       = L::GATHERED_ON_M ? args.n : args.m;
    const int N_TOTAL = L::GATHERED_ON_M ? args.m : args.n;
    const int K       = args.k;
    const int tp_size = args.nranks;
    const int tiles_m = M / BLOCK_ROW;
    const int tiles_n = N_TOTAL / BLOCK_COL;
    // Per-rank shard of the gathered axis and that axis in tiles: m_local/tiles_m for NN,
    // n_local/tiles_n for TN. The gathered axis is args.n either way; only its name moves.
    const int gath_local = (L::GATHERED_ON_M ? M : N_TOTAL) / tp_size;
    const int tiles_gath = L::GATHERED_ON_M ? tiles_m : tiles_n;

    const int k_iters = K / BLOCK_K;
    int scale_K = K / 32;

    // The scale region is sized in CommOverlapP2PBase::initialize. Decline rather than mis-copy if
    // its chunking disagrees with this kernel's view.
    if (args.scale_chunk_bytes &&
        args.scale_chunk_bytes != static_cast<size_t>(gath_local) * static_cast<size_t>(scale_K)) {
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

    // Lane-native scale buffers: A = 256 words/tile, B = 512 (hi/lo pair).
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

    // Weight scales are rank-local: no dependency on the gather, so pack them up front. scale_A is
    // always the weight's; only the <STEP, NG> geometry and the destination follow the slot.
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

    // Second all-gather region, if the caller supplied one. Its own registered region, so its own
    // peer table; our slot is the destination, as in the bulk path.
    PeerPtrs aux_peers{};
    char *aux_dst = nullptr;
    if (args.aux_ag != nullptr) {
        const std::vector<void *> *aux_bases = peer_bases(args.aux_ag->peer_ub, args.peer_count);
        if (!aux_bases) return false;
        for (int c = 0; c < tp_size; c++) {
            aux_peers.base[c] =
                static_cast<fp8e4m3 *>((*aux_bases)[(args.peer_first + c) % args.peer_count]);
        }
        aux_peers.base[args.rank] = static_cast<fp8e4m3 *>(args.aux_ag->dst);
        aux_dst                   = static_cast<char *>(args.aux_ag->dst);
    }

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

    // The aux region has its own signal space -- nothing else advances it, since the communicator
    // that owns it never runs an overlap method of its own.
    if (args.aux_ag != nullptr && args.arrive_peers && args.aux_ag->arrive_local) {
        ag_ready_kernel<<<1, 64, 0, args.stream>>>(
            static_cast<void *const *>(const_cast<void *>(args.arrive_peers)),
            args.aux_ag->arrive_offset, static_cast<const char *>(args.aux_ag->arrive_local),
            args.aux_ag->arrive_stride, args.aux_ag->arrive_value, args.peer_first,
            args.peer_count, tp_size, ag_ready_warn_ticks());
    }

    // Aux scales, up front and unpacked: nothing in this kernel reads them, so there is no
    // lane-native format to interleave into, and the consumer is a later GEMM in TE layout.
    if (args.aux_ag != nullptr && args.aux_ag->scale_chunk_bytes) {
        ScalePeers sp{};
        for (int c = 0; c < tp_size; c++) {
            sp.base[c] = reinterpret_cast<const char *>(aux_peers.base[c]);
        }
        constexpr int SCALE_GATHER_U       = 4;
        constexpr int SCALE_GATHER_THREADS = 256;
        const size_t lines     = args.aux_ag->scale_chunk_bytes / 16;
        const size_t per_block = static_cast<size_t>(SCALE_GATHER_THREADS) * SCALE_GATHER_U;
        int grid_x = static_cast<int>((lines + per_block - 1) / per_block);
        if (grid_x < 1) grid_x = 1;
        if (grid_x > 256) grid_x = 256;
        gather_scales<SCALE_GATHER_U, false>
            <<<dim3(grid_x, tp_size), SCALE_GATHER_THREADS, 0, args.stream>>>(
                aux_dst, sp, args.rank, tp_size, args.aux_ag->scale_base_offset,
                args.aux_ag->scale_chunk_bytes);
    }

    // Activation scales live in the userbuffer, so they can only be packed after ag_ready_kernel
    // has established that the peers finished writing their chunks. Row-wise in both layouts.
    // scale_chunk_bytes != 0 means the peers' raw scales are reachable, which interleaving needs;
    // it is always true for a fused MXFP8 AG buffer, so the guard is defensive.
    const int interleave = args.scale_chunk_bytes ? interleave_scales_enabled(K) : 0;

    if (args.scale_chunk_bytes && !interleave) {
        ScalePeers sp{};
        for (int c = 0; c < tp_size; c++) sp.base[c] = reinterpret_cast<const char *>(peers.base[c]);

        constexpr int SCALE_GATHER_U = 4;
        constexpr int SCALE_GATHER_THREADS = 256;
        // NT stores off: launch_pack_scales reads this region back in the very next kernel.
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

    // CBSZ names the kernel's A-slot format and BLGP its B-slot, so the dtypes travel with the
    // pointers; args.a/b_dtype stay BLAS-canonical (weight / gathered activation).
    const KittensDType cbsz_dt = L::GATHERED_ON_M ? args.b_dtype : args.a_dtype;
    const KittensDType blgp_dt = L::GATHERED_ON_M ? args.a_dtype : args.b_dtype;
    auto launch = L::launch_fn(M, N_TOTAL, K, cbsz_dt, blgp_dt);
    if (!launch) return false;   // unexpected operand format pair

    fp8e4m3 *const d_weight = static_cast<fp8e4m3 *>(const_cast<void *>(args.A));
    fp8e4m3 *const d_gath   = static_cast<fp8e4m3 *>(args.ub);
    launch(
        M, N_TOTAL, K, L::GATHERED_ON_M ? d_gath : d_weight,
        L::GATHERED_ON_M ? d_weight : d_gath, static_cast<bf16 *>(args.D),
        packed_sa, packed_sb, static_cast<TileDesc *>(plan.queue), plan.num_tiles,
        tile_counter, peers, arrive, args.rank, tp_size, GATH_WG, gath_local, args.chunk_bytes,
        plan.xcd_bucket, buckets, bucket_ctr,
        args.scale_base_offset, args.scale_chunk_bytes, interleave, aux_peers, aux_dst,
        args.stream);
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

// Bulk sibling of run_mxfp8: the all-gather is NOT associated with the GEMM. It lands in
// args.gather_dst while the GEMM reads args.ub and args.A, so no compute block waits on an
// arrival and the gathered tensor's scales are moved but never packed.
bool run_bulk_mxfp8(const KittensAgGemmArgs &args) {
    using L = Mxfp8Nn;
    using namespace hk_mxfp8_ag_nn;
    using kittens::bf16;
    using kittens::fp8e4m3;

    constexpr int BLOCK_K = hk_mxfp8_ag_nn::BLOCK_K;

    // Same NN kernel as run_mxfp8<Mxfp8Nn>: A slot = gathered activation on M, B slot = weight.
    const int M       = args.n;
    const int N_TOTAL = args.m;
    const int K       = args.k;
    const int tp_size = args.nranks;
    const int m_local = M / tp_size;
    const int tiles_m = M / BLOCK_ROW;
    const int tiles_n = N_TOTAL / BLOCK_COL;
    const int k_iters = K / BLOCK_K;
    const int scale_K = K / 32;

    // The A slot holds the gathered activation, so the CBSZ/BLGP codes swap with the pointers.
    auto bfn = get_persistent_bulk_fn(M, N_TOTAL, K, args.b_dtype, args.a_dtype);
    if (!bfn) return false;   // unexpected operand format pair

    std::lock_guard<std::mutex> lock(g_mu);

    // The work queue depends only on output geometry and TP split, so bulk and fused share a plan.
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

    // The GEMM's own operand scales, unrelated to the gathered tensor: no dependency on the gather.
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
        gath_wg_bulk(K, tp_size), m_local, args.chunk_bytes, plan.xcd_bucket, buckets,
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
    // args.n is the gathered token axis, so that is what must split evenly across the TP group.
    // The predicates are value-identical under the TN/NN naming swap, since BLOCK_ROW == BLOCK_COL.
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
    // The gathered region IS the [M,K] A operand, so a mis-sized one declines rather than gather
    // a fraction of itself.
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
