/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/
#pragma once

#include "hip/hip_runtime.h"
#include "kittens.cuh"
#include "overlap_common.cuh"
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace hk_rs_nt {

using namespace kittens;
using namespace hk_overlap;

constexpr int BLOCK_SIZE       = 256;
constexpr int HALF_BLOCK_SIZE  = BLOCK_SIZE / 2;
constexpr int WARPS_M          = 2;
constexpr int WARPS_N          = 4;
constexpr int REG_BLOCK_M      = BLOCK_SIZE / WARPS_M;
constexpr int REG_BLOCK_N      = BLOCK_SIZE / WARPS_N;
constexpr int HALF_REG_BLOCK_M = REG_BLOCK_M / 2;
constexpr int HALF_REG_BLOCK_N = REG_BLOCK_N / 2;

static_assert(WARPS_M * WARPS_N == NUM_WARPS, "bulk warp grid disagrees with hk_overlap");


constexpr int RS_COMM_WG_DEFAULT = 8;
constexpr int RS_BAND_ROWS       = 256;

using _gl_A = gl<bf16, -1, -1, -1, -1>;
using _gl_B = gl<bf16, -1, -1, -1, -1>;
using _gl_C = gl<bf16, -1, -1, -1, -1>;

using G = kittens::group<NUM_WARPS>;

// Per-PE pointer to each peer's Userbuffers region
struct PeerPtrs {
    bf16 *base[8];
};


__device__ __forceinline__
void rs_pull_fold(int w, int nred, int bands, int gband, int tp_size, int my_pe,
                  const PeerPtrs &peers, size_t shard_elems, size_t band_elems)
{
    typedef int v4i __attribute__((ext_vector_type(4)));
    const size_t lines = band_elems / 8;                  // one int4 = 8 bf16
    const size_t soff  = (size_t)my_pe * shard_elems;     // our shard, same offset on every rank
    bf16 *const out = peers.base[my_pe] + soff;

    for (int b0 = 0; b0 < bands; b0 += gband) {
        const int g = (b0 + gband <= bands) ? gband : (bands - b0);
        const size_t lines_g = lines * (size_t)g;
        const size_t per_g   = (lines_g + (size_t)nred - 1) / (size_t)nred;
        const size_t l0      = (size_t)w * per_g;
        if (l0 >= lines_g) continue;
        const size_t l1   = (l0 + per_g < lines_g) ? (l0 + per_g) : lines_g;
        const size_t boff = (size_t)b0 * band_elems;
        v4i *const dst = (v4i *)(out + boff);
        for (size_t l = l0 + threadIdx.x; l < l1; l += blockDim.x) {
            v4i acc;
#pragma unroll 1
            for (int s = 0; s < tp_size; s++) {
                const bf16 *base = peers.base[s] + soff + boff;
                const v4i v = ((const v4i *)base)[l];
                if (s == 0) {
                    acc = v;
                } else {
                    __hip_bfloat16 *a = reinterpret_cast<__hip_bfloat16 *>(&acc);
                    const __hip_bfloat16 *x = reinterpret_cast<const __hip_bfloat16 *>(&v);
#pragma unroll
                    for (int j = 0; j < 8; j++) a[j] = bf16_add(a[j], x[j]);
                }
            }
            __builtin_nontemporal_store(acc, &dst[l]);
        }
    }
    // Rendezvous the workgroup and retire its stores before the early return.
    __syncthreads();
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    __syncthreads();
}

struct rs_globals {
    _gl_A a;                       // dY, stored [K=tokens, M=out_local]
    _gl_B b;                       // X,  stored [K=tokens, N=hidden]
    _gl_C c;                       // dW,        [M=out_local, N=hidden]
    gl<float, -1, -1, -1, -1> w;   // fp32 split-K partials [splits*M, N]; unused at splits==1
    int splits;
    int nred;
    int bands;
    int gband;                     // bands folded per cycle; gband == bands is one pass
    int tp_size;
    int my_pe;
    PeerPtrs peers;
    size_t shard_elems;            // (tokens/tp) * hidden
    size_t band_elems;             // RS_BAND_ROWS * hidden
    hipStream_t stream;
    int M = c.rows();
    int N = c.cols();
    int K = a.rows();
    dim3 grid()  { return dim3((N / BLOCK_SIZE) * (M / BLOCK_SIZE) * splits + nred); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};

__global__ __launch_bounds__(NUM_THREADS, 2)
void wgrad_rs_tk(const rs_globals g, int M, int N, int K) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    using ST_AB = st_bf<K_STEP, HALF_BLOCK_SIZE, st_16x32_s>;   // natural [K, X]
    ST_AB (&As)[2][2] = al.allocate<ST_AB, 2, 2>();
    ST_AB (&Bs)[2][2] = al.allocate<ST_AB, 2, 2>();

    rt<bf16, K_STEP, HALF_REG_BLOCK_M, col_l, rt_16x32_s> A_tile;
    rt<bf16, K_STEP, HALF_REG_BLOCK_N, col_l, rt_16x32_s> B_tile_0;
    rt<bf16, K_STEP, HALF_REG_BLOCK_N, col_l, rt_16x32_s> B_tile_1;
    rt_fl<HALF_REG_BLOCK_M, HALF_REG_BLOCK_N, col_l, rt_32x32_s> C_accum[2][2];
    zero(C_accum[0][0]);
    zero(C_accum[0][1]);
    zero(C_accum[1][0]);
    zero(C_accum[1][1]);

    if ((int)blockIdx.x < g.nred) {
        rs_pull_fold((int)blockIdx.x, g.nred, g.bands, g.gband, g.tp_size, g.my_pe, g.peers,
                     g.shard_elems, g.band_elems);
        return;
    }

    int wgid = ((blockIdx.y * gridDim.x) + blockIdx.x) - g.nred;
    const int NUM_WGS  = (int)(gridDim.x * gridDim.y) - g.nred;

    // Split-K 
    const int splits   = g.splits;
    const int split_id = wgid % splits;
    wgid /= splits;
    const int WGM = 8;
    // Swizzle chiplet so that wgids are in the same XCD.
    wgid = chiplet_transform_chunked(wgid, NUM_WGS / splits, NUM_XCDS, 64);
    // Swizzle for better L2 within the same XCD.
    const int num_pid_m = ceil_div(M, BLOCK_SIZE);
    const int num_pid_n = ceil_div(N, BLOCK_SIZE);
    const int num_wgid_in_group = WGM * num_pid_n;
    int group_id = wgid / num_wgid_in_group;
    int first_pid_m = group_id * WGM;
    int group_size_m = min(num_pid_m - first_pid_m, WGM);
    int pid_m = first_pid_m + ((wgid % num_wgid_in_group) % group_size_m);
    int pid_n = (wgid % num_wgid_in_group) / group_size_m;
    int row = pid_m;
    int col = pid_n;

    const int warp_id = kittens::warpid();
    const int warp_row = warp_id / 4;
    const int warp_col = warp_id % 4;
    const int num_tiles = (K / K_STEP) / splits;
    const int kt_base   = split_id * num_tiles;

    const bf16* a_base = (bf16*)&g.a[{0, 0, 0, 0}];
    const bf16* b_base = (bf16*)&g.b[{0, 0, 0, 0}];
    const int a_row_stride = M * (int)sizeof(bf16);
    const int b_row_stride = N * (int)sizeof(bf16);
    i32x4 a_srsrc_base = make_srsrc(a_base, K * a_row_stride, a_row_stride);
    i32x4 b_srsrc_base = make_srsrc(b_base, K * b_row_stride, b_row_stride);

    const int wid = warpid() % NUM_WARPS;

    constexpr int B_INSTS = (HALF_BLOCK_SIZE * K_STEP * (int)sizeof(bf16)) / (16 * NUM_THREADS);
    uint32_t b_lane_off[B_INSTS];
    {
        const int lane = kittens::laneid();
#pragma unroll
        for (int i = 0; i < B_INSTS; i++) {
            const int q  = (wid + i * NUM_WARPS) * 64 + lane;
            const int W  = q / 32, within = q % 32;
            const int nb = W / 4,  w = W % 4;
            const int r  = within / 2, c0 = (within % 2) * 8;
            const int n  = nb * 16 + c0;
            const int k  = 32 * (w / 2) + 8 * (r / 4) + (r % 4) + 4 * (w % 2);
            b_lane_off[i] = (uint32_t)(((size_t)k * N + n) * sizeof(bf16));
        }
    }
    constexpr int epw_ab = (16 / sizeof(bf16)) * kittens::WARP_THREADS;

#define LDSB(t, i, j) static_cast<uint32_t>(__builtin_amdgcn_readfirstlane(static_cast<uint32_t>( \
        reinterpret_cast<uintptr_t>(&t[i][j].data[0]) + wid * epw_ab * sizeof(bf16))))
    uint32_t a_lds[2][2] = {{LDSB(As,0,0), LDSB(As,0,1)}, {LDSB(As,1,0), LDSB(As,1,1)}};
    uint32_t b_lds[2][2] = {{LDSB(Bs,0,0), LDSB(Bs,0,1)}, {LDSB(Bs,1,0), LDSB(Bs,1,1)}};
#undef LDSB

    using T = typename ST_AB::dtype;
    constexpr int bpt  = ST_AB::underlying_subtile_bytes_per_thread;
    constexpr int bpm  = bpt * NUM_THREADS;
    constexpr int cps  = K_STEP * BLOCK_SIZE * sizeof(T) / bpm;
    uint32_t sw_A[cps/2], sw_B[cps/2];
    G::prefill_swizzled_offsets(As[0][0], g.a, sw_A);
    G::prefill_swizzled_offsets(Bs[0][0], g.b, sw_B);


    int tic = 0;
    int toc = 1;


    G::load(Bs[tic][0], g.b, {0, 0, kt_base + (0), col * 2 + 0}, sw_B, b_srsrc_base, b_base, b_lds[tic][0]);
    G::load(As[tic][0], g.a, {0, 0, kt_base + (0), row * 2 + 0}, sw_A, a_srsrc_base, a_base, a_lds[tic][0]);
    G::load(Bs[tic][1], g.b, {0, 0, kt_base + (0), col * 2 + 1}, sw_B, b_srsrc_base, b_base, b_lds[tic][1]);
    G::load(As[tic][1], g.a, {0, 0, kt_base + (0), row * 2 + 1}, sw_A, a_srsrc_base, a_base, a_lds[tic][1]);

    if (warp_row == 1) {
        __builtin_amdgcn_s_barrier();
    }

    asm volatile("s_waitcnt vmcnt(4)");
    __builtin_amdgcn_s_barrier();

    G::load(Bs[toc][0], g.b, {0, 0, kt_base + (1), col * 2 + 0}, sw_B, b_srsrc_base, b_base, b_lds[toc][0]);
    G::load(As[toc][0], g.a, {0, 0, kt_base + (1), row * 2 + 0}, sw_A, a_srsrc_base, a_base, a_lds[toc][0]);
    G::load(Bs[toc][1], g.b, {0, 0, kt_base + (1), col * 2 + 1}, sw_B, b_srsrc_base, b_base, b_lds[toc][1]);

    asm volatile("s_waitcnt vmcnt(6)");
    __builtin_amdgcn_s_barrier();

#pragma unroll
    for (int tile = 0; tile < num_tiles - 2; tile += 2) {
        load(B_tile_0, subtile_inplace<K_STEP, HALF_REG_BLOCK_N>(Bs[0][0], {0, warp_col}));
        load(A_tile, subtile_inplace<K_STEP, HALF_REG_BLOCK_M>(As[0][0], {0, warp_row}));
        G::load(As[1][1], g.a, {0, 0, kt_base + (tile + 1), row * 2 + 1}, sw_A, a_srsrc_base, a_base, a_lds[1][1]);
        asm volatile("s_waitcnt lgkmcnt(8)");
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_AtB(C_accum[0][0], A_tile, B_tile_0, C_accum[0][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        load(B_tile_1, subtile_inplace<K_STEP, HALF_REG_BLOCK_N>(Bs[0][1], {0, warp_col}));
        G::load(Bs[0][0], g.b, {0, 0, kt_base + (tile + 2), col * 2 + 0}, sw_B, b_srsrc_base, b_base, b_lds[0][0]);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_AtB(C_accum[0][1], A_tile, B_tile_1, C_accum[0][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        load(A_tile, subtile_inplace<K_STEP, HALF_REG_BLOCK_M>(As[0][1], {0, warp_row}));
        G::load(As[0][0], g.a, {0, 0, kt_base + (tile + 2), row * 2 + 0}, sw_A, a_srsrc_base, a_base, a_lds[0][0]);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_AtB(C_accum[1][0], A_tile, B_tile_0, C_accum[1][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        asm volatile("s_waitcnt vmcnt(10)" ::: "memory");
        load(B_tile_0, subtile_inplace<K_STEP, HALF_REG_BLOCK_N>(Bs[1][0], {0, warp_col}));
        G::load(Bs[0][1], g.b, {0, 0, kt_base + (tile + 2), col * 2 + 1}, sw_B, b_srsrc_base, b_base, b_lds[0][1]);
        asm volatile("s_waitcnt vmcnt(6)");
        __builtin_amdgcn_s_barrier();

        __builtin_amdgcn_s_setprio(1);
        mma_AtB(C_accum[1][1], A_tile, B_tile_1, C_accum[1][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();


        load(A_tile, subtile_inplace<K_STEP, HALF_REG_BLOCK_M>(As[1][0], {0, warp_row}));
        G::load(As[0][1], g.a, {0, 0, kt_base + (tile + 2), row * 2 + 1}, sw_A, a_srsrc_base, a_base, a_lds[0][1]);
        asm volatile("s_waitcnt lgkmcnt(8)");
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_AtB(C_accum[0][0], A_tile, B_tile_0, C_accum[0][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        load(B_tile_1, subtile_inplace<K_STEP, HALF_REG_BLOCK_N>(Bs[1][1], {0, warp_col}));
        G::load(Bs[1][0], g.b, {0, 0, kt_base + (tile + 3), col * 2 + 0}, sw_B, b_srsrc_base, b_base, b_lds[1][0]);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_AtB(C_accum[0][1], A_tile, B_tile_1, C_accum[0][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        load(A_tile, subtile_inplace<K_STEP, HALF_REG_BLOCK_M>(As[1][1], {0, warp_row}));
        G::load(As[1][0], g.a, {0, 0, kt_base + (tile + 3), row * 2 + 0}, sw_A, a_srsrc_base, a_base, a_lds[1][0]);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_AtB(C_accum[1][0], A_tile, B_tile_0, C_accum[1][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        G::load(Bs[1][1], g.b, {0, 0, kt_base + (tile + 3), col * 2 + 1}, sw_B, b_srsrc_base, b_base, b_lds[1][1]);
        asm volatile("s_waitcnt vmcnt(6)");
        __builtin_amdgcn_s_barrier();

        __builtin_amdgcn_s_setprio(1);
        mma_AtB(C_accum[1][1], A_tile, B_tile_1, C_accum[1][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
    }

    {
        int tile = num_tiles - 2;

        load(B_tile_0, subtile_inplace<K_STEP, HALF_REG_BLOCK_N>(Bs[tic][0], {0, warp_col}));
        load(A_tile, subtile_inplace<K_STEP, HALF_REG_BLOCK_M>(As[tic][0], {0, warp_row}));
        G::load(As[toc][1], g.a, {0, 0, kt_base + (tile + 1), row * 2 + 1}, sw_A, a_srsrc_base, a_base, a_lds[toc][1]);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");

        __builtin_amdgcn_s_setprio(1);
        mma_AtB(C_accum[0][0], A_tile, B_tile_0, C_accum[0][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        load(B_tile_1, subtile_inplace<K_STEP, HALF_REG_BLOCK_N>(Bs[tic][1], {0, warp_col}));
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_AtB(C_accum[0][1], A_tile, B_tile_1, C_accum[0][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        load(A_tile, subtile_inplace<K_STEP, HALF_REG_BLOCK_M>(As[tic][1], {0, warp_row}));
        asm volatile("s_waitcnt vmcnt(4)");
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_AtB(C_accum[1][0], A_tile, B_tile_0, C_accum[1][0]);
        mma_AtB(C_accum[1][1], A_tile, B_tile_1, C_accum[1][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        tic^=1, toc^=1;
    }

    {
        load(B_tile_0, subtile_inplace<K_STEP, HALF_REG_BLOCK_N>(Bs[tic][0], {0, warp_col}));
        load(A_tile, subtile_inplace<K_STEP, HALF_REG_BLOCK_M>(As[tic][0], {0, warp_row}));
        asm volatile("s_waitcnt vmcnt(2)");
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_AtB(C_accum[0][0], A_tile, B_tile_0, C_accum[0][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        load(B_tile_1, subtile_inplace<K_STEP, HALF_REG_BLOCK_N>(Bs[tic][1], {0, warp_col}));
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_AtB(C_accum[0][1], A_tile, B_tile_1, C_accum[0][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        load(A_tile, subtile_inplace<K_STEP, HALF_REG_BLOCK_M>(As[tic][1], {0, warp_row}));
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_AtB(C_accum[1][0], A_tile, B_tile_0, C_accum[1][0]);
        mma_AtB(C_accum[1][1], A_tile, B_tile_1, C_accum[1][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
    }

    if (warp_row == 0) {
        __builtin_amdgcn_s_barrier();
    }

    const int cr0 = (row * 2) * WARPS_M + warp_row;
    const int cr1 = cr0 + WARPS_M;
    const int cc0 = col * 2 * WARPS_N + warp_col;
    const int cc1 = cc0 + WARPS_N;
    if (splits == 1) {
        store(g.c, C_accum[0][0], {0, 0, cr0, cc0});
        store(g.c, C_accum[0][1], {0, 0, cr0, cc1});
        store(g.c, C_accum[1][0], {0, 0, cr1, cc0});
        store(g.c, C_accum[1][1], {0, 0, cr1, cc1});
    } else {
        const int so = split_id * (M / HALF_REG_BLOCK_M);
        store(g.w, C_accum[0][0], {0, 0, so + cr0, cc0});
        store(g.w, C_accum[0][1], {0, 0, so + cr0, cc1});
        store(g.w, C_accum[1][0], {0, 0, so + cr1, cc0});
        store(g.w, C_accum[1][1], {0, 0, so + cr1, cc1});
    }
}

// Deterministic split-K sum
__global__ void rs_reduce_splits(const float *__restrict__ w, uint16_t *__restrict__ c,
                                 size_t n, int splits) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acc = 0.f;
    for (int s = 0; s < splits; s++) acc += w[(size_t)s * n + i];
    union { float f; unsigned u; } cv; cv.f = acc;
    c[i] = (uint16_t)(cv.u >> 16);
}

static inline int select_split_k_shape(int M, int N, int K) {
    const int nt_tiles = (M / BLOCK_SIZE) * (N / BLOCK_SIZE);
    int splits = (nt_tiles <= 64) ? 4 : 1;
    if ((K / K_STEP) % splits != 0) splits = 1;
    return splits;
}


static inline bool bands_tile_shard(size_t shard_elems, int bands, size_t band_elems) {
    if (bands <= 0 || band_elems == 0) return false;
    if (band_elems % 8) return false;
    return (size_t)bands * band_elems == shard_elems;
}

static inline void dispatch(rs_globals g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void *)wgrad_rs_tk, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    wgrad_rs_tk<<<g.grid(), g.block(), mem_size, g.stream>>>(g, g.M, g.N, g.K);
    if (g.splits > 1) {
        const size_t n = (size_t)g.M * g.N;
        rs_reduce_splits<<<(unsigned)((n + 255) / 256), 256, 0, g.stream>>>(
            (const float *)g.w.raw_ptr, (uint16_t *)g.c.raw_ptr, n, g.splits);
    }
}

}  // namespace hk_rs_nt
