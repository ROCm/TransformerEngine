/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

#include "hip/hip_runtime.h"
#include <cstdint>
#include <type_traits>
#include "common.cuh"

namespace te_kittens::cdna5 {

constexpr int K_STEP        = 32;
constexpr int WARPS_M       = 4;
constexpr int WARPS_N       = 2;
constexpr int WARP_M        = BLOCK_M / WARPS_M;
constexpr int WARP_N        = BLOCK_N / WARPS_N;
constexpr int CLUSTER_DIM   = 4;
constexpr int SCHED_DS_READ = 0x100;
constexpr int SCHED_WMMA    = 0x008;
constexpr int LDS_BYTES     = 4 * sizeof(kittens::st_bf<BF16_BLOCK_K, BLOCK_M>);

static_assert(BLOCK_N * staged_stride(sizeof(float)) <= LDS_BYTES && LDS_BYTES <= LDS_MAX_BYTES);

template <bool TRANS, int KS, typename RT, typename ST>
__device__ __forceinline__ void load_k_step(RT &dst, const ST &src, uint32_t off) {
    if constexpr (TRANS) {
        typedef short v8s __attribute__((ext_vector_type(8)));
        const uint32_t base = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(src.data)) + off;
#pragma unroll
        for (int ti = 0; ti < RT::height; ti++) {
#pragma unroll
            for (int x = 0; x < 2; x++) {
                *reinterpret_cast<v8s *>(&dst.tiles[ti][0].data[4 * x]) = __builtin_amdgcn_ds_load_tr16_b128_v8i16(
                    lds_ptr<v8s>(base + 2 * ST::idx(KS * K_STEP + 8 * x, 16 * ti)));
            }
        }
    } else {
        kittens::load(dst, src, int(off) + KS * K_STEP);
    }
}

template <bool TRANSA, bool TRANSB, bool MC, typename T, Epi E, bool ACC, typename OutT, typename SideT>
__global__ __cluster_dims__(MC ? CLUSTER_DIM : 0, MC ? CLUSTER_DIM : 0, MC ? 1 : 0) __launch_bounds__(NUM_THREADS, 1)
void bf16_gemm_kernel(const kittens::gl<T, -1, -1, -1, -1> ga, const kittens::gl<T, -1, -1, -1, -1> gb,
                      OutT *__restrict__ D, SideT *__restrict__ bias, SideT *__restrict__ aux, float alpha, float beta,
                      int M, int N, int K) {
    using ST_A = kittens::st<T, TRANSA ? BLOCK_M : BF16_BLOCK_K, TRANSA ? BF16_BLOCK_K : BLOCK_M,
                             kittens::ducks::st_shape::st_16x32_padded<>>;
    using ST_B = kittens::st<T, TRANSB ? BF16_BLOCK_K : BLOCK_N, TRANSB ? BLOCK_N : BF16_BLOCK_K,
                             kittens::ducks::st_shape::st_16x32_padded<>>;
    using RT_A = kittens::rt<T, WARP_M, K_STEP, kittens::ducks::rt_layout::row, kittens::ducks::rt_shape::rt_16x32>;
    using RT_B = kittens::rt<T, WARP_N, K_STEP, kittens::ducks::rt_layout::row, kittens::ducks::rt_shape::rt_16x32>;
    using RT_C = kittens::rt_fl<WARP_M, WARP_N, kittens::col_l, kittens::rt_16x16_s>;
    using IT   = std::conditional_t<std::is_same_v<T, kittens::half>, _Float16, __bf16>;
    using StT  = std::conditional_t<(E == NONE || E == BGRADB) && !ACC && std::is_same_v<OutT, IT>, OutT, float>;
    typedef IT v16in __attribute__((ext_vector_type(16)));
    typedef float v16f __attribute__((ext_vector_type(16)));
    constexpr bool CLUSTER = MC && ARCH_MCAST;

    extern __shared__ kittens::alignment_dummy __shm[];
    kittens::shared_allocator al(reinterpret_cast<int *>(&__shm[0]));
    ST_A &as0 = al.allocate_in<kittens::segment<0>, ST_A>();
    ST_B &bs0 = al.allocate_in<kittens::segment<0>, ST_B>();
    ST_A &as1 = al.allocate_in<kittens::segment<0>, ST_A>();
    ST_B &bs1 = al.allocate_in<kittens::segment<0>, ST_B>();
    ST_A *As[2] = {&as0, &as1};
    ST_B *Bs[2] = {&bs0, &bs1};

    kittens::sched::lock_simd();

    RT_C c;
    kittens::zero(c);
    float bsum[E == BGRADB ? RT_C::width : 1] = {};

    const int tile_m  = blockIdx.x;
    const int tile_n  = blockIdx.y;
    const int tiles_m = ceil_div(M, BLOCK_M);
    const int tiles_n = ceil_div(N, BLOCK_N);
    const int wid     = kittens::warpid();
    const int warp_r  = wid / WARPS_N;
    const int warp_c  = wid % WARPS_N;
    const int num_kb  = ceil_div(K, BF16_BLOCK_K);
    int kb0      = 0;
    int k_blocks = num_kb;
    if constexpr (E == SPLIT) {
        const int splits = gridDim.z;
        const int z      = blockIdx.z;
        kb0      = num_kb * z / splits;
        k_blocks = num_kb * (z + 1) / splits - kb0;
    }
    if (k_blocks <= 0) {
        return;
    }
    // A workgroup that ends leaves the cluster barrier's member count (ISA 5.6.6).
    if (tile_m >= tiles_m || tile_n >= tiles_n) {
        return;
    }

    const int lane = kittens::laneid();
    const int g    = lane / 8;
    const uint32_t a_off = TRANSA ? warp_r * WARP_M * BF16_BLOCK_K
                                  : 2 * ST_A::idx(lane % 8 + 16 * (g / 2), warp_r * WARP_M + 8 * (g % 2));
    const uint32_t b_off = TRANSB ? 2 * ST_B::idx(lane % 8 + 16 * (g / 2), warp_c * WARP_N + 8 * (g % 2))
                                  : warp_c * WARP_N * BF16_BLOCK_K;

    uint32_t a_mask = 0;
    uint32_t b_mask = 0;
    if constexpr (CLUSTER) {
        const uint32_t cx   = blockIdx.x % CLUSTER_DIM;
        const uint32_t cy   = blockIdx.y % CLUSTER_DIM;
        const uint32_t in_x = min(tiles_m - int(blockIdx.x - cx), CLUSTER_DIM);
        const uint32_t in_y = min(tiles_n - int(blockIdx.y - cy), CLUSTER_DIM);
        a_mask = (cluster_col_bits(CLUSTER_DIM, CLUSTER_DIM) << cx) & ((1u << (CLUSTER_DIM * in_y)) - 1u);
        b_mask = ((1u << in_x) - 1u) << (CLUSTER_DIM * cy);
    }

    const int m_rem  = M - tile_m * BLOCK_M;
    const int n_rem  = N - tile_n * BLOCK_N;
    const int k_head = K % BF16_BLOCK_K;
    const int back   = k_head ? BF16_BLOCK_K - k_head : 0;
    auto ga_k = ga;
    auto gb_k = gb;
    ga_k.raw_ptr -= size_t(back) * (TRANSA ? 1 : M);
    gb_k.raw_ptr -= size_t(back) * (TRANSB ? N : 1);
    const auto &ga0 = kb0 == 0 ? ga : ga_k;
    const auto &gb0 = kb0 == 0 ? gb : gb_k;
    const int kx0 = kb0 == 0 && k_head ? k_head : K;
    if (wid == 0) {
        kittens::tdm::load_async(*As[0], ga0, {0, 0, TRANSA ? tile_m : kb0, TRANSA ? kb0 : tile_m},
                                 TRANSA ? m_rem : kx0, TRANSA ? kx0 : m_rem, TRANSA ? K : M, a_mask, 1);
    }
    if (wid == 1) {
        kittens::tdm::load_async(*Bs[0], gb0, {0, 0, TRANSB ? kb0 : tile_n, TRANSB ? tile_n : kb0},
                                 TRANSB ? kx0 : n_rem, TRANSB ? n_rem : kx0, TRANSB ? N : K, b_mask, 1);
    }
    kittens::sync::wait_tdm<0>();
    kittens::sched::compiler_fence();
    if constexpr (CLUSTER) {
        kittens::cluster::sync();
    } else {
        kittens::sync::arrive();
        kittens::sync::wait();
    }
    kittens::sched::compiler_fence();

    RT_A a0, a1;
    RT_B b0, b1;
    load_k_step<!TRANSA, 0>(a0, *As[0], a_off);
    load_k_step<TRANSB, 0>(b0, *Bs[0], b_off);

    for (int kb = 0; kb < k_blocks; kb++) {
        const int cur = kb % 2;
        const int nxt = (kb + 1) % 2;

        const int fill_k = kb0 + (kb + 1 < k_blocks ? kb + 1 : 0);
        const uint32_t count = kb + 1 < k_blocks ? 1u : 0u;
        if (wid == 0) {
            kittens::tdm::load_async(*As[nxt], ga_k, {0, 0, TRANSA ? tile_m : fill_k, TRANSA ? fill_k : tile_m},
                                     TRANSA ? m_rem : K, TRANSA ? K : m_rem, TRANSA ? K : M, a_mask, count);
        }
        if (wid == 1) {
            kittens::tdm::load_async(*Bs[nxt], gb_k, {0, 0, TRANSB ? fill_k : tile_n, TRANSB ? tile_n : fill_k},
                                     TRANSB ? K : n_rem, TRANSB ? n_rem : K, TRANSB ? N : K, b_mask, count);
        }
        kittens::sched::compiler_fence();

        load_k_step<!TRANSA, 1>(a1, *As[cur], a_off);
        load_k_step<TRANSB, 1>(b1, *Bs[cur], b_off);
        kittens::mma_ABt(c, a0, b0, c);
#pragma unroll
        for (int i = 0; i < 4; i++) {
            __builtin_amdgcn_sched_group_barrier(SCHED_DS_READ, 6, 0);
            __builtin_amdgcn_sched_group_barrier(SCHED_WMMA, 8, 0);
        }
        kittens::sched::compiler_fence();
        if constexpr (E == BGRADB) {
            if (tile_m == 0 && warp_r == 0) {
#pragma unroll
                for (int t = 0; t < RT_B::height; t++) {
                    const v16f f = __builtin_convertvector(*reinterpret_cast<const v16in *>(b0.tiles[t][0].data), v16f);
#pragma unroll
                    for (int e = 0; e < 16; e += 2) {
                        bsum[t] += f[e] + f[e + 1];
                    }
                }
            }
        }

        load_k_step<!TRANSA, 2>(a0, *As[cur], a_off);
        load_k_step<TRANSB, 2>(b0, *Bs[cur], b_off);
        kittens::mma_ABt(c, a1, b1, c);
#pragma unroll
        for (int i = 0; i < 4; i++) {
            __builtin_amdgcn_sched_group_barrier(SCHED_DS_READ, 6, 0);
            __builtin_amdgcn_sched_group_barrier(SCHED_WMMA, 8, 0);
        }
        kittens::sched::compiler_fence();
        if constexpr (E == BGRADB) {
            if (tile_m == 0 && warp_r == 1) {
#pragma unroll
                for (int t = 0; t < RT_B::height; t++) {
                    const v16f f = __builtin_convertvector(*reinterpret_cast<const v16in *>(b1.tiles[t][0].data), v16f);
#pragma unroll
                    for (int e = 0; e < 16; e += 2) {
                        bsum[t] += f[e] + f[e + 1];
                    }
                }
            }
        }

        load_k_step<!TRANSA, 3>(a1, *As[cur], a_off);
        load_k_step<TRANSB, 3>(b1, *Bs[cur], b_off);
        kittens::mma_ABt(c, a0, b0, c);
#pragma unroll
        for (int i = 0; i < 4; i++) {
            __builtin_amdgcn_sched_group_barrier(SCHED_DS_READ, 6, 0);
            __builtin_amdgcn_sched_group_barrier(SCHED_WMMA, 8, 0);
        }
        kittens::sched::compiler_fence();
        if constexpr (E == BGRADB) {
            if (tile_m == 0 && warp_r == 2) {
#pragma unroll
                for (int t = 0; t < RT_B::height; t++) {
                    const v16f f = __builtin_convertvector(*reinterpret_cast<const v16in *>(b0.tiles[t][0].data), v16f);
#pragma unroll
                    for (int e = 0; e < 16; e += 2) {
                        bsum[t] += f[e] + f[e + 1];
                    }
                }
            }
        }

        kittens::mma_ABt_base<false, false>(c.tiles[0][0], a1.tiles[0][0], b1.tiles[0][0], c.tiles[0][0]);
        kittens::mma_ABt_base<true, false>(c.tiles[0][1], a1.tiles[0][0], b1.tiles[1][0], c.tiles[0][1]);
        kittens::mma_ABt_base<true, false>(c.tiles[0][2], a1.tiles[0][0], b1.tiles[2][0], c.tiles[0][2]);
        kittens::mma_ABt_base<true, false>(c.tiles[0][3], a1.tiles[0][0], b1.tiles[3][0], c.tiles[0][3]);
        kittens::mma_ABt_base<true, false>(c.tiles[0][4], a1.tiles[0][0], b1.tiles[4][0], c.tiles[0][4]);
        kittens::mma_ABt_base<true, false>(c.tiles[0][5], a1.tiles[0][0], b1.tiles[5][0], c.tiles[0][5]);
        kittens::mma_ABt_base<true, false>(c.tiles[0][6], a1.tiles[0][0], b1.tiles[6][0], c.tiles[0][6]);
        kittens::mma_ABt_base<true, false>(c.tiles[0][7], a1.tiles[0][0], b1.tiles[7][0], c.tiles[0][7]);
        kittens::mma_ABt_base<false, false>(c.tiles[1][7], a1.tiles[1][0], b1.tiles[7][0], c.tiles[1][7]);
        kittens::mma_ABt_base<true, false>(c.tiles[1][6], a1.tiles[1][0], b1.tiles[6][0], c.tiles[1][6]);
        kittens::mma_ABt_base<true, false>(c.tiles[1][5], a1.tiles[1][0], b1.tiles[5][0], c.tiles[1][5]);
        kittens::mma_ABt_base<true, false>(c.tiles[1][4], a1.tiles[1][0], b1.tiles[4][0], c.tiles[1][4]);
        kittens::sched::compiler_fence();

        kittens::sync::wait_ds<0>();
        kittens::sync::wait_tdm<0>();
        kittens::sched::compiler_fence();
        kittens::sync::arrive();
        if (CLUSTER && wid == 0) {
            kittens::cluster::arrive();
        }
        kittens::sched::compiler_fence();
        kittens::sync::wait();
        kittens::sched::compiler_fence();

        load_k_step<!TRANSA, 0>(a0, *As[nxt], a_off);
        load_k_step<TRANSB, 0>(b0, *Bs[nxt], b_off);
        kittens::mma_ABt_base<true, false>(c.tiles[1][3], a1.tiles[1][0], b1.tiles[3][0], c.tiles[1][3]);
        kittens::mma_ABt_base<true, false>(c.tiles[1][2], a1.tiles[1][0], b1.tiles[2][0], c.tiles[1][2]);
        kittens::mma_ABt_base<true, false>(c.tiles[1][1], a1.tiles[1][0], b1.tiles[1][0], c.tiles[1][1]);
        kittens::mma_ABt_base<true, false>(c.tiles[1][0], a1.tiles[1][0], b1.tiles[0][0], c.tiles[1][0]);
        kittens::mma_ABt_base<false, false>(c.tiles[2][0], a1.tiles[2][0], b1.tiles[0][0], c.tiles[2][0]);
        kittens::mma_ABt_base<true, false>(c.tiles[2][1], a1.tiles[2][0], b1.tiles[1][0], c.tiles[2][1]);
        kittens::mma_ABt_base<true, false>(c.tiles[2][2], a1.tiles[2][0], b1.tiles[2][0], c.tiles[2][2]);
        kittens::mma_ABt_base<true, false>(c.tiles[2][3], a1.tiles[2][0], b1.tiles[3][0], c.tiles[2][3]);
        kittens::mma_ABt_base<true, false>(c.tiles[2][4], a1.tiles[2][0], b1.tiles[4][0], c.tiles[2][4]);
        kittens::mma_ABt_base<true, false>(c.tiles[2][5], a1.tiles[2][0], b1.tiles[5][0], c.tiles[2][5]);
        kittens::mma_ABt_base<true, false>(c.tiles[2][6], a1.tiles[2][0], b1.tiles[6][0], c.tiles[2][6]);
        kittens::mma_ABt_base<true, false>(c.tiles[2][7], a1.tiles[2][0], b1.tiles[7][0], c.tiles[2][7]);
        kittens::mma_ABt_base<false, false>(c.tiles[3][7], a1.tiles[3][0], b1.tiles[7][0], c.tiles[3][7]);
        kittens::mma_ABt_base<true, false>(c.tiles[3][6], a1.tiles[3][0], b1.tiles[6][0], c.tiles[3][6]);
        kittens::mma_ABt_base<true, false>(c.tiles[3][5], a1.tiles[3][0], b1.tiles[5][0], c.tiles[3][5]);
        kittens::mma_ABt_base<true, false>(c.tiles[3][4], a1.tiles[3][0], b1.tiles[4][0], c.tiles[3][4]);
        kittens::mma_ABt_base<true, false>(c.tiles[3][3], a1.tiles[3][0], b1.tiles[3][0], c.tiles[3][3]);
        kittens::mma_ABt_base<true, false>(c.tiles[3][2], a1.tiles[3][0], b1.tiles[2][0], c.tiles[3][2]);
        kittens::mma_ABt_base<true, false>(c.tiles[3][1], a1.tiles[3][0], b1.tiles[1][0], c.tiles[3][1]);
        kittens::mma_ABt_base<true, false>(c.tiles[3][0], a1.tiles[3][0], b1.tiles[0][0], c.tiles[3][0]);
#pragma unroll
        for (int i = 0; i < 4; i++) {
            __builtin_amdgcn_sched_group_barrier(SCHED_DS_READ, 6, 0);
            __builtin_amdgcn_sched_group_barrier(SCHED_WMMA, 5, 0);
        }
        kittens::sched::compiler_fence();
        if constexpr (CLUSTER) {
            kittens::cluster::wait();
        }
        kittens::sched::compiler_fence();
        if constexpr (E == BGRADB) {
            if (tile_m == 0 && warp_r == 3) {
#pragma unroll
                for (int t = 0; t < RT_B::height; t++) {
                    const v16f f = __builtin_convertvector(*reinterpret_cast<const v16in *>(b1.tiles[t][0].data), v16f);
#pragma unroll
                    for (int e = 0; e < 16; e += 2) {
                        bsum[t] += f[e] + f[e + 1];
                    }
                }
            }
        }
    }

    kittens::sync::wait_tdm<0>();
    kittens::sync::wait_ds<0>();

    const int row0 = warp_r * WARP_M + 8 * (lane / 16);
    const int col0 = warp_c * WARP_N + lane % 16;
    const bool edge = (tile_m + 1) * BLOCK_M > M || (tile_n + 1) * BLOCK_N > N || M % 8 != 0;
    if constexpr (E == SPLIT) {
        float *ws = reinterpret_cast<float *>(D) + size_t(blockIdx.z) * M * N + size_t(tile_n) * BLOCK_N * M +
                    size_t(tile_m) * BLOCK_M;
#pragma unroll
        for (int i = 0; i < RT_C::height; i++) {
#pragma unroll
            for (int j = 0; j < RT_C::width; j++) {
                const auto &d = c.tiles[i][j].data;
                const int row = row0 + 16 * i;
                const int col = col0 + 16 * j;
                if (!edge || (row < m_rem && col < n_rem)) {
                    *reinterpret_cast<v8f *>(ws + size_t(col) * M + row) =
                        v8f{d[0].x, d[0].y, d[1].x, d[1].y, d[2].x, d[2].y, d[3].x, d[3].y};
                }
            }
        }
        return;
    }

    const uint32_t lds0 = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&__shm[0]));
    if constexpr (E == BGRADB) {
        if (tile_m == 0) {
            kittens::sync::wait_ds<0>();
            kittens::sync::sync();
#pragma unroll
            for (int t = 0; t < RT_C::width; t++) {
                const int p = (warp_r * 2 + lane / 16) * BLOCK_N + col0 + t * 16;
                *lds_ptr<float>(lds0 + p * sizeof(float)) = bsum[t];
            }
            kittens::sync::wait_ds<0>();
            kittens::sync::sync();
            float sum = 0.f;
#pragma unroll
            for (int p = 0; p < WARPS_M * 2; p++) {
                sum += *lds_ptr<float>(lds0 + (p * BLOCK_N + threadIdx.x) * sizeof(float));
            }
            if (int(threadIdx.x) < n_rem) {
                bias[tile_n * BLOCK_N + threadIdx.x] = static_cast<SideT>(sum);
            }
        }
    }

    kittens::sync::wait_ds<0>();
    kittens::sync::sync();
#pragma unroll
    for (int i = 0; i < RT_C::height; i++) {
#pragma unroll
        for (int j = 0; j < RT_C::width; j++) {
            const auto &d = c.tiles[i][j].data;
            const v8f v{d[0].x, d[0].y, d[1].x, d[1].y, d[2].x, d[2].y, d[3].x, d[3].y};
            const uint32_t addr = lds0 + (col0 + 16 * j) * staged_stride(sizeof(StT)) + (row0 + 16 * i) * sizeof(StT);
            *lds_ptr<vec8<StT>>(addr) = __builtin_convertvector(v, vec8<StT>);
        }
    }
    kittens::sync::wait_ds<0>();
    kittens::sync::sync();
    if (edge) {
        copy_out<StT, E, ACC, OutT, SideT, !TRANSA, true>(lds0, D, bias, aux, alpha, beta, M, N, tile_m, tile_n);
    } else {
        copy_out<StT, E, ACC, OutT, SideT, !TRANSA, false>(lds0, D, bias, aux, alpha, beta, M, N, tile_m, tile_n);
    }
}

template <bool TRANSA, bool TRANSB, typename T, Epi E, bool ACC, typename OutT, typename SideT>
static void launch_gemm(const GemmArgs &args, void *d, int splits, bool mcast) {
    const auto kernel = mcast ? bf16_gemm_kernel<TRANSA, TRANSB, true, T, E, ACC, OutT, SideT>
                              : bf16_gemm_kernel<TRANSA, TRANSB, false, T, E, ACC, OutT, SideT>;
    kittens_check(hipFuncSetAttribute(reinterpret_cast<const void *>(kernel),
                                      hipFuncAttributeMaxDynamicSharedMemorySize, LDS_BYTES),
                  "gfx1250 bf16_gemm_kernel LDS");
    const kittens::gl<T, -1, -1, -1, -1> ga(static_cast<T *>(const_cast<void *>(args.A)), 1, 1,
                                            TRANSA ? args.M : args.K, TRANSA ? args.K : args.M);
    const kittens::gl<T, -1, -1, -1, -1> gb(static_cast<T *>(const_cast<void *>(args.B)), 1, 1,
                                            TRANSB ? args.K : args.N, TRANSB ? args.N : args.K);
    const int side = mcast ? CLUSTER_DIM : 1;
    const dim3 grid(ceil_div(ceil_div(args.M, BLOCK_M), side) * side, ceil_div(ceil_div(args.N, BLOCK_N), side) * side,
                    splits);
    hipLaunchKernelGGL(kernel, grid, dim3(NUM_THREADS), LDS_BYTES, args.stream, ga, gb, static_cast<OutT *>(d),
                       static_cast<SideT *>(const_cast<void *>(args.bias)), static_cast<SideT *>(args.aux_gelu),
                       args.alpha, args.beta, args.M, args.N, args.K);
    kittens_check(hipGetLastError(), "gfx1250 bf16_gemm_kernel");
}

template <bool TRANSA, bool TRANSB, typename T>
static bool dispatch_gemm(const GemmArgs &args, void *workspace, int splits, Epi epi, bool mcast) {
    using IT = std::conditional_t<std::is_same_v<T, kittens::half>, _Float16, __bf16>;
    const int in_dtype = std::is_same_v<T, kittens::half> ? KITTENS_FLOAT16 : KITTENS_BFLOAT16;
    const int side     = args.bias != nullptr ? args.bias_dtype : args.aux_dtype;
    const bool acc = args.beta != 0.f;
    const bool f32 = args.out_dtype == KITTENS_FLOAT32;
    if (!f32 && args.out_dtype != in_dtype) {
        return false;
    }
    switch (epi) {
    case NONE:
        if (splits > 1) {
            launch_gemm<TRANSA, TRANSB, T, SPLIT, false, float, float>(args, workspace, splits, mcast);
            splitk_reduce(args, workspace, splits);
        } else if (f32 && acc) {
            launch_gemm<TRANSA, TRANSB, T, NONE, true, float, float>(args, args.C, 1, mcast);
        } else if (f32) {
            launch_gemm<TRANSA, TRANSB, T, NONE, false, float, float>(args, args.C, 1, mcast);
        } else if (acc) {
            launch_gemm<TRANSA, TRANSB, T, NONE, true, IT, IT>(args, args.C, 1, mcast);
        } else {
            launch_gemm<TRANSA, TRANSB, T, NONE, false, IT, IT>(args, args.C, 1, mcast);
        }
        return true;
    case BIAS:
        if (side != in_dtype || (acc && (f32 || !TRANSA))) {
            return false;
        }
        if (f32) {
            launch_gemm<TRANSA, TRANSB, T, BIAS, false, float, IT>(args, args.C, 1, mcast);
        } else if (!acc) {
            launch_gemm<TRANSA, TRANSB, T, BIAS, false, IT, IT>(args, args.C, 1, mcast);
        } else if constexpr (TRANSA) {
            launch_gemm<TRANSA, TRANSB, T, BIAS, true, IT, IT>(args, args.C, 1, mcast);
        }
        return true;
    case GELU_AUX:
        if constexpr (TRANSA) {
            if (side == in_dtype && !acc && !f32) {
                launch_gemm<TRANSA, TRANSB, T, GELU_AUX, false, IT, IT>(args, args.C, 1, mcast);
                return true;
            }
        }
        return false;
    case GELU_AUX_BIAS:
        if constexpr (TRANSA) {
            if (side == in_dtype && !acc && !f32) {
                launch_gemm<TRANSA, TRANSB, T, GELU_AUX_BIAS, false, IT, IT>(args, args.C, 1, mcast);
                return true;
            }
        }
        return false;
    case DGELU:
        if constexpr (!TRANSA && !TRANSB) {
            if (side == in_dtype && !acc && !f32) {
                launch_gemm<TRANSA, TRANSB, T, DGELU, false, IT, IT>(args, args.C, 1, mcast);
                return true;
            }
        }
        return false;
    case BGRADB:
        if constexpr (TRANSB) {
            if (side != args.out_dtype) {
                return false;
            }
            if (f32 && acc) {
                launch_gemm<TRANSA, TRANSB, T, BGRADB, true, float, float>(args, args.C, 1, mcast);
            } else if (f32) {
                launch_gemm<TRANSA, TRANSB, T, BGRADB, false, float, float>(args, args.C, 1, mcast);
            } else if (acc) {
                launch_gemm<TRANSA, TRANSB, T, BGRADB, true, IT, IT>(args, args.C, 1, mcast);
            } else {
                launch_gemm<TRANSA, TRANSB, T, BGRADB, false, IT, IT>(args, args.C, 1, mcast);
            }
            return true;
        }
        return false;
    default:
        return false;
    }
}

template <bool TRANSA, bool TRANSB>
bool launch_bf16(const GemmArgs &args, void *workspace, int splits, Epi epi, bool mcast) {
    if (args.a_dtype == KITTENS_FLOAT16) {
        return dispatch_gemm<TRANSA, TRANSB, kittens::half>(args, workspace, splits, epi, mcast);
    }
    return dispatch_gemm<TRANSA, TRANSB, kittens::bf16>(args, workspace, splits, epi, mcast);
}

template bool launch_bf16<true, false>(const GemmArgs &args, void *workspace, int splits, Epi epi, bool mcast);
template bool launch_bf16<false, false>(const GemmArgs &args, void *workspace, int splits, Epi epi, bool mcast);
template bool launch_bf16<false, true>(const GemmArgs &args, void *workspace, int splits, Epi epi, bool mcast);

}  // namespace te_kittens::cdna5
