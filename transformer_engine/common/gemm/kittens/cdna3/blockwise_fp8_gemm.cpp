/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

#include <type_traits>
#include "kittens.cuh"
#include "blockwise_fp8_gemm.h"

namespace blockwise_gfx942 {

#include "blockwise_fp8_gemm_helper.cuh"

constexpr int NUM_WARPS   = 8;
constexpr int WARPS_ROW   = 2;
constexpr int WARPS_COL   = 4;
constexpr int BLOCK_M     = 128;
constexpr int BLOCK_N     = 256;
constexpr int BLOCK_K     = 128;
constexpr int REG_M       = BLOCK_M / 4;
constexpr int REG_N       = BLOCK_N / 4;
constexpr int MFMA_K      = 32;
constexpr int SCALE_BLOCK = 128;
constexpr int NUM_THREADS = NUM_WARPS * kittens::WARP_THREADS;
constexpr size_t SMEM_BYTES = (BLOCK_M * BLOCK_K + BLOCK_N * BLOCK_K) * sizeof(kittens::fp8e4m3);

template <typename T> using _gl_A_t = kittens::gl<T, -1, -1, -1, -1>;
template <typename T> using _gl_B_t = kittens::gl<T, -1, -1, -1, -1>;
template <typename OType> using _gl_C_t = kittens::gl<OType, -1, -1, -1, -1>;
using _gl_SA = kittens::gl<float,   -1, -1, -1, -1>;
using _gl_SB = kittens::gl<float,   -1, -1, -1, -1>;

using G = kittens::group<NUM_WARPS>;

template <typename AType, typename BType, typename OType>
struct micro_globals {
    _gl_A_t<AType> a;
    _gl_B_t<BType> b;
    _gl_C_t<OType> c;
    _gl_SA scale_a;
    _gl_SB scale_b;
    hipStream_t stream;
    const void *bias;
    int bias_dtype;
    const void *gelu_aux;
    int gelu_aux_dtype;
    const OType *c_in;
    float beta;
    int M() const { return (int)c.rows(); }
    int N() const { return (int)c.cols(); }
    int K() const { return (int)a.cols(); }
    dim3 grid()  { return dim3(((N() + BLOCK_N - 1) / BLOCK_N) * ((M() + BLOCK_M - 1) / BLOCK_M)); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return SMEM_BYTES; }
};

template <typename AType, typename BType, typename OType, bool IS_PARTIAL_M, bool IS_PARTIAL_N,
          bool IS_1D2D, GemmEpilogue EPILOGUE = GemmEpilogue::DEFAULT,
          bool IS_PARTIAL_K = false>
__global__ __launch_bounds__(NUM_THREADS, 2)
void micro_tk(const micro_globals<AType, BType, OType> g) {
    extern __shared__ kittens::alignment_dummy __shm[];
    kittens::shared_allocator al((int*)&__shm[0]);
    kittens::st<AType, BLOCK_M, BLOCK_K> (&As) = al.allocate<kittens::st<AType, BLOCK_M, BLOCK_K>>();
    kittens::st<BType, BLOCK_N, BLOCK_K> (&Bs) = al.allocate<kittens::st<BType, BLOCK_N, BLOCK_K>>();

    kittens::rt<AType, REG_M, MFMA_K> at[5];
    kittens::rt<BType, REG_N, MFMA_K> bt[3];
    kittens::rt_fl<REG_M, REG_N, kittens::ducks::rt_layout::col> C_accum[2];
    kittens::rt_fl<REG_M, REG_N, kittens::ducks::rt_layout::col> partial[2];
    for (int i = 0; i < 2; i++) { kittens::zero(C_accum[i]); }

    const int M = (int)g.c.rows();
    const int N = (int)g.c.cols();
    const int K = (int)g.a.cols();

    int wgid = (blockIdx.y * gridDim.x) + blockIdx.x;
    const int NUM_WGS = gridDim.x * gridDim.y;
    constexpr int WGM = 4;
    wgid = kittens::chiplet_transform_chunked(wgid, NUM_WGS, kittens::NUM_XCDS, WGM*WGM);
    
    const int num_pid_m = kittens::ceil_div(M, BLOCK_M);
    const int num_pid_n = kittens::ceil_div(N, BLOCK_N);
    int num_wgid_in_group = WGM * num_pid_n;
    int group_id = wgid / num_wgid_in_group;
    int first_pid_m = group_id * WGM;
    int group_size_m = min(num_pid_m - first_pid_m, WGM);
    int pid_m = first_pid_m + ((wgid % num_wgid_in_group) % group_size_m);
    int pid_n = (wgid % num_wgid_in_group) / group_size_m;
    const int row = pid_m;
    const int col = pid_n;

    const bool is_last_m = IS_PARTIAL_M && (row * BLOCK_M + BLOCK_M > M);
    const bool is_last_n = IS_PARTIAL_N && (col * BLOCK_N + BLOCK_N > N);

    const int warp_id = kittens::warpid();
    const int warp_row = warp_id / WARPS_COL;
    const int warp_col = warp_id % WARPS_COL;

    const int num_k_steps = kittens::ceil_div(K, BLOCK_K);
    const bool is_k_partial = IS_PARTIAL_K && (K % BLOCK_K != 0);

    const float *sa_block = g.scale_a.raw_ptr + row * BLOCK_M;

    const int local_m0 = warp_row * REG_M;
    const int local_m1 = (warp_row + WARPS_ROW) * REG_M;
    const int tid = threadIdx.x;

    const uint32_t sa_range = (uint32_t)((M - row * BLOCK_M) * 4);

    const int n_scale_blocks = kittens::ceil_div(N, SCALE_BLOCK);
    const int sb_block0 = col * (BLOCK_N / SCALE_BLOCK) + warp_col / 2;
    const bool sb_valid = (!is_last_n) || (sb_block0 < n_scale_blocks);
    const float *sb_base = g.scale_b.raw_ptr + (sb_valid ? sb_block0 : 0) * num_k_steps;
    int32x4_lds_t sb_srsrc = make_buf_res((const void*)sb_base, (uint32_t)num_k_steps * 4);

    const int sb_col0 = col * BLOCK_N;
    const float *sb_block = g.scale_b.raw_ptr + sb_col0;
    const int local_n = warp_col * REG_N;
    const uint32_t sb_range = (uint32_t)((N - sb_col0) * 4);

    const bool is_first_k_partial = is_k_partial && (num_k_steps == 1);
    if (is_first_k_partial || is_last_m) {
        load_tile_masked<NUM_THREADS>(As, g.a, row, 0, M, K);
    } else {
        G::load(As, g.a, {0, 0, row, 0});
    }
    if (is_first_k_partial || is_last_n) {
        load_tile_masked<NUM_THREADS>(Bs, g.b, col, 0, N, K);
    } else {
        G::load(Bs, g.b, {0, 0, col, 0});
    }

    // Prologue
    float sb_cur = llvm_amdgcn_s_buffer_load_f32(sb_srsrc, 0, 0);
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();

    if (warp_row == 1) {
        __builtin_amdgcn_s_barrier();
    }

    #pragma unroll
    for (int k_step = 0; k_step < num_k_steps - 1; ++k_step) {

        constexpr int A_ELEMS_PER_THREAD = (BLOCK_M * BLOCK_K) / NUM_THREADS;
        constexpr int B_ELEMS_PER_THREAD = (BLOCK_N * BLOCK_K) / NUM_THREADS;
        float4 a_buffer_next[A_ELEMS_PER_THREAD * sizeof(AType) / sizeof(float4)];
        float4 b_buffer_next[B_ELEMS_PER_THREAD * sizeof(BType) / sizeof(float4)];

        kittens::zero(partial[0]); kittens::zero(partial[1]);

        const bool is_next_k_partial = is_k_partial && (k_step + 1 == num_k_steps - 1);
        float sa_reg0[REG_M / 16 * 4];
        float sa_reg1[REG_M / 16 * 4];

        // Cluster 0
        if (!is_last_n && !is_next_k_partial)
            kittens::load_global_to_register_buffer<2, false, NUM_THREADS>(b_buffer_next, B_ELEMS_PER_THREAD, g.b, {0, 0, col, k_step + 1}, Bs);
        float sb_next;
        float sb_reg[REG_N / 16]; // Unused in the 1d2d path; declared here for 1d1d
        kittens::load(at[0], kittens::subtile_inplace<REG_M, MFMA_K>(As, {warp_row, 0}));
        kittens::load(at[1], kittens::subtile_inplace<REG_M, MFMA_K>(As, {warp_row + WARPS_ROW, 0}));
        kittens::load(bt[0], kittens::subtile_inplace<REG_N, MFMA_K>(Bs, {warp_col, 0}));
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 1
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        kittens::mma_ABt(partial[0], at[0], bt[0], partial[0]);
        kittens::mma_ABt(partial[1], at[1], bt[0], partial[1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 2
        kittens::load(bt[1], kittens::subtile_inplace<REG_N, MFMA_K>(Bs, {warp_col, 1}));
        kittens::load(at[2], kittens::subtile_inplace<REG_M, MFMA_K>(As, {warp_row, 1}));
        kittens::load(at[3], kittens::subtile_inplace<REG_M, MFMA_K>(As, {warp_row + WARPS_ROW, 1}));
        kittens::load(bt[0], kittens::subtile_inplace<REG_N, MFMA_K>(Bs, {warp_col, 2}));
        kittens::load(at[0], kittens::subtile_inplace<REG_M, MFMA_K>(As, {warp_row, 2}));
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 3
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        kittens::mma_ABt(partial[0], at[2], bt[1], partial[0]);
        kittens::mma_ABt(partial[1], at[3], bt[1], partial[1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 4
        if (!is_last_m && !is_next_k_partial)
            kittens::load_global_to_register_buffer<2, false, NUM_THREADS>(a_buffer_next, A_ELEMS_PER_THREAD, g.a, {0, 0, row, k_step + 1}, As);
        kittens::load(at[1], kittens::subtile_inplace<REG_M, MFMA_K>(As, {warp_row + WARPS_ROW, 2}));
        kittens::load(bt[2], kittens::subtile_inplace<REG_N, MFMA_K>(Bs, {warp_col, 3}));
        kittens::load(at[4], kittens::subtile_inplace<REG_M, MFMA_K>(As, {warp_row, 3}));
        kittens::load(at[3], kittens::subtile_inplace<REG_M, MFMA_K>(As, {warp_row + WARPS_ROW, 3}));
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 5
        __builtin_amdgcn_s_setprio(1);
        kittens::mma_ABt(partial[0], at[0], bt[0], partial[0]);
        kittens::mma_ABt(partial[1], at[1], bt[0], partial[1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 6
        asm volatile("s_waitcnt lgkmcnt(0)");
        if (is_next_k_partial || is_last_m) {
            load_tile_masked<NUM_THREADS>(As, g.a, row, k_step + 1, M, K);
        } else {
            kittens::store_register_buffer_to_shared<NUM_THREADS>(As, a_buffer_next);
        }
        if (is_next_k_partial || is_last_n) {
            load_tile_masked<NUM_THREADS>(Bs, g.b, col, k_step + 1, N, K);
        }
        load_scale_global_reg<REG_M / 16>(sa_reg0, sa_block + k_step * M, local_m0, sa_range);
        load_scale_global_reg<REG_M / 16>(sa_reg1, sa_block + k_step * M, local_m1, sa_range);
        if constexpr (IS_1D2D)
            sb_next = llvm_amdgcn_s_buffer_load_f32(sb_srsrc, (k_step + 1) * 4, 0);
        else
            load_scaleB_global_reg<REG_N / 16>(sb_reg, sb_block + k_step * N, local_n, sb_range);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 7
        __builtin_amdgcn_s_setprio(1);
        kittens::mma_ABt(partial[0], at[4], bt[2], partial[0]);
        kittens::mma_ABt(partial[1], at[3], bt[2], partial[1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 8
        if (!(is_next_k_partial || is_last_n))
            kittens::store_register_buffer_to_shared<NUM_THREADS>(Bs, b_buffer_next);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 9
        if constexpr (IS_1D2D) {
            apply_block_scale_1d2d(C_accum[0], partial[0], sa_reg0, sb_cur);
            apply_block_scale_1d2d(C_accum[1], partial[1], sa_reg1, sb_cur);
            sb_cur = sb_next;
        } else {
            apply_block_scale_1d1d(C_accum[0], partial[0], sa_reg0, sb_reg);
            apply_block_scale_1d1d(C_accum[1], partial[1], sa_reg1, sb_reg);
        }
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

    }

    // Epilogue
    kittens::zero(partial[0]); kittens::zero(partial[1]);
    __builtin_amdgcn_sched_barrier(0);
    kittens::load(bt[0], kittens::subtile_inplace<REG_N, MFMA_K>(Bs, {warp_col, 0}));
    kittens::load(at[0], kittens::subtile_inplace<REG_M, MFMA_K>(As, {warp_row, 0}));
    kittens::load(at[1], kittens::subtile_inplace<REG_M, MFMA_K>(As, {warp_row + WARPS_ROW, 0}));
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    __builtin_amdgcn_s_setprio(1);
    kittens::mma_ABt(partial[0], at[0], bt[0], partial[0]);
    kittens::mma_ABt(partial[1], at[1], bt[0], partial[1]);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    kittens::load(bt[1], kittens::subtile_inplace<REG_N, MFMA_K>(Bs, {warp_col, 1}));
    kittens::load(at[2], kittens::subtile_inplace<REG_M, MFMA_K>(As, {warp_row, 1}));
    kittens::load(at[3], kittens::subtile_inplace<REG_M, MFMA_K>(As, {warp_row + WARPS_ROW, 1}));
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    __builtin_amdgcn_s_setprio(1);
    kittens::mma_ABt(partial[0], at[2], bt[1], partial[0]);
    kittens::mma_ABt(partial[1], at[3], bt[1], partial[1]);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    kittens::load(bt[0], kittens::subtile_inplace<REG_N, MFMA_K>(Bs, {warp_col, 2}));
    kittens::load(at[0], kittens::subtile_inplace<REG_M, MFMA_K>(As, {warp_row, 2}));
    kittens::load(at[1], kittens::subtile_inplace<REG_M, MFMA_K>(As, {warp_row + WARPS_ROW, 2}));
    kittens::load(bt[1], kittens::subtile_inplace<REG_N, MFMA_K>(Bs, {warp_col, 3}));
    kittens::load(at[2], kittens::subtile_inplace<REG_M, MFMA_K>(As, {warp_row, 3}));
    kittens::load(at[3], kittens::subtile_inplace<REG_M, MFMA_K>(As, {warp_row + WARPS_ROW, 3}));
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    __builtin_amdgcn_s_setprio(1);
    kittens::mma_ABt(partial[0], at[0], bt[0], partial[0]);
    kittens::mma_ABt(partial[1], at[1], bt[0], partial[1]);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    __builtin_amdgcn_s_setprio(1);
    kittens::mma_ABt(partial[0], at[2], bt[1], partial[0]);
    kittens::mma_ABt(partial[1], at[3], bt[1], partial[1]);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    {
        const int last = num_k_steps - 1;
        float sa_reg0[REG_M / 16 * 4];
        float sa_reg1[REG_M / 16 * 4];
        load_scale_global_reg<REG_M / 16>(sa_reg0, sa_block + last * M, local_m0, sa_range);
        load_scale_global_reg<REG_M / 16>(sa_reg1, sa_block + last * M, local_m1, sa_range);
        if constexpr (IS_1D2D) {
            apply_block_scale_1d2d(C_accum[0], partial[0], sa_reg0, sb_cur);
            apply_block_scale_1d2d(C_accum[1], partial[1], sa_reg1, sb_cur);
        } else {
            float sb_reg[REG_N / 16];
            load_scaleB_global_reg<REG_N / 16>(sb_reg, sb_block + last * N, local_n, sb_range);
            apply_block_scale_1d1d(C_accum[0], partial[0], sa_reg0, sb_reg);
            apply_block_scale_1d1d(C_accum[1], partial[1], sa_reg1, sb_reg);
        }
    }

    if (warp_row == 0) {
        __builtin_amdgcn_s_barrier();
    }

    if constexpr (EPILOGUE != GemmEpilogue::DEFAULT) {
        apply_epilogue<OType, EPILOGUE>(
            C_accum[0], row * 4 + warp_row,     col * 4 + warp_col, M, N,
            g.bias, g.bias_dtype, g.gelu_aux, g.gelu_aux_dtype, g.c_in, g.beta);
        apply_epilogue<OType, EPILOGUE>(
            C_accum[1], row * 4 + warp_row + WARPS_ROW, col * 4 + warp_col, M, N,
            g.bias, g.bias_dtype, g.gelu_aux, g.gelu_aux_dtype, g.c_in, g.beta);
    }

    if constexpr (std::is_same_v<OType, kittens::bf16>) {
        apply_rtne_bias(C_accum[0]);
        apply_rtne_bias(C_accum[1]);
    }
    if (is_last_m || is_last_n) {
        store_masked(g.c.raw_ptr, C_accum[0], row * 4 + warp_row,     col * 4 + warp_col, M, N);
        store_masked(g.c.raw_ptr, C_accum[1], row * 4 + warp_row + WARPS_ROW, col * 4 + warp_col, M, N);
    } else {
        kittens::store(g.c, C_accum[0], {0, 0, row * 4 + warp_row,     col * 4 + warp_col});
        kittens::store(g.c, C_accum[1], {0, 0, row * 4 + warp_row + WARPS_ROW, col * 4 + warp_col});
    }
}

#define BOOL_SWITCH(val, NAME, ...) \
    if (val) { constexpr bool NAME = true; __VA_ARGS__ } \
    else { constexpr bool NAME = false; __VA_ARGS__ }

static GemmEpilogue select_epilogue(bool has_bias, bool has_gelu, bool has_beta) {
    if (has_gelu) return has_beta ? GemmEpilogue::GELU_AUX_BETA : GemmEpilogue::GELU_AUX;
    if (has_bias) return has_beta ? GemmEpilogue::BIAS_BETA     : GemmEpilogue::BIAS;
    return has_beta ? GemmEpilogue::BETA : GemmEpilogue::DEFAULT;
}

template <bool IS_1D2D, typename AType, typename BType, typename OType,
          GemmEpilogue EPILOGUE, bool IS_PARTIAL_K>
static void dispatch_micro_epilogue(micro_globals<AType, BType, OType> g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    const bool is_partial_m = (g.M() % BLOCK_M != 0);
    const bool is_partial_n = (g.N() % BLOCK_N != 0);
    auto launch = [&](auto kern) {
        hipFuncSetAttribute((void*)kern, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
        kern<<<g.grid(), g.block(), mem_size, g.stream>>>(g);
    };
    BOOL_SWITCH(is_partial_m, IS_PARTIAL_M,
        BOOL_SWITCH(is_partial_n, IS_PARTIAL_N,
            launch(micro_tk<AType, BType, OType, IS_PARTIAL_M, IS_PARTIAL_N, IS_1D2D, EPILOGUE, IS_PARTIAL_K>);
        )
    )
}

template <bool IS_1D2D, typename AType, typename BType, typename OType, bool IS_PARTIAL_K>
static void dispatch_micro_k(micro_globals<AType, BType, OType> g,
                             bool has_bias, bool has_gelu, bool has_beta) {
    switch (select_epilogue(has_bias, has_gelu, has_beta)) {
        case GemmEpilogue::DEFAULT:
            dispatch_micro_epilogue<IS_1D2D, AType, BType, OType, GemmEpilogue::DEFAULT, IS_PARTIAL_K>(g); break;
        case GemmEpilogue::BIAS:
            dispatch_micro_epilogue<IS_1D2D, AType, BType, OType, GemmEpilogue::BIAS, IS_PARTIAL_K>(g); break;
        case GemmEpilogue::GELU_AUX:
            dispatch_micro_epilogue<IS_1D2D, AType, BType, OType, GemmEpilogue::GELU_AUX, IS_PARTIAL_K>(g); break;
        case GemmEpilogue::BETA:
            dispatch_micro_epilogue<IS_1D2D, AType, BType, OType, GemmEpilogue::BETA, IS_PARTIAL_K>(g); break;
        case GemmEpilogue::BIAS_BETA:
            dispatch_micro_epilogue<IS_1D2D, AType, BType, OType, GemmEpilogue::BIAS_BETA, IS_PARTIAL_K>(g); break;
        case GemmEpilogue::GELU_AUX_BETA:
            dispatch_micro_epilogue<IS_1D2D, AType, BType, OType, GemmEpilogue::GELU_AUX_BETA, IS_PARTIAL_K>(g); break;
    }
}

template <bool IS_1D2D, typename AType, typename BType, typename OType>
static void dispatch_micro(micro_globals<AType, BType, OType> g,
                           bool has_bias, bool has_gelu, bool has_beta, bool has_partial_k) {
    BOOL_SWITCH(has_partial_k, IS_PARTIAL_K,
        dispatch_micro_k<IS_1D2D, AType, BType, OType, IS_PARTIAL_K>(g, has_bias, has_gelu, has_beta);
    )
}

void kittens_blockwise_fp8_gemm_impl_cdna3(
    const void *A, const void *B, void *C,
    const void *scale_A, const void *scale_B,
    int M, int N, int K,
    bool transa, bool transb,
    int a_dtype, int b_dtype,
    int a_scaling_mode, int b_scaling_mode,
    int out_dtype,
    const void *bias, int bias_dtype,
    const void *gelu_aux, int gelu_aux_dtype,
    const void *c_in, float beta,
    hipStream_t stream) {

    // Dispatch passes canonical (A=weight/2D, B=activation/1D, M/N=user)
    // The kernel uses swapped layout
    const void *kA = B,          *kB = A;
    const void *ksa = scale_B,   *ksb = scale_A;
    void       *kC = C;
    const int   kM = N,          kN = M;
    const int   ka_mode = b_scaling_mode, kb_mode = a_scaling_mode;
    const int   ka_dtype = b_dtype,       kb_dtype = a_dtype;
    (void)transa; (void)transb;

    const bool is_1d2d   = (kb_mode == KITTENS_BLOCK_SCALING_2D);
    const bool has_bias  = (bias != nullptr);
    const bool has_gelu  = (gelu_aux != nullptr);
    const bool has_beta  = (c_in != nullptr);
    const bool has_partial_k = (K % BLOCK_K != 0);
    const int  k_blocks  = (K + BLOCK_K - 1) / BLOCK_K;

    auto run = [&]<typename AType, typename BType, typename OType>() {
        micro_globals<AType, BType, OType> g = {
            _gl_A_t<AType>(reinterpret_cast<AType*>(const_cast<void*>(kA)), 1, 1, kM, K),
            _gl_B_t<BType>(reinterpret_cast<BType*>(const_cast<void*>(kB)), 1, 1, kN, K),
            _gl_C_t<OType>(reinterpret_cast<OType*>(kC), 1, 1, kM, kN),
            _gl_SA(reinterpret_cast<float*>(const_cast<void*>(ksa)), 1, 1, k_blocks, kM),
            is_1d2d
                ? _gl_SB(reinterpret_cast<float*>(const_cast<void*>(ksb)), 1, 1, kittens::ceil_div(kN, SCALE_BLOCK), k_blocks)
                : _gl_SB(reinterpret_cast<float*>(const_cast<void*>(ksb)), 1, 1, k_blocks, kN),
            stream,
            bias, bias_dtype, gelu_aux, gelu_aux_dtype,
            reinterpret_cast<const OType*>(c_in), beta,
        };
        if (is_1d2d) dispatch_micro<true,  AType, BType, OType>(g, has_bias, has_gelu, has_beta, has_partial_k);
        else         dispatch_micro<false, AType, BType, OType>(g, has_bias, has_gelu, has_beta, has_partial_k);
    };

    const bool a_e5m2 = (ka_dtype == KITTENS_FP8E5M2);
    const bool b_e5m2 = (kb_dtype == KITTENS_FP8E5M2);
    auto run_ab = [&]<typename OType>() {
        if      (!a_e5m2 && !b_e5m2) run.template operator()<kittens::fp8e4m3, kittens::fp8e4m3, OType>();
        else if ( a_e5m2 && !b_e5m2) run.template operator()<kittens::fp8e5m2, kittens::fp8e4m3, OType>();
        else                         run.template operator()<kittens::fp8e4m3, kittens::fp8e5m2, OType>();
    };
    if      (out_dtype == KITTENS_FLOAT32) run_ab.template operator()<float>();
    else if (out_dtype == KITTENS_FLOAT16) run_ab.template operator()<kittens::half>();
    else                                   run_ab.template operator()<kittens::bf16>();
}

#undef BOOL_SWITCH

}  // namespace blockwise_gfx942
