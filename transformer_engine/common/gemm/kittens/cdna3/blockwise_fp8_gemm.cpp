/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

#include <type_traits>
#include "kittens.cuh"
#include "blockwise_fp8_gemm.h"
#include "../../../util/math.h"
using namespace kittens;

constexpr int BLOCK_M     = 128;
constexpr int BLOCK_N     = 256;
constexpr int BLOCK_K     = 128; // GEMM K dimension block size
constexpr int REG_M       = BLOCK_M / 4;
constexpr int REG_N       = BLOCK_N / 4;
constexpr int MFMA_K      = 32;
constexpr int SCALE_BLOCK = 128; // blockwise scale granularity

#define NUM_WARPS 8
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

template <typename T> using _gl_A_t = gl<T, -1, -1, -1, -1>;
template <typename T> using _gl_B_t = gl<T, -1, -1, -1, -1>;
template <typename OType> using _gl_C_t = gl<OType, -1, -1, -1, -1>;
using _gl_SA = gl<float,   -1, -1, -1, -1>;
using _gl_SB = gl<float,   -1, -1, -1, -1>;

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
    size_t dynamic_shared_memory() { return 49152; }
};

// --- SGPR/reg scale helpers ---
typedef int int32x4_lds_t __attribute__((ext_vector_type(4)));
struct __attribute__((packed)) buf_res { const void *ptr; uint32_t range; uint32_t config; };
__device__ inline int32x4_lds_t make_buf_res(const void *ptr, uint32_t size) {
    buf_res r{ptr, size, 0x00020000u};
    return __builtin_bit_cast(int32x4_lds_t, r);
}
extern "C" __device__ __uint128_t
llvm_amdgcn_raw_buffer_load_b128(int32x4_lds_t rsrc, int voffset, int soffset,
                                 int aux) __asm("llvm.amdgcn.raw.buffer.load.v4f32");
extern "C" __device__ float
llvm_amdgcn_s_buffer_load_f32(int32x4_lds_t rsrc, int offset,
                              int cachepolicy) __asm("llvm.amdgcn.s.buffer.load.f32");
extern "C" __device__ float
llvm_amdgcn_raw_buffer_load_f32(int32x4_lds_t rsrc, int voffset, int soffset,
                                int aux) __asm("llvm.amdgcn.raw.buffer.load.f32");

// SRD range_bytes bounds OOB rows to 0 (partial-M safe).
template <int HEIGHT>
__device__ inline void load_scale_global_reg(float (&sa_reg)[HEIGHT * 4], const float *sa_base,
                                             int local_m_base, uint32_t range_bytes) {
    const int lane = kittens::laneid();
    const int row_g = 4 * (lane / 16);
    int32x4_lds_t srsrc = make_buf_res((const void*)sa_base, range_bytes);
    #pragma unroll
    for (int i = 0; i < HEIGHT; i++) {
        const int m0 = local_m_base + i * 16 + row_g;
        __uint128_t raw = llvm_amdgcn_raw_buffer_load_b128(srsrc, m0 * 4, 0, 0);
        *reinterpret_cast<float4*>(&sa_reg[i * 4]) = *reinterpret_cast<float4*>(&raw);
    }
}

// 1D1D scale_B: per-N vector (lane-dependent col), so VGPR not SGPR.
// SRD range bounds OOB cols to 0 (partial-N safe).
template <int WIDTH>
__device__ inline void load_scaleB_global_reg(float (&sb_reg)[WIDTH], const float *sb_base,
                                              int local_n_base, uint32_t range_bytes) {
    const int lane = kittens::laneid();
    const int col_l = lane % 16;
    int32x4_lds_t srsrc = make_buf_res((const void*)sb_base, range_bytes);
    #pragma unroll
    for (int j = 0; j < WIDTH; j++) {
        const int n0 = local_n_base + j * 16 + col_l;
        sb_reg[j] = llvm_amdgcn_raw_buffer_load_f32(srsrc, n0 * 4, 0, 0);
    }
}

__device__ inline float rtne_bias(float v) {
    uint32_t bits = __builtin_bit_cast(uint32_t, v);
    if ((bits & 0x7f800000u) == 0x7f800000u) return v;
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return __builtin_bit_cast(float, bits);
}

__device__ inline float read_elem(const void *p, int dtype, int idx) {
    if (dtype == 6) return __bfloat162float(reinterpret_cast<const __hip_bfloat16 *>(p)[idx]);
    if (dtype == 5) return __half2float(reinterpret_cast<const __half *>(p)[idx]);
    return reinterpret_cast<const float *>(p)[idx];
}

template <typename OType>
__device__ inline float rtne_cast_roundtrip(float v) {
    if constexpr (std::is_same_v<OType, float>) {
        return v;
    } else {
        return static_cast<float>(base_types::convertor<OType, float>::convert(rtne_bias(v)));
    }
}

template <typename AccType>
__device__ inline void apply_rtne_bias(AccType &Cacc) {
    #pragma unroll
    for (int i = 0; i < AccType::height; i++) {
        #pragma unroll
        for (int j = 0; j < AccType::width; j++) {
            Cacc.tiles[i][j].data[0].x = rtne_bias(Cacc.tiles[i][j].data[0].x);
            Cacc.tiles[i][j].data[0].y = rtne_bias(Cacc.tiles[i][j].data[0].y);
            Cacc.tiles[i][j].data[1].x = rtne_bias(Cacc.tiles[i][j].data[1].x);
            Cacc.tiles[i][j].data[1].y = rtne_bias(Cacc.tiles[i][j].data[1].y);
        }
    }
}

template <typename OType, typename AccType>
__device__ inline void store_masked(OType *c_ptr, const AccType &Cacc,
                                      int Rtile, int Ctile, int M, int N) {
    const int lane = kittens::laneid();
    const int m_base = Rtile * AccType::rows + 4 * (lane / 16);
    const int n_base = Ctile * AccType::cols + (lane % 16);
    #pragma unroll
    for (int i = 0; i < AccType::height; i++) {
        const int m0 = m_base + i * 16;
        #pragma unroll
        for (int j = 0; j < AccType::width; j++) {
            const int col = n_base + j * 16;
            if (col >= N) continue;
            const float v0 = Cacc.tiles[i][j].data[0].x;
            const float v1 = Cacc.tiles[i][j].data[0].y;
            const float v2 = Cacc.tiles[i][j].data[1].x;
            const float v3 = Cacc.tiles[i][j].data[1].y;
            if (m0 + 0 < M) c_ptr[(m0 + 0) * N + col] = base_types::convertor<OType, float>::convert(v0);
            if (m0 + 1 < M) c_ptr[(m0 + 1) * N + col] = base_types::convertor<OType, float>::convert(v1);
            if (m0 + 2 < M) c_ptr[(m0 + 2) * N + col] = base_types::convertor<OType, float>::convert(v2);
            if (m0 + 3 < M) c_ptr[(m0 + 3) * N + col] = base_types::convertor<OType, float>::convert(v3);
        }
    }
}

template <typename OType, bool HAS_BIAS, bool HAS_GELU, bool HAS_BETA, typename AccType>
__device__ inline void apply_epilogue(
    AccType &Cacc, int Rtile, int Ctile, int M, int N,
        const void *bias, int bias_dtype,
        const void *gelu_aux, int gelu_aux_dtype,
        const OType *c_in, float beta) {
    const int lane = kittens::laneid();
    const int m_base = Rtile * AccType::rows + 4 * (lane / 16);
    const int n_base = Ctile * AccType::cols + (lane % 16);
    #pragma unroll
    for (int i = 0; i < AccType::height; i++) {
        const int m0 = m_base + i * 16;
        #pragma unroll
        for (int j = 0; j < AccType::width; j++) {
            const int col = n_base + j * 16;
            if (col >= N) continue;
            float v[4] = {
                Cacc.tiles[i][j].data[0].x, Cacc.tiles[i][j].data[0].y,
                Cacc.tiles[i][j].data[1].x, Cacc.tiles[i][j].data[1].y,
            };
            float bias_v = 0.f;
            if constexpr (HAS_BIAS) bias_v = read_elem(bias, bias_dtype, col);
            #pragma unroll
            for (int r = 0; r < 4; r++) {
                const int m_g = m0 + r;
                if (m_g >= M) continue;
                float x = v[r];
                if constexpr (HAS_BIAS) x += bias_v;
                if constexpr (HAS_BETA) {
                    x = rtne_cast_roundtrip<OType>(x);
                    x += beta * static_cast<float>(c_in[m_g * N + col]);
                }
                if constexpr (HAS_GELU) {
                    x *= transformer_engine::dgelu<float, float>(
                        read_elem(gelu_aux, gelu_aux_dtype, m_g * N + col), {});
                }
                v[r] = x;
            }
            Cacc.tiles[i][j].data[0].x = v[0];
            Cacc.tiles[i][j].data[0].y = v[1];
            Cacc.tiles[i][j].data[1].x = v[2];
            Cacc.tiles[i][j].data[1].y = v[3];
        }
    }
}

template <typename ST, typename GL>
__device__ inline void load_tile_masked(ST &dst, const GL &src, int row_blk,
                                        int k_blk, int row_dim, int K) {
    using T = typename ST::dtype;
    constexpr int elem_per_memcpy = sizeof(float4) / sizeof(T);
    constexpr int elem_per_half_memcpy = sizeof(float2) / sizeof(T);
    constexpr int memcpy_per_row = ST::cols / elem_per_memcpy;
    constexpr int total = (ST::rows * ST::cols) / elem_per_memcpy;
    const int row_stride = src.template stride<2>();
    const int row_base = row_blk * ST::rows;
    const int k_base = k_blk * ST::cols;
    kittens::coord<> uc = kittens::coord<ST>(0, 0, row_blk, k_blk).template unit_coord<2, 3>();
    T *src_ptr = (T *)&src[uc];
    uint32_t dst_ptr = reinterpret_cast<uintptr_t>(&dst.data[0]);
    const int tid = threadIdx.x;
    #pragma unroll
    for (int idx = tid; idx < total; idx += NUM_THREADS) {
        const int row = idx / memcpy_per_row;
        const int col = (idx % memcpy_per_row) * elem_per_memcpy;
        float4 v = {0.f, 0.f, 0.f, 0.f};
        if (row_base + row < row_dim && k_base + col < K)
            v = load_global_vec4((float4 *)(src_ptr + (row * row_stride + col)));
        store_shared_vec(dst.idx(dst_ptr, {row, col}), {v.x, v.y});
        store_shared_vec(dst.idx(dst_ptr, {row, col + elem_per_half_memcpy}), {v.z, v.w});
    }
    asm volatile("s_waitcnt lgkmcnt(0)");
}

template <typename AccType>
__device__ inline void apply_block_scale_1d2d(
    AccType &Cacc, const AccType &partial, const float *sa_lds, float sb, int local_m_base) {
    const int lane = kittens::laneid();
    const int row_g = 4 * (lane / 16);
    #pragma unroll
    for (int i = 0; i < AccType::height; i++) {
        const int m0 = local_m_base + i * 16 + row_g;
        const float s0 = sa_lds[m0 + 0] * sb;
        const float s1 = sa_lds[m0 + 1] * sb;
        const float s2 = sa_lds[m0 + 2] * sb;
        const float s3 = sa_lds[m0 + 3] * sb;
        #pragma unroll
        for (int j = 0; j < AccType::width; j++) {
            Cacc.tiles[i][j].data[0].x += partial.tiles[i][j].data[0].x * s0;
            Cacc.tiles[i][j].data[0].y += partial.tiles[i][j].data[0].y * s1;
            Cacc.tiles[i][j].data[1].x += partial.tiles[i][j].data[1].x * s2;
            Cacc.tiles[i][j].data[1].y += partial.tiles[i][j].data[1].y * s3;
        }
    }
}

template <typename AccType>
__device__ inline void apply_block_scale_1d2d_reg(
    AccType &Cacc, const AccType &partial, const float (&sa_reg)[AccType::height * 4], float sb) {
    #pragma unroll
    for (int i = 0; i < AccType::height; i++) {
        const float s0 = sa_reg[i * 4 + 0] * sb;
        const float s1 = sa_reg[i * 4 + 1] * sb;
        const float s2 = sa_reg[i * 4 + 2] * sb;
        const float s3 = sa_reg[i * 4 + 3] * sb;
        #pragma unroll
        for (int j = 0; j < AccType::width; j++) {
            Cacc.tiles[i][j].data[0].x += partial.tiles[i][j].data[0].x * s0;
            Cacc.tiles[i][j].data[0].y += partial.tiles[i][j].data[0].y * s1;
            Cacc.tiles[i][j].data[1].x += partial.tiles[i][j].data[1].x * s2;
            Cacc.tiles[i][j].data[1].y += partial.tiles[i][j].data[1].y * s3;
        }
    }
}

template <typename AccType>
__device__ inline void apply_block_scale_1d1d(
    AccType &Cacc, const AccType &partial, const float *sa_lds, const float *sb_lds,
        int local_m_base, int local_n_base) {
    const int lane = kittens::laneid();
    const int row_g = 4 * (lane / 16);
    const int col_l = lane % 16;
    #pragma unroll
    for (int i = 0; i < AccType::height; i++) {
        const int m0 = local_m_base + i * 16 + row_g;
        const float a0 = sa_lds[m0 + 0];
        const float a1 = sa_lds[m0 + 1];
        const float a2 = sa_lds[m0 + 2];
        const float a3 = sa_lds[m0 + 3];
        #pragma unroll
        for (int j = 0; j < AccType::width; j++) {
            const float sb = sb_lds[local_n_base + j * 16 + col_l];
            Cacc.tiles[i][j].data[0].x += partial.tiles[i][j].data[0].x * (a0 * sb);
            Cacc.tiles[i][j].data[0].y += partial.tiles[i][j].data[0].y * (a1 * sb);
            Cacc.tiles[i][j].data[1].x += partial.tiles[i][j].data[1].x * (a2 * sb);
            Cacc.tiles[i][j].data[1].y += partial.tiles[i][j].data[1].y * (a3 * sb);
        }
    }
}

template <typename AccType>
__device__ inline void apply_block_scale_1d1d_reg(
    AccType &Cacc, const AccType &partial, const float (&sa_reg)[AccType::height * 4],
    const float (&sb_reg)[AccType::width]) {
    #pragma unroll
    for (int i = 0; i < AccType::height; i++) {
        const float a0 = sa_reg[i * 4 + 0];
        const float a1 = sa_reg[i * 4 + 1];
        const float a2 = sa_reg[i * 4 + 2];
        const float a3 = sa_reg[i * 4 + 3];
        #pragma unroll
        for (int j = 0; j < AccType::width; j++) {
            const float sb = sb_reg[j];
            Cacc.tiles[i][j].data[0].x += partial.tiles[i][j].data[0].x * (a0 * sb);
            Cacc.tiles[i][j].data[0].y += partial.tiles[i][j].data[0].y * (a1 * sb);
            Cacc.tiles[i][j].data[1].x += partial.tiles[i][j].data[1].x * (a2 * sb);
            Cacc.tiles[i][j].data[1].y += partial.tiles[i][j].data[1].y * (a3 * sb);
        }
    }
}

template <typename AType, typename BType, typename OType, bool IS_PARTIAL_M, bool IS_PARTIAL_N,
          bool HAS_BIAS = false, bool HAS_GELU = false, bool HAS_BETA = false,
          bool IS_PARTIAL_K = false>
__global__ __launch_bounds__(NUM_THREADS, 2)
void micro_tk_1d2d(const micro_globals<AType, BType, OType> g) {
#if defined(__gfx942__)
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st<AType, BLOCK_M, BLOCK_K> (&As) = al.allocate<st<AType, BLOCK_M, BLOCK_K>>();
    st<BType, BLOCK_N, BLOCK_K> (&Bs) = al.allocate<st<BType, BLOCK_N, BLOCK_K>>();
    __shared__ float smem_sa[2][BLOCK_M];

    rt<AType, REG_M, MFMA_K> at[5];
    rt<BType, REG_N, MFMA_K> bt[3];
    rt_fl<REG_M, REG_N, ducks::rt_layout::col> C_accum[2];
    rt_fl<REG_M, REG_N, ducks::rt_layout::col> partial[2];
    for (int i = 0; i < 2; i++) { zero(C_accum[i]); }

    const int M = (int)g.c.rows();
    const int N = (int)g.c.cols();
    const int K = (int)g.a.cols();

    int wgid = (blockIdx.y * gridDim.x) + blockIdx.x;
    const int NUM_WGS = gridDim.x * gridDim.y;
    constexpr int WGM = 4;
    wgid = chiplet_transform_chunked(wgid, NUM_WGS, NUM_XCDS, WGM*WGM);
    
    const int num_pid_m = ceil_div(M, BLOCK_M);
    const int num_pid_n = ceil_div(N, BLOCK_N);
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
    const int warp_row = warp_id / 4;
    const int warp_col = warp_id % 4;

    const int num_k_steps = ceil_div(K, BLOCK_K);
    const bool is_k_partial = IS_PARTIAL_K && (K % BLOCK_K != 0);

    const float *sa_block = g.scale_a.raw_ptr + row * BLOCK_M;

    const int n_scale_blocks = ceil_div(N, SCALE_BLOCK);
    const int sb_block0 = col * (BLOCK_N / SCALE_BLOCK) + warp_col / 2;
    const bool sb_valid = (!is_last_n) || (sb_block0 < n_scale_blocks);
    const float *sb_base = g.scale_b.raw_ptr + (sb_valid ? sb_block0 : 0) * num_k_steps;
    int32x4_lds_t sb_srsrc = make_buf_res((const void*)sb_base, (uint32_t)num_k_steps * 4);
    const int local_m0 = warp_row * REG_M;
    const int local_m1 = (warp_row + 2) * REG_M;
    const int tid = threadIdx.x;
    // scale_A SRD: bounds OOB rows (partial-M) to 0 automatically.
    const uint32_t sa_range = (uint32_t)((M - row * BLOCK_M) * 4);

    const bool is_first_k_partial = is_k_partial && (num_k_steps == 1);
    if (is_first_k_partial || is_last_m) load_tile_masked(As, g.a, row, 0, M, K);
    else                         G::load(As, g.a, {0, 0, row, 0});
    if (is_first_k_partial || is_last_n) load_tile_masked(Bs, g.b, col, 0, N, K);
    else                         G::load(Bs, g.b, {0, 0, col, 0});

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

        zero(partial[0]); zero(partial[1]);

        const bool is_next_k_partial = is_k_partial && (k_step + 1 == num_k_steps - 1);
        float sa_reg0[REG_M / 16 * 4];
        float sa_reg1[REG_M / 16 * 4];

        // Cluster 0
        if (!is_last_n && !is_next_k_partial)
            load_global_to_register_buffer<2, false, NUM_THREADS>(b_buffer_next, B_ELEMS_PER_THREAD, g.b, {0, 0, col, k_step + 1}, Bs);
        float sb_next;
        load(at[0], subtile_inplace<REG_M, MFMA_K>(As, {warp_row, 0}));
        load(at[1], subtile_inplace<REG_M, MFMA_K>(As, {warp_row + 2, 0}));
        load(bt[0], subtile_inplace<REG_N, MFMA_K>(Bs, {warp_col, 0}));
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 1
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(partial[0], at[0], bt[0], partial[0]);
        mma_ABt(partial[1], at[1], bt[0], partial[1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 2
        load(bt[1], subtile_inplace<REG_N, MFMA_K>(Bs, {warp_col, 1}));
        load(at[2], subtile_inplace<REG_M, MFMA_K>(As, {warp_row, 1}));
        load(at[3], subtile_inplace<REG_M, MFMA_K>(As, {warp_row + 2, 1}));
        load(bt[0], subtile_inplace<REG_N, MFMA_K>(Bs, {warp_col, 2}));
        load(at[0], subtile_inplace<REG_M, MFMA_K>(As, {warp_row, 2}));
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 3
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(partial[0], at[2], bt[1], partial[0]);
        mma_ABt(partial[1], at[3], bt[1], partial[1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 4
        if (!is_last_m && !is_next_k_partial)
            load_global_to_register_buffer<2, false, NUM_THREADS>(a_buffer_next, A_ELEMS_PER_THREAD, g.a, {0, 0, row, k_step + 1}, As);
        load(at[1], subtile_inplace<REG_M, MFMA_K>(As, {warp_row + 2, 2}));
        load(bt[2], subtile_inplace<REG_N, MFMA_K>(Bs, {warp_col, 3}));
        load(at[4], subtile_inplace<REG_M, MFMA_K>(As, {warp_row, 3}));
        load(at[3], subtile_inplace<REG_M, MFMA_K>(As, {warp_row + 2, 3}));
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 5
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(partial[0], at[0], bt[0], partial[0]);
        mma_ABt(partial[1], at[1], bt[0], partial[1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 6
        asm volatile("s_waitcnt lgkmcnt(0)");
        if (is_next_k_partial || is_last_m) load_tile_masked(As, g.a, row, k_step + 1, M, K);
        else                         store_register_buffer_to_shared<NUM_THREADS>(As, a_buffer_next);
        if (is_next_k_partial || is_last_n) load_tile_masked(Bs, g.b, col, k_step + 1, N, K);
        load_scale_global_reg<REG_M / 16>(sa_reg0, sa_block + k_step * M, local_m0, sa_range);
        load_scale_global_reg<REG_M / 16>(sa_reg1, sa_block + k_step * M, local_m1, sa_range);
        sb_next = llvm_amdgcn_s_buffer_load_f32(sb_srsrc, (k_step + 1) * 4, 0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 7
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(partial[0], at[4], bt[2], partial[0]);
        mma_ABt(partial[1], at[3], bt[2], partial[1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 8
        if (!(is_next_k_partial || is_last_n))
            store_register_buffer_to_shared<NUM_THREADS>(Bs, b_buffer_next);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 9
        apply_block_scale_1d2d_reg(C_accum[0], partial[0], sa_reg0, sb_cur);
        apply_block_scale_1d2d_reg(C_accum[1], partial[1], sa_reg1, sb_cur);
        sb_cur = sb_next;
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

    }

    // Epilogue
    zero(partial[0]); zero(partial[1]);
    __builtin_amdgcn_sched_barrier(0);
    load(bt[0], subtile_inplace<REG_N, MFMA_K>(Bs, {warp_col, 0}));
    load(at[0], subtile_inplace<REG_M, MFMA_K>(As, {warp_row, 0}));
    load(at[1], subtile_inplace<REG_M, MFMA_K>(As, {warp_row + 2, 0}));
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    __builtin_amdgcn_s_setprio(1);
    mma_ABt(partial[0], at[0], bt[0], partial[0]);
    mma_ABt(partial[1], at[1], bt[0], partial[1]);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    load(bt[1], subtile_inplace<REG_N, MFMA_K>(Bs, {warp_col, 1}));
    load(at[2], subtile_inplace<REG_M, MFMA_K>(As, {warp_row, 1}));
    load(at[3], subtile_inplace<REG_M, MFMA_K>(As, {warp_row + 2, 1}));
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    __builtin_amdgcn_s_setprio(1);
    mma_ABt(partial[0], at[2], bt[1], partial[0]);
    mma_ABt(partial[1], at[3], bt[1], partial[1]);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    load(bt[0], subtile_inplace<REG_N, MFMA_K>(Bs, {warp_col, 2}));
    load(at[0], subtile_inplace<REG_M, MFMA_K>(As, {warp_row, 2}));
    load(at[1], subtile_inplace<REG_M, MFMA_K>(As, {warp_row + 2, 2}));
    load(bt[1], subtile_inplace<REG_N, MFMA_K>(Bs, {warp_col, 3}));
    load(at[2], subtile_inplace<REG_M, MFMA_K>(As, {warp_row, 3}));
    load(at[3], subtile_inplace<REG_M, MFMA_K>(As, {warp_row + 2, 3}));
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    __builtin_amdgcn_s_setprio(1);
    mma_ABt(partial[0], at[0], bt[0], partial[0]);
    mma_ABt(partial[1], at[1], bt[0], partial[1]);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    __builtin_amdgcn_s_setprio(1);
    mma_ABt(partial[0], at[2], bt[1], partial[0]);
    mma_ABt(partial[1], at[3], bt[1], partial[1]);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    {
        const int last = num_k_steps - 1;
        float sa_reg0[REG_M / 16 * 4];
        float sa_reg1[REG_M / 16 * 4];
        load_scale_global_reg<REG_M / 16>(sa_reg0, sa_block + last * M, local_m0, sa_range);
        load_scale_global_reg<REG_M / 16>(sa_reg1, sa_block + last * M, local_m1, sa_range);
        apply_block_scale_1d2d_reg(C_accum[0], partial[0], sa_reg0, sb_cur);
        apply_block_scale_1d2d_reg(C_accum[1], partial[1], sa_reg1, sb_cur);
    }

    if (warp_row == 0) {
        __builtin_amdgcn_s_barrier();
    }

    if constexpr (HAS_BIAS || HAS_GELU || HAS_BETA) {
        apply_epilogue<OType, HAS_BIAS, HAS_GELU, HAS_BETA>(
            C_accum[0], row * 4 + warp_row,     col * 4 + warp_col, M, N,
            g.bias, g.bias_dtype, g.gelu_aux, g.gelu_aux_dtype, g.c_in, g.beta);
        apply_epilogue<OType, HAS_BIAS, HAS_GELU, HAS_BETA>(
            C_accum[1], row * 4 + warp_row + 2, col * 4 + warp_col, M, N,
            g.bias, g.bias_dtype, g.gelu_aux, g.gelu_aux_dtype, g.c_in, g.beta);
    }

    if constexpr (std::is_same_v<OType, bf16>) {
        apply_rtne_bias(C_accum[0]);
        apply_rtne_bias(C_accum[1]);
    }
    if (is_last_m || is_last_n) {
        store_masked(g.c.raw_ptr, C_accum[0], row * 4 + warp_row,     col * 4 + warp_col, M, N);
        store_masked(g.c.raw_ptr, C_accum[1], row * 4 + warp_row + 2, col * 4 + warp_col, M, N);
    } else {
        store(g.c, C_accum[0], {0, 0, row * 4 + warp_row,     col * 4 + warp_col});
        store(g.c, C_accum[1], {0, 0, row * 4 + warp_row + 2, col * 4 + warp_col});
    }
#endif  // __gfx942__
}

template <typename AType, typename BType, typename OType, bool IS_PARTIAL_M, bool IS_PARTIAL_N,
          bool HAS_BIAS = false, bool HAS_GELU = false, bool HAS_BETA = false,
          bool IS_PARTIAL_K = false>
__global__ __launch_bounds__(NUM_THREADS, 2)
void micro_tk_1d1d(const micro_globals<AType, BType, OType> g) {
#if defined(__gfx942__)
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st<AType, BLOCK_M, BLOCK_K> (&As) = al.allocate<st<AType, BLOCK_M, BLOCK_K>>();
    st<BType, BLOCK_N, BLOCK_K> (&Bs) = al.allocate<st<BType, BLOCK_N, BLOCK_K>>();

    rt<AType, REG_M, MFMA_K> at[5];
    rt<BType, REG_N, MFMA_K> bt[3];
    rt_fl<REG_M, REG_N, ducks::rt_layout::col> C_accum[2];
    rt_fl<REG_M, REG_N, ducks::rt_layout::col> partial[2];
    for (int i = 0; i < 2; i++) { zero(C_accum[i]); }

    const int M = (int)g.c.rows();
    const int N = (int)g.c.cols();
    const int K = (int)g.a.cols();

    int wgid = (blockIdx.y * gridDim.x) + blockIdx.x;
    const int NUM_WGS = gridDim.x * gridDim.y;
    constexpr int WGM = 4;
    wgid = chiplet_transform_chunked(wgid, NUM_WGS, NUM_XCDS, WGM*WGM);

    const int num_pid_m = ceil_div(M, BLOCK_M);
    const int num_pid_n = ceil_div(N, BLOCK_N);
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
    const int warp_row = warp_id / 4;
    const int warp_col = warp_id % 4;

    const int num_k_steps = ceil_div(K, BLOCK_K);
    const bool is_k_partial = IS_PARTIAL_K && (K % BLOCK_K != 0);

    const float *sa_block = g.scale_a.raw_ptr + row * BLOCK_M;

    const int sb_col0 = col * BLOCK_N;
    const float *sb_block = g.scale_b.raw_ptr + sb_col0;
    const int local_m0 = warp_row * REG_M;
    const int local_m1 = (warp_row + 2) * REG_M;
    const int local_n  = warp_col * REG_N;
    const int tid = threadIdx.x;
    // scale SRD ranges: bound OOB rows/cols (partial-M/N) to 0 automatically.
    const uint32_t sa_range = (uint32_t)((M - row * BLOCK_M) * 4);
    const uint32_t sb_range = (uint32_t)((N - sb_col0) * 4);

    // Prologue
    const bool is_first_k_partial = is_k_partial && (num_k_steps == 1);
    if (is_first_k_partial || is_last_m) load_tile_masked(As, g.a, row, 0, M, K);
    else                         G::load(As, g.a, {0, 0, row, 0});
    if (is_first_k_partial || is_last_n) load_tile_masked(Bs, g.b, col, 0, N, K);
    else                         G::load(Bs, g.b, {0, 0, col, 0});
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

        zero(partial[0]); zero(partial[1]);

        const bool is_next_k_partial = is_k_partial && (k_step + 1 == num_k_steps - 1);

        // Cluster 0
        if (!is_last_n && !is_next_k_partial)
            load_global_to_register_buffer<2, false, NUM_THREADS>(b_buffer_next, B_ELEMS_PER_THREAD, g.b, {0, 0, col, k_step + 1}, Bs);
        float sa_reg0[REG_M / 16 * 4];
        float sa_reg1[REG_M / 16 * 4];
        float sb_reg[REG_N / 16];
        load(at[0], subtile_inplace<REG_M, MFMA_K>(As, {warp_row, 0}));
        load(at[1], subtile_inplace<REG_M, MFMA_K>(As, {warp_row + 2, 0}));
        load(bt[0], subtile_inplace<REG_N, MFMA_K>(Bs, {warp_col, 0}));
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 1
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(partial[0], at[0], bt[0], partial[0]);
        mma_ABt(partial[1], at[1], bt[0], partial[1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 2
        load(bt[1], subtile_inplace<REG_N, MFMA_K>(Bs, {warp_col, 1}));
        load(at[2], subtile_inplace<REG_M, MFMA_K>(As, {warp_row, 1}));
        load(at[3], subtile_inplace<REG_M, MFMA_K>(As, {warp_row + 2, 1}));
        load(bt[0], subtile_inplace<REG_N, MFMA_K>(Bs, {warp_col, 2}));
        load(at[0], subtile_inplace<REG_M, MFMA_K>(As, {warp_row, 2}));
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 3
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(partial[0], at[2], bt[1], partial[0]);
        mma_ABt(partial[1], at[3], bt[1], partial[1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 4
        if (!is_last_m && !is_next_k_partial)
            load_global_to_register_buffer<2, false, NUM_THREADS>(a_buffer_next, A_ELEMS_PER_THREAD, g.a, {0, 0, row, k_step + 1}, As);
        load(at[1], subtile_inplace<REG_M, MFMA_K>(As, {warp_row + 2, 2}));
        load(bt[2], subtile_inplace<REG_N, MFMA_K>(Bs, {warp_col, 3}));
        load(at[4], subtile_inplace<REG_M, MFMA_K>(As, {warp_row, 3}));
        load(at[3], subtile_inplace<REG_M, MFMA_K>(As, {warp_row + 2, 3}));
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 5
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(partial[0], at[0], bt[0], partial[0]);
        mma_ABt(partial[1], at[1], bt[0], partial[1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 6
        asm volatile("s_waitcnt lgkmcnt(0)");
        if (is_next_k_partial || is_last_m) load_tile_masked(As, g.a, row, k_step + 1, M, K);
        else                         store_register_buffer_to_shared<NUM_THREADS>(As, a_buffer_next);
        if (is_next_k_partial || is_last_n) load_tile_masked(Bs, g.b, col, k_step + 1, N, K);
        load_scale_global_reg<REG_M / 16>(sa_reg0, sa_block + k_step * M, local_m0, sa_range);
        load_scale_global_reg<REG_M / 16>(sa_reg1, sa_block + k_step * M, local_m1, sa_range);
        load_scaleB_global_reg<REG_N / 16>(sb_reg, sb_block + k_step * N, local_n, sb_range);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 7
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(partial[0], at[4], bt[2], partial[0]);
        mma_ABt(partial[1], at[3], bt[2], partial[1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 8
        if (!(is_next_k_partial || is_last_n))
            store_register_buffer_to_shared<NUM_THREADS>(Bs, b_buffer_next);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 9
        apply_block_scale_1d1d_reg(C_accum[0], partial[0], sa_reg0, sb_reg);
        apply_block_scale_1d1d_reg(C_accum[1], partial[1], sa_reg1, sb_reg);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

    }

    // Epilogue
    zero(partial[0]); zero(partial[1]);
    __builtin_amdgcn_sched_barrier(0);
    load(bt[0], subtile_inplace<REG_N, MFMA_K>(Bs, {warp_col, 0}));
    load(at[0], subtile_inplace<REG_M, MFMA_K>(As, {warp_row, 0}));
    load(at[1], subtile_inplace<REG_M, MFMA_K>(As, {warp_row + 2, 0}));
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    __builtin_amdgcn_s_setprio(1);
    mma_ABt(partial[0], at[0], bt[0], partial[0]);
    mma_ABt(partial[1], at[1], bt[0], partial[1]);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    load(bt[1], subtile_inplace<REG_N, MFMA_K>(Bs, {warp_col, 1}));
    load(at[2], subtile_inplace<REG_M, MFMA_K>(As, {warp_row, 1}));
    load(at[3], subtile_inplace<REG_M, MFMA_K>(As, {warp_row + 2, 1}));
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    __builtin_amdgcn_s_setprio(1);
    mma_ABt(partial[0], at[2], bt[1], partial[0]);
    mma_ABt(partial[1], at[3], bt[1], partial[1]);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    load(bt[0], subtile_inplace<REG_N, MFMA_K>(Bs, {warp_col, 2}));
    load(at[0], subtile_inplace<REG_M, MFMA_K>(As, {warp_row, 2}));
    load(at[1], subtile_inplace<REG_M, MFMA_K>(As, {warp_row + 2, 2}));
    load(bt[1], subtile_inplace<REG_N, MFMA_K>(Bs, {warp_col, 3}));
    load(at[2], subtile_inplace<REG_M, MFMA_K>(As, {warp_row, 3}));
    load(at[3], subtile_inplace<REG_M, MFMA_K>(As, {warp_row + 2, 3}));
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    __builtin_amdgcn_s_setprio(1);
    mma_ABt(partial[0], at[0], bt[0], partial[0]);
    mma_ABt(partial[1], at[1], bt[0], partial[1]);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    __builtin_amdgcn_s_setprio(1);
    mma_ABt(partial[0], at[2], bt[1], partial[0]);
    mma_ABt(partial[1], at[3], bt[1], partial[1]);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    {
        const int last = num_k_steps - 1;
        float sa_reg0[REG_M / 16 * 4];
        float sa_reg1[REG_M / 16 * 4];
        float sb_reg[REG_N / 16];
        load_scale_global_reg<REG_M / 16>(sa_reg0, sa_block + last * M, local_m0, sa_range);
        load_scale_global_reg<REG_M / 16>(sa_reg1, sa_block + last * M, local_m1, sa_range);
        load_scaleB_global_reg<REG_N / 16>(sb_reg, sb_block + last * N, local_n, sb_range);
        apply_block_scale_1d1d_reg(C_accum[0], partial[0], sa_reg0, sb_reg);
        apply_block_scale_1d1d_reg(C_accum[1], partial[1], sa_reg1, sb_reg);
    }

    if (warp_row == 0) {
        __builtin_amdgcn_s_barrier();
    }

    if constexpr (HAS_BIAS || HAS_GELU || HAS_BETA) {
        apply_epilogue<OType, HAS_BIAS, HAS_GELU, HAS_BETA>(
            C_accum[0], row * 4 + warp_row,     col * 4 + warp_col, M, N,
            g.bias, g.bias_dtype, g.gelu_aux, g.gelu_aux_dtype, g.c_in, g.beta);
        apply_epilogue<OType, HAS_BIAS, HAS_GELU, HAS_BETA>(
            C_accum[1], row * 4 + warp_row + 2, col * 4 + warp_col, M, N,
            g.bias, g.bias_dtype, g.gelu_aux, g.gelu_aux_dtype, g.c_in, g.beta);
    }

    if constexpr (std::is_same_v<OType, bf16>) {
        apply_rtne_bias(C_accum[0]);
        apply_rtne_bias(C_accum[1]);
    }
    if (is_last_m || is_last_n) {
        store_masked(g.c.raw_ptr, C_accum[0], row * 4 + warp_row,     col * 4 + warp_col, M, N);
        store_masked(g.c.raw_ptr, C_accum[1], row * 4 + warp_row + 2, col * 4 + warp_col, M, N);
    } else {
        store(g.c, C_accum[0], {0, 0, row * 4 + warp_row,     col * 4 + warp_col});
        store(g.c, C_accum[1], {0, 0, row * 4 + warp_row + 2, col * 4 + warp_col});
    }
#endif  // __gfx942__
}

template <bool IS_1D2D, typename AType, typename BType, typename OType,
          bool HAS_BIAS, bool HAS_GELU, bool HAS_BETA, bool IS_PARTIAL_K>
static void dispatch_micro_epilogue(micro_globals<AType, BType, OType> g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    const bool pm = (g.M() % BLOCK_M != 0);
    const bool pn = (g.N() % BLOCK_N != 0);
    auto launch = [&](auto kern) {
        hipFuncSetAttribute((void*)kern, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
        kern<<<g.grid(), g.block(), mem_size, g.stream>>>(g);
    };
    auto kernel = [&]<bool PM, bool PN>() {
        if constexpr (IS_1D2D)
            return micro_tk_1d2d<AType, BType, OType, PM, PN, HAS_BIAS, HAS_GELU, HAS_BETA, IS_PARTIAL_K>;
        else
            return micro_tk_1d1d<AType, BType, OType, PM, PN, HAS_BIAS, HAS_GELU, HAS_BETA, IS_PARTIAL_K>;
    };
    if      (!pm && !pn) launch(kernel.template operator()<false, false>());
    else if ( pm && !pn) launch(kernel.template operator()<true,  false>());
    else if (!pm &&  pn) launch(kernel.template operator()<false, true >());
    else                 launch(kernel.template operator()<true,  true >());
}

template <bool IS_1D2D, typename AType, typename BType, typename OType, bool KP>
static void dispatch_micro_k(micro_globals<AType, BType, OType> g,
                             bool has_bias, bool has_gelu, bool has_beta) {
    if (has_gelu) {
        if (has_beta) dispatch_micro_epilogue<IS_1D2D, AType, BType, OType, false, true, true, KP>(g);
        else          dispatch_micro_epilogue<IS_1D2D, AType, BType, OType, false, true, false, KP>(g);
    } else if (has_bias) {
        if (has_beta) dispatch_micro_epilogue<IS_1D2D, AType, BType, OType, true, false, true, KP>(g);
        else          dispatch_micro_epilogue<IS_1D2D, AType, BType, OType, true, false, false, KP>(g);
    } else {
        if (has_beta) dispatch_micro_epilogue<IS_1D2D, AType, BType, OType, false, false, true, KP>(g);
        else          dispatch_micro_epilogue<IS_1D2D, AType, BType, OType, false, false, false, KP>(g);
    }
}

template <bool IS_1D2D, typename AType, typename BType, typename OType>
static void dispatch_micro(micro_globals<AType, BType, OType> g,
                           bool has_bias, bool has_gelu, bool has_beta, bool has_partial_k) {
    if (has_partial_k) dispatch_micro_k<IS_1D2D, AType, BType, OType, true >(g, has_bias, has_gelu, has_beta);
    else               dispatch_micro_k<IS_1D2D, AType, BType, OType, false>(g, has_bias, has_gelu, has_beta);
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

    // Kernel body uses the swapped convention (kernel-A=activation/1D, kM=user N);
    // dispatch passes canonical (A=weight/2D, M=user M). Swap back here (as impl_cdna4).
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
                ? _gl_SB(reinterpret_cast<float*>(const_cast<void*>(ksb)), 1, 1, ceil_div(kN, SCALE_BLOCK), k_blocks)
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
#ifdef NVTE_HK_FAST_BUILD
    // Fast dev build: instantiate only e4m3xe4m3 + bf16 out (1/9 of the
    // template instances). Other dtype combos fall back to this instance.
    // For correctness/perf testing build WITHOUT NVTE_HK_FAST_BUILD.
    (void)a_e5m2; (void)b_e5m2; (void)out_dtype;
    run.template operator()<fp8e4m3, fp8e4m3, bf16>();
#else
    auto run_ab = [&]<typename OType>() {
        if      (!a_e5m2 && !b_e5m2) run.template operator()<fp8e4m3, fp8e4m3, OType>();
        else if ( a_e5m2 && !b_e5m2) run.template operator()<fp8e5m2, fp8e4m3, OType>();
        else                         run.template operator()<fp8e4m3, fp8e5m2, OType>();
    };
    if      (out_dtype == KITTENS_FLOAT32) run_ab.template operator()<float>();
    else if (out_dtype == KITTENS_FLOAT16) run_ab.template operator()<half>();
    else                                   run_ab.template operator()<bf16>();
#endif
}
