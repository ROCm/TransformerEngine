/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

#include "kittens.cuh"
#include "../kittens_common.h"
#include "../kittens_kernel_common.cuh"
#include "mxfp8_gemm_helper.cuh"
#include "xcd_sched_common.cuh"
#include <algorithm>
#include <cstdlib>
#include <map>
#include <mutex>
#include <utility>
#include <vector>

namespace te_kittens::cdna4::mxfp8 {

constexpr int NUM_WARPS = 8;
constexpr int NUM_THREADS = NUM_WARPS * kittens::WARP_THREADS;
constexpr int WARPS_ROW = 2;
constexpr int WARPS_COL = 4;
constexpr int BLOCK_ROW = 256;
constexpr int BLOCK_COL = 256;
constexpr int BLOCK_K   = 128;
constexpr int HALF_ROW  = BLOCK_ROW / 2;
constexpr int HALF_COL  = BLOCK_COL / 2;
constexpr int REG_M     = BLOCK_ROW / WARPS_ROW / 2;
constexpr int REG_N     = BLOCK_COL / WARPS_COL / 2;

using gl_f32_rt  = kittens::gl<float, 1, 1, -1, -1>;
using gl_fp16_rt = kittens::gl<half,  1, 1, -1, -1>;

using gl_bf16_rt = kittens::gl<kittens::bf16,    1, 1, -1, -1>;

// fp8e4m3 is used for all FP8 data (both e4m3 and e5m2) in HipKittens
// The MFMA instruction's cbsz/blgp bits select the actual format at compute time
// See mma_ABt_scaled for implementation details.
using gl_fp8_rt   = kittens::gl<kittens::fp8e4m3, 1, 1, -1, -1>;
using gl_scale_rt = kittens::gl<kittens::fp8e8m0, -1, 1, 16, 64>;

using G = kittens::group<NUM_WARPS>;

// Scale tile shared by mxfp8 kernels; one fp8e8m0_4 per (group, lane)
using ST_Scale = kittens::st<kittens::fp8e8m0, 16, 64, kittens::st_16x64_s>;

// Reads the pre-packed lane-native scale for group lg on this lane
__device__ __forceinline__ kittens::fp8e8m0_4 lane_rd(const ST_Scale &s, int lg) {
    return reinterpret_cast<const uint32_t *>(s.data)[lg * 64 + kittens::laneid()];
}

template <bool EDGE>
__device__ inline float read_bias(const void *bias, int bias_dtype, int idx, int len) {
    if constexpr (EDGE) {
        idx = min(idx, len - 1);
    }
    if (bias_dtype == 1) {
        return __bfloat162float(reinterpret_cast<const __hip_bfloat16 *>(bias)[idx]);
    } else if (bias_dtype == 2) {
        return __half2float(reinterpret_cast<const __half *>(bias)[idx]);
    }
    return reinterpret_cast<const float *>(bias)[idx];
}

__device__ uint16_t llvm_amdgcn_raw_buffer_load_b16(kittens::i32x4 srsrc, uint32_t voffset, uint32_t soffset,
                                                    uint32_t coherency) __asm("llvm.amdgcn.raw.buffer.load.i16");

// Reads/writes C/aux for tiles on matrix edge with elements outside the matrix skipped. Relies on buffer range check.
template <bool STORE, bool EDGE, typename GL, typename RT>
__device__ __forceinline__ void access_c(const GL &gl, RT &tile, const kittens::coord<RT> &idx) {
    using U  = typename GL::dtype;
    using U2 = typename kittens::base_types::packing<U>::packed_type;
    static_assert(RT::base_tile_stride == 4 && RT::base_tile_num_strides == 1);
    if constexpr (!EDGE) {
        if constexpr (STORE) {
            kittens::store(gl, tile, idx);
        } else {
            kittens::load(tile, gl, idx);
        }
        return;
    }
    const int rows = gl.rows();
    const int cols = gl.cols();
    const int lane = kittens::laneid();
    const int row0 = __builtin_amdgcn_readfirstlane(idx.r * RT::rows);
    const int col0 = __builtin_amdgcn_readfirstlane(idx.c * RT::cols);
    U *org = gl.raw_ptr + (size_t)row0 * cols + col0;
    const int left = max(min(rows - row0, RT::rows) * cols - col0, 0);
    const kittens::i32x4 srd = std::bit_cast<kittens::i32x4>(kittens::make_buffer_resource(
        reinterpret_cast<uintptr_t>(org), (uint32_t)(left * sizeof(U)), 0x00020000));
    if constexpr (kittens::ducks::rt::row_layout<RT>) {
#pragma unroll
        for (int i = 0; i < RT::height; i++) {
#pragma unroll
            for (int j = 0; j < RT::width; j++) {
                const int r = i * RT::base_tile_rows + lane % RT::base_tile_rows;
                const int c = j * RT::base_tile_cols + 4 * (lane / RT::base_tile_rows);
                const uint32_t off = row0 + r < rows && col0 + c < cols ? (r * cols + c) * sizeof(U) : 0x80000000u;
                U2 tmp[2];
                if constexpr (STORE) {
                    tmp[0] = kittens::base_types::convertor<U2, float2>::convert(tile.tiles[i][j].data[0]);
                    tmp[1] = kittens::base_types::convertor<U2, float2>::convert(tile.tiles[i][j].data[1]);
                    if constexpr (sizeof(U) == 4) {
                        kittens::llvm_amdgcn_raw_buffer_store_b128(*reinterpret_cast<__uint128_t *>(tmp),
                                                                   srd, off, 0, 0);
                    } else {
                        kittens::llvm_amdgcn_raw_buffer_store_b64(*reinterpret_cast<uint64_t *>(tmp), srd, off, 0, 0);
                    }
                } else {
                    if constexpr (sizeof(U) == 4) {
                        *reinterpret_cast<__uint128_t *>(tmp) =
                            kittens::llvm_amdgcn_raw_buffer_load_b128(srd, off, 0, 0);
                    } else {
                        *reinterpret_cast<uint64_t *>(tmp) = kittens::llvm_amdgcn_raw_buffer_load_b64(srd, off, 0, 0);
                    }
                    tile.tiles[i][j].data[0] = kittens::base_types::convertor<float2, U2>::convert(tmp[0]);
                    tile.tiles[i][j].data[1] = kittens::base_types::convertor<float2, U2>::convert(tmp[1]);
                }
            }
        }
    } else {
#pragma unroll
        for (int i = 0; i < RT::height; i++) {
#pragma unroll
            for (int j = 0; j < RT::width; j++) {
                const int r = 4 * (lane / RT::base_tile_cols);
                const int c = j * RT::base_tile_cols + lane % RT::base_tile_cols;
                // Rows past gl already fall outside the SRD range; only the column needs the explicit drop.
                const uint32_t off = col0 + c < cols ? (r * cols + c) * sizeof(U) : 0x80000000u;
#pragma unroll
                for (int l = 0; l < 4; l++) {
                    const uint32_t row_off = (i * RT::base_tile_rows + l) * cols * sizeof(U);
                    float &v = l & 1 ? tile.tiles[i][j].data[l / 2].y : tile.tiles[i][j].data[l / 2].x;
                    if constexpr (STORE) {
                        const U u = kittens::base_types::convertor<U, float>::convert(v);
                        if constexpr (sizeof(U) == 4) {
                            kittens::llvm_amdgcn_raw_buffer_store_b32(std::bit_cast<uint32_t>(u), srd, off, row_off, 0);
                        } else {
                            kittens::llvm_amdgcn_raw_buffer_store_b16(std::bit_cast<uint16_t>(u), srd, off, row_off, 0);
                        }
                    } else {
                        U u;
                        if constexpr (sizeof(U) == 4) {
                            u = std::bit_cast<U>(kittens::llvm_amdgcn_raw_buffer_load_b32(srd, off, row_off, 0));
                        } else {
                            u = std::bit_cast<U>(llvm_amdgcn_raw_buffer_load_b16(srd, off, row_off, 0));
                        }
                        v = kittens::base_types::convertor<float, U>::convert(u);
                    }
                }
            }
        }
    }
}

// Like G::prefill_swizzled_offsets, but lanes with K index below skip load zeros.
template <bool K_ROWS, typename ST, int COPIES>
__device__ __forceinline__ void prefill_k_offsets(uint32_t (&sw)[COPIES], int row_stride, int skip) {
    static_assert(ST::underlying_subtiles_per_row == 1 && sizeof(typename ST::dtype) == 1);
    const int lane = kittens::laneid();
    const int warp = kittens::warpid() % NUM_WARPS;
#pragma unroll
    for (int i = 0; i < COPIES; i++) {
        const int byte        = lane * LANE_LOAD_BYTES + (warp + i * NUM_WARPS) * WAVE_LOAD_BYTES;
        const int subtile     = byte / ST::underlying_subtile_bytes;
        const int subtile_off = byte % ST::underlying_subtile_bytes;
        const uint32_t swizzled = ST::swizzle({subtile_off / ST::underlying_subtile_row_bytes,
                                               subtile_off % ST::underlying_subtile_row_bytes});
        const int row = swizzled / ST::underlying_subtile_row_bytes + subtile * ST::underlying_subtile_rows;
        const int col = swizzled % ST::underlying_subtile_row_bytes;
        sw[i] = (K_ROWS ? row : col) < skip ? 0x80000000u : row * row_stride + col;
    }
}

enum struct GemmEpilogue {
    DEFAULT,
    BIAS,
    GELU_AUX,
    GELU_AUX_BIAS,
};

enum struct OutDtype {
    FP32,
    BF16,
    FP16,
};

template <typename T>
__device__ __forceinline__ int hk_upper_bound(const T *arr, int n, T val) {
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        int mid = (lo + hi + 1) / 2;
        if (arr[mid] <= val) { lo = mid; } else { hi = mid - 1; }
    }
    return lo;
}

// Shared epilogue. TRANSPOSE_OUT selects the TN/NT form, where C is [N, M] and the col_l
// accumulators transpose before the store; otherwise (NN, flipped operand slots) the accumulator
// already carries C's orientation and stores straight through. BLAS bias has length m, which is
// the accumulator's row axis when transposing and its column axis when not, so the two gathers
// differ in which register index the bias is constant across.
template<GemmEpilogue EPILOGUE, bool ACCUMULATE, bool TRANSPOSE_OUT, bool EDGE, typename RT_C, typename RT_C_T,
         typename OutGL, typename AuxGLType>
__device__ __forceinline__ void gemm_epilogue(
    RT_C &cA, RT_C &cB, RT_C &cC, RT_C &cD,
    const OutGL &C, const AuxGLType &AuxGL,
    const void *__restrict__ bias, int bias_dtype,
    int block_m, int block_row, int block_col, int warp_m, int warp_n) {

    constexpr bool HAS_BIAS = EPILOGUE == GemmEpilogue::BIAS || EPILOGUE == GemmEpilogue::GELU_AUX_BIAS;
    constexpr bool HAS_GELU = EPILOGUE == GemmEpilogue::GELU_AUX || EPILOGUE == GemmEpilogue::GELU_AUX_BIAS;

    const int r_lo = block_row * WARPS_ROW * 2 + warp_m;
    const int r_hi = block_row * WARPS_ROW * 2 + WARPS_ROW + warp_m;
    const int c_lo = block_col * WARPS_COL * 2 + warp_n;
    const int c_hi = block_col * WARPS_COL * 2 + WARPS_COL + warp_n;
    auto mk = [](int r, int c) {
        if constexpr (TRANSPOSE_OUT) { return kittens::coord<RT_C_T>{0, 0, c, r}; }
        else                         { return kittens::coord<RT_C_T>{0, 0, r, c}; }
    };
    auto out_coord_A = mk(r_lo, c_lo);
    auto out_coord_B = mk(r_lo, c_hi);
    auto out_coord_C = mk(r_hi, c_lo);
    auto out_coord_D = mk(r_hi, c_hi);

    if constexpr (HAS_BIAS) {
        if constexpr (TRANSPOSE_OUT) {
            int m_base_lo = block_m + warp_m * REG_M;
            int m_base_hi = block_m + (WARPS_ROW + warp_m) * REG_M;
            int lane      = kittens::laneid();
            int row_off   = cA.base_tile_stride * (lane / cA.base_tile_cols);
    #pragma unroll
            for (int i = 0; i < cA.height; i++) {
    #pragma unroll
                for (int j = 0; j < cA.width; j++) {
    #pragma unroll
                    for (int kk = 0; kk < cA.base_tile_num_strides; kk++) {
    #pragma unroll
                        for (int l = 0; l < cA.base_tile_stride / 2; l++) {
                            int idx    = l + kk * cA.base_tile_stride / 2;
                            int m_lo_x = m_base_lo + i * 16 + row_off + l * 2;
                            int m_hi_x = m_base_hi + i * 16 + row_off + l * 2;
                            float b_lo_x = read_bias<EDGE>(bias, bias_dtype, m_lo_x, C.cols());
                            float b_lo_y = read_bias<EDGE>(bias, bias_dtype, m_lo_x + 1, C.cols());
                            float b_hi_x = read_bias<EDGE>(bias, bias_dtype, m_hi_x, C.cols());
                            float b_hi_y = read_bias<EDGE>(bias, bias_dtype, m_hi_x + 1, C.cols());
                            cA.tiles[i][j].data[idx].x += b_lo_x; cA.tiles[i][j].data[idx].y += b_lo_y;
                            cB.tiles[i][j].data[idx].x += b_lo_x; cB.tiles[i][j].data[idx].y += b_lo_y;
                            cC.tiles[i][j].data[idx].x += b_hi_x; cC.tiles[i][j].data[idx].y += b_hi_y;
                            cD.tiles[i][j].data[idx].x += b_hi_x; cD.tiles[i][j].data[idx].y += b_hi_y;
                        }
                    }
                }
            }
        } else {
            // block_m is the weight-extent (BLAS m) base of this tile, i.e. the accumulator's N base.
            int n_base_lo = block_m + warp_n * REG_N;
            int n_base_hi = block_m + (WARPS_COL + warp_n) * REG_N;
            int lane      = kittens::laneid();
            int col_off   = lane % cA.base_tile_cols;
    #pragma unroll
            for (int j = 0; j < cA.width; j++) {
                int    n_x   = j * cA.base_tile_cols + col_off;
                float  b_lo  = read_bias<EDGE>(bias, bias_dtype, n_base_lo + n_x, C.cols());
                float  b_hi  = read_bias<EDGE>(bias, bias_dtype, n_base_hi + n_x, C.cols());
    #pragma unroll
                for (int i = 0; i < cA.height; i++) {
    #pragma unroll
                    for (int kk = 0; kk < cA.base_tile_num_strides; kk++) {
    #pragma unroll
                        for (int l = 0; l < cA.base_tile_stride / 2; l++) {
                            int idx = l + kk * cA.base_tile_stride / 2;
                            cA.tiles[i][j].data[idx].x += b_lo; cA.tiles[i][j].data[idx].y += b_lo;
                            cC.tiles[i][j].data[idx].x += b_lo; cC.tiles[i][j].data[idx].y += b_lo;
                            cB.tiles[i][j].data[idx].x += b_hi; cB.tiles[i][j].data[idx].y += b_hi;
                            cD.tiles[i][j].data[idx].x += b_hi; cD.tiles[i][j].data[idx].y += b_hi;
                        }
                    }
                }
            }
        }
    }

    if constexpr (HAS_GELU) {
        if constexpr (TRANSPOSE_OUT) {
            RT_C_T tA, tB, tC, tD;
            kittens::transpose(tA, cA); kittens::transpose(tB, cB); kittens::transpose(tC, cC); kittens::transpose(tD, cD);
            access_c<true, EDGE>(AuxGL, tA, out_coord_A); access_c<true, EDGE>(AuxGL, tB, out_coord_B);
            access_c<true, EDGE>(AuxGL, tC, out_coord_C); access_c<true, EDGE>(AuxGL, tD, out_coord_D);
        } else {
            access_c<true, EDGE>(AuxGL, cA, out_coord_A); access_c<true, EDGE>(AuxGL, cB, out_coord_B);
            access_c<true, EDGE>(AuxGL, cC, out_coord_C); access_c<true, EDGE>(AuxGL, cD, out_coord_D);
        }
        kittens::gelu(cA, cA); kittens::gelu(cB, cB); kittens::gelu(cC, cC); kittens::gelu(cD, cD);
    }

    if constexpr (TRANSPOSE_OUT) {
        RT_C_T oA, oB, oC, oD;
        kittens::transpose(oA, cA); kittens::transpose(oB, cB); kittens::transpose(oC, cC); kittens::transpose(oD, cD);

        if constexpr (ACCUMULATE) {
            RT_C_T eA, eB, eC, eD;
            access_c<false, EDGE>(C, eA, out_coord_A); access_c<false, EDGE>(C, eB, out_coord_B);
            access_c<false, EDGE>(C, eC, out_coord_C); access_c<false, EDGE>(C, eD, out_coord_D);
            kittens::add(oA, oA, eA); kittens::add(oB, oB, eB);
            kittens::add(oC, oC, eC); kittens::add(oD, oD, eD);
        }

        access_c<true, EDGE>(C, oA, out_coord_A); access_c<true, EDGE>(C, oB, out_coord_B);
        access_c<true, EDGE>(C, oC, out_coord_C); access_c<true, EDGE>(C, oD, out_coord_D);
    } else {
        if constexpr (ACCUMULATE) {
            RT_C eA, eB, eC, eD;
            access_c<false, EDGE>(C, eA, out_coord_A); access_c<false, EDGE>(C, eB, out_coord_B);
            access_c<false, EDGE>(C, eC, out_coord_C); access_c<false, EDGE>(C, eD, out_coord_D);
            kittens::add(cA, cA, eA); kittens::add(cB, cB, eB);
            kittens::add(cC, cC, eC); kittens::add(cD, cD, eD);
        }

        access_c<true, EDGE>(C, cA, out_coord_A); access_c<true, EDGE>(C, cB, out_coord_B);
        access_c<true, EDGE>(C, cC, out_coord_C); access_c<true, EDGE>(C, cD, out_coord_D);
    }
}

template <GemmEpilogue EPILOGUE, int CBSZ, int BLGP, bool ACCUMULATE = false, bool EDGE = false, bool SPLIT_K = false,
          typename OutGL, typename AuxGLType>
__global__ __launch_bounds__(NUM_THREADS, 2)
void mxfp8_gemm_tn_kernel(const gl_fp8_rt A, const gl_fp8_rt B, const OutGL C, const AuxGLType AuxGL,
    const gl_scale_rt scale_A_gl, const gl_scale_rt scale_B_gl,
    [[maybe_unused]] const void *__restrict__ bias, [[maybe_unused]] int bias_dtype,
    int N, int K, int total_m_tiles, int tiles_N, int splits) {

    static_assert(!SPLIT_K || (EPILOGUE == GemmEpilogue::DEFAULT && !ACCUMULATE),
                  "Split-K only supports the DEFAULT epilogue without accumulation");

    // EDGE takes the K remainder at the head: operands start shift columns early and block 0 masks them.
    int k_iters   = EDGE && !SPLIT_K ? max(kittens::ceil_div(K, BLOCK_K), 2) : K / BLOCK_K / (SPLIT_K ? splits : 1);
    int shift     = EDGE && !SPLIT_K ? k_iters * BLOCK_K - K : 0;
    int sa_stride = total_m_tiles;

    using ST_A     = kittens::st_fp8e4m3<HALF_ROW, BLOCK_K, kittens::st_16x128_s>;
    using ST_B     = kittens::st_fp8e4m3<HALF_COL, BLOCK_K, kittens::st_16x128_s>;
    using RT_A     = kittens::rt_fp8e4m3<REG_M, BLOCK_K>;
    using RT_B     = kittens::rt_fp8e4m3<REG_N, BLOCK_K>;
    using RT_C     = kittens::rt_fl<REG_M, REG_N, kittens::col_l, kittens::rt_16x16_s>;
    using RT_C_T   = kittens::rt_fl<REG_N, REG_M, kittens::row_l, kittens::rt_16x16_s>;

    __shared__ ST_A As[2][2];
    __shared__ ST_B Bs[2][2];

    // B needs 8 scale groups = 2 tiles
    __shared__ ST_Scale scale_smem[2][3];

    RT_A a;
    RT_B b0, b1;
    RT_C cA, cB, cC, cD;
    kittens::zero(cA); kittens::zero(cB); kittens::zero(cC); kittens::zero(cD);

    const int NUM_XCDS = 8;
    const int WGM      = 8;
    int wgid           = kittens::chiplet_transform_chunked(blockIdx.x, gridDim.x, NUM_XCDS, WGM * WGM);
    int num_wgid_in_group = WGM * tiles_N * (SPLIT_K ? splits : 1);
    int group_id     = wgid / num_wgid_in_group;
    int first_pid_m  = group_id * WGM;
    int group_size_m = min(total_m_tiles - first_pid_m, WGM);
    int m_tile       = first_pid_m + ((wgid % num_wgid_in_group) % group_size_m);
    int n_tile       = (wgid % num_wgid_in_group) / group_size_m;

    int split_id = 0;
    if constexpr (SPLIT_K) {
        split_id = n_tile / tiles_N;
        n_tile -= split_id * tiles_N;
    }

    int block_m      = m_tile * BLOCK_ROW;
    int block_row    = m_tile;
    int block_col    = n_tile;
    [[maybe_unused]] int sa_batch = split_id * k_iters * total_m_tiles + m_tile;
    [[maybe_unused]] int sb_batch = split_id * k_iters * tiles_N + n_tile;
    [[maybe_unused]] int sb_stride = tiles_N;

    int warp_m = kittens::warpid() / WARPS_COL;
    int warp_n = kittens::warpid() % WARPS_COL;

    using T = kittens::fp8e4m3;

    gl_fp8_rt A_local(A);
    gl_fp8_rt B_local(B);
    if constexpr (SPLIT_K) {
        A_local.raw_ptr += split_id * k_iters * BLOCK_K;
        B_local.raw_ptr += split_id * k_iters * BLOCK_K;
    }

    const uint32_t a_row_stride = A_local.template stride<2>() * sizeof(T);
    const uint32_t b_row_stride = B_local.template stride<2>() * sizeof(T);
    const T *a_org = &A_local[{0, 0, block_m, 0}] - shift;
    const T *b_org = &B_local[{0, 0, n_tile * BLOCK_COL, 0}] - shift;
    const int a_rows   = EDGE ? A_local.rows() - block_m : BLOCK_ROW;
    const int b_rows   = EDGE ? B_local.rows() - n_tile * BLOCK_COL : BLOCK_COL;
    const int a_range0 = min(a_rows, HALF_ROW) * a_row_stride + shift;
    const int a_range1 = max(min(a_rows - HALF_ROW, HALF_ROW), 0) * a_row_stride + shift;
    const int b_range0 = min(b_rows, HALF_COL) * b_row_stride + shift;
    const int b_range1 = max(min(b_rows - HALF_COL, HALF_COL), 0) * b_row_stride + shift;
    const TileLoad a_dma = prefill_tile_load<NUM_WARPS>(a_org, HALF_ROW * a_row_stride, BLOCK_K * sizeof(T), a_range0,
                                                        a_row_stride, As, shift - BLOCK_K, a_range1);
    const TileLoad b_dma = prefill_tile_load<NUM_WARPS>(b_org, HALF_COL * b_row_stride, BLOCK_K * sizeof(T), b_range0,
                                                        b_row_stride, Bs, shift - BLOCK_K, b_range1);
    const TileLoad a_k0  = prefill_tile_load<NUM_WARPS>(a_org, HALF_ROW * a_row_stride, BLOCK_K * sizeof(T), a_range0,
                                                        a_row_stride, As, shift, a_range1);
    const TileLoad b_k0  = prefill_tile_load<NUM_WARPS>(b_org, HALF_COL * b_row_stride, BLOCK_K * sizeof(T), b_range0,
                                                        b_row_stride, Bs, shift, b_range1);
    const ScaleLoad s_dma = prefill_scale_load(scale_A_gl, scale_B_gl, sa_batch, 2 * sb_batch, sa_stride,
                                               2 * sb_stride, scale_smem);

    const uint32_t a_frag0 = row_frag_addr<RT_A>(As[0][0], warp_m, 0);
    const uint32_t a_frag1 = row_frag_addr<RT_A>(As[0][0], warp_m, 1);
    const uint32_t b_frag0 = row_frag_addr<RT_B>(Bs[0][0], warp_n, 0);
    const uint32_t b_frag1 = row_frag_addr<RT_B>(Bs[0][0], warp_n, 1);

    const int n_iters = k_iters - 2;
    const bool s0 = n_iters & 1;

#include "mxfp8_tn_preload.inc"

#include "mxfp8_tn_mainloop.inc"

    OutGL C_local(C);
    if constexpr (SPLIT_K) {
        C_local.raw_ptr += (size_t)split_id * N * C_local.template stride<2>();
    }

    gemm_epilogue<EPILOGUE, ACCUMULATE, true, EDGE, RT_C, RT_C_T>(cA, cB, cC, cD, C_local, AuxGL, bias, bias_dtype,
        block_m, block_row, block_col, warp_m, warp_n);
}

template <GemmEpilogue EPILOGUE, int CBSZ, int BLGP, bool ACCUMULATE = false, bool EDGE = false, bool SPLIT_K = false,
          typename OutGL, typename AuxGLType>
__global__ __launch_bounds__(NUM_THREADS, 2) void mxfp8_gemm_nn_kernel(const gl_fp8_rt A, const gl_fp8_rt B, const OutGL C,
    const AuxGLType AuxGL, const gl_scale_rt scale_A_gl, const gl_scale_rt scale_B_gl,
    [[maybe_unused]] const void *__restrict__ bias, [[maybe_unused]] int bias_dtype,
    int N, int K, int total_m_tiles, int tiles_N, int splits) {

    static_assert(!SPLIT_K || (EPILOGUE == GemmEpilogue::DEFAULT && !ACCUMULATE),
                  "Split-K only supports the DEFAULT epilogue without accumulation");

    int k_iters = EDGE && !SPLIT_K ? max(kittens::ceil_div(K, BLOCK_K), 2) : K / BLOCK_K / (SPLIT_K ? splits : 1);
    int shift   = EDGE && !SPLIT_K ? k_iters * BLOCK_K - K : 0;

    // FLIPPED operand slots: the A slot holds the activation (K-contiguous [M,K], plain reads),
    // the B slot the weight (K-major [K,N], ds_read_*_tr via b_frag). The transposing operand
    // therefore sits on N, replicated by WARPS_ROW=2 warps instead of M's WARPS_COL=4, and the
    // accumulator lands already in C's [M,N] orientation so the writeback needs no transpose.
    using ST_A     = kittens::st_fp8e4m3<HALF_ROW, BLOCK_K, kittens::st_16x128_s>;
    using ST_B     = kittens::st_fp8e4m3<BLOCK_K, HALF_COL, kittens::st_16x128_s>;
    using RT_A     = kittens::rt_fp8e4m3<REG_M, BLOCK_K>;
    using RT_B     = kittens::rt<kittens::fp8e4m3, REG_N, BLOCK_K, kittens::col_l, kittens::rt_16x128_s>;
    using RT_C     = kittens::rt_fl<REG_M, REG_N, kittens::col_l, kittens::rt_16x16_s>;

    __shared__ ST_A As[2][2];
    __shared__ alignas(ST_B::underlying_subtile_bytes) ST_B Bs[2][2];
    // B needs 8 scale groups = 2 tiles
    __shared__ ST_Scale scale_smem[2][3];

    RT_A a;
    RT_B b0, b1;
    RT_C cA, cB, cC, cD;
    kittens::zero(cA); kittens::zero(cB); kittens::zero(cC); kittens::zero(cD);

    const int NUM_XCDS = 8;
    const int WGM      = 8;
    int wgid           = kittens::chiplet_transform_chunked(blockIdx.x, gridDim.x, NUM_XCDS, WGM * WGM);
    int num_wgid_in_group = WGM * tiles_N * (SPLIT_K ? splits : 1);
    int group_id     = wgid / num_wgid_in_group;
    int first_pid_m  = group_id * WGM;
    int group_size_m = min(total_m_tiles - first_pid_m, WGM);
    int m_tile       = first_pid_m + ((wgid % num_wgid_in_group) % group_size_m);
    int n_tile       = (wgid % num_wgid_in_group) / group_size_m;

    int split_id = 0;
    if constexpr (SPLIT_K) {
        split_id = n_tile / tiles_N;
        n_tile -= split_id * tiles_N;
    }

    // m_tile indexes the weight (BLAS M), n_tile the activation (BLAS N). The activation feeds
    // the A slot, so the accumulator's M axis is n_tile and its N axis m_tile -- C's memory order.
    int a_half0      = n_tile * 2;
    int a_half1      = a_half0 + 1;
    int b_half0      = m_tile * 2;
    int b_half1      = b_half0 + 1;
    int block_m      = m_tile * BLOCK_ROW;   // bias base: bias runs along the accumulator's N axis
    int block_row    = n_tile;
    int block_col    = m_tile;
    // Scale slots follow the operands: scale_A_gl is the activation's <64,4> buffer keyed by
    // n_tile, scale_B_gl the weight's <32,8> hi/lo pair keyed by m_tile.
    [[maybe_unused]] int sa_batch  = split_id * k_iters * tiles_N + n_tile;
    [[maybe_unused]] int sa_stride = tiles_N;
    [[maybe_unused]] int sb_batch  = split_id * k_iters * total_m_tiles + m_tile;
    [[maybe_unused]] int sb_stride = total_m_tiles;

    int warp_m = kittens::warpid() / WARPS_COL;
    int warp_n = kittens::warpid() % WARPS_COL;

    using T = kittens::fp8e4m3;

    gl_fp8_rt A_local(A);
    gl_fp8_rt B_local(B);
    if constexpr (SPLIT_K) {
        A_local.raw_ptr += split_id * k_iters * BLOCK_K;
        B_local.raw_ptr += (size_t)split_id * k_iters * BLOCK_K * B_local.template stride<2>();
    }
    A_local.raw_ptr -= shift;
    B_local.raw_ptr -= (size_t)shift * B_local.template stride<2>();

    const int a_row_stride = A_local.template stride<2>() * sizeof(T);
    const int b_row_stride = B_local.template stride<2>() * sizeof(T);
    constexpr int bpt = ST_A::underlying_subtile_bytes_per_thread;
    constexpr int bpm = bpt * NUM_THREADS;
    constexpr int copies_A = HALF_ROW * BLOCK_K * sizeof(T) / bpm;
    constexpr int copies_B = BLOCK_K * HALF_COL * sizeof(T) / bpm;
    uint32_t sw_A[copies_A], sw_B[copies_B], sw_A0[copies_A], sw_B0[copies_B];
    prefill_k_offsets<false, ST_A>(sw_A, a_row_stride, shift - BLOCK_K);
    prefill_k_offsets<true, ST_B>(sw_B, b_row_stride, shift - BLOCK_K);
    prefill_k_offsets<false, ST_A>(sw_A0, a_row_stride, shift);
    prefill_k_offsets<true, ST_B>(sw_B0, b_row_stride, shift);

    const T *a_base = (const T *)&A_local[{0, 0, 0, 0}];
    const T *b_base = (const T *)&B_local[{0, 0, 0, 0}];
    kittens::i32x4 a_srd = kittens::make_srsrc(a_base, A_local.rows() * a_row_stride + shift, EDGE ? 0 : a_row_stride);
    const int k_rows = B_local.rows() + shift - split_id * k_iters * BLOCK_K;
    kittens::i32x4 b_srd = kittens::make_srsrc(b_base, k_rows * b_row_stride, EDGE ? 0 : b_row_stride);

    const int wid = kittens::warpid() % NUM_WARPS;
    constexpr int elem_per_warp = (16 / sizeof(T)) * kittens::WARP_THREADS;
    uint32_t a_lds[2][2], b_lds[2][2];
    for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) {
        a_lds[i][j] = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
            reinterpret_cast<uintptr_t>(&As[i][j].data[0]) + wid * elem_per_warp * sizeof(T)));
        b_lds[i][j] = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
            reinterpret_cast<uintptr_t>(&Bs[i][j].data[0]) + wid * elem_per_warp * sizeof(T)));
    }

    const uint32_t a_frag0 = row_frag_addr<RT_A>(As[0][0], warp_m, 0);
    const uint32_t a_frag1 = row_frag_addr<RT_A>(As[0][0], warp_m, 1);
    uint32_t b_frag[RT_B::height];
    col_frag_addrs<RT_B>(b_frag, Bs[0][0], warp_n * REG_N);

    const int n_iters = k_iters - 2;
    const bool s0 = n_iters & 1;

#include "mxfp8_nn_preload.inc"

#include "mxfp8_nn_mainloop.inc"

    OutGL C_local(C);
    if constexpr (SPLIT_K) {
        C_local.raw_ptr += (size_t)split_id * N * C_local.template stride<2>();
    }

    gemm_epilogue<EPILOGUE, ACCUMULATE, false, EDGE, RT_C, RT_C>(cA, cB, cC, cD, C_local, AuxGL, bias, bias_dtype,
        block_m, block_row, block_col, warp_m, warp_n);
}

template <bool SPLIT_K, GemmEpilogue EPILOGUE, int CBSZ, int BLGP, bool ACCUMULATE = false, bool EDGE = false,
          typename OutGL, typename AuxGLType>
__global__ __launch_bounds__(NUM_THREADS, 2) void mxfp8_gemm_nt_kernel(const gl_fp8_rt A, const gl_fp8_rt B,
    const OutGL C, const AuxGLType AuxGL, const gl_scale_rt scale_A_gl, const gl_scale_rt scale_B_gl,
    [[maybe_unused]] const void *__restrict__ bias, [[maybe_unused]] int bias_dtype,
    int N, int K, int total_m_tiles, int tiles_N, int splits) {

    static_assert(!SPLIT_K || (EPILOGUE == GemmEpilogue::DEFAULT && !ACCUMULATE),
                  "Split-K only supports the DEFAULT epilogue without accumulation");

    int k_iters   = EDGE && !SPLIT_K ? max(kittens::ceil_div(K, BLOCK_K), 2) : K / BLOCK_K / (SPLIT_K ? splits : 1);
    int shift     = EDGE && !SPLIT_K ? k_iters * BLOCK_K - K : 0;
    int sa_stride = total_m_tiles;

    using ST_A     = kittens::st_fp8e4m3<BLOCK_K, HALF_ROW, kittens::st_16x128_s>;
    using ST_B     = kittens::st_fp8e4m3<BLOCK_K, HALF_COL, kittens::st_16x128_s>;
    using RT_A     = kittens::rt<kittens::fp8e4m3, REG_M, BLOCK_K, kittens::col_l, kittens::rt_16x128_s>;
    using RT_B     = kittens::rt<kittens::fp8e4m3, REG_N, BLOCK_K, kittens::col_l, kittens::rt_16x128_s>;
    using RT_C     = kittens::rt_fl<REG_M, REG_N, kittens::col_l, kittens::rt_16x16_s>;
    using RT_C_T   = kittens::rt_fl<REG_N, REG_M, kittens::row_l, kittens::rt_16x16_s>;

    __shared__ alignas(ST_A::underlying_subtile_bytes) ST_A As[2][2];
    __shared__ alignas(ST_B::underlying_subtile_bytes) ST_B Bs[2][2];
    // B needs 8 scale groups = 2 tiles
    __shared__ ST_Scale scale_smem[2][3];

    RT_A a;
    RT_B b0, b1;
    RT_C cA, cB, cC, cD;
    kittens::zero(cA); kittens::zero(cB); kittens::zero(cC); kittens::zero(cD);

    const int NUM_XCDS = 8;
    const int WGM      = 8;
    int wgid           = kittens::chiplet_transform_chunked(blockIdx.x, gridDim.x, NUM_XCDS, WGM * WGM);
    int num_wgid_in_group = WGM * tiles_N * (SPLIT_K ? splits : 1);
    int group_id     = wgid / num_wgid_in_group;
    int first_pid_m  = group_id * WGM;
    int group_size_m = min(total_m_tiles - first_pid_m, WGM);
    int m_tile       = first_pid_m + ((wgid % num_wgid_in_group) % group_size_m);
    int n_tile       = (wgid % num_wgid_in_group) / group_size_m;

    int split_id = 0;
    if constexpr (SPLIT_K) {
        split_id = n_tile / tiles_N;
        n_tile -= split_id * tiles_N;
    }

    int a_row_tile   = m_tile;
    int block_m      = m_tile * BLOCK_ROW;
    int block_row    = m_tile;
    int block_col    = n_tile;
    [[maybe_unused]] int sa_batch = SPLIT_K ? split_id * k_iters * total_m_tiles + m_tile : m_tile;
    [[maybe_unused]] int sb_batch = SPLIT_K ? split_id * k_iters * tiles_N + n_tile : n_tile;
    [[maybe_unused]] int sb_stride = tiles_N;

    int warp_m = kittens::warpid() / WARPS_COL;
    int warp_n = kittens::warpid() % WARPS_COL;

    using T = kittens::fp8e4m3;

    gl_fp8_rt A_local(A);
    gl_fp8_rt B_local(B);
    if constexpr (SPLIT_K) {
        A_local.raw_ptr += (size_t)split_id * k_iters * BLOCK_K * A_local.template stride<2>();
        B_local.raw_ptr += (size_t)split_id * k_iters * BLOCK_K * B_local.template stride<2>();
    }
    A_local.raw_ptr -= (size_t)shift * A_local.template stride<2>();
    B_local.raw_ptr -= (size_t)shift * B_local.template stride<2>();

    const int a_row_stride = A_local.template stride<2>() * sizeof(T);
    const int b_row_stride = B_local.template stride<2>() * sizeof(T);
    constexpr int bpt = ST_A::underlying_subtile_bytes_per_thread;
    constexpr int bpm = bpt * NUM_THREADS;
    constexpr int copies_A = BLOCK_K * HALF_ROW * sizeof(T) / bpm;
    constexpr int copies_B = BLOCK_K * HALF_COL * sizeof(T) / bpm;
    uint32_t sw_A[copies_A], sw_B[copies_B], sw_A0[copies_A], sw_B0[copies_B];
    prefill_k_offsets<true, ST_A>(sw_A, a_row_stride, shift - BLOCK_K);
    prefill_k_offsets<true, ST_B>(sw_B, b_row_stride, shift - BLOCK_K);
    prefill_k_offsets<true, ST_A>(sw_A0, a_row_stride, shift);
    prefill_k_offsets<true, ST_B>(sw_B0, b_row_stride, shift);

    // SRDs come from the *_local views so the split-K path picks up per-split base pointers. EDGE drops
    // the stride, which would otherwise let the soffset tile displacement skip the range check.
    const int k_rows = K + shift - split_id * k_iters * BLOCK_K;
    const T *a_base = (const T *)&A_local[{0, 0, 0, 0}];
    const T *b_base = (const T *)&B_local[{0, 0, 0, 0}];
    kittens::i32x4 a_srd = kittens::make_srsrc(a_base, k_rows * a_row_stride, EDGE ? 0 : a_row_stride);
    kittens::i32x4 b_srd = kittens::make_srsrc(b_base, k_rows * b_row_stride, EDGE ? 0 : b_row_stride);

    const int wid = kittens::warpid() % NUM_WARPS;
    constexpr int elem_per_warp = (16 / sizeof(T)) * kittens::WARP_THREADS;
    uint32_t a_lds[2][2], b_lds[2][2];
    for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) {
        a_lds[i][j] = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
            reinterpret_cast<uintptr_t>(&As[i][j].data[0]) + wid * elem_per_warp * sizeof(T)));
        b_lds[i][j] = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
            reinterpret_cast<uintptr_t>(&Bs[i][j].data[0]) + wid * elem_per_warp * sizeof(T)));
    }

    uint32_t a_frag[RT_A::height];
    col_frag_addrs<RT_A>(a_frag, As[0][0], warp_m * REG_M);
    uint32_t b_frag[RT_B::height];
    col_frag_addrs<RT_B>(b_frag, Bs[0][0], warp_n * REG_N);

    const int n_iters = k_iters - 2;
    const bool s0 = n_iters & 1;

    load_srd(Bs[s0][0], B_local, {0, 0, 0, n_tile * 2    }, sw_B0, b_srd, b_base, b_lds[s0][0]);
    load_srd(As[s0][0], A_local, {0, 0, 0, a_row_tile * 2    }, sw_A0, a_srd, a_base, a_lds[s0][0]);
    load_srd(Bs[s0][1], B_local, {0, 0, 0, n_tile * 2 + 1}, sw_B0, b_srd, b_base, b_lds[s0][1]);
    load_srd(As[s0][1], A_local, {0, 0, 0, a_row_tile * 2 + 1}, sw_A0, a_srd, a_base, a_lds[s0][1]);

    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();

    load_srd(As[!s0][0], A_local, {0, 0, 1, a_row_tile * 2    }, sw_A, a_srd, a_base, a_lds[!s0][0]);
    load_srd(Bs[!s0][0], B_local, {0, 0, 1, n_tile * 2    }, sw_B, b_srd, b_base, b_lds[!s0][0]);
    load_srd(Bs[!s0][1], B_local, {0, 0, 1, n_tile * 2 + 1}, sw_B, b_srd, b_base, b_lds[!s0][1]);
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();

    G::load(scale_smem[s0][0], scale_A_gl, {sa_batch, 0, 0, 0});
    G::load(scale_smem[s0][1], scale_B_gl, {2 * sb_batch,     0, 0, 0});
    G::load(scale_smem[s0][2], scale_B_gl, {2 * sb_batch + 1, 0, 0, 0});
    asm volatile("s_waitcnt vmcnt(0)");
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();

    if (warp_m == 1) {
        __builtin_amdgcn_s_barrier();
    }

    int k = 0;
    if (s0) {
        constexpr int TIC = 1;
        constexpr int TOC = TIC ^ 1;
        G::load(scale_smem[TOC][0], scale_A_gl, {(k + 1) * sa_stride + sa_batch, 0, 0, 0});
        G::load(scale_smem[TOC][1], scale_B_gl, {2 * ((k + 1) * sb_stride + sb_batch),     0, 0, 0});
        G::load(scale_smem[TOC][2], scale_B_gl, {2 * ((k + 1) * sb_stride + sb_batch) + 1, 0, 0, 0});

        load_cols<ST_B, 2 * TIC>(b0, b_frag);
        load_cols<ST_A, 2 * TIC>(a, a_frag);
        load_srd(As[TOC][1], A_local, {0, 0, k + 1, a_row_tile * 2 + 1}, sw_A, a_srd, a_base, a_lds[TOC][1]);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();

        kittens::fp8e8m0_4 sa_h0 = lane_rd(scale_smem[TIC][0], warp_m);
        kittens::fp8e8m0_4 sb_h0 = lane_rd(scale_smem[TIC][1], warp_n);
        __builtin_amdgcn_s_setprio(2);
        kittens::mma_ABt_scaled<CBSZ, BLGP>(cA, a, b0, cA, &sa_h0, &sb_h0);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        kittens::fp8e8m0_4 sb_h1 = lane_rd(scale_smem[TIC][2], warp_n);
        load_cols<ST_B, 2 * TIC + 1>(b1, b_frag);
        load_srd(As[TIC][0], A_local, {0, 0, k + 2, a_row_tile * 2    }, sw_A, a_srd, a_base, a_lds[TIC][0]);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();

        __builtin_amdgcn_s_setprio(2);
        kittens::mma_ABt_scaled<CBSZ, BLGP>(cB, a, b1, cB, &sa_h0, &sb_h1);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        kittens::fp8e8m0_4 sa_h1 = lane_rd(scale_smem[TIC][0], 2 + warp_m);
        load_cols<ST_A, 2 * TIC + 1>(a, a_frag);
        load_srd(Bs[TIC][0], B_local, {0, 0, k + 2, n_tile * 2    }, sw_B, b_srd, b_base, b_lds[TIC][0]);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();

        __builtin_amdgcn_s_setprio(2);
        kittens::mma_ABt_scaled<CBSZ, BLGP>(cC, a, b0, cC, &sa_h1, &sb_h0);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        load_srd(Bs[TIC][1], B_local, {0, 0, k + 2, n_tile * 2 + 1}, sw_B, b_srd, b_base, b_lds[TIC][1]);
        asm volatile("s_waitcnt vmcnt(6)"); // wait for toc data; next-iter prefetches in flight
        __builtin_amdgcn_s_barrier();

        __builtin_amdgcn_s_setprio(2);
        kittens::mma_ABt_scaled<CBSZ, BLGP>(cD, a, b1, cD, &sa_h1, &sb_h1);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        k++;
    }
    while (k < n_iters) {
        {
            constexpr int TIC = 0;
            constexpr int TOC = TIC ^ 1;
            G::load(scale_smem[TOC][0], scale_A_gl, {(k + 1) * sa_stride + sa_batch, 0, 0, 0});
            G::load(scale_smem[TOC][1], scale_B_gl, {2 * ((k + 1) * sb_stride + sb_batch),     0, 0, 0});
            G::load(scale_smem[TOC][2], scale_B_gl, {2 * ((k + 1) * sb_stride + sb_batch) + 1, 0, 0, 0});

            load_cols<ST_B, 2 * TIC>(b0, b_frag);
            load_cols<ST_A, 2 * TIC>(a, a_frag);
            load_srd(As[TOC][1], A_local, {0, 0, k + 1, a_row_tile * 2 + 1}, sw_A, a_srd, a_base, a_lds[TOC][1]);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_barrier();

            kittens::fp8e8m0_4 sa_h0 = lane_rd(scale_smem[TIC][0], warp_m);
            kittens::fp8e8m0_4 sb_h0 = lane_rd(scale_smem[TIC][1], warp_n);
            __builtin_amdgcn_s_setprio(2);
            kittens::mma_ABt_scaled<CBSZ, BLGP>(cA, a, b0, cA, &sa_h0, &sb_h0);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            kittens::fp8e8m0_4 sb_h1 = lane_rd(scale_smem[TIC][2], warp_n);
            load_cols<ST_B, 2 * TIC + 1>(b1, b_frag);
            load_srd(As[TIC][0], A_local, {0, 0, k + 2, a_row_tile * 2    }, sw_A, a_srd, a_base, a_lds[TIC][0]);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_barrier();

            __builtin_amdgcn_s_setprio(2);
            kittens::mma_ABt_scaled<CBSZ, BLGP>(cB, a, b1, cB, &sa_h0, &sb_h1);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            kittens::fp8e8m0_4 sa_h1 = lane_rd(scale_smem[TIC][0], 2 + warp_m);
            load_cols<ST_A, 2 * TIC + 1>(a, a_frag);
            load_srd(Bs[TIC][0], B_local, {0, 0, k + 2, n_tile * 2    }, sw_B, b_srd, b_base, b_lds[TIC][0]);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_barrier();

            __builtin_amdgcn_s_setprio(2);
            kittens::mma_ABt_scaled<CBSZ, BLGP>(cC, a, b0, cC, &sa_h1, &sb_h0);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            load_srd(Bs[TIC][1], B_local, {0, 0, k + 2, n_tile * 2 + 1}, sw_B, b_srd, b_base, b_lds[TIC][1]);
            asm volatile("s_waitcnt vmcnt(6)");
            __builtin_amdgcn_s_barrier();

            __builtin_amdgcn_s_setprio(2);
            kittens::mma_ABt_scaled<CBSZ, BLGP>(cD, a, b1, cD, &sa_h1, &sb_h1);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }
        k++;
        {
            constexpr int TIC = 1;
            constexpr int TOC = TIC ^ 1;
            G::load(scale_smem[TOC][0], scale_A_gl, {(k + 1) * sa_stride + sa_batch, 0, 0, 0});
            G::load(scale_smem[TOC][1], scale_B_gl, {2 * ((k + 1) * sb_stride + sb_batch),     0, 0, 0});
            G::load(scale_smem[TOC][2], scale_B_gl, {2 * ((k + 1) * sb_stride + sb_batch) + 1, 0, 0, 0});

            load_cols<ST_B, 2 * TIC>(b0, b_frag);
            load_cols<ST_A, 2 * TIC>(a, a_frag);
            load_srd(As[TOC][1], A_local, {0, 0, k + 1, a_row_tile * 2 + 1}, sw_A, a_srd, a_base, a_lds[TOC][1]);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_barrier();

            kittens::fp8e8m0_4 sa_h0 = lane_rd(scale_smem[TIC][0], warp_m);
            kittens::fp8e8m0_4 sb_h0 = lane_rd(scale_smem[TIC][1], warp_n);
            __builtin_amdgcn_s_setprio(2);
            kittens::mma_ABt_scaled<CBSZ, BLGP>(cA, a, b0, cA, &sa_h0, &sb_h0);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            kittens::fp8e8m0_4 sb_h1 = lane_rd(scale_smem[TIC][2], warp_n);
            load_cols<ST_B, 2 * TIC + 1>(b1, b_frag);
            load_srd(As[TIC][0], A_local, {0, 0, k + 2, a_row_tile * 2    }, sw_A, a_srd, a_base, a_lds[TIC][0]);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_barrier();

            __builtin_amdgcn_s_setprio(2);
            kittens::mma_ABt_scaled<CBSZ, BLGP>(cB, a, b1, cB, &sa_h0, &sb_h1);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            kittens::fp8e8m0_4 sa_h1 = lane_rd(scale_smem[TIC][0], 2 + warp_m);
            load_cols<ST_A, 2 * TIC + 1>(a, a_frag);
            load_srd(Bs[TIC][0], B_local, {0, 0, k + 2, n_tile * 2    }, sw_B, b_srd, b_base, b_lds[TIC][0]);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_barrier();

            __builtin_amdgcn_s_setprio(2);
            kittens::mma_ABt_scaled<CBSZ, BLGP>(cC, a, b0, cC, &sa_h1, &sb_h0);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            load_srd(Bs[TIC][1], B_local, {0, 0, k + 2, n_tile * 2 + 1}, sw_B, b_srd, b_base, b_lds[TIC][1]);
            asm volatile("s_waitcnt vmcnt(6)");
            __builtin_amdgcn_s_barrier();

            __builtin_amdgcn_s_setprio(2);
            kittens::mma_ABt_scaled<CBSZ, BLGP>(cD, a, b1, cD, &sa_h1, &sb_h1);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }
        k++;
    }

    { // Epilogue k = k_iters - 2
        constexpr int TIC = 0;
        constexpr int TOC = TIC ^ 1;
        int k = k_iters - 2;
        G::load(scale_smem[TOC][0], scale_A_gl, {(k + 1) * sa_stride + sa_batch, 0, 0, 0});
        G::load(scale_smem[TOC][1], scale_B_gl, {2 * ((k + 1) * sb_stride + sb_batch),     0, 0, 0});
        G::load(scale_smem[TOC][2], scale_B_gl, {2 * ((k + 1) * sb_stride + sb_batch) + 1, 0, 0, 0});
        asm volatile("s_waitcnt vmcnt(0)");
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        kittens::fp8e8m0_4 sa_h0 = lane_rd(scale_smem[TIC][0], warp_m);
        kittens::fp8e8m0_4 sa_h1 = lane_rd(scale_smem[TIC][0], 2 + warp_m);
        kittens::fp8e8m0_4 sb_h0 = lane_rd(scale_smem[TIC][1], warp_n);
        kittens::fp8e8m0_4 sb_h1 = lane_rd(scale_smem[TIC][2], warp_n);

        load_cols<ST_B, 2 * TIC>(b0, b_frag);
        load_cols<ST_A, 2 * TIC>(a, a_frag);
        load_srd(As[TOC][1], A_local, {0, 0, k + 1, a_row_tile * 2 + 1}, sw_A, a_srd, a_base, a_lds[TOC][1]);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_setprio(2);
        kittens::mma_ABt_scaled<CBSZ, BLGP>(cA, a, b0, cA, &sa_h0, &sb_h0);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        load_cols<ST_B, 2 * TIC + 1>(b1, b_frag);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_setprio(2);
        kittens::mma_ABt_scaled<CBSZ, BLGP>(cB, a, b1, cB, &sa_h0, &sb_h1);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        load_cols<ST_A, 2 * TIC + 1>(a, a_frag);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_setprio(2);
        kittens::mma_ABt_scaled<CBSZ, BLGP>(cC, a, b0, cC, &sa_h1, &sb_h0);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        load_cols<ST_B, 2 * TOC>(b0, b_frag);
        asm volatile("s_waitcnt vmcnt(4)"); // wait for toc data; As[toc][1] still in flight
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_setprio(2);
        kittens::mma_ABt_scaled<CBSZ, BLGP>(cD, a, b1, cD, &sa_h1, &sb_h1);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    }

    { // Final epilogue k = k_iters - 1
        constexpr int TIC = 1;
        asm volatile("s_waitcnt vmcnt(0)");
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        kittens::fp8e8m0_4 sa_h0 = lane_rd(scale_smem[TIC][0], warp_m);
        kittens::fp8e8m0_4 sa_h1 = lane_rd(scale_smem[TIC][0], 2 + warp_m);
        kittens::fp8e8m0_4 sb_h0 = lane_rd(scale_smem[TIC][1], warp_n);
        kittens::fp8e8m0_4 sb_h1 = lane_rd(scale_smem[TIC][2], warp_n);

        load_cols<ST_A, 2 * TIC>(a, a_frag);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_setprio(2);
        kittens::mma_ABt_scaled<CBSZ, BLGP>(cA, a, b0, cA, &sa_h0, &sb_h0);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        load_cols<ST_B, 2 * TIC + 1>(b1, b_frag);
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_setprio(2);
        kittens::mma_ABt_scaled<CBSZ, BLGP>(cB, a, b1, cB, &sa_h0, &sb_h1);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        load_cols<ST_A, 2 * TIC + 1>(a, a_frag);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_setprio(2);
        kittens::mma_ABt_scaled<CBSZ, BLGP>(cC, a, b0, cC, &sa_h1, &sb_h0);
        kittens::mma_ABt_scaled<CBSZ, BLGP>(cD, a, b1, cD, &sa_h1, &sb_h1);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
    }

    OutGL C_local(C);
    if constexpr (SPLIT_K) {
        C_local.raw_ptr += (size_t)split_id * N * C_local.template stride<2>();
    }

    gemm_epilogue<EPILOGUE, ACCUMULATE, true, EDGE, RT_C, RT_C_T>(cA, cB, cC, cD, C_local, AuxGL, bias, bias_dtype,
        block_m, block_row, block_col, warp_m, warp_n);
}

template<bool TRANSA, bool TRANSB, GemmEpilogue EPILOGUE, int CBSZ, int BLGP, bool ACCUMULATE, typename AuxGLType>
static void launch_gemm_typed(
    const void *A, const void *B, void *C,
    const uint32_t *packed_sa, const uint32_t *packed_sb,
    const void *bias, int bias_dtype, AuxGLType aux_gl,
    int M, int N, int K, OutDtype out_dtype, hipStream_t stream) {

    int tiles_M = kittens::ceil_div(M, BLOCK_ROW);
    int tiles_N = kittens::ceil_div(N, BLOCK_COL);
    int grid    = tiles_M * tiles_N;
    int k_iters = std::max(kittens::ceil_div(K, BLOCK_K), 2);
    bool edge = M % BLOCK_ROW || N % BLOCK_COL || K % BLOCK_K || K < 2 * BLOCK_K;

    gl_fp8_rt gl_A((kittens::fp8e4m3 *)A, nullptr, nullptr,
                   TRANSA ? (size_t)M : (size_t)K, TRANSA ? (size_t)K : (size_t)M);
    gl_fp8_rt gl_B((kittens::fp8e4m3 *)B, nullptr, nullptr,
                   TRANSB ? (size_t)K : (size_t)N, TRANSB ? (size_t)N : (size_t)K);
    gl_scale_rt gl_SA(reinterpret_cast<kittens::fp8e8m0 *>(const_cast<uint32_t *>(packed_sa)),
                      k_iters * tiles_M, nullptr, nullptr, nullptr);
    // B scale buffer is 2 tiles for the hi/lo group split
    gl_scale_rt gl_SB(reinterpret_cast<kittens::fp8e8m0 *>(const_cast<uint32_t *>(packed_sb)),
                      2 * k_iters * tiles_N, nullptr, nullptr, nullptr);

    auto launch = [&](auto gl_C) {
        KITTENS_BOOL_SWITCH(edge, EDGE,
            if constexpr (TRANSA && !TRANSB) {
                mxfp8_gemm_tn_kernel<EPILOGUE, CBSZ, BLGP, ACCUMULATE, EDGE><<<grid, NUM_THREADS, 0, stream>>>(
                    gl_A, gl_B, gl_C, aux_gl, gl_SA, gl_SB, bias, bias_dtype, N, K, tiles_M, tiles_N, 1);
            } else if constexpr (!TRANSA && !TRANSB) {
                // FLIPPED slots: the A slot takes B's tensor (the activation, K-contiguous [N,K]) and
                // the B slot takes A's tensor (the weight, K-major [K,M]); the scale buffers follow.
                // gl_C and aux_gl are unchanged -- the accumulator lands in C's existing [N,M] memory.
                gl_fp8_rt gl_A_nn((kittens::fp8e4m3 *)B, nullptr, nullptr, (size_t)N, (size_t)K);
                gl_fp8_rt gl_B_nn((kittens::fp8e4m3 *)A, nullptr, nullptr, (size_t)K, (size_t)M);
                gl_scale_rt gl_SA_nn(reinterpret_cast<kittens::fp8e8m0 *>(const_cast<uint32_t *>(packed_sa)),
                                     k_iters * tiles_N, nullptr, nullptr, nullptr);
                gl_scale_rt gl_SB_nn(reinterpret_cast<kittens::fp8e8m0 *>(const_cast<uint32_t *>(packed_sb)),
                                     2 * k_iters * tiles_M, nullptr, nullptr, nullptr);
                mxfp8_gemm_nn_kernel<EPILOGUE, CBSZ, BLGP, ACCUMULATE, EDGE><<<grid, NUM_THREADS, 0, stream>>>(
                    gl_A_nn, gl_B_nn, gl_C, aux_gl, gl_SA_nn, gl_SB_nn, bias, bias_dtype, N, K, tiles_M, tiles_N, 1);
            } else {
                mxfp8_gemm_nt_kernel<false, EPILOGUE, CBSZ, BLGP, ACCUMULATE, EDGE><<<grid, NUM_THREADS, 0, stream>>>(
                    gl_A, gl_B, gl_C, aux_gl, gl_SA, gl_SB, bias, bias_dtype, N, K, tiles_M, tiles_N, 1);
            }
        ) // NOLINT(*)
    };
    if (out_dtype == OutDtype::BF16) {
        launch(gl_bf16_rt((kittens::bf16 *)C, nullptr, nullptr, (size_t)N, (size_t)M));
    } else if (out_dtype == OutDtype::FP16) {
        launch(gl_fp16_rt((half *)C, nullptr, nullptr, (size_t)N, (size_t)M));
    } else {
        launch(gl_f32_rt((float *)C, nullptr, nullptr, (size_t)N, (size_t)M));
    }
}

template<bool TRANSA, bool TRANSB, GemmEpilogue EPILOGUE, int CBSZ, int BLGP, bool ACCUMULATE>
static void launch_gemm(
    const void *A, const void *B, void *C,
    const uint32_t *packed_sa, const uint32_t *packed_sb,
    const void *bias, int bias_dtype, void *aux_gelu,
    int M, int N, int K, OutDtype out_dtype, OutDtype aux_dtype, hipStream_t stream) {

    if (aux_gelu && aux_dtype == OutDtype::BF16) {
        gl_bf16_rt aux_gl((kittens::bf16 *)aux_gelu, nullptr, nullptr, (size_t)N, (size_t)M);
        launch_gemm_typed<TRANSA, TRANSB, EPILOGUE, CBSZ, BLGP, ACCUMULATE>(A, B, C, packed_sa, packed_sb,
            bias, bias_dtype, aux_gl, M, N, K, out_dtype, stream);
    } else if (aux_gelu && aux_dtype == OutDtype::FP16) {
        gl_fp16_rt aux_gl((half *)aux_gelu, nullptr, nullptr, (size_t)N, (size_t)M);
        launch_gemm_typed<TRANSA, TRANSB, EPILOGUE, CBSZ, BLGP, ACCUMULATE>(A, B, C, packed_sa, packed_sb,
            bias, bias_dtype, aux_gl, M, N, K, out_dtype, stream);
    } else {
        static float _ = 0.f;
        gl_f32_rt aux_gl(aux_gelu ? (float *)aux_gelu : &_, nullptr, nullptr,
                         aux_gelu ? (size_t)N : 1, aux_gelu ? (size_t)M : 1);
        launch_gemm_typed<TRANSA, TRANSB, EPILOGUE, CBSZ, BLGP, ACCUMULATE>(A, B, C, packed_sa, packed_sb,
            bias, bias_dtype, aux_gl, M, N, K, out_dtype, stream);
    }
}

template<bool TRANSA, bool TRANSB, bool ACCUMULATE>
static void dispatch_gemm(
    GemmEpilogue epilogue, int a_fp8, int b_fp8,
    const void *A, const void *B, void *C,
    const uint32_t *packed_sa, const uint32_t *packed_sb,
    const void *bias, int bias_dtype, void *aux_gelu,
    int M, int N, int K, OutDtype out_dtype, OutDtype aux_dtype, hipStream_t stream) {

    // CBSZ/BLGP: 0 = e4m3, 1 = e5m2. CBSZ names the kernel's A-SLOT format and BLGP its B-slot,
    // so for NN -- whose slots hold the opposite tensors -- the codes swap with them. Only mixed
    // e4m3/e5m2 pairs expose a mistake here.
    constexpr bool NN_SLOTS_SWAPPED = !TRANSA && !TRANSB;
    const int a_slot_fp8 = NN_SLOTS_SWAPPED ? b_fp8 : a_fp8;
    const int b_slot_fp8 = NN_SLOTS_SWAPPED ? a_fp8 : b_fp8;

    auto dispatch = [&]<GemmEpilogue EPI>() {
        if (a_slot_fp8 == 0 && b_slot_fp8 == 0) {
            launch_gemm<TRANSA, TRANSB, EPI, 0, 0, ACCUMULATE>(A, B, C, packed_sa, packed_sb,
                bias, bias_dtype, aux_gelu, M, N, K, out_dtype, aux_dtype, stream);
        } else if (a_slot_fp8 == 0 && b_slot_fp8 == 1) {
            launch_gemm<TRANSA, TRANSB, EPI, 0, 1, ACCUMULATE>(A, B, C, packed_sa, packed_sb,
                bias, bias_dtype, aux_gelu, M, N, K, out_dtype, aux_dtype, stream);
        } else if (a_slot_fp8 == 1 && b_slot_fp8 == 0) {
            launch_gemm<TRANSA, TRANSB, EPI, 1, 0, ACCUMULATE>(A, B, C, packed_sa, packed_sb,
                bias, bias_dtype, aux_gelu, M, N, K, out_dtype, aux_dtype, stream);
        } else {
            launch_gemm<TRANSA, TRANSB, EPI, 1, 1, ACCUMULATE>(A, B, C, packed_sa, packed_sb,
                bias, bias_dtype, aux_gelu, M, N, K, out_dtype, aux_dtype, stream);
        }
    };
    switch (epilogue) {
        case GemmEpilogue::DEFAULT:       dispatch.template operator()<GemmEpilogue::DEFAULT>(); break;
        case GemmEpilogue::BIAS:          dispatch.template operator()<GemmEpilogue::BIAS>(); break;
        case GemmEpilogue::GELU_AUX:      dispatch.template operator()<GemmEpilogue::GELU_AUX>(); break;
        case GemmEpilogue::GELU_AUX_BIAS: dispatch.template operator()<GemmEpilogue::GELU_AUX_BIAS>(); break;
    }
}

constexpr int SPLITK_MAX_SPLITS     = 8;
constexpr int SPLITK_MIN_K          = 512;
constexpr int SPLITK_WAVE_CTAS      = 256;
constexpr int SPLITK_ITER_WEIGHT    = 16;
constexpr int SPLITK_REDUCE_THREADS = 256;
constexpr size_t SPLITK_MAX_PARTIAL_BYTES = (size_t)64 << 20;

template <typename OutT>
__global__ __launch_bounds__(SPLITK_REDUCE_THREADS)
void splitk_reduce_kernel(const float4 *__restrict__ partials, OutT *__restrict__ out, size_t n4, int splits) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n4) {
        return;
    }
    float4 acc = partials[i];
    for (int s = 1; s < splits; s++) {
        float4 v = partials[(size_t)s * n4 + i];
        acc.x += v.x;
        acc.y += v.y;
        acc.z += v.z;
        acc.w += v.w;
    }
    out[4 * i]     = static_cast<OutT>(acc.x);
    out[4 * i + 1] = static_cast<OutT>(acc.y);
    out[4 * i + 2] = static_cast<OutT>(acc.z);
    out[4 * i + 3] = static_cast<OutT>(acc.w);
}

static size_t splitk_workspace_bytes(int M, int N, int splits) {
    return (size_t)splits * M * N * sizeof(float);
}

// Split count minimizing ITER_WEIGHT * waves * K iterations per split + splits * tiles (the partials' traffic).
static int select_splitk(int M, int N, int K) {
    const size_t tiles = (size_t)kittens::ceil_div(M, BLOCK_ROW) * kittens::ceil_div(N, BLOCK_COL);
    size_t best_cost = 0;
    int best_splits = 1;
    for (int splits = 1; splits <= SPLITK_MAX_SPLITS; splits++) {
        if (splits > 1 && (K % (splits * BLOCK_K) != 0 || K / splits < SPLITK_MIN_K ||
                           splitk_workspace_bytes(M, N, splits) > SPLITK_MAX_PARTIAL_BYTES)) {
            continue;
        }
        const size_t waves = (tiles * splits + SPLITK_WAVE_CTAS - 1) / SPLITK_WAVE_CTAS;
        const size_t cost  = SPLITK_ITER_WEIGHT * waves * (K / (splits * BLOCK_K)) + (splits > 1 ? splits * tiles : 0);
        if (splits == 1 || cost < best_cost) {
            best_cost = cost;
            best_splits = splits;
        }
    }
    return best_splits;
}

template<bool TRANSA, bool TRANSB, int CBSZ, int BLGP>
static void launch_splitk(
    const void *A, const void *B, void *C,
    const uint32_t *packed_sa, const uint32_t *packed_sb,
    int M, int N, int K, int splits, OutDtype out_dtype, uint8_t *ws, hipStream_t stream) {

    const int tiles_M = kittens::ceil_div(M, BLOCK_ROW);
    const int tiles_N = kittens::ceil_div(N, BLOCK_COL);
    const int k_iters = K / BLOCK_K;
    const int grid    = tiles_M * tiles_N * splits;

    float *partials = (float *)ws;

    gl_fp8_rt gl_A((kittens::fp8e4m3 *)A, nullptr, nullptr,
                   TRANSA ? (size_t)M : (size_t)K, TRANSA ? (size_t)K : (size_t)M);
    gl_fp8_rt gl_B((kittens::fp8e4m3 *)B, nullptr, nullptr,
                   TRANSB ? (size_t)K : (size_t)N, TRANSB ? (size_t)N : (size_t)K);
    gl_scale_rt gl_SA(reinterpret_cast<kittens::fp8e8m0 *>(const_cast<uint32_t *>(packed_sa)),
                      k_iters * tiles_M, nullptr, nullptr, nullptr);
    gl_scale_rt gl_SB(reinterpret_cast<kittens::fp8e8m0 *>(const_cast<uint32_t *>(packed_sb)),
                      2 * k_iters * tiles_N, nullptr, nullptr, nullptr);
    gl_f32_rt gl_C(partials, nullptr, nullptr, (size_t)N, (size_t)M);
    static float dummy_aux = 0.f;
    gl_f32_rt aux_gl(&dummy_aux, nullptr, nullptr, 1, 1);

    KITTENS_BOOL_SWITCH(M % BLOCK_ROW || N % BLOCK_COL, EDGE,
        if constexpr (TRANSA && !TRANSB) {
            mxfp8_gemm_tn_kernel<GemmEpilogue::DEFAULT, CBSZ, BLGP, false, EDGE, true>
                <<<grid, NUM_THREADS, 0, stream>>>(
                gl_A, gl_B, gl_C, aux_gl, gl_SA, gl_SB, nullptr, 0, N, K, tiles_M, tiles_N, splits);
        } else if constexpr (!TRANSA && !TRANSB) {
            gl_fp8_rt gl_A_nn((kittens::fp8e4m3 *)B, nullptr, nullptr, (size_t)N, (size_t)K);
            gl_fp8_rt gl_B_nn((kittens::fp8e4m3 *)A, nullptr, nullptr, (size_t)K, (size_t)M);
            gl_scale_rt gl_SA_nn(reinterpret_cast<kittens::fp8e8m0 *>(const_cast<uint32_t *>(packed_sa)),
                                 k_iters * tiles_N, nullptr, nullptr, nullptr);
            gl_scale_rt gl_SB_nn(reinterpret_cast<kittens::fp8e8m0 *>(const_cast<uint32_t *>(packed_sb)),
                                 2 * k_iters * tiles_M, nullptr, nullptr, nullptr);
            mxfp8_gemm_nn_kernel<GemmEpilogue::DEFAULT, CBSZ, BLGP, false, EDGE, true>
                <<<grid, NUM_THREADS, 0, stream>>>(
                gl_A_nn, gl_B_nn, gl_C, aux_gl, gl_SA_nn, gl_SB_nn, nullptr, 0, N, K, tiles_M, tiles_N, splits);
        } else {
            mxfp8_gemm_nt_kernel<true, GemmEpilogue::DEFAULT, CBSZ, BLGP, false, EDGE>
                <<<grid, NUM_THREADS, 0, stream>>>(
                gl_A, gl_B, gl_C, aux_gl, gl_SA, gl_SB, nullptr, 0, N, K, tiles_M, tiles_N, splits);
        }
    ) // NOLINT(*)

    const size_t n4 = (size_t)M * N * sizeof(float) / sizeof(float4);
    const int reduce_grid = kittens::ceil_div((int)n4, SPLITK_REDUCE_THREADS);
    const float4 *partials4 = (const float4 *)partials;
    if (out_dtype == OutDtype::BF16) {
        splitk_reduce_kernel<<<reduce_grid, SPLITK_REDUCE_THREADS, 0, stream>>>(
            partials4, (kittens::bf16 *)C, n4, splits);
    } else if (out_dtype == OutDtype::FP16) {
        splitk_reduce_kernel<<<reduce_grid, SPLITK_REDUCE_THREADS, 0, stream>>>(
            partials4, (half *)C, n4, splits);
    } else {
        splitk_reduce_kernel<<<reduce_grid, SPLITK_REDUCE_THREADS, 0, stream>>>(
            partials4, (float *)C, n4, splits);
    }
}

static bool check_tn_constraints(int M, int N, int K) {
    return M > 0 && N > 0 && K > 0 && M % 16 == 0 && N % 16 == 0 && K % 32 == 0;
}

static GemmEpilogue select_epilogue(const void *bias, void *aux_gelu) {
    if (bias && aux_gelu) return GemmEpilogue::GELU_AUX_BIAS;
    if (aux_gelu)         return GemmEpilogue::GELU_AUX;
    if (bias)             return GemmEpilogue::BIAS;
    return GemmEpilogue::DEFAULT;
}

template<bool TRANSA, bool TRANSB, bool ACCUMULATE>
static bool mxfp8_gemm_impl(
    const void *A, const void *B, void *C,
    const void *scale_A, const void *scale_B,
    int M, int N, int K,
    int a_fp8_code, int b_fp8_code,
    const void *bias, int bias_dtype_code,
    void *aux_gelu, int out_dtype_code, int aux_dtype_code,
    void *workspace, size_t workspace_size,
    hipStream_t stream) {

    if (!check_tn_constraints(M, N, K)) return false;

    int k_iters  = std::max(kittens::ceil_div(K, BLOCK_K), 2);
    int kb_shift = (k_iters * BLOCK_K - K) / 32;
    int scale_K  = K / 32;
    int tiles_M  = kittens::ceil_div(M, BLOCK_ROW);
    int tiles_N  = kittens::ceil_div(N, BLOCK_COL);
    if (!(TRANSA && !TRANSB) && (K % BLOCK_K || K < 2 * BLOCK_K) &&
        (size_t)std::max(M, N) * k_iters * BLOCK_K > 0x80000000ull) {
        return false;
    }

    // Lane-native scale buffers: the A slot is 256 words/tile, the B slot 512 (hi/lo pair). NN's
    // slots are flipped, so the activation (dim N) feeds the A slot; every other layout is A->A.
    constexpr bool NN_FLIP = !TRANSA && !TRANSB;
    int a_slot_tiles = NN_FLIP ? tiles_N : tiles_M;
    int b_slot_tiles = NN_FLIP ? tiles_M : tiles_N;
    size_t sa_bytes = kittens_align_up((size_t)k_iters * a_slot_tiles * 256 * sizeof(uint32_t), 256);
    size_t sb_bytes = kittens_align_up((size_t)k_iters * b_slot_tiles * 512 * sizeof(uint32_t), 256);
    if (workspace_size < sa_bytes + sb_bytes) return false;

    auto *packed_sa = (uint32_t *)workspace;
    auto *packed_sb = (uint32_t *)((uint8_t *)workspace + sa_bytes);

    // COLWISE describes the raw scale buffer's own layout and stays welded to its tensor
    // (scale_A -> !TRANSA, scale_B -> TRANSB); only the <STEP,NG> geometry follows the slot.
    if constexpr (NN_FLIP) {
        launch_pack_scales<TRANSB, 64, 4>((const uint8_t *)scale_B, packed_sa, N, scale_K, k_iters, stream, kb_shift);
        launch_pack_scales<!TRANSA, 32, 8>((const uint8_t *)scale_A, packed_sb, M, scale_K, k_iters, stream, kb_shift);
    } else {
        launch_pack_scales<!TRANSA, 64, 4>((const uint8_t *)scale_A, packed_sa, M, scale_K, k_iters, stream, kb_shift);
        launch_pack_scales<TRANSB, 32, 8>((const uint8_t *)scale_B, packed_sb, N, scale_K, k_iters, stream, kb_shift);
    }

    GemmEpilogue ep = select_epilogue(bias, aux_gelu);
    if constexpr (!ACCUMULATE) {
        int splits = ep == GemmEpilogue::DEFAULT ? select_splitk(M, N, K) : 1;
        if (splits > 1 && workspace_size >= sa_bytes + sb_bytes + splitk_workspace_bytes(M, N, splits)) {
            KITTENS_BOOL_SWITCH(NN_FLIP ? b_fp8_code : a_fp8_code, CBSZ,
                KITTENS_BOOL_SWITCH(NN_FLIP ? a_fp8_code : b_fp8_code, BLGP,
                    launch_splitk<TRANSA, TRANSB, CBSZ, BLGP>(A, B, C, packed_sa, packed_sb, M, N, K, splits,
                        static_cast<OutDtype>(out_dtype_code), (uint8_t *)workspace + sa_bytes + sb_bytes, stream);
            )) // NOLINT(*)
            return true;
        }
    }
    dispatch_gemm<TRANSA, TRANSB, ACCUMULATE>(ep, a_fp8_code, b_fp8_code,
        A, B, C, packed_sa, packed_sb, bias, bias_dtype_code,
        aux_gelu, M, N, K,
        static_cast<OutDtype>(out_dtype_code),
        static_cast<OutDtype>(aux_dtype_code), stream);
    return true;
}

static int out_code(int dt) {
    switch (dt) {
    case KITTENS_FLOAT32:  return 0;
    case KITTENS_BFLOAT16: return 1;
    case KITTENS_FLOAT16:  return 2;
    default: assert(0 && "unexpected output dtype"); return 0;
    }
}

static bool mxfp8_gemm(
    const void *A, const void *B, void *C,
    const void *scale_A, const void *scale_B,
    int M, int N, int K,
    bool transa, bool transb,
    int a_dtype, int b_dtype,
    const void *bias, int bias_dtype,
    void *aux_gelu, int out_dtype, int aux_dtype,
    float beta,
    void *workspace, size_t workspace_size,
    hipStream_t stream) {

    int a_fp8   = fp8_code(static_cast<KittensDType>(a_dtype));
    int b_fp8   = fp8_code(static_cast<KittensDType>(b_dtype));
    int out_dc  = out_code(out_dtype);
    int bias_dc = bias ? out_code(bias_dtype) : 0;
    int aux_dc  = aux_gelu ? out_code(aux_dtype) : 0;
    bool accumulate = beta != 0.0f;

    bool result = false;
    KITTENS_BOOL_SWITCH(transa, TRANSA,
        KITTENS_BOOL_SWITCH(transb, TRANSB,
            KITTENS_BOOL_SWITCH(accumulate, ACCUMULATE,
                if constexpr (!(TRANSA && TRANSB)) {
                    result = mxfp8_gemm_impl<TRANSA, TRANSB, ACCUMULATE>(A, B, C, scale_A, scale_B, M, N, K,
                        a_fp8, b_fp8, bias, bias_dc, aux_gelu, out_dc, aux_dc,
                        workspace, workspace_size, stream);
                } else {
                    assert(0 && "mxfp8_gemm: TT layout is not supported");
                }
    ))) // NOLINT(*)
    return result;
}

// Traces why a grouped launch was declined
static void warn_fallback(const char *tag, const char *reason) {
    static bool enabled = [] {
        const char *v = std::getenv("NVTE_CUTLASS_GROUPED_GEMM_WARN_FALLBACK");
        return v && v[0] == '1';
    }();
    if (enabled) {
        fprintf(stderr, "[%s] falling back: %s\n", tag, reason);
    }
}

constexpr int SCHED_CHUNK      = 64;
constexpr int SCHED_CTR_GLOBAL = NUM_XCDS_AFF;
constexpr int SCHED_CTR_BYTES  = 256;
constexpr int MAX_TAB_EXPERTS  = 64;

struct GroupedTab {
    const void *a[MAX_TAB_EXPERTS];
    const void *b[MAX_TAB_EXPERTS];
    void *c[MAX_TAB_EXPERTS];
    int tile_off[MAX_TAB_EXPERTS + 1];
    int rows[MAX_TAB_EXPERTS];
};

struct WgradExpertInfo {
    const void *a_ptr;
    const void *b_ptr;
    void *d_ptr;
    int k_iters;
    int sa_tile_offset;
    int sb_tile_offset;
    int shift;
};

struct WgradTab {
    WgradExpertInfo info[MAX_TAB_EXPERTS];
};

struct PackTab {
    const uint8_t *src[MAX_TAB_EXPERTS];
    int off[MAX_TAB_EXPERTS + 1];
    int k_iters[MAX_TAB_EXPERTS];
    int tokens[MAX_TAB_EXPERTS];
};

enum struct ScalePack {
    UNIFORM,
    CONCAT,
    VARYING,
};

struct GroupedNoAux {};

template <bool COLWISE, int STEP, int NG, ScalePack MODE, bool BOUNDED = false>
__global__ void pack_scales_tab_kernel(const PackTab tab, uint32_t *__restrict__ ln, int num_experts, int dim,
    int scale_K, int k_iters, int tiles_per_col, int expert_stride) {

    int tile_id = blockIdx.x;
    int e       = blockIdx.y;
    if constexpr (MODE == ScalePack::CONCAT) {
        e = hk_upper_bound(tab.off, num_experts, tile_id / k_iters);
        int tpc = tab.off[e + 1] - tab.off[e];
        tile_id -= k_iters * tab.off[e];
        pack_scale_tile<COLWISE, STEP, NG, BOUNDED>(tab.src[e], ln + (size_t)k_iters * tab.off[e] * NG * 64,
                                                    BOUNDED ? tab.tokens[e] : tpc * 256, scale_K, tile_id / tpc,
                                                    tile_id % tpc, tile_id);
    } else if constexpr (MODE == ScalePack::VARYING) {
        int e_k_iters = tab.k_iters[e];
        if (tile_id >= e_k_iters * tiles_per_col) {
            return;
        }
        int e_scale_K = BOUNDED ? tab.tokens[e] / 32 : e_k_iters * 4;
        pack_scale_tile<COLWISE, STEP, NG, BOUNDED>(tab.src[e], ln + tab.off[e], dim, e_scale_K,
                                                    tile_id / tiles_per_col, tile_id % tiles_per_col, tile_id,
                                                    e_k_iters * 4 - e_scale_K);
    } else {
        pack_scale_tile<COLWISE, STEP, NG>(tab.src[e], ln + (size_t)e * expert_stride, dim, scale_K,
                                           tile_id / tiles_per_col, tile_id % tiles_per_col, tile_id);
    }
}

template <int CBSZ, int BLGP, bool EDGE, typename OutGL>
__global__ __launch_bounds__(NUM_THREADS, 2)
void mxfp8_grouped_tn_kernel(const gl_fp8_rt A, const gl_fp8_rt B, const OutGL C,
    const gl_scale_rt scale_A_gl, const gl_scale_rt scale_B_gl, const GroupedTab tab, int num_experts,
    int K, int total_m_tiles, int tiles_N, int num_tiles, int *__restrict__ ctr) {

    int k_iters   = K / BLOCK_K;
    int sa_stride = total_m_tiles;

    using ST_A     = kittens::st_fp8e4m3<HALF_ROW, BLOCK_K, kittens::st_16x128_s>;
    using ST_B     = kittens::st_fp8e4m3<HALF_COL, BLOCK_K, kittens::st_16x128_s>;
    using RT_A     = kittens::rt_fp8e4m3<REG_M, BLOCK_K>;
    using RT_B     = kittens::rt_fp8e4m3<REG_N, BLOCK_K>;
    using RT_C     = kittens::rt_fl<REG_M, REG_N, kittens::col_l, kittens::rt_16x16_s>;
    using RT_C_T   = kittens::rt_fl<REG_N, REG_M, kittens::row_l, kittens::rt_16x16_s>;

    __shared__ ST_A As_s[2][2];
    __shared__ ST_B Bs_s[2][2];
    __shared__ ST_Scale scale_s[2][3];
    __shared__ int s_wgid;
    __shared__ XcdBuckets buckets;

    int warp_m = __builtin_amdgcn_readfirstlane(kittens::warpid() / WARPS_COL);
    int warp_n = __builtin_amdgcn_readfirstlane(kittens::warpid() % WARPS_COL);

    using T = kittens::fp8e4m3;

#include "xcd_sched.inc"
        uint32_t lds_zero;
        asm volatile("s_mov_b32 %0, 0" : "=s"(lds_zero));
        auto &As = *(decltype(As_s) *)(__attribute__((address_space(3))) decltype(As_s) *)(uintptr_t)
                   (lds_zero + (uint32_t)reinterpret_cast<uintptr_t>(&As_s));
        auto &Bs = *(decltype(Bs_s) *)(__attribute__((address_space(3))) decltype(Bs_s) *)(uintptr_t)
                   (lds_zero + (uint32_t)reinterpret_cast<uintptr_t>(&Bs_s));
        auto &scale_smem = *(decltype(scale_s) *)(__attribute__((address_space(3))) decltype(scale_s) *)
                           (uintptr_t)(lds_zero + (uint32_t)reinterpret_cast<uintptr_t>(&scale_s));

        const uint32_t a_frag0 = row_frag_addr<RT_A>(As[0][0], warp_m, 0);
        const uint32_t a_frag1 = row_frag_addr<RT_A>(As[0][0], warp_m, 1);
        const uint32_t b_frag0 = row_frag_addr<RT_B>(Bs[0][0], warp_n, 0);
        const uint32_t b_frag1 = row_frag_addr<RT_B>(Bs[0][0], warp_n, 1);

        RT_A a;
        RT_B b0, b1;
        RT_C cA, cB, cC, cD;
        kittens::zero(cA); kittens::zero(cB); kittens::zero(cC); kittens::zero(cD);

        const int WGM = 8;
        int num_wgid_in_group = WGM * tiles_N;
        int group_id          = wgid / num_wgid_in_group;
        int first_pid_m       = group_id * WGM;
        int group_size_m      = min(total_m_tiles - first_pid_m, WGM);
        int m_tile            = first_pid_m + ((wgid % num_wgid_in_group) % group_size_m);
        int n_tile            = (wgid % num_wgid_in_group) / group_size_m;

        int expert_id = hk_upper_bound(tab.tile_off, num_experts, n_tile);
        n_tile -= tab.tile_off[expert_id];

        int block_m   = m_tile * BLOCK_ROW;
        int block_row = m_tile;
        int block_col = n_tile;
        int sa_batch  = expert_id * k_iters * total_m_tiles + m_tile;
        int sb_batch  = k_iters * tab.tile_off[expert_id] + n_tile;
        int sb_stride = tab.tile_off[expert_id + 1] - tab.tile_off[expert_id];

        gl_fp8_rt A_local(A);
        gl_fp8_rt B_local(B);
        A_local.raw_ptr = (T *)tab.a[expert_id];
        B_local.raw_ptr = (T *)tab.b[expert_id];

        const uint32_t a_row_stride = A_local.template stride<2>() * sizeof(T);
        const uint32_t b_row_stride = B_local.template stride<2>() * sizeof(T);
        const int b_rows = EDGE ? tab.rows[expert_id] - n_tile * BLOCK_COL : BLOCK_COL;
        const TileLoad a_dma = prefill_tile_load<NUM_WARPS>(&A_local[{0, 0, block_m, 0}], HALF_ROW * a_row_stride,
                                                            BLOCK_K * sizeof(T), HALF_ROW * a_row_stride,
                                                            a_row_stride, As);
        const TileLoad b_dma = prefill_tile_load<NUM_WARPS>(&B_local[{0, 0, n_tile * BLOCK_COL, 0}],
                                                            HALF_COL * b_row_stride, BLOCK_K * sizeof(T),
                                                            min(b_rows, HALF_COL) * b_row_stride, b_row_stride, Bs, 0,
                                                            max(min(b_rows - HALF_COL, HALF_COL), 0) * b_row_stride);
        const TileLoad &a_k0 = a_dma;
        const TileLoad &b_k0 = b_dma;
        const ScaleLoad s_dma = prefill_scale_load(scale_A_gl, scale_B_gl, sa_batch, 2 * sb_batch, sa_stride,
                                                   2 * sb_stride, scale_smem);

        const int n_iters = k_iters - 2;
        const bool s0 = n_iters & 1;

#include "mxfp8_tn_preload.inc"

#define MXFP8_TN_OPAQUE_PEEL 1
#include "mxfp8_tn_mainloop.inc"
#undef MXFP8_TN_OPAQUE_PEEL

        if (warp_m == 0) {
            __builtin_amdgcn_s_barrier();
        }

        OutGL C_local(C);
        C_local.raw_ptr = (typename OutGL::dtype *)tab.c[expert_id];
        if constexpr (EDGE) {
            C_local.rows_internal = tab.rows[expert_id];
        }

        gemm_epilogue<GemmEpilogue::DEFAULT, false, true, EDGE, RT_C, RT_C_T>(cA, cB, cC, cD, C_local,
            GroupedNoAux{}, nullptr, 0, block_m, block_row, block_col, warp_m, warp_n);
    }
}

template <int CBSZ, int BLGP, bool EDGE, typename OutGL>
__global__ __launch_bounds__(NUM_THREADS, 2)
void mxfp8_grouped_nn_kernel(const gl_fp8_rt A, const gl_fp8_rt B, const OutGL C,
    const gl_scale_rt scale_A_gl, const gl_scale_rt scale_B_gl, const GroupedTab tab, int num_experts,
    int K, int total_m_tiles, int tiles_N, int num_tiles, int *__restrict__ ctr) {

    int k_iters = K / BLOCK_K;

    using ST_A     = kittens::st_fp8e4m3<HALF_ROW, BLOCK_K, kittens::st_16x128_s>;
    using ST_B     = kittens::st_fp8e4m3<BLOCK_K, HALF_COL, kittens::st_16x128_s>;
    using RT_A     = kittens::rt_fp8e4m3<REG_M, BLOCK_K>;
    using RT_B     = kittens::rt<kittens::fp8e4m3, REG_N, BLOCK_K, kittens::col_l, kittens::rt_16x128_s>;
    using RT_C     = kittens::rt_fl<REG_M, REG_N, kittens::col_l, kittens::rt_16x16_s>;

    __shared__ ST_A As_s[2][2];
    __shared__ alignas(ST_B::underlying_subtile_bytes) ST_B Bs_s[2][2];
    __shared__ ST_Scale scale_s[2][3];
    __shared__ int s_wgid;
    __shared__ XcdBuckets buckets;

    int warp_m = __builtin_amdgcn_readfirstlane(kittens::warpid() / WARPS_COL);
    int warp_n = __builtin_amdgcn_readfirstlane(kittens::warpid() % WARPS_COL);

    using T = kittens::fp8e4m3;

    constexpr int bpt      = ST_A::underlying_subtile_bytes_per_thread;
    constexpr int bpm      = bpt * NUM_THREADS;
    constexpr int copies_A = HALF_ROW * BLOCK_K * sizeof(T) / bpm;
    constexpr int copies_B = BLOCK_K * HALF_COL * sizeof(T) / bpm;
    uint32_t sw_A[copies_A], sw_B[copies_B];
    G::prefill_swizzled_offsets(As_s[0][0], A, sw_A);
    G::prefill_swizzled_offsets(Bs_s[0][0], B, sw_B);
    uint32_t (&sw_A0)[copies_A] = sw_A;
    uint32_t (&sw_B0)[copies_B] = sw_B;

#include "xcd_sched.inc"
        uint32_t lds_zero;
        asm volatile("s_mov_b32 %0, 0" : "=s"(lds_zero));
        auto &As = *(decltype(As_s) *)(__attribute__((address_space(3))) decltype(As_s) *)(uintptr_t)
                   (lds_zero + (uint32_t)reinterpret_cast<uintptr_t>(&As_s));
        auto &Bs = *(decltype(Bs_s) *)(__attribute__((address_space(3))) decltype(Bs_s) *)(uintptr_t)
                   (lds_zero + (uint32_t)reinterpret_cast<uintptr_t>(&Bs_s));
        auto &scale_smem = *(decltype(scale_s) *)(__attribute__((address_space(3))) decltype(scale_s) *)
                           (uintptr_t)(lds_zero + (uint32_t)reinterpret_cast<uintptr_t>(&scale_s));

        const int wid = kittens::warpid() % NUM_WARPS;
        constexpr int elem_per_warp = (16 / sizeof(T)) * kittens::WARP_THREADS;
        uint32_t a_lds[2][2], b_lds[2][2];
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                a_lds[i][j] = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
                    reinterpret_cast<uintptr_t>(&As[i][j].data[0]) + wid * elem_per_warp * sizeof(T)));
                b_lds[i][j] = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
                    reinterpret_cast<uintptr_t>(&Bs[i][j].data[0]) + wid * elem_per_warp * sizeof(T)));
            }
        }

        const uint32_t a_frag0 = row_frag_addr<RT_A>(As[0][0], warp_m, 0);
        const uint32_t a_frag1 = row_frag_addr<RT_A>(As[0][0], warp_m, 1);
        uint32_t b_frag[RT_B::height];
        col_frag_addrs<RT_B>(b_frag, Bs[0][0], warp_n * REG_N);

        RT_A a;
        RT_B b0, b1;
        RT_C cA, cB, cC, cD;
        kittens::zero(cA); kittens::zero(cB); kittens::zero(cC); kittens::zero(cD);

        const int WGM = 8;
        int num_wgid_in_group = WGM * tiles_N;
        int group_id          = wgid / num_wgid_in_group;
        int first_pid_m       = group_id * WGM;
        int group_size_m      = min(total_m_tiles - first_pid_m, WGM);
        int m_tile            = first_pid_m + ((wgid % num_wgid_in_group) % group_size_m);
        int n_tile            = (wgid % num_wgid_in_group) / group_size_m;

        int expert_id = hk_upper_bound(tab.tile_off, num_experts, n_tile);
        n_tile -= tab.tile_off[expert_id];

        int a_half0   = n_tile * 2;
        int a_half1   = a_half0 + 1;
        int b_half0   = m_tile * 2;
        int b_half1   = b_half0 + 1;
        int block_m   = m_tile * BLOCK_ROW;
        int block_row = n_tile;
        int block_col = m_tile;
        int sa_batch  = k_iters * tab.tile_off[expert_id] + n_tile;
        int sa_stride = tab.tile_off[expert_id + 1] - tab.tile_off[expert_id];
        int sb_batch  = expert_id * k_iters * total_m_tiles + m_tile;
        int sb_stride = total_m_tiles;

        gl_fp8_rt A_local(A);
        gl_fp8_rt B_local(B);
        A_local.raw_ptr = (T *)tab.a[expert_id];
        B_local.raw_ptr = (T *)tab.b[expert_id];

        const T *a_base = (const T *)&A_local[{0, 0, 0, 0}];
        const T *b_base = (const T *)&B_local[{0, 0, 0, 0}];
        const int a_row_stride = A_local.template stride<2>() * sizeof(T);
        const int b_row_stride = B_local.template stride<2>() * sizeof(T);
        const int a_rows = EDGE ? tab.rows[expert_id] : A_local.rows();
        kittens::i32x4 a_srd = kittens::make_srsrc(a_base, a_rows * a_row_stride, EDGE ? 0 : a_row_stride);
        kittens::i32x4 b_srd = kittens::make_srsrc(b_base, B_local.rows() * b_row_stride, b_row_stride);

        const int n_iters = k_iters - 2;
        const bool s0 = n_iters & 1;

#include "mxfp8_nn_preload.inc"

#define MXFP8_NN_OPAQUE_PEEL 1
#include "mxfp8_nn_mainloop.inc"
#undef MXFP8_NN_OPAQUE_PEEL

        if (warp_m == 0) {
            __builtin_amdgcn_s_barrier();
        }

        OutGL C_local(C);
        C_local.raw_ptr = (typename OutGL::dtype *)tab.c[expert_id];
        if constexpr (EDGE) {
            C_local.rows_internal = tab.rows[expert_id];
        }

        gemm_epilogue<GemmEpilogue::DEFAULT, false, false, EDGE, RT_C, RT_C>(cA, cB, cC, cD, C_local,
            GroupedNoAux{}, nullptr, 0, block_m, block_row, block_col, warp_m, warp_n);
    }
}

static int persistent_grid(const void *kernel, int num_tiles, int *ctr, hipStream_t stream) {
    static std::mutex mu;
    static std::map<std::pair<int, const void *>, int> resident;
    int dev = 0;
    hipGetDevice(&dev);
    int wgs = 0;
    {
        std::lock_guard<std::mutex> lock(mu);
        auto it = resident.find({dev, kernel});
        if (it == resident.end()) {
            int occ = 0, cus = 0;
            hipOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kernel, NUM_THREADS, 0);
            hipDeviceGetAttribute(&cus, hipDeviceAttributeMultiprocessorCount, dev);
            it = resident.emplace(std::make_pair(dev, kernel), std::max(1, occ) * std::max(1, cus)).first;
        }
        wgs = std::min(num_tiles, it->second);
    }
    if (num_tiles > SCHED_ROUNDS * wgs) {
        hipMemsetAsync(ctr, 0, SCHED_CTR_BYTES, stream);
    }
    return wgs;
}

template <typename OutGL>
static void grouped_mxfp8_gemm_chunk(const void *const *A_array, const void *const *B_array,
    void *const *C_array, const void *const *scale_A_array, const void *const *scale_B_array, const int *N_array,
    int num_experts, int M, int K, int max_N, bool nn, int a_fp8, int b_fp8, int *ctr, uint32_t *sa_pk,
    uint32_t *sb_pk, hipStream_t stream) {

    int tiles_M = M / BLOCK_COL;
    int k_iters = K / BLOCK_K;
    int scale_K = K / 32;

    GroupedTab tab{};
    PackTab    wtab{};
    PackTab    atab{};
    int n_tiles = 0;
    bool edge = false;
    for (int g = 0; g < num_experts; g++) {
        tab.a[g]        = nn ? B_array[g] : A_array[g];
        tab.b[g]        = nn ? A_array[g] : B_array[g];
        tab.c[g]        = C_array[g];
        tab.tile_off[g] = n_tiles;
        tab.rows[g]     = N_array[g];
        wtab.src[g]     = (const uint8_t *)scale_A_array[g];
        atab.src[g]     = (const uint8_t *)scale_B_array[g];
        atab.off[g]     = n_tiles;
        atab.tokens[g]  = N_array[g];
        n_tiles += kittens::ceil_div(N_array[g], BLOCK_COL);
        edge = edge || N_array[g] % BLOCK_COL != 0;
    }
    tab.tile_off[num_experts] = n_tiles;
    atab.off[num_experts]     = n_tiles;
    if (n_tiles == 0) {
        return;
    }

    dim3 wgrid(k_iters * tiles_M, num_experts);
    if (nn) {
        pack_scales_tab_kernel<true, 32, 8, ScalePack::UNIFORM><<<wgrid, 8 * 64, 0, stream>>>(
            wtab, sb_pk, num_experts, M, scale_K, k_iters, tiles_M, 2 * k_iters * M);
    } else {
        pack_scales_tab_kernel<false, 64, 4, ScalePack::UNIFORM><<<wgrid, 4 * 64, 0, stream>>>(
            wtab, sa_pk, num_experts, M, scale_K, k_iters, tiles_M, k_iters * M);
    }
    KITTENS_BOOL_SWITCH(edge, EDGE,
        if (nn) {
            pack_scales_tab_kernel<false, 64, 4, ScalePack::CONCAT, EDGE><<<k_iters * n_tiles, 4 * 64, 0, stream>>>(
                atab, sa_pk, num_experts, 0, scale_K, k_iters, 0, 0);
        } else {
            pack_scales_tab_kernel<false, 32, 8, ScalePack::CONCAT, EDGE><<<k_iters * n_tiles, 8 * 64, 0, stream>>>(
                atab, sb_pk, num_experts, 0, scale_K, k_iters, 0, 0);
        }
    ) // NOLINT(*)

    gl_fp8_rt gl_A((kittens::fp8e4m3 *)tab.a[0], nullptr, nullptr, nn ? (size_t)max_N : (size_t)M, (size_t)K);
    gl_fp8_rt gl_B((kittens::fp8e4m3 *)tab.b[0], nullptr, nullptr, nn ? (size_t)K : (size_t)max_N,
                   nn ? (size_t)M : (size_t)K);
    gl_scale_rt gl_SA(reinterpret_cast<kittens::fp8e8m0 *>(sa_pk),
                      nn ? k_iters * n_tiles : num_experts * k_iters * tiles_M, nullptr, nullptr, nullptr);
    gl_scale_rt gl_SB(reinterpret_cast<kittens::fp8e8m0 *>(sb_pk),
                      nn ? 2 * num_experts * k_iters * tiles_M : 2 * k_iters * n_tiles, nullptr, nullptr, nullptr);
    OutGL gl_C((typename OutGL::dtype *)C_array[0], nullptr, nullptr, (size_t)max_N, (size_t)M);

    const int a_slot_fp8 = nn ? b_fp8 : a_fp8;
    const int b_slot_fp8 = nn ? a_fp8 : b_fp8;
    const int num_tiles  = tiles_M * n_tiles;
    KITTENS_BOOL_SWITCH(edge, EDGE,
        KITTENS_BOOL_SWITCH(a_slot_fp8, CBSZ,
            KITTENS_BOOL_SWITCH(b_slot_fp8, BLGP,
                if (nn) {
                    const int wgs = persistent_grid((const void *)mxfp8_grouped_nn_kernel<CBSZ, BLGP, EDGE, OutGL>,
                                                    num_tiles, ctr, stream);
                    mxfp8_grouped_nn_kernel<CBSZ, BLGP, EDGE, OutGL><<<wgs, NUM_THREADS, 0, stream>>>(
                        gl_A, gl_B, gl_C, gl_SA, gl_SB, tab, num_experts, K, tiles_M, n_tiles, num_tiles, ctr);
                } else {
                    const int wgs = persistent_grid((const void *)mxfp8_grouped_tn_kernel<CBSZ, BLGP, EDGE, OutGL>,
                                                    num_tiles, ctr, stream);
                    mxfp8_grouped_tn_kernel<CBSZ, BLGP, EDGE, OutGL><<<wgs, NUM_THREADS, 0, stream>>>(
                        gl_A, gl_B, gl_C, gl_SA, gl_SB, tab, num_experts, K, tiles_M, n_tiles, num_tiles, ctr);
                }
    ))) // NOLINT(*)
}

static bool grouped_mxfp8_gemm(
    const void *const *A_array, const void *const *B_array, void *const *C_array,
    const void *const *scale_A_array, const void *const *scale_B_array,
    int M, const int *N_array, int K, int num_experts,
    bool transa, bool transb, int a_dtype, int b_dtype, int out_dtype,
    void *workspace, size_t workspace_size, hipStream_t stream) {

    if (transa && transb) { warn_fallback("HK-grouped", "TT layout not supported"); return false; }
    if (!transa && transb) { warn_fallback("HK-grouped", "NT layout: use grouped_mxfp8_wgrad"); return false; }
    if (M % BLOCK_ROW != 0) { warn_fallback("HK-grouped", "M not 256-aligned"); return false; }
    if (K % BLOCK_K != 0 || K < 256) { warn_fallback("HK-grouped", "K not 128-aligned or < 256"); return false; }
    if (num_experts <= 0) { warn_fallback("HK-grouped", "num_experts <= 0"); return false; }
    if (out_dtype != KITTENS_BFLOAT16 && out_dtype != KITTENS_FLOAT16 && out_dtype != KITTENS_FLOAT32) {
        return false;
    }

    int total_tiles = 0;
    int max_N       = 0;
    bool edge = false;
    for (int g = 0; g < num_experts; g++) {
        if (N_array[g] % 32 != 0) { warn_fallback("HK-grouped", "N_array not 32-aligned"); return false; }
        total_tiles += kittens::ceil_div(N_array[g], BLOCK_COL);
        max_N = std::max(max_N, N_array[g]);
        edge = edge || N_array[g] % BLOCK_COL != 0;
    }
    if (total_tiles == 0) {
        return true;
    }
    // Unaligned NN drops the activation rows past an expert by the range check so needs 31-bit offset
    if (edge && !transa && kittens_align_up(max_N, BLOCK_COL) * K >= 0x80000000ull) {
        warn_fallback("HK-grouped", "expert too large for unaligned NN");
        return false;
    }

    int k_iters = K / BLOCK_K;
    size_t act_words    = (size_t)k_iters * total_tiles * BLOCK_COL;
    size_t weight_words = (size_t)k_iters * std::min(num_experts, MAX_TAB_EXPERTS) * M;
    size_t sa_pk_bytes  = kittens_align_up((transa ? weight_words : act_words) * sizeof(uint32_t), 256);
    size_t sb_pk_bytes  = kittens_align_up(2 * (transa ? act_words : weight_words) * sizeof(uint32_t), 256);
    if (workspace_size < SCHED_CTR_BYTES + sa_pk_bytes + sb_pk_bytes) {
        warn_fallback("HK-grouped", "workspace too small"); return false;
    }

    auto *ctr   = (int *)workspace;
    auto *sa_pk = (uint32_t *)((uint8_t *)workspace + SCHED_CTR_BYTES);
    auto *sb_pk = (uint32_t *)((uint8_t *)workspace + SCHED_CTR_BYTES + sa_pk_bytes);
    int a_fp8 = fp8_code(static_cast<KittensDType>(a_dtype));
    int b_fp8 = fp8_code(static_cast<KittensDType>(b_dtype));

    for (int e0 = 0; e0 < num_experts; e0 += MAX_TAB_EXPERTS) {
        int ne = std::min(MAX_TAB_EXPERTS, num_experts - e0);
        if (out_dtype == KITTENS_BFLOAT16) {
            grouped_mxfp8_gemm_chunk<gl_bf16_rt>(A_array + e0, B_array + e0, C_array + e0, scale_A_array + e0,
                scale_B_array + e0, N_array + e0, ne, M, K, max_N, !transa, a_fp8, b_fp8, ctr, sa_pk, sb_pk, stream);
        } else if (out_dtype == KITTENS_FLOAT16) {
            grouped_mxfp8_gemm_chunk<gl_fp16_rt>(A_array + e0, B_array + e0, C_array + e0, scale_A_array + e0,
                scale_B_array + e0, N_array + e0, ne, M, K, max_N, !transa, a_fp8, b_fp8, ctr, sa_pk, sb_pk, stream);
        } else {
            grouped_mxfp8_gemm_chunk<gl_f32_rt>(A_array + e0, B_array + e0, C_array + e0, scale_A_array + e0,
                scale_B_array + e0, N_array + e0, ne, M, K, max_N, !transa, a_fp8, b_fp8, ctr, sa_pk, sb_pk, stream);
        }
    }
    return true;
}

template <int CBSZ, int BLGP, bool ACCUMULATE, bool EDGE, typename OutGL>
__global__ __launch_bounds__(NUM_THREADS, 2)
void mxfp8_wgrad_nt_kernel(
    const gl_fp8_rt A, const gl_fp8_rt B, const OutGL C,
    const gl_scale_rt scale_A_gl, const gl_scale_rt scale_B_gl,
    const WgradTab tab, int tiles_M, int tiles_N, int tiles_per_expert,
    int num_tiles, int *__restrict__ ctr) {

    using ST_A     = kittens::st_fp8e4m3<BLOCK_K, HALF_ROW, kittens::st_16x128_s>;
    using ST_B     = kittens::st_fp8e4m3<BLOCK_K, HALF_COL, kittens::st_16x128_s>;
    using RT_A     = kittens::rt<kittens::fp8e4m3, REG_M, BLOCK_K, kittens::col_l, kittens::rt_16x128_s>;
    using RT_B     = kittens::rt<kittens::fp8e4m3, REG_N, BLOCK_K, kittens::col_l, kittens::rt_16x128_s>;
    using RT_C     = kittens::rt_fl<REG_M, REG_N, kittens::col_l, kittens::rt_16x16_s>;
    using RT_C_T   = kittens::rt_fl<REG_N, REG_M, kittens::row_l, kittens::rt_16x16_s>;

    constexpr bool DMA = ACCUMULATE || EDGE;

    __shared__ alignas(ST_A::underlying_subtile_bytes) ST_A As[2][2];
    __shared__ alignas(ST_B::underlying_subtile_bytes) ST_B Bs[2][2];
    __shared__ ST_Scale scale_smem[2][3];
    __shared__ int s_wgid;

    int warp_m = __builtin_amdgcn_readfirstlane(kittens::warpid() / WARPS_COL);
    int warp_n = __builtin_amdgcn_readfirstlane(kittens::warpid() % WARPS_COL);

    using T = kittens::fp8e4m3;

    constexpr int bpt = ST_A::underlying_subtile_bytes_per_thread;
    constexpr int bpm = bpt * NUM_THREADS;
    constexpr int copies_A = BLOCK_K * HALF_ROW * sizeof(T) / bpm;
    constexpr int copies_B = BLOCK_K * HALF_COL * sizeof(T) / bpm;
    uint32_t sw_A[copies_A], sw_B[copies_B];
    if constexpr (!DMA) {
        G::prefill_swizzled_offsets(As[0][0], A, sw_A);
        G::prefill_swizzled_offsets(Bs[0][0], B, sw_B);
    }

    uint32_t a_frag[RT_A::height];
    col_frag_addrs<RT_A>(a_frag, As[0][0], warp_m * REG_M);
    uint32_t b_frag[RT_B::height];
    col_frag_addrs<RT_B>(b_frag, Bs[0][0], warp_n * REG_N);

    const bool static_sched = num_tiles <= SCHED_ROUNDS * (int)gridDim.x;
    int iter = 0;
    int wgid = 0;
    int zero;
    asm volatile("s_mov_b32 %0, 0" : "=s"(zero));
    if (zero != 0) {
        asm volatile("" : "+s"(wgid));
        goto tile;
    }
    asm volatile("" : "+s"(iter));
#pragma unroll 1
    for (;; iter++) {
        if (threadIdx.x == 0) {
            const int i = static_sched ? (int)blockIdx.x + iter * (int)gridDim.x : atomicAdd(&ctr[SCHED_CTR_GLOBAL], 1);
            s_wgid = i < num_tiles ? kittens::chiplet_transform_chunked(i, num_tiles, NUM_XCDS_AFF, SCHED_CHUNK)
                                   : num_tiles;
        }
        __syncthreads();
        wgid = __builtin_amdgcn_readfirstlane(s_wgid);
        if (wgid >= num_tiles) {
            break;
        }
    tile:
        RT_A a;
        RT_B b0, b1;
        RT_C cA, cB, cC, cD;
        kittens::zero(cA); kittens::zero(cB); kittens::zero(cC); kittens::zero(cD);

        const int WGM = 8;
        int expert_id  = wgid / tiles_per_expert;
        int local_tile = wgid % tiles_per_expert;

        int num_wgid_in_group = WGM * tiles_N;
        int group_id          = local_tile / num_wgid_in_group;
        int first_pid_m       = group_id * WGM;
        int group_size_m      = min(tiles_M - first_pid_m, WGM);
        int m_tile            = first_pid_m + ((local_tile % num_wgid_in_group) % group_size_m);
        int n_tile            = (local_tile % num_wgid_in_group) / group_size_m;

        WgradExpertInfo info = tab.info[expert_id];
        int k_iters = __builtin_amdgcn_readfirstlane(info.k_iters);
        int sa_base = info.sa_tile_offset;
        int sb_base = info.sb_tile_offset;

        int a_row_tile = m_tile;
        int block_m    = m_tile * BLOCK_ROW;
        int block_row  = m_tile;
        int block_col  = n_tile;

        if constexpr (!ACCUMULATE) {
            if (k_iters == 0) {
                using U = typename OutGL::dtype;
                constexpr int VEC_PER_ROW = BLOCK_COL * sizeof(U) / sizeof(uint4);
                const size_t ld = C.cols();
                U *tile = (U *)info.d_ptr + (size_t)block_col * BLOCK_ROW * ld + (size_t)block_row * BLOCK_COL;
                for (int i = threadIdx.x; i < BLOCK_ROW * VEC_PER_ROW; i += NUM_THREADS) {
                    uint4 *row = reinterpret_cast<uint4 *>(tile + (size_t)(i / VEC_PER_ROW) * ld);
                    row[i % VEC_PER_ROW] = uint4{0, 0, 0, 0};
                }
                __syncthreads();
                continue;
            }
        }

        const int shift = EDGE ? __builtin_amdgcn_readfirstlane(info.shift) : 0;

        gl_fp8_rt A_local(A);
        A_local.raw_ptr = (T *)info.a_ptr - (size_t)shift * A_local.template stride<2>();

        gl_fp8_rt B_local(B);
        B_local.raw_ptr = (T *)info.b_ptr - (size_t)shift * B_local.template stride<2>();

        [[maybe_unused]] TileLoad a_dma, b_dma, a_k0, b_k0;
        [[maybe_unused]] ScaleLoad s_dma;
        if constexpr (DMA) {
            const uint32_t a_row_stride = A_local.template stride<2>() * sizeof(T);
            const uint32_t b_row_stride = B_local.template stride<2>() * sizeof(T);
            const uint32_t a_range      = (k_iters * BLOCK_K - 1) * a_row_stride + HALF_ROW;
            const uint32_t b_range      = (k_iters * BLOCK_K - 1) * b_row_stride + HALF_COL;
            a_dma = prefill_tile_load<NUM_WARPS, EDGE>(&A_local[{0, 0, 0, block_m}], HALF_ROW * sizeof(T),
                                                       BLOCK_K * a_row_stride, a_range, a_row_stride, As,
                                                       EDGE ? shift - BLOCK_K : 0);
            b_dma = prefill_tile_load<NUM_WARPS, EDGE>(&B_local[{0, 0, 0, n_tile * BLOCK_COL}], HALF_COL * sizeof(T),
                                                       BLOCK_K * b_row_stride, b_range, b_row_stride, Bs,
                                                       EDGE ? shift - BLOCK_K : 0);
            if constexpr (EDGE) {
                a_k0 = prefill_tile_load<NUM_WARPS, true>(&A_local[{0, 0, 0, block_m}], HALF_ROW * sizeof(T),
                                                          BLOCK_K * a_row_stride, a_range, a_row_stride, As, shift);
                b_k0 = prefill_tile_load<NUM_WARPS, true>(&B_local[{0, 0, 0, n_tile * BLOCK_COL}],
                                                          HALF_COL * sizeof(T), BLOCK_K * b_row_stride, b_range,
                                                          b_row_stride, Bs, shift);
            } else {
                a_k0 = a_dma;
                b_k0 = b_dma;
            }
            s_dma = prefill_scale_load(scale_A_gl, scale_B_gl, sa_base + m_tile, 2 * (sb_base + n_tile), tiles_M,
                                       2 * tiles_N, scale_smem);
        }

        const int n_iters = k_iters - 2;
        const bool s0 = n_iters & 1;

        if constexpr (DMA) {
            tile_load_m0(b_k0, s0, 0, 0);
            tile_load_m0(a_k0, s0, 0, 0);
            tile_load_m0(b_k0, s0, 1, 0);
            tile_load_m0(a_k0, s0, 1, 0);
        } else {
            G::load(Bs[s0][0], B_local, {0, 0, 0, n_tile * 2    }, sw_B);
            G::load(As[s0][0], A_local, {0, 0, 0, a_row_tile * 2    }, sw_A);
            G::load(Bs[s0][1], B_local, {0, 0, 0, n_tile * 2 + 1}, sw_B);
            G::load(As[s0][1], A_local, {0, 0, 0, a_row_tile * 2 + 1}, sw_A);
        }

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();

        if constexpr (DMA) {
            tile_load_m0(a_dma, !s0, 0, 1);
            tile_load_m0(b_dma, !s0, 0, 1);
            tile_load_m0(b_dma, !s0, 1, 1);
        } else {
            G::load(As[!s0][0], A_local, {0, 0, 1, a_row_tile * 2    }, sw_A);
            G::load(Bs[!s0][0], B_local, {0, 0, 1, n_tile * 2    }, sw_B);
            G::load(Bs[!s0][1], B_local, {0, 0, 1, n_tile * 2 + 1}, sw_B);
        }
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();

        if constexpr (DMA) {
            scale_load_m0(s_dma, s0, 0);
        } else {
            G::load(scale_smem[s0][0], scale_A_gl, {sa_base + m_tile, 0, 0, 0});
            G::load(scale_smem[s0][1], scale_B_gl, {2 * (sb_base + n_tile),     0, 0, 0});
            G::load(scale_smem[s0][2], scale_B_gl, {2 * (sb_base + n_tile) + 1, 0, 0, 0});
        }
        asm volatile("s_waitcnt vmcnt(0)");
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();

        if (warp_m == 1) {
            __builtin_amdgcn_s_barrier();
        }
        if constexpr (DMA) {
            asm volatile("s_mov_b32 m0, %0" :: "s"(a_dma.lds[!s0][1]) : "memory", "m0");
        }

        int k = 0;
        if (s0) {
            constexpr int TIC = 1;
            constexpr int TOC = TIC ^ 1;
            if constexpr (!DMA) {
                G::load(scale_smem[TOC][0], scale_A_gl, {sa_base + (k + 1) * tiles_M + m_tile, 0, 0, 0});
                G::load(scale_smem[TOC][1], scale_B_gl, {2 * (sb_base + (k + 1) * tiles_N + n_tile),     0, 0, 0});
                G::load(scale_smem[TOC][2], scale_B_gl, {2 * (sb_base + (k + 1) * tiles_N + n_tile) + 1, 0, 0, 0});
            }

            load_cols<ST_B, 2 * TIC>(b0, b_frag);
            load_cols<ST_A, 2 * TIC>(a, a_frag);
            if constexpr (DMA) {
                tile_load(a_dma, 1, k + 1);
            } else {
                G::load(As[TOC][1], A_local, {0, 0, k + 1, a_row_tile * 2 + 1}, sw_A);
            }
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_barrier();

            if constexpr (DMA) {
                asm volatile("s_mov_b32 m0, %0" :: "s"(s_dma.lds[TOC]) : "memory", "m0");
            }
            kittens::fp8e8m0_4 sa_h0 = lane_rd(scale_smem[TIC][0], warp_m);
            kittens::fp8e8m0_4 sb_h0 = lane_rd(scale_smem[TIC][1], warp_n);
            if constexpr (DMA) {
                scale_load(s_dma, k + 1);
            }
            __builtin_amdgcn_s_setprio(2);
            kittens::mma_ABt_scaled<CBSZ, BLGP>(cA, a, b0, cA, &sa_h0, &sb_h0);
            if constexpr (DMA) {
                __builtin_amdgcn_sched_barrier(0);
                asm volatile("s_mov_b32 m0, %0" :: "s"(a_dma.lds[TIC][0]) : "memory", "m0");
            }
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            kittens::fp8e8m0_4 sb_h1 = lane_rd(scale_smem[TIC][2], warp_n);
            load_cols<ST_B, 2 * TIC + 1>(b1, b_frag);
            if constexpr (DMA) {
                tile_load(a_dma, 0, k + 2);
            } else {
                G::load(As[TIC][0], A_local, {0, 0, k + 2, a_row_tile * 2    }, sw_A);
            }
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_barrier();

            __builtin_amdgcn_s_setprio(2);
            kittens::mma_ABt_scaled<CBSZ, BLGP>(cB, a, b1, cB, &sa_h0, &sb_h1);
            if constexpr (DMA) {
                __builtin_amdgcn_sched_barrier(0);
                asm volatile("s_mov_b32 m0, %0" :: "s"(b_dma.lds[TIC][0]) : "memory", "m0");
            }
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            kittens::fp8e8m0_4 sa_h1 = lane_rd(scale_smem[TIC][0], 2 + warp_m);
            load_cols<ST_A, 2 * TIC + 1>(a, a_frag);
            if constexpr (DMA) {
                tile_load(b_dma, 0, k + 2);
            } else {
                G::load(Bs[TIC][0], B_local, {0, 0, k + 2, n_tile * 2    }, sw_B);
            }
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_barrier();

            __builtin_amdgcn_s_setprio(2);
            kittens::mma_ABt_scaled<CBSZ, BLGP>(cC, a, b0, cC, &sa_h1, &sb_h0);
            if constexpr (DMA) {
                __builtin_amdgcn_sched_barrier(0);
                asm volatile("s_mov_b32 m0, %0" :: "s"(b_dma.lds[TIC][1]) : "memory", "m0");
            }
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            if constexpr (DMA) {
                tile_load(b_dma, 1, k + 2);
            } else {
                G::load(Bs[TIC][1], B_local, {0, 0, k + 2, n_tile * 2 + 1}, sw_B);
            }
            asm volatile("s_waitcnt vmcnt(6)"); // wait for toc data; next-iter prefetches in flight
            __builtin_amdgcn_s_barrier();

            __builtin_amdgcn_s_setprio(2);
            kittens::mma_ABt_scaled<CBSZ, BLGP>(cD, a, b1, cD, &sa_h1, &sb_h1);
            if constexpr (DMA) {
                __builtin_amdgcn_sched_barrier(0);
                asm volatile("s_mov_b32 m0, %0" :: "s"(a_dma.lds[TIC][1]) : "memory", "m0");
            }
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            k++;
        }
        while (k < n_iters) {
            {
                constexpr int TIC = 0;
                constexpr int TOC = TIC ^ 1;
                if constexpr (!DMA) {
                    G::load(scale_smem[TOC][0], scale_A_gl, {sa_base + (k + 1) * tiles_M + m_tile, 0, 0, 0});
                    G::load(scale_smem[TOC][1], scale_B_gl, {2 * (sb_base + (k + 1) * tiles_N + n_tile),     0, 0, 0});
                    G::load(scale_smem[TOC][2], scale_B_gl, {2 * (sb_base + (k + 1) * tiles_N + n_tile) + 1, 0, 0, 0});
                }

                load_cols<ST_B, 2 * TIC>(b0, b_frag);
                load_cols<ST_A, 2 * TIC>(a, a_frag);
                if constexpr (DMA) {
                    tile_load(a_dma, 1, k + 1);
                } else {
                    G::load(As[TOC][1], A_local, {0, 0, k + 1, a_row_tile * 2 + 1}, sw_A);
                }
                asm volatile("s_waitcnt lgkmcnt(0)");
                __builtin_amdgcn_s_barrier();

                if constexpr (DMA) {
                    asm volatile("s_mov_b32 m0, %0" :: "s"(s_dma.lds[TOC]) : "memory", "m0");
                }
                kittens::fp8e8m0_4 sa_h0 = lane_rd(scale_smem[TIC][0], warp_m);
                kittens::fp8e8m0_4 sb_h0 = lane_rd(scale_smem[TIC][1], warp_n);
                if constexpr (DMA) {
                    scale_load(s_dma, k + 1);
                }
                __builtin_amdgcn_s_setprio(2);
                kittens::mma_ABt_scaled<CBSZ, BLGP>(cA, a, b0, cA, &sa_h0, &sb_h0);
                if constexpr (DMA) {
                    __builtin_amdgcn_sched_barrier(0);
                    asm volatile("s_mov_b32 m0, %0" :: "s"(a_dma.lds[TIC][0]) : "memory", "m0");
                }
                __builtin_amdgcn_s_setprio(0);
                __builtin_amdgcn_s_barrier();
                __builtin_amdgcn_sched_barrier(0);

                kittens::fp8e8m0_4 sb_h1 = lane_rd(scale_smem[TIC][2], warp_n);
                load_cols<ST_B, 2 * TIC + 1>(b1, b_frag);
                if constexpr (DMA) {
                    tile_load(a_dma, 0, k + 2);
                } else {
                    G::load(As[TIC][0], A_local, {0, 0, k + 2, a_row_tile * 2    }, sw_A);
                }
                asm volatile("s_waitcnt lgkmcnt(0)");
                __builtin_amdgcn_s_barrier();

                __builtin_amdgcn_s_setprio(2);
                kittens::mma_ABt_scaled<CBSZ, BLGP>(cB, a, b1, cB, &sa_h0, &sb_h1);
                if constexpr (DMA) {
                    __builtin_amdgcn_sched_barrier(0);
                    asm volatile("s_mov_b32 m0, %0" :: "s"(b_dma.lds[TIC][0]) : "memory", "m0");
                }
                __builtin_amdgcn_s_setprio(0);
                __builtin_amdgcn_s_barrier();

                kittens::fp8e8m0_4 sa_h1 = lane_rd(scale_smem[TIC][0], 2 + warp_m);
                load_cols<ST_A, 2 * TIC + 1>(a, a_frag);
                if constexpr (DMA) {
                    tile_load(b_dma, 0, k + 2);
                } else {
                    G::load(Bs[TIC][0], B_local, {0, 0, k + 2, n_tile * 2    }, sw_B);
                }
                asm volatile("s_waitcnt lgkmcnt(0)");
                __builtin_amdgcn_s_barrier();

                __builtin_amdgcn_s_setprio(2);
                kittens::mma_ABt_scaled<CBSZ, BLGP>(cC, a, b0, cC, &sa_h1, &sb_h0);
                if constexpr (DMA) {
                    __builtin_amdgcn_sched_barrier(0);
                    asm volatile("s_mov_b32 m0, %0" :: "s"(b_dma.lds[TIC][1]) : "memory", "m0");
                }
                __builtin_amdgcn_s_setprio(0);
                __builtin_amdgcn_s_barrier();
                __builtin_amdgcn_sched_barrier(0);

                if constexpr (DMA) {
                    tile_load(b_dma, 1, k + 2);
                } else {
                    G::load(Bs[TIC][1], B_local, {0, 0, k + 2, n_tile * 2 + 1}, sw_B);
                }
                asm volatile("s_waitcnt vmcnt(6)");
                __builtin_amdgcn_s_barrier();

                __builtin_amdgcn_s_setprio(2);
                kittens::mma_ABt_scaled<CBSZ, BLGP>(cD, a, b1, cD, &sa_h1, &sb_h1);
                if constexpr (DMA) {
                    __builtin_amdgcn_sched_barrier(0);
                    asm volatile("s_mov_b32 m0, %0" :: "s"(a_dma.lds[TIC][1]) : "memory", "m0");
                }
                __builtin_amdgcn_s_setprio(0);
                __builtin_amdgcn_s_barrier();
            }
            k++;
            {
                constexpr int TIC = 1;
                constexpr int TOC = TIC ^ 1;
                if constexpr (!DMA) {
                    G::load(scale_smem[TOC][0], scale_A_gl, {sa_base + (k + 1) * tiles_M + m_tile, 0, 0, 0});
                    G::load(scale_smem[TOC][1], scale_B_gl, {2 * (sb_base + (k + 1) * tiles_N + n_tile),     0, 0, 0});
                    G::load(scale_smem[TOC][2], scale_B_gl, {2 * (sb_base + (k + 1) * tiles_N + n_tile) + 1, 0, 0, 0});
                }

                load_cols<ST_B, 2 * TIC>(b0, b_frag);
                load_cols<ST_A, 2 * TIC>(a, a_frag);
                if constexpr (DMA) {
                    tile_load(a_dma, 1, k + 1);
                } else {
                    G::load(As[TOC][1], A_local, {0, 0, k + 1, a_row_tile * 2 + 1}, sw_A);
                }
                asm volatile("s_waitcnt lgkmcnt(0)");
                __builtin_amdgcn_s_barrier();

                if constexpr (DMA) {
                    asm volatile("s_mov_b32 m0, %0" :: "s"(s_dma.lds[TOC]) : "memory", "m0");
                }
                kittens::fp8e8m0_4 sa_h0 = lane_rd(scale_smem[TIC][0], warp_m);
                kittens::fp8e8m0_4 sb_h0 = lane_rd(scale_smem[TIC][1], warp_n);
                if constexpr (DMA) {
                    scale_load(s_dma, k + 1);
                }
                __builtin_amdgcn_s_setprio(2);
                kittens::mma_ABt_scaled<CBSZ, BLGP>(cA, a, b0, cA, &sa_h0, &sb_h0);
                if constexpr (DMA) {
                    __builtin_amdgcn_sched_barrier(0);
                    asm volatile("s_mov_b32 m0, %0" :: "s"(a_dma.lds[TIC][0]) : "memory", "m0");
                }
                __builtin_amdgcn_s_setprio(0);
                __builtin_amdgcn_s_barrier();
                __builtin_amdgcn_sched_barrier(0);

                kittens::fp8e8m0_4 sb_h1 = lane_rd(scale_smem[TIC][2], warp_n);
                load_cols<ST_B, 2 * TIC + 1>(b1, b_frag);
                if constexpr (DMA) {
                    tile_load(a_dma, 0, k + 2);
                } else {
                    G::load(As[TIC][0], A_local, {0, 0, k + 2, a_row_tile * 2    }, sw_A);
                }
                asm volatile("s_waitcnt lgkmcnt(0)");
                __builtin_amdgcn_s_barrier();

                __builtin_amdgcn_s_setprio(2);
                kittens::mma_ABt_scaled<CBSZ, BLGP>(cB, a, b1, cB, &sa_h0, &sb_h1);
                if constexpr (DMA) {
                    __builtin_amdgcn_sched_barrier(0);
                    asm volatile("s_mov_b32 m0, %0" :: "s"(b_dma.lds[TIC][0]) : "memory", "m0");
                }
                __builtin_amdgcn_s_setprio(0);
                __builtin_amdgcn_s_barrier();

                kittens::fp8e8m0_4 sa_h1 = lane_rd(scale_smem[TIC][0], 2 + warp_m);
                load_cols<ST_A, 2 * TIC + 1>(a, a_frag);
                if constexpr (DMA) {
                    tile_load(b_dma, 0, k + 2);
                } else {
                    G::load(Bs[TIC][0], B_local, {0, 0, k + 2, n_tile * 2    }, sw_B);
                }
                asm volatile("s_waitcnt lgkmcnt(0)");
                __builtin_amdgcn_s_barrier();

                __builtin_amdgcn_s_setprio(2);
                kittens::mma_ABt_scaled<CBSZ, BLGP>(cC, a, b0, cC, &sa_h1, &sb_h0);
                if constexpr (DMA) {
                    __builtin_amdgcn_sched_barrier(0);
                    asm volatile("s_mov_b32 m0, %0" :: "s"(b_dma.lds[TIC][1]) : "memory", "m0");
                }
                __builtin_amdgcn_s_setprio(0);
                __builtin_amdgcn_s_barrier();
                __builtin_amdgcn_sched_barrier(0);

                if constexpr (DMA) {
                    tile_load(b_dma, 1, k + 2);
                } else {
                    G::load(Bs[TIC][1], B_local, {0, 0, k + 2, n_tile * 2 + 1}, sw_B);
                }
                asm volatile("s_waitcnt vmcnt(6)");
                __builtin_amdgcn_s_barrier();

                __builtin_amdgcn_s_setprio(2);
                kittens::mma_ABt_scaled<CBSZ, BLGP>(cD, a, b1, cD, &sa_h1, &sb_h1);
                if constexpr (DMA) {
                    __builtin_amdgcn_sched_barrier(0);
                    asm volatile("s_mov_b32 m0, %0" :: "s"(a_dma.lds[TIC][1]) : "memory", "m0");
                }
                __builtin_amdgcn_s_setprio(0);
                __builtin_amdgcn_s_barrier();
            }
            k++;
        }

        // Opaque single-trip loops make the peeled stages' MFMAs accumulate in place; without them the
        // persistent kernel copies the accumulators and spills.
        int once = 1;
        asm volatile("" : "+s"(once));
        for (int o = once; o > 0; o--) { // Epilogue k = k_iters - 2
            constexpr int TIC = 0;
            constexpr int TOC = TIC ^ 1;
            int k = k_iters - 2;
            if constexpr (DMA) {
                scale_load_m0(s_dma, TOC, k + 1);
            } else {
                G::load(scale_smem[TOC][0], scale_A_gl, {sa_base + (k + 1) * tiles_M + m_tile, 0, 0, 0});
                G::load(scale_smem[TOC][1], scale_B_gl, {2 * (sb_base + (k + 1) * tiles_N + n_tile),     0, 0, 0});
                G::load(scale_smem[TOC][2], scale_B_gl, {2 * (sb_base + (k + 1) * tiles_N + n_tile) + 1, 0, 0, 0});
            }
            asm volatile("s_waitcnt vmcnt(0)");
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_barrier();
            kittens::fp8e8m0_4 sa_h0 = lane_rd(scale_smem[TIC][0], warp_m);
            kittens::fp8e8m0_4 sa_h1 = lane_rd(scale_smem[TIC][0], 2 + warp_m);
            kittens::fp8e8m0_4 sb_h0 = lane_rd(scale_smem[TIC][1], warp_n);
            kittens::fp8e8m0_4 sb_h1 = lane_rd(scale_smem[TIC][2], warp_n);

            load_cols<ST_B, 2 * TIC>(b0, b_frag);
            load_cols<ST_A, 2 * TIC>(a, a_frag);
            if constexpr (DMA) {
                tile_load_m0(a_dma, TOC, 1, k + 1);
            } else {
                G::load(As[TOC][1], A_local, {0, 0, k + 1, a_row_tile * 2 + 1}, sw_A);
            }
            __builtin_amdgcn_s_barrier();

            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_setprio(2);
            kittens::mma_ABt_scaled<CBSZ, BLGP>(cA, a, b0, cA, &sa_h0, &sb_h0);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            load_cols<ST_B, 2 * TIC + 1>(b1, b_frag);
            __builtin_amdgcn_s_barrier();

            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_setprio(2);
            kittens::mma_ABt_scaled<CBSZ, BLGP>(cB, a, b1, cB, &sa_h0, &sb_h1);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_cols<ST_A, 2 * TIC + 1>(a, a_frag);
            __builtin_amdgcn_s_barrier();

            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_setprio(2);
            kittens::mma_ABt_scaled<CBSZ, BLGP>(cC, a, b0, cC, &sa_h1, &sb_h0);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_cols<ST_B, 2 * TOC>(b0, b_frag);
            asm volatile("s_waitcnt vmcnt(4)"); // wait for toc data; As[toc][1] still in flight
            __builtin_amdgcn_s_barrier();

            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_setprio(2);
            kittens::mma_ABt_scaled<CBSZ, BLGP>(cD, a, b1, cD, &sa_h1, &sb_h1);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);
        }

        for (int o = once; o > 0; o--) { // Final epilogue k = k_iters - 1
            constexpr int TIC = 1;
            asm volatile("s_waitcnt vmcnt(0)");
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_barrier();
            kittens::fp8e8m0_4 sa_h0 = lane_rd(scale_smem[TIC][0], warp_m);
            kittens::fp8e8m0_4 sa_h1 = lane_rd(scale_smem[TIC][0], 2 + warp_m);
            kittens::fp8e8m0_4 sb_h0 = lane_rd(scale_smem[TIC][1], warp_n);
            kittens::fp8e8m0_4 sb_h1 = lane_rd(scale_smem[TIC][2], warp_n);

            load_cols<ST_A, 2 * TIC>(a, a_frag);
            __builtin_amdgcn_s_barrier();

            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_setprio(2);
            kittens::mma_ABt_scaled<CBSZ, BLGP>(cA, a, b0, cA, &sa_h0, &sb_h0);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_cols<ST_B, 2 * TIC + 1>(b1, b_frag);
            asm volatile("s_waitcnt vmcnt(0)");
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_setprio(2);
            kittens::mma_ABt_scaled<CBSZ, BLGP>(cB, a, b1, cB, &sa_h0, &sb_h1);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_cols<ST_A, 2 * TIC + 1>(a, a_frag);
            __builtin_amdgcn_s_barrier();

            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_setprio(2);
            kittens::mma_ABt_scaled<CBSZ, BLGP>(cC, a, b0, cC, &sa_h1, &sb_h0);
            kittens::mma_ABt_scaled<CBSZ, BLGP>(cD, a, b1, cD, &sa_h1, &sb_h1);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }

        if (warp_m == 0) {
            __builtin_amdgcn_s_barrier();
        }

        OutGL C_local(C);
        C_local.raw_ptr = (typename OutGL::dtype *)info.d_ptr;

        gemm_epilogue<GemmEpilogue::DEFAULT, ACCUMULATE, true, false, RT_C, RT_C_T>(cA, cB, cC, cD, C_local,
            GroupedNoAux{}, nullptr, 0, block_m, block_row, block_col, warp_m, warp_n);
    }
}

static int wgrad_k_iters(int M_g) {
    return M_g == 0 ? 0 : std::max(kittens::ceil_div(M_g, BLOCK_K), 2);
}

template <typename OutGL>
static void grouped_mxfp8_wgrad_chunk(const void *const *A_array, const void *const *B_array,
    void *const *D_array, const void *const *scale_A_array, const void *const *scale_B_array, const int *M_array,
    const int *order, int num_experts, int N, int K, int max_M, bool accumulate, int a_fp8, int b_fp8, int *ctr,
    uint32_t *sa_pk, uint32_t *sb_pk, hipStream_t stream) {

    int tiles_M = N / BLOCK_ROW;
    int tiles_N = K / BLOCK_COL;

    WgradTab tab{};
    PackTab  atab{};
    PackTab  btab{};
    int num_active  = 0;
    int max_k_iters = 0;
    int sa_tiles    = 0;
    int sb_tiles    = 0;
    bool edge = false;
    for (int i = 0; i < num_experts; i++) {
        int g       = order[i];
        int k_iters = wgrad_k_iters(M_array[g]);
        int shift   = k_iters * BLOCK_K - M_array[g];
        tab.info[i] = {A_array[g], B_array[g], D_array[g], k_iters, sa_tiles, sb_tiles, shift};
        if (k_iters == 0) {
            continue;
        }
        edge = edge || k_iters * BLOCK_K != M_array[g];
        atab.src[num_active]     = (const uint8_t *)scale_A_array[g];
        atab.off[num_active]     = sa_tiles * 256;
        atab.k_iters[num_active] = k_iters;
        atab.tokens[num_active]  = M_array[g];
        btab.src[num_active]     = (const uint8_t *)scale_B_array[g];
        btab.off[num_active]     = sb_tiles * 512;
        btab.k_iters[num_active] = k_iters;
        btab.tokens[num_active]  = M_array[g];
        num_active++;
        max_k_iters = std::max(max_k_iters, k_iters);
        sa_tiles   += k_iters * tiles_M;
        sb_tiles   += k_iters * tiles_N;
    }

    if (num_active > 0) {
        KITTENS_BOOL_SWITCH(edge, EDGE,
            pack_scales_tab_kernel<true, 64, 4, ScalePack::VARYING, EDGE>
                <<<dim3(max_k_iters * tiles_M, num_active), 4 * 64, 0, stream>>>(
                    atab, sa_pk, num_active, N, 0, 0, tiles_M, 0);
            pack_scales_tab_kernel<true, 32, 8, ScalePack::VARYING, EDGE>
                <<<dim3(max_k_iters * tiles_N, num_active), 8 * 64, 0, stream>>>(
                    btab, sb_pk, num_active, K, 0, 0, tiles_N, 0);
        ) // NOLINT(*)
    }

    gl_scale_rt gl_SA(reinterpret_cast<kittens::fp8e8m0 *>(sa_pk), sa_tiles, nullptr, nullptr, nullptr);
    gl_scale_rt gl_SB(reinterpret_cast<kittens::fp8e8m0 *>(sb_pk), 2 * sb_tiles, nullptr, nullptr, nullptr);
    gl_fp8_rt gl_A((kittens::fp8e4m3 *)A_array[order[0]], nullptr, nullptr, (size_t)max_M, (size_t)N);
    gl_fp8_rt gl_B((kittens::fp8e4m3 *)B_array[order[0]], nullptr, nullptr, (size_t)max_M, (size_t)K);
    OutGL gl_D((typename OutGL::dtype *)D_array[order[0]], nullptr, nullptr, (size_t)K, (size_t)N);

    const int num_tiles = num_experts * tiles_M * tiles_N;
    KITTENS_BOOL_SWITCH(edge, EDGE,
        KITTENS_BOOL_SWITCH(accumulate, ACCUMULATE,
            KITTENS_BOOL_SWITCH(a_fp8, CBSZ,
                KITTENS_BOOL_SWITCH(b_fp8, BLGP,
                    const void *kernel = (const void *)mxfp8_wgrad_nt_kernel<CBSZ, BLGP, ACCUMULATE, EDGE, OutGL>;
                    const int wgs = persistent_grid(kernel, num_tiles, ctr, stream);
                    mxfp8_wgrad_nt_kernel<CBSZ, BLGP, ACCUMULATE, EDGE, OutGL><<<wgs, NUM_THREADS, 0, stream>>>(
                        gl_A, gl_B, gl_D, gl_SA, gl_SB, tab, tiles_M, tiles_N, tiles_M * tiles_N, num_tiles, ctr);
    )))) // NOLINT(*)
}

static bool grouped_mxfp8_wgrad(const void *const *A_array, const void *const *B_array, void *const *D_array,
    const void *const *scale_A_array, const void *const *scale_B_array,
    int N, int K, const int *M_array, int num_experts,
    int a_dtype, int b_dtype, int out_dtype, bool accumulate,
    void *workspace, size_t workspace_size, hipStream_t stream) {

    if (N % BLOCK_ROW != 0) { warn_fallback("HK-wgrad", "N not 256-aligned"); return false; }
    if (K % BLOCK_COL != 0) { warn_fallback("HK-wgrad", "K not 256-aligned"); return false; }
    if (num_experts <= 0)    { warn_fallback("HK-wgrad", "num_experts <= 0"); return false; }
    if (out_dtype != KITTENS_BFLOAT16 && out_dtype != KITTENS_FLOAT16 && out_dtype != KITTENS_FLOAT32) {
        return false;
    }

    std::vector<std::pair<int, int>> keyed;
    keyed.reserve(num_experts);
    size_t total_sa_entries = 0;
    size_t total_sb_entries = 0;
    int max_M = 0;
    bool edge = false;
    for (int g = 0; g < num_experts; g++) {
        int M_g = M_array[g];
        if (M_g % 32 != 0) {
            warn_fallback("HK-wgrad", "M_i not 32-aligned");
            return false;
        }
        if (M_g == 0 && accumulate) {
            continue;
        }
        keyed.push_back({-M_g, g});
        total_sa_entries += (size_t)wgrad_k_iters(M_g) * N;
        total_sb_entries += (size_t)2 * wgrad_k_iters(M_g) * K;
        max_M = std::max(max_M, M_g);
        edge = edge || wgrad_k_iters(M_g) * BLOCK_K != M_g;
    }
    // The head shift masks rows with an out-of-range offset, so every expert's bytes need 31-bit offsets.
    if (edge && (size_t)wgrad_k_iters(max_M) * BLOCK_K * std::max(N, K) >= 0x80000000ull) {
        warn_fallback("HK-wgrad", "expert too large");
        return false;
    }
    if (keyed.empty()) {
        return true;
    }
    std::sort(keyed.begin(), keyed.end());
    std::vector<int> order(keyed.size());
    for (size_t i = 0; i < keyed.size(); i++) {
        order[i] = keyed[i].second;
    }

    size_t sa_pk_bytes = kittens_align_up(total_sa_entries * sizeof(uint32_t), 256);
    size_t sb_pk_bytes = kittens_align_up(total_sb_entries * sizeof(uint32_t), 256);
    if (workspace_size < SCHED_CTR_BYTES + sa_pk_bytes + sb_pk_bytes) {
        warn_fallback("HK-wgrad", "workspace too small");
        return false;
    }

    auto *ctr   = (int *)workspace;
    auto *sa_pk = (uint32_t *)((uint8_t *)workspace + SCHED_CTR_BYTES);
    auto *sb_pk = (uint32_t *)((uint8_t *)workspace + SCHED_CTR_BYTES + sa_pk_bytes);
    int a_fp8 = fp8_code(static_cast<KittensDType>(a_dtype));
    int b_fp8 = fp8_code(static_cast<KittensDType>(b_dtype));

    const int n = (int)order.size();
    for (int c0 = 0; c0 < n; c0 += MAX_TAB_EXPERTS) {
        int ne = std::min(MAX_TAB_EXPERTS, n - c0);
        if (out_dtype == KITTENS_BFLOAT16) {
            grouped_mxfp8_wgrad_chunk<gl_bf16_rt>(A_array, B_array, D_array, scale_A_array, scale_B_array, M_array,
                order.data() + c0, ne, N, K, max_M, accumulate, a_fp8, b_fp8, ctr, sa_pk, sb_pk, stream);
        } else if (out_dtype == KITTENS_FLOAT16) {
            grouped_mxfp8_wgrad_chunk<gl_fp16_rt>(A_array, B_array, D_array, scale_A_array, scale_B_array, M_array,
                order.data() + c0, ne, N, K, max_M, accumulate, a_fp8, b_fp8, ctr, sa_pk, sb_pk, stream);
        } else {
            grouped_mxfp8_wgrad_chunk<gl_f32_rt>(A_array, B_array, D_array, scale_A_array, scale_B_array, M_array,
                order.data() + c0, ne, N, K, max_M, accumulate, a_fp8, b_fp8, ctr, sa_pk, sb_pk, stream);
        }
    }
    return true;
}

class MXFP8GemmCdna4 final : public MXFP8GemmBackend {
 public:
    bool gemm(const MXFP8GemmArgs &args) override {
        return mxfp8_gemm(args.A, args.B, args.C, args.scale_A, args.scale_B,
                          args.M, args.N, args.K, args.transa, args.transb,
                          args.a_dtype, args.b_dtype, args.bias, args.bias_dtype,
                          args.aux_gelu, args.out_dtype, args.aux_dtype, args.beta,
                          args.workspace, args.workspace_size, args.stream);
    }

    bool grouped_gemm(const MXFP8GroupedGemmArgs &args) override {
        return grouped_mxfp8_gemm(args.A_array, args.B_array, args.C_array,
                                  args.scale_A_array, args.scale_B_array,
                                  args.M, args.N_array, args.K, args.num_experts,
                                  args.transa, args.transb,
                                  args.a_dtype, args.b_dtype, args.out_dtype,
                                  args.workspace, args.workspace_size, args.stream);
    }

    bool grouped_wgrad(const MXFP8WgradArgs &args) override {
        return grouped_mxfp8_wgrad(args.A_array, args.B_array, args.D_array,
                                   args.scale_A_array, args.scale_B_array,
                                   args.N, args.K, args.M_array, args.num_experts,
                                   args.a_dtype, args.b_dtype, args.out_dtype, args.accumulate,
                                   args.workspace, args.workspace_size, args.stream);
    }
};

}  // namespace te_kittens::cdna4::mxfp8

MXFP8GemmBackend *MXFP8GemmBackend::get_cdna4() {
    static te_kittens::cdna4::mxfp8::MXFP8GemmCdna4 impl;
    return &impl;
}
