/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#pragma once

#include "../util/rocm_device_utils.cuh"
#include "rocm_cast_transpose.cuh"

constexpr int kMCTMaxTensors = 256;

struct RocmMultiCastTransposeArgs {
  const void *input_list[kMCTMaxTensors];
  void       *output_c_list[kMCTMaxTensors];
  void       *output_t_list[kMCTMaxTensors];
  const void *scale_list[kMCTMaxTensors];
  void       *amax_list[kMCTMaxTensors];
  void       *scale_inv_list[kMCTMaxTensors];
  int   num_rows_list[kMCTMaxTensors];
  int   row_length_list[kMCTMaxTensors];
  int   block_range[kMCTMaxTensors + 1];
  int   num_tensors;
};

template <bool IS_EDGE, int LOAD_SIZE, int STORE_SIZE, int WARPS_PER_TILE,
          typename IType, typename OType>
__device__ __forceinline__ void
mct_cast_store(
    const IType *__restrict__ input,
    OType       *__restrict__ output_c,
    const int                 row_length,
    const int                 num_rows,
    const float               scale,
    float                    &amax,
    NTVec<OType, STORE_SIZE / sizeof(OType)> (&local_t)[LOAD_SIZE / sizeof(IType)][ROCM_CT_WARP_SIZE / WARPS_PER_TILE],
    const int tidx, const int tidy,
    const int row_base, const int col_base)
{
    constexpr int NVEC_IN   = LOAD_SIZE / sizeof(IType);
    constexpr int NVEC_OUT  = STORE_SIZE / sizeof(OType);
    constexpr int NUM_ITERS = ROCM_CT_WARP_SIZE / WARPS_PER_TILE;

    using IVec  = NTVec<IType, NVEC_IN>;
    using OVecC = NTVec<OType, NVEC_IN>;

#pragma unroll
    for (int iter = 0; iter < NUM_ITERS; iter++) {
        const int i1 = tidy + iter * WARPS_PER_TILE;
        const int j1 = tidx;
#pragma unroll
        for (int i2 = 0; i2 < NVEC_OUT; i2++) {
            const int row = row_base + i1 * NVEC_OUT + i2;
            const int col = col_base + j1 * NVEC_IN;

            IVec  in;
            OVecC out_c;

            if (IS_EDGE && row >= num_rows) {
#pragma unroll
                for (int j2 = 0; j2 < NVEC_IN; j2++) in.val[j2] = IType(0);
            } else {
                in.load(&input[row * row_length + col]);
            }

#ifdef HAS_PACK_4xFLOAT8
            if constexpr (sizeof(OType) == 1) {
#pragma unroll
                for (int j2 = 0; j2 < NVEC_IN; j2 += 4) {
                    const float v0 = static_cast<float>(in.val[j2]);
                    const float v1 = (j2+1 < NVEC_IN) ? static_cast<float>(in.val[j2+1]) : 0.0f;
                    const float v2 = (j2+2 < NVEC_IN) ? static_cast<float>(in.val[j2+2]) : 0.0f;
                    const float v3 = (j2+3 < NVEC_IN) ? static_cast<float>(in.val[j2+3]) : 0.0f;
                    if (!IS_EDGE || row < num_rows)
                        amax = fmaxf(amax, fmaxf(fmaxf(fabsf(v0), fabsf(v1)),
                                                  fmaxf(fabsf(v2), fabsf(v3))));
                    uint32_t packed = rocm_pack_4xfloat8<OType>(
                        v0 * scale, v1 * scale, v2 * scale, v3 * scale);
                    uint8_t *bytes = reinterpret_cast<uint8_t *>(&packed);
#pragma unroll
                    for (int k = 0; k < 4 && j2 + k < NVEC_IN; k++) {
                        out_c.val[j2 + k] = reinterpret_cast<OType &>(bytes[k]);
                        local_t[j2 + k][iter].val[i2] = out_c.val[j2 + k];
                    }
                }
            } else
#endif
            {
#pragma unroll
                for (int j2 = 0; j2 < NVEC_IN; j2++) {
                    const float v = static_cast<float>(in.val[j2]);
                    if (!IS_EDGE || row < num_rows)
                        amax = fmaxf(amax, fabsf(v));
                    const OType o = static_cast<OType>(v * scale);
                    out_c.val[j2] = o;
                    local_t[j2][iter].val[i2] = o;
                }
            }

            if (!IS_EDGE || row < num_rows)
                out_c.nt_store(&output_c[row * row_length + col]);
        }
    }
}

template <bool IS_EDGE, int LOAD_SIZE, int STORE_SIZE, int WARPS_PER_TILE,
          typename IType, typename OType>
__device__ __forceinline__ void
mct_transpose_store(
    OType *__restrict__ output_t,
    const int           num_rows,
    NTVec<OType, STORE_SIZE / sizeof(OType)> (&smem)[ROCM_CT_WARP_SIZE][ROCM_CT_WARP_SIZE + 1],
    NTVec<OType, STORE_SIZE / sizeof(OType)> (&local_t)[LOAD_SIZE / sizeof(IType)][ROCM_CT_WARP_SIZE / WARPS_PER_TILE],
    const int tidx, const int tidy,
    const int row_base, const int col_base)
{
    constexpr int NVEC_IN   = LOAD_SIZE / sizeof(IType);
    constexpr int NVEC_OUT  = STORE_SIZE / sizeof(OType);
    constexpr int NUM_ITERS = ROCM_CT_WARP_SIZE / WARPS_PER_TILE;

#pragma unroll
    for (int j2 = 0; j2 < NVEC_IN; j2++) {
#pragma unroll
        for (int iter = 0; iter < NUM_ITERS; iter++) {
            smem[tidx][tidy + iter * WARPS_PER_TILE] = local_t[j2][iter];
        }
        __syncthreads();
#pragma unroll
        for (int iter = 0; iter < NUM_ITERS; iter++) {
            const int i1  = tidx;
            const int j1  = tidy + iter * WARPS_PER_TILE;
            const int row = row_base + i1 * NVEC_OUT;
            const int col = col_base + j1 * NVEC_IN + j2;

            if (IS_EDGE && row + NVEC_OUT > num_rows) {
                if (row < num_rows) {
                    for (int k = 0; k < NVEC_OUT && row + k < num_rows; k++)
                        output_t[col * num_rows + row + k] = smem[j1][i1].val[k];
                }
            } else {
                smem[j1][i1].nt_store(&output_t[col * num_rows + row]);
            }
        }
        if (j2 + 1 < NVEC_IN) {
            __syncthreads();
        }
    }
}

template <int LOAD_SIZE, int STORE_SIZE, int WARPS_PER_TILE,
          typename IType, typename OType>
__global__ void __launch_bounds__(ROCM_CT_WARP_SIZE * WARPS_PER_TILE)
rocm_multi_cast_transpose_kernel(RocmMultiCastTransposeArgs args) {
    constexpr int NVEC_IN   = LOAD_SIZE / sizeof(IType);
    constexpr int NVEC_OUT  = STORE_SIZE / sizeof(OType);
    constexpr int TILE_COLS = ROCM_CT_WARP_SIZE * NVEC_IN;
    constexpr int TILE_ROWS = ROCM_CT_WARP_SIZE * NVEC_OUT;
    constexpr int NUM_ITERS = ROCM_CT_WARP_SIZE / WARPS_PER_TILE;

    using OVecT = NTVec<OType, NVEC_OUT>;

    const int tid  = threadIdx.x;
    const int tidx = tid % ROCM_CT_WARP_SIZE;
    const int tidy = tid / ROCM_CT_WARP_SIZE;
    const int bid  = blockIdx.x;

    int lo = 0, hi = args.num_tensors - 1;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (args.block_range[mid + 1] <= bid) lo = mid + 1;
        else                                  hi = mid;
    }

    const int tensor_id  = lo;
    const int local_bid  = bid - args.block_range[tensor_id];
    const int num_rows   = args.num_rows_list[tensor_id];
    const int row_length = args.row_length_list[tensor_id];

    const IType *__restrict__ input = reinterpret_cast<const IType *>(args.input_list[tensor_id]);
    OType *__restrict__ output_c    = reinterpret_cast<OType *>(args.output_c_list[tensor_id]);
    OType *__restrict__ output_t    = reinterpret_cast<OType *>(args.output_t_list[tensor_id]);

    const float *__restrict__ scale_ptr = reinterpret_cast<const float *>(args.scale_list[tensor_id]);
    float *__restrict__ amax_ptr        = reinterpret_cast<float *>(args.amax_list[tensor_id]);
    float *__restrict__ scale_inv_ptr   = reinterpret_cast<float *>(args.scale_inv_list[tensor_id]);

    const int tiles_m  = (num_rows + TILE_ROWS - 1) / TILE_ROWS;
    const int tile_m   = local_bid % tiles_m;
    const int tile_n   = local_bid / tiles_m;
    const int row_base = tile_m * TILE_ROWS;
    const int col_base = tile_n * TILE_COLS;

    const bool is_edge = (row_base + TILE_ROWS > num_rows);

    const float scale = (scale_ptr != nullptr) ? *scale_ptr : 1.0f;
    float amax = 0.0f;

    __shared__ OVecT smem[ROCM_CT_WARP_SIZE][ROCM_CT_WARP_SIZE+1];

    OVecT local_t[NVEC_IN][NUM_ITERS];

    if (is_edge) {
        mct_cast_store<true, LOAD_SIZE, STORE_SIZE, WARPS_PER_TILE, IType, OType>(
            input, output_c, row_length, num_rows, scale, amax, local_t,
            tidx, tidy, row_base, col_base);
        mct_transpose_store<true, LOAD_SIZE, STORE_SIZE, WARPS_PER_TILE, IType, OType>(
            output_t, num_rows, smem, local_t,
            tidx, tidy, row_base, col_base);
    } else {
        mct_cast_store<false, LOAD_SIZE, STORE_SIZE, WARPS_PER_TILE, IType, OType>(
            input, output_c, row_length, num_rows, scale, amax, local_t,
            tidx, tidy, row_base, col_base);
        mct_transpose_store<false, LOAD_SIZE, STORE_SIZE, WARPS_PER_TILE, IType, OType>(
            output_t, num_rows, smem, local_t,
            tidx, tidy, row_base, col_base);
    }

    if (amax_ptr != nullptr) {
        amax = rocm_block_reduce_max<WARPS_PER_TILE>(amax, tidy);
        if (tid == 0) {
            rocm_atomicMaxFloat(amax_ptr, amax);
        }
    }

    if (local_bid == 0 && tid == 0 && scale_inv_ptr != nullptr) {
        *scale_inv_ptr = __frcp_rn(scale);
    }
}

template <typename IType, typename OType>
void rocm_multi_cast_transpose_dispatch(size_t num_tensors, const IType *const *input_list, OType *const *output_c_list,
                                        OType *const *output_t_list, const float *const *scale_list, float *const *amax_list,
                                        float *const *scale_inv_list, const size_t *num_rows_list, 
                                        const size_t *row_length_list, hipStream_t stream) {
    constexpr int WPT       = 16;
    constexpr int BLK       = ROCM_CT_WARP_SIZE * WPT;
    constexpr int ISZ       = sizeof(IType);
    constexpr int OSZ       = sizeof(OType);
    constexpr int LOAD_SZ   = 16;
    constexpr int STORE_SZ  = 8;
    constexpr int TILE_COLS = ROCM_CT_WARP_SIZE * (LOAD_SZ / ISZ);
    constexpr int TILE_ROWS = ROCM_CT_WARP_SIZE * (STORE_SZ / OSZ);

    size_t i = 0;

    while (i < num_tensors) {
        RocmMultiCastTransposeArgs args;
        args.block_range[0] = 0;

        int total_blocks = 0;
        int packed       = 0;

        while (i < num_tensors && packed < kMCTMaxTensors) {
            int rows = num_rows_list[i];
            int cols = row_length_list[i];

            if (cols % TILE_COLS != 0 || rows == 0) {
                if (rows > 0 && cols > 0) {
                    size_t done = rocm_cast_transpose_dispatch<IType, OType>(input_list[i], nullptr, output_c_list[i], 
                                                                             output_t_list[i], scale_list[i], 
                                                                             amax_list[i], scale_inv_list[i], cols, 
                                                                             rows, stream);
                    if (done < static_cast<size_t>(rows)) {
                        size_t rem = rows - done;
                        hipLaunchKernelGGL(
                            (rocm_cast_transpose_remainder_kernel<IType, OType>),
                            dim3((rem * cols + 255) / 256), dim3(256), 0, stream,
                            input_list[i] + done * cols, nullptr,
                            output_c_list[i] + done * cols, output_t_list[i] + done,
                            scale_list[i], amax_list[i], scale_inv_list[i],
                            rem, cols, cols, rows);
                    }
                }
                i++;
                continue;
            }

            int tiles_m = (rows + TILE_ROWS - 1) / TILE_ROWS;
            int tiles_n = cols / TILE_COLS;
            int tiles   = tiles_m * tiles_n;

            args.input_list[packed]      = reinterpret_cast<const void *>(input_list[i]);
            args.output_c_list[packed]   = reinterpret_cast<void *>(output_c_list[i]);
            args.output_t_list[packed]   = reinterpret_cast<void *>(output_t_list[i]);
            args.scale_list[packed]      = reinterpret_cast<const void *>(scale_list[i]);
            args.amax_list[packed]       = amax_list[i];
            args.scale_inv_list[packed]  = scale_inv_list[i];
            args.num_rows_list[packed]   = rows;
            args.row_length_list[packed] = cols;
            total_blocks += tiles;
            args.block_range[packed + 1] = total_blocks;
            packed++;
            i++;
        }

        if (total_blocks > 0) {
            args.num_tensors = packed;
            hipLaunchKernelGGL(
                (rocm_multi_cast_transpose_kernel<LOAD_SZ, STORE_SZ, WPT, IType, OType>),
                dim3(total_blocks), dim3(BLK), 0, stream,
                args);
        }
    }
}
