/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#include "rocm_norm.h"

#include <algorithm>

#include "../common.h"
#include "../util/cuda_runtime.h"
#include "../util/system.h"
#include "rocm_norm_bwd.cuh"
#include "rocm_norm_fwd.cuh"

namespace transformer_engine {
namespace normalization {
namespace rocm {
namespace {

using FwdKernel = void (*)(ForwardKernelParams, RowSplit, float *);
using Mxfp8Kernel = void (*)(ForwardKernelParams, Mxfp8Out);
using BwdKernel = void (*)(BackwardKernelParams, RowSplit);
using PlanKernel = std::function<void(LaunchParams<ForwardKernelParams> &, const bool)>;
using BwdPlanKernel = std::function<void(LaunchParams<BackwardKernelParams> &, const bool)>;

bool is_float(const DType t) {
  return t == DType::kFloat32 || t == DType::kBFloat16 || t == DType::kFloat16;
}

// Whether the ROCm kernels support these types, which the TE kernels support too: weights of the
// input type, and outputs of it, FP8 or, from FP32, 16-bit.
bool types_supported(const DType wtype, const DType itype, const DType otype) {
  return wtype == itype && is_float(itype) &&
         (otype == itype || is_fp8_dtype(otype) || (itype == DType::kFloat32 && is_float(otype)));
}

// The split of the kernels, which spreads few rows over more waves, for about TARGET_WAVES_PER_CU
// waves per CU.
RowSplit row_split(const size_t rows, const size_t cols, const DType itype, const int sm_count,
                   const int max_vecs_per_lane = MAX_VECS_PER_LANE) {
  constexpr size_t TARGET_WAVES_PER_CU = 16;
  const size_t min_waves_per_row =
      DIVUP(TARGET_WAVES_PER_CU * sm_count, std::max<size_t>(rows, 1));
  return make_row_split(cols, typeToSize(itype),
                        static_cast<int>(std::min<size_t>(min_waves_per_row, MAX_WAVES)),
                        max_vecs_per_lane);
}

template <int MAX_WAVES_PER_BLOCK>
FwdKernel fwd_kernel(const bool rms, const DType itype, const DType otype) {
  FwdKernel kernel = nullptr;
  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
      itype, IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(
          otype, OType,
          if constexpr (std::is_same_v<IType, OType> || is_fp8<OType>::value ||
                        std::is_same_v<IType, fp32>) {
            if (rms) {
              kernel = norm_fwd_kernel<true, IType, IType, OType, MAX_WAVES_PER_BLOCK>;
            } else {
              kernel = norm_fwd_kernel<false, IType, IType, OType, MAX_WAVES_PER_BLOCK>;
            }
          }););
  return kernel;
}

// norm_fwd_kernel writing the statistics only
template <int MAX_WAVES_PER_BLOCK>
FwdKernel fwd_stats_kernel(const bool rms, const DType itype) {
  FwdKernel kernel = nullptr;
  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
      itype, IType,
      kernel = rms ? norm_fwd_kernel<true, IType, IType, IType, MAX_WAVES_PER_BLOCK, true>
                   : norm_fwd_kernel<false, IType, IType, IType, MAX_WAVES_PER_BLOCK, true>;);
  return kernel;
}

template <bool COLWISE>
Mxfp8Kernel mxfp8_kernel(const bool rms, const DType itype, const DType otype) {
  Mxfp8Kernel kernel = nullptr;
  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
      itype, IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
          otype, OType,
          kernel = rms ? norm_mxfp8_kernel<true, IType, IType, OType, COLWISE>
                       : norm_mxfp8_kernel<false, IType, IType, OType, COLWISE>;););
  return kernel;
}

// Sets the dynamic LDS of kernel and returns how many of its blocks of threads fit on a CU.
template <typename Kernel>
int blocks_per_cu(const Kernel kernel, const int threads, const size_t smem) {
  NVTE_CHECK_CUDA(hipFuncSetAttribute(reinterpret_cast<const void *>(kernel),
                                      hipFuncAttributeMaxDynamicSharedMemorySize, smem));
  int blocks = 0;
  NVTE_CHECK_CUDA(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, threads, smem));
  return blocks;
}

// The persistent blocks of a norm_fwd_kernel for the split, at least one to set the scale inverse,
// or 0 where none fits on a CU.
size_t fwd_blocks(const FwdKernel kernel, const RowSplit &split, const size_t rows,
                  const int sm_count) {
  const int occupancy =
      blocks_per_cu(kernel, block_waves(split) * WAVE_SIZE, block_slice_bytes(split));
  if (occupancy == 0) {
    return 0;
  }
  return std::max<size_t>(1, std::min(DIVUP(rows, static_cast<size_t>(split.rows_per_step)),
                                      static_cast<size_t>(occupancy) * sm_count));
}

// The statistics of norm_fwd_kernel, then norm_mxfp8_kernel, with columnwise output where the plan
// is for training
PlanKernel get_fwd_mxfp8_kernel(const bool rms, const DType itype, const DType otype,
                                const size_t rows, const size_t cols, const RowSplit &split,
                                const int sm_count) {
  const FwdKernel stats_kernel = block_waves(split) <= SMALL_BLOCK_WAVES
                                     ? fwd_stats_kernel<SMALL_BLOCK_WAVES>(rms, itype)
                                     : fwd_stats_kernel<MAX_WAVES>(rms, itype);
  const size_t stats_blocks = fwd_blocks(stats_kernel, split, rows, sm_count);
  if (stats_blocks == 0) {
    return {};
  }
  const int stats_threads = block_waves(split) * WAVE_SIZE;
  const size_t stats_smem = block_slice_bytes(split);
  const Mxfp8Kernel rowwise_kernel = mxfp8_kernel<false>(rms, itype, otype);
  const Mxfp8Kernel both_kernel = mxfp8_kernel<true>(rms, itype, otype);
  const dim3 grid(DIVUP(cols, mxfp8_tdm::CHUNK_DIM_X), DIVUP(rows, mxfp8_tdm::CHUNK_DIM_Y));
  return [=](LaunchParams<ForwardKernelParams> &lp, const bool configure) {
    if (configure || rows == 0) {
      return;
    }
    const Tensor &z = *lp.z_tensor;
    NVTE_CHECK(z.data.dptr != nullptr, "MXFP8 normalization output must have rowwise data.");
    stats_kernel<<<stats_blocks, stats_threads, stats_smem, lp.stream>>>(lp.params, split,
                                                                          nullptr);
    NVTE_CHECK_CUDA(hipGetLastError());
    const Mxfp8Out out{z.data.dptr,
                       z.columnwise_data.dptr,
                       static_cast<e8m0_t *>(z.scale_inv.dptr),
                       static_cast<e8m0_t *>(z.columnwise_scale_inv.dptr),
                       z.scale_inv.shape[1],
                       lp.training ? z.columnwise_scale_inv.shape[1] : 1};
    const Mxfp8Kernel kernel = lp.training ? both_kernel : rowwise_kernel;
    kernel<<<grid, mxfp8_tdm::THREADS_PER_CHUNK, 0, lp.stream>>>(lp.params, out);
    NVTE_CHECK_CUDA(hipGetLastError());
  };
}

template <int MAX_WAVES_PER_BLOCK>
BwdKernel bwd_kernel(const bool rms, const bool add, const DType itype) {
  BwdKernel kernel = nullptr;
  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
      itype, IType,
      if (!rms) {
        kernel = norm_bwd_kernel<false, IType, MAX_WAVES_PER_BLOCK, false>;
      } else if (add) {
        kernel = norm_bwd_kernel<true, IType, MAX_WAVES_PER_BLOCK, true>;
      } else {
        kernel = norm_bwd_kernel<true, IType, MAX_WAVES_PER_BLOCK, false>;
      });
  return kernel;
}

}  // namespace

PlanKernel get_fwd_kernel(const NVTE_Norm_Type norm_type, const DType wtype, const DType itype,
                          const DType otype, const LaunchParams<ForwardKernelParams> &launch_params,
                          const bool aligned) {
  const auto &params = launch_params.params;
  if (!aligned || !types_supported(wtype, itype, otype) ||
      !dispatch::common::use_tdm(params.rows, params.cols, {})) {
    return {};
  }
  const bool rms = norm_type == NVTE_Norm_Type::RMSNorm;
  const RowSplit split =
      row_split(params.rows, params.cols, itype, launch_params.multiprocessorCount);
  if (split.waves_per_row == 0) {
    return {};
  }
  // The TE path for MXFP8 writes a float buffer and quantizes it separately.
  if (params.mxfp8_out) {
    return get_fwd_mxfp8_kernel(rms, itype, otype, params.rows, params.cols, split,
                                launch_params.multiprocessorCount);
  }
  // Where a row fits in one of their blocks, the TE kernels are faster for inputs of up to 16 MiB
  // and LayerNorm rows under 3 KB.
  constexpr size_t TE_MAX_COLS = 8192;
  constexpr size_t TE_MAX_BYTES = size_t{16} << 20;
  constexpr size_t TE_MAX_LN_ROW_BYTES = 3072;
  const size_t row_bytes = params.cols * typeToSize(itype);
  if (static_cast<size_t>(params.cols) <= TE_MAX_COLS &&
      (params.rows * row_bytes <= TE_MAX_BYTES || (!rms && row_bytes < TE_MAX_LN_ROW_BYTES))) {
    return {};
  }
  const int threads = block_waves(split) * WAVE_SIZE;
  const FwdKernel kernel = block_waves(split) <= SMALL_BLOCK_WAVES
                               ? fwd_kernel<SMALL_BLOCK_WAVES>(rms, itype, otype)
                               : fwd_kernel<MAX_WAVES>(rms, itype, otype);
  const size_t smem = block_slice_bytes(split);
  const size_t blocks =
      fwd_blocks(kernel, split, params.rows, launch_params.multiprocessorCount);
  if (blocks == 0) {
    return {};
  }
  return [=](LaunchParams<ForwardKernelParams> &lp, const bool configure) {
    if (configure) {
      return;
    }
    float *const amax = static_cast<float *>(lp.params.amax);
    float *const amax_slots = amax != nullptr ? rocm_amax_slots(blocks, lp.stream) : nullptr;
    kernel<<<blocks, threads, smem, lp.stream>>>(lp.params, split, amax_slots);
    NVTE_CHECK_CUDA(hipGetLastError());
    if (amax_slots != nullptr) {
      rocm_amax_finalize(amax_slots, amax, nullptr, lp.stream);
    }
  };
}

BwdPlanKernel get_bwd_kernel(const NVTE_Norm_Type norm_type, const NVTE_Norm_Stage norm_stage,
                             const DType wtype, const DType itype, const DType otype,
                             const LaunchParams<BackwardKernelParams> &launch_params,
                             const bool aligned) {
  const auto &params = launch_params.params;
  if (!aligned || wtype != itype || otype != itype || !is_float(itype) ||
      !dispatch::common::use_tdm(params.rows, params.cols, {})) {
    return {};
  }
  const bool rms = norm_type == NVTE_Norm_Type::RMSNorm;
  const bool add = norm_stage == NVTE_Norm_Stage::BackwardAdd;
  if (add && !rms) {
    return {};
  }
  const RowSplit split = row_split(params.rows, params.cols, itype,
                                   launch_params.multiprocessorCount, BWD_VECS_PER_LANE);
  if (split.waves_per_row == 0) {
    return {};
  }
  const int threads = block_waves(split) * WAVE_SIZE;
  const BwdKernel kernel = block_waves(split) <= SMALL_BLOCK_WAVES
                               ? bwd_kernel<SMALL_BLOCK_WAVES>(rms, add, itype)
                               : bwd_kernel<MAX_WAVES>(rms, add, itype);
  const size_t smem = bwd_block_slice_bytes(split, add);
  const int occupancy = blocks_per_cu(kernel, threads, smem);
  if (occupancy == 0) {
    return {};
  }
  // Persistent blocks, at least one to zero the partials
  const size_t rows = params.rows;
  const size_t cols = params.cols;
  const size_t blocks = std::max<size_t>(
      1, std::min(DIVUP(rows, static_cast<size_t>(split.rows_per_step)),
                  static_cast<size_t>(occupancy) * launch_params.multiprocessorCount));
  const size_t partial_rows = blocks;
  return [=](LaunchParams<BackwardKernelParams> &lp, const bool configure) {
    if (configure) {
      lp.dgamma_part_bytes = partial_rows * cols * sizeof(float);
      return;
    }
    kernel<<<blocks, threads, smem, lp.stream>>>(lp.params, split);
    NVTE_CHECK_CUDA(hipGetLastError());
    TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
        itype, WType,
        dispatch::common::reduce_dbias_tdm(static_cast<WType *>(lp.params.dgamma),
                                           static_cast<const float *>(lp.params.dgamma_part), cols,
                                           1, partial_rows, nullptr, nullptr, lp.stream);
        if (!rms) {
          dispatch::common::reduce_dbias_tdm(static_cast<WType *>(lp.params.dbeta),
                                             static_cast<const float *>(lp.params.dbeta_part),
                                             cols, 1, partial_rows, nullptr, nullptr, lp.stream);
        });
  };
}

}  // namespace rocm
}  // namespace normalization
}  // namespace transformer_engine

bool nvte_norm_fwd_mxfp8_fused_supported(const NVTETensor x, const NVTETensor gamma) {
  using namespace transformer_engine;
  using namespace transformer_engine::normalization::rocm;
  const Tensor &input = *convertNVTETensorCheck(x);
  const Tensor &weight = *convertNVTETensorCheck(gamma);
  const auto [rows, cols] = input.flat_2d_dims();
  return types_supported(weight.dtype(), input.dtype(), DType::kFloat8E4M3) &&
         dispatch::common::use_tdm(rows, cols, {input.data.dptr, weight.data.dptr}) &&
         make_row_split(cols, typeToSize(input.dtype())).waves_per_row != 0;
}
