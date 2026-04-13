/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "utils.cuh"
#include "transpose/cast_transpose_tile_kernel.cuh"

using namespace transformer_engine;
using namespace transformer_engine::transpose;

namespace {

// Parameters
using CType = float;
using IType = __ITYPE__;
using OType = __OTYPE__;
constexpr size_t load_size = __LOAD_SIZE__;
constexpr size_t store_size = __STORE_SIZE__;
constexpr size_t warps_per_tile = __WARPS_PER_TILE__;
constexpr size_t block_size = __BLOCK_SIZE__;

}  // namespace

__global__ void __launch_bounds__(block_size) cast_transpose_optimized_kernel(
    const IType* __restrict__ const input, const CType* __restrict__ const noop,
    OType* __restrict__ const output_c, OType* __restrict__ const output_t,
    const CType* __restrict__ const scale_ptr, CType* __restrict__ const amax_ptr,
    CType* __restrict__ const scale_inv_ptr, const size_t row_length, const size_t num_rows) {
  if (noop != nullptr && noop[0] == 1.0f) return;

  cast_transpose_tile_impl<NvOps, load_size, store_size, warps_per_tile,
                           THREADS_PER_WARP, IType, OType>(
      input, output_c, output_t, scale_ptr, amax_ptr, scale_inv_ptr,
      num_rows, row_length, num_rows);
}
