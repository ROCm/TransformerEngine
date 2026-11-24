/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
 * Portions adapted from HipKittens, Copyright (c) 2024 HazyResearch, MIT License
 * (see 3rdparty/hipkittens/LICENSE).
*************************************************************************/
// MoE backward NN building blocks for the dgrad2 and dgrad1 tiles.
#pragma once

namespace hk_moe {

// A-side LDS-DMA load passing the real LDS pointer through the intrinsic: kittens G::load sets m0 in separate asm,
// and the backend's own m0 init for hand-written ds_* asm can land in between and redirect the DMA.
template <ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD = coord<ST>>
__device__ __forceinline__ void moe_nn_load_A(ST &dst, const GL &src, const COORD &idx,
                                              const uint32_t *__restrict__ swizzled_offsets, i32x4 srd,
                                              const void *base_ptr, uint32_t lds_base) {
  using T = typename ST::dtype;
  constexpr int bytes_per_thread = 16;
  constexpr int bytes_per_memcpy = bytes_per_thread * NUM_THREADS;
  constexpr int memcpy_per_tile  = (ST::rows * ST::cols * sizeof(T)) / bytes_per_memcpy;
  static_assert(memcpy_per_tile > 0, "A tile too small for one group memcpy");

  coord<> unit_coord = idx.template unit_coord<2, 3>();
  const T *__restrict__ gptr = (const T *)&src[unit_coord];
  const uint32_t SOFF = (uint32_t)((const char *)gptr - (const char *)base_ptr);

  uint32_t lds_cur = lds_base;
#pragma unroll
  for (int i = 0; i < memcpy_per_tile; i++) {
    llvm_amdgcn_raw_buffer_load_lds(srd, (as3_uint32_ptr)(uintptr_t)lds_cur, 16, swizzled_offsets[i], SOFF, 0,
                                    static_cast<int>(coherency::cache_all));
    lds_cur += bytes_per_memcpy;
  }
}

struct NnArrivalGate {
  const int *sig;
  const int *cnt_by_src;
  const int *start_by_src;
  hk_comm::CDispatch push;
};

template <int OFF, typename RT, typename ST>
__device__ __forceinline__ void moe_nn_read_A(RT &dst, const ST &src) {
  static_assert(RT::base_tile_stride == 8 && std::is_same_v<typename RT::T, bf16>, "bf16, stride 8");
  static_assert(ST::underlying_subtile_rows >= RT::base_tile_rows && ST::underlying_subtile_cols >= RT::base_tile_cols,
                "shared subtile >= register subtile");
  constexpr int packing = base_types::packing<typename RT::dtype>::num();
  constexpr int rps     = ST::underlying_subtile_cols / RT::base_tile_cols;
  constexpr int rpc     = ST::underlying_subtile_rows / RT::base_tile_rows;
  const int laneid     = kittens::laneid();
  const int row_offset = laneid % RT::base_tile_rows;
  const int col_offset = RT::base_tile_stride * (laneid / RT::base_tile_rows);
  const uint32_t src_ptr = reinterpret_cast<uintptr_t>(&src.data[0]);
#pragma unroll
  for (int k = 0; k < RT::base_tile_num_strides; k++) {
#pragma unroll
    for (int i = 0; i < rpc; i++) {
#pragma unroll
      for (int j = 0; j < rps; j++) {
        const int row = i * RT::base_tile_rows + row_offset;
        const int col = j * RT::base_tile_cols + col_offset + k * RT::base_tile_elements_per_stride_group;
        const uint32_t addr = src_ptr + src.swizzle({row, col});
        const int idx = k * RT::base_tile_stride / packing;
#pragma unroll
        for (int ii = 0; ii < ST::subtiles_per_col; ii++) {
#pragma unroll
          for (int jj = 0; jj < ST::subtiles_per_row; jj++) {
            const int offset = (ii * ST::underlying_subtiles_per_row + jj) * ST::underlying_subtile_bytes;
            asm volatile("ds_read_b128 %0, %1 offset:%2\n"
                         : "=v"(*reinterpret_cast<float4 *>(&dst.tiles[ii * rpc + i][jj * rps + j].data[idx]))
                         : "v"(addr), "i"(offset + OFF)
                         : "memory");
          }
        }
      }
    }
  }
}

constexpr int MOE_BWD_MAX_EXPERTS = 128;

}
