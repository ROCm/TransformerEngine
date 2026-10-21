/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/
#pragma once
//#include "hip/hip_runtime.h" // prevent hipification of this rocm_ file

// gfx1250 Tensor Data Mover (TDM): asynchronous 2D tile copies between global memory and LDS.
// Descriptors are packed here rather than with hip/amd_detail/amd_gfx1250_TDM.h, whose stride
// setters truncate (ROCm/rocm-systems#13274).

#include <cstddef>
#include <cstdint>

#if defined(__gfx1250__) && __has_builtin(__builtin_amdgcn_tensor_load_to_lds) && \
    __has_builtin(__builtin_amdgcn_tensor_store_from_lds) &&                       \
    __has_builtin(__builtin_amdgcn_s_wait_tensorcnt)
#define NVTE_ROCM_HAS_TDM 1
#endif

#ifdef NVTE_ROCM_HAS_TDM
namespace transformer_engine {
namespace tdm {
namespace detail {

// The builtins' operand signedness has changed between compiler versions; integer vectors of
// either signedness convert implicitly.
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
typedef uint32_t u32x8 __attribute__((ext_vector_type(8)));

template <bool STORE, typename T, size_t TILE_ROWS, size_t TILE_COLS>
__device__ __forceinline__ void copy_2d(const T (&lds)[TILE_ROWS][TILE_COLS], const T *gmem,
                                        uint32_t rows, uint32_t cols, uint64_t ld, uint32_t row,
                                        uint32_t col) {
  static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8,
                "TDM elements are 1, 2, 4 or 8 bytes");
  static_assert(TILE_ROWS > 0 && TILE_ROWS <= 0xFFFF && TILE_COLS > 0 && TILE_COLS <= 0xFFFF,
                "TDM tile dimensions are 16-bit");
  constexpr uint32_t data_size = sizeof(T) == 1 ? 0 : sizeof(T) == 2 ? 1 : sizeof(T) == 4 ? 2 : 3;

  // The descriptor addresses the tile origin and the tensor extents remaining from there, so the
  // TDM zero-fills (load) or skips (store) everything outside the tensor.
  const uint64_t addr = reinterpret_cast<uintptr_t>(gmem) + (row * ld + col) * sizeof(T);
  const uint32_t dim0 = col < cols ? cols - col : 0;
  const uint32_t dim1 = row < rows ? rows - row : 0;

  u32x4 g0;
  g0[0] = 1;                                                         // count
  g0[1] = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds));  // low bits: LDS offset
  g0[2] = static_cast<uint32_t>(addr);
  g0[3] = (static_cast<uint32_t>(addr >> 32) & 0x1FFFFFF) | (2u << 30);  // type = 2

  u32x8 g1;
  g1[0] = data_size << 16;
  g1[1] = (dim0 & 0xFFFF) << 16;
  g1[2] = (dim0 >> 16) | ((dim1 & 0xFFFF) << 16);
  g1[3] = (dim1 >> 16) | (static_cast<uint32_t>(TILE_COLS) << 16);
  g1[4] = static_cast<uint32_t>(TILE_ROWS);
  g1[5] = static_cast<uint32_t>(ld);
  g1[6] = static_cast<uint32_t>(ld >> 32) & 0xFFFF;
  g1[7] = 0;

  // Descriptors are scalar operands.
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    g0[i] = __builtin_amdgcn_readfirstlane(g0[i]);
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    g1[i] = __builtin_amdgcn_readfirstlane(g1[i]);
  }
  const u32x4 z4 = {0, 0, 0, 0};
  const u32x8 z8 = {0, 0, 0, 0, 0, 0, 0, 0};

  // The TDM does not order against this wave's outstanding LDS accesses.
  asm volatile("s_wait_dscnt 0x0" ::: "memory");
  if constexpr (STORE) {
    __builtin_amdgcn_tensor_store_from_lds(g0, g1, z4, z4, z8, 0);
  } else {
    __builtin_amdgcn_tensor_load_to_lds(g0, g1, z4, z4, z8, 0);
  }
}

}  // namespace detail

// Starts copying the tile at (row, col) of a rows x cols row-major tensor with a row pitch of ld
// elements into the dense LDS array lds; elements outside the tensor read as zero. Arguments must
// be wave-uniform, and every calling wave issues its own copy. Complete it with wait().
template <typename T, size_t TILE_ROWS, size_t TILE_COLS>
__device__ __forceinline__ void load_2d(T (&lds)[TILE_ROWS][TILE_COLS], const T *gmem,
                                        uint32_t rows, uint32_t cols, uint64_t ld, uint32_t row,
                                        uint32_t col) {
  detail::copy_2d<false>(lds, gmem, rows, cols, ld, row, col);
}

// Inverse of load_2d: elements outside the tensor are not written. lds must stay unmodified
// until wait() returns.
template <typename T, size_t TILE_ROWS, size_t TILE_COLS>
__device__ __forceinline__ void store_2d(T *gmem, const T (&lds)[TILE_ROWS][TILE_COLS],
                                         uint32_t rows, uint32_t cols, uint64_t ld, uint32_t row,
                                         uint32_t col) {
  detail::copy_2d<true>(lds, gmem, rows, cols, ld, row, col);
}

// Waits until at most N copies issued by this wave are in flight. Other waves of the block see
// the loaded data only after a subsequent barrier.
template <int N>
__device__ __forceinline__ void wait() {
  __builtin_amdgcn_s_wait_tensorcnt(N);
  asm volatile("" ::: "memory");  // the compiler does not see the TDM's LDS writes
}

}  // namespace tdm
}  // namespace transformer_engine
#endif  // NVTE_ROCM_HAS_TDM
