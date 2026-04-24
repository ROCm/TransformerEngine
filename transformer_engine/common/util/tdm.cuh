/*************************************************************************
 * Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

/*! \file tdm.cuh
 *  \brief TDM (Tensor Data Mover) wrappers for gfx1250.
 *
 *  AMD's TDM is the gfx1250 equivalent of NVIDIA's TMA (Tensor Memory
 *  Accelerator). It provides asynchronous bulk copies between global memory
 *  and LDS (shared memory) using hardware descriptors.
 *
 *  Key differences from TMA:
 *   - Descriptors are constructed on-device (no host-side CUtensorMap).
 *   - The instruction is wave-level (EXEC mask ignored).
 *   - Synchronization uses TENSORcnt + s_wait_tensorcnt (not mbarrier).
 *   - A single counter tracks both loads and stores in issue order.
 */

#ifndef TRANSFORMER_ENGINE_TDM_CUH_
#define TRANSFORMER_ENGINE_TDM_CUH_

#ifdef __HIP_PLATFORM_AMD__

#if defined(__gfx1250__)
#include <hip/amd_detail/amd_gfx1250_TDM.h>
#endif

namespace transformer_engine {
namespace tdm {

#if defined(__gfx1250__)

// ---------------------------------------------------------------------------
// Data type mapping
// ---------------------------------------------------------------------------

//! Returns log2(sizeof(element_in_bytes)) for the TDM dataSize field.
//! 8-bit -> 0, 16-bit -> 1, 32-bit -> 2, 64-bit -> 3.
//! For sub-byte types (NVFP4), treat as packed uint8 and pass 0.
__device__ __forceinline__ constexpr uint32_t get_data_size_from_bits(size_t type_num_bits) {
  // type_num_bits: 8 -> 0, 16 -> 1, 32 -> 2, 64 -> 3
  return (type_num_bits <= 8) ? 0 : (type_num_bits <= 16) ? 1 : (type_num_bits <= 32) ? 2 : 3;
}

// ---------------------------------------------------------------------------
// Wave guard
// ---------------------------------------------------------------------------

//! Returns true for threads in the first wavefront (wave 0) of the block.
//! TDM instructions are wave-level -- only wave 0 should issue them.
__device__ __forceinline__ bool is_tdm_wave() {
  const int linear_tid = threadIdx.x + threadIdx.y * blockDim.x;
  return (linear_tid < 32);
}

// ---------------------------------------------------------------------------
// Core 2D load: global memory -> LDS
// ---------------------------------------------------------------------------

//! Set up a 2D D# descriptor (groups 0+1) and issue a TDM load.
//!
//! @param global_base      Raw device pointer to tensor base.
//! @param lds_byte_offset  LDS destination byte offset (from shared ptr cast).
//! @param tensor_w         Full tensor width in elements.
//! @param tensor_h         Full tensor height in elements.
//! @param tile_dim_x       Tile width to load (elements, inner/columns).
//! @param tile_dim_y       Tile height to load (elements, outer/rows).
//! @param stride_elements  Row stride in elements.
//! @param data_size        log2(sizeof(element)): 0=1B, 1=2B, 2=4B, 3=8B.
//! @param tile_col         Tile start column offset in elements.
//! @param tile_row         Tile start row offset in elements.
__device__ __forceinline__
void load_2d_to_lds(const void* global_base,
                    uint32_t lds_byte_offset,
                    uint32_t tensor_w,
                    uint32_t tensor_h,
                    uint32_t tile_dim_x,
                    uint32_t tile_dim_y,
                    uint32_t stride_elements,
                    uint32_t data_size,
                    uint32_t tile_col,
                    uint32_t tile_row) {
  gfx1250_TDM_GROUP0 g0;
  gfx1250_TDM_GROUP1 g1;

  g0.ldsAddr(lds_byte_offset);

  // Compute global address of the tile's top-left element.
  const size_t elem_bytes = 1u << data_size;
  const char* base = reinterpret_cast<const char*>(global_base);
  const char* tile_start = base +
      (static_cast<size_t>(tile_row) * stride_elements + tile_col) * elem_bytes;
  g0.globalAddr(reinterpret_cast<uintptr_t>(tile_start));

  g1.dataSize(data_size);
  // Clamp remaining extent to avoid uint32_t underflow when a prefetch tile origin
  // falls past the tensor boundary (e.g. last block in a non-tile-aligned dimension).
  g1.tensorDim0(tile_col < tensor_w ? tensor_w - tile_col : 0u);
  g1.tensorDim1(tile_row < tensor_h ? tensor_h - tile_row : 0u);
  g1.tileDim0(tile_dim_x);
  g1.tileDim1(tile_dim_y);
  g1.tensorDim0Stride(stride_elements);

  __builtin_amdgcn_tensor_load_to_lds_d2(g0.m_bitfield, g1.m_bitfield, /*cachepolicy=*/0);
}

// ---------------------------------------------------------------------------
// Core 2D store: LDS -> global memory
// ---------------------------------------------------------------------------

//! Set up a 2D D# descriptor and issue a TDM store.
//! Parameters mirror load_2d_to_lds but direction is LDS->global.
__device__ __forceinline__
void store_2d_from_lds(void* global_base,
                       uint32_t lds_byte_offset,
                       uint32_t tensor_w,
                       uint32_t tensor_h,
                       uint32_t tile_dim_x,
                       uint32_t tile_dim_y,
                       uint32_t stride_elements,
                       uint32_t data_size,
                       uint32_t tile_col,
                       uint32_t tile_row) {
  gfx1250_TDM_GROUP0 g0;
  gfx1250_TDM_GROUP1 g1;

  g0.ldsAddr(lds_byte_offset);

  const size_t elem_bytes = 1u << data_size;
  char* base = reinterpret_cast<char*>(global_base);
  char* tile_start = base +
      (static_cast<size_t>(tile_row) * stride_elements + tile_col) * elem_bytes;
  g0.globalAddr(reinterpret_cast<uintptr_t>(tile_start));

  g1.dataSize(data_size);
  g1.tensorDim0(tile_col < tensor_w ? tensor_w - tile_col : 0u);
  g1.tensorDim1(tile_row < tensor_h ? tensor_h - tile_row : 0u);
  g1.tileDim0(tile_dim_x);
  g1.tileDim1(tile_dim_y);
  g1.tensorDim0Stride(stride_elements);

  __builtin_amdgcn_tensor_store_from_lds_d2(g0.m_bitfield, g1.m_bitfield, /*cachepolicy=*/0);
}

// ---------------------------------------------------------------------------
// Wait helpers (argument must be compile-time immediate)
// ---------------------------------------------------------------------------

__device__ __forceinline__ void wait_tensorcnt_0() {
  __builtin_amdgcn_s_wait_tensorcnt(0);
}

__device__ __forceinline__ void wait_tensorcnt_1() {
  __builtin_amdgcn_s_wait_tensorcnt(1);
}

__device__ __forceinline__ void wait_tensorcnt_2() {
  __builtin_amdgcn_s_wait_tensorcnt(2);
}

__device__ __forceinline__ void wait_tensorcnt_3() {
  __builtin_amdgcn_s_wait_tensorcnt(3);
}

__device__ __forceinline__ void wait_tensorcnt_4() {
  __builtin_amdgcn_s_wait_tensorcnt(4);
}

// ---------------------------------------------------------------------------
// Higher-level helpers (matching ptx.cuh copy_2d_to_shared interface)
// ---------------------------------------------------------------------------
// These handle the is_tdm_wave() guard internally.
// The caller is responsible for __syncthreads() AFTER calling these,
// matching the TMA pattern where mbarrier_wait + syncthreads follows.

//! Load a single 2D tile from global to shared via TDM.
//! Only wave 0 issues the instruction; other waves are no-ops.
//!
//! @param lds_dst       Shared memory destination pointer.
//! @param global_base   Raw device pointer to tensor base.
//! @param chunk_x       Tile column offset (elements).
//! @param chunk_y       Tile row offset (elements).
//! @param tile_dim_x    Tile width (elements).
//! @param tile_dim_y    Tile height (elements).
//! @param tensor_w      Full tensor width (elements).
//! @param tensor_h      Full tensor height (elements).
//! @param stride        Row stride (elements).
//! @param data_size     log2(sizeof(element)).
__device__ __forceinline__
void copy_2d_to_shared(void* lds_dst,
                       const void* global_base,
                       uint32_t chunk_x,
                       uint32_t chunk_y,
                       uint32_t tile_dim_x,
                       uint32_t tile_dim_y,
                       uint32_t tensor_w,
                       uint32_t tensor_h,
                       uint32_t stride,
                       uint32_t data_size) {
  if (is_tdm_wave()) {
    uint32_t lds_off = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(lds_dst));
    load_2d_to_lds(global_base, lds_off,
                   tensor_w, tensor_h,
                   tile_dim_x, tile_dim_y,
                   stride, data_size,
                   chunk_x, chunk_y);
  }
}

//! Load two 2D tiles from (possibly different) tensors into shared via TDM.
__device__ __forceinline__
void copy_2d_to_shared_x2(void* dst1, const void* src1, uint32_t cx1, uint32_t cy1,
                           void* dst2, const void* src2, uint32_t cx2, uint32_t cy2,
                           uint32_t tile_dim_x, uint32_t tile_dim_y,
                           uint32_t tensor_w, uint32_t tensor_h,
                           uint32_t stride, uint32_t data_size) {
  if (is_tdm_wave()) {
    uint32_t lds_off1 = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(dst1));
    load_2d_to_lds(src1, lds_off1,
                   tensor_w, tensor_h,
                   tile_dim_x, tile_dim_y,
                   stride, data_size,
                   cx1, cy1);

    uint32_t lds_off2 = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(dst2));
    load_2d_to_lds(src2, lds_off2,
                   tensor_w, tensor_h,
                   tile_dim_x, tile_dim_y,
                   stride, data_size,
                   cx2, cy2);
  }
}

//! Load three 2D tiles from (possibly different) tensors into shared via TDM.
__device__ __forceinline__
void copy_2d_to_shared_x3(void* dst1, const void* src1, uint32_t cx1, uint32_t cy1,
                           void* dst2, const void* src2, uint32_t cx2, uint32_t cy2,
                           void* dst3, const void* src3, uint32_t cx3, uint32_t cy3,
                           uint32_t tile_dim_x, uint32_t tile_dim_y,
                           uint32_t tensor_w, uint32_t tensor_h,
                           uint32_t stride, uint32_t data_size) {
  if (is_tdm_wave()) {
    uint32_t lds_off1 = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(dst1));
    load_2d_to_lds(src1, lds_off1,
                   tensor_w, tensor_h,
                   tile_dim_x, tile_dim_y,
                   stride, data_size,
                   cx1, cy1);

    uint32_t lds_off2 = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(dst2));
    load_2d_to_lds(src2, lds_off2,
                   tensor_w, tensor_h,
                   tile_dim_x, tile_dim_y,
                   stride, data_size,
                   cx2, cy2);

    uint32_t lds_off3 = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(dst3));
    load_2d_to_lds(src3, lds_off3,
                   tensor_w, tensor_h,
                   tile_dim_x, tile_dim_y,
                   stride, data_size,
                   cx3, cy3);
  }
}

//! Store a 2D tile from shared to global via TDM.
//! Only wave 0 issues the instruction.
//! Caller must ensure all threads have finished writing to LDS (via __syncthreads())
//! BEFORE calling this.
__device__ __forceinline__
void store_2d_to_global(const void* lds_src,
                        void* global_base,
                        uint32_t chunk_x,
                        uint32_t chunk_y,
                        uint32_t tile_dim_x,
                        uint32_t tile_dim_y,
                        uint32_t tensor_w,
                        uint32_t tensor_h,
                        uint32_t stride,
                        uint32_t data_size) {
  if (is_tdm_wave()) {
    uint32_t lds_off = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(lds_src));
    store_2d_from_lds(global_base, lds_off,
                      tensor_w, tensor_h,
                      tile_dim_x, tile_dim_y,
                      stride, data_size,
                      chunk_x, chunk_y);
  }
}

#else  // !defined(__gfx1250__)

// Stubs for non-gfx1250 AMD targets -- these should never be called.
__device__ __forceinline__ bool is_tdm_wave() { return false; }
__device__ __forceinline__ void wait_tensorcnt_0() {}
__device__ __forceinline__ void wait_tensorcnt_1() {}
__device__ __forceinline__ void wait_tensorcnt_2() {}
__device__ __forceinline__ void wait_tensorcnt_3() {}
__device__ __forceinline__ void wait_tensorcnt_4() {}

__device__ __forceinline__ constexpr uint32_t get_data_size_from_bits(size_t type_num_bits) {
  return (type_num_bits <= 8) ? 0 : (type_num_bits <= 16) ? 1 : (type_num_bits <= 32) ? 2 : 3;
}

__device__ __forceinline__
void copy_2d_to_shared(void*, const void*, uint32_t, uint32_t,
                       uint32_t, uint32_t, uint32_t, uint32_t,
                       uint32_t, uint32_t) {}

__device__ __forceinline__
void copy_2d_to_shared_x2(void*, const void*, uint32_t, uint32_t,
                           void*, const void*, uint32_t, uint32_t,
                           uint32_t, uint32_t, uint32_t, uint32_t,
                           uint32_t, uint32_t) {}

__device__ __forceinline__
void copy_2d_to_shared_x3(void*, const void*, uint32_t, uint32_t,
                           void*, const void*, uint32_t, uint32_t,
                           void*, const void*, uint32_t, uint32_t,
                           uint32_t, uint32_t, uint32_t, uint32_t,
                           uint32_t, uint32_t) {}

__device__ __forceinline__
void store_2d_to_global(const void*, void*, uint32_t, uint32_t,
                        uint32_t, uint32_t, uint32_t, uint32_t,
                        uint32_t, uint32_t) {}

#endif  // defined(__gfx1250__)

}  // namespace tdm
}  // namespace transformer_engine

#endif  // __HIP_PLATFORM_AMD__

#endif  // TRANSFORMER_ENGINE_TDM_CUH_
