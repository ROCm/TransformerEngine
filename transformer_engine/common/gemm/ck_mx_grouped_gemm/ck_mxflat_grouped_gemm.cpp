/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "../ck_gemm_common.h"

#include "ck_tile/host.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/flatmm.hpp"
#include "ck_tile/ops/flatmm/kernel/mx_flatmm_kernel.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/gemm/kernel/grouped_gemm_kernel.hpp"

namespace ck_tile {
namespace core {
namespace arch {
using TargetId = amdgcn_target_id;
}  // namespace arch
}  // namespace core
}  // namespace ck_tile

// MXFlatmm standalone uses e8m0_bexp_t for FlatmmScalePointer.
using ScaleType = ck_tile::e8m0_bexp_t;

static constexpr int ScaleBlockSize = 32;

struct MXFlatmmConfigBase16 {
  static constexpr ck_tile::index_t M_Tile = 128;
  static constexpr ck_tile::index_t N_Tile = 256;
  static constexpr ck_tile::index_t K_Tile = 256;

  static constexpr ck_tile::index_t M_Warp = 1;
  static constexpr ck_tile::index_t N_Warp = 4;
  static constexpr ck_tile::index_t K_Warp = 1;

  static constexpr ck_tile::index_t M_Warp_Tile = 16;
  static constexpr ck_tile::index_t N_Warp_Tile = 16;
  static constexpr ck_tile::index_t K_Warp_Tile = 128;

  static constexpr bool kPadM = false;
  static constexpr bool kPadN = false;
  static constexpr bool kPadK = false;

  static constexpr bool TransposeC = false;
  static constexpr bool UseStructuredSparsity = false;

  static constexpr int kBlockPerCu = 1;
  static constexpr int TilePartitionerGroupNum = 8;
  static constexpr int TilePartitionerM01 = 4;
  static constexpr auto Scheduler = ck_tile::GemmPipelineScheduler::Default;
  static constexpr ck_tile::index_t NumWaveGroups = 1;
  static constexpr bool DoubleSmemBuffer = false;

  static constexpr int N_Repeat = N_Tile / N_Warp_Tile / N_Warp;
  static constexpr bool TiledMMAPermuteN = false;
};

template <ck_tile::core::arch::TargetId Arch, typename FlatmmConfig>
struct MXFlatmmArchTraits {
  static constexpr int BlockedXDLN_PerWarp = 2;
  using Config = FlatmmConfig;

  template <typename MXPipelineProblem>
  using MXFlatmmPipeline = ck_tile::MXFlatmmPipelineAGmemBGmemCRegV1<MXPipelineProblem>;
};

using MXTraits =
    MXFlatmmArchTraits<ck_tile::core::arch::TargetId::GFX950, MXFlatmmConfigBase16>;

using ScaleA = ck_tile::FlatmmScalePointer<1, ScaleBlockSize, ScaleType>;
using ScaleB = ck_tile::FlatmmScalePointer<1, ScaleBlockSize, ScaleType>;

static size_t align_up_size(size_t x, size_t a) {
  return (x + a - 1) / a * a;
}

static ck_tile::index_t round_up_index(ck_tile::index_t x, ck_tile::index_t tile) {
  return ((x + tile - 1) / tile) * tile;
}

// -----------------------------------------------------------------------------
// Device-side MXFlatmm preshuffle kernels, copied from the working standalone.
// -----------------------------------------------------------------------------

template <typename dtype, int NLane>
__global__ void preshuffle_weight_kernel(const dtype* __restrict__ src,
                                         dtype* __restrict__ dst,
                                         int K,
                                         int N) {
  constexpr int packed_size = ck_tile::numeric_traits<dtype>::PackedSize;
  const int KPack = std::is_same_v<dtype, ck_tile::pk_fp6x16_t> ? 32 : 16 * packed_size;
  const int KLane = 64 / NLane;
  const int K0 = K / (KLane * KPack);

  const int linear = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = N * (K / packed_size);
  if (linear >= total) return;

  const int n = linear / (K / packed_size);
  const int k = (linear % (K / packed_size)) * packed_size;

  const int n0 = n / NLane;
  const int n1 = n % NLane;

  const int k0 = k / (KLane * KPack);
  const int tempk = k % (KLane * KPack);
  const int k1 = tempk / KPack;
  const int k2 = tempk % KPack;

  const int outputIndex = n0 * KPack * NLane * KLane * K0 +
                          k0 * KPack * NLane * KLane +
                          k1 * KPack * NLane + n1 * KPack + k2;

  dst[outputIndex] = src[k * N + n];
}

template <typename dtype, bool KLast, int XdlMNThread>
__global__ void preshuffle_scale_kernel(const dtype* __restrict__ src,
                                        dtype* __restrict__ dst,
                                        int MN,
                                        int K) {
  constexpr int MNXdlPack = 2;
  constexpr int KXdlPack = 2;
  constexpr int XdlKThread = 64 / XdlMNThread;

  const int MN_padded =
      ((MN + XdlMNThread * MNXdlPack - 1) / (XdlMNThread * MNXdlPack)) *
      (XdlMNThread * MNXdlPack);
  const int K0 = K / KXdlPack / XdlKThread;

  const int linear = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = MN_padded * K;
  if (linear >= total) return;

  const int n = linear / K;
  const int k = linear % K;

  const int n0 = n / (XdlMNThread * MNXdlPack);
  const int tempn = n % (XdlMNThread * MNXdlPack);
  const int n1 = tempn % XdlMNThread;
  const int n2 = tempn / XdlMNThread;

  const int k0 = k / (XdlKThread * KXdlPack);
  const int tempk = k % (XdlKThread * KXdlPack);
  const int k1 = tempk % XdlKThread;
  const int k2 = tempk / XdlKThread;

  const int outputIndex =
      n0 * MNXdlPack * KXdlPack * XdlMNThread * XdlKThread * K0 +
      k0 * MNXdlPack * KXdlPack * XdlMNThread * XdlKThread +
      k1 * MNXdlPack * KXdlPack * XdlMNThread + n1 * MNXdlPack * KXdlPack +
      k2 * MNXdlPack + n2;

  dtype value{};
  if (n < MN) {
    if constexpr (KLast) {
      value = src[n * K + k];
    } else {
      value = src[k * MN + n];
    }
  }
  dst[outputIndex] = value;
}

template <typename T>
void launch_weight_preshuffle_once(const T* src, T* dst, int K, int N, hipStream_t stream) {
  constexpr int threads = 256;
  const int total = N * (K / ck_tile::numeric_traits<T>::PackedSize);
  const int blocks = (total + threads - 1) / threads;
  hipLaunchKernelGGL((preshuffle_weight_kernel<T, 16>),
                     dim3(blocks),
                     dim3(threads),
                     0,
                     stream,
                     src,
                     dst,
                     K,
                     N);
  NVTE_CHECK_CUDA(hipGetLastError());
}

template <typename T, bool KLast>
void launch_scale_preshuffle_once(const T* src,
                                  T* dst,
                                  int MN,
                                  int K,
                                  hipStream_t stream) {
  constexpr int threads = 256;
  constexpr int XdlMNThread = 16;
  constexpr int MNXdlPack = 2;
  const int MN_padded =
      ((MN + XdlMNThread * MNXdlPack - 1) / (XdlMNThread * MNXdlPack)) *
      (XdlMNThread * MNXdlPack);
  const int total = MN_padded * K;
  const int blocks = (total + threads - 1) / threads;
  hipLaunchKernelGGL((preshuffle_scale_kernel<T, KLast, XdlMNThread>),
                     dim3(blocks),
                     dim3(threads),
                     0,
                     stream,
                     src,
                     dst,
                     MN,
                     K);
  NVTE_CHECK_CUDA(hipGetLastError());
}


// TE integration variant of the standalone preshuffle.  The standalone builds
// contiguous per-expert HostTensors before copying to device.  TE can hand us
// either [K, N] or [N, K] physical storage depending on the normalized transB
// presentation, so make the source read explicit instead of assuming src[k*N+n].
template <typename dtype, int NLane>
__global__ void preshuffle_weight_kernel_strided(const dtype* __restrict__ src,
                                                 dtype* __restrict__ dst,
                                                 int K,
                                                 int N,
                                                 int64_t src_stride0,
                                                 int64_t src_stride1,
                                                 bool src_is_nk) {
  constexpr int packed_size = ck_tile::numeric_traits<dtype>::PackedSize;
  const int KPack = std::is_same_v<dtype, ck_tile::pk_fp6x16_t> ? 32 : 16 * packed_size;
  const int KLane = 64 / NLane;
  const int K0 = K / (KLane * KPack);

  const int linear = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = N * (K / packed_size);
  if (linear >= total) return;

  const int n = linear / (K / packed_size);
  const int k = (linear % (K / packed_size)) * packed_size;

  const int n0 = n / NLane;
  const int n1 = n % NLane;

  const int k0 = k / (KLane * KPack);
  const int tempk = k % (KLane * KPack);
  const int k1 = tempk / KPack;
  const int k2 = tempk % KPack;

  const int outputIndex = n0 * KPack * NLane * KLane * K0 +
                          k0 * KPack * NLane * KLane +
                          k1 * KPack * NLane + n1 * KPack + k2;

  // MXFlatmm wants logical B as [K, N].  If TE gave us transposed physical
  // storage [N, K], read src[n, k]; otherwise read src[k, n].
  dst[outputIndex] = src_is_nk ? src[n * src_stride0 + k * src_stride1]
                               : src[k * src_stride0 + n * src_stride1];
}

template <typename T>
void launch_weight_preshuffle_strided_once(const T* src,
                                           T* dst,
                                           int K,
                                           int N,
                                           int64_t src_stride0,
                                           int64_t src_stride1,
                                           bool src_is_nk,
                                           hipStream_t stream) {
  constexpr int threads = 256;
  const int total = N * (K / ck_tile::numeric_traits<T>::PackedSize);
  const int blocks = (total + threads - 1) / threads;
  hipLaunchKernelGGL((preshuffle_weight_kernel_strided<T, 16>),
                     dim3(blocks),
                     dim3(threads),
                     0,
                     stream,
                     src,
                     dst,
                     K,
                     N,
                     src_stride0,
                     src_stride1,
                     src_is_nk);
  NVTE_CHECK_CUDA(hipGetLastError());
}

template <typename dtype, bool KLast, int XdlMNThread>
__global__ void preshuffle_scale_kernel_strided(const dtype* __restrict__ src,
                                                dtype* __restrict__ dst,
                                                int MN_src,
                                                int MN_dst,
                                                int K,
                                                int64_t src_stride0,
                                                int64_t src_stride1) {
  constexpr int MNXdlPack = 2;
  constexpr int KXdlPack = 2;
  constexpr int XdlKThread = 64 / XdlMNThread;

  const int MN_padded =
      ((MN_dst + XdlMNThread * MNXdlPack - 1) / (XdlMNThread * MNXdlPack)) *
      (XdlMNThread * MNXdlPack);
  const int K0 = K / KXdlPack / XdlKThread;

  const int linear = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = MN_padded * K;
  if (linear >= total) return;

  const int n = linear / K;
  const int k = linear % K;

  const int n0 = n / (XdlMNThread * MNXdlPack);
  const int tempn = n % (XdlMNThread * MNXdlPack);
  const int n1 = tempn % XdlMNThread;
  const int n2 = tempn / XdlMNThread;

  const int k0 = k / (XdlKThread * KXdlPack);
  const int tempk = k % (XdlKThread * KXdlPack);
  const int k1 = tempk % XdlKThread;
  const int k2 = tempk / XdlKThread;

  const int outputIndex =
      n0 * MNXdlPack * KXdlPack * XdlMNThread * XdlKThread * K0 +
      k0 * MNXdlPack * KXdlPack * XdlMNThread * XdlKThread +
      k1 * MNXdlPack * KXdlPack * XdlMNThread + n1 * MNXdlPack * KXdlPack +
      k2 * MNXdlPack + n2;

  dtype value{};
  if (n < MN_src) {
    if constexpr (KLast) {
      value = src[n * src_stride0 + k * src_stride1];
    } else {
      value = src[k * src_stride0 + n * src_stride1];
    }
  }
  dst[outputIndex] = value;
}

template <typename T, bool KLast>
void launch_scale_preshuffle_strided_once(const T* src,
                                          T* dst,
                                          int MN_src,
                                          int MN_dst,
                                          int K,
                                          int64_t src_stride0,
                                          int64_t src_stride1,
                                          hipStream_t stream) {
  constexpr int threads = 256;
  constexpr int XdlMNThread = 16;
  constexpr int MNXdlPack = 2;
  const int MN_padded =
      ((MN_dst + XdlMNThread * MNXdlPack - 1) / (XdlMNThread * MNXdlPack)) *
      (XdlMNThread * MNXdlPack);
  const int total = MN_padded * K;
  const int blocks = (total + threads - 1) / threads;
  hipLaunchKernelGGL((preshuffle_scale_kernel_strided<T, KLast, XdlMNThread>),
                     dim3(blocks),
                     dim3(threads),
                     0,
                     stream,
                     src,
                     dst,
                     MN_src,
                     MN_dst,
                     K,
                     src_stride0,
                     src_stride1);
  NVTE_CHECK_CUDA(hipGetLastError());
}

template <typename T>
__global__ void stage_a_rowmajor_kernel(const T* __restrict__ src,
                                        T* __restrict__ dst,
                                        int M,
                                        int M_padded,
                                        int K,
                                        int64_t src_stride0,
                                        int64_t src_stride1,
                                        bool src_is_km) {
  const int linear = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = M_padded * K;
  if (linear >= total) return;

  const int m = linear / K;
  const int k = linear % K;

  T value{};
  if (m < M) {
    value = src_is_km ? src[k * src_stride0 + m * src_stride1]
                      : src[m * src_stride0 + k * src_stride1];
  }
  dst[m * K + k] = value;
}

template <typename T>
void launch_stage_a_rowmajor_once(const T* src,
                                  T* dst,
                                  int M,
                                  int M_padded,
                                  int K,
                                  int64_t src_stride0,
                                  int64_t src_stride1,
                                  bool src_is_km,
                                  hipStream_t stream) {
  constexpr int threads = 256;
  const int total = M_padded * K;
  const int blocks = (total + threads - 1) / threads;
  hipLaunchKernelGGL((stage_a_rowmajor_kernel<T>),
                     dim3(blocks),
                     dim3(threads),
                     0,
                     stream,
                     src,
                     dst,
                     M,
                     M_padded,
                     K,
                     src_stride0,
                     src_stride1,
                     src_is_km);
  NVTE_CHECK_CUDA(hipGetLastError());
}

template <typename T>
__global__ void copy_c_unpad_kernel(const T* __restrict__ src,
                                    T* __restrict__ dst,
                                    int M,
                                    int N,
                                    int64_t dst_stride) {
  const int linear = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = M * N;
  if (linear >= total) return;

  const int m = linear / N;
  const int n = linear % N;
  dst[m * dst_stride + n] = src[m * N + n];
}

template <typename T>
void launch_copy_c_unpad_once(const T* src,
                              T* dst,
                              int M,
                              int N,
                              int64_t dst_stride,
                              hipStream_t stream) {
  constexpr int threads = 256;
  const int total = M * N;
  const int blocks = (total + threads - 1) / threads;
  hipLaunchKernelGGL((copy_c_unpad_kernel<T>),
                     dim3(blocks),
                     dim3(threads),
                     0,
                     stream,
                     src,
                     dst,
                     M,
                     N,
                     dst_stride);
  NVTE_CHECK_CUDA(hipGetLastError());
}


// -----------------------------------------------------------------------------
// Grouped MXFlatmm preprocessing kernels.
//
// The original TE integration launched staging/preshuffle kernels once per expert:
//   E x A staging + E x B preshuffle + E x A-scale preshuffle + E x B-scale
// preshuffle (+ optional E x C copyback).  These descriptor-driven wrappers keep
// the exact same per-element transforms, but launch one kernel per transform type
// across all experts.
// -----------------------------------------------------------------------------

template <typename T>
struct GroupedStageADesc {
  const T* src;
  T* dst;
  int M;
  int M_padded;
  int K;
  int64_t src_stride0;
  int64_t src_stride1;
  bool src_is_km;
  int blocks;
};

template <typename T>
__global__ void grouped_stage_a_rowmajor_kernel(const GroupedStageADesc<T>* __restrict__ descs,
                                                int group_count) {
  constexpr int threads = 256;
  const int g = static_cast<int>(blockIdx.x);
  if (g >= group_count) return;

  const auto desc = descs[g];
  const int block_linear = static_cast<int>(blockIdx.y);
  if (block_linear >= desc.blocks) return;

  const int linear = block_linear * threads + static_cast<int>(threadIdx.x);
  const int total = desc.M_padded * desc.K;
  if (linear >= total) return;

  const int m = linear / desc.K;
  const int k = linear % desc.K;

  T value{};
  if (m < desc.M) {
    value = desc.src_is_km ? desc.src[k * desc.src_stride0 + m * desc.src_stride1]
                           : desc.src[m * desc.src_stride0 + k * desc.src_stride1];
  }
  desc.dst[m * desc.K + k] = value;
}

template <typename T>
void launch_grouped_stage_a_rowmajor(const std::vector<GroupedStageADesc<T>>& descs_host,
                                     GroupedStageADesc<T>* descs_dev,
                                     hipStream_t stream) {
  if (descs_host.empty()) return;

  int max_blocks = 0;
  for (const auto& desc : descs_host) max_blocks = std::max(max_blocks, desc.blocks);
  if (max_blocks <= 0) return;

  NVTE_CHECK_CUDA(hipMemcpyAsync(descs_dev,
                                 descs_host.data(),
                                 descs_host.size() * sizeof(GroupedStageADesc<T>),
                                 hipMemcpyHostToDevice,
                                 stream));

  hipLaunchKernelGGL((grouped_stage_a_rowmajor_kernel<T>),
                     dim3(static_cast<unsigned int>(descs_host.size()),
                          static_cast<unsigned int>(max_blocks),
                          1),
                     dim3(256),
                     0,
                     stream,
                     descs_dev,
                     static_cast<int>(descs_host.size()));
  NVTE_CHECK_CUDA(hipGetLastError());
}

template <typename T>
struct GroupedWeightPreshuffleDesc {
  const T* src;
  T* dst;
  int K;
  int N;
  int64_t src_stride0;
  int64_t src_stride1;
  bool src_is_nk;
  int blocks;
};

template <typename T, int NLane>
__global__ void grouped_preshuffle_weight_kernel(const GroupedWeightPreshuffleDesc<T>* __restrict__ descs,
                                                 int group_count) {
  constexpr int threads = 256;
  constexpr int packed_size = ck_tile::numeric_traits<T>::PackedSize;
  constexpr int KPack = std::is_same_v<T, ck_tile::pk_fp6x16_t> ? 32 : 16 * packed_size;
  constexpr int KLane = 64 / NLane;

  const int g = static_cast<int>(blockIdx.x);
  if (g >= group_count) return;

  const auto desc = descs[g];
  const int block_linear = static_cast<int>(blockIdx.y);
  if (block_linear >= desc.blocks) return;

  const int linear = block_linear * threads + static_cast<int>(threadIdx.x);
  const int total = desc.N * (desc.K / packed_size);
  if (linear >= total) return;

  const int K0 = desc.K / (KLane * KPack);
  const int n = linear / (desc.K / packed_size);
  const int k = (linear % (desc.K / packed_size)) * packed_size;

  const int n0 = n / NLane;
  const int n1 = n % NLane;

  const int k0 = k / (KLane * KPack);
  const int tempk = k % (KLane * KPack);
  const int k1 = tempk / KPack;
  const int k2 = tempk % KPack;

  const int outputIndex = n0 * KPack * NLane * KLane * K0 +
                          k0 * KPack * NLane * KLane +
                          k1 * KPack * NLane + n1 * KPack + k2;

  desc.dst[outputIndex] = desc.src_is_nk ? desc.src[n * desc.src_stride0 + k * desc.src_stride1]
                                         : desc.src[k * desc.src_stride0 + n * desc.src_stride1];
}

template <typename T>
void launch_grouped_weight_preshuffle(const std::vector<GroupedWeightPreshuffleDesc<T>>& descs_host,
                                      GroupedWeightPreshuffleDesc<T>* descs_dev,
                                      hipStream_t stream) {
  if (descs_host.empty()) return;

  int max_blocks = 0;
  for (const auto& desc : descs_host) max_blocks = std::max(max_blocks, desc.blocks);
  if (max_blocks <= 0) return;

  NVTE_CHECK_CUDA(hipMemcpyAsync(descs_dev,
                                 descs_host.data(),
                                 descs_host.size() * sizeof(GroupedWeightPreshuffleDesc<T>),
                                 hipMemcpyHostToDevice,
                                 stream));

  hipLaunchKernelGGL((grouped_preshuffle_weight_kernel<T, 16>),
                     dim3(static_cast<unsigned int>(descs_host.size()),
                          static_cast<unsigned int>(max_blocks),
                          1),
                     dim3(256),
                     0,
                     stream,
                     descs_dev,
                     static_cast<int>(descs_host.size()));
  NVTE_CHECK_CUDA(hipGetLastError());
}

struct GroupedScalePreshuffleDesc {
  const ScaleType* src;
  ScaleType* dst;
  int MN_src;
  int MN_dst;
  int K;
  int64_t src_stride0;
  int64_t src_stride1;
  bool k_last;
  int blocks;
};

template <int XdlMNThread>
__global__ void grouped_preshuffle_scale_kernel(const GroupedScalePreshuffleDesc* __restrict__ descs,
                                                int group_count) {
  constexpr int threads = 256;
  constexpr int MNXdlPack = 2;
  constexpr int KXdlPack = 2;
  constexpr int XdlKThread = 64 / XdlMNThread;

  const int g = static_cast<int>(blockIdx.x);
  if (g >= group_count) return;

  const auto desc = descs[g];
  const int block_linear = static_cast<int>(blockIdx.y);
  if (block_linear >= desc.blocks) return;

  const int linear = block_linear * threads + static_cast<int>(threadIdx.x);
  const int MN_padded = ((desc.MN_dst + XdlMNThread * MNXdlPack - 1) /
                         (XdlMNThread * MNXdlPack)) *
                        (XdlMNThread * MNXdlPack);
  const int total = MN_padded * desc.K;
  if (linear >= total) return;

  const int n = linear / desc.K;
  const int k = linear % desc.K;

  const int n0 = n / (XdlMNThread * MNXdlPack);
  const int tempn = n % (XdlMNThread * MNXdlPack);
  const int n1 = tempn % XdlMNThread;
  const int n2 = tempn / XdlMNThread;

  const int k0 = k / (XdlKThread * KXdlPack);
  const int tempk = k % (XdlKThread * KXdlPack);
  const int k1 = tempk % XdlKThread;
  const int k2 = tempk / XdlKThread;

  const int K0 = desc.K / KXdlPack / XdlKThread;
  const int outputIndex =
      n0 * MNXdlPack * KXdlPack * XdlMNThread * XdlKThread * K0 +
      k0 * MNXdlPack * KXdlPack * XdlMNThread * XdlKThread +
      k1 * MNXdlPack * KXdlPack * XdlMNThread + n1 * MNXdlPack * KXdlPack +
      k2 * MNXdlPack + n2;

  ScaleType value{};
  if (n < desc.MN_src) {
    value = desc.k_last ? desc.src[n * desc.src_stride0 + k * desc.src_stride1]
                        : desc.src[k * desc.src_stride0 + n * desc.src_stride1];
  }
  desc.dst[outputIndex] = value;
}

inline void launch_grouped_scale_preshuffle(const std::vector<GroupedScalePreshuffleDesc>& descs_host,
                                            GroupedScalePreshuffleDesc* descs_dev,
                                            hipStream_t stream) {
  if (descs_host.empty()) return;

  int max_blocks = 0;
  for (const auto& desc : descs_host) max_blocks = std::max(max_blocks, desc.blocks);
  if (max_blocks <= 0) return;

  NVTE_CHECK_CUDA(hipMemcpyAsync(descs_dev,
                                 descs_host.data(),
                                 descs_host.size() * sizeof(GroupedScalePreshuffleDesc),
                                 hipMemcpyHostToDevice,
                                 stream));

  hipLaunchKernelGGL((grouped_preshuffle_scale_kernel<16>),
                     dim3(static_cast<unsigned int>(descs_host.size()),
                          static_cast<unsigned int>(max_blocks),
                          1),
                     dim3(256),
                     0,
                     stream,
                     descs_dev,
                     static_cast<int>(descs_host.size()));
  NVTE_CHECK_CUDA(hipGetLastError());
}

template <typename T>
struct GroupedCopyCDesc {
  const T* src;
  T* dst;
  int M;
  int N;
  int64_t dst_stride;
  int blocks;
};

template <typename T>
__global__ void grouped_copy_c_unpad_kernel(const GroupedCopyCDesc<T>* __restrict__ descs,
                                            int group_count) {
  constexpr int threads = 256;
  const int g = static_cast<int>(blockIdx.x);
  if (g >= group_count) return;

  const auto desc = descs[g];
  const int block_linear = static_cast<int>(blockIdx.y);
  if (block_linear >= desc.blocks) return;

  const int linear = block_linear * threads + static_cast<int>(threadIdx.x);
  const int total = desc.M * desc.N;
  if (linear >= total) return;

  const int m = linear / desc.N;
  const int n = linear % desc.N;
  desc.dst[m * desc.dst_stride + n] = desc.src[m * desc.N + n];
}

template <typename T>
void launch_grouped_copy_c_unpad(const std::vector<GroupedCopyCDesc<T>>& descs_host,
                                 GroupedCopyCDesc<T>* descs_dev,
                                 hipStream_t stream) {
  if (descs_host.empty()) return;

  int max_blocks = 0;
  for (const auto& desc : descs_host) max_blocks = std::max(max_blocks, desc.blocks);
  if (max_blocks <= 0) return;

  NVTE_CHECK_CUDA(hipMemcpyAsync(descs_dev,
                                 descs_host.data(),
                                 descs_host.size() * sizeof(GroupedCopyCDesc<T>),
                                 hipMemcpyHostToDevice,
                                 stream));

  hipLaunchKernelGGL((grouped_copy_c_unpad_kernel<T>),
                     dim3(static_cast<unsigned int>(descs_host.size()),
                          static_cast<unsigned int>(max_blocks),
                          1),
                     dim3(256),
                     0,
                     stream,
                     descs_dev,
                     static_cast<int>(descs_host.size()));
  NVTE_CHECK_CUDA(hipGetLastError());
}

namespace ck_tile {

template <class ScaleM = FlatmmScalePointer<-1>,
          class ScaleN = FlatmmScalePointer<-1>,
          index_t NumDTensor = 0>
struct GroupedMXFlatmmHostArgs {
  CK_TILE_HOST explicit GroupedMXFlatmmHostArgs(
      const void* a_ptr_,
      const void* b_ptr_,
      const std::array<const void*, NumDTensor>& ds_ptr_,
      void* e_ptr_,
      index_t k_batch_,
      index_t M_,
      index_t N_,
      index_t K_,
      index_t stride_A_,
      index_t stride_B_,
      const std::array<index_t, NumDTensor>& stride_Ds_,
      index_t stride_E_,
      ScaleM scale_m_ = ScaleM{},
      ScaleN scale_n_ = ScaleN{})
      : a_ptr(a_ptr_),
        b_ptr(b_ptr_),
        ds_ptr(ds_ptr_),
        e_ptr(e_ptr_),
        k_batch(k_batch_),
        M(M_),
        N(N_),
        K(K_),
        stride_A(stride_A_),
        stride_B(stride_B_),
        stride_Ds(stride_Ds_),
        stride_E(stride_E_),
        scale_m(scale_m_),
        scale_n(scale_n_) {}

  const void* a_ptr;
  const void* b_ptr;
  const std::array<const void*, NumDTensor> ds_ptr{};
  void* e_ptr;
  index_t k_batch;
  index_t M;
  index_t N;
  index_t K;
  index_t stride_A;
  index_t stride_B;
  const std::array<index_t, NumDTensor> stride_Ds{};
  index_t stride_E;
  ScaleM scale_m;
  ScaleN scale_n;
};

template <class ScaleM = FlatmmScalePointer<-1>,
          class ScaleN = FlatmmScalePointer<-1>,
          index_t NumDTensor = 0>
struct FlatmmTransKernelArg {
  FlatmmKernelArgs<ScaleM, ScaleN, NumDTensor> group_karg;
  index_t block_start;
  index_t block_end;

  FlatmmTransKernelArg() = delete;

  FlatmmTransKernelArg(FlatmmKernelArgs<ScaleM, ScaleN, NumDTensor>&& karg,
                       index_t bl_start,
                       index_t bl_end)
      : group_karg{std::move(karg)}, block_start{bl_start}, block_end{bl_end} {}
};

template <typename TilePartitioner_, typename MXFlatmmPipeline_, typename EpiloguePipeline_>
struct GroupedMXFlatmmKernel {
  using Base = MXFlatmmKernel<TilePartitioner_, MXFlatmmPipeline_, EpiloguePipeline_>;

  using TilePartitioner = remove_cvref_t<TilePartitioner_>;
  using MXFlatmmPipeline = remove_cvref_t<MXFlatmmPipeline_>;
  using EpiloguePipeline = remove_cvref_t<EpiloguePipeline_>;
  using DsDataType = remove_cvref_t<typename EpiloguePipeline::DsDataType>;
  static constexpr index_t NumDTensor_ = DsDataType::size();

  static constexpr index_t kBlockSize = MXFlatmmPipeline::BlockSize;
  static constexpr bool UsePersistentKernel = false;

  CK_TILE_HOST static auto BlockSize() -> dim3 {
    if (is_wave32()) return dim3(kBlockSize / 2);
    return dim3(kBlockSize);
  }

  template <class ScaleM, class ScaleN>
  CK_TILE_HOST static auto GridSize(
      const std::vector<GroupedMXFlatmmHostArgs<ScaleM, ScaleN, NumDTensor_>>& gemm_descs)
      -> dim3 {
    index_t grid_size = 0;
    for (const auto& it_desc : gemm_descs) {
      grid_size += TilePartitioner::GridSize(it_desc.M, it_desc.N) * it_desc.k_batch;
    }
    return dim3(grid_size, 1, 1);
  }

  template <class ScaleM, class ScaleN>
  CK_TILE_HOST static auto MakeKargs(
      const std::vector<GroupedMXFlatmmHostArgs<ScaleM, ScaleN, NumDTensor_>>& gemm_descs)
      -> std::vector<FlatmmTransKernelArg<ScaleM, ScaleN, NumDTensor_>> {
    std::vector<FlatmmTransKernelArg<ScaleM, ScaleN, NumDTensor_>> out;
    out.reserve(gemm_descs.size());

    index_t grid_size = 0;
    for (const auto& desc : gemm_descs) {
      const index_t M = desc.M;
      const index_t N = desc.N;
      const index_t K = desc.K;
      if (M == 0 || N == 0 || K == 0) continue;

      const index_t grid_size_grp = TilePartitioner::GridSize(M, N) * desc.k_batch;
      const index_t block_start = grid_size;
      const index_t block_end = grid_size + grid_size_grp;
      grid_size += grid_size_grp;

      auto karg = FlatmmKernelArgs<ScaleM, ScaleN, NumDTensor_>{
          desc.a_ptr,
          desc.b_ptr,
          desc.ds_ptr,
          desc.e_ptr,
          desc.M,
          desc.N,
          desc.K,
          desc.stride_A,
          desc.stride_B,
          desc.stride_Ds,
          desc.stride_E,
          desc.k_batch,
          desc.scale_m,
          desc.scale_n};
      out.emplace_back(std::move(karg), block_start, block_end);
    }
    return out;
  }

  template <class ScaleM, class ScaleN>
  CK_TILE_HOST static bool IsSupportedArgument(
      const std::vector<FlatmmTransKernelArg<ScaleM, ScaleN, NumDTensor_>>& kargs) {
    for (const auto& karg : kargs) {
      if (!Base::IsSupportedArgument(karg.group_karg)) return false;
    }
    return true;
  }

  template <class ScaleM, class ScaleN>
  CK_TILE_DEVICE index_t FindGroupId(
      const FlatmmTransKernelArg<ScaleM, ScaleN, NumDTensor_>* gemm_desc_ptr,
      index_t block_id,
      index_t group_count) const {
    index_t left = 0;
    index_t right = group_count;
    index_t group_id = index_t((left + right) >> 1);

    while ((!(block_id >= gemm_desc_ptr[group_id].block_start &&
              block_id < gemm_desc_ptr[group_id].block_end)) &&
           left <= right) {
      if (block_id < gemm_desc_ptr[group_id].block_start) {
        right = group_id;
      } else {
        left = group_id;
      }
      group_id = index_t((left + right) >> 1);
    }
    return group_id;
  }

  template <bool U = UsePersistentKernel, typename = std::enable_if_t<!U>>
  CK_TILE_DEVICE void operator()(const void CK_TILE_CONSTANT_ADDRESS_SPACE* gemm_descs_const,
                                 index_t group_count) const {
    using ScaleA_ = FlatmmScalePointer<1, 32, e8m0_bexp_t>;
    using ScaleB_ = FlatmmScalePointer<1, 32, e8m0_bexp_t>;
    using TransArg = FlatmmTransKernelArg<ScaleA_, ScaleB_, NumDTensor_>;

    const index_t block_id = ck_tile::get_block_1d_id();
    const auto gemm_desc_ptr = reinterpret_cast<const TransArg*>(
        cast_pointer_to_generic_address_space(gemm_descs_const));

    const index_t group_id = FindGroupId(gemm_desc_ptr, block_id, group_count);
    const auto& kargs = gemm_desc_ptr[group_id];
    const index_t local_block_id = block_id - kargs.block_start;
    const auto grid_size_2d = TilePartitioner::GridSize(kargs.group_karg.M, kargs.group_karg.N);
    const auto partition_idx = local_block_id % grid_size_2d;
    Base{}(kargs.group_karg, partition_idx);
  }
};

}  // namespace ck_tile

namespace transformer_engine {

template <typename MXFlatmmArchTraitsT,
          typename ADataType,
          typename BDataType,
          typename DsDatatype,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename CLayout,
          typename ScaleAType,
          typename ScaleBType,
          bool UsePersistentKernel = true,
          typename CDEElementWise = ck_tile::element_wise::PassThrough>
bool invoke_grouped_mx_flatmm_raw(
    const std::vector<ck_tile::index_t>& Ms_host,
    const std::vector<ck_tile::index_t>& Ns_host,
    const std::vector<ck_tile::index_t>& Ks_host,
    const std::vector<const void*>& a_ptrs_host,
    const std::vector<const void*>& b_ptrs_host,
    const std::vector<void*>& c_ptrs_host,
    const std::vector<ck_tile::index_t>& stride_A_host,
    const std::vector<ck_tile::index_t>& stride_B_host,
    const std::vector<ck_tile::index_t>& stride_C_host,
    const std::vector<ScaleAType>& scale_a_host,
    const std::vector<ScaleBType>& scale_b_host,
    const ck_tile::stream_config& stream_cfg,
    void* kargs_workspace,
    size_t kargs_workspace_bytes) {
  using FlatmmConfig = typename MXFlatmmArchTraitsT::Config;
  using FlatmmShape =
      ck_tile::TileGemmShape<ck_tile::sequence<FlatmmConfig::M_Tile,
                                               FlatmmConfig::N_Tile,
                                               FlatmmConfig::K_Tile>,
                             ck_tile::sequence<FlatmmConfig::M_Warp,
                                               FlatmmConfig::N_Warp,
                                               FlatmmConfig::K_Warp>,
                             ck_tile::sequence<FlatmmConfig::M_Warp_Tile,
                                               FlatmmConfig::N_Warp_Tile,
                                               FlatmmConfig::K_Warp_Tile>>;

  using TilePartitioner =
      ck_tile::GemmSpatiallyLocalTilePartitioner<FlatmmShape,
                                                 FlatmmConfig::TilePartitionerGroupNum,
                                                 FlatmmConfig::TilePartitionerM01>;

  using Traits = ck_tile::TileGemmTraits<FlatmmConfig::kPadM,
                                         FlatmmConfig::kPadN,
                                         FlatmmConfig::kPadK,
                                         ALayout,
                                         BLayout,
                                         CLayout,
                                         FlatmmConfig::NumWaveGroups>;

  using GemmPipelineProblem =
      ck_tile::GemmPipelineProblem<ADataType, BDataType, AccDataType, FlatmmShape, Traits>;
  using BaseFlatmmPipeline =
      ck_tile::BaseFlatmmPipelineAGmemBGmemCRegV1<GemmPipelineProblem>;

  if (Ks_host.empty()) {
    NVTE_WARN("ck_tile_mxflat_grouped_gemm: no active groups.");
    return false;
  }

  const ck_tile::index_t K0 = Ks_host.front();
  for (std::size_t i = 1; i < Ks_host.size(); ++i) {
    if (Ks_host[i] != K0) {
      NVTE_WARN("ck_tile_mxflat_grouped_gemm: grouped MXFlatmm requires uniform K.");
      return false;
    }
  }

  const ck_tile::index_t k_grain = FlatmmConfig::K_Tile;
  const ck_tile::index_t k_split = (K0 + k_grain - 1) / k_grain * k_grain;
  const ck_tile::index_t num_loop = TilePartitioner::GetLoopNum(k_split);
  const bool has_hot_loop = BaseFlatmmPipeline::BlockHasHotloop(num_loop);
  const ck_tile::TailNumber tail_num = BaseFlatmmPipeline::GetBlockLoopTailNum(num_loop);

  return BaseFlatmmPipeline::template TailHandler<true>(
      [&](auto has_hot_loop_, auto tail_num_) {
        constexpr bool has_hot_loop_v = has_hot_loop_.value;
        constexpr auto tail_num_v = tail_num_.value;

        using MXGemmTraits =
            ck_tile::TileGemmUniversalTraits<FlatmmConfig::kPadM,
                                             FlatmmConfig::kPadN,
                                             FlatmmConfig::kPadK,
                                             FlatmmConfig::DoubleSmemBuffer,
                                             ALayout,
                                             BLayout,
                                             CLayout,
                                             FlatmmConfig::TransposeC,
                                             FlatmmConfig::UseStructuredSparsity,
                                             UsePersistentKernel,
                                             FlatmmConfig::NumWaveGroups,
                                             true>;

        using MXPipelineProblem =
            ck_tile::MXFlatmmPipelineProblem<ADataType,
                                             BDataType,
                                             AccDataType,
                                             FlatmmShape,
                                             MXGemmTraits,
                                             FlatmmConfig::Scheduler,
                                             has_hot_loop_v,
                                             tail_num_v>;

        using MXFlatmmPipeline =
            typename MXFlatmmArchTraitsT::template MXFlatmmPipeline<MXPipelineProblem>;

        constexpr int BlockedXDLN_PerWarp = MXFlatmmArchTraitsT::BlockedXDLN_PerWarp;

        using GemmEpilogue =
            ck_tile::CShuffleEpilogue<ck_tile::CShuffleEpilogueProblem<ADataType,
                                                                       ADataType,
                                                                       DsDatatype,
                                                                       AccDataType,
                                                                       CDataType,
                                                                       DsLayout,
                                                                       CLayout,
                                                                       CDEElementWise,
                                                                       TilePartitioner::MPerBlock,
                                                                       TilePartitioner::NPerBlock,
                                                                       FlatmmConfig::M_Warp,
                                                                       FlatmmConfig::N_Warp,
                                                                       FlatmmConfig::M_Warp_Tile,
                                                                       FlatmmConfig::N_Warp_Tile,
                                                                       FlatmmConfig::K_Warp_Tile,
                                                                       MXPipelineProblem::TransposeC,
                                                                       FlatmmConfig::NumWaveGroups,
                                                                       false,
                                                                       1,
                                                                       FlatmmConfig::TiledMMAPermuteN,
                                                                       BlockedXDLN_PerWarp>>;

        using UnderlyingKernel =
            ck_tile::MXFlatmmKernel<TilePartitioner, MXFlatmmPipeline, GemmEpilogue>;
        using GroupedKernel =
            ck_tile::GroupedMXFlatmmKernel<TilePartitioner, MXFlatmmPipeline, GemmEpilogue>;
        using Desc = ck_tile::GroupedMXFlatmmHostArgs<ScaleAType, ScaleBType, 0>;

        std::vector<Desc> descs;
        descs.reserve(Ms_host.size());

        for (std::size_t i = 0; i < Ms_host.size(); ++i) {
          ck_tile::FlatmmKernelArgs<ScaleAType, ScaleBType, 0> impl_kargs{
              a_ptrs_host[i],
              b_ptrs_host[i],
              {},
              c_ptrs_host[i],
              Ms_host[i],
              Ns_host[i],
              Ks_host[i],
              stride_A_host[i],
              stride_B_host[i],
              {},
              stride_C_host[i],
              1,
              scale_a_host[i],
              scale_b_host[i]};

          if (!UnderlyingKernel::IsSupportedArgument(impl_kargs)) {
            NVTE_WARN("ck_tile_mxflat_grouped_gemm: unsupported per-group MXFlatmm arguments.");
            return false;
          }

          descs.emplace_back(a_ptrs_host[i],
                             b_ptrs_host[i],
                             std::array<const void*, 0>{},
                             c_ptrs_host[i],
                             1,
                             Ms_host[i],
                             Ns_host[i],
                             Ks_host[i],
                             stride_A_host[i],
                             stride_B_host[i],
                             std::array<ck_tile::index_t, 0>{},
                             stride_C_host[i],
                             scale_a_host[i],
                             scale_b_host[i]);
        }

        auto kargs = GroupedKernel::MakeKargs(descs);
        if (!GroupedKernel::IsSupportedArgument(kargs)) {
          NVTE_WARN("ck_tile_mxflat_grouped_gemm: unsupported grouped MXFlatmm arguments.");
          return false;
        }

        const size_t needed_workspace =
            kargs.size() * sizeof(typename decltype(kargs)::value_type);
        if (!kargs_workspace || kargs_workspace_bytes < needed_workspace) {
          NVTE_WARN("ck_tile_mxflat_grouped_gemm: insufficient kargs workspace. Needed bytes=",
                    needed_workspace,
                    ", available bytes=",
                    kargs_workspace_bytes);
          return false;
        }

        NVTE_CHECK_CUDA(hipMemcpyAsync(kargs_workspace,
                                       kargs.data(),
                                       needed_workspace,
                                       hipMemcpyHostToDevice,
                                       stream_cfg.stream_id_));

        const dim3 grids = GroupedKernel::GridSize(descs);
        const dim3 blocks = GroupedKernel::BlockSize();
        const auto d_workspace_const =
            ck_tile::cast_pointer_to_constant_address_space(kargs_workspace);

        ck_tile::ignore = ck_tile::launch_kernel(
            stream_cfg,
            ck_tile::make_kernel<FlatmmConfig::kBlockPerCu>(
                GroupedKernel{},
                grids,
                blocks,
                0,
                d_workspace_const,
                static_cast<ck_tile::index_t>(kargs.size())));

        return true;
      },
      has_hot_loop,
      tail_num);
}

template <typename AType, typename BType, typename CType, typename AccType = float>
bool invoke_mxflat_grouped_gemm_from_te(const CKGemmRunContext& ctx,
                                        const ck_tile::stream_config& stream_cfg,
                                        char* ws_cursor,
                                        size_t ws_remaining) {
  using FlatmmCfg = typename MXTraits::Config;
  using Row = RowMajor;
  using Col = ColMajor;

  // Preserve the TE workspace contract: ctx.workspace/ctx.workspace_bytes are
  // the single scratch region for this path.  Match the hipBLASLt/FP4 pattern by
  // carving MXFlatmm preprocessing buffers from the END of the workspace and
  // shrinking the remaining front region before passing it to the grouped-kargs
  // upload path.
  //
  // Layout after carving:
  //   [ grouped-kargs workspace ... | B preshuffle / A-scale / B-scale buffers ]
  uint8_t* const ws_base = reinterpret_cast<uint8_t*>(ws_cursor);
  size_t kargs_workspace_bytes = ws_remaining;

  auto carve_workspace_from_end = [&](size_t bytes, size_t alignment, const char* label) -> void* {
    if (bytes == 0) {
      return nullptr;
    }
    kargs_workspace_bytes = (kargs_workspace_bytes / alignment) * alignment;
    if (kargs_workspace_bytes < bytes) {
      NVTE_WARN("ck_tile_mxflat_grouped_gemm: insufficient workspace while carving ",
                label,
                ". Needed bytes=",
                bytes,
                ", available bytes=",
                kargs_workspace_bytes,
                ".");
      return nullptr;
    }
    kargs_workspace_bytes -= bytes;
    return ws_base + kargs_workspace_bytes;
  };

  std::vector<ck_tile::index_t> Ms_host;
  std::vector<ck_tile::index_t> Ns_host;
  std::vector<ck_tile::index_t> Ks_host;
  std::vector<ck_tile::index_t> stride_A_host;
  std::vector<ck_tile::index_t> stride_B_host;
  std::vector<ck_tile::index_t> stride_C_host;
  std::vector<const void*> a_ptrs_host;
  std::vector<const void*> b_ptrs_host;
  std::vector<void*> c_ptrs_host;
  std::vector<ScaleA> scale_a_host;
  std::vector<ScaleB> scale_b_host;

  std::vector<GroupedStageADesc<AType>> a_stage_descs;
  std::vector<GroupedWeightPreshuffleDesc<BType>> b_preshuffle_descs;
  std::vector<GroupedScalePreshuffleDesc> a_scale_preshuffle_descs;
  std::vector<GroupedScalePreshuffleDesc> b_scale_preshuffle_descs;
  std::vector<GroupedCopyCDesc<CType>> c_copyback_descs;

  Ms_host.reserve(ctx.group_num);
  Ns_host.reserve(ctx.group_num);
  Ks_host.reserve(ctx.group_num);
  stride_A_host.reserve(ctx.group_num);
  stride_B_host.reserve(ctx.group_num);
  stride_C_host.reserve(ctx.group_num);
  a_ptrs_host.reserve(ctx.group_num);
  b_ptrs_host.reserve(ctx.group_num);
  c_ptrs_host.reserve(ctx.group_num);
  scale_a_host.reserve(ctx.group_num);
  scale_b_host.reserve(ctx.group_num);
  a_stage_descs.reserve(ctx.group_num);
  b_preshuffle_descs.reserve(ctx.group_num);
  a_scale_preshuffle_descs.reserve(ctx.group_num);
  b_scale_preshuffle_descs.reserve(ctx.group_num);
  c_copyback_descs.reserve(ctx.group_num);

  for (int i = 0; i < ctx.group_num; ++i) {
    const Tensor* const A_te = convertNVTETensorCheck(ctx.A[i]);
    const Tensor* const B_te = convertNVTETensorCheck(ctx.B[i]);
    Tensor* D_te = convertNVTETensorCheck(ctx.D[i]);

    const SimpleTensor* a_src = nullptr;
    if (ctx.use_a_columnwise_data) {
      NVTE_CHECK(A_te->has_columnwise_data(),
                 "ck_tile_mxflat_grouped_gemm: ctx.use_a_columnwise_data=true but "
                 "columnwise_data is absent.");
      a_src = &A_te->columnwise_data;
    } else {
      a_src = &A_te->data;
    }

    const SimpleTensor* b_src = nullptr;
    if (ctx.use_b_columnwise_data) {
      NVTE_CHECK(B_te->has_columnwise_data(),
                 "ck_tile_mxflat_grouped_gemm: ctx.use_b_columnwise_data=true but "
                 "columnwise_data is absent.");
      b_src = &B_te->columnwise_data;
    } else {
      b_src = &B_te->data;
    }

    const auto& a = *a_src;
    const auto& b = *b_src;
    const auto& d = data_view(*D_te);

    int64_t Ad0 = 0, Ad1 = 0, Bd0 = 0, Bd1 = 0, Dd0 = 0, Dd1 = 0;

    if (ctx.use_a_columnwise_data) {
      if (!get_columnwise_storage_2d_dims(A_te->columnwise_data, Ad0, Ad1)) {
        NVTE_ERROR("ck_tile_mxflat_grouped_gemm: expected 2D columnwise_data for A in group ",
                   i);
      }
    } else {
      if (!get_flat_2d_dims(*A_te, Ad0, Ad1)) {
        NVTE_ERROR("ck_tile_mxflat_grouped_gemm: expected rank>=2 for A in group ", i);
      }
    }

    if (ctx.use_b_columnwise_data) {
      if (!get_columnwise_storage_2d_dims(B_te->columnwise_data, Bd0, Bd1)) {
        NVTE_ERROR("ck_tile_mxflat_grouped_gemm: expected 2D columnwise_data for B in group ",
                   i);
      }
    } else {
      if (!get_flat_2d_dims(*B_te, Bd0, Bd1)) {
        NVTE_ERROR("ck_tile_mxflat_grouped_gemm: expected rank>=2 for B in group ", i);
      }
    }

    if (!get_flat_2d_dims(*D_te, Dd0, Dd1)) {
      NVTE_ERROR("ck_tile_mxflat_grouped_gemm: expected rank>=2 for D in group ", i);
    }

    const auto& a_scales =
        ctx.use_a_columnwise_data ? A_te->columnwise_scale_inv : A_te->scale_inv;
    const auto& b_scales =
        ctx.use_b_columnwise_data ? B_te->columnwise_scale_inv : B_te->scale_inv;

    if (a_scales.shape.size() != 2 || b_scales.shape.size() != 2) {
      NVTE_ERROR("ck_tile_mxflat_grouped_gemm: expected A/B scale_inv tensors to be rank-2.");
    }

    NVTE_CHECK(a_scales.dtype == DType::kFloat8E8M0,
               "ck_tile_mxflat_grouped_gemm: A scale dtype must be Float8E8M0.");
    NVTE_CHECK(b_scales.dtype == DType::kFloat8E8M0,
               "ck_tile_mxflat_grouped_gemm: B scale dtype must be Float8E8M0.");

    const int64_t M = ctx.transA ? Ad1 : Ad0;
    const int64_t K = ctx.transA ? Ad0 : Ad1;
    const int64_t N = ctx.transB ? Bd0 : Bd1;
    const int64_t Kb = ctx.transB ? Bd1 : Bd0;

    if (K % ScaleBlockSize != 0) {
      NVTE_ERROR("ck_tile_mxflat_grouped_gemm: K must be a multiple of ScaleBlockSize in group ",
                 i);
    }

    const int64_t KScale = K / ScaleBlockSize;
    if (Kb != K) {
      NVTE_ERROR("ck_tile_mxflat_grouped_gemm: K mismatch between A and B in group ", i);
    }
    if (Dd0 != M || Dd1 != N) {
      NVTE_ERROR("ck_tile_mxflat_grouped_gemm: D shape mismatch in group ", i);
    }

    const bool a_scales_m_k =
        (a_scales.shape[0] == M && a_scales.shape[1] == KScale);
    const bool a_scales_k_m =
        (a_scales.shape[0] == KScale && a_scales.shape[1] == M);
    if (!a_scales_m_k && !a_scales_k_m) {
      NVTE_ERROR("ck_tile_mxflat_grouped_gemm: expected A scales shape [M, KScale] or "
                 "[KScale, M].");
    }

    const bool b_scales_n_k =
        (b_scales.shape[0] == N && b_scales.shape[1] == KScale);
    const bool b_scales_k_n =
        (b_scales.shape[0] == KScale && b_scales.shape[1] == N);
    if (!b_scales_n_k && !b_scales_k_n) {
      NVTE_ERROR("ck_tile_mxflat_grouped_gemm: expected B scales shape [N, KScale] or "
                 "[KScale, N].");
    }

    if (K % FlatmmCfg::K_Tile != 0) {
      NVTE_WARN("ck_tile_mxflat_grouped_gemm: K is not a multiple of MXFlatmm K_Tile. K=", K);
      return false;
    }

    const ck_tile::index_t M_padded =
        round_up_index(static_cast<ck_tile::index_t>(M), FlatmmCfg::M_Tile);

    const ck_tile::index_t stride_A = static_cast<ck_tile::index_t>(Ad1);
    const ck_tile::index_t stride_B = static_cast<ck_tile::index_t>(Bd1);
    const ck_tile::index_t stride_E = static_cast<ck_tile::index_t>(Dd1);

    const size_t a_stage_bytes =
        align_up_size(static_cast<size_t>(M_padded) * static_cast<size_t>(K) * sizeof(AType), 16);
    const size_t b_shuf_bytes =
        align_up_size(static_cast<size_t>(K) * static_cast<size_t>(N) * sizeof(BType), 16);
    const size_t c_stage_bytes =
        (M_padded == static_cast<ck_tile::index_t>(M))
            ? 0
            : align_up_size(static_cast<size_t>(M_padded) * static_cast<size_t>(N) * sizeof(CType), 16);
    const size_t a_scale_shuf_bytes =
        align_up_size(static_cast<size_t>(round_up_index(M_padded, 32)) *
                          static_cast<size_t>(KScale) * sizeof(ScaleType),
                      16);
    const size_t b_scale_shuf_bytes =
        align_up_size(static_cast<size_t>(round_up_index(static_cast<ck_tile::index_t>(N), 32)) *
                          static_cast<size_t>(KScale) * sizeof(ScaleType),
                      16);

    auto* a_stage = reinterpret_cast<AType*>(
        carve_workspace_from_end(a_stage_bytes, 16, "A padded/staged buffer"));
    auto* b_shuf = reinterpret_cast<BType*>(
        carve_workspace_from_end(b_shuf_bytes, 16, "B preshuffle buffer"));
    auto* c_stage = reinterpret_cast<CType*>(
        carve_workspace_from_end(c_stage_bytes, 16, "C padded output buffer"));
    auto* a_scale_shuf = reinterpret_cast<ScaleType*>(
        carve_workspace_from_end(a_scale_shuf_bytes, 16, "A scale preshuffle buffer"));
    auto* b_scale_shuf = reinterpret_cast<ScaleType*>(
        carve_workspace_from_end(b_scale_shuf_bytes, 16, "B scale preshuffle buffer"));

    if (a_stage == nullptr || b_shuf == nullptr || a_scale_shuf == nullptr ||
        b_scale_shuf == nullptr || (c_stage_bytes != 0 && c_stage == nullptr)) {
      return false;
    }

    constexpr int preprocess_threads = 256;

    // Build grouped preprocessing descriptors.  These preserve the exact same
    // per-expert transforms as the original one-launch-per-expert path, but are
    // launched after this loop as one grouped kernel per transform type.
    const int a_stage_total = static_cast<int>(M_padded) * static_cast<int>(K);
    a_stage_descs.push_back(GroupedStageADesc<AType>{
        reinterpret_cast<const AType*>(a.dptr),
        a_stage,
        static_cast<int>(M),
        static_cast<int>(M_padded),
        static_cast<int>(K),
        static_cast<int64_t>(Ad1),
        1,
        ctx.transA,
        (a_stage_total + preprocess_threads - 1) / preprocess_threads});

    const int b_preshuffle_total =
        static_cast<int>(N) *
        (static_cast<int>(K) / ck_tile::numeric_traits<BType>::PackedSize);
    b_preshuffle_descs.push_back(GroupedWeightPreshuffleDesc<BType>{
        reinterpret_cast<const BType*>(b.dptr),
        b_shuf,
        static_cast<int>(K),
        static_cast<int>(N),
        static_cast<int64_t>(Bd1),
        1,
        ctx.transB,
        (b_preshuffle_total + preprocess_threads - 1) / preprocess_threads});

    const auto make_scale_blocks = [=](int MN_dst, int K) {
      constexpr int XdlMNThread = 16;
      constexpr int MNXdlPack = 2;
      const int MN_padded =
          ((MN_dst + XdlMNThread * MNXdlPack - 1) / (XdlMNThread * MNXdlPack)) *
          (XdlMNThread * MNXdlPack);
      const int total = MN_padded * K;
      return (total + preprocess_threads - 1) / preprocess_threads;
    };

    a_scale_preshuffle_descs.push_back(GroupedScalePreshuffleDesc{
        reinterpret_cast<const ScaleType*>(a_scales.dptr),
        a_scale_shuf,
        static_cast<int>(M),
        static_cast<int>(M_padded),
        static_cast<int>(KScale),
        static_cast<int64_t>(a_scales.shape[1]),
        1,
        a_scales_m_k,
        make_scale_blocks(static_cast<int>(M_padded), static_cast<int>(KScale))});

    b_scale_preshuffle_descs.push_back(GroupedScalePreshuffleDesc{
        reinterpret_cast<const ScaleType*>(b_scales.dptr),
        b_scale_shuf,
        static_cast<int>(N),
        static_cast<int>(N),
        static_cast<int>(KScale),
        static_cast<int64_t>(b_scales.shape[1]),
        1,
        !b_scales_k_n,
        make_scale_blocks(static_cast<int>(N), static_cast<int>(KScale))});

    void* c_ptr_for_kernel = (c_stage_bytes == 0) ? d.dptr : static_cast<void*>(c_stage);
    const ck_tile::index_t stride_C_for_kernel =
        (c_stage_bytes == 0) ? stride_E : static_cast<ck_tile::index_t>(N);

    a_ptrs_host.push_back(a_stage);
    b_ptrs_host.push_back(b_shuf);
    c_ptrs_host.push_back(c_ptr_for_kernel);
    Ms_host.push_back(M_padded);
    Ns_host.push_back(static_cast<ck_tile::index_t>(N));
    Ks_host.push_back(static_cast<ck_tile::index_t>(K));
    stride_A_host.push_back(static_cast<ck_tile::index_t>(K));
    stride_B_host.push_back(stride_B);
    stride_C_host.push_back(stride_C_for_kernel);
    scale_a_host.push_back(ScaleA{a_scale_shuf, M_padded});
    scale_b_host.push_back(ScaleB{b_scale_shuf, static_cast<ck_tile::index_t>(N)});

    if (c_stage_bytes != 0) {
      const int c_copy_total = static_cast<int>(M) * static_cast<int>(N);
      c_copyback_descs.push_back(GroupedCopyCDesc<CType>{
          c_stage,
          reinterpret_cast<CType*>(d.dptr),
          static_cast<int>(M),
          static_cast<int>(N),
          static_cast<int64_t>(stride_E),
          (c_copy_total + 256 - 1) / 256});
    }
  }


  auto* a_stage_descs_dev = reinterpret_cast<GroupedStageADesc<AType>*>(
      carve_workspace_from_end(a_stage_descs.size() * sizeof(GroupedStageADesc<AType>),
                               alignof(GroupedStageADesc<AType>),
                               "A staging descriptor buffer"));
  auto* b_preshuffle_descs_dev = reinterpret_cast<GroupedWeightPreshuffleDesc<BType>*>(
      carve_workspace_from_end(b_preshuffle_descs.size() * sizeof(GroupedWeightPreshuffleDesc<BType>),
                               alignof(GroupedWeightPreshuffleDesc<BType>),
                               "B preshuffle descriptor buffer"));
  auto* a_scale_descs_dev = reinterpret_cast<GroupedScalePreshuffleDesc*>(
      carve_workspace_from_end(a_scale_preshuffle_descs.size() * sizeof(GroupedScalePreshuffleDesc),
                               alignof(GroupedScalePreshuffleDesc),
                               "A scale preshuffle descriptor buffer"));
  auto* b_scale_descs_dev = reinterpret_cast<GroupedScalePreshuffleDesc*>(
      carve_workspace_from_end(b_scale_preshuffle_descs.size() * sizeof(GroupedScalePreshuffleDesc),
                               alignof(GroupedScalePreshuffleDesc),
                               "B scale preshuffle descriptor buffer"));
  auto* c_copyback_descs_dev = reinterpret_cast<GroupedCopyCDesc<CType>*>(
      carve_workspace_from_end(c_copyback_descs.size() * sizeof(GroupedCopyCDesc<CType>),
                               alignof(GroupedCopyCDesc<CType>),
                               "C copyback descriptor buffer"));

  if ((a_stage_descs.empty() == false && a_stage_descs_dev == nullptr) ||
      (b_preshuffle_descs.empty() == false && b_preshuffle_descs_dev == nullptr) ||
      (a_scale_preshuffle_descs.empty() == false && a_scale_descs_dev == nullptr) ||
      (b_scale_preshuffle_descs.empty() == false && b_scale_descs_dev == nullptr) ||
      (c_copyback_descs.empty() == false && c_copyback_descs_dev == nullptr)) {
    return false;
  }

  launch_grouped_stage_a_rowmajor<AType>(a_stage_descs, a_stage_descs_dev, ctx.stream);
  launch_grouped_weight_preshuffle<BType>(b_preshuffle_descs, b_preshuffle_descs_dev, ctx.stream);
  launch_grouped_scale_preshuffle(a_scale_preshuffle_descs, a_scale_descs_dev, ctx.stream);
  launch_grouped_scale_preshuffle(b_scale_preshuffle_descs, b_scale_descs_dev, ctx.stream);

  const bool launched = invoke_grouped_mx_flatmm_raw<MXTraits,
                                                  AType,
                                                  BType,
                                                  ck_tile::tuple<>,
                                                  AccType,
                                                  CType,
                                                  Row,
                                                  Col,
                                                  ck_tile::tuple<>,
                                                  Row>(Ms_host,
                                                       Ns_host,
                                                       Ks_host,
                                                       a_ptrs_host,
                                                       b_ptrs_host,
                                                       c_ptrs_host,
                                                       stride_A_host,
                                                       stride_B_host,
                                                       stride_C_host,
                                                       scale_a_host,
                                                       scale_b_host,
                                                       stream_cfg,
                                                       ws_base,
                                                       kargs_workspace_bytes);
  if (!launched) {
    return false;
  }

  // If any group needed padded C workspace, copy only the real M rows back to TE's D
  // with one grouped copyback launch.
  launch_grouped_copy_c_unpad<CType>(c_copyback_descs, c_copyback_descs_dev, ctx.stream);

  return true;
}

}  // namespace transformer_engine

bool ck_tile_mxflat_grouped_gemm(const NVTETensor* A,
                                 const NVTETensor* B,
                                 NVTETensor* D,
                                 int group_num,
                                 bool transA,
                                 bool transB,
                                 NVTETensor* workspace,
                                 bool accumulate,
                                 hipStream_t stream) {
  if (accumulate || group_num <= 0) {
    return true;
  }

  using namespace transformer_engine;

  void* ws_ptr = nullptr;
  size_t ws_bytes = 0;
  if (workspace) {
    auto* ws_te = convertNVTETensorCheck(*workspace);
    ws_ptr = ws_te->data.dptr;
    ws_bytes = ws_te->data.numel() * typeToSize(ws_te->data.dtype);
  }

  const auto norm = normalize_gemm_inputs(A, B, transA, transB);
  const NVTETensor* A_use = norm.A;
  const NVTETensor* B_use = norm.B;
  bool transA_use = norm.transA;
  bool transB_use = norm.transB;

  bool use_a_colwise_data = false;
  bool use_b_colwise_data = false;

  Tensor* A0_te = convertNVTETensorCheck(A_use[0]);
  Tensor* B0_te = convertNVTETensorCheck(B_use[0]);
  Tensor* D0_te = convertNVTETensorCheck(D[0]);

  const auto a_dtype = A0_te->dtype();
  const auto b_dtype = B0_te->dtype();
  const auto d_dtype = D0_te->dtype();

  // Keep the same normalized NT presentation policy as ck_mx_grouped_gemm.cpp.
  const auto presentation = select_ck_fp8_nt_presentation(true,
                                                          transA_use,
                                                          transB_use,
                                                          A0_te->has_columnwise_data(),
                                                          B0_te->has_columnwise_data());

  transA_use = presentation.transA;
  transB_use = presentation.transB;
  use_a_colwise_data = presentation.use_a_colwise_data;
  use_b_colwise_data = presentation.use_b_colwise_data;

  int64_t a0 = 0, a1 = 0;
  int64_t b0 = 0, b1 = 0;
  int64_t d0 = 0, d1 = 0;

  if (use_a_colwise_data) {
    if (!get_columnwise_storage_2d_dims(A0_te->columnwise_data, a0, a1)) {
      NVTE_ERROR("ck_tile_mxflat_grouped_gemm: expected 2D columnwise_data for A_use[0]");
      return false;
    }
  } else {
    if (!get_flat_2d_dims(*A0_te, a0, a1)) {
      NVTE_ERROR("ck_tile_mxflat_grouped_gemm: expected rank>=2 for normalized A_use[0]");
      return false;
    }
  }

  if (use_b_colwise_data) {
    if (!get_columnwise_storage_2d_dims(B0_te->columnwise_data, b0, b1)) {
      NVTE_ERROR("ck_tile_mxflat_grouped_gemm: expected 2D columnwise_data for B_use[0]");
      return false;
    }
  } else {
    if (!get_flat_2d_dims(*B0_te, b0, b1)) {
      NVTE_ERROR("ck_tile_mxflat_grouped_gemm: expected rank>=2 for normalized B_use[0]");
      return false;
    }
  }

  if (!get_flat_2d_dims(*D0_te, d0, d1)) {
    NVTE_ERROR("ck_tile_mxflat_grouped_gemm: expected rank>=2 for D[0]");
    return false;
  }

  const int64_t m = transA_use ? a1 : a0;
  const int64_t kA = transA_use ? a0 : a1;
  const int64_t kB = transB_use ? b1 : b0;
  const int64_t n = transB_use ? b0 : b1;

  if (kA != kB) {
    NVTE_ERROR("ck_tile_mxflat_grouped_gemm: normalized GEMM K mismatch: op(A_use) is ",
               m,
               "x",
               kA,
               ", op(B_use) is ",
               kB,
               "x",
               n);
    return false;
  }

  if (d0 != m || d1 != n) {
    NVTE_ERROR("ck_tile_mxflat_grouped_gemm: D shape mismatch for normalized GEMM. D is ",
               d0,
               "x",
               d1,
               " but expected ",
               m,
               "x",
               n);
    return false;
  }

  CKGemmRunContext ctx = {A_use,
                          B_use,
                          D,
                          static_cast<int>(n),
                          group_num,
                          transA_use,
                          transB_use,
                          ws_ptr,
                          ws_bytes,
                          stream,
                          use_a_colwise_data,
                          use_b_colwise_data,
                          accumulate};

  const ck_tile::stream_config s{ctx.stream};

  bool ok = false;
  char* ws_cursor = static_cast<char*>(ws_ptr);
  size_t ws_remaining = ws_bytes;

  // MXFlatmm policy in CK is specialized for ck_tile::fp8_t, matching the standalone
  // ck_mx_flat_grouped_gemm path. Do not instantiate this pipeline with the raw
  // TE switch type for FP8, because that maps to the compiler storage type and
  // trips MXFlatmmPolicy unsupported-datatype static assertions.
  TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(a_dtype, a_te_type, {
    TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(b_dtype, b_te_type, {
      TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(d_dtype, d_te_type, {
        using AType = ck_tile::fp8_t;
        using BType = ck_tile::fp8_t;
        using CType = typename TETypeToCKType<d_te_type>::type;
        ok = invoke_mxflat_grouped_gemm_from_te<AType, BType, CType>(
            ctx, s, ws_cursor, ws_remaining);
      });
    });
  });

  return ok;
}
