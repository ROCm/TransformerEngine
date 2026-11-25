/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/
#pragma once
//#include "hip/hip_runtime.h" //dummy include to prevent hipification adding this header

// Where the ROCm MXFP8 kernels write the E8M0 scales of a tensor: compact, or pre-swizzled for the
// gfx1250 GEMM. Like the other rocm_ headers, this expects its includer to have included the common
// headers and util/cuda_runtime.h.

#include <algorithm>
#include <vector>

namespace transformer_engine {
namespace dispatch {
namespace mxfp8 {

// Where scale k of row m of a rowwise scaling, or of column m of a colwise one, lies, with strides
// for m, for groups of four k and for k within a group.
struct ScaleLayout {
  size_t m_stride;
  size_t group_stride;
  size_t k_stride;
  bool preswizzled = false;

  __host__ __device__ __forceinline__ size_t offset(const size_t m, const size_t k) const {
    return m * m_stride + k / 4 * group_stride + k % 4 * k_stride;
  }
};

// The layout of [m, k] rowwise or [k, m] colwise scales with rows stride apart.
__host__ __device__ __forceinline__ ScaleLayout compact_scale_layout(const size_t stride,
                                                                     const bool colwise) {
  return colwise ? ScaleLayout{1, 4 * stride, stride} : ScaleLayout{stride, 4, 1};
}

// The layout of the scales of padded_m m pre-swizzled for the gfx1250 GEMM, as
// swizzle_scaling_factors_mx lays them out: each m has the four k of a group together.
__host__ __device__ __forceinline__ ScaleLayout preswizzled_scale_layout(const size_t padded_m) {
  return {4, 4 * padded_m, 1, true};
}

// The layout of the [rows, scale cols] rowwise or [scale rows, cols] colwise scales of a tensor.
inline ScaleLayout scale_layout(const SimpleTensor &scales, const bool colwise,
                                const bool preswizzled) {
  const size_t stride = scales.shape.size() == 2 ? scales.shape[1] : 1;
  if (preswizzled) {
    NVTE_CHECK(scales.shape.size() == 2 && scales.shape[colwise ? 0 : 1] % 4 == 0,
               "Pre-swizzled MXFP8 scales need their scale dimension padded to a multiple of 4.");
    return preswizzled_scale_layout(colwise ? stride : scales.shape[0]);
  }
  return compact_scale_layout(stride, colwise);
}

// The scale tensors of one fill launch.
constexpr int SCALE_FILL_BATCH = 128;
struct ScaleFillArgs {
  uint32_t *scales[SCALE_FILL_BATCH];
  size_t words[SCALE_FILL_BATCH];
};

// Fills scale tensor blockIdx.y of args with E8M0 127 (1.0), the padding of pre-swizzled scales,
// unless noop is set.
template <typename = void>
__global__ void fill_scales_kernel(const ScaleFillArgs args, const float *noop) {
  if (noop != nullptr && noop[0] == 1.0f) {
    return;
  }
  uint32_t *const scales = args.scales[blockIdx.y];
  const size_t words = args.words[blockIdx.y];
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < words; i += gridDim.x * blockDim.x) {
    scales[i] = 0x7F7F7F7F;
  }
}

// The kernels write pre-swizzled scales only within the tensor, so where scale tensors have
// padding, this fills them beforehand, unless noop is set, with one launch per SCALE_FILL_BATCH.
class ScalePadding {
 public:
  ScalePadding(const float *noop, hipStream_t stream) : noop_(noop), stream_(stream) {}

  // Adds scales, valid_m x valid_k of them within the tensor.
  template <typename = void>
  void add(const SimpleTensor &scales, const size_t valid_m, const size_t valid_k,
           const bool colwise) {
    const size_t padded_m = colwise ? scales.shape[1] : scales.shape[0];
    const size_t padded_k = colwise ? scales.shape[0] : scales.shape[1];
    const size_t words = padded_m * padded_k / 4;
    if ((padded_m == valid_m && padded_k == valid_k) || words == 0) {
      return;
    }
    if (num_ == SCALE_FILL_BATCH) {
      fill();
    }
    args_.scales[num_] = static_cast<uint32_t *>(scales.dptr);
    args_.words[num_] = words;
    max_words_ = std::max(max_words_, words);
    num_++;
  }

  // Launches the fills of the added scales, before the kernels that write the scales.
  template <typename = void>
  void fill() {
    if (num_ == 0) {
      return;
    }
    const dim3 grid(std::min<size_t>(DIVUP(max_words_, size_t{256}), 1024), num_);
    fill_scales_kernel<<<grid, 256, 0, stream_>>>(args_, noop_);
    NVTE_CHECK_CUDA(hipGetLastError());
    num_ = 0;
    max_words_ = 0;
  }

 private:
  ScaleFillArgs args_{};
  int num_ = 0;
  size_t max_words_ = 0;
  const float *noop_;
  hipStream_t stream_;
};

// The layout of scales with valid_m x valid_k of them within the tensor; adds pre-swizzled ones to
// padding.
template <typename = void>
ScaleLayout prepare_scales(const SimpleTensor &scales, const bool colwise, const bool preswizzled,
                           const size_t valid_m, const size_t valid_k, ScalePadding &padding) {
  const ScaleLayout layout = scale_layout(scales, colwise, preswizzled);
  if (preswizzled) {
    padding.add(scales, valid_m, valid_k, colwise);
  }
  return layout;
}

// Whether the scales of output are to be pre-swizzled for the gfx1250 GEMM.
inline bool preswizzled_scales(const Tensor &output) {
  return output.with_gemm_swizzled_scales && cuda::sm_arch() == 125;
}

// For kernels that only write compact scales: quantize_compact quantizes into a copy of output with
// temporary compact scales, which then go pre-swizzled to output's, unless noop is set.
template <typename QuantizeCompact>
void quantize_preswizzled(Tensor *output, const float *noop, hipStream_t stream,
                          QuantizeCompact &&quantize_compact) {
  struct Temporaries {
    std::vector<void *> ptrs;
    hipStream_t stream;
    ~Temporaries() {
      for (void *ptr : ptrs) {
        (void)hipFreeAsync(ptr, stream);
      }
    }
  } temporaries{{}, stream};
  Tensor compact = *output;
  compact.with_gemm_swizzled_scales = false;
  for (SimpleTensor *scales : {&compact.scale_inv, &compact.columnwise_scale_inv}) {
    if (scales->has_data()) {
      NVTE_CHECK_CUDA(hipMallocAsync(&scales->dptr, scales->buffer_size_bytes(), stream));
      temporaries.ptrs.push_back(scales->dptr);
    }
  }
  quantize_compact(&compact);
  // The swizzle takes one scaling at a time.
  for (const bool colwise : {false, true}) {
    Tensor from = compact;
    Tensor to = *output;
    for (Tensor *t : {&from, &to}) {
      (colwise ? t->data : t->columnwise_data) = SimpleTensor();
      (colwise ? t->scale_inv : t->columnwise_scale_inv) = SimpleTensor();
    }
    if ((colwise ? from.columnwise_scale_inv : from.scale_inv).has_data()) {
      swizzle_scaling_factors_mx(&from, &to, stream, noop);
    }
  }
}

}  // namespace mxfp8
}  // namespace dispatch
}  // namespace transformer_engine
