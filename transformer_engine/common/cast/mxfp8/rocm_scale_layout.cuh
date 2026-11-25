/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/
#pragma once
//#include "hip/hip_runtime.h" //dummy include to prevent hipification adding this header

// Where the ROCm MXFP8 kernels write the E8M0 scales of a tensor. Like the other rocm_ headers,
// this expects its includer to have included the common headers.

namespace transformer_engine {
namespace dispatch {
namespace mxfp8 {

// Where scale k of row m of a rowwise scaling, or of column m of a colwise one, lies, with strides
// for m, for groups of four k and for k within a group.
struct ScaleLayout {
  size_t m_stride;
  size_t group_stride;
  size_t k_stride;

  __host__ __device__ __forceinline__ size_t offset(const size_t m, const size_t k) const {
    return m * m_stride + k / 4 * group_stride + k % 4 * k_stride;
  }
};

// The layout of [m, k] rowwise or [k, m] colwise scales with rows stride apart.
__host__ __device__ __forceinline__ ScaleLayout compact_scale_layout(const size_t stride,
                                                                     const bool colwise) {
  return colwise ? ScaleLayout{1, 4 * stride, stride} : ScaleLayout{stride, 4, 1};
}

// The layout of the [rows, scale cols] rowwise or [scale rows, cols] colwise scales of a tensor.
inline ScaleLayout scale_layout(const SimpleTensor &scales, const bool colwise) {
  return compact_scale_layout(scales.shape.size() == 2 ? scales.shape[1] : 1, colwise);
}

}  // namespace mxfp8
}  // namespace dispatch
}  // namespace transformer_engine
