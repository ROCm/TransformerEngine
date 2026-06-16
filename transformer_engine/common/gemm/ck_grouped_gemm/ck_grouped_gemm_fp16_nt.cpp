/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#include "ck_grouped_gemm_fp16_impl.h"

namespace transformer_engine {
namespace grouped_gemm {

template <GPUArch Arch>
bool ck_tile_grouped_gemm_fp16_dispatch_nt(DType a_dtype, DType d_dtype,
                                           bool need_m_pad, bool need_k_pad,
                                           const GroupedGemmRunContext& ctx) {
  return ck_tile_grouped_gemm_fp16_dispatch_layout<Arch, RowMajor, ColMajor>(
      a_dtype, d_dtype, need_m_pad, need_k_pad, ctx);
}

}  // namespace grouped_gemm
}  // namespace transformer_engine
