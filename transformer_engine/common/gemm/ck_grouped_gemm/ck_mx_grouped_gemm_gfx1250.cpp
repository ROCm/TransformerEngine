/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#include "ck_mx_grouped_gemm_impl.h"

namespace transformer_engine {
namespace grouped_gemm {

bool ck_tile_mx_grouped_gemm_dispatch_gfx1250(DType a_dtype, DType b_dtype, DType d_dtype,
                                              const GroupedGemmRunContext& ctx) {
  return ck_tile_mx_grouped_gemm_impl<GPUArch::GFX1250>(a_dtype, b_dtype, d_dtype, ctx);
}

}  // namespace grouped_gemm
}  // namespace transformer_engine
