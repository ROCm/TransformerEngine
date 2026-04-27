/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#pragma once

namespace transformer_engine {
namespace grouped_gemm {

bool ck_tile_grouped_gemm_fp8_dispatch(DType a_dtype,
                                    DType b_dtype,
                                    DType d_dtype,
                                    const CKGemmRunContext& ctx);

}  // namespace grouped_gemm
}  // namespace transformer_engine
