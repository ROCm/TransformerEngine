/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#include "ck_grouped_gemm_common.h"
#include "ck_grouped_gemm_fp16.h"
#include "ck_grouped_gemm_fp16_impl.h"

namespace transformer_engine {
namespace grouped_gemm {

bool ck_tile_grouped_gemm_fp16_dispatch(DType a_dtype,
                                        DType b_dtype,
                                        DType d_dtype,
                                        const GroupedGemmRunContext& ctx) {
  // Check M and K alignment across all groups.
  // All tile configs share the same M_Tile (256) and K_Tile (64).
  constexpr ck_tile::index_t M_Tile = TileCfg_256x256x64::M_Tile;
  constexpr ck_tile::index_t K_Tile = TileCfg_256x256x64::K_Tile;

  bool need_m_pad = false;
  bool need_k_pad = false;

  for (int i = 0; i < ctx.group_num; ++i) {
    const transformer_engine::Tensor* A_te =
        transformer_engine::convertNVTETensorCheck(ctx.A[i]);
    int64_t Ad0 = 0, Ad1 = 0;
    if (get_flat_2d_dims(*A_te, Ad0, Ad1)) {
      const int64_t M = ctx.transA ? Ad1 : Ad0;
      const int64_t K = ctx.transA ? Ad0 : Ad1;

      if (M % M_Tile != 0)
        need_m_pad = true;
      if (K % K_Tile != 0)
        need_k_pad = true;
      if (need_m_pad && need_k_pad)
        break;
    }
  }

  // FIXME: CK tile kernel produces incorrect results with kPadK + ColMajor B.
  // Workaround: use B's column-wise storage buffer (RowMajor) with transB=false,
  // which preserves the same logical GEMM while avoiding the buggy path.
  // Fall back to cuBLAS only if the column-wise buffer is unavailable.
  if (need_k_pad && ctx.transB) {
    // Check all B tensors have columnwise_data available.
    bool all_have_columnwise = true;
    for (int i = 0; i < ctx.group_num; ++i) {
      const transformer_engine::Tensor* B_te =
          transformer_engine::convertNVTETensorCheck(ctx.B[i]);
      if (!B_te->has_columnwise_data()) {
        all_have_columnwise = false;
        break;
      }
    }
    if (!all_have_columnwise) {
      return false;
    }
    // Dispatch with B's columnwise buffer as RowMajor (transB=false).
    GroupedGemmRunContext ctx_b_colwise = ctx;
    ctx_b_colwise.transB = false;
    ctx_b_colwise.use_b_columnwise_data = true;
    if (!ctx_b_colwise.transA) {
      return ck_tile_grouped_gemm_fp16_dispatch_nn(a_dtype, d_dtype, need_m_pad, need_k_pad, ctx_b_colwise);
    } else {
      return ck_tile_grouped_gemm_fp16_dispatch_tn(a_dtype, d_dtype, need_m_pad, need_k_pad, ctx_b_colwise);
    }
  }

  // Dispatch to per-layout translation unit.
  if (!ctx.transA && !ctx.transB) {
    return ck_tile_grouped_gemm_fp16_dispatch_nn(a_dtype, d_dtype, need_m_pad, need_k_pad, ctx);
  } else if (!ctx.transA && ctx.transB) {
    return ck_tile_grouped_gemm_fp16_dispatch_nt(a_dtype, d_dtype, need_m_pad, need_k_pad, ctx);
  } else if (ctx.transA && !ctx.transB) {
    return ck_tile_grouped_gemm_fp16_dispatch_tn(a_dtype, d_dtype, need_m_pad, need_k_pad, ctx);
  } else {
    return ck_tile_grouped_gemm_fp16_dispatch_tt(a_dtype, d_dtype, need_m_pad, need_k_pad, ctx);
  }
}

}  // namespace grouped_gemm
}  // namespace transformer_engine
