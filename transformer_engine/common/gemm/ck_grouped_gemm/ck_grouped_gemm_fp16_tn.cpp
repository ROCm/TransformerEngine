/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#include "ck_grouped_gemm_fp16_impl.h"

namespace transformer_engine {
namespace grouped_gemm {

bool ck_tile_grouped_gemm_fp16_dispatch_tn(DType a_dtype, DType d_dtype,
                                           bool need_m_pad, bool need_k_pad,
                                           const GroupedGemmRunContext& ctx) {
  const ck_tile::stream_config s{ctx.stream};
  std::unique_ptr<RunnerInterface> runner = nullptr;

  using ALayout = ColMajor;
  using BLayout = RowMajor;

  TRANSFORMER_ENGINE_TYPE_SWITCH_16BIT(a_dtype, a_te_type, {
    using AType = typename TETypeToCKType<a_te_type>::type;
    using BType = typename TETypeToCKType<a_te_type>::type;
    using CLayout = RowMajor;

    TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(d_dtype, d_te_type, {
      using CType = typename TETypeToCKType<d_te_type>::type;

      TRANSFORMER_ENGINE_SWITCH_CONDITION(need_m_pad, kPadM, {
        TRANSFORMER_ENGINE_SWITCH_CONDITION(need_k_pad, kPadK, {
          if (ctx.N % 256 == 0) {
            MAKE_RUNNER(TileCfg_256x256x64, kPadM, false, kPadK);
          } else if (ctx.N % 128 == 0) {
            MAKE_RUNNER(TileCfg_256x128x64, kPadM, false, kPadK);
          } else {
            MAKE_RUNNER(TileCfg_256x128x64, kPadM, true, kPadK);
          }
        });
      });
    });
  });

  if (!runner) {
    return false;
  }
  return runner->run(s, ctx);
}

#undef MAKE_RUNNER

}  // namespace grouped_gemm
}  // namespace transformer_engine
