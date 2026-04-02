/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#include "ck_grouped_gemm_common.h"
#include "ck_grouped_gemm_fp16_impl.h"

namespace transformer_engine {
namespace grouped_gemm {

#define MAKE_RUNNER(TileCfg_)                                          \
    TRANSFORMER_ENGINE_SWITCH_CONDITION(ctx.accumulate, accum_option, {\
        using Runner = GroupedGemmRunner<                               \
            AType, BType, CType,                                       \
            ALayout, BLayout, CLayout,                                 \
            TileCfg_, accum_option>;                                   \
        runner = std::make_unique<Runner>();                           \
    })

template <typename teA, typename teB, typename ALayout, typename BLayout>
static std::unique_ptr<RunnerInterface> make_fp16_runner_typed(DType d_dtype, const GroupedGemmRunContext& ctx) {
  std::unique_ptr<RunnerInterface> runner = nullptr;
  using AType = typename TETypeToCKType<teA>::type;
  using BType = typename TETypeToCKType<teB>::type;
  using CLayout = RowMajor;

  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(d_dtype, d_te_type, {
      using CType = typename TETypeToCKType<d_te_type>::type;
      
      if (ctx.N % 256 == 0) {
          MAKE_RUNNER(TileCfg_256x256x64);
      } else if (ctx.N % 128 == 0) {
          MAKE_RUNNER(TileCfg_256x128x64);
      } else {
          MAKE_RUNNER(TileCfg_256x128x64_padding);
      }
  });
  return runner;
}

#undef MAKE_RUNNER

static std::unique_ptr<RunnerInterface> 
make_fp16_runner(DType a_dtype,
                 DType b_dtype,
                 DType d_dtype,
                 const GroupedGemmRunContext& ctx) {

    TRANSFORMER_ENGINE_SWITCH_CONDITION(ctx.transA, kTransA, {
        using ALayout = std::conditional_t<kTransA, ColMajor, RowMajor>;

        TRANSFORMER_ENGINE_SWITCH_CONDITION(ctx.transB, kTransB, {
            using BLayout = std::conditional_t<kTransB, ColMajor, RowMajor>;

            TRANSFORMER_ENGINE_TYPE_SWITCH_16BIT(a_dtype, a_type, {
                return make_fp16_runner_typed<a_type, a_type, ALayout, BLayout>(d_dtype, ctx);
            });
        });
    });

    return nullptr;
}

bool ck_tile_grouped_gemm_fp16_dispatch(DType a_dtype,
                                       DType b_dtype,
                                       DType d_dtype,
                                       const GroupedGemmRunContext& ctx) {
    const ck_tile::stream_config s{ctx.stream};

    auto runner = make_fp16_runner(
        a_dtype, b_dtype, d_dtype, ctx);

    if (!runner) {
        return false;
    }

    return runner->run(s, ctx);
}

}  // namespace grouped_gemm
}  // namespace transformer_engine
