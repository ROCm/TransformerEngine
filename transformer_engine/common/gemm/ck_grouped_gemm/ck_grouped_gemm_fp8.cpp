/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#include "ck_grouped_gemm_common.h"
#include "ck_grouped_gemm_fp8_impl.h"
#include "common/util/cuda_runtime.h"

namespace transformer_engine {
namespace grouped_gemm {

enum class GPUArch {
  GFX942,
  GFX950,
  UNKNOWN
};

static inline GPUArch detect_gpu_arch() {
  int arch = cuda::sm_arch(0);

  if (arch == 94) {
    return GPUArch::GFX942;
  }
  if (arch == 95) {
    return GPUArch::GFX950;
  }
  return GPUArch::UNKNOWN;
}

template <GPUArch Arch>
struct FP8TileCfg;

template <>
struct FP8TileCfg<GPUArch::GFX942> {
  using type = TileCfg_128x128x128_32x32x16_2x2x1;
};

template <>
struct FP8TileCfg<GPUArch::GFX950> {
  using type = TileCfg_128x128x128_16x16x128_2x2x1;
};

template <GPUArch Arch, typename teA, typename teB, typename ALayout, typename BLayout>
static std::unique_ptr<RunnerInterface> make_fp8_runner_typed(DType d_dtype,
                                                       const GroupedGemmRunContext& ctx) {
  std::unique_ptr<RunnerInterface> runner = nullptr;

  using AType = typename TETypeToCKType<teA>::type;
  using BType = typename TETypeToCKType<teB>::type;
  using CTypeLayout = RowMajor;
  using TileCfg = typename FP8TileCfg<Arch>::type;

  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(d_dtype, d_te_type, {
    using CType = typename TETypeToCKType<d_te_type>::type;
    using Runner = QuantGroupedGemmRunner<
        AType, BType, CType,
        ALayout, BLayout, CTypeLayout,
        TileCfg, ck_tile::memory_operation_enum::set>;
    runner = std::make_unique<Runner>();
  });

  return runner;
}

template <GPUArch Arch>
static std::unique_ptr<RunnerInterface> make_fp8_runner_impl(DType a_dtype,
                                                      DType b_dtype,
                                                      DType d_dtype,
                                                      const GroupedGemmRunContext& ctx) {

    TRANSFORMER_ENGINE_SWITCH_CONDITION(ctx.transA, kTransA, {
        using ALayout = std::conditional_t<kTransA, ColMajor, RowMajor>;

        TRANSFORMER_ENGINE_SWITCH_CONDITION(ctx.transB, kTransB, {
            using BLayout = std::conditional_t<kTransB, ColMajor, RowMajor>;

            TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(a_dtype, a_type, {
                TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(b_dtype, b_type, {
                    return make_fp8_runner_typed<Arch, a_type, b_type, ALayout, BLayout>(
                        d_dtype, ctx);
                });
            });
        });
    });

    return nullptr;
}

static inline std::unique_ptr<RunnerInterface> make_fp8_runner_gfx942(DType a_dtype,
                                                        DType b_dtype,
                                                        DType d_dtype,
                                                        const GroupedGemmRunContext& ctx) {
  return make_fp8_runner_impl<GPUArch::GFX942>(a_dtype, b_dtype, d_dtype, ctx);
}

static inline std::unique_ptr<RunnerInterface> make_fp8_runner_gfx950(DType a_dtype,
                                                        DType b_dtype,
                                                        DType d_dtype,
                                                        const GroupedGemmRunContext& ctx) {
  return make_fp8_runner_impl<GPUArch::GFX950>(a_dtype, b_dtype, d_dtype, ctx);
}

static std::unique_ptr<RunnerInterface> 
make_fp8_runner(DType a_dtype,
                DType b_dtype,
                DType d_dtype,
                const GroupedGemmRunContext& ctx) {
  switch (detect_gpu_arch()) {
    case GPUArch::GFX942:
      return make_fp8_runner_gfx942(a_dtype, b_dtype, d_dtype, ctx);
    case GPUArch::GFX950:
      return make_fp8_runner_gfx950(a_dtype, b_dtype, d_dtype, ctx);
    default:
      NVTE_ERROR("ck_tile_grouped_gemm: available architectures = {gfx942, gfx950}");
      return nullptr;
  }
}

bool ck_tile_grouped_gemm_fp8_dispatch(DType a_dtype,
                                       DType b_dtype,
                                       DType d_dtype,
                                       const GroupedGemmRunContext& ctx) {
  const ck_tile::stream_config s{ctx.stream};
  auto runner = make_fp8_runner(a_dtype, b_dtype, d_dtype, ctx);
  if (!runner) {
    return false;
  }
  return runner->run(s, ctx);
}

}  // namespace grouped_gemm
}  // namespace transformer_engine
