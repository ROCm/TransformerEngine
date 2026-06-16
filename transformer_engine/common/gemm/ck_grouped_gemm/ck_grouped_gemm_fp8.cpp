/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#include "ck_grouped_gemm_common.h"
#include "ck_grouped_gemm_fp8.h"
#include "common/util/cuda_runtime.h"

#include "ck_tile/ops/gemm_quant/kernel/grouped_gemm_quant_kernel.hpp"
#include "ck_tile/ops/gemm_quant/pipeline/gemm_group_quant_utils.hpp"
#include "ck_tile/ops/gemm_quant/pipeline/gemm_quant_pipeline_problem.hpp"
#include "ck_tile/ops/gemm_quant/pipeline/tile_gemm_quant_traits.hpp"

namespace transformer_engine {
namespace grouped_gemm {


struct TileCfg_256x256x128_16x16x128_2x2x1 {
  static constexpr ck_tile::index_t M_Tile = 256;
  static constexpr ck_tile::index_t N_Tile = 256;
  static constexpr ck_tile::index_t K_Tile = 128;

  static constexpr ck_tile::index_t M_Warp = 2;
  static constexpr ck_tile::index_t N_Warp = 2;
  static constexpr ck_tile::index_t K_Warp = 1;

  static constexpr ck_tile::index_t M_Warp_Tile = 16;
  static constexpr ck_tile::index_t N_Warp_Tile = 16;
  static constexpr ck_tile::index_t K_Warp_Tile = 128;

  static constexpr bool kPadM = false;
  static constexpr bool kPadN = false;
  static constexpr bool kPadK = false;

  static constexpr bool DoubleSmemBuffer = false;

  static constexpr ck_tile::index_t TilePartitionerGroupNum = 16;
  static constexpr ck_tile::index_t TilePartitionerM01 = 8;
};

struct TileCfg_128x128x128_16x16x64_2x2x1 {
  static constexpr ck_tile::index_t M_Tile = 128;
  static constexpr ck_tile::index_t N_Tile = 128;
  static constexpr ck_tile::index_t K_Tile = 128;

  static constexpr ck_tile::index_t M_Warp = 2;
  static constexpr ck_tile::index_t N_Warp = 2;
  static constexpr ck_tile::index_t K_Warp = 1;

  static constexpr ck_tile::index_t M_Warp_Tile = 16;
  static constexpr ck_tile::index_t N_Warp_Tile = 16;
  static constexpr ck_tile::index_t K_Warp_Tile = 64;

  static constexpr bool kPadM = false;
  static constexpr bool kPadN = false;
  static constexpr bool kPadK = false;

  static constexpr bool DoubleSmemBuffer = false;

  static constexpr ck_tile::index_t TilePartitionerGroupNum = 16;
  static constexpr ck_tile::index_t TilePartitionerM01 = 8;
};

struct TileCfg_128x128x128_16x16x128_2x2x1
    : TileCfg_256x256x128_16x16x128_2x2x1 {
  static constexpr ck_tile::index_t M_Tile = 128;
  static constexpr ck_tile::index_t N_Tile = 128;
};

struct TileCfg_256x256x128_16x16x128_2x2x1_kpad
    : TileCfg_256x256x128_16x16x128_2x2x1 {
  static constexpr bool kPadK = true;
};

struct TileCfg_128x128x128_16x16x128_2x2x1_kpad
    : TileCfg_128x128x128_16x16x128_2x2x1 {
  static constexpr bool kPadK = true;
};

struct TileCfg_128x128x128_16x16x128_2x2x1_npad
    : TileCfg_128x128x128_16x16x128_2x2x1 {
  static constexpr bool kPadN = true;
};

struct TileCfg_128x128x128_16x16x128_2x2x1_nkpad
    : TileCfg_128x128x128_16x16x128_2x2x1 {
  static constexpr bool kPadN = true;
  static constexpr bool kPadK = true;
};

// gfx950 device compilation cannot instantiate the literal 32x32x16 FP8 tile
// configuration due to an unsupported warp GEMM dispatcher configuration.
// See: ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp for supported variants.
//
// To preserve the existing type name in shared template code, this struct
// inherits from the gfx950-safe 128x128x128 16x16x128 configuration in the
// gfx950 device compilation path, effectively reusing those parameters without
// redefining them.
//
// In all other compilation paths, the struct overrides the relevant fields to
// provide the intended 32x32x16 configuration.
#if defined(__gfx950__)
struct TileCfg_128x128x128_32x32x16_2x2x1
    : TileCfg_128x128x128_16x16x128_2x2x1 {};
#else
struct TileCfg_128x128x128_32x32x16_2x2x1
    : TileCfg_128x128x128_16x16x128_2x2x1 {
  static constexpr ck_tile::index_t M_Warp_Tile = 32;
  static constexpr ck_tile::index_t N_Warp_Tile = 32;
  static constexpr ck_tile::index_t K_Warp_Tile = 16;

  static constexpr ck_tile::index_t TilePartitionerGroupNum = 8;
  static constexpr ck_tile::index_t TilePartitionerM01 = 4;
};
#endif

// FP8 currently supports overwrite only.
// Preserve MemOp here for a future accumulate path.
template <typename AType,
          typename BType,
          typename CType,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          typename TileCfg,
          ck_tile::memory_operation_enum MemOp,
          typename AccType = float>
class QuantGroupedGemmRunner : public RunnerInterface {
 public:
  static constexpr ck_tile::QuantType QuantMode = ck_tile::QuantType::TensorQuant;

  using GemmShape = GroupedGemmShape<TileCfg>;
  using Partitioner = GroupedGemmPartitioner<TileCfg>;

  using AQLayout = RowMajor;
  using BQLayout = RowMajor;

  using UniversalTraits =
      ck_tile::TileGemmQuantTraits<TileCfg::kPadM,
                                   TileCfg::kPadN,
                                   TileCfg::kPadK,
                                   false,
                                   false,
                                   false,
                                   ALayout,
                                   BLayout,
                                   CLayout,
                                   QuantMode,
                                   AQLayout,
                                   BQLayout,
                                   false,
                                   TileCfg::DoubleSmemBuffer,
                                   false>;

  using Problem =
      ck_tile::GemmRowColTensorQuantPipelineProblem<AType,
                                                    BType,
                                                    AccType,
                                                    AccType,
                                                    GemmShape,
                                                    UniversalTraits,
                                                    false,
                                                    AccType>;

  using Pipeline = ck_tile::GemmPipelineAgBgCrCompV3<Problem>;

  using Epilogue =
      ck_tile::CShuffleEpilogue<ck_tile::CShuffleEpilogueProblem<
          AType,
          BType,
          ck_tile::tuple<>,
          AccType,
          CType,
          ck_tile::tuple<>,
          CLayout,
          ck_tile::element_wise::PassThrough,
          Partitioner::MPerBlock,
          Partitioner::NPerBlock,
          TileCfg::M_Warp,
          TileCfg::N_Warp,
          TileCfg::M_Warp_Tile,
          TileCfg::N_Warp_Tile,
          TileCfg::K_Warp_Tile,
          Problem::TransposeC>>;

  using Kernel =
      ck_tile::QuantGroupedGemmKernel<Partitioner, Pipeline, Epilogue, QuantMode>;
  using HostArgs = ck_tile::QuantGroupedGemmHostArgs;

 public:
  static std::vector<HostArgs> build_descs(const GroupedGemmRunContext& ctx) {
    if (!has_sufficient_workspace<Kernel>(ctx)) {
      return {};
    }

    std::vector<HostArgs> descs;
    descs.reserve(ctx.group_num);

    for (int i = 0; i < ctx.group_num; ++i) {
      const transformer_engine::Tensor* const A_te =
          transformer_engine::convertNVTETensorCheck(ctx.A[i]);
      const transformer_engine::Tensor* const B_te =
          transformer_engine::convertNVTETensorCheck(ctx.B[i]);
      transformer_engine::Tensor* D_te =
          transformer_engine::convertNVTETensorCheck(ctx.D[i]);

      const transformer_engine::SimpleTensor* a_src = nullptr;
      if (ctx.use_a_columnwise_data) {
        NVTE_CHECK(A_te->has_columnwise_data(), "ck_tile_grouped_gemm: ctx.use_a_columnwise_data=true but columnwise_data is absent.");
        a_src = &A_te->columnwise_data;
      } else {
        a_src = &A_te->data;
      }

      const auto& a = *a_src;
      const auto& d = data_view(*D_te);

      const transformer_engine::SimpleTensor* b_src = nullptr;
      if (ctx.use_b_columnwise_data) {
        NVTE_CHECK(B_te->has_columnwise_data(), "ck_tile_grouped_gemm: ctx.use_b_columnwise_data=true but columnwise_data is absent.");
        b_src = &B_te->columnwise_data;
      } else {
        b_src = &B_te->data;
      }

      const auto& b = *b_src;

      int64_t Ad0 = 0, Ad1 = 0, Bd0 = 0, Bd1 = 0, Dd0 = 0, Dd1 = 0;

      if (ctx.use_a_columnwise_data) {
        if (!get_columnwise_storage_2d_dims(A_te->columnwise_data, Ad0, Ad1)) {
          NVTE_ERROR("ck_tile_grouped_gemm: expected 2D columnwise_data for A in group ", i);
        }
      } else {
        if (!get_flat_2d_dims(*A_te, Ad0, Ad1)) {
          NVTE_ERROR("ck_tile_grouped_gemm: expected rank>=2 for normalized A in group ", i);
        }
      }

      if (ctx.use_b_columnwise_data) {
        if (!get_columnwise_storage_2d_dims(B_te->columnwise_data, Bd0, Bd1)) {
          NVTE_ERROR("ck_tile_grouped_gemm: expected 2D columnwise_data for B in group ", i);
        }
      } else {
        if (!get_flat_2d_dims(*B_te, Bd0, Bd1)) {
          NVTE_ERROR("ck_tile_grouped_gemm: expected rank>=2 for normalized B in group ", i);
        }
      }

      if (!get_flat_2d_dims(*D_te, Dd0, Dd1)) {
        NVTE_ERROR("ck_tile_grouped_gemm: expected rank>=2 for normalized D in group ", i);
      }

      const int64_t M = ctx.transA ? Ad1 : Ad0;
      const int64_t K = ctx.transA ? Ad0 : Ad1;
      const int64_t N = ctx.transB ? Bd0 : Bd1;
      const int64_t Kb = ctx.transB ? Bd1 : Bd0;

      if (Kb != K) {
        NVTE_ERROR("ck_tile_grouped_gemm: K mismatch between A and B in group ", i,
                   ". op(A)=", M, "x", K, ", op(B)=", Kb, "x", N);
      }

      if (Dd0 != M || Dd1 != N) {
        NVTE_ERROR("ck_tile_grouped_gemm: D shape mismatch in group ", i,
                   ". D=", Dd0, "x", Dd1, ", expected=", M, "x", N);
      }

      const ck_tile::index_t stride_A = static_cast<ck_tile::index_t>(Ad1);
      const ck_tile::index_t stride_B = static_cast<ck_tile::index_t>(Bd1);
      const ck_tile::index_t stride_E = static_cast<ck_tile::index_t>(Dd1);

      ck_tile::index_t AQK = 1;
      ck_tile::index_t BQK = 1;
      ck_tile::index_t stride_AQ = 1;
      ck_tile::index_t stride_BQ = 1;

      const auto& aq = scale_inv_view(*A_te);
      const auto& bq = scale_inv_view(*B_te);

      descs.emplace_back(a.dptr,
                         b.dptr,
                         d.dptr,
                         aq.dptr,
                         bq.dptr,
                         1,
                         M,
                         N,
                         K,
                         AQK,
                         BQK,
                         stride_A,
                         stride_B,
                         stride_E,
                         stride_AQ,
                         stride_BQ);
    }

    return descs;
  }

  bool run(const ck_tile::stream_config& stream_cfg,
           const GroupedGemmRunContext& ctx) override {
    auto descs = build_descs(ctx);
    if (descs.empty()) {
      return false;
    }

    return launch_grouped_gemm_kernel<Kernel>(descs, ctx, stream_cfg);
  }
};

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

template <>
struct FP8TileCfg<GPUArch::GFX1250> {
  using type = TileCfg_128x128x128_16x16x64_2x2x1;
};

struct FP8GroupedShapeAlignment {
  bool all_n_256_aligned = true;
  bool all_n_128_aligned = true;
  bool all_k_128_aligned = true;
};

static FP8GroupedShapeAlignment get_fp8_grouped_shape_alignment(
    const GroupedGemmRunContext& ctx) {
  FP8GroupedShapeAlignment alignment;

  for (int i = 0; i < ctx.group_num; ++i) {
    const transformer_engine::Tensor* const A_te =
        transformer_engine::convertNVTETensorCheck(ctx.A[i]);
    const transformer_engine::Tensor* const B_te =
        transformer_engine::convertNVTETensorCheck(ctx.B[i]);

    int64_t Ad0 = 0, Ad1 = 0, Bd0 = 0, Bd1 = 0;

    if (ctx.use_a_columnwise_data) {
      if (!get_columnwise_storage_2d_dims(A_te->columnwise_data, Ad0, Ad1)) {
        NVTE_ERROR("ck_tile_grouped_gemm: expected 2D columnwise_data for A in group ", i);
      }
    } else {
      if (!get_flat_2d_dims(*A_te, Ad0, Ad1)) {
        NVTE_ERROR("ck_tile_grouped_gemm: expected rank>=2 for normalized A in group ", i);
      }
    }

    if (ctx.use_b_columnwise_data) {
      if (!get_columnwise_storage_2d_dims(B_te->columnwise_data, Bd0, Bd1)) {
        NVTE_ERROR("ck_tile_grouped_gemm: expected 2D columnwise_data for B in group ", i);
      }
    } else {
      if (!get_flat_2d_dims(*B_te, Bd0, Bd1)) {
        NVTE_ERROR("ck_tile_grouped_gemm: expected rank>=2 for normalized B in group ", i);
      }
    }

    const int64_t K = ctx.transA ? Ad0 : Ad1;
    const int64_t N = ctx.transB ? Bd0 : Bd1;

    if (N % 256 != 0) {
      alignment.all_n_256_aligned = false;
    }
    if (N % 128 != 0) {
      alignment.all_n_128_aligned = false;
    }
    if (K % 128 != 0) {
      alignment.all_k_128_aligned = false;
    }

    if (!alignment.all_n_256_aligned &&
        !alignment.all_n_128_aligned &&
        !alignment.all_k_128_aligned) {
      break;
    }
  }

  return alignment;
}

#define MAKE_FP8_RUNNER(TileCfg_)                                      \
  using Runner = QuantGroupedGemmRunner<AType,                     \
                                        BType,                     \
                                        CType,                     \
                                        ALayout,                   \
                                        BLayout,                   \
                                        CTypeLayout,               \
                                        TileCfg_,                  \
                                        ck_tile::memory_operation_enum::set>; \
  runner = std::make_unique<Runner>()

template <GPUArch Arch>
static bool ck_tile_grouped_gemm_fp8_dispatch_arch(DType a_dtype,
                                                   DType b_dtype,
                                                   DType d_dtype,
                                                   const GroupedGemmRunContext& ctx) {
  const ck_tile::stream_config s{ctx.stream};
  std::unique_ptr<RunnerInterface> runner = nullptr;

  using CTypeLayout = RowMajor;

  // FP8 grouped GEMM is only compiled for CK's preferred NT presentation:
  //   transA=false, transB=true
  // which maps to:
  //   ALayout=RowMajor, BLayout=ColMajor.
  //
  // The caller is responsible for rewriting other FP8 layouts into this form
  // using columnwise_data when needed. Reject anything that did not normalize
  // successfully so we do not instantiate unreachable/unsupported layout variants.
  if (ctx.transA || !ctx.transB) {
    return false;
  }

  using ALayout = RowMajor;
  using BLayout = ColMajor;

  TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(a_dtype, a_te_type, {
    using AType = typename TETypeToCKType<a_te_type>::type;

    TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(b_dtype, b_te_type, {
      using BType = typename TETypeToCKType<b_te_type>::type;

      TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(d_dtype, d_te_type, {
        using CType = typename TETypeToCKType<d_te_type>::type;

      if constexpr (Arch == GPUArch::GFX950) {
          const auto alignment = get_fp8_grouped_shape_alignment(ctx);

          if (alignment.all_n_256_aligned) {
            if (alignment.all_k_128_aligned) {
              MAKE_FP8_RUNNER(TileCfg_256x256x128_16x16x128_2x2x1);
            } else {
              MAKE_FP8_RUNNER(TileCfg_128x128x128_16x16x128_2x2x1_kpad);
            }
          } else if (alignment.all_n_128_aligned) {
            if (alignment.all_k_128_aligned) {
              MAKE_FP8_RUNNER(TileCfg_128x128x128_16x16x128_2x2x1);
            } else {
              MAKE_FP8_RUNNER(TileCfg_128x128x128_16x16x128_2x2x1_kpad);
            }
          } else if (alignment.all_k_128_aligned) {
            MAKE_FP8_RUNNER(TileCfg_128x128x128_16x16x128_2x2x1_npad);
          } else {
            MAKE_FP8_RUNNER(TileCfg_128x128x128_16x16x128_2x2x1_nkpad);
          }
        } else {
          using TileCfg = typename FP8TileCfg<Arch>::type;
          MAKE_FP8_RUNNER(TileCfg);
        }
      });
    });
  });

  if (!runner) {
    return false;
  }

  return runner->run(s, ctx);
}

#undef MAKE_FP8_RUNNER

bool ck_tile_grouped_gemm_fp8_dispatch(DType a_dtype,
                                       DType b_dtype,
                                       DType d_dtype,
                                       const GroupedGemmRunContext& ctx) {
  switch (detect_gpu_arch()) {
#if defined(__gfx942__)
    case GPUArch::GFX942:
      return ck_tile_grouped_gemm_fp8_dispatch_arch<GPUArch::GFX942>(a_dtype, b_dtype, d_dtype, ctx);
#endif
#if defined(__gfx950__)
    case GPUArch::GFX950:
      return ck_tile_grouped_gemm_fp8_dispatch_arch<GPUArch::GFX950>(a_dtype, b_dtype, d_dtype, ctx);
#endif
#if defined(__gfx1250__)
    case GPUArch::GFX1250:
      return ck_tile_grouped_gemm_fp8_dispatch_arch<GPUArch::GFX1250>(a_dtype, b_dtype, d_dtype, ctx);
#endif

    default:
      NVTE_ERROR("ck_tile_grouped_gemm: available architectures = {gfx942, gfx950, gfx1250}");
      return false;
  }
}

}  // namespace grouped_gemm
}  // namespace transformer_engine
