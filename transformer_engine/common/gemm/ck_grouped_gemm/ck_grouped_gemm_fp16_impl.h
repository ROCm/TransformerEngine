/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#pragma once

#include "ck_grouped_gemm_common.h"

namespace transformer_engine {
namespace grouped_gemm {

// -------------------------
// Tile configs: FP16/BF16
// -------------------------

struct TileCfg_256x256x64 {
  static constexpr ck_tile::index_t M_Tile = 256;
  static constexpr ck_tile::index_t N_Tile = 256;
  static constexpr ck_tile::index_t K_Tile = 64;

  static constexpr ck_tile::index_t M_Warp = 2;
  static constexpr ck_tile::index_t N_Warp = 2;
  static constexpr ck_tile::index_t K_Warp = 1;

  static constexpr ck_tile::index_t M_Warp_Tile = 32;
  static constexpr ck_tile::index_t N_Warp_Tile = 32;
  static constexpr ck_tile::index_t K_Warp_Tile = 16;

  static constexpr bool kPadM = false;
  static constexpr bool kPadN = false;
  static constexpr bool kPadK = false;

  static constexpr bool DoubleSmemBuffer = false;

  static constexpr ck_tile::index_t TilePartitionerGroupNum = 8;
  static constexpr ck_tile::index_t TilePartitionerM01 = 4;
};

struct TileCfg_256x128x64 : TileCfg_256x256x64 {
  static constexpr ck_tile::index_t N_Tile = 128;
};

template <typename Base, bool PadM_, bool PadN_, bool PadK_>
struct WithPadding : Base {
  static constexpr bool kPadM = PadM_;
  static constexpr bool kPadN = PadN_;
  static constexpr bool kPadK = PadK_;
};

template <typename AType,
          typename BType,
          typename CType,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          typename TileCfg,
          bool Accumulate,
          typename AccType = float>
class GroupedGemmRunner : public RunnerInterface {
 public:
  using GemmShape = GroupedGemmShape<TileCfg>;
  using Partitioner = GroupedGemmPartitioner<TileCfg>;

  using UniversalTraits =
      ck_tile::PersistentTileGemmUniversalTraits<TileCfg::kPadM,
                                                 TileCfg::kPadN,
                                                 TileCfg::kPadK,
                                                 TileCfg::DoubleSmemBuffer,
                                                 ALayout,
                                                 BLayout,
                                                 CLayout>;

  static constexpr ck_tile::GemmPipelineScheduler Scheduler =
      ck_tile::GemmPipelineScheduler::Intrawave;

  using Problem =
      ck_tile::UniversalGemmPipelineProblem<AType,
                                            BType,
                                            AccType,
                                            GemmShape,
                                            UniversalTraits,
                                            Scheduler>;

  using Pipeline = ck_tile::GemmPipelineAgBgCrCompV3<Problem>;

  using ET = EpilogueTraits<CType, CLayout, Accumulate>;

  using Epilogue =
      ck_tile::CShuffleEpilogue<ck_tile::CShuffleEpilogueProblem<
          AType,
          BType,
          typename ET::DsDataType,
          AccType,
          CType,
          typename ET::DsLayout,
          CLayout,
          typename ET::ElemOp,
          Partitioner::MPerBlock,
          Partitioner::NPerBlock,
          TileCfg::M_Warp,
          TileCfg::N_Warp,
          TileCfg::M_Warp_Tile,
          TileCfg::N_Warp_Tile,
          TileCfg::K_Warp_Tile,
          Problem::TransposeC>>;

  using Kernel = ck_tile::GroupedGemmKernel<Partitioner, Pipeline, Epilogue>;

  // GroupedGemmHostArgs<1> for the MultiD accumulate path, <0> for the overwrite path.
  using HostArgs = std::conditional_t<Accumulate,
                                      ck_tile::GroupedGemmHostArgs<1>,
                                      ck_tile::GroupedGemmHostArgs<0>>;

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

      const auto& a = data_view(*A_te);

      const transformer_engine::SimpleTensor* b_src = nullptr;
      if (ctx.use_b_columnwise_data) {
        b_src = &B_te->columnwise_data;
      } else {
        b_src = &B_te->data;
      }
      const auto& b = *b_src;

      const auto& d = data_view(*D_te);

      int64_t Ad0 = 0, Ad1 = 0, Bd0 = 0, Bd1 = 0, Dd0 = 0, Dd1 = 0;
      if (!get_flat_2d_dims(*A_te, Ad0, Ad1)) {
        NVTE_ERROR("ck_tile_grouped_gemm: expected rank>=2 for A in group ", i);
      }
      if (ctx.use_b_columnwise_data) {
        if (!get_columnwise_storage_2d_dims(B_te->columnwise_data, Bd0, Bd1)) {
          NVTE_ERROR("ck_tile_grouped_gemm: expected 2D columnwise_data for B in group ", i);
        }
      } else {
        if (!get_flat_2d_dims(*B_te, Bd0, Bd1)) {
          NVTE_ERROR("ck_tile_grouped_gemm: expected rank>=2 for B in group ", i);
        }
      }
      if (!get_flat_2d_dims(*D_te, Dd0, Dd1)) {
        NVTE_ERROR("ck_tile_grouped_gemm: expected rank>=2 for D in group ", i);
      }

      const int64_t M = ctx.transA ? Ad1 : Ad0;
      const int64_t K = ctx.transA ? Ad0 : Ad1;
      const int64_t N = ctx.transB ? Bd0 : Bd1;
      const int64_t Kb = ctx.transB ? Bd1 : Bd0;

      if (Kb != K) {
        NVTE_ERROR("ck_tile_grouped_gemm: K mismatch between A and B in group ", i);
      }

      if (Dd0 != M || Dd1 != N) {
        NVTE_ERROR("ck_tile_grouped_gemm: D shape mismatch in group ", i);
      }

      const ck_tile::index_t stride_A = static_cast<ck_tile::index_t>(Ad1);
      const ck_tile::index_t stride_B = static_cast<ck_tile::index_t>(Bd1);
      const ck_tile::index_t stride_E = static_cast<ck_tile::index_t>(Dd1);

      if constexpr (Accumulate) {
        descs.emplace_back(a.dptr,
                           b.dptr,
                           std::array<const void*, 1>{d.dptr},
                           d.dptr,
                           1,
                           M,
                           N,
                           K,
                           stride_A,
                           stride_B,
                           std::array<ck_tile::index_t, 1>{stride_E},
                           stride_E);
      } else {
        descs.emplace_back(a.dptr,
                           b.dptr,
                           std::array<const void*, 0>{},
                           d.dptr,
                           1,
                           M,
                           N,
                           K,
                           stride_A,
                           stride_B,
                           std::array<ck_tile::index_t, 0>{},
                           stride_E);
      }
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

#define MAKE_RUNNER(BaseCfg_, kPadM_, kPadN_, kPadK_)                  \
  TRANSFORMER_ENGINE_SWITCH_CONDITION(ctx.accumulate, accum_option, {  \
    using Runner = GroupedGemmRunner<AType,                            \
                                     BType,                            \
                                     CType,                            \
                                     ALayout,                          \
                                     BLayout,                          \
                                     CLayout,                          \
                                     WithPadding<BaseCfg_, kPadM_, kPadN_, kPadK_>, \
                                     accum_option>;                    \
    runner = std::make_unique<Runner>();                               \
  })

// Templated dispatch on A/B layouts, shared by all layout-specific .cpp files.
template <typename ALayout, typename BLayout>
bool ck_tile_grouped_gemm_fp16_dispatch_layout(DType a_dtype, DType d_dtype,
                                               bool need_m_pad, bool need_k_pad,
                                               const GroupedGemmRunContext& ctx) {
  const ck_tile::stream_config s{ctx.stream};
  std::unique_ptr<RunnerInterface> runner = nullptr;

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

// Per-layout dispatch function signature.
// Each layout file (NN, NT, TN, TT) implements one of these.
bool ck_tile_grouped_gemm_fp16_dispatch_nn(DType a_dtype, DType d_dtype,
                                           bool need_m_pad, bool need_k_pad,
                                           const GroupedGemmRunContext& ctx);
bool ck_tile_grouped_gemm_fp16_dispatch_nt(DType a_dtype, DType d_dtype,
                                           bool need_m_pad, bool need_k_pad,
                                           const GroupedGemmRunContext& ctx);
bool ck_tile_grouped_gemm_fp16_dispatch_tn(DType a_dtype, DType d_dtype,
                                           bool need_m_pad, bool need_k_pad,
                                           const GroupedGemmRunContext& ctx);
bool ck_tile_grouped_gemm_fp16_dispatch_tt(DType a_dtype, DType d_dtype,
                                           bool need_m_pad, bool need_k_pad,
                                           const GroupedGemmRunContext& ctx);

}  // namespace grouped_gemm
}  // namespace transformer_engine
