/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#pragma once

#include <hip/hip_runtime.h>

#include <array>
#include <type_traits>
#include <vector>
#include <memory>

#include <transformer_engine/transformer_engine.h>
#include "../ck_gemm_common.h"

namespace transformer_engine {
namespace grouped_gemm {

template <typename TileCfg>
using GroupedGemmShape = ck_tile::TileGemmShape<
    ck_tile::sequence<TileCfg::M_Tile, TileCfg::N_Tile, TileCfg::K_Tile>,
    ck_tile::sequence<TileCfg::M_Warp, TileCfg::N_Warp, TileCfg::K_Warp>,
    ck_tile::sequence<TileCfg::M_Warp_Tile, TileCfg::N_Warp_Tile, TileCfg::K_Warp_Tile>>;

template <typename TileCfg>
using GroupedGemmPartitioner = ck_tile::GemmSpatiallyLocalTilePartitioner<
    GroupedGemmShape<TileCfg>,
    TileCfg::TilePartitionerGroupNum,
    TileCfg::TilePartitionerM01>;

// Selects epilogue traits based on whether we are accumulating (D += A*B) or not (D = A*B).
// For accumulate=true, the existing D buffer is passed as a MultiD input tensor and combined
// via element_wise::Add. For accumulate=false, no extra input is needed and PassThrough is used.
template <typename CType, typename CLayout, bool Accumulate>
struct EpilogueTraits {
  using DsDataType = ck_tile::tuple<>;
  using DsLayout   = ck_tile::tuple<>;
  using ElemOp     = ck_tile::element_wise::PassThrough;
};

template <typename CType, typename CLayout>
struct EpilogueTraits<CType, CLayout, true> {
  using DsDataType = ck_tile::tuple<CType>;
  using DsLayout   = ck_tile::tuple<CLayout>;
  using ElemOp     = ck_tile::element_wise::Add;
};

template <typename Kernel, typename DescContainer>
static inline bool launch_grouped_gemm_kernel(const DescContainer& descs,
                                              const CKGemmRunContext& ctx,
                                              const ck_tile::stream_config& stream_cfg) {
  constexpr int kBlockPerCu = 1;

  const dim3 blocks = Kernel::BlockSize();
  const dim3 grids  = Kernel::GridSize(descs);
  auto kargs = Kernel::MakeKargs(descs);

  if (!Kernel::IsSupportedArgument(kargs)) {
    NVTE_WARN("ck_tile_grouped_gemm: CK_Tile kernel arguments not supported for this config. "
              "Falling back.");
    return false;
  }

  NVTE_CHECK_CUDA(hipMemcpyAsync(ctx.workspace,
                                  kargs.data(),
                                  kargs.size() * sizeof(typename decltype(kargs)::value_type),
                                  hipMemcpyHostToDevice,
                                  ctx.stream));

  ck_tile::launch_kernel(
      stream_cfg, ck_tile::make_kernel<kBlockPerCu>(
                      Kernel{}, grids, blocks, 0,
                      ck_tile::cast_pointer_to_constant_address_space(ctx.workspace),
                      ctx.group_num));
  return true;
}

class RunnerInterface {
public:
    virtual ~RunnerInterface() = default;
    virtual bool run(const ck_tile::stream_config& stream_cfg,
                     const CKGemmRunContext& ctx) = 0;
};

}  // namespace grouped_gemm
}  // namespace transformer_engine
