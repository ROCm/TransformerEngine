/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#include <hip/hip_runtime.h>

#include <stdexcept>
#include <vector>
#include <array>
#include <memory>

#include "../ck_gemm_common.h"

#include "ck_tile/host.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/gemm/kernel/grouped_gemm_kernel.hpp"
#include "ck_tile/ops/gemm_mx/kernel/gemm_mx_kernel.hpp"
#include "ck_tile/ops/gemm_mx/kernel/scale_pointer.hpp"
#include "ck_tile/ops/gemm_mx/pipeline/gemm_pipeline_ag_bg_cr_comp_async.hpp"

using ck_tile::address_space_enum;
using ck_tile::safe_underlying_type_t;

using ScaleType = ck_tile::e8m0_t;
using ScaleM = ck_tile::MXScalePointer<ScaleType, 1, 32>;
using ScaleN = ck_tile::MXScalePointer<ScaleType, 1, 32>;

static constexpr int ScaleBlockSize = 32;

struct MXGemmConfig
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 512;

    static constexpr ck_tile::index_t M_Warp = 1;
    static constexpr ck_tile::index_t N_Warp = 4;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = 128;

    static constexpr bool kPadM = false;
    static constexpr bool kPadN = false;
    static constexpr bool kPadK = false;

    static constexpr bool TransposeC            = false;
    static constexpr bool UseStructuredSparsity = false;

    static constexpr int kBlockPerCu                = 1;
    static constexpr int TilePartitionerGroupNum     = 8;
    static constexpr int TilePartitionerM01          = 4;
    static constexpr auto Scheduler                 = ck_tile::GemmPipelineScheduler::Intrawave;
    static constexpr ck_tile::index_t NumWaveGroups = 1;
    static constexpr bool DoubleSmemBuffer          = false;
    static constexpr bool Preshuffle                = false;

    static constexpr int N_Repeat          = N_Tile / N_Warp_Tile / N_Warp;
    static constexpr bool TiledMMAPermuteN = false;
};

struct MXfp8_GemmConfig_256x64x128 : MXGemmConfig
{
    static constexpr ck_tile::index_t M_Tile = 256;
    static constexpr ck_tile::index_t N_Tile = 64;
    static constexpr ck_tile::index_t K_Tile = 128;
};

// GEMM config with 16x16 warp tile
struct MXfp8_GemmConfig_64x64x256 : MXGemmConfig
{
    static constexpr ck_tile::index_t M_Tile = 64;
    static constexpr ck_tile::index_t N_Tile = 64;
    static constexpr ck_tile::index_t K_Tile = 256;
};


template <typename ScaleM, typename ScaleN, ck_tile::index_t NumDTensor = 0>
struct MXGroupedGemmHostArgs : public ck_tile::GroupedGemmHostArgs<NumDTensor>
{
    using Base = ck_tile::GroupedGemmHostArgs<NumDTensor>;

    CK_TILE_HOST explicit MXGroupedGemmHostArgs(const void* a_ptr_,
                                                ScaleM scale_m_,
                                                const void* b_ptr_,
                                                ScaleN scale_n_,
                                                const std::array<const void*, NumDTensor>& ds_ptr_,
                                                void* e_ptr_,
                                                ck_tile::index_t k_batch_,
                                                ck_tile::index_t M_,
                                                ck_tile::index_t N_,
                                                ck_tile::index_t K_,
                                                ck_tile::index_t stride_A_,
                                                ck_tile::index_t stride_B_,
                                                const std::array<ck_tile::index_t, NumDTensor>& stride_Ds_,
                                                ck_tile::index_t stride_E_)
        : Base(a_ptr_,
               b_ptr_,
               ds_ptr_,
               e_ptr_,
               k_batch_,
               M_,
               N_,
               K_,
               stride_A_,
               stride_B_,
               stride_Ds_,
               stride_E_),
          scale_m(scale_m_),
          scale_n(scale_n_)
    {
    }

    ScaleM scale_m;
    ScaleN scale_n;
};

template <typename Karg>
struct MXGemmTransKernelArg
{
    Karg group_karg;
    ck_tile::index_t block_start;
    ck_tile::index_t block_end;

    CK_TILE_HOST MXGemmTransKernelArg(Karg&& karg_, ck_tile::index_t block_start_, ck_tile::index_t block_end_)
        : group_karg(std::move(karg_)), block_start(block_start_), block_end(block_end_)
    {
    }
};

template <typename TilePartitioner_,
          typename MXGemmPipeline_,
          typename EpiloguePipeline_,
          typename ScaleM_,
          typename ScaleN_,
          ck_tile::index_t NumDTensor_ = 0>
struct MXGroupedGemmKernel
{
    using Base            = ck_tile::MXGemmKernel<TilePartitioner_, MXGemmPipeline_, EpiloguePipeline_>;
    using TilePartitioner = ck_tile::remove_cvref_t<TilePartitioner_>;
    using HostArgs        = MXGroupedGemmHostArgs<ScaleM_, ScaleN_, NumDTensor_>;
    using KernelArgs      = typename Base::template KernelArgs<ScaleM_, ScaleN_>;
    using TransKernelArg  = MXGemmTransKernelArg<KernelArgs>;

    static constexpr ck_tile::index_t kBlockSize  = Base::KernelBlockSize;
    static constexpr ck_tile::index_t kBlockPerCu = Base::kBlockPerCu;

    CK_TILE_HOST static auto BlockSize() -> dim3 { return Base::BlockSize(); }

    CK_TILE_HOST static auto GridSize(const std::vector<HostArgs>& descs) -> dim3
    {
        ck_tile::index_t grid_size = 0;
        for(const auto& d : descs)
            grid_size += TilePartitioner::GridSize(d.M, d.N);
        return dim3(grid_size, 1, 1);
    }

    CK_TILE_HOST static auto MakeKargs(const std::vector<HostArgs>& descs)
    {
        std::vector<TransKernelArg> out;
        out.reserve(descs.size());

        ck_tile::index_t grid_size = 0;
        for(const auto& g : descs)
        {
            if(g.M == 0 || g.N == 0 || g.K == 0)
                continue;

            const ck_tile::index_t grid_size_grp = TilePartitioner::GridSize(g.M, g.N);
            const ck_tile::index_t block_start   = grid_size;
            const ck_tile::index_t block_end     = grid_size + grid_size_grp;
            grid_size += grid_size_grp;

            auto karg = Base::MakeKernelArgs(std::array<const void*, 1>{g.a_ptr},
                                             std::array<const void*, 1>{g.b_ptr},
                                             std::array<const void*, NumDTensor_>{g.ds_ptr},
                                             g.e_ptr,
                                             g.k_batch,
                                             g.M,
                                             g.N,
                                             g.K,
                                             std::array<ck_tile::index_t, 1>{g.stride_A},
                                             std::array<ck_tile::index_t, 1>{g.stride_B},
                                             g.stride_Ds,
                                             g.stride_E,
                                             g.scale_m,
                                             g.scale_n);
            out.emplace_back(std::move(karg), block_start, block_end);
        }
        return out;
    }

    CK_TILE_HOST static bool IsSupportedArgument(const std::vector<TransKernelArg>& kargs)
    {
        for(const auto& k : kargs)
            if(!Base::IsSupportedArgument(k.group_karg))
                return false;
        return true;
    }

    CK_TILE_DEVICE ck_tile::index_t FindGroupId(const TransKernelArg* ptr,
                                       ck_tile::index_t block_id,
                                       ck_tile::index_t group_count) const
    {
        ck_tile::index_t left = 0;
        ck_tile::index_t right = group_count;
        ck_tile::index_t group_id = (left + right) >> 1;

        while((!(block_id >= ptr[group_id].block_start && block_id < ptr[group_id].block_end)) &&
              left <= right)
        {
            if(block_id < ptr[group_id].block_start)
                right = group_id;
            else
                left = group_id;
            group_id = (left + right) >> 1;
        }
        return group_id;
    }

    CK_TILE_DEVICE void operator()(const void CK_TILE_CONSTANT_ADDRESS_SPACE* descs_const,
                                   ck_tile::index_t group_count) const
    {
        const ck_tile::index_t block_id = ck_tile::get_block_1d_id();
        const auto ptr = reinterpret_cast<const TransKernelArg*>(
            ck_tile::cast_pointer_to_generic_address_space(descs_const));

        const ck_tile::index_t group_id = FindGroupId(ptr, block_id, group_count);
        const auto& kargs = ptr[group_id];
        const ck_tile::index_t local_partition_idx = block_id - kargs.block_start;
        Base{}(kargs.group_karg, local_partition_idx);
    }
};

using MXGroupedHostDesc = MXGroupedGemmHostArgs<ScaleM, ScaleN>;

template <typename ScaleType, bool KLast, int MNPack, int KPack, int XdlMNThread, int XdlKThread>
__global__ void pack_scales_mnxk_kernel(const ScaleType* __restrict__ src,
                                        int32_t* __restrict__ dst,
                                        int MN,
                                        int K_scale,
                                        int stride_dim0,
                                        int stride_dim1)
{
    const int MN_packed = MN / MNPack;
    const int K_packed  = K_scale / KPack;
    const int total     = MN_packed * K_packed;

    const int linear = blockIdx.x * blockDim.x + threadIdx.x;
    if(linear >= total)
        return;

    const int packed_mn = linear / K_packed;
    const int packed_k  = linear % K_packed;

    int32_t val         = 0;
    const int mn_lane   = packed_mn % XdlMNThread;
    const int mn_group  = packed_mn / XdlMNThread;
    const int k_lane    = packed_k % XdlKThread;
    const int k_group   = packed_k / XdlKThread;

    for(int ik = 0; ik < KPack; ++ik)
    {
        for(int imn = 0; imn < MNPack; ++imn)
        {
            const int byteIdx = ik * MNPack + imn;
            const int orig_mn = mn_group * XdlMNThread * MNPack + imn * XdlMNThread + mn_lane;
            const int orig_k  = k_group * XdlKThread * KPack + ik * XdlKThread + k_lane;

            ScaleType v{};
            if constexpr(KLast)
            {
                // src is logical [MN, K_scale]
                v = src[orig_mn * stride_dim0 + orig_k * stride_dim1];
            }
            else
            {
                // src is logical [K_scale, MN]
                v = src[orig_k * stride_dim0 + orig_mn * stride_dim1];
            }
            val |= (static_cast<int32_t>(v.get()) << (byteIdx * 8));
        }
    }

    dst[packed_mn * K_packed + packed_k] = val;
}

template <typename ScaleType, bool KLast, int MNPack, int KPack, int XdlMNThread, int XdlKThread>
void launch_pack_scales(const ScaleType* src,
                        int32_t* dst,
                        int MN,
                        int K_scale,
                        int stride_dim0,
                        int stride_dim1,
                        hipStream_t stream) {
  constexpr int threads = 256;
  const int total = (MN / MNPack) * (K_scale / KPack);
  const int blocks = (total + threads - 1) / threads;

  hipLaunchKernelGGL(
      (pack_scales_mnxk_kernel<ScaleType, KLast, MNPack, KPack, XdlMNThread, XdlKThread>),
      dim3(blocks),
      dim3(threads),
      0,
      stream,
      src,
      dst,
      MN,
      K_scale,
      stride_dim0,
      stride_dim1);

  NVTE_CHECK_CUDA(hipGetLastError());
}


struct GroupedScalePackJob {
  const ScaleType* src;
  int32_t* dst;
  int MN;
  int K_scale;
  int stride_dim0;
  int stride_dim1;
  int total;
};

template <typename ScaleT, bool KLast, int MNPack, int KPack, int XdlMNThread, int XdlKThread>
__global__ void grouped_pack_scales_mnxk_kernel(const GroupedScalePackJob* __restrict__ jobs) {
  const int group = blockIdx.y;
  const auto job = jobs[group];

  const int linear = blockIdx.x * blockDim.x + threadIdx.x;
  if (linear >= job.total) return;

  const int K_packed = job.K_scale / KPack;
  const int packed_mn = linear / K_packed;
  const int packed_k  = linear % K_packed;

  int32_t val        = 0;
  const int mn_lane  = packed_mn % XdlMNThread;
  const int mn_group = packed_mn / XdlMNThread;
  const int k_lane   = packed_k % XdlKThread;
  const int k_group  = packed_k / XdlKThread;

  for (int ik = 0; ik < KPack; ++ik) {
    for (int imn = 0; imn < MNPack; ++imn) {
      const int byteIdx = ik * MNPack + imn;
      const int orig_mn = mn_group * XdlMNThread * MNPack + imn * XdlMNThread + mn_lane;
      const int orig_k  = k_group * XdlKThread * KPack + ik * XdlKThread + k_lane;

      ScaleT v{};
      if constexpr (KLast) {
        // src is logical [MN, K_scale]
        v = job.src[orig_mn * job.stride_dim0 + orig_k * job.stride_dim1];
      } else {
        // src is logical [K_scale, MN]
        v = job.src[orig_k * job.stride_dim0 + orig_mn * job.stride_dim1];
      }
      val |= (static_cast<int32_t>(v.get()) << (byteIdx * 8));
    }
  }

  job.dst[packed_mn * K_packed + packed_k] = val;
}

template <typename ScaleT, bool KLast, int MNPack, int KPack, int XdlMNThread, int XdlKThread>
bool launch_grouped_pack_scales(const std::vector<GroupedScalePackJob>& jobs_host,
                                char*& ws_cursor,
                                size_t& ws_remaining,
                                hipStream_t stream,
                                const char* label) {
  if (jobs_host.empty()) return true;

  auto align_up = [](size_t x, size_t a) -> size_t { return (x + a - 1) / a * a; };

  const size_t jobs_bytes = align_up(jobs_host.size() * sizeof(GroupedScalePackJob), 16);
  if (ws_remaining < jobs_bytes) {
    NVTE_WARN("ck_tile_mx_grouped_gemm: insufficient workspace for grouped scale-pack job descriptors for ",
              label,
              ". Needed bytes=",
              jobs_bytes,
              ", available bytes=",
              ws_remaining);
    return false;
  }

  auto* jobs_dev = reinterpret_cast<GroupedScalePackJob*>(ws_cursor);
  ws_cursor += jobs_bytes;
  ws_remaining -= jobs_bytes;

  NVTE_CHECK_CUDA(hipMemcpyAsync(jobs_dev,
                                 jobs_host.data(),
                                 jobs_host.size() * sizeof(GroupedScalePackJob),
                                 hipMemcpyHostToDevice,
                                 stream));

  int max_total = 0;
  for (const auto& job : jobs_host) max_total = std::max(max_total, job.total);

  constexpr int threads = 256;
  const int blocks_x = (max_total + threads - 1) / threads;

  hipLaunchKernelGGL(
      (grouped_pack_scales_mnxk_kernel<ScaleT, KLast, MNPack, KPack, XdlMNThread, XdlKThread>),
      dim3(blocks_x, static_cast<unsigned int>(jobs_host.size()), 1),
      dim3(threads),
      0,
      stream,
      jobs_dev);

  NVTE_CHECK_CUDA(hipGetLastError());
  return true;
}

namespace transformer_engine {

template <typename GemmConfig, typename AType, typename BType, typename CType, typename AccType=float>
bool invoke_mx_grouped_gemm(const std::vector<MXGroupedHostDesc>& descs,
                            const CKGemmRunContext& ctx,
                            const ck_tile::stream_config& stream_cfg,
                            void* kargs_workspace,
                            size_t kargs_workspace_bytes) {
    using GemmShape =
        ck_tile::TileGemmShape<ck_tile::sequence<GemmConfig::M_Tile, GemmConfig::N_Tile, GemmConfig::K_Tile>,
                               ck_tile::sequence<GemmConfig::M_Warp, GemmConfig::N_Warp, GemmConfig::K_Warp>,
                               ck_tile::sequence<GemmConfig::M_Warp_Tile, GemmConfig::N_Warp_Tile, GemmConfig::K_Warp_Tile>>;
    
    using TilePartitioner =
        ck_tile::GemmSpatiallyLocalTilePartitioner<GemmShape, GemmConfig::TilePartitionerGroupNum, GemmConfig::TilePartitionerM01>;

    using GemmUniversalTraits = ck_tile::TileGemmUniversalTraits<GemmConfig::kPadM,
                                                                 GemmConfig::kPadN,
                                                                 GemmConfig::kPadK,
                                                                 GemmConfig::DoubleSmemBuffer,
                                                                 RowMajor,
                                                                 ColMajor,
                                                                 RowMajor,
                                                                 GemmConfig::TransposeC,
                                                                 GemmConfig::UseStructuredSparsity,
                                                                 true,
                                                                 GemmConfig::NumWaveGroups,
                                                                 GemmConfig::Preshuffle>;

    using UniversalGemmProblem =
        ck_tile::UniversalGemmPipelineProblem<AType, BType, AccType, GemmShape, GemmUniversalTraits, GemmConfig::Scheduler>;
    using GemmPipeline = ck_tile::MXGemmPipelineAgBgCrCompAsync<UniversalGemmProblem>;
    using GemmEpilogue = ck_tile::CShuffleEpilogue<
        ck_tile::CShuffleEpilogueProblem<AType,
                                         BType,
                                         ck_tile::tuple<>,
                                         AccType,
                                         CType,
                                         ck_tile::tuple<>,
                                         RowMajor,
                                         ck_tile::element_wise::PassThrough,
                                         TilePartitioner::MPerBlock,
                                         TilePartitioner::NPerBlock,
                                         GemmConfig::M_Warp,
                                         GemmConfig::N_Warp,
                                         GemmConfig::M_Warp_Tile,
                                         GemmConfig::N_Warp_Tile,
                                         GemmConfig::K_Warp_Tile,
                                         UniversalGemmProblem::TransposeC>>;
    using GroupedKernel = MXGroupedGemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue, ScaleM, ScaleN>;
    auto kargs = GroupedKernel::MakeKargs(descs);
    if(!GroupedKernel::IsSupportedArgument(kargs))
        throw std::runtime_error("Grouped MX GEMM arguments are not supported on this path.");

    const size_t needed_workspace =
        kargs.size() * sizeof(typename decltype(kargs)::value_type);

    if (!kargs_workspace || kargs_workspace_bytes < needed_workspace) {
    NVTE_WARN("ck_tile_mx_grouped_gemm: insufficient kargs workspace. Needed bytes=",
                needed_workspace, ", available bytes=", kargs_workspace_bytes);
    return false;
    }

    NVTE_CHECK_CUDA(hipMemcpyAsync(kargs_workspace,
                                    kargs.data(),
                                    needed_workspace,
                                    hipMemcpyHostToDevice,
                                    ctx.stream));

    const dim3 grids = GroupedKernel::GridSize(descs);
    const dim3 blocks = GroupedKernel::BlockSize();

    ck_tile::ignore = ck_tile::launch_kernel(
        stream_cfg,
        ck_tile::make_kernel<GemmConfig::kBlockPerCu>(
            GroupedKernel{},
            grids,
            blocks,
            0,
            ck_tile::cast_pointer_to_constant_address_space(kargs_workspace),
            static_cast<ck_tile::index_t>(kargs.size())));

    return true;

}

} // namespace transformer_engine

bool ck_tile_mx_grouped_gemm(const NVTETensor* A,
                          const NVTETensor* B,
                          NVTETensor* D,
                          int group_num,
                          bool transA,
                          bool transB,
                          NVTETensor* workspace,
                          bool accumulate,//ignored for now
                          hipStream_t stream) {
  if (accumulate || group_num <= 0) {
    return true;
  }

  using namespace transformer_engine;

  void* ws_ptr = nullptr;
  size_t ws_bytes = 0;
  if (workspace) {
    auto* ws_te = convertNVTETensorCheck(*workspace);
    ws_ptr = ws_te->data.dptr;
    ws_bytes = ws_te->data.numel() * typeToSize(ws_te->data.dtype);
  }

  // Normalize input mats
  const auto norm = normalize_gemm_inputs(A, B, transA, transB);
  const NVTETensor* A_use = norm.A;
  const NVTETensor* B_use = norm.B;
  bool transA_use = norm.transA;
  bool transB_use = norm.transB;

  bool use_a_colwise_data = false;
  bool use_b_colwise_data = false;

  Tensor* A0_te = convertNVTETensorCheck(A_use[0]);
  Tensor* B0_te = convertNVTETensorCheck(B_use[0]);
  Tensor* D0_te = convertNVTETensorCheck(D[0]);

  const auto a_dtype = convertNVTETensorCheck(A_use[0])->dtype();
  const auto b_dtype = convertNVTETensorCheck(B_use[0])->dtype();
  const auto d_dtype = D0_te->dtype();

  // Select CK's preferred FP8 NT presentation when columnwise storage is available.
  const auto presentation = select_ck_fp8_nt_presentation(
    true,
    transA_use,
    transB_use,
    A0_te->has_columnwise_data(),
    B0_te->has_columnwise_data());

  transA_use = presentation.transA;
  transB_use = presentation.transB;
  use_a_colwise_data = presentation.use_a_colwise_data;
  use_b_colwise_data = presentation.use_b_colwise_data;

  int64_t a0 = 0, a1 = 0;
  int64_t b0 = 0, b1 = 0;
  int64_t d0 = 0, d1 = 0;

  if (use_a_colwise_data) {
    if (!get_columnwise_storage_2d_dims(A0_te->columnwise_data, a0, a1)) {
      NVTE_ERROR("ck_tile_grouped_gemm: expected 2D columnwise_data for A_use[0]");
      return false;
    }
  } else {
    if (!get_flat_2d_dims(*A0_te, a0, a1)) {
      NVTE_ERROR("ck_tile_grouped_gemm: expected rank>=2 for normalized A_use[0]");
      return false;
    }
  }

  if (use_b_colwise_data) {
    if (!get_columnwise_storage_2d_dims(B0_te->columnwise_data, b0, b1)) {
      NVTE_ERROR("ck_tile_grouped_gemm: expected 2D columnwise_data for B_use[0]");
      return false;
    }
  } else {
    if (!get_flat_2d_dims(*B0_te, b0, b1)) {
      NVTE_ERROR("ck_tile_grouped_gemm: expected rank>=2 for normalized B_use[0]");
      return false;
    }
  }

  if (!get_flat_2d_dims(*D0_te, d0, d1)) {
    NVTE_ERROR("ck_tile_grouped_gemm: expected rank>=2 for D[0]");
    return false;
  }

  const int64_t m  = transA_use ? a1 : a0;
  const int64_t kA = transA_use ? a0 : a1;

  const int64_t kB = transB_use ? b1 : b0;
  const int64_t n  = transB_use ? b0 : b1;

  if (kA != kB) {
    NVTE_ERROR("ck_tile_grouped_gemm: normalized GEMM K mismatch: op(A_use) is ",
               m, "x", kA, ", op(B_use) is ", kB, "x", n);
    return false;
  }

  if (d0 != m || d1 != n) {
    NVTE_ERROR("ck_tile_grouped_gemm: D shape mismatch for normalized GEMM. "
               "D is ", d0, "x", d1, " but expected ", m, "x", n);
    return false;
  }

  CKGemmRunContext ctx = {
    A_use,
    B_use,
    D,
    static_cast<int>(n),
    group_num,
    transA_use,
    transB_use,
    ws_ptr,
    ws_bytes,
    stream,
    use_a_colwise_data,
    use_b_colwise_data,
    accumulate};

  const ck_tile::stream_config s{ctx.stream};

  using MxGemmConfig = MXfp8_GemmConfig_64x64x256;//MXfp8_GemmConfig_256x64x128;

  std::vector<MXGroupedHostDesc> descs;
  descs.reserve(group_num);

  std::vector<GroupedScalePackJob> a_scale_pack_k_last_jobs;
  std::vector<GroupedScalePackJob> a_scale_pack_k_first_jobs;
  std::vector<GroupedScalePackJob> b_scale_pack_k_last_jobs;
  std::vector<GroupedScalePackJob> b_scale_pack_k_first_jobs;
  a_scale_pack_k_last_jobs.reserve(group_num);
  a_scale_pack_k_first_jobs.reserve(group_num);
  b_scale_pack_k_last_jobs.reserve(group_num);
  b_scale_pack_k_first_jobs.reserve(group_num);

  auto align_up = [](size_t x, size_t a) -> size_t {
    return (x + a - 1) / a * a;
  };

  char* ws_cursor = static_cast<char*>(ws_ptr);
  size_t ws_remaining = ws_bytes;
  for (int i = 0; i < group_num; i++) {
    const transformer_engine::Tensor* const A_te =
        transformer_engine::convertNVTETensorCheck(ctx.A[i]);
    const transformer_engine::Tensor* const B_te =
        transformer_engine::convertNVTETensorCheck(ctx.B[i]);
    transformer_engine::Tensor* D_te =
        transformer_engine::convertNVTETensorCheck(ctx.D[i]);
    
    const transformer_engine::SimpleTensor* a_src = nullptr;
    if (ctx.use_a_columnwise_data) {
        NVTE_CHECK(A_te->has_columnwise_data(),
        "ck_tile_mx_grouped_gemm: ctx.use_a_columnwise_data=true but columnwise_data is absent.");
        a_src = &A_te->columnwise_data;
    } else {
        a_src = &A_te->data;
    }

    const transformer_engine::SimpleTensor* b_src = nullptr;
    if (ctx.use_b_columnwise_data) {
        NVTE_CHECK(B_te->has_columnwise_data(),
        "ck_tile_mx_grouped_gemm: ctx.use_b_columnwise_data=true but columnwise_data is absent.");
        b_src = &B_te->columnwise_data;
    } else {
        b_src = &B_te->data;
    }

    const auto& a = *a_src;
    const auto& b = *b_src;
    const auto& d = data_view(*D_te);
    int64_t Ad0 = 0, Ad1 = 0, Bd0 = 0, Bd1 = 0, Dd0 = 0, Dd1 = 0;

    if (ctx.use_a_columnwise_data) {
        if (!get_columnwise_storage_2d_dims(A_te->columnwise_data, Ad0, Ad1)) {
        NVTE_ERROR("ck_tile_mx_grouped_gemm: expected 2D columnwise_data for A in group ", i);
        }
    } else {
        if (!get_flat_2d_dims(*A_te, Ad0, Ad1)) {
        NVTE_ERROR("ck_tile_mx_grouped_gemm: expected rank>=2 for A in group ", i);
        }
    }

    if (ctx.use_b_columnwise_data) {
        if (!get_columnwise_storage_2d_dims(B_te->columnwise_data, Bd0, Bd1)) {
        NVTE_ERROR("ck_tile_mx_grouped_gemm: expected 2D columnwise_data for B in group ", i);
        }
    } else {
        if (!get_flat_2d_dims(*B_te, Bd0, Bd1)) {
        NVTE_ERROR("ck_tile_mx_grouped_gemm: expected rank>=2 for B in group ", i);
        }
    }

    if (!get_flat_2d_dims(*D_te, Dd0, Dd1)) {
        NVTE_ERROR("ck_tile_mx_grouped_gemm: expected rank>=2 for D in group ", i);
    }

    const auto& a_scales =
        ctx.use_a_columnwise_data ? A_te->columnwise_scale_inv : A_te->scale_inv;
    const auto& b_scales =
        ctx.use_b_columnwise_data ? B_te->columnwise_scale_inv : B_te->scale_inv;

    if (a_scales.shape.size() != 2 || b_scales.shape.size() != 2) {
        NVTE_ERROR("ck_tile_mx_grouped_gemm: expected A/B scale_inv tensors to be rank-2.");
    }

    NVTE_CHECK(a_scales.dtype == DType::kFloat8E8M0,
            "ck_tile_mx_grouped_gemm: A scale dtype must be Float8E8M0.");
    NVTE_CHECK(b_scales.dtype == DType::kFloat8E8M0,
            "ck_tile_mx_grouped_gemm: B scale dtype must be Float8E8M0.");

    const int64_t M = ctx.transA ? Ad1 : Ad0;
    const int64_t K = ctx.transA ? Ad0 : Ad1;
    const int64_t N = ctx.transB ? Bd0 : Bd1;
    const int64_t Kb = ctx.transB ? Bd1 : Bd0;
    if (K % ScaleBlockSize != 0) {
        NVTE_ERROR("ck_tile_mx_grouped_gemm: K must be a multiple of ScaleBlockSize for MX GEMM", i);
    }
    const int KScale = static_cast<int>(K / ScaleBlockSize);
    if (Kb != K) {
        NVTE_ERROR("ck_tile_mx_grouped_gemm: K mismatch between A and B in group ", i);
    }
    if (Dd0 != M || Dd1 != N) {
        NVTE_ERROR("ck_tile_mx_grouped_gemm: D shape mismatch in group ", i);
    }
    const ck_tile::index_t stride_A = static_cast<ck_tile::index_t>(Ad1);
    const ck_tile::index_t stride_B = static_cast<ck_tile::index_t>(Bd1);
    const ck_tile::index_t stride_E = static_cast<ck_tile::index_t>(Dd1);

    constexpr ck_tile::index_t MPerXdl = MxGemmConfig::M_Warp_Tile;
    constexpr ck_tile::index_t NPerXdl = MxGemmConfig::N_Warp_Tile;
    constexpr ck_tile::index_t KPerXdl = MxGemmConfig::K_Warp_Tile;

    constexpr ck_tile::index_t MIterPerWarp =
        MxGemmConfig::M_Tile / (MxGemmConfig::M_Warp * MPerXdl);
    constexpr ck_tile::index_t NIterPerWarp =
        MxGemmConfig::N_Tile / (MxGemmConfig::N_Warp * NPerXdl);
    constexpr ck_tile::index_t KIterPerWarp =
        MxGemmConfig::K_Tile / KPerXdl;

    constexpr ck_tile::index_t MXdlPackEff =
        (MIterPerWarp >= 2 && MIterPerWarp % 2 == 0) ? 2 : 1;
    constexpr ck_tile::index_t NXdlPackEff =
        (NIterPerWarp >= 2 && NIterPerWarp % 2 == 0) ? 2 : 1;
    constexpr ck_tile::index_t KXdlPackEff =
        (KIterPerWarp >= 2 && KIterPerWarp % 2 == 0) ? 2 : 1;

    constexpr ck_tile::index_t XdlMNThread = MxGemmConfig::M_Warp_Tile;
    constexpr ck_tile::index_t XdlKThread = 64 / XdlMNThread;

    const bool a_scales_m_k =
        (a_scales.shape[0] == M && a_scales.shape[1] == KScale);
    const bool a_scales_k_m =
        (a_scales.shape[0] == KScale && a_scales.shape[1] == M);

    if (!a_scales_m_k && !a_scales_k_m) {
    NVTE_ERROR("ck_tile_mx_grouped_gemm: expected A scales shape [M, KScale] or [KScale, M].");
    }

    const bool b_scales_n_k =
        (b_scales.shape[0] == N && b_scales.shape[1] == KScale);
    const bool b_scales_k_n =
        (b_scales.shape[0] == KScale && b_scales.shape[1] == N);

    if (!b_scales_n_k && !b_scales_k_n) {
    NVTE_ERROR("ck_tile_mx_grouped_gemm: expected B scales shape [N, KScale] or [KScale, N].");
    }

    if (M % MXdlPackEff != 0 || N % NXdlPackEff != 0 || KScale % KXdlPackEff != 0) {
    NVTE_ERROR("ck_tile_mx_grouped_gemm: scale pack dimensions are not divisible by pack factors.");
    }

    const size_t a_pack_elems =
        static_cast<size_t>(M / MXdlPackEff) * static_cast<size_t>(KScale / KXdlPackEff);
    const size_t b_pack_elems =
        static_cast<size_t>(N / NXdlPackEff) * static_cast<size_t>(KScale / KXdlPackEff);

    const size_t a_pack_bytes = align_up(a_pack_elems * sizeof(int32_t), 16);
    const size_t b_pack_bytes = align_up(b_pack_elems * sizeof(int32_t), 16);

    if (ws_remaining < a_pack_bytes + b_pack_bytes) {
      NVTE_WARN("ck_tile_mx_grouped_gemm: insufficient workspace for packed scales. Needed bytes=",
                a_pack_bytes + b_pack_bytes, ", available bytes=", ws_remaining);
      return false;
    }

    auto* p_scale_a = reinterpret_cast<ScaleType*>(ws_cursor);
    ws_cursor += a_pack_bytes;
    ws_remaining -= a_pack_bytes;

    auto* p_scale_b = reinterpret_cast<ScaleType*>(ws_cursor);
    ws_cursor += b_pack_bytes;
    ws_remaining -= b_pack_bytes;

    const int a_total = static_cast<int>(a_pack_elems);
    const int b_total = static_cast<int>(b_pack_elems);

    if (a_scales_m_k) {
      // physical/logical [M, KScale]
      a_scale_pack_k_last_jobs.push_back(GroupedScalePackJob{
          reinterpret_cast<const ScaleType*>(a_scales.dptr),
          reinterpret_cast<int32_t*>(p_scale_a),
          static_cast<int>(M),
          static_cast<int>(KScale),
          static_cast<int>(a_scales.shape[1]),
          1,
          a_total});
    } else {
      // physical [KScale, M], but pack kernel expects logical [M, KScale]
      a_scale_pack_k_last_jobs.push_back(GroupedScalePackJob{
          reinterpret_cast<const ScaleType*>(a_scales.dptr),
          reinterpret_cast<int32_t*>(p_scale_a),
          static_cast<int>(M),
          static_cast<int>(KScale),
          1,
          static_cast<int>(M),
          a_total});
    }

    if (b_scales_k_n) {
      // physical/logical [KScale, N]
      b_scale_pack_k_first_jobs.push_back(GroupedScalePackJob{
          reinterpret_cast<const ScaleType*>(b_scales.dptr),
          reinterpret_cast<int32_t*>(p_scale_b),
          static_cast<int>(N),
          static_cast<int>(KScale),
          static_cast<int>(b_scales.shape[1]),
          1,
          b_total});
    } else {
      // physical/logical [N, KScale]
      b_scale_pack_k_first_jobs.push_back(GroupedScalePackJob{
          reinterpret_cast<const ScaleType*>(b_scales.dptr),
          reinterpret_cast<int32_t*>(p_scale_b),
          static_cast<int>(N),
          static_cast<int>(KScale),
          1,
          static_cast<int>(b_scales.shape[1]),
          b_total});
    }

    descs.emplace_back(
        a.dptr,
        ScaleM{p_scale_a},
        b.dptr,
        ScaleN{p_scale_b},
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

  constexpr ck_tile::index_t MPerXdlGrouped = MxGemmConfig::M_Warp_Tile;
  constexpr ck_tile::index_t NPerXdlGrouped = MxGemmConfig::N_Warp_Tile;
  constexpr ck_tile::index_t KPerXdlGrouped = MxGemmConfig::K_Warp_Tile;
  constexpr ck_tile::index_t MIterPerWarpGrouped =
      MxGemmConfig::M_Tile / (MxGemmConfig::M_Warp * MPerXdlGrouped);
  constexpr ck_tile::index_t NIterPerWarpGrouped =
      MxGemmConfig::N_Tile / (MxGemmConfig::N_Warp * NPerXdlGrouped);
  constexpr ck_tile::index_t KIterPerWarpGrouped = MxGemmConfig::K_Tile / KPerXdlGrouped;
  constexpr ck_tile::index_t MXdlPackEffGrouped =
      (MIterPerWarpGrouped >= 2 && MIterPerWarpGrouped % 2 == 0) ? 2 : 1;
  constexpr ck_tile::index_t NXdlPackEffGrouped =
      (NIterPerWarpGrouped >= 2 && NIterPerWarpGrouped % 2 == 0) ? 2 : 1;
  constexpr ck_tile::index_t KXdlPackEffGrouped =
      (KIterPerWarpGrouped >= 2 && KIterPerWarpGrouped % 2 == 0) ? 2 : 1;
  constexpr ck_tile::index_t XdlMNThreadGrouped = MxGemmConfig::M_Warp_Tile;
  constexpr ck_tile::index_t XdlKThreadGrouped = 64 / XdlMNThreadGrouped;

  // Run scale preprocessing as grouped kernels instead of launching one A-scale
  // pack and one B-scale pack kernel per expert.  This keeps the same packed
  // scale layout and workspace contract, but removes unnecessary serialized
  // preprocessing launches before the grouped MXGemm kernel.
  if (!launch_grouped_pack_scales<ScaleType, true, MXdlPackEffGrouped, KXdlPackEffGrouped, XdlMNThreadGrouped, XdlKThreadGrouped>(
          a_scale_pack_k_last_jobs, ws_cursor, ws_remaining, stream, "A scales KLast")) {
    return false;
  }
  if (!launch_grouped_pack_scales<ScaleType, false, MXdlPackEffGrouped, KXdlPackEffGrouped, XdlMNThreadGrouped, XdlKThreadGrouped>(
          a_scale_pack_k_first_jobs, ws_cursor, ws_remaining, stream, "A scales KFirst")) {
    return false;
  }
  if (!launch_grouped_pack_scales<ScaleType, true, NXdlPackEffGrouped, KXdlPackEffGrouped, XdlMNThreadGrouped, XdlKThreadGrouped>(
          b_scale_pack_k_last_jobs, ws_cursor, ws_remaining, stream, "B scales KLast")) {
    return false;
  }
  if (!launch_grouped_pack_scales<ScaleType, false, NXdlPackEffGrouped, KXdlPackEffGrouped, XdlMNThreadGrouped, XdlKThreadGrouped>(
          b_scale_pack_k_first_jobs, ws_cursor, ws_remaining, stream, "B scales KFirst")) {
    return false;
  }

  bool ok = false;
  TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(a_dtype, a_te_type, {
    using AType = typename TETypeToCKType<a_te_type>::type;
    TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(b_dtype, b_te_type, {
      using BType = typename TETypeToCKType<b_te_type>::type;
      TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(d_dtype, d_te_type, {
        using CType = typename TETypeToCKType<d_te_type>::type;
          ok = invoke_mx_grouped_gemm<MxGemmConfig, AType, BType, CType>(
            descs, ctx, s, ws_cursor, ws_remaining);
      });
    });
  });

  return ok;
}
