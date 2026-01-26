#include <hip/hip_runtime.h>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"

static inline int get_num_cu_for_stream(hipStream_t stream) {
  int device = -1;
  hipError_t st = hipGetDevice(&device);
  if (st != hipSuccess)
    return 0;

  hipDeviceProp_t prop{};
  st = hipGetDeviceProperties(&prop, device);
  if (st != hipSuccess)
    return 0;

  return prop.multiProcessorCount;
}

// Map TE DType to CK_Tile scalar type
template <transformer_engine::DType TeDtype>
struct TeDTypeToCk;

template <> struct TeDTypeToCk<transformer_engine::DType::kFloat16> {
  using type = ck_tile::half_t;
};
template <> struct TeDTypeToCk<transformer_engine::DType::kBFloat16> {
  using type = ck_tile::bfloat16_t;
};

// TE Tensor -> SimpleTensor view
static inline const transformer_engine::SimpleTensor& data_view(const transformer_engine::Tensor& t) {
  // For GEMM we want the "data" view (rowwise)
  return t.data;
}

// CK_Tile runner

using RowMajor = ck_tile::tensor_layout::gemm::RowMajor;
using ColMajor = ck_tile::tensor_layout::gemm::ColumnMajor;

struct TileCfg_basic {
  static constexpr ck_tile::index_t M_Tile = 256;
  static constexpr ck_tile::index_t N_Tile = 128;
  static constexpr ck_tile::index_t K_Tile = 64;

  static constexpr ck_tile::index_t M_Warp = 2;
  static constexpr ck_tile::index_t N_Warp = 2;
  static constexpr ck_tile::index_t K_Warp = 1;

  static constexpr ck_tile::index_t M_Warp_Tile = 32;
  static constexpr ck_tile::index_t N_Warp_Tile = 32;
  static constexpr ck_tile::index_t K_Warp_Tile = 16;

  static constexpr bool kPadM = true;
  static constexpr bool kPadN = true;
  static constexpr bool kPadK = true;

  static constexpr bool DoubleSmemBuffer = false;

  // Spatially-local partitioner parameters
  static constexpr ck_tile::index_t TileParitionerGroupNum = 8;
  static constexpr ck_tile::index_t TileParitionerM01      = 1;
};

template <typename Kernel>
inline void launch_grouped_kernel(const ck_tile::stream_config& stream_cfg,
                                  ck_tile::index_t group_num,
                                  void* args_ptr,
                                  uint32_t num_cu) {
  constexpr int kBlockPerCu = 1;
  const dim3 blocks = Kernel::BlockSize();
  dim3 grids = Kernel::MaxOccupancyGridSize(stream_cfg);
  grids.x = std::min<ck_tile::index_t>(grids.x, static_cast<ck_tile::index_t>(num_cu));
  ck_tile::launch_kernel(
      stream_cfg,
      ck_tile::make_kernel<kBlockPerCu>(
          Kernel{}, grids, blocks, 0,
          ck_tile::cast_pointer_to_constant_address_space(args_ptr),
          group_num));
}

template <typename AType, typename BType, typename CType,
          typename ALayout, typename BLayout, typename CLayout,
          typename TileCfg, typename AccType = float>
class Runner{
public:
  using GemmShape = ck_tile::TileGemmShape<
      ck_tile::sequence<TileCfg::M_Tile, TileCfg::N_Tile, TileCfg::K_Tile>,
      ck_tile::sequence<TileCfg::M_Warp, TileCfg::N_Warp, TileCfg::K_Warp>,
      ck_tile::sequence<TileCfg::M_Warp_Tile, TileCfg::N_Warp_Tile, TileCfg::K_Warp_Tile>>;

  using Partitioner = ck_tile::GemmSpatiallyLocalTilePartitioner<
      GemmShape, TileCfg::TileParitionerGroupNum, TileCfg::TileParitionerM01>;

  using UniversalTraits = ck_tile::PersistentTileGemmUniversalTraits<
      TileCfg::kPadM, TileCfg::kPadN, TileCfg::kPadK,
      TileCfg::DoubleSmemBuffer, ALayout, BLayout, CLayout>;

  static constexpr ck_tile::GemmPipelineScheduler Scheduler =
      ck_tile::GemmPipelineScheduler::Intrawave;

  using Problem = ck_tile::UniversalGemmPipelineProblem<
      AType, BType, AccType, GemmShape, UniversalTraits, Scheduler>;

  using Pipeline = ck_tile::GemmPipelineAgBgCrCompV3<Problem>;

  static constexpr ck_tile::memory_operation_enum MemOp = ck_tile::memory_operation_enum::set;

  using Epilogue = ck_tile::CShuffleEpilogue<
      ck_tile::CShuffleEpilogueProblem<
          AType, BType, ck_tile::tuple<>, AccType,
          CType, ck_tile::tuple<>, CLayout,
          ck_tile::element_wise::PassThrough,
          Partitioner::MPerBlock, Partitioner::NPerBlock,
          TileCfg::M_Warp, TileCfg::N_Warp,
          TileCfg::M_Warp_Tile, TileCfg::N_Warp_Tile, TileCfg::K_Warp_Tile,
          Problem::TransposeC, MemOp>>;

  using Kernel = ck_tile::GroupedGemmKernel<Partitioner, Pipeline, Epilogue>;

  void run(const ck_tile::stream_config& stream_cfg,
           ck_tile::index_t group_num,
           void* args_ptr,
           uint32_t num_cu) {
    launch_grouped_kernel<Kernel>(stream_cfg, group_num, args_ptr, num_cu);
  }
};

// Arg builder kernel

template <typename AType, typename BType, typename CType>
__global__ void build_args_kernel(ck_tile::GemmTransKernelArg<>* args,
                                  const void* const* a_ptrs,
                                  const void* const* b_ptrs,
                                  void* const* d_ptrs,
                                  const int64_t* ms,
                                  const int64_t* ns,
                                  const int64_t* ks,
                                  ck_tile::index_t group_num,
                                  ck_tile::index_t strideA,
                                  ck_tile::index_t strideB,
                                  ck_tile::index_t strideD,
                                  ck_tile::index_t k_batch) {
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= group_num)
    return;

  // CK_Tile's grouped arg uses arrays for As/Bs
  const_cast<std::array<const void*, 1>&>(args[gid].group_karg.as_ptr)[0] =
      static_cast<const void*>(a_ptrs[gid]);
  const_cast<std::array<const void*, 1>&>(args[gid].group_karg.bs_ptr)[0] =
      static_cast<const void*>(b_ptrs[gid]);

  args[gid].group_karg.e_ptr = d_ptrs[gid];

  args[gid].group_karg.M = static_cast<ck_tile::index_t>(ms[gid]);
  args[gid].group_karg.N = static_cast<ck_tile::index_t>(ns[gid]);
  args[gid].group_karg.K = static_cast<ck_tile::index_t>(ks[gid]);

  args[gid].group_karg.stride_As[0] = strideA;
  args[gid].group_karg.stride_Bs[0] = strideB;
  args[gid].group_karg.stride_E     = strideD;
  args[gid].group_karg.k_batch      = k_batch;
}

bool grouped_gemm_ck_tile(const transformer_engine::Tensor* const* A,
                          const transformer_engine::Tensor* const* B,
                          transformer_engine::Tensor* const* D,
                          int group_num,
                          bool transA,
                          bool transB,
                          void* workspace,
                          size_t workspace_bytes,
                          hipStream_t stream,
                          uint32_t num_cu_override = 0) {
  // TE sometimes passes (A=weight, B=input, transA=1, transB=0) for y = x * W^T
  // CK_Tile expects the left operand to be the activation matrix
  // So for (transA && !transB), swap A/B and turn it into (!transA && transB)
  const transformer_engine::Tensor* const* A_use = A;
  const transformer_engine::Tensor* const* B_use = B;
  bool transA_use = transA;
  bool transB_use = transB;
  if (transA && !transB) {
    A_use = B;
    B_use = A;
    transA_use = false;
    transB_use = true;
  }

  if (!( (!transA_use && !transB_use) || (!transA_use && transB_use) )) {
    NVTE_ERROR("grouped_gemm_ck_tile: only NN/NT/TN supported.");
    return false;
  }

  // DType routing: allow fp16/bf16 for now
  const auto a_dtype = A_use[0]->dtype();
  const auto b_dtype = B_use[0]->dtype();
  const auto d_dtype = D[0]->dtype();
  if (a_dtype != b_dtype || a_dtype != d_dtype) {
    NVTE_ERROR("grouped_gemm_ck_tile: dtype mismatch A/B/D.");
    return false;
  }
  if (!(a_dtype == transformer_engine::DType::kFloat16 ||
        a_dtype == transformer_engine::DType::kBFloat16)) {
    NVTE_ERROR("grouped_gemm_ck_tile: only fp16/bf16 supported.");
    return false;
  }

  // Workspace layout:
  // [0] device arrays of pointers (A_ptrs, B_ptrs, D_ptrs)
  // [1] device arrays of int64 (M, N, K)
  // [2] ck_tile::GemmTransKernelArg<>[group_num]
  const size_t ptr_arr_bytes = sizeof(void*) * static_cast<size_t>(group_num);
  const size_t i64_arr_bytes = sizeof(int64_t) * static_cast<size_t>(group_num);

  const size_t off_a_ptrs = 0;
  const size_t off_b_ptrs = off_a_ptrs + ptr_arr_bytes;
  const size_t off_d_ptrs = off_b_ptrs + ptr_arr_bytes;
  const size_t off_ms     = off_d_ptrs + ptr_arr_bytes;
  const size_t off_ns     = off_ms + i64_arr_bytes;
  const size_t off_ks     = off_ns + i64_arr_bytes;

  const size_t off_args   = ck_tile::integer_divide_ceil(off_ks + i64_arr_bytes, size_t(16)) * 16;

  const size_t args_bytes = sizeof(ck_tile::GemmTransKernelArg<>) * static_cast<size_t>(group_num);
  const size_t needed = off_args + args_bytes;

  if (workspace == nullptr || workspace_bytes < needed) {
    NVTE_ERROR("grouped_gemm_ck_tile: insufficient workspace. Needed bytes=", needed);
    return false;
  }

  auto* base = static_cast<uint8_t*>(workspace);

  void**  d_a_ptrs = reinterpret_cast<void**>(base + off_a_ptrs);
  void**  d_b_ptrs = reinterpret_cast<void**>(base + off_b_ptrs);
  void**  d_d_ptrs = reinterpret_cast<void**>(base + off_d_ptrs);
  int64_t* d_ms    = reinterpret_cast<int64_t*>(base + off_ms);
  int64_t* d_ns    = reinterpret_cast<int64_t*>(base + off_ns);
  int64_t* d_ks    = reinterpret_cast<int64_t*>(base + off_ks);

  auto* d_args = reinterpret_cast<ck_tile::GemmTransKernelArg<>*>(base + off_args);

  // Build host-side staging buffers and memcpy to device
  std::vector<void*>  h_a_ptrs(group_num);
  std::vector<void*>  h_b_ptrs(group_num);
  std::vector<void*>  h_d_ptrs(group_num);
  std::vector<int64_t> h_ms(group_num);
  std::vector<int64_t> h_ns(group_num);
  std::vector<int64_t> h_ks(group_num);

  // Infer global N/K from group 0
  const auto& a0 = data_view(*A_use[0]);
  const auto& b0 = data_view(*B_use[0]);
  const auto& d0 = data_view(*D[0]);
  if (a0.shape.size() != 2 || b0.shape.size() != 2 || d0.shape.size() != 2) {
    NVTE_ERROR("grouped_gemm_ck_tile: expected 2D tensors.");
    return false;
  }

  printf("grouped_gemm_ck_tile gg0 A=[%zu,%zu] B=[%zu,%zu] D=[%zu,%zu] transA=%d transB=%d\n",
              a0.shape[0], a0.shape[1],
              b0.shape[0], b0.shape[1],
              d0.shape[0], d0.shape[1],
              (int)transA_use, (int)transB_use);

  // Infer logical M/K from A depending on transA
  // - NN/NT: A stored [M,K]
  // - TN:    A stored [K,M] row-major, interpret as ColMajor [M,K]
  const int64_t m0 = transA_use ? static_cast<int64_t>(a0.shape[1]) : static_cast<int64_t>(a0.shape[0]);
  const int64_t k0 = transA_use ? static_cast<int64_t>(a0.shape[0]) : static_cast<int64_t>(a0.shape[1]);

  const int64_t n0 = transB_use ? static_cast<int64_t>(b0.shape[0])
                            : static_cast<int64_t>(b0.shape[1]);
  const int64_t kb = transB_use ? static_cast<int64_t>(b0.shape[1])
                            : static_cast<int64_t>(b0.shape[0]);
  if (kb != k0) {
    NVTE_ERROR("grouped_gemm_ck_tile: K mismatch between A and B in group 0.");
    return false;
  }
  if (static_cast<int64_t>(d0.shape[0]) != m0 || static_cast<int64_t>(d0.shape[1]) != n0) {
    NVTE_ERROR("grouped_gemm_ck_tile: D shape mismatch in group 0.");
    return false;
  }

  for (int i = 0; i < group_num; ++i) {
    const auto& ai = data_view(*A_use[i]);
    const auto& bi = data_view(*B_use[i]);
    const auto& di = data_view(*D[i]);

    if (ai.shape.size() != 2 || bi.shape.size() != 2 || di.shape.size() != 2) {
      NVTE_ERROR("grouped_gemm_ck_tile: expected all groups to be 2D.");
      return false;
    }

    const int64_t mi = transA_use ? static_cast<int64_t>(ai.shape[1]) : static_cast<int64_t>(ai.shape[0]);
    const int64_t ki = transA_use ? static_cast<int64_t>(ai.shape[0]) : static_cast<int64_t>(ai.shape[1]);
    const int64_t ni = transB_use ? static_cast<int64_t>(bi.shape[0])
                              : static_cast<int64_t>(bi.shape[1]);
    const int64_t kbi = transB_use ? static_cast<int64_t>(bi.shape[1])
                               : static_cast<int64_t>(bi.shape[0]);

    if (ki != k0 || ni != n0 || kbi != k0) {
      NVTE_ERROR("grouped_gemm_ck_tile: N/K must be constant across groups.");
      return false;
    }
    if (static_cast<int64_t>(di.shape[0]) != mi || static_cast<int64_t>(di.shape[1]) != n0) {
      NVTE_ERROR("grouped_gemm_ck_tile: D shape mismatch in group ", i);
      return false;
    }

    h_a_ptrs[i] = ai.dptr;
    h_b_ptrs[i] = bi.dptr;
    h_d_ptrs[i] = di.dptr;
    h_ms[i] = mi;
    h_ns[i] = n0;
    h_ks[i] = k0;
  }

  HIP_CHECK_ERROR(hipMemcpyAsync(d_a_ptrs, h_a_ptrs.data(), ptr_arr_bytes, hipMemcpyHostToDevice,
                                reinterpret_cast<hipStream_t>(stream)));
  HIP_CHECK_ERROR(hipMemcpyAsync(d_b_ptrs, h_b_ptrs.data(), ptr_arr_bytes, hipMemcpyHostToDevice,
                                reinterpret_cast<hipStream_t>(stream)));
  HIP_CHECK_ERROR(hipMemcpyAsync(d_d_ptrs, h_d_ptrs.data(), ptr_arr_bytes, hipMemcpyHostToDevice,
                                reinterpret_cast<hipStream_t>(stream)));
  HIP_CHECK_ERROR(hipMemcpyAsync(d_ms, h_ms.data(), i64_arr_bytes, hipMemcpyHostToDevice,
                                reinterpret_cast<hipStream_t>(stream)));
  HIP_CHECK_ERROR(hipMemcpyAsync(d_ns, h_ns.data(), i64_arr_bytes, hipMemcpyHostToDevice,
                                reinterpret_cast<hipStream_t>(stream)));
  HIP_CHECK_ERROR(hipMemcpyAsync(d_ks, h_ks.data(), i64_arr_bytes, hipMemcpyHostToDevice,
                                reinterpret_cast<hipStream_t>(stream)));

  // Leading dimensions for CK layouts:
  // A is row-major [M,K] and we only support transA=false -> ALayout=RowMajor, strideA=K
  // B is row-major [K,N] if NN -> BLayout=RowMajor, strideB=N
  // B is row-major [N,K] if NT -> BLayout=ColMajor (logical [K,N]), strideB=K
  const ck_tile::index_t strideA = static_cast<ck_tile::index_t>(transA_use ? m0 : k0);
  const ck_tile::index_t strideB = static_cast<ck_tile::index_t>(transB_use ? k0 : n0);
  const ck_tile::index_t strideD = static_cast<ck_tile::index_t>(n0);

  // Build CK arg structs on device
  {
    const int threads = 256;
    const int blocks  = (group_num + threads - 1) / threads;
    const ck_tile::index_t k_batch = 1;
    if (a_dtype == transformer_engine::DType::kFloat16) {
      using AType = TeDTypeToCk<transformer_engine::DType::kFloat16>::type;
      using BType = AType;
      using CType = AType;
      hipLaunchKernelGGL((build_args_kernel<AType, BType, CType>),
                         dim3(blocks), dim3(threads), 0,
                         reinterpret_cast<hipStream_t>(stream),
                         d_args,
                         const_cast<const void* const*>(reinterpret_cast<void* const*>(d_a_ptrs)),
                         const_cast<const void* const*>(reinterpret_cast<void* const*>(d_b_ptrs)),
                         reinterpret_cast<void* const*>(d_d_ptrs),
                         d_ms, d_ns, d_ks,
                         static_cast<ck_tile::index_t>(group_num),
                         strideA, strideB, strideD,
                         k_batch);
    } else {
      using AType = TeDTypeToCk<transformer_engine::DType::kBFloat16>::type;
      using BType = AType;
      using CType = AType;
      hipLaunchKernelGGL((build_args_kernel<AType, BType, CType>),
                         dim3(blocks), dim3(threads), 0,
                         reinterpret_cast<hipStream_t>(stream),
                         d_args,
                         const_cast<const void* const*>(reinterpret_cast<void* const*>(d_a_ptrs)),
                         const_cast<const void* const*>(reinterpret_cast<void* const*>(d_b_ptrs)),
                         reinterpret_cast<void* const*>(d_d_ptrs),
                         d_ms, d_ns, d_ks,
                         static_cast<ck_tile::index_t>(group_num),
                         strideA, strideB, strideD,
                         k_batch);
    }
  }

  // Runner selection
  const uint32_t num_cu = (num_cu_override != 0) ? num_cu_override
                                                : static_cast<uint32_t>(get_num_cu_for_stream(stream));
  const ck_tile::stream_config stream_cfg{reinterpret_cast<hipStream_t>(stream)};

  // Choose layouts based on transB
  if (a_dtype == transformer_engine::DType::kFloat16) {
    using T = TeDTypeToCk<transformer_engine::DType::kFloat16>::type;

    if (!transB_use) {
      // NN: A RowMajor, B RowMajor, D RowMajor
      Runner<T, T, T, RowMajor, RowMajor, RowMajor, TileCfg_basic> runner;
      runner.run(stream_cfg, static_cast<ck_tile::index_t>(group_num), d_args, num_cu);
    } else {
      // NT: B is stored as [N,K] row-major -> treat as ColMajor logical [K,N]
      Runner<T, T, T, RowMajor, ColMajor, RowMajor, TileCfg_basic> runner;
      runner.run(stream_cfg, static_cast<ck_tile::index_t>(group_num), d_args, num_cu);
    }
  } else {
    using T = TeDTypeToCk<transformer_engine::DType::kBFloat16>::type;

    if (!transB_use) {
      Runner<T, T, T, RowMajor, RowMajor, RowMajor, TileCfg_basic> runner;
      runner.run(stream_cfg, static_cast<ck_tile::index_t>(group_num), d_args, num_cu);
    } else {
      Runner<T, T, T, RowMajor, ColMajor, RowMajor, TileCfg_basic> runner;
      runner.run(stream_cfg, static_cast<ck_tile::index_t>(group_num), d_args, num_cu);
    }
  }

  return true;
}

bool grouped_gemm_ck_tile(const NVTETensor* A,
                          const NVTETensor* B,
                          NVTETensor* D,
                          int group_num,
                          bool transA,
                          bool transB,
                          NVTETensor* workspace,
                          hipStream_t stream) {
  if (group_num <= 0)
    return true;

  // Convert A/B/D arrays into TE Tensor* arrays
  std::vector<const transformer_engine::Tensor*> A_te(group_num);
  std::vector<const transformer_engine::Tensor*> B_te(group_num);
  std::vector<transformer_engine::Tensor*>       D_te(group_num);

  for (int i = 0; i < group_num; ++i) {
    A_te[i] = transformer_engine::convertNVTETensorCheck(A[i]);
    B_te[i] = transformer_engine::convertNVTETensorCheck(B[i]);
    D_te[i] = transformer_engine::convertNVTETensorCheck(D[i]);
  }

  // Workspace pointer + bytes
  void*  ws_ptr   = nullptr;
  size_t ws_bytes = 0;
  if (workspace) {
    auto* ws_te = transformer_engine::convertNVTETensorCheck(*workspace);
    ws_ptr   = ws_te->data.dptr;
    ws_bytes = ws_te->data.numel() * transformer_engine::typeToSize(ws_te->data.dtype);
  }

  return grouped_gemm_ck_tile(A_te.data(), B_te.data(), D_te.data(),
                              group_num, transA, transB,
                              ws_ptr, ws_bytes,
                              stream);
}
