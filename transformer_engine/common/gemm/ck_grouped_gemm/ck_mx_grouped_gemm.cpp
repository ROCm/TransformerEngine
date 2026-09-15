/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#include "ck_grouped_gemm_common.h"
#include "ck_mx_grouped_gemm_impl.h"

namespace transformer_engine {
namespace grouped_gemm {

bool ck_tile_mx_grouped_gemm(const NVTETensor *A,
                             const NVTETensor *B,
                             NVTETensor *D,
                             int group_num,
                             bool transA,
                             bool transB,
                             NVTETensor *workspace,
                             bool accumulate,  // ignored for now
                             hipStream_t stream) {

  const bool warn_fallback =
    getenv<bool>("NVTE_CUTLASS_GROUPED_GEMM_WARN_FALLBACK", false);

  if (group_num <= 0) {
    return true;
  }

  // Normalize input mats
  // I.e., swap A and B, as well as transa and transb.
  const NVTETensor *A_use = B;
  const NVTETensor *B_use = A;
  bool transA_use = transB;
  bool transB_use = transA;

  // Note: for MXFP8, row-wise and col-wise data are scaled along different
  // dims, with the mat interpreted in row-major.
  // Use the operand transpose flags to select the correct view.
  // Scale view needs to match data view.
  const bool use_a_colwise_data = transA_use;
  const bool use_b_colwise_data = !transB_use;

  Tensor *A0_te = convertNVTETensorCheck(A_use[0]);
  Tensor *B0_te = convertNVTETensorCheck(B_use[0]);

  // Validate scale type / data type combination.
  // Expected input data format: fp8/bf8 (e4m3/e5m2)
  // Expected scale data format: e8m0
  const auto *D0 = convertNVTETensorCheck(D[0]);

  const auto &A0_data = use_a_colwise_data ? A0_te->columnwise_data : A0_te->data;
  const auto &B0_data = use_b_colwise_data ? B0_te->columnwise_data : B0_te->data;
  const auto &A0_scale = use_a_colwise_data ? A0_te->columnwise_scale_inv : A0_te->scale_inv;
  const auto &B0_scale = use_b_colwise_data ? B0_te->columnwise_scale_inv : B0_te->scale_inv;

  NVTE_CHECK(A0_data.dptr != nullptr,
             "ck_tile_mx_grouped_gemm: A[0] data is not initialized");
  NVTE_CHECK(B0_data.dptr != nullptr,
             "ck_tile_mx_grouped_gemm: B[0] data is not initialized");
  NVTE_CHECK(A0_scale.dptr != nullptr,
             "ck_tile_mx_grouped_gemm: A[0] scale_inv is not initialized");
  NVTE_CHECK(B0_scale.dptr != nullptr,
             "ck_tile_mx_grouped_gemm: B[0] scale_inv is not initialized");

  const auto a_scale_dtype = A0_scale.dtype;
  const auto b_scale_dtype = B0_scale.dtype;
  NVTE_CHECK(a_scale_dtype == DType::kFloat8E8M0,
             "ck_tile_mx_grouped_gemm: A scale_inv dtype must be Float8E8M0, got ",
             static_cast<int>(a_scale_dtype));

  NVTE_CHECK(b_scale_dtype == DType::kFloat8E8M0,
             "ck_tile_mx_grouped_gemm: B scale_inv dtype must be Float8E8M0, got ",
             static_cast<int>(b_scale_dtype));

  const auto a_dtype = A0_data.dtype;
  const auto b_dtype = B0_data.dtype;
  const auto d_dtype = D0->dtype();
  NVTE_CHECK(is_fp8_dtype(a_dtype), "ck_tile_mx_grouped_gemm: A dtype must be FP8");
  NVTE_CHECK(is_fp8_dtype(b_dtype), "ck_tile_mx_grouped_gemm: B dtype must be FP8");

  void *ws_ptr = nullptr;
  size_t ws_bytes = 0;
  if (workspace) {
    auto *ws_te = convertNVTETensorCheck(*workspace);
    ws_ptr = ws_te->data.dptr;
    ws_bytes = ws_te->data.numel() * typeToSize(ws_te->data.dtype);
  }

  GroupedGemmRunContext ctx{
    .A = A_use,
    .B = B_use,
    .D = D,
    .N = 0,
    .arch = detect_gpu_arch(),
    .group_num = group_num,
    .transA = transA_use,
    .transB = transB_use,
    .workspace = ws_ptr,
    .workspace_bytes = ws_bytes,
    .stream = stream,
    .use_a_columnwise_data = use_a_colwise_data,
    .use_b_columnwise_data = use_b_colwise_data,
    .accumulate = false,
  };

  // Dispatch to per-architecture translation unit. Each backend is only built
  // when its architecture is targeted, so guard on the availability macro the
  // build defines for it.
  switch (ctx.arch) {
#ifdef NVTE_HAVE_MX_GROUPED_GEMM_GFX1250
    case GPUArch::GFX1250:
      return ck_tile_mx_grouped_gemm_dispatch_gfx1250(a_dtype, b_dtype, d_dtype, ctx);
#endif
#ifdef NVTE_HAVE_MX_GROUPED_GEMM_GFX950
    case GPUArch::GFX950:
      return ck_tile_mx_grouped_gemm_dispatch_gfx950(a_dtype, b_dtype, d_dtype, ctx);
#endif
    default:
      if (warn_fallback) {
        NVTE_WARN("ck_tile_mx_grouped_gemm: no MX grouped GEMM kernel built for this "
                  "architecture. Falling back.");
      }
      return false;
  }
}

}  // namespace grouped_gemm
}  // namespace transformer_engine

bool ck_tile_mx_grouped_gemm(const NVTETensor *A,
                             const NVTETensor *B,
                             NVTETensor *D,
                             int group_num,
                             bool transA,
                             bool transB,
                             NVTETensor *workspace,
                             bool accumulate,
                             hipStream_t stream) {
  return transformer_engine::grouped_gemm::ck_tile_mx_grouped_gemm(
    A, B, D, group_num, transA, transB, workspace, accumulate, stream);
}
