/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#include "../ck_gemm_common.h"
#include "ck_grouped_gemm_common.h"
#include "ck_grouped_gemm_fp16.h"
#include "ck_grouped_gemm_fp8.h"

bool ck_tile_grouped_gemm(const NVTETensor* A,
                          const NVTETensor* B,
                          NVTETensor* D,
                          int group_num,
                          bool transA,
                          bool transB,
                          NVTETensor* workspace,
                          bool accumulate,
                          hipStream_t stream) {
  if (group_num <= 0) {
    return true;
  }

  using namespace transformer_engine;
  using namespace transformer_engine::grouped_gemm;

  void* ws_ptr = nullptr;
  size_t ws_bytes = 0;
  if (workspace) {
    auto* ws_te = convertNVTETensorCheck(*workspace);
    ws_ptr = ws_te->data.dptr;
    ws_bytes = ws_te->data.numel() * typeToSize(ws_te->data.dtype);
  }

  const auto norm = normalize_gemm_inputs(A, B, transA, transB);
  const NVTETensor* A_use = norm.A;
  const NVTETensor* B_use = norm.B;
  bool transA_use = norm.transA;
  bool transB_use = norm.transB;

  bool use_a_colwise_data = false;
  bool use_b_colwise_data = false;

  const auto caller_a_dtype = convertNVTETensorCheck(A[0])->dtype();
  const bool is_8bit_float = is_fp8_dtype(caller_a_dtype);
  const bool is_16bit_float = is_fp16_dtype(caller_a_dtype);

  Tensor* A0_te = convertNVTETensorCheck(A_use[0]);
  Tensor* B0_te = convertNVTETensorCheck(B_use[0]);

  // Currently the accumulate path is only supported for fp16.
  if (accumulate && is_8bit_float) {
    return false;
  }

  // Select CK's preferred FP8 NT presentation when columnwise storage is available.
  const auto presentation = select_ck_fp8_nt_presentation(
    is_8bit_float,
    transA_use,
    transB_use,
    A0_te->has_columnwise_data(),
    B0_te->has_columnwise_data());

  transA_use = presentation.transA;
  transB_use = presentation.transB;
  use_a_colwise_data = presentation.use_a_colwise_data;
  use_b_colwise_data = presentation.use_b_colwise_data;

  const auto a_dtype = convertNVTETensorCheck(A_use[0])->dtype();
  const auto b_dtype = convertNVTETensorCheck(B_use[0])->dtype();

  Tensor* D0_te = convertNVTETensorCheck(D[0]);
  const auto d_dtype = D0_te->dtype();

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

  if (is_16bit_float) {
    return ck_tile_grouped_gemm_fp16_dispatch(a_dtype, b_dtype, d_dtype, ctx);
  } else if (is_8bit_float) {
    return ck_tile_grouped_gemm_fp8_dispatch(a_dtype, b_dtype, d_dtype, ctx);
  }

  NVTE_WARN("ck_tile_grouped_gemm: input dtype is neither fp16 nor fp8.");
  return false;
}
