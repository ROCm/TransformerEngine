/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

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

  // Normalize similar to upstream
  // See https://github.com/NVIDIA/TransformerEngine/blob/59f6f3876767d07045152bfae07b5dd4c54e1725/transformer_engine/common/gemm/cutlass_grouped_gemm.cu#L54-L68
  // I.e., swap A and B, as well as transa and transb.
  const NVTETensor* A_use = B;
  const NVTETensor* B_use = A;
  bool transA_use = transB;
  bool transB_use = transA;
  bool use_a_colwise_data = false;
  bool use_b_colwise_data = false;

  const auto caller_a_dtype = convertNVTETensorCheck(A[0])->dtype();
  const bool is_8bit_float = is_fp8_dtype(caller_a_dtype);
  const bool is_16bit_float = is_fp16_dtype(caller_a_dtype);
  
  Tensor* A0_te = convertNVTETensorCheck(A_use[0]);
  Tensor* B0_te = convertNVTETensorCheck(B_use[0]);

  // Currently the accumulate path is only supported on fp16
  if (accumulate && is_8bit_float) {
  	return false;
  }

  // FP8 special handling.
  //
  // A_use/B_use and transA_use/transB_use have already gone through the
  // upstream-style grouped GEMM normalization above. This block only rewrites
  // that normalized presentation into the CK FP8 preferred NT presentation by selecting
  // `columnwise_data` when needed.
  //
  // CK FP8 target presentation:
  //   A_use: N
  //   B_use: T
  //
  // The outer condition checks whether this NT presentation is possible:
  //   - A_use is already N, or can be made N using columnwise_data
  //   - B_use is already T, or can be made T using columnwise_data
  //
  // Then each operand is rewritten independently only if needed:
  //   NN -> rewrite B only
  //   TN -> rewrite A and B
  //   NT -> already in target form
  //   TT -> rewrite A only
  //
  // This preserves the intended math and only changes the physical
  // storage/transpose-flag encoding seen by CK.
  if (is_8bit_float) {
    const bool has_a_col = A0_te->has_columnwise_data();
    const bool has_b_col = B0_te->has_columnwise_data();

    if ((!transA_use || has_a_col) && (transB_use || has_b_col)) {
      if (transA_use) {
        use_a_colwise_data = true;
        transA_use = false;
      }

      if (!transB_use) {
        use_b_colwise_data = true;
        transB_use = true;
      }
    }
  }

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

  GroupedGemmRunContext ctx = {
      A_use,
      B_use,
      D,
      static_cast<int>(n),
      static_cast<int>(kA),
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
