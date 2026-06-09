/*************************************************************************
 * Stub for ck_tile_grouped_gemm when USE_CK_GEMM=OFF.
 * Returning false makes cublaslt_gemm fall back to the cuBLAS path.
 ************************************************************************/

#include <hip/hip_runtime.h>

#include "transformer_engine/transformer_engine.h"

bool ck_tile_grouped_gemm(const NVTETensor* /*A*/,
                          const NVTETensor* /*B*/,
                          NVTETensor* /*D*/,
                          int /*group_num*/,
                          bool /*transA*/,
                          bool /*transB*/,
                          NVTETensor* /*workspace*/,
                          bool /*accumulate*/,
                          hipStream_t /*stream*/) {
  return false;
}
