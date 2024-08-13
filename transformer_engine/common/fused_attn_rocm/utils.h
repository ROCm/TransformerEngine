#ifndef TRANSFORMER_ENGINE_FUSED_ATTN_UTILS_H_
#define TRANSFORMER_ENGINE_FUSED_ATTN_UTILS_H_

#include "transformer_engine/fused_attn.h"
#include "transformer_engine/transformer_engine.h"

namespace transformer_engine {
namespace fused_attn_rocm {

enum NVTE_QKV_Matrix {
    NVTE_Q_Matrix           = 0,  // queries
    NVTE_K_Matrix           = 1,  // keys
    NVTE_K_Matrix_Transpose = 2,  // keys transposed
    NVTE_V_Matrix           = 3,  // values
    NVTE_V_Matrix_Transpose = 4,  // value matrix transposed
    NVTE_S_Matrix           = 5,  // output of GEMM1
    NVTE_O_Matrix           = 6,  // final output
};

std::string get_datatype_str(DType dtype);

void generateMatrixStrides(int64_t b, int64_t h, int64_t s_q, int64_t s_kv, int64_t d,
                           int64_t *strideA, NVTE_QKV_Layout layout, NVTE_QKV_Matrix matrix);

__global__ void cu_seqlens_to_offsets(size_t b, size_t h, size_t d, int32_t *cu_seqlens_q,
                                      int32_t *actual_seqlens_q, int32_t *qkv_ragged_offset,
                                      int32_t *o_ragged_offset);

__global__ void cu_seqlens_to_actual_seqlens(size_t b, int32_t const *const q_cu_seqlens,
                                             int32_t const *const kv_cu_seqlens, int32_t *q_seqlens,
                                             int32_t *kv_seqlens);

}  // namespace fused_attn_rocm
}  // namespace transformer_engine

#endif
