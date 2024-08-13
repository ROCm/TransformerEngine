#include "utils.h"

#include "../common.h"
#include "transformer_engine/fused_attn.h"

namespace transformer_engine {
namespace fused_attn_rocm {

std::string get_datatype_str(DType dtype) {
    switch (dtype) {
        case DType::kFloat16:
            return "fp16";
        case DType::kBFloat16:
            return "bf16";
        default:
            NVTE_ERROR("Unexpected QKV_type");
    }
}

// get matrix strides based on matrix type
void generateMatrixStrides(int64_t b, int64_t h, int64_t s_q, int64_t s_kv, int64_t d,
                           int64_t *strideA, NVTE_QKV_Layout layout, NVTE_QKV_Matrix matrix) {
    constexpr int batch_dim_idx  = 0;
    constexpr int head_dim_idx   = 1;
    constexpr int seqlen_dim_idx = 2;
    constexpr int hidden_dim_idx = 3;

    constexpr int seqlen_transpose_dim_idx = 3;
    constexpr int hidden_transpose_dim_idx = 2;

    constexpr int seqlen_q_dim_idx  = 2;
    constexpr int seqlen_kv_dim_idx = 3;

    switch (layout) {
        case NVTE_QKV_Layout::NVTE_SB3HD:
            if ((matrix == NVTE_QKV_Matrix::NVTE_Q_Matrix) ||
                (matrix == NVTE_QKV_Matrix::NVTE_K_Matrix) ||
                (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix)) {
                strideA[batch_dim_idx]  = 3 * h * d;
                strideA[head_dim_idx]   = d;
                strideA[seqlen_dim_idx] = b * 3 * h * d;
                strideA[hidden_dim_idx] = 1;
            } else if ((matrix == NVTE_QKV_Matrix::NVTE_K_Matrix_Transpose) ||
                       (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix_Transpose)) {
                strideA[batch_dim_idx]            = 3 * h * d;
                strideA[head_dim_idx]             = d;
                strideA[seqlen_transpose_dim_idx] = b * 3 * h * d;
                strideA[hidden_transpose_dim_idx] = 1;
            } else if (matrix == NVTE_QKV_Matrix::NVTE_O_Matrix) {
                strideA[batch_dim_idx]  = h * d;
                strideA[head_dim_idx]   = d;
                strideA[seqlen_dim_idx] = b * h * d;
                strideA[hidden_dim_idx] = 1;
            }
            break;
        case NVTE_QKV_Layout::NVTE_SBH3D:
            if ((matrix == NVTE_QKV_Matrix::NVTE_Q_Matrix) ||
                (matrix == NVTE_QKV_Matrix::NVTE_K_Matrix) ||
                (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix)) {
                strideA[batch_dim_idx]  = 3 * h * d;
                strideA[head_dim_idx]   = 3 * d;
                strideA[seqlen_dim_idx] = b * 3 * h * d;
                strideA[hidden_dim_idx] = 1;
            } else if ((matrix == NVTE_QKV_Matrix::NVTE_K_Matrix_Transpose) ||
                       (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix_Transpose)) {
                strideA[batch_dim_idx]            = 3 * h * d;
                strideA[head_dim_idx]             = 3 * d;
                strideA[seqlen_transpose_dim_idx] = b * 3 * h * d;
                strideA[hidden_transpose_dim_idx] = 1;
            } else if (matrix == NVTE_QKV_Matrix::NVTE_O_Matrix) {
                strideA[batch_dim_idx]  = h * d;
                strideA[head_dim_idx]   = d;
                strideA[seqlen_dim_idx] = b * h * d;
                strideA[hidden_dim_idx] = 1;
            }
            break;
        case NVTE_QKV_Layout::NVTE_SBHD_SB2HD:
            if ((matrix == NVTE_QKV_Matrix::NVTE_K_Matrix) ||
                (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix)) {
                strideA[batch_dim_idx]  = 2 * h * d;
                strideA[head_dim_idx]   = d;
                strideA[seqlen_dim_idx] = b * 2 * h * d;
                strideA[hidden_dim_idx] = 1;
            } else if ((matrix == NVTE_QKV_Matrix::NVTE_K_Matrix_Transpose) ||
                       (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix_Transpose)) {
                strideA[batch_dim_idx]            = 2 * h * d;
                strideA[head_dim_idx]             = d;
                strideA[seqlen_transpose_dim_idx] = b * 2 * h * d;
                strideA[hidden_transpose_dim_idx] = 1;
            } else if ((matrix == NVTE_QKV_Matrix::NVTE_Q_Matrix) ||
                       (matrix == NVTE_QKV_Matrix::NVTE_O_Matrix)) {
                strideA[batch_dim_idx]  = h * d;
                strideA[head_dim_idx]   = d;
                strideA[seqlen_dim_idx] = b * h * d;
                strideA[hidden_dim_idx] = 1;
            }
            break;
        case NVTE_QKV_Layout::NVTE_SBHD_SBH2D:
            if ((matrix == NVTE_QKV_Matrix::NVTE_K_Matrix) ||
                (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix)) {
                strideA[batch_dim_idx]  = 2 * h * d;
                strideA[head_dim_idx]   = 2 * d;
                strideA[seqlen_dim_idx] = b * 2 * h * d;
                strideA[hidden_dim_idx] = 1;
            } else if ((matrix == NVTE_QKV_Matrix::NVTE_K_Matrix_Transpose) ||
                       (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix_Transpose)) {
                strideA[batch_dim_idx]            = 2 * h * d;
                strideA[head_dim_idx]             = 2 * d;
                strideA[seqlen_transpose_dim_idx] = b * 2 * h * d;
                strideA[hidden_transpose_dim_idx] = 1;
            } else if ((matrix == NVTE_QKV_Matrix::NVTE_Q_Matrix) ||
                       (matrix == NVTE_QKV_Matrix::NVTE_O_Matrix)) {
                strideA[batch_dim_idx]  = h * d;
                strideA[head_dim_idx]   = d;
                strideA[seqlen_dim_idx] = b * h * d;
                strideA[hidden_dim_idx] = 1;
            }
            break;
        case NVTE_QKV_Layout::NVTE_SBHD_SBHD_SBHD:
            if ((matrix == NVTE_QKV_Matrix::NVTE_Q_Matrix) ||
                (matrix == NVTE_QKV_Matrix::NVTE_K_Matrix) ||
                (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix) ||
                (matrix == NVTE_QKV_Matrix::NVTE_O_Matrix)) {
                strideA[batch_dim_idx]  = h * d;
                strideA[head_dim_idx]   = d;
                strideA[seqlen_dim_idx] = b * h * d;
                strideA[hidden_dim_idx] = 1;
            } else if ((matrix == NVTE_QKV_Matrix::NVTE_K_Matrix_Transpose) ||
                       (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix_Transpose)) {
                strideA[batch_dim_idx]            = h * d;
                strideA[head_dim_idx]             = d;
                strideA[seqlen_transpose_dim_idx] = b * h * d;
                strideA[hidden_transpose_dim_idx] = 1;
            }
            break;
        case NVTE_QKV_Layout::NVTE_BS3HD:
        case NVTE_QKV_Layout::NVTE_T3HD:
            if ((matrix == NVTE_QKV_Matrix::NVTE_Q_Matrix) ||
                (matrix == NVTE_QKV_Matrix::NVTE_K_Matrix) ||
                (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix)) {
                strideA[batch_dim_idx]  = s_q * 3 * h * d;
                strideA[head_dim_idx]   = d;
                strideA[seqlen_dim_idx] = 3 * h * d;
                strideA[hidden_dim_idx] = 1;
            } else if ((matrix == NVTE_QKV_Matrix::NVTE_K_Matrix_Transpose) ||
                       (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix_Transpose)) {
                strideA[batch_dim_idx]            = s_q * 3 * h * d;
                strideA[head_dim_idx]             = d;
                strideA[seqlen_transpose_dim_idx] = 3 * h * d;
                strideA[hidden_transpose_dim_idx] = 1;
            } else if (matrix == NVTE_QKV_Matrix::NVTE_O_Matrix) {
                strideA[batch_dim_idx]  = s_q * h * d;
                strideA[head_dim_idx]   = d;
                strideA[seqlen_dim_idx] = h * d;
                strideA[hidden_dim_idx] = 1;
            }
            break;
        case NVTE_QKV_Layout::NVTE_BSH3D:
        case NVTE_QKV_Layout::NVTE_TH3D:
            if ((matrix == NVTE_QKV_Matrix::NVTE_Q_Matrix) ||
                (matrix == NVTE_QKV_Matrix::NVTE_K_Matrix) ||
                (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix)) {
                strideA[batch_dim_idx]  = s_q * 3 * h * d;
                strideA[head_dim_idx]   = 3 * d;
                strideA[seqlen_dim_idx] = 3 * h * d;
                strideA[hidden_dim_idx] = 1;
            } else if ((matrix == NVTE_QKV_Matrix::NVTE_K_Matrix_Transpose) ||
                       (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix_Transpose)) {
                strideA[batch_dim_idx]            = s_q * 3 * h * d;
                strideA[head_dim_idx]             = 3 * d;
                strideA[seqlen_transpose_dim_idx] = 3 * h * d;
                strideA[hidden_transpose_dim_idx] = 1;
            } else if (matrix == NVTE_QKV_Matrix::NVTE_O_Matrix) {
                strideA[batch_dim_idx]  = s_q * h * d;
                strideA[head_dim_idx]   = d;
                strideA[seqlen_dim_idx] = h * d;
                strideA[hidden_dim_idx] = 1;
            }
            break;
        case NVTE_QKV_Layout::NVTE_BSHD_BS2HD:
        case NVTE_QKV_Layout::NVTE_THD_T2HD:
            if ((matrix == NVTE_QKV_Matrix::NVTE_K_Matrix) ||
                (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix)) {
                strideA[batch_dim_idx]  = s_kv * 2 * h * d;
                strideA[head_dim_idx]   = d;
                strideA[seqlen_dim_idx] = 2 * h * d;
                strideA[hidden_dim_idx] = 1;
            } else if ((matrix == NVTE_QKV_Matrix::NVTE_K_Matrix_Transpose) ||
                       (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix_Transpose)) {
                strideA[batch_dim_idx]            = s_kv * 2 * h * d;
                strideA[head_dim_idx]             = d;
                strideA[seqlen_transpose_dim_idx] = 2 * h * d;
                strideA[hidden_transpose_dim_idx] = 1;
            } else if ((matrix == NVTE_QKV_Matrix::NVTE_Q_Matrix) ||
                       (matrix == NVTE_QKV_Matrix::NVTE_O_Matrix)) {
                strideA[batch_dim_idx]  = s_q * h * d;
                strideA[head_dim_idx]   = d;
                strideA[seqlen_dim_idx] = h * d;
                strideA[hidden_dim_idx] = 1;
            }
            break;
        case NVTE_QKV_Layout::NVTE_BSHD_BSH2D:
        case NVTE_QKV_Layout::NVTE_THD_TH2D:
            if ((matrix == NVTE_QKV_Matrix::NVTE_K_Matrix) ||
                (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix)) {
                strideA[batch_dim_idx]  = s_kv * 2 * h * d;
                strideA[head_dim_idx]   = 2 * d;
                strideA[seqlen_dim_idx] = 2 * h * d;
                strideA[hidden_dim_idx] = 1;
            } else if ((matrix == NVTE_QKV_Matrix::NVTE_K_Matrix_Transpose) ||
                       (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix_Transpose)) {
                strideA[batch_dim_idx]            = s_kv * 2 * h * d;
                strideA[head_dim_idx]             = 2 * d;
                strideA[seqlen_transpose_dim_idx] = 2 * h * d;
                strideA[hidden_transpose_dim_idx] = 1;
            } else if ((matrix == NVTE_QKV_Matrix::NVTE_Q_Matrix) ||
                       (matrix == NVTE_QKV_Matrix::NVTE_O_Matrix)) {
                strideA[batch_dim_idx]  = s_q * h * d;
                strideA[head_dim_idx]   = d;
                strideA[seqlen_dim_idx] = h * d;
                strideA[hidden_dim_idx] = 1;
            }
            break;
        case NVTE_QKV_Layout::NVTE_BSHD_BSHD_BSHD:
        case NVTE_QKV_Layout::NVTE_THD_THD_THD:
            if ((matrix == NVTE_QKV_Matrix::NVTE_Q_Matrix) ||
                (matrix == NVTE_QKV_Matrix::NVTE_O_Matrix)) {
                strideA[batch_dim_idx]  = s_q * h * d;
                strideA[head_dim_idx]   = d;
                strideA[seqlen_dim_idx] = h * d;
                strideA[hidden_dim_idx] = 1;
            } else if ((matrix == NVTE_QKV_Matrix::NVTE_K_Matrix) ||
                       (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix)) {
                strideA[batch_dim_idx]  = s_kv * h * d;
                strideA[head_dim_idx]   = d;
                strideA[seqlen_dim_idx] = h * d;
                strideA[hidden_dim_idx] = 1;
            } else if ((matrix == NVTE_QKV_Matrix::NVTE_K_Matrix_Transpose) ||
                       (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix_Transpose)) {
                strideA[batch_dim_idx]            = s_kv * h * d;
                strideA[head_dim_idx]             = d;
                strideA[seqlen_transpose_dim_idx] = h * d;
                strideA[hidden_transpose_dim_idx] = 1;
            }
            break;
    }

    if (matrix == NVTE_QKV_Matrix::NVTE_S_Matrix) {
        strideA[seqlen_kv_dim_idx] = 1;
        strideA[seqlen_q_dim_idx]  = s_kv;
        strideA[head_dim_idx]      = s_q * s_kv;
        strideA[batch_dim_idx]     = h * s_q * s_kv;
    }
}

// convert cu_seqlens_q to qkv/o_ragged_offset and actual_seqlens_q
__global__ void cu_seqlens_to_offsets(size_t b, size_t h, size_t d, int32_t *cu_seqlens_q,
                                      int32_t *actual_seqlens_q, int32_t *qkv_ragged_offset,
                                      int32_t *o_ragged_offset) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < b) {
        actual_seqlens_q[tid] = cu_seqlens_q[tid + 1] - cu_seqlens_q[tid];
    }
    if (tid < b + 1) {
        qkv_ragged_offset[tid] = cu_seqlens_q[tid] * 3 * h * d;
        o_ragged_offset[tid]   = cu_seqlens_q[tid] * h * d;
    }
}

// convert cu_seqlens to actual_seqlens
__global__ void cu_seqlens_to_actual_seqlens(size_t b, int32_t const *const q_cu_seqlens,
                                             int32_t const *const kv_cu_seqlens, int32_t *q_seqlens,
                                             int32_t *kv_seqlens) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < b) {
        q_seqlens[tid]  = q_cu_seqlens[tid + 1] - q_cu_seqlens[tid];
        kv_seqlens[tid] = kv_cu_seqlens[tid + 1] - kv_cu_seqlens[tid];
    }
}

}  // namespace fused_attn_rocm
}  // namespace transformer_engine
