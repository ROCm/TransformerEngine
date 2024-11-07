/*************************************************************************
 * Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#include "transformer_engine/fused_attn.h"
#include "../common.h"
#include "utils.h"

namespace transformer_engine {
namespace fused_attn_rocm {

using namespace transformer_engine;

size_t nvte_dtype_size(DType t_dtype){
  switch(t_dtype){
    case DType::kByte: 
      return 1;
    case DType::kInt32: 
      return 4;
    case DType::kInt64: 
      return 8;
    case DType::kFloat32: 
      return 4;
    case DType::kFloat16: 
      return 2;
    case DType::kBFloat16: 
      return 2;
    case DType::kFloat8E4M3: 
    case DType::kFloat8E5M2: 
      return 1;
    default:
      return 1;
  }
  return 1;
}

// get matrix strides based on matrix type
void generateMatrixStrides(
            uint64_t b, uint64_t h,
            uint64_t s_q, uint64_t s_kv,
            uint64_t d, uint64_t* stride,
            NVTE_QKV_Layout layout, NVTE_QKV_Matrix matrix) {
    // AOTriton internally takes BHSD for implementation
    constexpr int batch_dim_idx   = 0;
    constexpr int head_dim_idx    = 1;
    constexpr int seqlen_dim_idx  = 2;
    constexpr int hidden_dim_idx  = 3;

    switch (layout) {
        case NVTE_QKV_Layout::NVTE_SB3HD:
            if ((matrix == NVTE_QKV_Matrix::NVTE_Q_Matrix)
                || (matrix == NVTE_QKV_Matrix::NVTE_K_Matrix)
                || (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix)) {
                    stride[batch_dim_idx] = 3 * h * d;
                    stride[head_dim_idx] = d;
                    stride[seqlen_dim_idx] = b * 3 * h * d;
                    stride[hidden_dim_idx] = 1;
            } else if (matrix == NVTE_QKV_Matrix::NVTE_O_Matrix) {
                    stride[batch_dim_idx] = h * d;
                    stride[head_dim_idx] = d;
                    stride[seqlen_dim_idx] = b * h * d;
                    stride[hidden_dim_idx] = 1;
            }
            break;
        case NVTE_QKV_Layout::NVTE_SBH3D:
            if ((matrix == NVTE_QKV_Matrix::NVTE_Q_Matrix)
                || (matrix == NVTE_QKV_Matrix::NVTE_K_Matrix)
                || (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix)) {
                    stride[batch_dim_idx] = 3 * h * d;
                    stride[head_dim_idx] = 3 * d;
                    stride[seqlen_dim_idx] = b * 3 * h * d;
                    stride[hidden_dim_idx] = 1;
            } else if (matrix == NVTE_QKV_Matrix::NVTE_O_Matrix) {
                    stride[batch_dim_idx] = h * d;
                    stride[head_dim_idx] = d;
                    stride[seqlen_dim_idx] = b * h * d;
                    stride[hidden_dim_idx] = 1;
            }
            break;
        case NVTE_QKV_Layout::NVTE_SBHD_SB2HD:
            if ((matrix == NVTE_QKV_Matrix::NVTE_K_Matrix)
                || (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix)) {
                    stride[batch_dim_idx] = 2 * h * d;
                    stride[head_dim_idx] = d;
                    stride[seqlen_dim_idx] = b * 2 * h * d;
                    stride[hidden_dim_idx] = 1;
            } else if ((matrix == NVTE_QKV_Matrix::NVTE_Q_Matrix)
                || (matrix == NVTE_QKV_Matrix::NVTE_O_Matrix)) {
                    stride[batch_dim_idx] = h * d;
                    stride[head_dim_idx] = d;
                    stride[seqlen_dim_idx] = b * h * d;
                    stride[hidden_dim_idx] = 1;
            }
            break;
        case NVTE_QKV_Layout::NVTE_SBHD_SBH2D:
            if ((matrix == NVTE_QKV_Matrix::NVTE_K_Matrix)
                || (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix)) {
                    stride[batch_dim_idx] = 2 * h * d;
                    stride[head_dim_idx] = 2 * d;
                    stride[seqlen_dim_idx] = b * 2 * h * d;
                    stride[hidden_dim_idx] = 1;
            } else if ((matrix == NVTE_QKV_Matrix::NVTE_Q_Matrix)
                || (matrix == NVTE_QKV_Matrix::NVTE_O_Matrix)) {
                    stride[batch_dim_idx] = h * d;
                    stride[head_dim_idx] = d;
                    stride[seqlen_dim_idx] = b * h * d;
                    stride[hidden_dim_idx] = 1;
            }
            break;
        case NVTE_QKV_Layout::NVTE_SBHD_SBHD_SBHD:
            if ((matrix == NVTE_QKV_Matrix::NVTE_Q_Matrix)
                || (matrix == NVTE_QKV_Matrix::NVTE_K_Matrix)
                || (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix)
                || (matrix == NVTE_QKV_Matrix::NVTE_O_Matrix)) {
                    stride[batch_dim_idx] = h * d;
                    stride[head_dim_idx] = d;
                    stride[seqlen_dim_idx] = b * h * d;
                    stride[hidden_dim_idx] = 1;
            }
            break;
        case NVTE_QKV_Layout::NVTE_BS3HD:
        case NVTE_QKV_Layout::NVTE_T3HD:
            if ((matrix == NVTE_QKV_Matrix::NVTE_Q_Matrix)
                || (matrix == NVTE_QKV_Matrix::NVTE_K_Matrix)
                || (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix)) {
                    stride[batch_dim_idx] = s_q * 3 * h * d;
                    stride[head_dim_idx] = d;
                    stride[seqlen_dim_idx] = 3 * h * d;
                    stride[hidden_dim_idx] = 1;
            } else if (matrix == NVTE_QKV_Matrix::NVTE_O_Matrix) {
                    stride[batch_dim_idx] = s_q * h * d;
                    stride[head_dim_idx] = d;
                    stride[seqlen_dim_idx] = h * d;
                    stride[hidden_dim_idx] = 1;
            }
            break;
        case NVTE_QKV_Layout::NVTE_BSH3D:
        case NVTE_QKV_Layout::NVTE_TH3D:
            if ((matrix == NVTE_QKV_Matrix::NVTE_Q_Matrix)
                 || (matrix == NVTE_QKV_Matrix::NVTE_K_Matrix)
                 || (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix)) {
                     stride[batch_dim_idx] = s_q * 3 * h * d;
                     stride[head_dim_idx] = 3 * d;
                     stride[seqlen_dim_idx] = 3 * h * d;
                     stride[hidden_dim_idx] = 1;
             } else if (matrix == NVTE_QKV_Matrix::NVTE_O_Matrix) {
                     stride[batch_dim_idx] = s_q * h * d;
                     stride[head_dim_idx] = d;
                     stride[seqlen_dim_idx] = h * d;
                     stride[hidden_dim_idx] = 1;
             }
             break;
        case NVTE_QKV_Layout::NVTE_BSHD_BS2HD:
        case NVTE_QKV_Layout::NVTE_THD_T2HD:
            if ((matrix == NVTE_QKV_Matrix::NVTE_K_Matrix)
                 || (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix)) {
                     stride[batch_dim_idx] = s_kv * 2 * h * d;
                     stride[head_dim_idx] = d;
                     stride[seqlen_dim_idx] = 2 * h * d;
                     stride[hidden_dim_idx] = 1;
             } else if ((matrix == NVTE_QKV_Matrix::NVTE_Q_Matrix)
                 || (matrix == NVTE_QKV_Matrix::NVTE_O_Matrix)) {
                     stride[batch_dim_idx] = s_q * h * d;
                     stride[head_dim_idx] = d;
                     stride[seqlen_dim_idx] = h * d;
                     stride[hidden_dim_idx] = 1;
             }
             break;
        case NVTE_QKV_Layout::NVTE_BSHD_BSH2D:
        case NVTE_QKV_Layout::NVTE_THD_TH2D:
            if ((matrix == NVTE_QKV_Matrix::NVTE_K_Matrix)
                 || (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix)) {
                     stride[batch_dim_idx] = s_kv * 2 * h * d;
                     stride[head_dim_idx] = 2 * d;
                     stride[seqlen_dim_idx] = 2 * h * d;
                     stride[hidden_dim_idx] = 1;
             } else if ((matrix == NVTE_QKV_Matrix::NVTE_Q_Matrix)
                 || (matrix == NVTE_QKV_Matrix::NVTE_O_Matrix)) {
                     stride[batch_dim_idx] = s_q * h * d;
                     stride[head_dim_idx] = d;
                     stride[seqlen_dim_idx] = h * d;
                     stride[hidden_dim_idx] = 1;
             }
             break;
        case NVTE_QKV_Layout::NVTE_BSHD_BSHD_BSHD:
        case NVTE_QKV_Layout::NVTE_THD_THD_THD:
            if ((matrix == NVTE_QKV_Matrix::NVTE_Q_Matrix)
                || (matrix == NVTE_QKV_Matrix::NVTE_O_Matrix)) {
                    stride[batch_dim_idx] = s_q * h * d;
                    stride[head_dim_idx] = d;
                    stride[seqlen_dim_idx] = h * d;
                    stride[hidden_dim_idx] = 1;
            } else if ((matrix == NVTE_QKV_Matrix::NVTE_K_Matrix)
                || (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix)) {
                    stride[batch_dim_idx] = s_kv * h * d;
                    stride[head_dim_idx] = d;
                    stride[seqlen_dim_idx] = h * d;
                    stride[hidden_dim_idx] = 1;
            }
            break;
    }
}

}  // namespace fused_attn_rocm
}  // namespace transformer_engine
