/*************************************************************************
 * Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

/*! \file utils.h
 *  \brief Enums and functions for fused attention in rocm.
 */


#ifndef TRANSFORMER_ENGINE_FUSED_ATTN_ROCM_UTILS_H_
#define TRANSFORMER_ENGINE_FUSED_ATTN_ROCM_UTILS_H_

#include "transformer_engine/fused_attn.h"
#include "transformer_engine/transformer_engine.h"


namespace transformer_engine {
namespace fused_attn_rocm {

using namespace transformer_engine;

enum NVTE_QKV_Matrix {
  NVTE_Q_Matrix            = 0,  // queries
  NVTE_K_Matrix            = 1,  // keys
  NVTE_V_Matrix            = 2,  // values
  NVTE_O_Matrix            = 3,  // final output
};

void generateMatrixStrides(
            uint64_t b, uint64_t h,
            uint64_t s_q, uint64_t s_kv,
            uint64_t d, uint64_t* stride,
            NVTE_QKV_Layout layout, NVTE_QKV_Matrix matrix);

size_t nvte_dtype_size(DType t_dtype);


}  // namespace fused_attn_rocm
}  // namespace transformer_engine

#endif
