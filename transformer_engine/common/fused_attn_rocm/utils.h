/*************************************************************************
 * Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
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

class FusedAttnOffsetManager {
 public:
  static FusedAttnOffsetManager &Instance() {
    static thread_local FusedAttnOffsetManager instance;
    return instance;
  }

  size_t GetAndUpdateOffset(size_t increment) {
    size_t ret = offset_;
    offset_ += increment;
    return ret;
  }

  FusedAttnOffsetManager(FusedAttnOffsetManager const &) = delete;
  void operator=(FusedAttnOffsetManager const &) = delete;

 private:
  FusedAttnOffsetManager() {}
  size_t offset_ = 0;
};

void PopulateRngStateAsync(void *rng_state_dst, 
                           const void *const seed,
                           size_t q_max_seqlen, 
                           size_t kv_max_seqlen,
                           NVTE_Fused_Attn_Backend backend,
                           cudaStream_t stream);

uint32_t GetRuntimeNumSegments(void *cu_seqlen, void *workspace, size_t len, cudaStream_t stream);

}  // namespace fused_attn_rocm
}  // namespace transformer_engine

#endif
