#ifndef TRANSFORMER_ENGINE_COMMON_FUSED_ATTN_ROCM_CK_FMHA_H_
#define TRANSFORMER_ENGINE_COMMON_FUSED_ATTN_ROCM_CK_FMHA_H_

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

#include "memory_manager.h"

void ck_fused_attn_fwd_impl(int64_t b, int64_t h, int64_t hg, int64_t s_q, int64_t s_kv, int64_t d,
                            int64_t bias_b, int64_t bias_h, bool is_training, float scaling_factor,
                            float dropout_probability, uint64_t drop_seed, uint64_t drop_offset,
                            uint32_t bias_type, uint32_t mask_type, void *devPtrQ, void *devPtrK,
                            void *devPtrV, void *devPtrBias, void *devPtrSoftmaxStats,
                            void *devPtrO, void *devPtrCuSeqlensQ, void *devPtrCuSeqlensKV,
                            const std::string &data_type, hipStream_t stream);

void ck_fused_attn_bwd_impl(int64_t b, int64_t h, int64_t hg, int64_t s_q, int64_t s_kv, int64_t d,
                            int64_t bias_b, int64_t bias_h, float scaling_factor,
                            float dropout_probability, uint64_t drop_seed, uint64_t drop_offset,
                            uint32_t bias_type, uint32_t mask_type, void *devPtrQ,
                            void *devPtrKTranspose, void *devPtrVTranspose, void *devPtrO,
                            void *devPtrSoftmaxStats, void *devPtrBias, void *devPtrdQ,
                            void *devPtrdK, void *devPtrdV, void *devPtrdO, void *devPtrdBias,
                            void *devPtrCuSeqlensQ, void *devPtrCuSeqlensKV,
                            const std::string &data_type, void *workspace, size_t *workspace_size,
                            bool deterministic, bool ext_asm, bool asm_atomic_fp32,
                            bool asm_no_coex, bool asm_rtz_cvt, hipStream_t stream);

#endif
