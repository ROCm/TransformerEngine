#ifndef TRANSFORMER_ENGINE_COMMON_FUSED_ATTN_ROCM_CK_FMHA_H_
#define TRANSFORMER_ENGINE_COMMON_FUSED_ATTN_ROCM_CK_FMHA_H_

#include <hip/hip_runtime.h>

#include <iostream>
#include <cstdint>
#include <string>

#define CHECK_HIP_ERROR(expr)                                                            \
    do {                                                                                 \
        const hipError_t error_code = (expr);                                            \
        if (error_code != hipSuccess) {                                                  \
            std::string error_str = hipGetErrorString(error_code);                       \
            std::cerr << "HIP error: " << error_str << " in " << __FILE__ << " at line " \
                      << __LINE__ << std::endl;                                          \
            throw std::runtime_error(error_str);                                         \
        }                                                                                \
    } while (0)

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
                            bool deterministic, hipStream_t stream);

#endif
