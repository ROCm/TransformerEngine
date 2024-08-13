#ifndef TRANSFORMER_ENGINE_COMMON_FUSED_ATTN_ROCM_CK_FMHA_H_
#define TRANSFORMER_ENGINE_COMMON_FUSED_ATTN_ROCM_CK_FMHA_H_

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <string>

#define CHECK_HIP_ERROR(cmd)                                                        \
    do {                                                                            \
        hipError_t error_code = cmd;                                                \
        if (error_code != hipSuccess) {                                             \
            fprintf(stderr, "HIP error: '%s'(%d) in %s at line %d\n",               \
                    hipGetErrorString(error_code), error_code, __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                     \
        }                                                                           \
    } while (0)

void ck_fused_attn_fwd_impl(int64_t b, int64_t h, int64_t hg, int64_t s_q, int64_t s_kv, int64_t d,
                            int64_t bias_b, int64_t bias_h, bool is_training, float scaling_factor,
                            float dropout_probability, uint64_t drop_seed, uint64_t drop_offset,
                            uint32_t bias_value, uint32_t mask_value, void *devPtrQ, void *devPtrK,
                            void *devPtrV, void *devPtrBias, void *devPtrSoftmaxStats,
                            void *devPtrO, void *devPtrCuSeqlensQ, void *devPtrCuSeqlensKV,
                            const std::string &data_type, hipStream_t stream);

void ck_fused_attn_bwd_impl(int64_t b, int64_t h, int64_t hg, int64_t s_q, int64_t s_kv, int64_t d,
                            int64_t bias_b, int64_t bias_h, float scaling_factor,
                            float dropout_probability, uint64_t drop_seed, uint64_t drop_offset,
                            uint32_t bias_value, uint32_t mask_value, void *devPtrQ,
                            void *devPtrKTranspose, void *devPtrVTranspose, void *devPtrO,
                            void *devPtrSoftmaxStats, void *devPtrBias, void *devPtrdQ,
                            void *devPtrdK, void *devPtrdV, void *devPtrdO, void *devPtrdBias,
                            void *devPtrCuSeqlensQ, void *devPtrCuSeqlensKV,
                            const std::string &data_type, void *devPtrSpace, bool deterministic,
                            hipStream_t stream);

#endif
