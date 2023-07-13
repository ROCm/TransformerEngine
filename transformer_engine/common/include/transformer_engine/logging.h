/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_LOGGING_H_
#define TRANSFORMER_ENGINE_LOGGING_H_

#include <cuda_runtime_api.h>
#ifdef __HIP_PLATFORM_HCC__
#define ROCBLAS_BETA_FEATURES_API
#include <rocblas/rocblas.h>
#define USE_HIPBLASLT
#include <hipblaslt/hipblaslt.h>
#else
#include <cublas_v2.h>
#endif
#include <string>
#include <stdexcept>

#define NVTE_ERROR(x) \
    do { \
        throw std::runtime_error(std::string(__FILE__ ":") + std::to_string(__LINE__) +            \
                                 " in function " + __func__ + ": " + x);                           \
    } while (false)

#define NVTE_CHECK(x, ...)                                                                         \
    do {                                                                                           \
        if (!(x)) {                                                                                \
            NVTE_ERROR(std::string("Assertion failed: "  #x ". ") + std::string(__VA_ARGS__));     \
        }                                                                                          \
    } while (false)

namespace {

inline void check_cuda_(cudaError_t status) {
    if ( status != cudaSuccess ) {
        NVTE_ERROR("CUDA Error: " + std::string(cudaGetErrorString(status)));
    }
}

#ifdef __HIP_PLATFORM_HCC__
#ifdef USE_HIPBLASLT
inline void check_cublas_(hipblasStatus_t status) {
    if ( status != HIPBLAS_STATUS_SUCCESS ) {
        NVTE_ERROR("HIPBLASLT Error: " + std::to_string((int)status) );
    }
}
#else
inline void check_cublas_(cublasStatus_t status) {
    if ( status != rocblas_status_success ) {
        NVTE_ERROR("ROCBLAS Error: " + std::string(rocblas_status_to_string(status)));
    }
}
#endif
#else
inline void check_cublas_(cublasStatus_t status) {
    if ( status != CUBLAS_STATUS_SUCCESS ) {
        NVTE_ERROR("CUBLAS Error: " + std::string(cublasGetStatusString(status)));
    }
}
#endif

}  // namespace

#define NVTE_CHECK_CUDA(ans) { check_cuda_(ans); }

#define NVTE_CHECK_CUBLAS(ans) { check_cublas_(ans); }

#endif  // TRANSFORMER_ENGINE_LOGGING_H_
