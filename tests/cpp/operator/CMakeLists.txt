# This file was modified for portability to AMDGPU
# Copyright (c) 2022-2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

list(APPEND test_cuda_sources
            test_cast.cu
            test_cast_current_scaling.cu
            test_cast_dbias.cu
            test_cast_dbias_dgelu.cu
            test_cast_gated_swiglu.cu
            test_cast_mxfp8_gated_swiglu.cu
            test_qdq.cu
            test_cast_mxfp8.cu
            test_dequantize_mxfp8.cu
            test_transpose.cu
            test_cast_transpose.cu
            test_cast_transpose_current_scaling.cu
            test_cast_transpose_dbias.cu
            test_cast_transpose_dbias_dgelu.cu
            test_cast_transpose_dgeglu.cu
            test_act.cu
            test_normalization.cu
            test_normalization_mxfp8.cu
            test_memset.cu
            test_multi_cast_transpose.cu
            test_multi_padding.cu
            test_multi_unpadding.cu
            test_causal_softmax.cu
            test_swap_first_dims.cu
	          ../test_common.cu)
if(USE_CUDA)
  list(APPEND test_cuda_sources
              test_cast_nvfp4_transpose.cu
              test_cast_float8blockwise.cu
              test_swizzle.cu)
else()
  list(APPEND test_cuda_sources
              test_cublaslt_gemm.cu)
endif()

if(USE_CUDA)
  add_executable(test_operator ${test_cuda_sources})
else()
  TE_GetHipifiedSources("${test_cuda_sources}" ${CMAKE_CURRENT_SOURCE_DIR} test_hip_sources)
  message("${message_line}")
  message(STATUS "test_operator hipified sources: ${test_hip_sources}")

  add_executable(test_operator ${test_hip_sources})
endif()

# Find required packages
find_package(OpenMP REQUIRED)
if(USE_CUDA)
  list(APPEND test_operator_LINKER_LIBS CUDA::cudart GTest::gtest_main ${TE_LIB} CUDA::nvrtc CUDNN::cudnn)
  target_link_libraries(test_operator PUBLIC ${test_operator_LINKER_LIBS} OpenMP::OpenMP_CXX)
else()
  target_link_libraries(test_operator PUBLIC hip::host hip::device GTest::gtest_main ${TE_LIB} OpenMP::OpenMP_CXX hiprand)
endif()
target_compile_options(test_operator PRIVATE -O2 -fopenmp)

include(GoogleTest)
gtest_discover_tests(test_operator DISCOVERY_TIMEOUT 600)
