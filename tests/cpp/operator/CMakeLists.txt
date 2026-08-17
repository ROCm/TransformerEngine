# This file was modified for portability to AMDGPU
# Copyright (c) 2022-2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

add_executable(test_operator
               test_cast.cu
               test_cast_current_scaling.cu
               test_cast_dbias.cu
               test_cast_dbias_dgelu.cu
               test_cast_gated_swiglu.cu
               test_cast_mxfp8_gated_swiglu.cu
               test_qdq.cu
               test_cast_mxfp8.cu
               test_cast_mxfp8_grouped.cu
               test_cast_nvfp4_transpose.cu
               test_cast_float8blockwise.cu
               test_dequantize_mxfp8.cu
               test_dequantize_mxfp8_grouped.cu
               test_dequantize_nvfp4.cu
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
               test_splits_to_offsets.cu
               test_multi_cast_transpose.cu
               test_multi_padding.cu
               test_multi_unpadding.cu
               test_causal_softmax.cu
               test_swizzle.cu
               test_multi_swizzle.cu
               test_swap_first_dims.cu
               test_grouped_gemm.cu #CUDA-only test
               ../test_common.cu)

if(USE_ROCM)
  get_target_property(test_cuda_sources test_operator SOURCES)
  # Remove CUDA-only tests and add ROCm specific ones
  list(REMOVE_ITEM test_cuda_sources
              test_grouped_gemm.cu)
  list(APPEND test_cuda_sources
              test_multi_tensor_transpose_bhsd.cu
              test_dequantize_nvfp4.cu
              test_cublaslt_gemm.cu
              test_cast_mxfp4_transpose.cu
              test_multi_quantize_mxfp8.cu)

  list(FIND CMAKE_HIP_ARCHITECTURES "gfx1250" _gfx1250_idx)
  if(NOT _gfx1250_idx EQUAL -1)
    list(APPEND test_cuda_sources test_ck_grouped_mxfp8.cu)
    target_include_directories(test_operator BEFORE PRIVATE
              ${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/composable_kernel/include)
  endif()

  TE_GetHipifiedSources("${test_cuda_sources}" ${CMAKE_CURRENT_SOURCE_DIR} test_hip_sources)
  TE_AddHipifyDeps("${test_cuda_sources}" ${CMAKE_CURRENT_SOURCE_DIR})
  message("${message_line}")
  message(STATUS "test_operator hipified sources: ${test_hip_sources}")
  set_target_properties(test_operator PROPERTIES SOURCES "${test_hip_sources}")
endif()

# Find required packages
find_package(OpenMP REQUIRED)
if(USE_CUDA)
  list(APPEND test_operator_LINKER_LIBS CUDA::cudart GTest::gtest_main ${TE_LIB} CUDA::nvrtc CUDNN::cudnn)
  target_link_libraries(test_operator PUBLIC ${test_operator_LINKER_LIBS} OpenMP::OpenMP_CXX)
else()
  target_link_libraries(test_operator PUBLIC GTest::gtest_main ${TE_LIB} OpenMP::OpenMP_CXX ${TE_ROCM_TEST_LIBS})
endif()
target_compile_options(test_operator PRIVATE -O2 -fopenmp)

include(GoogleTest)
gtest_discover_tests(test_operator DISCOVERY_TIMEOUT 600)
