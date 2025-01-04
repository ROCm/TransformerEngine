# This file was modified for portability to AMDGPU
# Copyright (c) 2022-2024, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

list(APPEND test_cuda_sources
            test_qdq.cu
            test_cast_transpose.cu
            test_transpose.cu
            test_cast_transpose_dbias.cu
            test_cast_transpose_dbias_dgelu.cu
            test_cast_transpose_dgeglu.cu
            test_act.cu
            test_layernorm.cu
            test_rmsnorm.cu
            test_multi_cast_transpose.cu
            test_multi_padding.cu
            test_causal_softmax.cu
            ../test_common.cu)
if(USE_ROCM)
  list(APPEND test_cuda_sources
              test_cublaslt_gemm.cu)
endif()

if(USE_CUDA)
  add_executable(test_operator ${test_cuda_sources})
else()
  message("${message_line}")
  message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
  message(STATUS "PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")

  set(TE ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
  set(THIRDPARTY ${TE}/3rdparty)
  list(APPEND CMAKE_MODULE_PATH "${THIRDPARTY}/hipify_torch/cmake")
  include(Hipify)
  message(STATUS "CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")

  file(REAL_PATH ../../../transformer_engine/common/include header_include_dir1)
  file(REAL_PATH ../../../transformer_engine/common header_include_dir2)
  set(header_include_dir ${header_include_dir1} ${header_include_dir2})

  message(STATUS "CUDA_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")
  message(STATUS "HEADER_INCLUDE_DIR: ${header_include_dir}")
  set(cuda_source_dir ${PROJECT_SOURCE_DIR} )
  hipify(CUDA_SOURCE_DIR ${cuda_source_dir} 
    HEADER_INCLUDE_DIR ${header_include_dir}
    CUSTOM_MAP_FILE "${TE}/hipify_custom_map.json"
  )
  get_hipified_list("${test_cuda_sources}" test_hip_sources)
  message("${message_line}")
  message(STATUS "nvte tests hipified sources: ${test_hip_sources}")

  add_executable(test_operator ${test_hip_sources})
endif()

if(USE_CUDA)
  list(APPEND test_operator_LINKER_LIBS CUDA::cudart GTest::gtest_main ${TE_LIB} CUDA::nvrtc CUDNN::cudnn)
  target_link_libraries(test_operator PUBLIC ${test_operator_LINKER_LIBS})
  target_compile_options(test_operator PRIVATE -O2)
else()
  target_link_libraries(test_operator PUBLIC hip::host hip::device GTest::gtest_main ${TE_LIB})
  target_compile_options(test_operator PRIVATE -O2)
endif()

include(GoogleTest)
gtest_discover_tests(test_operator PROPERTIES DISCOVERY_TIMEOUT 600)
