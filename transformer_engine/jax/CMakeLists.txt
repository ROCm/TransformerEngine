# This file was modified for portability to AMDGPU
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

if(USE_CUDA)
  pybind11_add_module(
    transformer_engine_jax
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/extensions.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/modules.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/utils.cu
  )
else()
  message("${message_line}")
  message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
  message(STATUS "PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")

  set(TE ${CMAKE_CURRENT_SOURCE_DIR}/../..)
  set(THIRDPARTY ${TE}/3rdparty)
  list(APPEND CMAKE_MODULE_PATH "${THIRDPARTY}/hipify_torch/cmake")
  include(Hipify)
  set(header_include_dir
      ${CMAKE_CURRENT_SOURCE_DIR}/..
      ${CMAKE_CURRENT_SOURCE_DIR}/../common
      ${CMAKE_CURRENT_SOURCE_DIR}/../common/include
      ${CMAKE_CURRENT_SOURCE_DIR}/../common/util
      ${CMAKE_CURRENT_SOURCE_DIR}/../common/rmsnorm
      ${CMAKE_CURRENT_SOURCE_DIR}/../common/layer_norm
      ${CMAKE_CURRENT_SOURCE_DIR}/csrc)
 
  message(STATUS "CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")
  message(STATUS "HIPIFY CUDA_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
  hipify(CUDA_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
      HEADER_INCLUDE_DIR ${header_include_dir}
      IGNORES "*/amd_detail/*"
      IGNORES "*/aotriton/*"
      CUSTOM_MAP_FILE "${TE}/hipify_custom_map.json"
  )
  get_hipified_list("${transformer_engine_SOURCES}" te_hip_sources)

  message("${message_line}")
  message(STATUS "nvte hipified sources: ${te_hip_sources}")

  pybind11_add_module(
    transformer_engine_jax
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/extensions_hip.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/modules_hip.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/utils.hip
  )
endif()

if(USE_CUDA)
  target_link_libraries(transformer_engine_jax PRIVATE CUDA::cudart CUDA::cublas CUDA::cublasLt transformer_engine)
else()
  find_package(hip)
  find_package(rocblas)
  target_link_libraries(transformer_engine_jax PRIVATE roc::rocblas hip::host hip::device transformer_engine)
endif()

install(TARGETS transformer_engine_jax DESTINATION .)
