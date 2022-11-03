# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

cmake_minimum_required(VERSION 3.18)

option(USE_CUDA "Use CUDA" ON)
option(USE_ROCM "Use ROCm" OFF)

if(((EXISTS "/opt/rocm/") OR (EXISTS $ENV{ROCM_PATH}))
   AND NOT (EXISTS "/bin/nvcc"))
  message("AMD GPU detected.")
  set(USE_CUDA OFF)
  set(USE_ROCM ON)
endif()

if (USE_ROCM)
  add_definitions(-DUSE_ROCM)
endif()

if (USE_CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 80 90)
  endif()
endif()

set(message_line
    "-------------------------------------------------------------")
message(STATUS "USE_ROCM ${USE_ROCM}")

set(CMAKE_CXX_STANDARD 17)
if (USE_CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
else()
  set(CMAKE_HIP_STANDARD 17)
  set(CMAKE_HIP_STANDARD_REQUIRED ON)
endif()

if (USE_CUDA)
  project(transformer_engine LANGUAGES CUDA CXX)
else()
  project(transformer_engine LANGUAGES HIP CXX)
endif()

if (USE_CUDA)
  list(APPEND CMAKE_CUDA_FLAGS "--threads 4")
  if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    list(APPEND CMAKE_CUDA_FLAGS "-G")
  endif()
else()
  list(APPEND CMAKE_HIP_FLAGS "-parallel-jobs=4")
  if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    list(APPEND CMAKE_HIP_FLAGS "-G")
  endif()
endif()

if (USE_CUDA)
add_library(transformer_engine SHARED
                               transformer_engine.cpp
                               transpose/cast_transpose.cu
                               transpose/transpose.cu
                               transpose/cast_transpose_fusion.cu
                               activation/gelu.cu
                               gemm/cublaslt_gemm.cu
                               layer_norm/ln_api.cpp
                               layer_norm/ln_bwd_semi_cuda_kernel.cu
                               layer_norm/ln_fwd_cuda_kernel.cu
                               util/cast.cu)
else()
  list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/../../3rdparty/hipify_torch/cmake")
  set(message_line
      "-------------------------------------------------------------")
  message(STATUS "PROJECT_SOURCE_DIR= ${PROJECT_SOURCE_DIR}")
  message(STATUS "CMAKE_MODULE_PATH= ${CMAKE_MODULE_PATH}")
  include(Hipify)
  ##TODO: Redirect the hipified files to somewhere outside src. Maybe in the build directory?
  ## Somehow, hipify(CUDA_SOURCE_DIR ${PROJECT_SOURCE_DIR} HIP_SOURCE_DIR ${PROJECT_SOURCE_DIR}/hip)
  ## didn't work - get_hipified_list could not get the right list
  hipify(CUDA_SOURCE_DIR ${PROJECT_SOURCE_DIR} HEADER_INCLUDE_DIR
         "${PROJECT_SOURCE_DIR}/include")
  list(APPEND cuda_srcs_to_be_hipified
                               transformer_engine.cpp
                               transpose/cast_transpose.cu
                               transpose/transpose.cu
                               transpose/cast_transpose_fusion.cu
                               activation/gelu.cu
                               gemm/cublaslt_gemm.cu
                               layer_norm/ln_api.cpp
                               layer_norm/ln_bwd_semi_cuda_kernel.cu
                               layer_norm/ln_fwd_cuda_kernel.cu
                               util/cast.cu)
  get_hipified_list("${cuda_srcs_to_be_hipified}" hipified_cuda_srcs)
  message(STATUS "cuda_srcs_to_be_hipified= ${cuda_srcs_to_be_hipified}")
  message(STATUS "hipified_cuda_srcs= ${hipified_cuda_srcs}")
  add_library(transformer_engine SHARED ${hipified_cuda_srcs})
  #add_library(transformer_engine SHARED "${cuda_srcs_to_be_hipified}")
endif()

##TODO: This might need to be modified as well...
target_include_directories(transformer_engine PUBLIC "${PROJECT_SOURCE_DIR}/include")

if (USE_CUDA)
  find_package(CUDAToolkit REQUIRED cublas)
else()
  find_package(hip)
  find_package(rocblas)
endif()

if (USE_CUDA)
  list(APPEND transformer_engine_LINKER_LIBS CUDA::cublas CUDA::cudart)
else()
  list(APPEND transformer_engine_LINKER_LIBS roc::rocblas hip::host hip::device)
endif()
target_link_libraries(transformer_engine PUBLIC ${transformer_engine_LINKER_LIBS})


if (USE_CUDA)
  target_include_directories(transformer_engine PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
else()
  IF(NOT DEFINED ENV{ROCM_PATH})
    SET(ROCM_PATH /opt/rocm)
  ELSE()
    SET(ROCM_PATH $ENV{ROCM_PATH})
  ENDIF()
  if(NOT DEFINED ENV{ROCM_INCLUDE_DIRS})
    set(ROCM_INCLUDE_DIRS ${ROCM_PATH}/include)
  else()
    set(ROCM_INCLUDE_DIRS $ENV{ROCM_INCLUDE_DIRS})
  endif()
  target_include_directories(transformer_engine PRIVATE ${ROCM_INCLUDE_DIRS})
endif()
