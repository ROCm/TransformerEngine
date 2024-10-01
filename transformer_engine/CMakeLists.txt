# This file was modified for portability to AMDGPU
# Copyright (c) 2022-2024, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

cmake_minimum_required(VERSION 3.18)

option(USE_CUDA "Use CUDA" OFF)
option(USE_ROCM "Use ROCm" OFF)

set(message_line
  "-------------------------------------------------------------")
message("${message_line}")

if(USE_CUDA)
  message(STATUS "USE_CUDA ${USE_CUDA}")
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 80 89 90)
  endif()

  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)

  project(transformer_engine LANGUAGES CUDA CXX)
  # Check if CUDA is available
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "CUDA compiler not found!")
  endif()

  # Output the CUDA compiler version
  message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER}")
  message(STATUS "CUDA version: ${CMAKE_CUDA_COMPILER_VERSION}")

  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --threads 4")
  if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G")
  endif()
elseif(USE_ROCM)
  message(STATUS "USE_ROCM ${USE_ROCM}")
  if(DEFINED ENV{ROCM_PATH})
    set(ROCM_PATH $ENV{ROCM_PATH})
  elseif(EXISTS "/opt/rocm")
    set(ROCM_PATH "/opt/rocm")
  else()
    message(FATAL_ERROR "ROCm not found")
  endif()

  if(EXISTS "${ROCM_PATH}/.info/version")
    file(READ "${ROCM_PATH}/.info/version" ROCM_VERSION)
    string(STRIP "${ROCM_VERSION}" ROCM_VERSION)
    message(STATUS "ROCm detected: ${ROCM_VERSION}")
  else()
    message(WARNING "ROCm version not found")
  endif()

  add_definitions(-DUSE_ROCM)
  list(APPEND CMAKE_MODULE_PATH "${ROCM_PATH}/lib/cmake")

  # Show whether hipBLASLt is enabled
  option(USE_HIPBLASLT "Use hipBLASLt" OFF)
  message(STATUS "USE_HIPBLASLT ${USE_HIPBLASLT}")

  if(NOT DEFINED GPU_TARGETS)
    if (NOT DEFINED CMAKE_HIP_ARCHITECTURES)
      set(GPU_TARGETS gfx90a;gfx940;gfx941;gfx942)
    else()
      set(GPU_TARGETS ${CMAKE_HIP_ARCHITECTURES})
    endif()
  endif()
  if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES ${GPU_TARGETS})
  endif()
  message(STATUS "GPU_TARGETS: ${GPU_TARGETS}")
  message(STATUS "CMAKE_HIP_ARCHITECTURES: ${CMAKE_HIP_ARCHITECTURES}")

  set(CMAKE_CXX_STANDARD 17)

  project(transformer_engine LANGUAGES HIP CXX)
  # General HIP flags applied to all targets
  add_compile_options(
    -mavx2
    -mf16c
    -mfma
    -ffast-math
    -fno-gpu-rdc
    -Wno-defaulted-function-deleted)
  # build error will be dup-ed parallel-jobs times
  # list(APPEND CMAKE_HIP_FLAGS "-parallel-jobs=4")
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS_DEBUG "-mcmodel=large -O3 -ggdb")
    set(CMAKE_HIP_FLAGS_DEBUG "-mcmodel=large -O3 -ggdb")
  endif()
else()
  message(FATAL_ERROR "No GPU platform specified")
endif()

if(USE_CUDA)
  list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
  find_package(CUDAToolkit REQUIRED cublas nvToolsExt)
  find_package(CUDNN REQUIRED cudnn)
elseif(USE_ROCM)
  set(THREADS_PREFER_PTHREAD_FLAG ON)
  find_package(ROCM REQUIRED)
  find_package(HIP REQUIRED)
  find_package(Threads REQUIRED)
  if (USE_HIPBLASLT)
    add_definitions(-DUSE_HIPBLASLT)
    find_package(hipblaslt)
  else()
    find_package(rocblas)
  endif()
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

include_directories(${PROJECT_SOURCE_DIR})

add_subdirectory(common)
if(USE_CUDA)
  if(NVTE_WITH_USERBUFFERS)
    message(STATUS "userbuffers support enabled")
    add_subdirectory(pytorch/csrc/userbuffers)
  endif()
endif()

option(ENABLE_JAX "Enable JAX in the building workflow." OFF)
message(STATUS "JAX support: ${ENABLE_JAX}")
if(ENABLE_JAX)
  find_package(pybind11 CONFIG REQUIRED)
  add_subdirectory(jax)
endif()
