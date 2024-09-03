# This file was modified for portability to AMDGPU
# Copyright (c) 2022-2024, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

cmake_minimum_required(VERSION 3.18)

option(USE_CUDA "Use CUDA" ON)
option(USE_ROCM "Use ROCm" OFF)
option(USE_HIPBLASLT "Use HIPBLASLT" OFF)

# Determine ROCM_PATH
if(DEFINED ENV{ROCM_PATH} AND EXISTS "$ENV{ROCM_PATH}")
  set(ROCM_PATH "$ENV{ROCM_PATH}" CACHE PATH "Path to ROCm installation")
elseif(EXISTS "/opt/rocm")
  set(ROCM_PATH "/opt/rocm" CACHE PATH "Default ROCm installation path")
else()
  set(ROCM_PATH "" CACHE PATH "ROCm installation path not found")
endif()

# Check if ROCm is available and CUDA is not installed
if(ROCM_PATH AND NOT EXISTS "/bin/nvcc")
  message(STATUS "AMD GPU detected.")
  set(USE_ROCM ON)
  set(USE_CUDA OFF)
  add_definitions(-DUSE_ROCM)
  list(APPEND CMAKE_MODULE_PATH "${ROCM_PATH}/lib/cmake")
endif()

set(message_line
  "-------------------------------------------------------------")
message("${message_line}")
message(STATUS "USE_CUDA ${USE_CUDA}")
message(STATUS "USE_ROCM ${USE_ROCM}")
message(STATUS "USE_HIPBLASLT ${USE_HIPBLASLT}")

if(USE_CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 80 89 90)
  endif()
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  project(transformer_engine LANGUAGES CUDA CXX)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --threads 4")
  if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G")
  endif()
elseif(USE_ROCM)
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
    set(CMAKE_HIP_FLAGS_DEBUG "${CMAKE_HIP_FLAGS_DEBUG} -g")
  endif()
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
