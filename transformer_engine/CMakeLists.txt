# This file was modified for portability to AMDGPU
# Copyright (c) 2022-2024, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

cmake_minimum_required(VERSION 3.18)

option(USE_CUDA "Use CUDA" ON)
option(USE_ROCM "Use ROCm" OFF)
option(USE_HIPBLASLT "Use HIPBLASLT" OFF)

# Check if ROCM_PATH environment variable exists and is a valid path, or if the fallback path exists
set(ROCM_PATH_EXISTS FALSE)
if(DEFINED ENV{ROCM_PATH} AND EXISTS "$ENV{ROCM_PATH}")
  set(ROCM_PATH_EXISTS TRUE)
elseif(EXISTS "/opt/rocm")
  set(ROCM_PATH_EXISTS TRUE)
endif()

# Check if the nvcc compiler does not exist
if(ROCM_PATH_EXISTS AND NOT EXISTS "/bin/nvcc")
  message(STATUS "AMD GPU detected.")
  set(USE_ROCM ON)
  set(USE_CUDA OFF)

  # Disable Asserts In Code (Can't use asserts on HIP stack)
  add_definitions(-DNDEBUG)
  add_definitions(-DUSE_ROCM)

  if(DEFINED ENV{PYTORCH_ROCM_ARCH})
    set(ROCM_ARCH $ENV{PYTORCH_ROCM_ARCH})
  else()
    set(ROCM_ARCH gfx90a;gfx940;gfx941;gfx942)
  endif()

  set(CMAKE_HIP_ARCHITECTURES ${ROCM_ARCH})
  message(STATUS "ROCM_ARCH: ${ROCM_ARCH}")
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
else()
  set(CMAKE_CXX_STANDARD 17)
  project(transformer_engine LANGUAGES HIP CXX)
  # build error will be dup-ed parallel-jobs times
  # list(APPEND CMAKE_HIP_FLAGS "-parallel-jobs=4")
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    list(APPEND CMAKE_HIP_FLAGS "-g")
  endif()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_HIP_FLAGS}")

  set(HIP_HCC_FLAGS "${CMAKE_HIP_FLAGS} -mavx2 -mf16c -mfma -std=c++17")
  # Ask hcc to generate device code during compilation so we can use
  # host linker to link.
  set(HIP_HCC_FLAGS "${HIP_HCC_FLAGS} -O3 -ffast-math -fno-gpu-rdc -Wno-defaulted-function-deleted")
  foreach(rocm_arch ${TE_ROCM_ARCH})
    # if CMAKE_CXX_FLAGS has --offload-arch set already, better to rm first
    set(HIP_HCC_FLAGS "${HIP_HCC_FLAGS} --offload-arch=${rocm_arch}")
  endforeach()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${HIP_HCC_FLAGS}")
  if(DEFINED ENV{ROCM_PATH} AND NOT "$ENV{ROCM_PATH}" STREQUAL "")
    list(APPEND CMAKE_MODULE_PATH "$ENV{ROCM_PATH}")
  else()
    list(APPEND CMAKE_MODULE_PATH "/opt/rocm")
  endif()
endif()

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
if(USE_CUDA)
  find_package(CUDAToolkit REQUIRED cublas nvToolsExt)
  find_package(CUDNN REQUIRED cudnn)
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
