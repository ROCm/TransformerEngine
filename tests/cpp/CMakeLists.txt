# This file was modified for portability to AMDGPU
# Copyright (c) 2022-2024, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

cmake_minimum_required(VERSION 3.18)

option(USE_CUDA "Use CUDA" ON)
option(USE_ROCM "Use ROCm" OFF)
option(USE_HIPBLASLT "Use HIPBLASLT" OFF)

if(((EXISTS "/opt/rocm/") OR (EXISTS $ENV{ROCM_PATH})) AND NOT (EXISTS "/bin/nvcc"))
  message("AMD GPU detected.")
  set(USE_ROCM ON)
  set(USE_CUDA OFF)

  # Add HIP to the CMAKE Module Path
  # set(CMAKE_MODULE_PATH ${HIP_PATH}/cmake ${CMAKE_MODULE_PATH})
  # Disable Asserts In Code (Can't use asserts on HIP stack.)
  add_definitions(-DNDEBUG)
  add_definitions(-DUSE_ROCM)
  if(NOT DEFINED ENV{PYTORCH_ROCM_ARCH})
    SET(TE_ROCM_ARCH gfx90a;gfx940;gfx941;gfx942)
  else()
    SET(TE_ROCM_ARCH $ENV{PYTORCH_ROCM_ARCH})
  endif()
  SET(CMAKE_HIP_ARCHITECTURES ${TE_ROCM_ARCH})
else()
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 80 90)
  endif()
endif()

set(message_line
	"-------------------------------------------------------------")
message("${message_line}")
message(STATUS "USE_CUDA ${USE_CUDA}")
message(STATUS "USE_ROCM ${USE_ROCM}")
message(STATUS "USE_HIPBLASLT ${USE_HIPBLASLT}")


if(USE_CUDA)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  project(transformer_engine_tests LANGUAGES CUDA CXX)
else()
  set(CMAKE_CXX_STANDARD 17)
  project(transformer_engine_tests LANGUAGES HIP CXX)
  # Ask hcc to generate device code during compilation so we can use
  # host linker to link.
  set(HIP_HCC_FLAGS "${HIP_HCC_FLAGS} -fno-gpu-rdc -Wno-defaulted-function-deleted")
  foreach(rocm_arch ${TE_ROCM_ARCH})
    # if CMAKE_CXX_FLAGS has --offload-arch set already, better to rm first
    set(HIP_HCC_FLAGS "${HIP_HCC_FLAGS} --offload-arch=${rocm_arch}")
  endforeach()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${HIP_HCC_FLAGS}")
endif()

add_subdirectory(../../3rdparty/googletest ${PROJECT_BINARY_DIR}/googletest)

enable_testing()

include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

if(NOT DEFINED TE_LIB_PATH)
    execute_process(COMMAND bash -c "pip show transformer-engine | grep Location | cut -d ' ' -f 2 | tr -d '\n'"
                    OUTPUT_VARIABLE TE_LIB_PATH)
endif()

find_library(TE_LIB NAMES transformer_engine PATHS ${TE_LIB_PATH} ENV TE_LIB_PATH REQUIRED)

message(STATUS "Found transformer_engine library: ${TE_LIB}")
include_directories(../../transformer_engine/common/include)
include_directories(../../transformer_engine/common)
include_directories(${CMAKE_SOURCE_DIR})

if(USE_CUDA)
  find_package(CUDAToolkit REQUIRED)
else()
  find_package(hip REQUIRED)
  if(USE_HIPBLASLT)
    find_package(hipblaslt)
    add_definitions(-DUSE_HIPBLASLT)
  else()
    find_package(rocblas)
  endif()
endif()

add_subdirectory(operator)
add_subdirectory(util)
