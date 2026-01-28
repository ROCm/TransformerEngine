# This file was modified for portability to AMDGPU
# Copyright (c) 2022-2025, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

cmake_minimum_required(VERSION 3.18)

option(USE_ROCM "Use ROCm" ON)
set(USE_CUDA OFF)

if (USE_ROCM)
  #Check if ROCm exists
  if( ((EXISTS "/opt/rocm/") OR (EXISTS $ENV{ROCM_PATH})) )
    message("Using AMD Platform")
  else()
    message(FATAL_ERROR "Could not ROCm installation")
  endif()
else()
  #Check if CUDA exists
  if (EXISTS "$ENV{CUDA_HOME}/bin/nvcc")
    set(USE_ROCM OFF)
    set(USE_CUDA ON)
    message("Using NVIDIA Platform")
  else()
    message(FATAL_ERROR "Could not find NVCC at '$ENV{CUDA_HOME}/bin/nvcc'")
  endif()
endif()

if(USE_ROCM)
  # Add HIP to the CMAKE Module Path
  # set(CMAKE_MODULE_PATH ${HIP_PATH}/cmake ${CMAKE_MODULE_PATH})
  # Disable Asserts In Code (Can't use asserts on HIP stack.)
  add_definitions(-DNDEBUG)
  add_definitions(-DUSE_ROCM)
  if(NOT DEFINED ENV{NVTE_ROCM_ARCH})
    SET(CMAKE_HIP_ARCHITECTURES gfx942 gfx950)
  else()
    SET(CMAKE_HIP_ARCHITECTURES $ENV{NVTE_ROCM_ARCH})
  endif()
else()
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL 12.8)
      set(CMAKE_CUDA_ARCHITECTURES 75 80 89 90 100 120)
    else ()
      set(CMAKE_CUDA_ARCHITECTURES 75 80 89 90)
    endif()
  endif()
endif()

set(message_line
	"-------------------------------------------------------------")
message("${message_line}")
message(STATUS "USE_CUDA ${USE_CUDA}")
message(STATUS "USE_ROCM ${USE_ROCM}")

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
  set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -fno-gpu-rdc -Wno-defaulted-function-deleted -Wno-unused-result -Wno-unused-value -ftemplate-backtrace-limit=0")
endif()

add_subdirectory(../../3rdparty/googletest ${PROJECT_BINARY_DIR}/googletest)

enable_testing()

include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

if(NOT DEFINED TE_LIB_PATH)
    execute_process(COMMAND bash -c "python3 -c 'import transformer_engine as te; print(te.__file__)'"
                    OUTPUT_VARIABLE TE_LIB_FILE
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    get_filename_component(TE_LIB_PATH ${TE_LIB_FILE} DIRECTORY)
endif()

find_library(TE_LIB NAMES transformer_engine PATHS "${TE_LIB_PATH}/.." ${TE_LIB_PATH} ENV TE_LIB_PATH REQUIRED)

message(STATUS "Found transformer_engine library: ${TE_LIB}")
include_directories(../../transformer_engine/common/include)
include_directories(../../transformer_engine/common)
include_directories(../../transformer_engine)
include_directories(${CMAKE_SOURCE_DIR})

if(USE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  include(${CMAKE_SOURCE_DIR}/../../3rdparty/cudnn-frontend/cmake/cuDNN.cmake)
else()
  set(GPU_TARGETS ${CMAKE_HIP_ARCHITECTURES})
  find_package(hip REQUIRED)
endif()

add_subdirectory(operator)
add_subdirectory(util)
