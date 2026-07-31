# This file was modified for portability to AMDGPU
# Copyright (c) 2022-2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

cmake_minimum_required(VERSION 3.18)

option(USE_ROCM "Use ROCm" ON)
set(USE_CUDA OFF)

if (USE_ROCM)
  include("${CMAKE_CURRENT_SOURCE_DIR}/../../build_tools/rocm_utils.cmake")
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
  find_package(hiprand REQUIRED)
  find_package(hiprtc REQUIRED)
  find_package(hipblaslt REQUIRED)
  # ROCm libs used by libtransformer_engine.so must be direct link deps of test
  # binaries. DT_RUNPATH on the executable does not apply to transitive NEEDED
  # libs, so gtest_discover_tests fails loading e.g. libroctx64.so.4 otherwise.
  set(TE_ROCM_TEST_LIBS hip::host hip::device roctx64 hiprtc roc::hipblaslt hip::hiprand)
  list(APPEND CMAKE_BUILD_RPATH "${ROCM_PATH}/lib")
  # Emit RPATH (not RUNPATH) so ${ROCM_PATH}/lib also covers any remaining
  # transitive ROCm NEEDED entries from libtransformer_engine.so.
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--disable-new-dtags")
endif()

if(USE_ROCM)
  include("${CMAKE_CURRENT_SOURCE_DIR}/../../build_tools/hipify/hipify.cmake")
  TE_Hipify(${CMAKE_CURRENT_SOURCE_DIR})

  # Register original source files as configure dependencies so that
  # editing them triggers re-configuration and re-hipification.
  # SOURCES covers explicitly listed .cu/.cpp files; headers (.h, .cuh,
  # .hpp) are globbed from BASE_DIR since they are not enumerated in SOURCES.
  function(TE_AddHipifyDeps SOURCES BASE_DIR)
      set(_deps)
      foreach(_src ${SOURCES})
          get_filename_component(_abs "${BASE_DIR}/${_src}" ABSOLUTE)
          list(APPEND _deps "${_abs}")
      endforeach()
      file(GLOB _hdrs
          "${BASE_DIR}/*.h"
          "${BASE_DIR}/*.cuh"
          "${BASE_DIR}/*.hpp"
      )
      list(APPEND _deps ${_hdrs})
      set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_deps})
  endfunction()
endif()

add_subdirectory(operator)
add_subdirectory(util)
