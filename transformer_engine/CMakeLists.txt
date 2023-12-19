# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#                    2023 Advanced Micro Devices, Inc. All rights reserved.
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
  message(STATUS "TE_ROCM_ARCH: ${TE_ROCM_ARCH}")
 
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
  list(APPEND CMAKE_CUDA_FLAGS "--threads 4")
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
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

  add_definitions(-D__HIP_PLATFORM_HCC__=1)
  # add_definitions(-D__HIP_NO_HALF_OPERATORS__=1)
  # add_definitions(-D__HIP_NO_HALF_CONVERSIONS__=1)
  # add_definitions(-D__HIP_NO_BFLOAT16_CONVERSIONS__=1)
  # add_definitions(-D__HIP_NO_HALF2_OPERATORS__=1)

  set(HIP_HCC_FLAGS "${CMAKE_HIP_FLAGS} -mavx2 -mf16c -mfma -std=c++17")
  # Ask hcc to generate device code during compilation so we can use
  # host linker to link.
  set(HIP_HCC_FLAGS "${HIP_HCC_FLAGS} -fno-gpu-rdc -Wno-defaulted-function-deleted")
  foreach(rocm_arch ${TE_ROCM_ARCH})
    # if CMAKE_CXX_FLAGS has --offload-arch set already, better to rm first
    set(HIP_HCC_FLAGS "${HIP_HCC_FLAGS} --offload-arch=${rocm_arch}")
  endforeach()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${HIP_HCC_FLAGS}")
  list(APPEND CMAKE_MODULE_PATH "/opt/rocm")
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/")
if(USE_CUDA)
  find_package(CUDAToolkit REQUIRED cublas nvToolsExt)
  find_package(CUDNN REQUIRED cudnn)
endif()
find_package(Python COMPONENTS Interpreter Development REQUIRED)

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

option(ENABLE_TENSORFLOW "Enable TensorFlow in the building workflow." OFF)
message(STATUS "TensorFlow support: ${ENABLE_TENSORFLOW}")
if(ENABLE_TENSORFLOW)
  find_package(pybind11 CONFIG REQUIRED)
  add_subdirectory(tensorflow)
endif()
