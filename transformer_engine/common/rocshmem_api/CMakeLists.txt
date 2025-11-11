# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information
cmake_minimum_required (VERSION 3.21)
project(rocshmem LANGUAGES HIP)

find_package(hipblaslt REQUIRED)
find_package(hiprtc REQUIRED)
find_package(hip REQUIRED)
find_package(MPI REQUIRED)

if(NOT DEFINED ENV{NVTE_ROCM_ARCH})
  set(CMAKE_HIP_ARCHITECTURES gfx942 gfx950)
else()
  set(CMAKE_HIP_ARCHITECTURES $ENV{NVTE_ROCM_ARCH})
endif()

if(DEFINED ENV{ROCSHMEM_HOME})
  set(ROCSHMEM_HOME "$ENV{ROCSHMEM_HOME}" CACHE STRING "Location of ROCSHMEM installation")
else()
  set(ROCSHMEM_HOME "/opt/rocm" CACHE STRING "Location of ROCSHMEM installation (default)")
endif()

set(ROCSHMEM_INCLUDE_DIR "${ROCSHMEM_HOME}/include/rocshmem")
if(NOT EXISTS "${ROCSHMEM_INCLUDE_DIR}")
  set(ROCSHMEM_INCLUDE_DIR "${ROCSHMEM_HOME}/include")
endif()

add_library(rocshmemapi OBJECT rocshmem_waitkernel.hip)

target_compile_options(rocshmemapi PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-fgpu-rdc>
)

target_include_directories(rocshmemapi PUBLIC
        "${ROCSHMEM_INCLUDE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${MPI_INCLUDE_PATH}"
)

target_link_libraries(rocshmemapi PUBLIC
    "${ROCSHMEM_HOME}/lib/librocshmem.a"
    MPI::MPI_CXX
    hip::host
    hip::device
    roctx64
    hiprtc
    roc::hipblaslt
)

set_target_properties(rocshmemapi PROPERTIES
    CXX_STANDARD 17
    HIP_STANDARD 17
    POSITION_INDEPENDENT_CODE ON
    HIP_SEPARABLE_COMPILATION ON
)

set(ROCSHMEMAPI_INCLUDE_DIR "${ROCSHMEM_INCLUDE_DIR}" PARENT_SCOPE)
