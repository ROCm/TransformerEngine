# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

cmake_minimum_required (VERSION 3.21)
project(rocshmem LANGUAGES HIP)

find_package(hip REQUIRED)
find_package(MPI REQUIRED)

add_library(rocshmemapi OBJECT rocshmem_waitkernel.hip)

target_compile_options(rocshmemapi PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-fgpu-rdc>
)

set_target_properties(rocshmemapi PROPERTIES
    CXX_STANDARD 17
    HIP_STANDARD 17
    POSITION_INDEPENDENT_CODE ON
)

if(DEFINED ENV{ROCSHMEM_HOME})
  set(ROCSHMEM_HOME "$ENV{ROCSHMEM_HOME}" CACHE STRING "Location of ROCSHMEM installation")
else()
  set(ROCSHMEM_HOME "/opt/rocm" CACHE STRING "Location of ROCSHMEM installation (default)")
endif()
set(ROCSHMEM_LIBRARY_PATH "${ROCSHMEM_HOME}/lib/librocshmem.a")
if (EXISTS ${ROCSHMEM_LIBRARY_PATH})
  add_library(rocshmem STATIC IMPORTED)
  set_target_properties(rocshmem PROPERTIES
    IMPORTED_LOCATION "${ROCSHMEM_LIBRARY_PATH}"
    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
  )
else()
  message(FATAL_ERROR "ROCSHMEM library not found at ${ROCSHMEM_LIBRARY_PATH}. PLease set ROCSHMEM_HOME.")
endif()

set(ROCSHMEM_INCLUDE_DIR "${ROCSHMEM_HOME}/include/rocshmem")
if(NOT EXISTS "${ROCSHMEM_INCLUDE_DIR}")
  set(ROCSHMEM_INCLUDE_DIR "${ROCSHMEM_HOME}/include")
endif()

target_include_directories(rocshmemapi PUBLIC
        "${ROCSHMEM_INCLUDE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${MPI_INCLUDE_PATH}"
)

target_link_libraries(rocshmemapi PUBLIC
    rocshmem
    MPI::MPI_CXX
    hip::host
    hip::device
)
