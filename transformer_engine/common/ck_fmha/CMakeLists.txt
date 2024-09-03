# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.18)
set(CMAKE_CXX_STANDARD 17)
project(ck LANGUAGES HIP CXX)

file(MAKE_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/gen_src)
set(CK_GEN_DIR "${CMAKE_CURRENT_LIST_DIR}/gen_src")
set(CK_SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../3rdparty/composable_kernel")

execute_process(
  COMMAND python3 ${CK_SRC_DIR}/example/ck_tile/01_fmha/generate.py
  --api fwd --output_dir ${CK_GEN_DIR} --receipt 2
)
execute_process(
  COMMAND python3 ${CK_SRC_DIR}/example/ck_tile/01_fmha/generate.py
  --api bwd --output_dir ${CK_GEN_DIR} --receipt 2
)

# Copy the header files needed
# Glob all .hpp files in the source directory
file(GLOB HPP_FILES "${CK_SRC_DIR}/example/ck_tile/01_fmha/*.hpp")

# Copy each .hpp file to the destination directory
foreach(HPP_FILE ${HPP_FILES})
  get_filename_component(FILE_NAME ${HPP_FILE} NAME)
  file(COPY ${HPP_FILE} DESTINATION ${CMAKE_CURRENT_LIST_DIR}/src)
endforeach()

if(DEFINED ENV{PYTORCH_ROCM_ARCH})
  set(ROCM_ARCH $ENV{PYTORCH_ROCM_ARCH})
else()
  set(ROCM_ARCH gfx90a;gfx940;gfx941;gfx942)
endif()

set(CMAKE_HIP_ARCHITECTURES ${ROCM_ARCH})
message(STATUS "ROCM_ARCH: ${ROCM_ARCH}")

set(CK_FMHA_SOURCES)
list(APPEND CK_FMHA_SOURCES
       src/ck_fmha_fwd.cpp
       src/ck_fmha_bwd.cpp)

file(GLOB_RECURSE CK_FMHA_FILES "${CK_GEN_DIR}/*.cpp")
list(APPEND CK_FMHA_SOURCES ${CK_FMHA_FILES})

# Check if CK_PARALLEL_COMPILE_JOBS is not set
if(NOT DEFINED CK_PARALLEL_COMPILE_JOBS)
    # Set a default value
    set(CK_PARALLEL_COMPILE_JOBS 256)
endif()

if(CK_PARALLEL_COMPILE_JOBS)
    set_property(GLOBAL APPEND PROPERTY JOB_POOLS compile_job_pool=${CK_PARALLEL_COMPILE_JOBS})
    set(CMAKE_JOB_POOL_COMPILE compile_job_pool)
endif()

add_library(ck STATIC ${CK_FMHA_SOURCES})

add_library(ck_fmha_options INTERFACE)
target_compile_definitions(ck_fmha_options INTERFACE
  CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=3
  CK_TILE_FMHA_FWD_FAST_EXP2=1
  CK_USE_XDL
)

target_compile_definitions(ck_fmha_options INTERFACE
  CK_ENABLE_BF8
  CK_ENABLE_FP8
  CK_ENABLE_INT8
  CK_ENABLE_BF16
  CK_ENABLE_FP16
  CK_ENABLE_FP32
  CK_ENABLE_FP64
)

# Compiler-specific options
target_compile_options(ck_fmha_options INTERFACE
  $<$<CXX_COMPILER_ID:Clang>:
    -fcolor-diagnostics
    -fgpu-flush-denormals-to-zero
    -ftemplate-backtrace-limit=0
    -fPIC
    -Wno-gnu-line-marker
    -fno-offload-uniform-block
  >
)

set(CK_INCLUDE_DIR "${CK_SRC_DIR}/include")
message(STATUS "ck_include_dir: ${CK_INCLUDE_DIR}")

if(NOT EXISTS "${CK_INCLUDE_DIR}")
  message(FATAL_ERROR
          "Could not find CK API. "
          "Try running 'git submodule update --init' "
          "within the Transformer Engine source.")
endif()

target_include_directories(ck PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)
target_include_directories(ck PRIVATE ${CK_INCLUDE_DIR} ${CMAKE_CURRENT_LIST_DIR}/src)

list(APPEND CK_FMHA_LINKER_LIBS hip::host hip::device Threads::Threads roctx64)
target_link_libraries(ck PUBLIC ${CK_FMHA_LINKER_LIBS})
target_link_libraries(ck PRIVATE ck_fmha_options)

include(GNUInstallDirs)
message(STATUS "CMAKE_INSTALL_INCLUDEDIR: ${CMAKE_INSTALL_INCLUDEDIR}")
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include" DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS ck DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
