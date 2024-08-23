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

# copy the header files needed
file(COPY_FILE ${CK_SRC_DIR}/example/ck_tile/01_fmha/bias.hpp ${CMAKE_CURRENT_LIST_DIR}/src/bias.hpp)
file(COPY_FILE ${CK_SRC_DIR}/example/ck_tile/01_fmha/mask.hpp ${CMAKE_CURRENT_LIST_DIR}/src/mask.hpp)
file(COPY_FILE ${CK_SRC_DIR}/example/ck_tile/01_fmha/fmha_fwd.hpp ${CMAKE_CURRENT_LIST_DIR}/src/fmha_fwd.hpp)
file(COPY_FILE ${CK_SRC_DIR}/example/ck_tile/01_fmha/fmha_bwd.hpp ${CMAKE_CURRENT_LIST_DIR}/src/fmha_bwd.hpp)

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
set(CK_FMHA_COMPILE_OPTIONS)
list(APPEND CK_FMHA_COMPILE_OPTIONS
  -DCK_TILE_FMHA_FWD_FAST_EXP2=1
  -DCK_ENABLE_BF8
  -DCK_ENABLE_FP8
  -DCK_ENABLE_INT8
  -DCK_ENABLE_BF16
  -DCK_ENABLE_FP16
  -DCK_ENABLE_FP32
  -DCK_ENABLE_FP64
  -DCK_USE_XDL
  -mllvm
  -enable-post-misched=0
  -fPIC
  -fgpu-flush-denormals-to-zero
  -fno-offload-uniform-block
  -Wno-undefined-func-template
  -Wno-float-equal
  -Wunused-variable)

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
target_compile_options(ck PRIVATE ${CK_FMHA_COMPILE_OPTIONS})

include(GNUInstallDirs)
message(STATUS "CMAKE_INSTALL_INCLUDEDIR: ${CMAKE_INSTALL_INCLUDEDIR}")
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include" DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS ck DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
