# Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# This file is for temporary use
cmake_minimum_required(VERSION 3.21)
set(CMAKE_CXX_STANDARD 17)
project(ck_fused_attn LANGUAGES HIP CXX)
# generate fused attn kernels, both fwd/bwd

set(CK_FUSED_ATTN_TARGET_GPUS "gfx950" CACHE STRING "Target Architecture to build ck_fused_attn backend")

if (CK_FUSED_ATTN_INSTALL_PREFIX)
  set(_CK_FUSED_ATTN_INSTALL_PREFIX ${CK_FUSED_ATTN_INSTALL_PREFIX}/ )
else()
  set(_CK_FUSED_ATTN_INSTALL_PREFIX . )
endif()

set(__AITER_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../3rdparty/aiter")
set(__AITER_TEST_DIR "${__AITER_SOURCE_DIR}/op_tests/cpp/mha")
set(__CK_SOURCE_DIR "${__AITER_SOURCE_DIR}/3rdparty/composable_kernel")

#compile the libmha_fwd.so and libmha_bwd.so
set(ENV{AITER_LOG_MORE} 1)
#execute_process(
#  COMMAND python3 ${__AITER_TEST_DIR}/compile.py
#)

set(ck_fused_attn_SOURCES)
list(APPEND ck_fused_attn_SOURCES
       src/ck_fused_attn_fwd.cpp
       src/ck_fused_attn_bwd.cpp
       src/ck_fused_attn_utils.cpp)

message(STATUS "Found the following fused attention files:")
foreach(file ${ck_fused_attn_SOURCES})
  message(STATUS " ${file}")
endforeach()

add_library(ck_fused_attn SHARED ${ck_fused_attn_SOURCES})
set(CK_FUSED_ATTN_COMPILE_OPTIONS)
list(APPEND CK_FUSED_ATTN_COMPILE_OPTIONS -std=c++17 -O3 -DUSE_ROCM=1 -DCK_TILE_FMHA_FWD_SPLITKV_API=1 --offload-arch=native)
foreach(rocm_arch ${CK_FUSED_ATTN_TARGET_GPUS})
  list(APPEND CK_FUSED_ATTN_COMPILE_OPTIONS --offload-arch=${rocm_arch})
endforeach()

set(CK_INCLUDE_DIR "${__CK_SOURCE_DIR}/include")
message(STATUS "ck_include_dir: ${CK_INCLUDE_DIR}")

if(NOT EXISTS "${CK_INCLUDE_DIR}")
  message(FATAL_ERROR
          "Could not find CK API. "
          "Try running 'git submodule update --init --recursive' "
          "within the Transformer Engine source.")
endif()

set(AITER_INCLUDE_DIR "${__AITER_SOURCE_DIR}/csrc/include")
message(STATUS "aiter_include_dir: ${AITER_INCLUDE_DIR}")

if(NOT EXISTS "${AITER_INCLUDE_DIR}")
  message(FATAL_ERROR
          "Could not find AITER API. "
          "Try running 'git submodule update --init --recursive' "
          "within the Transformer Engine source.")
endif()

target_include_directories(ck_fused_attn PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(ck_fused_attn PRIVATE ${CK_INCLUDE_DIR} ${__CK_SOURCE_DIR}/example/ck_tile/01_fmha)
target_include_directories(ck_fused_attn PRIVATE ${AITER_INCLUDE_DIR})

find_package(hip)
list(APPEND ck_fused_attn_LINKER_LIBS hip::host hip::device roctx64 ${__AITER_TEST_DIR}/libmha_fwd.so ${__AITER_TEST_DIR}/libmha_bwd.so)
target_link_libraries(ck_fused_attn PUBLIC ${ck_fused_attn_LINKER_LIBS})
target_compile_options(ck_fused_attn PRIVATE ${CK_FUSED_ATTN_COMPILE_OPTIONS})
set_target_properties(ck_fused_attn PROPERTIES INSTALL_RPATH "$ORIGIN")

include(GNUInstallDirs)
message("CMAKE_INSTALL_INCLUDEDIR ${CMAKE_INSTALL_INCLUDEDIR}")
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/ck_fused_attn" DESTINATION ${CMAKE_INSTALL_PREFIX}/${_CK_FUSED_ATTN_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR})
install(FILES ${__AITER_TEST_DIR}/libmha_fwd.so ${__AITER_TEST_DIR}/libmha_bwd.so DESTINATION ${CMAKE_INSTALL_PREFIX}/${_CK_FUSED_ATTN_INSTALL_PREFIX}/lib)
install(TARGETS ck_fused_attn DESTINATION ${CMAKE_INSTALL_PREFIX}/${_CK_FUSED_ATTN_INSTALL_PREFIX}/lib)
