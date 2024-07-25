# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.18)
set(CMAKE_CXX_STANDARD 17)
project(ck_fused_attn LANGUAGES HIP CXX)
# generate ck fused attn kernels, both fwd/bwd

set(ck_fused_attn_SOURCES)
list(APPEND ck_fused_attn_SOURCES
       src/ck_fused_attn.cpp)
#fwd list
execute_process(
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/generate.py
  --direction fwd --list_blobs ${CMAKE_CURRENT_SOURCE_DIR}/gen_src/fwd_blob_list.txt -r 1
)
#fwd kernels
execute_process(
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/generate.py
  --direction fwd --output_dir ${CMAKE_CURRENT_SOURCE_DIR}/gen_src -r 1
)

#bwd list
execute_process(
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/generate.py
  --direction bwd --list_blobs ${CMAKE_CURRENT_SOURCE_DIR}/gen_src/bwd_blob_list.txt -r 1
)
#bwd kernels
execute_process(
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/generate.py
  --direction bwd --output_dir ${CMAKE_CURRENT_SOURCE_DIR}/gen_src -r 1
)


file(GLOB_RECURSE CK_FA_FILES "${CMAKE_CURRENT_SOURCE_DIR}/gen_src/*.cpp")
list(APPEND ck_fused_attn_SOURCES ${CK_FA_FILES})
message(STATUS "Found the following CK fused attention files:")
foreach(file ${CK_FA_FILES})
  message(STATUS " ${file}")
endforeach()

add_library(ck_fused_attn STATIC ${ck_fused_attn_SOURCES})
set(CK_FUSED_ATTN_COMPILE_OPTIONS)
#list(APPEND CK_FUSED_ATTN_COMPILE_OPTIONS -Wno-undefined-func-template -DCK_TILE_FMHA_FWD_FAST_EXP2=1 -fgpu-flush-denormals-to-zero -Wno-float-equal)
list(APPEND CK_FUSED_ATTN_COMPILE_OPTIONS -Wno-undefined-func-template -DCK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=0 -fgpu-flush-denormals-to-zero -Wno-float-equal)
set(CMAKE_CXX_FLAGS "-std=c++17 -O3 -ftemplate-backtrace-limit=0  -fPIC  -Wno-gnu-line-marker -Wunused-variable -Werror")

set(CK_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/composable_kernel/include")

if(NOT EXISTS "${CK_INCLUDE_DIR}")
  message(FATAL_ERROR
          "Could not find CK API. "
          "Try running 'git submodule update --init --recursive' "
          "within the Transformer Engine source.")
endif()

target_include_directories(ck_fused_attn PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(ck_fused_attn PRIVATE ${CK_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src)

find_package(hip)
list(APPEND ck_fused_attn_LINKER_LIBS hip::host hip::device roctx64)
target_link_libraries(ck_fused_attn PUBLIC ${ck_fused_attn_LINKER_LIBS})
target_compile_options(ck_fused_attn PRIVATE ${CK_FUSED_ATTN_COMPILE_OPTIONS})

include(GNUInstallDirs)
message("CMAKE_INSTALL_INCLUDEDIR ${CMAKE_INSTALL_INCLUDEDIR}")
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/ck_fused_attn" DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS ck_fused_attn DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
