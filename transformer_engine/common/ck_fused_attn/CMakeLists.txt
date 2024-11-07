# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.18)
set(CMAKE_CXX_STANDARD 17)
project(ck_fused_attn LANGUAGES HIP CXX)
# generate ck fused attn kernels, both fwd/bwd

set(CK_FUSED_ATTN_TARGET_GPUS "gfx942" CACHE STRING "Target Architecture to build ck_fused_attn backend")

# remove all previously generated kernel files in gen_src
file(REMOVE_RECURSE ${CMAKE_CURRENT_SOURCE_DIR}/gen_src)
# create the empty gen_src again
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/gen_src)

set(__CK_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../3rdparty/composable_kernel")

#fwd kernels list
execute_process(
  COMMAND python3 ${__CK_SOURCE_DIR}/example/ck_tile/01_fmha/generate.py
  --api fwd --list_blobs ${CMAKE_CURRENT_SOURCE_DIR}/gen_src/fwd_blob_list.txt
)
#bwd kernels list
execute_process(
  COMMAND python3 ${__CK_SOURCE_DIR}/example/ck_tile/01_fmha/generate.py
  --api bwd --list_blobs ${CMAKE_CURRENT_SOURCE_DIR}/gen_src/bwd_blob_list.txt
)

# NOTE: for cmake, the FMHA_FWD_GEN_BLOBS/FMHA_BWD_GEN_BLOBS files must be in the same directory
#       as current cmake list, otherwise will not figure out the dependency properly
file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/gen_src/fwd_blob_list.txt FMHA_FWD_GEN_BLOBS)
file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/gen_src/bwd_blob_list.txt FMHA_BWD_GEN_BLOBS)

# generate the actual fwd kernel cpp files
execute_process(
  COMMAND python3 ${__CK_SOURCE_DIR}/example/ck_tile/01_fmha/generate.py
  --api fwd --output_dir ${CMAKE_CURRENT_SOURCE_DIR}/gen_src
)

# generate the actual bwd kernel cpp files
execute_process(
  COMMAND python3 ${__CK_SOURCE_DIR}/example/ck_tile/01_fmha/generate.py
  --api bwd --output_dir ${CMAKE_CURRENT_SOURCE_DIR}/gen_src --receipt 3
)

# copy the header files needed
file(COPY_FILE ${__CK_SOURCE_DIR}/example/ck_tile/01_fmha/bias.hpp ${CMAKE_CURRENT_SOURCE_DIR}/src/bias.hpp)
file(COPY_FILE ${__CK_SOURCE_DIR}/example/ck_tile/01_fmha/mask.hpp ${CMAKE_CURRENT_SOURCE_DIR}/src/mask.hpp)
file(COPY_FILE ${__CK_SOURCE_DIR}/example/ck_tile/01_fmha/fmha_fwd.hpp ${CMAKE_CURRENT_SOURCE_DIR}/src/fmha_fwd.hpp)
file(COPY_FILE ${__CK_SOURCE_DIR}/example/ck_tile/01_fmha/fmha_bwd.hpp ${CMAKE_CURRENT_SOURCE_DIR}/src/fmha_bwd.hpp)

set(ck_fused_attn_SOURCES)
list(APPEND ck_fused_attn_SOURCES
       src/ck_fused_attn_fwd.cpp
       src/ck_fused_attn_bwd.cpp
       src/ck_fused_attn_utils.cpp)

file(GLOB_RECURSE CK_FA_FILES "${CMAKE_CURRENT_SOURCE_DIR}/gen_src/*.cpp")
list(APPEND ck_fused_attn_SOURCES ${CK_FA_FILES})
message(STATUS "Found the following CK fused attention files:")
foreach(file ${CK_FA_FILES})
  message(STATUS " ${file}")
endforeach()

add_library(ck_fused_attn STATIC ${ck_fused_attn_SOURCES})
set(CK_FUSED_ATTN_COMPILE_OPTIONS)
list(APPEND CK_FUSED_ATTN_COMPILE_OPTIONS -Wno-undefined-func-template -DCK_TILE_FMHA_FWD_FAST_EXP2=1 -DCK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=0 -fgpu-flush-denormals-to-zero -Wno-float-equal -ftemplate-backtrace-limit=0  -fPIC  -Wno-gnu-line-marker -Wunused-variable -Werror)
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
