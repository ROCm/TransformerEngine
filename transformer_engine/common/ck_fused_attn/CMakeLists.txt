# Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# This file is for temporary use
cmake_minimum_required(VERSION 3.21)
set(CMAKE_CXX_STANDARD 17)
project(ck_fused_attn LANGUAGES HIP CXX)
# generate fused attn kernels, both fwd/bwd

set(CK_FUSED_ATTN_TARGET_GPUS "gfx942,gfx950" CACHE STRING "Target Architecture to build ck_fused_attn backend")

# remove files that should be regenerated
file(REMOVE_RECURSE ${CMAKE_CURRENT_BINARY_DIR}/gen_src/tmp ${CMAKE_CURRENT_BINARY_DIR}/gen_src/blob_list.txt)

# create gen_src and gen_src/tmp directories if needed
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/gen_src/tmp)

set(__AITER_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../3rdparty/aiter")
set(__CK_SOURCE_DIR "${__AITER_SOURCE_DIR}/3rdparty/composable_kernel")

#fwd kernels list
execute_process(
  COMMAND python3 ${__CK_SOURCE_DIR}/example/ck_tile/01_fmha/generate.py
  --api fwd --list_blobs ${CMAKE_CURRENT_BINARY_DIR}/gen_src/fwd_blob_list.txt --receipt 600
)
execute_process(
  COMMAND python3 ${__CK_SOURCE_DIR}/example/ck_tile/01_fmha/generate.py
  --api fwd_splitkv --list_blobs ${CMAKE_CURRENT_BINARY_DIR}/gen_src/fwd_splitkv_blob_list.txt --receipt 600
)
#bwd kernels list
execute_process(
  COMMAND python3 ${__CK_SOURCE_DIR}/example/ck_tile/01_fmha/generate.py
  --api bwd --list_blobs ${CMAKE_CURRENT_BINARY_DIR}/gen_src/bwd_blob_list.txt --receipt 600
)

file(STRINGS ${CMAKE_CURRENT_BINARY_DIR}/gen_src/fwd_blob_list.txt FMHA_FWD_GEN_BLOBS)
file(STRINGS ${CMAKE_CURRENT_BINARY_DIR}/gen_src/fwd_splitkv_blob_list.txt FMHA_FWD_SPLITKV_GEN_BLOBS)
file(STRINGS ${CMAKE_CURRENT_BINARY_DIR}/gen_src/bwd_blob_list.txt FMHA_BWD_GEN_BLOBS)

# generate the actual fwd kernel cpp files
execute_process(
  COMMAND python3 ${__CK_SOURCE_DIR}/example/ck_tile/01_fmha/generate.py
  --api fwd --output_dir ${CMAKE_CURRENT_BINARY_DIR}/gen_src/tmp --receipt 600
)

execute_process(
  COMMAND python3 ${__CK_SOURCE_DIR}/example/ck_tile/01_fmha/generate.py
  --api fwd_splitkv --output_dir ${CMAKE_CURRENT_BINARY_DIR}/gen_src/tmp --receipt 600
)

execute_process(
  COMMAND python3 ${__AITER_SOURCE_DIR}/csrc/cpp_itfs/mha_fwd_generate.py
  --receipt 3 --output_dir ${CMAKE_CURRENT_BINARY_DIR}/gen_src/tmp --receipt 3
)

# generate the actual bwd kernel cpp files
execute_process(
  COMMAND python3 ${__CK_SOURCE_DIR}/example/ck_tile/01_fmha/generate.py
  --api bwd --output_dir ${CMAKE_CURRENT_BINARY_DIR}/gen_src/tmp --receipt 600
)

execute_process(
  COMMAND python3 ${__AITER_SOURCE_DIR}/csrc/py_itfs_cu/fmha_bwd_pre_post_kernel.py
  --receipt 1 --filter *@*_ndeterministic@*_nbias*_dropout*_ndeterministic* --output_dir ${CMAKE_CURRENT_BINARY_DIR}/gen_src/tmp --receipt 3
)

execute_process(
  COMMAND python3 ${__AITER_SOURCE_DIR}/csrc/cpp_itfs/mha_bwd_generate.py
  --receipt 3 --output_dir ${CMAKE_CURRENT_BINARY_DIR}/gen_src/tmp --receipt 3
)

set(ck_fused_attn_SOURCES)
list(APPEND ck_fused_attn_SOURCES
       src/ck_fused_attn_fwd.cpp
       src/ck_fused_attn_bwd.cpp
       src/ck_fused_attn_utils.cpp)

foreach(blob ${FMHA_FWD_GEN_BLOBS})
  file(RELATIVE_PATH blob_path ${CMAKE_CURRENT_BINARY_DIR}/gen_src ${blob})
  file(COPY_FILE ${CMAKE_CURRENT_BINARY_DIR}/gen_src/tmp/${blob_path} ${blob} ONLY_IF_DIFFERENT)
endforeach()
list(APPEND ck_fused_attn_SOURCES ${FMHA_FWD_GEN_BLOBS})

foreach(blob ${FMHA_FWD_SPLITKV_GEN_BLOBS})
  file(RELATIVE_PATH blob_path ${CMAKE_CURRENT_BINARY_DIR}/gen_src ${blob})
  file(COPY_FILE ${CMAKE_CURRENT_BINARY_DIR}/gen_src/tmp/${blob_path} ${blob} ONLY_IF_DIFFERENT)
endforeach()

list(APPEND ck_fused_attn_SOURCES ${FMHA_FWD_SPLITKV_GEN_BLOBS})
foreach(blob ${FMHA_BWD_GEN_BLOBS})
  file(RELATIVE_PATH blob_path ${CMAKE_CURRENT_BINARY_DIR}/gen_src ${blob})
  file(COPY_FILE ${CMAKE_CURRENT_BINARY_DIR}/gen_src/tmp/${blob_path} ${blob} ONLY_IF_DIFFERENT)
endforeach()
list(APPEND ck_fused_attn_SOURCES ${FMHA_BWD_GEN_BLOBS})

set(MHA_BWD_SRC "${CMAKE_CURRENT_BINARY_DIR}/gen_src/mha_bwd.cpp")
set(MHA_FWD_SRC "${CMAKE_CURRENT_BINARY_DIR}/gen_src/mha_fwd.cpp")
set(ASM_MHA_BWD_SRC "${CMAKE_CURRENT_BINARY_DIR}/gen_src/asm_fmha_bwd_v3.cpp")

# TODO: other generated files need to be added to `ck_fused_attn_SOURCES`
file(RELATIVE_PATH blob_path ${CMAKE_CURRENT_BINARY_DIR}/gen_src ${MHA_BWD_SRC})
file(COPY_FILE ${CMAKE_CURRENT_BINARY_DIR}/gen_src/tmp/${blob_path} ${MHA_BWD_SRC} ONLY_IF_DIFFERENT)

file(RELATIVE_PATH blob_path ${CMAKE_CURRENT_BINARY_DIR}/gen_src ${MHA_FWD_SRC})
file(COPY_FILE ${CMAKE_CURRENT_BINARY_DIR}/gen_src/tmp/${blob_path} ${MHA_FWD_SRC} ONLY_IF_DIFFERENT)

file(RELATIVE_PATH blob_path ${CMAKE_CURRENT_BINARY_DIR}/gen_src ${ASM_MHA_BWD_SRC})
file(COPY_FILE ${CMAKE_CURRENT_BINARY_DIR}/gen_src/tmp/${blob_path} ${ASM_MHA_BWD_SRC} ONLY_IF_DIFFERENT)
list(APPEND ck_fused_attn_SOURCES ${MHA_BWD_SRC} ${MHA_FWD_SRC} ${ASM_MHA_BWD_SRC})

# remove all previously generated temporary files
file(REMOVE_RECURSE ${CMAKE_CURRENT_BINARY_DIR}/gen_src/tmp)

message(STATUS "Found the following fused attention files:")
foreach(file ${ck_fused_attn_SOURCES})
  message(STATUS " ${file}")
endforeach()

add_library(ck_fused_attn STATIC ${ck_fused_attn_SOURCES})
set(CK_FUSED_ATTN_COMPILE_OPTIONS)
list(APPEND CK_FUSED_ATTN_COMPILE_OPTIONS -Wno-undefined-func-template -DCK_TILE_FMHA_FWD_FAST_EXP2=1 -DCK_TILE_FMHA_FWD_SPLITKV_API=1 -DCK_TILE_FMHA_FWD_APPENDKV_API=0 -DCK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=${CK_FUSED_ATTN_FLOAT_TO_BFLOAT16_DEFAULT} -fgpu-flush-denormals-to-zero -Wno-float-equal -ftemplate-backtrace-limit=0  -fPIC  -Wno-gnu-line-marker -Wunused-variable -Wuninitialized -Werror)
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
target_include_directories(ck_fused_attn PRIVATE ${AITER_INCLUDE_DIR}/mha_fwd.h ${AITER_INCLUDE_DIR}/mha_bwd.h)

find_package(hip)
list(APPEND ck_fused_attn_LINKER_LIBS hip::host hip::device roctx64)
target_link_libraries(ck_fused_attn PUBLIC ${ck_fused_attn_LINKER_LIBS})
target_compile_options(ck_fused_attn PRIVATE ${CK_FUSED_ATTN_COMPILE_OPTIONS})
