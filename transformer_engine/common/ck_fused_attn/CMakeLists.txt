# Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#TODO: compile to a shared library
cmake_minimum_required(VERSION 3.28)
set(CMAKE_CXX_STANDARD 20)
#TODO: remove after figuring out how to install clang-scan-deps
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
project(ck_fused_attn LANGUAGES HIP CXX)
# generate fused attn kernels, both fwd/bwd

# remove files that should be regenerated
file(REMOVE_RECURSE ${CMAKE_CURRENT_BINARY_DIR}/gen_src/tmp ${CMAKE_CURRENT_BINARY_DIR}/gen_src/blob_list.txt)

# create gen_src and gen_src/tmp directories if needed
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/gen_src/tmp)

set(__AITER_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../3rdparty/aiter")
set(__CK_SOURCE_DIR "${__AITER_SOURCE_DIR}/3rdparty/composable_kernel")

# so far, there are only gfx942 and gfx950 v3 kernels
SET(V3_ASM_ARCHS_SUPPORTED "gfx942;gfx950")

# filter unsupported gfx archs for v3 kernels in aiter
SET(V3_ASM_ARCHS "")
# generate fwd/bwd v3 kernels for each requested rocm arch
foreach(CK_TARGET_ARCH IN LISTS CMAKE_HIP_ARCHITECTURES)
  string(TOLOWER "${CK_TARGET_ARCH}" CK_TARGET_ARCH_LOWER)
  if(${CK_TARGET_ARCH_LOWER} IN_LIST V3_ASM_ARCHS_SUPPORTED)
    list(APPEND V3_ASM_ARCHS "${CK_TARGET_ARCH_LOWER}")
  else()
    message(WARNING "${CK_TARGET_ARCH} not supported with aiter v3 asm kernels")
  endif()
endforeach()
message(STATUS "AITER V3_ASM_ARCHS: ${V3_ASM_ARCHS}")

# pass GPU_TARGET as env into aiter codegen
list(JOIN V3_ASM_ARCHS ";" V3_ASM_ARCHS_STR)
set(ENV{GPU_ARCHS} "${V3_ASM_ARCHS_STR}")

# generate v2 (CK) kernels
# fwd kernels list
execute_process(
  COMMAND python3 ${__CK_SOURCE_DIR}/example/ck_tile/01_fmha/generate.py
  --api fwd --list_blobs ${CMAKE_CURRENT_BINARY_DIR}/gen_src/fwd_blob_list.txt --receipt 600
)
execute_process(
  COMMAND python3 ${__CK_SOURCE_DIR}/example/ck_tile/01_fmha/generate.py
  --api fwd_splitkv --list_blobs ${CMAKE_CURRENT_BINARY_DIR}/gen_src/fwd_splitkv_blob_list.txt --receipt 600
)
execute_process(
  COMMAND python3 ${__CK_SOURCE_DIR}/example/ck_tile/01_fmha/generate.py
  --api batch_prefill --list_blobs ${CMAKE_CURRENT_BINARY_DIR}/gen_src/fwd_batch_prefill_blob_list.txt --receipt 600
)

# bwd kernels list
execute_process(
  COMMAND python3 ${__CK_SOURCE_DIR}/example/ck_tile/01_fmha/generate.py
  --api bwd --list_blobs ${CMAKE_CURRENT_BINARY_DIR}/gen_src/bwd_blob_list.txt --receipt 600
)

file(STRINGS ${CMAKE_CURRENT_BINARY_DIR}/gen_src/fwd_blob_list.txt FMHA_FWD_GEN_BLOBS)
file(STRINGS ${CMAKE_CURRENT_BINARY_DIR}/gen_src/fwd_splitkv_blob_list.txt FMHA_FWD_SPLITKV_GEN_BLOBS)
file(STRINGS ${CMAKE_CURRENT_BINARY_DIR}/gen_src/fwd_batch_prefill_blob_list.txt FMHA_FWD_BATCH_PREFILL_GEN_BLOBS)
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
  COMMAND python3 ${__CK_SOURCE_DIR}/example/ck_tile/01_fmha/generate.py
  --api batch_prefill --output_dir ${CMAKE_CURRENT_BINARY_DIR}/gen_src/tmp --receipt 600
)

# generate the aiter fwd interface cpp file
execute_process(
  COMMAND python3 ${__AITER_SOURCE_DIR}/csrc/cpp_itfs/mha_fwd_generate.py
  --output_dir ${CMAKE_CURRENT_BINARY_DIR}/gen_src/tmp --receipt 5
)

# generate the actual bwd kernel cpp files
execute_process(
  COMMAND python3 ${__CK_SOURCE_DIR}/example/ck_tile/01_fmha/generate.py
  --api bwd --output_dir ${CMAKE_CURRENT_BINARY_DIR}/gen_src/tmp --receipt 600
)

# generate the aiter bwd interface cpp file
execute_process(
  COMMAND python3 ${__AITER_SOURCE_DIR}/csrc/py_itfs_cu/fmha_bwd_pre_post_kernel_generate.py
  --filter *@*_ndeterministic@*_nbias*_dropout*_ndeterministic* --output_dir ${CMAKE_CURRENT_BINARY_DIR}/gen_src/tmp
)

execute_process(
  COMMAND python3 ${__AITER_SOURCE_DIR}/csrc/cpp_itfs/mha_bwd_generate.py
  --receipt 3 --output_dir ${CMAKE_CURRENT_BINARY_DIR}/gen_src/tmp
)

# generate fwd/bwd v3 kernels for each requested rocm arch
foreach(CK_TARGET_ARCH IN LISTS V3_ASM_ARCHS)
  execute_process(
    COMMAND python3 ${__AITER_SOURCE_DIR}/hsa/${CK_TARGET_ARCH}/fmha_v3_fwd/codegen.py
    --output_dir ${CMAKE_CURRENT_BINARY_DIR}/gen_src/tmp
  )
  execute_process(
    COMMAND python3 ${__AITER_SOURCE_DIR}/hsa/${CK_TARGET_ARCH}/fmha_v3_bwd/codegen.py
    --output_dir ${CMAKE_CURRENT_BINARY_DIR}/gen_src/tmp
  )
endforeach()

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

foreach(blob ${FMHA_FWD_BATCH_PREFILL_GEN_BLOBS})
  file(RELATIVE_PATH blob_path ${CMAKE_CURRENT_BINARY_DIR}/gen_src ${blob})
  file(COPY_FILE ${CMAKE_CURRENT_BINARY_DIR}/gen_src/tmp/${blob_path} ${blob} ONLY_IF_DIFFERENT)
endforeach()
list(APPEND ck_fused_attn_SOURCES ${FMHA_FWD_BATCH_PREFILL_GEN_BLOBS})

foreach(blob ${FMHA_BWD_GEN_BLOBS})
  file(RELATIVE_PATH blob_path ${CMAKE_CURRENT_BINARY_DIR}/gen_src ${blob})
  file(COPY_FILE ${CMAKE_CURRENT_BINARY_DIR}/gen_src/tmp/${blob_path} ${blob} ONLY_IF_DIFFERENT)
endforeach()
list(APPEND ck_fused_attn_SOURCES ${FMHA_BWD_GEN_BLOBS})

# add generated cpp files into ck_fused_attn_sources
set(MHA_BWD_SRC "${CMAKE_CURRENT_BINARY_DIR}/gen_src/mha_bwd.cpp")
set(MHA_FWD_SRC "${CMAKE_CURRENT_BINARY_DIR}/gen_src/mha_fwd.cpp")

file(RELATIVE_PATH blob_path ${CMAKE_CURRENT_BINARY_DIR}/gen_src ${MHA_BWD_SRC})
file(COPY_FILE ${CMAKE_CURRENT_BINARY_DIR}/gen_src/tmp/${blob_path} ${MHA_BWD_SRC} ONLY_IF_DIFFERENT)

file(RELATIVE_PATH blob_path ${CMAKE_CURRENT_BINARY_DIR}/gen_src ${MHA_FWD_SRC})
file(COPY_FILE ${CMAKE_CURRENT_BINARY_DIR}/gen_src/tmp/${blob_path} ${MHA_FWD_SRC} ONLY_IF_DIFFERENT)

list(APPEND ck_fused_attn_SOURCES ${MHA_BWD_SRC} ${MHA_FWD_SRC})

foreach(CK_TARGET_ARCH IN LISTS V3_ASM_ARCHS)
  set(ASM_MHA_FWD_SRC "${CMAKE_CURRENT_BINARY_DIR}/gen_src/asm_fmha_fwd_v3_${CK_TARGET_ARCH}.cpp")
  set(ASM_MHA_BWD_SRC "${CMAKE_CURRENT_BINARY_DIR}/gen_src/asm_fmha_bwd_v3_${CK_TARGET_ARCH}.cpp")

  file(RELATIVE_PATH blob_path ${CMAKE_CURRENT_BINARY_DIR}/gen_src ${ASM_MHA_BWD_SRC})
  file(COPY_FILE ${CMAKE_CURRENT_BINARY_DIR}/gen_src/tmp/${blob_path} ${ASM_MHA_BWD_SRC} ONLY_IF_DIFFERENT)
  
  file(RELATIVE_PATH blob_path ${CMAKE_CURRENT_BINARY_DIR}/gen_src ${ASM_MHA_FWD_SRC})
  file(COPY_FILE ${CMAKE_CURRENT_BINARY_DIR}/gen_src/tmp/${blob_path} ${ASM_MHA_FWD_SRC} ONLY_IF_DIFFERENT)
  list(APPEND ck_fused_attn_SOURCES ${ASM_MHA_BWD_SRC} ${ASM_MHA_FWD_SRC})
endforeach()

# remove all previously generated temporary files
file(REMOVE_RECURSE ${CMAKE_CURRENT_BINARY_DIR}/gen_src/tmp)

message(STATUS "Found the following fused attention files:")
foreach(file ${ck_fused_attn_SOURCES})
  message(STATUS " ${file}")
endforeach()

add_library(ck_fused_attn STATIC ${ck_fused_attn_SOURCES})
set(CK_FUSED_ATTN_COMPILE_OPTIONS)
list(APPEND CK_FUSED_ATTN_COMPILE_OPTIONS -Wno-undefined-func-template -DCK_TILE_FMHA_FWD_FAST_EXP2=1 -DCK_TILE_FMHA_FWD_SPLITKV_API=1 -DCK_TILE_FMHA_FWD_APPENDKV_API=0 -DCK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=${CK_FUSED_ATTN_FLOAT_TO_BFLOAT16_DEFAULT} -fgpu-flush-denormals-to-zero -Wno-float-equal -ftemplate-backtrace-limit=0  -fPIC  -Wno-gnu-line-marker -Wunused-variable -Wuninitialized "SHELL:-mllvm -enable-post-misched=0" "SHELL:-mllvm -amdgpu-early-inline-all=true" "SHELL:-mllvm -amdgpu-function-calls=false" "SHELL:-mllvm -amdgpu-coerce-illegal-types=1" "SHELL:-mllvm --amdgpu-kernarg-preload-count=16")

foreach(CK_TARGET_ARCH IN LISTS CMAKE_HIP_ARCHITECTURES)
  list(APPEND CK_FUSED_ATTN_COMPILE_OPTIONS --offload-arch=${CK_TARGET_ARCH})
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
list(APPEND ck_fused_attn_LINKER_LIBS hip::host hip::device roctx64)
target_link_libraries(ck_fused_attn PUBLIC ${ck_fused_attn_LINKER_LIBS})
target_compile_options(ck_fused_attn PRIVATE ${CK_FUSED_ATTN_COMPILE_OPTIONS})
