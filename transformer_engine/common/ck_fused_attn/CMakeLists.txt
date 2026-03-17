# Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.21)
set(CMAKE_CXX_STANDARD 17)
project(ck_fused_attn LANGUAGES HIP CXX)


set(AITER_MHA_INSTALL_PREFIX "transformer_engine" CACHE STRING "aiter mha shared lib install prefix in TE")

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

if(DEFINED AITER_MHA_PATH)
  message(STATUS "[AITER-BUILD] Using AITER_MHA_PATH=${AITER_MHA_PATH}")
  # use pre-built libmha_fwd.so libmha_bwd.so
  set(__AITER_MHA_PATH ${AITER_MHA_PATH})
else()
  set(__AITER_MHA_PATH "")
  include("${CMAKE_CURRENT_LIST_DIR}/aiter_prebuilt.cmake")
  get_prebuilt_aiter(__AITER_MHA_PATH)

  if(__AITER_MHA_PATH STREQUAL "")
    # If not available, fallback: Build from source
    message(STATUS "[AITER-BUILD] Building aiter from source.")
    get_default_aiter_cache_dir(__AITER_MHA_PATH)
    execute_process(
      COMMAND bash ${CMAKE_CURRENT_LIST_DIR}/aiter_build.sh
              --aiter-dir ${__AITER_SOURCE_DIR}
              --install-dir ${__AITER_MHA_PATH}
              --gpu-archs "${V3_ASM_ARCHS_STR}"
              --ck-tile-bf16 ${CK_FUSED_ATTN_FLOAT_TO_BFLOAT16_DEFAULT}
    )
  else()
    message(STATUS "[AITER-BUILD] Using pre-built AITER from ${__AITER_MHA_PATH}")
  endif()
endif()

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
list(APPEND CK_FUSED_ATTN_COMPILE_OPTIONS
    -DCK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=${CK_FUSED_ATTN_FLOAT_TO_BFLOAT16_DEFAULT})

foreach(ARCH IN LISTS V3_ASM_ARCHS)
  list(APPEND CK_FUSED_ATTN_COMPILE_OPTIONS --offload-arch=${ARCH})
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
list(APPEND ck_fused_attn_LINKER_LIBS hip::host hip::device roctx64 ${__AITER_MHA_PATH}/libmha_fwd.so ${__AITER_MHA_PATH}/libmha_bwd.so)
target_link_libraries(ck_fused_attn PUBLIC ${ck_fused_attn_LINKER_LIBS})
target_compile_options(ck_fused_attn PRIVATE ${CK_FUSED_ATTN_COMPILE_OPTIONS})
set_target_properties(ck_fused_attn PROPERTIES INSTALL_RPATH "$ORIGIN")

install(FILES ${__AITER_MHA_PATH}/libmha_fwd.so ${__AITER_MHA_PATH}/libmha_bwd.so DESTINATION ${CMAKE_INSTALL_PREFIX}/${AITER_MHA_INSTALL_PREFIX}/lib)
install(TARGETS ck_fused_attn DESTINATION ${CMAKE_INSTALL_PREFIX}/${AITER_MHA_INSTALL_PREFIX}/lib)
