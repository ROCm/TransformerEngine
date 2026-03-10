# Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.21)
set(CMAKE_CXX_STANDARD 17)
project(ck_fused_attn LANGUAGES HIP CXX)


set(AITER_MHA_INSTALL_PREFIX "transformer_engine" CACHE STRING "aiter mha shared lib install prefix in TE")

set(__AITER_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../3rdparty/aiter")
set(__AITER_TEST_DIR "${__AITER_SOURCE_DIR}/op_tests/cpp/mha")
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
  # use pre-built libraries
  set(__AITER_MHA_PATH ${AITER_MHA_PATH})
else()
  set(AITER_CACHE_VALID FALSE)
  set(AITER_PREBUILT_DOWNLOAD_SUCCESS FALSE)
  include("${CMAKE_CURRENT_LIST_DIR}/aiter_prebuilt.cmake")

  # If cached path already exists and is valid
  is_aiter_cache_valid(AITER_CACHE_VALID)

  if(NOT AITER_CACHE_VALID)
    # Try downloading prebuilt files if NVTE_AITER_PREBUILT_BASE_URL is set.
    download_aiter_prebuilt(AITER_PREBUILT_DOWNLOAD_SUCCESS)
    
    # If not downloaded, Fallback: Build from source
    if(NOT AITER_PREBUILT_DOWNLOAD_SUCCESS)
      message(STATUS " [AITER-BUILD] Building aiter from source.")
      execute_process(
        COMMAND bash ${CMAKE_CURRENT_LIST_DIR}/aiter_build.sh
                --aiter-dir ${__AITER_SOURCE_DIR}
                --aiter-test-dir ${__AITER_TEST_DIR}
                --gpu-archs "${V3_ASM_ARCHS_STR}"
                --ck-tile-bf16 ${CK_FUSED_ATTN_FLOAT_TO_BFLOAT16_DEFAULT}
      )
      # libmha_fwd.so and libmha_bwd.so will be under 3rdparty/aiter/op_tests/cpp/mha 
      cache_local_aiter_build(${__AITER_TEST_DIR})
    endif()
  endif()
  set(__AITER_MHA_PATH "${EXTRACT_DIR}")
  message(STATUS "[AITER-BUILD] Using __AITER_MHA_PATH=${__AITER_MHA_PATH}")
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
target_link_options(ck_fused_attn PRIVATE -Wl,--exclude-libs,ALL)

set(__AITER_MHA_FWD_LIB "${__AITER_MHA_PATH}/libmha_fwd.a")
set(__AITER_MHA_BWD_LIB "${__AITER_MHA_PATH}/libmha_bwd.a")
find_package(hip)
list(APPEND ck_fused_attn_LINKER_LIBS hip::host hip::device roctx64 ${__AITER_MHA_FWD_LIB} ${__AITER_MHA_BWD_LIB})

target_link_libraries(ck_fused_attn PUBLIC ${ck_fused_attn_LINKER_LIBS})
target_compile_options(ck_fused_attn PRIVATE ${CK_FUSED_ATTN_COMPILE_OPTIONS})
set_target_properties(ck_fused_attn PROPERTIES INSTALL_RPATH "$ORIGIN")

install(TARGETS ck_fused_attn DESTINATION ${CMAKE_INSTALL_PREFIX}/${AITER_MHA_INSTALL_PREFIX}/lib)
