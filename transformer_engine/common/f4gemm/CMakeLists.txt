# Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.21)
set(CMAKE_CXX_STANDARD 17)
project(f4gemm_qola LANGUAGES HIP CXX)

set(F4GEMM_INSTALL_PREFIX "transformer_engine" CACHE STRING "f4gemm shared lib install prefix in TE")

set(__AITER_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../3rdparty/QoLA/3rdparty/aiter")

if(NOT Python_EXECUTABLE)
  find_package(Python COMPONENTS Interpreter QUIET)
endif()

# Only gfx942 and gfx950 support v3 ASM kernels
SET(V3_ASM_ARCHS_SUPPORTED "gfx942;gfx950")
SET(V3_ASM_ARCHS "")
foreach(CK_TARGET_ARCH IN LISTS CMAKE_HIP_ARCHITECTURES)
  string(TOLOWER "${CK_TARGET_ARCH}" CK_TARGET_ARCH_LOWER)
  if(${CK_TARGET_ARCH_LOWER} IN_LIST V3_ASM_ARCHS_SUPPORTED)
    list(APPEND V3_ASM_ARCHS "${CK_TARGET_ARCH_LOWER}")
  else()
    message(WARNING "${CK_TARGET_ARCH} not supported with f4gemm ASM kernels")
  endif()
endforeach()
list(JOIN V3_ASM_ARCHS ";" V3_ASM_ARCHS_STR)
message(STATUS "[F4GEMM] V3_ASM_ARCHS: ${V3_ASM_ARCHS}")

if(DEFINED AITER_F4GEMM_PATH)
  message(STATUS "[F4GEMM-BUILD] Using AITER_F4GEMM_PATH=${AITER_F4GEMM_PATH}")
  set(__AITER_F4GEMM_PATH ${AITER_F4GEMM_PATH})
elseif(DEFINED AITER_MHA_PATH
       AND EXISTS "${AITER_MHA_PATH}/te_module_gemm_a4w4_asm.so"
       AND EXISTS "${AITER_MHA_PATH}/te_module_gemm_a4w4_blockscale.so")
  # Reuse the MHA prebuilt path if it also contains f4gemm files
  message(STATUS "[F4GEMM-BUILD] Using AITER_MHA_PATH for f4gemm: ${AITER_MHA_PATH}")
  set(__AITER_F4GEMM_PATH ${AITER_MHA_PATH})
else()
  set(__AITER_F4GEMM_PATH "")
  include("${CMAKE_CURRENT_LIST_DIR}/../ck_fused_attn/aiter_prebuilt.cmake")
  get_prebuilt_f4gemm(__AITER_F4GEMM_PATH)

  if(__AITER_F4GEMM_PATH STREQUAL "")
    # Build from source via QoLA
    message(STATUS "[F4GEMM-BUILD] Building f4gemm kernels via QoLA.")
    get_default_aiter_cache_dir(__QOLA_OUTPUT_DIR)
    set(__QOLA_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../3rdparty/QoLA")
    set(__QOLA_MANIFEST "${CMAKE_CURRENT_LIST_DIR}/../manifests/f4gemm.toml")
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=${__QOLA_DIR}:$ENV{PYTHONPATH}"
              ${Python_EXECUTABLE} -m qola.cli build
              --manifest ${__QOLA_MANIFEST}
              --aiter-root ${__AITER_SOURCE_DIR}
              --output-dir ${__QOLA_OUTPUT_DIR}
              --arch "${V3_ASM_ARCHS_STR}"
      RESULT_VARIABLE QOLA_F4GEMM_BUILD_RESULT
    )
    if(NOT QOLA_F4GEMM_BUILD_RESULT EQUAL 0)
      message(FATAL_ERROR "[F4GEMM-BUILD] QoLA build failed.")
    endif()
    set(__AITER_F4GEMM_PATH "${__QOLA_OUTPUT_DIR}/lib")
  else()
    message(STATUS "[F4GEMM-BUILD] Using pre-built f4gemm from ${__AITER_F4GEMM_PATH}")
  endif()
endif()

install(FILES ${__AITER_F4GEMM_PATH}/te_module_gemm_a4w4_asm.so
              ${__AITER_F4GEMM_PATH}/te_module_gemm_a4w4_blockscale.so
        DESTINATION ${CMAKE_INSTALL_PREFIX}/${F4GEMM_INSTALL_PREFIX}/lib
        OPTIONAL)
