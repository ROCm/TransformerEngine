# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.21)
# Note: HIP language needed so CMAKE_HIP_ARCHITECTURES is available in this scope.
# We don't compile HIP code directly here but need the arch list for filtering.
project(ck_f4gemm LANGUAGES HIP CXX)

set(AITER_F4GEMM_INSTALL_PREFIX "transformer_engine" CACHE STRING "f4gemm install prefix in TE")
set(__AITER_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../3rdparty/aiter")

# ---- Guard: PyTorch is required (f4gemm modules are pybind11 extensions) ----
if(NOT Python_EXECUTABLE)
  find_package(Python COMPONENTS Interpreter QUIET)
endif()

if(Python_EXECUTABLE)
  execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch; print(torch.__version__)"
    RESULT_VARIABLE TORCH_CHECK_RESULT
    OUTPUT_QUIET ERROR_QUIET
  )
  if(NOT TORCH_CHECK_RESULT EQUAL 0)
    message(STATUS "[AITER-F4GEMM] PyTorch not found; skipping f4gemm build (pybind11 modules require torch).")
    return()
  endif()
else()
  message(STATUS "[AITER-F4GEMM] Python interpreter not found; skipping f4gemm build.")
  return()
endif()

# ---- Guard: gfx950 only ----
set(F4GEMM_ARCHS "")
foreach(ARCH IN LISTS CMAKE_HIP_ARCHITECTURES)
  string(TOLOWER "${ARCH}" ARCH_LOWER)
  if("${ARCH_LOWER}" STREQUAL "gfx950")
    list(APPEND F4GEMM_ARCHS "gfx950")
  endif()
endforeach()

if(NOT F4GEMM_ARCHS)
  message(STATUS "[AITER-F4GEMM] No gfx950 arch requested; skipping f4gemm build.")
  return()
endif()

message(STATUS "[AITER-F4GEMM] Building f4gemm for archs: ${F4GEMM_ARCHS}")

# ---- 3-tier acquisition: explicit path > prebuilt > source build ----
if(DEFINED AITER_F4GEMM_PATH)
  message(STATUS "[AITER-F4GEMM] Using explicit AITER_F4GEMM_PATH=${AITER_F4GEMM_PATH}")
  set(__F4GEMM_PATH ${AITER_F4GEMM_PATH})
else()
  set(__F4GEMM_PATH "")

  # Try prebuilt cache (reuse aiter_prebuilt.cmake from ck_fused_attn)
  include("${CMAKE_CURRENT_LIST_DIR}/../ck_fused_attn/aiter_prebuilt.cmake")

  # Check if prebuilt cache has f4gemm content
  set(__F4GEMM_CACHE_VALID FALSE)
  foreach(ROCM_VER_PARAM IN LISTS ROCM_VER_MAJOR ROCM_VER)
    get_aiter_cache_key("${ROCM_VER_PARAM}" _KEY _CACHE_DIR)
    if(EXISTS "${_CACHE_DIR}/f4gemm/module_gemm_a4w4_asm.so")
      set(__F4GEMM_PATH "${_CACHE_DIR}")
      set(__F4GEMM_CACHE_VALID TRUE)
      message(STATUS "[AITER-F4GEMM] Found cached f4gemm at ${_CACHE_DIR}")
      break()
    endif()
  endforeach()

  if(NOT __F4GEMM_CACHE_VALID)
    # Try downloading prebuilt
    get_prebuilt_aiter(__PREBUILT_DIR)
    if(NOT "${__PREBUILT_DIR}" STREQUAL "")
      if(EXISTS "${__PREBUILT_DIR}/f4gemm/module_gemm_a4w4_asm.so")
        set(__F4GEMM_PATH "${__PREBUILT_DIR}")
        message(STATUS "[AITER-F4GEMM] Using downloaded prebuilt f4gemm from ${__PREBUILT_DIR}")
      endif()
    endif()
  endif()

  if("${__F4GEMM_PATH}" STREQUAL "")
    # Fallback: build from source
    message(STATUS "[AITER-F4GEMM] Building f4gemm from source.")
    get_default_aiter_cache_dir(__F4GEMM_PATH)
    execute_process(
      COMMAND bash ${CMAKE_CURRENT_LIST_DIR}/../ck_fused_attn/aiter_build.sh
              --aiter-dir ${__AITER_SOURCE_DIR}
              --install-dir ${__F4GEMM_PATH}
              --gpu-archs "gfx950"
              --mode f4gemm
      RESULT_VARIABLE F4GEMM_BUILD_RESULT
    )
    if(NOT F4GEMM_BUILD_RESULT EQUAL 0)
      message(WARNING "[AITER-F4GEMM] f4gemm build failed (exit code ${F4GEMM_BUILD_RESULT}). Skipping.")
      return()
    endif()
  endif()
endif()

# ---- Install f4gemm Python extension modules + tuned CSV ----
if(EXISTS "${__F4GEMM_PATH}/f4gemm/module_gemm_a4w4_asm.so")
  install(FILES
    ${__F4GEMM_PATH}/f4gemm/module_gemm_a4w4_asm.so
    DESTINATION ${CMAKE_INSTALL_PREFIX}/${AITER_F4GEMM_INSTALL_PREFIX}/lib/f4gemm)
endif()

if(EXISTS "${__F4GEMM_PATH}/f4gemm/module_gemm_a4w4_blockscale.so")
  install(FILES
    ${__F4GEMM_PATH}/f4gemm/module_gemm_a4w4_blockscale.so
    DESTINATION ${CMAKE_INSTALL_PREFIX}/${AITER_F4GEMM_INSTALL_PREFIX}/lib/f4gemm)
endif()

if(EXISTS "${__F4GEMM_PATH}/f4gemm/a4w4_blockscale_tuned_gemm.csv")
  install(FILES
    ${__F4GEMM_PATH}/f4gemm/a4w4_blockscale_tuned_gemm.csv
    DESTINATION ${CMAKE_INSTALL_PREFIX}/${AITER_F4GEMM_INSTALL_PREFIX}/lib/f4gemm)
endif()

# ---- Install .co blobs into unified aiter/ tree ----
set(__F4GEMM_CO_DIR "${__F4GEMM_PATH}/aiter/gfx950/f4gemm")
if(NOT EXISTS "${__F4GEMM_CO_DIR}")
  # Fallback: copy directly from AITER source tree
  set(__F4GEMM_CO_DIR "${__AITER_SOURCE_DIR}/hsa/gfx950/f4gemm")
endif()

if(EXISTS "${__F4GEMM_CO_DIR}")
  install(DIRECTORY
    ${__F4GEMM_CO_DIR}
    DESTINATION ${CMAKE_INSTALL_PREFIX}/${AITER_F4GEMM_INSTALL_PREFIX}/lib/aiter/gfx950/
    PATTERN "codegen.py" EXCLUDE)
else()
  message(WARNING "[AITER-F4GEMM] f4gemm .co blob directory not found; ASM kernels will not be available.")
endif()
