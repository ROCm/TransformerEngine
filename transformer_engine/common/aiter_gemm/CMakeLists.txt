# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.21)
set(CMAKE_CXX_STANDARD 17)
project(aiter_gemm LANGUAGES HIP CXX)

set(AITER_GEMM_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/transformer_engine/lib")

# a4w4 kernels are gfx950-only (ASM f4gemm .co blobs ship for gfx950). Drop
# unsupported arches; a runtime guard covers dispatch.
set(__AG_SUPPORTED_ARCHS "gfx950")
set(__AG_ARCHS)
foreach(__arch ${CMAKE_HIP_ARCHITECTURES})
  if(__arch IN_LIST __AG_SUPPORTED_ARCHS)
    list(APPEND __AG_ARCHS ${__arch})
  else()
    message(WARNING "[aiter_gemm] Skipping unsupported arch ${__arch} for a4w4 GEMM build.")
  endif()
endforeach()
if(NOT __AG_ARCHS)
  message(FATAL_ERROR
          "[aiter_gemm] No supported architectures (need one of: ${__AG_SUPPORTED_ARCHS}). "
          "Re-run the build with NVTE_AITER_GEMM=0 to disable the a4w4 GEMM backend.")
endif()

set(__QOLA_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../3rdparty/QoLA")
set(__AITER_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/qola/third_party/aiter")
if(DEFINED ENV{NVTE_AITER_SOURCE_DIR} AND NOT $ENV{NVTE_AITER_SOURCE_DIR} STREQUAL "")
  set(__AITER_SOURCE_DIR $ENV{NVTE_AITER_SOURCE_DIR})
  message(STATUS "[aiter_gemm] Using AITER source from NVTE_AITER_SOURCE_DIR=${__AITER_SOURCE_DIR}. Disable AITER checkout.")
  set(__SKIP_AITER_CHECKOUT TRUE)
else()
  set(__SKIP_AITER_CHECKOUT FALSE)
endif()
set(AITER_INCLUDE_DIR "${__AITER_SOURCE_DIR}/csrc/include")

if(NOT Python_EXECUTABLE)
  find_package(Python COMPONENTS Interpreter QUIET)
endif()

# Resolve the manifest-pinned AITER commit (defines AITER_SHA).
set(__QOLA_MANIFEST "${CMAKE_CURRENT_LIST_DIR}/qola_manifest.toml")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${__QOLA_MANIFEST}")
file(STRINGS "${__QOLA_MANIFEST}" __AITER_COMMIT_LINES
     REGEX "^[ \t]*aiter_commit[ \t]*=[ \t]*\"[^\"]+\"")
list(LENGTH __AITER_COMMIT_LINES __AITER_COMMIT_COUNT)
if(NOT __AITER_COMMIT_COUNT EQUAL 1)
  message(FATAL_ERROR
          "Expected exactly one 'aiter_commit = \"...\"' line in ${__QOLA_MANIFEST}.")
endif()
list(GET __AITER_COMMIT_LINES 0 __AITER_COMMIT_LINE)
string(REGEX MATCH "\"([^\"]+)\"" _UNUSED "${__AITER_COMMIT_LINE}")
set(AITER_SHA "${CMAKE_MATCH_1}")

if(Python_EXECUTABLE AND NOT __SKIP_AITER_CHECKOUT)
  execute_process(
    COMMAND sh -c
      "PYTHONPATH=\"${__QOLA_DIR}:$PYTHONPATH\" '${Python_EXECUTABLE}' -m qola.cli checkout \
      --manifest '${__QOLA_MANIFEST}' \
      --aiter-root '${__AITER_SOURCE_DIR}'"
    RESULT_VARIABLE AITER_CHECKOUT_RESULT
    OUTPUT_VARIABLE AITER_CHECKOUT_OUTPUT
    ERROR_VARIABLE AITER_CHECKOUT_ERROR)
  if(NOT AITER_CHECKOUT_RESULT EQUAL 0)
    message(FATAL_ERROR
            "[aiter_gemm] Failed to sync AITER source at ${__AITER_SOURCE_DIR} to ${AITER_SHA}.\n"
            "${AITER_CHECKOUT_OUTPUT}\n${AITER_CHECKOUT_ERROR}")
  endif()
  message(STATUS "[aiter_gemm] Synced ${__AITER_SOURCE_DIR} to ${AITER_SHA}")
endif()

if(NOT EXISTS "${AITER_INCLUDE_DIR}")
  message(FATAL_ERROR
          "[aiter_gemm] Could not find AITER API at ${AITER_INCLUDE_DIR}.")
endif()

# Obtain the torch-free kernel libs: prebuilt bypass or build from source (QoLA).
set(__AITER_GEMM_LIB_PATH "")
set(__QOLA_INCLUDE_DIR "")
if(DEFINED ENV{AITER_GEMM_PATH})
  message(STATUS "[aiter_gemm] Using prebuilt libs from AITER_GEMM_PATH=$ENV{AITER_GEMM_PATH}")
  set(__AITER_GEMM_LIB_PATH "$ENV{AITER_GEMM_PATH}/lib")
  set(__QOLA_INCLUDE_DIR "$ENV{AITER_GEMM_PATH}/include")
else()
  list(JOIN __AG_ARCHS ";" GPU_ARCHS_STR)
  set(__QOLA_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/qola")
  message(STATUS "[aiter_gemm] Building a4w4 GEMM kernels for ${GPU_ARCHS_STR} via QoLA.")
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=${__QOLA_DIR}:$ENV{PYTHONPATH}"
            ${Python_EXECUTABLE} -m qola.cli build
            --manifest ${__QOLA_MANIFEST}
            --aiter-root ${__AITER_SOURCE_DIR}
            --output-dir ${__QOLA_BUILD_DIR}
            --arch "${GPU_ARCHS_STR}"
            --skip-checkout
    RESULT_VARIABLE QOLA_BUILD_RESULT)
  if(NOT QOLA_BUILD_RESULT EQUAL 0)
    message(FATAL_ERROR "[aiter_gemm] QoLA build failed.")
  endif()
  set(__AITER_GEMM_LIB_PATH "${__QOLA_BUILD_DIR}/lib")
  set(__QOLA_INCLUDE_DIR "${__QOLA_BUILD_DIR}/include")
endif()

if(NOT EXISTS "${__QOLA_INCLUDE_DIR}/qola_config.h")
  message(FATAL_ERROR "[aiter_gemm] Could not find QoLA public headers at ${__QOLA_INCLUDE_DIR}.")
endif()

add_library(aiter_gemm SHARED src/aiter_gemm_a4w4.cpp)
target_include_directories(aiter_gemm PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(aiter_gemm PRIVATE ${AITER_INCLUDE_DIR} ${__QOLA_INCLUDE_DIR})

find_package(hip)
target_link_directories(aiter_gemm PUBLIC ${__AITER_GEMM_LIB_PATH})
target_link_libraries(aiter_gemm PUBLIC
    hip::host hip::device
    -l:te_libgemm_a4w4_blockscale.so
    -l:te_libgemm_a4w4_asm.so)
set_target_properties(aiter_gemm PROPERTIES INSTALL_RPATH "$ORIGIN")

if(NOT "${__AITER_GEMM_LIB_PATH}" STREQUAL "${AITER_GEMM_INSTALL_DIR}")
  install(FILES
          ${__AITER_GEMM_LIB_PATH}/te_libgemm_a4w4_blockscale.so
          ${__AITER_GEMM_LIB_PATH}/te_libgemm_a4w4_asm.so
          DESTINATION ${AITER_GEMM_INSTALL_DIR})
endif()
install(TARGETS aiter_gemm DESTINATION ${AITER_GEMM_INSTALL_DIR})
