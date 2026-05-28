# Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.21)
set(CMAKE_CXX_STANDARD 17)
project(ck_fused_attn LANGUAGES HIP CXX)


set(AITER_MHA_INSTALL_PREFIX "transformer_engine" CACHE STRING "aiter mha shared lib install prefix in TE")

#Corresponding runtime check is in nvte_get_fused_attn_backend()
list(FIND CMAKE_HIP_ARCHITECTURES "gfx1250" _gfx1250_idx)
if(NOT _gfx1250_idx EQUAL -1)
  message(WARNING
    "Removing unsupported gfx1250 from CMAKE_HIP_ARCHITECTURES for ck_fused_attn build.")
  list(REMOVE_ITEM CMAKE_HIP_ARCHITECTURES "gfx1250")
  list(LENGTH CMAKE_HIP_ARCHITECTURES _hip_arch_count)
  if(_hip_arch_count EQUAL 0)
    message(FATAL_ERROR
            "No supported architectures remain for the ck_fused_attn build. "
            "Re-run the build with FUSED_ATTN_CK backend disabled.")
  endif()
  set(GPU_TARGETS ${CMAKE_HIP_ARCHITECTURES})
endif()
set(__QOLA_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../3rdparty/QoLA")
set(__AITER_SOURCE_DIR "${__QOLA_DIR}/build/third_party/aiter")
set(__CK_SOURCE_DIR "${__AITER_SOURCE_DIR}/3rdparty/composable_kernel")
set(CK_INCLUDE_DIR "${__CK_SOURCE_DIR}/include")
set(AITER_INCLUDE_DIR "${__AITER_SOURCE_DIR}/csrc/include")

if(NOT Python_EXECUTABLE)
  find_package(Python COMPONENTS Interpreter QUIET)
endif()

# Resolve the manifest-pinned AITER commit (defines AITER_SHA) and bring the
# QoLA-managed AITER source tree to that commit before any consumer reads it
# (header validation below, header includes for the .cpp build later, and
# QoLA's own kernel build if the prebuilt cache misses).
include("${CMAKE_CURRENT_LIST_DIR}/aiter_prebuilt.cmake")

if(Python_EXECUTABLE)
  set(__QOLA_MANIFEST "${CMAKE_CURRENT_LIST_DIR}/qola_manifest.toml")
  # Redirect GIT_CONFIG_GLOBAL to a tempfile carrying `safe.directory = *` so
  # git operations inside the QoLA-managed AITER tree (and its recursive
  # submodules) work in containerized builds where the bind-mounted .git is
  # owned by a different UID than the build process. Mirrors the pattern in
  # transformer_engine/common/CMakeLists.txt:get_git_commit().
  execute_process(
    COMMAND sh -c
      "tmp=$(mktemp /tmp/gitconfig.XXXXXX) || exit 1; \
GIT_CONFIG_GLOBAL=$tmp git config --global --add safe.directory '*' >/dev/null 2>&1; \
GIT_CONFIG_GLOBAL=$tmp PYTHONPATH=\"${__QOLA_DIR}:$PYTHONPATH\" '${Python_EXECUTABLE}' -m qola.cli checkout \
--manifest '${__QOLA_MANIFEST}' \
--aiter-root '${__AITER_SOURCE_DIR}'; \
rc=$?; rm -f \"$tmp\"; exit $rc"
    RESULT_VARIABLE AITER_CHECKOUT_RESULT
    OUTPUT_VARIABLE AITER_CHECKOUT_OUTPUT
    ERROR_VARIABLE AITER_CHECKOUT_ERROR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE
  )
  if(NOT AITER_CHECKOUT_RESULT EQUAL 0)
    message(FATAL_ERROR
            "Failed to sync AITER source tree at ${__AITER_SOURCE_DIR} to "
            "manifest-pinned commit ${AITER_SHA}.\n"
            "${AITER_CHECKOUT_OUTPUT}\n${AITER_CHECKOUT_ERROR}")
  endif()
  message(STATUS "[AITER] Synced ${__AITER_SOURCE_DIR} to ${AITER_SHA}")

  execute_process(
    COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/check_aiter_mha_args.py
            --mode both
            --te-dir "${CMAKE_CURRENT_LIST_DIR}/../../.."
            --aiter-root "${__AITER_SOURCE_DIR}"
    RESULT_VARIABLE AITER_ARG_CHECK_RESULT
    OUTPUT_VARIABLE AITER_ARG_CHECK_OUTPUT
    ERROR_VARIABLE AITER_ARG_CHECK_ERROR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE
  )

  if(NOT AITER_ARG_CHECK_RESULT EQUAL 0)
    message(FATAL_ERROR
            "AITER API validation failed in check_aiter_mha_args.py.\n"
            "${AITER_ARG_CHECK_OUTPUT}\n${AITER_ARG_CHECK_ERROR}")
  endif()
  message(STATUS "AITER API validation passed via check_aiter_mha_args.py")
else()
  message(WARNING "Python interpreter not found; skipping AITER source-tree sync and API validation.")
endif()

# Sanity-check the resolved include directories now that `qola checkout` has
# materialized the AITER tree.
message(STATUS "ck_include_dir: ${CK_INCLUDE_DIR}")
if(NOT EXISTS "${CK_INCLUDE_DIR}")
  message(FATAL_ERROR
          "Could not find CK API at ${CK_INCLUDE_DIR}. "
          "Re-run the build to let `qola checkout` clone AITER and its "
          "composable_kernel submodule.")
endif()

message(STATUS "aiter_include_dir: ${AITER_INCLUDE_DIR}")
if(NOT EXISTS "${AITER_INCLUDE_DIR}")
  message(FATAL_ERROR
          "Could not find AITER API at ${AITER_INCLUDE_DIR}. "
          "Re-run the build to let `qola checkout` clone AITER.")
endif()

if(DEFINED AITER_MHA_PATH)
  message(STATUS "[AITER-BUILD] Using AITER_MHA_PATH=${AITER_MHA_PATH}")
  # use pre-built te_libmha_fwd.so te_libmha_bwd.so
  set(__AITER_MHA_PATH ${AITER_MHA_PATH})
else()
  set(__AITER_MHA_PATH "")
  get_prebuilt_aiter(__AITER_MHA_PATH)

  if(__AITER_MHA_PATH STREQUAL "")
    # If not available, fallback: Build from source via QoLA
    list(JOIN CMAKE_HIP_ARCHITECTURES ";" GPU_ARCHS_STR)
    message(STATUS "[AITER-BUILD] Building AITER kernels for ${GPU_ARCHS_STR} via QoLA.")
    set(__QOLA_BUILD_DIR "${__QOLA_DIR}/build")
    # Same GIT_CONFIG_GLOBAL trick as the earlier `qola.cli checkout` call:
    # `qola.cli build` re-invokes ensure_aiter_commit internally and will hit
    # the same dubious-ownership trap without it.
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=${__QOLA_DIR}:$ENV{PYTHONPATH}"
              ${Python_EXECUTABLE} -m qola.cli build
              --manifest ${__QOLA_MANIFEST}
              --aiter-root ${__AITER_SOURCE_DIR}
              --output-dir ${__QOLA_BUILD_DIR}
              --arch "${GPU_ARCHS_STR}"
      RESULT_VARIABLE QOLA_BUILD_RESULT
    )
    if(NOT QOLA_BUILD_RESULT EQUAL 0)
      message(FATAL_ERROR "[AITER-BUILD] QoLA build failed.")
    endif()

    # Copy the final .so libs and exported public headers into the aiter
    # prebuilt cache so downstream consumers see a self-contained tree.
    get_default_aiter_cache_dir(__QOLA_CACHE_DIR)
    set(__QOLA_CACHE_LIB "${__QOLA_CACHE_DIR}/lib")
    file(MAKE_DIRECTORY ${__QOLA_CACHE_LIB})
    file(GLOB __QOLA_BUILT_LIBS "${__QOLA_BUILD_DIR}/lib/*.so")
    file(COPY ${__QOLA_BUILT_LIBS} DESTINATION ${__QOLA_CACHE_LIB})
    file(COPY "${__QOLA_BUILD_DIR}/include" DESTINATION "${__QOLA_CACHE_DIR}")
    set(__AITER_MHA_PATH "${__QOLA_CACHE_LIB}")
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
    -DCK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=${CK_FUSED_ATTN_FLOAT_TO_BFLOAT16_DEFAULT}
    -DENABLE_CK=1)

# Public QoLA headers ship alongside the .so libs in ${__AITER_MHA_PATH}/../include
# (emitted by qola.cli build, or copied from the QoLA build dir above for the
# source-build path).
set(__QOLA_INCLUDE_DIR "${__AITER_MHA_PATH}/../include")
if(NOT EXISTS "${__QOLA_INCLUDE_DIR}/qola_config.h")
  message(FATAL_ERROR "Could not find QoLA public headers at ${__QOLA_INCLUDE_DIR}.")
endif()

target_include_directories(ck_fused_attn PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(ck_fused_attn PRIVATE ${CK_INCLUDE_DIR} ${__CK_SOURCE_DIR}/example/ck_tile/01_fmha)
target_include_directories(ck_fused_attn PRIVATE ${AITER_INCLUDE_DIR})
target_include_directories(ck_fused_attn PRIVATE ${__QOLA_INCLUDE_DIR})

find_package(hip)
target_link_directories(ck_fused_attn PUBLIC ${__AITER_MHA_PATH})
list(APPEND ck_fused_attn_LINKER_LIBS hip::host hip::device roctx64 -l:te_libmha_fwd.so -l:te_libmha_bwd.so)
target_link_libraries(ck_fused_attn PUBLIC ${ck_fused_attn_LINKER_LIBS})
target_compile_options(ck_fused_attn PRIVATE ${CK_FUSED_ATTN_COMPILE_OPTIONS})
set_target_properties(ck_fused_attn PROPERTIES INSTALL_RPATH "$ORIGIN")

install(FILES ${__AITER_MHA_PATH}/te_libmha_fwd.so ${__AITER_MHA_PATH}/te_libmha_bwd.so DESTINATION ${CMAKE_INSTALL_PREFIX}/${AITER_MHA_INSTALL_PREFIX}/lib)
install(TARGETS ck_fused_attn DESTINATION ${CMAKE_INSTALL_PREFIX}/${AITER_MHA_INSTALL_PREFIX}/lib)
