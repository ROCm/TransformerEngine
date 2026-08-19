# Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.21)
set(CMAKE_CXX_STANDARD 17)
project(ck_fused_attn LANGUAGES HIP CXX)


set(AITER_MHA_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/transformer_engine/lib")

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
set(__AITER_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/qola/third_party/aiter")
if (DEFINED ENV{NVTE_AITER_SOURCE_DIR} AND NOT $ENV{NVTE_AITER_SOURCE_DIR} STREQUAL "")
  set(__AITER_SOURCE_DIR $ENV{NVTE_AITER_SOURCE_DIR})
  message(STATUS "Using AITER source from NVTE_AITER_SOURCE_DIR=${__AITER_SOURCE_DIR}. Disable AITER checkout.")
  set(__SKIP_AITER_CHECKOUT TRUE)
else()
  set(__SKIP_AITER_CHECKOUT FALSE)
endif()
set(__CK_SOURCE_DIR "${__AITER_SOURCE_DIR}/3rdparty/composable_kernel")
set(CK_INCLUDE_DIR "${__CK_SOURCE_DIR}/include")
set(AITER_INCLUDE_DIR "${__AITER_SOURCE_DIR}/csrc/include")

if(NOT Python_EXECUTABLE)
  find_package(Python COMPONENTS Interpreter QUIET)
  if (NOT Python_EXECUTABLE)
    #Don't fail the entire build if Python isn't found since it may pass if pre-built AITER is used
    message(ERROR "Python interpreter not found. Some build steps will be skipped or fail. Please install Python and ensure it is on the PATH.")
  endif()
endif()

# Resolve the manifest-pinned AITER commit (defines AITER_SHA) and bring the
# QoLA-managed AITER source tree to that commit before any consumer reads it
# (header validation below, header includes for the .cpp build later, and
# QoLA's own kernel build if the prebuilt cache misses).
set(__QOLA_MANIFEST "${CMAKE_CURRENT_LIST_DIR}/qola_manifest.toml")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${__QOLA_MANIFEST}")
file(STRINGS "${__QOLA_MANIFEST}" __AITER_COMMIT_LINES
     REGEX "^[ \t]*aiter_commit[ \t]*=[ \t]*\"[^\"]+\"")
list(LENGTH __AITER_COMMIT_LINES __AITER_COMMIT_COUNT)
if(NOT __AITER_COMMIT_COUNT EQUAL 1)
  message(FATAL_ERROR
          "Expected exactly one 'aiter_commit = \"...\"' line in "
          "${__QOLA_MANIFEST}, found ${__AITER_COMMIT_COUNT}.")
endif()
list(GET __AITER_COMMIT_LINES 0 __AITER_COMMIT_LINE)
string(REGEX MATCH "\"([^\"]+)\"" _UNUSED "${__AITER_COMMIT_LINE}")
set(AITER_SHA "${CMAKE_MATCH_1}")
if("${AITER_SHA}" STREQUAL "")
  message(FATAL_ERROR
          "Failed to parse 'aiter_commit = \"...\"' line in ${__QOLA_MANIFEST}.")
endif()
if (NOT __SKIP_AITER_CHECKOUT)
  execute_process(
    COMMAND sh -c
      "PYTHONPATH=\"${__QOLA_DIR}:$PYTHONPATH\" '${Python_EXECUTABLE}' -m qola.cli checkout \
--manifest '${__QOLA_MANIFEST}' \
--aiter-root '${__AITER_SOURCE_DIR}'"
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
endif()

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

set(__AITER_CACHE_DIR "")
set(__AITER_MHA_PATH "")
set(__QOLA_INCLUDE_DIR "")
if(NOT "$ENV{NVTE_CK_JIT}" STREQUAL "0")
  set(__USE_CK_JIT TRUE)
else()
  set(__USE_CK_JIT FALSE)
endif()

if(DEFINED ENV{AITER_MHA_PATH})
  message(STATUS "[AITER-BUILD] Using AITER_MHA_PATH=$ENV{AITER_MHA_PATH}")
  # use pre-built libraries and includes from a location specified by the user
  set(__AITER_CACHE_DIR $ENV{AITER_MHA_PATH})
elseif(NOT __USE_CK_JIT) # CK_JIT always builds from source; skip the prebuilt cache
  include("${CMAKE_CURRENT_LIST_DIR}/aiter_prebuilt.cmake")
  get_prebuilt_aiter(__AITER_CACHE_DIR)
elseif(DEFINED ENV{NVTE_AITER_PREBUILT_BASE_URL})
  message(WARNING "[AITER-BUILD] NVTE_AITER_PREBUILT_BASE_URL is set but will be ignored because CK_JIT is enabled.")
endif()

if(__AITER_CACHE_DIR STREQUAL "")
  # If not available or not requested, build from source via QoLA
  list(JOIN CMAKE_HIP_ARCHITECTURES ";" GPU_ARCHS_STR)
  message(STATUS "[AITER-BUILD] Building AITER kernels for ${GPU_ARCHS_STR} via QoLA.")
  if(__USE_CK_JIT)
    message(STATUS "[AITER-BUILD] CK_JIT is enabled; will build CK kernels via CK_JIT.")
    set(__CK_JIT_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/ck_jit")
    set(__QOLA_BUILD_DIR "${__CK_JIT_BUILD_DIR}/qola") #Need it under ck_jit to clean on full build
    if(DEFINED ENV{NVTE_CK_JIT_DIR})
      set(__CK_JIT_SOURCE_DIR $ENV{NVTE_CK_JIT_DIR})
    else()
      set(__CK_JIT_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../3rdparty/ck_jit")
    endif()
    execute_process(
      COMMAND ${Python_EXECUTABLE} "${__CK_JIT_SOURCE_DIR}/ck_jit_build.py" full
          --with-qola
          --qola-dir ${__QOLA_DIR}
          --qola-manifest ${__QOLA_MANIFEST}
          --qola-output "${__QOLA_BUILD_DIR}"
          --gpu-archs "${GPU_ARCHS_STR}"
          --aiter-dir ${__AITER_SOURCE_DIR}
          --tmp-dir "${__CK_JIT_BUILD_DIR}"
          --install-dir ${AITER_MHA_INSTALL_DIR}
          --jit-name "te_ck_jit"
      RESULT_VARIABLE CKJIT_BUILD_RESULT
    )
    if(NOT CKJIT_BUILD_RESULT EQUAL 0)
      message(FATAL_ERROR "[AITER-BUILD] CK-JIT build failed.")
    endif()
    # CK_JIT emits the libs straight into the install dir; nothing to cache.
    set(__AITER_MHA_PATH ${AITER_MHA_INSTALL_DIR})
    set(__QOLA_INCLUDE_DIR "${__QOLA_BUILD_DIR}/include")
  else()
    set(__QOLA_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/qola")
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=${__QOLA_DIR}:$ENV{PYTHONPATH}"
              ${Python_EXECUTABLE} -m qola.cli build
              --manifest ${__QOLA_MANIFEST}
              --aiter-root ${__AITER_SOURCE_DIR}
              --output-dir ${__QOLA_BUILD_DIR}
              --arch "${GPU_ARCHS_STR}"
              --skip-checkout
      RESULT_VARIABLE QOLA_BUILD_RESULT
    )
    if(NOT QOLA_BUILD_RESULT EQUAL 0)
      message(FATAL_ERROR "[AITER-BUILD] QoLA build failed.")
    endif()

    # Copy the final .so libs and exported public headers into the aiter
    # prebuilt cache so downstream consumers see a self-contained tree.
    get_default_aiter_cache_dir(__AITER_CACHE_DIR)
    set(__QOLA_CACHE_LIB "${__AITER_CACHE_DIR}/lib")
    file(MAKE_DIRECTORY ${__QOLA_CACHE_LIB})
    file(GLOB __QOLA_BUILT_LIBS "${__QOLA_BUILD_DIR}/lib/*.so")
    file(COPY ${__QOLA_BUILT_LIBS} DESTINATION ${__QOLA_CACHE_LIB})
    file(COPY "${__QOLA_BUILD_DIR}/include" DESTINATION "${__AITER_CACHE_DIR}")
    set(__AITER_MHA_PATH "${__QOLA_CACHE_LIB}")
    set(__QOLA_INCLUDE_DIR "${__AITER_CACHE_DIR}/include")
  endif()
else()
  message(STATUS "[AITER-BUILD] Using pre-built AITER from ${__AITER_CACHE_DIR}")
  set(__AITER_MHA_PATH "${__AITER_CACHE_DIR}/lib")
  set(__QOLA_INCLUDE_DIR "${__AITER_CACHE_DIR}/include")
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
    -DENABLE_CK=1 -DFA_WITH_NATIVE_SPLITKV=1)

# Public QoLA headers ship alongside the .so libs in ${__AITER_MHA_PATH}/../include
# (emitted by qola.cli build, or copied from the QoLA build dir above for the
# source-build path).
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

if (NOT "${__AITER_MHA_PATH}" STREQUAL "${AITER_MHA_INSTALL_DIR}")
  install(FILES ${__AITER_MHA_PATH}/te_libmha_fwd.so ${__AITER_MHA_PATH}/te_libmha_bwd.so DESTINATION ${AITER_MHA_INSTALL_DIR})
endif()
install(TARGETS ck_fused_attn DESTINATION ${AITER_MHA_INSTALL_DIR})
