# Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.21)
set(CMAKE_CXX_STANDARD 20)
project(ck_fused_attn LANGUAGES HIP CXX)


set(AITER_MHA_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/transformer_engine/lib")

# QoLA itself and the AITER source tree are both resolved once by the parent
# scope (transformer_engine/common) and shared by every QoLA module group, so
# this directory only consumes them -- see the NVTE_QOLA_DIR override there and
# qola_checkout_aiter() in QoLA.cmake for the NVTE_AITER_SOURCE_DIR override.
set(__QOLA_DIR "${TE_QOLA_DIR}")
if(NOT TE_AITER_SOURCE_DIR)
  message(FATAL_ERROR
          "TE_AITER_SOURCE_DIR is not set; ck_fused_attn expects the shared "
          "QoLA AITER checkout performed by transformer_engine/common.")
endif()
set(__AITER_SOURCE_DIR "${TE_AITER_SOURCE_DIR}")
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

# TE keeps a single QoLA manifest so the whole build shares one AITER commit,
# one patch set and one checkout.  Each consumer builds only its own module
# group (see `qola build --group`).  aiter_prebuilt.cmake keys its cache on the
# manifest-pinned commit, so resolve it here as AITER_SHA.
set(__QOLA_MANIFEST "${TE_QOLA_MANIFEST}")
set(__QOLA_GROUP "ck_fused_attn")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${__QOLA_MANIFEST}")
qola_read_manifest_commit("${__QOLA_MANIFEST}" AITER_SHA)

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

# qola_checkout_aiter() already validated AITER's own headers; the CK submodule
# under the checkout is this backend's extra requirement.
message(STATUS "ck_include_dir: ${CK_INCLUDE_DIR}")
if(NOT EXISTS "${CK_INCLUDE_DIR}")
  message(FATAL_ERROR
          "Could not find CK API at ${CK_INCLUDE_DIR}. "
          "Re-run the build to let `qola checkout` clone AITER and its "
          "composable_kernel submodule.")
endif()
message(STATUS "aiter_include_dir: ${AITER_INCLUDE_DIR}")

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
    # ck_jit_build.py shells out to `qola build` with a fixed argument list and
    # cannot forward --group, so the group is passed through the environment.
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E env "QOLA_BUILD_GROUPS=${__QOLA_GROUP}"
          ${Python_EXECUTABLE} "${__CK_JIT_SOURCE_DIR}/ck_jit_build.py" full
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
    qola_build_modules(
      GROUP           ${__QOLA_GROUP}
      MANIFEST        ${__QOLA_MANIFEST}
      BUILD_DIR       ${__QOLA_BUILD_DIR}
      AITER_DIR       ${__AITER_SOURCE_DIR}
      ARCHS           ${CMAKE_HIP_ARCHITECTURES}
      LIBS            te_libmha_fwd.so te_libmha_bwd.so
      OUT_INCLUDE_DIR __QOLA_BUILT_INCLUDE_DIR
      OUT_LIB_DIR     __QOLA_BUILT_LIB_DIR)

    # Copy the final .so libs and exported public headers into the aiter
    # prebuilt cache so downstream consumers see a self-contained tree.
    get_default_aiter_cache_dir(__AITER_CACHE_DIR)
    set(__QOLA_CACHE_LIB "${__AITER_CACHE_DIR}/lib")
    file(MAKE_DIRECTORY ${__QOLA_CACHE_LIB})
    file(GLOB __QOLA_BUILT_LIBS "${__QOLA_BUILT_LIB_DIR}/*.so")
    file(COPY ${__QOLA_BUILT_LIBS} DESTINATION ${__QOLA_CACHE_LIB})
    file(COPY "${__QOLA_BUILT_INCLUDE_DIR}" DESTINATION "${__AITER_CACHE_DIR}")
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
       src/ck_fused_attn_utils.cpp
       src/ck_fused_attn_smallseq.cpp)

message(STATUS "Found the following fused attention files:")
foreach(file ${ck_fused_attn_SOURCES})
  message(STATUS " ${file}")
endforeach()

add_library(ck_fused_attn SHARED ${ck_fused_attn_SOURCES})
set(CK_FUSED_ATTN_COMPILE_OPTIONS)
#-Wno-unknown-warning-option is to suppress warning about -Wlifetime-safety-lifetimebound-violation
#pragma in CK headers which is not supported by the compiler used to build TE.
list(APPEND CK_FUSED_ATTN_COMPILE_OPTIONS
    -DCK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=${CK_FUSED_ATTN_FLOAT_TO_BFLOAT16_DEFAULT}
    -DENABLE_CK=1 -DFA_WITH_NATIVE_SPLITKV=1 -DFA_WITH_SINK=1 -Wno-unused-value -Wno-unknown-warning-option)

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
target_include_directories(ck_fused_attn PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/small_seq_kernels")

find_package(hip)
target_link_directories(ck_fused_attn PUBLIC ${__AITER_MHA_PATH})
list(APPEND ck_fused_attn_LINKER_LIBS hip::host hip::device roctx64 -l:te_libmha_fwd.so -l:te_libmha_bwd.so)
target_link_libraries(ck_fused_attn PUBLIC ${ck_fused_attn_LINKER_LIBS})
target_compile_options(ck_fused_attn PRIVATE ${CK_FUSED_ATTN_COMPILE_OPTIONS})
set_target_properties(ck_fused_attn PROPERTIES INSTALL_RPATH "$ORIGIN")

# In CK_JIT mode the AITER libs are emitted directly into AITER_MHA_INSTALL_DIR
if (NOT "${__AITER_MHA_PATH}" STREQUAL "${AITER_MHA_INSTALL_DIR}")
  install(FILES ${__AITER_MHA_PATH}/te_libmha_fwd.so ${__AITER_MHA_PATH}/te_libmha_bwd.so DESTINATION ${AITER_MHA_INSTALL_DIR})
endif()
install(TARGETS ck_fused_attn DESTINATION ${AITER_MHA_INSTALL_DIR})
