# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

cmake_minimum_required(VERSION 3.21)
include(FetchContent)
find_package(hip REQUIRED)

function(fetch_aiter_prebuilt __AITER_MHA_PATH_VAR)
  # Base URL is mandatory
  if(NOT DEFINED ENV{NVTE_AITER_PREBUILT_BASE_URL} OR "$ENV{NVTE_AITER_PREBUILT_BASE_URL}" STREQUAL "")
    message(SEND_ERROR " [AITER-PREBUILT] ENV variable NVTE_AITER_PREBUILT_BASE_URL must be set.")
    return()
  endif()

  # Build a unique key based on ROCm version + AITER commit
  set(ROCM_VER "${hip_VERSION_MAJOR}.${hip_VERSION_MINOR}")
  execute_process(
    COMMAND git -C "${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/aiter" rev-parse HEAD
    OUTPUT_VARIABLE AITER_SHA OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  set(KEY "rocm-${ROCM_VER}_aiter-${AITER_SHA}")
  set(BASE_URL "$ENV{NVTE_AITER_PREBUILT_BASE_URL}")
  string(REGEX REPLACE "/$" "" BASE_URL "${BASE_URL}")
  set(DOWNLOAD_URL "${BASE_URL}/aiter-prebuilts/${KEY}.tar.gz")
  set(SHA_URL      "${DOWNLOAD_URL}.sha256")

  # Prepare local cache
  set(CACHE_ROOT "${CMAKE_BINARY_DIR}/aiter-prebuilts")
  set(EXTRACT_DIR "${CACHE_ROOT}/${KEY}")
  set(SHA_PATH    "/tmp/aiter_prebuilt_${KEY}.sha256")

  # Try local cache first
  if(EXISTS "${EXTRACT_DIR}/libmha_fwd.so" AND EXISTS "${EXTRACT_DIR}/libmha_fwd.so")
    message(STATUS "[AITER-PREBUILT] ${EXTRACT_DIR} already exists. Skipping download.")
    set(${__AITER_MHA_PATH_VAR} "${EXTRACT_DIR}" PARENT_SCOPE)
    message(STATUS "[AITER-PREBUILT] Using ${__AITER_MHA_PATH_VAR}='${EXTRACT_DIR}'")
    return()
  endif()

  # Download SHA file
  file(DOWNLOAD "${SHA_URL}" "${SHA_PATH}" STATUS sha_status LOG sha_log)
  list(GET sha_status 0 sha_code)
  if(NOT sha_code EQUAL 0)
    file(REMOVE "${SHA_PATH}")
    message(WARNING " [AITER-PREBUILT] Prebuild file with Key=${KEY} doesn't exist in the NVTE_AITER_PREBUILT_BASE_URL provided.")
    return()
  endif()

  FetchContent_Declare(
    aiter_prebuilt
    URL "${DOWNLOAD_URL}"
    SOURCE_DIR "${EXTRACT_DIR}"
    DOWNLOAD_EXTRACT_TIMESTAMP FALSE
  )

  # Download & extract prebuilt files
  message(STATUS "[AITER-PREBUILT] Downloading ${KEY}.tar.gz ...")
  file(MAKE_DIRECTORY "${CACHE_ROOT}")
  FetchContent_MakeAvailable(aiter_prebuilt)
  message(STATUS "[AITER-PREBUILT] Successfully downloaded to ${EXTRACT_DIR}")
  set(${__AITER_MHA_PATH_VAR} "${EXTRACT_DIR}" PARENT_SCOPE)
  message(STATUS "[AITER-PREBUILT] Using ${__AITER_MHA_PATH_VAR}='${EXTRACT_DIR}'")
endfunction()