# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

cmake_minimum_required(VERSION 3.21)
include(FetchContent)

# Check env vars
function(require_env_var VAR)
  if(NOT DEFINED ENV{${VAR}} OR "$ENV{${VAR}}" STREQUAL "")
    message(SEND_ERROR " [AITER-PREBUILT] Environment variable ${VAR} must be set.")
    return()
  endif()
endfunction()

function(fetch_aiter_prebuilt __AITER_MHA_PATH_VAR)
  # Required env vars
  require_env_var("NVTE_PREBUILT_BASE_URL")
  require_env_var("NVTE_ARTIFACTORY_USER")
  require_env_var("NVTE_ARTIFACTORY_PASSWORD")

  execute_process(COMMAND git rev-parse --show-toplevel
                  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
                  OUTPUT_VARIABLE TE_ROOT OUTPUT_STRIP_TRAILING_WHITESPACE)
  
  # Build a unique key based on ROCm version + AITER commit
  set(ROCM_PATH "$ENV{ROCM_PATH}")
  if("${ROCM_PATH}" STREQUAL "")
    set(ROCM_PATH "/opt/rocm")
  endif()
  file(READ "${ROCM_PATH}/.info/version" ROCM_VER_CONTENT)
  string(STRIP "${ROCM_VER_CONTENT}" ROCM_VER)

  execute_process(
    COMMAND git -C "${TE_ROOT}/3rdparty/aiter" rev-parse HEAD
    OUTPUT_VARIABLE AITER_SHA OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  set(KEY "rocm-${ROCM_VER}_aiter-${AITER_SHA}")
  set(BASE_URL "$ENV{NVTE_PREBUILT_BASE_URL}")
  string(REGEX REPLACE "/$" "" BASE_URL "${BASE_URL}")
  set(ARTIFACT_URL "${BASE_URL}/aiter-prebuilts/${KEY}.tar.gz")
  set(SHA_URL      "${BASE_URL}.sha256")

  # Prepare local cache
  set(CACHE_ROOT "${TE_ROOT}/.cache/aiter-prebuilts")
  set(EXTRACT_DIR "${CACHE_ROOT}/${KEY}")
  set(SHA_PATH    "${CACHE_ROOT}/${KEY}.sha256")

  message(STATUS "[AITER-PREBUILT] Key: ${KEY}")
  message(STATUS "[AITER-PREBUILT] Artifact: ${ARTIFACT_URL}")

  # Try local cache first
  if(EXISTS "${EXTRACT_DIR}/libmha_fwd.so" AND EXISTS "${EXTRACT_DIR}/libmha_fwd.so")
    message(STATUS "[AITER-PREBUILT] ${EXTRACT_DIR} already exists. Skipping download.")
    set(${__AITER_MHA_PATH_VAR} "${EXTRACT_DIR}" PARENT_SCOPE)
    message(STATUS "[AITER-PREBUILT] Using ${__AITER_MHA_PATH_VAR}='${EXTRACT_DIR}'")
    return()
  else()
    file(MAKE_DIRECTORY "${CACHE_ROOT}")
  endif()

  # # Download SHA file
  file(DOWNLOAD "${SHA_URL}" "${SHA_PATH}" STATUS sha_status LOG sha_log SHOW_PROGRESS)
  list(GET sha_status 0 sha_code)
  if(NOT sha_code EQUAL 0)
    file(REMOVE "${SHA_PATH}")
    message(WARNING " [AITER-PREBUILT] Prebuild file with Key=${KEY} doesn't exist in the artifactory.")
    return()
  endif()

  file(READ "${SHA_PATH}" expected_hash)
  string(STRIP "${expected_hash}" expected_hash)

  # Define FetchContent block
  set(FETCHCONTENT_BASE_DIR "${CACHE_ROOT}/_deps")

  FetchContent_Declare(
    aiter_prebuilt
    URL "${ARTIFACT_URL}"
    URL_HASH "SHA256=${expected_hash}"
    SOURCE_DIR "${EXTRACT_DIR}"
    HTTP_USERNAME "$ENV{NVTE_ARTIFACTORY_USER}"
    HTTP_PASSWORD "$ENV{NVTE_ARTIFACTORY_PASSWORD}"
    DOWNLOAD_EXTRACT_TIMESTAMP FALSE
  )

  # --- Download & extract prebuilt files ---
  message(STATUS "[AITER-PREBUILT] Downloading ${ARTIFACT_URL} ...")
  FetchContent_MakeAvailable(aiter_prebuilt)
  file(REMOVE_RECURSE "${FETCHCONTENT_BASE_DIR}")
  message(STATUS "[AITER-PREBUILT] Successfully downloaded to ${EXTRACT_DIR}")
  set(${__AITER_MHA_PATH_VAR} "${EXTRACT_DIR}" PARENT_SCOPE)
  message(STATUS "[AITER-PREBUILT] Using ${__AITER_MHA_PATH_VAR}='${EXTRACT_DIR}'")
endfunction()