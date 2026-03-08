# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

cmake_minimum_required(VERSION 3.21)
include(FetchContent)
# Check if CMP0135 exists (introduced in CMake 3.24)
if(POLICY CMP0135)
  # For CMake >= 3.24, set NEW behavior for DOWNLOAD_EXTRACT_TIMESTAMP explicitly
  cmake_policy(SET CMP0135 NEW)
endif()

# Extract ROCm version
set(ROCM_PATH "$ENV{ROCM_PATH}")
if("${ROCM_PATH}" STREQUAL "")
  set(ROCM_PATH "/opt/rocm")
endif()
file(READ "${ROCM_PATH}/.info/version" ROCM_VER_CONTENT)
string(STRIP "${ROCM_VER_CONTENT}" ROCM_VER_CONTENT)
string(REGEX MATCH "^[0-9]+\\.[0-9]+" ROCM_VER "${ROCM_VER_CONTENT}")
string(REGEX MATCH "^[0-9]+" ROCM_VER_MAJOR "${ROCM_VER}")

# AITER commit
get_git_commit("${CMAKE_CURRENT_LIST_DIR}/../../../3rdparty/aiter" AITER_SHA)

# Cache key & local paths
set(AITER_CACHE_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../build/aiter-prebuilts")

function(get_aiter_cache_key ROCM_VER_PARAM KEY_VAR CACHE_DIR_VAR)
  set(_KEY "rocm-${ROCM_VER_PARAM}_aiter-${AITER_SHA}")
  set(${KEY_VAR} ${_KEY} PARENT_SCOPE)
  set(${CACHE_DIR_VAR} "${AITER_CACHE_ROOT}/${_KEY}" PARENT_SCOPE)
endfunction()

# Validate existing cache path
function(is_aiter_cache_valid ROCM_VER_PARAM CACHE_VALID)
  get_aiter_cache_key("${ROCM_VER_PARAM}" KEY EXTRACT_DIR)
  if(EXISTS "${EXTRACT_DIR}/libmha_fwd.so" AND EXISTS "${EXTRACT_DIR}/libmha_bwd.so")
    set(${CACHE_VALID} TRUE PARENT_SCOPE)
    message(STATUS "[AITER-PREBUILT] Found Cached build files at ${EXTRACT_DIR}")
  endif()
endfunction()

# Main function to get prebuilt aiter libs. 
# It checks cache validity first, if invalid, tries to download.
function(get_prebuilt_aiter PREBUILT_DIR_VAR)
  set(RESULT FALSE)
  foreach(ROCM_VER_PARAM IN LISTS ROCM_VER ROCM_VER_MAJOR)
    is_aiter_cache_valid("${ROCM_VER_PARAM}" RESULT)
    if(RESULT)
      get_aiter_cache_key("${ROCM_VER_PARAM}" _UNUSED CACHE_DIR)
      set(${PREBUILT_DIR_VAR} "${CACHE_DIR}" PARENT_SCOPE)
      return()
    endif()
  endforeach()
  
  # Cache is invalid/outdated - clean it and some build files
  file(REMOVE_RECURSE "${AITER_CACHE_ROOT}")
  file(REMOVE_RECURSE "${CMAKE_BINARY_DIR}/_deps")

  foreach(ROCM_VER_PARAM IN LISTS ROCM_VER ROCM_VER_MAJOR)
    download_aiter_prebuilt("${ROCM_VER_PARAM}" RESULT)
    if(RESULT)
      get_aiter_cache_key("${ROCM_VER_PARAM}" _UNUSED CACHE_DIR)
      set(${PREBUILT_DIR_VAR} "${CACHE_DIR}" PARENT_SCOPE)
      return()
    endif()
  endforeach()
endfunction()

# Cache locally built libs
function(get_default_aiter_cache_dir CACHE_DIR_VAR)
  #Use only ROCM major version for local cache key to maximize cache reuse across minor versions
  get_aiter_cache_key("${ROCM_VER_MAJOR}" _UNUSED EXTRACT_DIR)
  set(${CACHE_DIR_VAR} "${EXTRACT_DIR}" PARENT_SCOPE)
endfunction()

# Download prebuilt tgz file
function(download_aiter_prebuilt ROCM_VER_PARAM DOWNLOAD_SUCCESS)
  if(NOT DEFINED ENV{NVTE_AITER_PREBUILT_BASE_URL} OR "$ENV{NVTE_AITER_PREBUILT_BASE_URL}" STREQUAL "")
    return()
  endif()

  get_aiter_cache_key("${ROCM_VER_PARAM}" KEY EXTRACT_DIR)
  set(FILE_URL "$ENV{NVTE_AITER_PREBUILT_BASE_URL}/${KEY}.tar.gz")
  message(STATUS "[AITER-PREBUILT] NVTE_AITER_PREBUILT_BASE_URL is set - Attempting to download ${KEY}.tar.gz ...")

  # Check if ${KEY}.tar.gz exists in the URL provided.
  file(DOWNLOAD "${FILE_URL}.sha256" "/tmp/aiter_prebuilt_sha256.txt" STATUS sha_status LOG sha_log)
  list(GET sha_status 0 sha_code)
  if(NOT sha_code EQUAL 0)
    message(STATUS " [AITER-PREBUILT] File with Key=${KEY} is not available at the NVTE_AITER_PREBUILT_BASE_URL provided.")
    return()
  endif()
  file(READ "/tmp/aiter_prebuilt_sha256.txt" AITER_SHA_CONTENT)
  string(STRIP "${AITER_SHA_CONTENT}" AITER_SHA_CONTENT)
  
  FetchContent_Declare(
    aiter_prebuilt
    URL "${FILE_URL}"
    URL_HASH SHA256=${AITER_SHA_CONTENT}
    SOURCE_DIR "${EXTRACT_DIR}"
  )

  # Download & extract prebuilt files
  FetchContent_MakeAvailable(aiter_prebuilt)
  message(STATUS "[AITER-PREBUILT] Successfully downloaded to ${EXTRACT_DIR}")
  set(${DOWNLOAD_SUCCESS} TRUE PARENT_SCOPE)
endfunction()
