# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

cmake_minimum_required(VERSION 3.21)
include(FetchContent)

# Extract ROCm version
set(ROCM_PATH "$ENV{ROCM_PATH}")
if("${ROCM_PATH}" STREQUAL "")
  set(ROCM_PATH "/opt/rocm")
endif()
file(READ "${ROCM_PATH}/.info/version" ROCM_VER_CONTENT)
string(STRIP "${ROCM_VER_CONTENT}" ROCM_VER_CONTENT)
string(REGEX MATCH "^[0-9]+\\.[0-9]+" ROCM_VER "${ROCM_VER_CONTENT}")

# AITER commit
execute_process(
  COMMAND git -C "${CMAKE_SOURCE_DIR}/../../3rdparty/aiter" rev-parse HEAD
  OUTPUT_VARIABLE AITER_SHA OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Cache key & local paths
set(KEY "rocm-${ROCM_VER}_aiter-${AITER_SHA}")
set(CACHE_ROOT "${CMAKE_SOURCE_DIR}/../../build/aiter-prebuilts")
set(EXTRACT_DIR "${CACHE_ROOT}/${KEY}")

# Initialize FILE_URL
function(init_aiter_base_url )
  
endfunction()

# Validate existing cache path
function(is_aiter_cache_valid CACHE_VALID)
  if(EXISTS "${__AITER_MHA_PATH}/libmha_fwd.so" AND EXISTS "${__AITER_MHA_PATH}/libmha_bwd.so")
    set(EXPECTED_CACHE_PATH "${EXTRACT_DIR}")    
    if(__AITER_MHA_PATH STREQUAL EXPECTED_CACHE_PATH)
      set(${CACHE_VALID} TRUE PARENT_SCOPE)
      message(STATUS "[AITER-PREBUILT] Using Cached __AITER_MHA_PATH=${__AITER_MHA_PATH}")
      return()
    endif()
  endif()

  # Cache is invalid/outdated - clean it
  file(REMOVE_RECURSE "${EXTRACT_DIR}")
  unset(__AITER_MHA_PATH CACHE)
endfunction()

# Cache locally built libs
function(cache_local_aiter_build SOURCE_DIR)
  file(MAKE_DIRECTORY "${EXTRACT_DIR}")
  message(STATUS "[AITER-PREBUILT] Caching locally built libs to ${EXTRACT_DIR}")
  file(COPY "${SOURCE_DIR}/libmha_fwd.so" "${SOURCE_DIR}/libmha_bwd.so" DESTINATION "${EXTRACT_DIR}")
endfunction()

# Download prebuilt zip file
function(download_aiter_prebuilt DOWNLOAD_SUCCESS)
  if(NOT DEFINED ENV{NVTE_AITER_PREBUILT_BASE_URL} OR "$ENV{NVTE_AITER_PREBUILT_BASE_URL}" STREQUAL "")
    return()
  endif()

  set(BASE_URL "$ENV{NVTE_AITER_PREBUILT_BASE_URL}")
  string(REGEX REPLACE "/$" "" BASE_URL "${BASE_URL}")
  set(FILE_URL "${BASE_URL}/${KEY}.tar.gz")

  message(STATUS "[AITER-PREBUILT] NVTE_AITER_PREBUILT_BASE_URL is set - Attempting to download ${KEY}.tar.gz ...")

  # Check if ${KEY}.tar.gz exists in the URL provided.
  file(DOWNLOAD "${FILE_URL}.sha256" "/tmp/aiter_prebuilt_sha256.txt" STATUS sha_status LOG sha_log)
  list(GET sha_status 0 sha_code)
  if(NOT sha_code EQUAL 0)
    message(WARNING " [AITER-PREBUILT] Prebuild file with Key=${KEY} not available in the NVTE_AITER_PREBUILT_BASE_URL provided.")
    return()
  endif()
  file(READ "/tmp/aiter_prebuilt_sha256.txt" AITER_SHA_CONTENT)
  string(STRIP "${AITER_SHA_CONTENT}" AITER_SHA_CONTENT)
  
  file(MAKE_DIRECTORY "${CACHE_ROOT}")
  FetchContent_Declare(
    aiter_prebuilt
    URL "${FILE_URL}"
    URL_HASH SHA256=${AITER_SHA_CONTENT}
    SOURCE_DIR "${EXTRACT_DIR}"
    DOWNLOAD_EXTRACT_TIMESTAMP FALSE
  )

  # Download & extract prebuilt files
  FetchContent_MakeAvailable(aiter_prebuilt)
  
  if(EXISTS "${EXTRACT_DIR}/libmha_fwd.so" AND EXISTS "${EXTRACT_DIR}/libmha_bwd.so")
    message(STATUS "[AITER-PREBUILT] Successfully downloaded.")
    set(${DOWNLOAD_SUCCESS} TRUE PARENT_SCOPE)
  else()
    file(REMOVE_RECURSE "${CACHE_ROOT}")
    message(STATUS "[AITER-PREBUILT] Download unsuccessfull.")
  endif()
endfunction()

# Upload prebuilt
# Requires: NVTE_AITER_PREBUILT_BASE_URL
# Uses __AITER_MHA_PATH contents to create zip file and upload
function(upload_aiter_prebuilt)  
  if(NOT DEFINED ENV{NVTE_AITER_PREBUILT_BASE_URL} OR "$ENV{NVTE_AITER_PREBUILT_BASE_URL}" STREQUAL "")
    message(FATAL_ERROR " [AITER-PREBUILT] ENV variable NVTE_AITER_PREBUILT_BASE_URL must be set.")
  endif()

  set(BASE_URL "$ENV{NVTE_AITER_PREBUILT_BASE_URL}")
  string(REGEX REPLACE "/$" "" BASE_URL "${BASE_URL}")
  set(FILE_URL "${BASE_URL}/${KEY}.tar.gz")

  # Locate .so files
  if (NOT EXISTS  "${__AITER_MHA_PATH}/libmha_fwd.so")
    message(FATAL_ERROR "[AITER-PREBUILT] Missing libmha_fwd.so")
  endif()
  if (NOT EXISTS  "${__AITER_MHA_PATH}/libmha_fwd.so")
    message(FATAL_ERROR "[AITER-PREBUILT] Missing libmha_bwd.so")
  endif()

  # Create archive
  file(ARCHIVE_CREATE
       OUTPUT "/tmp/${KEY}.tar.gz"
       PATHS "${KEY}"
       WORKING_DIRECTORY "/tmp"
       FORMAT "gnutar"
       COMPRESSION "GZip")

  message(STATUS "[AITER-PREBUILT] Uploading /tmp/${KEY}.tar.gz ...")
  set(TOKEN $ENV{NVTE_AITER_ACCESS_TOKEN})
  if(NOT TOKEN STREQUAL "")
    file(UPLOAD "/tmp/${KEY}.tar.gz" "${FILE_URL}"
         HTTPHEADER "Authorization: Bearer ${TOKEN}"
         SHOW_PROGRESS
         STATUS upload_status
         LOG upload_log
         INACTIVITY_TIMEOUT 60
         TIMEOUT 300)
  else()
    file(UPLOAD "/tmp/${KEY}.tar.gz" "${FILE_URL}"
         SHOW_PROGRESS
         STATUS upload_status
         LOG upload_log
         INACTIVITY_TIMEOUT 60
         TIMEOUT 300)
  endif()

  list(GET upload_status 0 upload_code)
  list(GET upload_status 1 upload_err)
  if(upload_code EQUAL 0)
    message(STATUS "[AITER-PREBUILT] Upload complete.")
  else()
    message(FATAL_ERROR "[AITER-PREBUILT] Upload failed.")
  endif()

  file(REMOVE "/tmp/${KEY}.tar.gz")
endfunction()

# ------------------------------------------------------
# Script-mode entry point (for manual upload)
# Usage: cmake -P aiter_prebuilt.cmake -DACTION=upload
# ------------------------------------------------------
if (CMAKE_SCRIPT_MODE_FILE)
  if (DEFINED ACTION AND ACTION STREQUAL "upload")
    upload_aiter_prebuilt()
  else()
    message(FATAL_ERROR "[AITER-PREBUILT] Invalid ACTION=${ACTION}. Use upload.")
  endif()
endif()