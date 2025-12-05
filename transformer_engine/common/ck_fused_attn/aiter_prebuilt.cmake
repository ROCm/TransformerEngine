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
file(REAL_PATH "${CMAKE_CURRENT_LIST_DIR}/../../../3rdparty/aiter" AITER_DIR)
execute_process(
  COMMAND sh -c "git config --global --add safe.directory ${AITER_DIR} 2>/dev/null || true && git -C ${AITER_DIR} rev-parse HEAD"
  OUTPUT_VARIABLE AITER_SHA
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Cache key & local paths
set(KEY "rocm-${ROCM_VER}_aiter-${AITER_SHA}")
set(CACHE_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../build/aiter-prebuilts")
set(EXTRACT_DIR "${CACHE_ROOT}/${KEY}")

# Validate existing cache path
function(is_aiter_cache_valid CACHE_VALID)
  if(EXISTS "${EXTRACT_DIR}/libmha_fwd.so" AND EXISTS "${EXTRACT_DIR}/libmha_bwd.so")
    set(${CACHE_VALID} TRUE PARENT_SCOPE)
    message(STATUS "[AITER-PREBUILT] Found Cached build files at ${EXTRACT_DIR}")
    return()
  endif()

  # Cache is invalid/outdated - clean it
  file(REMOVE_RECURSE "${CACHE_ROOT}")
  file(REMOVE_RECURSE "${CMAKE_BINARY_DIR}/_deps")
endfunction()

# Cache locally built libs
function(cache_local_aiter_build SOURCE_DIR)
  file(MAKE_DIRECTORY "${EXTRACT_DIR}")
  message(STATUS "[AITER-PREBUILT] Caching locally built libs to ${EXTRACT_DIR}")
  file(COPY "${SOURCE_DIR}/libmha_fwd.so" "${SOURCE_DIR}/libmha_bwd.so" DESTINATION "${EXTRACT_DIR}")
endfunction()

# Download prebuilt tgz file
function(download_aiter_prebuilt DOWNLOAD_SUCCESS)
  if(NOT DEFINED ENV{NVTE_AITER_PREBUILT_BASE_URL} OR "$ENV{NVTE_AITER_PREBUILT_BASE_URL}" STREQUAL "")
    return()
  endif()

  set(FILE_URL "$ENV{NVTE_AITER_PREBUILT_BASE_URL}/${KEY}.tar.gz")
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
  message(STATUS "[AITER-PREBUILT] Successfully downloaded.")
  set(${DOWNLOAD_SUCCESS} TRUE PARENT_SCOPE)
endfunction()

# Create prebuilt tgz file to upload
function(create_upload_files)
  # Locate .so files
  if (NOT EXISTS  "${EXTRACT_DIR}/libmha_fwd.so")
    message(FATAL_ERROR "[AITER-PREBUILT] Missing libmha_fwd.so")
  endif()
  if (NOT EXISTS  "${EXTRACT_DIR}/libmha_bwd.so")
    message(FATAL_ERROR "[AITER-PREBUILT] Missing libmha_bwd.so")
  endif()

  # Output paths
  set(OUTPUT_TGZ "/tmp/${KEY}.tar.gz")
  set(OUTPUT_SHA "/tmp/${KEY}.tar.gz.sha256")

  message(STATUS "[AITER-PREBUILT] Creating prebuilt files...")
  # Create archive
  file(ARCHIVE_CREATE
       OUTPUT "${OUTPUT_TGZ}"
       PATHS "${KEY}"
       WORKING_DIRECTORY "${CACHE_ROOT}"
       FORMAT "gnutar"
       COMPRESSION "GZip")

  # Compute SHA256
  file(SHA256 "${OUTPUT_TGZ}" ARCHIVE_HASH)
  file(WRITE "${OUTPUT_SHA}" "${ARCHIVE_HASH}")
  message(STATUS "[AITER-PREBUILT] tgz and sha256 files generated successfully:")
  message(STATUS "  ${OUTPUT_TGZ}")
  message(STATUS "  ${OUTPUT_SHA}")
endfunction()

# ------------------------------------------------------
# Script-mode entry point (to create upload files)
# Usage: cmake -DACTION=upload -P /path/to/aiter_prebuilt.cmake
# ------------------------------------------------------
if (CMAKE_SCRIPT_MODE_FILE)
  if (DEFINED ACTION AND ACTION STREQUAL "upload")
    create_upload_files()
  else()
    message(FATAL_ERROR "[AITER-PREBUILT] Invalid ACTION=${ACTION}. Use upload.")
  endif()
endif()