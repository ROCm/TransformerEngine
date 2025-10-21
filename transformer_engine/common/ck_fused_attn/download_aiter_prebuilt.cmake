# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

cmake_minimum_required(VERSION 3.21)
include(FetchContent)

# Base URL is mandatory
if(NOT DEFINED ENV{NVTE_AITER_PREBUILT_BASE_URL} OR "$ENV{NVTE_AITER_PREBUILT_BASE_URL}" STREQUAL "")
  message(SEND_ERROR " [AITER-PREBUILT] ENV variable NVTE_AITER_PREBUILT_BASE_URL must be set.")
  return()
endif()

execute_process(
  COMMAND git rev-parse --show-toplevel
  OUTPUT_VARIABLE TE_ROOT
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Build a unique key based on ROCm version + AITER commit
set(ROCM_PATH "$ENV{ROCM_PATH}")
if("${ROCM_PATH}" STREQUAL "")
  set(ROCM_PATH "/opt/rocm")
endif()

file(READ "${ROCM_PATH}/.info/version" ROCM_VER_CONTENT)
string(STRIP "${ROCM_VER_CONTENT}" ROCM_VER_CONTENT)
string(REGEX MATCH "^[0-9]+\\.[0-9]+" ROCM_VER "${ROCM_VER_CONTENT}")

execute_process(
    COMMAND git -C "${TE_ROOT}/3rdparty/aiter" rev-parse HEAD
    OUTPUT_VARIABLE AITER_SHA OUTPUT_STRIP_TRAILING_WHITESPACE
  )

set(KEY "rocm-${ROCM_VER}_aiter-${AITER_SHA}")
set(BASE_URL "$ENV{NVTE_AITER_PREBUILT_BASE_URL}")
string(REGEX REPLACE "/$" "" BASE_URL "${BASE_URL}")
set(FILE_URL "${BASE_URL}/aiter-prebuilts/${KEY}.tar.gz")

set(CACHE_ROOT "${TE_ROOT}/build/aiter-prebuilts")
set(EXTRACT_DIR "${CACHE_ROOT}/${KEY}")

# Download prebuilt zip file
function(download_aiter_prebuilt __AITER_MHA_PATH_VAR)
  # Try local cache first
  if(EXISTS "${EXTRACT_DIR}/libmha_fwd.so" AND EXISTS "${EXTRACT_DIR}/libmha_fwd.so")
    message(STATUS "[AITER-PREBUILT] ${EXTRACT_DIR} already exists. Skipping download.")
    set(${__AITER_MHA_PATH_VAR} "${EXTRACT_DIR}" PARENT_SCOPE)
    message(STATUS "[AITER-PREBUILT] Using ${__AITER_MHA_PATH_VAR}='${EXTRACT_DIR}'")
    return()
  endif()

  # Check if prebuilt files exist in the URL provided.
  file(DOWNLOAD "${FILE_URL}.sha256" "/tmp/aiter_prebuilt_${KEY}.sha256" STATUS sha_status LOG sha_log)
  list(GET sha_status 0 sha_code)
  if(NOT sha_code EQUAL 0)
    message(WARNING " [AITER-PREBUILT] Prebuild file with Key=${KEY} doesn't exist in the NVTE_AITER_PREBUILT_BASE_URL provided.")
    return()
  endif()

  file(MAKE_DIRECTORY "${CACHE_ROOT}")
  FetchContent_Declare(
    aiter_prebuilt
    URL "${FILE_URL}"
    SOURCE_DIR "${EXTRACT_DIR}"
    DOWNLOAD_EXTRACT_TIMESTAMP FALSE
  )

  # Download & extract prebuilt files
  message(STATUS "[AITER-PREBUILT] Downloading ${KEY}.tar.gz ...")
  FetchContent_MakeAvailable(aiter_prebuilt)
  message(STATUS "[AITER-PREBUILT] Successfully downloaded to ${EXTRACT_DIR}")
  set(${__AITER_MHA_PATH_VAR} "${EXTRACT_DIR}" PARENT_SCOPE)
  message(STATUS "[AITER-PREBUILT] Using ${__AITER_MHA_PATH_VAR}='${EXTRACT_DIR}'")
endfunction()

# Upload prebuilt zip file
function(upload_aiter_prebuilt)  
  file(GLOB FWD_LIST
      "${TE_ROOT}/build/lib.*-cpython-*/transformer_engine/lib/libmha_fwd.so")
  file(GLOB BWD_LIST
      "${TE_ROOT}/build/lib.*-cpython-*/transformer_engine/lib/libmha_bwd.so")
  list(GET FWD_LIST 0 FWD)
  list(GET BWD_LIST 0 BWD)

  # Locate .so files
  if (NOT EXISTS  ${FWD})
    message(FATAL_ERROR "[AITER-PREBUILT] Missing libmha_fwd.so")
  endif()
  if (NOT EXISTS  ${BWD})
    message(FATAL_ERROR "[AITER-PREBUILT] Missing libmha_bwd.so")
  endif()

  set(TMP_DIR "/tmp/aiter-prebuilts")
  file(MAKE_DIRECTORY "${TMP_DIR}/${KEY}")
  file(COPY "${FWD}" "${BWD}" DESTINATION "${TMP_DIR}/${KEY}")
  
  file(ARCHIVE_CREATE
       OUTPUT "${TMP_DIR}/${KEY}.tar.gz"
       PATHS "${KEY}"
       WORKING_DIRECTORY "${TMP_DIR}"
       FORMAT "gnutar"
       COMPRESSION "GZip")

  message(STATUS "[AITER-PREBUILT] Uploading ${TMP_DIR}/${KEY}.tar.gz ...")
  set(TOKEN $ENV{NVTE_AITER_ACCESS_TOKEN})
  if(NOT TOKEN STREQUAL "")
    file(UPLOAD "${TMP_DIR}/${KEY}.tar.gz" "${FILE_URL}"
         HTTPHEADER "Authorization: Bearer ${TOKEN}"
         SHOW_PROGRESS
         STATUS upload_status
         LOG upload_log
         INACTIVITY_TIMEOUT 60
         TIMEOUT 300)
  else()
    file(UPLOAD "${TMP_DIR}/${KEY}.tar.gz" "${FILE_URL}"
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

  file(REMOVE_RECURSE ${TMP_DIR})
endfunction()

if (CMAKE_SCRIPT_MODE_FILE)
  if (DEFINED ACTION AND ACTION STREQUAL "upload")
    upload_aiter_prebuilt()
  else()
    message(FATAL_ERROR "[AITER-PREBUILT] Invalid ACTION=${ACTION}. Use upload.")
  endif()
endif()