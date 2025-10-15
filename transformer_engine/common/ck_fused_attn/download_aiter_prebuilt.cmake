# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

cmake_minimum_required(VERSION 3.21)

# Check env vars
function(require_env_var VAR)
  if(NOT DEFINED ENV{${VAR}} OR "$ENV{${VAR}}" STREQUAL "")
    message(SEND_ERROR "[AITER-PREBUILT] [Error] Environment variable ${VAR} must be set.")
  endif()
endfunction()

function(read_file_trim file_path out_var)
  file(READ "${file_path}" content)
  string(STRIP "${content}" stripped)
  set(${out_var} "${stripped}" PARENT_SCOPE)
endfunction()

function(check_file_hash file_path expected_hash result_var)
  file(SHA256 "${file_path}" actual_value)
  if(NOT "${actual_value}" STREQUAL "${expected_hash}")
    message(WARNING "[AITER-PREBUILT] SHA256 mismatch:
            Expected: ${expected_hash}
            Actual:   ${actual_value}")
    set(${result_var} FALSE PARENT_SCOPE)
  else()
    message(STATUS "[AITER-PREBUILT] SHA256 verified successfully.")
    set(${result_var} TRUE PARENT_SCOPE)
  endif()
endfunction()

# Download a file with retries
function(download_with_retries URL OUTPUT_PATH RETRIES)
  set(download_retry_codes 7 6 8 15 28 35)
  set(status_code 7)
  foreach(i RANGE ${RETRIES})
    if(status_code IN_LIST download_retry_codes)
      if(i GREATER 0)
        math(EXPR sleep_seconds "5 * ${i}")
        message(STATUS "[AITER-PREBUILT] Retrying download in ${sleep_seconds}s (attempt #${i})...")
        execute_process(COMMAND "${CMAKE_COMMAND}" -E sleep "${sleep_seconds}")
      endif()
      message(STATUS "[AITER-PREBUILT] Downloading attempt #${i} ...")
      file(
        DOWNLOAD "${URL}" "${OUTPUT_PATH}"
        SHOW_PROGRESS
        STATUS status
        LOG log
      )
      list(GET status 0 status_code)
      
      if(status_code EQUAL 22 OR status_code EQUAL 23 OR status_code EQUAL 404)
        message(WARNING "[AITER-PREBUILT] Artifact not found at ${URL} (status=${status_code}).")
        set(${OUTPUT_PATH}_NOT_FOUND TRUE PARENT_SCOPE)
        return()
      endif()

      if(status_code EQUAL 0)
        message(STATUS "[AITER-PREBUILT] Successfully downloaded ${OUTPUT_PATH}")
        return()
      else()
        message(WARNING "[AITER-PREBUILT] Failed (status=${status_code}), log:\n${log}")
      endif()
    endif()
  endforeach()
  message(WARNING "[AITER-PREBUILT] All ${RETRIES} download attempts failed for ${URL}")
endfunction()

# Extract tarball
function(extract_tarball TAR_PATH DEST_DIR)
  message(STATUS "[AITER-PREBUILT] Extracting ${TAR_PATH} -> ${DEST_DIR}")
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E tar xzf "${TAR_PATH}"
    WORKING_DIRECTORY "${DEST_DIR}"
    RESULT_VARIABLE extract_result
  )
  if(NOT extract_result EQUAL 0)
    message(SEND_ERROR "[AITER-PREBUILT] Failed to extract ${TAR_PATH}")
  endif()
endfunction()

# Main function to handle prebuilt fetch
function(fetch_aiter_prebuilt __AITER_MHA_PATH_VAR)
  # Required env vars
  require_env_var("NVTE_PREBUILT_BASE_URL")
  require_env_var("NVTE_ARTIFACTORY_USER")
  require_env_var("NVTE_ARTIFACTORY_PASSWORD")

  message(STATUS "[AITER-PREBUILT] Started downloading...")

  # Required info to derive key
  execute_process(COMMAND git rev-parse --show-toplevel
                  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
                  OUTPUT_VARIABLE TE_ROOT OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(ROCM_PATH "$ENV{ROCM_PATH}")
  if("${ROCM_PATH}" STREQUAL "")
    set(ROCM_PATH "/opt/rocm")
  endif()
  file(READ "${ROCM_PATH}/.info/version" ROCM_VER_CONTENT)
  string(STRIP "${ROCM_VER_CONTENT}" ROCM_VER)
  execute_process(
    COMMAND git -C "${TE_ROOT}/3rdparty/aiter" rev-parse --short=9 HEAD
    OUTPUT_VARIABLE AITER_SHA OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(DEFINED ENV{NVTE_ROCM_ARCH})
    set(ARCHS "$ENV{NVTE_ROCM_ARCH}")
  else()
    set(ARCHS "gfx942-gfx9580")
  endif()

  # Construct key
  set(KEY "rocm-${ROCM_VER}_archs-${ARCHS}_aiter-${AITER_SHA}")
  set(REMOTE_PATH "aiter-prebuilts/${KEY}.tar.gz")
  set(BASE_URL "$ENV{NVTE_PREBUILT_BASE_URL}")
  string(REGEX REPLACE "/$" "" BASE_URL "${BASE_URL}")
  set(ARTIFACT_URL "${BASE_URL}/${REMOTE_PATH}")
  set(SHA_URL "${ARTIFACT_URL}.sha256")

  # Cache
  set(CACHE_ROOT "${TE_ROOT}/.cache/aiter-prebuilts")
  file(MAKE_DIRECTORY "${CACHE_ROOT}")
  set(TAR_PATH "${CACHE_ROOT}/${KEY}.tar.gz")
  set(SHA_PATH "${CACHE_ROOT}/${KEY}/${KEY}.sha256")
  set(EXTRACT_DIR "${CACHE_ROOT}/${KEY}")

  message(STATUS "[AITER-PREBUILT] Constructed key: ${KEY}")

  # if already cached and valid, skip
  if(EXISTS "${EXTRACT_DIR}" AND EXISTS "${TAR_PATH}" AND EXISTS "${SHA_PATH}")
    read_file_trim("${SHA_PATH}" expected_hash)
    check_file_hash("${TAR_PATH}" "${expected_hash}" valid_hash)
    if(valid_hash)
      message(STATUS "[AITER-PREBUILT] ${EXTRACT_DIR} already exists. Skipping download.")
      set(${__AITER_MHA_PATH_VAR} "${EXTRACT_DIR}" PARENT_SCOPE)
      return()
    else()
      file(REMOVE "${TAR_PATH}" "${SHA_PATH}")
      file(REMOVE_RECURSE "${EXTRACT_DIR}")
    endif()
  endif()

  # download .tar.gz and .sha256
  message(STATUS "[AITER-PREBUILT] Download URL: ${SHA_URL}")
  download_with_retries("${SHA_URL}" "${SHA_PATH}" 5)
  if(${SHA_PATH}_NOT_FOUND)
    if(EXISTS "${EXTRACT_DIR}")
      file(REMOVE_RECURSE "${EXTRACT_DIR}")
    endif()
    message(SEND_ERROR "[AITER-PREBUILT] SHA file missing — probably new ROCm/AITER version.")
    return()
  endif()

  message(STATUS "[AITER-PREBUILT] Download URL: ${ARTIFACT_URL}")  
  download_with_retries("${ARTIFACT_URL}" "${TAR_PATH}" 5)

  # Extract
  extract_tarball("${TAR_PATH}" "${CACHE_ROOT}")

  # Set __AITER_MHA_PATH_VAR to extracted path
  set(${__AITER_MHA_PATH_VAR} "${EXTRACT_DIR}" PARENT_SCOPE)
  message(STATUS "[AITER-PREBUILT] Extraction completed at ${EXTRACT_DIR}")
endfunction()
