# Copyright (c) 2022-2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

include_guard(GLOBAL)

#Determine ROCM_PATH
if(NOT "$ENV{ROCM_PATH}" STREQUAL "")
    set(ROCM_PATH "$ENV{ROCM_PATH}")
else()
    set(_ROCM_SDK_ROOT "")
    find_program(ROCM_SDK_CLI rocm-sdk)
    if(ROCM_SDK_CLI)
        execute_process(
            COMMAND ${ROCM_SDK_CLI} path --root
            OUTPUT_VARIABLE _ROCM_SDK_ROOT
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
    endif()
    if(NOT _ROCM_SDK_ROOT STREQUAL "" AND EXISTS "${_ROCM_SDK_ROOT}/bin/hipcc")
        set(ROCM_PATH "${_ROCM_SDK_ROOT}")
    elseif(EXISTS "/opt/rocm/core")
        set(ROCM_PATH "/opt/rocm/core")
    elseif(EXISTS "/opt/rocm")
        set(ROCM_PATH "/opt/rocm")
    else()
        message(FATAL_ERROR "Could not find ROCm installation")
    endif()
endif()

list(PREPEND CMAKE_PREFIX_PATH "${ROCM_PATH}/lib/cmake")
message(STATUS "ROCM_PATH: ${ROCM_PATH}")

#Configure target GPU architectures
if(NOT DEFINED ENV{NVTE_ROCM_ARCH})
    SET(CMAKE_HIP_ARCHITECTURES gfx942 gfx950)
    # gfx1250 is not in the default set; when building natively on such a machine
    # add it so it is targeted without having to set NVTE_ROCM_ARCH. Skipped
    # silently if rocminfo or a GPU is unavailable.
    find_program(NVTE_ROCMINFO_EXECUTABLE rocminfo HINTS "${ROCM_PATH}/bin")
    if(NVTE_ROCMINFO_EXECUTABLE)
        execute_process(
            COMMAND "${NVTE_ROCMINFO_EXECUTABLE}"
            OUTPUT_VARIABLE _rocminfo_output
            ERROR_QUIET
            RESULT_VARIABLE _rocminfo_result
        )
        if(_rocminfo_result EQUAL 0 AND _rocminfo_output MATCHES "gfx1250")
            message(STATUS "Detected gfx1250; adding to CMAKE_HIP_ARCHITECTURES")
            list(APPEND CMAKE_HIP_ARCHITECTURES gfx1250)
        endif()
    endif()
else()
    # Accept comma separated list for NVTE_ROCM_ARCH
    string(REPLACE "," ";" HIP_ARCH_LIST "$ENV{NVTE_ROCM_ARCH}")
    SET(CMAKE_HIP_ARCHITECTURES ${HIP_ARCH_LIST})
endif()
