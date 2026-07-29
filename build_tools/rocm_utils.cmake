# Copyright (c) 2022-2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

include_guard(GLOBAL)

#Determine ROCM_PATH
if(NOT "$ENV{ROCM_PATH}" STREQUAL "")
    set(ROCM_PATH "$ENV{ROCM_PATH}")
else()
    find_program(ROCM_SDK_CLI rocm-sdk)
    if(ROCM_SDK_CLI)
        execute_process(
            COMMAND ${ROCM_SDK_CLI} path --root
            OUTPUT_VARIABLE _ROCM_SDK_ROOT
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            RESULT_VARIABLE _ROCM_SDK_RESULT
        )
        if(_ROCM_SDK_RESULT EQUAL 0 AND EXISTS "${_ROCM_SDK_ROOT}/bin/hipcc")
            set(ROCM_PATH "${_ROCM_SDK_ROOT}")
        endif()
    endif()
    if(NOT DEFINED ROCM_PATH)
        if(EXISTS "/opt/rocm/core")
            set(ROCM_PATH "/opt/rocm/core")
        else()
            set(ROCM_PATH "/opt/rocm")
        endif()
    endif()
endif()

list(PREPEND CMAKE_PREFIX_PATH "${ROCM_PATH}/lib/cmake")
message(STATUS "ROCM_PATH: ${ROCM_PATH}")

#Configure target GPU architectures
if(NOT DEFINED ENV{NVTE_ROCM_ARCH})
    SET(CMAKE_HIP_ARCHITECTURES gfx942 gfx950)
else()
    # Accept comma separated list for NVTE_ROCM_ARCH
    string(REPLACE "," ";" HIP_ARCH_LIST "$ENV{NVTE_ROCM_ARCH}")
    SET(CMAKE_HIP_ARCHITECTURES ${HIP_ARCH_LIST})
endif()
