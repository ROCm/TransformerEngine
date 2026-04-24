# Copyright (c) 2022-2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

#Determine ROCM_PATH
if(NOT "$ENV{ROCM_PATH}" STREQUAL "")
    set(ROCM_PATH "$ENV{ROCM_PATH}")
elseif(EXISTS "/opt/rocm/core")
    set(ROCM_PATH "/opt/rocm/core")
else()
    set(ROCM_PATH "/opt/rocm")
endif()

#Configure target GPU architectures
if(NOT DEFINED ENV{NVTE_ROCM_ARCH})
    SET(CMAKE_HIP_ARCHITECTURES gfx942 gfx950)
else()
    # Accept comma separated list for NVTE_ROCM_ARCH
    string(REPLACE "," ";" HIP_ARCH_LIST "$ENV{NVTE_ROCM_ARCH}")
    SET(CMAKE_HIP_ARCHITECTURES ${HIP_ARCH_LIST})
endif()
