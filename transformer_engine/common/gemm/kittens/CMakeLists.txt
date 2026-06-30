# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

cmake_minimum_required(VERSION 3.21)

# HipKittens GEMM kernels are split by target architecture into per-device
# folders (a TEMPORARY layout until the hipkittens cdna3/cdna4 submodule
# branches are merged upstream):
#   cdna3/  gfx942: blockwise FP8 GEMM   (3rdparty/hipkittens_cdna3, -DKITTENS_CDNA3)
#   cdna4/  gfx950: blockwise + MXFP8    (3rdparty/hipkittens,       -DKITTENS_CDNA4)
# Each source is added only when its arch is in CMAKE_HIP_ARCHITECTURES, and
# compiled with its own HipKittens include dir + define.
function(try_enable_hipkittens_gemm)
    list(FIND CMAKE_HIP_ARCHITECTURES "gfx942" _gfx942_index)
    list(FIND CMAKE_HIP_ARCHITECTURES "gfx950" _gfx950_index)
    if(_gfx942_index EQUAL -1 AND _gfx950_index EQUAL -1)
        message(STATUS "HipKittens GEMM disabled (neither gfx942 nor gfx950 in CMAKE_HIP_ARCHITECTURES)")
        set(USE_HIPKITTENS_GEMM OFF PARENT_SCOPE)
        return()
    endif()

    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-std=c++20" HAS_CXX20)
    if(NOT HAS_CXX20)
        message(WARNING "HipKittens GEMMs require C++20")
        set(USE_HIPKITTENS_GEMM OFF PARENT_SCOPE)
        return()
    endif()

    set(HIPKITTENS_CDNA3_INCLUDE_DIR
        "${CMAKE_CURRENT_SOURCE_DIR}/../../../../3rdparty/hipkittens_cdna3/include")
    set(HIPKITTENS_CDNA4_INCLUDE_DIR
        "${CMAKE_CURRENT_SOURCE_DIR}/../../../../3rdparty/hipkittens/include")

    # Unified host launcher (pure host, no HipKittens headers) — always built.
    set(kittens_gemm_SOURCES blockwise_fp8_gemm.cpp)
    if(NOT _gfx942_index EQUAL -1)
        if(NOT EXISTS "${HIPKITTENS_CDNA3_INCLUDE_DIR}/kittens.cuh")
            message(FATAL_ERROR
                "Could not find HipKittens (CDNA3) headers at ${HIPKITTENS_CDNA3_INCLUDE_DIR}. "
                "Try running 'git submodule update --init --recursive'.")
        endif()
        list(APPEND kittens_gemm_SOURCES cdna3/blockwise_fp8_gemm.cpp)
    endif()
    if(NOT _gfx950_index EQUAL -1)
        if(NOT EXISTS "${HIPKITTENS_CDNA4_INCLUDE_DIR}/kittens.cuh")
            message(FATAL_ERROR
                "Could not find HipKittens (CDNA4) headers at ${HIPKITTENS_CDNA4_INCLUDE_DIR}. "
                "Try running 'git submodule update --init --recursive'.")
        endif()
        list(APPEND kittens_gemm_SOURCES cdna4/blockwise_fp8_gemm.cpp cdna4/mxfp8_gemm.cpp)
    endif()

    set(CMAKE_CXX_STANDARD 20)
    project(kittens_gemm LANGUAGES HIP CXX)

    add_library(kittens_gemm SHARED ${kittens_gemm_SOURCES})
    set_source_files_properties(${kittens_gemm_SOURCES} PROPERTIES LANGUAGE HIP)

    find_package(hip)
    target_include_directories(kittens_gemm PRIVATE ${HIP_INCLUDE_DIRS})
    target_link_libraries(kittens_gemm PUBLIC hip::host hip::device)

    # Dev-only fast build: NVTE_HK_FAST_BUILD=1 instantiates only one dtype
    # combo (e4m3xe4m3, bf16 out) to cut kittens compile time ~9x.
    set(_kittens_fast_def "")
    if(NOT "$ENV{NVTE_HK_FAST_BUILD}" STREQUAL "" AND NOT "$ENV{NVTE_HK_FAST_BUILD}" STREQUAL "0")
        set(_kittens_fast_def -DNVTE_HK_FAST_BUILD)
        message(STATUS "HipKittens GEMM: FAST dev build (single dtype instance)")
    endif()

    # Per-source include dir + define so each kernel sees its own HipKittens.
    if(NOT _gfx942_index EQUAL -1)
        set_property(SOURCE cdna3/blockwise_fp8_gemm.cpp APPEND PROPERTY
            INCLUDE_DIRECTORIES "${HIPKITTENS_CDNA3_INCLUDE_DIR}")
        set_property(SOURCE cdna3/blockwise_fp8_gemm.cpp APPEND PROPERTY
            COMPILE_OPTIONS -DKITTENS_CDNA3 ${_kittens_fast_def} -fno-gpu-rdc -O3)
        set_source_files_properties(cdna3/blockwise_fp8_gemm.cpp PROPERTIES
            HIP_ARCHITECTURES "gfx942")
    endif()
    if(NOT _gfx950_index EQUAL -1)
        foreach(_cdna4_src cdna4/blockwise_fp8_gemm.cpp cdna4/mxfp8_gemm.cpp)
            set_property(SOURCE ${_cdna4_src} APPEND PROPERTY
                INCLUDE_DIRECTORIES "${HIPKITTENS_CDNA4_INCLUDE_DIR}")
            set_property(SOURCE ${_cdna4_src} APPEND PROPERTY
                COMPILE_OPTIONS -DKITTENS_CDNA4 -fno-gpu-rdc -O3)
            set_source_files_properties(${_cdna4_src} PROPERTIES
                HIP_ARCHITECTURES "gfx950")
        endforeach()
    endif()

    install(TARGETS kittens_gemm
        DESTINATION ${CMAKE_INSTALL_PREFIX}/transformer_engine/lib)
endfunction()

try_enable_hipkittens_gemm()
