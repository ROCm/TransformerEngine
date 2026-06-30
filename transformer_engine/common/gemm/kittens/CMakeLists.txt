# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

cmake_minimum_required(VERSION 3.21)

function(try_enable_hipkittens_gemm)
    list(FIND CMAKE_HIP_ARCHITECTURES "gfx942" _gfx942_index)
    if(_gfx942_index EQUAL -1)
        message(STATUS "HipKittens GEMM disabled (gfx942 not in CMAKE_HIP_ARCHITECTURES)")
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

    set(HIPKITTENS_INCLUDE_DIR
        "${CMAKE_CURRENT_SOURCE_DIR}/../../../../3rdparty/hipkittens_cdna3/include")
    if(NOT EXISTS "${HIPKITTENS_INCLUDE_DIR}/kittens.cuh")
        message(FATAL_ERROR
            "Could not find HipKittens headers at ${HIPKITTENS_INCLUDE_DIR}. "
            "Try running 'git submodule update --init --recursive' "
            "within the Transformer Engine source.")
    endif()

    set(CMAKE_CXX_STANDARD 20)
    project(kittens_gemm LANGUAGES HIP CXX)

    set(kittens_gemm_SOURCES
        blockwise_fp8_gemm.cpp)

    add_library(kittens_gemm SHARED ${kittens_gemm_SOURCES})
    set_source_files_properties(${kittens_gemm_SOURCES} PROPERTIES LANGUAGE HIP)
    set_target_properties(kittens_gemm PROPERTIES HIP_ARCHITECTURES "gfx942")

    set(KITTENS_GEMM_COMPILE_OPTIONS
        -DKITTENS_CDNA3
        -fno-gpu-rdc
        -O3)

    find_package(hip)
    target_include_directories(kittens_gemm PRIVATE ${HIPKITTENS_INCLUDE_DIR})
    target_include_directories(kittens_gemm PRIVATE ${HIP_INCLUDE_DIRS})
    target_link_libraries(kittens_gemm PUBLIC hip::host hip::device)
    target_compile_options(kittens_gemm PRIVATE ${KITTENS_GEMM_COMPILE_OPTIONS})

    install(TARGETS kittens_gemm
        DESTINATION ${CMAKE_INSTALL_PREFIX}/transformer_engine/lib)
endfunction()

try_enable_hipkittens_gemm()
