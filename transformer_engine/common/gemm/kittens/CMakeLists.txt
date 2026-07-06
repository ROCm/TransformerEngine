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

    set(CMAKE_CXX_STANDARD 20)
    project(kittens_gemm LANGUAGES HIP CXX)
    find_package(hip)

    # Arch-specific kernels are compiled as separate OBJECT libraries, each
    # pinned to its own GPU arch via the per-TARGET HIP_ARCHITECTURES property
    # (per-SOURCE HIP_ARCHITECTURES is ignored by CMake's HIP language, which is
    # why cdna3 was previously getting gfx950 code objects too). The final
    # shared library links the host launcher + whichever arch objects apply.
    set(_kittens_arch_objs "")

    if(NOT _gfx942_index EQUAL -1)
        if(NOT EXISTS "${HIPKITTENS_CDNA3_INCLUDE_DIR}/kittens.cuh")
            message(FATAL_ERROR
                "Could not find HipKittens (CDNA3) headers at ${HIPKITTENS_CDNA3_INCLUDE_DIR}. "
                "Try running 'git submodule update --init --recursive'.")
        endif()
        add_library(kittens_gemm_cdna3 OBJECT cdna3/blockwise_fp8_gemm.cpp)
        set_source_files_properties(cdna3/blockwise_fp8_gemm.cpp PROPERTIES LANGUAGE HIP)
        set_target_properties(kittens_gemm_cdna3 PROPERTIES
            HIP_ARCHITECTURES "gfx942" POSITION_INDEPENDENT_CODE ON)
        target_include_directories(kittens_gemm_cdna3 PRIVATE
            ${HIP_INCLUDE_DIRS} "${HIPKITTENS_CDNA3_INCLUDE_DIR}")
        target_compile_options(kittens_gemm_cdna3 PRIVATE
            -DKITTENS_CDNA3 -fno-gpu-rdc -O3)
        target_link_libraries(kittens_gemm_cdna3 PRIVATE hip::host hip::device)
        list(APPEND _kittens_arch_objs $<TARGET_OBJECTS:kittens_gemm_cdna3>)
    endif()

    if(NOT _gfx950_index EQUAL -1)
        if(NOT EXISTS "${HIPKITTENS_CDNA4_INCLUDE_DIR}/kittens.cuh")
            message(FATAL_ERROR
                "Could not find HipKittens (CDNA4) headers at ${HIPKITTENS_CDNA4_INCLUDE_DIR}. "
                "Try running 'git submodule update --init --recursive'.")
        endif()
        add_library(kittens_gemm_cdna4 OBJECT cdna4/blockwise_fp8_gemm.cpp cdna4/mxfp8_gemm.cpp)
        set_source_files_properties(cdna4/blockwise_fp8_gemm.cpp cdna4/mxfp8_gemm.cpp
            PROPERTIES LANGUAGE HIP)
        set_target_properties(kittens_gemm_cdna4 PROPERTIES
            HIP_ARCHITECTURES "gfx950" POSITION_INDEPENDENT_CODE ON)
        target_include_directories(kittens_gemm_cdna4 PRIVATE
            ${HIP_INCLUDE_DIRS} "${HIPKITTENS_CDNA4_INCLUDE_DIR}")
        target_compile_options(kittens_gemm_cdna4 PRIVATE
            -DKITTENS_CDNA4 -fno-gpu-rdc -O3)
        target_link_libraries(kittens_gemm_cdna4 PRIVATE hip::host hip::device)
        list(APPEND _kittens_arch_objs $<TARGET_OBJECTS:kittens_gemm_cdna4>)
    endif()

    # Unified host launcher (pure host, no HipKittens headers) + arch objects.
    add_library(kittens_gemm SHARED blockwise_fp8_gemm.cpp ${_kittens_arch_objs})
    set_source_files_properties(blockwise_fp8_gemm.cpp PROPERTIES LANGUAGE HIP)
    target_include_directories(kittens_gemm PRIVATE ${HIP_INCLUDE_DIRS})
    target_link_libraries(kittens_gemm PUBLIC hip::host hip::device)

    install(TARGETS kittens_gemm
        DESTINATION ${CMAKE_INSTALL_PREFIX}/transformer_engine/lib)
endfunction()

try_enable_hipkittens_gemm()
