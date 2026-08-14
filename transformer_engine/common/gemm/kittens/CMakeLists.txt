# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

cmake_minimum_required(VERSION 3.21)

set(KITTENS_SUPPORTED_ARCHS gfx942 gfx950)

set(_kittens_enabled_archs "")
foreach(_arch IN LISTS KITTENS_SUPPORTED_ARCHS)
    if(_arch IN_LIST CMAKE_HIP_ARCHITECTURES)
        list(APPEND _kittens_enabled_archs ${_arch})
    endif()
endforeach()

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++20" HAS_CXX20)

if(NOT _kittens_enabled_archs)
    message(STATUS "HipKittens GEMM disabled (none of ${KITTENS_SUPPORTED_ARCHS} in CMAKE_HIP_ARCHITECTURES)")
    set(USE_HIPKITTENS_GEMM OFF PARENT_SCOPE)
elseif(NOT HAS_CXX20)
    message(WARNING "HipKittens GEMMs require C++20")
    set(USE_HIPKITTENS_GEMM OFF PARENT_SCOPE)
else()
    set(HIPKITTENS_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../3rdparty/hipkittens/include")

    set(CMAKE_CXX_STANDARD 20)
    project(kittens_gemm LANGUAGES HIP CXX)
    find_package(hip)

    include_directories("${ROCM_PATH}/include/hip")

    set(_kittens_arch_objs "")
    set(_kittens_have_defs "")

    function(kittens_add_arch)
        cmake_parse_arguments(A "" "TAG;GFX" "SOURCES;FLAGS;SOURCE_FLAGS" ${ARGN})
        if(NOT "${A_GFX}" IN_LIST _kittens_enabled_archs)
            return()
        endif()
        if(NOT EXISTS "${HIPKITTENS_INCLUDE_DIR}/kittens.cuh")
            message(FATAL_ERROR
                "Could not find HipKittens headers at ${HIPKITTENS_INCLUDE_DIR}. "
                "Try running 'git submodule update --init --recursive'.")
        endif()

        string(TOUPPER "${A_TAG}" _tag_upper)
        set(_target "kittens_gemm_${A_TAG}")

        add_library(${_target} OBJECT ${A_SOURCES})
        set_source_files_properties(${A_SOURCES} PROPERTIES LANGUAGE HIP)

        # SOURCE_FLAGS is a flat <source> <flags> list
        list(LENGTH A_SOURCE_FLAGS _sf_len)
        if(_sf_len GREATER 0)
            math(EXPR _sf_last "${_sf_len} - 2")
            foreach(_i RANGE 0 ${_sf_last} 2)
                math(EXPR _j "${_i} + 1")
                list(GET A_SOURCE_FLAGS ${_i} _sf_src)
                list(GET A_SOURCE_FLAGS ${_j} _sf_opts)
                if(NOT "${_sf_src}" IN_LIST A_SOURCES)
                    message(FATAL_ERROR
                        "kittens_add_arch(TAG ${A_TAG}): SOURCE_FLAGS names ${_sf_src}, "
                        "which is not in SOURCES.")
                endif()
                set_source_files_properties(${_sf_src} PROPERTIES COMPILE_FLAGS "${_sf_opts}")
            endforeach()
        endif()

        set_target_properties(${_target} PROPERTIES
            HIP_ARCHITECTURES "${A_GFX}" POSITION_INDEPENDENT_CODE ON)
        target_include_directories(${_target} PRIVATE
            ${HIP_INCLUDE_DIRS} "${HIPKITTENS_INCLUDE_DIR}")
        target_compile_options(${_target} PRIVATE
            -DKITTENS_${_tag_upper} -fno-gpu-rdc -O3 ${A_FLAGS})
        target_link_libraries(${_target} PRIVATE hip::host hip::device)

        set(_kittens_arch_objs ${_kittens_arch_objs} $<TARGET_OBJECTS:${_target}> PARENT_SCOPE)
        set(_kittens_have_defs ${_kittens_have_defs} KITTENS_HAVE_${_tag_upper} PARENT_SCOPE)
    endfunction()

    kittens_add_arch(TAG cdna3 GFX gfx942
                     SOURCES cdna3/blockwise_fp8_gemm.cpp)

    # -ffast-math folds fpext(fptrunc(x)) back to x, so any deliberate narrow-then-widen
    # in blockwise_fp8_gemm.cpp has to be made opaque to the optimizer -- see
    # round_to_out_dtype() in cdna4/blockwise_fp8_gemm_helper.cuh.
    kittens_add_arch(TAG cdna4 GFX gfx950
                     SOURCES cdna4/blockwise_fp8_gemm.cpp cdna4/mxfp8_gemm.cpp
                     FLAGS -gline-tables-only
                     SOURCE_FLAGS cdna4/blockwise_fp8_gemm.cpp "-ffast-math")

    add_library(kittens_gemm SHARED kittens_common.cpp ${_kittens_arch_objs})
    set_target_properties(kittens_gemm PROPERTIES LINKER_LANGUAGE HIP)
    target_include_directories(kittens_gemm PRIVATE ${HIP_INCLUDE_DIRS})
    target_compile_definitions(kittens_gemm PRIVATE ${_kittens_have_defs})
    target_link_libraries(kittens_gemm PUBLIC hip::host hip::device)

    install(TARGETS kittens_gemm
        DESTINATION ${CMAKE_INSTALL_PREFIX}/transformer_engine/lib)
endif()
