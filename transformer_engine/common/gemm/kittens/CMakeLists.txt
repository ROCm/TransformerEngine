# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

cmake_minimum_required(VERSION 3.21)

# gfx1250-strict is gfx1250 A0; it builds the gfx1250 sources.
set(KITTENS_SUPPORTED_ARCHS gfx942 gfx950 gfx1250 gfx1250-strict)

# The gfx1250 kernels need cluster launches and the gfx1250 WMMA-scale/LDS-transpose builtins.
function(kittens_check_gfx1250_toolchain arch result)
    set(CMAKE_HIP_ARCHITECTURES ${arch})
    include(CheckSourceCompiles)
    check_source_compiles(HIP [[
        #include <hip/hip_runtime.h>
        typedef int v2i __attribute__((ext_vector_type(2)));
        typedef int v16i __attribute__((ext_vector_type(16)));
        typedef float v8f __attribute__((ext_vector_type(8)));
        typedef __attribute__((address_space(3))) v2i lds_v2i;
        __global__ __cluster_dims__(2, 1, 1) void k(v8f *c, const v16i *a, lds_v2i *s) {
            v2i t = __builtin_amdgcn_ds_load_tr8_b64_v2i32(s);
            *c = __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(
                0, *a, 0, *a, (short)0, *c, 0, 0, t.x, 0, 0, t.y, false, false);
        }
        int main() { return 0; }]] ${result})
endfunction()
foreach(_arch gfx1250 gfx1250-strict)
    if(_arch IN_LIST CMAKE_HIP_ARCHITECTURES)
        string(MAKE_C_IDENTIFIER "KITTENS_${_arch}_TOOLCHAIN" _toolchain)
        kittens_check_gfx1250_toolchain(${_arch} ${_toolchain})
        if(NOT ${_toolchain})
            message(STATUS "HipKittens GEMM for ${_arch} disabled (HIP compiler lacks ${_arch} support)")
            list(REMOVE_ITEM KITTENS_SUPPORTED_ARCHS ${_arch})
        endif()
    endif()
endforeach()

set(_kittens_enabled_archs "")
foreach(_arch IN LISTS KITTENS_SUPPORTED_ARCHS)
    if(_arch IN_LIST CMAKE_HIP_ARCHITECTURES)
        list(APPEND _kittens_enabled_archs ${_arch})
    endif()
endforeach()

if(NOT _kittens_enabled_archs)
    message(STATUS "HipKittens GEMM disabled (none of ${KITTENS_SUPPORTED_ARCHS} in CMAKE_HIP_ARCHITECTURES)")
    set(USE_HIPKITTENS_GEMM OFF PARENT_SCOPE)
else()
    set(HIPKITTENS_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../3rdparty/hipkittens/include")

    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_HIP_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_HIP_STANDARD_REQUIRED ON)
    project(kittens_gemm LANGUAGES HIP CXX)
    find_package(hip)

    include_directories("${ROCM_PATH}/include/hip")

    set(_kittens_arch_objs "")
    set(_kittens_have_defs "")

    function(kittens_add_arch)
        cmake_parse_arguments(A "" "TAG" "GFX;SOURCES;FLAGS;SOURCE_FLAGS" ${ARGN})
        set(_gfx "")
        foreach(_arch IN LISTS A_GFX)
            if(_arch IN_LIST _kittens_enabled_archs)
                list(APPEND _gfx ${_arch})
            endif()
        endforeach()
        if(NOT _gfx)
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
            HIP_ARCHITECTURES "${_gfx}" POSITION_INDEPENDENT_CODE ON)
        target_include_directories(${_target} PRIVATE
            ${HIP_INCLUDE_DIRS} "${HIPKITTENS_INCLUDE_DIR}")
        target_compile_options(${_target} PRIVATE
            -DKITTENS_${_tag_upper} -fno-gpu-rdc -O3 -Wno-unused-value ${A_FLAGS})
        target_link_libraries(${_target} PRIVATE hip::host hip::device)

        set(_kittens_arch_objs ${_kittens_arch_objs} $<TARGET_OBJECTS:${_target}> PARENT_SCOPE)
        set(_kittens_have_defs ${_kittens_have_defs} KITTENS_HAVE_${_tag_upper} PARENT_SCOPE)
    endfunction()

    kittens_add_arch(TAG cdna3 GFX gfx942
                     SOURCES cdna3/blockwise_fp8_gemm.cpp)

    kittens_add_arch(TAG cdna4 GFX gfx950
                     SOURCES cdna4/blockwise_fp8_gemm.cpp cdna4/mxfp8_gemm.cpp
                             cdna4/overlap/comm_gemm.cpp cdna4/grouped16.cpp cdna4/grouped16_tn.cpp
                             cdna4/grouped16_nn.cpp cdna4/grouped16_nt.cpp
                     FLAGS -gline-tables-only)

    # NVTE_HK_GFX1250_MCAST=0/1 at build time fixes TDM multicast on or off (default: off on A0).
    set(_gfx1250_flags -DKITTENS_CDNA5)
    if(DEFINED ENV{NVTE_HK_GFX1250_MCAST})
        list(APPEND _gfx1250_flags -DNVTE_HK_GFX1250_MCAST=$ENV{NVTE_HK_GFX1250_MCAST})
    endif()
    kittens_add_arch(TAG gfx1250 GFX gfx1250 gfx1250-strict
                     SOURCES gfx1250/gemm.cpp gfx1250/mxfp8_tn.cpp gfx1250/mxfp8_nn.cpp
                             gfx1250/mxfp8_nt.cpp gfx1250/bf16_tn.cpp gfx1250/bf16_nn.cpp
                             gfx1250/bf16_nt.cpp
                     FLAGS ${_gfx1250_flags})

    add_library(kittens_gemm SHARED kittens_common.cpp ${_kittens_arch_objs})
    set_target_properties(kittens_gemm PROPERTIES LINKER_LANGUAGE HIP)
    target_include_directories(kittens_gemm PRIVATE ${HIP_INCLUDE_DIRS})
    target_compile_definitions(kittens_gemm PRIVATE ${_kittens_have_defs})
    target_link_libraries(kittens_gemm PUBLIC hip::host hip::device)
    te_set_rocm_rpath(kittens_gemm "$ORIGIN")

    install(TARGETS kittens_gemm
        DESTINATION ${CMAKE_INSTALL_PREFIX}/transformer_engine/lib)
endif()
