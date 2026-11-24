# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

# KOSMOS MoE (expert-parallel MoE forward and backward, one launch each) for gfx950: libkosmos_moe.so.

cmake_minimum_required(VERSION 3.21)

if(NOT "gfx950" IN_LIST CMAKE_HIP_ARCHITECTURES)
    message(STATUS "KOSMOS MoE disabled (gfx950 not in CMAKE_HIP_ARCHITECTURES)")
    set(USE_KOSMOS_MOE OFF PARENT_SCOPE)
    return()
endif()

# KOSMOS's own HipKittens pin (3rdparty/hipkittens_kosmos: HazyResearch/HipKittens alexmagro_earlyclobber, as KOSMOS
# pins it), separate from TE's 3rdparty/hipkittens.
set(HIPKITTENS_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/hipkittens_kosmos/include")
if(NOT EXISTS "${HIPKITTENS_INCLUDE_DIR}/kittens.cuh")
    message(FATAL_ERROR "Could not find HipKittens headers at ${HIPKITTENS_INCLUDE_DIR}. "
                        "Try running 'git submodule update --init --recursive'.")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_HIP_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD_REQUIRED ON)
project(kosmos_moe LANGUAGES HIP CXX)
find_package(hip)

# Kernel translation units built once per activation (SwiGLU; -DMOE_ACT_CLAMP10=1).
set(_kmoe_act_tus
    moe_plain moe_ks moe_ks_pin moe_shared moe_shared_bwdg moe_shared_ks moe_shared_ks_pin
    moe_mx moe_mx_ks moe_mx_ks_pin moe_mx_q8 moe_mx_q8_ks moe_mx_q8_ks_pin
    moe_mx_shared moe_mx_shared_bwdg moe_mx_shared_ks moe_mx_shared_ks_pin
    moe_mx_shared_q8 moe_mx_shared_q8_bwdg moe_mx_shared_q8_ks moe_mx_shared_q8_ks_pin)
# Fused-router translation units, built once per activation and per score function (softmax; -DMOE_ROUTER_SCORE=n).
set(_kmoe_router_tus
    moe_router moe_router_ks moe_shared_router moe_shared_router_bwdg moe_shared_router_ks
    moe_mx_router moe_mx_router_ks moe_mx_router_q8 moe_mx_router_q8_ks
    moe_mx_shared_router moe_mx_shared_router_bwdg moe_mx_shared_router_ks
    moe_mx_shared_router_q8 moe_mx_shared_router_q8_bwdg moe_mx_shared_router_q8_ks)

# Backward kernels at the register limit whose over-capacity round loop spills under MachineLICM.
foreach(_tu IN LISTS _kmoe_act_tus _kmoe_router_tus)
    if(_tu MATCHES "^moe_(ks_pin|shared_ks_pin|shared_router|mx)")
        set_source_files_properties(moe/${_tu}.cu PROPERTIES COMPILE_OPTIONS "-mllvm;-disable-machine-licm")
    endif()
endforeach()

set(_kmoe_flags -DKITTENS_CDNA4 -fno-gpu-rdc -O3 -Wno-unused-value -Wno-deprecated-declarations)
set(_kmoe_objs "")

function(kmoe_add_variant)
    cmake_parse_arguments(A "" "NAME" "TUS;DEFS" ${ARGN})
    set(_srcs "")
    foreach(_tu IN LISTS A_TUS)
        list(APPEND _srcs moe/${_tu}.cu)
    endforeach()
    set(_target "kosmos_moe_${A_NAME}")
    add_library(${_target} OBJECT ${_srcs})
    set_source_files_properties(${_srcs} PROPERTIES LANGUAGE HIP)
    set_target_properties(${_target} PROPERTIES HIP_ARCHITECTURES gfx950 POSITION_INDEPENDENT_CODE ON)
    target_include_directories(${_target} PRIVATE ${HIP_INCLUDE_DIRS} "${HIPKITTENS_INCLUDE_DIR}")
    target_compile_definitions(${_target} PRIVATE ${A_DEFS})
    target_compile_options(${_target} PRIVATE ${_kmoe_flags})
    target_link_libraries(${_target} PRIVATE hip::host hip::device)
    set(_kmoe_objs ${_kmoe_objs} $<TARGET_OBJECTS:${_target}> PARENT_SCOPE)
endfunction()

kmoe_add_variant(NAME a0_s0 TUS ${_kmoe_act_tus} ${_kmoe_router_tus})
kmoe_add_variant(NAME a1_s0 TUS ${_kmoe_act_tus} ${_kmoe_router_tus} DEFS MOE_ACT_CLAMP10=1)
foreach(_s 1 2 3)
    kmoe_add_variant(NAME a0_s${_s} TUS ${_kmoe_router_tus} DEFS MOE_ROUTER_SCORE=${_s})
    kmoe_add_variant(NAME a1_s${_s} TUS ${_kmoe_router_tus} DEFS MOE_ACT_CLAMP10=1 MOE_ROUTER_SCORE=${_s})
endforeach()

# The host library: moe.cu (API, planning and launch glue) and the communicator.
kmoe_add_variant(NAME host TUS moe)
add_library(kosmos_moe SHARED utils/kosmos_comm.cpp ${_kmoe_objs})
set_source_files_properties(utils/kosmos_comm.cpp PROPERTIES LANGUAGE CXX)
set_target_properties(kosmos_moe PROPERTIES LINKER_LANGUAGE HIP CXX_VISIBILITY_PRESET default)
target_include_directories(kosmos_moe PRIVATE ${HIP_INCLUDE_DIRS})
target_compile_options(kosmos_moe PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)
target_link_libraries(kosmos_moe PUBLIC hip::host hip::device)
te_set_rocm_rpath(kosmos_moe "$ORIGIN")

install(TARGETS kosmos_moe DESTINATION ${CMAKE_INSTALL_PREFIX}/transformer_engine/lib)
