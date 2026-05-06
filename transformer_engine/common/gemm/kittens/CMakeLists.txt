# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

cmake_minimum_required(VERSION 3.21)
set(CMAKE_CXX_STANDARD 20)
project(kittens_gemm LANGUAGES HIP CXX)

set(HIPKITTENS_INCLUDE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../3rdparty/hipkittens/include")
if(NOT EXISTS "${HIPKITTENS_INCLUDE_DIR}/kittens.cuh")
    message(FATAL_ERROR
        "Could not find HipKittens headers at ${HIPKITTENS_INCLUDE_DIR}. "
        "Try running 'git submodule update --init --recursive' "
        "within the Transformer Engine source.")
endif()

set(kittens_gemm_SOURCES
    mxfp8_gemm.hip)

add_library(kittens_gemm SHARED ${kittens_gemm_SOURCES})
set_target_properties(kittens_gemm PROPERTIES HIP_ARCHITECTURES "gfx950")

set(KITTENS_GEMM_COMPILE_OPTIONS
    -DKITTENS_CDNA4
    -fno-gpu-rdc
    -O3)

find_package(hip)
target_include_directories(kittens_gemm PRIVATE ${HIPKITTENS_INCLUDE_DIR})
target_include_directories(kittens_gemm PRIVATE ${HIP_INCLUDE_DIRS})
target_link_libraries(kittens_gemm PUBLIC hip::host hip::device)
target_compile_options(kittens_gemm PRIVATE ${KITTENS_GEMM_COMPILE_OPTIONS})

install(TARGETS kittens_gemm
    DESTINATION ${CMAKE_INSTALL_PREFIX}/transformer_engine/lib)
