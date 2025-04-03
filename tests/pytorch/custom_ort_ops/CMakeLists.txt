# This file was modified for portability to AMDGPU
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

cmake_minimum_required(VERSION 3.21)
project(custom_ort_ops LANGUAGES CXX)

option(USE_ROCM "Use ROCm" ON)

# Dependencies
if (USE_ROCM)
    enable_language(HIP)
    find_package(hip REQUIRED)
else()
    find_package(CUDAToolkit REQUIRED)
endif()

set(ONNX_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/onnxruntime/include)
if(NOT EXISTS "${ONNX_INCLUDE_DIR}")
    message(FATAL_ERROR
            "Could not find ONNX Runtime headers. "
            "Please clone https://github.com/microsoft/onnxruntime "
            "into TransformerEngine/tests/pytorch/onnx.")
endif()
include_directories(${ONNX_INCLUDE_DIR})

# Configure library
add_library(custom_ort_ops SHARED custom_op_library.cc)
if (USE_ROCM)
    set_source_files_properties(custom_op_library.cc PROPERTIES LANGUAGE HIP)
    target_link_libraries(custom_ort_ops PUBLIC hip::host)
else()
    target_link_libraries(custom_ort_ops PUBLIC CUDA::cudart)
    target_include_directories(custom_ort_ops PUBLIC
                               ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()
target_include_directories(custom_ort_ops PRIVATE
                           ${ONNX_INCLUDE_DIR}/onnxruntime
                           ${ONNX_INCLUDE_DIR}/onnxruntime/core/session)

# Install library
install(TARGETS custom_ort_ops DESTINATION .)
