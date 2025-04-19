# This file was modified for portability to AMDGPU
# Copyright (c) 2022-2025, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

if(USE_CUDA)
add_executable(test_util
               test_nvrtc.cpp
               test_string.cpp
               ../test_common.cu)

<<<<<<< HEAD
target_link_libraries(test_util PUBLIC CUDA::cudart GTest::gtest_main ${TE_LIB} CUDA::nvrtc CUDNN::cudnn)
else()
add_executable(test_util
               test_nvrtc_hip.cpp
               test_string.cpp
               ../test_common.hip)

target_link_libraries(test_util PUBLIC hip::host hip::device GTest::gtest_main ${TE_LIB})
endif()

target_compile_options(test_util PRIVATE -O2)

include(GoogleTest)
gtest_discover_tests(test_util PROPERTIES DISCOVERY_TIMEOUT 600)
=======

find_package(OpenMP REQUIRED)
target_link_libraries(test_util PUBLIC CUDA::cudart GTest::gtest_main ${TE_LIB} CUDA::nvrtc CUDNN::cudnn  OpenMP::OpenMP_CXX)
target_compile_options(test_util PRIVATE -O2 -fopenmp)

include(GoogleTest)
gtest_discover_tests(test_util DISCOVERY_TIMEOUT 600)
>>>>>>> af7b2b44dd6173c9b3049f306c0773a938feceae
