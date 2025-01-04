# This file was modified for portability to AMDGPU
# Copyright (c) 2022-2024, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

if(USE_CUDA)
add_executable(test_util
               test_nvrtc.cpp
               test_string.cpp
               ../test_common.cu)

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
