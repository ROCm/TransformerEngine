# This file was modified for portability to AMDGPU
# Copyright (c) 2022-2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

list(APPEND test_cuda_sources
               test_nvrtc.cpp
               test_string.cpp
               ../test_common.cu)
if(USE_CUDA)
    add_executable(test_util ${test_cuda_sources})
else()  
    TE_GetHipifiedSources("${test_cuda_sources}" ${CMAKE_CURRENT_SOURCE_DIR} test_hip_sources)
    TE_AddHipifyDeps("${test_cuda_sources}" ${CMAKE_CURRENT_SOURCE_DIR})
    message("${message_line}")
    message(STATUS "test_util hipified sources: ${test_hip_sources}")

    add_executable(test_util ${test_hip_sources})
endif()

find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)

if(USE_CUDA)
target_link_libraries(test_util PUBLIC CUDA::cudart GTest::gtest_main ${TE_LIB} CUDA::nvrtc CUDNN::cudnn OpenMP::OpenMP_CXX Threads::Threads)
else()
target_link_libraries(test_util PUBLIC GTest::gtest_main ${TE_LIB} OpenMP::OpenMP_CXX ${TE_ROCM_TEST_LIBS} Threads::Threads)
endif()
target_compile_options(test_util PRIVATE -O2 -fopenmp)

include(GoogleTest)
# MALLOC_PERTURB_ makes glibc overwrite freed memory, so a destructor that reads a destroyed object
# at exit crashes the test process, which ctest reports as a failure, instead of passing silently.
gtest_discover_tests(test_util DISCOVERY_TIMEOUT 600 PROPERTIES ENVIRONMENT "MALLOC_PERTURB_=165")
