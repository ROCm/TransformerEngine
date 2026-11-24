# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

# The fused expert-parallel MoE (nvte_rocm_moe_*, KOSMOS kernels): ROCm gfx950, 8 GPUs. A separate executable, as every
# test forks one process per GPU.
set(test_kosmos_moe_cuda_sources test_kosmos_moe.cu)
add_executable(test_kosmos_moe ${test_kosmos_moe_cuda_sources})

TE_GetHipifiedSources("${test_kosmos_moe_cuda_sources}" ${CMAKE_CURRENT_SOURCE_DIR} test_kosmos_moe_hip_sources)
TE_AddHipifyDeps("${test_kosmos_moe_cuda_sources}" ${CMAKE_CURRENT_SOURCE_DIR})
message(STATUS "test_kosmos_moe hipified sources: ${test_kosmos_moe_hip_sources}")
set_target_properties(test_kosmos_moe PROPERTIES SOURCES "${test_kosmos_moe_hip_sources}")

# kosmos_moe_quantize_mxfp8 (the MXFP8 weights) is called directly from libkosmos_moe.so.
find_library(KOSMOS_MOE_LIB NAMES kosmos_moe
             PATHS "${TE_LIB_PATH}/lib" "${TE_LIB_PATH}/../transformer_engine/lib" ENV TE_LIB_PATH
             NO_DEFAULT_PATH REQUIRED)
message(STATUS "Found kosmos_moe library: ${KOSMOS_MOE_LIB}")

find_package(OpenMP REQUIRED)
target_link_libraries(test_kosmos_moe PUBLIC GTest::gtest_main ${TE_LIB} ${KOSMOS_MOE_LIB} OpenMP::OpenMP_CXX
                      ${TE_ROCM_TEST_LIBS})
target_compile_options(test_kosmos_moe PRIVATE -O2 -fopenmp)

# Listed when ctest runs (not at build time: the binary is not run by the build). Each test forks 8 ranks.
include(GoogleTest)
gtest_discover_tests(test_kosmos_moe DISCOVERY_MODE PRE_TEST DISCOVERY_TIMEOUT 600
                     PROPERTIES TIMEOUT 2400 RUN_SERIAL TRUE)
