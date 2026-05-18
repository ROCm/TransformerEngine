cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED CMAKE_CXX_COMPILER)
  set(CMAKE_CXX_COMPILER hipcc)
endif()

include("${CMAKE_CURRENT_SOURCE_DIR}/../../build_tools/rocm_utils.cmake")

project(transformer_engine_benchmarks LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(HIP REQUIRED)

include(FetchContent)
FetchContent_Declare(
  benchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.8.3
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable benchmark tests" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "Disable gtest in benchmark" FORCE)
FetchContent_MakeAvailable(benchmark)

set(TESTS_CPP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../tests/cpp)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../build_tools/hipify/hipify.cmake")
TE_Hipify(${TESTS_CPP_DIR})

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/../../transformer_engine/common/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../../transformer_engine/common
  ${CMAKE_CURRENT_SOURCE_DIR}/../../transformer_engine
  ${CMAKE_CURRENT_SOURCE_DIR}/utils
  ${TESTS_CPP_DIR}
)

set(COMMON_COMPILE_OPTIONS
  -Wall
  -Wextra
  -O3
  -DNDEBUG
  -DUSE_ROCM
)

find_library(TRANSFORMER_ENGINE_LIB
  NAMES transformer_engine
  PATHS ${CMAKE_CURRENT_SOURCE_DIR}/../..
        ${CMAKE_CURRENT_SOURCE_DIR}/../../build/cmake
        ${CMAKE_CURRENT_SOURCE_DIR}/../../build/lib
        /usr/local/lib
        $ENV{HOME}/.local/lib
  NO_DEFAULT_PATH
)

if(NOT TRANSFORMER_ENGINE_LIB)
  message(WARNING "TransformerEngine library not found in expected paths. Trying system paths...")
  find_library(TRANSFORMER_ENGINE_LIB NAMES transformer_engine)
endif()

if(TRANSFORMER_ENGINE_LIB)
  message(STATUS "Found TransformerEngine library: ${TRANSFORMER_ENGINE_LIB}")
else()
  message(FATAL_ERROR "TransformerEngine library not found. Please build TransformerEngine first:\n"
                      "  cd ${CMAKE_CURRENT_SOURCE_DIR}/../..\n"
                      "  pip install -e . --no-build-isolation\n"
                      "Searched paths:\n"
                      "  ${CMAKE_CURRENT_SOURCE_DIR}/../..\n"
                      "  ${CMAKE_CURRENT_SOURCE_DIR}/../../build/cmake\n"
                      "  ${CMAKE_CURRENT_SOURCE_DIR}/../../build/lib")
endif()

function(add_te_benchmark TARGET_NAME SOURCE_FILE)
  add_executable(${TARGET_NAME} ${SOURCE_FILE} ${TESTS_CPP_DIR}/test_common.hip)
  target_compile_options(${TARGET_NAME} PRIVATE ${COMMON_COMPILE_OPTIONS})
  target_compile_definitions(${TARGET_NAME} PRIVATE NVTE_ROCM_BENCHMARK)
  target_link_libraries(${TARGET_NAME} PRIVATE
    benchmark::benchmark
    ${TRANSFORMER_ENGINE_LIB}
    hiprand
  )
endfunction()

add_te_benchmark(bench_quantize_mxfp8_fused cast/bench_quantize_mxfp8_fused.cpp)
add_te_benchmark(bench_dequantize_mxfp8 cast/bench_dequantize_mxfp8.cpp)
add_te_benchmark(bench_gated_mxfp8 cast/bench_gated_mxfp8.cpp)
add_te_benchmark(bench_casttranspose cast/bench_casttranspose.cpp)
add_te_benchmark(bench_normalization normalization/bench_normalization.cpp)
