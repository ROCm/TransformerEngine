cmake_minimum_required(VERSION 3.21)

# Declare project with both CXX and HIP languages.
# Requires hip-lang CMake package (available under ${ROCM_PATH}/lib/cmake).
project(crossattn_hip_kernel LANGUAGES CXX HIP)

# ---------------------------------------------------------------------------
# ROCm / HIP setup
# ---------------------------------------------------------------------------

if(NOT DEFINED ROCM_PATH)
    set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to ROCm installation")
endif()

list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}/lib/cmake")
find_package(hip REQUIRED CONFIG)

# GPU architecture — override with -DGPU_TARGETS=gfx906 etc.
# set(GPU_TARGETS "gfx950" CACHE STRING "GPU architecture targets")
set(GPU_TARGETS "gfx942" CACHE STRING "GPU architecture targets")
set(CMAKE_HIP_ARCHITECTURES "${GPU_TARGETS}")

# ---------------------------------------------------------------------------
# Language standards
# ---------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Kernel headers
#
# The MFMA kernel headers (attn_*.h) are vendored once for the Transformer
# Engine build under transformer_engine/common/fused_attn_rocm/small_seq_kernels.
# These reference tests include them directly from that canonical location
# instead of keeping a duplicate copy here.
# ---------------------------------------------------------------------------

set(KERNEL_INCLUDE_DIR
    "${CMAKE_SOURCE_DIR}/../../../transformer_engine/common/fused_attn_rocm/small_seq_kernels"
    CACHE PATH "Path to the vendored small-seq MFMA kernel headers")

# ---------------------------------------------------------------------------
# CPU reference library
#
# Compiled as plain C++ (no HIP device code) and linked by both test
# executables. Avoids duplicating CPU reference compilation.
# ---------------------------------------------------------------------------

add_library(attn_ref STATIC
    ref/attn_fwd_ref.cpp
    ref/attn_bwd_ref.cpp
)
# Mark as HIP so clang++ handles hip_bfloat16 conversions (float<->bfloat16)
# correctly. No device kernels are compiled; clang just enables the host-side
# HIP type support.
set_source_files_properties(
    ref/attn_fwd_ref.cpp
    ref/attn_bwd_ref.cpp
    PROPERTIES LANGUAGE HIP
)
target_include_directories(attn_ref PUBLIC
    ${KERNEL_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/ref
)
target_compile_options(attn_ref PRIVATE --offload-arch=${GPU_TARGETS})
target_link_libraries(attn_ref PUBLIC hip::host)

# ---------------------------------------------------------------------------
# test_fwd executable
#
# test_fwd.cpp includes attn_fwd.h which contains __global__ kernels.
# Mark it as LANGUAGE HIP so clang++ uses -x hip and sees <<< >>> syntax.
# ---------------------------------------------------------------------------

add_executable(test_fwd tests/test_fwd.cpp)
set_source_files_properties(tests/test_fwd.cpp PROPERTIES LANGUAGE HIP)

target_include_directories(test_fwd PRIVATE
    ${KERNEL_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/ref
    ${CMAKE_SOURCE_DIR}/tests
)
target_link_libraries(test_fwd PRIVATE attn_ref hip::host)
target_compile_options(test_fwd PRIVATE -O3 --offload-arch=${GPU_TARGETS})

# ---------------------------------------------------------------------------
# test_bwd executable
# ---------------------------------------------------------------------------

add_executable(test_bwd tests/test_bwd.cpp)
set_source_files_properties(tests/test_bwd.cpp PROPERTIES LANGUAGE HIP)

target_include_directories(test_bwd PRIVATE
    ${KERNEL_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/ref
    ${CMAKE_SOURCE_DIR}/tests
)
target_link_libraries(test_bwd PRIVATE attn_ref hip::host)
target_compile_options(test_bwd PRIVATE -O3 --offload-arch=${GPU_TARGETS})

# ---------------------------------------------------------------------------
# test_fwd_mfma executable
#
# Tests the fused MFMA forward kernel (attn_fwd_mfma.h).
# ---------------------------------------------------------------------------

add_executable(test_fwd_mfma tests/test_fwd_mfma.cpp)
set_source_files_properties(tests/test_fwd_mfma.cpp PROPERTIES LANGUAGE HIP)

target_include_directories(test_fwd_mfma PRIVATE
    ${KERNEL_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/ref
    ${CMAKE_SOURCE_DIR}/tests
)
target_link_libraries(test_fwd_mfma PRIVATE attn_ref hip::host)
target_compile_options(test_fwd_mfma PRIVATE -O3 --offload-arch=${GPU_TARGETS})

# ---------------------------------------------------------------------------
# test_fwd_mfma_16x16 executable
#
# Tests the fused MFMA 16x16x16 forward kernel (attn_fwd_mfma_16x16.h).
# ---------------------------------------------------------------------------

add_executable(test_fwd_mfma_16x16 tests/test_fwd_mfma_16x16.cpp)
set_source_files_properties(tests/test_fwd_mfma_16x16.cpp PROPERTIES LANGUAGE HIP)

target_include_directories(test_fwd_mfma_16x16 PRIVATE
    ${KERNEL_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/ref
    ${CMAKE_SOURCE_DIR}/tests
)
target_link_libraries(test_fwd_mfma_16x16 PRIVATE attn_ref hip::host)
target_compile_options(test_fwd_mfma_16x16 PRIVATE -O3 --offload-arch=${GPU_TARGETS})

# ---------------------------------------------------------------------------
# test_mfma_head_dims executable
#
# Forward MFMA 16x16 correctness for head_dim 128, 256, 512 (small config).
# ---------------------------------------------------------------------------

add_executable(test_mfma_head_dims tests/test_mfma_head_dims.cpp)
set_source_files_properties(tests/test_mfma_head_dims.cpp PROPERTIES LANGUAGE HIP)

target_include_directories(test_mfma_head_dims PRIVATE
    ${KERNEL_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/ref
    ${CMAKE_SOURCE_DIR}/tests
)
target_link_libraries(test_mfma_head_dims PRIVATE attn_ref hip::host)
target_compile_options(test_mfma_head_dims PRIVATE -O3 --offload-arch=${GPU_TARGETS})

# ---------------------------------------------------------------------------
# test_fwd_mfma_multiq executable
#
# Tests multi-Q dispatch across 4x4x4 and 16x16x16 MFMA kernels.
# ---------------------------------------------------------------------------

add_executable(test_fwd_mfma_multiq tests/test_fwd_mfma_multiq.cpp)
set_source_files_properties(tests/test_fwd_mfma_multiq.cpp PROPERTIES LANGUAGE HIP)

target_include_directories(test_fwd_mfma_multiq PRIVATE
    ${KERNEL_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/ref
    ${CMAKE_SOURCE_DIR}/tests
)
target_link_libraries(test_fwd_mfma_multiq PRIVATE attn_ref hip::host)
target_compile_options(test_fwd_mfma_multiq PRIVATE -O3 --offload-arch=${GPU_TARGETS})

# ---------------------------------------------------------------------------
# test_bwd_mfma_16x16 executable
#
# Tests the MFMA 16x16x16 backward kernels (attn_bwd_mfma_16x16.h).
# ---------------------------------------------------------------------------

add_executable(test_bwd_mfma_16x16 tests/test_bwd_mfma_16x16.cpp)
set_source_files_properties(tests/test_bwd_mfma_16x16.cpp PROPERTIES LANGUAGE HIP)

target_include_directories(test_bwd_mfma_16x16 PRIVATE
    ${KERNEL_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/ref
    ${CMAKE_SOURCE_DIR}/tests
)
target_link_libraries(test_bwd_mfma_16x16 PRIVATE attn_ref hip::host)
target_compile_options(test_bwd_mfma_16x16 PRIVATE -O3 --offload-arch=${GPU_TARGETS})

# ---------------------------------------------------------------------------
# test_varlen_mfma_16x16 executable
#
# Unified varlen test for MFMA 16x16x16 forward + backward kernels.
# ---------------------------------------------------------------------------

add_executable(test_varlen_mfma_16x16 tests/test_varlen_mfma_16x16.cpp)
set_source_files_properties(tests/test_varlen_mfma_16x16.cpp PROPERTIES LANGUAGE HIP)

target_include_directories(test_varlen_mfma_16x16 PRIVATE
    ${KERNEL_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/ref
    ${CMAKE_SOURCE_DIR}/tests
)
target_link_libraries(test_varlen_mfma_16x16 PRIVATE attn_ref hip::host)
target_compile_options(test_varlen_mfma_16x16 PRIVATE -O3 --offload-arch=${GPU_TARGETS})

# ---------------------------------------------------------------------------
# test_small_seq_sweep executable
#
# Small-sequence sweep benchmark (seqlen 1..17, bs=2048, fwd+bwd, TE format).
# ---------------------------------------------------------------------------

add_executable(test_small_seq_sweep tests/test_small_seq_sweep.cpp)
set_source_files_properties(tests/test_small_seq_sweep.cpp PROPERTIES LANGUAGE HIP)

target_include_directories(test_small_seq_sweep PRIVATE
    ${KERNEL_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/ref
    ${CMAKE_SOURCE_DIR}/tests
)
target_link_libraries(test_small_seq_sweep PRIVATE attn_ref hip::host)
target_compile_options(test_small_seq_sweep PRIVATE -O3 --offload-arch=${GPU_TARGETS})
