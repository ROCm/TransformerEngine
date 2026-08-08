# This file was modified for portability to AMDGPU
# Copyright (c) 2022-2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

cmake_minimum_required(VERSION 3.21)

option(USE_ROCM "Use ROCm" ON)
option(USE_FUSED_ATTN_AOTRITON "Use aotriton backend" ON)
option(USE_FUSED_ATTN_CK "Use ck backend" ON)
option(USE_HIPKITTENS_GEMM "Use HipKittens blockwise FP8 and MXFP8 GEMM kernels" ON)
set(USE_CUDA OFF)

if (USE_ROCM)
  include("${CMAKE_CURRENT_SOURCE_DIR}/../../build_tools/rocm_utils.cmake")
else()
  #Check if CUDA exists
  if (EXISTS "$ENV{CUDA_HOME}/bin/nvcc")
    set(USE_ROCM OFF)
    set(USE_CUDA ON)
    message("Using NVIDIA Platform")
  else()
    message(FATAL_ERROR "Could not find NVCC at '$ENV{CUDA_HOME}/bin/nvcc'")
  endif()

endif()

# Language options
if(USE_CUDA)
# Removed indent to minimize code diff with NV upstream
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G")
endif()

# Hide non-necessary symbols in shared object.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/libtransformer_engine.version")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/libtransformer_engine.version")

# Transformer Engine library
project(transformer_engine LANGUAGES CUDA CXX)

# CUDA Toolkit
find_package(CUDAToolkit REQUIRED)
if (CUDAToolkit_VERSION VERSION_LESS 12.1)
  message(FATAL_ERROR "CUDA 12.1+ is required, but found CUDA ${CUDAToolkit_VERSION}")
endif()

# Process GPU architectures
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL 13.0)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 89 90 100 120)
  elseif (CUDAToolkit_VERSION VERSION_GREATER_EQUAL 12.8)
    set(CMAKE_CUDA_ARCHITECTURES 70 80 89 90 100 120)
  else ()
    set(CMAKE_CUDA_ARCHITECTURES 70 80 89 90)
  endif()
endif()

# Process CMAKE_CUDA_ARCHITECTURES to separate standard, generic, and specific architectures.
# - NVTE_STANDARD_ARCHS: pre-Blackwell archs (e.g. 75, 80, 89, 90). Applied to all CUDA sources.
# - NVTE_GENERIC_ARCHS: Blackwell family heads (e.g. 100, 120). Applied to non-arch-specific sources only.
# - NVTE_SPECIFIC_ARCHS: Blackwell specific targets (e.g. 100a, 120f). Applied to arch-specific sources only.
set(NVTE_STANDARD_ARCHS)
set(NVTE_GENERIC_ARCHS)
set(NVTE_SPECIFIC_ARCHS)

# Check for architecture 100
list(FIND CMAKE_CUDA_ARCHITECTURES "100" arch_100_index)
if(NOT arch_100_index EQUAL -1)
  list(REMOVE_ITEM CMAKE_CUDA_ARCHITECTURES "100")
  list(APPEND NVTE_GENERIC_ARCHS "100")
  list(APPEND NVTE_SPECIFIC_ARCHS "100a")
  if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 12.9)
    list(APPEND NVTE_SPECIFIC_ARCHS "103a")
  endif()
endif()

# Check for architecture 101 (if we see this we are in toolkit <= 12.9)
list(FIND CMAKE_CUDA_ARCHITECTURES "101" arch_101_index)
if(NOT arch_101_index EQUAL -1)
  list(REMOVE_ITEM CMAKE_CUDA_ARCHITECTURES "101")
  list(APPEND NVTE_GENERIC_ARCHS "101")
  list(APPEND NVTE_SPECIFIC_ARCHS "101a")
endif()

# Check for architecture 110 (if we see this we are in toolkit >= 13.0)
list(FIND CMAKE_CUDA_ARCHITECTURES "110" arch_110_index)
if(NOT arch_110_index EQUAL -1)
  list(REMOVE_ITEM CMAKE_CUDA_ARCHITECTURES "110")
  list(APPEND NVTE_GENERIC_ARCHS "110")
  list(APPEND NVTE_SPECIFIC_ARCHS "110f")
endif()

# Check for architecture 120
list(FIND CMAKE_CUDA_ARCHITECTURES "120" arch_120_index)
if(NOT arch_120_index EQUAL -1)
  list(REMOVE_ITEM CMAKE_CUDA_ARCHITECTURES "120")
  list(APPEND NVTE_GENERIC_ARCHS "120")
  if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 12.9)
    list(APPEND NVTE_SPECIFIC_ARCHS "120f")
  else()
    list(APPEND NVTE_SPECIFIC_ARCHS "120a")
  endif()
endif()

# Move remaining standard (pre-Blackwell) architectures into NVTE_STANDARD_ARCHS.
# These are applied to all CUDA sources (both generic and arch-specific).
set(NVTE_STANDARD_ARCHS ${CMAKE_CUDA_ARCHITECTURES})

# cuDNN frontend API
set(CUDNN_FRONTEND_INCLUDE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/cudnn-frontend/include")
if(NOT EXISTS "${CUDNN_FRONTEND_INCLUDE_DIR}")
    message(FATAL_ERROR
            "Could not find cuDNN frontend API at ${CUDNN_FRONTEND_INCLUDE_DIR}. "
            "Try running 'git submodule update --init --recursive' "
            "within the Transformer Engine source.")
endif()
include(${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/cudnn-frontend/cmake/cuDNN.cmake)

set(CUTLASS_INCLUDE_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/cutlass/include")
set(CUTLASS_TOOLS_INCLUDE_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/cutlass/tools/util/include")

else()
  set(CMAKE_CXX_STANDARD 17)
  #define GPU ARCH before enabling HIP to avoid potential failures with arch detection
  project(transformer_engine LANGUAGES HIP CXX)

  # If using ROCm, ninja must be installed to build TE.
  if (NOT CMAKE_GENERATOR STREQUAL "Ninja")
    message(FATAL_ERROR "This project requires the Ninja build system. Install it using 'pip install ninja'.")
  endif()

  # Disable Asserts In Code (Can't use asserts on HIP stack.)
  add_definitions(-DNDEBUG)
  add_definitions(-DUSE_ROCM)

  # build error will be dup-ed parallel-jobs times
  # set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -parallel-jobs=4")
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -g")
  endif()
endif()

set(message_line "-------------------------------------------------------------")
message("${message_line}")
message(STATUS "USE_ROCM ${USE_ROCM}")
if(USE_ROCM)
  message(STATUS "CMAKE_HIP_ARCHITECTURES: ${CMAKE_HIP_ARCHITECTURES}")
  message(STATUS "USE_FUSED_ATTN_CK ${USE_FUSED_ATTN_CK}")
  message(STATUS "USE_FUSED_ATTN_AOTRITON ${USE_FUSED_ATTN_AOTRITON}")
endif()

# Python
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

function(find_nccl_version OUT_VERSION OUT_INCLUDE_DIR)
  find_path(_nvte_nccl_include_dir
            NAMES nccl.h
            PATH_SUFFIXES include
            REQUIRED)

  file(STRINGS "${_nvte_nccl_include_dir}/nccl.h" _nvte_nccl_major_line
       REGEX "^#define NCCL_MAJOR[ \t]+[0-9]+$")
  file(STRINGS "${_nvte_nccl_include_dir}/nccl.h" _nvte_nccl_minor_line
       REGEX "^#define NCCL_MINOR[ \t]+[0-9]+$")
  file(STRINGS "${_nvte_nccl_include_dir}/nccl.h" _nvte_nccl_patch_line
       REGEX "^#define NCCL_PATCH[ \t]+[0-9]+$")

  string(REGEX REPLACE "^#define NCCL_MAJOR[ \t]+([0-9]+)$" "\\1"
         _nvte_nccl_major "${_nvte_nccl_major_line}")
  string(REGEX REPLACE "^#define NCCL_MINOR[ \t]+([0-9]+)$" "\\1"
         _nvte_nccl_minor "${_nvte_nccl_minor_line}")
  string(REGEX REPLACE "^#define NCCL_PATCH[ \t]+([0-9]+)$" "\\1"
         _nvte_nccl_patch "${_nvte_nccl_patch_line}")

  if ("${_nvte_nccl_major}" STREQUAL ""
      OR "${_nvte_nccl_minor}" STREQUAL ""
      OR "${_nvte_nccl_patch}" STREQUAL "")
    message(FATAL_ERROR
            "Failed to parse NCCL version from ${_nvte_nccl_include_dir}/nccl.h")
  endif()

  set(${OUT_VERSION}
      "${_nvte_nccl_major}.${_nvte_nccl_minor}.${_nvte_nccl_patch}"
      PARENT_SCOPE)
  set(${OUT_INCLUDE_DIR} "${_nvte_nccl_include_dir}" PARENT_SCOPE)
endfunction()

function(find_cublasmp_version OUT_VERSION OUT_INCLUDE_DIR SEARCH_DIR)
  find_path(_nvte_cublasmp_include_dir
            NAMES cublasmp.h
            HINTS "${SEARCH_DIR}/include"
            PATH_SUFFIXES include
            REQUIRED)

  file(STRINGS "${_nvte_cublasmp_include_dir}/cublasmp.h" _nvte_cublasmp_major_line
       REGEX "^#define CUBLASMP_VER_MAJOR[ \t]+[0-9]+$")
  file(STRINGS "${_nvte_cublasmp_include_dir}/cublasmp.h" _nvte_cublasmp_minor_line
       REGEX "^#define CUBLASMP_VER_MINOR[ \t]+[0-9]+$")
  file(STRINGS "${_nvte_cublasmp_include_dir}/cublasmp.h" _nvte_cublasmp_patch_line
       REGEX "^#define CUBLASMP_VER_PATCH[ \t]+[0-9]+$")

  string(REGEX REPLACE "^#define CUBLASMP_VER_MAJOR[ \t]+([0-9]+)$" "\\1"
         _nvte_cublasmp_major "${_nvte_cublasmp_major_line}")
  string(REGEX REPLACE "^#define CUBLASMP_VER_MINOR[ \t]+([0-9]+)$" "\\1"
         _nvte_cublasmp_minor "${_nvte_cublasmp_minor_line}")
  string(REGEX REPLACE "^#define CUBLASMP_VER_PATCH[ \t]+([0-9]+)$" "\\1"
         _nvte_cublasmp_patch "${_nvte_cublasmp_patch_line}")

  if ("${_nvte_cublasmp_major}" STREQUAL ""
      OR "${_nvte_cublasmp_minor}" STREQUAL ""
      OR "${_nvte_cublasmp_patch}" STREQUAL "")
    message(FATAL_ERROR
            "Failed to parse cuBLASMp version from ${_nvte_cublasmp_include_dir}/cublasmp.h")
  endif()

  set(${OUT_VERSION}
      "${_nvte_cublasmp_major}.${_nvte_cublasmp_minor}.${_nvte_cublasmp_patch}"
      PARENT_SCOPE)
  set(${OUT_INCLUDE_DIR} "${_nvte_cublasmp_include_dir}" PARENT_SCOPE)
endfunction()

# Configure Transformer Engine library
include_directories(${PROJECT_SOURCE_DIR}/..)
set(transformer_engine_SOURCES)
set(transformer_engine_cpp_sources)
set(transformer_engine_cuda_sources)
set(transformer_engine_cuda_arch_specific_sources)

# Source files in both cuda and rocm
list(APPEND transformer_engine_cpp_sources
     cudnn_utils.cpp #CUDA-only
     transformer_engine.cpp
     fused_attn/fused_attn.cpp #CUDA-only
     gemm/config.cpp
     normalization/common.cpp
     normalization/layernorm/ln_api.cpp
     normalization/rmsnorm/rmsnorm_api.cpp
     util/cuda_driver.cpp
     util/cuda_nvml.cpp #CUDA-only
     util/cuda_runtime.cpp
     util/multi_stream.cpp
     util/rtc.cpp
     comm_gemm/comm_gemm.cpp
     comm_gemm_overlap/userbuffers/ipcsocket.cc
     comm_gemm_overlap/userbuffers/userbuffers-host.cpp
     comm_gemm_overlap/comm_gemm_overlap.cpp
     newton_schulz/newton_schulz.cpp #CUDA-only
     )

# Sources that only apply to CUDA builds (removed when building for ROCm)
set(cuda_only_cpp_sources
     cudnn_utils.cpp
     fused_attn/fused_attn.cpp
     util/cuda_nvml.cpp
     newton_schulz/newton_schulz.cpp)

list(APPEND transformer_engine_cuda_sources
     common.cu
     multi_tensor/adam.cu
     multi_tensor/l2norm.cu
     multi_tensor/scale.cu
     multi_tensor/sgd.cu
     transpose/cast_transpose.cu
     transpose/transpose.cu
     transpose/cast_transpose_fusion.cu
     transpose/transpose_fusion.cu
     transpose/multi_cast_transpose.cu
     transpose/quantize_transpose_vector_blockwise.cu
     transpose/swap_first_dims.cu
     dropout/dropout.cu
     fused_attn/context_parallel.cu
     fused_attn/kv_cache.cu
     fused_attn/fused_attn_f16_arbitrary_seqlen.cu #CUDA-only
     fused_attn/fused_attn_fp8.cu #CUDA-only
     fused_attn/utils.cu #CUDA-only
     gemm/cublaslt_gemm.cu
     gemm/cublaslt_grouped_gemm.cu
     normalization/layernorm/ln_bwd_semi_cuda_kernel.cu
     normalization/layernorm/ln_fwd_cuda_kernel.cu
     normalization/rmsnorm/rmsnorm_bwd_semi_cuda_kernel.cu
     normalization/rmsnorm/rmsnorm_fwd_cuda_kernel.cu
     permutation/permutation.cu
     util/utils.cu
     util/padding.cu
     util/splits_to_offsets.cu
     util/topk.cu
     swizzle/swizzle.cu
     swizzle/swizzle_block_scaling.cu #CUDA-only
     fused_softmax/scaled_masked_softmax.cu
     fused_softmax/scaled_upper_triang_masked_softmax.cu
     fused_softmax/scaled_aligned_causal_masked_softmax.cu
     fused_rope/fused_rope.cu
     fused_router/fused_moe_aux_loss.cu
     fused_router/fused_score_for_moe_aux_loss.cu
     fused_router/fused_topk_with_score_function.cu
     recipe/current_scaling.cu
     recipe/delayed_scaling.cu
     recipe/fp8_block_scaling.cu
     comm_gemm_overlap/userbuffers/userbuffers.cu)

set(cuda_only_cuda_sources
     fused_attn/fused_attn_f16_arbitrary_seqlen.cu
     fused_attn/fused_attn_fp8.cu
     fused_attn/utils.cu
     swizzle/swizzle_block_scaling.cu
     util/topk.cu) #CUDA-only

list(APPEND transformer_engine_cuda_arch_specific_sources
     fused_attn/flash_attn.cu
     activation/gelu.cu
     activation/gelu_dbias.cu
     activation/gelu_grouped.cu
     activation/gelu_grouped_dbias.cu
     activation/glu.cu
     activation/relu.cu
     activation/relu_dbias.cu
     activation/relu_grouped.cu
     activation/relu_grouped_dbias.cu
     activation/swiglu.cu
     activation/swiglu_dbias.cu
     activation/swiglu_grouped.cu
     activation/swiglu_grouped_dbias.cu
     cast/cast.cu
     cast/cast_dbias.cu
     cast/cast_grouped.cu
     cast/cast_grouped_dbias.cu
     gemm/cutlass_grouped_gemm.cu #CUDA-only
     hadamard_transform/group_hadamard_transform.cu #CUDA-only
     hadamard_transform/graph_safe_group_hadamard_transform.cu #CUDA-only
     hadamard_transform/hadamard_transform.cu
     hadamard_transform/hadamard_transform_cast_fusion.cu #CUDA-only
     hadamard_transform/group_hadamard_transform_cast_fusion.cu #CUDA-only
     hadamard_transform/group_row_cast_col_hadamard_transform_cast_fusion.cu #CUDA-only
     hadamard_transform/graph_safe_group_row_cast_col_hadamard_transform_cast_fusion.cu #CUDA-only
     multi_tensor/compute_scale.cu
     recipe/mxfp8_scaling.cu
     recipe/nvfp4.cu
     transpose/quantize_transpose_square_blockwise.cu
     transpose/quantize_transpose_vector_blockwise_fp4.cu)

set(cuda_only_cuda_arch_specific_sources
     gemm/cutlass_grouped_gemm.cu
     hadamard_transform/group_hadamard_transform.cu
     hadamard_transform/graph_safe_group_hadamard_transform.cu
     hadamard_transform/hadamard_transform_cast_fusion.cu
     hadamard_transform/row_cast_col_hadamard_transform_cast_fusion.cu
     hadamard_transform/group_hadamard_transform_cast_fusion.cu
     hadamard_transform/group_row_cast_col_hadamard_transform_cast_fusion.cu
     hadamard_transform/graph_safe_group_row_cast_col_hadamard_transform_cast_fusion.cu)

# Compiling the files with the worst compilation time first to hopefully overlap
# better with the faster-compiling cpp files
list(APPEND transformer_engine_SOURCES ${transformer_engine_cuda_arch_specific_sources}
                                       ${transformer_engine_cuda_sources}
                                       ${transformer_engine_cpp_sources})

if(USE_ROCM)
# Remove CUDA-only sources when not building for CUDA
  list(REMOVE_ITEM transformer_engine_SOURCES
    ${cuda_only_cpp_sources}
    ${cuda_only_cuda_sources}
    ${cuda_only_cuda_arch_specific_sources})

  #ROCm specific source codes
  list(INSERT transformer_engine_SOURCES 0
      amd_detail/system.cpp
      comm_gemm_overlap/rocm_comm_gemm_overlap.cpp
      fused_attn_rocm/fused_attn.cpp
      fused_attn_rocm/fused_attn_aotriton.cpp
      fused_attn_rocm/fused_attn_ck.cpp
      fused_attn_rocm/utils.cpp
      gemm/ck_grouped_gemm/ck_grouped_gemm.cpp
      gemm/ck_grouped_gemm/ck_grouped_gemm_fp8.cpp
      gemm/ck_grouped_gemm/ck_grouped_gemm_fp16.cpp
      gemm/ck_grouped_gemm/ck_grouped_gemm_fp16_nn.cpp
      gemm/ck_grouped_gemm/ck_grouped_gemm_fp16_nt.cpp
      gemm/ck_grouped_gemm/ck_grouped_gemm_fp16_tn.cpp
      gemm/ck_grouped_gemm/ck_grouped_gemm_fp16_tt.cpp
      gemm/ck_grouped_gemm/ck_mx_grouped_gemm.cpp
      gemm/rocm_gemm.cu)
endif()

if(USE_CUDA)
# Set compile options for CUDA sources with generic architectures.
# These get standard archs (pre-Blackwell) + generic Blackwell family heads.
foreach(cuda_source IN LISTS transformer_engine_cuda_sources)
  set(arch_compile_options)
  foreach(arch IN LISTS NVTE_STANDARD_ARCHS)
    list(APPEND arch_compile_options "--generate-code=arch=compute_${arch},code=sm_${arch}")
  endforeach()
  foreach(arch IN LISTS NVTE_GENERIC_ARCHS)
    list(APPEND arch_compile_options "--generate-code=arch=compute_${arch},code=sm_${arch}")
  endforeach()

  if(arch_compile_options)
    set_property(
      SOURCE ${cuda_source}
      APPEND
      PROPERTY
      COMPILE_OPTIONS ${arch_compile_options}
    )
  endif()
endforeach()

# Set compile options for CUDA sources with arch-specific features.
# These get standard archs (pre-Blackwell) + Blackwell specific targets (a/f suffix).
# They must NOT get generic Blackwell archs, as they use family/arch-specific PTX features.
foreach(cuda_source IN LISTS transformer_engine_cuda_arch_specific_sources)
  set(arch_compile_options)
  foreach(arch IN LISTS NVTE_STANDARD_ARCHS)
    list(APPEND arch_compile_options "--generate-code=arch=compute_${arch},code=sm_${arch}")
  endforeach()
  foreach(arch IN LISTS NVTE_SPECIFIC_ARCHS)
    list(APPEND arch_compile_options "--generate-code=arch=compute_${arch},code=sm_${arch}")
  endforeach()

  if(arch_compile_options)
    set_property(
      SOURCE ${cuda_source}
      APPEND
      PROPERTY
      COMPILE_OPTIONS ${arch_compile_options}
    )
  endif()
endforeach()

add_library(transformer_engine SHARED ${transformer_engine_SOURCES})
# Disable CMake's automatic architecture flag injection.
# All architectures are handled explicitly via per-source COMPILE_OPTIONS
# using NVTE_STANDARD_ARCHS, NVTE_GENERIC_ARCHS, and NVTE_SPECIFIC_ARCHS above.
set_target_properties(transformer_engine PROPERTIES CUDA_ARCHITECTURES OFF)

else() #USE_ROCM

  # process source code files
  include("${CMAKE_CURRENT_SOURCE_DIR}/../../build_tools/hipify/hipify.cmake")
  TE_Hipify(${CMAKE_CURRENT_SOURCE_DIR})
  TE_GetHipifiedSources("${transformer_engine_SOURCES}" ${CMAKE_CURRENT_SOURCE_DIR} te_hip_sources)

  message("${message_line}")
  message(STATUS "nvte hipified sources: ${te_hip_sources}")

  add_library(transformer_engine SHARED ${te_hip_sources})

  # Workaround for TheRock installation that moved some headers from system-wide location
  # to rocm_sysdeps but missing it in the default include path.
  target_include_directories(transformer_engine SYSTEM PRIVATE "${ROCM_PATH}/lib/rocm_sysdeps/include")
endif()
target_include_directories(transformer_engine PUBLIC
                           "${CMAKE_CURRENT_SOURCE_DIR}/include")

if (USE_CUDA)
# Grouped GEMM kernels require SM90a
set_property(
  SOURCE gemm/cutlass_grouped_gemm.cu
  APPEND
  PROPERTY
  COMPILE_OPTIONS "--generate-code=arch=compute_90a,code=sm_90a")

# CUTLASS kernels could cause hang in debug build
set(CUTLASS_KERNEL_SOURCES
    gemm/cutlass_grouped_gemm.cu
    hadamard_transform/group_hadamard_transform_cast_fusion.cu
    hadamard_transform/group_row_cast_col_hadamard_transform_cast_fusion.cu
    hadamard_transform/hadamard_transform_cast_fusion.cu)
set_property(
  SOURCE ${CUTLASS_KERNEL_SOURCES}
  APPEND
  PROPERTY
  COMPILE_OPTIONS "-g0;-dopt=on")
else()
  set(CK_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/composable_kernel)
  target_include_directories(transformer_engine PRIVATE ${CK_ROOT}/include)
endif() #USE_CUDA

# Configure dependencies
if (USE_CUDA)
target_link_libraries(transformer_engine PUBLIC
                      CUDA::cublas
                      CUDA::cudart
                      CUDNN::cudnn_all)

target_include_directories(transformer_engine PRIVATE
                           ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_include_directories(transformer_engine SYSTEM PRIVATE
                           ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/cccl)
target_include_directories(transformer_engine PRIVATE "${CUDNN_FRONTEND_INCLUDE_DIR}")
target_include_directories(transformer_engine PRIVATE
                          ${CUTLASS_INCLUDE_DIR}
                          ${CUTLASS_TOOLS_INCLUDE_DIR})
endif()

# Compiling Userbuffers with native MPI bootstrapping requires linking against MPI
option(NVTE_UB_WITH_MPI "Bootstrap Userbuffers with MPI" OFF)
if (NVTE_UB_WITH_MPI)
    # OpenMPI C++ headers are deprecated -- flag unused w/ MPICH
    add_definitions(-DOMPI_SKIP_MPICXX)
    
    target_include_directories(transformer_engine PRIVATE "$ENV{MPI_HOME}/include")
    target_link_directories(transformer_engine PRIVATE "$ENV{MPI_HOME}/lib")
    target_link_libraries(transformer_engine PUBLIC mpi)
    target_compile_definitions(transformer_engine PUBLIC NVTE_UB_WITH_MPI)
endif()

if(USE_CUDA)
option(NVTE_ENABLE_NVSHMEM "Compile with NVSHMEM library" OFF)
if (NVTE_ENABLE_NVSHMEM)
    add_subdirectory(nvshmem_api)
    target_link_libraries(transformer_engine PUBLIC nvshmemapi)
    target_include_directories(transformer_engine PUBLIC ${NVSHMEMAPI_INCLUDE_DIR})
endif()

option(NVTE_WITH_CUBLASMP "Use cuBLASMp for tensor parallel GEMMs" OFF)
if (NVTE_WITH_CUBLASMP)

    target_compile_definitions(transformer_engine PRIVATE NVTE_WITH_CUBLASMP)
    target_include_directories(transformer_engine PRIVATE ${CUBLASMP_DIR}/include)
    find_nccl_version(NCCL_VERSION NCCL_INCLUDE_DIR)
    find_cublasmp_version(CUBLASMP_VERSION CUBLASMP_INCLUDE_DIR ${CUBLASMP_DIR})
    find_library(CUBLASMP_LIB
                 NAMES cublasmp libcublasmp.so libcublasmp.so.0
                 PATHS ${CUBLASMP_DIR}
                 PATH_SUFFIXES lib lib64 lib/aarch64-linux-gnu lib/sbsa-linux-gnu lib/x86_64-linux-gnu
                 REQUIRED)
    find_library(NCCL_LIB
                 NAMES nccl libnccl
                 PATH_SUFFIXES lib
                 REQUIRED)
    # cuBLASMp 0.8 is the first release with CUDA-graph-safe overlap algos,
    # and NCCL 2.30 is the first release with graph-safe one-sided RMA
    # primitives (ncclPutSignal/ncclWaitSignal) that those algos use.
    if (CUBLASMP_VERSION VERSION_LESS 0.8.0)
      message(FATAL_ERROR
              "NVTE_WITH_CUBLASMP requires cuBLASMp >= 0.8.0, but found cuBLASMp "
              "${CUBLASMP_VERSION} in ${CUBLASMP_INCLUDE_DIR}/cublasmp.h")
    endif()
    if (NCCL_VERSION VERSION_LESS 2.30.0)
      message(FATAL_ERROR
              "NVTE_WITH_CUBLASMP requires NCCL >= 2.30.0 (for graph-capture-safe "
              "one-sided RMA primitives used by cuBLASMp's overlap algorithms), but "
              "found NCCL ${NCCL_VERSION} in ${NCCL_INCLUDE_DIR}/nccl.h")
    endif()
    target_link_libraries(transformer_engine PUBLIC ${NCCL_LIB} ${CUBLASMP_LIB})
    message(STATUS "Using cuBLASMp ${CUBLASMP_VERSION} at: ${CUBLASMP_DIR}")
    message(STATUS "Using NCCL ${NCCL_VERSION} at: ${NCCL_LIB}")
endif()

option(NVTE_WITH_CUSOLVERMP "Use cuSolverMp for distributed Newton-Schulz" OFF)
if (NVTE_WITH_CUSOLVERMP)
    target_compile_definitions(transformer_engine PRIVATE NVTE_WITH_CUSOLVERMP)
    target_include_directories(transformer_engine PRIVATE ${CUSOLVERMP_DIR}/include)
    find_library(CUSOLVERMP_LIB
                 NAMES cusolverMp libcusolverMp
                 PATHS ${CUSOLVERMP_DIR}
                 PATH_SUFFIXES lib
                 REQUIRED)
    find_library(NCCL_LIB
                 NAMES nccl libnccl
                 PATH_SUFFIXES lib
                 REQUIRED)
    target_link_libraries(transformer_engine PRIVATE ${NCCL_LIB} ${CUSOLVERMP_LIB})
    message(STATUS "Using cuSolverMp at: ${CUSOLVERMP_DIR}")
endif()

# -- NCCL EP (on by default, HT mode only) ---------------------------------
# Set -DNVTE_WITH_NCCL_EP=OFF (or NVTE_WITH_NCCL_EP=0 in setup.py) to
# skip NCCL EP entirely - useful on older images whose system NCCL is below
# the 2.30.4 EP minimum.
option(NVTE_WITH_NCCL_EP "Build NCCL EP into libtransformer_engine.so" ON)
if(NVTE_WITH_NCCL_EP)
# SM>=90 and NCCL>=2.30.4 are gated at runtime in EPBackend::initialize.
# -- NCCL EP headers --------------------------------------------------------
# Headers + libs are produced by the in-tree 3rdparty/nccl submodule build
# (auto-built by setup.py via build_nccl_ep_submodule).
set(NCCL_EP_SUBMODULE_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/nccl")
set(NCCL_EP_INCLUDE_DIR "${NCCL_EP_SUBMODULE_ROOT}/build/include")
if(NOT EXISTS "${NCCL_EP_INCLUDE_DIR}/nccl_ep.h")
  message(FATAL_ERROR
    "NCCL EP header not found at ${NCCL_EP_INCLUDE_DIR}/nccl_ep.h. "
    "Run `git submodule update --init --recursive` and rebuild TE.")
endif()
message(STATUS "NCCL EP headers: ${NCCL_EP_INCLUDE_DIR}")

# -- libnccl_ep.a -----------------------------------------------------------
# Statically linked into libtransformer_engine.so. EPBackend::initialize checks
# NCCL >= 2.30.4 before any nccl_ep call, so the newer NCCL symbols nccl_ep
# imports stay unresolved (and harmless) under default ELF lazy binding when
# the gate trips. LD_BIND_NOW environments lose this property.
set(NCCL_EP_LIB_DIR "${NCCL_EP_SUBMODULE_ROOT}/build/lib")
find_file(NCCL_EP_LIB
    NAMES libnccl_ep.a
    HINTS ${NCCL_EP_LIB_DIR}
    NO_DEFAULT_PATH
    REQUIRED)

# -- NCCL core: nccl.h + libnccl.so -----------------------------------------
# setup.py passes -DNCCL_INCLUDE_DIR; standalone CMake falls back to probing
# well-known NCCL install prefixes.
find_path(NCCL_INCLUDE_DIR nccl.h
    HINTS /opt/nvidia/nccl/include /usr/local/nccl/include)
if(NOT NCCL_INCLUDE_DIR)
  message(FATAL_ERROR
    "nccl.h not found. Pass -DNCCL_INCLUDE_DIR=<prefix>/include.")
endif()
if(NOT NCCL_LIB)
  find_library(NCCL_LIB
      NAMES nccl libnccl
      PATH_SUFFIXES lib lib64
      REQUIRED)
endif()

target_include_directories(transformer_engine PRIVATE
    ${NCCL_EP_INCLUDE_DIR}
    ${NCCL_INCLUDE_DIR})

# libnccl.so direct symbols (ncclGetVersion etc.) come from libnccl_ep.a's
# DT_NEEDED chain plus this TU's own references. CUDA::cuda_driver must follow
# the static archive on the link line so --as-needed records libcuda.so.1.
target_link_libraries(transformer_engine PUBLIC ${NCCL_LIB})
target_link_libraries(transformer_engine PRIVATE
    -Wl,--whole-archive ${NCCL_EP_LIB} -Wl,--no-whole-archive
    CUDA::cuda_driver)

target_sources(transformer_engine PRIVATE
    ep/ep_backend.cpp
    ep/ep_api.cpp)
target_compile_definitions(transformer_engine PRIVATE NVTE_WITH_NCCL_EP)

message(STATUS "NCCL EP enabled (static link): ${NCCL_EP_LIB}")
message(STATUS "NCCL EP include: ${NCCL_EP_INCLUDE_DIR}")
else()
  # NCCL EP off: ep_api.cpp's #else branch exports throwing nvte_ep_* stubs.
  target_sources(transformer_engine PRIVATE ep/ep_api.cpp)
  message(STATUS "NCCL EP disabled (NVTE_WITH_NCCL_EP=OFF) - using nvte_ep_* stubs")
endif()

# Number of philox4x32 rounds for stochastic rounding (build-time constant).
set(NVTE_BUILD_NUM_PHILOX_ROUNDS_STR $ENV{NVTE_BUILD_NUM_PHILOX_ROUNDS})
if (NOT NVTE_BUILD_NUM_PHILOX_ROUNDS_STR)
  set(NVTE_BUILD_NUM_PHILOX_ROUNDS_STR "10")
endif()
if (NOT NVTE_BUILD_NUM_PHILOX_ROUNDS_STR MATCHES "^[1-9][0-9]*$")
  message(FATAL_ERROR
          "Environment variable NVTE_BUILD_NUM_PHILOX_ROUNDS must be a positive integer, "
          "but got '${NVTE_BUILD_NUM_PHILOX_ROUNDS_STR}'.")
endif()
set(NVTE_BUILD_NUM_PHILOX_ROUNDS ${NVTE_BUILD_NUM_PHILOX_ROUNDS_STR})

target_compile_definitions(transformer_engine
                           PUBLIC NVTE_BUILD_NUM_PHILOX_ROUNDS=${NVTE_BUILD_NUM_PHILOX_ROUNDS})
message(STATUS "Philox rounds for stochastic rounding: ${NVTE_BUILD_NUM_PHILOX_ROUNDS}")

# Hack to enable dynamic loading in cuDNN frontend
target_compile_definitions(transformer_engine PUBLIC NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING)

else() # USE_ROCM
  option(NVTE_ENABLE_ROCSHMEM "Compile with ROCSHMEM library" OFF)
  if (NVTE_ENABLE_ROCSHMEM)
    add_subdirectory(rocshmem_api)
    target_link_options(transformer_engine PRIVATE
        -fgpu-rdc
    )
    target_link_libraries(transformer_engine PUBLIC 
        -Wl,--whole-archive
        rocshmemapi
        -Wl,--no-whole-archive
    )
  endif()
  set(GPU_TARGETS ${CMAKE_HIP_ARCHITECTURES})

  function(get_git_commit REPO_DIR COMMIT_ID_VAR)
    file(REAL_PATH "${REPO_DIR}" REPO_DIR)
    execute_process(
      COMMAND sh -c "export GIT_CONFIG_GLOBAL=$(mktemp /tmp/gitconfig.XXXXXX);
      git config --global --add safe.directory ${REPO_DIR} && git -C ${REPO_DIR} rev-parse HEAD;
      rm $GIT_CONFIG_GLOBAL"
      OUTPUT_VARIABLE ${COMMIT_ID_VAR}
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    SET(${COMMIT_ID_VAR} ${${COMMIT_ID_VAR}} PARENT_SCOPE)
  endfunction()

  if(USE_FUSED_ATTN_AOTRITON)
    add_subdirectory(aotriton ${CMAKE_CURRENT_BINARY_DIR}/aotriton)
  endif()

  if(USE_FUSED_ATTN_CK)
    add_subdirectory(ck_fused_attn ${CMAKE_CURRENT_BINARY_DIR}/ck_fused_attn)
  endif()

  if(USE_HIPKITTENS_GEMM)
    add_subdirectory(gemm/kittens ${CMAKE_CURRENT_BINARY_DIR}/kittens)
  endif()

  find_package(hip)
  list(APPEND transformer_engine_LINKER_LIBS hip::host hip::device roctx64)
  find_package(hiprtc)
  list(APPEND transformer_engine_LINKER_LIBS hiprtc)
  find_package(hipblaslt)
  list(APPEND transformer_engine_LINKER_LIBS roc::hipblaslt)
  if(USE_FUSED_ATTN_AOTRITON)
    target_compile_definitions(transformer_engine PUBLIC USE_FUSED_ATTN_AOTRITON)
    list(APPEND transformer_engine_LINKER_LIBS aotriton)
  endif()
  if(USE_FUSED_ATTN_CK)
    target_compile_definitions(transformer_engine PUBLIC USE_FUSED_ATTN_CK)
    list(APPEND transformer_engine_LINKER_LIBS ck_fused_attn)
  endif()
  if(USE_HIPKITTENS_GEMM)
    target_compile_definitions(transformer_engine PUBLIC USE_HIPKITTENS_GEMM)
    target_compile_definitions(transformer_engine PUBLIC ${KITTENS_HAVE_DEFS})
    list(APPEND transformer_engine_LINKER_LIBS kittens_gemm)
  endif()
  target_link_libraries(transformer_engine PUBLIC ${transformer_engine_LINKER_LIBS})
endif()

# Helper functions to make header files with C++ strings
function(make_string_header STRING STRING_NAME)
    configure_file(util/string_header.h.in
                   "string_headers/${STRING_NAME}.h"
                   @ONLY)
endfunction()
function(make_string_header_from_file file_ STRING_NAME)
    file(READ "${file_}" STRING)
    configure_file(util/string_header.h.in
                   "string_headers/${STRING_NAME}.h"
                   @ONLY)
endfunction()

if(USE_CUDA)
# Header files with C++ strings
list(GET CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES 0 cuda_include_path)
make_string_header("${cuda_include_path}"
                   string_path_cuda_include)
make_string_header_from_file(transpose/rtc/cast_transpose_fusion.cu
                             string_code_transpose_rtc_cast_transpose_fusion_cu)
make_string_header_from_file(transpose/rtc/cast_transpose.cu
                             string_code_transpose_rtc_cast_transpose_cu)
make_string_header_from_file(transpose/rtc/transpose.cu
                             string_code_transpose_rtc_transpose_cu)
make_string_header_from_file(transpose/rtc/swap_first_dims.cu
                             string_code_transpose_rtc_swap_first_dims_cu)
make_string_header_from_file(utils.cuh
                             string_code_utils_cuh)
else()
  make_string_header_from_file(utils_hip.cuh
                              string_code_utils_cuh)
  make_string_header_from_file(transpose/rtc/cast_transpose_fusion.hip
                              string_code_transpose_rtc_cast_transpose_fusion_cu)
  make_string_header_from_file(transpose/rtc/cast_transpose.hip
                              string_code_transpose_rtc_cast_transpose_cu)
  make_string_header_from_file(transpose/rtc/transpose.hip
                              string_code_transpose_rtc_transpose_cu)
  make_string_header_from_file(transpose/rtc/swap_first_dims.hip
                              string_code_transpose_rtc_swap_first_dims_cu)
  make_string_header_from_file(amd_detail/hip_float8.h
                              string_code_amd_detail_hip_float8_h)
endif()
make_string_header_from_file(util/math.h
                             string_code_util_math_h)
target_include_directories(transformer_engine PRIVATE
                           "${CMAKE_CURRENT_BINARY_DIR}/string_headers")

# Compiler options
if(USE_CUDA)
set(nvte_sources_with_fast_math)
list(APPEND nvte_sources_with_fast_math fused_softmax/scaled_masked_softmax.cu
                                        fused_softmax/scaled_upper_triang_masked_softmax.cu
                                        fused_softmax/scaled_aligned_causal_masked_softmax.cu
                                        multi_tensor/adam.cu
                                        multi_tensor/compute_scale.cu
                                        multi_tensor/l2norm.cu
                                        multi_tensor/scale.cu
                                        multi_tensor/sgd.cu
                                        fused_attn/flash_attn.cu
                                        fused_attn/context_parallel.cu
                                        fused_attn/kv_cache.cu)

option(NVTE_BUILD_ACTIVATION_WITH_FAST_MATH "Compile activation kernels with --use_fast_math option" OFF)
if (NVTE_BUILD_ACTIVATION_WITH_FAST_MATH)
  list(APPEND nvte_sources_with_fast_math activation/gelu.cu
                                          activation/gelu_dbias.cu
                                          activation/gelu_grouped.cu
                                          activation/gelu_grouped_dbias.cu
                                          activation/glu.cu
                                          activation/relu.cu
                                          activation/relu_dbias.cu
                                          activation/relu_grouped.cu
                                          activation/relu_grouped_dbias.cu
                                          activation/swiglu.cu
                                          activation/swiglu_dbias.cu
                                          activation/swiglu_grouped.cu
                                          activation/swiglu_grouped_dbias.cu)
endif()

foreach(cuda_source IN LISTS nvte_sources_with_fast_math)
  set_property(
    SOURCE ${cuda_source}
    APPEND
    PROPERTY
    COMPILE_OPTIONS "--use_fast_math")
endforeach()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# Number of parallel build jobs
if($ENV{MAX_JOBS})
  set(BUILD_JOBS_STR $ENV{MAX_JOBS})
elseif($ENV{NVTE_BUILD_MAX_JOBS})
  set(BUILD_JOBS_STR $ENV{NVTE_BUILD_MAX_JOBS})
else()
  set(BUILD_JOBS_STR "max")
endif()
message(STATUS "Parallel build jobs: ${BUILD_JOBS_STR}")

# Number of threads per parallel build job
set(BUILD_THREADS_PER_JOB $ENV{NVTE_BUILD_THREADS_PER_JOB})
if (NOT BUILD_THREADS_PER_JOB)
  set(BUILD_THREADS_PER_JOB 1)
endif()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --threads ${BUILD_THREADS_PER_JOB}")
message(STATUS "Threads per parallel build job: ${BUILD_THREADS_PER_JOB}")
else()
  set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -O3")
  set(HIP_HCC_FLAGS "${CMAKE_HIP_FLAGS} -mavx2 -mf16c -mfma -std=c++17")
  # Ask hcc to generate device code during compilation so we can use
  # host linker to link.
  set(HIP_HCC_FLAGS "${HIP_HCC_FLAGS} -fno-gpu-rdc -Wno-defaulted-function-deleted")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${HIP_HCC_FLAGS}")
  # Hide non-necessary symbols in shared object.
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/libtransformer_engine.version")
endif()

# Install library
install(TARGETS transformer_engine DESTINATION .)
if (USE_ROCM)
  set_target_properties(transformer_engine PROPERTIES INSTALL_RPATH "$ORIGIN/lib;$ORIGIN/transformer_engine/lib")
  file(READ "${ROCM_PATH}/.info/version" ROCM_VER)
  string(STRIP "${ROCM_VER}" ROCM_VER)
  string(REGEX MATCH "^[0-9]+\\.[0-9]+" ROCM_VER "${ROCM_VER}")
  get_git_commit("${TE}" TE_COMMIT_ID)
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/build_info.txt"
    "ROCM_VERSION: ${ROCM_VER}\n"
    "GPU_TARGETS: ${CMAKE_HIP_ARCHITECTURES}\n"
    "COMMIT_ID: ${TE_COMMIT_ID}\n"
  )
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/build_info.txt" DESTINATION "transformer_engine/")
endif()
