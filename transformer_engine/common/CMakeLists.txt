# This file was modified for portability to AMDGPU
# Copyright (c) 2022-2025, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

cmake_minimum_required(VERSION 3.21)

option(USE_ROCM "Use ROCm" ON)
option(USE_FUSED_ATTN_AOTRITON "Use aotriton backend" ON)
option(USE_FUSED_ATTN_AOTRITON_BUILD_GPU_KERNELS "Build AOTriton GPU kernels" OFF)
option(USE_FUSED_ATTN_CK "Use ck backend" ON)
set(USE_CUDA OFF)

if (USE_ROCM)
  #Check if ROCm exists
  if( ((EXISTS "/opt/rocm/") OR (EXISTS $ENV{ROCM_PATH})) )
    message("Using AMD Platform")
  else()
    message(FATAL_ERROR "Could not find ROCm installation")
  endif()

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
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL 13.0)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 89 90 100 120)
  elseif (CUDAToolkit_VERSION VERSION_GREATER_EQUAL 12.8)
    set(CMAKE_CUDA_ARCHITECTURES 70 80 89 90 100 120)
  else ()
    set(CMAKE_CUDA_ARCHITECTURES 70 80 89 90)
  endif()
endif()
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
if (CUDAToolkit_VERSION VERSION_LESS 12.0)
  message(FATAL_ERROR "CUDA 12.0+ is required, but found CUDA ${CUDAToolkit_VERSION}")
endif()

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
  project(transformer_engine LANGUAGES HIP CXX)

  # If using ROCm, ninja must be installed to build TE.
  if (NOT CMAKE_GENERATOR STREQUAL "Ninja")
    message(FATAL_ERROR "This project requires the Ninja build system. Install it using 'pip install ninja'.")
  endif()

  # Disable Asserts In Code (Can't use asserts on HIP stack.)
  add_definitions(-DNDEBUG)
  add_definitions(-DUSE_ROCM)

  if(NOT DEFINED ENV{NVTE_ROCM_ARCH})
    SET(CMAKE_HIP_ARCHITECTURES gfx942 gfx950)
  else()
    SET(CMAKE_HIP_ARCHITECTURES $ENV{NVTE_ROCM_ARCH})
  endif()

  # build error will be dup-ed parallel-jobs times
  # set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -parallel-jobs=4")
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -g")
  endif()

  list(APPEND CMAKE_MODULE_PATH "/opt/rocm")
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

# Configure Transformer Engine library
include_directories(${PROJECT_SOURCE_DIR}/..)
set(transformer_engine_SOURCES)
# Source files in both cuda and rocm
list(APPEND transformer_engine_SOURCES
     transformer_engine.cpp
     common.cu
     multi_tensor/adam.cu
     multi_tensor/compute_scale.cu
     multi_tensor/l2norm.cu
     multi_tensor/scale.cu
     multi_tensor/sgd.cu
     transpose/cast_transpose.cu
     transpose/transpose.cu
     transpose/cast_transpose_fusion.cu
     transpose/transpose_fusion.cu
     transpose/multi_cast_transpose.cu
     transpose/swap_first_dims.cu
     activation/gelu.cu
     dropout/dropout.cu
     fused_attn/flash_attn.cu
     fused_attn/context_parallel.cu
     fused_attn/kv_cache.cu
     activation/relu.cu
     activation/swiglu.cu
     gemm/cublaslt_gemm.cu
     gemm/cutlass_grouped_gemm.cu
     normalization/common.cpp
     normalization/layernorm/ln_api.cpp
     normalization/layernorm/ln_bwd_semi_cuda_kernel.cu
     normalization/layernorm/ln_fwd_cuda_kernel.cu
     normalization/rmsnorm/rmsnorm_api.cpp
     normalization/rmsnorm/rmsnorm_bwd_semi_cuda_kernel.cu
     normalization/rmsnorm/rmsnorm_fwd_cuda_kernel.cu
     permutation/permutation.cu
     util/cast.cu
     util/padding.cu
     util/cuda_driver.cpp
     util/cuda_runtime.cpp
     util/multi_stream.cpp
     util/rtc.cpp
     swizzle/swizzle.cu
     fused_softmax/scaled_masked_softmax.cu
     fused_softmax/scaled_upper_triang_masked_softmax.cu
     fused_softmax/scaled_aligned_causal_masked_softmax.cu
     fused_rope/fused_rope.cu
     fused_router/fused_moe_aux_loss.cu
     fused_router/fused_score_for_moe_aux_loss.cu
     fused_router/fused_topk_with_score_function.cu
     recipe/current_scaling.cu
     recipe/delayed_scaling.cu
     recipe/fp8_block_scaling.cu)
if(USE_CUDA)
# Removed indent to minimize code diff with NV upstream
# Files unique in cuda building
list(APPEND transformer_engine_SOURCES
     cudnn_utils.cpp
     transpose/quantize_transpose_square_blockwise.cu
     transpose/quantize_transpose_vector_blockwise.cu
     fused_attn/fused_attn_f16_max512_seqlen.cu
     fused_attn/fused_attn_f16_arbitrary_seqlen.cu
     fused_attn/fused_attn_fp8.cu
     fused_attn/fused_attn.cpp
     fused_attn/utils.cu
     util/cuda_nvml.cpp
     comm_gemm_overlap/userbuffers/ipcsocket.cc
     comm_gemm_overlap/userbuffers/userbuffers-host.cpp
     comm_gemm_overlap/userbuffers/userbuffers.cu
     comm_gemm_overlap/comm_gemm_overlap.cpp)

if (NVTE_WITH_CUBLASMP)
list(APPEND transformer_engine_SOURCES
     comm_gemm/comm_gemm.cpp)
endif()

add_library(transformer_engine SHARED ${transformer_engine_SOURCES})
else()
  list(APPEND transformer_engine_SOURCES
       fused_attn_rocm/fused_attn.cpp
       fused_attn_rocm/fused_attn_aotriton.cpp
       fused_attn_rocm/fused_attn_ck.cpp
       fused_attn_rocm/utils.cpp
       gemm/rocm_gemm.cu
       amd_detail/system.cpp)

  # process source code files
  set(TE ${CMAKE_CURRENT_SOURCE_DIR}/../..)
  set(THIRDPARTY ${TE}/3rdparty)
  list(APPEND CMAKE_MODULE_PATH "${THIRDPARTY}/hipify_torch/cmake")
  include(Hipify)
  message(STATUS "CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")

  set(header_include_dir
      ${CMAKE_CURRENT_SOURCE_DIR}/..
      ${CMAKE_CURRENT_SOURCE_DIR}/include 
      ${CMAKE_CURRENT_SOURCE_DIR}/util
      ${CMAKE_CURRENT_SOURCE_DIR})
  message(STATUS "HIPIFY CUDA_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
  message(STATUS "HIPIFY HEADER_INCLUDE_DIR: ${header_include_dir}")
  hipify(CUDA_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
      HEADER_INCLUDE_DIR ${header_include_dir}
      IGNORES "*/amd_detail/*"
      IGNORES "*/aotriton/*"
      IGNORES "*/ck_fused_attn/*"
      IGNORES "*/pytorch/csrc/*"
      IGNORES "*/jax/csrc/*"
      IGNORES "*/rocshmem_api/*"
      CUSTOM_MAP_FILE "${TE}/hipify_custom_map.json"
      NO_MATH_REPLACE
  )
  get_hipified_list("${transformer_engine_SOURCES}" te_hip_sources)
  message("${message_line}")
  message(STATUS "nvte hipified sources: ${te_hip_sources}")

  add_library(transformer_engine SHARED ${te_hip_sources})
  target_include_directories(transformer_engine PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
endif()

target_include_directories(transformer_engine PUBLIC
                           "${CMAKE_CURRENT_SOURCE_DIR}/include")

if (USE_CUDA)
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER 12.0)
  set_source_files_properties(
    "gemm/cutlass_grouped_gemm.cu"
    PROPERTIES
    COMPILE_FLAGS
    "-gencode arch=compute_90a,code=sm_90a")
else()
  message(FATAL_ERROR "cutlass gemm/cutlass_grouped_gemm.cu kernel required sm 90a")
endif()
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

# Compiling Userbuffers with native MPI bootstrapping requires linking against MPI
option(NVTE_UB_WITH_MPI "Bootstrap Userbuffers with MPI" OFF)
if (NVTE_UB_WITH_MPI)
    find_package(MPI REQUIRED)
    target_link_libraries(transformer_engine PUBLIC MPI::MPI_CXX)
    target_include_directories(transformer_engine PRIVATE ${MPI_CXX_INCLUDES})
    target_compile_definitions(transformer_engine PUBLIC NVTE_UB_WITH_MPI)
endif()

option(NVTE_ENABLE_NVSHMEM "Compile with NVSHMEM library" OFF)
if (NVTE_ENABLE_NVSHMEM)
    add_subdirectory(nvshmem_api)
    target_link_libraries(transformer_engine PUBLIC nvshmemapi)
    target_include_directories(transformer_engine PUBLIC ${NVSHMEMAPI_INCLUDE_DIR})
endif()

option(NVTE_WITH_CUBLASMP "Use cuBLASMp for tensor parallel GEMMs" OFF)
if (NVTE_WITH_CUBLASMP)
    target_compile_definitions(transformer_engine PRIVATE NVTE_WITH_CUBLASMP)
    target_include_directories(transformer_engine PRIVATE ${CUBLASMP_DIR}/include ${NVSHMEM_DIR}/include)
    find_library(CUBLASMP_LIB
                 NAMES cublasmp libcublasmp
                 PATHS ${CUBLASMP_DIR}
                 PATH_SUFFIXES lib
                 REQUIRED)
    find_library(NVSHMEM_HOST_LIB
                 NAMES nvshmem_host libnvshmem_host.so.3
                 PATHS ${NVSHMEM_DIR}
                 PATH_SUFFIXES lib
                 REQUIRED)
  target_link_libraries(transformer_engine PUBLIC ${CUBLASMP_LIB} ${NVSHMEM_HOST_LIB})
  message(STATUS "Using cuBLASMp at: ${CUBLASMP_DIR}")
  message(STATUS "Using nvshmem at: ${NVSHMEM_DIR}")
endif()

# Hack to enable dynamic loading in cuDNN frontend
target_compile_definitions(transformer_engine PUBLIC NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING)

else()
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
  make_string_header_from_file(amd_detail/hip_f8_impl.h
                               string_code_amd_detail_hip_f8_impl_h)
endif()
make_string_header_from_file(util/math.h
                             string_code_util_math_h)
target_include_directories(transformer_engine PRIVATE
                           "${CMAKE_CURRENT_BINARY_DIR}/string_headers")

# Compiler options
set_source_files_properties(fused_softmax/scaled_masked_softmax.cu
                            fused_softmax/scaled_upper_triang_masked_softmax.cu
                            fused_softmax/scaled_aligned_causal_masked_softmax.cu
                            multi_tensor/adam.cu
                            multi_tensor/compute_scale.cu
                            multi_tensor/l2norm.cu
                            multi_tensor/scale.cu
                            multi_tensor/sgd.cu
                            fused_attn/flash_attn.cu
                            fused_attn/context_parallel.cu
                            fused_attn/kv_cache.cu
                            PROPERTIES
                            COMPILE_OPTIONS "--use_fast_math")
if(USE_CUDA)
option(NVTE_BUILD_ACTIVATION_WITH_FAST_MATH "Compile activation kernels with --use_fast_math option" OFF)
if (NVTE_BUILD_ACTIVATION_WITH_FAST_MATH)
  set_source_files_properties(activation/gelu.cu
                              activation/relu.cu
                              activation/swiglu.cu
                              util/cast.cu
                              PROPERTIES
                              COMPILE_OPTIONS "--use_fast_math")
endif()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# Number of parallel build jobs
if(ENV{MAX_JOBS})
  set(BUILD_JOBS_STR "$ENV{MAX_JOBS}")
elseif(ENV{NVTE_BUILD_MAX_JOBS})
  set(BUILD_JOBS_STR "$ENV{NVTE_BUILD_MAX_JOBS}")
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
  if("$ENV{ROCM_PATH}" STREQUAL "")
    set(ROCM_PATH "/opt/rocm")
  else()
    set(ROCM_PATH "$ENV{ROCM_PATH}")
  endif()
  file(READ "${ROCM_PATH}/.info/version" ROCM_VER)
  string(STRIP "${ROCM_VER}" ROCM_VER)
  string(REGEX MATCH "^[0-9]+\\.[0-9]+" ROCM_VER "${ROCM_VER}")
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/build_info.txt"
    "ROCM_VERSION: ${ROCM_VER}\n"
    "GPU_TARGETS: ${CMAKE_HIP_ARCHITECTURES}\n"
  )
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/build_info.txt" DESTINATION "transformer_engine/")
endif()
