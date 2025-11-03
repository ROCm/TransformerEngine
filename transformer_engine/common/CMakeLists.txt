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
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL 12.8)
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
if(USE_CUDA)
  list(APPEND transformer_engine_SOURCES
       cudnn_utils.cpp
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
       transpose/quantize_transpose_square_blockwise.cu
       transpose/quantize_transpose_vector_blockwise.cu
       activation/gelu.cu
       fused_attn/flash_attn.cu
       fused_attn/context_parallel.cu
       fused_attn/kv_cache.cu
       fused_attn/fused_attn_f16_max512_seqlen.cu
       fused_attn/fused_attn_f16_arbitrary_seqlen.cu
       activation/relu.cu
       activation/swiglu.cu
       fused_attn/fused_attn_fp8.cu
       fused_attn/fused_attn.cpp
       fused_attn/utils.cu
       gemm/cublaslt_gemm.cu
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
       util/cuda_nvml.cpp
       util/cuda_runtime.cpp
       util/rtc.cpp
       swizzle/swizzle.cu
       fused_softmax/scaled_masked_softmax.cu
       fused_softmax/scaled_upper_triang_masked_softmax.cu
       fused_softmax/scaled_aligned_causal_masked_softmax.cu
       fused_rope/fused_rope.cu
       recipe/current_scaling.cu
       recipe/delayed_scaling.cu
       recipe/fp8_block_scaling.cu
       comm_gemm_overlap/userbuffers/ipcsocket.cc
       comm_gemm_overlap/userbuffers/userbuffers-host.cpp
       comm_gemm_overlap/userbuffers/userbuffers.cu
       comm_gemm_overlap/comm_gemm_overlap.cpp)
  add_library(transformer_engine SHARED ${transformer_engine_SOURCES})
else()
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
       activation/gelu.cu
       activation/relu.cu
       activation/swiglu.cu
       fused_attn/flash_attn.cu
       fused_attn/context_parallel.cu
       fused_attn/kv_cache.cu
       fused_attn_rocm/fused_attn.cpp
       fused_attn_rocm/fused_attn_aotriton.cpp
       fused_attn_rocm/fused_attn_ck.cpp
       fused_attn_rocm/utils.cpp
       gemm/cublaslt_gemm.cu
       gemm/rocm_gemm.cu
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
       util/rtc.cpp
       amd_detail/system.cpp
       swizzle/swizzle.cu
       fused_softmax/scaled_masked_softmax.cu
       fused_softmax/scaled_upper_triang_masked_softmax.cu
       fused_softmax/scaled_aligned_causal_masked_softmax.cu
       fused_rope/fused_rope.cu
       recipe/current_scaling.cu
       recipe/delayed_scaling.cu
       recipe/fp8_block_scaling.cu)

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
      CUSTOM_MAP_FILE "${TE}/hipify_custom_map.json"
  )
  get_hipified_list("${transformer_engine_SOURCES}" te_hip_sources)
  message("${message_line}")
  message(STATUS "nvte hipified sources: ${te_hip_sources}")

  add_library(transformer_engine SHARED ${te_hip_sources})
  target_include_directories(transformer_engine PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
endif()
target_include_directories(transformer_engine PUBLIC
                           "${CMAKE_CURRENT_SOURCE_DIR}/include")



# Configure dependencies
if (USE_CUDA)
  target_link_libraries(transformer_engine PUBLIC
                      CUDA::cublas
                      CUDA::cudart
                      CUDNN::cudnn_all)
  target_include_directories(transformer_engine PRIVATE
                           ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_include_directories(transformer_engine PRIVATE "${CUDNN_FRONTEND_INCLUDE_DIR}")

  # Compiling Userbuffers with native MPI bootstrapping requires linking against MPI
  # Changed
  option(NVTE_UB_WITH_MPI "Bootstrap Userbuffers with MPI" OFF)
  if (NVTE_UB_WITH_MPI)
      find_package(MPI REQUIRED)
      target_link_libraries(transformer_engine PUBLIC MPI::MPI_CXX)
      target_include_directories(transformer_engine PRIVATE ${MPI_CXX_INCLUDES})
      target_compile_definitions(transformer_engine PUBLIC NVTE_UB_WITH_MPI)
  endif()
  
  # Hack to enable dynamic loading in cuDNN frontend
  target_compile_definitions(transformer_engine PUBLIC NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING)
else()
  set(GPU_TARGETS ${CMAKE_HIP_ARCHITECTURES})

  if(USE_FUSED_ATTN_AOTRITON)

    set(__AOTRITON_INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/aotriton")
    set(__AOTRITON_SUFFIX "_TEprivate")
  
    if(NOT DEFINED AOTRITON_PATH)
      if(USE_FUSED_ATTN_AOTRITON_BUILD_GPU_KERNELS)
        set(AOTRITON_NOIMAGE_MODE OFF)
      else()
        set(AOTRITON_NOIMAGE_MODE ON)
      endif()

      set(__AOTRITON_VER "0.11b")
      set(__AOTRITON_SHA256
          "a2a974e0ad929a5e5827c0f896c59bda4872459cbaf8dd8e0a00407f404491cf"  # rocm7.0
          )
      set(__AOTRITON_IMAGE_LIST
          "amd-gfx942"
          "amd-gfx950"
        )
      set(__AOTRITON_IMAGE_SHA256_LIST
        "3a06a99971dddb7703a30378f1c5d6b41468d926ea51821156d1b6857b985bc4" # amd-gfx942
        "27fc21f6761d57987a700436de8cf29cbdd9eeee91318dfed596eeb147d219ad" # amd-gfx950
        )
      set(__AOTRITON_BASE_URL "https://github.com/ROCm/aotriton/releases/download/")  # @lint-ignore
      set(__AOTRITON_Z "gz")
      # Set the default __AOTRITON_LIB path
      set(__AOTRITON_LIB "lib/libaotriton_v2.so")
      include(ExternalProject)

      function(aotriton_download_image image project)
        list(FIND __AOTRITON_IMAGE_LIST ${image} index)
        list(GET __AOTRITON_IMAGE_SHA256_LIST ${index} __AOTRITON_IMAGE_SHA256)

        string(CONCAT __AOTRITON_FILE
              "aotriton-${__AOTRITON_VER}-images-"
              "${image}.tar.${__AOTRITON_Z}")
        string(CONCAT __AOTRITON_URL
              "${__AOTRITON_BASE_URL}"
              "${__AOTRITON_VER}/${__AOTRITON_FILE}")

        # Set up directories
        set(__AOTRITON_DOWNLOAD_DIR ${CMAKE_CURRENT_BINARY_DIR}/aotriton_download-${image})
        set(__AOTRITON_EXTRACT_DIR ${CMAKE_CURRENT_BINARY_DIR}/aotriton_image-${image})

        ExternalProject_Add(${project}
          URL "${__AOTRITON_URL}"
          URL_HASH SHA256=${__AOTRITON_IMAGE_SHA256}
          DOWNLOAD_DIR ${__AOTRITON_DOWNLOAD_DIR}
          SOURCE_DIR ${__AOTRITON_EXTRACT_DIR}
          CONFIGURE_COMMAND ""
          BUILD_COMMAND ""
          INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_directory
          "${__AOTRITON_EXTRACT_DIR}"
          "${__AOTRITON_INSTALL_DIR}"
          BUILD_BYPRODUCTS
          "${__AOTRITON_INSTALL_DIR}/lib/aotriton.images/${image}/__signature__"
        )
        message(STATUS "Download AOTriton pre-compiled GPU images from ${__AOTRITON_URL}.")
      endfunction()

      function(aotriton_build_from_source noimage)
        if(noimage)
          SET(RECURSIVE "OFF")
        else()
          SET(RECURSIVE "ON")
        endif()
        message(STATUS "No-image mode: ${noimage}.")
        ExternalProject_Add(aotriton_external
          LIST_SEPARATOR ","
          SOURCE_DIR ${TE}/3rdparty/aotriton
          CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${__AOTRITON_INSTALL_DIR}
          -DAOTRITON_TARGET_ARCH=${ARCH_LIST_COMMA_STR}
          -DGPU_TARGETS=${ARCH_LIST_COMMA_STR}
          -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
          -DAOTRITON_NO_PYTHON=ON
          -DAOTRITON_NAME_SUFFIX=${__AOTRITON_SUFFIX}
          -DAOTRITON_NOIMAGE_MODE=${AOTRITON_NOIMAGE_MODE}
          BUILD_BYPRODUCTS "${__AOTRITON_INSTALL_DIR}/lib/libaotriton${__AOTRITON_SUFFIX}_v2.so"
        )
        message(STATUS "Adding AOTriton library.")
        add_dependencies(aotriton aotriton_external)
        target_link_libraries(aotriton INTERFACE ${__AOTRITON_INSTALL_DIR}/lib/libaotriton${__AOTRITON_SUFFIX}_v2.so)
        target_include_directories(aotriton INTERFACE ${__AOTRITON_INSTALL_DIR}/include)
        install(DIRECTORY
                ${__AOTRITON_INSTALL_DIR}/lib
                DESTINATION ${CMAKE_INSTALL_PREFIX}/transformer_engine
                PATTERN "cmake" EXCLUDE
        )
      endfunction()
      string(REPLACE ";" "," ARCH_LIST_COMMA_STR "${CMAKE_HIP_ARCHITECTURES}")
      add_library(aotriton INTERFACE)
      message(STATUS "Building AOTriton from source.")
      aotriton_build_from_source(${AOTRITON_NOIMAGE_MODE})
      if(NOT USE_FUSED_ATTN_AOTRITON_BUILD_GPU_KERNELS)
        message(STATUS "Downloading AOTriton GPU Kernels.")
        set(__AOTRITON_CHAINED_IMAGE "aotriton_external")
        foreach(image ${__AOTRITON_IMAGE_LIST})
          string(SUBSTRING ${image} 7 -1 gfx_pattern)
          string(REPLACE "x" "." gfx_regex ${gfx_pattern})
          foreach(target ${ARCH_LIST_COMMA_STR})
            if(target MATCHES ${gfx_regex})
              message(STATUS "Downloading AOTriton image ${image}.")
              set(__AOTRITON_DOWNLOAD_TARGET aotriton_image_${gfx_pattern})
              aotriton_download_image(${image} ${__AOTRITON_DOWNLOAD_TARGET})
              add_dependencies(${__AOTRITON_CHAINED_IMAGE} ${__AOTRITON_DOWNLOAD_TARGET})
              set(__AOTRITON_CHAINED_IMAGE ${__AOTRITON_DOWNLOAD_TARGET})
              break()
            endif()
          endforeach()
        endforeach()
      else()
      endif()


    else()
      # Use aotriton built during initial TE building/installation
      # When only need rebuild TE library itself
      unset(AOTRITON_LIB CACHE)
      find_library(AOTRITON_LIB NAMES aotriton aotriton${__AOTRITON_SUFFIX}_v2 PATHS ${AOTRITON_PATH}/lib REQUIRED NO_DEFAULT_PATH)
      add_library( aotriton SHARED IMPORTED )
      set_target_properties( aotriton PROPERTIES IMPORTED_LOCATION ${AOTRITON_LIB} )
      target_include_directories(aotriton INTERFACE ${AOTRITON_PATH}/include)
    endif()
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

if(USE_CUDA)
  option(NVTE_ENABLE_NVSHMEM "Compile with NVSHMEM library" OFF)
  if (NVTE_ENABLE_NVSHMEM)
      add_subdirectory(nvshmem_api)
      target_link_libraries(transformer_engine PUBLIC nvshmemapi)
      target_include_directories(transformer_engine PUBLIC ${NVSHMEMAPI_INCLUDE_DIR})
  endif()

# Hack to enable dynamic loading in cuDNN frontend
target_compile_definitions(transformer_engine PUBLIC NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING)
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
set_target_properties(transformer_engine PROPERTIES INSTALL_RPATH "$ORIGIN/lib;$ORIGIN/transformer_engine/lib")

if (USE_ROCM)
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/build_info.txt"
    "ROCM_VERSION: ${hip_VERSION_MAJOR}.${hip_VERSION_MINOR}\n"
    "GPU_TARGETS: ${CMAKE_HIP_ARCHITECTURES}\n"
  )
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/build_info.txt" DESTINATION "transformer_engine/")
endif()
