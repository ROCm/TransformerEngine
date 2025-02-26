# This file was modified for portability to AMDGPU
# Copyright (c) 2022-2025, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

cmake_minimum_required(VERSION 3.21)

option(USE_ROCM "Use ROCm" ON)
option(USE_HIPBLASLT "Use HIPBLASLT" ON)
option(USE_ROCBLAS "Use ROCBLAS" ON)
option(USE_FUSED_ATTN_AOTRITON "Use aotriton backend" ON)
option(USE_FUSED_ATTN_AOTRITON_BUILD_GPU_KERNELS "Build AOTriton GPU kernels" OFF)
option(USE_FUSED_ATTN_CK "Use ck backend" ON)
set(USE_CUDA OFF)

if (USE_ROCM)
  #Check if ROCm exists
  if( ((EXISTS "/opt/rocm/") OR (EXISTS $ENV{ROCM_PATH})) )
    message("Using AMD Platform")
  else()
    message(FATAL_ERROR "Could not find HIPCC at '/opt/rocm'")
  endif()
    
  if (NOT USE_HIPBLASLT AND NOT USE_ROCBLAS)
    message(FATAL_ERROR "Need specify at least one GEMM library to use: HIPBLASLT or ROCBLAS")
  endif()

else()
  #Check if CUDA exists
  if (EXISTS "/bin/nvcc")
    set(USE_ROCM OFF)
    set(USE_CUDA ON)
    message("Using NVIDIA Platform")
  else()
    message(FATAL_ERROR "Could not find NVCC at '/bin/nvcc'")
  endif()
  
endif()

# Language options
if(USE_CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 80 89 90)
  endif()
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G")
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
      "${CMAKE_SOURCE_DIR}/../../3rdparty/cudnn-frontend/include")
  if(NOT EXISTS "${CUDNN_FRONTEND_INCLUDE_DIR}")
      message(FATAL_ERROR
              "Could not find cuDNN frontend API. "
              "Try running 'git submodule update --init --recursive' "
              "within the Transformer Engine source.")
  endif()
  include(${CMAKE_SOURCE_DIR}/../../3rdparty/cudnn-frontend/cmake/cuDNN.cmake)
else()
  set(CMAKE_CXX_STANDARD 17)
  project(transformer_engine LANGUAGES HIP CXX)

  # Disable Asserts In Code (Can't use asserts on HIP stack.)
  add_definitions(-DNDEBUG)
  add_definitions(-DUSE_ROCM)

  if(NOT DEFINED ENV{NVTE_ROCM_ARCH})
    SET(CMAKE_HIP_ARCHITECTURES gfx942)
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
  message(STATUS "USE_HIPBLASLT ${USE_HIPBLASLT} USE_ROCBLAS ${USE_ROCBLAS}")
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
       pycudnn.cpp
       transformer_engine.cpp
       common.cu
       transpose/cast_transpose.cu
       transpose/transpose.cu
       transpose/cast_transpose_fusion.cu
       transpose/transpose_fusion.cu
       transpose/multi_cast_transpose.cu
       activation/gelu.cu
       fused_attn/fused_attn_f16_max512_seqlen.cu
       fused_attn/fused_attn_f16_arbitrary_seqlen.cu
       activation/relu.cu
       activation/swiglu.cu
       fused_attn/fused_attn_fp8.cu
       fused_attn/fused_attn.cpp
       fused_attn/utils.cu
       gemm/cublaslt_gemm.cu
       layer_norm/ln_api.cpp
       layer_norm/ln_bwd_semi_cuda_kernel.cu
       layer_norm/ln_fwd_cuda_kernel.cu
       permutation/permutation.cu
       rmsnorm/rmsnorm_api.cpp
       rmsnorm/rmsnorm_bwd_semi_cuda_kernel.cu
       rmsnorm/rmsnorm_fwd_cuda_kernel.cu
       util/cast.cu
       util/padding.cu
       util/cuda_driver.cpp
       util/cuda_runtime.cpp
       util/rtc.cpp
       util/system.cpp
       fused_softmax/scaled_masked_softmax.cu
       fused_softmax/scaled_upper_triang_masked_softmax.cu
       fused_softmax/scaled_aligned_causal_masked_softmax.cu
       fused_rope/fused_rope.cu
       recipe/delayed_scaling.cu)
  add_library(transformer_engine SHARED ${transformer_engine_SOURCES})
else()
  list(APPEND transformer_engine_SOURCES
       transformer_engine.cpp
       common.cu
       transpose/cast_transpose.cu
       transpose/transpose.cu
       transpose/cast_transpose_fusion.cu
       transpose/transpose_fusion.cu
       transpose/multi_cast_transpose.cu
       activation/gelu.cu
       activation/relu.cu
       activation/swiglu.cu
       fused_attn_rocm/fused_attn.cpp
       fused_attn_rocm/fused_attn_aotriton.cpp
       fused_attn_rocm/fused_attn_ck.cpp
       fused_attn_rocm/utils.cpp
       gemm/cublaslt_gemm.cu
       gemm/rocm_gemm.cu
       layer_norm/ln_api.cpp
       layer_norm/ln_bwd_semi_cuda_kernel.cu
       layer_norm/ln_fwd_cuda_kernel.cu
       permutation/permutation.cu
       rmsnorm/rmsnorm_api.cpp
       rmsnorm/rmsnorm_bwd_semi_cuda_kernel.cu
       rmsnorm/rmsnorm_fwd_cuda_kernel.cu
       util/cast.cu
       util/padding.cu
       util/cuda_driver.cpp
       util/cuda_runtime.cpp
       util/rtc.cpp
       util/system.cpp
       fused_softmax/scaled_masked_softmax.cu
       fused_softmax/scaled_upper_triang_masked_softmax.cu
       fused_softmax/scaled_aligned_causal_masked_softmax.cu
       fused_rope/fused_rope.cu
       recipe/delayed_scaling.cu)

  # process source code files
  set(TE ${CMAKE_CURRENT_SOURCE_DIR}/../..)
  set(THIRDPARTY ${TE}/3rdparty)
  list(APPEND CMAKE_MODULE_PATH "${THIRDPARTY}/hipify_torch/cmake")
  include(Hipify)
  message(STATUS "CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")

  set(header_include_dir
      ${CMAKE_CURRENT_SOURCE_DIR}/include 
      ${CMAKE_CURRENT_SOURCE_DIR}/util
      ${CMAKE_CURRENT_SOURCE_DIR}/rmsnorm
      ${CMAKE_CURRENT_SOURCE_DIR}/layer_norm 
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
endif()
target_include_directories(transformer_engine PUBLIC
                           "${CMAKE_CURRENT_SOURCE_DIR}/include")

# Configure dependencies
if (USE_CUDA)
  target_link_libraries(transformer_engine PUBLIC
                      CUDA::cublas
                      CUDA::cudart)
  target_include_directories(transformer_engine PRIVATE
                           ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_include_directories(transformer_engine PRIVATE "${CUDNN_FRONTEND_INCLUDE_DIR}")
  
  # Hack to enable dynamic loading in cuDNN frontend
  target_compile_definitions(transformer_engine PUBLIC NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING)
else()
  if(USE_FUSED_ATTN_AOTRITON)
    # This is for GPU kernel downloading
    # The AOTriton C++ runtime will be built from ../../3rdparty/aotriton
    # Hence there is no need to add multiple ROCM version here
    set(__AOTRITON_INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/aotriton")
    set(__AOTRITON_SUFFIX "_TEprivate")
    if(NOT DEFINED AOTRITON_PATH)
      # # Install aotriton fused attn
      if(USE_FUSED_ATTN_AOTRITON_BUILD_GPU_KERNELS)
        set(AOTRITON_NOIMAGE_MODE OFF)
      else()
        set(AOTRITON_NOIMAGE_MODE ON)
      endif()
      set(aotriton_target_gpus "")
      # GPU Targets
      macro(translate_arch_to_gpu_names aotriton_target_gpus)
        set(gfx90a_key MI200)
        set(gfx942_key MI300X)
        set(gfx1100_key Navi31)

        foreach(X IN LISTS CMAKE_HIP_ARCHITECTURES)
          set(key ${X})
          string(APPEND key "_key")
          string(APPEND aotriton_target_gpus ${${key}})
          string(APPEND aotriton_target_gpus "|")
        endforeach()
      endmacro()
      translate_arch_to_gpu_names(aotriton_target_gpus)
      include(ExternalProject)
      ExternalProject_Add(aotriton_external
        SOURCE_DIR ../../3rdparty/aotriton
        LIST_SEPARATOR |
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${__AOTRITON_INSTALL_DIR}
        -DTARGET_GPUS=${aotriton_target_gpus}
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DAOTRITON_NO_PYTHON=ON
        -DAOTRITON_NAME_SUFFIX=${__AOTRITON_SUFFIX}
        -DAOTRITON_NOIMAGE_MODE=${AOTRITON_NOIMAGE_MODE}
        BUILD_BYPRODUCTS "${__AOTRITON_INSTALL_DIR}/lib/libaotriton${__AOTRITON_SUFFIX}_v2.so"
        )
      add_library(aotriton INTERFACE)
      add_dependencies(aotriton aotriton_external)
      target_link_libraries(aotriton INTERFACE ${__AOTRITON_INSTALL_DIR}/lib/libaotriton${__AOTRITON_SUFFIX}_v2.so)
      target_include_directories(aotriton INTERFACE ${__AOTRITON_INSTALL_DIR}/include)
      if(NOT USE_FUSED_ATTN_AOTRITON_BUILD_GPU_KERNELS)
        set(__AOTRITON_VER "0.8.2b")
        set(__AOTRITON_SHA256 "66445e6b0209b9f4080743b839cc9d424054dc5c8d07363f9f27f109231c324a")
        string(CONCAT __AOTRITON_URL "https://github.com/ROCm/aotriton/releases/download/"
                                     "${__AOTRITON_VER}/aotriton-"
                                     "${__AOTRITON_VER}-manylinux_2_28"
                                     "_x86_64-rocm6.2"
                                     "-shared.tar.gz")
        ExternalProject_Add(aotriton_images
                            URL "${__AOTRITON_URL}"
                            URL_HASH SHA256=${__AOTRITON_SHA256}
                            SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/aotriton_tarball
                            CONFIGURE_COMMAND ""
                            BUILD_COMMAND ""
                            INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_directory
                            "${CMAKE_CURRENT_BINARY_DIR}/aotriton_tarball/lib/aotriton.images"
                            "${__AOTRITON_INSTALL_DIR}/lib/aotriton.images")
        add_dependencies(aotriton aotriton_images)
      endif()
      install(DIRECTORY
              ${__AOTRITON_INSTALL_DIR}/lib
              DESTINATION ${CMAKE_INSTALL_PREFIX}/transformer_engine
              PATTERN "cmake" EXCLUDE
              PATTERN "libaotriton${__AOTRITON_SUFFIX}_v2.so" EXCLUDE)
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
    if(NOT DEFINED CK_FUSED_ATTN_PATH)
      set(CK_FUSED_ATTN_TARGET_GPUS ${CMAKE_HIP_ARCHITECTURES} CACHE STRING "Target arch to compile ck fused attn backend")
      set(CK_FUSED_ATTN_INSTALL_PREFIX "transformer_engine" CACHE STRING "ck_fused_attn install prefix in TE")      
      add_subdirectory(ck_fused_attn ${CMAKE_CURRENT_BINARY_DIR}/ck_fused_attn)
    else()
      # Use CK built during initial TE building/installation
      # When only need rebuild TE library itself
      unset(CK_FUSED_ATTN_LIB CACHE)
      find_library(CK_FUSED_ATTN_LIB NAMES ck_fused_attn PATHS ${CK_FUSED_ATTN_PATH}/lib REQUIRED NO_DEFAULT_PATH)
      add_library( ck_fused_attn STATIC IMPORTED )
      set_target_properties( ck_fused_attn PROPERTIES IMPORTED_LOCATION ${CK_FUSED_ATTN_LIB} )
      target_include_directories(ck_fused_attn INTERFACE ${CK_FUSED_ATTN_PATH}/include)
    endif()
  endif()

  find_package(hip)
  list(APPEND transformer_engine_LINKER_LIBS hip::host hip::device roctx64)
  if(USE_HIPBLASLT)
    find_package(hipblaslt)
    target_compile_definitions(transformer_engine PUBLIC USE_HIPBLASLT)
    list(APPEND transformer_engine_LINKER_LIBS roc::hipblaslt)
  endif()
  if(USE_ROCBLAS)
    find_package(rocblas)
    target_compile_definitions(transformer_engine PUBLIC USE_ROCBLAS)
    list(APPEND transformer_engine_LINKER_LIBS roc::rocblas)
  endif()
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
                            PROPERTIES
                            COMPILE_OPTIONS "--use_fast_math")
if(USE_CUDA)
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
  foreach(rocm_arch ${CMAKE_HIP_ARCHITECTURES})
    # if CMAKE_CXX_FLAGS has --offload-arch set already, better to rm first
    set(HIP_HCC_FLAGS "${HIP_HCC_FLAGS} --offload-arch=${rocm_arch}")
  endforeach()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${HIP_HCC_FLAGS}")
  # Hide non-necessary symbols in shared object.
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/libtransformer_engine.version")
endif()

# Install library
install(TARGETS transformer_engine DESTINATION .)
set_target_properties(transformer_engine PROPERTIES INSTALL_RPATH "$ORIGIN/lib;$ORIGIN/transformer_engine/lib")
