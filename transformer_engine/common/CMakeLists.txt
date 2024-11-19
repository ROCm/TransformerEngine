# This file was modified for portability to AMDGPU
# Copyright (c) 2022-2024, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

cmake_minimum_required(VERSION 3.21)

option(USE_ROCM "Use ROCm" OFF)
option(USE_HIPBLASLT "Use HIPBLASLT" ON)
option(USE_ROCBLAS "Use ROCBLAS" ON)
option(USE_FUSED_ATTN_AOTRITON "Use aotriton backend" ON)
option(USE_FUSED_ATTN_CK "Use ck backend" ON)

if(NOT USE_ROCM)
  if(((EXISTS "/opt/rocm/") OR (EXISTS $ENV{ROCM_PATH})) AND NOT (EXISTS "/bin/nvcc"))
    message("AMD GPU detected.")
    set(USE_ROCM ON)
  endif()
endif()

if (USE_ROCM)
  if (NOT USE_HIPBLASLT AND NOT USE_ROCBLAS)
    message(FATAL_ERROR "Need specify at least one GEMM library to use: HIPBLASLT or ROCBLAS")
  endif()
  unset(USE_CUDA)
else()
  set(USE_CUDA TRUE)
endif()

if(USE_CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 80 89 90)
  endif()

  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)

  project(transformer_engine LANGUAGES CUDA CXX)

  if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G")
  endif()

  find_package(CUDAToolkit REQUIRED)

  # Check for cuDNN frontend API
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

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
include_directories(${PROJECT_SOURCE_DIR}/..)

# Configure Transformer Engine library
set(transformer_engine_SOURCES)
if(USE_CUDA)
  list(APPEND transformer_engine_SOURCES
       pycudnn.cpp
       transformer_engine.cpp
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
       rmsnorm/rmsnorm_api.cpp
       rmsnorm/rmsnorm_bwd_semi_cuda_kernel.cu
       rmsnorm/rmsnorm_fwd_cuda_kernel.cu
       util/cast.cu
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
       rmsnorm/rmsnorm_api.cpp
       rmsnorm/rmsnorm_bwd_semi_cuda_kernel.cu
       rmsnorm/rmsnorm_fwd_cuda_kernel.cu
       util/cast.cu
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
  message("${message_line}")
  message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
  message(STATUS "PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")

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

# process include header files
target_include_directories(transformer_engine PUBLIC
                           "${CMAKE_CURRENT_SOURCE_DIR}/include")

# Configure dependencies
if (USE_CUDA)
  target_compile_definitions(transformer_engine PUBLIC NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING)

  target_link_libraries(transformer_engine PUBLIC
                      CUDA::cublas
                      CUDA::cuda_driver
                      CUDA::cudart)
  target_include_directories(transformer_engine PRIVATE
                           ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_include_directories(transformer_engine PRIVATE "${CUDNN_FRONTEND_INCLUDE_DIR}")
else()
  if(USE_FUSED_ATTN_AOTRITON)
    if(NOT DEFINED AOTRITON_PATH)
      # Install aotriton fused attn
      set(AOTRITON_NO_PYTHON ON)
      set(AOTRITON_COMPRESS_KERNEL OFF)
      set(AOTRITON_NO_SHARED ON)
      set(TARGET_GPUS "MI300X" CACHE STRING "Target arch to compile aotriton fused attn backend (use trade names)")
      add_subdirectory(../../3rdparty/aotriton ${CMAKE_CURRENT_BINARY_DIR}/aotriton)
    else()
      # Use aotriton built during initial TE building/installation
      # When only need rebuild TE library itself
      unset(AOTRITON_LIB CACHE)
      find_library(AOTRITON_LIB NAMES aotriton aotriton_v2 PATHS ${AOTRITON_PATH}/lib REQUIRED NO_DEFAULT_PATH)
      add_library( aotriton STATIC IMPORTED )
      set_target_properties( aotriton PROPERTIES IMPORTED_LOCATION ${AOTRITON_LIB} )
      target_include_directories(aotriton INTERFACE ${AOTRITON_PATH}/include)
    endif()
  endif()

  if(USE_FUSED_ATTN_CK)
    if(NOT DEFINED CK_FUSED_ATTN_PATH)
      set(CK_FUSED_ATTN_TARGET_GPUS ${CMAKE_HIP_ARCHITECTURES} CACHE STRING "Target arch to compile ck fused attn backend")
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

# Make header files with C++ strings
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
endif()

# Install library
install(TARGETS transformer_engine DESTINATION .)

