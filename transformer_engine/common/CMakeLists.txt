# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# process source code files
set(transformer_engine_SOURCES)
if(USE_CUDA)
list(APPEND transformer_engine_SOURCES
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
     fused_softmax/scaled_upper_triang_masked_softmax.cu)
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
     gemm/cublaslt_gemm.cu
     layer_norm/ln_api.cpp
     layer_norm/ln_bwd_semi_cuda_kernel.cu
     layer_norm/ln_fwd_cuda_kernel.cu
     rmsnorm/rmsnorm_api.cpp
     rmsnorm/rmsnorm_bwd_semi_cuda_kernel.cu
     rmsnorm/rmsnorm_fwd_cuda_kernel.cu
     util/cast.cu
     fused_softmax/scaled_masked_softmax.cu
     fused_softmax/scaled_upper_triang_masked_softmax.cu)
endif()

if(USE_CUDA)
  add_library(transformer_engine SHARED ${transformer_engine_SOURCES})
else()
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
      HEADER_INCLUDE_DIR ${header_include_dir})
  get_hipified_list("${transformer_engine_SOURCES}" te_hip_sources)
  message("${message_line}")
  message(STATUS "nvte hipified sources: ${te_hip_sources}")

  add_library(transformer_engine SHARED ${te_hip_sources})
endif()

# process include header files
target_include_directories(transformer_engine PUBLIC
                           "${CMAKE_CURRENT_SOURCE_DIR}/include")
if(USE_CUDA)
  target_include_directories(transformer_engine PRIVATE
                             ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_include_directories(transformer_engine PRIVATE "${CUDNN_FRONTEND_INCLUDE_DIR}")
  
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
  list(GET CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES 0 cuda_include_path)
  make_string_header("${cuda_include_path}"
                     string_path_cuda_include)
  make_string_header_from_file(utils.cuh
                               string_code_utils_cuh)
  make_string_header_from_file(transpose/rtc/transpose.cu
                               string_code_transpose_rtc_transpose_cu)
  target_include_directories(transformer_engine PRIVATE
                             "${CMAKE_CURRENT_BINARY_DIR}/string_headers")
else()

endif()

# find and process package dependency
if(USE_CUDA)
  find_package(CUDAToolkit REQUIRED cublas)
  # Check for cuDNN frontend API
  set(CUDNN_FRONTEND_INCLUDE_DIR
      "${CMAKE_SOURCE_DIR}/../3rdparty/cudnn-frontend/include")
  if(NOT EXISTS "${CUDNN_FRONTEND_INCLUDE_DIR}")
      message(FATAL_ERROR
              "Could not find cuDNN frontend API. "
              "Try running 'git submodule update --init --recursive' "
              "within the Transformer Engine source.")
  endif()

  list(APPEND transformer_engine_LINKER_LIBS 
       CUDA::cublas
       CUDA::cuda_driver
       CUDA::cudart
       CUDA::nvrtc
       CUDA::nvToolsExt
       CUDNN::cudnn)
else()
  find_package(hip)
  find_package(rocblas)
  if(USE_HIPBLASLT)
    find_package(hipblaslt)
    target_compile_definitions(transformer_engine PUBLIC USE_HIPBLASLT)
    list(APPEND transformer_engine_LINKER_LIBS roc::rocblas hip::host hip::device roc::hipblaslt)
  else()
    list(APPEND transformer_engine_LINKER_LIBS roc::rocblas hip::host hip::device)
  endif()
endif()

target_link_libraries(transformer_engine PUBLIC ${transformer_engine_LINKER_LIBS})

# Compiler options
if(USE_CUDA)
  set_source_files_properties(fused_softmax/scaled_masked_softmax.cu
                              fused_softmax/scaled_upper_triang_masked_softmax.cu
                              PROPERTIES
                              COMPILE_OPTIONS "--use_fast_math")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
else()
endif()

# Install library
install(TARGETS transformer_engine DESTINATION .)
