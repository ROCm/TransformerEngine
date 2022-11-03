# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

cmake_minimum_required(VERSION 3.18)

set(message_line
    "-------------------------------------------------------------")
message("${message_line}")

option(USE_ROCM "Use ROCm" OFF)
if(((EXISTS "/opt/rocm/") OR (EXISTS $ENV{ROCM_PATH})) AND NOT (EXISTS "/bin/nvcc"))
   message("AMD GPU detected.")
   set(USE_ROCM ON)
endif()

message("${message_line}")
message("USE_ROCM ${USE_ROCM}")

#if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
#  set(CMAKE_CUDA_ARCHITECTURES 70 80 90)
#endif()

set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CUDA_STANDARD 17)
#set(CMAKE_CUDA_STANDARD_REQUIRED ON)


if(USE_ROCM)
  project(
    transformer_engine
    LANGUAGES CXX)
else()
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  project(
    transformer_engine
    LANGUAGES CUDA CXX)
endif()


# find_package(Torch REQUIRED)
# find_package(PythonExtensions REQUIRED)

if(DEFINED GLIBCXX_USE_CXX11_ABI)
  if(${GLIBCXX_USE_CXX11_ABI} EQUAL 1)
    set(CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
  endif()
  message("${CMAKE_CXX_FLAGS}")
endif()

#
# Toch Cuda Extensions are normally compiled with the flags below. However we
# disabled -D__CUDA_NO_HALF_CONVERSIONS__ here as it caused "error: no suitable
# constructor exists to convert from "int" to "__half" errors in
# gen_embedding_forward_quantized_split_[un]weighted_codegen_cuda.cu
#

set(TORCH_CUDA_OPTIONS
    --expt-relaxed-constexpr -D__CUDA_NO_HALF_OPERATORS__
    # -D__CUDA_NO_HALF_CONVERSIONS__
    -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__)


message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")

# project(transformer_engine LANGUAGES CUDA CXX)
# set(TE ${CMAKE_CURRENT_SOURCE_DIR}/../..)
set(TE ${PROJECT_SOURCE_DIR}/../..)
set(THIRDPARTY ${TE}/third_party)

if(USE_ROCM)
  list(APPEND CMAKE_MODULE_PATH "${TE}/cmake"
       "${THIRDPARTY}/hipify_torch/cmake")
  include(Hip)
  include(Hipify)

  message("${message_line}")
  message(STATUS "hip found ${HIP_FOUND}")
endif()


#list(APPEND CMAKE_CUDA_FLAGS "--threads 4")
#if (CMAKE_BUILD_TYPE STREQUAL "Debug")
#  list(APPEND CMAKE_CUDA_FLAGS "-G")
#endif()

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${HIP_HCC_FLAGS}")
message(STATUS "HIP_HCC_FLAGS: ${HIP_HCC_FLAGS}")

set(TE_HIP_FILES
    ${PROJECT_SOURCE_DIR}/transformer_engine.cpp
    ${PROJECT_SOURCE_DIR}/transpose/cast_transpose.hip
    ${PROJECT_SOURCE_DIR}/transpose/transpose.hip
    ${PROJECT_SOURCE_DIR}/transpose/cast_transpose_fusion.hip
    ${PROJECT_SOURCE_DIR}/activation/gelu.hip
    ${PROJECT_SOURCE_DIR}/gemm/cublaslt_gemm.hip
    ${PROJECT_SOURCE_DIR}/layer_norm/ln_api_hip.cpp
    ${PROJECT_SOURCE_DIR}/layer_norm/ln_bwd_semi_hip_kernel.hip
    ${PROJECT_SOURCE_DIR}/layer_norm/ln_fwd_hip_kernel.hip
    ${PROJECT_SOURCE_DIR}/util/cast.hip)

set_source_files_properties(${TE_HIP_FILES} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)

#add_library(transformer_engine SHARED
#                               transformer_engine.cpp
#                               transpose/cast_transpose.hip
#                               transpose/transpose.hip
#                               transpose/cast_transpose_fusion.hip
#                               activation/gelu.hip
#                               gemm/cublaslt_gemm.hip
#                               layer_norm/ln_api_hip.cpp
#                               layer_norm/ln_bwd_semi_hip_kernel.hip
#                               layer_norm/ln_fwd_hip_kernel.hip
#                               util/cast.hip)

hip_add_library(transformer_engine
	SHARED
        ${TE_HIP_FILES}
        ${FBGEMM_HIP_HCC_LIBRARIES}
	HIPCC_OPTIONS
        ${HIP_HCC_FLAGS})
target_include_directories(transformer_engine PUBLIC "${PROJECT_SOURCE_DIR}/include" ${FBGEMM_HIP_INCLUDE})

# find_package(CUDAToolkit REQUIRED cublas)
find_package(rocblas REQUIRED)

list(APPEND transformer_engine_LINKER_LIBS rocblas rocdart)
target_link_libraries(transformer_engine PUBLIC ${transformer_engine_LINKER_LIBS})

target_include_directories(transformer_engine PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
