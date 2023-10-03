# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

if(USE_CUDA)
  pybind11_add_module(
      transformer_engine_tensorflow
      ${CMAKE_CURRENT_SOURCE_DIR}/csrc/extensions.cu
  )
else()
  message("${message_line}")
  message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
  message(STATUS "PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")

  set(TE ${CMAKE_CURRENT_SOURCE_DIR}/../..)
  set(THIRDPARTY ${TE}/3rdparty)
  list(APPEND CMAKE_MODULE_PATH "${THIRDPARTY}/hipify_torch/cmake")
  include(Hipify)
  message(STATUS "CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")

  message(STATUS "HIPIFY CUDA_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
  hipify(CUDA_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
  get_hipified_list("${transformer_engine_SOURCES}" te_hip_sources)
  message("${message_line}")
  message(STATUS "nvte hipified sources: ${te_hip_sources}")
  pybind11_add_module(
      transformer_engine_tensorflow
      ${CMAKE_CURRENT_SOURCE_DIR}/csrc/extensions.hip
  )
endif()
set_target_properties(transformer_engine_tensorflow PROPERTIES LINKER_LANGUAGE CXX)

add_library(
    _get_stream SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/get_stream_op.cpp
)

# Includes
execute_process(COMMAND ${Python_EXECUTABLE} -c "import tensorflow as tf; print(tf.sysconfig.get_include())"
                OUTPUT_VARIABLE Tensorflow_INCLUDE_DIRS OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${Python_EXECUTABLE} -c "import numpy as np; print(np.get_include())"
                OUTPUT_VARIABLE Numpy_INCLUDE_DIRS OUTPUT_STRIP_TRAILING_WHITESPACE)

target_include_directories(transformer_engine_tensorflow PRIVATE
    ${Tensorflow_INCLUDE_DIRS}
    ${Tensorflow_INCLUDE_DIRS}/external/farmhash_archive/src
    ${Numpy_INCLUDE_DIRS})

if(USE_ROCM)
  target_include_directories(transformer_engine_tensorflow PUBLIC /opt)
  target_include_directories(_get_stream PUBLIC /opt)
endif()
    
target_include_directories(_get_stream PRIVATE ${Tensorflow_INCLUDE_DIRS})

# Libraries
execute_process(COMMAND ${Python_EXECUTABLE} -c "import tensorflow as tf; print(tf.__file__)"
                OUTPUT_VARIABLE Tensorflow_LIB_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)
get_filename_component(Tensorflow_LIB_PATH ${Tensorflow_LIB_PATH} DIRECTORY)
list(APPEND TF_LINKER_LIBS "${Tensorflow_LIB_PATH}/libtensorflow_framework.so.2")
list(APPEND TF_LINKER_LIBS "${Tensorflow_LIB_PATH}/libtensorflow_cc.so.2")
list(APPEND TF_LINKER_LIBS "${Tensorflow_LIB_PATH}/python/_pywrap_tensorflow_internal.so")

if(USE_CUDA)
  target_link_libraries(
      transformer_engine_tensorflow PRIVATE
      ${TF_LINKER_LIBS} CUDA::cudart CUDA::cublas CUDA::cublasLt transformer_engine
  )
else()
  find_package(hip)
  find_package(rocblas)
  list(APPEND transformer_engine_tensorflow_LINKER_LIBS roc::rocblas hip::host hip::device)
  message("${message_line}")
  message(STATUS "TF_LINKER_LIBS: ${TF_LINKER_LIBS}")
  message(STATUS "transformer_engine_tensorflow_LINKER_LIBS: ${transformer_engine_tensorflow_LINKER_LIBS}")
  target_link_libraries(
      transformer_engine_tensorflow PRIVATE
      ${TF_LINKER_LIBS} ${transformer_engine_tensorflow_LINKER_LIBS} transformer_engine
  )
endif()

target_link_libraries(_get_stream PRIVATE ${TF_LINKER_LIBS})
if(USE_ROCM)
  target_compile_definitions(_get_stream PUBLIC TENSORFLOW_USE_ROCM)
  target_link_libraries(_get_stream PRIVATE roc::rocblas hip::host hip::device)
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

# Install library
install(TARGETS transformer_engine_tensorflow DESTINATION .)
install(TARGETS _get_stream DESTINATION .)
