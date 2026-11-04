# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

# Builds the gfx1250 (MI450) attention library of 3rdparty/te-mi450-attn-kernels and exposes it as `mi450_attn`.
#
# The library is built by its own build (kernels/build.py, run through build_mi450_attn.py) into this build
# directory: one shared library holding every backend and the dispatcher, behind the C ABI of kernels/attn_abi.h.
# Its exported symbols are namespaced; TE fixes the namespace (MI450_ATTN_NAMESPACE) so it knows the library's name
# and compiles against kernels/attn_abi.h with ATTN_NS defined to it. The build step runs on every build and is a
# no-op when nothing changed.
#
# The submodule is private and is not checked out by `git submodule update --init --recursive`
# (`update = none` in .gitmodules). To build this backend, check it out explicitly:
#   git -c submodule.3rdparty/te-mi450-attn-kernels.update=checkout submodule update --init --recursive 3rdparty/te-mi450-attn-kernels

cmake_minimum_required(VERSION 3.21)

set(MI450_ATTN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/te-mi450-attn-kernels")
set(MI450_ATTN_NAMESPACE "te")

set(_mi450_arch "")
foreach(_arch IN LISTS CMAKE_HIP_ARCHITECTURES)
  if(_arch MATCHES "^gfx1250")
    set(_mi450_arch ${_arch})
    break()
  endif()
endforeach()

if(NOT _mi450_arch)
  message(STATUS "MI450 fused attn disabled (no gfx1250 in CMAKE_HIP_ARCHITECTURES)")
  set(USE_FUSED_ATTN_MI450 OFF PARENT_SCOPE)
  return()
endif()
if(NOT EXISTS "${MI450_ATTN_ROOT}/kernels/attn_abi.h" OR NOT EXISTS "${MI450_ATTN_ROOT}/kernels/build.py" OR
   NOT EXISTS "${MI450_ATTN_ROOT}/3rdparty/hipkittens/include/kittens.cuh")
  message(STATUS "MI450 fused attn disabled (3rdparty/te-mi450-attn-kernels is not checked out recursively)")
  set(USE_FUSED_ATTN_MI450 OFF PARENT_SCOPE)
  return()
endif()
if(NOT Python_EXECUTABLE)
  find_package(Python COMPONENTS Interpreter REQUIRED)
endif()

set(_mi450_build_dir "${CMAKE_CURRENT_BINARY_DIR}/${_mi450_arch}")
set(MI450_ATTN_LIB
    "${_mi450_build_dir}/lib/${MI450_ATTN_NAMESPACE}/libattn_${MI450_ATTN_NAMESPACE}.so")

add_custom_target(mi450_attn_build ALL
  COMMAND ${CMAKE_COMMAND} -E env "ROCM_PATH=${ROCM_PATH}"
          ${Python_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/build_mi450_attn.py"
          --root "${MI450_ATTN_ROOT}"
          --arch ${_mi450_arch}
          --build-dir "${_mi450_build_dir}"
          --namespace ${MI450_ATTN_NAMESPACE}
  BYPRODUCTS "${MI450_ATTN_LIB}"
  COMMENT "Building the MI450 attention library (3rdparty/te-mi450-attn-kernels) for ${_mi450_arch}"
  VERBATIM)

add_library(mi450_attn INTERFACE)
add_dependencies(mi450_attn mi450_attn_build)
target_include_directories(mi450_attn INTERFACE "${MI450_ATTN_ROOT}/kernels")
target_compile_definitions(mi450_attn INTERFACE ATTN_NS=${MI450_ATTN_NAMESPACE})
target_link_libraries(mi450_attn INTERFACE "${MI450_ATTN_LIB}")

# Next to the other attention libraries; libtransformer_engine.so finds it through its $ORIGIN/lib rpath.
install(FILES "${MI450_ATTN_LIB}" DESTINATION ${CMAKE_INSTALL_PREFIX}/transformer_engine/lib)

message(STATUS "MI450 fused attn: ${MI450_ATTN_ROOT} for ${_mi450_arch}, namespace ${MI450_ATTN_NAMESPACE}")
