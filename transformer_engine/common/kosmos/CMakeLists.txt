# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information
# libkosmos_moe.a, the MoE training library: the Makefile's objects with the Makefile's compiler flags.

cmake_minimum_required(VERSION 3.21)
project(kosmos_moe LANGUAGES HIP)

set(KOSMOS_GPU_ARCH gfx950 CACHE STRING "GPU architecture (the Makefile's GPU_ARCH)")
if(NOT KOSMOS_HIPKITTENS_INCLUDE)
  set(KOSMOS_HIPKITTENS_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/hipkittens/include")
endif()
if(NOT EXISTS "${KOSMOS_HIPKITTENS_INCLUDE}/kittens.cuh")
  message(FATAL_ERROR "KOSMOS: no HipKittens headers at ${KOSMOS_HIPKITTENS_INCLUDE} (git submodule update --init)")
endif()
if(NOT KOSMOS_ROCM_PATH)
  if(ROCM_PATH)
    set(KOSMOS_ROCM_PATH "${ROCM_PATH}")
  elseif(DEFINED ENV{ROCM_PATH})
    set(KOSMOS_ROCM_PATH "$ENV{ROCM_PATH}")
  else()
    cmake_path(NORMAL_PATH CMAKE_HIP_COMPILER_ROCM_ROOT OUTPUT_VARIABLE KOSMOS_ROCM_PATH)
  endif()
endif()

# Kernels are validated with exactly these flags: nothing an including project sets for its own targets applies here.
set_property(DIRECTORY PROPERTY COMPILE_DEFINITIONS "")
set_property(DIRECTORY PROPERTY COMPILE_OPTIONS "")
set_property(DIRECTORY PROPERTY INCLUDE_DIRECTORIES "")
set(CMAKE_HIP_FLAGS "")
foreach(_config DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
  set(CMAKE_HIP_FLAGS_${_config} "")
endforeach()

function(kosmos_moe_objects name)
  cmake_parse_arguments(A "" "" "SOURCES;DEFINES;OPTIONS;INCLUDES" ${ARGN})
  add_library(${name} OBJECT ${A_SOURCES})
  set_source_files_properties(${A_SOURCES} PROPERTIES LANGUAGE HIP)
  set_target_properties(${name} PROPERTIES HIP_ARCHITECTURES ${KOSMOS_GPU_ARCH} HIP_STANDARD 20
                        HIP_STANDARD_REQUIRED ON HIP_EXTENSIONS OFF POSITION_INDEPENDENT_CODE ON)
  # CMake's HIP runtime target would add -D__HIP_ROCclr__=1, which hipcc does not pass
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.28)
    set_target_properties(${name} PROPERTIES HIP_RUNTIME_LIBRARY NONE)
  endif()
  target_compile_definitions(${name} PRIVATE ${A_DEFINES})
  target_compile_options(${name} PRIVATE ${A_OPTIONS} ${A_INCLUDES})
endfunction()

# MOE_LIB_FLAGS, -I as options so that none is dropped as an implicit include directory
set(_kosmos_moe_options -fno-gpu-rdc -O3 -Wno-unused-value -Wno-deprecated-declarations)
set(_kosmos_moe_includes -I${KOSMOS_HIPKITTENS_INCLUDE} -I${KOSMOS_ROCM_PATH}/include/hip -I${KOSMOS_ROCM_PATH}/include)
set(_kosmos_kernels src/moe/moe_train_api.cu src/moe/moe_train_api_ks.cu)
kosmos_moe_objects(kosmos_moe_swiglu SOURCES src/moe/moe.cu ${_kosmos_kernels} DEFINES KITTENS_CDNA4
                   OPTIONS ${_kosmos_moe_options} INCLUDES ${_kosmos_moe_includes})
kosmos_moe_objects(kosmos_moe_clamp10 SOURCES ${_kosmos_kernels} DEFINES KITTENS_CDNA4 MOE_ACT_CLAMP10=1
                   OPTIONS ${_kosmos_moe_options} INCLUDES ${_kosmos_moe_includes})
# COMM_FLAGS (host code; hipcc compiles it as HIP)
kosmos_moe_objects(kosmos_moe_comm SOURCES src/utils/kosmos_comm.cpp OPTIONS -O3 INCLUDES -I${KOSMOS_ROCM_PATH}/include)

add_library(kosmos_moe STATIC $<TARGET_OBJECTS:kosmos_moe_swiglu> $<TARGET_OBJECTS:kosmos_moe_clamp10>
            $<TARGET_OBJECTS:kosmos_moe_comm>)
set_target_properties(kosmos_moe PROPERTIES LINKER_LANGUAGE HIP)
