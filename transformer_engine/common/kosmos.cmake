# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

# KOSMOS fused expert-parallel MoE (gfx950), included by CMakeLists.txt when USE_KOSMOS_MOE is on (the
# default; NVTE_CMAKE_EXTRA_ARGS="-DUSE_KOSMOS_MOE=OFF" leaves it out).
#
# kosmos/ is a byte-identical copy of KOSMOS (kosmos/UPSTREAM, kosmos/SHA256SUMS). Its library is built
# by KOSMOS's own Makefile, so the flags (per-object defines, -disable-machine-licm on the objects that
# need it) are exactly the ones KOSMOS was validated with, against its own HipKittens pin
# (3rdparty/hipkittens_kosmos). The Python module transformer_engine_kosmos is KOSMOS's own binding,
# python/kosmos.cpp, compiled under that module name. Neither is linked into libtransformer_engine,
# so importing Transformer Engine never loads KOSMOS.

if(NOT "gfx950" IN_LIST CMAKE_HIP_ARCHITECTURES)
  message(STATUS "KOSMOS MoE disabled (gfx950 not in CMAKE_HIP_ARCHITECTURES)")
  set(USE_KOSMOS_MOE OFF)
  return()
endif()

set(KOSMOS_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/kosmos")
set(KOSMOS_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/kosmos")
set(KOSMOS_HIPKITTENS_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/hipkittens_kosmos/include")
set(KOSMOS_LIBRARY "${KOSMOS_BUILD_DIR}/libkosmos_moe.a")
if(NOT EXISTS "${KOSMOS_HIPKITTENS_INCLUDE}/kittens.cuh")
  message(FATAL_ERROR
    "Could not find KOSMOS's HipKittens headers at ${KOSMOS_HIPKITTENS_INCLUDE}. "
    "Try running 'git submodule update --init --recursive', or configure with -DUSE_KOSMOS_MOE=OFF.")
endif()
find_program(KOSMOS_MAKE NAMES make gmake REQUIRED)
if(DEFINED ENV{MAX_JOBS})
  set(_kosmos_jobs "$ENV{MAX_JOBS}")
else()
  cmake_host_system_information(RESULT _kosmos_jobs QUERY NUMBER_OF_LOGICAL_CORES)
endif()

# The archive only: KOSMOS's `make moe` also links moe_linkcheck against the unversioned
# libamdhip64.so, which TheRock's pip SDK does not ship; linking transformer_engine_kosmos below
# checks the same symbols. make decides what is stale, so the step always runs.
include(ExternalProject)
ExternalProject_Add(kosmos_moe_lib
  SOURCE_DIR "${KOSMOS_SOURCE_DIR}"
  BINARY_DIR "${KOSMOS_BUILD_DIR}"
  CONFIGURE_COMMAND ""
  BUILD_COMMAND "${KOSMOS_MAKE}" -C "${KOSMOS_SOURCE_DIR}" -j${_kosmos_jobs} "${KOSMOS_LIBRARY}"
                "KOSMOS_BUILD=${KOSMOS_BUILD_DIR}" "HK_INC=${KOSMOS_HIPKITTENS_INCLUDE}"
                "ROCM_PATH=${ROCM_PATH}" "HIPCXX=${ROCM_PATH}/bin/hipcc" GPU_ARCH=gfx950
  INSTALL_COMMAND ""
  BUILD_ALWAYS ON
  BUILD_BYPRODUCTS "${KOSMOS_LIBRARY}"
  USES_TERMINAL_BUILD ON)

# python/kosmos.cpp declares PYBIND11_MODULE(kosmos, mod); the module is built from a renamed copy so
# that the vendored file stays byte-identical. configure_file rewrites the copy only when it changes.
set(_kosmos_binding_in "${KOSMOS_SOURCE_DIR}/python/kosmos.cpp")
set(_kosmos_binding "${KOSMOS_BUILD_DIR}/transformer_engine_kosmos.cpp")
file(READ "${_kosmos_binding_in}" _kosmos_src)
string(REPLACE "PYBIND11_MODULE(kosmos, mod)" "PYBIND11_MODULE(transformer_engine_kosmos, mod)"
       _kosmos_renamed "${_kosmos_src}")
if(_kosmos_renamed STREQUAL _kosmos_src)
  message(FATAL_ERROR "${_kosmos_binding_in} no longer declares PYBIND11_MODULE(kosmos, mod)")
endif()
file(WRITE "${KOSMOS_BUILD_DIR}/transformer_engine_kosmos.cpp.in" "${_kosmos_renamed}")
configure_file("${KOSMOS_BUILD_DIR}/transformer_engine_kosmos.cpp.in" "${_kosmos_binding}" COPYONLY)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_kosmos_binding_in}")

execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import pybind11; print(pybind11.get_include())"
  OUTPUT_VARIABLE _kosmos_pybind11_include
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _kosmos_pybind11_rc)
if(NOT _kosmos_pybind11_rc EQUAL 0)
  message(FATAL_ERROR "KOSMOS MoE needs pybind11 in ${Python_EXECUTABLE} (pip install pybind11), "
                      "or configure with -DUSE_KOSMOS_MOE=OFF.")
endif()

# The same flags as KOSMOS's own `make python`: HIP, C++20, -O2, hidden visibility, the archive's
# symbols kept out of the module's dynamic symbol table.
Python_add_library(transformer_engine_kosmos MODULE WITH_SOABI "${_kosmos_binding}")
set_source_files_properties("${_kosmos_binding}" PROPERTIES LANGUAGE HIP)
set_target_properties(transformer_engine_kosmos PROPERTIES
  HIP_ARCHITECTURES gfx950
  HIP_STANDARD 20
  HIP_VISIBILITY_PRESET hidden
  CXX_VISIBILITY_PRESET hidden)
target_include_directories(transformer_engine_kosmos PRIVATE
  "${KOSMOS_SOURCE_DIR}/src/moe" "${_kosmos_pybind11_include}")
target_compile_options(transformer_engine_kosmos PRIVATE -O2)
add_dependencies(transformer_engine_kosmos kosmos_moe_lib)
target_link_libraries(transformer_engine_kosmos PRIVATE "${KOSMOS_LIBRARY}" hip::host)
target_link_options(transformer_engine_kosmos PRIVATE "LINKER:--exclude-libs,ALL")
# Inside the transformer_engine package (transformer_engine.transformer_engine_kosmos): a top-level
# module installed by CMake is not reachable through an editable install's import hooks.
te_set_rocm_rpath(transformer_engine_kosmos)
install(TARGETS transformer_engine_kosmos DESTINATION transformer_engine)
message(STATUS "KOSMOS MoE: ${KOSMOS_SOURCE_DIR} with HipKittens ${KOSMOS_HIPKITTENS_INCLUDE}")
