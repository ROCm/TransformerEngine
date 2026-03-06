# Copyright (c) 2022-2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.21)
set(CMAKE_CXX_STANDARD 17)
project(aotriton LANGUAGES CXX)

# The AOTriton C++ runtime will be built from {TE}/3rdparty/aotriton
# Hence there is no need to add multiple ROCM version here

if(DEFINED ENV{AOTRITON_PATH})
    set(AOTRITON_PATH $ENV{AOTRITON_PATH})
endif()

set(__AOTRITON_INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/aotriton")
set(__AOTRITON_SUFFIX "_TEprivate")

if(NOT DEFINED AOTRITON_PATH)

    set(__AOTRITON_VER "0.11.2b")
    set(__AOTRITON_IMAGE_LIST
        "amd-gfx942"
        "amd-gfx950"
    )
    set(__AOTRITON_IMAGE_SHA256_LIST
        "0a7bcee19d3bb6d548732248c3234f7b92736c2ab7a7aae65294b87a7fd64c06" # amd-gfx942
        "c1ba3bfe84217fd67df3dd1f8b67c80a7f7b33d0ad4d74b41d6567036e032ace" # amd-gfx950
    )
    set(__AOTRITON_BASE_URL "https://github.com/ROCm/aotriton/releases/download/")  # @lint-ignore
    include(ExternalProject)

    # Download GPU kernels for a specific target
    function(aotriton_download_image image project)
        list(FIND __AOTRITON_IMAGE_LIST ${image} index)
        list(GET __AOTRITON_IMAGE_SHA256_LIST ${index} __AOTRITON_IMAGE_SHA256)

        string(CONCAT __AOTRITON_FILE
                "aotriton-${__AOTRITON_VER}-images-"
                "${image}.tar.gz")
        string(CONCAT __AOTRITON_URL
                "${__AOTRITON_BASE_URL}"
                "${__AOTRITON_VER}/${__AOTRITON_FILE}")

        # Set up directories
        set(__AOTRITON_DOWNLOAD_DIR ${CMAKE_CURRENT_BINARY_DIR}/aotriton/download/${image})
        set(__AOTRITON_EXTRACT_DIR ${CMAKE_CURRENT_BINARY_DIR}/aotriton/image/${image})

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

    # Build the AOTriton runtime from source with custom suffix to avoid
    # potential conflict with libaotriton as provided by PyTorch
    function(aotriton_build_from_source)
        get_git_commit(${TE}/3rdparty/aotriton AOTRITON_SHA)
        ExternalProject_Add(aotriton_external
            LIST_SEPARATOR ","
            SOURCE_DIR ${TE}/3rdparty/aotriton
            CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${__AOTRITON_INSTALL_DIR}
            -DAOTRITON_TARGET_ARCH=${ARCH_LIST_COMMA_STR}
            -DGPU_TARGETS=${ARCH_LIST_COMMA_STR}
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DAOTRITON_NO_PYTHON=ON
            -DAOTRITON_NAME_SUFFIX=${__AOTRITON_SUFFIX}
            -DAOTRITON_NOIMAGE_MODE=ON
            -DTE_AOTRITON_COMMIT_SHA1=${AOTRITON_SHA}
            -DCMAKE_PROJECT_INCLUDE=${CMAKE_CURRENT_LIST_DIR}/aotriton_custom.cmake
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

    add_library(aotriton INTERFACE)
    message(STATUS "Building AOTriton from source.")
    string(REPLACE ";" "," ARCH_LIST_COMMA_STR "${CMAKE_HIP_ARCHITECTURES}")
    aotriton_build_from_source()

    # Download GPU kernels if needed
    if(NOT USE_FUSED_ATTN_AOTRITON_BUILD_GPU_KERNELS)
    message(STATUS "Downloading AOTriton GPU Kernels.")
    foreach(image ${__AOTRITON_IMAGE_LIST})
        string(SUBSTRING ${image} 7 -1 gfx_pattern)
        string(REPLACE "x" "." gfx_regex ${gfx_pattern})
        foreach(target ${ARCH_LIST_COMMA_STR})
        if(target MATCHES ${gfx_regex})
            message(STATUS "Downloading AOTriton image ${image}.")
            set(__AOTRITON_DOWNLOAD_TARGET aotriton_image_${gfx_pattern})
            aotriton_download_image(${image} ${__AOTRITON_DOWNLOAD_TARGET})
            add_dependencies(aotriton_external ${__AOTRITON_DOWNLOAD_TARGET})
            break()
        endif()
        endforeach()
    endforeach()
    else()
    endif()

else()
    # Use aotriton built during initial TE building/installation
    # When only need rebuild TE library itself
    message(STATUS "Using existing AOTriton lib at $ENV{AOTRITON_PATH}")
    unset(AOTRITON_LIB CACHE)
    find_library(AOTRITON_LIB NAMES aotriton aotriton${__AOTRITON_SUFFIX}_v2 PATHS ${AOTRITON_PATH} REQUIRED NO_DEFAULT_PATH)
    add_library( aotriton SHARED IMPORTED )
    set_target_properties( aotriton PROPERTIES IMPORTED_LOCATION ${AOTRITON_LIB} )
    target_include_directories(aotriton INTERFACE ${AOTRITON_PATH}/include)
endif()
