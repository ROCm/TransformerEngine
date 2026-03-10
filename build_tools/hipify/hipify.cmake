# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

set(_HIPIFY_CMAKE_DIR "${CMAKE_CURRENT_LIST_DIR}")


function(TE_Hipify SRC_DIR)
    # Create result file
    set(hipify_result "${CMAKE_BINARY_DIR}/hipify_result.json")
    
    # Call Python script
    execute_process(
        COMMAND python3 "${_HIPIFY_CMAKE_DIR}/hipify.py" hipify
        --src-dir "${SRC_DIR}"
        --hipify-result "${hipify_result}"
        RESULT_VARIABLE script_result
    )
    
    if(NOT script_result EQUAL 0)
        message(FATAL_ERROR "Python script failed with code ${script_result}")
    endif()
endfunction()


function(TE_GetHipifiedSources SOURCE_LIST BASE_PATH OUTPUT_VARIABLE)
    # Create a temporary file
    string(RANDOM LENGTH 8 RANDOM_SUFFIX)
    set(list_file "${CMAKE_BINARY_DIR}/source_list_${RANDOM_SUFFIX}.txt")
    
    # Write list to temp file
    string(REPLACE ";" "\n" list_content "${SOURCE_LIST}")
    file(WRITE "${list_file}" "${list_content}")
    
    set(hipify_result "${CMAKE_BINARY_DIR}/hipify_result.json")

    # Call Python script
    execute_process(
        COMMAND python3 "${_HIPIFY_CMAKE_DIR}/hipify.py" get_sources
        --hipify-result "${hipify_result}"
        --sources "${list_file}"
        --base-path "${BASE_PATH}"
        RESULT_VARIABLE script_result
    )
    
    if(NOT script_result EQUAL 0)
        message(FATAL_ERROR "Python script failed with code ${script_result}")
    endif()

    # Read result from output file
    file(READ "${list_file}" result_content)
    string(REPLACE "\n" ";" result_content "${result_content}")
    
    # Clean up temp files
    file(REMOVE "${list_file}")
    
    # Set output variable in parent scope
    set(${OUTPUT_VARIABLE} "${result_content}" PARENT_SCOPE)
endfunction()
