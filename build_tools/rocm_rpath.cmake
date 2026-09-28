# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

# RUNPATH configuration for TheRock (rocm-sdk) builds.
include_guard(GLOBAL)

option(TE_ROCM_RPATH
       "Bake RUNPATHs into the ROCm shared objects" OFF)

# site-packages of the building interpreter. Editable installs leave the shared
# objects in the source tree, where no $ORIGIN path reaches site-packages.
set(TE_ROCM_PURELIB "" CACHE STRING
    "site-packages of the interpreter driving the build")

# Build the list of RUNPATH entries.
function(te_rocm_rpath out_var)
  set(_core _rocm_sdk_core/lib _rocm_sdk_libraries/lib)

  # TE ships shared objects at transformer_engine/ and transformer_engine/lib/.
  set(_rpaths "")
  foreach(_origin_to_site "$ORIGIN/.." "$ORIGIN/../..")
    foreach(_lib IN LISTS _core)
      list(APPEND _rpaths "${_origin_to_site}/${_lib}")
    endforeach()
  endforeach()

  if(TE_ROCM_PURELIB)
    foreach(_lib IN LISTS _core)
      list(APPEND _rpaths "${TE_ROCM_PURELIB}/${_lib}")
    endforeach()
  endif()

  set(${out_var} "${_rpaths}" PARENT_SCOPE)
endfunction()

# Set INSTALL_RPATH on a target: caller's entries first, then the ROCm paths.
#
#   te_set_rocm_rpath(<target> [<rpath entry>...])
function(te_set_rocm_rpath target)
  set(_rpath ${ARGN})
  if(TE_ROCM_RPATH)
    te_rocm_rpath(_rocm_rpath)
    list(APPEND _rpath ${_rocm_rpath})
  endif()
  set_target_properties(${target} PROPERTIES
    INSTALL_RPATH "${_rpath}"
    # Build-tree link directories only exist on the build machine.
    INSTALL_RPATH_USE_LINK_PATH OFF)
endfunction()

# Rewrite the RUNPATH of shared objects TE copies in rather than links, which on
# two of their three delivery paths arrive already built (prebuilt AITER cache,
# or AITER_MHA_PATH) and so have no link step to carry one. DT_RUNPATH is not
# inherited, so without this they depend on libtransformer_engine.so being
# loaded first. Paths may be globs and resolve at install time, so call this
# after the install() rule that places the files.
#
#   te_patchelf_rocm_rpath(<path or glob>...)
function(te_patchelf_rocm_rpath)
  if(NOT TE_ROCM_RPATH)
    return()
  endif()

  find_program(TE_PATCHELF_EXECUTABLE NAMES patchelf)
  if(NOT TE_PATCHELF_EXECUTABLE)
    message(FATAL_ERROR
            "patchelf is required to build against the rocm-sdk packages but was "
            "not found. Install it with 'pip install patchelf' or "
            "'apt-get install patchelf'.")
  endif()

  te_rocm_rpath(_rocm_rpath)
  list(INSERT _rocm_rpath 0 "$ORIGIN")
  string(REPLACE ";" ":" _rpath "${_rocm_rpath}")

  foreach(_pattern IN LISTS ARGN)
    install(CODE "
      file(GLOB _te_rpath_files \"${_pattern}\")
      foreach(_te_rpath_file IN LISTS _te_rpath_files)
        if(NOT IS_SYMLINK \"\${_te_rpath_file}\")
          message(STATUS \"Setting ROCm RUNPATH: \${_te_rpath_file}\")
          execute_process(
            COMMAND \"${TE_PATCHELF_EXECUTABLE}\" --set-rpath \"${_rpath}\" \"\${_te_rpath_file}\"
            COMMAND_ERROR_IS_FATAL ANY)
        endif()
      endforeach()")
  endforeach()
endfunction()

