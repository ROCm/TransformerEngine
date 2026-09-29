# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

# RUNPATH configuration for the ROCm shared objects.
include_guard(GLOBAL)

# Inplace builds leave the objects in the source tree, out of $ORIGIN's reach.
# Empty otherwise: a redistributable wheel must not carry a build-machine path.
set(TE_ROCM_PURELIB "" CACHE STRING
    "site-packages of the building interpreter (inplace builds only)")

# Build the list of RUNPATH entries.
function(te_rocm_rpath out_var)
  set(_core _rocm_sdk_core/lib _rocm_sdk_libraries/lib)

  # TheRock tarball layout, next to the wheel.
  set(_rpaths "$ORIGIN/../rocm/lib" "$ORIGIN/../../rocm/lib")

  # rocm-sdk packages. TE ships objects at two depths and one list goes on every
  # object, so emit both; the loader skips entries whose directories are absent.
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

  # System install, last so the wheel's own ROCm always wins.
  list(APPEND _rpaths "/opt/rocm/lib")

  set(${out_var} "${_rpaths}" PARENT_SCOPE)
endfunction()

# Set INSTALL_RPATH on a target: caller's entries first, then the ROCm paths.
#
#   te_set_rocm_rpath(<target> [<rpath entry>...])
function(te_set_rocm_rpath target)
  te_rocm_rpath(_rocm_rpath)
  set(_rpath ${ARGN} ${_rocm_rpath})
  set_target_properties(${target} PROPERTIES
    INSTALL_RPATH "${_rpath}"
    # Build-tree link directories only exist on the build machine.
    INSTALL_RPATH_USE_LINK_PATH OFF)
endfunction()

# Rewrite the RUNPATH of shared objects TE copies in rather than links, so has no
# link step to set one on. DT_RUNPATH is not inherited, so without this they rely
# on libtransformer_engine.so being loaded first. Paths may be globs and resolve
# at install time, so call this after the install() rule that places the files.
#
#   te_patchelf_rocm_rpath(<path or glob>...)
function(te_patchelf_rocm_rpath)
  find_program(TE_PATCHELF_EXECUTABLE NAMES patchelf)
  if(NOT TE_PATCHELF_EXECUTABLE)
    message(FATAL_ERROR
            "patchelf is required to bake RUNPATHs into the prebuilt ROCm shared "
            "objects but was not found.")
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
