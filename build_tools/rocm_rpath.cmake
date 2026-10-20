# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

# RUNPATH configuration for the ROCm shared objects.
include_guard(GLOBAL)

# Set for builds that leave the objects in the source tree -- `pip install -e .`
# and `setup.py build_ext --inplace` -- where no $ORIGIN entry reaches site-packages
set(TE_ROCM_PURELIB "" CACHE STRING
    "site-packages of the building interpreter (inplace builds only)")

# Build the list of RUNPATH entries.
function(te_rocm_rpath out_var)
  # Layouts rocm-sdk takes inside site-packages.
  set(_sdk_libdirs rocm/lib _rocm_sdk_core/lib _rocm_sdk_libraries/lib)

  set(_rpaths "")
  if(TE_ROCM_PURELIB)
    # Append absolute rpath as the object stays in the source tree, where no
    # $ORIGIN hop reaches site-packages.
    foreach(_lib IN LISTS _sdk_libdirs)
      list(APPEND _rpaths "${TE_ROCM_PURELIB}/${_lib}")
    endforeach()
  else()
    # Assumes rocm-sdk is in the same site-packages as TE. Two hops, because the
    # objects install at two depths:
    #   $ORIGIN/..     transformer_engine/libtransformer_engine.so
    #   $ORIGIN/../..  transformer_engine/lib/{libck_fused_attn,libaotriton}*.so
    # Splits across interpreters or a venv boundary have no fixed relative form
    # and are left to the user's LD_LIBRARY_PATH.
    foreach(_origin_to_site "$ORIGIN/.." "$ORIGIN/../..")
      foreach(_lib IN LISTS _sdk_libdirs)
        list(APPEND _rpaths "${_origin_to_site}/${_lib}")
      endforeach()
    endforeach()
  endif()

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
