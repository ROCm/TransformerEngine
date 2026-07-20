/* Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved. */
/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

// libtorch compatibility shim.
//
// The prebuilt, closed-source xAttention core (lib/<arch>/xattention_tileiras.a)
// was compiled against a newer libtorch that exposes the PyObject-preservation
// methods
//     c10::TensorImpl::incref_pyobject() const
//     c10::TensorImpl::decref_pyobject() const
// These do not exist in the torch 2.8 shipped in the build container, so the
// binding .so fails to load with an undefined-symbol error.
//
// These methods bump/drop the refcount of a TensorImpl's associated Python
// wrapper to keep it alive while C++ holds a reference. During a synchronous
// xAttention kernel call the caller (Python) already holds live references to
// all input tensors, and outputs are freshly created, so a no-op is safe for
// this use case. We emit the exact mangled names as *weak* aliases to a no-op
// so that a genuine libtorch definition (on a matching torch version) would
// take precedence.
//
// This shim is only needed until the closed core is repackaged against the
// container's torch version.

extern "C" {

void _xattn_noop_pyobject(const void *) {}

// c10::TensorImpl::incref_pyobject() const
__attribute__((weak, alias("_xattn_noop_pyobject"))) void
_ZNK3c1010TensorImpl15incref_pyobjectEv(const void *);

// c10::TensorImpl::decref_pyobject() const
__attribute__((weak, alias("_xattn_noop_pyobject"))) void
_ZNK3c1010TensorImpl15decref_pyobjectEv(const void *);

}  // extern "C"
