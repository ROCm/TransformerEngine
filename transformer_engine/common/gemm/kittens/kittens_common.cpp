/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

#include "hip/hip_runtime.h"
#include "kittens_common.h"

BlockwiseGemmBackend *BlockwiseGemmBackend::get_for_arch(int sm_arch) {
#ifdef KITTENS_HAVE_CDNA4
    if (sm_arch == 95) {
        return get_cdna4();
    }
#endif
#ifdef KITTENS_HAVE_CDNA3
    if (sm_arch == 94) {
        return get_cdna3();
    }
#endif
    static_cast<void>(sm_arch);
    return nullptr;
}

MXFP8GemmBackend *MXFP8GemmBackend::get_for_arch(int sm_arch) {
#ifdef KITTENS_HAVE_CDNA4
    if (sm_arch == 95) {
        return get_cdna4();
    }
#endif
    static_cast<void>(sm_arch);
    return nullptr;
}
