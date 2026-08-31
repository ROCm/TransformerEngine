/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

#include "hip/hip_runtime.h"
#include "comm_gemm.h"
#include "kittens_common.h"

#ifdef KITTENS_HAVE_CDNA4
bool kittens_fused_ag_gemm_bf16_cdna4(const KittensAgGemmArgs &args);
bool kittens_bulk_ag_gemm_bf16_cdna4(const KittensAgGemmArgs &args);
bool kittens_bulk_rs_gemm_bf16_cdna4(const KittensRsGemmArgs &args);
bool kittens_fused_rs_gemm_bf16_cdna4(const KittensRsGemmArgs &args);
bool kittens_fused_rs_gemm_eligible_cdna4(const KittensRsGemmArgs &args);
size_t kittens_fused_rs_region_bytes_cdna4(size_t chunk_bytes, int tp_size);
void kittens_persistent_plans_reset_cdna4();
#endif

bool kittens_fused_ag_gemm_supported(int sm_arch) {
#ifdef KITTENS_HAVE_CDNA4
    if (sm_arch == 95) {
        return true;
    }
#endif
    static_cast<void>(sm_arch);
    return false;
}

void kittens_comm_gemm_reset() {
#ifdef KITTENS_HAVE_CDNA4
    kittens_persistent_plans_reset_cdna4();
#endif
}

bool kittens_fused_ag_gemm_bf16(const KittensAgGemmArgs &args) {
#ifdef KITTENS_HAVE_CDNA4
    return kittens_fused_ag_gemm_bf16_cdna4(args);
#else
    static_cast<void>(args);
    return false;
#endif
}

bool kittens_bulk_ag_gemm_bf16(const KittensAgGemmArgs &args) {
#ifdef KITTENS_HAVE_CDNA4
    return kittens_bulk_ag_gemm_bf16_cdna4(args);
#else
    static_cast<void>(args);
    return false;
#endif
}

bool kittens_bulk_rs_gemm_supported(int sm_arch) {
#ifdef KITTENS_HAVE_CDNA4
    if (sm_arch == 95) {
        return true;
    }
#endif
    static_cast<void>(sm_arch);
    return false;
}

bool kittens_bulk_rs_gemm_bf16(const KittensRsGemmArgs &args) {
#ifdef KITTENS_HAVE_CDNA4
    return kittens_bulk_rs_gemm_bf16_cdna4(args);
#else
    static_cast<void>(args);
    return false;
#endif
}

bool kittens_fused_rs_gemm_supported(int sm_arch) {
#ifdef KITTENS_HAVE_CDNA4
    if (sm_arch == 95) {
        return true;
    }
#endif
    static_cast<void>(sm_arch);
    return false;
}


size_t kittens_fused_rs_region_bytes(size_t chunk_bytes, int tp_size) {
#ifdef KITTENS_HAVE_CDNA4
    return kittens_fused_rs_region_bytes_cdna4(chunk_bytes, tp_size);
#else
    static_cast<void>(chunk_bytes);
    static_cast<void>(tp_size);
    return 0;
#endif
}

bool kittens_fused_rs_gemm_eligible(const KittensRsGemmArgs &args) {
#ifdef KITTENS_HAVE_CDNA4
    return kittens_fused_rs_gemm_eligible_cdna4(args);
#else
    static_cast<void>(args);
    return false;
#endif
}

bool kittens_fused_rs_gemm_bf16(const KittensRsGemmArgs &args) {
#ifdef KITTENS_HAVE_CDNA4
    return kittens_fused_rs_gemm_bf16_cdna4(args);
#else
    static_cast<void>(args);
    return false;
#endif
}

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
