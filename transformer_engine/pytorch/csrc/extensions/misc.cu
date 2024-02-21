/*************************************************************************
<<<<<<< HEAD
 * This file was modified for portability to AMDGPU
 * Copyright (c) 2023-2024, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
=======
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
>>>>>>> upstream/main
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "extensions.h"
#ifdef NVTE_WITH_USERBUFFERS
#include "comm_gemm_overlap.h"
#endif  // NVTE_WITH_USERBUFFERS

#ifndef USE_ROCM
size_t get_cublasLt_version() {
    return cublasLtGetVersion();
}

<<<<<<< HEAD
=======
size_t get_cudnn_version() {
    return cudnnGetVersion();
}


>>>>>>> upstream/main
bool userbuf_comm_available() {  // TODO(ksivamani) check on python side
#ifdef NVTE_WITH_USERBUFFERS
    return true;
#else
    return false;
#endif
}
#endif

void placeholder() {}  // TODO(ksivamani) clean this up
