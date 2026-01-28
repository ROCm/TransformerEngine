/*************************************************************************
 * This file was modified for portability to AMDGPU
 * Copyright (c) 2023-2025, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../extensions.h"

namespace transformer_engine::pytorch {
#ifndef USE_ROCM
size_t get_cublasLt_version() { return cublasLtGetVersion(); }

size_t get_cudnn_version() { return cudnnGetVersion(); }
#endif
void placeholder() {}
}  // namespace transformer_engine::pytorch
