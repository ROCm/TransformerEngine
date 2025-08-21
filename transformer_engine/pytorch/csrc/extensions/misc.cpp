/*************************************************************************
 * This file was modified for portability to AMDGPU
 * Copyright (c) 2023-2025, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "extensions.h"

<<<<<<< HEAD
#ifndef USE_ROCM
size_t get_cublasLt_version() { return cublasLtGetVersion(); }

size_t get_cudnn_version() { return cudnnGetVersion(); }
#endif
void placeholder() {}  // TODO(ksivamani) clean this up
=======
namespace transformer_engine::pytorch {

size_t get_cublasLt_version() { return cublasLtGetVersion(); }

size_t get_cudnn_version() { return cudnnGetVersion(); }

}  // namespace transformer_engine::pytorch
>>>>>>> 42b51c40c4e39adce9640cf98f8a3f5869f5f270
