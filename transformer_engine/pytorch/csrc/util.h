/*************************************************************************
 * This file was modified for portability to AMDGPU
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_PYTORCH_CSRC_UTIL_H_
#define TRANSFORMER_ENGINE_PYTORCH_CSRC_UTIL_H_

#ifndef USE_ROCM

#include <torch/extension.h>

#include <optional>

#include "transformer_engine/transformer_engine.h"

/*! \brief Swizzle the scaling factor of the input tensor.
 *
 * The returned swizzled scaling factor tensor should be kept alive during the GEMM.
 */
std::optional<at::Tensor> swizzle_scaling_factors(transformer_engine::TensorWrapper &input,
                                                  bool rowwise);

/*! \brief Swizzle the scaling factor of the input tensors.
 *
 * The returned swizzled scaling factor tensors should be kept alive during the GEMMs.
 */
std::optional<at::Tensor> multi_tensor_swizzle_scaling_factors(
    std::vector<transformer_engine::TensorWrapper> &inputs, bool rowwise);

#endif  //!USE_ROCM

#endif  //!USE_ROCM

#endif  // TRANSFORMER_ENGINE_PYTORCH_CSRC_UTIL_H_
