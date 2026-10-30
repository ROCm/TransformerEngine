/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/
#ifndef TRANSFORMER_ENGINE_COMMON_NORMALIZATION_ROCM_NORM_H_
#define TRANSFORMER_ENGINE_COMMON_NORMALIZATION_ROCM_NORM_H_

// The ROCm normalization kernels, which normalization plans prefer where they apply.

#include <functional>

#include "common.h"

namespace transformer_engine {
namespace normalization {
namespace rocm {

// The ROCm forward kernel for the rows, columns and output of launch_params, as a plan's kernel
// function, or an empty function where it does not apply. It needs tensors aligned as
// is_ptr_aligned checks them, and no workspace.
std::function<void(LaunchParams<ForwardKernelParams> &, const bool)> get_fwd_kernel(
    NVTE_Norm_Type norm_type, DType wtype, DType itype, DType otype,
    const LaunchParams<ForwardKernelParams> &launch_params, bool aligned);

// The ROCm backward kernels for a Backward or BackwardAdd stage, as get_fwd_kernel returns the
// forward one. They take dgamma and dbeta partials from the plan's workspace.
std::function<void(LaunchParams<BackwardKernelParams> &, const bool)> get_bwd_kernel(
    NVTE_Norm_Type norm_type, NVTE_Norm_Stage norm_stage, DType wtype, DType itype, DType otype,
    const LaunchParams<BackwardKernelParams> &launch_params, bool aligned);

}  // namespace rocm
}  // namespace normalization
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_NORMALIZATION_ROCM_NORM_H_
