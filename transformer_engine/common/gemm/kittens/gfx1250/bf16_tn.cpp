/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

#include "hip/hip_runtime.h"
#include "bf16_gemm.cuh"

namespace te_kittens::gfx1250 {

// Forward, D in the input type or fp32. Bias: Linear and LayerNormLinear; beta: te.ops
// ForwardLinear{Bias,Scale}Add; GELU: LayerNormMLP fc1 without bias-GELU fusion. fp32 D: the MoE
// router (bias in the input type) and fp32 residuals.
template <>
struct bf16_gemm::Instances<TN> {
    using variants = List<Variant<NONE, false, In>, Variant<NONE, true, In>,
                          Variant<BIAS, false, In>, Variant<BIAS, true, In>,
                          Variant<GELU_AUX, false, In>, Variant<GELU_AUX_BIAS, false, In>,
                          Variant<NONE, false, float>, Variant<NONE, true, float>,
                          Variant<BIAS, false, float, In>>;
};

template bool launch_bf16<TN>(const Launch &l);

}  // namespace te_kittens::gfx1250
