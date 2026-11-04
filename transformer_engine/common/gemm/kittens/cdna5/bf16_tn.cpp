/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

#include "hip/hip_runtime.h"
#include "bf16_gemm.cuh"

namespace te_kittens::cdna5 {

// Forward, D in the input type or fp32: bias (Linear), beta (te.ops ForwardLinear*Add), GELU (LayerNormMLP fc1),
// fp32 D with an input-type bias (MoE router).
template <>
struct bf16_gemm::Instances<TN> {
    using variants = List<Variant<NONE, false, In>, Variant<NONE, true, In>,
                          Variant<BIAS, false, In>, Variant<BIAS, true, In>,
                          Variant<GELU_AUX, false, In>, Variant<GELU_AUX_BIAS, false, In>,
                          Variant<NONE, false, float>, Variant<NONE, true, float>,
                          Variant<BIAS, false, float, In>>;
};

template bool launch_bf16<TN>(const Launch &l);

}  // namespace te_kittens::cdna5
