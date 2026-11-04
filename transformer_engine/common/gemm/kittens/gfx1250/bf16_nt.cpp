/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

#include "hip/hip_runtime.h"
#include "bf16_gemm.cuh"

namespace te_kittens::gfx1250 {

// Wgrad into a weight gradient of the input type or fp32 main_grad, with beta for gradient
// accumulation; BGRADB also writes dbias (Linear, LayerNormLinear, LayerNormMLP with bias).
template <>
struct bf16_gemm::Instances<NT> {
    using variants = List<Variant<NONE, false, In>, Variant<NONE, true, In>,
                          Variant<NONE, false, float>, Variant<NONE, true, float>,
                          Variant<BGRADB, false, In>, Variant<BGRADB, true, In>,
                          Variant<BGRADB, false, float>, Variant<BGRADB, true, float>>;
};

template bool launch_bf16<NT>(const Launch &l);

}  // namespace te_kittens::gfx1250
