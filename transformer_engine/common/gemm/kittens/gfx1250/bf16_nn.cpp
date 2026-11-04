/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

#include "hip/hip_runtime.h"
#include "bf16_gemm.cuh"

namespace te_kittens::gfx1250 {

// Dgrad, D in the input type or fp32. Beta: te.ops BackwardLinearAdd; DGELU: LayerNormMLP fc2
// dgrad without bias-GELU fusion.
template <>
struct bf16_gemm::Instances<NN> {
    using variants = List<Variant<NONE, false, In>, Variant<NONE, true, In>,
                          Variant<DGELU, false, In>, Variant<NONE, false, float>,
                          Variant<NONE, true, float>>;
};

template bool launch_bf16<NN>(const Launch &l);

}  // namespace te_kittens::gfx1250
