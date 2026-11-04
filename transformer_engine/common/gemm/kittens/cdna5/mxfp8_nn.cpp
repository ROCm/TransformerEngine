/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

#include "hip/hip_runtime.h"
#include "mxfp8_gemm.cuh"

namespace te_kittens::cdna5 {

// Dgrad: e4m3/e5m2 x e4m3/e5m2. Beta: te.ops BackwardLinearAdd into a
// grad_input of any dtype.
template <>
struct mxfp8_gemm::Instances<NN> {
    using formats  = List<Formats<0, 0>, Formats<0, 1>, Formats<1, 0>, Formats<1, 1>>;
    using variants = List<Variant<NONE, false, __bf16>, Variant<NONE, false, _Float16>,
                          Variant<NONE, false, float>, Variant<NONE, true, __bf16>,
                          Variant<NONE, true, _Float16>, Variant<NONE, true, float>>;
};

template bool launch_mxfp8<NN>(const Launch &l);

}  // namespace te_kittens::cdna5
