/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

#include "hip/hip_runtime.h"
#include "mxfp8_gemm.cuh"

namespace te_kittens::gfx1250 {

// Wgrad: x e4m3 x dy e4m3 or e5m2 (HYBRID). Beta: gradient accumulation into main_grad (fp32) or
// a 16-bit weight gradient.
template <>
struct mxfp8_gemm::Instances<NT> {
    using formats  = List<Formats<0, 0>, Formats<0, 1>>;
    using variants = List<Variant<NONE, false, __bf16>, Variant<NONE, false, _Float16>,
                          Variant<NONE, false, float>, Variant<NONE, true, __bf16>,
                          Variant<NONE, true, _Float16>, Variant<NONE, true, float>>;
};

template bool launch_mxfp8<NT>(const Launch &l);

}  // namespace te_kittens::gfx1250
