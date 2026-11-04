/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

#include "hip/hip_runtime.h"
#include "mxfp8_gemm.cuh"

namespace te_kittens::cdna5 {

// Forward, e4m3/e5m2 x e4m3/e5m2: bias (Linear; bf16 with fp32 D), beta (te.ops ForwardLinear*Add),
// GELU (LayerNormMLP fc1 with NVTE_GEMM_GELU_FUSION=1; aux = bias dtype, bf16 without a bias).
template <>
struct mxfp8_gemm::Instances<TN> {
    using formats  = List<Formats<0, 0>, Formats<0, 1>, Formats<1, 0>, Formats<1, 1>>;
    using variants = List<Variant<NONE, false, __bf16>, Variant<NONE, false, _Float16>,
                          Variant<NONE, false, float>, Variant<NONE, true, __bf16>,
                          Variant<NONE, true, _Float16>, Variant<NONE, true, float>,
                          Variant<BIAS, false, __bf16>, Variant<BIAS, false, _Float16>,
                          Variant<BIAS, false, float>, Variant<BIAS, false, float, __bf16>,
                          Variant<BIAS, true, __bf16>, Variant<BIAS, true, _Float16>,
                          Variant<BIAS, true, float>, Variant<BIAS, true, float, __bf16>,
                          Variant<GELU_AUX, false, __bf16>,
                          Variant<GELU_AUX, false, _Float16, __bf16>,
                          Variant<GELU_AUX, false, float, __bf16>,
                          Variant<GELU_AUX_BIAS, false, __bf16>,
                          Variant<GELU_AUX_BIAS, false, _Float16>,
                          Variant<GELU_AUX_BIAS, false, float, __bf16>>;
};

template bool launch_mxfp8<TN>(const Launch &l);

}  // namespace te_kittens::cdna5
