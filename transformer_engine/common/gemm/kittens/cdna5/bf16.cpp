/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

#include "hip/hip_runtime.h"
#include "bf16_gemm.cuh"

namespace te_kittens::cdna5 {

template bool launch_bf16<TN>(const GemmArgs &args, void *workspace, int splits, Epi epi, bool mcast);
template bool launch_bf16<NN>(const GemmArgs &args, void *workspace, int splits, Epi epi, bool mcast);
template bool launch_bf16<NT>(const GemmArgs &args, void *workspace, int splits, Epi epi, bool mcast);

}  // namespace te_kittens::cdna5
