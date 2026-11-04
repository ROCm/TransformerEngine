/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

/* gfx950 BF16/FP16 grouped GEMM (Grouped16GemmArgs) in one persistent launch per 64 experts.
 * Contract: the reduction weight dim (TN: K, NN: N) % 64 == 0, the other weight dims % 8 == 0,
 * any m_e >= 0, 16 B aligned pointers, operand extents below 2 GiB. Anything else is declined. */

#pragma once

#include "hip/hip_runtime.h"
#include "../kittens_common.h"

namespace te_kittens {
namespace cdna4 {
namespace grouped16 {

// Returns false, before any launch, for anything outside the contract; *why says why.
bool gemm(const Grouped16GemmArgs &args, const char **why);

}  // namespace grouped16
}  // namespace cdna4
}  // namespace te_kittens
