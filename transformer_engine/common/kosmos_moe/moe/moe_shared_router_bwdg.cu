/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/
// libkosmos_moe.a: the shared-expert backward with the fused router and the sigmoid gate (moe_shared_router.cu with
// MoeShGate).
#define MOE_SHARED_GATE 1
#include "moe_shared_router.cu"
