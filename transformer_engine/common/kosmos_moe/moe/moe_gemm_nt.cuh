/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/
// MoE backward wgrad NT building blocks.
#pragma once

namespace hk_moe {

static_assert(EXPERT_PADDING % K_STEP_64 == 0,
              "roff_e / K_STEP_64 must be exact and round_up(rows_e, K_STEP_64) must stay inside the "
              "EXPERT_PADDING slab: EXPERT_PADDING must be a multiple of K_STEP_64.");

constexpr int NT_REG_M = BLOCK_ROW / WARPS_ROW;
constexpr int NT_REG_N = BLOCK_COL / WARPS_COL;

}
