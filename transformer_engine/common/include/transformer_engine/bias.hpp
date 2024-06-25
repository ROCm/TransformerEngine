// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ostream>
#include <string>
#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha.hpp"

// keep sync with BlockAttentionBiasEnum
enum class bias_enum
{
    no_bias          = 0,
    elementwise_bias = 1,
    alibi            = 2,
};

struct bias_info
{
    bias_enum type;
    /*
     * simple dispatch logic
     *
     * if type == elementwise_bias:
     *      if rank_info == 0:
     *           bias is 1*1*s*s
     *      elif rank_info == 1:
     *           bias is 1*h*s*s
     *      elif rank_info == 2:
     *           bias is b*h*s*s
     *
     * elif type == alibi:
     *       if rank_info == 0:
     *           alibi in 1*h
     *       elif rank_info == 1:
     *           alibi in b*h
     */
    int rank_info;
};
