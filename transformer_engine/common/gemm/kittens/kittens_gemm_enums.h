/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

#pragma once

// Values match NVTEDType in transformer_engine.h
enum KittensDType {
    KITTENS_FLOAT32  = 4,
    KITTENS_FLOAT16  = 5,
    KITTENS_BFLOAT16 = 6,
    KITTENS_FP8E4M3  = 7,
    KITTENS_FP8E5M2  = 8,
};

// Values match NVTEScalingMode in transformer_engine.h
enum KittensScalingMode {
    KITTENS_BLOCK_SCALING_1D = 2,
    KITTENS_BLOCK_SCALING_2D = 3,
};
