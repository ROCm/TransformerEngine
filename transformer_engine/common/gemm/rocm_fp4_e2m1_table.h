/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_GEMM_ROCM_FP4_E2M1_TABLE_H_
#define TRANSFORMER_ENGINE_COMMON_GEMM_ROCM_FP4_E2M1_TABLE_H_

// FP4 (E2M1) code-point value table, indexed by the 4-bit code (OCP microscaling).
#define NVTE_ROCM_FP4_E2M1_VALUES                          \
  {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,         \
   -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f}

#endif  // TRANSFORMER_ENGINE_COMMON_GEMM_ROCM_FP4_E2M1_TABLE_H_
