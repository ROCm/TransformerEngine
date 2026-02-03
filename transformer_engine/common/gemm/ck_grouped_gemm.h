/* Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved. */

bool grouped_gemm_ck_tile(const NVTETensor* A,
                          const NVTETensor* B,
                          NVTETensor* D,
                          int group_num,
                          bool transA,
                          bool transB,
                          NVTETensor* workspace,
                          bool accumulate,
                          hipStream_t stream);
