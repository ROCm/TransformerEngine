/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#pragma once

bool ck_tile_grouped_gemm(const NVTETensor *A,
                          const NVTETensor *B,
                          NVTETensor *D,
                          int group_num,
                          bool transA,
                          bool transB,
                          NVTETensor *workspace,
                          bool accumulate,
                          hipStream_t stream);

bool ck_tile_mx_grouped_gemm(const NVTETensor *A,
                             const NVTETensor *B,
                             NVTETensor *D,
                             int group_num,
                             bool transA,
                             bool transB,
                             NVTETensor *workspace,
                             bool accumulate,
                             hipStream_t stream);
