/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

#include "grouped_f16_gemm_helper.cuh"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

static void warn_fallback(const char *reason) {
    const char *v = std::getenv("NVTE_CUTLASS_GROUPED_GEMM_WARN_FALLBACK");
    if (v && v[0] == '1') {
        fprintf(stderr, "[HK-grouped_f16] falling back: %s\n", reason);
    }
}

bool kittens_grouped_f16_gemm_cdna4(const GroupedF16GemmArgs &args) {
    using namespace te_kittens::cdna4;
    using namespace te_kittens::cdna4::grouped_f16;
    const bool transa = args.transa;
    const bool transb = args.transb;
    const int N = args.N;
    const int K = args.K;
    if (args.out_dtype != args.in_dtype && args.out_dtype != KITTENS_FLOAT32) {
        warn_fallback("output must be the input type or fp32");
        return false;
    }
    if (N <= 0 || K <= 0 || N % BLOCK_ROW != 0 || K % BLOCK_COL != 0) {
        warn_fallback("weight dims must be multiples of 256");
        return false;
    }
    bool bias = false;
    if (args.bias_array != nullptr) {
        for (int g = 0; g < args.num_experts; g++) {
            bias = bias || args.bias_array[g] != nullptr;
        }
    }
    if (bias) {
        if (!transa && !transb) {
            warn_fallback("bias with NN");
            return false;
        }
        if (transa == args.grad) {
            warn_fallback("bias is fprop's (TN) or dbias is wgrad's (NT, grad)");
            return false;
        }
        if (transa && args.bias_dtype != args.in_dtype) {
            warn_fallback("bias type must be the input type");
            return false;
        }
        if (transb && args.bias_dtype != args.out_dtype) {
            warn_fallback("dbias type must be the output type");
            return false;
        }
    }
    const size_t es_out  = args.out_dtype == KITTENS_FLOAT32 ? 4 : 2;
    const size_t es_max  = std::max<size_t>(2, es_out);
    const size_t lim     = size_t(1) << 31;
    const size_t max_row = size_t(std::max(N, K)) * es_max;
    if (size_t(N) * K * es_max >= lim) {
        warn_fallback("weight extent >= 2 GiB");
        return false;
    }

    std::vector<std::pair<int, int>> keyed;
    keyed.reserve(args.num_experts);
    for (int g = 0; g < args.num_experts; g++) {
        const int64_t m = args.M_array[g];
        const void *bias_g = bias ? args.bias_array[g] : nullptr;
        if (m == 0) {
            if (transb && ((!args.accumulate && args.D_array[g] == nullptr) || (bias && bias_g == nullptr))) {
                warn_fallback("null output of an empty expert");
                return false;
            }
            continue;
        }
        if (size_t(m) * max_row >= lim) {
            warn_fallback("activation extent >= 2 GiB");
            return false;
        }
        if (args.A_array[g] == nullptr || args.B_array[g] == nullptr || args.D_array[g] == nullptr ||
            (bias && bias_g == nullptr)) {
            warn_fallback("null operand");
            return false;
        }
        keyed.push_back({transb ? -int(m) : 0, g});
    }
    const uintptr_t base = reinterpret_cast<uintptr_t>(args.workspace);
    int *ctr = reinterpret_cast<int *>(kittens_align_up(base, 256));
    if (args.workspace == nullptr || reinterpret_cast<uintptr_t>(ctr) - base + 256 > args.workspace_size) {
        warn_fallback("workspace too small");
        return false;
    }
    std::sort(keyed.begin(), keyed.end());

    const int n = (int)keyed.size();
    for (int c0 = 0; c0 < n; c0 += MAX_TAB_EXPERTS) {
        const int ne = std::min(MAX_TAB_EXPERTS, n - c0);
        GroupedTab tab{};
        for (int i = 0; i < ne; i++) {
            const int g         = keyed[c0 + i].second;
            const int rows      = (int)args.M_array[g];
            const int row_tiles = (rows + BLOCK_ROW - 1) / BLOCK_ROW;
            int tiles = (N / BLOCK_ROW) * (K / BLOCK_COL);
            if (transa) {
                tiles = row_tiles * (N / BLOCK_COL);
            } else if (!transb) {
                tiles = row_tiles * (K / BLOCK_COL);
            }
            tab.a[i]            = args.A_array[g];
            tab.b[i]            = args.B_array[g];
            tab.d[i]            = args.D_array[g];
            tab.bias[i]         = bias ? args.bias_array[g] : nullptr;
            tab.rows[i]         = rows;
            tab.tile_off[i + 1] = tab.tile_off[i] + tiles;
        }
        const int units = tab.tile_off[ne] + (transb && bias ? ne * (N / BLOCK_ROW) : 0);
        const int grid  = std::min(units, args.cu_count);
        hipMemsetAsync(ctr, 0, NUM_XCDS_AFF * sizeof(int), args.stream);
        if (transa) {
            launch_grouped_f16_tn(tab, ne, N, K, grid, ctr, args.in_dtype, args.out_dtype, bias, args.accumulate,
                                  args.stream);
        } else if (!transb) {
            launch_grouped_f16_nn(tab, ne, N, K, grid, ctr, args.in_dtype, args.out_dtype, args.accumulate,
                                  args.stream);
        } else {
            launch_grouped_f16_nt(tab, ne, N, K, grid, ctr, args.in_dtype, args.out_dtype, bias, args.accumulate,
                                  args.stream);
        }
    }

    for (int g = 0; g < args.num_experts; g++) {
        if (!transb || args.M_array[g] != 0) {
            continue;
        }
        if (!args.accumulate) {
            hipMemsetAsync(args.D_array[g], 0, size_t(N) * K * es_out, args.stream);
        }
        if (bias) {
            hipMemsetAsync(const_cast<void *>(args.bias_array[g]), 0, size_t(N) * es_out, args.stream);
        }
    }
    return true;
}
