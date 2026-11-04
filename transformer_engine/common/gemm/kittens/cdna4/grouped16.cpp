/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

// Host entry of the gfx950 BF16/FP16 grouped GEMM: contract checks, expert tables (64 per launch), launches.

#include "hip/hip_runtime.h"
#include "grouped16_common.cuh"

#include <algorithm>
#include <vector>

namespace te_kittens {
namespace cdna4 {
namespace grouped16 {

namespace {

struct PlanExpert {
    const void *a, *b;
    void *d;
    const void *bias;
    int rows, tiles;
};

struct Plan {
    Layout layout;
    bool bias;
    std::vector<PlanExpert> ex;   // launch order: NT longest first
    std::vector<int> empty;       // NT experts with no rows (D zeroed unless accumulating, dbias zeroed)
    int *ctr;                     // claim counters, 256 B aligned in the workspace
};

int cdiv(int a, int b) {
    return (a + b - 1) / b;
}

bool plan(const Grouped16GemmArgs &g, Plan &p, const char **why) {
    auto no = [&](const char *r) {
        *why = r;
        return false;
    };
    if (g.transa && !g.transb) {
        p.layout = L_TN;
    } else if (!g.transa && !g.transb) {
        p.layout = L_NN;
    } else if (!g.transa && g.transb) {
        p.layout = L_NT;
    } else {
        return no("TT layout");
    }
    if (g.in_dtype != KITTENS_BFLOAT16 && g.in_dtype != KITTENS_FLOAT16) {
        return no("inputs must be bf16 or fp16");
    }
    if (g.out_dtype != g.in_dtype && g.out_dtype != KITTENS_FLOAT32) {
        return no("output must be the input type or fp32");
    }
    if (g.num_gemms <= 0 || g.m_splits == nullptr || g.A == nullptr || g.B == nullptr || g.D == nullptr) {
        return no("no experts");
    }
    const int red = p.layout == L_TN ? g.K : p.layout == L_NN ? g.N : K_STEP;
    if (g.N <= 0 || g.K <= 0 || g.N % 8 != 0 || g.K % 8 != 0 || red % K_STEP != 0) {
        return no("weight dims must be multiples of 8, the reduction one of 64");
    }
    p.bias = false;
    if (g.bias != nullptr) {
        for (int e = 0; e < g.num_gemms; e++) {
            p.bias = p.bias || g.bias[e] != nullptr;
        }
    }
    if (p.bias) {
        if (p.layout == L_NN) {
            return no("bias with NN");
        }
        if ((p.layout == L_TN) == g.grad) {
            return no("bias is fprop's (TN) or dbias is wgrad's (NT, grad)");
        }
        if (p.layout == L_TN && g.bias_dtype != g.in_dtype) {
            return no("bias type must be the input type");
        }
        if (p.layout == L_NT && g.bias_dtype != g.out_dtype) {
            return no("dbias type must be the output type");
        }
    }
    const size_t es_out = g.out_dtype == KITTENS_FLOAT32 ? 4 : 2;
    const size_t lim    = size_t(1) << 31;
    if (size_t(g.N) * g.K * es_out >= lim) {
        return no("weight extent >= 2 GiB");
    }
    const size_t max_row = size_t(std::max(g.N, g.K)) * es_out;
    auto aligned = [](const void *q) { return reinterpret_cast<uintptr_t>(q) % 16 == 0; };
    p.ex.clear();
    p.empty.clear();
    for (int e = 0; e < g.num_gemms; e++) {
        const int64_t m = g.m_splits[e];
        if (m < 0) {
            return no("negative split");
        }
        const void *bias = p.bias ? g.bias[e] : nullptr;
        if (m == 0) {
            if (p.layout == L_NT) {
                if ((!g.accumulate && g.D[e] == nullptr) || (p.bias && bias == nullptr)) {
                    return no("null output of an empty expert");
                }
                p.empty.push_back(e);
            }
            continue;
        }
        if (size_t(m) * max_row >= lim) {
            return no("activation extent >= 2 GiB");
        }
        if (g.A[e] == nullptr || g.B[e] == nullptr || g.D[e] == nullptr || (p.bias && bias == nullptr)) {
            return no("null operand");
        }
        if (!aligned(g.A[e]) || !aligned(g.B[e]) || !aligned(g.D[e]) || (bias && !aligned(bias))) {
            return no("operand not 16 B aligned");
        }
        const int row_tiles = cdiv(int(m), BLOCK_ROW);
        const int tiles     = p.layout == L_TN   ? row_tiles * cdiv(g.N, BLOCK_COL)
                              : p.layout == L_NN ? row_tiles * cdiv(g.K, BLOCK_COL)
                                                 : cdiv(g.N, BLOCK_ROW) * cdiv(g.K, BLOCK_COL);
        p.ex.push_back({g.A[e], g.B[e], g.D[e], bias, int(m), tiles});
    }
    const uintptr_t base = reinterpret_cast<uintptr_t>(g.workspace);
    const uintptr_t ctr  = (base + 255) & ~uintptr_t(255);
    if (g.workspace == nullptr || ctr - base + 256 > g.workspace_size) {
        return no("workspace too small");
    }
    p.ctr = reinterpret_cast<int *>(ctr);
    if (p.layout == L_NT) {
        std::stable_sort(p.ex.begin(), p.ex.end(),
                         [](const PlanExpert &x, const PlanExpert &y) { return x.rows > y.rows; });
    }
    return true;
}

int device_cus() {
    static int cus[64] = {};
    int dev = 0;
    hk_check(hipGetDevice(&dev), "hipGetDevice");
    if (dev < 64 && cus[dev] > 0) {
        return cus[dev];
    }
    int n = 0;
    hk_check(hipDeviceGetAttribute(&n, hipDeviceAttributeMultiprocessorCount, dev), "CU count");
    if (dev < 64) {
        cus[dev] = n;
    }
    return n;
}

}  // namespace

bool gemm(const Grouped16GemmArgs &g, const char **why) {
    Plan p;
    if (!plan(g, p, why)) {
        return false;
    }
    const int nt = int((p.ex.size() + MAX_EXPERTS - 1) / MAX_EXPERTS);
    std::vector<Table> tables(nt);
    std::vector<int> counts(nt);
    for (int c = 0; c < nt; c++) {
        Table &t = tables[c];
        t = Table{};
        const int n = int(std::min<size_t>(MAX_EXPERTS, p.ex.size() - size_t(c) * MAX_EXPERTS));
        counts[c] = n;
        for (int i = 0; i < n; i++) {
            const PlanExpert &x = p.ex[size_t(c) * MAX_EXPERTS + i];
            t.a[i]            = x.a;
            t.b[i]            = x.b;
            t.d[i]            = x.d;
            t.bias[i]         = x.bias;
            t.rows[i]         = x.rows;
            t.unit_off[i + 1] = t.unit_off[i] + x.tiles;
        }
    }
    const Launch l{tables.data(), counts.data(), nt, g.N, g.K, g.alpha, g.accumulate ? g.beta : 0.f,
                   g.cu_count > 0 ? g.cu_count : device_cus(), p.ctr, g.stream};
    if (nt > 0) {
        if (p.layout == L_TN) {
            launch_tn(l, g.in_dtype, g.out_dtype, p.bias, g.accumulate);
        } else if (p.layout == L_NN) {
            launch_nn(l, g.in_dtype, g.out_dtype, g.accumulate);
        } else {
            launch_nt(l, g.in_dtype, g.out_dtype, p.bias, g.accumulate);
        }
    }
    const size_t es_out = g.out_dtype == KITTENS_FLOAT32 ? 4 : 2;
    for (int e : p.empty) {
        if (!g.accumulate) {
            hk_check(hipMemsetAsync(g.D[e], 0, size_t(g.N) * g.K * es_out, g.stream), "grouped16 zero dW");
        }
        if (p.bias) {
            hk_check(hipMemsetAsync(const_cast<void *>(g.bias[e]), 0, size_t(g.N) * es_out, g.stream),
                     "grouped16 zero dbias");
        }
    }
    return true;
}

}  // namespace grouped16
}  // namespace cdna4
}  // namespace te_kittens
