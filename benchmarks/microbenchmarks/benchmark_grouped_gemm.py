#!/usr/bin/env python
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import torch
import transformer_engine.pytorch as te
from utils import (
    DTYPE_LIST,
    time_func,
    compute_tflops,
    make_forward_backward_metric_records,
    run_benchmarks,
    make_input,
)

BENCHMARK_LABEL = "Grouped GEMM"

def generate_grouped_gemm_group_lens(b, m, balance: bool):
    if balance:
        return torch.full((b,), m, dtype=torch.int64)
    else:
        dist = 0.2 + 0.8 * torch.rand(b)
        dist /= dist.sum()
        group_lens = (dist * b * m).to(torch.int64)
        error = b * m - group_lens.sum()
        group_lens[-1] += error
        return group_lens

# Grouped GEMM scales with expert count B, so we sweep smaller M values than
# the dense GEMM benchmarks to keep the working set and runtime reasonable.
GROUPED_GEMM_M_SIZE_LIST = [512, 1024, 2048, 4096]
EP_SIZE_LIST = [32, 16, 8]


def _generate_moe_test_cases(
    name_prefix: str,
    n_routed_experts: int,
    moe_intermediate_size: int,
    hidden_size: int,
    skip_shapes=None,
):
    test_cases = []
    shapes_dict = {
        f"{name_prefix}-GateUP": (2 * moe_intermediate_size, hidden_size),
        f"{name_prefix}-Down": (hidden_size, moe_intermediate_size),
    }
    if skip_shapes:
        for s in skip_shapes:
            shapes_dict.pop(f"{name_prefix}-{s}", None)

    for ep in EP_SIZE_LIST:
        if n_routed_experts % ep != 0:
            continue
        B = n_routed_experts // ep
        if B < 1:
            continue
        for M in GROUPED_GEMM_M_SIZE_LIST:
            for name, (N, K) in shapes_dict.items():
                for dtype in DTYPE_LIST:
                    test_cases.append(
                        {
                            "Case": name,
                            "B": B,
                            "M": M,
                            "N": N,
                            "K": K,
                            "dtype": dtype,
                        }
                    )
    return test_cases


def generate_deepseekv3_test_cases():
    # DSV3-GateUP hangs on some hardware; only benchmark DSV3-Down.
    return _generate_moe_test_cases(
        "DSV3", n_routed_experts=256, moe_intermediate_size=2048, hidden_size=7168,
        skip_shapes=["GateUP"],
    )


def generate_deepseekv2_test_cases():
    return _generate_moe_test_cases(
        "DSV2", n_routed_experts=160, moe_intermediate_size=1536, hidden_size=5120
    )


def generate_deepseekv2_lite_test_cases():
    return _generate_moe_test_cases(
        "DSV2-Lite", n_routed_experts=64, moe_intermediate_size=1408, hidden_size=2048
    )


def generate_grok_v2_test_cases():
    return _generate_moe_test_cases(
        "Grok-V2", n_routed_experts=8, moe_intermediate_size=16384, hidden_size=8192
    )


def bench_grouped_gemm(Case, B, M, N, K, dtype):
    device = "cuda"

    group_lens = generate_grouped_gemm_group_lens(B, M, balance=True)
    m_splits = [int(v) for v in group_lens.tolist()]
    m_splits_tensor = torch.tensor(m_splits, dtype=torch.int32, device=device)
    sum_M = sum(m_splits)

    grouped_linear = te.GroupedLinear(
        B,
        K,
        N,
        bias=False,
        params_dtype=dtype,
        device=device,
    )
    # Rotate the activation buffer (on by default) so back-to-back grouped GEMMs 
    # read different memory; GroupedLinear splits it internally per m_splits.
    next_x = make_input((sum_M, K), dtype, device=device, requires_grad=True)

    def fwd_func_te():
        return grouped_linear(next_x(), m_splits, m_splits_tensor=m_splits_tensor)

    out_te = fwd_func_te()
    grad_out = torch.randn_like(out_te)

    def fwd_bwd_func_te():
        xb = next_x()
        out = grouped_linear(xb, m_splits, m_splits_tensor=m_splits_tensor)
        out.backward(grad_out)
        xb.grad = None
        for param in grouped_linear.parameters():
            param.grad = None

    fwd_bwd_func_te()

    fwd_total_flops = 2 * sum_M * N * K
    bwd_total_flops = 2 * fwd_total_flops

    fwd_te_ms, fwd_measurement = time_func(fwd_func_te)
    fwd_bwd_te_ms, fwd_bwd_measurement = time_func(fwd_bwd_func_te)
    bwd_te_ms = fwd_bwd_te_ms - fwd_te_ms

    fwd_te_tflops = compute_tflops(fwd_total_flops, fwd_te_ms)
    bwd_te_tflops = compute_tflops(bwd_total_flops, bwd_te_ms)

    return make_forward_backward_metric_records(
        BENCHMARK_LABEL,
        "TFLOPS",
        fwd_te_ms,
        fwd_te_tflops,
        bwd_te_ms,
        bwd_te_tflops,
        backward_derived=True,
        fwd_measurement=fwd_measurement,
        fwd_bwd_measurement=fwd_bwd_measurement,
    )


if __name__ == "__main__":
    test_cases = (
        generate_deepseekv2_lite_test_cases()
        + generate_deepseekv2_test_cases()
        + generate_deepseekv3_test_cases()
        + generate_grok_v2_test_cases()
    )

    run_benchmarks(
        test_cases=test_cases,
        bench_fn=bench_grouped_gemm,
        param_columns=["Case", "B", "M", "N", "K", "dtype"],
    )
