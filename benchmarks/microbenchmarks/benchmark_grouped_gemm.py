#!/usr/bin/env python
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import torch
from utils import (
    DTYPE_LIST,
    time_func,
    compute_tflops,
    make_forward_backward_metric_records,
    run_benchmarks,
    rotating,
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


def make_fwd_bwd_funcs_te(next_xs, w, group_lens, activation_dtype):
    from transformer_engine.pytorch.cpp_extensions import general_grouped_gemm

    B = int(group_lens.numel())
    N = int(w.shape[1])
    K = int(w.shape[2])

    m_splits = [int(v) for v in group_lens.tolist()]
    assert len(m_splits) == B
    sum_M = sum(m_splits)

    weights = [w[i] for i in range(B)]
    device = w.device

    # next_xs() yields the per-expert split of the (current) activation buffer.
    # The splits are precomputed once per buffer (see bench_grouped_gemm), so the
    # timed region below only measures general_grouped_gemm, not reshape/split.
    out = torch.empty((sum_M, N), device=device, dtype=activation_dtype)

    def fwd_func_te():
        general_grouped_gemm(
            A=weights,
            B=next_xs(),
            out=[out],
            quantization_params=[None] * B,
            out_dtype=activation_dtype,
            single_output=True,
            m_splits=m_splits,
            use_bias=False,
            bias=None,
            layout="TN",
        )
        return out

    dx = torch.empty((sum_M, K), device=device, dtype=activation_dtype)
    dxs = list(torch.split(dx, m_splits))

    dw_stacked = torch.empty((B, N, K), device=device, dtype=activation_dtype)
    dws = [dw_stacked[i] for i in range(B)]

    def bwd_func_te(grad_out):
        go = grad_out.view(-1, grad_out.shape[-1])
        splits = torch.split(go, m_splits)

        general_grouped_gemm(
            A=weights,
            B=splits,
            out=dxs,
            quantization_params=[None] * B,
            out_dtype=activation_dtype,
            single_output=False,
            layout="NN",
            m_splits=m_splits,
            grad=False,
            use_bias=False,
            bias=None,
        )

        general_grouped_gemm(
            A=next_xs(),
            B=splits,
            out=dws,
            quantization_params=[None] * B,
            out_dtype=activation_dtype,
            single_output=False,
            layout="NT",
            m_splits=m_splits,
            grad=False,
            use_bias=False,
            bias=None,
            accumulate=False,
        )

        return dx, dw_stacked

    return fwd_func_te, bwd_func_te


def bench_grouped_gemm(Case, B, M, N, K, dtype):
    device = "cuda"

    w = torch.randn((B, N, K), dtype=dtype, device=device)
    group_lens = generate_grouped_gemm_group_lens(B, M, balance=True).to(device)
    m_splits = [int(v) for v in group_lens.tolist()]

    # Rotation ring of *pre-split* activation buffers, built once at setup, so
    # the timed region only measures the grouped GEMM (not reshape/split). With
    # rotation off this is a single cached split, matching the original behavior.
    elem_bytes = torch.empty(0, dtype=dtype).element_size()

    def _build_xs():
        x = torch.randn((B * M, K), dtype=dtype, device=device)
        return list(torch.split(x.reshape(-1, x.shape[-1]), m_splits))

    next_xs = rotating(_build_xs, bytes_per_buffer=B * M * K * elem_bytes)

    fwd_func_te, bwd_func_te_inner = make_fwd_bwd_funcs_te(
        next_xs, w, group_lens, activation_dtype=dtype
    )

    out_te = fwd_func_te()
    grad_out = torch.randn_like(out_te)
    bwd_func_te = lambda: bwd_func_te_inner(grad_out)

    fwd_total_flops = 2 * B * M * N * K
    bwd_total_flops = 2 * fwd_total_flops

    fwd_te_ms, fwd_measurement = time_func(fwd_func_te)
    bwd_te_ms, bwd_measurement = time_func(bwd_func_te)

    fwd_te_tflops = compute_tflops(fwd_total_flops, fwd_te_ms)
    bwd_te_tflops = compute_tflops(bwd_total_flops, bwd_te_ms)

    return make_forward_backward_metric_records(
        BENCHMARK_LABEL,
        "TFLOPS",
        fwd_te_ms,
        fwd_te_tflops,
        bwd_te_ms,
        bwd_te_tflops,
        fwd_measurement=fwd_measurement,
        bwd_measurement=bwd_measurement,
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
