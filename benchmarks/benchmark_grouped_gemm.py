#!/usr/bin/env python
###############################################################################
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os
import torch
import torch.utils.benchmark as benchmark

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
    
M_SIZE_LIST = [512, 1024, 2048, 4096]#, 8192, 16384]
EP_SIZE_LIST = [32, 16, 8]


def _generate_moe_test_cases(
    name_prefix: str,
    n_routed_experts: int,
    moe_intermediate_size: int,
    hidden_size: int,
):
    test_cases = []
    shapes_dict = {
        f"{name_prefix}-GateUP": (2 * moe_intermediate_size, hidden_size),
        f"{name_prefix}-Down": (hidden_size, moe_intermediate_size),
    }

    for ep in EP_SIZE_LIST:
        if n_routed_experts % ep != 0:
            continue
        B = n_routed_experts // ep
        if B < 1:
            continue
        for M in M_SIZE_LIST:
            for name, (N, K) in shapes_dict.items():
                for dtype in [torch.bfloat16]:
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
    return _generate_moe_test_cases(
        "DSV3", n_routed_experts=256, moe_intermediate_size=2048, hidden_size=7168
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


def make_fwd_bwd_funcs_te(x, w, group_lens, activation_dtype):
    from transformer_engine.pytorch.module.base import get_multi_stream_cublas_workspace
    from transformer_engine.pytorch.cpp_extensions import general_grouped_gemm

    B = int(group_lens.numel())
    N = int(w.shape[1])
    K = int(w.shape[2])

    m_splits = [int(v) for v in group_lens.tolist()]
    assert len(m_splits) == B
    sum_M = sum(m_splits)
    assert x.numel() > 0 and x.shape[0] == sum_M

    x_view = x.reshape(-1, x.shape[-1])
    xs = list(torch.split(x_view, m_splits))
    weights = [w[i] for i in range(B)]

    workspaces = get_multi_stream_cublas_workspace()

    # Forward output buffer
    out = torch.empty((sum_M, N), device=x.device, dtype=activation_dtype)

    def fwd_func_te():
        general_grouped_gemm(
            A=weights,
            B=xs,
            out=[out],
            out_dtype=activation_dtype,
            workspaces=workspaces,
            single_output=True,
            m_splits=m_splits,
            use_bias=False,
            bias=None,
            layout="TN",
        )
        return out

    # dx buffers
    dx = torch.empty((sum_M, K), device=x.device, dtype=activation_dtype)
    dxs = list(torch.split(dx, m_splits))

    # dw buffers
    dw_stacked = torch.empty((B, N, K), device=x.device, dtype=activation_dtype)
    dws = [dw_stacked[i] for i in range(B)]

    def bwd_func_te(grad_out):
        go = grad_out.view(-1, grad_out.shape[-1])
        splits = torch.split(go, m_splits)

        general_grouped_gemm(
            A=weights,
            B=splits,
            out=dxs,
            out_dtype=activation_dtype,
            workspaces=workspaces,
            single_output=False,
            layout="NN",
            m_splits=m_splits,
            grad=False,
            use_bias=False,
            bias=None,
        )

        general_grouped_gemm(
            A=xs,
            B=splits,
            out=dws,
            out_dtype=activation_dtype,
            workspaces=workspaces,
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


def bench_grouped_gemm(B, M, N, K, dtype):
    device = "cuda"

    x = torch.randn((B * M, K), dtype=dtype, device=device, requires_grad=True)
    w = torch.randn((B, N, K), dtype=dtype, device=device, requires_grad=True)
    group_lens = generate_grouped_gemm_group_lens(B, M, balance=True).to(device)
    print("group_lens: ", group_lens)

    os.environ["NVTE_USE_CUTLASS_GROUPED_GEMM"] = "1"
    os.environ["NVTE_CUTLASS_GROUPED_GEMM_WARN_FALLBACK"] = "1"

    # TE grouped (CK_Tile)
    x_te = x.clone().detach()
    w_te = w.clone().detach()
    fwd_func_te, bwd_func_te_inner = make_fwd_bwd_funcs_te(
        x_te, w_te, group_lens, activation_dtype=dtype
    )

    out_te = fwd_func_te()
    grad_out = torch.randn_like(out_te)
    bwd_func_te = lambda: bwd_func_te_inner(grad_out)
    dx_te, dw_te = bwd_func_te()

    # FLOPs
    fwd_total_flops = 2 * B * M * N * K
    bwd_total_flops = 2 * fwd_total_flops

    # Warmup
    for _ in range(20):
        fwd_func_te()
        bwd_func_te()

    torch.cuda.synchronize()

    # Benchmark
    n_iters = 100

    fwd_te_ms = benchmark.Timer(stmt="fn()", globals={"fn": fwd_func_te}).timeit(n_iters).mean * 1e3
    bwd_te_ms = benchmark.Timer(stmt="fn()", globals={"fn": bwd_func_te}).timeit(n_iters).mean * 1e3

    fwd_te_tflops = fwd_total_flops / (fwd_te_ms * 1e-3) / 1e12
    bwd_te_tflops = bwd_total_flops / (bwd_te_ms * 1e-3) / 1e12

    print(f"TE (CK_Tile)     Forward  {fwd_te_ms:.3f} ms | {fwd_te_tflops:.2f} TFLOPS")
    print(f"TE (CK_Tile)     Backward {bwd_te_ms:.3f} ms | {bwd_te_tflops:.2f} TFLOPS")

    return {
        "TE (CK_Tile) Forward Time (ms)": f"{fwd_te_ms:.2f}",
        "TE (CK_Tile) Forward TFLOPS": f"{fwd_te_tflops:.2f}",
        "TE (CK_Tile) Backward Time (ms)": f"{bwd_te_ms:.2f}",
        "TE (CK_Tile) Backward TFLOPS": f"{bwd_te_tflops:.2f}",
    }


if __name__ == "__main__":
    import pandas as pd

    test_cases = (
        generate_deepseekv2_lite_test_cases()
        + generate_deepseekv2_test_cases()
        + generate_deepseekv3_test_cases()
        + generate_grok_v2_test_cases()
    )

    columns = [
        "Case", "B", "M", "N", "K", "dtype",
        "TE (CK_Tile) Forward Time (ms)",
        "TE (CK_Tile) Forward TFLOPS",
        "TE (CK_Tile) Backward Time (ms)",
        "TE (CK_Tile) Backward TFLOPS",
    ]
    rows = []

    # Warmup run
    c = test_cases[0]
    print(f"\n{'='*50}")
    print(f"WARMUP: {c}")
    print(f"{'='*50}")
    bench_grouped_gemm(B=c["B"], M=c["M"], N=c["N"], K=c["K"], dtype=c["dtype"])

    for case in test_cases:
        print(f"\n{'='*50}")
        print(f"Testing: {case}")
        print(f"{'='*50}")
        try:
            metrics = bench_grouped_gemm(
                B=case["B"], M=case["M"], N=case["N"], K=case["K"], dtype=case["dtype"]
            )
            row = {
                "Case": case["Case"],
                "B": case["B"],
                "M": case["M"],
                "N": case["N"],
                "K": case["K"],
                "dtype": str(case["dtype"]),
                **metrics,
            }
            rows.append(row)
        except Exception as e:
            print(f"FAILED: {case}: {e}")
            raise

    results = pd.DataFrame(rows, columns=columns)

    out_csv = "benchmark_grouped_gemm.csv"
    results.to_csv(out_csv, index=False)
    print(f"\nResults saved to {out_csv}")
