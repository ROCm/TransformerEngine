#!/usr/bin/env python
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Grouped GEMM micro-benchmark using te.GroupedLinear across precisions and backends.

Runs under pytest (see conftest.py). Sweeps MoE grouped-GEMM shapes over BF16,
FP8, and MXFP8, crossed with the selectable kernel backend for each precision
(hipBLASLt / CK_Tile / Triton / HipKittens) and forward/backward direction.

    pytest benchmark_grouped_gemm.py --csv
    pytest benchmark_grouped_gemm.py -k "mxfp8 and hipkittens"
"""

import pytest
import torch
import transformer_engine.pytorch as te
from utils import (
    DTYPE_LIST,
    apply_backend_env,
    build_recipes,
    compute_tflops,
    direction_records,
    make_input,
)

BENCHMARK_LABEL = "Grouped GEMM"

# bf16/fp8/mxfp8 grouped GEMM (no grouped MXFP4 kernel yet -- that's a separate PR).
RECIPES = build_recipes(names=("bf16", "fp8", "mxfp8"))

# Env recipes to force a grouped-GEMM kernel backend (None unsets the var). Per the
# C++ dispatch (cublaslt_gemm.cu / rocm_gemm.cu): all-unset -> multi-stream hipBLASLt;
# CUTLASS+CK -> CK; CUTLASS+HK -> HipKittens; NVTE_USE_GROUPED_GEMM_TRITON routes bf16
# to the Triton grouped GEMM.
_CUTLASS = "NVTE_USE_CUTLASS_GROUPED_GEMM"
_CK = "NVTE_USE_CK_GROUPED_GEMM"
_HK = "NVTE_USE_HIPKITTENS_GROUPED_GEMM"
_TRITON = "NVTE_USE_GROUPED_GEMM_TRITON"

GROUPED_BACKENDS = {
    "hipblaslt":  {_CUTLASS: None, _CK: None, _HK: None, _TRITON: None},
    "ck_tile":    {_CUTLASS: "1", _CK: "1", _HK: None, _TRITON: None},
    "hipkittens": {_CUTLASS: "1", _CK: None, _HK: "1", _TRITON: None},
    "triton":     {_CUTLASS: None, _CK: None, _HK: None, _TRITON: "1"},
}

# Backends with a real choice per precision (the supported-backends table).
_BACKENDS_BY_PRECISION = {
    "bf16": ["hipblaslt", "ck_tile", "triton"],
    "fp8": ["hipblaslt", "ck_tile"],
    "mxfp8": ["hipblaslt", "hipkittens"],
}


def _backends_for(recipe):
    return _BACKENDS_BY_PRECISION.get(recipe, ["hipblaslt"])

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
                    for recipe in RECIPES:
                        test_cases.append(
                            {
                                "Case": name,
                                "B": B,
                                "M": M,
                                "N": N,
                                "K": K,
                                "dtype": dtype,
                                "recipe": recipe,
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


def bench_grouped_gemm(Case, B, M, N, K, dtype, recipe, Direction):
    device = "cuda"

    fp8_recipe = RECIPES[recipe]
    use_fp8 = fp8_recipe is not None

    group_lens = generate_grouped_gemm_group_lens(B, M, balance=True)
    m_splits = [int(v) for v in group_lens.tolist()]
    m_splits_tensor = torch.tensor(m_splits, dtype=torch.int32, device=device)
    sum_M = sum(m_splits)

    grouped_linear = te.GroupedLinear(
        B, K, N, bias=False, params_dtype=dtype, device=device,
    )
    # Rotate the activation buffer (on by default) so back-to-back grouped GEMMs
    # read different memory; GroupedLinear splits it internally per m_splits.
    next_x = make_input((sum_M, K), dtype, device=device, requires_grad=True)

    def fwd_func():
        with te.autocast(enabled=use_fp8, recipe=fp8_recipe):
            return grouped_linear(next_x(), m_splits, m_splits_tensor=m_splits_tensor)

    out_te = fwd_func()
    grad_out = torch.randn_like(out_te)

    def fwd_bwd_func():
        xb = next_x()
        with te.autocast(enabled=use_fp8, recipe=fp8_recipe):
            out = grouped_linear(xb, m_splits, m_splits_tensor=m_splits_tensor)
            out.backward(grad_out)
        xb.grad = None
        for param in grouped_linear.parameters():
            param.grad = None

    fwd_total_flops = 2 * sum_M * N * K
    return direction_records(
        Direction, BENCHMARK_LABEL, "TFLOPS", compute_tflops,
        fwd_func, fwd_bwd_func, fwd_total_flops, 2 * fwd_total_flops,
    )


def generate_cases():
    """MoE grouped-GEMM cases crossed with per-precision backend and direction."""
    base = (
        generate_deepseekv2_lite_test_cases()
        + generate_deepseekv2_test_cases()
        + generate_deepseekv3_test_cases()
        + generate_grok_v2_test_cases()
    )
    cases = []
    for b in base:
        for backend in _backends_for(b["recipe"]):
            for direction in ("fwd", "bwd"):
                cases.append({**b, "Backend": backend, "Direction": direction})
    return cases


def _case_id(c):
    return f"{c['Case']}-{c['recipe']}-{c['Backend']}-{c['Direction']}-B{c['B']}-M{c['M']}"


def pytest_generate_tests(metafunc):
    if "case" in metafunc.fixturenames:
        cases = generate_cases()
        metafunc.parametrize("case", cases, ids=[_case_id(c) for c in cases])


@pytest.mark.benchmark
def test_grouped_gemm(microbench, case, monkeypatch):
    backend = case["Backend"]
    if backend in ("ck_tile", "hipkittens") and case["B"] <= 1:
        pytest.skip(f"{backend} grouped GEMM needs num_groups > 1")
    if backend == "hipkittens" and (case["N"] % 256 or case["K"] % 256):
        pytest.skip("HipKittens grouped GEMM needs 256-aligned expert dims")
    apply_backend_env(monkeypatch, GROUPED_BACKENDS[backend])
    microbench.run(
        case,
        lambda: bench_grouped_gemm(
            case["Case"], case["B"], case["M"], case["N"], case["K"],
            case["dtype"], case["recipe"], case["Direction"],
        ),
    )


if __name__ == "__main__":
    import sys
    # Make the file runnable directly: python benchmark_grouped_gemm.py [--csv -k ...].
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
