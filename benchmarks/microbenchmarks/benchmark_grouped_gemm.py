#!/usr/bin/env python
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Grouped GEMM micro-benchmark using te.GroupedLinear across precisions and backends.

Run with ``python benchmark_grouped_gemm.py`` (a pytest module under the hood;
see conftest.py). Sweeps MoE grouped-GEMM shapes over BF16, FP8, MXFP8, and
NVFP4, crossed with the selectable kernel backend for each precision (hipBLASLt /
CK_Tile / Triton / HipKittens) and forward/backward direction.

    python benchmark_grouped_gemm.py --csv
    python benchmark_grouped_gemm.py -k "mxfp8 and hipkittens"
"""

import os
import sys
import tempfile

import pytest
import torch
import transformer_engine.pytorch as te
from transformer_engine.pytorch.utils import get_device_compute_capability
from utils import (
    DTYPE_LIST,
    apply_backend_env,
    build_recipes,
    compute_tflops,
    direction_records,
    make_input,
    te_honors_env,
)

BENCHMARK_LABEL = "Grouped GEMM"

# FIXME: Add mxfp4 when https://github.com/ROCm/TransformerEngine/pull/745 is merged
RECIPES = build_recipes(names=("bf16", "fp8", "mxfp8", "nvfp4"))

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

_BACKENDS_BY_PRECISION = {
    "bf16": ["hipblaslt", "ck_tile", "triton"],
    "fp8": ["hipblaslt", "ck_tile"],
    "mxfp8": ["hipblaslt", "hipkittens", "ck_tile"],
    "nvfp4": ["hipblaslt"],
}


# (Case, recipe, backend, direction, B, M) grouped-GEMM configs that hang hipBLASLt on
# gfx950
_GFX950_HANG_CONFIGS = {
    ("DSV3-GateUP", "bf16", "hipblaslt", "fwd", 8, 1024),
    ("DSV3-GateUP", "bf16", "hipblaslt", "bwd", 8, 1024),
    ("DSV3-GateUP", "bf16", "hipblaslt", "fwd", 16, 1024),
    ("DSV3-GateUP", "bf16", "hipblaslt", "bwd", 16, 1024),
    ("DSV3-GateUP", "bf16", "hipblaslt", "fwd", 32, 1024),
    ("DSV3-GateUP", "bf16", "hipblaslt", "bwd", 32, 1024),
    ("DSV3-Down", "nvfp4", "hipblaslt", "bwd", 8, 512),
    ("DSV3-Down", "nvfp4", "hipblaslt", "bwd", 16, 512),
    ("DSV3-Down", "nvfp4", "hipblaslt", "bwd", 32, 512),
}


# (Case, recipe, backend, direction, B, M) nvfp4 grouped configs that fail on gfx950 with
# 'NVFP4 GEMM requires ... workspace': FP4 dequant scratch exceeds the fixed grouped-GEMM
# workspace.
_GFX950_XFAIL_CONFIGS = {
    ("DSV2-GateUP", "nvfp4", "hipblaslt", "fwd", 5, 1024),
    ("DSV2-GateUP", "nvfp4", "hipblaslt", "bwd", 5, 1024),
    ("DSV2-Down", "nvfp4", "hipblaslt", "bwd", 5, 2048),
    ("DSV2-GateUP", "nvfp4", "hipblaslt", "fwd", 5, 4096),
    ("DSV2-GateUP", "nvfp4", "hipblaslt", "bwd", 5, 4096),
    ("DSV2-GateUP", "nvfp4", "hipblaslt", "fwd", 10, 1024),
    ("DSV2-GateUP", "nvfp4", "hipblaslt", "bwd", 10, 1024),
    ("DSV2-Down", "nvfp4", "hipblaslt", "bwd", 10, 2048),
    ("DSV2-GateUP", "nvfp4", "hipblaslt", "fwd", 10, 4096),
    ("DSV2-GateUP", "nvfp4", "hipblaslt", "bwd", 10, 4096),
    ("DSV2-GateUP", "nvfp4", "hipblaslt", "fwd", 20, 1024),
    ("DSV2-GateUP", "nvfp4", "hipblaslt", "bwd", 20, 1024),
    ("DSV2-Down", "nvfp4", "hipblaslt", "bwd", 20, 2048),
    ("DSV2-GateUP", "nvfp4", "hipblaslt", "fwd", 20, 4096),
    ("DSV2-GateUP", "nvfp4", "hipblaslt", "bwd", 20, 4096),
    ("DSV3-GateUP", "nvfp4", "hipblaslt", "fwd", 8, 1024),
    ("DSV3-GateUP", "nvfp4", "hipblaslt", "bwd", 8, 1024),
    ("DSV3-Down", "nvfp4", "hipblaslt", "bwd", 8, 1024),
    ("DSV3-GateUP", "nvfp4", "hipblaslt", "fwd", 8, 2048),
    ("DSV3-GateUP", "nvfp4", "hipblaslt", "bwd", 8, 2048),
    ("DSV3-GateUP", "nvfp4", "hipblaslt", "fwd", 8, 4096),
    ("DSV3-GateUP", "nvfp4", "hipblaslt", "bwd", 8, 4096),
    ("DSV3-Down", "nvfp4", "hipblaslt", "bwd", 8, 4096),
    ("DSV3-GateUP", "nvfp4", "hipblaslt", "fwd", 16, 1024),
    ("DSV3-GateUP", "nvfp4", "hipblaslt", "bwd", 16, 1024),
    ("DSV3-Down", "nvfp4", "hipblaslt", "bwd", 16, 1024),
    ("DSV3-GateUP", "nvfp4", "hipblaslt", "fwd", 16, 2048),
    ("DSV3-GateUP", "nvfp4", "hipblaslt", "bwd", 16, 2048),
    ("DSV3-GateUP", "nvfp4", "hipblaslt", "fwd", 16, 4096),
    ("DSV3-GateUP", "nvfp4", "hipblaslt", "bwd", 16, 4096),
    ("DSV3-Down", "nvfp4", "hipblaslt", "bwd", 16, 4096),
    ("DSV3-GateUP", "nvfp4", "hipblaslt", "fwd", 32, 1024),
    ("DSV3-GateUP", "nvfp4", "hipblaslt", "bwd", 32, 1024),
    ("DSV3-Down", "nvfp4", "hipblaslt", "bwd", 32, 1024),
    ("DSV3-GateUP", "nvfp4", "hipblaslt", "fwd", 32, 2048),
    ("DSV3-GateUP", "nvfp4", "hipblaslt", "bwd", 32, 2048),
    ("DSV3-GateUP", "nvfp4", "hipblaslt", "fwd", 32, 4096),
    ("DSV3-GateUP", "nvfp4", "hipblaslt", "bwd", 32, 4096),
    ("DSV3-Down", "nvfp4", "hipblaslt", "bwd", 32, 4096),
    ("Grok-V2-GateUP", "nvfp4", "hipblaslt", "fwd", 1, 512),
    ("Grok-V2-GateUP", "nvfp4", "hipblaslt", "bwd", 1, 512),
    ("Grok-V2-Down", "nvfp4", "hipblaslt", "fwd", 1, 512),
    ("Grok-V2-Down", "nvfp4", "hipblaslt", "bwd", 1, 512),
    ("Grok-V2-GateUP", "nvfp4", "hipblaslt", "fwd", 1, 1024),
    ("Grok-V2-GateUP", "nvfp4", "hipblaslt", "bwd", 1, 1024),
    ("Grok-V2-Down", "nvfp4", "hipblaslt", "fwd", 1, 1024),
    ("Grok-V2-Down", "nvfp4", "hipblaslt", "bwd", 1, 1024),
    ("Grok-V2-GateUP", "nvfp4", "hipblaslt", "fwd", 1, 2048),
    ("Grok-V2-GateUP", "nvfp4", "hipblaslt", "bwd", 1, 2048),
    ("Grok-V2-Down", "nvfp4", "hipblaslt", "fwd", 1, 2048),
    ("Grok-V2-Down", "nvfp4", "hipblaslt", "bwd", 1, 2048),
    ("Grok-V2-GateUP", "nvfp4", "hipblaslt", "fwd", 1, 4096),
    ("Grok-V2-GateUP", "nvfp4", "hipblaslt", "bwd", 1, 4096),
    ("Grok-V2-Down", "nvfp4", "hipblaslt", "fwd", 1, 4096),
    ("Grok-V2-Down", "nvfp4", "hipblaslt", "bwd", 1, 4096),
}


def _backends_for(recipe):
    return _BACKENDS_BY_PRECISION.get(recipe, ["hipblaslt"])


def _grouped_fallback_count(fn):
    """Run *fn* once with grouped-GEMM fallback warnings on; return how many of its
    grouped GEMMs fell back to hipBLASLt. Counts the dispatcher's per-call NVTE_WARN
    (HipKittens' own messages are latched once per process in C++); it is C++ stderr,
    so capture at the fd level rather than via ``warnings``."""
    prev = os.environ.get("NVTE_CUTLASS_GROUPED_GEMM_WARN_FALLBACK")
    os.environ["NVTE_CUTLASS_GROUPED_GEMM_WARN_FALLBACK"] = "1"
    saved = os.dup(2)
    try:
        with tempfile.TemporaryFile(mode="w+b") as tmp:
            sys.stderr.flush()
            os.dup2(tmp.fileno(), 2)
            try:
                fn()
            finally:
                sys.stderr.flush()
                os.dup2(saved, 2)
            tmp.seek(0)
            out = tmp.read().decode(errors="ignore")
        return out.count("Fallback to cuBLAS grouped GEMM")
    finally:
        os.close(saved)
        if prev is None:
            os.environ.pop("NVTE_CUTLASS_GROUPED_GEMM_WARN_FALLBACK", None)
        else:
            os.environ["NVTE_CUTLASS_GROUPED_GEMM_WARN_FALLBACK"] = prev

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
    return _generate_moe_test_cases(
        "DSV3", n_routed_experts=256, moe_intermediate_size=2048, hidden_size=7168,
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

    if os.environ.get(_CUTLASS) == "1":
        # bwd is derived as fwd_bwd - fwd, so only fallbacks beyond the forward pass count.
        fwd_fallbacks = _grouped_fallback_count(fwd_func)
        fell_back = (fwd_fallbacks > 0 if Direction == "fwd"
                     else _grouped_fallback_count(fwd_bwd_func) > fwd_fallbacks)
        if fell_back:
            name = "HipKittens" if os.environ.get(_HK) == "1" else "CK"
            pytest.skip(f"{name} grouped GEMM fell back to hipBLASLt for this config")

    fwd_total_flops = 2 * sum_M * N * K
    # hipBLASLt grouped GEMM runs the experts across compute streams; the profiler
    # under-counts concurrent kernels, so measure elapsed device time by makespan.
    return direction_records(
        Direction, BENCHMARK_LABEL, "TFLOPS", compute_tflops,
        fwd_func, fwd_bwd_func, fwd_total_flops, 2 * fwd_total_flops,
        kernel_method="event",
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
def test_grouped_gemm(request, microbench, case, monkeypatch):
    backend = case["Backend"]
    sig = (case["Case"], case["recipe"], backend, case["Direction"], case["B"], case["M"])
    if get_device_compute_capability() == (9, 5) and sig in _GFX950_XFAIL_CONFIGS:
        request.node.add_marker(pytest.mark.xfail(
            reason="nvfp4 grouped FP4 dequant workspace too small on gfx950",
            raises=RuntimeError, strict=False))
    if get_device_compute_capability() == (9, 5) and sig in _GFX950_HANG_CONFIGS:
        pytest.skip("known hipBLASLt grouped-GEMM hang on gfx950")
    if backend in ("ck_tile", "hipkittens") and case["B"] <= 1:
        pytest.skip(f"{backend} grouped GEMM needs num_groups > 1")
    if backend == "hipkittens" and (case["N"] % 256 or case["K"] % 256):
        pytest.skip("HipKittens grouped GEMM needs 256-aligned expert dims")
    if backend == "ck_tile" and case["recipe"] == "mxfp8" and get_device_compute_capability() != (12, 5):
        pytest.skip("CK MXFP8 grouped GEMM is gfx1250-only")
    # Skip a forced backend when the build doesn't honor the toggles it enables,
    # so old builds show no data instead of silently measuring hipBLASLt.
    required = [k for k, v in GROUPED_BACKENDS[backend].items() if v is not None]
    if required and not all(te_honors_env(k) for k in required):
        pytest.skip(f"{backend} grouped GEMM backend not available in this TE build")
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
