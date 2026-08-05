#!/usr/bin/env python
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Compare permute-free gather-GEMM vs permute + grouped GEMM backends on ROCm.

Backends:
  - permute_free: TE permute-free (FlyDSL) route-list gather-in-GEMM (align + GEMM), no permute
  - hipblaslt:    TE multistream hipBLASLt grouped GEMM after moe_permute
  - ck:           TE CK grouped GEMM after moe_permute (NVTE_USE_CUTLASS_GROUPED_GEMM=1)
  - triton:       AITER Triton grouped GEMM after moe_permute

Run from this directory::

    python benchmark_perm_free_grouped_gemm.py
    python benchmark_perm_free_grouped_gemm.py --quick --csv
"""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Tuple

import torch
from torch.utils.cpp_extension import IS_HIP_EXTENSION

from benchmark_grouped_gemm import (
    EP_SIZE_LIST,
    GROUPED_GEMM_M_SIZE_LIST,
    generate_deepseekv2_lite_test_cases,
    generate_deepseekv2_test_cases,
    generate_deepseekv3_test_cases,
    generate_grok_v2_test_cases,
    generate_qwen3_235b_test_cases,
)
from utils import compute_tflops, make_metric_record, make_parser, run_benchmarks, time_func

DEFAULT_TOPK = 8
BACKENDS = ("permute_free", "hipblaslt", "ck", "triton")
# ``permute_free_act`` is an opt-in, GateUP-only backend: the permute-free FC1 gather-GEMM
# with the gated SiLU activation (``silu(gate) * up``) fused into the epilogue. It is not in
# the default sweep (it only applies to gate+up shapes); request it via ``--backends``.
KNOWN_BACKENDS = BACKENDS + ("permute_free_act",)

# Training phases benchmarked (and reported) separately in --train mode.
PHASES = ("fwd", "dgrad", "wgrad")

# Short backend labels for the per-phase --train metric names.
_BACKEND_SHORT = {
    "permute_free": "PermuteFree",
    "permute_free_act": "PermuteFreeAct",
    "hipblaslt": "hipBLASLt",
    "ck": "CK",
    "triton": "Triton",
}


def _require_rocm():
    if not IS_HIP_EXTENSION or not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires ROCm (HIP extension) and a CUDA device.")


def _make_routing(num_tokens: int, num_experts: int, topk: int, device: str, seed: int):
    from transformer_engine.pytorch.moe import MoERoutingMetadata

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    logits = torch.randn(num_tokens, num_experts, device=device, generator=gen)
    probs = torch.softmax(logits, dim=-1)
    topk_weights, topk_ids = torch.topk(probs, k=topk, dim=-1)
    topk_ids = topk_ids.to(torch.int32)
    topk_weights = topk_weights.to(torch.float32)
    # MoERoutingMetadata now takes a boolean routing_map as its primary input; build it
    # from the sampled topk_ids so the permute-free and permute backends see identical
    # routing. Passing ``topk`` tightens the block-padded over-allocation (em_max). topk_ids/
    # topk_weights are attached for the permute (moe_permute) backends.
    routing_map = torch.zeros(num_tokens, num_experts, dtype=torch.bool, device=device)
    routing_map.scatter_(1, topk_ids.to(torch.long), True)
    routing = MoERoutingMetadata(routing_map=routing_map, num_experts=num_experts, topk=topk)
    routing.topk_ids = topk_ids
    routing.topk_weights = topk_weights
    return routing


def _m_splits_from_topk(topk_ids: torch.Tensor, num_experts: int) -> List[int]:
    counts = torch.bincount(topk_ids.reshape(-1).long(), minlength=num_experts)
    return [int(v) for v in counts.tolist()]


def _permute_hidden(
    hidden: torch.Tensor, topk_ids: torch.Tensor, num_tokens: int, topk: int
) -> torch.Tensor:
    from transformer_engine.pytorch import moe_permute

    num_out_tokens = num_tokens * topk
    permuted, _ = moe_permute(hidden, topk_ids, num_out_tokens, map_type="index")
    return permuted


def _grouped_gemm_hip(
    permuted: torch.Tensor,
    weights: torch.Tensor,
    m_splits: List[int],
    *,
    use_ck: bool,
) -> torch.Tensor:
    from transformer_engine.pytorch.cpp_extensions import general_grouped_gemm

    prev = os.environ.get("NVTE_USE_CUTLASS_GROUPED_GEMM")
    os.environ["NVTE_USE_CUTLASS_GROUPED_GEMM"] = "1" if use_ck else "0"
    try:
        b = len(m_splits)
        n = int(weights.shape[1])
        sum_m = sum(m_splits)
        xs = list(torch.split(permuted.view(sum_m, -1), m_splits))
        weight_list = [weights[i] for i in range(b)]
        out = torch.empty((sum_m, n), device=permuted.device, dtype=permuted.dtype)
        general_grouped_gemm(
            A=weight_list,
            B=xs,
            out=[out],
            quantization_params=[None] * b,
            out_dtype=permuted.dtype,
            single_output=True,
            m_splits=m_splits,
            use_bias=False,
            bias=None,
            layout="TN",
        )
        return out
    finally:
        if prev is None:
            os.environ.pop("NVTE_USE_CUTLASS_GROUPED_GEMM", None)
        else:
            os.environ["NVTE_USE_CUTLASS_GROUPED_GEMM"] = prev


def _grouped_gemm_triton(
    permuted: torch.Tensor,
    weights: torch.Tensor,
    m_splits: List[int],
) -> torch.Tensor:
    from transformer_engine.pytorch.triton_kernels.grouped_gemm import general_grouped_gemm_triton

    b = len(m_splits)
    n = int(weights.shape[1])
    sum_m = sum(m_splits)
    xs = list(torch.split(permuted.view(sum_m, -1), m_splits))
    weight_list = [weights[i] for i in range(b)]
    out = torch.empty((sum_m, n), device=permuted.device, dtype=permuted.dtype)
    general_grouped_gemm_triton(
        A=weight_list,
        B=xs,
        out=[out],
        quantization_params=[None] * b,
        out_dtype=permuted.dtype,
        single_output=True,
        m_splits=m_splits,
        use_bias=False,
        bias=None,
        layout="TN",
    )
    return out


def _permute_free_gemm(
    hidden: torch.Tensor,
    weights: torch.Tensor,
    routing,
) -> torch.Tensor:
    from transformer_engine.pytorch.moe import (
        permute_free_grouped_gemm_bf16,
    )

    return permute_free_grouped_gemm_bf16(hidden, weights, routing)


def _permute_free_act_gemm(
    hidden: torch.Tensor,
    weights: torch.Tensor,
    routing,
) -> torch.Tensor:
    """Permute-free FC1 gather-GEMM + standalone SiLU gated activation (no in-kernel fusion).

    ``weights`` is the gate+up projection ``[E, 2F, K]``; returns the F-wide activated buffer.
    """
    from transformer_engine.pytorch.moe import (
        permute_free_gated_act_recompute,
        permute_free_grouped_gemm_bf16,
    )

    preact = permute_free_grouped_gemm_bf16(hidden, weights, routing)
    return permute_free_gated_act_recompute(preact, routing, activation="silu")


def _build_backend_fns(
    hidden: torch.Tensor,
    weights: torch.Tensor,
    routing,
    m_splits: List[int],
    num_tokens: int,
    topk: int,
    is_gated: bool = False,
) -> Dict[str, Callable[[], torch.Tensor]]:
    fns: Dict[str, Callable[[], torch.Tensor]] = {}

    def hipblaslt_fn():
        permuted = _permute_hidden(hidden, routing.topk_ids, num_tokens, topk)
        return _grouped_gemm_hip(permuted, weights, m_splits, use_ck=False)

    def ck_fn():
        permuted = _permute_hidden(hidden, routing.topk_ids, num_tokens, topk)
        return _grouped_gemm_hip(permuted, weights, m_splits, use_ck=True)

    def triton_fn():
        permuted = _permute_hidden(hidden, routing.topk_ids, num_tokens, topk)
        return _grouped_gemm_triton(permuted, weights, m_splits)

    fns["hipblaslt"] = hipblaslt_fn
    fns["ck"] = ck_fn
    fns["triton"] = triton_fn
    fns["permute_free"] = lambda: _permute_free_gemm(hidden, weights, routing)
    # GateUP-only: permute-free FC1 with the fused gated SiLU epilogue.
    if is_gated:
        fns["permute_free_act"] = lambda: _permute_free_act_gemm(hidden, weights, routing)
    return fns


def _build_train_phase_fns(
    hidden: torch.Tensor,
    weights: torch.Tensor,
    routing,
    m_splits: List[int],
    num_tokens: int,
    topk: int,
    dtype: torch.dtype,
) -> Dict[str, Dict[str, Callable[[], torch.Tensor]]]:
    """Per-phase (fwd / dgrad / wgrad) closures for each backend.

    Returns ``{backend: {phase: fn}}`` so each phase can be timed independently.

    - permute_free: gather fwd + gather dgrad + fused wgrad (FlyDSL / Triton), matching the
      ``_GroupedLinear`` permute-free autograd path. Every kernel gathers from token-major
      tensors, so no standalone permute/unpermute is charged.
    - hipblaslt/ck: the traditional permuted grouped GEMMs. The token permute is charged to
      ``fwd`` and the dgrad unpermute to ``dgrad`` (both produce a token-major-equivalent
      result); ``wgrad`` reuses the operands already permuted in fwd, so it is GEMM-only --
      mirroring a real training iteration where the permute is paid once. (Triton GG is
      excluded from --train: its grouped backend does not expose the NN/NT grad layouts.)
    """
    from transformer_engine.pytorch import moe_permute, moe_unpermute
    from transformer_engine.pytorch.cpp_extensions import general_grouped_gemm
    from transformer_engine.pytorch.moe import (
        permute_free_grouped_gemm_bf16,
        permute_free_grouped_gemm_bf16_dgrad,
        permute_free_grouped_gemm_bf16_wgrad,
    )

    device = hidden.device
    b = len(m_splits)
    n = int(weights.shape[1])
    k = int(weights.shape[2])
    total_m = sum(m_splits)
    weight_list = [weights[i] for i in range(b)]
    weights_shape = (b, n, k)

    # Run the permute-free forward once (outside timing) to finalize the block-padded align on
    # ``routing`` (the fwd enforces a v3 block-size floor, which fixes em_max) and to learn the
    # padded slot extent. The dgrad/wgrad now consume the block-padded ``[em_max, out_features]``
    # slot gradient -- i.e. the gradient of this fwd output -- not a compact ``[num_routes]`` one.
    em_max = int(permute_free_grouped_gemm_bf16(hidden, weights, routing).shape[0])

    # Upstream gradients. ``grad_out_pf`` is the block-padded slot grad for the permute-free
    # dgrad/wgrad; the traditional path keeps its compact expert-sorted ``[total_m, n]`` grad.
    # Allocated once, outside the timed region.
    grad_out_pf = torch.randn(em_max, n, dtype=dtype, device=device)
    grad_out_perm = torch.randn(total_m, n, dtype=dtype, device=device)
    grad_splits = list(torch.split(grad_out_perm, m_splits))

    # Pre-permuted activations reused by the traditional dgrad/wgrad phases. The permute
    # itself is (re)charged inside the traditional fwd phase below.
    permuted_hidden, row_id_map = moe_permute(
        hidden, routing.topk_ids, total_m, map_type="index"
    )
    xs_perm = list(torch.split(permuted_hidden.view(total_m, -1), m_splits))

    # --- permute-free phases (gather-in-GEMM; token-major in/out) ---
    def pf_fwd():
        return permute_free_grouped_gemm_bf16(hidden, weights, routing)

    def pf_dgrad():
        return permute_free_grouped_gemm_bf16_dgrad(grad_out_pf, weights, routing)

    def pf_wgrad():
        return permute_free_grouped_gemm_bf16_wgrad(
            hidden, grad_out_pf, weights_shape, routing
        )

    # --- traditional (permute + grouped GEMM) phases ---
    def _with_ck_env(use_ck: bool, fn: Callable[[], torch.Tensor]):
        def run():
            prev = os.environ.get("NVTE_USE_CUTLASS_GROUPED_GEMM")
            os.environ["NVTE_USE_CUTLASS_GROUPED_GEMM"] = "1" if use_ck else "0"
            try:
                return fn()
            finally:
                if prev is None:
                    os.environ.pop("NVTE_USE_CUTLASS_GROUPED_GEMM", None)
                else:
                    os.environ["NVTE_USE_CUTLASS_GROUPED_GEMM"] = prev

        return run

    def _trad_fwd():
        permuted, _ = moe_permute(hidden, routing.topk_ids, total_m, map_type="index")
        xs = list(torch.split(permuted.view(total_m, -1), m_splits))
        out = torch.empty((total_m, n), device=device, dtype=dtype)
        general_grouped_gemm(
            A=weight_list, B=xs, out=[out], quantization_params=[None] * b,
            out_dtype=dtype, single_output=True, m_splits=m_splits,
            use_bias=False, bias=None, layout="TN",
        )
        return out

    def _trad_dgrad():
        dx_buf = torch.empty((total_m, k), device=device, dtype=dtype)
        dxs = list(torch.split(dx_buf, m_splits))
        general_grouped_gemm(
            A=weight_list, B=grad_splits, out=dxs, quantization_params=[None] * b,
            out_dtype=dtype, single_output=False, m_splits=m_splits, grad=False,
            use_bias=False, bias=None, layout="NN",
        )
        return moe_unpermute(dx_buf, row_id_map, map_type="index")

    def _trad_wgrad():
        dw = torch.empty((b, n, k), device=device, dtype=dtype)
        dws = [dw[i] for i in range(b)]
        general_grouped_gemm(
            A=xs_perm, B=grad_splits, out=dws, quantization_params=[None] * b,
            out_dtype=dtype, single_output=False, m_splits=m_splits, grad=False,
            use_bias=False, bias=None, layout="NT",
        )
        return dw

    def _trad_phase_fns(use_ck: bool) -> Dict[str, Callable[[], torch.Tensor]]:
        return {
            "fwd": _with_ck_env(use_ck, _trad_fwd),
            "dgrad": _with_ck_env(use_ck, _trad_dgrad),
            "wgrad": _with_ck_env(use_ck, _trad_wgrad),
        }

    return {
        "permute_free": {"fwd": pf_fwd, "dgrad": pf_dgrad, "wgrad": pf_wgrad},
        "hipblaslt": _trad_phase_fns(use_ck=False),
        "ck": _trad_phase_fns(use_ck=True),
    }


def bench_moe_grouped_gemm_backends(
    Case: str,
    B: int,
    M: int,
    N: int,
    K: int,
    dtype: torch.dtype,
    topk: int = DEFAULT_TOPK,
    seed: int = 0,
    backends: Tuple[str, ...] = BACKENDS,
    train: bool = False,
):
    _require_rocm()
    device = "cuda"
    num_tokens = M
    num_experts = B

    effective_topk = min(topk, num_experts)
    if effective_topk < 1:
        raise ValueError(f"Need at least one expert (B={num_experts}).")

    hidden = torch.randn(num_tokens, K, dtype=dtype, device=device)
    weights = torch.randn(num_experts, N, K, dtype=dtype, device=device)
    routing = _make_routing(num_tokens, num_experts, effective_topk, device, seed)
    m_splits = _m_splits_from_topk(routing.topk_ids, num_experts)

    total_m = num_tokens * effective_topk

    if train:
        # Per-phase analysis: time fwd / dgrad / wgrad independently for each backend.
        phase_fns = _build_train_phase_fns(
            hidden, weights, routing, m_splits, num_tokens, effective_topk, dtype
        )
        # Each of fwd / dgrad / wgrad performs a ~2*total_m*N*K FLOP GEMM.
        phase_flops = 2 * total_m * N * K

        # Warmup every selected phase (also builds the permute-free align buffers so the
        # dgrad/wgrad timings reflect cache reuse, as in a real iteration).
        for name in backends:
            if name not in phase_fns:
                continue
            for phase in PHASES:
                phase_fns[name][phase]()
        torch.cuda.synchronize()

        # Emit records phase-major so the printout groups fwd, then dgrad, then wgrad.
        records = []
        for phase in PHASES:
            for name in backends:
                if name not in phase_fns:
                    continue
                ms, measurement = time_func(phase_fns[name][phase])
                records.append(
                    make_metric_record(
                        f"{phase} \u00b7 {_BACKEND_SHORT[name]}",
                        ms,
                        "TFLOPS",
                        compute_tflops(phase_flops, ms),
                        measurement=measurement,
                    )
                )
        return records

    # GateUP shapes carry a gate+up projection (out_features == 2F, even), which is the only
    # case where the fused gated-activation permute-free backend applies.
    is_gated = Case.endswith("GateUP")
    backend_fns = _build_backend_fns(
        hidden, weights, routing, m_splits, num_tokens, effective_topk, is_gated=is_gated
    )
    fwd_flops = 2 * total_m * N * K

    # Permute-free backends emit a padded/route-major (or F-wide, for the activation fusion)
    # buffer, so their output shape differs from the permuted [total_m, N] result.
    _permute_free_names = ("permute_free", "permute_free_act")

    # Warmup + correctness spot-check (permute-free vs hipblaslt layout differs; skip allclose)
    for name in backends:
        if name not in backend_fns:
            continue
        out = backend_fns[name]()
        torch.cuda.synchronize()
        if name not in _permute_free_names:
            assert out.shape == (total_m, N), f"{name}: bad shape {out.shape}"

    records = []
    for name in backends:
        if name not in backend_fns:
            continue
        label = {
            "permute_free": "PermuteFree Gather-GEMM",
            "permute_free_act": "PermuteFree Gather-GEMM + SiLU",
            "hipblaslt": "Permute+hipBLASLt Grouped GEMM",
            "ck": "Permute+CK Grouped GEMM",
            "triton": "Permute+AITER Triton Grouped GEMM",
        }[name]
        ms, measurement = time_func(backend_fns[name])
        records.append(
            make_metric_record(
                label,
                ms,
                "TFLOPS",
                compute_tflops(fwd_flops, ms),
                measurement=measurement,
            )
        )
    return records


def _filter_test_cases(test_cases, quick: bool):
    if not quick:
        return test_cases
    allowed_m = {512, 1024}
    filtered = [c for c in test_cases if c["M"] in allowed_m and c["Case"].endswith("-Down")]
    return filtered[:4] if filtered else test_cases[:2]


def main():
    base = make_parser(description="Benchmark MoE grouped GEMM backends on ROCm.")
    base.add_argument("--quick", action="store_true", help="Run a small subset of shapes.")
    base.add_argument(
        "--train",
        action="store_true",
        help="Benchmark a full training iteration (fwd + dgrad + wgrad) instead of fwd only.",
    )
    base.add_argument("--topk", type=int, default=DEFAULT_TOPK, help="MoE top-k routing.")
    base.add_argument(
        "--backends",
        type=str,
        default=",".join(BACKENDS),
        help=f"Comma-separated backends (default: {','.join(BACKENDS)}).",
    )
    base.add_argument(
        "--case-prefix",
        type=str,
        default=None,
        help="Only run cases whose Case name starts with this prefix (e.g. DSV3).",
    )
    base.add_argument(
        "--include-dsv3-gateup",
        action="store_true",
        help="Also benchmark DSV3-GateUP (skipped by default; known to run on gfx950).",
    )
    args = base.parse_args()

    _require_rocm()

    test_cases = (
        generate_deepseekv2_lite_test_cases()
        + generate_deepseekv2_test_cases()
        + generate_deepseekv3_test_cases(include_gateup=args.include_dsv3_gateup)
        + generate_grok_v2_test_cases()
        + generate_qwen3_235b_test_cases()
    )
    test_cases = _filter_test_cases(test_cases, args.quick)
    if args.case_prefix:
        prefix = args.case_prefix
        test_cases = [
            c
            for c in test_cases
            if c["Case"].startswith(prefix)
            and not (prefix == "DSV2-" and c["Case"].startswith("DSV2-Lite"))
        ]
    for case in test_cases:
        case["topk"] = min(args.topk, case["B"])
        case["seed"] = 42

    selected_backends = tuple(b.strip() for b in args.backends.split(",") if b.strip())
    for b in selected_backends:
        if b not in KNOWN_BACKENDS:
            raise ValueError(f"Unknown backend {b!r}, expected one of {KNOWN_BACKENDS}")

    def bench_fn(**case):
        return bench_moe_grouped_gemm_backends(
            **case, backends=selected_backends, train=args.train
        )

    run_benchmarks(
        test_cases=test_cases,
        bench_fn=bench_fn,
        param_columns=["Case", "B", "M", "N", "K", "dtype", "topk"],
        args=args,
    )


if __name__ == "__main__":
    main()
