#!/usr/bin/env python
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Low-precision casting micro-benchmark.

Benchmarks quantization (BF16 -> low precision) and dequantization
(low precision -> BF16) for the formats used by TE training recipes:

  * FP8 per-tensor scaling: E4M3 (activations/weights) and E5M2 (gradients)
  * MXFP8 block scaling:    E4M3 and E5M2 (32-elem blocks, E8M0 scales)
  * NVFP4 block scaling:    E2M1 (16-elem blocks, E4M3 scales), no RHT
  * MXFP4 block scaling:    E2M1 (32-elem blocks, E8M0 scales)

Rowwise-only casts are measured (one output tensor), so the numbers reflect the
core cast kernel; training additionally computes the columnwise/transpose copy.
The NVFP4 random-Hadamard-transform fused cast is covered separately by
benchmarks/benchmark_rht_cast.py. Formats unsupported on the current device
are skipped.

These casts are memory-bound; we report GB/s (input + output bytes).
Output: benchmark_casting.csv (written to cwd)
"""

import pytest
import torch
import transformer_engine
import transformer_engine_torch as tex
from transformer_engine.pytorch import Float8Quantizer, MXFP8Quantizer, NVFP4Quantizer
from transformer_engine.pytorch.tensor.mxfp4_tensor import MXFP4Quantizer
from transformer_engine.pytorch.quantization import (
    check_fp8_support,
    check_mxfp4_support,
    check_mxfp8_support,
    check_nvfp4_support,
)
from utils import (
    MODEL_HIDDEN_SIZES, M_SIZE_LIST,
    apply_backend_env, time_func, compute_gbps, make_metric_record,
    make_input, rotating,
)

TE_FP8_E4M3 = tex.DType.kFloat8E4M3
TE_FP8_E5M2 = tex.DType.kFloat8E5M2
TE_FP4_E2M1 = tex.DType.kFloat4E2M1

CAST_LABEL = "Cast"


def _fp8_quantizer(fp8_dtype):
    """Per-tensor delayed-scaling FP8 quantizer factory (needs scale/amax buffers)."""

    def build():
        scale = torch.ones(1, dtype=torch.float32, device="cuda")
        amax = torch.zeros(1, dtype=torch.float32, device="cuda")
        return Float8Quantizer(scale, amax, fp8_dtype)

    return build


# Per-format cast specs:
#   (name, quantizer factory, quantized bytes/elem, support check, dequant supported).
# "quantized bytes/elem" = packed data + block-scale bytes per element:
#   FP8   : 1.0 data, per-tensor scale ~ 0          -> 1.0
#   MXFP8 : 1.0 data + E8M0 1 byte / 32-elem block  -> 1 + 1/32
#   NVFP4 : 0.5 data + E4M3 1 byte / 16-elem block  -> 0.5 + 1/16
#   MXFP4 : 0.5 data + E8M0 1 byte / 32-elem block  -> 0.5 + 1/32
# MXFP4 has no packed-FP4 dequantize kernel yet, so it runs the quantize direction only.
_CAST_FORMATS = (
    ("fp8-e4m3", _fp8_quantizer(TE_FP8_E4M3), 1.0, check_fp8_support, True),
    ("fp8-e5m2", _fp8_quantizer(TE_FP8_E5M2), 1.0, check_fp8_support, True),
    (
        "mxfp8-e4m3",
        lambda: MXFP8Quantizer(TE_FP8_E4M3, rowwise=True, columnwise=False),
        1.0 + 1.0 / 32,
        check_mxfp8_support,
        True,
    ),
    (
        "mxfp8-e5m2",
        lambda: MXFP8Quantizer(TE_FP8_E5M2, rowwise=True, columnwise=False),
        1.0 + 1.0 / 32,
        check_mxfp8_support,
        True,
    ),
    (
        "nvfp4",
        lambda: NVFP4Quantizer(
            fp4_dtype=TE_FP4_E2M1, rowwise=True, columnwise=False, with_rht=False
        ),
        0.5 + 1.0 / 16,
        check_nvfp4_support,
        True,
    ),
    (
        "mxfp4",
        lambda: MXFP4Quantizer(fp4_dtype=TE_FP4_E2M1, rowwise=True, columnwise=False),
        0.5 + 1.0 / 32,
        check_mxfp4_support,
        False,
    ),
)

DIRECTIONS = ("quantize", "dequantize")


def _active_formats():
    """Filter cast formats to those supported on the current device."""
    formats = []
    for name, make_quantizer, q_bytes_per_elem, support_check, dequant_supported in _CAST_FORMATS:
        supported, reason = support_check()
        if not supported:
            print(f"Skipping {name} casts: {reason}")
            continue
        formats.append((name, make_quantizer, q_bytes_per_elem, dequant_supported))
    return formats


# Backend axis (None unsets, so "default" is the native path even if the ambient
# env has a toggle set). "triton" flips the Triton kernel for the op being timed:
# quantize -> NVTE_USE_CAST_TRANSPOSE_TRITON, dequantize -> NVTE_USE_DEQUANTIZE_TRITON.
CAST_BACKENDS = {
    "default": {"NVTE_USE_CAST_TRANSPOSE_TRITON": None, "NVTE_USE_DEQUANTIZE_TRITON": None},
    "triton": {"NVTE_USE_CAST_TRANSPOSE_TRITON": "1", "NVTE_USE_DEQUANTIZE_TRITON": "1"},
}

_FORMATS = None


def _triton_applies(fmt, direction):
    # Cast-transpose Triton covers FP8/MXFP8/MXFP4 quantize (not NVFP4); the
    # dequantize Triton path exists only for MXFP8 (mxfp8_tensor_storage).
    if direction == "quantize":
        return fmt != "nvfp4"
    return fmt.startswith("mxfp8")


def _backends_for(fmt, direction):
    return ["default", "triton"] if _triton_applies(fmt, direction) else ["default"]


def _formats():
    """{format_name: (quantizer_factory, quantized_bytes/elem, dequant_supported)}."""
    global _FORMATS
    if _FORMATS is None:
        _FORMATS = {
            name: (make_quantizer, q_bytes, dequant_supported)
            for name, make_quantizer, q_bytes, dequant_supported in _active_formats()
        }
    return _FORMATS


def generate_cases():
    """Cross models x cast format x direction x backend x M."""
    cases = []
    for model_name, hidden in MODEL_HIDDEN_SIZES:
        for fmt_name, (_mk, _qb, dequant_supported) in _formats().items():
            for direction in DIRECTIONS:
                if direction == "dequantize" and not dequant_supported:
                    continue
                for backend in _backends_for(fmt_name, direction):
                    for M in M_SIZE_LIST:
                        cases.append({
                            "Case": model_name,
                            "Format": fmt_name,
                            "Direction": direction,
                            "Backend": backend,
                            "M": M,
                            "hidden_size": hidden,
                        })
    return cases


def _case_id(c):
    return f"{c['Case']}-{c['Format']}-{c['Direction']}-{c['Backend']}-M{c['M']}"


def bench_cast(Format, Direction, M, hidden_size):
    device = "cuda"

    make_quantizer, q_bytes_per_elem, _deq = _formats()[Format]
    numel = M * hidden_size
    quantizer = make_quantizer()

    if Direction == "quantize":
        next_x = make_input((M, hidden_size), torch.bfloat16, device=device)
        out = quantizer(next_x())
        cast_func = lambda: quantizer.quantize(next_x(), out=out)
        total_bytes = int(numel * (2 + q_bytes_per_elem))  # BF16 read + quantized write
    else:
        # Rotate a ring of quantized tensors (packed bytes can't be inferred, so hint).
        next_q = rotating(
            lambda: quantizer(
                torch.randn(M, hidden_size, dtype=torch.bfloat16, device=device)
            ),
            bytes_per_buffer=int(numel * q_bytes_per_elem),
        )
        cast_func = lambda: next_q().dequantize()
        total_bytes = int(numel * (q_bytes_per_elem + 2))  # quantized read + BF16 write

    ms, measurement = time_func(cast_func, method="blocked")
    return [make_metric_record(
        CAST_LABEL, ms, "GB/s", compute_gbps(total_bytes, ms), measurement=measurement,
    )]


def pytest_generate_tests(metafunc):
    if "case" in metafunc.fixturenames:
        cases = generate_cases()
        metafunc.parametrize("case", cases, ids=[_case_id(c) for c in cases])


@pytest.mark.benchmark
def test_cast(microbench, case, monkeypatch):
    apply_backend_env(monkeypatch, CAST_BACKENDS[case["Backend"]])
    microbench.run(
        case,
        lambda: bench_cast(case["Format"], case["Direction"], case["M"], case["hidden_size"]),
    )


if __name__ == "__main__":
    import sys
    # Make the file runnable directly: python benchmark_casting.py [--csv -k ...].
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
