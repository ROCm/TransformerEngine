#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Quantization (BF16 -> FP8) and dequantization (FP8 -> BF16) benchmarks.

Covers E4M3 (activations/weights) and E5M2 (gradients). These casts are
memory-bound, so we report GB/s (input + output bytes).
"""

import torch
from transformer_engine.pytorch import Float8CurrentScalingQuantizer
from transformer_engine_torch import DType as TE_DType

from driver import BenchBase, run_as_main
from models import M_SIZES, hidden_sizes

HIDDEN = hidden_sizes()

# cast name -> (direction, fp8 dtype)
CAST_CONFIGS = {
    "BF16_to_E4M3": ("quantize", TE_DType.kFloat8E4M3),
    "E4M3_to_BF16": ("dequantize", TE_DType.kFloat8E4M3),
    "BF16_to_E5M2": ("quantize", TE_DType.kFloat8E5M2),
    "E5M2_to_BF16": ("dequantize", TE_DType.kFloat8E5M2),
}


class BenchCasting(BenchBase):
    params = [M_SIZES, list(HIDDEN), list(CAST_CONFIGS)]
    param_names = ["M", "model", "cast"]

    def setup(self, M, model, cast):
        hidden = HIDDEN[model]
        direction, fp8_dtype = CAST_CONFIGS[cast]
        quantizer = Float8CurrentScalingQuantizer(
            fp8_dtype=fp8_dtype, device=torch.device("cuda"),
            rowwise=True, columnwise=False,
        )
        if direction == "dequantize":
            x = quantizer.quantize(torch.randn(M, hidden, dtype=torch.bfloat16, device="cuda"))
            self._call = lambda: x.dequantize(dtype=torch.bfloat16)
        else:
            x = torch.randn(M, hidden, dtype=torch.bfloat16, device="cuda")
            self._call = lambda: quantizer.quantize(x)

    def work_cast(self, M, model, cast):
        # quantize: read BF16 (2B) + write FP8 (1B) + scale; dequantize: the
        # reverse -- 3 bytes/element either way.
        return {"bytes": M * HIDDEN[model] * 3}

    def time_cast(self, M, model, cast):
        return self._time(self._call)


if __name__ == "__main__":
    run_as_main(__file__)
