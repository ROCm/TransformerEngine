#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Benchmarks quantization (BF16 -> FP8) and dequantization (FP8 -> BF16) for
both E4M3 (activations/weights) and E5M2 (gradients) formats.

Shapes are (M, hidden_size) matching activation tensors from the shared
dense-model configs. These casts are memory-bound; we report GB/s.
"""

import torch
from transformer_engine.pytorch import Float8CurrentScalingQuantizer
from transformer_engine_torch import DType as TE_DType

from driver import time_func
from shapes import M_SIZES, hidden_sizes

# Default to the shared per-architecture hidden sizes; mutate to add custom
# entries (e.g. HIDDEN_SIZES["MyModel"] = 5120).
HIDDEN_SIZES = hidden_sizes()

CAST_CONFIGS = {
    "BF16_to_E4M3": ("quantize", TE_DType.kFloat8E4M3),
    "E4M3_to_BF16": ("dequantize", TE_DType.kFloat8E4M3),
    "BF16_to_E5M2": ("quantize", TE_DType.kFloat8E5M2),
    "E5M2_to_BF16": ("dequantize", TE_DType.kFloat8E5M2),
}


class BenchCasting:
    params = [M_SIZES, list(HIDDEN_SIZES), list(CAST_CONFIGS)]
    param_names = ["M", "model", "cast"]
    timeout = 120

    def setup(self, M, model, cast):
        hidden = HIDDEN_SIZES[model]
        direction, fp8_dtype = CAST_CONFIGS[cast]
        self.direction = direction
        quantizer = Float8CurrentScalingQuantizer(
            fp8_dtype=fp8_dtype,
            device=torch.device("cuda"),
            rowwise=True,
            columnwise=False,
        )
        if direction == "dequantize":
            bf16_tensor = torch.randn(M, hidden, dtype=torch.bfloat16, device="cuda")
            self.x = quantizer.quantize(bf16_tensor)
        else:
            self.x = torch.randn(M, hidden, dtype=torch.bfloat16, device="cuda")
            self.quantizer = quantizer

    def work_cast(self, M, model, cast):
        hidden = HIDDEN_SIZES[model]
        # Read input (1B FP8 or 2B BF16) + write output + scale (~hidden bytes total)
        # Approximated as 3 bytes per element either direction.
        return {"bytes": M * hidden * 3}

    def time_cast(self, M, model, cast):
        if self.direction == "quantize":
            return time_func(lambda: self.quantizer.quantize(self.x))
        return time_func(lambda: self.x.dequantize(dtype=torch.bfloat16))


if __name__ == "__main__":
    from driver import main
    main(__file__)
