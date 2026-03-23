#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""FP8 casting micro-benchmarks.

Memory-bound quantization/dequantization between BF16 and FP8 formats
using Transformer Engine's quantized tensor infrastructure.
"""

import torch
from transformer_engine.pytorch import Float8CurrentScalingQuantizer
from transformer_engine_torch import DType as TE_DType

HIDDEN_SIZES = {
    "Llama3-8B": 4096,
    "Llama3-70B": 8192,
    "Llama3-405B": 16384,
    "Qwen2.5-7B": 3584,
    "Qwen2.5-72B": 8192,
}

CAST_CONFIGS = {
    "BF16_to_E4M3": ("quantize", TE_DType.kFloat8E4M3),
    "E4M3_to_BF16": ("dequantize", TE_DType.kFloat8E4M3),
    "BF16_to_E5M2": ("quantize", TE_DType.kFloat8E5M2),
    "E5M2_to_BF16": ("dequantize", TE_DType.kFloat8E5M2),
}


class BenchCasting:
    params = [[1024, 2048, 4096, 8192], list(HIDDEN_SIZES), list(CAST_CONFIGS)]
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
        self._evt = [torch.cuda.Event(enable_timing=True) for _ in range(2)]

    def time_cast(self, M, model, cast):
        self._evt[0].record()
        if self.direction == "quantize":
            self.quantizer.quantize(self.x)
        else:
            self.x.dequantize(dtype=torch.bfloat16)
        self._evt[1].record()
        torch.cuda.synchronize()
        return self._evt[0].elapsed_time(self._evt[1]) / 1000

if __name__ == "__main__":
    from driver import run_as_main
    run_as_main(__file__)
