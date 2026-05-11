#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Benchmarks quantization (BF16 -> FP8) and dequantization (FP8 -> BF16) for
both E4M3 (activations/weights) and E5M2 (gradients) formats.

Shapes are (M, hidden_size) matching the activation tensors from models:
  - Llama 3.1 8B, 70B, 405B
  - Qwen 2.5  7B, 72B

These casts are memory-bound; we report GB/s (input + output bytes).

Sources for model configs:
  https://huggingface.co/meta-llama/Llama-3.1-8B/blob/main/config.json
  https://huggingface.co/meta-llama/Llama-3.1-70B/blob/main/config.json
  https://huggingface.co/meta-llama/Llama-3.1-405B/blob/main/config.json
  https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json
  https://huggingface.co/Qwen/Qwen2.5-72B-Instruct/blob/main/config.json
"""

import torch
from transformer_engine.pytorch import Float8CurrentScalingQuantizer
from transformer_engine_torch import DType as TE_DType

from driver import time_func

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
    from driver import run_as_main
    run_as_main(__file__)
