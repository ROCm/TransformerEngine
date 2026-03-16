###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""FP8 casting micro-benchmarks.

Memory-bound quantization/dequantization between BF16 and FP8 formats.
"""

import torch

if hasattr(torch, "float8_e4m3fnuz"):
    FP8_E4M3 = torch.float8_e4m3fnuz
    FP8_E5M2 = torch.float8_e5m2fnuz
else:
    FP8_E4M3 = torch.float8_e4m3fn
    FP8_E5M2 = torch.float8_e5m2

HIDDEN_SIZES = {
    "Llama3-8B": 4096,
    "Llama3-70B": 8192,
    "Llama3-405B": 16384,
    "Qwen2.5-7B": 3584,
    "Qwen2.5-72B": 8192,
}

CAST_CONFIGS = {
    "BF16_to_E4M3": (torch.bfloat16, FP8_E4M3),
    "E4M3_to_BF16": (FP8_E4M3, torch.bfloat16),
    "BF16_to_E5M2": (torch.bfloat16, FP8_E5M2),
    "E5M2_to_BF16": (FP8_E5M2, torch.bfloat16),
}


class BenchCasting:
    params = [[1024, 2048, 4096, 8192], list(HIDDEN_SIZES), list(CAST_CONFIGS)]
    param_names = ["M", "model", "cast"]
    timeout = 120

    def setup(self, M, model, cast):
        hidden = HIDDEN_SIZES[model]
        src_dtype, self.dst_dtype = CAST_CONFIGS[cast]
        if src_dtype in (FP8_E4M3, FP8_E5M2):
            self.x = torch.randn(M, hidden, dtype=torch.bfloat16, device="cuda").to(src_dtype)
        else:
            self.x = torch.randn(M, hidden, dtype=src_dtype, device="cuda")

    def time_cast(self, M, model, cast):
        self.x.to(self.dst_dtype)
        torch.cuda.synchronize()
