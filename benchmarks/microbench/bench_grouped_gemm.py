#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Grouped GEMM benchmarks via te.GroupedLinear.

MoE model configurations with GateUp and Down projections.
Configurations are based on:
https://github.com/AMD-AGI/Primus-Turbo/blob/main/benchmark/ops/config.py
"""

import torch
import transformer_engine.pytorch as te

from driver import time_func

# (n_routed_experts, moe_intermediate_size, hidden_size)
MOE_MODELS = {
    "DSV2-Lite": (64, 1408, 2048),
    "DSV2":      (160, 1536, 5120),
    "DSV3":      (256, 2048, 7168),
    "Grok-V2":   (8, 16384, 8192),
}

# Build (config_key -> (num_gemms, N, K)) mapping
CONFIGS = {}
for model, (n_experts, inter, hidden) in MOE_MODELS.items():
    for ep in [32, 16, 8]:
        if n_experts % ep != 0:
            continue
        B = n_experts // ep
        CONFIGS[f"{model}_EP{ep}-GateUp"] = (B, 2 * inter, hidden)
        CONFIGS[f"{model}_EP{ep}-Down"] = (B, hidden, inter)


class BenchGroupedGemm:
    params = [[512, 1024, 2048, 4096], list(CONFIGS)]
    param_names = ["M", "config"]
    timeout = 300

    def setup(self, M, config):
        B, N, K = CONFIGS[config]
        dtype = torch.bfloat16

        self.module = te.GroupedLinear(
            num_gemms=B, in_features=K, out_features=N, bias=False,
        ).to(device="cuda", dtype=dtype)

        self.xs = [
            torch.randn(M, K, dtype=dtype, device="cuda", requires_grad=True)
            for _ in range(B)
        ]
        outs = self.module(self.xs)
        self.grad_outs = [torch.randn_like(o) for o in outs]

    def work_forward(self, M, config):
        B, N, K = CONFIGS[config]
        return {"flops": B * 2 * M * N * K}

    def work_forward_backward(self, M, config):
        B, N, K = CONFIGS[config]
        return {"flops": B * 3 * 2 * M * N * K}

    def time_forward(self, M, config):
        return time_func(lambda: self.module(self.xs))

    def time_forward_backward(self, M, config):
        def fn():
            outs = self.module(self.xs)
            torch.autograd.backward(outs, self.grad_outs)
        return time_func(fn)


if __name__ == "__main__":
    from driver import main
    main(__file__)
