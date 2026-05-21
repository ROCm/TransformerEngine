#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Grouped GEMM benchmarks via te.GroupedLinear.

MoE model configurations with GateUp and Down projections, swept over a
range of expert-parallel sizes.
"""

import torch
import transformer_engine.pytorch as te

from driver import time_func
from shapes import grouped_gemm_configs

# Grouped GEMM scales with B, so we sweep smaller M than dense benchmarks
# to keep the working set and runtime reasonable.
M_SIZES = [512, 1024, 2048, 4096]

# Default to the shared MoE configs; mutate to add custom shapes
# (e.g. CONFIGS["MyMoE_EP4-GateUp"] = (B, N, K)).
CONFIGS = grouped_gemm_configs()


class BenchGroupedGemm:
    params = [M_SIZES, list(CONFIGS)]
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
