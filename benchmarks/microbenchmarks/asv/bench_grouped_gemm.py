#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Grouped GEMM benchmarks via te.GroupedLinear (MoE GateUp / Down)."""

import torch
import transformer_engine.pytorch as te

from driver import BenchBase, run_as_main
from models import M_SIZES_MOE, grouped_gemm_configs

CONFIGS = grouped_gemm_configs()  # name -> (num_gemms, N, K)


class BenchGroupedGemm(BenchBase):
    params = [M_SIZES_MOE, list(CONFIGS)]
    param_names = ["M", "config"]

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
        self.grad_outs = [torch.randn_like(o) for o in self.module(self.xs)]

    def work_forward(self, M, config):
        B, N, K = CONFIGS[config]
        return {"flops": B * 2 * M * N * K}

    def work_forward_backward(self, M, config):
        B, N, K = CONFIGS[config]
        return {"flops": B * 3 * 2 * M * N * K}

    def time_forward(self, M, config):
        return self._time(lambda: self.module(self.xs))

    def time_forward_backward(self, M, config):
        t = self._time(lambda: torch.autograd.backward(self.module(self.xs), self.grad_outs))
        for x in self.xs:
            x.grad = None
        for p in self.module.parameters():
            p.grad = None
        return t


if __name__ == "__main__":
    run_as_main(__file__)
