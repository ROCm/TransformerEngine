#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""FP8 GEMM benchmarks via te.Linear under fp8_autocast.

Same shapes as bench_gemm.py but with FP8 (HYBRID) quantized compute.
"""

import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling, Format

from driver import BenchBase, run_as_main
from models import M_SIZES, gemm_shapes

SHAPES = gemm_shapes()
FP8_RECIPE = DelayedScaling(
    fp8_format=Format.HYBRID, amax_history_len=16, amax_compute_algo="max",
)


class BenchGemmFP8(BenchBase):
    params = [M_SIZES, list(SHAPES)]
    param_names = ["M", "shape"]

    def setup(self, M, shape):
        N, K = SHAPES[shape]
        dtype = torch.bfloat16
        self.linear = te.Linear(K, N, bias=False).to(device="cuda", dtype=dtype)
        self.x = torch.randn(M, K, dtype=dtype, device="cuda", requires_grad=True)
        self.grad_out = torch.randn(M, N, dtype=dtype, device="cuda")

    def work_forward(self, M, shape):
        N, K = SHAPES[shape]
        return {"flops": 2 * M * N * K}

    def work_forward_backward(self, M, shape):
        N, K = SHAPES[shape]
        return {"flops": 3 * 2 * M * N * K}

    def _forward(self):
        with te.fp8_autocast(enabled=True, fp8_recipe=FP8_RECIPE):
            return self.linear(self.x)

    def time_forward(self, M, shape):
        return self._time(self._forward)

    def time_forward_backward(self, M, shape):
        t = self._time(lambda: self._forward().backward(self.grad_out))
        self.x.grad = None
        self.linear.weight.grad = None
        return t


if __name__ == "__main__":
    run_as_main(__file__)
