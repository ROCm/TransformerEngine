#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""BF16 GEMM benchmarks via te.Linear.

Shapes are the four transformer projections (QKV, AttnOut, GateUp, Down)
derived from the models in models.py.
"""

import torch
import transformer_engine.pytorch as te

from driver import BenchBase, run_as_main
from models import M_SIZES, gemm_shapes

SHAPES = gemm_shapes()


class BenchGemm(BenchBase):
    params = [M_SIZES, list(SHAPES)]
    param_names = ["M", "shape"]

    def setup(self, M, shape):
        N, K = SHAPES[shape]
        dtype = torch.bfloat16
        self.linear = te.Linear(K, N, bias=False).to(device="cuda", dtype=dtype)
        self.x = torch.randn(M, K, dtype=dtype, device="cuda", requires_grad=True)
        self.grad_out = torch.randn_like(self.linear(self.x))

    def work_forward(self, M, shape):
        N, K = SHAPES[shape]
        return {"flops": 2 * M * N * K}

    def work_forward_backward(self, M, shape):
        N, K = SHAPES[shape]
        return {"flops": 3 * 2 * M * N * K}

    def time_forward(self, M, shape):
        return self._time(lambda: self.linear(self.x))

    def time_forward_backward(self, M, shape):
        t = self._time(lambda: self.linear(self.x).backward(self.grad_out))
        self.x.grad = None
        self.linear.weight.grad = None
        return t


if __name__ == "__main__":
    run_as_main(__file__)
