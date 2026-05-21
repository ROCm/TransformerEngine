#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""BF16 GEMM benchmarks via te.Linear.

GEMM shapes derived from transformer layer projections:
  QKV, AttnOut, GateUp (SwiGLU), Down.
"""

import torch
import transformer_engine.pytorch as te

from driver import time_func
from shapes import M_SIZES, gemm_shapes

# Default to the shared dense-model projection shapes; mutate this dict to
# add custom shapes (e.g. SHAPES["MyModel-QKV"] = (N, K)).
SHAPES = gemm_shapes()


class BenchGemm:
    params = [M_SIZES, list(SHAPES)]
    param_names = ["M", "shape"]
    timeout = 300

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
        return time_func(lambda: self.linear(self.x))

    def time_forward_backward(self, M, shape):
        def fn():
            out = self.linear(self.x)
            out.backward(self.grad_out)
        return time_func(fn)


if __name__ == "__main__":
    from driver import main
    main(__file__)
