#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
FP8 GEMM benchmarks via te.Linear under fp8_autocast.

Same shapes as bench_gemm.py but with FP8 quantized compute. Each model
contributes four GEMM shapes:
  QKV projection     (column-parallel)  N = (Qheads + 2*KVheads)*head_dim / TP, K = hidden
  Attention output   (row-parallel)     N = hidden, K = Qheads*head_dim / TP
  MLP Gate+Up        (column-parallel)  N = 2*intermediate / TP, K = hidden  (SwiGLU)
  MLP Down           (row-parallel)     N = hidden, K = intermediate / TP
"""

import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling, Format

from driver import time_func
from shapes import M_SIZES, gemm_shapes

SHAPES = gemm_shapes()

FP8_RECIPE = DelayedScaling(
    fp8_format=Format.HYBRID, amax_history_len=16, amax_compute_algo="max",
)


class BenchGemmFP8:
    params = [M_SIZES, list(SHAPES)]
    param_names = ["M", "shape"]
    timeout = 300

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

    def time_forward(self, M, shape):
        def fn():
            with te.fp8_autocast(enabled=True, fp8_recipe=FP8_RECIPE):
                self.linear(self.x)
        return time_func(fn)

    def time_forward_backward(self, M, shape):
        def fn():
            with te.fp8_autocast(enabled=True, fp8_recipe=FP8_RECIPE):
                out = self.linear(self.x)
            out.backward(self.grad_out)
        return time_func(fn)


if __name__ == "__main__":
    from driver import main
    main(__file__)
