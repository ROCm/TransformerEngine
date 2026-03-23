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

# (hidden, intermediate, num_q_heads, num_kv_heads, head_dim, tp)
MODELS = {
    "Llama3-8B_TP1":   (4096, 14336, 32, 8, 128, 1),
    "Llama3-8B_TP8":   (4096, 14336, 32, 8, 128, 8),
    "Llama3-70B_TP8":  (8192, 28672, 64, 8, 128, 8),
    "Llama3-405B_TP8": (16384, 53248, 128, 8, 128, 8),
    "Qwen2.5-7B_TP1":  (3584, 18944, 28, 4, 128, 1),
    "Qwen2.5-72B_TP8": (8192, 29568, 64, 8, 128, 8),
}

# Pre-compute (N, K) for each GEMM shape
SHAPES = {}
for _name, (h, inter, nq, nkv, hd, tp) in MODELS.items():
    SHAPES[f"{_name}-QKV"] = ((nq * hd + 2 * nkv * hd) // tp, h)
    SHAPES[f"{_name}-AttnOut"] = (h, (nq * hd) // tp)
    SHAPES[f"{_name}-GateUp"] = ((2 * inter) // tp, h)
    SHAPES[f"{_name}-Down"] = (h, inter // tp)


class BenchGemm:
    params = [[1024, 2048, 4096, 8192], list(SHAPES)]
    param_names = ["M", "shape"]
    timeout = 300

    def setup(self, M, shape):
        N, K = SHAPES[shape]
        dtype = torch.bfloat16
        self.linear = te.Linear(K, N, bias=False).to(device="cuda", dtype=dtype)
        self.x = torch.randn(M, K, dtype=dtype, device="cuda", requires_grad=True)
        self.grad_out = torch.randn_like(self.linear(self.x))

    def time_forward(self, M, shape):
        self.linear(self.x)
        torch.cuda.synchronize()

    def time_forward_backward(self, M, shape):
        out = self.linear(self.x)
        out.backward(self.grad_out)
        self.x.grad = None
        self.linear.weight.grad = None
        torch.cuda.synchronize()

if __name__ == "__main__":
    from direct_run import run_as_main
    run_as_main(__file__)
