#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""BF16 GEMM benchmarks via te.Linear.

GEMM shapes derived from transformer layer projections:
  QKV, AttnOut, GateUp (SwiGLU), Down.

Model configuration sources:
- Llama 3 8B (hidden=4096, intermediate=14336, heads=32, kv_heads=8, head_dim=128)
  https://huggingface.co/meta-llama/Llama-3.1-8B/blob/main/config.json

- Llama 3 70B (hidden=8192, intermediate=28672, heads=64, kv_heads=8, head_dim=128)
  https://huggingface.co/meta-llama/Llama-3.1-70B/blob/main/config.json

- Llama 3 405B (hidden=16384, intermediate=53248, heads=128, kv_heads=8, head_dim=128)
  https://huggingface.co/meta-llama/Llama-3.1-405B/blob/main/config.json

- Qwen 2.5 7B (hidden=3584, intermediate=18944, heads=28, kv_heads=4, head_dim=128)
  https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json

- Qwen 2.5 72B (hidden=8192, intermediate=29568, heads=64, kv_heads=8, head_dim=128)
  https://huggingface.co/Qwen/Qwen2.5-72B-Instruct/blob/main/config.json
"""

import torch
import transformer_engine.pytorch as te

from driver import time_func

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
    from driver import run_as_main
    run_as_main(__file__)
