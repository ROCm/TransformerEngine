#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
FP8 GEMM benchmarks via te.Linear under fp8_autocast.

Same shapes as bench_gemm.py but with FP8 quantized compute:
  - Llama 3   8B (TP=1, TP=8), 70B (TP=8), 405B (TP=8)
  - Qwen 2.5  7B (TP=1), 72B (TP=8)

Each model contributes four GEMM shapes:
  QKV projection     (column-parallel)  N = (Qheads + 2*KVheads)*head_dim / TP, K = hidden
  Attention output   (row-parallel)     N = hidden, K = Qheads*head_dim / TP
  MLP Gate+Up        (column-parallel)  N = 2*intermediate / TP, K = hidden  (SwiGLU)
  MLP Down           (row-parallel)     N = hidden, K = intermediate / TP

Sources for model configs:
  https://huggingface.co/meta-llama/Llama-3.1-8B/blob/main/config.json
  https://huggingface.co/meta-llama/Llama-3.1-70B/blob/main/config.json
  https://huggingface.co/meta-llama/Llama-3.1-405B/blob/main/config.json
  https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json
  https://huggingface.co/Qwen/Qwen2.5-72B-Instruct/blob/main/config.json
"""

import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling, Format

# (hidden, intermediate, num_q_heads, num_kv_heads, head_dim, tp)
MODELS = {
    "Llama3-8B_TP1":   (4096, 14336, 32, 8, 128, 1),
    "Llama3-8B_TP8":   (4096, 14336, 32, 8, 128, 8),
    "Llama3-70B_TP8":  (8192, 28672, 64, 8, 128, 8),
    "Llama3-405B_TP8": (16384, 53248, 128, 8, 128, 8),
    "Qwen2.5-7B_TP1":  (3584, 18944, 28, 4, 128, 1),
    "Qwen2.5-72B_TP8": (8192, 29568, 64, 8, 128, 8),
}

SHAPES = {}
for _name, (h, inter, nq, nkv, hd, tp) in MODELS.items():
    SHAPES[f"{_name}-QKV"] = ((nq * hd + 2 * nkv * hd) // tp, h)
    SHAPES[f"{_name}-AttnOut"] = (h, (nq * hd) // tp)
    SHAPES[f"{_name}-GateUp"] = ((2 * inter) // tp, h)
    SHAPES[f"{_name}-Down"] = (h, inter // tp)

FP8_RECIPE = DelayedScaling(
    fp8_format=Format.HYBRID, amax_history_len=16, amax_compute_algo="max",
)


class BenchGemmFP8:
    params = [[1024, 2048, 4096, 8192], list(SHAPES)]
    param_names = ["M", "shape"]
    timeout = 300
    _inner = 1
    _scratch = None

    def setup(self, M, shape):
        N, K = SHAPES[shape]
        dtype = torch.bfloat16
        self.linear = te.Linear(K, N, bias=False).to(device="cuda", dtype=dtype)
        self.x = torch.randn(M, K, dtype=dtype, device="cuda", requires_grad=True)
        self.grad_out = torch.randn(M, N, dtype=dtype, device="cuda")
        self._evt = [torch.cuda.Event(enable_timing=True) for _ in range(2)]

    def work_forward(self, M, shape):
        N, K = SHAPES[shape]
        return {"flops": 2 * M * N * K}

    def work_forward_backward(self, M, shape):
        N, K = SHAPES[shape]
        return {"flops": 3 * 2 * M * N * K}

    def time_forward(self, M, shape):
        if self._scratch is not None:
            self._scratch.fill_(1.0)
        self._evt[0].record()
        with te.fp8_autocast(enabled=True, fp8_recipe=FP8_RECIPE):
            for _ in range(self._inner):
                self.linear(self.x)
        self._evt[1].record()
        torch.cuda.synchronize()
        return self._evt[0].elapsed_time(self._evt[1]) / 1000 / self._inner

    def time_forward_backward(self, M, shape):
        if self._scratch is not None:
            self._scratch.fill_(1.0)
        self._evt[0].record()
        for _ in range(self._inner):
            with te.fp8_autocast(enabled=True, fp8_recipe=FP8_RECIPE):
                out = self.linear(self.x)
            out.backward(self.grad_out)
        self._evt[1].record()
        torch.cuda.synchronize()
        self.x.grad = None
        self.linear.weight.grad = None
        return self._evt[0].elapsed_time(self._evt[1]) / 1000 / self._inner

if __name__ == "__main__":
    from driver import run_as_main
    run_as_main(__file__)
