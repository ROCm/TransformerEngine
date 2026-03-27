#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Attention micro-benchmark using te.DotProductAttention.

Benchmarks fused multi-head attention (with flash attention backend) for
model configurations with grouped-query attention (GQA).

Models:
  - Llama 3   8B (TP=1, TP=8), 70B (TP=8), 405B (TP=8)
  - Qwen 2.5  7B (TP=1), 72B (TP=8)

Forward FLOPs = 4 * batch * num_q_heads * seq_len^2 * head_dim
  (two matmuls: Q@K^T and attn@V, each contributing 2*b*h*s^2*d)
Backward FLOPs = 2 * Forward FLOPs (approximately)

Sources for model configs:
  https://huggingface.co/meta-llama/Llama-3.1-8B/blob/main/config.json
  https://huggingface.co/meta-llama/Llama-3.1-70B/blob/main/config.json
  https://huggingface.co/meta-llama/Llama-3.1-405B/blob/main/config.json
  https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json
  https://huggingface.co/Qwen/Qwen2.5-72B-Instruct/blob/main/config.json

Forward FLOPs  = 4 * batch * num_q_heads * seq_len^2 * head_dim
Backward FLOPs ~ 2x forward
"""

import torch
import transformer_engine.pytorch as te

BATCH = 2

# (num_q_heads, num_kv_heads, head_dim, tp)
MODELS = {
    "Llama3-8B_TP1":   (32, 8, 128, 1),
    "Llama3-8B_TP8":   (32, 8, 128, 8),
    "Llama3-70B_TP8":  (64, 8, 128, 8),
    "Llama3-405B_TP8": (128, 8, 128, 8),
    "Qwen2.5-7B_TP1":  (28, 4, 128, 1),
    "Qwen2.5-72B_TP8": (64, 8, 128, 8),
}


class BenchAttention:
    params = [[1024, 2048, 4096, 8192], list(MODELS)]
    param_names = ["seq_len", "model"]
    timeout = 300

    def setup(self, seq_len, model):
        n_q, n_kv, hd, tp = MODELS[model]
        qh, kvh = n_q // tp, n_kv // tp
        dtype = torch.bfloat16

        self.attn = te.DotProductAttention(
            num_attention_heads=qh, kv_channels=hd,
            num_gqa_groups=kvh, attn_mask_type="causal",
        ).to(device="cuda", dtype=dtype)

        self.q = torch.randn(seq_len, BATCH, qh, hd, dtype=dtype, device="cuda", requires_grad=True)
        self.k = torch.randn(seq_len, BATCH, kvh, hd, dtype=dtype, device="cuda", requires_grad=True)
        self.v = torch.randn(seq_len, BATCH, kvh, hd, dtype=dtype, device="cuda", requires_grad=True)
        self.grad_out = torch.randn_like(self.attn(self.q, self.k, self.v))
        self._evt = [torch.cuda.Event(enable_timing=True) for _ in range(2)]

    def work_forward(self, seq_len, model):
        n_q, n_kv, hd, tp = MODELS[model]
        qh = n_q // tp
        return {"flops": 4 * BATCH * qh * seq_len * seq_len * hd}

    def work_forward_backward(self, seq_len, model):
        n_q, n_kv, hd, tp = MODELS[model]
        qh = n_q // tp
        return {"flops": 3 * 4 * BATCH * qh * seq_len * seq_len * hd}

    def time_forward(self, seq_len, model):
        self._evt[0].record()
        self.attn(self.q, self.k, self.v)
        self._evt[1].record()
        torch.cuda.synchronize()
        return self._evt[0].elapsed_time(self._evt[1]) / 1000

    def time_forward_backward(self, seq_len, model):
        self._evt[0].record()
        out = self.attn(self.q, self.k, self.v)
        out.backward(self.grad_out)
        self._evt[1].record()
        torch.cuda.synchronize()
        self.q.grad = self.k.grad = self.v.grad = None
        return self._evt[0].elapsed_time(self._evt[1]) / 1000

if __name__ == "__main__":
    from driver import run_as_main
    run_as_main(__file__)
