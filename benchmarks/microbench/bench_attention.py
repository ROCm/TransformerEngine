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

Forward FLOPs = 4 * batch * num_q_heads * seq_len^2 * head_dim
  (two matmuls: Q@K^T and attn@V, each contributing 2*b*h*s^2*d)
Backward FLOPs = 2 * Forward FLOPs (approximately)
"""

import torch
import transformer_engine.pytorch as te

from driver import time_func
from shapes import M_SIZES, attention_configs

BATCH = 2

# Default to the shared dense-model configs; mutate this dict to add custom
# attention shapes (e.g. CONFIGS["MyModel"] = (qh, kvh, head_dim, tp)).
CONFIGS = attention_configs()


class BenchAttention:
    params = [M_SIZES, list(CONFIGS)]
    param_names = ["seq_len", "model"]
    timeout = 300

    def setup(self, seq_len, model):
        n_q, n_kv, hd, tp = CONFIGS[model]
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

    def work_forward(self, seq_len, model):
        n_q, _n_kv, hd, tp = CONFIGS[model]
        qh = n_q // tp
        return {"flops": 4 * BATCH * qh * seq_len * seq_len * hd}

    def work_forward_backward(self, seq_len, model):
        n_q, _n_kv, hd, tp = CONFIGS[model]
        qh = n_q // tp
        return {"flops": 3 * 4 * BATCH * qh * seq_len * seq_len * hd}

    def time_forward(self, seq_len, model):
        return time_func(lambda: self.attn(self.q, self.k, self.v))

    def time_forward_backward(self, seq_len, model):
        def fn():
            out = self.attn(self.q, self.k, self.v)
            out.backward(self.grad_out)
        return time_func(fn)


if __name__ == "__main__":
    from driver import main
    main(__file__)
