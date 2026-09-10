#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Attention benchmarks via te.DotProductAttention (causal, GQA).

Forward FLOPs  = 4 * batch * num_q_heads * seq_len^2 * head_dim
  (Q@K^T and attn@V, each 2*b*h*s^2*d).
Backward FLOPs ~= 2 * Forward FLOPs.
"""

import torch
import transformer_engine.pytorch as te

from driver import BenchBase, run_as_main
from models import M_SIZES, attention_configs

BATCH = 2
MODELS = attention_configs()  # name -> (num_q_heads, num_kv_heads, head_dim, tp)


class BenchAttention(BenchBase):
    params = [M_SIZES, list(MODELS)]  # M_SIZES used as seq_len
    param_names = ["seq_len", "model"]

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

    def work_forward(self, seq_len, model):
        n_q, _, hd, tp = MODELS[model]
        return {"flops": 4 * BATCH * (n_q // tp) * seq_len * seq_len * hd}

    def work_forward_backward(self, seq_len, model):
        n_q, _, hd, tp = MODELS[model]
        return {"flops": 3 * 4 * BATCH * (n_q // tp) * seq_len * seq_len * hd}

    def time_forward(self, seq_len, model):
        return self._time(lambda: self.attn(self.q, self.k, self.v))

    def time_forward_backward(self, seq_len, model):
        t = self._time(lambda: self.attn(self.q, self.k, self.v).backward(self.grad_out))
        self.q.grad = self.k.grad = self.v.grad = None
        return t


if __name__ == "__main__":
    run_as_main(__file__)
