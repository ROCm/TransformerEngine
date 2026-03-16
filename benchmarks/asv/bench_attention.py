###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Fused multi-head attention (GQA) benchmarks via te.DotProductAttention.

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

    def time_forward(self, seq_len, model):
        self.attn(self.q, self.k, self.v)
        torch.cuda.synchronize()

    def time_forward_backward(self, seq_len, model):
        out = self.attn(self.q, self.k, self.v)
        out.backward(self.grad_out)
        self.q.grad = self.k.grad = self.v.grad = None
        torch.cuda.synchronize()
