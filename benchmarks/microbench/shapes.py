#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Shared model configs and shape helpers for microbenchmarks.

Each helper returns a fresh dict — callers may mutate it (add custom entries,
drop unwanted ones) without affecting other benchmarks. Callers that want a
different set of base models can pass ``models=`` to override the default.

Sources:
  https://huggingface.co/meta-llama/Llama-3.1-8B/blob/main/config.json
  https://huggingface.co/meta-llama/Llama-3.1-70B/blob/main/config.json
  https://huggingface.co/meta-llama/Llama-3.1-405B/blob/main/config.json
  https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json
  https://huggingface.co/Qwen/Qwen2.5-72B-Instruct/blob/main/config.json
  MoE configs: https://github.com/AMD-AGI/Primus-Turbo/blob/main/benchmark/ops/config.py
"""

# Default token-count (batch * seq_len) sweep used by most benches.
M_SIZES = [1024, 2048, 4096, 8192]

# Dense transformer configs: (hidden, intermediate, num_q_heads, num_kv_heads, head_dim, tp)
DENSE_MODELS = {
    "Llama3-8B_TP1":   (4096, 14336, 32, 8, 128, 1),
    "Llama3-8B_TP8":   (4096, 14336, 32, 8, 128, 8),
    "Llama3-70B_TP8":  (8192, 28672, 64, 8, 128, 8),
    "Llama3-405B_TP8": (16384, 53248, 128, 8, 128, 8),
    "Qwen2.5-7B_TP1":  (3584, 18944, 28, 4, 128, 1),
    "Qwen2.5-72B_TP8": (8192, 29568, 64, 8, 128, 8),
}

# MoE configs: (n_routed_experts, moe_intermediate_size, hidden_size)
MOE_MODELS = {
    "DSV2-Lite": (64, 1408, 2048),
    "DSV2":      (160, 1536, 5120),
    "DSV3":      (256, 2048, 7168),
    "Grok-V2":   (8, 16384, 8192),
}

# Default expert-parallel sweep for grouped GEMM.
EP_SIZES = [32, 16, 8]


def gemm_shapes(models=None):
    """Per-projection ``(N, K)`` shapes derived from dense transformer configs.

    Returns ``{"<model>-QKV": (N, K), "<model>-AttnOut": ..., "-GateUp", "-Down"}``.
    """
    shapes = {}
    for name, (h, inter, nq, nkv, hd, tp) in (models or DENSE_MODELS).items():
        shapes[f"{name}-QKV"]     = ((nq * hd + 2 * nkv * hd) // tp, h)
        shapes[f"{name}-AttnOut"] = (h, (nq * hd) // tp)
        shapes[f"{name}-GateUp"]  = ((2 * inter) // tp, h)
        shapes[f"{name}-Down"]    = (h, inter // tp)
    return shapes


def attention_configs(models=None):
    """Per-model attention configs as ``(num_q_heads, num_kv_heads, head_dim, tp)``."""
    return {
        name: (nq, nkv, hd, tp)
        for name, (_h, _i, nq, nkv, hd, tp) in (models or DENSE_MODELS).items()
    }


def hidden_sizes(models=None):
    """Per-architecture ``{name: hidden}`` mapping, deduplicated across TP variants."""
    out = {}
    for name, (h, *_) in (models or DENSE_MODELS).items():
        base = name.split("_TP")[0]
        out.setdefault(base, h)
    return out


def grouped_gemm_configs(models=None, ep_sizes=None):
    """Grouped GEMM ``(B, N, K)`` configs for MoE GateUp + Down projections.

    ``B = n_routed_experts // ep`` for each EP that divides ``n_routed_experts``;
    other EPs are silently skipped.
    """
    configs = {}
    for name, (n_experts, inter, hidden) in (models or MOE_MODELS).items():
        for ep in (ep_sizes or EP_SIZES):
            if n_experts % ep != 0:
                continue
            B = n_experts // ep
            configs[f"{name}_EP{ep}-GateUp"] = (B, 2 * inter, hidden)
            configs[f"{name}_EP{ep}-Down"]   = (B, hidden, inter)
    return configs
