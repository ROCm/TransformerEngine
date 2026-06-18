###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Shared model configurations and shape derivations for the microbenchmarks.

Single source of truth for the model shapes every ``bench_*.py`` sweeps over,
so a new model is added in one place. Config sources:

  - Llama 3.1 8B   https://huggingface.co/meta-llama/Llama-3.1-8B/blob/main/config.json
  - Llama 3.1 70B  https://huggingface.co/meta-llama/Llama-3.1-70B/blob/main/config.json
  - Llama 3.1 405B https://huggingface.co/meta-llama/Llama-3.1-405B/blob/main/config.json
  - Qwen 2.5 7B    https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json
  - Qwen 2.5 72B   https://huggingface.co/Qwen/Qwen2.5-72B-Instruct/blob/main/config.json
  - MoE configs    https://github.com/AMD-AGI/Primus-Turbo/blob/main/benchmark/ops/config.py
"""

# Token-count (batch * seq_len) sweeps shared across suites.
M_SIZES = [1024, 2048, 4096, 8192]
M_SIZES_MOE = [512, 1024, 2048, 4096]

# Dense transformer models, keyed by "<family>_TP<tp>".
# Value = (hidden, intermediate, num_q_heads, num_kv_heads, head_dim, tp).
MODELS = {
    "Llama3-8B_TP1":   (4096,  14336,  32, 8, 128, 1),
    "Llama3-8B_TP8":   (4096,  14336,  32, 8, 128, 8),
    "Llama3-70B_TP8":  (8192,  28672,  64, 8, 128, 8),
    "Llama3-405B_TP8": (16384, 53248, 128, 8, 128, 8),
    "Qwen2.5-7B_TP1":  (3584,  18944,  28, 4, 128, 1),
    "Qwen2.5-72B_TP8": (8192,  29568,  64, 8, 128, 8),
}

# MoE models for grouped GEMM: (num_routed_experts, moe_intermediate, hidden).
MOE_MODELS = {
    "DSV2-Lite": (64, 1408, 2048),
    "DSV2":      (160, 1536, 5120),
    "DSV3":      (256, 2048, 7168),
    "Grok-V2":   (8, 16384, 8192),
}


def attention_configs(models=MODELS):
    """Return ``{name: (num_q_heads, num_kv_heads, head_dim, tp)}``."""
    return {name: cfg[2:6] for name, cfg in models.items()}


def gemm_shapes(models=MODELS):
    """Return ``{shape_name: (N, K)}`` for the four transformer projections.

    Each model contributes QKV, AttnOut, GateUp (SwiGLU), and Down GEMMs.
    """
    shapes = {}
    for name, (hidden, inter, n_q, n_kv, hd, tp) in models.items():
        shapes[f"{name}-QKV"] = ((n_q * hd + 2 * n_kv * hd) // tp, hidden)
        shapes[f"{name}-AttnOut"] = (hidden, (n_q * hd) // tp)
        shapes[f"{name}-GateUp"] = ((2 * inter) // tp, hidden)
        shapes[f"{name}-Down"] = (hidden, inter // tp)
    return shapes


def grouped_gemm_configs(models=MOE_MODELS, eps=(32, 16, 8)):
    """Return ``{config_name: (num_gemms, N, K)}`` for MoE GateUp/Down GEMMs.

    One entry per (model, expert-parallel size) where the experts divide evenly.
    """
    configs = {}
    for model, (n_experts, inter, hidden) in models.items():
        for ep in eps:
            if n_experts % ep != 0:
                continue
            num_gemms = n_experts // ep
            configs[f"{model}_EP{ep}-GateUp"] = (num_gemms, 2 * inter, hidden)
            configs[f"{model}_EP{ep}-Down"] = (num_gemms, hidden, inter)
    return configs


def hidden_sizes(models=MODELS):
    """Return ``{model_family: hidden}`` (TP-independent) for element-wise benches."""
    out = {}
    for name, cfg in models.items():
        family = name.split("_TP")[0]
        out.setdefault(family, cfg[0])
    return out


def unique_hidden_sizes(models=MODELS):
    """Return the sorted distinct hidden dimensions across all models."""
    return sorted(set(hidden_sizes(models).values()))
