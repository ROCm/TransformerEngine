# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information
"""KOSMOS fused expert-parallel MoE: bf16, EP = 8 inside one node, gfx950 (MI350X / MI355X).

The whole routed-expert part of an MoE layer (dispatch, GEMM1 + SwiGLU, GEMM2, combine and the
weighted fold; in the backward also both weight gradients) runs as one KOSMOS kernel per pass::

    from transformer_engine.pytorch import kosmos

    w1 = kosmos.to_kosmos_w1(w1_gate_up)        # once, when the weights are loaded
    y = kosmos.kosmos_moe(ep_group, x, topk_idx, topk_weights, w1, w2)

What a training call keeps for its backward lives in ordinary tensors that autograd owns, so
activation recompute, saved-tensor offload and memory snapshots apply to it; KOSMOS itself holds
a fixed amount per configuration, its symmetric buffer and scratch (:func:`memory_report`).

KOSMOS is part of libtransformer_engine (the C API ``nvte_rocm_moe_*``), bound as
``transformer_engine_torch.rocm_moe``; :func:`is_available` says whether this build has it.
"""

from .functional import (
    destroy,
    is_available,
    kosmos_moe,
    memory_report,
    status,
)
from .layout import GLU_INTERLEAVE, from_kosmos_w1, to_kosmos_w1

__all__ = [
    "kosmos_moe",
    "to_kosmos_w1",
    "from_kosmos_w1",
    "status",
    "memory_report",
    "destroy",
    "is_available",
    "GLU_INTERLEAVE",
]
