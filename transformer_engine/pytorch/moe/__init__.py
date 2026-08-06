# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Permute-free MoE grouped GEMM (FlyDSL) integration for PyTorch."""

from .permute_free_grouped_gemm import (
    MoERoutingMetadata,
    PermuteFreeBackwardResult,
    PermuteFreeMetadata,
    is_permute_free_grouped_gemm_enabled,
    permute_free_grouped_gemm_backward,
    permute_free_grouped_gemm_bf16,
    permute_free_grouped_gemm_bf16_dgrad,
    permute_free_grouped_gemm_bf16_fc2,
    permute_free_grouped_gemm_bf16_fc2_dgrad,
    permute_free_grouped_gemm_bf16_fc2_wgrad,
    permute_free_grouped_gemm_bf16_wgrad,
    permute_free_grouped_gemm_forward,
    permute_free_gated_act_bwd,
    permute_free_gated_act_recompute,
    prepare_moe_align,
)

__all__ = [
    "MoERoutingMetadata",
    "PermuteFreeBackwardResult",
    "PermuteFreeMetadata",
    "is_permute_free_grouped_gemm_enabled",
    "permute_free_grouped_gemm_backward",
    "permute_free_grouped_gemm_bf16",
    "permute_free_grouped_gemm_bf16_dgrad",
    "permute_free_grouped_gemm_bf16_fc2",
    "permute_free_grouped_gemm_bf16_fc2_dgrad",
    "permute_free_grouped_gemm_bf16_fc2_wgrad",
    "permute_free_grouped_gemm_bf16_wgrad",
    "permute_free_grouped_gemm_forward",
    "permute_free_gated_act_bwd",
    "permute_free_gated_act_recompute",
    "prepare_moe_align",
]
