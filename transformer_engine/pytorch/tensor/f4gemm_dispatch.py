# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Python dispatch for QoLA f4gemm pybind11 modules.

Loads the pybind11 .so files built by QoLA and calls the AITER kernel
functions directly with torch.Tensor arguments.
"""

from __future__ import annotations

import functools
import importlib.util
from pathlib import Path
from typing import Optional

import torch

def _load_pybind_module(so_name: str, module_name: str):
    """Load a pybind11 .so as a Python module."""
    so_path = None
    p = Path(__file__).resolve().parent.parent.parent / "lib" / so_name
    if p.is_file():
        so_path = str(p)
    else:
        raise FileNotFoundError(f"Could not find {so_name}.")
    spec = importlib.util.spec_from_file_location(module_name, so_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@functools.lru_cache(maxsize=1)
def _get_asm_fn():
    mod = _load_pybind_module("te_module_gemm_a4w4_asm.so", "te_module_gemm_a4w4_asm")
    return mod.gemm_a4w4_asm


@functools.lru_cache(maxsize=1)
def _get_blockscale_fn():
    mod = _load_pybind_module(
        "te_module_gemm_a4w4_blockscale.so", "te_module_gemm_a4w4_blockscale"
    )
    return mod.gemm_a4w4_blockscale


def gemm_a4w4_asm(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    out: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> torch.Tensor:
    """Dispatch A4W4 ASM GEMM via QoLA pybind11 module.

    Args:
        A: [M, K/2] packed FP4 pairs (uint8)
        B: [N, K/2] packed FP4 pairs (uint8)
        A_scale: [M_padded, K/32_padded] E8M0 block scales (uint8)
        B_scale: [N_padded, K/32_padded] E8M0 block scales (uint8)
        out: [M, N] bf16 output buffer
        bias: optional bias tensor, or None
        alpha, beta: scaling factors

    Returns:
        The out tensor, filled with the GEMM result.
    """
    fn = _get_asm_fn()
    # AITER expects float4_e2m1fn_x2 data and float8_e8m0fnu scales
    A = A.view(torch.float4_e2m1fn_x2)
    B = B.view(torch.float4_e2m1fn_x2)
    A_scale = A_scale.view(torch.float8_e8m0fnu)
    B_scale = B_scale.view(torch.float8_e8m0fnu)
    # kernelName="" lets AITER auto-select via heuristic
    fn(A, B, A_scale, B_scale, out, "", bias, alpha, beta)
    return out


def gemm_a4w4_blockscale(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    out: torch.Tensor,
    splitK: int = 0,
) -> torch.Tensor:
    """Dispatch A4W4 blockscale CK GEMM via QoLA pybind11 module.

    Args:
        A: [M, K/2] packed FP4 pairs (uint8)
        B: [N, K/2] packed FP4 pairs (uint8)
        A_scale: [M_padded, K/32_padded] E8M0 block scales
        B_scale: [N_padded, K/32_padded] E8M0 block scales
        out: [M, N] fp16 or bf16 output buffer
        splitK: log2 of split factor (0 = no split)

    Returns:
        The out tensor, filled with the GEMM result.
    """
    fn = _get_blockscale_fn()
    M = A.shape[0]
    N = B.shape[0]
    K = A.shape[-1] * 2
    num_scale_blocks = K // 32
    A = A.view(torch.float4_e2m1fn_x2)
    B = B.view(torch.float4_e2m1fn_x2)
    # Blockscale kernel expects unpadded scales [M, K/32] / [N, K/32]
    A_scale = A_scale[:M, :num_scale_blocks].view(torch.float8_e8m0fnu)
    B_scale = B_scale[:N, :num_scale_blocks].view(torch.float8_e8m0fnu)
    fn(A, B, A_scale, B_scale, out, splitK)
    return out
