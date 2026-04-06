# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Vendored f4gemm dispatch logic for AITER's A4W4 GEMM kernels.

This module provides lazy loading of pre-built f4gemm extension modules
and a dispatch function that routes to either the ASM or CK blockscale
kernel path based on a tuned configuration CSV.

Adapted from aiter/ops/gemm_op_a4w4.py — no runtime dependency on the
AITER Python package.
"""

import csv
import functools
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

import torch


def _get_f4gemm_lib_dir() -> Path:
    """Locate transformer_engine/lib/f4gemm/ relative to the installed package."""
    spec = importlib.util.find_spec("transformer_engine")
    if spec is None or spec.origin is None:
        raise ImportError("transformer_engine package not found")
    return Path(spec.origin).parent / "lib" / "f4gemm"


@functools.lru_cache(maxsize=1)
def _get_cu_num() -> int:
    """Get the number of compute units on the current GPU."""
    cu_num = int(os.environ.get("CU_NUM", 0))
    if cu_num > 0:
        return cu_num
    try:
        result = subprocess.run(
            ["rocminfo"], capture_output=True, text=True, check=True
        )
        for block in re.split(r"Agent\s*\d+", result.stdout):
            if "GPU" not in block:
                continue
            for line in block.split("\n"):
                if "Device Type" in line and "GPU" in line:
                    match = re.search(r"Compute Unit\s*:\s*(\d+)", block)
                    if match:
                        return int(match.group(1))
                    break
    except Exception:
        pass
    raise RuntimeError("Could not determine GPU compute unit count from rocminfo")


@functools.lru_cache(maxsize=1)
def _load_f4gemm_config(csv_path: str) -> dict:
    """Load the tuned GEMM config CSV into a lookup dict.

    Returns dict keyed by (cu_num, M, N, K) -> {"kernelName": str, "splitK": int, ...}
    """
    config_dict = {}
    try:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (
                    int(row["cu_num"]),
                    int(row["M"]),
                    int(row["N"]),
                    int(row["K"]),
                )
                if key not in config_dict:
                    config_dict[key] = {
                        "kernelName": row["kernelName"],
                        "splitK": int(row["splitK"]) if row.get("splitK") else 0,
                    }
    except (FileNotFoundError, KeyError):
        pass
    return config_dict


def _get_gemm_config(M: int, N: int, K: int) -> Optional[dict]:
    """Look up the best kernel config for the given shape.

    Tries exact M first, then padded M values (matching AITER's heuristic).
    """
    f4gemm_dir = _get_f4gemm_lib_dir()
    csv_path = str(f4gemm_dir / "a4w4_blockscale_tuned_gemm.csv")
    config_dict = _load_f4gemm_config(csv_path)

    if not config_dict:
        return None

    cu_num = _get_cu_num()

    # Try exact M, then common padded M values (multiples of 32)
    candidates = [M]
    padded_32 = ((M + 31) // 32) * 32
    if padded_32 != M:
        candidates.append(padded_32)

    for m_val in candidates:
        config = config_dict.get((cu_num, m_val, N, K))
        if config is not None:
            return config

    return None


@functools.lru_cache(maxsize=1)
def _load_f4gemm_modules():
    """Lazy-load the pre-built f4gemm pybind11 extension modules.

    AITER_ASM_DIR is already set by ck_fused_attn_utils.cpp's auto-discovery
    (pointing to transformer_engine/lib/aiter/), so f4gemm .co blobs are
    found automatically under <AITER_ASM_DIR>/gfx950/f4gemm/.

    Returns (module_asm, module_blockscale) or raises ImportError.
    """
    f4gemm_dir = _get_f4gemm_lib_dir()

    if not f4gemm_dir.exists():
        raise ImportError(
            f"f4gemm lib directory not found at {f4gemm_dir}. "
            "Was TransformerEngine built with NVTE_AITER_F4GEMM=1 on gfx950?"
        )

    # Ensure AITER_ASM_DIR is set (fallback if MHA layer hasn't been loaded yet)
    if "AITER_ASM_DIR" not in os.environ:
        aiter_asm_dir = f4gemm_dir.parent / "aiter"
        if aiter_asm_dir.exists():
            os.environ["AITER_ASM_DIR"] = str(aiter_asm_dir)

    # Add f4gemm dir to sys.path for module import
    f4gemm_str = str(f4gemm_dir)
    if f4gemm_str not in sys.path:
        sys.path.insert(0, f4gemm_str)

    module_asm = None
    module_blockscale = None

    try:
        import module_gemm_a4w4_asm as _asm  # type: ignore[import-not-found]
        module_asm = _asm
    except ImportError:
        pass

    try:
        import module_gemm_a4w4_blockscale as _bs  # type: ignore[import-not-found]
        module_blockscale = _bs
    except ImportError:
        pass

    if module_asm is None and module_blockscale is None:
        raise ImportError(
            f"No f4gemm modules found in {f4gemm_dir}. "
            "Expected module_gemm_a4w4_asm.so and/or module_gemm_a4w4_blockscale.so"
        )

    return module_asm, module_blockscale


def f4gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    out: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    alpha: float = 1.0,
    beta: float = 0.0,
    bpreshuffle: bool = True,
) -> torch.Tensor:
    """Dispatch A4W4 GEMM to the best available kernel (ASM or CK blockscale).

    Args:
        A: [M, K/2] uint8, packed FP4 pairs
        B: [N, K/2] uint8, packed FP4 pairs
        A_scale: [M, K/32] uint8, E8M0 block scales (padded)
        B_scale: [N, K/32] uint8, E8M0 block scales (padded)
        out: [M_padded, N] output tensor (M_padded = ceil(M/32)*32)
        bias: optional [1, N] or [M, N] bias
        alpha, beta: scaling factors for ASM path
        bpreshuffle: whether B matrix scales are pre-shuffled

    Returns:
        Output tensor, sliced to [M, N] if M was padded.
    """
    module_asm, module_blockscale = _load_f4gemm_modules()

    m = A.shape[0]
    n = B.shape[0]
    k = A.shape[-1] * 2

    config = _get_gemm_config(m, n, k)

    kernel_name = config["kernelName"] if config else ""
    split_k = config.get("splitK", 0) if config else 0

    # Route to CK blockscale if config says so (non-mangled kernel name)
    if config is not None and "_ZN" not in kernel_name and module_blockscale is not None:
        split_k = split_k if split_k else 0
        return module_blockscale.gemm_a4w4_blockscale(
            A.view(m, k // 2), B, A_scale, B_scale, out, split_k
        )[:m]

    # Route to ASM path
    if module_asm is not None:
        assert out.shape[0] % 32 == 0, (
            "Dim0 of f4gemm ASM output must be padded to multiples of 32"
        )
        log2_k_split = split_k if split_k else None
        module_asm.gemm_a4w4_asm(
            A.view(m, k // 2),
            B,
            A_scale,
            B_scale,
            out,
            kernel_name,
            bias,
            alpha,
            beta,
            bpreshuffle,
            log2_k_split,
        )
        return out[:m].view(*A.shape[:-1], n)

    raise RuntimeError(
        "No suitable f4gemm kernel module available for the given configuration"
    )
