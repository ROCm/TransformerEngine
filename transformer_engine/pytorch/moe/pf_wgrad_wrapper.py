# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL permute-free MoE weight-gradient (wgrad) op wrapper; only the tile is autotuned."""

from __future__ import annotations

import torch

from ..flydsl_kernels.gemm.gemm_common_utils import require_launch_size
from ..flydsl_kernels.permute_free_grouped_gemm.pf_wgrad import (
    WGRAD_BLOCK_M,
    compile_moe_wgrad_v2,
)
from ..flydsl_kernels.tensor_shim import _run_compiled, ptr_arg

__all__ = ["flydsl_moe_wgrad", "flydsl_moe_wgrad_autotuned", "WGRAD_BLOCK_M"]


def flydsl_moe_wgrad(
    x: torch.Tensor,
    grad: torch.Tensor,
    dw: torch.Tensor,
    sorted_slot_ids: torch.Tensor,
    block_start: torch.Tensor,
    blocks_per_expert: torch.Tensor,
    grad_base: torch.Tensor,
    *,
    num_recv_tokens: int,
    block_n: int = 128,
    block_k: int = 128,
    warps_n: int = 2,
    warps_k: int = 2,
    accumulate: bool = False,
    swap_gather: bool = False,
) -> None:
    """Compute FC1 grouped wgrad ``grad[slot]^T @ x[token(slot)]`` into ``dw``, per expert.

    ``x`` is token-ordered ``[num_recv_tokens, K]`` (gathered), ``grad`` is block-padded
    ``[R_block, N]``, and ``grad_base[e] = block_start[e] * block_size_m``. ``swap_gather``
    (FC2) swaps which operand is gathered, still writing ``dw[E, N, K]``.
    """
    num_experts, N, K = dw.shape

    assert x.dtype == grad.dtype == torch.bfloat16
    assert dw.dtype in (torch.bfloat16, torch.float32)
    assert x.is_contiguous() and grad.is_contiguous() and dw.is_contiguous()

    # ``dw`` offsets are int32; ``x``/``grad`` are rebased per expert in i64, so need no guard.
    require_launch_size(
        "permute-free wgrad",
        ("dw", dw),
    )

    out_dtype = "fp32" if dw.dtype == torch.float32 else "bf16"
    exe = compile_moe_wgrad_v2(
        block_n=int(block_n),
        block_k=int(block_k),
        warps_n=int(warps_n),
        warps_k=int(warps_k),
        accumulate=bool(accumulate),
        swap_gather=bool(swap_gather),
        out_dtype=out_dtype,
    )

    _run_compiled(
        exe,
        ptr_arg(dw),
        ptr_arg(x),
        ptr_arg(grad),
        ptr_arg(sorted_slot_ids),
        ptr_arg(block_start),
        ptr_arg(blocks_per_expert),
        ptr_arg(grad_base),
        int(N),
        int(K),
        int(num_recv_tokens),
        int(num_experts),
        int(sorted_slot_ids.numel()),
        torch.cuda.current_stream(),
    )


# Tile configs the autotuner sweeps (block_n, block_k, warps_n, warps_k).
_AUTOTUNE_TILES = [
    (64, 64, 1, 1),
    (128, 64, 1, 1),
    (128, 128, 2, 2),
    (128, 128, 4, 2),
    (256, 128, 4, 2),
    (256, 256, 4, 4),
    (128, 256, 2, 4),
    (256, 128, 2, 2),
    (256, 256, 2, 4),
    (256, 256, 4, 2),
]

_wgrad_autotuner = None


def _wgrad_run(
    dw,
    x,
    grad,
    sorted_slot_ids,
    block_start,
    blocks_per_expert,
    grad_base,
    N,
    K,
    num_recv_tokens,
    num_experts,
    block_n=128,
    block_k=128,
    warps_n=2,
    warps_k=2,
    accumulate=False,
    swap_gather=False,
    out_dtype="bf16",
):
    """Autotuner dispatch target: compile (lru-cached) and launch one tile config."""
    exe = compile_moe_wgrad_v2(
        block_n=int(block_n),
        block_k=int(block_k),
        warps_n=int(warps_n),
        warps_k=int(warps_k),
        accumulate=bool(accumulate),
        swap_gather=bool(swap_gather),
        out_dtype=out_dtype,
    )
    _run_compiled(
        exe,
        ptr_arg(dw),
        ptr_arg(x),
        ptr_arg(grad),
        ptr_arg(sorted_slot_ids),
        ptr_arg(block_start),
        ptr_arg(blocks_per_expert),
        ptr_arg(grad_base),
        int(N),
        int(K),
        int(num_recv_tokens),
        int(num_experts),
        int(sorted_slot_ids.numel()),
        torch.cuda.current_stream(),
    )


def _get_autotuner(warmup=10, rep=30):
    """Build the shape-keyed Autotuner lazily (one instance, disk-cached results)."""
    global _wgrad_autotuner
    if _wgrad_autotuner is None:
        from flydsl.autotune import Autotuner, Config

        configs = [
            Config(block_n=bn, block_k=bk, warps_n=wn, warps_k=wk)
            for (bn, bk, wn, wk) in _AUTOTUNE_TILES
        ]
        _wgrad_autotuner = Autotuner(
            _wgrad_run,
            configs,
            # Each kernel variant has its own best tile, so key on the variant flags too.
            key=["x", "N", "num_experts", "swap_gather", "accumulate", "out_dtype"],
            warmup=warmup,
            rep=rep,
        )
    return _wgrad_autotuner


def _select_wgrad_config(
    x, grad, sorted_slot_ids, block_start, blocks_per_expert, grad_base,
    N, K, num_recv_tokens, num_experts,
    *,
    accumulate=False,
    swap_gather=False,
    out_dtype="bf16",
):
    """Return the autotuned ``(block_n, block_k, warps_n, warps_k)`` for this variant."""
    tuner = _get_autotuner()
    scratch_dtype = torch.float32 if out_dtype == "fp32" else torch.bfloat16
    scratch = torch.empty(num_experts, N, K, device=x.device, dtype=scratch_dtype)
    args = (
        scratch, x, grad, sorted_slot_ids, block_start, blocks_per_expert, grad_base,
        int(N), int(K), int(num_recv_tokens), int(num_experts),
    )
    variant = {"accumulate": bool(accumulate), "swap_gather": bool(swap_gather), "out_dtype": out_dtype}
    key = tuner._make_key(args, variant)
    if key not in tuner.cache:
        tuner(*args, **variant)
    cfg = tuner.cache[key].kwargs
    return cfg["block_n"], cfg["block_k"], cfg["warps_n"], cfg["warps_k"]


def flydsl_moe_wgrad_autotuned(
    x: torch.Tensor,
    grad: torch.Tensor,
    dw: torch.Tensor,
    sorted_slot_ids: torch.Tensor,
    block_start: torch.Tensor,
    blocks_per_expert: torch.Tensor,
    grad_base: torch.Tensor,
    *,
    num_recv_tokens: int,
    accumulate: bool = False,
    swap_gather: bool = False,
) -> None:
    """Shape-autotuned variant of :func:`flydsl_moe_wgrad`.

    The first call per shape/variant benchmarks ``_AUTOTUNE_TILES`` and caches the fastest
    (in memory and under ``~/.flydsl/autotune/``).
    """
    num_experts, N, K = dw.shape

    assert x.dtype == grad.dtype == torch.bfloat16
    assert dw.dtype in (torch.bfloat16, torch.float32)
    assert x.is_contiguous() and grad.is_contiguous() and dw.is_contiguous()

    out_dtype = "fp32" if dw.dtype == torch.float32 else "bf16"
    block_n, block_k, warps_n, warps_k = _select_wgrad_config(
        x, grad, sorted_slot_ids, block_start, blocks_per_expert, grad_base,
        N, K, num_recv_tokens, num_experts,
        accumulate=bool(accumulate),
        swap_gather=bool(swap_gather),
        out_dtype=out_dtype,
    )
    flydsl_moe_wgrad(
        x,
        grad,
        dw,
        sorted_slot_ids,
        block_start,
        blocks_per_expert,
        grad_base,
        num_recv_tokens=int(num_recv_tokens),
        block_n=block_n,
        block_k=block_k,
        warps_n=warps_n,
        warps_k=warps_k,
        accumulate=bool(accumulate),
        swap_gather=bool(swap_gather),
    )
