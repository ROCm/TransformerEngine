# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL permute-free MoE weight-gradient (wgrad) op wrapper.

Mirrors the route-list wgrad contract so the same routing metadata
(``sorted_slot_ids`` holding the received-token row per slot, plus ``block_start`` /
``blocks_per_expert`` and the per-expert block-padded grad base ``grad_base``) can be reused
verbatim.

Fixed kernel configuration (matches ``pf_wgrad.py``): FC1 route-list wgrad with block-padded
``grad`` + token-gathered ``x``, bf16 into ``dw`` (overwrite or accumulate), DMA + XOR
chunk swizzle fill, 3-stage LDS pipeline. Only the workgroup tile geometry is selectable
/ autotuned.
"""

from __future__ import annotations

import torch

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

    See the permute-free wgrad API for the argument contract: ``x`` is
    ``[num_recv_tokens, K]`` (gathered by received-token row), ``grad`` is the block-padded
    ``[em_max, N]`` per-slot gradient, ``sorted_slot_ids`` maps each block-padded slot to its
    received-token row (sentinel ``num_recv_tokens`` for padding), and ``grad_base[e]`` is the
    block-padded first-slot row of expert ``e`` (``block_start[e] * block_size_m``).

    ``swap_gather`` selects the FC2 wgrad variant: ``grad`` is the token-space ``[num_recv, N]``
    operand (gathered) and ``x`` is the block-padded ``[em_max, K]`` operand (contiguous walk),
    emitting the native ``dw[E, N, K]`` orientation with no transpose.

    ``block_n``/``block_k`` and ``warps_n``/``warps_k`` select the workgroup tile; the
    defaults (``128x128`` over ``2x2`` warps) are a strong general config on CDNA4.
    """
    num_experts, N, K = dw.shape

    assert x.dtype == grad.dtype == torch.bfloat16
    assert dw.dtype in (torch.bfloat16, torch.float32)
    assert x.is_contiguous() and grad.is_contiguous() and dw.is_contiguous()

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
    """Dispatch target for the FlyDSL autotuner: compile (lru-cached) + launch one tile.

    ``swap_gather`` / ``accumulate`` / ``out_dtype`` are forwarded to the compile so the sweep
    benchmarks the *actual* kernel variant being launched (the FC1 non-swap and FC2 swap kernels
    gather a different operand and write ``dw`` in a different orientation, so their optimal tile
    can differ even at the same ``(N, K)``).
    """
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
            # ``swap_gather`` / ``accumulate`` / ``out_dtype`` gate distinct kernel variants
            # (different gather operand, store orientation, and epilogue), each with its own
            # best tile -- key on them so FC1 (non-swap) and FC2 (swap) don't share a tile.
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
    """Return the autotuned ``(block_n, block_k, warps_n, warps_k)`` for this problem.

    The sweep is run for the *specific* ``swap_gather`` / ``accumulate`` / ``out_dtype`` variant
    being launched (they are part of the cache key), so the tile is benchmarked against the real
    kernel rather than a stand-in.
    """
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

    First call for a given ``(x.shape, N, num_experts, swap_gather, accumulate, out_dtype)``
    benchmarks every tile in ``_AUTOTUNE_TILES`` and caches the fastest (in-memory + on disk
    under ``~/.flydsl/autotune/``). ``swap_gather`` (FC1 non-swap vs FC2 swap) is part of the
    key and is compiled into the benchmarked kernel, so each variant gets its own measured-best
    tile instead of inheriting the other's.
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
