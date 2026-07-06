# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Permute-free grouped GEMM for MoE (dense bf16, AITER Triton gather kernel)."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import triton.language as tl

from ..moe_routing import MoERoutingMetadata

__all__ = [
    "MoERoutingMetadata",
    "permute_free_grouped_gemm_bf16",
    "permute_free_grouped_gemm_bf16_dgrad",
    "permute_free_wgrad_sorted_inputs",
    "moe_align_block_size",
    "prepare_moe_align",
    "get_default_moe_kernel_config",
    "is_permute_free_grouped_gemm_enabled",
]


def prepare_moe_align(metadata: MoERoutingMetadata, block_m: int) -> MoERoutingMetadata:
    """Run ``moe_align_block_size`` and cache buffers on ``metadata``."""
    if (
        metadata.sorted_token_ids is not None
        and metadata.expert_ids is not None
        and metadata.num_tokens_post_padded is not None
    ):
        return metadata

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        metadata.topk_ids,
        num_experts=metadata.num_experts,
        block_size=block_m,
    )
    metadata.sorted_token_ids = sorted_token_ids
    metadata.expert_ids = expert_ids
    metadata.num_tokens_post_padded = num_tokens_post_padded
    metadata.block_size_m = block_m
    return metadata


def is_permute_free_grouped_gemm_enabled() -> bool:
    import os

    from torch.utils.cpp_extension import IS_HIP_EXTENSION

    return IS_HIP_EXTENSION and os.getenv("NVTE_PERMUTE_FREE_GROUPED_GEMM", "0") == "1"


def get_default_moe_kernel_config(num_tokens: int) -> Dict[str, Any]:
    """Return Triton tile config for dense bf16 MoE GEMM."""
    try:
        from aiter.ops.triton.utils.moe_config_utils import get_optimal_moe_config

        return get_optimal_moe_config(torch.bfloat16, M=num_tokens)
    except ImportError:
        pass

    if num_tokens <= 32:
        block_m = 16
    elif num_tokens <= 96:
        block_m = 32
    elif num_tokens <= 512:
        block_m = 64
    else:
        block_m = 128

    block_n = 64 if num_tokens <= 64 else 128
    block_k = 128 if num_tokens <= 64 else 64
    group_m = 16 if num_tokens // max(block_m, 1) > 128 else 1

    return {
        "BLOCK_SIZE_M": block_m,
        "BLOCK_SIZE_N": block_n,
        "BLOCK_SIZE_K": block_k,
        "GROUP_SIZE_M": group_m,
        "num_warps": 4 if num_tokens <= 128 else 8,
        "num_stages": 2,
    }


def moe_align_block_size(
    topk_ids: torch.Tensor,
    *,
    num_experts: int,
    block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build sorted token / expert buffers for gather-GEMM."""
    from aiter.ops.triton.moe.moe_align_block_size import moe_align_block_size_triton

    topk_ids_i32 = topk_ids.to(torch.int32)
    device = topk_ids_i32.device
    numel = topk_ids_i32.numel()

    # The align kernel only writes valid slots; padding slots must be pre-filled
    # with the sentinel ``numel`` so the GEMM kernel masks them via
    # ``offs_token < num_valid_tokens`` (otherwise stale values collide with slot 0).
    sorted_token_ids = torch.full(
        (numel + num_experts * (block_size - 1),),
        numel,
        dtype=torch.int32,
        device=device,
    )
    expert_ids = torch.empty(
        numel + num_experts,
        dtype=torch.int32,
        device=device,
    )
    num_tokens_post_padded = torch.empty(1, dtype=torch.int32, device=device)

    moe_align_block_size_triton(
        topk_ids_i32,
        num_experts,
        block_size,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
    )
    return sorted_token_ids, expert_ids, num_tokens_post_padded


def _stack_expert_weights(weights: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
    if isinstance(weights, torch.Tensor):
        if weights.dim() != 3:
            raise ValueError(
                f"Stacked expert weights must be 3D [num_experts, out, in], got {weights.shape}."
            )
        return weights
    if not weights:
        raise ValueError("At least one expert weight tensor is required.")
    return torch.stack(weights, dim=0)


def permute_free_grouped_gemm_bf16(
    hidden_states: torch.Tensor,
    weights: torch.Tensor | list[torch.Tensor],
    routing: MoERoutingMetadata,
    *,
    config: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """Gather-in-GEMM expert linear for dense bf16 MoE.

    Parameters
    ----------
    hidden_states:
        Unpermuted activations, shape ``[num_tokens, in_features]``, bf16, contiguous.
    weights:
        Expert weights ``[num_experts, out_features, in_features]`` or list of ``[out, in]``.
    routing:
        ``MoERoutingMetadata`` with ``topk_ids``, ``topk_weights``, and ``num_experts``.
    config:
        Optional Triton kernel config override.

    Returns
    -------
    torch.Tensor
        Shape ``[num_tokens, topk, out_features]``, bf16.
    """
    if hidden_states.dtype != torch.bfloat16:
        raise TypeError(
            f"permute_free_grouped_gemm_bf16 requires bf16 input, got {hidden_states.dtype}."
        )
    if not hidden_states.is_contiguous():
        hidden_states = hidden_states.contiguous()

    weights_stacked = _stack_expert_weights(weights)
    if weights_stacked.stride(-1) != 1:
        weights_stacked = weights_stacked.contiguous()

    num_tokens, in_features = hidden_states.shape
    num_experts, out_features, in_k = weights_stacked.shape
    if in_k != in_features:
        raise ValueError(
            f"Weight in_features ({in_k}) does not match hidden_states ({in_features})."
        )
    if num_experts != routing.num_experts:
        raise ValueError(
            f"num_experts mismatch: weights have {num_experts}, routing has {routing.num_experts}."
        )

    topk_ids = routing.topk_ids.to(torch.int32)
    topk_weights = routing.topk_weights.to(torch.float32)
    top_k = topk_ids.size(1)

    kernel_config = config or get_default_moe_kernel_config(num_tokens)
    block_size_m = int(kernel_config["BLOCK_SIZE_M"])

    routing = prepare_moe_align(routing, block_size_m)
    assert routing.sorted_token_ids is not None
    assert routing.expert_ids is not None
    assert routing.num_tokens_post_padded is not None

    output = torch.empty(
        (num_tokens, top_k, out_features),
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )

    from aiter.ops.triton.moe.moe_op import fused_moe

    fused_moe(
        hidden_states,
        weights_stacked,
        output,
        None,
        None,
        None,
        topk_weights,
        topk_ids,
        routing.sorted_token_ids,
        routing.expert_ids,
        routing.num_tokens_post_padded,
        routing.mul_routed_weight,
        top_k,
        tl.bfloat16,
        use_fp8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        block_shape=None,
        config=kernel_config,
    )
    return output


def permute_free_grouped_gemm_bf16_dgrad(
    grad_output: torch.Tensor,
    weights: torch.Tensor | list[torch.Tensor],
    routing: MoERoutingMetadata,
    *,
    config: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """Gather-in-GEMM dgrad for permute-free bf16 MoE.

    Computes ``dX[m] = sum_t grad_output[m, t] @ W_e`` where ``e = topk_ids[m, t]``.

    The same AITER gather kernel is reused in its pre-expanded (``top_k=1``) mode:
    ``grad_output`` is treated as ``[num_tokens * topk, out_features]`` (one row per
    routed slot) and the weights are transposed to ``[num_experts, in_features,
    out_features]`` so the contraction runs over ``out_features``. The per-slot
    partials are then summed over ``topk`` to recover token-major ``dX``.

    Parameters
    ----------
    grad_output:
        Gradient of the forward output, shape ``[num_tokens, topk, out_features]``, bf16.
    weights:
        Expert weights ``[num_experts, out_features, in_features]`` or list of ``[out, in]``.
    routing:
        ``MoERoutingMetadata`` carrying the align buffers from the forward pass. The
        cached ``block_size_m`` is reused so the dgrad grid matches ``expert_ids``.

    Returns
    -------
    torch.Tensor
        ``dX``, shape ``[num_tokens, in_features]``, bf16.
    """
    if grad_output.dtype != torch.bfloat16:
        raise TypeError(
            f"permute_free_grouped_gemm_bf16_dgrad requires bf16 grad, got {grad_output.dtype}."
        )
    if grad_output.dim() != 3:
        raise ValueError(
            f"grad_output must be [num_tokens, topk, out_features], got {grad_output.shape}."
        )

    weights_stacked = _stack_expert_weights(weights)
    num_experts, out_features, in_features = weights_stacked.shape

    num_tokens, top_k, n = grad_output.shape
    if n != out_features:
        raise ValueError(
            f"grad_output out_features ({n}) does not match weights ({out_features})."
        )
    if num_experts != routing.num_experts:
        raise ValueError(
            f"num_experts mismatch: weights have {num_experts}, routing has {routing.num_experts}."
        )

    # Contract over out_features by handing the kernel the [E, in, out] *transposed view*.
    # ``fused_moe`` addresses the weights purely through per-dim strides (stride_be/bk/bn), so
    # no physical copy is needed -- this only relabels strides. Materializing the transpose
    # (``.contiguous()``) would add a token-independent O(E*N*K) copy that dominates dgrad at
    # high expert count. Only copy if the stacked weights are themselves non-contiguous, which
    # would otherwise make the view's strides degenerate.
    if weights_stacked.stride(-1) != 1:
        weights_stacked = weights_stacked.contiguous()
    weights_t = weights_stacked.transpose(1, 2)
    grad_flat = grad_output.reshape(num_tokens * top_k, n).contiguous()

    # Reuse the forward align buffers (and their block size). Build them if absent.
    if (
        routing.sorted_token_ids is None
        or routing.expert_ids is None
        or routing.num_tokens_post_padded is None
        or routing.block_size_m is None
    ):
        fwd_config = config or get_default_moe_kernel_config(num_tokens)
        routing = prepare_moe_align(routing, int(fwd_config["BLOCK_SIZE_M"]))
    block_size_m = int(routing.block_size_m)

    topk_ids = routing.topk_ids.to(torch.int32)
    topk_weights = routing.topk_weights.to(torch.float32)

    # BLOCK_SIZE_M must match the align buffers; other tile dims are free.
    dgrad_config = get_default_moe_kernel_config(num_tokens * top_k)
    dgrad_config = {**dgrad_config, "BLOCK_SIZE_M": block_size_m}

    # The kernel writes to a 3D output [rows, 1, in_features] using C.stride(1)/(2);
    # with top_k=1 each routed slot maps to one output row.
    dgrad_part = torch.empty(
        (num_tokens * top_k, 1, in_features),
        dtype=torch.bfloat16,
        device=grad_output.device,
    )

    from aiter.ops.triton.moe.moe_op import fused_moe

    fused_moe(
        grad_flat,
        weights_t,
        dgrad_part,
        None,
        None,
        None,
        topk_weights,
        topk_ids,
        routing.sorted_token_ids,
        routing.expert_ids,
        routing.num_tokens_post_padded,
        routing.mul_routed_weight,
        1,  # pre-expanded: one grad row per routed slot
        tl.bfloat16,
        use_fp8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        block_shape=None,
        config=dgrad_config,
    )
    return dgrad_part.view(num_tokens, top_k, in_features).sum(dim=1)



def permute_free_wgrad_sorted_inputs(
    hidden_states: torch.Tensor,
    grad_output: torch.Tensor,
    routing: MoERoutingMetadata,
) -> Tuple[list, list, list]:
    """Build expert-contiguous inputs for the default grouped-GEMM wgrad path.

    The forward leaves activations unpermuted, so wgrad (which accumulates per
    expert) needs both operands sorted by expert. A single ``argsort`` of
    ``topk_ids`` reorders the (expanded) activations and gradients consistently.

    Returns
    -------
    (input_chunks, grad_chunks, counts):
        ``input_chunks[e]`` is ``[count_e, in_features]`` (gathered ``hidden_states``),
        ``grad_chunks[e]`` is ``[count_e, out_features]`` (sorted ``grad_output``),
        and ``counts`` is the per-expert token count (``m_splits``).
    """
    topk_ids = routing.topk_ids
    num_tokens, top_k = topk_ids.shape
    flat = topk_ids.reshape(-1).to(torch.int64)
    perm = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=routing.num_experts)
    counts_list = [int(c) for c in counts.tolist()]

    x = hidden_states.reshape(num_tokens, -1)
    dy = grad_output.reshape(num_tokens * top_k, -1)
    # slot s -> token s // top_k ; perm sorts slots by expert.
    x_sorted = x[perm // top_k].contiguous()
    dy_sorted = dy[perm].contiguous()

    return (
        list(torch.split(x_sorted, counts_list)),
        list(torch.split(dy_sorted, counts_list)),
        counts_list,
    )
