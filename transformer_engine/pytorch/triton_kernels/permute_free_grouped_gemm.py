# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Permute-free route-list grouped GEMM for MoE (bf16).

Route-list contract (post-dispatch, no topk / no route weights):
- Activations ``A = [num_recv_tokens, H]``; boolean ``routing_map = [num_recv_tokens,
  num_local_experts]`` marks which local expert(s) each received token feeds.
- TE builds one expert-sorted ``sorted_slot_ids`` (received-token row per route) plus a
  compact-output map (``route_start``/``block_start``), then runs the gather-GEMM.
- FC1 fwd output is statically padded ``[em_max, out_features]`` in expert order (valid rows
  are the compact route range ``[0, num_routes)``, tail is inert zero padding); dgrad returns
  ``dA = [num_recv_tokens, in_features]`` (scatter-add of the per-route gradients). ``em_max``
  is the sync-free upper bound ``T * min(topk, num_local_experts)`` (block-padded); pass
  ``MoERoutingMetadata.topk`` to tighten it from the dense ``T * num_local_experts``.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import torch
import triton.language as tl

from ..moe_routing import MoERoutingMetadata, PermuteFreeMetadata
from .route_list_align import route_list_align, route_list_scan
from .route_list_moe_gemm import fused_route_list_moe
from .route_list_moe_wgrad import (
    fused_route_list_moe_wgrad,
    get_default_route_list_wgrad_config,
)

__all__ = [
    "MoERoutingMetadata",
    "PermuteFreeMetadata",
    "permute_free_grouped_gemm_bf16",
    "permute_free_grouped_gemm_bf16_dgrad",
    "permute_free_grouped_gemm_bf16_wgrad",
    "permute_free_grouped_gemm_bf16_fc2",
    "permute_free_grouped_gemm_bf16_fc2_dgrad",
    "permute_free_grouped_gemm_bf16_fc2_wgrad",
    "moe_align_route_list",
    "prepare_moe_align",
    "get_default_moe_kernel_config",
    "is_permute_free_grouped_gemm_enabled",
]

# Contraction step (routes accumulated per dot) for the route-list wgrad kernel. The wgrad
# align is padded to this block size, independent of the fwd/dgrad BLOCK_SIZE_M.
_WGRAD_CONTRACT_M = 32


def is_permute_free_grouped_gemm_enabled() -> bool:
    from torch.utils.cpp_extension import IS_HIP_EXTENSION

    return IS_HIP_EXTENSION and os.getenv("NVTE_PERMUTE_FREE_GROUPED_GEMM", "0") == "1"


def get_default_moe_kernel_config(num_tokens: int) -> Dict[str, Any]:
    """Return Triton tile config for the route-list bf16 MoE GEMM."""
    # aiter's tuned config is an optional accelerator. Fall back to the built-in heuristic on
    # any import-time failure -- not just ImportError: aiter can raise RuntimeError (e.g. its
    # gluon kernels enforcing a minimum triton version) while probing this optional path.
    try:
        from aiter.ops.triton.utils.moe_config_utils import get_optimal_moe_config

        return get_optimal_moe_config(torch.bfloat16, M=num_tokens)
    except Exception:  # pylint: disable=broad-except
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


def moe_align_route_list(
    routing_map: torch.Tensor,
    *,
    num_experts: int,
    block_size: int,
    scan=None,
    max_routes_per_token: int | None = None,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Build the expert-sorted route-list buffers from a boolean routing map.

    Thin wrapper over the fused, host-sync-free Triton builder :func:`route_list_align`
    (one kernel launch + tiny per-expert prefix sums; no ``nonzero``/``argsort``/``.item()``).
    Index buffers are over-allocated to static, shape-derived bounds so their sizes never
    depend on device data; the real block-padded extent comes back as the device scalar
    ``num_tokens_post_padded``.

    Parameters
    ----------
    routing_map:
        Boolean ``[num_recv_tokens, num_experts]``; True where a received token feeds a
        local expert.
    num_experts:
        Local expert count.
    block_size:
        ``BLOCK_SIZE_M`` (fwd/dgrad) or ``CONTRACT_M`` (wgrad) the layout is padded to.
    scan:
        Optional ``(counts, within)`` from :func:`route_list_scan`, shared across block
        sizes so the block-independent scan is only paid once per routing map.
    max_routes_per_token:
        Host-known top-k bound; tightens the sync-free over-allocation from ``T * E`` to
        ``T * min(max_routes_per_token, E)``.

    Returns
    -------
    ``(sorted_slot_ids, expert_ids, num_tokens_post_padded, block_start, blocks_per_expert,
       route_start, route_to_token)``. Index tensors are ``int32``; ``num_tokens_post_padded``
    (``[1]``) is a device scalar.
    """
    return route_list_align(
        routing_map,
        num_experts=num_experts,
        block_size=block_size,
        scan=scan,
        max_routes_per_token=max_routes_per_token,
    )


def _ensure_route_scan(metadata: MoERoutingMetadata):
    """Compute + cache the block-independent scan (counts, within) once per routing map."""
    if metadata.route_counts is None or metadata.route_within is None:
        metadata.route_counts, metadata.route_within = route_list_scan(
            metadata.routing_map, num_experts=metadata.num_experts
        )
    return metadata.route_counts, metadata.route_within


def prepare_moe_align(metadata: MoERoutingMetadata, block_m: int) -> MoERoutingMetadata:
    """Build and cache the fwd/dgrad route-list align buffers on ``metadata`` (sync-free)."""
    if metadata.sorted_slot_ids is not None and metadata.block_size_m == block_m:
        return metadata

    (
        sorted_slot_ids,
        expert_ids,
        num_tokens_post_padded,
        block_start,
        _blocks_per_expert,
        route_start,
        route_to_token,
    ) = moe_align_route_list(
        metadata.routing_map,
        num_experts=metadata.num_experts,
        block_size=block_m,
        scan=_ensure_route_scan(metadata),
        max_routes_per_token=metadata.topk,
    )
    metadata.sorted_slot_ids = sorted_slot_ids
    metadata.expert_ids = expert_ids
    metadata.num_tokens_post_padded = num_tokens_post_padded
    metadata.block_start = block_start
    metadata.route_start = route_start
    metadata.route_to_token = route_to_token
    metadata.block_size_m = block_m
    return metadata


def _prepare_wgrad_align(
    metadata: MoERoutingMetadata, contract_m: int
) -> MoERoutingMetadata:
    """Build and cache the block-``contract_m`` align buffers for the wgrad kernel."""
    if (
        metadata.wgrad_sorted_slot_ids is not None
        and metadata.wgrad_block_size == contract_m
    ):
        return metadata

    (
        sorted_slot_ids,
        _expert_ids,
        _num_tokens_post_padded,
        block_start,
        blocks_per_expert,
        route_start,
        _route_to_token,
    ) = moe_align_route_list(
        metadata.routing_map,
        num_experts=metadata.num_experts,
        block_size=contract_m,
        scan=_ensure_route_scan(metadata),
        max_routes_per_token=metadata.topk,
    )
    metadata.wgrad_sorted_slot_ids = sorted_slot_ids
    metadata.wgrad_block_start = block_start
    metadata.wgrad_blocks_per_expert = blocks_per_expert
    metadata.wgrad_block_size = contract_m
    # route_start is block-size-independent; keep whichever is already cached.
    if metadata.route_start is None:
        metadata.route_start = route_start
    return metadata


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
    """Route-list gather-in-GEMM (FC1 forward) for bf16 MoE.

    Computes ``C[route] = A[route_to_token[route]] @ W[e]^T`` for each expert-sorted route,
    writing directly into a compact ``[num_routes, out_features]`` output.

    Parameters
    ----------
    hidden_states:
        Received activations ``[num_recv_tokens, in_features]``, bf16, contiguous.
    weights:
        Expert weights ``[num_experts, out_features, in_features]`` or list of ``[out, in]``.
    routing:
        ``MoERoutingMetadata`` carrying (or able to build) the route-list align buffers.

    Returns
    -------
    torch.Tensor
        Statically padded ``[em_max, out_features]``, bf16 (expert-contiguous), where
        ``em_max`` is the sync-free upper bound ``T * min(topk, num_experts)`` (block-padded;
        set ``routing.topk`` to tighten it from the dense ``T * num_experts``). The valid
        rows are the compact route range ``[0, num_routes)``; the tail ``[num_routes, em_max)``
        is inert zero padding. Consumers read the compact range via the routing metadata
        (``num_tokens_post_padded`` / ``route_start``), so no host sync is needed here.
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

    num_recv_tokens, in_features = hidden_states.shape
    num_experts, out_features, in_k = weights_stacked.shape
    if in_k != in_features:
        raise ValueError(
            f"Weight in_features ({in_k}) does not match hidden_states ({in_features})."
        )
    if num_experts != routing.num_experts:
        raise ValueError(
            f"num_experts mismatch: weights have {num_experts}, routing has {routing.num_experts}."
        )

    kernel_config = config or get_default_moe_kernel_config(num_recv_tokens)
    block_size_m = int(kernel_config["BLOCK_SIZE_M"])

    routing = prepare_moe_align(routing, block_size_m)

    # Worst-case (sync-free) allocation: size the output to the block-padded upper bound
    # em_max = sorted_slot_ids.shape[0], which is derived purely from shapes (num_recv_tokens,
    # num_experts, block_size) and is always >= num_routes. This avoids the device->host sync
    # (.item()) that a compact [num_routes, N] allocation would require. The gather-GEMM writes
    # only the compact route range [0, num_routes) (out_row is built from the compact
    # route_start); the tail stays zero and is masked out by the sorted_slot_ids sentinel and
    # expert_ids == -1, so it is never touched.
    em_max = routing.sorted_slot_ids.shape[0]
    output = torch.zeros(
        (em_max, out_features),
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )

    fused_route_list_moe(
        hidden_states,
        weights_stacked,
        output,
        routing.sorted_slot_ids,
        routing.expert_ids,
        routing.num_tokens_post_padded,
        routing.block_start,
        routing.route_start,
        num_recv_tokens=num_recv_tokens,
        compute_type=tl.bfloat16,
        config=kernel_config,
        index_a_by_route_pos=False,
    )
    return output


def permute_free_grouped_gemm_bf16_dgrad(
    grad_output: torch.Tensor,
    weights: torch.Tensor | list[torch.Tensor],
    routing: MoERoutingMetadata,
    *,
    config: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """Route-list gather-in-GEMM dgrad (FC1 backward wrt input).

    ``grad_output`` is the compact per-route gradient ``[num_routes, out_features]``. The
    kernel contracts over ``out_features`` to produce per-route ``dX = [num_routes, in]``,
    which is then scatter-added back onto the received tokens.

    Returns
    -------
    torch.Tensor
        ``dA``, shape ``[num_recv_tokens, in_features]``, bf16.
    """
    if grad_output.dtype != torch.bfloat16:
        raise TypeError(
            f"permute_free_grouped_gemm_bf16_dgrad requires bf16 grad, got {grad_output.dtype}."
        )
    if grad_output.dim() != 2:
        raise ValueError(
            f"grad_output must be compact [num_routes, out_features], got {grad_output.shape}."
        )
    if not grad_output.is_contiguous():
        grad_output = grad_output.contiguous()

    weights_stacked = _stack_expert_weights(weights)
    num_experts, out_features, in_features = weights_stacked.shape
    if grad_output.shape[-1] != out_features:
        raise ValueError(
            f"grad_output out_features ({grad_output.shape[-1]}) "
            f"does not match weights ({out_features})."
        )
    if num_experts != routing.num_experts:
        raise ValueError(
            f"num_experts mismatch: weights have {num_experts}, routing has {routing.num_experts}."
        )

    # Contract over out_features via the [E, in, out] transposed *view* (stride relabel only).
    if weights_stacked.stride(-1) != 1:
        weights_stacked = weights_stacked.contiguous()
    weights_t = weights_stacked.transpose(1, 2)

    if routing.sorted_slot_ids is None or routing.block_size_m is None:
        fwd_config = config or get_default_moe_kernel_config(routing.num_recv_tokens)
        routing = prepare_moe_align(routing, int(fwd_config["BLOCK_SIZE_M"]))
    block_size_m = int(routing.block_size_m)

    dgrad_config = get_default_moe_kernel_config(int(grad_output.shape[0]))
    dgrad_config = {**dgrad_config, "BLOCK_SIZE_M": block_size_m}

    # Fused scatter dgrad (sync-free, no padding work): the gather-GEMM accumulates each
    # per-route dX directly onto its received-token row via an in-kernel atomic add into a
    # compact [num_recv_tokens, in] fp32 buffer. Block-padded / sentinel positions are masked
    # out inside the kernel, so padding is never scattered -- there is no [routes_max, in]
    # intermediate and no separate index_add. fp32 accumulation keeps multi-route adds
    # accurate; cast back to bf16 at the end.
    dA = torch.zeros(
        (routing.num_recv_tokens, in_features),
        dtype=torch.float32,
        device=grad_output.device,
    )
    fused_route_list_moe(
        grad_output,
        weights_t,
        dA,
        routing.sorted_slot_ids,
        routing.expert_ids,
        routing.num_tokens_post_padded,
        routing.block_start,
        routing.route_start,
        num_recv_tokens=routing.num_recv_tokens,
        compute_type=tl.bfloat16,
        config=dgrad_config,
        index_a_by_route_pos=True,
        scatter_to_token=True,
    )
    return dA.to(torch.bfloat16)


def permute_free_grouped_gemm_bf16_wgrad(
    hidden_states: torch.Tensor,
    grad_output: torch.Tensor,
    weights_shape,
    routing: MoERoutingMetadata,
    *,
    config: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """Route-list fused weight-gradient (FC1 backward wrt weights).

    Computes ``dW[e] = sum_{route in e} grad[route]^T @ A[route_to_token[route]]`` by
    gathering the activation operand (received-token row) and reading the compact grad row
    inside a single Triton kernel.

    Parameters
    ----------
    hidden_states:
        Received activations ``[num_recv_tokens, in_features]``, bf16.
    grad_output:
        Compact per-route gradient ``[num_routes, out_features]``, bf16.
    weights_shape:
        ``(num_experts, out_features, in_features)``.

    Returns
    -------
    torch.Tensor
        Stacked weight gradient ``[num_experts, out_features, in_features]``, bf16.
    """
    if hidden_states.dtype != torch.bfloat16 or grad_output.dtype != torch.bfloat16:
        raise TypeError("permute_free_grouped_gemm_bf16_wgrad requires bf16 inputs.")
    if grad_output.dim() != 2:
        raise ValueError(
            f"grad_output must be compact [num_routes, out_features], got {grad_output.shape}."
        )

    num_experts, out_features, in_features = (int(v) for v in weights_shape)
    if grad_output.shape[-1] != out_features:
        raise ValueError(
            f"grad_output out_features ({grad_output.shape[-1]}) "
            f"does not match weights ({out_features})."
        )
    if hidden_states.shape[-1] != in_features:
        raise ValueError(
            f"hidden_states in_features ({hidden_states.shape[-1]}) "
            f"does not match weights ({in_features})."
        )
    if num_experts != routing.num_experts:
        raise ValueError(
            f"num_experts mismatch: weights have {num_experts}, routing has {routing.num_experts}."
        )

    x = hidden_states
    if not x.is_contiguous():
        x = x.contiguous()
    if not grad_output.is_contiguous():
        grad_output = grad_output.contiguous()

    routing = _prepare_wgrad_align(routing, _WGRAD_CONTRACT_M)

    dW = torch.zeros(
        (num_experts, out_features, in_features),
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )
    fused_route_list_moe_wgrad(
        x,
        grad_output,
        dW,
        routing.wgrad_sorted_slot_ids,
        routing.wgrad_block_start,
        routing.wgrad_blocks_per_expert,
        routing.route_start,
        num_recv_tokens=routing.num_recv_tokens,
        config=get_default_route_list_wgrad_config(_WGRAD_CONTRACT_M),
        contract_m=_WGRAD_CONTRACT_M,
    )
    return dW


def permute_free_grouped_gemm_bf16_fc2(
    fc2_input: torch.Tensor,
    weights: torch.Tensor | list[torch.Tensor],
    routing: MoERoutingMetadata,
    *,
    config: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """Route-list FC2 forward with fused scatter-to-token (bf16 MoE).

    ``fc2_input`` is the route-ordered activation ``[em_max, in_features]`` (FC1's padded
    output; valid rows are the compact range ``[0, num_routes)``). For each route the kernel
    reads its own row (``index_a_by_route_pos=True``), computes ``fc2_input[route] @ W2[e]^T``
    and atomic-accumulates the result onto its received-token row
    (``scatter_to_token=True``). Because the router probs are applied upstream (at the
    activation), this fused scatter-add *is* the token combine.

    Parameters
    ----------
    fc2_input:
        Route-ordered activations ``[em_max, in_features]``, bf16.
    weights:
        Expert weights ``[num_experts, out_features, in_features]`` (W2) or list of ``[out,
        in]``.
    routing:
        ``MoERoutingMetadata`` carrying (or able to build) the route-list align buffers.

    Returns
    -------
    torch.Tensor
        ``[num_recv_tokens, out_features]``, bf16 (token order); ready for the cross-rank
        combine. No separate unpermute is needed.
    """
    if fc2_input.dtype != torch.bfloat16:
        raise TypeError(
            f"permute_free_grouped_gemm_bf16_fc2 requires bf16 input, got {fc2_input.dtype}."
        )
    if not fc2_input.is_contiguous():
        fc2_input = fc2_input.contiguous()

    weights_stacked = _stack_expert_weights(weights)
    if weights_stacked.stride(-1) != 1:
        weights_stacked = weights_stacked.contiguous()

    num_experts, out_features, in_features = weights_stacked.shape
    if fc2_input.shape[-1] != in_features:
        raise ValueError(
            f"fc2_input in_features ({fc2_input.shape[-1]}) does not match weights "
            f"({in_features})."
        )
    if num_experts != routing.num_experts:
        raise ValueError(
            f"num_experts mismatch: weights have {num_experts}, routing has {routing.num_experts}."
        )

    if routing.sorted_slot_ids is None or routing.block_size_m is None:
        fwd_config = config or get_default_moe_kernel_config(routing.num_recv_tokens)
        routing = prepare_moe_align(routing, int(fwd_config["BLOCK_SIZE_M"]))
    block_size_m = int(routing.block_size_m)
    kernel_config = get_default_moe_kernel_config(int(fc2_input.shape[0]))
    kernel_config = {**kernel_config, "BLOCK_SIZE_M": block_size_m}

    # Fused token scatter: atomic-accumulate each per-route result onto its received-token
    # row in a compact [num_recv_tokens, out] fp32 buffer (padding/sentinel positions are
    # masked out inside the kernel). fp32 keeps the multi-route adds accurate; cast to bf16.
    out = torch.zeros(
        (routing.num_recv_tokens, out_features),
        dtype=torch.float32,
        device=fc2_input.device,
    )
    fused_route_list_moe(
        fc2_input,
        weights_stacked,
        out,
        routing.sorted_slot_ids,
        routing.expert_ids,
        routing.num_tokens_post_padded,
        routing.block_start,
        routing.route_start,
        num_recv_tokens=routing.num_recv_tokens,
        compute_type=tl.bfloat16,
        config=kernel_config,
        index_a_by_route_pos=True,
        scatter_to_token=True,
    )
    return out.to(torch.bfloat16)


def permute_free_grouped_gemm_bf16_fc2_dgrad(
    grad_output: torch.Tensor,
    weights: torch.Tensor | list[torch.Tensor],
    routing: MoERoutingMetadata,
    *,
    config: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """FC2 dgrad: gather the token-space grad back into the compact route buffer.

    ``grad_output`` is the token-space gradient ``[num_recv_tokens, out_features]`` (the grad
    of FC2's fused-scatter forward output). For each route the kernel gathers its received
    token's grad row (``index_a_by_route_pos=False``) and computes ``grad[token] @ W2[e]``
    (contracting over ``out_features``), writing the compact per-route
    ``d(fc2_input) = [em_max, in_features]``.

    Returns
    -------
    torch.Tensor
        ``[em_max, in_features]``, bf16 (route order; valid rows are ``[0, num_routes)``).
    """
    if grad_output.dtype != torch.bfloat16:
        raise TypeError(
            f"permute_free_grouped_gemm_bf16_fc2_dgrad requires bf16 grad, got "
            f"{grad_output.dtype}."
        )
    if grad_output.dim() != 2:
        raise ValueError(
            f"grad_output must be [num_recv_tokens, out_features], got {grad_output.shape}."
        )
    if not grad_output.is_contiguous():
        grad_output = grad_output.contiguous()

    weights_stacked = _stack_expert_weights(weights)
    num_experts, out_features, in_features = weights_stacked.shape
    if grad_output.shape[-1] != out_features:
        raise ValueError(
            f"grad_output out_features ({grad_output.shape[-1]}) does not match weights "
            f"({out_features})."
        )
    if num_experts != routing.num_experts:
        raise ValueError(
            f"num_experts mismatch: weights have {num_experts}, routing has {routing.num_experts}."
        )

    # Contract over out_features via the [E, in, out] transposed view (stride relabel only).
    if weights_stacked.stride(-1) != 1:
        weights_stacked = weights_stacked.contiguous()
    weights_t = weights_stacked.transpose(1, 2)

    if routing.sorted_slot_ids is None or routing.block_size_m is None:
        fwd_config = config or get_default_moe_kernel_config(routing.num_recv_tokens)
        routing = prepare_moe_align(routing, int(fwd_config["BLOCK_SIZE_M"]))
    block_size_m = int(routing.block_size_m)
    dgrad_config = get_default_moe_kernel_config(int(grad_output.shape[0]))
    dgrad_config = {**dgrad_config, "BLOCK_SIZE_M": block_size_m}

    em_max = routing.sorted_slot_ids.shape[0]
    dgrad = torch.zeros(
        (em_max, in_features),
        dtype=torch.bfloat16,
        device=grad_output.device,
    )
    fused_route_list_moe(
        grad_output,
        weights_t,
        dgrad,
        routing.sorted_slot_ids,
        routing.expert_ids,
        routing.num_tokens_post_padded,
        routing.block_start,
        routing.route_start,
        num_recv_tokens=grad_output.shape[0],
        compute_type=tl.bfloat16,
        config=dgrad_config,
        index_a_by_route_pos=False,
        scatter_to_token=False,
    )
    return dgrad


def permute_free_grouped_gemm_bf16_fc2_wgrad(
    fc2_input: torch.Tensor,
    grad_output: torch.Tensor,
    weights_shape,
    routing: MoERoutingMetadata,
    *,
    config: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """FC2 wgrad: ``dW2[e] = sum_{route in e} grad[token(route)]^T @ fc2_input[route]``.

    Reuses the FC1 route-list wgrad kernel with the two operands swapped: the token-gathered
    operand is the token-space ``grad_output`` and the route-read operand is the compact
    ``fc2_input``. The kernel returns ``[E, F, H]`` (= ``dW2^T`` per expert), so we transpose
    the last two dims back to ``[E, H, F]``.

    Parameters
    ----------
    fc2_input:
        Route-ordered activations ``[em_max, in_features(F)]``, bf16.
    grad_output:
        Token-space gradient ``[num_recv_tokens, out_features(H)]``, bf16.
    weights_shape:
        ``(num_experts, out_features(H), in_features(F))`` -- the W2 shape.
    """
    num_experts, out_features, in_features = (int(v) for v in weights_shape)
    # Call the FC1 wgrad with swapped roles: hidden_states = token-gathered grad (K=H),
    # grad_output(route-read) = fc2_input (N=F). Result dW is [E, F, H] = dW2^T.
    dW_t = permute_free_grouped_gemm_bf16_wgrad(
        grad_output,
        fc2_input,
        (num_experts, in_features, out_features),
        routing,
        config=config,
    )
    return dW_t.transpose(1, 2).contiguous()
