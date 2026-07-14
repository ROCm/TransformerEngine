# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Permute-free route-list grouped GEMM for MoE (bf16).

Route-list contract (post-dispatch, no topk / no route weights):
- Activations ``A = [num_recv_tokens, H]``; boolean ``routing_map = [num_recv_tokens,
  num_local_experts]`` marks which local expert(s) each received token feeds.
- TE builds one expert-sorted ``sorted_slot_ids`` (received-token row per route) plus a
  compact-output map (``route_start``/``block_start``), then runs the gather-GEMM.
- FC1 fwd output is worst-case padded ``[em_max, out_features]`` in expert order (valid rows
  are the compact route range ``[0, num_routes)``, tail is inert zero padding); dgrad returns
  ``dA = [num_recv_tokens, in_features]`` (scatter-add of the per-route gradients).
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import torch
import triton.language as tl

from ..moe_routing import MoERoutingMetadata, PermuteFreeMetadata
from .route_list_align import route_list_align, route_list_scan
from .route_combine import route_gather_combine
from .route_list_moe_gemm import fused_route_list_moe, fused_gated_act_prob_bwd
from .route_list_moe_wgrad import fused_route_list_moe_wgrad
from .route_prob import _expert_per_route

__all__ = [
    "MoERoutingMetadata",
    "PermuteFreeMetadata",
    "permute_free_grouped_gemm_bf16",
    "permute_free_grouped_gemm_bf16_dgrad",
    "permute_free_grouped_gemm_bf16_wgrad",
    "permute_free_gated_act_bwd",
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
    max_routes_per_token: Optional[int] = None,
    build_inverse_map: bool = False,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
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
        Host-known upper bound (the router top-k) on the number of experts any token routes
        to. When provided, tightens the static over-allocation from ``T * num_experts`` to
        ``T * min(max_routes_per_token, num_experts)`` (still sync-free).

    Returns
    -------
    ``(sorted_slot_ids, expert_ids, num_tokens_post_padded, block_start, blocks_per_expert,
       route_start, route_to_token, token_routes, token_route_count)``. Index tensors are
    ``int32``; ``num_tokens_post_padded`` (``[1]``) is a device scalar. The last two are the
    token->routes inverse map, ``None`` unless ``build_inverse_map``.
    """
    return route_list_align(
        routing_map,
        num_experts=num_experts,
        block_size=block_size,
        scan=scan,
        max_routes_per_token=max_routes_per_token,
        build_inverse_map=build_inverse_map,
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
    if (
        metadata.sorted_slot_ids is not None
        and metadata.block_size_m == block_m
        and metadata.token_routes is not None
    ):
        return metadata

    counts, within = _ensure_route_scan(metadata)
    # The token->routes inverse map (used by the contention-free gather-combine in FC2 fwd /
    # FC1 dgrad) is emitted by the same place kernel (``build_inverse_map=True``), so it costs
    # no extra launch. Block-independent, so it is built once here on the fwd align.
    (
        sorted_slot_ids,
        expert_ids,
        num_tokens_post_padded,
        block_start,
        _blocks_per_expert,
        route_start,
        route_to_token,
        token_routes,
        token_route_count,
    ) = moe_align_route_list(
        metadata.routing_map,
        num_experts=metadata.num_experts,
        block_size=block_m,
        scan=(counts, within),
        max_routes_per_token=metadata.topk,
        build_inverse_map=True,
    )
    metadata.sorted_slot_ids = sorted_slot_ids
    metadata.expert_ids = expert_ids
    metadata.num_tokens_post_padded = num_tokens_post_padded
    metadata.block_start = block_start
    metadata.route_start = route_start
    metadata.route_to_token = route_to_token
    metadata.block_size_m = block_m
    metadata.token_routes = token_routes
    metadata.token_route_count = token_route_count
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
        _token_routes,
        _token_route_count,
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
    activation: Optional[str] = None,
    dispatched_probs: Optional[torch.Tensor] = None,
    return_preact: bool = False,
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
    activation:
        When set (``"silu"`` / ``"gelu"``), fuses the **gated** activation into the GEMM
        epilogue: ``out_features`` is the gate+up width (``2F``), the output is the ``F``-wide
        activated buffer ``act(gate) * up``, and the separate activation kernel is skipped. The
        weight output dim must be laid out as ``[gate | up]``.
    dispatched_probs:
        Optional ``[num_recv_tokens, num_experts]`` gating probabilities. When given (with a
        fused ``activation``), each route's ``prob[token, expert]`` is multiplied into the
        activation in-kernel, skipping the separate route-prob-apply pass.
    return_preact:
        When set (only valid with a fused ``activation``), also allocate and fill a
        ``[em_max, 2F]`` buffer with the raw ``[gate | up]`` pre-activation and return it
        alongside the activated output. The backward needs it to reconstruct the 2F GEMM-output
        gradient; keep ``False`` for inference to skip the extra store.

    Returns
    -------
    torch.Tensor
        Worst-case padded ``[em_max, out_features]`` (or ``[em_max, F]`` with a fused
        ``activation``), bf16 (expert-contiguous). The valid rows are the compact route range
        ``[0, num_routes)``; the tail ``[num_routes, em_max)`` is inert, *uninitialized*
        padding (the buffer is ``torch.empty``, never zeroed). Consumers MUST read only the
        compact range via the routing metadata (``num_tokens_post_padded`` / ``route_start``);
        the tail holds garbage and must never be read. No host sync is needed here.
        When ``return_preact`` is set, returns ``(output, preact)`` where ``preact`` is the
        ``[em_max, 2F]`` raw ``[gate | up]`` buffer (same compact-range validity contract).
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

    gated_act = activation is not None
    if gated_act and out_features % 2 != 0:
        raise ValueError(
            f"Gated activation requires an even (gate+up) out_features, got {out_features}."
        )
    # Fused gated activation halves the stored width (2F -> F).
    stored_features = out_features // 2 if gated_act else out_features

    kernel_config = config or get_default_moe_kernel_config(num_recv_tokens)
    block_size_m = int(kernel_config["BLOCK_SIZE_M"])

    routing = prepare_moe_align(routing, block_size_m)

    # Worst-case (sync-free) allocation: size the output to the block-padded upper bound
    # em_max = sorted_slot_ids.shape[0], which is derived purely from shapes (num_recv_tokens,
    # num_experts, block_size) and is always >= num_routes. This avoids the device->host sync
    # (.item()) that a compact [num_routes, N] allocation would require. The gather-GEMM writes
    # only the compact route range [0, num_routes) (out_row is built from the compact
    # route_start); the tail [num_routes, em_max) is never visited (bounded by
    # num_tokens_post_padded, sentinel-masked). Since the tail is never read by consumers, the
    # buffer is left uninitialized (torch.empty) -- skipping the em_max*N zero-init memset,
    # which dominates the padding cost, at no correctness change.
    em_max = routing.sorted_slot_ids.shape[0]
    output = torch.empty(
        (em_max, stored_features),
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )

    if return_preact and not gated_act:
        raise ValueError("return_preact requires a fused activation.")
    preact = (
        torch.empty((em_max, out_features), dtype=torch.bfloat16, device=hidden_states.device)
        if return_preact
        else None
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
        activation=activation,
        dispatched_probs=dispatched_probs,
        preact_out=preact,
    )
    if return_preact:
        return output, preact
    return output


def permute_free_gated_act_bwd(
    grad_output: torch.Tensor,
    preact: torch.Tensor,
    routing: MoERoutingMetadata,
    *,
    activation: str,
    dispatched_probs: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Activation (+ route-prob) backward for the fused permute-free FC1 epilogue.

    Reconstructs the raw ``2F`` GEMM-output gradient ``dpre = [d_gate | d_up]`` (and, when the
    forward fused the route prob, the prob gradient) from the saved ``[em_max, 2F]``
    pre-activation. The returned ``dpre`` feeds the *unchanged* route-list dgrad / wgrad, so the
    FC1 backward stays symmetric with the non-activation path; the caller owns the autograd
    boundary and routes ``dprob`` back to ``dispatched_probs``.

    Parameters
    ----------
    grad_output:
        ``[em_max, F]`` grad wrt the fused FC1 output (route/padded layout).
    preact:
        ``[em_max, 2F]`` raw ``[gate | up]`` pre-activation saved by the forward.
    dispatched_probs:
        ``[num_recv_tokens, E]`` gating probs (or ``None`` for a silu/gelu-only fusion).

    Returns
    -------
    ``(dpre, dprob)`` -- ``dpre`` is ``[em_max, 2F]`` bf16; ``dprob`` matches
    ``dispatched_probs`` (or ``None`` when no probs were fused).
    """
    routes_max = int(routing.route_to_token.shape[0])
    token = routing.route_to_token.to(torch.int32)
    expert = _expert_per_route(routing, routes_max)
    return fused_gated_act_prob_bwd(
        grad_output.contiguous(),
        preact,
        token,
        expert,
        num_recv_tokens=routing.num_recv_tokens,
        activation=activation,
        dispatched_probs=dispatched_probs,
    )


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

    # Two-stage dgrad reduction (contention-free): (1) plain compact store of each per-route
    # dX[route] = grad[route] @ W1[e] into an [em_max, in] bf16 buffer (coalesced, no atomics),
    # then (2) a token-parallel gather-combine summing each token's route rows. This replaces
    # the fused atomic scatter-to-token, whose per-token contention dominated the FC1 dgrad.
    em_max = routing.sorted_slot_ids.shape[0]
    compact = torch.empty(
        (em_max, in_features),
        dtype=torch.bfloat16,
        device=grad_output.device,
    )
    fused_route_list_moe(
        grad_output,
        weights_t,
        compact,
        routing.sorted_slot_ids,
        routing.expert_ids,
        routing.num_tokens_post_padded,
        routing.block_start,
        routing.route_start,
        num_recv_tokens=routing.num_recv_tokens,
        compute_type=tl.bfloat16,
        config=dgrad_config,
        index_a_by_route_pos=True,
        scatter_to_token=False,
    )
    return route_gather_combine(
        compact,
        routing.token_routes,
        routing.token_route_count,
        routing.num_recv_tokens,
        out_dtype=torch.bfloat16,
    )


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
        config=None,  # autotune BLOCK_SIZE_N/K / num_warps / num_stages per shape
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

    # Two-stage combine (contention-free): (1) plain compact store of each per-route result
    # y[route] = fc2_input[route] @ W2[e] into an [em_max, out] bf16 buffer (coalesced, no
    # atomics), then (2) a token-parallel gather-combine that sums each token's route rows.
    # This replaces the fused atomic scatter-to-token, whose per-token atomic contention made
    # the FC2 forward ~2x the FC1 forward. Gather has no contention, so the combine is ~free.
    em_max = routing.sorted_slot_ids.shape[0]
    compact = torch.empty(
        (em_max, out_features),
        dtype=torch.bfloat16,
        device=fc2_input.device,
    )
    fused_route_list_moe(
        fc2_input,
        weights_stacked,
        compact,
        routing.sorted_slot_ids,
        routing.expert_ids,
        routing.num_tokens_post_padded,
        routing.block_start,
        routing.route_start,
        num_recv_tokens=routing.num_recv_tokens,
        compute_type=tl.bfloat16,
        config=kernel_config,
        index_a_by_route_pos=True,
        scatter_to_token=False,
    )
    return route_gather_combine(
        compact,
        routing.token_routes,
        routing.token_route_count,
        routing.num_recv_tokens,
        out_dtype=torch.bfloat16,
    )


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

    # Padded route-order output; the gather-GEMM writes only the compact [0, num_routes)
    # range and never visits the tail (bounded by num_tokens_post_padded, sentinel-masked).
    # Left uninitialized (torch.empty) to skip the em_max*N zero-init; consumers read only the
    # compact range via the routing metadata, so the garbage tail is never observed.
    em_max = routing.sorted_slot_ids.shape[0]
    dgrad = torch.empty(
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
