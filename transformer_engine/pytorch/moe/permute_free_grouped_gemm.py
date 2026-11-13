# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Permute-free route-list grouped GEMM for MoE (bf16).

Layout vocabulary:
  * **token-ordered** ``[num_recv_tokens, F]`` — one row per received token.
  * **dense route-ordered** ``[num_routes, F]`` — one row per ``(token, expert)`` route.
  * **block-padded route-ordered** ``[R_block, F]`` — expert-sorted routes with per-expert
    ``block_m`` padding (sync-free over-allocated upper bound).

Implements the permute-free MoE approach from MoEBlaze ("MoEBlaze: Breaking the Memory Wall for
Efficient MoE Training on Modern GPUs", MLSys 2026); the gather-GEMM kernels are ported from the
MegaMOE reference kernel.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from ..utils import packed_3d_view
from .moe_routing import MoERoutingMetadata, PermuteFreeMetadata
from .pf_helper_kernels import (
    fused_gated_act_prob_bwd,
    fused_gated_act_prob_fwd,
    route_gather_combine,
    route_list_align,
    route_list_scan,
)

__all__ = [
    "MoERoutingMetadata",
    "PermuteFreeMetadata",
    "permute_free_grouped_gemm_bf16",
    "permute_free_grouped_gemm_bf16_dgrad",
    "permute_free_grouped_gemm_bf16_wgrad",
    "permute_free_grouped_gemm_forward",
    "permute_free_grouped_gemm_backward",
    "PermuteFreeBackwardResult",
    "prepare_moe_align",
    "is_permute_free_grouped_gemm_enabled",
]

# Minimum gather/dgrad align block_m (128x256 MFMA floor).
_FLYDSL_MIN_BLOCK_M = 128
_FLYDSL_FWD_BLOCK_M = 256


def _get_flydsl_fwd():
    """Return the permute-free FlyDSL gather-GEMM launcher."""
    from .pf_fwd_wrapper import flydsl_moe_fwd

    return flydsl_moe_fwd


def _expand_expert_ids_per_slot(
    expert_ids: torch.Tensor,
    block_m: int,
    routes_max: int,
) -> torch.Tensor:
    """Broadcast block-level ``expert_ids`` to per-slot ids ``[routes_max]``."""
    block_m = int(block_m)
    pos = torch.arange(routes_max, device=expert_ids.device, dtype=torch.int64)
    # Clamp covers the over-allocated tail.
    block_idx = (pos // block_m).clamp(max=expert_ids.numel() - 1)
    return expert_ids[block_idx].to(torch.int32)


def _expert_per_route(routing: MoERoutingMetadata, routes_max: int) -> torch.Tensor:
    """Per-route local expert id ``[routes_max]`` from the route-list block metadata."""
    if (
        routing.expert_ids is None
        or routing.block_size_m is None
        or routing.block_size_m <= 0
    ):
        raise ValueError("_expert_per_route requires prepared routing align buffers.")
    if (
        routing.slot_expert_ids is not None
        and routing.slot_expert_ids.shape[0] == routes_max
    ):
        return routing.slot_expert_ids
    slot_expert_ids = _expand_expert_ids_per_slot(
        routing.expert_ids, routing.block_size_m, routes_max
    )
    routing.slot_expert_ids = slot_expert_ids
    return slot_expert_ids


def _get_flydsl_wgrad():
    """Return the FC1 FlyDSL wgrad launcher."""
    from .pf_wgrad_wrapper import flydsl_moe_wgrad_autotuned

    return flydsl_moe_wgrad_autotuned


def _wgrad_contract_m() -> int:
    """Wgrad align slot step; must equal the kernel's ``WGRAD_BLOCK_M``."""
    from .pf_wgrad_wrapper import WGRAD_BLOCK_M

    return int(WGRAD_BLOCK_M)


def _pf_moe_fwd(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    routing: MoERoutingMetadata,
    *,
    block_m: int,
    index_a_by_route_pos: bool = False,
    dgrad: bool = False,
) -> None:
    """Run the route-list gather-GEMM (forward or dgrad) via the FlyDSL permute-free kernels."""
    launch = _get_flydsl_fwd()
    block_m = int(block_m)
    launch(
        A,
        B,
        C,
        routing.sorted_slot_ids,
        routing.expert_ids,
        block_m=block_m,
        index_a_by_route_pos=index_a_by_route_pos,
        dgrad=dgrad,
    )


def _fwd_align_block_size_m(num_tokens: int) -> int:
    """Forward align ``block_m`` (128 or 256); shared by FC1 fwd/dgrad and FC2 fwd."""
    return _FLYDSL_FWD_BLOCK_M if num_tokens >= _FLYDSL_MIN_BLOCK_M else _FLYDSL_MIN_BLOCK_M


def _ensure_fwd_align(
    routing: MoERoutingMetadata,
) -> int:
    """Return ``routing.block_size_m``, building fwd align buffers when missing."""
    _invalidate_align_if_stale(routing)
    if routing.sorted_slot_ids is not None and routing.block_size_m is not None:
        return int(routing.block_size_m)
    block_m = _fwd_align_block_size_m(routing.num_recv_tokens)
    prepare_moe_align(routing, block_m)
    return block_m


def is_permute_free_grouped_gemm_enabled() -> bool:
    from torch.utils.cpp_extension import IS_HIP_EXTENSION

    if not (IS_HIP_EXTENSION and os.getenv("NVTE_PERMUTE_FREE_GROUPED_GEMM", "0") == "1"):
        return False
    # The kernels emit gfx950-only instructions (e.g. ``ds_read_tr16_b64``).
    from transformer_engine.pytorch.utils import get_device_compute_capability

    return get_device_compute_capability() == (9, 5)


def is_permute_free_exact_routes_enabled() -> bool:
    """Size route buffers from ``sum(m_splits)``, not the static bound (breaks CUDA graphs)."""
    return os.getenv("NVTE_PERMUTE_FREE_EXACT_ROUTES", "0") == "1"


def moe_align_route_list(
    routing_map: torch.Tensor,
    *,
    num_experts: int,
    block_size: int,
    scan=None,
    topk: Optional[int] = None,
    num_routes: Optional[int] = None,
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

    Parameters
    ----------
    routing_map:
        Boolean ``[num_recv_tokens, num_experts]``; True where a received token feeds a
        local expert.
    num_experts:
        Local expert count.
    block_size:
        ``BLOCK_SIZE_M`` (fwd/dgrad) or ``CONTRACT_M`` (wgrad) the layout is padded to.
    scan, topk, num_routes, build_inverse_map:
        See :func:`route_list_align`.

    Returns
    -------
    ``(sorted_slot_ids, expert_ids, num_tokens_post_padded, block_start, blocks_per_expert,
       token_routes, token_route_count)``. Index tensors are ``int32``;
    ``num_tokens_post_padded`` (``[1]``) is a device scalar. The last two are the token->routes
    inverse map, ``None`` unless ``build_inverse_map``.
    """
    return route_list_align(
        routing_map,
        num_experts=num_experts,
        block_size=block_size,
        scan=scan,
        topk=topk,
        num_routes=num_routes,
        build_inverse_map=build_inverse_map,
    )


def _exact_num_routes(metadata: MoERoutingMetadata) -> Optional[int]:
    """Caller-supplied exact route count, if any.

    Checked against ``routing_map`` (a device sync) under ``NVTE_PERMUTE_FREE_VALIDATE_ROUTES=1``.
    """
    num_routes = metadata.num_routes
    if num_routes is None:
        return None
    num_routes = int(num_routes)
    if os.getenv("NVTE_PERMUTE_FREE_VALIDATE_ROUTES", "0") == "1":
        actual = int(metadata.routing_map.sum().item())
        if num_routes != actual:
            raise ValueError(
                f"num_routes ({num_routes}) does not match routing_map.sum() ({actual}). "
                "A value below the true count silently drops routes."
            )
    return num_routes


def _ensure_route_scan(metadata: MoERoutingMetadata):
    """Compute + cache the block-independent scan (counts, within) once per routing map."""
    if metadata.route_counts is None or metadata.route_within is None:
        metadata.route_counts, metadata.route_within = route_list_scan(
            metadata.routing_map, num_experts=metadata.num_experts
        )
    return metadata.route_counts, metadata.route_within


def _routing_map_signature(metadata: MoERoutingMetadata):
    """``(id, _version)`` of ``routing_map`` -- the cache key for all align-derived buffers."""
    rmap = metadata.routing_map
    return (id(rmap), rmap._version)


def _stamp_align_signature(metadata: MoERoutingMetadata) -> None:
    """Record the ``routing_map`` signature the currently-cached align buffers were built from."""
    metadata.align_routing_id, metadata.align_routing_version = _routing_map_signature(metadata)


def _invalidate_align_if_stale(metadata: MoERoutingMetadata) -> None:
    """Drop cached align/scan buffers if ``routing_map`` was swapped or mutated in place.

    In-place reuse is the CUDA-graph pattern; without this the stale align routes silently wrong.
    """
    if metadata.align_routing_id is None:
        return  # nothing built yet
    if (metadata.align_routing_id, metadata.align_routing_version) == _routing_map_signature(
        metadata
    ):
        return  # unchanged since last build
    metadata.sorted_slot_ids = None
    metadata.expert_ids = None
    metadata.slot_expert_ids = None
    metadata.num_tokens_post_padded = None
    metadata.block_start = None
    metadata.token_routes = None
    metadata.token_route_count = None
    metadata.block_size_m = None
    metadata.route_counts = None
    metadata.route_within = None
    if metadata.wgrad_align is not None:
        metadata.wgrad_align.sorted_slot_ids = None
        metadata.wgrad_align.block_start = None
        metadata.wgrad_align.blocks_per_expert = None
        metadata.wgrad_align.block_size = None
    metadata.align_routing_id = None
    metadata.align_routing_version = None


def prepare_moe_align(metadata: MoERoutingMetadata, block_m: int) -> MoERoutingMetadata:
    """Build and cache the fwd/dgrad route-list align buffers on ``metadata`` (sync-free)."""
    _invalidate_align_if_stale(metadata)
    if (
        metadata.sorted_slot_ids is not None
        and metadata.block_size_m == block_m
        and metadata.token_routes is not None
    ):
        if metadata.slot_expert_ids is None:
            metadata.slot_expert_ids = _expand_expert_ids_per_slot(
                metadata.expert_ids,
                block_m,
                metadata.sorted_slot_ids.shape[0],
            )
        return metadata

    counts, within = _ensure_route_scan(metadata)

    (
        sorted_slot_ids,
        expert_ids,
        num_tokens_post_padded,
        block_start,
        _blocks_per_expert,
        token_routes,
        token_route_count,
    ) = moe_align_route_list(
        metadata.routing_map,
        num_experts=metadata.num_experts,
        block_size=block_m,
        scan=(counts, within),
        topk=metadata.topk,
        num_routes=_exact_num_routes(metadata),
        build_inverse_map=True,
    )
    metadata.sorted_slot_ids = sorted_slot_ids
    metadata.expert_ids = expert_ids
    metadata.slot_expert_ids = _expand_expert_ids_per_slot(
        expert_ids, block_m, sorted_slot_ids.shape[0]
    )
    metadata.num_tokens_post_padded = num_tokens_post_padded
    metadata.block_start = block_start
    metadata.block_size_m = block_m
    metadata.token_routes = token_routes
    metadata.token_route_count = token_route_count
    _stamp_align_signature(metadata)
    return metadata


def _prepare_wgrad_align(
    metadata: MoERoutingMetadata, contract_m: int
) -> MoERoutingMetadata:
    """Build and cache the wgrad align in the shared :class:`WgradAlign` holder."""
    cache = metadata.wgrad_align
    if cache.sorted_slot_ids is not None and cache.block_size == contract_m:
        return metadata

    (
        sorted_slot_ids,
        _expert_ids,
        _num_tokens_post_padded,
        block_start,
        blocks_per_expert,
        _token_routes,
        _token_route_count,
    ) = moe_align_route_list(
        metadata.routing_map,
        num_experts=metadata.num_experts,
        block_size=contract_m,
        scan=_ensure_route_scan(metadata),
        topk=metadata.topk,
        num_routes=_exact_num_routes(metadata),
    )
    cache.sorted_slot_ids = sorted_slot_ids
    cache.block_start = block_start
    cache.blocks_per_expert = blocks_per_expert
    cache.block_size = contract_m
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
    grouped = packed_3d_view(weights)
    return grouped if grouped is not None else torch.stack(weights, dim=0)


def permute_free_grouped_gemm_bf16(
    hidden_states: torch.Tensor,
    weights: torch.Tensor | list[torch.Tensor],
    routing: MoERoutingMetadata,
) -> torch.Tensor:
    """Route-list gather-in-GEMM (FC1 forward) for bf16 MoE.

    Computes ``C[slot] = A[sorted_slot_ids[slot]] @ W[expert(slot)]^T`` into the block-padded
    output. No activation is fused; FC2 applies it.

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
        Block-padded ``[R_block, out_features]``, bf16. Rows with
        ``sorted_slot_ids[slot] < num_recv_tokens`` are valid; padding rows hold garbage.
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

    in_features = hidden_states.shape[1]
    num_experts, out_features, in_k = weights_stacked.shape
    if in_k != in_features:
        raise ValueError(
            f"Weight in_features ({in_k}) does not match hidden_states ({in_features})."
        )
    if num_experts != routing.num_experts:
        raise ValueError(
            f"num_experts mismatch: weights have {num_experts}, routing has {routing.num_experts}."
        )

    block_size_m = _ensure_fwd_align(routing)

    # R_block is a shape-derived upper bound, so no device->host sync is needed.
    R_block = routing.sorted_slot_ids.shape[0]
    output = torch.empty(
        (R_block, out_features),
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )

    _pf_moe_fwd(
        hidden_states,
        weights_stacked,
        output,
        routing,
        block_m=block_size_m,
        index_a_by_route_pos=False,
    )
    return output


def _validate_dispatched_probs(
    routing: MoERoutingMetadata, dispatched_probs: Optional[torch.Tensor]
) -> None:
    """Enforce the ``[num_recv_tokens, num_local_experts]`` layout the kernels index into."""
    if dispatched_probs is None:
        return
    expected = (routing.num_recv_tokens, int(routing.num_experts))
    if tuple(dispatched_probs.shape) != expected:
        raise ValueError(
            "dispatched_probs must be [num_recv_tokens, num_local_experts] = "
            f"{expected} (indexed by local expert id), got {tuple(dispatched_probs.shape)}. "
            "The Megatron [T, topk] post-dispatch layout is not accepted here."
        )


def permute_free_gated_act_bwd(
    grad_output: torch.Tensor,
    preact: torch.Tensor,
    routing: MoERoutingMetadata,
    *,
    activation: str,
    dispatched_probs: Optional[torch.Tensor] = None,
    return_fc2_input: bool = False,
):
    """Gated-activation (+ route-prob) backward: ``dpre = [d_gate | d_up]`` from the ``2F`` preact.

    Parameters
    ----------
    grad_output:
        ``[R_block, F]`` grad wrt the activation output.
    preact:
        ``[R_block, 2F]`` raw ``[gate | up]`` pre-activation saved by the forward.
    dispatched_probs:
        ``[num_recv_tokens, E]`` gating probs, or ``None``.
    return_fc2_input:
        Also return the forward activation, saving the FC2 wgrad a recompute pass.

    Returns
    -------
    ``(dpre, dprob)`` by default, or ``(dpre, dprob, fc2_input)`` when ``return_fc2_input`` --
    ``dpre`` is ``[R_block, 2F]`` bf16; ``dprob`` matches ``dispatched_probs`` (or
    ``None`` when no probs were fused); ``fc2_input`` is ``[R_block, F]`` bf16.
    """
    # Rows are padded slots, so token/expert must be the per-slot arrays, not dense route arrays.
    _validate_dispatched_probs(routing, dispatched_probs)
    R_block = int(routing.sorted_slot_ids.shape[0])
    token = routing.sorted_slot_ids.to(torch.int32)
    expert = _expert_per_route(routing, R_block)
    dpre, dprob, fc2_input = fused_gated_act_prob_bwd(
        grad_output.contiguous(),
        preact,
        token,
        expert,
        num_recv_tokens=routing.num_recv_tokens,
        activation=activation,
        dispatched_probs=dispatched_probs,
        num_routes_bound=routing.num_tokens_post_padded,
        emit_act=return_fc2_input,
    )
    if return_fc2_input:
        return dpre, dprob, fc2_input
    return dpre, dprob


def permute_free_gated_act_fwd(
    preact: torch.Tensor,
    routing: MoERoutingMetadata,
    *,
    activation: str,
    dispatched_probs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply ``act(gate) * up [* prob]`` to the ``[R_block, 2F]`` preact; returns ``[R_block, F]``.

    Used for the FC2 forward input and to recompute it for the FC2 wgrad. Padding rows are zero.
    """
    _validate_dispatched_probs(routing, dispatched_probs)
    R_block = int(routing.sorted_slot_ids.shape[0])
    token = routing.sorted_slot_ids.to(torch.int32)
    expert = _expert_per_route(routing, R_block)
    return fused_gated_act_prob_fwd(
        preact,
        token,
        expert,
        num_recv_tokens=routing.num_recv_tokens,
        activation=activation,
        dispatched_probs=dispatched_probs,
        num_routes_bound=routing.num_tokens_post_padded,
    )


def permute_free_grouped_gemm_bf16_dgrad(
    grad_output: torch.Tensor,
    weights: torch.Tensor | list[torch.Tensor],
    routing: MoERoutingMetadata,
) -> torch.Tensor:
    """Route-list gather-in-GEMM dgrad (FC1 backward wrt input).

    ``grad_output`` is block-padded route-ordered ``[R_block, out_features]``; per-slot ``dX`` is
    gather-combined back to token-ordered ``dA``.

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
            f"grad_output must be dense route-ordered [num_routes, out_features], got {grad_output.shape}."
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

    if weights_stacked.stride(-1) != 1:
        weights_stacked = weights_stacked.contiguous()

    block_size_m = _ensure_fwd_align(routing)

    R_block = routing.sorted_slot_ids.shape[0]
    route_buf = torch.empty(
        (R_block, in_features),
        dtype=torch.bfloat16,
        device=grad_output.device,
    )
    _pf_moe_fwd(
        grad_output,
        weights_stacked,
        route_buf,
        routing,
        block_m=block_size_m,
        index_a_by_route_pos=True,
        dgrad=True,
    )
    return route_gather_combine(
        route_buf,
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
    out: Optional[torch.Tensor] = None,
    accumulate: bool = False,
    swap_gather: bool = False,
) -> torch.Tensor:
    """Route-list fused weight-gradient (FC1 backward wrt weights), FlyDSL-only.

    Computes ``dW[e] = sum_{slot in e} grad[slot]^T @ A[sorted_slot_ids[slot]]``.

    Parameters
    ----------
    hidden_states:
        Received activations ``[num_recv_tokens, in_features]``, bf16.
    grad_output:
        Block-padded route-ordered gradient ``[R_block, out_features]``, bf16.
    weights_shape:
        ``(num_experts, out_features, in_features)``.
    out:
        Optional bf16 or fp32 ``[E, out, in]`` destination written in-kernel
        (``+=`` if ``accumulate`` else ``=``).
    accumulate:
        With ``out``: add into it rather than overwrite.
    swap_gather:
        FC2 mode: gather token-space ``grad_output`` and read block-padded ``hidden_states``.

    Returns
    -------
    torch.Tensor
        The wgrad ``[num_experts, out_features, in_features]`` -- ``out`` when supplied, else a
        freshly allocated bf16 tensor.
    """
    if hidden_states.dtype != torch.bfloat16 or grad_output.dtype != torch.bfloat16:
        raise TypeError("permute_free_grouped_gemm_bf16_wgrad requires bf16 inputs.")
    if grad_output.dim() != 2:
        raise ValueError(
            f"grad_output must be dense route-ordered [num_routes, out_features], got {grad_output.shape}."
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

    contract_m = _wgrad_contract_m()
    # The contiguous operand uses the fwd align layout, so its per-expert base needs that align.
    if routing.block_start is None or routing.block_size_m is None:
        _ensure_fwd_align(routing)
    routing = _prepare_wgrad_align(routing, contract_m)
    grad_base = (routing.block_start.to(torch.int64) * int(routing.block_size_m)).to(torch.int32)

    flydsl_wgrad = _get_flydsl_wgrad()

    if out is not None:
        assert tuple(out.shape) == (num_experts, out_features, in_features), (
            f"wgrad out shape {tuple(out.shape)} != {(num_experts, out_features, in_features)}"
        )
        if out.dtype not in (torch.bfloat16, torch.float32):
            raise TypeError(
                f"FlyDSL wgrad requires bf16 or fp32 out, got {out.dtype}."
            )
        flydsl_wgrad(
            x,
            grad_output,
            out,
            routing.wgrad_align.sorted_slot_ids,
            routing.wgrad_align.block_start,
            routing.wgrad_align.blocks_per_expert,
            grad_base,
            num_recv_tokens=routing.num_recv_tokens,
            accumulate=bool(accumulate),
            swap_gather=bool(swap_gather),
        )
        return out

    dW = torch.zeros(
        (num_experts, out_features, in_features),
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )
    flydsl_wgrad(
        x,
        grad_output,
        dW,
        routing.wgrad_align.sorted_slot_ids,
        routing.wgrad_align.block_start,
        routing.wgrad_align.blocks_per_expert,
        grad_base,
        num_recv_tokens=routing.num_recv_tokens,
        accumulate=bool(accumulate),
        swap_gather=bool(swap_gather),
    )
    return dW


def permute_free_grouped_gemm_bf16_fc2(
    fc2_input: torch.Tensor,
    weights: torch.Tensor | list[torch.Tensor],
    routing: MoERoutingMetadata,
) -> torch.Tensor:
    """Route-list FC2 forward with gather-combine to token-ordered output (bf16 MoE).

    Block-padded route-ordered ``fc2_input`` goes through the GEMM, and the result is
    gather-combined into token-ordered ``[num_recv_tokens, out]``.

    Parameters
    ----------
    fc2_input:
        Block-padded route-ordered activations ``[R_block, in_features(F)]``, bf16.
    weights:
        Expert weights ``[num_experts, H, F]`` (W2) or list of ``[H, F]``.
    routing:
        ``MoERoutingMetadata`` carrying (or able to build) the route-list align buffers.

    Returns
    -------
    torch.Tensor
        Token-ordered ``[num_recv_tokens, out_features]``, bf16.
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

    block_size_m = _ensure_fwd_align(routing)

    R_block = routing.sorted_slot_ids.shape[0]
    route_buf = torch.empty(
        (R_block, out_features),
        dtype=torch.bfloat16,
        device=fc2_input.device,
    )
    _pf_moe_fwd(
        fc2_input,
        weights_stacked,
        route_buf,
        routing,
        block_m=block_size_m,
        index_a_by_route_pos=True,
    )
    return route_gather_combine(
        route_buf,
        routing.token_routes,
        routing.token_route_count,
        routing.num_recv_tokens,
        out_dtype=torch.bfloat16,
    )


def permute_free_grouped_gemm_bf16_fc2_dgrad(
    grad_output: torch.Tensor,
    weights: torch.Tensor | list[torch.Tensor],
    routing: MoERoutingMetadata,
) -> torch.Tensor:
    """FC2 dgrad: gather **token-ordered** grad back into **block-padded route-ordered** dX.

    Returns
    -------
    torch.Tensor
        ``[R_block, in_features]``, bf16; padding rows hold garbage.
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

    if weights_stacked.stride(-1) != 1:
        weights_stacked = weights_stacked.contiguous()

    block_size_m = _ensure_fwd_align(routing)

    R_block = routing.sorted_slot_ids.shape[0]
    dx = torch.empty(
        (R_block, in_features),
        dtype=torch.bfloat16,
        device=grad_output.device,
    )
    _pf_moe_fwd(
        grad_output,
        weights_stacked,
        dx,
        routing,
        block_m=block_size_m,
        index_a_by_route_pos=False,
        dgrad=True,
    )
    return dx


def permute_free_grouped_gemm_bf16_fc2_wgrad(
    fc2_input: torch.Tensor,
    grad_output: torch.Tensor,
    weights_shape,
    routing: MoERoutingMetadata,
    *,
    out: Optional[torch.Tensor] = None,
    accumulate: bool = False,
    preact: Optional[torch.Tensor] = None,
    dispatched_probs: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
) -> torch.Tensor:
    """FC2 wgrad: ``dW2[e] = sum_{route in e} grad[token(route)]^T @ fc2_input[route]``.

    Runs the wgrad kernel with ``swap_gather``: ``grad_output`` is gathered, ``fc2_input`` read
    contiguously.

    Parameters
    ----------
    fc2_input:
        Block-padded route-ordered activations ``[R_block, in_features(F)]``, bf16.
    grad_output:
        Token-space gradient ``[num_recv_tokens, out_features(H)]``, bf16.
    weights_shape:
        ``(num_experts, out_features(H), in_features(F))`` -- the W2 shape.
    out / accumulate:
        Optional ``[E, H, F]`` destination (``+=`` if ``accumulate``); returned when given.
    preact / dispatched_probs / activation:
        If ``preact`` (``[R_block, 2F]``) is given, ``fc2_input`` is ignored and recomputed from
        it with :func:`permute_free_gated_act_fwd`.
    """
    num_experts, out_features, in_features = (int(v) for v in weights_shape)

    if preact is not None:
        if activation is None:
            raise ValueError("permute_free FC2 wgrad recompute requires an activation.")
        fc2_input = permute_free_gated_act_fwd(
            preact, routing, activation=activation, dispatched_probs=dispatched_probs
        )

    weights_shape = (num_experts, out_features, in_features)  # [E, H, F]
    if out is not None and out.dtype in (torch.bfloat16, torch.float32):
        return permute_free_grouped_gemm_bf16_wgrad(
            fc2_input,
            grad_output,
            weights_shape,
            routing,
            out=out,
            accumulate=accumulate,
            swap_gather=True,
        )
    dW = permute_free_grouped_gemm_bf16_wgrad(
        fc2_input,
        grad_output,
        weights_shape,
        routing,
        swap_gather=True,
    )  # [E, H, F] bf16
    if out is None:
        return dW
    # Other sink dtypes can't be written in-kernel.
    if accumulate:
        out.add_(dW)
    else:
        out.copy_(dW)
    return out


# ---------------------------------------------------------------------------
# Direction-aware dispatch on ``routing.route_space``.
# ---------------------------------------------------------------------------
@dataclass
class PermuteFreeBackwardResult:
    """Gradients from :func:`permute_free_grouped_gemm_backward`.

    Fields are ``None`` when not requested. ``wgrad_stacked`` is ``[E, out, in]``.
    """

    dgrad: Optional[torch.Tensor] = None
    grad_probs: Optional[torch.Tensor] = None
    wgrad_stacked: Optional[torch.Tensor] = None
    # True when the wgrad was already written into ``wgrad_out``.
    wgrad_applied: bool = False


def permute_free_grouped_gemm_forward(
    hidden_states: torch.Tensor,
    weights: torch.Tensor | list[torch.Tensor],
    routing: MoERoutingMetadata,
    *,
    activation: Optional[str] = None,
    dispatched_probs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Dispatch the permute-free grouped-GEMM forward from the routing direction + fusion hints.

    ``routing.route_space`` selects the direction:

    - ``True`` (FC2): optional gated activation, GEMM, then gather-combine to ``[num_recv, out]``.
    - ``False`` (FC1): gather-in-GEMM to block-padded ``[R_block, out]``.
    """
    if getattr(routing, "route_space", False):
        fc2_input = hidden_states
        if activation is not None:
            fc2_input = permute_free_gated_act_fwd(
                hidden_states,
                routing,
                activation=activation,
                dispatched_probs=dispatched_probs,
            )
        return permute_free_grouped_gemm_bf16_fc2(fc2_input, weights, routing)
    return permute_free_grouped_gemm_bf16(hidden_states, weights, routing)


def permute_free_grouped_gemm_backward(
    grad_output: torch.Tensor,
    *,
    routing: MoERoutingMetadata,
    weights: list[torch.Tensor],
    num_gemms: int,
    hidden_states: Optional[torch.Tensor] = None,
    requires_dgrad: bool = False,
    requires_wgrad: bool = False,
    dispatched_probs: Optional[torch.Tensor] = None,
    fc2_activation: Optional[str] = None,
    wgrad_out: Optional[torch.Tensor] = None,
    wgrad_accumulate: bool = False,
) -> PermuteFreeBackwardResult:
    """Dispatch the permute-free grouped-GEMM backward (mirror of the forward dispatch).

    On FC2, ``hidden_states`` is the saved ``2F`` preact and ``fc2_activation`` runs the
    gated-activation backward. ``wgrad_out`` receives the wgrad in place (``+=`` if
    ``wgrad_accumulate``), setting ``result.wgrad_applied``.
    """
    grad_output = grad_output.contiguous()
    route_space = getattr(routing, "route_space", False)
    dgrad = None
    grad_probs = None
    wgrad_stacked = None
    wgrad_applied = False

    if route_space:
        recompute = fc2_activation is not None and hidden_states is not None
        # Prob grads come from the act-bwd too, so run it even when the input needs no grad.
        need_probs = dispatched_probs is not None and dispatched_probs.requires_grad
        run_act_bwd = requires_dgrad or (need_probs and recompute)
        fused_fc2_input = None
        if run_act_bwd:
            weights_stacked = _stack_expert_weights(weights)
            dgrad_f = permute_free_grouped_gemm_bf16_fc2_dgrad(
                grad_output, weights_stacked, routing
            )
            if recompute:
                if requires_wgrad:
                    dgrad, grad_probs, fused_fc2_input = permute_free_gated_act_bwd(
                        dgrad_f,
                        hidden_states,
                        routing,
                        activation=fc2_activation,
                        dispatched_probs=dispatched_probs,
                        return_fc2_input=True,
                    )
                else:
                    dgrad, grad_probs = permute_free_gated_act_bwd(
                        dgrad_f,
                        hidden_states,
                        routing,
                        activation=fc2_activation,
                        dispatched_probs=dispatched_probs,
                    )
                if not need_probs:
                    grad_probs = None
            else:
                dgrad = dgrad_f
            if not requires_dgrad:
                dgrad = None
        if requires_wgrad:
            weights_shape = (num_gemms, weights[0].size(0), weights[0].size(1))
            if fused_fc2_input is not None:
                dW = permute_free_grouped_gemm_bf16_fc2_wgrad(
                    fused_fc2_input, grad_output, weights_shape, routing,
                    out=wgrad_out, accumulate=wgrad_accumulate,
                )  # [E, H, F]
            else:
                dW = permute_free_grouped_gemm_bf16_fc2_wgrad(
                    hidden_states, grad_output, weights_shape, routing,
                    out=wgrad_out, accumulate=wgrad_accumulate,
                    preact=hidden_states if recompute else None,
                    dispatched_probs=dispatched_probs if recompute else None,
                    activation=fc2_activation if recompute else None,
                )  # [E, H, F]
            wgrad_stacked = dW
            wgrad_applied = wgrad_out is not None
    else:
        if requires_dgrad:
            weights_stacked = _stack_expert_weights(weights)
            dgrad = permute_free_grouped_gemm_bf16_dgrad(grad_output, weights_stacked, routing)
        if requires_wgrad:
            weights_shape = (num_gemms, weights[0].size(0), weights[0].size(1))
            dW = permute_free_grouped_gemm_bf16_wgrad(
                hidden_states, grad_output, weights_shape, routing,
                out=wgrad_out, accumulate=wgrad_accumulate,
            )  # [E, N, H]
            wgrad_stacked = dW
            wgrad_applied = wgrad_out is not None

    return PermuteFreeBackwardResult(
        dgrad=dgrad, grad_probs=grad_probs, wgrad_stacked=wgrad_stacked,
        wgrad_applied=wgrad_applied,
    )
