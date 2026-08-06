# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Permute-free route-list grouped GEMM for MoE (bf16).

- TE builds one expert-sorted ``sorted_slot_ids`` (received-token row per block-padded slot)
  plus the per-expert ``block_start`` (block units), then runs the gather-GEMM.
- FC1 fwd output is worst-case padded ``[T * min(topk, E), out_features]`` in expert order (valid rows
  are the compact route range ``[0, num_routes)``, tail is inert zero padding); dgrad returns
  ``dA = [num_recv_tokens, in_features]`` (scatter-add of the per-route gradients).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

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

_WGRAD_CONTRACT_M = 32

# Minimum v3 gather/dgrad block_m (128x256 MFMA floor).
_FLYDSL_MIN_BLOCK_M = 128
_FLYDSL_FWD_BLOCK_M = 256


def _get_flydsl_fwd():
    """Return ``(flydsl_moe_fwd_autotuned, flydsl_moe_fwd_supported)``."""
    from .pf_fwd_wrapper import flydsl_moe_fwd_autotuned, flydsl_moe_fwd_supported

    return flydsl_moe_fwd_autotuned, flydsl_moe_fwd_supported


def _expand_expert_ids_per_slot(
    expert_ids: torch.Tensor,
    block_m: int,
    routes_max: int,
) -> torch.Tensor:
    """Broadcast block-level ``expert_ids`` to per-slot ids ``[routes_max]``.

    ``prepare_moe_align`` lays routes out expert-by-expert, padding each expert's count up to
    a multiple of ``block_m``. ``expert_ids[b]`` records which expert owns M-block ``b`` (an
    expert with more than ``block_m`` routes spans several consecutive blocks, all with the same
    id). FlyDSL GEMM reads ``expert_ids`` at block granularity; the standalone gated-activation
    kernels index row-by-row and need ``expert[slot]`` to look up ``dispatched_probs[token, e]``.

    This is a lookup, not ``expert = slot // block_m``: ``slot // block_m`` is the block index,
    then ``expert_ids[block_idx]`` is the owner assigned during align.
    """
    block_m = int(block_m)
    pos = torch.arange(routes_max, device=expert_ids.device, dtype=torch.int64)
    # Map each padded slot to its M-block; clamp covers the static over-allocated tail.
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


def _pf_moe_fwd(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    routing: MoERoutingMetadata,
    *,
    num_recv_tokens: int,
    block_m: int,
    index_a_by_route_pos: bool = False,
) -> None:
    """Run the route-list gather-GEMM (forward or dgrad) via FlyDSL autotuning."""
    autotuned, supported = _get_flydsl_fwd()
    block_m = int(block_m)
    if not supported(A, B, block_m=block_m):
        raise RuntimeError(
            "FlyDSL grouped GEMM does not support these operands "
            f"(block_m={block_m}, A={tuple(A.shape)}, B={tuple(B.shape)})."
        )
    autotuned(
        A,
        B,
        C,
        routing.sorted_slot_ids,
        routing.expert_ids,
        routing.block_start,
        num_recv_tokens=num_recv_tokens,
        block_m=block_m,
        index_a_by_route_pos=index_a_by_route_pos,
    )


def _fwd_align_block_size_m(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    num_tokens: int,
) -> int:
    """Pick the forward align/kernel ``block_m``, favoring the backend that will run.

    ``block_m`` is baked into ``prepare_moe_align`` (it sets the row padding) and is *shared* by a
    layer's FC1 fwd, FC1 dgrad and FC2 fwd (they reuse ``routing.block_size_m``), so it is chosen
    once here. For ``num_tokens >= _FLYDSL_MIN_BLOCK_M`` (128), offer :data:`_FLYDSL_FWD_BLOCK_M`
    (256, MegaMOE tile) alongside the v3 floor; smaller batches stay at 128 only.
    """
    default_block_m = (
        _FLYDSL_FWD_BLOCK_M if num_tokens >= _FLYDSL_MIN_BLOCK_M else _FLYDSL_MIN_BLOCK_M
    )
    candidates = {c for c in (_FLYDSL_MIN_BLOCK_M, default_block_m) if c <= default_block_m}
    from .pf_fwd_wrapper import _v3_enabled

    # The v3 gather/dgrad tile requires block_m >= 128 (128x256 MFMA minimum), so on a
    # small-token tier where the default would be < 128 we must still floor the align
    # block_m to 128 rather than emit a sub-tile the kernel cannot run.
    if _v3_enabled():
        candidates = {c for c in candidates if c >= _FLYDSL_MIN_BLOCK_M} or {_FLYDSL_MIN_BLOCK_M}

    # Prefer the in-tree FlyDSL block_m picker when available.
    try:
        from .pf_fwd_wrapper import flydsl_moe_fwd_pick_block_m
    except Exception:  # pylint: disable=broad-except
        flydsl_moe_fwd_pick_block_m = None
    if flydsl_moe_fwd_pick_block_m is not None:
        picked = flydsl_moe_fwd_pick_block_m(A, B, candidates=tuple(candidates))
        if picked is not None:
            return picked
    return default_block_m


def _ensure_fwd_align(
    routing: MoERoutingMetadata,
    A: torch.Tensor,
    B: torch.Tensor,
) -> int:
    """Return ``routing.block_size_m``, building fwd align buffers when missing."""
    if routing.sorted_slot_ids is not None and routing.block_size_m is not None:
        return int(routing.block_size_m)
    block_m = _fwd_align_block_size_m(A, B, num_tokens=routing.num_recv_tokens)
    prepare_moe_align(routing, block_m)
    return block_m


def is_permute_free_grouped_gemm_enabled() -> bool:
    from torch.utils.cpp_extension import IS_HIP_EXTENSION

    return IS_HIP_EXTENSION and os.getenv("NVTE_PERMUTE_FREE_GROUPED_GEMM", "0") == "1"


def moe_align_route_list(
    routing_map: torch.Tensor,
    *,
    num_experts: int,
    block_size: int,
    scan=None,
    topk: Optional[int] = None,
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
    scan:
        Optional ``(counts, within)`` from :func:`route_list_scan`, shared across block
        sizes so the block-independent scan is only paid once per routing map.
    topk:
        Host-known upper bound (the router top-k) on the number of experts any token routes
        to. When provided, tightens the static over-allocation from ``T * num_experts`` to
        ``T * min(topk, num_experts)`` (still sync-free).

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
        build_inverse_map=True,
    )
    metadata.sorted_slot_ids = sorted_slot_ids  # [T * min(topk, E)] int32: block-padded slot -> token
    metadata.expert_ids = expert_ids  # [blocks_max] int32: expert owning each block (-1 past end)
    metadata.slot_expert_ids = _expand_expert_ids_per_slot(
        expert_ids, block_m, sorted_slot_ids.shape[0]
    )
    metadata.num_tokens_post_padded = num_tokens_post_padded  # [1] int32 device scalar: padded extent
    metadata.block_start = block_start  # [E] int32: first block index of each expert (block units)
    metadata.block_size_m = block_m  # int: BLOCK_SIZE_M the layout is padded to
    metadata.token_routes = token_routes  # [T, min(topk, E)] int32: token -> its compact route ids
    metadata.token_route_count = token_route_count  # [T] int32: number of routes per token
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
        _token_routes,
        _token_route_count,
    ) = moe_align_route_list(
        metadata.routing_map,
        num_experts=metadata.num_experts,
        block_size=contract_m,
        scan=_ensure_route_scan(metadata),
        topk=metadata.topk,
    )
    metadata.wgrad_sorted_slot_ids = sorted_slot_ids
    metadata.wgrad_block_start = block_start
    metadata.wgrad_blocks_per_expert = blocks_per_expert
    metadata.wgrad_block_size = contract_m
    return metadata


def _try_view_grouped(weights: list[torch.Tensor]) -> Optional[torch.Tensor]:
    """Return a zero-copy ``[E, out, in]`` view when the per-expert tensors are contiguous,
    uniformly-strided, sequential slices of **one shared storage** (e.g. a single grouped weight
    buffer).

    Returns ``None`` when the layout is not a single contiguous block, so the caller can fall
    back to ``torch.stack``. Note that sequential ``data_ptr``s are not sufficient: separate
    parameters can be allocated back-to-back yet own distinct storages, so ``as_strided`` from
    the first tensor would overrun its storage. We therefore require a single shared storage.
    """
    w0 = weights[0]
    if not w0.is_contiguous():
        return None
    out, in_ = w0.shape
    stride = out * in_
    itemsize = w0.element_size()
    base_ptr = w0.data_ptr()
    storage_ptr = w0.untyped_storage().data_ptr()
    for i, w in enumerate(weights):
        if (
            w.shape != w0.shape
            or w.dtype != w0.dtype
            or not w.is_contiguous()
            or w.untyped_storage().data_ptr() != storage_ptr
            or w.data_ptr() != base_ptr + i * stride * itemsize
        ):
            return None
    # Guard: the shared storage must actually span all E experts from w0's offset.
    needed = (w0.storage_offset() + len(weights) * stride) * itemsize
    if w0.untyped_storage().nbytes() < needed:
        return None
    # All experts form one contiguous [E, out, in] block starting at w0's storage offset.
    return torch.as_strided(w0, (len(weights), out, in_), (stride, in_, 1))


def _stack_expert_weights(weights: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
    if isinstance(weights, torch.Tensor):
        if weights.dim() != 3:
            raise ValueError(
                f"Stacked expert weights must be 3D [num_experts, out, in], got {weights.shape}."
            )
        return weights
    if not weights:
        raise ValueError("At least one expert weight tensor is required.")
    grouped = _try_view_grouped(weights)
    return grouped if grouped is not None else torch.stack(weights, dim=0)


def permute_free_grouped_gemm_bf16(
    hidden_states: torch.Tensor,
    weights: torch.Tensor | list[torch.Tensor],
    routing: MoERoutingMetadata,
) -> torch.Tensor:
    """Route-list gather-in-GEMM (FC1 forward) for bf16 MoE.

    Computes ``C[slot] = A[sorted_slot_ids[slot]] @ W[expert(slot)]^T`` for each block-padded
    slot, writing directly into the ``[em_max, out_features]`` block-padded output. Gated
    activation is **not** fused here; FC1 emits the raw ``2F`` ``[gate | up]`` pre-activation
    and the standalone gated-act helpers apply ``act(gate) * up [* prob]`` on FC2.

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
        Block-padded ``[em_max, out_features]``, bf16 (expert-contiguous, each expert padded to
        ``block_size``). The valid rows are the block-padded slots whose
        ``sorted_slot_ids[slot] < num_recv_tokens``; the padding slots carry inert dead values.
        Consumers locate each expert's rows via the routing metadata:

        - ``block_start[e]`` (block units): expert ``e``'s rows occupy padded slots
          ``[block_start[e] * block_size, ...)``, with within-rank offset matching each route.
        - ``num_tokens_post_padded = sum_e ceil(counts[e] / block_size) * block_size``: the
          block-padded route count (each expert's row count rounded up to ``block_size``).
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

    block_size_m = _ensure_fwd_align(routing, hidden_states, weights_stacked)

    # Worst-case (sync-free) allocation: size the output to the block-padded upper bound
    # em_max = sorted_slot_ids.shape[0], which is derived purely from shapes (num_recv_tokens,
    # num_experts, block_size) and is always >= num_routes. This avoids the device->host sync
    # (.item()) that a compact [num_routes, N] allocation would require.
    em_max = routing.sorted_slot_ids.shape[0]
    output = torch.empty(
        (em_max, out_features),
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )

    _pf_moe_fwd(
        hidden_states,
        weights_stacked,
        output,
        routing,
        num_recv_tokens=num_recv_tokens,
        block_m=block_size_m,
        index_a_by_route_pos=False,
    )
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
    forward fused the route prob, the prob gradient) from the saved ``[T * min(topk, E), 2F]``
    pre-activation. The returned ``dpre`` feeds the *unchanged* route-list dgrad / wgrad, so the
    FC1 backward stays symmetric with the non-activation path; the caller owns the autograd
    boundary and routes ``dprob`` back to ``dispatched_probs``.

    Parameters
    ----------
    grad_output:
        ``[T * min(topk, E), F]`` grad wrt the fused FC1 output (route/padded layout).
    preact:
        ``[T * min(topk, E), 2F]`` raw ``[gate | up]`` pre-activation saved by the forward.
    dispatched_probs:
        ``[num_recv_tokens, E]`` gating probs (or ``None`` for a silu/gelu-only fusion).

    Returns
    -------
    ``(dpre, dprob)`` -- ``dpre`` is ``[T * min(topk, E), 2F]`` bf16; ``dprob`` matches
    ``dispatched_probs`` (or ``None`` when no probs were fused).
    """
    # Block-padded canonical layout: grad_out (FC2 dgrad) and preact both live in the [em_max]
    # padded slot order, and the kernel indexes grad_out/preact/dpre by the same row as
    # token/expert -- so we must feed the padded slot arrays (sorted_slot_ids + slot expert),
    # not the compact route arrays, or the padded valid slots beyond routes_max never get a dpre
    # row (and each route pairs a correct token with the wrong padded grad row).
    em_max = int(routing.sorted_slot_ids.shape[0])
    token = routing.sorted_slot_ids.to(torch.int32)
    expert = _expert_per_route(routing, em_max)
    # ``num_tokens_post_padded`` is a device scalar bounding the real padded extent, so the tail
    # programs exit early (sync-free) instead of streaming the padding through HBM.
    return fused_gated_act_prob_bwd(
        grad_output.contiguous(),
        preact,
        token,
        expert,
        num_recv_tokens=routing.num_recv_tokens,
        activation=activation,
        dispatched_probs=dispatched_probs,
        num_routes_bound=routing.num_tokens_post_padded,
    )


def permute_free_gated_act_recompute(
    preact: torch.Tensor,
    routing: MoERoutingMetadata,
    *,
    activation: str,
    dispatched_probs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Rebuild the fused FC1 activation ``act(gate) * up * prob`` from the saved pre-activation.

    Lets the backward checkpoint only the ``2F`` pre-activation (``[T * min(topk, E), 2F]``) and
    reconstruct the ``F``-wide activation just-in-time for the FC2 wgrad, feeding the
    *unchanged* full-speed wgrad kernel a transient buffer (freed right after) instead of
    persisting the activation across the fwd/bwd boundary. Uses the same per-route routing
    arrays as :func:`permute_free_gated_act_bwd`.

    Returns
    -------
    torch.Tensor
        ``act``, shape ``[em_max, F]``, bf16 (block-padded slot layout, matching the FC1
        forward output and the layout the wgrad route-reads).
    """
    # Block-padded canonical layout: emit one row per padded slot (keyed by sorted_slot_ids) so
    # the rebuilt activation lines up with the [em_max] grad the wgrad route-reads. Padding slots
    # (token sentinel >= num_recv_tokens) are masked to zero by the kernel.
    em_max = int(routing.sorted_slot_ids.shape[0])
    token = routing.sorted_slot_ids.to(torch.int32)
    expert = _expert_per_route(routing, em_max)
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

    block_size_m = _ensure_fwd_align(routing, grad_output, weights_t)

    # Two-stage dgrad reduction (contention-free): (1) plain compact store of each per-route
    # dX[route] = grad[route] @ W1[e] into an [T * min(topk, E), in] bf16 buffer (coalesced, no atomics),
    # then (2) a token-parallel gather-combine summing each token's route rows. This replaces
    # the fused atomic scatter-to-token, whose per-token contention dominated the FC1 dgrad.
    em_max = routing.sorted_slot_ids.shape[0]
    compact = torch.empty(
        (em_max, in_features),
        dtype=torch.bfloat16,
        device=grad_output.device,
    )
    _pf_moe_fwd(
        grad_output,
        weights_t,
        compact,
        routing,
        num_recv_tokens=routing.num_recv_tokens,
        block_m=block_size_m,
        index_a_by_route_pos=True,
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
    out: Optional[torch.Tensor] = None,
    accumulate: bool = False,
    swap_gather: bool = False,
) -> torch.Tensor:
    """Route-list fused weight-gradient (FC1 backward wrt weights), FlyDSL-only.

    Computes ``dW[e] = sum_{slot in e} grad[slot]^T @ A[sorted_slot_ids[slot]]`` by gathering
    the activation operand (received-token row) and reading the block-padded grad row (base
    ``block_start[e] * block_size_m`` + within-rank) in the FlyDSL kernel.

    Parameters
    ----------
    hidden_states:
        Received activations ``[num_recv_tokens, in_features]``, bf16.
    grad_output:
        Compact per-route gradient ``[num_routes, out_features]``, bf16.
    weights_shape:
        ``(num_experts, out_features, in_features)``.
    out:
        Optional ``[E, out, in]`` destination (bf16 or fp32) to fold the wgrad into directly
        (``+=`` if ``accumulate`` else ``=``). An fp32 sink (e.g. ``main_grad``) accumulates
        in-kernel via the fp32-store kernel variant -- no bf16 scratch + separate fold.
    accumulate:
        With ``out``: add into it rather than overwrite.
    swap_gather:
        FC2 wgrad mode. Gathers ``grad_output`` (token-space ``[num_recv, out_features]``) and
        walks ``hidden_states`` (block-padded ``[em_max, in_features]``) contiguously, emitting the
        native ``[E, out_features, in_features]`` layout directly (no transpose).

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

    contract_m = _WGRAD_CONTRACT_M
    # The grad operand lives in the forward's block-padded [em_max] slot layout (its row for
    # route (e, w) is block_start[e]*block_size_m + w). Ensure the forward align is present so we
    # can pass that padded per-expert base to the kernel; routes are contiguous within an expert,
    # so base + within-rank indexes the correct padded grad row.
    if routing.block_start is None or routing.block_size_m is None:
        stub_w = torch.empty(
            num_experts, out_features, in_features,
            device=hidden_states.device, dtype=torch.bfloat16,
        )
        _ensure_fwd_align(routing, hidden_states, stub_w)
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
            routing.wgrad_sorted_slot_ids,
            routing.wgrad_block_start,
            routing.wgrad_blocks_per_expert,
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
        routing.wgrad_sorted_slot_ids,
        routing.wgrad_block_start,
        routing.wgrad_blocks_per_expert,
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
    """Route-list FC2 forward with gather-combine to token order (bf16 MoE).

    ``fc2_input`` is the route-ordered ``F``-wide FC1 activation (or a transient buffer rebuilt
    from the saved ``2F`` pre-activation via :func:`permute_free_gated_act_recompute`). For each
    route the kernel reads its row (``index_a_by_route_pos=True``) and computes the compact
    per-route GEMM; a separate contention-free gather-combine pass sums each token's routes
    into its output row.

    Parameters
    ----------
    fc2_input:
        Route-ordered activations ``[em_max, in_features(F)]``, bf16.
    weights:
        Expert weights ``[num_experts, out_features(H), in_features(F)]`` (W2) or list of ``[H, F]``.
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

    block_size_m = _ensure_fwd_align(routing, fc2_input, weights_stacked)

    # Two-stage combine (contention-free): (1) plain compact store of each per-route result
    # y[route] = fc2_input[route] @ W2[e] into an [T * min(topk, E), out] bf16 buffer (coalesced, no
    # atomics), then (2) a token-parallel gather-combine that sums each token's route rows.
    # This replaces the fused atomic scatter-to-token, whose per-token atomic contention made
    # the FC2 forward ~2x the FC1 forward. Gather has no contention, so the combine is ~free.
    em_max = routing.sorted_slot_ids.shape[0]
    compact = torch.empty(
        (em_max, out_features),
        dtype=torch.bfloat16,
        device=fc2_input.device,
    )
    _pf_moe_fwd(
        fc2_input,
        weights_stacked,
        compact,
        routing,
        num_recv_tokens=routing.num_recv_tokens,
        block_m=block_size_m,
        index_a_by_route_pos=True,
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
) -> torch.Tensor:
    """FC2 dgrad: gather the token-space grad back into the compact route buffer.

    ``grad_output`` is the token-space gradient ``[num_recv_tokens, out_features]`` (the grad
    of FC2's fused-scatter forward output). For each route the kernel gathers its received
    token's grad row (``index_a_by_route_pos=False``) and computes ``grad[token] @ W2[e]``
    (contracting over ``out_features``), writing the compact per-route
    ``d(fc2_input) = [T * min(topk, E), in_features]``.

    Returns
    -------
    torch.Tensor
        ``[T * min(topk, E), in_features]``, bf16 (route order; valid rows are ``[0, num_routes)``).
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

    block_size_m = _ensure_fwd_align(routing, grad_output, weights_t)

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
    _pf_moe_fwd(
        grad_output,
        weights_t,
        dgrad,
        routing,
        num_recv_tokens=grad_output.shape[0],
        block_m=block_size_m,
        index_a_by_route_pos=False,
    )
    return dgrad


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

    The two operands play mirror roles to FC1: ``grad_output`` is token-space and must be
    token-gathered, ``fc2_input`` is route-ordered and read contiguously. Which one the kernel
    gathers decides the output orientation:

    * **FC2 wgrad** uses ``swap_gather``: the FlyDSL kernel token-gathers ``grad_output`` and
      walks the block-padded ``fc2_input`` contiguously, writing the native ``[E, H, F]`` layout
      directly (no transpose, and ``out``/``accumulate`` fold straight into a bf16 destination).

    Parameters
    ----------
    fc2_input:
        Route-ordered activations ``[T * min(topk, E), in_features(F)]``, bf16.
    grad_output:
        Token-space gradient ``[num_recv_tokens, out_features(H)]``, bf16.
    weights_shape:
        ``(num_experts, out_features(H), in_features(F))`` -- the W2 shape.
    out / accumulate:
        Optional ``[E, H, F]`` destination; ``+=`` if ``accumulate`` else ``=``. Returned when given.
    preact / dispatched_probs / activation:
        Recompute-from-preact mode. When ``preact`` (``[T * min(topk, E), 2F]`` raw ``[gate | up]``)
        is given, ``fc2_input`` is ignored and the ``F``-wide activation ``act(gate)*up*prob`` is
        re-materialised here into a transient buffer (freed after the wgrad), so the backward can
        checkpoint only the ``2F`` preact instead of the ``F``-wide activation. ``activation`` picks
        the nonlinearity (``"silu"``/``"gelu"``); ``dispatched_probs`` (``[num_recv, E]``) is the
        fused route-prob table (omit when the forward did not fuse probs). The wgrad kernel itself
        is unchanged and runs at full stored-act speed.
    """
    num_experts, out_features, in_features = (int(v) for v in weights_shape)

    # Recompute-from-preact: rebuild the transient activation for the (unchanged) wgrad.
    if preact is not None:
        if activation is None:
            raise ValueError("permute_free FC2 wgrad recompute requires an activation.")
        fc2_input = permute_free_gated_act_recompute(
            preact, routing, activation=activation, dispatched_probs=dispatched_probs
        )

    # FC2 wgrad via swap_gather: gather grad_output [num_recv, H] (N=H) and walk the block-padded
    # fc2_input [em_max, F] (K=F) contiguously -> native [E, H, F], no transpose.
    weights_shape = (num_experts, out_features, in_features)  # [E, H, F]
    if out is not None and out.dtype in (torch.bfloat16, torch.float32):
        # bf16 or fp32 sink: the kernel writes/accumulates straight into it (fp32 via the
        # fp32-store variant), so no scratch dW and no separate fold pass.
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
    # Exotic sink dtype the kernel can't write directly: fold the bf16 result here.
    if accumulate:
        out.add_(dW)
    else:
        out.copy_(dW)
    return out


# ---------------------------------------------------------------------------
# Direction-aware dispatch: pick the FC1/FC2 fwd/bwd kernels from the routing
# metadata so callers (e.g. GroupedLinear) don't have to branch on route_space /
# activation themselves.
# ---------------------------------------------------------------------------
@dataclass
class PermuteFreeBackwardResult:
    """Gradients from :func:`permute_free_grouped_gemm_backward`.

    ``dgrad`` / ``wgrad_stacked`` are ``None`` when the corresponding gradient was not requested.
    ``wgrad_stacked`` is the weight gradient as a single ``[E, out, in]`` tensor (the natural
    kernel output): accumulate it directly into a grouped param's ``main_grad``, or split it into
    per-expert views (``list(wgrad_stacked)``) for the positional autograd return. ``grad_probs``
    is the route-prob gradient from the standalone FC2 gated-activation backward (``None``
    otherwise).
    """

    dgrad: Optional[torch.Tensor] = None
    grad_probs: Optional[torch.Tensor] = None
    wgrad_stacked: Optional[torch.Tensor] = None
    # True when the wgrad was written directly into the caller-provided ``wgrad_out``;
    # False when ``wgrad_stacked`` is a fresh tensor the caller still needs to sink.
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

    - ``True`` (FC2): route-ordered ``2F`` pre-activation (FC1 output). The gated activation
      ``act(gate)*up[*prob]`` is applied in a standalone pass into an ``F``-wide transient, then
      a plain gather-GEMM + gather-combine -> ``[num_recv, out]``.
    - ``False`` (FC1): gather-in-GEMM -> padded ``[em_max, out]`` raw ``[gate | up]`` (``2F``).
    """
    if getattr(routing, "route_space", False):
        fc2_input = hidden_states
        if activation is not None:
            fc2_input = permute_free_gated_act_recompute(
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

    ``routing.route_space`` picks the FC2 vs FC1 dgrad/wgrad kernels. On the FC2 path,
    ``fc2_activation`` runs the standalone gated-activation backward after ``fc2_dgrad``;
    ``hidden_states`` is the saved ``2F`` pre-activation for act-bwd / wgrad recompute.

    ``wgrad_out`` (optional ``[E, out, in]`` accumulator) folds the wgrad straight into the
    caller's buffer (``+=`` if ``wgrad_accumulate`` else ``=``) instead of returning a fresh
    tensor. When used, ``result.wgrad_applied`` is ``True`` and ``result.wgrad_stacked`` is
    that same buffer.
    """
    grad_output = grad_output.contiguous()
    route_space = getattr(routing, "route_space", False)
    dgrad = None
    grad_probs = None
    wgrad_stacked = None
    wgrad_applied = False

    if route_space:
        # FC2: grad is token-space [num_recv, out]. dgrad gathers back to the route buffer,
        # then the standalone gated-activation backward maps dL/dF -> dL/d(2F) (+ dprob).
        if requires_dgrad:
            weights_stacked = _stack_expert_weights(weights)  # [E, H, F], zero-copy when grouped
            dgrad_f = permute_free_grouped_gemm_bf16_fc2_dgrad(
                grad_output, weights_stacked, routing
            )
            if fc2_activation is not None and hidden_states is not None:
                dgrad, grad_probs = permute_free_gated_act_bwd(
                    dgrad_f,
                    hidden_states,
                    routing,
                    activation=fc2_activation,
                    dispatched_probs=dispatched_probs,
                )
                if not (dispatched_probs is not None and dispatched_probs.requires_grad):
                    grad_probs = None
            else:
                dgrad = dgrad_f
        if requires_wgrad:
            weights_shape = (num_gemms, weights[0].size(0), weights[0].size(1))
            recompute = fc2_activation is not None and hidden_states is not None
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
        # FC1: grad is the padded [T * min(topk, E), 2F] route buffer (from FC2 act-bwd); dgrad
        # scatters the input grad back to received-token rows. No FC1 activation backward.
        if requires_dgrad:
            weights_stacked = _stack_expert_weights(weights)  # [E, N, H], zero-copy when grouped
            dgrad = permute_free_grouped_gemm_bf16_dgrad(grad_output, weights_stacked, routing)
        if requires_wgrad:
            weights_shape = (num_gemms, weights[0].size(0), weights[0].size(1))
            # bf16 or fp32 ``main_grad`` sink both accumulate in-kernel (fp32 via the fp32-store
            # variant); no scratch dW + separate fold.
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
