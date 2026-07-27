# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Permute-free route-list grouped GEMM for MoE (bf16).

Route-list contract (post-dispatch, no topk / no route weights):
- Activations ``A = [num_recv_tokens, H]``; boolean ``routing_map = [num_recv_tokens,
  num_local_experts]`` marks which local expert(s) each received token feeds.
- TE builds one expert-sorted ``sorted_slot_ids`` (received-token row per route) plus a
  compact-output map (``route_start``/``block_start``), then runs the gather-GEMM.
- FC1 fwd output is worst-case padded ``[T * min(topk, E), out_features]`` in expert order (valid rows
  are the compact route range ``[0, num_routes)``, tail is inert zero padding); dgrad returns
  ``dA = [num_recv_tokens, in_features]`` (scatter-add of the per-route gradients).
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import triton.language as tl

from ..moe_routing import MoERoutingMetadata, PermuteFreeMetadata
from .route_list_align import route_list_align, route_list_scan
from .route_combine import route_gather_combine
from .route_list_moe_gemm import (
    fused_route_list_moe,
    fused_gated_act_prob_bwd,
    fused_gated_act_prob_fwd,
)
from .route_list_moe_wgrad import fused_route_list_moe_wgrad
from .route_prob import _expert_per_route

__all__ = [
    "MoERoutingMetadata",
    "PermuteFreeMetadata",
    "permute_free_grouped_gemm_bf16",
    "permute_free_grouped_gemm_bf16_dgrad",
    "permute_free_grouped_gemm_bf16_wgrad",
    "permute_free_grouped_gemm_forward",
    "permute_free_grouped_gemm_backward",
    "PermuteFreeForwardResult",
    "PermuteFreeBackwardResult",
    "prepare_moe_align",
    "get_default_moe_kernel_config",
    "is_permute_free_grouped_gemm_enabled",
]

# Contraction step (routes accumulated per dot) for the route-list wgrad kernel. The wgrad
# align is padded to this block size, independent of the fwd/dgrad BLOCK_SIZE_M.
_WGRAD_CONTRACT_M = 32

# Max forward align/kernel ``block_m`` for the *gated* FlyDSL fwd. It stages a ``2F`` [gate|up]
# B-tile in LDS, so a larger Triton-optimal ``block_m`` (e.g. 256) overflows the 160 KB budget
# for wide-N tiles and forces FlyDSL onto a tiny (64x64) tile (losing most of its edge). Capping
# at 128 keeps a fast wide-N tile. See ``_fwd_align_block_size_m``.
_FLYDSL_GATED_BLOCK_M = 128

# Resolved lazily on first wgrad call: a callable (FlyDSL wgrad wrapper) if AITER's FlyDSL
# kernel imports, otherwise ``False`` (fall back to the Triton kernel). ``None`` = unresolved.
_FLYDSL_WGRAD: Any = None


def _get_flydsl_wgrad():
    """Resolve the AITER FlyDSL route-list wgrad kernel, or ``False`` if it can't be imported.

    The FlyDSL wrapper shares the exact ``fused_route_list_moe_wgrad`` contract, so it drops
    in as a faster backend on ROCm. If the import fails (AITER/FlyDSL not installed, or the
    import raises for any reason) we cache the failure and warn exactly once, then use the
    Triton kernel for the rest of the process.
    """
    global _FLYDSL_WGRAD
    if _FLYDSL_WGRAD is None:
        try:
            from aiter.ops.flydsl.moe_wgrad import flydsl_moe_wgrad_autotuned

            _FLYDSL_WGRAD = flydsl_moe_wgrad_autotuned
        except Exception as exc:  # noqa: BLE001 -- any import failure -> Triton fallback
            _FLYDSL_WGRAD = False
            warnings.warn(
                "permute-free wgrad: could not import the AITER FlyDSL wgrad kernel "
                f"({type(exc).__name__}: {exc}); falling back to the Triton wgrad kernel.",
                stacklevel=2,
            )
    return _FLYDSL_WGRAD


# Resolved lazily on first forward call: a ``(autotuned, supported)`` callable pair (FlyDSL
# fwd wrappers) if AITER's kernel imports, otherwise ``False`` (Triton fallback). ``None`` =
# unresolved.
_FLYDSL_FWD: Any = None


def _get_flydsl_fwd():
    """Resolve the AITER FlyDSL route-list forward gather-GEMM, or ``False`` if unavailable.

    Mirrors :func:`_get_flydsl_wgrad`: the FlyDSL fwd wrapper shares the exact
    ``fused_route_list_moe`` gather contract (same routing metadata, same fused gated
    activation / route-prob / pre-activation-save epilogue, and the transposed-B dgrad view),
    so it drops in as a faster bf16 backend on ROCm. Set ``NVTE_PERMUTE_FREE_FLYDSL_FWD=0`` to
    force the Triton kernel. Returns an ``(autotuned, supported)`` pair, or ``False``.
    """
    global _FLYDSL_FWD
    if _FLYDSL_FWD is None:
        if os.getenv("NVTE_PERMUTE_FREE_FLYDSL_FWD", "1") != "1":
            _FLYDSL_FWD = False
            return _FLYDSL_FWD
        try:
            from aiter.ops.flydsl.moe_fwd import (
                flydsl_moe_fwd_autotuned,
                flydsl_moe_fwd_supported,
            )

            _FLYDSL_FWD = (flydsl_moe_fwd_autotuned, flydsl_moe_fwd_supported)
        except Exception as exc:  # noqa: BLE001 -- any import failure -> Triton fallback
            _FLYDSL_FWD = False
            warnings.warn(
                "permute-free fwd: could not import the AITER FlyDSL fwd kernel "
                f"({type(exc).__name__}: {exc}); falling back to the Triton fwd kernel.",
                stacklevel=2,
            )
    return _FLYDSL_FWD


def _route_list_moe_fwd(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    routing: MoERoutingMetadata,
    *,
    num_recv_tokens: int,
    config: Dict[str, Any],
    index_a_by_route_pos: bool = False,
    activation: Optional[str] = None,
    dispatched_probs: Optional[torch.Tensor] = None,
    preact_out: Optional[torch.Tensor] = None,
    gated_a: bool = False,
) -> None:
    """Run the route-list forward gather-GEMM, preferring the FlyDSL backend.

    Drop-in for :func:`fused_route_list_moe`: dispatches to the AITER FlyDSL bf16 kernel when
    it's importable and can handle these operands (dtype / contraction contiguity / tile
    divisibility), else the Triton kernel. Both share the routing metadata and the fused
    gated-activation / route-prob / pre-activation-save epilogue, so the output is identical.
    """
    flydsl = _get_flydsl_fwd()
    block_m = int(config["BLOCK_SIZE_M"])
    if flydsl:
        autotuned, supported = flydsl
        if supported(A, B, block_m=block_m):
            autotuned(
                A,
                B,
                C,
                routing.sorted_slot_ids,
                routing.expert_ids,
                routing.block_start,
                routing.route_start,
                num_recv_tokens=num_recv_tokens,
                block_m=block_m,
                index_a_by_route_pos=index_a_by_route_pos,
                activation=activation,
                dispatched_probs=dispatched_probs,
                preact_out=preact_out,
                gated_a=gated_a,
            )
            return

    fused_route_list_moe(
        A,
        B,
        C,
        routing.sorted_slot_ids,
        routing.expert_ids,
        routing.num_tokens_post_padded,
        routing.block_start,
        routing.route_start,
        num_recv_tokens=num_recv_tokens,
        compute_type=tl.bfloat16,
        config=config,
        index_a_by_route_pos=index_a_by_route_pos,
        activation=activation,
        dispatched_probs=dispatched_probs,
        preact_out=preact_out,
        gated_a=gated_a,
    )


def _fwd_align_block_size_m(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    gated: bool,
    default_block_m: int,
) -> int:
    """Pick the forward align/kernel ``block_m``, favoring the backend that will run.

    ``block_m`` is baked into ``prepare_moe_align`` (it sets the row padding), so the fwd GEMM
    and the align must agree on it. The gated FlyDSL fwd is LDS-limited: a Triton-optimal
    ``block_m`` of 256 forces it onto a tiny tile (measured ~1.06x vs Triton) whereas 128 keeps
    a fast wide-N tile (~1.35x). When the FlyDSL gated path will actually be selected, cap
    ``block_m`` at :data:`_FLYDSL_GATED_BLOCK_M`; otherwise keep the (Triton-optimal) default.
    """
    if not gated or default_block_m <= _FLYDSL_GATED_BLOCK_M:
        return default_block_m
    flydsl = _get_flydsl_fwd()
    if not flydsl:
        return default_block_m
    _, supported = flydsl
    if supported(A, B, block_m=_FLYDSL_GATED_BLOCK_M):
        return _FLYDSL_GATED_BLOCK_M
    return default_block_m


def is_permute_free_grouped_gemm_enabled() -> bool:
    from torch.utils.cpp_extension import IS_HIP_EXTENSION

    return IS_HIP_EXTENSION and os.getenv("NVTE_PERMUTE_FREE_GROUPED_GEMM", "0") == "1"


def get_default_moe_kernel_config(num_tokens: int) -> Dict[str, Any]:
    """Return Triton tile config for the route-list bf16 MoE GEMM."""
    # aiter's tuned config is an optional accelerator.
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
       route_start, route_to_token, token_routes, token_route_count)``. Index tensors are
    ``int32``; ``num_tokens_post_padded`` (``[1]``) is a device scalar. The last two are the
    token->routes inverse map, ``None`` unless ``build_inverse_map``.
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
        return metadata

    counts, within = _ensure_route_scan(metadata)

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
        topk=metadata.topk,
        build_inverse_map=True,
    )
    metadata.sorted_slot_ids = sorted_slot_ids  # [T * min(topk, E)] int32: block-padded slot -> token
    metadata.expert_ids = expert_ids  # [blocks_max] int32: expert owning each block (-1 past end)
    metadata.num_tokens_post_padded = num_tokens_post_padded  # [1] int32 device scalar: padded extent
    metadata.block_start = block_start  # [E] int32: first block index of each expert (block units)
    metadata.route_start = route_start  # [E] int32: first compact route index of each expert
    metadata.route_to_token = route_to_token  # [routes_max] int32: compact route -> token
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
        route_start,
        _route_to_token,
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
    # route_start is block-size-independent; keep whichever is already cached.
    if metadata.route_start is None:
        metadata.route_start = route_start
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
        ``[T * min(topk, E), 2F]`` buffer with the raw ``[gate | up]`` pre-activation and return it
        alongside the activated output. The backward needs it to reconstruct the 2F GEMM-output
        gradient; keep ``False`` for inference to skip the extra store.

    Returns
    -------
    torch.Tensor
        Worst-case padded ``[T * min(topk, E), out_features]`` (or ``[T * min(topk, E), F]`` with a fused
        ``activation``), bf16 (expert-contiguous). The valid rows are the compact route range
        ``[0, num_routes)``; the tail ``[num_routes, em_max)`` is inert, *uninitialized*
        padding. Consumers MUST read only the compact range, using the routing metadata to
        locate each expert's rows:

        - ``route_start[e] = counts[0] + ... + counts[e-1]`` (``= cumsum(counts) - counts``):
          its starting offset in the packed output.
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

    gated_act = activation is not None
    if gated_act and out_features % 2 != 0:
        raise ValueError(
            f"Gated activation requires an even (gate+up) out_features, got {out_features}."
        )
    # Fused gated activation halves the stored width (2F -> F).
    stored_features = out_features // 2 if gated_act else out_features

    # Copy so the backend-aware block_m override never mutates the caller's config dict.
    kernel_config = dict(config or get_default_moe_kernel_config(num_recv_tokens))
    block_size_m = _fwd_align_block_size_m(
        hidden_states,
        weights_stacked,
        gated=gated_act,
        default_block_m=int(kernel_config["BLOCK_SIZE_M"]),
    )
    kernel_config["BLOCK_SIZE_M"] = block_size_m

    routing = prepare_moe_align(routing, block_size_m)

    # Worst-case (sync-free) allocation: size the output to the block-padded upper bound
    # em_max = sorted_slot_ids.shape[0], which is derived purely from shapes (num_recv_tokens,
    # num_experts, block_size) and is always >= num_routes. This avoids the device->host sync
    # (.item()) that a compact [num_routes, N] allocation would require.
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

    _route_list_moe_fwd(
        hidden_states,
        weights_stacked,
        output,
        routing,
        num_recv_tokens=num_recv_tokens,
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
    routes_max = int(routing.route_to_token.shape[0])
    token = routing.route_to_token.to(torch.int32)
    expert = _expert_per_route(routing, routes_max)
    # The route buffers are statically sized to the worst case ``routes_max = T * topk``, but
    # only the dense head ``[0, num_routes)`` is real (under EP this can be ~topk*E_local/E
    # smaller). ``num_tokens_post_padded`` is a device scalar >= num_routes (block-padded), so
    # it bounds the kernel to the real routes -- sync-free -- and lets the tail exit early.
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
        ``act``, shape ``[T * min(topk, E), F]``, bf16 (route/padded layout).
    """
    routes_max = int(routing.route_to_token.shape[0])
    token = routing.route_to_token.to(torch.int32)
    expert = _expert_per_route(routing, routes_max)
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
    # dX[route] = grad[route] @ W1[e] into an [T * min(topk, E), in] bf16 buffer (coalesced, no atomics),
    # then (2) a token-parallel gather-combine summing each token's route rows. This replaces
    # the fused atomic scatter-to-token, whose per-token contention dominated the FC1 dgrad.
    em_max = routing.sorted_slot_ids.shape[0]
    compact = torch.empty(
        (em_max, in_features),
        dtype=torch.bfloat16,
        device=grad_output.device,
    )
    _route_list_moe_fwd(
        grad_output,
        weights_t,
        compact,
        routing,
        num_recv_tokens=routing.num_recv_tokens,
        config=dgrad_config,
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
    config: Optional[Dict[str, Any]] = None,
    out: Optional[torch.Tensor] = None,
    accumulate: bool = False,
    swap_gather: bool = False,
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
    out:
        Optional ``[E, out, in]`` destination to write the wgrad into directly (bf16 or fp32).
        When given, the FlyDSL kernel folds the wgrad into it (``+=`` if ``accumulate`` else
        ``=``) with no scratch tensor / separate elementwise pass, and this tensor is returned.
    accumulate:
        With ``out``: add into it (``out += dW``) rather than overwrite (``out = dW``).
    swap_gather:
        Move the ``SORTED`` token-gather from ``hidden_states`` (K) to ``grad_output`` (N) -- the
        FC2 wgrad mode. ``hidden_states`` is then a route-ordered ``[num_routes, in]`` operand and
        ``grad_output`` a token-space ``[num_recv, out]`` operand, so the kernel emits ``dW`` in
        ``[E, out, in]`` directly (no transpose). FlyDSL-only; the Triton fallback doesn't
        support it.

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

    routing = _prepare_wgrad_align(routing, _WGRAD_CONTRACT_M)

    flydsl_wgrad = _get_flydsl_wgrad()
    if swap_gather and not flydsl_wgrad:
        # The Triton route-list wgrad has no swap-gather mode; the FC2 caller handles the
        # no-FlyDSL case itself (compute [F, H] then transpose), so it never asks for it here.
        raise RuntimeError(
            "swap_gather wgrad requires the FlyDSL backend; Triton fallback does not support it."
        )
    # ``out`` lets the caller fold the wgrad straight into an existing accumulator (a param's
    # ``main_grad`` / ``.grad``), skipping the scratch dW + separate add/copy. The FlyDSL kernel
    # supports a race-free accumulate/overwrite epilogue (fp32 or bf16) for this; without it (or
    # when it's unavailable) we fall back to computing a scratch bf16 dW and applying it.
    if out is not None:
        assert tuple(out.shape) == (num_experts, out_features, in_features), (
            f"wgrad out shape {tuple(out.shape)} != {(num_experts, out_features, in_features)}"
        )
        if flydsl_wgrad:
            out_dtype = "fp32" if out.dtype == torch.float32 else "bf16"
            flydsl_wgrad(
                x,
                grad_output,
                out,
                routing.wgrad_sorted_slot_ids,
                routing.wgrad_block_start,
                routing.wgrad_blocks_per_expert,
                routing.route_start,
                num_recv_tokens=routing.num_recv_tokens,
                accumulate=bool(accumulate),
                out_dtype=out_dtype,
                swap_gather=bool(swap_gather),
            )
            return out
        # No FlyDSL kernel: compute a scratch bf16 dW (overwrite), then apply into ``out``.
        dW = torch.zeros(
            (num_experts, out_features, in_features),
            dtype=torch.bfloat16,
            device=hidden_states.device,
        )
        fused_route_list_moe_wgrad(
            x, grad_output, dW,
            routing.wgrad_sorted_slot_ids, routing.wgrad_block_start,
            routing.wgrad_blocks_per_expert, routing.route_start,
            num_recv_tokens=routing.num_recv_tokens, config=None,
            contract_m=_WGRAD_CONTRACT_M,
        )
        if accumulate:
            out.add_(dW)
        else:
            out.copy_(dW)
        return out

    dW = torch.zeros(
        (num_experts, out_features, in_features),
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )
    if flydsl_wgrad:
        # FlyDSL backend (AITER): identical route-list contract, writes dW in place.
        # ``contract_m`` is baked into the FlyDSL kernel (WGRAD_BLOCK_M == _WGRAD_CONTRACT_M).
        flydsl_wgrad(
            x,
            grad_output,
            dW,
            routing.wgrad_sorted_slot_ids,
            routing.wgrad_block_start,
            routing.wgrad_blocks_per_expert,
            routing.route_start,
            num_recv_tokens=routing.num_recv_tokens,
            swap_gather=bool(swap_gather),
        )
    else:
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
    activation: Optional[str] = None,
    dispatched_probs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Route-list FC2 forward with gather-combine to token order (bf16 MoE).

    ``fc2_input`` is the route-ordered FC1 output. When ``activation`` is set (``"silu"`` /
    ``"gelu"``) it is the raw ``2F`` ``[gate | up]`` pre-activation and the kernel fuses
    ``act(gate) * up [* prob]`` on the ``A`` operand before the GEMM; otherwise it is the
    ``F``-wide activated buffer (legacy path). For each route the kernel reads its row
    (``index_a_by_route_pos=True``) and computes the compact per-route GEMM; a separate
    contention-free gather-combine pass sums each token's routes into its output row.

    Parameters
    ----------
    fc2_input:
        Route-ordered activations ``[T * min(topk, E), in_features]`` (``F``) or, when
        ``activation`` is set, the raw ``2F`` ``[gate | up]`` pre-activation from FC1.
    weights:
        Expert weights ``[num_experts, out_features, in_features]`` (W2) or list of ``[out,
        in]``.
    activation:
        When set, fuse ``act(gate) * up`` (+ optional ``dispatched_probs``) on the ``A``
        operand (FC2 prologue). Requires ``fc2_input.shape[-1] == 2 * in_features``.
    dispatched_probs:
        Optional ``[num_recv_tokens, num_experts]`` gating probabilities (fused in the FC2
        prologue when ``activation`` is set).
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
    gated_a = activation is not None
    if gated_a:
        if fc2_input.shape[-1] != 2 * in_features:
            raise ValueError(
                f"gated FC2 expects 2F preact input ({2 * in_features} cols), "
                f"got {fc2_input.shape[-1]}."
            )
    elif fc2_input.shape[-1] != in_features:
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
    _route_list_moe_fwd(
        fc2_input,
        weights_stacked,
        compact,
        routing,
        num_recv_tokens=routing.num_recv_tokens,
        config=kernel_config,
        index_a_by_route_pos=True,
        activation=activation,
        dispatched_probs=dispatched_probs,
        gated_a=gated_a,
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
    _route_list_moe_fwd(
        grad_output,
        weights_t,
        dgrad,
        routing,
        num_recv_tokens=grad_output.shape[0],
        config=dgrad_config,
        index_a_by_route_pos=False,
    )
    return dgrad


def permute_free_grouped_gemm_bf16_fc2_wgrad(
    fc2_input: torch.Tensor,
    grad_output: torch.Tensor,
    weights_shape,
    routing: MoERoutingMetadata,
    *,
    config: Optional[Dict[str, Any]] = None,
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

    * **FlyDSL** (``swap_gather``): gather ``grad_output`` on the N-operand and route-walk
      ``fc2_input`` on the K-operand, so the kernel emits ``dW2`` in ``[E, H, F]`` directly --
      coalesced, no transpose, and foldable straight into ``out`` (``main_grad`` / ``.grad``).
    * **Triton fallback** (no swap mode): compute ``[E, F, H] = dW2^T`` with operands swapped,
      then ``transpose(1, 2).contiguous()`` back to ``[E, H, F]`` and, if ``out`` is given,
      apply it there.

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

    if _get_flydsl_wgrad():
        # swap_gather: emit [E, H(out), F(in)] directly. hidden_states = fc2_input (route K=in),
        # grad_output = grad_output (token-gathered N=out).
        return permute_free_grouped_gemm_bf16_wgrad(
            fc2_input,
            grad_output,
            (num_experts, out_features, in_features),
            routing,
            config=config,
            out=out,
            accumulate=accumulate,
            swap_gather=True,
        )

    # Triton fallback: no swap mode. Compute [E, F, H] = dW2^T with operands swapped, transpose.
    dW_t = permute_free_grouped_gemm_bf16_wgrad(
        grad_output,
        fc2_input,
        (num_experts, in_features, out_features),
        routing,
        config=config,
    )
    dW = dW_t.transpose(1, 2).contiguous()  # [E, H, F]
    if out is None:
        return dW
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
class PermuteFreeForwardResult:
    """Output of :func:`permute_free_grouped_gemm_forward`.

    ``preact`` is the raw ``2F`` pre-activation saved for the FC1 gated-activation backward;
    it is ``None`` on every other path (FC2, or FC1 without a fused activation / without a
    backward).
    """

    out: torch.Tensor
    preact: Optional[torch.Tensor] = None


@dataclass
class PermuteFreeBackwardResult:
    """Gradients from :func:`permute_free_grouped_gemm_backward`.

    ``dgrad`` / ``wgrad_stacked`` are ``None`` when the corresponding gradient was not requested.
    ``wgrad_stacked`` is the weight gradient as a single ``[E, out, in]`` tensor (the natural
    kernel output): accumulate it directly into a grouped param's ``main_grad``, or split it into
    per-expert views (``list(wgrad_stacked)``) for the positional autograd return. ``grad_probs``
    is the route-prob gradient for the FC1 fused-prob path (``None`` otherwise).
    """

    dgrad: Optional[torch.Tensor] = None
    grad_probs: Optional[torch.Tensor] = None
    wgrad_stacked: Optional[torch.Tensor] = None
    # True when the wgrad was written directly into the caller-provided ``wgrad_out`` (FC1
    # path), so the caller must not re-apply it. False when ``wgrad_stacked`` is a fresh tensor
    # the caller still needs to sink (FC2 -- its transpose precludes an in-place accumulate --
    # or the plain positional-return path).
    wgrad_applied: bool = False


def permute_free_grouped_gemm_forward(
    hidden_states: torch.Tensor,
    weights: torch.Tensor | list[torch.Tensor],
    routing: MoERoutingMetadata,
    *,
    activation: Optional[str] = None,
    dispatched_probs: Optional[torch.Tensor] = None,
) -> PermuteFreeForwardResult:
    """Dispatch the permute-free grouped-GEMM forward from the routing direction + fusion hints.

    ``routing.route_space`` selects the direction:

    - ``True`` (FC2): route-ordered ``2F`` pre-activation (FC1 output), fused
      ``act(gate)*up[*prob]`` prologue + GEMM + gather-combine -> ``[num_recv, out]``.
    - ``False`` (FC1): gather-in-GEMM -> padded ``[T * min(topk, E), 2F]`` raw ``[gate | up]``
      pre-activation (no activation fusion here; the ``activation`` hint is consumed on FC2).
    """
    if getattr(routing, "route_space", False):
        return PermuteFreeForwardResult(
            permute_free_grouped_gemm_bf16_fc2(
                hidden_states,
                weights,
                routing,
                activation=activation,
                dispatched_probs=dispatched_probs,
            )
        )
    return PermuteFreeForwardResult(
        permute_free_grouped_gemm_bf16(hidden_states, weights, routing)
    )


def permute_free_grouped_gemm_backward(
    grad_output: torch.Tensor,
    *,
    routing: MoERoutingMetadata,
    weights: list[torch.Tensor],
    num_gemms: int,
    hidden_states: Optional[torch.Tensor] = None,
    requires_dgrad: bool = False,
    requires_wgrad: bool = False,
    fc1_activation: Optional[str] = None,
    preact: Optional[torch.Tensor] = None,
    dispatched_probs: Optional[torch.Tensor] = None,
    fc2_activation: Optional[str] = None,
    wgrad_out: Optional[torch.Tensor] = None,
    wgrad_accumulate: bool = False,
) -> PermuteFreeBackwardResult:
    """Dispatch the permute-free grouped-GEMM backward (mirror of the forward dispatch).

    ``routing.route_space`` picks the FC2 vs FC1 dgrad/wgrad kernels. On the FC1
    gated-activation path (``fc1_activation`` set) the raw ``2F`` GEMM-output gradient (and the
    route-prob gradient) is first reconstructed from the saved ``preact`` and fed to the
    *unchanged* dgrad/wgrad. ``hidden_states`` (the forward input) is only needed for wgrad.

    On the FC2 path (``route_space=True``), when ``fc2_activation`` is set together with
    ``preact`` the wgrad rebuilds its F-wide input (``act(gate)*up*prob``) in-flight from the FC1
    pre-activation instead of consuming a stored ``hidden_states`` -- the FC1->FC2 recompute
    handoff, letting the FC2 forward skip saving its F-wide input.

    ``wgrad_out`` (optional ``[E, out, in]`` accumulator) folds the wgrad straight into the
    caller's buffer (``+=`` if ``wgrad_accumulate`` else ``=``) instead of returning a fresh
    tensor. Both FC1 and FC2 support it (FC2 emits ``[E, H, F]`` directly via ``swap_gather`` on
    FlyDSL, or transposes + applies on the Triton fallback). When used, ``result.wgrad_applied``
    is ``True`` and ``result.wgrad_stacked`` is that same buffer.
    """
    grad_output = grad_output.contiguous()
    route_space = getattr(routing, "route_space", False)
    dgrad = None
    grad_probs = None
    wgrad_stacked = None
    wgrad_applied = False

    if route_space:
        # FC2: grad is token-space [num_recv, out]. dgrad gathers back to the compact route
        # buffer [T * min(topk, E), F] (GEMM operand), then the gated-activation backward maps
        # dL/dF -> dL/d(2F) (+ dprob) when the forward fused the FC2 prologue.
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
