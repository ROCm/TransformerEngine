# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.


"""Tests for the route-list permute-free bf16 MoE gather-GEMM kernels."""

import dataclasses

import pytest
import torch
from torch.utils.cpp_extension import IS_HIP_EXTENSION

from transformer_engine.pytorch import GroupedLinear
from transformer_engine.pytorch.moe import (
    MoERoutingMetadata,
    PermuteFreeMetadata,
    is_permute_free_grouped_gemm_enabled,
    permute_free_gated_act_bwd,
    permute_free_gated_act_fwd,
    permute_free_grouped_gemm_bf16,
    permute_free_grouped_gemm_bf16_dgrad,
    permute_free_grouped_gemm_bf16_fc2,
    permute_free_grouped_gemm_bf16_fc2_dgrad,
    permute_free_grouped_gemm_bf16_fc2_wgrad,
    permute_free_grouped_gemm_bf16_wgrad,
    prepare_moe_align,
)
from transformer_engine.pytorch.moe.permute_free_grouped_gemm import (
    _FLYDSL_FWD_BLOCK_M,
    _WGRAD_CONTRACT_M,
    _expert_per_route,
    _prepare_wgrad_align,
    moe_align_route_list,
)
from transformer_engine.pytorch.moe.pf_helper_kernels import route_list_scan
from utils import assert_close, dtype_tols

pytestmark = pytest.mark.skipif(
    not (IS_HIP_EXTENSION and torch.cuda.is_available()),
    reason="Permute-free grouped GEMM tests require ROCm and CUDA device.",
)

_TEST_SEED = 1

# Absolute floor for the elementwise bound, as a fraction of the reference's peak magnitude.
# ``dtype_tols`` only supplies a relative bound, which is meaningless on the gather-combine
# outputs: those sum several routes into one token, so cancellation leaves elements near zero
# whose error is set by the magnitude of the summands rather than by their own value. The
# worst floor measured over every comparison in this file (1116 of them, 6 seeds, gfx950) is
# 4.6e-3 x peak, i.e. ~1.2 bf16 ULP; allow 4 ULP, matching the bf16 rtol ``dtype_tols`` uses.
_BF16_ATOL_SCALE = 4 * 2**-8


@pytest.fixture(autouse=True)
def _cuda_sync_after_test():
    yield
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@pytest.fixture(autouse=True)
def _test_seed():
    torch.manual_seed(_TEST_SEED)
    yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _assert_bf16_close(actual: torch.Tensor, ref: torch.Tensor) -> None:
    """Elementwise-compare a bf16 kernel result against a higher-precision reference."""
    tols = dtype_tols(torch.bfloat16)
    tols["atol"] = max(tols["atol"], _BF16_ATOL_SCALE * ref.abs().max().item())
    assert_close(actual, ref, **tols)


def _random_routing_map(num_recv_tokens, num_experts, max_hits, device, seed=_TEST_SEED):
    """Boolean [num_recv_tokens, num_experts] with 1..max_hits local experts per token."""
    gen = torch.Generator(device=device).manual_seed(seed)
    routing_map = torch.zeros(num_recv_tokens, num_experts, dtype=torch.bool, device=device)
    for t in range(num_recv_tokens):
        k = int(torch.randint(1, max_hits + 1, (1,), device=device, generator=gen).item())
        experts = torch.randperm(num_experts, device=device, generator=gen)[:k]
        routing_map[t, experts] = True
    # Guarantee every expert owns at least one route so per-expert refs are exercised.
    for e in range(num_experts):
        if not routing_map[:, e].any():
            routing_map[int(e) % num_recv_tokens, e] = True
    return routing_map


def _dense_route_order(routing_map):
    """Expert-sorted (token, expert) dense route list, matching ``moe_align_route_list``."""
    tok, exp = routing_map.nonzero(as_tuple=True)
    order = torch.argsort(exp, stable=True)
    return tok[order].to(torch.int64), exp[order].to(torch.int64)


def _route_slots(routing, num_recv_tokens):
    """Per-slot indices of an aligned routing metadata.

    Returns ``(em_max, valid, tok, exp)``: the block-padded slot count, the mask of slots
    backing a real route, and the token / local-expert id of each slot (padding slots are
    clamped in range so they can be indexed and then masked out).
    """
    em_max = int(routing.sorted_slot_ids.shape[0])
    valid = routing.sorted_slot_ids < num_recv_tokens
    tok = routing.sorted_slot_ids.to(torch.int64).clamp_(0, num_recv_tokens - 1)
    exp = _expert_per_route(routing, em_max).to(torch.int64).clamp_min(0)
    return em_max, valid, tok, exp


def _randn_route_buffer(em_max, width, valid):
    """Block-padded route buffer: random bf16 on the valid slots, zeros on the padding."""
    buf = torch.zeros(em_max, width, device="cuda", dtype=torch.bfloat16)
    buf[valid] = torch.randn(int(valid.sum().item()), width, device="cuda", dtype=torch.bfloat16)
    return buf


def _sum_by_expert(per_slot, exp, weights_shape):
    """Reduce a per-route ``[em_max, out, in]`` outer product into ``[E, out, in]``."""
    ref = torch.zeros(weights_shape, device=per_slot.device, dtype=torch.float32)
    ref.index_add_(0, exp, per_slot)
    return ref


def _gated_act_ref(gate: torch.Tensor, activation: str) -> torch.Tensor:
    if activation == "silu":
        return torch.nn.functional.silu(gate)
    if activation == "gelu":
        return torch.nn.functional.gelu(gate, approximate="tanh")
    raise ValueError(f"unsupported activation: {activation}")


# ---------------------------------------------------------------------------
# Route-list gather-GEMM: fwd / dgrad / wgrad
# ---------------------------------------------------------------------------
# FlyDSL FC1 dgrad (route-read NN): out_features (N) must be a multiple of 64.
# FC1 fwd: in_features (K) >= 128 and % 64. FC2 fwd/dgrad mirror FC1 with swapped layouts.
_GEMM_SHAPE_CONFIGS = [
    # id, T, in, out, E, max_h, fc1_fwd (in, out), fc2_fwd (in, out), fc2_T
    ("orig", 128, 96, 128, 8, 4, (128, 256), (128, 128), None),
    ("mid", 96, 128, 128, 6, 3, None, None, None),
    ("wide", 192, 128, 192, 8, 2, (192, 384), (192, 192), None),
    ("few-experts", 64, 96, 128, 4, 1, (128, 512), (128, 128), 128),
    ("large", 256, 128, 256, 16, 4, (256, 512), (256, 256), None),
]


def _build_route_list_gemm_cases():
    cases = []
    for sid, t, inf, outf, e, max_hits, fc1_fwd, fc2_fwd, fc2_t in _GEMM_SHAPE_CONFIGS:
        fc1_fwd = fc1_fwd or (inf, outf)
        fc2_fwd = fc2_fwd or (inf, outf)
        t_fc2 = fc2_t or t
        specs = [
            ("fc1", "fwd", None, t, *fc1_fwd),
            ("fc1", "dgrad", None, t, inf, outf),
            ("fc1", "wgrad", None, t, inf, outf),
            ("fc2", "fwd", None, t_fc2, *fc2_fwd),
            ("fc2", "dgrad", None, t_fc2, inf, outf),
            ("fc2", "wgrad", "stored", t_fc2, inf, outf),
            ("fc2", "wgrad", "recompute", t_fc2, inf, outf),
        ]
        for layer, mode, variant, tokens, in_features, out_features in specs:
            variant_id = f"-{variant}" if variant else ""
            cases.append(
                pytest.param(
                    layer,
                    mode,
                    variant,
                    tokens,
                    in_features,
                    out_features,
                    e,
                    max_hits,
                    id=f"{layer}-{mode}{variant_id}-{sid}",
                )
            )
    return cases


@pytest.mark.parametrize(
    "layer,mode,wgrad_variant,num_recv_tokens,in_features,out_features,num_experts,max_hits",
    _build_route_list_gemm_cases(),
)
def test_route_list_gemm(
    layer,
    mode,
    wgrad_variant,
    num_recv_tokens,
    in_features,
    out_features,
    num_experts,
    max_hits,
):
    """Route-list grouped GEMM kernels: FC1 vs FC2 (mirrored token/route layouts).

    FC1: activation token-ordered, grad block-padded route-ordered.
    FC2: activation block-padded route-ordered, grad token-ordered.
    """
    routing_map = _random_routing_map(num_recv_tokens, num_experts, max_hits, "cuda")
    routing = MoERoutingMetadata(routing_map=routing_map, num_experts=num_experts)
    num_routes = int(routing_map.sum().item())
    weights_shape = (num_experts, out_features, in_features)
    weights = torch.randn(weights_shape, device="cuda", dtype=torch.bfloat16)

    # FC1 forward is the only entry point that builds the align itself (on demand, inside the
    # kernel wrapper), so read the slot layout back only after the call.
    if layer == "fc1" and mode == "fwd":
        hidden = torch.randn(num_recv_tokens, in_features, device="cuda", dtype=torch.bfloat16)
        out = permute_free_grouped_gemm_bf16(hidden, weights, routing)
        em_max, valid, tok, exp = _route_slots(routing, num_recv_tokens)
        assert out.shape == (em_max, out_features)
        assert int(valid.sum().item()) == num_routes
        ref = torch.einsum("rk,rnk->rn", hidden[tok].float(), weights[exp].float())
        _assert_bf16_close(out[valid], ref[valid])
        return

    prepare_moe_align(routing, _FLYDSL_FWD_BLOCK_M)
    em_max, valid, tok, exp = _route_slots(routing, num_recv_tokens)
    valid_f = valid[:, None].float()

    if mode == "fwd":
        # FC2 forward: route-ordered activation in, token-ordered gather-combine out.
        fc2_input = _randn_route_buffer(em_max, in_features, valid)
        out = permute_free_grouped_gemm_bf16_fc2(fc2_input, weights, routing)
        assert out.shape == (num_recv_tokens, out_features)
        per_slot = torch.einsum("rf,rhf->rh", fc2_input.float(), weights[exp].float()) * valid_f
        ref = torch.zeros(num_recv_tokens, out_features, device="cuda", dtype=torch.float32)
        ref.index_add_(0, tok, per_slot)
        _assert_bf16_close(out, ref)
        return

    if mode == "dgrad":
        if layer == "fc1":
            grad = _randn_route_buffer(em_max, out_features, valid)
            dgrad = permute_free_grouped_gemm_bf16_dgrad(grad, weights, routing)
            assert dgrad.shape == (num_recv_tokens, in_features)
            assert int(valid.sum().item()) == num_routes
            per_slot = torch.einsum("rn,rnk->rk", grad.float(), weights[exp].float()) * valid_f
            ref = torch.zeros(num_recv_tokens, in_features, device="cuda", dtype=torch.float32)
            ref.index_add_(0, tok, per_slot)
            _assert_bf16_close(dgrad, ref)
            return

        grad_output = torch.randn(
            num_recv_tokens, out_features, device="cuda", dtype=torch.bfloat16
        )
        dgrad = permute_free_grouped_gemm_bf16_fc2_dgrad(grad_output, weights, routing)
        assert dgrad.shape == (em_max, in_features)
        ref = torch.einsum("rh,rhf->rf", grad_output[tok].float(), weights[exp].float())
        _assert_bf16_close(dgrad[valid], ref[valid])
        return

    assert mode == "wgrad"
    if layer == "fc1":
        grad = _randn_route_buffer(em_max, out_features, valid)
        hidden = torch.randn(num_recv_tokens, in_features, device="cuda", dtype=torch.bfloat16)
        dW = permute_free_grouped_gemm_bf16_wgrad(hidden, grad, weights_shape, routing)
        assert dW.shape == weights_shape
        assert int(valid.sum().item()) == num_routes
        per_slot = torch.einsum("rn,rk->rnk", grad.float(), hidden[tok].float())
        ref = _sum_by_expert(per_slot * valid[:, None, None].float(), exp, weights_shape)
        _assert_bf16_close(dW, ref)
        return

    grad_output = torch.randn(num_recv_tokens, out_features, device="cuda", dtype=torch.bfloat16)

    if wgrad_variant == "stored":
        fc2_input = _randn_route_buffer(em_max, in_features, valid)
        per_slot = torch.einsum("rh,rf->rhf", grad_output[tok].float(), fc2_input.float())
        ref = _sum_by_expert(per_slot * valid[:, None, None].float(), exp, weights_shape)

        dW2 = permute_free_grouped_gemm_bf16_fc2_wgrad(
            fc2_input, grad_output, weights_shape, routing
        )
        assert dW2.shape == weights_shape
        _assert_bf16_close(dW2, ref)

        # ``out=`` folds the wgrad straight into a caller buffer (the fp32 ``main_grad`` sink),
        # overwriting or accumulating in-kernel.
        out = torch.zeros(weights_shape, device="cuda", dtype=torch.float32)
        permute_free_grouped_gemm_bf16_fc2_wgrad(
            fc2_input, grad_output, weights_shape, routing, out=out, accumulate=False
        )
        _assert_bf16_close(out, ref)
        permute_free_grouped_gemm_bf16_fc2_wgrad(
            fc2_input, grad_output, weights_shape, routing, out=out, accumulate=True
        )
        _assert_bf16_close(out, 2.0 * ref)
        return

    # Recompute-from-preact: the wgrad rebuilds the F-wide activation from the saved 2F preact
    # instead of consuming a stored one, and must match the stored-activation wgrad.
    assert wgrad_variant == "recompute"
    preact = _randn_route_buffer(em_max, 2 * in_features, valid)
    probs = torch.rand(num_recv_tokens, num_experts, device="cuda", dtype=torch.float32)
    act_ref = (
        _gated_act_ref(preact[:, :in_features].float(), "silu")
        * preact[:, in_features:].float()
        * probs[tok, exp][:, None]
        * valid_f
    )

    dW2_stored = permute_free_grouped_gemm_bf16_fc2_wgrad(
        act_ref.to(torch.bfloat16), grad_output, weights_shape, routing
    )
    dW2_recompute = permute_free_grouped_gemm_bf16_fc2_wgrad(
        None,
        grad_output,
        weights_shape,
        routing,
        preact=preact,
        dispatched_probs=probs,
        activation="silu",
    )
    ref = _sum_by_expert(
        torch.einsum("rh,rf->rhf", grad_output[tok].float(), act_ref), exp, weights_shape
    )

    _assert_bf16_close(dW2_stored, ref)
    _assert_bf16_close(dW2_recompute, ref)
    _assert_bf16_close(dW2_recompute, dW2_stored)


# ---------------------------------------------------------------------------
# Standalone gated activation (FC1 epilogue / FC2 prologue)
# ---------------------------------------------------------------------------
_GATED_ACT_SHAPES = [
    pytest.param(128, 128, 8, 4, id="orig"),
    pytest.param(96, 128, 6, 3, id="mid"),
    pytest.param(192, 192, 8, 2, id="wide"),
    pytest.param(128, 192, 4, 2, id="narrow-experts"),
    pytest.param(256, 256, 16, 4, id="large"),
    pytest.param(64, 128, 4, 1, id="few-experts"),
]
_GATED_ACT_CASES = [
    pytest.param("fwd", "silu", id="fwd-silu"),
    pytest.param("fwd", "gelu", id="fwd-gelu"),
    pytest.param("bwd", "silu", id="bwd-silu"),
    pytest.param("bwd", "gelu", id="bwd-gelu"),
]
_GATED_ACT_PROB_CASES = [
    pytest.param(True, id="probs"),
    pytest.param(False, id="no-probs"),
]


@pytest.mark.parametrize("use_probs", _GATED_ACT_PROB_CASES)
@pytest.mark.parametrize("num_recv_tokens,in_features,num_experts,max_hits", _GATED_ACT_SHAPES)
@pytest.mark.parametrize("mode,activation", _GATED_ACT_CASES)
def test_route_list_gated_act(
    mode, activation, use_probs, num_recv_tokens, in_features, num_experts, max_hits
):
    """Standalone gated activation ``act(gate)*up[*prob]`` vs PyTorch autograd.

    ``use_probs=False`` covers the prob-less fusion, where the backward returns no ``dprob``.
    """
    routing_map = _random_routing_map(num_recv_tokens, num_experts, max_hits, "cuda")
    routing = prepare_moe_align(
        MoERoutingMetadata(routing_map=routing_map, num_experts=num_experts), _FLYDSL_FWD_BLOCK_M
    )
    em_max, valid, tok, exp = _route_slots(routing, num_recv_tokens)
    valid_f = valid[:, None].float()

    if mode == "fwd":
        preact = _randn_route_buffer(em_max, 2 * in_features, valid)
        probs = (
            torch.rand(num_recv_tokens, num_experts, device="cuda", dtype=torch.float32)
            if use_probs
            else None
        )

        act = permute_free_gated_act_fwd(
            preact, routing, activation=activation, dispatched_probs=probs
        )
        assert act.shape == (em_max, in_features)

        gate, up = preact[:, :in_features].float(), preact[:, in_features:].float()
        act_ref = _gated_act_ref(gate, activation) * up * valid_f
        if use_probs:
            act_ref = act_ref * probs[tok, exp][:, None]
        _assert_bf16_close(act[valid], act_ref[valid])
        return

    gate = (torch.randn(em_max, in_features, device="cuda") * valid_f).requires_grad_(True)
    up = (torch.randn(em_max, in_features, device="cuda") * valid_f).requires_grad_(True)
    probs = None
    if use_probs:
        probs = torch.rand(
            num_recv_tokens, num_experts, device="cuda", dtype=torch.float32, requires_grad=True
        )
    grad_out = _randn_route_buffer(em_max, in_features, valid)

    act_ref = _gated_act_ref(gate, activation) * up * valid_f
    if use_probs:
        act_ref = act_ref * probs[tok, exp][:, None]
    (act_ref * grad_out.float()).sum().backward()
    dpre_ref = torch.cat([gate.grad, up.grad], dim=1)

    preact = torch.cat([gate.detach(), up.detach()], dim=1).to(torch.bfloat16)
    fused_probs = probs.detach() if use_probs else None
    dpre, dprob = permute_free_gated_act_bwd(
        grad_out, preact, routing, activation=activation, dispatched_probs=fused_probs
    )
    assert dpre.shape == (em_max, 2 * in_features)
    _assert_bf16_close(dpre[valid], dpre_ref[valid])
    if use_probs:
        assert dprob.shape == (num_recv_tokens, num_experts)
        _assert_bf16_close(dprob, probs.grad)
    else:
        assert dprob is None

    # ``return_fc2_input`` re-materialises the F-wide forward activation from values the
    # backward already holds, so the FC2 wgrad can skip a second pass over the 2F preact. It
    # must not perturb the gradients and must match a standalone forward.
    dpre_emit, dprob_emit, fc2_input = permute_free_gated_act_bwd(
        grad_out,
        preact,
        routing,
        activation=activation,
        dispatched_probs=fused_probs,
        return_fc2_input=True,
    )
    fc2_ref = permute_free_gated_act_fwd(
        preact, routing, activation=activation, dispatched_probs=fused_probs
    )
    assert fc2_input.shape == (em_max, in_features)
    assert torch.equal(dpre_emit[valid], dpre[valid])
    _assert_bf16_close(fc2_input[valid], fc2_ref[valid])
    if use_probs:
        _assert_bf16_close(dprob_emit, dprob)
    else:
        assert dprob_emit is None


# ---------------------------------------------------------------------------
# Routing / align infrastructure
# ---------------------------------------------------------------------------
_ALIGN_SHAPES = [
    pytest.param(128, 8, 4, id="orig"),
    pytest.param(96, 6, 3, id="mid"),
]


def _route_scan_ref(routing_map):
    """CPU reference for ``route_list_scan``: per-expert counts + within-expert ranks [E, T]."""
    vals = routing_map.to(torch.int32)
    counts = vals.sum(dim=0).to(torch.int32)
    within = (vals.cumsum(dim=0) - vals).to(torch.int32).T.contiguous()
    return counts, within


@pytest.mark.parametrize(
    "tighten_bound",
    [pytest.param(False, id="dense-bound"), pytest.param(True, id="topk-bound")],
)
@pytest.mark.parametrize("num_recv_tokens,num_experts,max_hits", _ALIGN_SHAPES)
def test_route_list_align_buffers(num_recv_tokens, num_experts, max_hits, tighten_bound):
    """``route_list_scan`` + ``moe_align_route_list`` build the expert-sorted route list.

    Covers the block-independent scan, the per-expert block layout, the token->route inverse
    map, and the ``topk`` hint tightening the (still sync-free) static over-allocation.
    """
    routing_map = _random_routing_map(num_recv_tokens, num_experts, max_hits, "cuda")
    routes_per_token = routing_map.sum(dim=1).to(torch.int32)
    assert int(routes_per_token.max().item()) <= max_hits
    topk = max_hits if tighten_bound else None

    counts, within = route_list_scan(routing_map, num_experts=num_experts)
    counts_ref, within_ref = _route_scan_ref(routing_map)
    assert counts.dtype == torch.int32 and within.dtype == torch.int32
    assert torch.equal(counts, counts_ref)
    assert torch.equal(within, within_ref)

    (
        sorted_slot_ids,
        _expert_ids,
        num_tokens_post_padded,
        block_start,
        blocks_per_expert,
        token_routes,
        token_route_count,
    ) = moe_align_route_list(
        routing_map,
        num_experts=num_experts,
        block_size=_FLYDSL_FWD_BLOCK_M,
        scan=(counts, within),
        topk=topk,
        build_inverse_map=True,
    )

    # Static (sync-free) over-allocation: T * min(topk, E) routes rounded up to whole blocks,
    # plus one partial block per expert. The topk hint shrinks it by up to E / topk.
    max_per_token = num_experts if topk is None else min(topk, num_experts)
    blocks_max = (
        num_recv_tokens * max_per_token + _FLYDSL_FWD_BLOCK_M - 1
    ) // _FLYDSL_FWD_BLOCK_M + num_experts
    assert sorted_slot_ids.shape[0] == blocks_max * _FLYDSL_FWD_BLOCK_M
    assert tighten_bound == (max_per_token < num_experts)

    # Per-expert block layout: ceil(count / block_m) blocks each, laid out back to back.
    blocks_ref = (counts_ref + _FLYDSL_FWD_BLOCK_M - 1) // _FLYDSL_FWD_BLOCK_M
    assert torch.equal(blocks_per_expert, blocks_ref)
    assert torch.equal(block_start, (blocks_ref.cumsum(0) - blocks_ref).to(torch.int32))
    assert int(num_tokens_post_padded.item()) == int(blocks_ref.sum().item()) * _FLYDSL_FWD_BLOCK_M

    # The valid slots are exactly the dense expert-sorted route list; the rest is padding.
    route_token, _ = _dense_route_order(routing_map)
    valid = sorted_slot_ids < num_recv_tokens
    assert torch.equal(sorted_slot_ids[valid].to(torch.int64), route_token)

    # Inverse map: token_routes[t, :count] are the padded slots owned by token t.
    assert token_routes.shape == (num_recv_tokens, max_per_token)
    assert torch.equal(token_route_count, routes_per_token)
    listed = torch.arange(max_per_token, device="cuda")[None, :] < routes_per_token[:, None]
    owner = sorted_slot_ids[token_routes.to(torch.int64)]
    tokens = torch.arange(num_recv_tokens, device="cuda", dtype=torch.int32)
    assert torch.equal(owner[listed], tokens[:, None].expand_as(owner)[listed])


def test_route_list_align_caching():
    """``prepare_moe_align`` / ``_prepare_wgrad_align`` build once and cache on the metadata.

    The fwd/dgrad align (``BLOCK_SIZE_M``) and the wgrad align (``CONTRACT_M``) are separate
    builds over the same routing map, sharing one block-independent scan; the wgrad holder is
    shared by reference with the FC2 ``route_space`` view so only one backward pays for it.
    """
    num_recv_tokens, num_experts, max_hits = 128, 8, 3
    routing_map = _random_routing_map(num_recv_tokens, num_experts, max_hits, "cuda")
    routing = PermuteFreeMetadata(
        routing_map=routing_map, num_experts=num_experts, topk=max_hits
    )
    num_routes = int(routing_map.sum().item())

    # Nothing is built until the align runs.
    assert routing.slot_expert_ids is None
    assert routing.route_counts is None
    prepare_moe_align(routing, _FLYDSL_FWD_BLOCK_M)
    em_max = int(routing.sorted_slot_ids.shape[0])
    scan_counts = routing.route_counts
    assert scan_counts is not None

    # slot_expert_ids is the block-level expert_ids broadcast to per-slot granularity, and
    # ``_expert_per_route`` hands back that cached tensor rather than recomputing it.
    assert routing.slot_expert_ids.shape == (em_max,)
    assert routing.slot_expert_ids.dtype == torch.int32
    block_idx = (torch.arange(em_max, device="cuda") // int(routing.block_size_m)).clamp(
        max=routing.expert_ids.numel() - 1
    )
    manual = routing.expert_ids[block_idx].to(torch.int32)
    assert torch.equal(routing.slot_expert_ids, manual)
    assert _expert_per_route(routing, em_max) is routing.slot_expert_ids

    # Valid slots carry the expert-sorted dense route experts.
    _, route_expert = _dense_route_order(routing_map)
    valid = routing.sorted_slot_ids < num_recv_tokens
    assert torch.equal(routing.slot_expert_ids[valid].to(torch.int64), route_expert)

    # Rebuild-on-early-return: drop the cache but keep the align, then re-prepare with the same
    # block_m (hits the cached-align early return) -- it must repopulate rather than stay None.
    routing.slot_expert_ids = None
    prepare_moe_align(routing, _FLYDSL_FWD_BLOCK_M)
    assert torch.equal(routing.slot_expert_ids, manual)

    # The FC2 route-space view shares the mutable wgrad-align holder, so whichever backward runs
    # first builds the block-CONTRACT_M buffers and the other reuses them.
    fc2_routing = dataclasses.replace(routing, route_space=True)
    assert fc2_routing.wgrad_align is routing.wgrad_align

    _prepare_wgrad_align(routing, _WGRAD_CONTRACT_M)
    wgrad_align = routing.wgrad_align
    assert wgrad_align.block_size == _WGRAD_CONTRACT_M
    assert wgrad_align.block_start is not None
    assert wgrad_align.blocks_per_expert is not None
    assert wgrad_align.sorted_slot_ids.shape[0] != em_max
    # The scan is block-size independent, so the second align reuses the cached one.
    assert routing.route_counts is scan_counts

    _prepare_wgrad_align(fc2_routing, _WGRAD_CONTRACT_M)
    assert fc2_routing.wgrad_align.sorted_slot_ids is wgrad_align.sorted_slot_ids

    # Both block sizes describe the same set of routes, only padded differently.
    for slots in (routing.sorted_slot_ids, wgrad_align.sorted_slot_ids):
        assert int((slots < num_recv_tokens).sum().item()) == num_routes


def test_is_permute_free_grouped_gemm_enabled(monkeypatch):
    """Env gate: True only on ROCm with ``NVTE_PERMUTE_FREE_GROUPED_GEMM=1``."""
    monkeypatch.setenv("NVTE_PERMUTE_FREE_GROUPED_GEMM", "1")
    assert is_permute_free_grouped_gemm_enabled() is True
    monkeypatch.setenv("NVTE_PERMUTE_FREE_GROUPED_GEMM", "0")
    assert is_permute_free_grouped_gemm_enabled() is False
    monkeypatch.delenv("NVTE_PERMUTE_FREE_GROUPED_GEMM", raising=False)
    assert is_permute_free_grouped_gemm_enabled() is False


# ---------------------------------------------------------------------------
# Module level: GroupedLinear end-to-end and weight-gradient sinks
# ---------------------------------------------------------------------------
# FlyDSL tiles: hidden/ffn >= 128 and multiples of 64 (block_k).
_FC1_FC2_PIPELINE_SHAPES = [
    pytest.param(128, 128, 128, 8, 3, id="orig"),
    pytest.param(96, 128, 128, 6, 3, id="mid"),
    pytest.param(192, 192, 192, 8, 2, id="wide"),
    pytest.param(128, 128, 192, 4, 2, id="narrow-experts"),
    pytest.param(256, 256, 256, 16, 4, id="large"),
]


def _stack_expert_weights(mod, num_experts):
    """Per-expert module weights as one CPU fp32 ``[E, out, in]`` tensor."""
    return torch.stack(
        [getattr(mod, f"weight{i}").detach().cpu().float() for i in range(num_experts)]
    )


def _stack_expert_grads(mod, num_experts, *, to_cpu=False):
    """Per-expert weight gradients as one fp32 ``[E, out, in]`` tensor."""
    grads = [getattr(mod, f"weight{i}").grad for i in range(num_experts)]
    assert all(g is not None for g in grads), "a per-expert weight received no gradient"
    return torch.stack([g.detach().cpu().float() if to_cpu else g.detach().float() for g in grads])


def _moe_gated_reference(inp, probs, w1, w2, routing_map, ffn):
    """Token-expert MoE forward in fp32: FC1 -> silu-gated act (* prob) -> FC2 -> route sum.

    Returns ``(preact, out)``; ``preact`` is the raw ``[T, E, 2F]`` FC1 output with
    ``retain_grad`` set so the caller can read its gradient after ``out.backward()``.
    """
    preact = torch.einsum("th,enh->ten", inp, w1)
    preact.retain_grad()
    gate, up = preact[..., :ffn], preact[..., ffn:]
    act = torch.nn.functional.silu(gate) * up * probs[..., None]
    routed = torch.einsum("tef,ehf->teh", act, w2)
    return preact, (routed * routing_map[..., None]).sum(dim=1)


@pytest.mark.parametrize(
    "num_recv_tokens,hidden,ffn,num_experts,max_hits", _FC1_FC2_PIPELINE_SHAPES
)
def test_fc1_fc2_gated_pipeline(monkeypatch, num_recv_tokens, hidden, ffn, num_experts, max_hits):
    """End-to-end FC1 raw 2F -> FC2 gated activation via GroupedLinear.

    Each layer's forward output and backward gradients are checked against a CPU fp32 autograd
    reference. The reference runs on the CPU between the GPU forward and backward so no device
    allocation perturbs the heap layout across that boundary.
    """
    monkeypatch.setenv("NVTE_PERMUTE_FREE_GROUPED_GEMM", "1")
    routing_map = _random_routing_map(num_recv_tokens, num_experts, max_hits, "cuda")
    fc1_meta = PermuteFreeMetadata(
        routing_map=routing_map, num_experts=num_experts, activation="silu"
    )
    prepare_moe_align(fc1_meta, _FLYDSL_FWD_BLOCK_M)
    fc2_meta = dataclasses.replace(fc1_meta, route_space=True)
    em_max, valid, _, _ = _route_slots(fc1_meta, num_recv_tokens)
    slot_token = fc1_meta.sorted_slot_ids[valid].cpu().to(torch.int64)
    slot_expert = _expert_per_route(fc1_meta, em_max)[valid].cpu().to(torch.int64)
    m_splits = [num_recv_tokens // num_experts] * num_experts
    m_splits[-1] += num_recv_tokens - sum(m_splits)

    inp = torch.randn(
        num_recv_tokens, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    probs = torch.rand(
        num_recv_tokens, num_experts, device="cuda", dtype=torch.float32, requires_grad=True
    )
    fc1 = GroupedLinear(
        num_experts, hidden, 2 * ffn, bias=False, params_dtype=torch.bfloat16, device="cuda"
    )
    fc2 = GroupedLinear(
        num_experts, ffn, hidden, bias=False, params_dtype=torch.bfloat16, device="cuda"
    )
    with torch.no_grad():
        for i in range(num_experts):
            getattr(fc1, f"weight{i}").normal_(0, 0.05)
            getattr(fc2, f"weight{i}").normal_(0, 0.05)

    preact = fc1(inp, m_splits, permute_free_metadata=fc1_meta)
    preact.retain_grad()
    out = fc2(preact, m_splits, permute_free_metadata=fc2_meta, dispatched_probs=probs)

    inp_ref = inp.detach().cpu().float().requires_grad_(True)
    probs_ref = probs.detach().cpu().float().requires_grad_(True)
    w1_ref = _stack_expert_weights(fc1, num_experts).requires_grad_(True)
    w2_ref = _stack_expert_weights(fc2, num_experts).requires_grad_(True)
    preact_ref, out_ref = _moe_gated_reference(
        inp_ref, probs_ref, w1_ref, w2_ref, routing_map.cpu().float(), ffn
    )

    assert preact.shape[1] == 2 * ffn
    _assert_bf16_close(preact[valid].detach().cpu(), preact_ref.detach()[slot_token, slot_expert])
    _assert_bf16_close(out.detach().cpu(), out_ref.detach())

    out_ref.sum().backward()
    out.float().sum().backward()

    _assert_bf16_close(inp.grad.cpu(), inp_ref.grad)
    _assert_bf16_close(probs.grad.cpu(), probs_ref.grad)
    _assert_bf16_close(preact.grad[valid].cpu(), preact_ref.grad[slot_token, slot_expert])
    for mod, ref in ((fc1, w1_ref), (fc2, w2_ref)):
        wgrad = _stack_expert_grads(mod, num_experts, to_cpu=True)
        _assert_bf16_close(wgrad, ref.grad)


@pytest.mark.parametrize(
    "fuse_wgrad_accumulation",
    [pytest.param(True, id="fused"), pytest.param(False, id="nonfused")],
)
def test_grouped_weight_grad_matches_ungrouped(monkeypatch, fuse_wgrad_accumulation):
    """``single_grouped_weight`` wgrad matches the ungrouped per-expert ``.grad``.

    The permute-free backward writes the whole ``[E, out, in]`` wgrad to the grouped param
    directly (the GEMM only sees detached per-expert views), so only the sink differs:

    * fused (``fuse_wgrad_accumulation=True``): it lands in ``main_grad`` (fp32).
    * nonfused: it lands in the autograd ``.grad`` (bf16) and accumulates across backwards.
    """
    monkeypatch.setenv("NVTE_PERMUTE_FREE_GROUPED_GEMM", "1")
    # single_grouped_weight requires this gate, else the module falls back to per-expert params.
    monkeypatch.setenv("NVTE_GROUPED_LINEAR_SINGLE_PARAM", "1")

    # The fwd/dgrad tiles need both contraction dims (in for fwd, out for dgrad) >= 128 and a
    # multiple of block_k (64); use 128-wide features.
    num_recv_tokens, in_features, out_features = 128, 128, 128
    num_experts, max_hits = 8, 3
    grouped_shape = (num_experts, out_features, in_features)

    routing_map = _random_routing_map(num_recv_tokens, num_experts, max_hits, "cuda")
    num_routes = int(routing_map.sum().item())
    m_splits = [num_recv_tokens // num_experts] * num_experts

    # inp requires grad (as in real training, where it comes from a prior layer): for the
    # grouped path the detached per-expert views do not drive autograd, so the activation is
    # what makes the Function output require grad and triggers the backward.
    inp = torch.randn(
        num_recv_tokens, in_features, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    # Shared initial weights (so the kernel dW is identical) and upstream gradient.
    weights = torch.randn(grouped_shape, device="cuda", dtype=torch.bfloat16)
    grad_output = torch.randn(num_routes, out_features, device="cuda", dtype=torch.bfloat16)

    def run(mod, weight_views):
        with torch.no_grad():
            for view, init in zip(weight_views, weights):
                view.copy_(init)
        routing = PermuteFreeMetadata(routing_map=routing_map, num_experts=num_experts)
        out = mod(inp, m_splits, permute_free_metadata=routing)
        # Block-padded canonical: the output is [em_max, out] in padded slot order, and the
        # valid (expert-ascending) slots line up with the dense upstream grad. Slicing
        # [:num_routes] would cut across the padding gaps, so gather the valid slots instead.
        valid = routing.sorted_slot_ids < num_recv_tokens
        (out[valid].float() * grad_output.float()).sum().backward()
        return routing, valid

    def make(*, single_grouped_weight, fuse):
        return GroupedLinear(
            num_experts,
            in_features,
            out_features,
            bias=False,
            params_dtype=torch.bfloat16,
            device="cuda",
            fuse_wgrad_accumulation=fuse,
            single_grouped_weight=single_grouped_weight,
        )

    # --- ungrouped reference: separate per-expert params, wgrad via autograd .grad ---
    ungrouped_mod = make(single_grouped_weight=False, fuse=False)
    run(ungrouped_mod, [getattr(ungrouped_mod, f"weight{i}") for i in range(num_experts)])
    grad_ref = _stack_expert_grads(ungrouped_mod, num_experts)

    # --- grouped: single param, wgrad sink selected by fuse_wgrad_accumulation ---
    grouped_mod = make(single_grouped_weight=True, fuse=fuse_wgrad_accumulation)
    grouped_weight = grouped_mod.weight
    if fuse_wgrad_accumulation:
        grouped_weight.main_grad = torch.zeros(grouped_shape, device="cuda", dtype=torch.float32)
        grouped_weight.grad_added_to_main_grad = False
    _, valid = run(grouped_mod, list(grouped_mod._get_weight_tensors()))

    if fuse_wgrad_accumulation:
        assert grouped_weight.grad is None
        assert getattr(grouped_weight, "grad_added_to_main_grad", False) is True
        grouped_grad = grouped_weight.main_grad.view(grouped_shape).float()
    else:
        assert getattr(grouped_weight, "grad_added_to_main_grad", False) is False
        assert tuple(grouped_weight.grad.shape) == grouped_shape
        grouped_grad = grouped_weight.grad.view(grouped_shape).float()

    # Same kernel dW on both sides, so the only difference is the sink dtype (exact for the bf16
    # .grad, one bf16 rounding of the reference for the fp32 main_grad): no absolute floor needed.
    assert_close(grouped_grad, grad_ref, **dtype_tols(torch.bfloat16))
    for e in range(num_experts):
        assert grouped_grad[e].abs().sum() > 0, f"expert {e} has a zero grad (frozen)."

    if not fuse_wgrad_accumulation:
        # A second backward without zero_grad accumulates in place (grad-accumulation semantics).
        routing = PermuteFreeMetadata(routing_map=routing_map, num_experts=num_experts)
        out = grouped_mod(inp, m_splits, permute_free_metadata=routing)
        (out[valid].float() * grad_output.float()).sum().backward()
        accumulated = grouped_weight.grad.view(grouped_shape).float()
        assert_close(accumulated, 2.0 * grad_ref, **dtype_tols(torch.bfloat16))
