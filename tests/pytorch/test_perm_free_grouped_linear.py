# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for the route-list permute-free bf16 MoE gather-GEMM kernels."""

import pytest
import torch
from torch.utils.cpp_extension import IS_HIP_EXTENSION

from transformer_engine.pytorch import GroupedLinear
from transformer_engine.pytorch.moe import (
    MoERoutingMetadata,
    PermuteFreeMetadata,
    get_default_moe_kernel_config,
    permute_free_grouped_gemm_backward,
    permute_free_grouped_gemm_bf16,
    permute_free_grouped_gemm_bf16_dgrad,
    permute_free_grouped_gemm_bf16_fc2_wgrad,
    permute_free_grouped_gemm_bf16_wgrad,
    prepare_moe_align,
)
from transformer_engine.pytorch.moe.permute_free_grouped_gemm import (
    _FLYDSL_FWD_BLOCK_M,
    _expert_per_route,
)

pytestmark = pytest.mark.skipif(
    not (IS_HIP_EXTENSION and torch.cuda.is_available()),
    reason="Permute-free grouped GEMM tests require ROCm and CUDA device.",
)


@pytest.fixture(autouse=True)
def _cuda_sync_after_test():
    yield
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _rel_l2(actual: torch.Tensor, ref: torch.Tensor) -> float:
    return ((actual.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-8)).item()


def _random_routing_map(num_recv_tokens, num_experts, max_hits, device, seed):
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


def _compact_route_order(routing_map):
    """Expert-sorted (token, expert) route lists, matching ``moe_align_route_list``."""
    tok, exp = routing_map.nonzero(as_tuple=True)
    order = torch.argsort(exp, stable=True)
    return tok[order].to(torch.int64), exp[order].to(torch.int64)

# ---------------------------------------------------------------------------
# Route-list gather-GEMM: fwd / dgrad / wgrad
# ---------------------------------------------------------------------------
def test_route_list_fwd():
    torch.manual_seed(7)
    num_recv_tokens, in_features, out_features = 128, 128, 256
    num_experts, max_hits = 8, 3

    routing_map = _random_routing_map(num_recv_tokens, num_experts, max_hits, "cuda", seed=1)
    routing = MoERoutingMetadata(routing_map=routing_map, num_experts=num_experts)

    hidden = torch.randn(num_recv_tokens, in_features, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(
        num_experts, out_features, in_features, device="cuda", dtype=torch.bfloat16
    )

    out_full = permute_free_grouped_gemm_bf16(hidden, weights, routing)

    route_to_token, route_expert = _compact_route_order(routing_map)
    num_routes = route_to_token.numel()
    # Block-padded canonical layout: the output is [em_max, out] in expert-sorted, block-padded
    # slot order. Padded slot s holds hidden[sorted_slot_ids[s]] @ W[expert(s)]^T for the valid
    # slots (sorted_slot_ids[s] < num_recv_tokens); padding slots carry inert dead values and
    # are dropped only at the final padded->token gather-combine.
    em_max = int(routing.sorted_slot_ids.shape[0])
    assert out_full.shape == (em_max, out_features)
    slot_token = routing.sorted_slot_ids
    slot_expert = _expert_per_route(routing, em_max)
    valid = slot_token < num_recv_tokens
    tok = slot_token.to(torch.int64).clamp_(0, num_recv_tokens - 1)
    ref = torch.einsum(
        "rk,rnk->rn",
        hidden[tok].float(),
        weights[slot_expert.to(torch.int64)].float(),
    )
    assert int(valid.sum().item()) == num_routes
    assert _rel_l2(out_full[valid], ref[valid]) < 2e-2


def test_route_list_dgrad():
    torch.manual_seed(11)
    num_recv_tokens, in_features, out_features = 128, 96, 128
    num_experts, max_hits = 8, 4

    routing_map = _random_routing_map(num_recv_tokens, num_experts, max_hits, "cuda", seed=2)
    routing = MoERoutingMetadata(routing_map=routing_map, num_experts=num_experts)
    num_routes = int(routing_map.sum().item())
    # Block-padded canonical: the incoming route grad lives in the same [em_max] block-padded
    # slot order as the FC1 forward output (the FC2 dgrad hands it back in this layout). Prepare
    # the align at the forward block size so the standalone dgrad reuses the same slot layout.
    routing = prepare_moe_align(routing, _FLYDSL_FWD_BLOCK_M)
    em_max = int(routing.sorted_slot_ids.shape[0])
    slot_token = routing.sorted_slot_ids
    slot_expert = _expert_per_route(routing, em_max)
    valid = slot_token < num_recv_tokens

    weights = torch.randn(
        num_experts, out_features, in_features, device="cuda", dtype=torch.bfloat16
    )
    grad = torch.zeros(em_max, out_features, device="cuda", dtype=torch.bfloat16)
    grad[valid] = torch.randn(
        int(valid.sum().item()), out_features, device="cuda", dtype=torch.bfloat16
    )

    dA = permute_free_grouped_gemm_bf16_dgrad(grad, weights, routing)
    assert dA.shape == (num_recv_tokens, in_features)

    # dA[t] = sum_{padded slots s with token==t, valid} grad[s] @ W[expert(s)]
    per_slot = torch.einsum(
        "rn,rnk->rk", grad.float(), weights[slot_expert.to(torch.int64)].float()
    )
    per_slot = per_slot * valid[:, None].float()
    tok = slot_token.to(torch.int64).clamp_(0, num_recv_tokens - 1)
    ref = torch.zeros(num_recv_tokens, in_features, device="cuda", dtype=torch.float32)
    ref.index_add_(0, tok, per_slot)
    assert int(valid.sum().item()) == num_routes
    assert _rel_l2(dA, ref) < 2e-2


def test_route_list_wgrad():
    torch.manual_seed(17)
    num_recv_tokens, in_features, out_features = 128, 96, 128
    num_experts, max_hits = 8, 4

    routing_map = _random_routing_map(num_recv_tokens, num_experts, max_hits, "cuda", seed=3)
    routing = MoERoutingMetadata(routing_map=routing_map, num_experts=num_experts)
    num_routes = int(routing_map.sum().item())
    # Block-padded canonical: the incoming route grad lives in the [em_max] block-padded slot
    # order (same layout the forward writes / FC2 dgrad hands back). Prepare the forward align so
    # the wgrad reads grad at block_start[e]*block_size_m + within-rank.
    routing = prepare_moe_align(routing, _FLYDSL_FWD_BLOCK_M)
    em_max = int(routing.sorted_slot_ids.shape[0])
    slot_token = routing.sorted_slot_ids
    slot_expert = _expert_per_route(routing, em_max)
    valid = slot_token < num_recv_tokens

    hidden = torch.randn(num_recv_tokens, in_features, device="cuda", dtype=torch.bfloat16)
    weights_shape = (num_experts, out_features, in_features)
    grad = torch.zeros(em_max, out_features, device="cuda", dtype=torch.bfloat16)
    grad[valid] = torch.randn(
        int(valid.sum().item()), out_features, device="cuda", dtype=torch.bfloat16
    )

    dW = permute_free_grouped_gemm_bf16_wgrad(hidden, grad, weights_shape, routing)
    assert dW.shape == weights_shape

    # dW[e] = sum_{valid slots s with expert==e} outer(grad[s], hidden[slot_token[s]])
    tok = slot_token.to(torch.int64).clamp_(0, num_recv_tokens - 1)
    per_slot = torch.einsum("rn,rk->rnk", grad.float(), hidden[tok].float())
    per_slot = per_slot * valid[:, None, None].float()
    ref = torch.zeros(num_experts, out_features, in_features, device="cuda", dtype=torch.float32)
    # Past-end padding slots carry expert id -1; clamp for the scatter (their rows are zeroed).
    ref.index_add_(0, slot_expert.to(torch.int64).clamp_min(0), per_slot)
    assert int(valid.sum().item()) == num_routes
    assert _rel_l2(dW, ref) < 2e-2


def test_route_list_fc2_wgrad():
    """FC2 wgrad via operand-swap + transpose; ``out``/``accumulate`` fold in bf16 or fp32."""
    torch.manual_seed(19)
    num_recv_tokens, in_features, out_features = 128, 96, 128  # F=in, H=out (W2 is [E, H, F])
    num_experts, max_hits = 8, 4

    routing_map = _random_routing_map(num_recv_tokens, num_experts, max_hits, "cuda", seed=4)
    routing = MoERoutingMetadata(routing_map=routing_map, num_experts=num_experts)
    num_routes = int(routing_map.sum().item())
    # Block-padded canonical: fc2_input (the FC1 output) lives in the [em_max] block-padded slot
    # order; grad_output stays token-space and is gathered internally.
    routing = prepare_moe_align(routing, _FLYDSL_FWD_BLOCK_M)
    em_max = int(routing.sorted_slot_ids.shape[0])
    slot_token = routing.sorted_slot_ids
    slot_expert = _expert_per_route(routing, em_max)
    valid = slot_token < num_recv_tokens

    fc2_input = torch.zeros(em_max, in_features, device="cuda", dtype=torch.bfloat16)
    fc2_input[valid] = torch.randn(
        int(valid.sum().item()), in_features, device="cuda", dtype=torch.bfloat16
    )
    grad_output = torch.randn(num_recv_tokens, out_features, device="cuda", dtype=torch.bfloat16)
    weights_shape = (num_experts, out_features, in_features)  # [E, H, F]

    # dW2[e] = sum_{valid slots s with expert==e} outer(grad_output[slot_token[s]], fc2_input[s])
    tok = slot_token.to(torch.int64).clamp_(0, num_recv_tokens - 1)
    per_slot = torch.einsum("rh,rf->rhf", grad_output[tok].float(), fc2_input.float())
    per_slot = per_slot * valid[:, None, None].float()
    ref = torch.zeros(num_experts, out_features, in_features, device="cuda", dtype=torch.float32)
    # Past-end padding slots carry expert id -1; clamp for the scatter (their rows are zeroed).
    ref.index_add_(0, slot_expert.to(torch.int64).clamp_min(0), per_slot)

    # Fresh output (no transpose needed downstream).
    dW2 = permute_free_grouped_gemm_bf16_fc2_wgrad(fc2_input, grad_output, weights_shape, routing)
    assert dW2.shape == weights_shape
    assert _rel_l2(dW2, ref) < 2e-2

    # Direct fp32 accumulate into an existing buffer: overwrite then add == 2x.
    out = torch.zeros(weights_shape, device="cuda", dtype=torch.float32)
    permute_free_grouped_gemm_bf16_fc2_wgrad(
        fc2_input, grad_output, weights_shape, routing, out=out, accumulate=False
    )
    assert _rel_l2(out, ref) < 2e-2
    permute_free_grouped_gemm_bf16_fc2_wgrad(
        fc2_input, grad_output, weights_shape, routing, out=out, accumulate=True
    )
    assert _rel_l2(out, 2.0 * ref) < 2e-2


def test_route_list_fc2_wgrad_recompute_from_preact():
    """FC2 wgrad recompute-from-preact: rebuild act = act(gate)*up*prob from the saved 2F
    pre-activation into a transient buffer, feed the unchanged wgrad, and match a stored-act
    reference (the backward then checkpoints only the 2F preact, never the F-wide act)."""
    from transformer_engine.pytorch.moe import (
        get_default_moe_kernel_config,
        prepare_moe_align,
    )

    torch.manual_seed(29)
    num_recv_tokens, in_features, out_features = 128, 96, 128  # F=in, H=out (W2 is [E, H, F])
    num_experts, max_hits = 8, 4

    routing_map = _random_routing_map(num_recv_tokens, num_experts, max_hits, "cuda", seed=6)
    routing = MoERoutingMetadata(routing_map=routing_map, num_experts=num_experts)
    prepare_moe_align(routing, _FLYDSL_FWD_BLOCK_M)

    # Block-padded canonical: preact / act / grad all live in the [em_max] block-padded slot
    # order (padding slots have token sentinel >= num_recv_tokens and are dropped by the kernel).
    em_max = int(routing.sorted_slot_ids.shape[0])
    num_routes = int(routing_map.sum().item())
    slot_token = routing.sorted_slot_ids
    slot_expert = _expert_per_route(routing, em_max)
    valid = slot_token < num_recv_tokens
    tok = slot_token.to(torch.int64).clamp_(0, num_recv_tokens - 1)
    exp = slot_expert.to(torch.int64).clamp_min(0)

    preact = torch.zeros(em_max, 2 * in_features, device="cuda", dtype=torch.bfloat16)
    preact[valid] = torch.randn(
        int(valid.sum().item()), 2 * in_features, device="cuda", dtype=torch.bfloat16
    )
    grad_output = torch.randn(num_recv_tokens, out_features, device="cuda", dtype=torch.bfloat16)
    probs = torch.rand(num_recv_tokens, num_experts, device="cuda", dtype=torch.float32)
    weights_shape = (num_experts, out_features, in_features)  # [E, H, F]

    # Reference act in the block-padded slot layout using the kernel's own slot arrays.
    g = preact[:, :in_features].float()
    u = preact[:, in_features:].float()
    p = probs[tok, exp]
    act_ref = torch.nn.functional.silu(g) * u * p[:, None]
    act_ref = act_ref * valid[:, None].float()
    act_stored = act_ref.to(torch.bfloat16)

    # Stored-act baseline: feed the materialized activation directly.
    dW2_stored = permute_free_grouped_gemm_bf16_fc2_wgrad(
        act_stored, grad_output, weights_shape, routing
    )
    # Recompute path: feed preact + probs, rebuild act in-flight.
    dW2_rc = permute_free_grouped_gemm_bf16_fc2_wgrad(
        None, grad_output, weights_shape, routing,
        preact=preact, dispatched_probs=probs, activation="silu",
    )

    # fp32 reference dW2[e] = sum_{valid slots s: exp==e} outer(grad_output[tok_s], act_ref[s]).
    ref = torch.zeros(num_experts, out_features, in_features, device="cuda", dtype=torch.float32)
    per_route = torch.einsum("rh,rf->rhf", grad_output[tok].float(), act_ref)
    ref.index_add_(0, exp, per_route)

    assert _rel_l2(dW2_stored, ref) < 2e-2
    assert _rel_l2(dW2_rc, ref) < 2e-2
    # Recompute should also track the stored-act path closely (same GEMM, act rebuilt).
    assert _rel_l2(dW2_rc, dW2_stored) < 1e-2


def test_fc2_backward_dispatch_recompute_matches_stored():
    """FC2 backward dispatch: wgrad from the saved 2F preact (+ fc2_activation) matches the
    legacy stored-F activation path."""
    from transformer_engine.pytorch.moe import (
        get_default_moe_kernel_config,
        permute_free_grouped_gemm_backward,
        prepare_moe_align,
    )

    torch.manual_seed(31)
    num_recv_tokens, in_features, out_features = 128, 96, 128  # F=in, H=out (W2 is [E, H, F])
    num_experts, max_hits = 8, 4

    routing_map = _random_routing_map(num_recv_tokens, num_experts, max_hits, "cuda", seed=8)
    routing = PermuteFreeMetadata(
        routing_map=routing_map, num_experts=num_experts, route_space=True, activation="silu"
    )
    prepare_moe_align(routing, _FLYDSL_FWD_BLOCK_M)

    # Block-padded canonical: preact / act live in the [em_max] block-padded slot order.
    em_max = int(routing.sorted_slot_ids.shape[0])
    slot_token = routing.sorted_slot_ids
    slot_expert = _expert_per_route(routing, em_max)
    valid = slot_token < num_recv_tokens
    tok = slot_token.to(torch.int64).clamp_(0, num_recv_tokens - 1)
    exp = slot_expert.to(torch.int64).clamp_min(0)

    preact = torch.zeros(em_max, 2 * in_features, device="cuda", dtype=torch.bfloat16)
    preact[valid] = torch.randn(
        int(valid.sum().item()), 2 * in_features, device="cuda", dtype=torch.bfloat16
    )
    grad_output = torch.randn(num_recv_tokens, out_features, device="cuda", dtype=torch.bfloat16)
    probs = torch.rand(num_recv_tokens, num_experts, device="cuda", dtype=torch.float32)
    weights = [
        torch.randn(out_features, in_features, device="cuda", dtype=torch.bfloat16)
        for _ in range(num_experts)
    ]

    g = preact[:, :in_features].float()
    u = preact[:, in_features:].float()
    p = probs[tok, exp]
    act_stored = (torch.nn.functional.silu(g) * u * p[:, None] * valid[:, None].float()).to(
        torch.bfloat16
    )

    stored = permute_free_grouped_gemm_backward(
        grad_output,
        routing=routing,
        weights=weights,
        num_gemms=num_experts,
        hidden_states=act_stored,
        requires_wgrad=True,
    )
    recompute = permute_free_grouped_gemm_backward(
        grad_output,
        routing=routing,
        weights=weights,
        num_gemms=num_experts,
        hidden_states=preact,
        requires_wgrad=True,
        dispatched_probs=probs,
        fc2_activation="silu",
    )
    assert recompute.wgrad_stacked.shape == (num_experts, out_features, in_features)
    assert _rel_l2(recompute.wgrad_stacked, stored.wgrad_stacked) < 1e-2


def test_fc1_fc2_gated_pipeline(monkeypatch):
    """End-to-end FC1 raw 2F -> FC2 fused activation: forward outputs and backward grads match a
    PyTorch reference through the permute-free GroupedLinear modules."""
    import dataclasses

    from transformer_engine.pytorch.moe import (
        get_default_moe_kernel_config,
        prepare_moe_align,
    )

    monkeypatch.setenv("NVTE_PERMUTE_FREE_GROUPED_GEMM", "1")
    torch.manual_seed(37)
    # FFN width is a multiple of the FlyDSL block_k (64) so the fused FC2 gated_a prologue
    # runs on FlyDSL (the default backend) rather than falling back to Triton.
    num_recv_tokens, hidden, ffn = 128, 128, 128
    num_experts, max_hits = 8, 3

    routing_map = _random_routing_map(num_recv_tokens, num_experts, max_hits, "cuda", seed=9)
    fc1_meta = PermuteFreeMetadata(
        routing_map=routing_map, num_experts=num_experts, activation="silu"
    )
    # Prepare at the v3-compatible forward block_m (>= 128, multiple of 128); the token-count
    # default (64 at 128 tokens) is below the v3 gather/dgrad tile minimum. FC1 and FC2 share the
    # same block-padded slot layout, so the derived fc2_meta inherits this align.
    prepare_moe_align(fc1_meta, _FLYDSL_FWD_BLOCK_M)
    fc2_meta = dataclasses.replace(fc1_meta, route_space=True)
    num_routes = int(routing_map.sum().item())
    m_splits = [num_recv_tokens // num_experts] * num_experts
    m_splits[-1] += num_recv_tokens - sum(m_splits)

    inp = torch.randn(num_recv_tokens, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    probs = torch.rand(num_recv_tokens, num_experts, device="cuda", dtype=torch.float32, requires_grad=True)

    fc1 = GroupedLinear(
        num_experts, hidden, 2 * ffn, bias=False, params_dtype=torch.bfloat16, device="cuda"
    )
    fc2 = GroupedLinear(
        num_experts, ffn, hidden, bias=False, params_dtype=torch.bfloat16, device="cuda"
    )
    torch.manual_seed(38)
    with torch.no_grad():
        for i in range(num_experts):
            getattr(fc1, f"weight{i}").normal_(0, 0.05)
            getattr(fc2, f"weight{i}").normal_(0, 0.05)

    preact = fc1(inp, m_splits, permute_free_metadata=fc1_meta)
    assert preact.shape[1] == 2 * ffn
    preact.retain_grad()  # non-leaf: keep its grad so we can sanity-check the FC2->FC1 edge
    out = fc2(preact, m_splits, permute_free_metadata=fc2_meta, dispatched_probs=probs)

    # Reference: FC1 raw 2F [gate|up] -> silu(gate)*up*prob -> FC2, summed over each token's
    # experts (the fused FC2 prologue must reproduce this token-space output). Computed on the
    # CPU so no GPU (re)allocation happens between the forward and the backward -- the FlyDSL
    # FC2 dgrad autotune is sensitive to heap layout shifts (a latent OOB read faults only when
    # a large device alloc/free reshuffles the pool between fwd and bwd).
    with torch.no_grad():
        w1 = torch.stack([getattr(fc1, f"weight{i}").cpu() for i in range(num_experts)]).float()
        w2 = torch.stack([getattr(fc2, f"weight{i}").cpu() for i in range(num_experts)]).float()
        pre_ref = torch.einsum("th,enh->ten", inp.detach().cpu().float(), w1)  # [T, E, 2F]
        gate, up = pre_ref[..., :ffn], pre_ref[..., ffn:]
        act = torch.nn.functional.silu(gate) * up * probs.detach().cpu().float()[..., None]
        y = torch.einsum("tef,ehf->teh", act, w2)  # [T, E, H]
        ref = (y * routing_map.cpu().float()[..., None]).sum(dim=1)  # [T, H]
    rel = (out.detach().cpu().float() - ref).norm() / ref.norm().clamp_min(1e-8)
    assert rel < 3e-2, rel.item()

    loss = out.float().sum()
    loss.backward()

    assert inp.grad is not None
    assert probs.grad is not None
    for i in range(num_experts):
        assert getattr(fc1, f"weight{i}").grad is not None
        assert getattr(fc2, f"weight{i}").grad is not None
        assert getattr(fc2, f"weight{i}").grad.abs().sum() > 0
    # Sanity: compact route head got a non-zero FC1 output grad (FC2 act-bwd -> FC1).
    assert preact.grad[:num_routes].abs().sum() > 0

    # Numerical check on probs.grad (locks the gated-act backward's per-slot (token, expert)
    # mapping): with loss = out.sum(), dL/dprob[t, e] = routing_map[t, e] * <silu(gate)*up[t,e],
    # sum_h w2[e, h, :]>. A wrong expert-per-slot map (e.g. compact route index misread as a
    # block-padded slot) silently corrupts this even though probs.grad stays non-zero.
    with torch.no_grad():
        act_noprob = torch.nn.functional.silu(gate) * up  # [T, E, F] (CPU float)
        w2sum = w2.sum(dim=1)  # [E, F] = sum over output features
        dprob_ref = routing_map.cpu().float() * torch.einsum("tef,ef->te", act_noprob, w2sum)
    rel_p = (probs.grad.detach().cpu() - dprob_ref).norm() / dprob_ref.norm().clamp_min(1e-8)
    assert rel_p < 3e-2, rel_p.item()


def test_route_list_fwd_dgrad_wgrad_consistency():
    """fwd + dgrad + wgrad against a single autograd reference in compact route space."""
    torch.manual_seed(23)
    # The v3 gather/dgrad tiles need both contraction dims (in for fwd, out for dgrad) to be a
    # multiple of the FlyDSL block_k (64) and >= 128 (K_ITERS >= 2), so use 128-wide features.
    num_recv_tokens, in_features, out_features = 96, 128, 128
    num_experts, max_hits = 6, 3

    routing_map = _random_routing_map(num_recv_tokens, num_experts, max_hits, "cuda", seed=5)
    routing = MoERoutingMetadata(routing_map=routing_map, num_experts=num_experts)
    route_to_token, route_expert = _compact_route_order(routing_map)
    num_routes = route_to_token.numel()
    # Prepare the fwd/dgrad align (block-padded slot layout) up front so we can map the compact
    # per-route grad into padded-slot order for the block-padded dgrad.
    routing = prepare_moe_align(routing, _FLYDSL_FWD_BLOCK_M)
    em_max = int(routing.sorted_slot_ids.shape[0])
    valid = routing.sorted_slot_ids < num_recv_tokens
    assert int(valid.sum().item()) == num_routes

    hidden = torch.randn(num_recv_tokens, in_features, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(
        num_experts, out_features, in_features, device="cuda", dtype=torch.bfloat16
    )

    # Forward output is block-padded [em_max]; the valid padded slots are expert-ascending, the
    # same order as the compact route list, so out_full[valid] lines up with the compact ref.
    out_full = permute_free_grouped_gemm_bf16(hidden, weights, routing)
    out = out_full[valid]
    # One per-route gradient in the block-padded [em_max] slot layout that both the dgrad and the
    # wgrad now route-read (padded slot = block_start[e]*block_size_m + within-rank). The valid
    # slots are expert-ascending, matching the compact autograd reference order. ``grad`` keeps a
    # compact [num_routes] copy only to seed the compact reference backward.
    grad = torch.randn(num_routes, out_features, device="cuda", dtype=torch.bfloat16)
    grad_bp = torch.zeros(em_max, out_features, device="cuda", dtype=torch.bfloat16)
    grad_bp[valid] = grad
    dA = permute_free_grouped_gemm_bf16_dgrad(grad_bp, weights, routing)
    dW = permute_free_grouped_gemm_bf16_wgrad(hidden, grad_bp, weights.shape, routing)

    # Autograd reference in compact route space.
    ref_hidden = hidden.float().clone().requires_grad_(True)
    ref_w = weights.float().clone().requires_grad_(True)
    ref_out = torch.einsum(
        "rk,rnk->rn", ref_hidden[route_to_token], ref_w[route_expert]
    )
    assert _rel_l2(out, ref_out) < 2e-2
    ref_out.backward(grad.float())

    assert _rel_l2(dA, ref_hidden.grad) < 2e-2
    assert _rel_l2(dW, ref_w.grad) < 2e-2


@pytest.mark.skip(reason="apply_route_probs not ported to transformer_engine.pytorch.moe yet")
def test_apply_route_probs_fwd_bwd():
    """Fused per-route prob apply (gather+multiply) vs an autograd advanced-index reference."""
    from transformer_engine.pytorch.moe import (
        get_default_moe_kernel_config,
        prepare_moe_align,
    )
    # apply_route_probs was not ported; kept for when route_prob helper lands in moe/.
    _ = get_default_moe_kernel_config, prepare_moe_align

    torch.manual_seed(31)
    num_recv_tokens, hidden_dim = 128, 192
    num_experts, max_hits = 8, 3

    routing_map = _random_routing_map(num_recv_tokens, num_experts, max_hits, "cuda", seed=9)
    routing = MoERoutingMetadata(routing_map=routing_map, num_experts=num_experts)
    cfg = get_default_moe_kernel_config(num_recv_tokens)
    prepare_moe_align(routing, int(cfg["BLOCK_SIZE_M"]))

    em_max = int(routing.sorted_slot_ids.shape[0])
    num_routes = int(routing_map.sum().item())
    act = torch.randn(em_max, hidden_dim, device="cuda", dtype=torch.bfloat16)
    probs = torch.rand(num_recv_tokens, num_experts, device="cuda", dtype=torch.float32)

    a1 = act.clone().requires_grad_(True)
    p1 = probs.clone().requires_grad_(True)
    out = apply_route_probs(a1, p1, routing)

    # Reference: differentiable advanced index over the block-padded slot order.
    tok = routing.sorted_slot_ids.to(torch.int64).clamp(0, num_recv_tokens - 1)
    exp = _expert_per_route(routing, em_max).to(torch.int64).clamp_min(0)
    valid = routing.sorted_slot_ids < num_recv_tokens
    a2 = act.clone().requires_grad_(True)
    p2 = probs.clone().requires_grad_(True)
    pr = torch.where(valid, p2[tok, exp], torch.zeros_like(p2[tok, exp]))
    ref = a2 * pr[:, None]

    assert _rel_l2(out[valid], ref[valid]) < 2e-2

    g = torch.randn(em_max, hidden_dim, device="cuda", dtype=torch.bfloat16)
    g[num_routes:] = 0
    out.backward(g)
    ref.backward(g)
    assert _rel_l2(a1.grad[:num_routes], a2.grad[:num_routes]) < 2e-2
    assert _rel_l2(p1.grad, p2.grad) < 2e-2


# ---------------------------------------------------------------------------
# Module-level weight-gradient accumulation: grouped weight (-> main_grad) vs.
# separate per-expert weights (-> autograd .grad).
# ---------------------------------------------------------------------------
def _make_grouped_linear(num_gemms, in_features, out_features, *, grouped, fuse_wgrad):
    mod = GroupedLinear(
        num_gemms,
        in_features,
        out_features,
        bias=False,
        params_dtype=torch.bfloat16,
        device="cuda",
        fuse_wgrad_accumulation=fuse_wgrad,
        single_grouped_weight=grouped,
    )
    return mod


def test_grouped_weight_main_grad_matches_ungrouped_grad(monkeypatch):
    """Permute-free wgrad lands in the grouped param's ``main_grad`` and matches the
    ungrouped autograd ``.grad`` expert-for-expert (same kernel dW, different sink)."""
    monkeypatch.setenv("NVTE_PERMUTE_FREE_GROUPED_GEMM", "1")
    # single_grouped_weight requires this gate, else the module falls back to per-expert params.
    monkeypatch.setenv("NVTE_GROUPED_LINEAR_SINGLE_PARAM", "1")
    torch.manual_seed(41)

    # The v3 fwd/dgrad tiles need both contraction dims (in for fwd, out for dgrad) >= 128 and a
    # multiple of block_k (64); use 128-wide features.
    num_recv_tokens, in_features, out_features = 128, 128, 128
    num_experts, max_hits = 8, 3

    routing_map = _random_routing_map(num_recv_tokens, num_experts, max_hits, "cuda", seed=13)
    num_routes = int(routing_map.sum().item())
    m_splits = [num_recv_tokens // num_experts] * num_experts

    # inp requires grad (as in real training, where it comes from a prior layer): for the
    # grouped path the detached per-expert views do not drive autograd, so the activation is
    # what makes the Function output require grad and triggers the backward.
    inp = torch.randn(
        num_recv_tokens, in_features, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    # Shared initial weights for both modules (so the kernel dW is identical).
    W = torch.randn(num_experts, out_features, in_features, device="cuda", dtype=torch.bfloat16)
    # Shared upstream gradient over the compact route range.
    g = torch.randn(num_routes, out_features, device="cuda", dtype=torch.bfloat16)

    # --- ungrouped: separate per-expert params, wgrad via autograd .grad ---
    mod_a = _make_grouped_linear(
        num_experts, in_features, out_features, grouped=False, fuse_wgrad=False
    )
    with torch.no_grad():
        for i in range(num_experts):
            getattr(mod_a, f"weight{i}").copy_(W[i])
    routing_a = PermuteFreeMetadata(routing_map=routing_map, num_experts=num_experts)
    out_a = mod_a(inp, m_splits, permute_free_metadata=routing_a)
    # Block-padded canonical: the module output is [em_max, out] in padded slot order; the valid
    # (expert-ascending) slots line up with the compact upstream grad g. Slicing [:num_routes]
    # would cut across the padding gaps, so gather the valid slots instead.
    valid_a = routing_a.sorted_slot_ids < num_recv_tokens
    (out_a[valid_a].float() * g.float()).sum().backward()
    grad_a = torch.stack(
        [getattr(mod_a, f"weight{i}").grad.float() for i in range(num_experts)], dim=0
    )

    # --- grouped: single grouped param, wgrad accumulated into main_grad ---
    mod_b = _make_grouped_linear(
        num_experts, in_features, out_features, grouped=True, fuse_wgrad=True
    )
    with torch.no_grad():
        for i, view in enumerate(mod_b._get_weight_tensors()):
            view.copy_(W[i])
    mod_b.weight.main_grad = torch.zeros(
        num_experts, out_features, in_features, device="cuda", dtype=torch.float32
    )
    mod_b.weight.grad_added_to_main_grad = False
    routing_b = PermuteFreeMetadata(routing_map=routing_map, num_experts=num_experts)
    out_b = mod_b(inp, m_splits, permute_free_metadata=routing_b)
    valid_b = routing_b.sorted_slot_ids < num_recv_tokens
    (out_b[valid_b].float() * g.float()).sum().backward()

    # The grouped param collects grad in main_grad, not in .grad.
    assert mod_b.weight.grad is None
    assert getattr(mod_b.weight, "grad_added_to_main_grad", False) is True
    main_grad = mod_b.weight.main_grad.view(num_experts, out_features, in_features).float()

    # Both paths fold the same bf16 kernel dW into their sink (grouped -> fp32 main_grad,
    # ungrouped -> bf16 .grad), so they match to within bf16 rounding (~1e-3 relative).
    assert _rel_l2(main_grad, grad_a) < 5e-3
    # And every expert actually received a non-zero gradient (guards against frozen experts).
    for e in range(num_experts):
        assert main_grad[e].abs().sum() > 0, f"expert {e} has a zero main_grad (frozen)."


def test_grouped_weight_nonfused_grad_matches_ungrouped(monkeypatch):
    """single_grouped_weight without fuse_wgrad_accumulation routes the [E, out, in] wgrad into
    the grouped param's autograd ``.grad`` (the GEMM only sees detached per-expert views), and it
    must match the ungrouped per-expert ``.grad`` expert-for-expert and accumulate across
    backwards."""
    monkeypatch.setenv("NVTE_PERMUTE_FREE_GROUPED_GEMM", "1")
    # single_grouped_weight requires this gate, else the module falls back to per-expert params.
    monkeypatch.setenv("NVTE_GROUPED_LINEAR_SINGLE_PARAM", "1")
    torch.manual_seed(43)

    # 128-wide features to satisfy the v3 fwd/dgrad tile (K>=128, multiple of block_k=64).
    num_recv_tokens, in_features, out_features = 128, 128, 128
    num_experts, max_hits = 8, 3
    m_splits = [num_recv_tokens // num_experts] * num_experts

    routing_map = _random_routing_map(num_recv_tokens, num_experts, max_hits, "cuda", seed=17)
    num_routes = int(routing_map.sum().item())
    # inp drives autograd for the grouped path (detached views carry no grad edge).
    inp = torch.randn(
        num_recv_tokens, in_features, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    W = torch.randn(num_experts, out_features, in_features, device="cuda", dtype=torch.bfloat16)
    g = torch.randn(num_routes, out_features, device="cuda", dtype=torch.bfloat16)

    # --- ungrouped reference: separate per-expert params, wgrad via autograd .grad ---
    mod_a = _make_grouped_linear(
        num_experts, in_features, out_features, grouped=False, fuse_wgrad=False
    )
    with torch.no_grad():
        for i in range(num_experts):
            getattr(mod_a, f"weight{i}").copy_(W[i])
    routing_a = PermuteFreeMetadata(routing_map=routing_map, num_experts=num_experts)
    out_a = mod_a(inp, m_splits, permute_free_metadata=routing_a)
    # Block-padded canonical: gather the valid (expert-ascending) slots rather than slicing
    # [:num_routes], which would cut across the padding gaps.
    valid_a = routing_a.sorted_slot_ids < num_recv_tokens
    (out_a[valid_a].float() * g.float()).sum().backward()
    grad_a = torch.stack(
        [getattr(mod_a, f"weight{i}").grad.float() for i in range(num_experts)], dim=0
    )

    # --- grouped, no fusion: wgrad accumulates into the grouped param's .grad ---
    mod_b = _make_grouped_linear(
        num_experts, in_features, out_features, grouped=True, fuse_wgrad=False
    )
    with torch.no_grad():
        for i, view in enumerate(mod_b._get_weight_tensors()):
            view.copy_(W[i])
    routing_b = PermuteFreeMetadata(routing_map=routing_map, num_experts=num_experts)
    out_b = mod_b(inp, m_splits, permute_free_metadata=routing_b)
    valid_b = routing_b.sorted_slot_ids < num_recv_tokens
    (out_b[valid_b].float() * g.float()).sum().backward()

    # Grad lands in the grouped param's .grad (not main_grad), shape [E, out, in].
    assert getattr(mod_b.weight, "grad_added_to_main_grad", False) is False
    assert mod_b.weight.grad is not None
    assert tuple(mod_b.weight.grad.shape) == (num_experts, out_features, in_features)
    grad_b = mod_b.weight.grad.view(num_experts, out_features, in_features).float()
    assert _rel_l2(grad_b, grad_a) < 1e-3
    for e in range(num_experts):
        assert grad_b[e].abs().sum() > 0, f"expert {e} has a zero grad (frozen)."

    # A second backward without zero_grad accumulates in place (grad-accumulation semantics).
    out_b2 = mod_b(inp, m_splits, permute_free_metadata=routing_b)
    (out_b2[valid_b].float() * g.float()).sum().backward()
    grad_b2 = mod_b.weight.grad.view(num_experts, out_features, in_features).float()
    assert _rel_l2(grad_b2, 2.0 * grad_a) < 1e-3
