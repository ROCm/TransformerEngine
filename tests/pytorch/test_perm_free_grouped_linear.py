# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for the route-list permute-free bf16 MoE gather-GEMM kernels."""

import pytest
import torch
from torch.utils.cpp_extension import IS_HIP_EXTENSION

from transformer_engine.pytorch import GroupedLinear
from transformer_engine.pytorch.moe_routing import (
    MoERoutingMetadata,
    PermuteFreeMetadata,
)
from transformer_engine.pytorch.triton_kernels.permute_free_grouped_gemm import (
    permute_free_grouped_gemm_bf16,
    permute_free_grouped_gemm_bf16_dgrad,
    permute_free_grouped_gemm_bf16_wgrad,
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
    # Output is worst-case padded to a static upper bound; only the compact route range
    # [0, num_routes) is valid. The tail [num_routes, em_max) is inert, uninitialized padding
    # (the buffer is torch.empty and the kernel never writes it), so it is not checked here.
    assert out_full.shape[0] >= num_routes
    assert out_full.shape[1] == out_features
    out = out_full[:num_routes]
    ref = torch.einsum(
        "rk,rnk->rn",
        hidden[route_to_token].float(),
        weights[route_expert].float(),
    )
    assert _rel_l2(out, ref) < 2e-2


def test_route_list_dgrad():
    torch.manual_seed(11)
    num_recv_tokens, in_features, out_features = 128, 96, 128
    num_experts, max_hits = 8, 4

    routing_map = _random_routing_map(num_recv_tokens, num_experts, max_hits, "cuda", seed=2)
    routing = MoERoutingMetadata(routing_map=routing_map, num_experts=num_experts)
    route_to_token, route_expert = _compact_route_order(routing_map)
    num_routes = route_to_token.numel()

    weights = torch.randn(
        num_experts, out_features, in_features, device="cuda", dtype=torch.bfloat16
    )
    grad = torch.randn(num_routes, out_features, device="cuda", dtype=torch.bfloat16)

    dA = permute_free_grouped_gemm_bf16_dgrad(grad, weights, routing)
    assert dA.shape == (num_recv_tokens, in_features)

    # dA[t] = sum_{routes r with token==t} grad[r] @ W[e_r]
    per_route = torch.einsum("rn,rnk->rk", grad.float(), weights[route_expert].float())
    ref = torch.zeros(num_recv_tokens, in_features, device="cuda", dtype=torch.float32)
    ref.index_add_(0, route_to_token, per_route)
    assert _rel_l2(dA, ref) < 2e-2


def test_route_list_wgrad():
    torch.manual_seed(17)
    num_recv_tokens, in_features, out_features = 128, 96, 128
    num_experts, max_hits = 8, 4

    routing_map = _random_routing_map(num_recv_tokens, num_experts, max_hits, "cuda", seed=3)
    routing = MoERoutingMetadata(routing_map=routing_map, num_experts=num_experts)
    route_to_token, route_expert = _compact_route_order(routing_map)
    num_routes = route_to_token.numel()

    hidden = torch.randn(num_recv_tokens, in_features, device="cuda", dtype=torch.bfloat16)
    weights_shape = (num_experts, out_features, in_features)
    grad = torch.randn(num_routes, out_features, device="cuda", dtype=torch.bfloat16)

    dW = permute_free_grouped_gemm_bf16_wgrad(hidden, grad, weights_shape, routing)
    assert dW.shape == weights_shape

    # dW[e] = sum_{routes r with expert==e} outer(grad[r], hidden[token_r])
    ref = torch.zeros(num_experts, out_features, in_features, device="cuda", dtype=torch.float32)
    per_route = torch.einsum(
        "rn,rk->rnk", grad.float(), hidden[route_to_token].float()
    )
    ref.index_add_(0, route_expert, per_route)
    assert _rel_l2(dW, ref) < 2e-2


def test_route_list_fwd_dgrad_wgrad_consistency():
    """fwd + dgrad + wgrad against a single autograd reference in compact route space."""
    torch.manual_seed(23)
    num_recv_tokens, in_features, out_features = 96, 64, 96
    num_experts, max_hits = 6, 3

    routing_map = _random_routing_map(num_recv_tokens, num_experts, max_hits, "cuda", seed=5)
    routing = MoERoutingMetadata(routing_map=routing_map, num_experts=num_experts)
    route_to_token, route_expert = _compact_route_order(routing_map)
    num_routes = route_to_token.numel()

    hidden = torch.randn(num_recv_tokens, in_features, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(
        num_experts, out_features, in_features, device="cuda", dtype=torch.bfloat16
    )

    # Forward output is worst-case padded; compare only the compact valid route range.
    out = permute_free_grouped_gemm_bf16(hidden, weights, routing)[:num_routes]
    grad = torch.randn(num_routes, out_features, device="cuda", dtype=torch.bfloat16)
    dA = permute_free_grouped_gemm_bf16_dgrad(grad, weights, routing)
    dW = permute_free_grouped_gemm_bf16_wgrad(hidden, grad, weights.shape, routing)

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


def test_apply_route_probs_fwd_bwd():
    """Fused per-route prob apply (gather+multiply) vs an autograd advanced-index reference."""
    from transformer_engine.pytorch.triton_kernels.permute_free_grouped_gemm import (
        get_default_moe_kernel_config,
        prepare_moe_align,
    )
    from transformer_engine.pytorch.triton_kernels.route_prob import (
        _expert_per_route,
        apply_route_probs,
    )

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

    # Reference: differentiable advanced index over the compact route order.
    routes_max = int(routing.route_to_token.shape[0])
    tok = routing.route_to_token.to(torch.int64).clamp(0, num_recv_tokens - 1)
    exp = _expert_per_route(routing, routes_max).to(torch.int64)
    valid = torch.arange(routes_max, device="cuda") < num_routes
    a2 = act.clone().requires_grad_(True)
    p2 = probs.clone().requires_grad_(True)
    pr = torch.where(valid, p2[tok, exp], torch.zeros_like(p2[tok, exp]))
    pr = torch.nn.functional.pad(pr, (0, em_max - routes_max))
    ref = a2 * pr[:, None]

    assert _rel_l2(out[:num_routes], ref[:num_routes]) < 2e-2

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
    torch.manual_seed(41)

    num_recv_tokens, in_features, out_features = 128, 64, 96
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
    (out_a[:num_routes].float() * g.float()).sum().backward()
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
    (out_b[:num_routes].float() * g.float()).sum().backward()

    # The grouped param collects grad in main_grad, not in .grad.
    assert mod_b.weight.grad is None
    assert getattr(mod_b.weight, "grad_added_to_main_grad", False) is True
    main_grad = mod_b.weight.main_grad.view(num_experts, out_features, in_features).float()

    # Same kernel dW -> both sinks must match (near bit-identical: bf16 dW upcast to fp32).
    assert _rel_l2(main_grad, grad_a) < 1e-3
    # And every expert actually received a non-zero gradient (guards against frozen experts).
    for e in range(num_experts):
        assert main_grad[e].abs().sum() > 0, f"expert {e} has a zero main_grad (frozen)."


def test_grouped_weight_nonfused_grad_matches_ungrouped(monkeypatch):
    """single_grouped_weight without fuse_wgrad_accumulation routes the [E, out, in] wgrad into
    the grouped param's autograd ``.grad`` (the GEMM only sees detached per-expert views), and it
    must match the ungrouped per-expert ``.grad`` expert-for-expert and accumulate across
    backwards."""
    monkeypatch.setenv("NVTE_PERMUTE_FREE_GROUPED_GEMM", "1")
    torch.manual_seed(43)

    num_recv_tokens, in_features, out_features = 128, 64, 96
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
    (out_a[:num_routes].float() * g.float()).sum().backward()
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
    (out_b[:num_routes].float() * g.float()).sum().backward()

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
    (out_b2[:num_routes].float() * g.float()).sum().backward()
    grad_b2 = mod_b.weight.grad.view(num_experts, out_features, in_features).float()
    assert _rel_l2(grad_b2, 2.0 * grad_a) < 1e-3
