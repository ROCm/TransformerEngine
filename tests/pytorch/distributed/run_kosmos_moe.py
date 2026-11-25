#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information
"""Eight-rank checks of transformer_engine.pytorch.kosmos (KOSMOS fused expert-parallel MoE).

    torchrun --nproc_per_node=8 run_kosmos_moe.py --test all [--shape small] [--json out.jsonl]

Every check is built to be able to fail: numerics are compared with an independent fp32 reference
on relative L2 *and* a magnitude ratio (cosine and relative L2 alone cannot see a pure scale
error), and mutations (weights in the wrong layout, a saved buffer zeroed between forward and
backward) must change what they feed. The multi-call checks exercise the orders a training loop
produces and compare each call bit for bit with the same call run on its own. The graph_* checks
capture the op's forward and backward with torch.cuda.graph and compare every replay bit for bit
with the same calls run eagerly; graph_first_call_in_capture makes a configuration's first call
inside capture, which must fail with the op's own error on every rank.

Every rank runs every check in the same order; a check never raises on one rank alone, because the
other ranks would wait in the next KOSMOS kernel forever. Results are gathered and judged on all
ranks alike. The exit code is 0 only if every requested check passed.
"""

import argparse
import gc
import json
import math
import os
import sys
import time
import traceback

import torch
import torch.distributed as dist

from transformer_engine.pytorch import kosmos
from transformer_engine.pytorch.kosmos import functional as kf
from transformer_engine.pytorch.kosmos import _instance as ki

SHAPES = {
    # small enough for an fp32 reference with every expert on every rank
    "small": dict(T=2048, H=2048, I=1024, E_loc=8, k=8),
    # I < 1024: KOSMOS pads each (source, expert) segment differently and sizes its crews
    # differently
    "small_i512": dict(T=1024, H=1024, I=512, E_loc=4, k=4),
    # muse's experts (E 256, top-8) at a quarter of its tokens and half its H and I
    "muse_small": dict(T=2048, H=2048, I=2048, E_loc=32, k=8),
    # muse itself; too big for the fp32 reference, so only the checks that compare the op with
    # itself (determinism, the graph_* checks)
    "muse": dict(T=8192, H=4096, I=4096, E_loc=32, k=8),
}
REL_L2_MAX = 1e-2
SCALE_TOL = 1e-2
NAMES = ("y", "dx", "dtw", "dw1", "dw2")
# The op's save_for_backward order (functional._KosmosMoeFn.forward): x, topk_weights, w1, w2,
# topk_idx (int32), then the saved buffers.
SAVED_INDEX = {"xp": 5, "g": 6, "a": 7, "y": 8, "route": 9}


class Env:
    def __init__(self, shape):
        self.group = dist.group.WORLD
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        self.dev = torch.device("cuda", torch.cuda.current_device())
        for key, val in SHAPES[shape].items():
            setattr(self, key, val)
        self.E = self.E_loc * self.world


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #
def _gen(env, seed):
    return torch.Generator(device=env.dev).manual_seed(seed * 1000 + env.rank)


def make_weights(env, seed):
    """This rank's experts, W1 in [gate; up] order, scaled to keep activations O(1)."""
    g = _gen(env, seed)
    w1 = torch.randn(env.E_loc, 2 * env.I, env.H, generator=g, device=env.dev) * (
        2.0 / math.sqrt(env.H)
    )
    w2 = torch.randn(env.E_loc, env.H, env.I, generator=g, device=env.dev) * (
        2.0 / math.sqrt(env.I)
    )
    return w1.bfloat16(), w2.bfloat16()


def make_tokens(env, seed, skew=0, exclude_expert=None, dropped=0.0, hot_expert=None, tokens=None):
    """x [T, H] bf16, topk_idx [T, k] int64, topk_w [T, k] fp32, dy [T, H] bf16."""
    T = env.T if tokens is None else tokens
    g = _gen(env, seed + 7)
    x = torch.randn(T, env.H, generator=g, device=env.dev).bfloat16()
    if skew:
        # KOSMOS's skewed draw: Gumbel noise, rank 0's experts biased by log(2 skew + 1)
        u = torch.rand(T, env.E, generator=g, device=env.dev).clamp(1e-7, 1 - 1e-7)
        score = -torch.log(-torch.log(u))
        score[:, : env.E_loc] += math.log(2 * skew + 1)
    else:
        score = torch.randn(T, env.E, generator=g, device=env.dev)
    if exclude_expert is not None:
        score[:, exclude_expert] = -float("inf")
    if hot_expert is not None:
        score[:, hot_expert] = float("inf")
    top, idx = torch.topk(score, env.k, dim=-1)
    w = torch.softmax(top.float(), dim=-1)
    if hot_expert is not None:
        w = torch.rand(T, env.k, generator=g, device=env.dev) + 0.1
    if dropped:
        drop = torch.rand(T, env.k, generator=g, device=env.dev) < dropped
        drop[:64] = True  # padding tokens: every route dropped
        idx = idx.masked_fill(drop, -1)
    dy = (torch.randn(T, env.H, generator=g, device=env.dev) / 8).bfloat16()
    return x, idx.to(torch.int64), w.float().contiguous(), dy


# --------------------------------------------------------------------------- #
# the op, and the fp32 reference
# --------------------------------------------------------------------------- #
def forward(env, x, idx, w, w1k, w2, **kw):
    leaves = [t.detach().clone().requires_grad_(True) for t in (x, w, w1k, w2)]
    y = kosmos.kosmos_moe(env.group, leaves[0], idx, leaves[1], leaves[2], leaves[3], **kw)
    return y, leaves


def backward(y, dy, leaves):
    y.backward(dy)
    torch.cuda.synchronize()
    return {
        "y": y.detach(),
        "dx": leaves[0].grad,
        "dtw": leaves[1].grad,
        "dw1": leaves[2].grad,
        "dw2": leaves[3].grad,
    }


def run_layer(env, x, idx, w, w1k, w2, dy, **kw):
    y, leaves = forward(env, x, idx, w, w1k, w2, **kw)
    return backward(y, dy, leaves)


def saved_buffers(y):
    """The saved buffers of the op call that produced ``y`` (without unpack hooks: the tensors
    autograd holds)."""
    st = y.grad_fn.saved_tensors
    return {name: st[i] for name, i in SAVED_INDEX.items()}


def saved_nbytes(y):
    return sum(t.numel() * t.element_size() for t in saved_buffers(y).values() if t is not None)


def reference(env, x, idx, w, w1_gu, w2, dy):
    """fp32, per (token, k) route, sharing no grouping, padding or layout with KOSMOS."""
    g1 = [torch.empty_like(w1_gu) for _ in range(env.world)]
    dist.all_gather(g1, w1_gu.contiguous(), group=env.group)
    g2 = [torch.empty_like(w2) for _ in range(env.world)]
    dist.all_gather(g2, w2.contiguous(), group=env.group)
    w1g = torch.cat(g1).float().requires_grad_(True)
    w2g = torch.cat(g2).float().requires_grad_(True)
    del g1, g2
    xf = x.float().requires_grad_(True)
    wf = w.clone().requires_grad_(True)
    out = torch.zeros(x.shape[0], env.H, device=env.dev)
    for j in range(env.k):
        col = idx[:, j]
        for e in torch.unique(col).tolist():
            if e < 0 or e >= env.E:
                continue
            sel = (col == e).nonzero(as_tuple=True)[0]
            h = xf[sel] @ w1g[e].t()
            gate, up = h.chunk(2, dim=-1)
            act = torch.nn.functional.silu(gate) * up
            out = out.index_add(0, sel, wf[sel, j : j + 1] * (act @ w2g[e].t()))
    out.backward(dy.float())
    dw1g = w1g.grad
    dw2g = w2g.grad
    dist.all_reduce(dw1g, group=env.group)
    dist.all_reduce(dw2g, group=env.group)
    mine = slice(env.rank * env.E_loc, (env.rank + 1) * env.E_loc)
    res = {
        "y": out.detach(),
        "dx": xf.grad,
        "dtw": wf.grad,
        "dw1": kosmos.to_kosmos_w1(dw1g[mine]),
        "dw2": dw2g[mine].clone(),
    }
    del w1g, w2g, dw1g, dw2g
    torch.cuda.empty_cache()
    return res


def metrics(a, r):
    a = a.float().flatten()
    r = r.float().flatten()
    finite = bool(torch.isfinite(a).all())
    if not finite:
        return {"finite": False, "rel_l2": float("inf"), "scale": float("nan")}
    rr = float(r @ r)
    if rr == 0.0:
        # an all-zero reference: the result must be exactly zero
        return {
            "finite": True,
            "rel_l2": 0.0 if float(a.abs().max()) == 0.0 else float("inf"),
            "scale": 1.0,
        }
    return {
        "finite": True,
        "rel_l2": float((a - r).norm() / math.sqrt(rr)),
        "scale": float(a @ r) / rr,
    }


def close(m):
    return m["finite"] and m["rel_l2"] < REL_L2_MAX and abs(m["scale"] - 1.0) < SCALE_TOL


def numerics(res, ref):
    """Per tensor: metrics, and whether they are within the bounds."""
    out = {}
    for n in NAMES:
        m = metrics(res[n], ref[n])
        m["ok"] = close(m)
        out[n] = m
    return out


def bitwise(a, b):
    return {n: bool(torch.equal(a[n], b[n])) for n in NAMES}


def allocated():
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated()


# --------------------------------------------------------------------------- #
# checks: each returns (passed, details) computed alike on every rank
# --------------------------------------------------------------------------- #
class Data:
    """Inputs and solo results shared by several checks (computed once)."""

    def __init__(self, env):
        self.env = env
        self.w1, self.w2 = make_weights(env, 1)
        self.w1k = kosmos.to_kosmos_w1(self.w1)
        self.tok = {s: make_tokens(env, s) for s in (11, 12, 13)}
        self._solo = {}
        self._ref = {}

    def solo(self, s):
        if s not in self._solo:
            x, idx, w, dy = self.tok[s]
            self._solo[s] = run_layer(self.env, x, idx, w, self.w1k, self.w2, dy)
        return self._solo[s]

    def ref(self, s):
        if s not in self._ref:
            x, idx, w, dy = self.tok[s]
            self._ref[s] = reference(self.env, x, idx, w, self.w1, self.w2, dy)
        return self._ref[s]


def _status(env):
    """Status since the last query, without raising: a check judges the bits itself, alike on
    every rank."""
    return kf.status(env.group, raise_on=())


def _errors(st):
    return st & (kf.STATUS_BAD_ID | kf.STATUS_DUPLICATE | kf.STATUS_ERROR)


def _all_ok(num):
    return all(m["ok"] for m in num.values())


def _default_rows(env):
    return kf._default_capacity_rows(env.T, env.k, env.E_loc)


def _private_instance(env, capacity_rows, tokens=None, allocator=None):
    """An instance of its own (the capacity is part of the key)."""
    return kf._INSTANCES.get(
        env.group,
        tokens=env.T if tokens is None else tokens,
        hidden=env.H,
        inter=env.I,
        local_experts=env.E_loc,
        topk=env.k,
        capacity_rows=capacity_rows,
        wgcap=256,
        allocator=allocator,
    )


def _main_instance(env):
    return _private_instance(env, None)


def check_numerics(env, d):
    """Expected: every tensor within rel L2 1e-2 and scale 1 +- 1e-2; status 0."""
    num = numerics(d.solo(11), d.ref(11))
    st = _status(env)
    return _all_ok(num) and st == 0, {"numerics": num, "status": st}


def check_mutation(env, d):
    """Weights handed over in [gate; up] order, unconverted: the numerics comparison must fail."""
    x, idx, w, dy = d.tok[11]
    res = run_layer(env, x, idx, w, d.w1, d.w2, dy)
    num = numerics(res, d.ref(11))
    _status(env)
    return not num["y"]["ok"], {"numerics_of_mutant": num}


def check_determinism(env, d):
    """Expected: two identical training calls and a no-grad call agree bit for bit."""
    a = d.solo(11)
    x, idx, w, dy = d.tok[11]
    b = run_layer(env, x, idx, w, d.w1k, d.w2, dy)
    with torch.no_grad():
        y = kosmos.kosmos_moe(env.group, x, idx, w, d.w1k, d.w2)
    torch.cuda.synchronize()
    eq = bitwise(a, b)
    eq["y_no_grad"] = bool(torch.equal(a["y"], y))
    return all(eq.values()), {"bitwise": eq, "status": _status(env)}


def check_two_routings(env, d):
    """Expected: routing B right after routing A is correct, and A again reproduces A bit for
    bit."""
    a1 = d.solo(11)
    b = d.solo(12)
    x, idx, w, dy = d.tok[11]
    a2 = run_layer(env, x, idx, w, d.w1k, d.w2, dy)
    num = numerics(b, d.ref(12))
    eq = bitwise(a1, a2)
    return _all_ok(num) and all(eq.values()), {"numerics_b": num, "a_again_bitwise": eq}


def _stored_forwards(env, d, seeds, order):
    """Forwards of `seeds` (each one layer call kept for its backward), then backwards in
    `order`."""
    pending = {}
    for s in seeds:
        x, idx, w, _ = d.tok[s]
        pending[s] = forward(env, x, idx, w, d.w1k, d.w2)
    out = {}
    for s in order:
        y, leaves = pending.pop(s)
        out[s] = backward(y, d.tok[s][3], leaves)
    return out


def _multi_call(env, d, seeds, order, steps=2):
    """Every step: status 0, every call within the reference bounds and bit for bit equal to the
    same call run on its own; the caching allocator holds the same bytes after every step."""
    for s in seeds:
        d.solo(s)
        d.ref(s)
    _status(env)
    details = {}
    ok = True
    mem = []
    for step in range(steps):
        out = _stored_forwards(env, d, seeds, order)
        st = _status(env)
        num = {s: numerics(out[s], d.ref(s)) for s in seeds}
        eq = {s: bitwise(out[s], d.solo(s)) for s in seeds}
        details[f"step{step + 1}"] = {"status": st, "numerics": num, "bitwise_vs_solo": eq}
        ok &= all(_all_ok(n) for n in num.values()) and st == 0
        ok &= all(all(e.values()) for e in eq.values())
        del out
        mem.append(allocated())
    details["allocated_after_step"] = mem
    ok &= len(set(mem)) == 1
    return ok, details


def check_three_layers(env, d):
    """Three layer calls, all forwards, then the backwards in reverse."""
    return _multi_call(env, d, (11, 12, 13), (13, 12, 11))


def check_inflight_fifo(env, d):
    """Two in-flight forwards of one layer (pipeline parallelism), backwards first-in first-out."""
    return _multi_call(env, d, (11, 12), (11, 12))


def check_inflight_lifo(env, d):
    """Two in-flight forwards of one layer, backwards last-in first-out."""
    return _multi_call(env, d, (11, 12), (12, 11))


def check_nograd_between(env, d):
    """A no-grad forward between a forward and its backward (whole-MoE recompute's first pass, or an
    evaluation): it must not disturb the stored forward."""
    x, idx, w, dy = d.tok[11]
    y, leaves = forward(env, x, idx, w, d.w1k, d.w2)
    xb, idxb, wb, _ = d.tok[12]
    with torch.no_grad():
        yb = kosmos.kosmos_moe(env.group, xb, idxb, wb, d.w1k, d.w2)
    a = backward(y, dy, leaves)
    eq = bitwise(a, d.solo(11))
    eq["no_grad_y"] = bool(torch.equal(yb, d.solo(12)["y"]))
    st = _status(env)
    return all(eq.values()) and st == 0, {"bitwise": eq, "status": st}


def check_forward_without_backward(env, d):
    """A training forward whose backward never runs, then training calls. Expected: dropping it
    returns the caching allocator to the bytes it held before the forward (its saved state was
    autograd's); the calls after it are within the bounds and bit for bit equal to their solo
    runs; status 0. Nothing to reclaim."""
    for s in (11, 12, 13):
        d.solo(s)
    _main_instance(env)
    _status(env)
    x, idx, w, dy = d.tok[11]
    before = allocated()
    y, leaves = forward(env, x, idx, w, d.w1k, d.w2)
    held = allocated() - before
    del y, leaves
    after = allocated()
    eq = {}
    num = {}
    for i, s in enumerate((12, 13, 11)):
        xs, idxs, ws, dys = d.tok[s]
        res = run_layer(env, xs, idxs, ws, d.w1k, d.w2, dys)
        eq[f"call{i}_{s}"] = bitwise(res, d.solo(s))
        num[f"call{i}_{s}"] = numerics(res, d.ref(s))
    st = _status(env)
    ok = after == before and held > 0 and st == 0
    ok &= all(all(e.values()) for e in eq.values()) and all(_all_ok(n) for n in num.values())
    return ok, {
        "allocated_before": before,
        "held_by_forward": held,
        "allocated_after_drop": after,
        "bitwise": eq,
        "numerics": num,
        "status": st,
    }


def check_save_tiers(env, d):
    """Each save tier forced. Expected: y, dx, dtopk_w, dW1, dW2 bit for bit equal to tier all; the
    saved bytes of a call are what the instance says for its tier, and all > xp > none; no error
    status bit."""
    inst = _main_instance(env)
    x, idx, w, dy = d.tok[11]
    full = d.solo(11)
    _status(env)
    details = {}
    ok = True
    sizes = {}
    for save in ("all", "xp", "none"):
        y, leaves = forward(env, x, idx, w, d.w1k, d.w2, save=save)
        sizes[save] = saved_nbytes(y)
        res = backward(y, dy, leaves)
        st = _status(env)
        eq = bitwise(res, full)
        want = inst.saved_bytes_per_call(inst.capacity_rows)[save]
        details[save] = {"bitwise_vs_all": eq, "status": st, "saved_bytes": sizes[save]}
        details[save]["saved_bytes_expected"] = want
        ok &= all(eq.values()) and not _errors(st) and sizes[save] == want
    ok &= sizes["all"] > sizes["xp"] > sizes["none"] > 0
    return ok, details


def check_saved_rows_below(env, d):
    """Saved rows (256) below the rows a call receives. Expected: the forward keeps nothing (status
    bit 32), outputs and gradients bit for bit equal to the full save, smaller saved buffers; no
    error bit."""
    inst = _main_instance(env)
    x, idx, w, dy = d.tok[11]
    full = d.solo(11)
    _status(env)
    y, leaves = forward(env, x, idx, w, d.w1k, d.w2, saved_rows=256)
    small = saved_nbytes(y)
    res = backward(y, dy, leaves)
    st = _status(env)
    eq = bitwise(res, full)
    big = inst.saved_bytes_per_call(inst.capacity_rows)["all"]
    fell_back = bool(st & kf.STATUS_KEPT_NOTHING)
    ok = all(eq.values()) and fell_back and not _errors(st) and small < big
    return ok, {"bitwise_vs_full": eq, "status": st, "saved_bytes": small, "full_bytes": big}


def _mutated_backward(env, d, name):
    """Forward (tier all), zero saved buffer `name` behind autograd's back, backward."""
    x, idx, w, dy = d.tok[11]
    y, leaves = forward(env, x, idx, w, d.w1k, d.w2)
    buf = saved_buffers(y)[name]
    # .data: a write autograd's version counter does not see, i.e. a silently corrupted buffer
    buf.data.zero_()
    return backward(y, dy, leaves)


def check_mutation_saved_xp(env, d):
    """Zero saved.xp between forward and backward. Expected: dW1 changes (the backward reads the
    caller's copy of the dispatched rows, not instance memory)."""
    res = _mutated_backward(env, d, "xp")
    _status(env)
    eq = bitwise(res, d.solo(11))
    return not eq["dw1"], {"bitwise_vs_unmutated": eq}


def check_mutation_saved_y(env, d):
    """Zero saved.y between forward and backward. Expected: dtopk_w changes (it is <dy, Y> per
    route, read from the caller's copy of Y)."""
    res = _mutated_backward(env, d, "y")
    _status(env)
    eq = bitwise(res, d.solo(11))
    return not eq["dtw"], {"bitwise_vs_unmutated": eq}


def check_inference_allocates_nothing(env, d):
    """A no-grad forward. Expected: the caching allocator's peak during the call exceeds what it
    held before by at most y and the int32 copy of topk_idx (nothing saved)."""
    _main_instance(env)
    x, idx, w, _ = d.tok[11]
    with torch.no_grad():
        kosmos.kosmos_moe(env.group, x, idx, w, d.w1k, d.w2)
        before = allocated()
        torch.cuda.reset_peak_memory_stats()
        y = kosmos.kosmos_moe(env.group, x, idx, w, d.w1k, d.w2)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - before
    allowed = y.numel() * y.element_size() + idx.numel() * 4 + 2 * 512
    return 0 < peak <= allowed, {"peak_over_before": peak, "allowed": allowed}


def check_skew_rounds(env, d):
    """Skewed routing with the capacity below the busiest rank's rows. Expected: status 8 (rounds),
    all five tensors bit for bit equal to a run with enough capacity, gradients within the
    reference bounds."""
    x, idx, w, dy = make_tokens(env, 21, skew=2)
    cnt = torch.bincount(idx.view(-1), minlength=env.E)
    dist.all_reduce(cnt, group=env.group)
    per_rank = cnt.view(env.world, -1).sum(dim=1)
    small = -(-int(per_rank.float().mean()) // 256) * 256 + 256 * env.E_loc
    big = -(-int(per_rank.max()) // 256) * 256 + 512 * env.E_loc
    ref = reference(env, x, idx, w, d.w1, d.w2, dy)
    _status(env)
    low = run_layer(env, x, idx, w, d.w1k, d.w2, dy, capacity_rows=small)
    st_low = _status(env)
    high = run_layer(env, x, idx, w, d.w1k, d.w2, dy, capacity_rows=big)
    st_high = _status(env)
    num = numerics(low, ref)
    eq = bitwise(low, high)
    ok = bool(st_low & kf.STATUS_ROUNDS) and not _errors(st_low) and st_high == 0
    ok &= _all_ok(num) and all(eq.values())
    for cap in (small, big):
        if cap != _default_rows(env):
            kf._INSTANCES.destroy_instance(_private_instance(env, cap))
    return ok, {
        "rows_per_rank": per_rank.tolist(),
        "capacity_low": small,
        "capacity_high": big,
        "status_low": st_low,
        "status_high": st_high,
        "numerics_low": num,
        "bitwise_low_vs_high": eq,
    }


def check_dropped_ids(env, d):
    """-1 ids (and 64 tokens with every route dropped). Expected: within the reference bounds,
    padding tokens' y and dx exactly zero, dtopk_w exactly zero at -1, status bit 1 reported (KOSMOS
    counts a dropped route as a bad id) and raised by status() unless raise_on leaves it out."""
    x, idx, w, dy = make_tokens(env, 31, dropped=0.1)
    res = run_layer(env, x, idx, w, d.w1k, d.w2, dy)
    ref = reference(env, x, idx, w, d.w1, d.w2, dy)
    num = numerics(res, ref)
    exact = {
        "pad_y_zero": bool((res["y"][:64] == 0).all()),
        "pad_dx_zero": bool((res["dx"][:64] == 0).all()),
        "dtw_zero_at_dropped": bool((res["dtw"][idx < 0] == 0).all()),
    }
    raised = False
    try:
        kosmos.status(env.group)
    except RuntimeError:
        raised = True
    # the bits were cleared by the query above; run once more to test raise_on
    run_layer(env, x, idx, w, d.w1k, d.w2, dy)
    st = kosmos.status(env.group, raise_on=(kf.STATUS_DUPLICATE, kf.STATUS_ERROR))
    ok = _all_ok(num) and all(exact.values()) and raised and st & kf.STATUS_BAD_ID
    return bool(ok), {
        "numerics": num,
        "exact": exact,
        "default_status_raised": raised,
        "status": st,
    }


def check_degenerate(env, d):
    """Every token's first route to one expert (global expert 0, rank 0) and an expert that gets no
    token at all (global expert E_loc + 1). Expected: within the reference bounds; the empty
    expert's dW1 and dW2 exactly zero."""
    empty = env.E_loc + 1
    x, idx, w, dy = make_tokens(env, 41, exclude_expert=empty, hot_expert=0)
    res = run_layer(env, x, idx, w, d.w1k, d.w2, dy)
    st = _status(env)
    ref = reference(env, x, idx, w, d.w1, d.w2, dy)
    num = numerics(res, ref)
    zero = True
    if env.rank == empty // env.E_loc:
        e = empty % env.E_loc
        zero = bool((res["dw1"][e] == 0).all()) and bool((res["dw2"][e] == 0).all())
    ok = _all_ok(num) and zero and not _errors(st)
    return ok, {"numerics": num, "empty_expert_grads_zero": zero, "status": st}


_LIFETIMES = 11
_MIB = 1 << 20


def _settled_free_bytes(env):
    """(free device bytes, caching-allocator reserved bytes) once every rank has let go of what it
    dropped. Peers map each other's symmetric buffers, so a rank's buffer is released only after
    every peer has unmapped it: hence the barrier."""
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    dist.barrier(group=env.group)
    torch.cuda.synchronize()
    return torch.cuda.mem_get_info()[0], torch.cuda.memory_reserved()


def _lifetimes(env, d, cap, alloc, solo, leak=False):
    """_LIFETIMES create / one training call / destroy cycles of an instance of its own. Returns
    whether each lifetime's results equal ``solo`` bit for bit, the bytes outside the caching
    allocator still held after each lifetime (relative to before the first), and the symmetric
    bytes. ``leak``: the mutant, whose destroy keeps the symmetric buffer (torch) or the whole
    instance (kosmos)."""
    x, idx, w, dy = d.tok[11]
    kept = []
    eq = []
    held = []
    sym = 0
    f_start, r_start = _settled_free_bytes(env)
    for _ in range(_LIFETIMES):
        inst = _private_instance(env, cap, allocator=alloc)
        sym = inst.symmetric_bytes
        eq.append(all(bitwise(_run_on(env, inst, x, idx, w, d.w1k, d.w2, dy), solo).values()))
        if leak and alloc == "kosmos":
            # never destroyed until the end, only forgotten by the cache
            kept.append(inst)
            kf._INSTANCES._instances = {
                k: v for k, v in kf._INSTANCES._instances.items() if v is not inst
            }
        else:
            if leak:
                kept.append(inst._symm)  # what torch 2.12 does: the buffer outlives the destroy
            kf._INSTANCES.destroy_instance(inst)
        del inst
        f1, r1 = _settled_free_bytes(env)
        held.append((f_start - f1) - (r1 - r_start))
    for k in kept:
        if isinstance(k, ki.KosmosInstance):
            k.destroy()
    kept.clear()
    _settled_free_bytes(env)
    return eq, held, sym


def _leaks(held, sym):
    """A per-destroy leak of X bytes grows the held bytes by X per lifetime. Torch symmetric memory
    may hold a freed buffer back until a later allocation, which moves the held bytes by up to one
    buffer but does not accumulate. So the smaller of the bytes held after the last two lifetimes,
    less those after the first (which also carries one-time costs), must stay below one buffer plus
    64 MiB: with 11 lifetimes a leak above (buffer + 64 MiB) / 9 per destroy fails, about 57 MiB at
    the small shape. Returns (leak, that growth)."""
    growth = min(held[-2:]) - held[0]
    return growth >= sym + 64 * _MIB, growth


def check_create_destroy_twice(env, d):
    """An instance created and destroyed _LIFETIMES times in one process, with KOSMOS's own
    symmetric-memory allocation and, when that is the one in use (torch >= 2.13 by default, or
    forced), torch's. Expected: every lifetime's results bit for bit equal to the main instance's,
    and no per-destroy leak (see _leaks). A mutant whose destroy keeps the buffer must be judged a
    leak, so the criterion is shown to fail where it must (torch 2.12 leaks one buffer per
    destroy)."""
    cap = _default_rows(env) + 768
    # one-time costs first (the KOSMOS code objects, the EP group's communicator), so that what
    # is measured below is only what an instance holds
    solo = d.solo(11)
    in_use = ki.resolve_allocator()
    allocators = ["kosmos"] + (["torch"] if in_use == "torch" else [])
    details = {"allocator_in_use": in_use}
    mine = {}
    ok = True
    for alloc in allocators + ["mutant"]:
        a = in_use if alloc == "mutant" else alloc
        eq, held, sym = _lifetimes(env, d, cap, a, solo, leak=alloc == "mutant")
        leak, growth = _leaks(held, sym)
        if alloc == "mutant":
            ok &= leak
        else:
            ok &= all(eq) and not leak
        details[alloc] = {
            "allocator": a,
            "bitwise_vs_main_instance": eq,
            "bytes_held_after_lifetime": held,
            "growth_after_first_bytes": growth,
            "judged_leak": leak,
            "symmetric_bytes": sym,
        }
        mine[alloc] = [round(v / _MIB, 1) for v in held]
    # every rank's held MiB per lifetime, so that a failure on any rank can be read from rank 0's
    # record
    per_rank = [None] * env.world
    dist.all_gather_object(per_rank, mine, group=env.group)
    details["held_mib_per_rank"] = per_rank
    return ok, details


def _main_grad_run(env, d, dtype, start_seed):
    """Three microbatches with fuse_wgrad_accumulation into main_grad of `dtype` that starts
    non-zero. Returns (main_grad - start per weight in fp32, flags, start)."""
    w1p = torch.nn.Parameter(d.w1k.clone())
    w2p = torch.nn.Parameter(d.w2.clone())
    g = _gen(env, start_seed)
    start = {}
    for name, p in (("dw1", w1p), ("dw2", w2p)):
        s0 = (torch.randn(p.shape, generator=g, device=env.dev) * 1e-3).to(dtype)
        p.main_grad = s0.clone()
        p.grad_added_to_main_grad = False
        start[name] = s0
    for s in (11, 12, 13):
        x, idx, w, dy = d.tok[s]
        xl = x.detach().clone().requires_grad_(True)
        y = kosmos.kosmos_moe(env.group, xl, idx, w, w1p, w2p, fuse_wgrad_accumulation=True)
        y.backward(dy)
    torch.cuda.synchronize()
    added = {
        "dw1": w1p.main_grad.float() - start["dw1"].float(),
        "dw2": w2p.main_grad.float() - start["dw2"].float(),
    }
    flags = {"w1": bool(w1p.grad_added_to_main_grad), "w2": bool(w2p.grad_added_to_main_grad)}
    return added, flags, (w1p, w2p)


def check_main_grad_acc(env, d):
    """fuse_wgrad_accumulation over three microbatches into a main_grad that starts non-zero:
    fp32 (KOSMOS acc 1) and bf16 (acc 2). Expected: what was added equals the fp32 reference's
    dW summed over the three calls (rel L2 1e-2, scale 1 +- 1e-2) and the op's own unfused dW summed
    the same way; grad_added_to_main_grad set. Then the fallback (only W1 has a main_grad, so the
    kernel cannot add both): W1.main_grad gets exactly start + dW1 of the unfused path, W2 gets an
    ordinary .grad equal to the unfused dW2, and W1 gets no .grad (it has no flag attribute)."""
    ref_sum = {n: sum(d.ref(s)[n].float() for s in (11, 12, 13)) for n in ("dw1", "dw2")}
    op_sum = {n: sum(d.solo(s)[n].float() for s in (11, 12, 13)) for n in ("dw1", "dw2")}
    details = {}
    ok = True
    for dtype, label in ((torch.float32, "fp32_acc1"), (torch.bfloat16, "bf16_acc2")):
        added, flags, _ = _main_grad_run(env, d, dtype, 51)
        m_ref = {n: metrics(added[n], ref_sum[n]) for n in added}
        m_op = {n: metrics(added[n], op_sum[n]) for n in added}
        ok &= all(close(m) for m in m_ref.values()) and all(close(m) for m in m_op.values())
        ok &= all(flags.values())
        details[label] = {"vs_reference": m_ref, "vs_unfused_dw": m_op, "flags": flags}
    # fallback: only W1 has main_grad
    w1p = torch.nn.Parameter(d.w1k.clone())
    w2p = torch.nn.Parameter(d.w2.clone())
    start = torch.full(w1p.shape, 0.5, dtype=torch.float32, device=env.dev)
    w1p.main_grad = start.clone()
    x, idx, w, dy = d.tok[11]
    y = kosmos.kosmos_moe(env.group, x, idx, w, w1p, w2p, fuse_wgrad_accumulation=True)
    y.backward(dy)
    torch.cuda.synchronize()
    fb = {
        "w1_main_grad_exact": bool(torch.equal(w1p.main_grad, start + d.solo(11)["dw1"].float())),
        "w2_grad_exact": w2p.grad is not None and bool(torch.equal(w2p.grad, d.solo(11)["dw2"])),
        "w1_grad_none": w1p.grad is None,
    }
    ok &= all(fb.values())
    details["fallback_w1_only"] = fb
    st = _status(env)
    ok &= not _errors(st)
    details["status"] = st
    return ok, details


def check_saved_tensor_hooks(env, d):
    """Every saved tensor moved to the host and back (CPU offload). Expected: bit for bit equal,
    and the five saved buffers among what was packed."""
    packed = []

    def pack(t):
        packed.append(t.numel() * t.element_size())
        return t.to("cpu", copy=True)

    def unpack(t):
        return t.to(env.dev)

    x, idx, w, dy = d.tok[11]
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        y, leaves = forward(env, x, idx, w, d.w1k, d.w2)
    res = backward(y, dy, leaves)
    eq = bitwise(res, d.solo(11))
    st = _status(env)
    return all(eq.values()) and len(packed) >= 10 and st == 0, {
        "bitwise": eq,
        "tensors_packed": len(packed),
        "bytes_packed": sum(packed),
        "status": st,
    }


def check_version_counter(env, d):
    """x, then W1, changed in place between forward and backward. Expected: autograd raises on every
    rank, both times (the saved tensors' version counters are checked)."""
    x, idx, w, dy = d.tok[11]
    raised = {}
    for name, which in (("x", 0), ("w1", 2)):
        y, leaves = forward(env, x, idx, w, d.w1k, d.w2)
        with torch.no_grad():
            leaves[which].mul_(2)
        raised[name] = False
        try:
            y.backward(dy)
        except RuntimeError as e:
            raised[name] = "modified by an inplace operation" in str(e)
        del y, leaves
    _status(env)
    return all(raised.values()), {"raised": raised}


def check_two_instances(env, d):
    """Two instances of different shapes (T and T/2 tokens) interleaved in one step: forward A,
    forward B, backward A, backward B. Expected: both bit for bit equal to their solo runs, status
    0."""
    half = env.T // 2
    xb, idxb, wb, dyb = make_tokens(env, 61, tokens=half)
    solo_b = run_layer(env, xb, idxb, wb, d.w1k, d.w2, dyb)
    ref_b = reference(env, xb, idxb, wb, d.w1, d.w2, dyb)
    xa, idxa, wa, dya = d.tok[11]
    solo_a = d.solo(11)
    _status(env)
    ya, la = forward(env, xa, idxa, wa, d.w1k, d.w2)
    yb, lb = forward(env, xb, idxb, wb, d.w1k, d.w2)
    a = backward(ya, dya, la)
    b = backward(yb, dyb, lb)
    n_inst = len(kf._INSTANCES.matching(env.group))
    eq = {"a": bitwise(a, solo_a), "b": bitwise(b, solo_b)}
    num_b = numerics(b, ref_b)
    st = _status(env)
    ok = all(all(e.values()) for e in eq.values()) and st == 0 and _all_ok(num_b)
    ok &= n_inst >= 2
    for inst in [v for _, v in kf._INSTANCES.matching(env.group) if v.tokens == half]:
        kf._INSTANCES.destroy_instance(inst)
    return ok, {"bitwise": eq, "numerics_b": num_b, "status": st, "instances": n_inst}


def _checkpointed(env, d, use_reentrant):
    """torch.utils.checkpoint around the op, three steps. Returns per step whether it matched the
    solo run bit for bit and the caching allocator's bytes after the step."""
    from torch.utils.checkpoint import checkpoint

    _main_instance(env)
    x, idx, w, dy = d.tok[11]
    solo = d.solo(11)
    _status(env)
    eq = []
    mem = []
    for _ in range(3):
        leaves = [t.detach().clone().requires_grad_(True) for t in (x, w, d.w1k, d.w2)]

        def layer(xx, ww, w1, w2):
            return kosmos.kosmos_moe(env.group, xx, idx, ww, w1, w2)

        y = checkpoint(layer, *leaves, use_reentrant=use_reentrant)
        res = backward(y, dy, leaves)
        eq.append(all(bitwise(res, solo).values()))
        del y, leaves, res
        mem.append(allocated())
    return eq, mem, _status(env)


def check_checkpoint_reentrant(env, d):
    """Activation recompute, reentrant (Megatron's style): the first pass runs without grad, the
    recompute with grad right before the backward. Expected: three steps bit for bit equal to the
    solo run, the same allocator bytes after each step, status 0."""
    eq, mem, st = _checkpointed(env, d, True)
    ok = all(eq) and len(set(mem)) == 1 and st == 0
    return ok, {"bitwise_per_step": eq, "allocated_after_step": mem, "status": st}


def check_checkpoint_nonreentrant(env, d):
    """Activation recompute, non-reentrant (torch.utils.checkpoint(use_reentrant=False)): the first
    pass runs with grad and its saved tensors are dropped at once; the recompute inside the backward
    is a training forward that refills them. Expected: three steps bit for bit equal to the solo
    run, the same allocator bytes after each step (nothing accumulates; there is no reclaim),
    status 0."""
    eq, mem, st = _checkpointed(env, d, False)
    ok = all(eq) and len(set(mem)) == 1 and st == 0
    return ok, {"bitwise_per_step": eq, "allocated_after_step": mem, "status": st}


def check_symm_vs_own(env, d):
    """The instance's symmetric buffer from torch symmetric memory against KOSMOS's own allocation,
    same configuration. Expected with torch >= 2.12: all five tensors bit for bit equal, both
    status 0, the torch instance's buffer counted in memory_report. Below 2.12: auto resolves to
    KOSMOS's own allocation and forcing torch raises. Below 2.13, auto resolves to KOSMOS's own
    allocation (see _instance.resolve_allocator)."""
    if not ki.torch_symmetric_memory_supported():
        os.environ[ki.SYMMETRIC_MEMORY_ENV] = "torch"
        raised = False
        try:
            ki.resolve_allocator()
        except RuntimeError:
            raised = True
        del os.environ[ki.SYMMETRIC_MEMORY_ENV]
        auto = ki.resolve_allocator()
        return raised and auto == "kosmos", {"torch": torch.__version__, "auto": auto}
    cap = _default_rows(env) + 1024
    x, idx, w, dy = d.tok[11]
    out = {}
    reports = {}
    for alloc in ("torch", "kosmos"):
        inst = _private_instance(env, cap, allocator=alloc)
        _status(env)
        # through the op's autograd function on this instance: kosmos_moe() would pick the
        # instance of the environment's allocator
        out[alloc] = _run_on(env, inst, x, idx, w, d.w1k, d.w2, dy)
        reports[alloc] = inst.memory_report()
        out[alloc]["status"] = inst.status()
        kf._INSTANCES.destroy_instance(inst)
    eq = bitwise(out["torch"], out["kosmos"])
    eq_main = bitwise(out["torch"], d.solo(11))
    ok = all(eq.values()) and out["torch"]["status"] == 0 and out["kosmos"]["status"] == 0
    ok &= reports["torch"]["symmetric_bytes"] >= reports["torch"]["instance_bytes"] > 0
    return ok, {
        "bitwise_torch_vs_kosmos": eq,
        "bitwise_torch_vs_main": eq_main,
        "status": {a: out[a]["status"] for a in out},
        "memory_report": reports,
    }


def _run_on(env, inst, x, idx, w, w1k, w2, dy):
    """One training call through the op's autograd function on a given instance."""
    leaves = [t.detach().clone().requires_grad_(True) for t in (x, w, w1k, w2)]
    y = kf._KosmosMoeFn.apply(
        leaves[0],
        leaves[1],
        leaves[2],
        leaves[3],
        kf._int32(idx),
        inst,
        "all",
        inst.capacity_rows,
        False,
    )
    return backward(y, dy, leaves)


def check_te_ep_reference(env, d):
    """Against TE's own pure-PyTorch EP reference (ep_reference.py, the reference NVIDIA's fused MoE
    is tested against), set to KOSMOS's semantics: router weights applied after GEMM2
    (apply_topk_in_fc1=False), weights in its layout (fc1 [E_loc, H, 2I] gate then up, fc2
    [E_loc, I, H]). Expected: y, dx and dtopk_w within the bounds."""
    from ep_reference import MoeEpReference

    x, idx, w, dy = d.tok[11]
    ref = MoeEpReference(
        num_experts=env.E,
        hidden_size=env.H,
        intermediate_size=env.I,
        top_k=env.k,
        ep_group=env.group,
        apply_topk_in_fc1=False,
        generate_c=True,
    )
    fc1 = d.w1.transpose(1, 2).contiguous()
    fc2 = d.w2.transpose(1, 2).contiguous()
    y_ref, fc1_c, meta = ref(x, fc1, fc2, idx, w)
    dx_ref, dtw_ref = ref.backward(dy, fc1, fc2, idx, w, fc1_c, meta)
    res = d.solo(11)
    num = {}
    for name, r in (("y", y_ref), ("dx", dx_ref), ("dtw", dtw_ref)):
        m = metrics(res[name], r)
        m["ok"] = close(m)
        num[name] = m
    return _all_ok(num), {"numerics": num}


def check_layout(env, d):
    """to_kosmos_w1 / from_kosmos_w1 are inverses, and to_kosmos_w1 puts gate rows 0..127 then up
    rows 0..127 in the first 256-row block."""
    w = torch.arange(2 * env.I, device=env.dev, dtype=torch.float32).view(1, -1, 1).expand(2, -1, 3)
    k = kosmos.to_kosmos_w1(w)
    first = k[0, :256, 0].tolist()
    want = list(range(128)) + [env.I + i for i in range(128)]
    ok = torch.equal(kosmos.from_kosmos_w1(k), w) and first == [float(v) for v in want]
    return ok, {}


# --------------------------------------------------------------------------- #
# CUDA graphs: the op's calls captured with torch.cuda.graph after eager calls of the same
# configuration (the first creates the instance, which is collective and host-blocking), every
# replay compared bit for bit with the same calls run eagerly on inputs of their own
# --------------------------------------------------------------------------- #
GRAPH_REPLAYS = 5


class _Microbatch:
    """One microbatch's inputs; x and topk_weights are leaves that require grad. A captured graph
    reads these addresses, so new values are copied in, never rebound."""

    def __init__(self, x, idx, w, dy):
        self.x = x.detach().clone().requires_grad_(True)
        self.idx = idx.clone()
        self.w = w.detach().clone().requires_grad_(True)
        self.dy = dy.clone()

    def load(self, x=None, idx=None, w=None):
        with torch.no_grad():
            for dst, src in ((self.x, x), (self.idx, idx), (self.w, w)):
                if src is not None:
                    dst.copy_(src)


def _graph_params(env, d, main_grad=False):
    """W1 and W2 as parameters. With ``main_grad``: Megatron's attributes, an fp32 ``main_grad``
    that starts non-zero; returns its start values too."""
    w1p = torch.nn.Parameter(d.w1k.clone())
    w2p = torch.nn.Parameter(d.w2.clone())
    start = None
    if main_grad:
        g = _gen(env, 71)
        start = []
        for p in (w1p, w2p):
            s0 = torch.randn(p.shape, generator=g, device=env.dev) * 1e-3
            p.main_grad = s0.clone()
            p.grad_added_to_main_grad = False
            start.append(s0)
    return w1p, w2p, start


def _layer_calls(env, mbs, order, w1p, w2p, fused=False, **kw):
    """The forwards of ``mbs`` in turn, then their backwards in ``order`` (indices into ``mbs``).
    Gradients by torch.autograd.grad, so nothing accumulates into .grad and the same code runs
    eagerly and under capture. ``fused``: dW is added into main_grad and not returned."""
    ys = [
        kosmos.kosmos_moe(
            env.group, mb.x, mb.idx, mb.w, w1p, w2p, fuse_wgrad_accumulation=fused, **kw
        )
        for mb in mbs
    ]
    out = [None] * len(mbs)
    for i in order:
        mb = mbs[i]
        wrt = (mb.x, mb.w) if fused else (mb.x, mb.w, w1p, w2p)
        grads = torch.autograd.grad(ys[i], wrt, grad_outputs=mb.dy)
        res = {"y": ys[i].detach(), "dx": grads[0], "dtw": grads[1]}
        if not fused:
            res["dw1"] = grads[2]
            res["dw2"] = grads[3]
        out[i] = res
    return out


def _eager(env, toks, order, w1p, w2p, fused=False, **kw):
    """_layer_calls run eagerly on microbatches with storage of their own; copies of the results."""
    mbs = [_Microbatch(*t) for t in toks]
    out = [
        {n: v.detach().clone() for n, v in r.items()}
        for r in _layer_calls(env, mbs, order, w1p, w2p, fused, **kw)
    ]
    torch.cuda.synchronize()
    return out


def _capture(fn):
    """``fn`` (which only enqueues work) captured into a CUDA graph after one warm-up run on a side
    stream, as torch.cuda.graph asks. Returns the graph and what ``fn`` returned during the
    capture: the graph's static outputs."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    torch.cuda.synchronize()
    return graph, out


def _replay(graph):
    graph.replay()
    torch.cuda.synchronize()


def _replays_queued(graph, outputs, n, before=None):
    """``n`` replays enqueued back to back with no host sync in between, as a training loop
    launches them; after each, a copy of ``outputs()`` taken on the stream. ``before(i)`` enqueues
    work ahead of replay i (copying new inputs in). Synchronizes once, at the end."""
    got = []
    for i in range(n):
        if before is not None:
            before(i)
        graph.replay()
        got.append({k: v.clone() for k, v in outputs().items()})
    torch.cuda.synchronize()
    return got


def _flat(outs):
    """Per-microbatch results as one dict: ``{"mb0.y": ..., "mb1.y": ...}``."""
    return {f"mb{i}.{n}": v for i, r in enumerate(outs) for n, v in r.items()}


def _same(a, b):
    """Bit for bit, per tensor name present in both."""
    return {n: bool(torch.equal(a[n], b[n])) for n in a if n in b}


def _all_same(eqs):
    return all(all(e.values()) for e in eqs)


def check_graph_one_layer(env, d):
    """One layer's forward and backward captured once and replayed GRAPH_REPLAYS times. Expected:
    every replay's y, dx, dtopk_w, dW1 and dW2 bit for bit equal to the same call run eagerly (two
    eager runs equal each other); status 0. Then fuse_wgrad_accumulation into an fp32 main_grad
    that starts non-zero (KOSMOS adds into it in the kernel): main_grad after replay i bit for bit
    equal to main_grad after eager step i from the same start, i = 1..GRAPH_REPLAYS (two eager
    series equal each other); y, dx and dtopk_w of every replay equal to the eager step's;
    status 0. Both parts again with the replays queued back to back, without a host sync between
    them (as a training loop launches them), compared after one sync at the end."""
    tok = [d.tok[11]]
    w1p, w2p, _ = _graph_params(env, d)
    eager = _eager(env, tok, [0], w1p, w2p)[0]
    eager_twice = _same(_eager(env, tok, [0], w1p, w2p)[0], eager)
    st_eager = _status(env)
    mbs = [_Microbatch(*tok[0])]
    graph, static = _capture(lambda: _layer_calls(env, mbs, [0], w1p, w2p))
    _status(env)
    replays = []
    for _ in range(GRAPH_REPLAYS):
        _replay(graph)
        replays.append(_same(static[0], eager))
    st_graph = _status(env)
    queued = [_same(q, eager) for q in _replays_queued(graph, lambda: static[0], GRAPH_REPLAYS)]
    st_queued = _status(env)
    ok = all(eager_twice.values()) and _all_same(replays) and _all_same(queued)
    ok &= st_eager == 0 and st_graph == 0 and st_queued == 0
    details = {
        "eager_twice_bitwise": eager_twice,
        "replays_bitwise_vs_eager": replays,
        "queued_replays_bitwise_vs_eager": queued,
        "status_eager": st_eager,
        "status_graph": st_graph,
        "status_queued": st_queued,
    }
    del graph, static, mbs

    w1f, w2f, start = _graph_params(env, d, main_grad=True)

    def reset():
        with torch.no_grad():
            for p, s0 in zip((w1f, w2f), start):
                p.main_grad.copy_(s0)

    def with_main_grad(res):
        return dict(res, mg1=w1f.main_grad, mg2=w2f.main_grad)

    reset()
    steps = []
    for _ in range(GRAPH_REPLAYS):
        res = _eager(env, tok, [0], w1f, w2f, fused=True)[0]
        steps.append({n: v.clone() for n, v in with_main_grad(res).items()})
    reset()
    series_twice = []
    for i in range(GRAPH_REPLAYS):
        res = _eager(env, tok, [0], w1f, w2f, fused=True)[0]
        series_twice.append(_same(with_main_grad(res), steps[i]))
    st_eager_f = _status(env)
    mbs = [_Microbatch(*tok[0])]
    graph, static = _capture(lambda: _layer_calls(env, mbs, [0], w1f, w2f, fused=True))
    _status(env)
    # the warm-up added into main_grad; the graph adds into the same storage
    reset()
    fused_replays = []
    for i in range(GRAPH_REPLAYS):
        _replay(graph)
        fused_replays.append(_same(with_main_grad(static[0]), steps[i]))
    st_graph_f = _status(env)
    reset()
    queued = _replays_queued(graph, lambda: with_main_grad(static[0]), GRAPH_REPLAYS)
    fused_queued = [_same(q, steps[i]) for i, q in enumerate(queued)]
    del queued
    st_queued_f = _status(env)
    # the steps do accumulate: main_grad after the last step differs from after the first
    accumulates = not torch.equal(steps[-1]["mg1"], steps[0]["mg1"])
    accumulates &= not torch.equal(steps[-1]["mg2"], steps[0]["mg2"])
    ok &= _all_same(series_twice) and _all_same(fused_replays) and _all_same(fused_queued)
    ok &= accumulates and st_eager_f == 0 and st_graph_f == 0 and st_queued_f == 0
    details["fused"] = {
        "eager_series_twice_bitwise": series_twice,
        "replays_bitwise_vs_eager_step": fused_replays,
        "queued_replays_bitwise_vs_eager_step": fused_queued,
        "status_queued": st_queued_f,
        "main_grad_accumulates": accumulates,
        "grad_added_to_main_grad": [
            bool(w1f.grad_added_to_main_grad),
            bool(w2f.grad_added_to_main_grad),
        ],
        "status_eager": st_eager_f,
        "status_graph": st_graph_f,
    }
    return ok, details


def check_graph_two_microbatches(env, d):
    """Two microbatches in flight in one graph: forward 1, forward 2, then the backwards first-in
    first-out (one graph) or last-in first-out (another graph). Expected: every replay bit for bit
    equal to the same calls run eagerly in the same order, which equal each microbatch run on its
    own, also with the replays queued back to back; status 0."""
    toks = [d.tok[11], d.tok[12]]
    w1p, w2p, _ = _graph_params(env, d)
    solo = [_eager(env, [t], [0], w1p, w2p)[0] for t in toks]
    _status(env)
    details = {}
    ok = True
    for label, order in (("fifo", [0, 1]), ("lifo", [1, 0])):
        eager = _eager(env, toks, order, w1p, w2p)
        st_eager = _status(env)
        mbs = [_Microbatch(*t) for t in toks]
        graph, static = _capture(
            lambda mbs=mbs, order=order: _layer_calls(env, mbs, order, w1p, w2p)
        )
        _status(env)
        replays = []
        for _ in range(GRAPH_REPLAYS):
            _replay(graph)
            replays.append([_same(static[i], eager[i]) for i in range(2)])
        st_graph = _status(env)
        queued = [
            _same(q, _flat(eager))
            for q in _replays_queued(graph, lambda static=static: _flat(static), GRAPH_REPLAYS)
        ]
        st_queued = _status(env)
        vs_solo = [_same(eager[i], solo[i]) for i in range(2)]
        ok &= all(_all_same(r) for r in replays) and _all_same(vs_solo) and _all_same(queued)
        ok &= st_eager == 0 and st_graph == 0 and st_queued == 0
        details[label] = {
            "replays_bitwise_vs_eager": replays,
            "queued_replays_bitwise_vs_eager": queued,
            "status_queued": st_queued,
            "eager_bitwise_vs_solo": vs_solo,
            "status_eager": st_eager,
            "status_graph": st_graph,
        }
        del graph, static, mbs
    return ok, details


def _rows_per_rank(env, idx):
    """Routed rows each rank receives (global, every rank's tokens), dropped routes excluded."""
    cnt = torch.bincount(idx[idx >= 0].view(-1), minlength=env.E)
    dist.all_reduce(cnt, group=env.group)
    return cnt.view(env.world, -1).sum(dim=1)


def _skewed_routing(env, seed, capacity, over):
    """A make_tokens Gumbel skew whose busiest rank fits the capacity with a row block of padding
    per expert to spare (over False: the largest such skew), or receives more rows than the
    capacity (over True: the smallest). Returns (skew, topk_idx, topk_weights, rows per rank), or
    Nones; the same on every rank."""
    skews = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
    for skew in skews if over else reversed(skews):
        _, idx, w, _ = make_tokens(env, seed, skew=skew)
        rows = _rows_per_rank(env, idx)
        busiest = int(rows.max())
        if (over and busiest > capacity) or (not over and busiest + 256 * env.E_loc <= capacity):
            return skew, idx, w, rows.tolist()
    return None, None, None, None


def check_graph_routing(env, d):
    """One graph captured with a balanced routing; before each replay another routing is copied
    into its static topk_idx and topk_weights: a second balanced draw, a skewed one that fits the
    capacity, one over the capacity (the call runs in rounds and keeps nothing), the first again.
    Expected: every replay bit for bit equal to the same call run eagerly, and the same status:
    0, 0, 0, 8 | 32, 0; the same with every copy and replay queued back to back, status 8 | 32
    over the five. Then a mutant replay without routing B copied in: it must differ from
    the eager call of B."""
    w1p, w2p, _ = _graph_params(env, d)
    cap = _main_instance(env).capacity_rows
    x, idx_a, w_a, dy = d.tok[11]
    _, idx_b, w_b, _ = d.tok[12]
    s_skew, idx_s, w_s, rows_s = _skewed_routing(env, 21, cap, over=False)
    s_over, idx_o, w_o, rows_o = _skewed_routing(env, 22, cap, over=True)
    details = {
        "capacity_rows": cap,
        "skewed": {"skew": s_skew, "rows_per_rank": rows_s},
        "over_capacity": {"skew": s_over, "rows_per_rank": rows_o},
    }
    if s_skew is None or s_over is None:
        return False, details
    over = kf.STATUS_ROUNDS | kf.STATUS_KEPT_NOTHING
    routings = (
        ("balanced_a", idx_a, w_a, 0),
        ("balanced_b", idx_b, w_b, 0),
        ("skewed", idx_s, w_s, 0),
        ("over_capacity", idx_o, w_o, over),
        ("balanced_a_again", idx_a, w_a, 0),
    )
    _status(env)
    mbs = [_Microbatch(x, idx_a, w_a, dy)]
    graph, static = _capture(lambda: _layer_calls(env, mbs, [0], w1p, w2p))
    _status(env)
    ok = True
    eager_of = {}
    for name, idx, w, want in routings:
        eager = _eager(env, [(x, idx, w, dy)], [0], w1p, w2p)[0]
        st_eager = _status(env)
        mbs[0].load(idx=idx, w=w)
        _replay(graph)
        st_graph = _status(env)
        eq = _same(static[0], eager)
        ok &= all(eq.values()) and st_eager == want and st_graph == want
        details[name] = dict(
            details.get(name, {}),
            bitwise_vs_eager=eq,
            status_eager=st_eager,
            status_graph=st_graph,
            status_expected=want,
        )
        eager_of[name] = eager
    # the same routings again, every copy and replay queued back to back: one status for all
    queued = _replays_queued(
        graph,
        lambda: static[0],
        len(routings),
        before=lambda i: mbs[0].load(idx=routings[i][1], w=routings[i][2]),
    )
    queued_eq = {r[0]: _same(q, eager_of[r[0]]) for r, q in zip(routings, queued)}
    del queued
    st_queued = _status(env)
    ok &= _all_same(queued_eq.values()) and st_queued == over
    details["queued"] = {
        "bitwise_vs_eager": queued_eq,
        "status": st_queued,
        "status_expected": over,
    }
    # mutant: a replay without routing B copied in (the static buffers still hold A)
    _replay(graph)
    eq = _same(static[0], eager_of["balanced_b"])
    _status(env)
    ok &= not eq["y"]
    details["mutant_routing_not_copied"] = {"bitwise_vs_eager_b": eq, "detected": not eq["y"]}
    return ok, details


def check_graph_save_tiers(env, d):
    """save="xp" and save="none" captured (the backward recomputes GEMM1, or sends x to the experts
    again). Expected: every replay bit for bit equal to the same call run eagerly with the same
    tier, which equals the eager save="all" call, also with the replays queued back to back;
    status 0."""
    tok = [d.tok[11]]
    w1p, w2p, _ = _graph_params(env, d)
    full = _eager(env, tok, [0], w1p, w2p)[0]
    _status(env)
    details = {}
    ok = True
    for save in ("xp", "none"):
        eager = _eager(env, tok, [0], w1p, w2p, save=save)[0]
        st_eager = _status(env)
        mbs = [_Microbatch(*tok[0])]
        graph, static = _capture(
            lambda mbs=mbs, save=save: _layer_calls(env, mbs, [0], w1p, w2p, save=save)
        )
        _status(env)
        replays = []
        for _ in range(GRAPH_REPLAYS):
            _replay(graph)
            replays.append(_same(static[0], eager))
        st_graph = _status(env)
        queued = [
            _same(q, eager)
            for q in _replays_queued(graph, lambda static=static: static[0], GRAPH_REPLAYS)
        ]
        st_queued = _status(env)
        vs_all = _same(eager, full)
        ok &= _all_same(replays) and _all_same(queued) and all(vs_all.values())
        ok &= st_eager == 0 and st_graph == 0 and st_queued == 0
        details[save] = {
            "replays_bitwise_vs_eager": replays,
            "queued_replays_bitwise_vs_eager": queued,
            "status_queued": st_queued,
            "eager_bitwise_vs_save_all": vs_all,
            "status_eager": st_eager,
            "status_graph": st_graph,
        }
        del graph, static, mbs
    return ok, details


def check_graph_mutation(env, d):
    """The graph checks' comparison must be able to fail: after capture one element of the static x
    is changed (x[5, 7] += 1) and the eager call is not re-run. Expected: the replay before the
    change equals the eager call; the replay after it differs in y (on every rank, each changes
    its own x); restoring x makes the replay equal again."""
    tok = [d.tok[11]]
    w1p, w2p, _ = _graph_params(env, d)
    eager = _eager(env, tok, [0], w1p, w2p)[0]
    _status(env)
    mbs = [_Microbatch(*tok[0])]
    graph, static = _capture(lambda: _layer_calls(env, mbs, [0], w1p, w2p))
    _status(env)
    _replay(graph)
    before = _same(static[0], eager)
    with torch.no_grad():
        mbs[0].x[5, 7] += 1.0
    _replay(graph)
    mutated = _same(static[0], eager)
    mbs[0].load(x=tok[0][0])
    _replay(graph)
    restored = _same(static[0], eager)
    st = _status(env)
    ok = all(before.values()) and not mutated["y"] and all(restored.values()) and st == 0
    return ok, {
        "before_bitwise": before,
        "mutated_bitwise": mutated,
        "restored_bitwise": restored,
        "status": st,
    }


def check_graph_first_call_in_capture(env, d):
    """A configuration's first call (which creates its instance: collective, host-blocking) made
    inside torch.cuda.graph capture, with no eager call before it. Expected on every rank: the op's
    RuntimeError saying the first call cannot be captured (another error fails the check), no
    hang, no instance created; then two eager calls of that configuration agree bit for bit;
    status 0."""
    # a capacity no other check uses, so that this configuration has no instance yet
    cap = _default_rows(env) + 1280
    guard_msg = "the first call of a configuration creates its instance"

    def instances():
        return [v for _, v in kf._INSTANCES.matching(env.group) if v.capacity_rows == cap]

    tok = [d.tok[11]]
    w1p, w2p, _ = _graph_params(env, d)
    _status(env)
    before = len(instances())
    mbs = [_Microbatch(*tok[0])]
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    err = None
    try:
        with torch.cuda.graph(graph):
            _layer_calls(env, mbs, [0], w1p, w2p, capacity_rows=cap)
    except Exception as e:  # pylint: disable=broad-except
        err = f"{type(e).__name__}: {e}"
    del graph, mbs
    guard = err is not None and err.startswith("RuntimeError: ") and guard_msg in err
    after_capture = len(instances())
    first = _eager(env, tok, [0], w1p, w2p, capacity_rows=cap)[0]
    second = _eager(env, tok, [0], w1p, w2p, capacity_rows=cap)[0]
    eq = _same(first, second)
    st = _status(env)
    after_eager = len(instances())
    for inst in instances():
        kf._INSTANCES.destroy_instance(inst)
    errors = [None] * env.world
    dist.all_gather_object(errors, err if err is None else err[:200], group=env.group)
    ok = guard and before == 0 and after_capture == 0 and after_eager == 1
    ok &= all(eq.values()) and st == 0
    return ok, {
        "capacity_rows": cap,
        "capture_error_per_rank": errors,
        "guard_error_raised": guard,
        "instances_before": before,
        "instances_after_capture": after_capture,
        "instances_after_eager": after_eager,
        "eager_twice_bitwise": eq,
        "status": st,
    }


CHECKS = {
    "layout": check_layout,
    "numerics": check_numerics,
    "mutation": check_mutation,
    "determinism": check_determinism,
    "two_routings": check_two_routings,
    "three_layers": check_three_layers,
    "inflight_fifo": check_inflight_fifo,
    "inflight_lifo": check_inflight_lifo,
    "nograd_between": check_nograd_between,
    "forward_without_backward": check_forward_without_backward,
    "save_tiers": check_save_tiers,
    "saved_rows_below": check_saved_rows_below,
    "mutation_saved_xp": check_mutation_saved_xp,
    "mutation_saved_y": check_mutation_saved_y,
    "inference_allocates_nothing": check_inference_allocates_nothing,
    "skew_rounds": check_skew_rounds,
    "dropped_ids": check_dropped_ids,
    "degenerate": check_degenerate,
    "create_destroy_twice": check_create_destroy_twice,
    "main_grad_acc": check_main_grad_acc,
    "saved_tensor_hooks": check_saved_tensor_hooks,
    "version_counter": check_version_counter,
    "two_instances": check_two_instances,
    "checkpoint_reentrant": check_checkpoint_reentrant,
    "checkpoint_nonreentrant": check_checkpoint_nonreentrant,
    "symm_vs_own": check_symm_vs_own,
    "te_ep_reference": check_te_ep_reference,
    "graph_one_layer": check_graph_one_layer,
    "graph_two_microbatches": check_graph_two_microbatches,
    "graph_routing": check_graph_routing,
    "graph_save_tiers": check_graph_save_tiers,
    "graph_mutation": check_graph_mutation,
    "graph_first_call_in_capture": check_graph_first_call_in_capture,
}


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return str(o)
    return o


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--test", default="all", help="comma-separated names, or all: " + ",".join(CHECKS)
    )
    p.add_argument("--shape", default="small", choices=sorted(SHAPES))
    p.add_argument("--json", default=None, help="rank 0 appends one JSON line per check")
    args = p.parse_args()
    names = list(CHECKS) if args.test == "all" else args.test.split(",")

    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local))
    env = Env(args.shape)
    if env.world != kf.EP_SIZE:
        raise SystemExit(f"needs {kf.EP_SIZE} ranks, got {env.world}")
    if env.rank == 0:
        mod = kf._ext.load()
        print(
            f"KOSMOS binding: {mod.__name__} from {getattr(mod, '__file__', '?')}; torch"
            f" {torch.__version__}; symmetric memory {ki.resolve_allocator()}",
            flush=True,
        )
    d = Data(env)
    failed = []
    for name in names:
        t0 = time.time()
        try:
            ok, details = CHECKS[name](env, d)
            err = None
        except Exception as e:  # pylint: disable=broad-except
            # identical on every rank for the failures a check provokes on purpose; anything else
            # is reported, and the next check may then hang, which the caller's timeout catches
            ok, details, err = False, {}, f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        flags = [None] * env.world
        dist.all_gather_object(flags, (bool(ok), err), group=env.group)
        passed = all(f[0] for f in flags)
        rec = {
            "check": name,
            "shape": args.shape,
            "passed": passed,
            "seconds": round(time.time() - t0, 1),
            "rank0": _jsonable(details),
            "ranks_failed": [r for r, f in enumerate(flags) if not f[0]],
            "errors": [f[1] for f in flags if f[1]][:1],
        }
        if env.rank == 0:
            print(
                f"KOSMOS_CHECK {name}: {'PASS' if passed else 'FAIL'} ({rec['seconds']} s)",
                flush=True,
            )
            if not passed:
                print(json.dumps(rec, indent=1)[:6000], flush=True)
            if args.json:
                with open(args.json, "a") as f:
                    f.write(json.dumps(rec) + "\n")
        if not passed:
            failed.append(name)
    if env.rank == 0:
        print(
            f"KOSMOS_CHECKS_DONE passed={len(names) - len(failed)}/{len(names)} failed={failed}",
            flush=True,
        )
    kosmos.destroy()
    dist.destroy_process_group()
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
