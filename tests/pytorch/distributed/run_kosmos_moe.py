#!/usr/bin/python3

# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

"""8-GPU checks of te.MoE (KOSMOS: fused router, routed experts and shared expert, one kernel launch for the
forward and one for the backward) against an fp32 torch reference:
torchrun --nproc_per_node 8 run_kosmos_moe.py --router <score> [--shared [--shared-gate]] [--mxfp8] ..."""

import argparse
import gc
import math
import os
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch.module.moe import fc1_from_gate_up, fc1_to_gate_up

EP = 8


def _max_over_ranks(v: float, device) -> float:
    t = torch.tensor([v], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


def rel_err(cand: torch.Tensor, ref: torch.Tensor) -> float:
    """max |c - r| / max |r| on this rank, then the max over the ranks; inf if non-finite."""
    c = cand.float()
    if not torch.isfinite(c).all():
        local = math.inf
    else:
        den = ref.abs().max().item()
        diff = (c - ref).abs().max().item()
        local = diff / den if den > 0 else diff
    return _max_over_ranks(local, cand.device)


def snr_db(cand: torch.Tensor, ref: torch.Tensor) -> float:
    """Signal-to-noise ratio of cand against ref in dB; the worst rank's."""
    c = cand.float()
    if not torch.isfinite(c).all():
        local = -math.inf
    else:
        noise = (c - ref).pow(2).sum().item()
        sig = ref.pow(2).sum().item()
        local = math.inf if noise == 0 else 10 * math.log10(sig / noise)
    return -_max_over_ranks(-local, cand.device)


class Checker:
    """Compares by relative error, or (snr > 0) by SNR; records failures."""

    def __init__(self, tol: float, snr: float):
        self.tol, self.snr, self.failures = tol, snr, []

    def __call__(self, name, cand, ref, snr=None):
        if self.snr > 0:
            lo = self.snr if snr is None else snr
            v = snr_db(cand, ref)
            ok = v >= lo
            msg = f"SNR {v:6.1f} dB (min {lo:g})"
        else:
            v = rel_err(cand, ref)
            ok = v <= self.tol
            msg = f"rel err {v:.3e} (tol {self.tol:g})"
        self.report(name, ok, msg)

    def report(self, name, ok, msg):
        if dist.get_rank() == 0:
            print(f"  {name:20s} {msg} {'ok' if ok else 'FAIL'}", flush=True)
        if not ok:
            self.failures.append(name)


def gather_cat(p: torch.Tensor) -> torch.Tensor:
    parts = [torch.empty_like(p) for _ in range(EP)]
    dist.all_gather(parts, p.detach().contiguous())
    return torch.cat(parts)


class _ClampedGLU(torch.autograd.Function):
    """silu(min(g, 10)) clamp(u, -10, 10), whose backward (as the kernel's) reads G rounded to bf16."""

    @staticmethod
    def forward(ctx, g, u):
        ctx.save_for_backward(g.bfloat16().float(), u.bfloat16().float())
        return F.silu(torch.clamp(g, max=10.0)) * torch.clamp(u, -10.0, 10.0)

    @staticmethod
    def backward(ctx, da):
        g, u = ctx.saved_tensors
        a = torch.clamp(g, max=10.0)
        s = torch.sigmoid(a)
        uc = torch.clamp(u, -10.0, 10.0)
        dg = torch.where(g <= 10.0, da * uc * s * (1 + a * (1 - s)), torch.zeros_like(g))
        du = torch.where(u.abs() <= 10.0, da * a * s, torch.zeros_like(u))
        return dg, du


def act_fn(h: torch.Tensor, I: int, clamp10: bool) -> torch.Tensor:
    g, u = h[:, :I], h[:, I:]
    if clamp10:
        return _ClampedGLU.apply(g, u)
    return F.silu(g) * u


def route_ref(x, Wr, rb, eb, cfg, T):
    """The fused router (Megatron semantics, as KOSMOS's CPU reference): routing, weights and aux loss."""
    E, k, score = Wr.shape[0], cfg["k"], cfg["router"]
    logits = x @ Wr.t()
    if rb is not None:
        logits = logits + rb
    if score == "sigmoid":
        s = torch.sigmoid(logits)
    elif score == "sqrtsoftplus":
        s = torch.sqrt(F.softplus(logits))
    elif score == "softmax_pre":
        s = torch.softmax(logits, dim=1)
    else:
        s = logits
    unit = score in ("sigmoid", "sqrtsoftplus")
    sel = (s + eb if (unit and eb is not None) else s).detach()
    if cfg["groups"]:
        G, gt = cfg["groups"], cfg["group_topk"]
        gs = sel.view(T, G, E // G).topk(k // gt, dim=2).values.sum(2)
        keep = gs.topk(gt, dim=1).indices
        mask = torch.zeros(T, G, dtype=torch.bool, device=x.device).scatter_(1, keep, True)
        sel = sel.masked_fill(~mask.repeat_interleave(E // G, dim=1), -math.inf)
    idx = sel.topk(k, dim=1).indices
    if score == "softmax":
        w = torch.softmax(logits.gather(1, idx), dim=1)
    elif unit and k > 1:
        num = s.gather(1, idx)
        w = num / (num.sum(1, keepdim=True) + 1e-20)
    else:
        w = s.gather(1, idx)
    w = w * cfg["scale"]
    aux = None
    nb = cfg["aux_groups"]
    if nb:
        if score == "softmax_pre":
            p = s
        elif unit:
            p = s / (s.sum(1, keepdim=True) + 1e-20)
        else:
            p = torch.softmax(logits, dim=1)
        am = p.detach().topk(k, dim=1).indices
        b = torch.arange(T, device=x.device) % nb
        cnt = torch.zeros(nb, E, device=x.device)
        ones = torch.ones(T * k, device=x.device)
        cnt.index_put_((b.repeat_interleave(k), am.reshape(-1)), ones, accumulate=True)
        psum = torch.zeros(nb, E, device=x.device).index_add(0, b, p)
        Tg = T // nb
        aux = E * cfg["aux_coeff"] / (k * Tg * Tg * nb) * (psum * cnt).sum()
    tpe = torch.bincount(idx.reshape(-1), minlength=E).to(torch.int32)
    return idx, w, aux, tpe


def experts_ref(x, idx, w, W1, W2, I, clamp10):
    """Routed experts: out[t] = sum_j w[t, j] Expert_{idx[t, j]}(x[t])."""
    out = torch.zeros_like(x)
    for e in range(W1.shape[0]):
        t, j = (idx == e).nonzero(as_tuple=True)
        if t.numel() == 0:
            continue
        y = act_fn(x[t] @ W1[e].t(), I, clamp10) @ W2[e].t()
        out = out.index_add(0, t, w[t, j].unsqueeze(1) * y)
    return out


def gpu_kernels(fn):
    """The device activities (kernels, copies, fills) fn issues."""
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        res = fn()
        torch.cuda.synchronize()
    return res, [e.name for e in prof.events() if e.device_type == DeviceType.CUDA]


def check_main_grad(args, moe, inputs, run, backward, leaves, chk, dev):
    """fuse_wgrad_accumulation: the backward kernel adds each weight gradient into main_grad itself (fp32, or bf16
    rounded once per element), still one launch each way. Without rounds the adds are bitwise those of the kernel's
    fp32 tiles: into zeros they round to the plain backward's bf16 gradients, and two backwards into a random
    main_grad equal the same two adds in torch. With rounds (an expert's rows split over passes, each pass added on
    its own) they are compared by error. A frozen weight's main_grad is left alone."""
    mdt = torch.float32 if args.main_grad == "fp32" else torch.bfloat16
    exact = args.cap_frac >= 1
    names = [n for n, _ in moe.named_parameters()]
    params = dict(moe.named_parameters())
    case = inputs()
    plain = run(*case)[3]  # the bf16 gradients of the plain backward
    moe.fuse_wgrad_accumulation = True
    gen = torch.Generator(device=dev).manual_seed(11)

    def set_main(zero, dtype=mdt):
        for n in names:
            p = params[n]
            p.main_grad = (torch.zeros if zero else torch.randn)(
                p.shape, dtype=torch.float32, device=dev, **({} if zero else {"generator": gen})
            ).to(dtype)
            p.grad_added_to_main_grad = False

    def bwd_once():
        xg, sgg = leaves(case[0], case[1])
        out, aux, _ = moe(xg, shared_gate=sgg)
        backward(out, aux, case[2], case[3])

    # Into fp32 zeros: the kernel's fp32 tile sums. Into main_grad of the checked dtype: rounded once (bf16).
    set_main(True, torch.float32)
    bwd_once()
    tiles = {n: params[n].main_grad.clone() for n in names}
    set_main(True)
    bwd_once()
    ok = all(params[n].grad_added_to_main_grad for n in names)
    chk.report("main_grad: grad_added_to_main_grad", _max_over_ranks(0.0 if ok else 1.0, dev) == 0,
               "set on every weight")
    for n in names:
        got = params[n].main_grad
        if exact:
            lo = torch.bfloat16 if torch.bfloat16 in (mdt, plain[n].dtype) else torch.float32
            same = torch.equal(got, tiles[n].to(mdt)) and torch.equal(got.to(lo), plain[n].to(lo))
            chk.report(f"main_grad {n} from 0", _max_over_ranks(0.0 if same else 1.0, dev) == 0,
                       "bitwise the plain gradient")
        else:
            chk(f"main_grad {n} from 0", got.float(), plain[n].float())

    # Two backwards into a random main_grad: the same adds in torch.
    set_main(False)
    start = {n: params[n].main_grad.clone() for n in names}
    bwd_once()
    bwd_once()
    for n in names:
        want = start[n]
        for _ in range(2):
            want = (want.float() + tiles[n].float()).to(mdt)
        got = params[n].main_grad
        if exact:
            same = torch.equal(got, want)
            chk.report(f"main_grad {n} += 2 calls", _max_over_ranks(0.0 if same else 1.0, dev) == 0,
                       "bitwise the same adds in torch")
        else:
            chk(f"main_grad {n} += 2 calls", (got.float() - start[n].float()), 2 * tiles[n].float())

    # Still one launch each way, nothing else on the device.
    set_main(True)
    xg, sgg = leaves(case[0], case[1])
    (out, aux, _), kf = gpu_kernels(lambda: moe(xg, shared_gate=sgg))
    _, kb = gpu_kernels(lambda: backward(out, aux, case[2], case[3]))
    extra = [n for n in kf if "moe_fwd" not in n] + [n for n in kb if "moe_bwd" not in n]
    bad = 1.0 if extra or len(kf) > 1 or len(kb) > 1 else 0.0
    if bad:
        print(f"[main_grad launches] forward {kf}, backward {kb}", flush=True)
    chk.report("main_grad: one launch each", _max_over_ranks(bad, dev) == 0, "no extra device work")

    # A frozen weight (fc2): its main_grad untouched, the others still accumulated.
    set_main(False)
    before = {n: params[n].main_grad.clone() for n in names}
    moe.fc2_weight.requires_grad_(False)
    try:
        bwd_once()
    finally:
        moe.fc2_weight.requires_grad_(True)
    same = torch.equal(moe.fc2_weight.main_grad, before["fc2_weight"]) and not torch.equal(
        moe.fc1_weight.main_grad, before["fc1_weight"]
    )
    chk.report("main_grad: frozen fc2", _max_over_ranks(0.0 if same else 1.0, dev) == 0, "untouched, fc1 added")

    # Refused before the launch (ValueError), the forward's saved state then released with the graph: a weight
    # without main_grad, and main_grad of mixed dtypes. Then a normal backward runs.
    for tag, setup in (
        ("no main_grad", lambda: delattr(moe.fc2_weight, "main_grad")),
        ("mixed dtypes", lambda: setattr(moe.fc2_weight, "main_grad", moe.fc2_weight.main_grad.to(
            torch.bfloat16 if mdt == torch.float32 else torch.float32))),
    ):
        set_main(True)
        setup()
        xg, sgg = leaves(case[0], case[1])
        out, aux, _ = moe(xg, shared_gate=sgg)
        try:
            backward(out, aux, case[2], case[3])
            refused = False
        except ValueError:
            refused = True
        del out, aux
        chk.report(f"main_grad: {tag} refused", refused, "ValueError")
    set_main(True)
    bwd_once()
    same = not exact or all(torch.equal(params[n].main_grad, tiles[n].to(mdt)) for n in names)
    chk.report("main_grad: backward after refusals", _max_over_ranks(0.0 if same else 1.0, dev) == 0, "as before")

    moe.fuse_wgrad_accumulation = False
    for n in names:
        del params[n].main_grad
        del params[n].grad_added_to_main_grad


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--router", default="softmax", choices=["softmax", "sigmoid", "sqrtsoftplus", "softmax_pre"])
    parser.add_argument("--groups", type=int, default=0)
    parser.add_argument("--group-topk", type=int, default=0)
    parser.add_argument("--aux-groups", type=int, default=1)
    parser.add_argument("--router-bias", action="store_true")
    parser.add_argument("--expert-bias", action="store_true")
    parser.add_argument("--shared", action="store_true")
    parser.add_argument("--shared-gate", action="store_true")
    parser.add_argument("--mxfp8", action="store_true")
    parser.add_argument("--fp8-combine", action="store_true")
    parser.add_argument("--clamp10", action="store_true")
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--ffn", type=int, default=256)
    parser.add_argument("--experts", type=int, default=16)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--tol", type=float, default=2e-2)
    parser.add_argument("--snr", type=float, default=22.0, help="mxfp8: minimum SNR (dB) vs the bf16 reference")
    parser.add_argument("--cap-frac", type=float, default=1.0, help="capacity / (T k); < 1 runs calls in rounds")
    parser.add_argument("--layers", type=int, default=1, help="> 1: also run that many modules on one workspace")
    parser.add_argument("--train-steps", type=int, default=0, help="> 0: also run that many SGD steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="SGD learning rate of --train-steps")
    parser.add_argument("--ckpt-steps", type=int, default=0, help="> 0: steps under non-reentrant checkpointing")
    parser.add_argument("--abandon-steps", type=int, default=0, help="> 0: steps that each discard a forward")
    parser.add_argument("--no-release", action="store_true", help="negative control: dropped forwards not released")
    parser.add_argument("--main-grad", choices=["fp32", "bf16"], help="also check fuse_wgrad_accumulation")
    args = parser.parse_args()
    if args.mxfp8 and args.clamp10:
        raise SystemExit("MXFP8 with the clamped SwiGLU: fp8 rounding moves many elements across the clamp limit, so a"
                         " bf16 reference is not meaningful; this combination is checked bitwise against KOSMOS")

    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    dist.init_process_group("nccl", device_id=dev)
    rank = dist.get_rank()
    if dist.get_world_size() != EP:
        raise SystemExit(f"needs {EP} ranks")
    H, I, E, k, T = args.hidden, args.ffn, args.experts, args.topk, args.tokens
    E_loc = E // EP
    cap = max(256, int(args.cap_frac * T * k) // 256 * 256) + (256 * E_loc if args.cap_frac >= 1 else 0)
    if args.mxfp8 and args.cap_frac >= 1:
        # MXFP8 with I < 1024 pads every (source, expert) segment to 32 rows: room for a call in one pass.
        cap = 2 * T * k
    rcfg = dict(
        k=k,
        router=args.router,
        groups=args.groups,
        group_topk=args.group_topk,
        aux_groups=args.aux_groups,
        aux_coeff=1e-2,
        scale=1.0,
    )
    torch.cuda.synchronize()
    alloc0, free0 = torch.cuda.memory_allocated(), torch.cuda.mem_get_info()[0]
    ws = te.MoEWorkspace(
        hidden_size=H,
        ffn_hidden_size=I,
        num_experts=E,
        topk=k,
        num_tokens=T,
        ep_group=dist.group.WORLD,
        capacity_rows=cap,
        router=args.router,
        router_groups=args.groups,
        router_group_topk=args.group_topk,
        aux_loss_groups=args.aux_groups,
        aux_loss_coeff=rcfg["aux_coeff"],
        shared_expert=args.shared,
        pool_bytes=4 * cap * 2 * H * 4,
        activation="swiglu_clamp10" if args.clamp10 else "swiglu",
        mxfp8=args.mxfp8,
        fp8_combine=args.fp8_combine,
    )
    torch.manual_seed(1234 + rank)
    wscale = 0.5 if args.clamp10 else 0.05

    def init(t):
        return torch.nn.init.normal_(t, 0, wscale)

    def make_moe():
        m = te.MoE(
            ws,
            init_method=init,
            output_layer_init_method=init,
            add_router_bias=args.router_bias,
            use_expert_bias=args.expert_bias,
        )
        with torch.no_grad():
            # The router and the shared expert are replicated: every rank holds rank 0's.
            for name in ("router_weight", "shared_fc1_weight", "shared_fc2_weight"):
                if getattr(m, name, None) is not None:
                    dist.broadcast(getattr(m, name), 0)
            if m.router_bias is not None:
                m.router_bias.normal_(0, 0.5)
                dist.broadcast(m.router_bias, 0)
            if m.expert_bias is not None:
                m.expert_bias.normal_(0, 0.1)
                dist.broadcast(m.expert_bias, 0)
        m.update_weights()
        return m

    moe = make_moe()
    gen = torch.Generator().manual_seed(77 + rank)
    # clamp10: elements next to the clamp limit may take either mask after the kernel's bf16 rounding of G
    # (KOSMOS's own reference treats them as ambiguous), so compare by SNR, which isolated flips barely move.
    chk = Checker(args.tol, args.snr if args.mxfp8 else 30.0 if args.clamp10 else 0.0)
    if rank == 0:
        feats = [args.router] + [f for f, on in (("shared", args.shared), ("mxfp8", args.mxfp8)) if on]
        print(f"router + experts ({', '.join(feats)}): H {H} I {I} E {E} k {k} T {T} cap {cap}", flush=True)

    def inputs():
        x = torch.randn(T, H, generator=gen).to(dev, torch.bfloat16)
        sg = torch.sigmoid(torch.randn(T, generator=gen)).to(dev) if args.shared_gate else None
        dout = torch.randn(T, H, generator=gen).to(dev, torch.bfloat16)
        daux = (torch.randn(1, generator=gen).abs() * 10).to(dev)
        return x, sg, dout, daux

    def leaves(x, sg, m=None):
        for p in (m or moe).parameters():
            p.grad = None
        return x.detach().requires_grad_(), None if sg is None else sg.detach().requires_grad_()

    def backward(out, aux, dout, daux):
        if aux is not None:
            torch.autograd.backward([out, aux], [dout, daux])
        else:
            out.backward(dout)

    def grads(xg, sgg, m=None):
        g = {"dx": xg.grad, "dsgate": None if sgg is None else sgg.grad}
        for name, p in (m or moe).named_parameters():
            g[name] = p.grad
        return g

    def run(x, sg, dout, daux, m=None):
        m = m or moe
        xg, sgg = leaves(x, sg, m)
        out, aux, tpe = m(xg, shared_gate=sgg)
        backward(out, aux, dout, daux)
        return out, aux, tpe, grads(xg, sgg, m)

    def reference(x, sg, dout, daux, m=None):
        """fp32 reference with every rank's experts; expert weight gradients reduced over the ranks."""
        mm = m or moe
        xr = x.float().requires_grad_()
        W1 = fc1_to_gate_up(gather_cat(mm.fc1_weight).float()).requires_grad_()
        W2 = gather_cat(mm.fc2_weight).float().requires_grad_()
        Wr = mm.router_weight.detach().float().requires_grad_()
        lv = {"fc1_weight": W1, "fc2_weight": W2, "router_weight": Wr}
        rb = None
        if mm.router_bias is not None:
            rb = mm.router_bias.detach().float().requires_grad_()
            lv["router_bias"] = rb
        idx, w, aux, tpe = route_ref(xr, Wr, rb, mm.expert_bias, rcfg, T)
        out = experts_ref(xr, idx, w, W1, W2, I, args.clamp10)
        sgr = None
        if args.shared:
            S1 = mm.shared_fc1_weight.detach().float().requires_grad_()
            S2 = mm.shared_fc2_weight.detach().float().requires_grad_()
            lv["shared_fc1_weight"], lv["shared_fc2_weight"] = S1, S2
            ys = act_fn(xr @ S1.t(), I, args.clamp10) @ S2.t()
            if sg is not None:
                sgr = sg.float().requires_grad_()
                ys = ys * sgr.unsqueeze(1)
            out = out + ys
        loss = (out * dout.float()).sum()
        if aux is not None:
            loss = loss + aux * daux.float().squeeze()
        loss.backward()
        g = {"dx": xr.grad, "dsgate": None if sgr is None else sgr.grad}
        for name, leaf in lv.items():
            gr = leaf.grad
            if name in ("fc1_weight", "fc2_weight"):
                dist.all_reduce(gr)
                if name == "fc1_weight":
                    gr = fc1_from_gate_up(gr)
                gr = gr[rank * E_loc : (rank + 1) * E_loc]
            g[name] = gr
        return out.detach(), None if aux is None else aux.detach(), tpe, g

    def compare(tag, case, m=None):
        if rank == 0:
            print(f"[{tag}]", flush=True)
        out, aux, tpe, g = run(*case, m=m)
        r_out, r_aux, r_tpe, rg = reference(*case, m=m)
        chk("out", out, r_out)
        nflip = _max_over_ranks(float((tpe != r_tpe).sum().item()), dev)
        chk.report("tokens_per_expert", nflip == 0, "equal" if nflip == 0 else f"{int(nflip)} experts differ")
        if r_aux is not None:
            chk("aux_loss", aux.reshape(()), r_aux)
        for name, ref in rg.items():
            if ref is not None:
                # MXFP8: the logit-bias gradient sums every token's noisy routing-weight gradient.
                lo = args.snr - 5 if (args.mxfp8 and name == "router_bias") else None
                chk(name, g[name], ref, snr=lo)

    compare("forward and backward", inputs())

    # Exactly one kernel launch for the forward and one for the backward; nothing else on the device. The profiler
    # can drop an activity record now and then: a missing kernel is measured again, any extra one fails at once.
    verdict = None
    for _ in range(3):
        x, sg, dout, daux = inputs()
        xg, sgg = leaves(x, sg)
        (out, aux, _), kf = gpu_kernels(lambda: moe(xg, shared_gate=sgg))
        _, kb = gpu_kernels(lambda: backward(out, aux, dout, daux))
        extra = [n for n in kf if "moe_fwd" not in n] + [n for n in kb if "moe_bwd" not in n]
        exact = len(kf) == 1 and len(kb) == 1 and not extra
        # Per rank: 2 extra device work, 1 a record missing, 0 exact; every rank's worst.
        state = _max_over_ranks(2.0 if extra or len(kf) > 1 or len(kb) > 1 else 0.0 if exact else 1.0, dev)
        if rank == 0 or state:
            print(f"[launches] rank {rank}: forward {[n[:60] for n in kf]}, backward {[n[:60] for n in kb]}",
                  flush=True)
        if state != 1:
            verdict = state == 0
            break
    chk.report("one launch each", bool(verdict), "forward 1, backward 1")

    # Two forwards in flight (a pipeline schedule), backwards in reverse order: out and dx bitwise equal to
    # each call run alone.
    a, b = inputs(), inputs()
    alone = [run(*c) for c in (a, b)]
    res = []
    for c in (a, b):
        xg, sgg = leaves(c[0], c[1])
        out, aux, _ = moe(xg, shared_gate=sgg)
        res.append((out, aux, xg))
    backward(res[1][0], res[1][1], b[2], b[3])
    backward(res[0][0], res[0][1], a[2], a[3])
    same = all(
        torch.equal(o.detach(), al[0]) and torch.equal(xg.grad, al[3]["dx"]) for (o, _, xg), al in zip(res, alone)
    )
    chk.report("in flight (LIFO)", _max_over_ranks(0.0 if same else 1.0, dev) == 0, "bitwise equal to alone")

    # Recompute: a no-grad forward (no saved state), then the same call with grad: bitwise equal outputs.
    case = inputs()
    with torch.no_grad():
        out_ng = moe(case[0], shared_gate=case[1])[0].clone()
    out_g = run(*case)[0]
    same = torch.equal(out_ng, out_g)
    chk.report("no-grad forward", _max_over_ranks(0.0 if same else 1.0, dev) == 0, "bitwise equal to training")

    # Determinism: the same call twice, bitwise equal outputs and gradients.
    case = inputs()
    r1, r2 = run(*case), run(*case)
    same = torch.equal(r1[0], r2[0]) and all(
        (r1[3][n] is None and r2[3][n] is None) or torch.equal(r1[3][n], r2[3][n]) for n in r1[3]
    )
    chk.report("determinism", _max_over_ranks(0.0 if same else 1.0, dev) == 0, "bitwise equal on every rank")

    if args.main_grad:
        check_main_grad(args, moe, inputs, run, backward, leaves, chk, dev)

    if args.layers > 1:
        # Several modules on one workspace (one pool, one set of call buffers), as the layers of a model: a
        # Megatron schedule (all forwards, then the backwards in reverse) and a random interleaving of the
        # layers' forward -> backward pairs; each layer's out and gradients bitwise equal to the layer alone.
        mods = [moe] + [make_moe() for _ in range(args.layers - 1)]
        cases = [inputs() for _ in mods]
        alone = [run(*c, m=mm) for c, mm in zip(cases, mods)]

        def check_against_alone(tag, results):
            same = all(
                torch.equal(r[0], al[0])
                and all((r[1][n] is None and al[3][n] is None) or torch.equal(r[1][n], al[3][n]) for n in r[1])
                for r, al in zip(results, alone)
            )
            chk.report(tag, _max_over_ranks(0.0 if same else 1.0, dev) == 0, "bitwise equal to each layer alone")

        fwd = []
        for c, mm in zip(cases, mods):
            xg, sgg = leaves(c[0], c[1], mm)
            out, aux, _ = mm(xg, shared_gate=sgg)
            fwd.append((out, aux, xg, sgg))
        results = [None] * len(mods)
        for i in reversed(range(len(mods))):
            out, aux, xg, sgg = fwd[i]
            backward(out, aux, cases[i][2], cases[i][3])
            results[i] = (out.detach(), grads(xg, sgg, mods[i]))
        check_against_alone(f"{len(mods)} layers, megatron", results)
        order = torch.randperm(2 * len(mods), generator=torch.Generator().manual_seed(5)).tolist()
        steps = []
        for o in order:  # a random merge of the layers' [forward, backward] sequences, same on every rank
            steps.append(o % len(mods))
        pending, results = {}, [None] * len(mods)
        for i in steps:
            if i not in pending and results[i] is None:
                xg, sgg = leaves(cases[i][0], cases[i][1], mods[i])
                out, aux, _ = mods[i](xg, shared_gate=sgg)
                pending[i] = (out, aux, xg, sgg)
            elif i in pending:
                out, aux, xg, sgg = pending.pop(i)
                backward(out, aux, cases[i][2], cases[i][3])
                results[i] = (out.detach(), grads(xg, sgg, mods[i]))
        for i, (out, aux, xg, sgg) in pending.items():
            backward(out, aux, cases[i][2], cases[i][3])
            results[i] = (out.detach(), grads(xg, sgg, mods[i]))
        check_against_alone(f"{len(mods)} layers, interleaved", results)
        compare(f"layer {len(mods) - 1} of {len(mods)} vs reference", inputs(), m=mods[-1])

    if args.train_steps:
        # SGD steps (MXFP8: update_weights after each): every step's forward and gradients vs the reference
        # at the current weights.
        for step in range(args.train_steps):
            compare(f"train step {step}", inputs())
            with torch.no_grad():
                for p in moe.parameters():
                    if p.grad is not None:
                        p.add_(p.grad.to(p.dtype), alpha=-args.lr)
            moe.update_weights()
            if args.mxfp8:
                # update_weights quantized the current weights: a fresh quantization is bitwise the same.
                kept = moe._qweights[1]
                moe.update_weights()
                again = moe._qweights[1]
                same = all(
                    a is None
                    or all(torch.equal(getattr(a, f), getattr(b, f)) for f in
                           ("_rowwise_data", "_rowwise_scale_inv", "_columnwise_data", "_columnwise_scale_inv"))
                    for a, b in zip(kept, again)
                )
                chk.report(f"step {step} weights fresh", _max_over_ranks(0.0 if same else 1.0, dev) == 0,
                           "re-quantization bitwise equal")
                w = moe.fc1_weight.float()
                if rank == 0:
                    print(f"  fc1 after step: std {w.std().item():.4f} max|w| {w.abs().max().item():.3f}", flush=True)

    if args.no_release:
        from transformer_engine.pytorch.module import moe as moe_mod  # pylint: disable=import-outside-toplevel

        moe_mod._Ticket.__del__ = lambda self: None

    def handles(m=None):
        m = m or moe
        n = len(m._free) + len(m._busy)
        lo = -_max_over_ranks(-float(n), dev)
        return n, lo == _max_over_ranks(float(n), dev)

    def kept_less(st):
        # A forward keeping Xp only or nothing for lack of pool room; a call in rounds (8) keeps nothing by design.
        return bool(st & (16 | 32)) and not st & 8

    if args.ckpt_steps:
        # Non-reentrant activation checkpointing: the first forward saves (its graph is discarded on the
        # backward), the recompute saves again; each step bitwise equal to the step without checkpointing, the
        # handles bounded and alike on every rank, and the pool never full (no forward keeping less).
        from torch.utils.checkpoint import checkpoint  # pylint: disable=import-outside-toplevel

        moe.status()
        most, bad, why = 0, False, set()
        for step in range(args.ckpt_steps):
            case = inputs()
            ref = run(*case)
            xg, sgg = leaves(case[0], case[1])
            out, aux, _ = checkpoint(lambda x, g_: moe(x, shared_gate=g_), xg, sgg, use_reentrant=False)
            backward(out, aux, case[2], case[3])
            g = grads(xg, sgg)
            same = torch.equal(out.detach(), ref[0]) and all(
                (g[n] is None and ref[3][n] is None) or torch.equal(g[n], ref[3][n]) for n in g
            )
            if _max_over_ranks(0.0 if same else 1.0, dev) > 0:
                why.add("not bitwise")
            n, alike = handles()
            most = max(most, n)
            if not alike:
                why.add("handles differ across ranks")
            st = moe.status()
            if os.environ.get("KMOE_DEBUG") == "1" and rank == 0:
                pi = moe.workspace.pool_info()
                print(f"  ckpt step {step}: status {st} handles {n} busy {len(moe._busy)} pool {pi}", flush=True)
            if kept_less(st):
                why.add(f"pool kept less (status {st})")
        bad = bool(why)
        if rank == 0 and why:
            print(f"  checkpoint: {sorted(why)}", flush=True)
        ok = not bad and most <= 4
        chk.report(f"checkpoint x{args.ckpt_steps}", ok, f"bitwise, handles <= {most} alike, pool never full")

    if args.abandon_steps:
        # Forwards whose graphs are discarded without a backward, each rank dropping its own later (rank r holds
        # one r % 3 more steps): released once every rank dropped it. Every step's normal call stays exact.
        held, most, why = [], 0, set()
        moe.status()
        dbg = os.environ.get("KMOE_DEBUG") == "1"

        def trace(msg):
            if dbg:
                st = [l.release_state() for l in moe._busy]
                print(f"[r{rank}] {msg} busy={len(moe._busy)} {st} free={len(moe._free)}", file=sys.stderr, flush=True)

        for step in range(args.abandon_steps):
            xg, sgg = leaves(*inputs()[:2])
            trace(f"step {step} abandon-fwd")
            held.append((step, moe(xg, shared_gate=sgg)))
            held = [(s0, o) for s0, o in held if step - s0 < rank % 3]
            case = inputs()
            trace(f"step {step} run1")
            r1 = run(*case)
            trace(f"step {step} run2")
            r2 = run(*case)
            trace(f"step {step} done")
            same = torch.equal(r1[0], r2[0]) and all(
                (r1[3][n] is None and r2[3][n] is None) or torch.equal(r1[3][n], r2[3][n]) for n in r1[3]
            )
            if _max_over_ranks(0.0 if same else 1.0, dev) > 0:
                why.add("not deterministic")
            n, alike = handles()
            most = max(most, n)
            if not alike:
                why.add("handles differ across ranks")
            st = moe.status()
            if kept_less(st):
                why.add(f"pool kept less (status {st})")
        bad = bool(why)
        if rank == 0 and why:
            print(f"  abandon: {sorted(why)}", flush=True)
        held.clear()
        for _ in range(2 * 4):
            run(*inputs())
        moe.workspace._ws.release_sync()
        run(*inputs())
        n, alike = handles()
        busy = int(_max_over_ranks(float(len(moe._busy)), dev))
        compare("after the abandoned forwards", inputs())
        ok = not bad and alike and busy == 0 and most <= 8
        chk.report(f"abandon x{args.abandon_steps}", ok, f"handles <= {most} alike, all reclaimed (busy {busy})")

    # Saved-tensor hooks: save_on_cpu offloads what autograd saved (x, the gate, Gs; the pool stays on the device):
    # bitwise equal to the plain call.
    case = inputs()
    plain = run(*case)
    with torch.autograd.graph.save_on_cpu(pin_memory=True):
        xg, sgg = leaves(case[0], case[1])
        out, aux, _ = moe(xg, shared_gate=sgg)
    backward(out, aux, case[2], case[3])
    got = grads(xg, sgg)
    same = torch.equal(out.detach(), plain[0]) and all(
        (got[n] is None and plain[3][n] is None) or torch.equal(got[n], plain[3][n]) for n in got
    )
    chk.report("save_on_cpu", _max_over_ranks(0.0 if same else 1.0, dev) == 0, "bitwise equal to the plain call")

    # Memory: every byte the fused MoE allocates (workspace buffers, the symmetric arenas, the saved-state pool,
    # layer handles) comes from PyTorch's caching allocator, so torch counts it.
    torch.cuda.synchronize()
    untracked = tex.moe_untracked_bytes()
    grew = torch.cuda.memory_allocated() - alloc0
    pool = ws.pool_info()["bytes"]
    if rank == 0:
        print(f"[memory] torch allocated +{grew / 2**20:.0f} MiB (pool {pool / 2**20:.0f} MiB); device used "
              f"+{(free0 - torch.cuda.mem_get_info()[0]) / 2**20:.0f} MiB; outside torch {untracked} B", flush=True)
    ok = untracked == 0 and grew >= pool > 0
    chk.report("memory visible to torch", _max_over_ranks(0.0 if ok else 1.0, dev) == 0,
               "nothing outside the caching allocator")

    # Refusals: a second backward of one forward (retain_graph=True), a weight changed in place between a forward and
    # its backward, and, with MXFP8, a forward after a weight change without update_weights.
    xg, sgg = leaves(case[0], case[1])
    out, aux, _ = moe(xg, shared_gate=sgg)
    if aux is not None:
        torch.autograd.backward([out, aux], [case[2], case[3]], retain_graph=True)
    else:
        out.backward(case[2], retain_graph=True)
    try:
        backward(out, aux, case[2], case[3])
        refused = False
    except RuntimeError:
        refused = True
    chk.report("second backward refused", refused, "RuntimeError")
    xg, sgg = leaves(case[0], case[1])
    out, aux, _ = moe(xg, shared_gate=sgg)
    w2 = moe.fc2_weight.detach().clone()
    with torch.no_grad():
        moe.fc2_weight.mul_(2.0)
    try:
        backward(out, aux, case[2], case[3])
        refused = False
    except RuntimeError:
        refused = True
    del out, aux
    with torch.no_grad():
        moe.fc2_weight.copy_(w2)
    moe.update_weights()
    chk.report("weight changed before backward refused", refused, "RuntimeError")
    if args.mxfp8:
        with torch.no_grad():
            moe.fc1_weight.mul_(1.0)
        try:
            moe(inputs()[0])
            refused = False
        except RuntimeError:
            refused = True
        chk.report("stale MXFP8 refused", refused, "RuntimeError")
        moe.update_weights()

    # A fresh module: its handle is set up at construction, so its very first forward and backward are one launch
    # each too. Its expert weights carry allreduce=False; its router and shared weights start as EP rank 0's (each
    # rank's seed differs, and this module is not broadcast by the test).
    fresh = te.MoE(ws, init_method=init, output_layer_init_method=init, add_router_bias=args.router_bias,
                   use_expert_bias=args.expert_bias)
    case = inputs()
    xg, sgg = leaves(case[0], case[1], fresh)
    (out, aux, _), kf = gpu_kernels(lambda: fresh(xg, shared_gate=sgg))
    _, kb = gpu_kernels(lambda: backward(out, aux, case[2], case[3]))
    extra = [n for n in kf if "moe_fwd" not in n] + [n for n in kb if "moe_bwd" not in n]
    bad = 1.0 if extra or len(kf) > 1 or len(kb) > 1 else 0.0
    if bad:
        print(f"[first call] rank {rank}: forward {kf}, backward {kb}", flush=True)
    chk.report("first call of a new module", _max_over_ranks(bad, dev) == 0, "one launch each")
    same = fresh.fc1_weight.allreduce is False and fresh.fc2_weight.allreduce is False
    for name in ("router_weight", "router_bias", "shared_fc1_weight", "shared_fc2_weight"):
        p = getattr(fresh, name, None)
        if p is not None:
            v = p.detach().float().sum().reshape(1)
            hi, lo = v.clone(), v.clone()
            dist.all_reduce(hi, op=dist.ReduceOp.MAX)
            dist.all_reduce(lo, op=dist.ReduceOp.MIN)
            same = same and bool(hi.item() == lo.item())
    chk.report("expert params marked, replicated equal", _max_over_ranks(0.0 if same else 1.0, dev) == 0,
               "fc1/fc2 allreduce=False; router and shared EP rank 0's")
    # Modules dropped by Python (forward and backward run, then deleted): finalize still destroys their handles.
    for _ in range(3):
        m = te.MoE(ws, init_method=init, output_layer_init_method=init, add_router_bias=args.router_bias,
                   use_expert_bias=args.expert_bias)
        run(*inputs(), m=m)
        del m
    del fresh, out, aux, xg, sgg
    gc.collect()

    status = moe.status()
    if rank == 0:
        print(f"status bits {status}; handles {len(moe._free) + len(moe._busy)}, busy {len(moe._busy)}", flush=True)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    pool = ws.pool_info()["bytes"]
    try:
        ws.finalize()
        done = True
    except RuntimeError as e:
        print(f"[finalize] rank {rank}: {e}", flush=True)
        done = False
    torch.cuda.synchronize()
    freed = before - torch.cuda.memory_allocated()
    ok = done and freed >= pool and tex.moe_untracked_bytes() == 0
    if rank == 0:
        print(f"[finalize] freed {freed / 2**20:.0f} MiB (pool {pool / 2**20:.0f} MiB)", flush=True)
    chk.report("finalize with dropped modules", _max_over_ranks(0.0 if ok else 1.0, dev) == 0,
               "every handle destroyed, memory back to torch")
    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print("KOSMOS MoE CHECK " + ("FAIL: " + ", ".join(chk.failures) if chk.failures else "PASS"), flush=True)
    return 1 if chk.failures else 0


if __name__ == "__main__":
    sys.exit(main())
