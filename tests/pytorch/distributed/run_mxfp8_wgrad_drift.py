# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
"""Does the MXFP8 wgrad copy add error that ACCUMULATES over training steps?

The byte-level checks prove the column-scaled copy of dY is the tensor we intended to produce.
They say nothing about training: they are one step, and they compare against a reference that
shares the same lossy behaviour. This runner answers the other question -- whether the error a
scheme introduces per step compounds over many steps -- which is what arXiv:2511.02302 Sec 4.1
establishes for its recipe by training 16B parameters for 200B tokens and overlaying the loss
curves. This is the cheap analogue: a few hundred SGD steps on one row-parallel Linear.

WHAT IS MEASURED

    d(n) = || W_fp8(n) - W_bf16(n) ||_F / || W_bf16(n) ||_F        (global, over all ranks)

one trajectory per process, against a bf16 reference trained in lockstep on identical data from
identical initial weights. Both arms take the same optimizer steps; only the precision of the
layer differs.

WHY THE REFERENCE IS BF16 AND NOT THE OTHER MXFP8 MODE

    The question is whether deriving the column-scaled copy is worse than gathering it. But two
    MXFP8 trajectories that quantize dY differently diverge from each other chaotically -- any
    step-1 difference is amplified by the optimizer -- so || W_inflight - W_gather || grows
    whether or not either is worse. It measures chaos, not quality.

    bf16 is a fixed point both can be measured against. Each mode gets its own d(n), and the
    comparison is between those two curves. MXFP8's inherent drift from bf16 is common to both
    and cancels out of that comparison.

WHY ONE MODE PER PROCESS

    NVTE_AG_WGRAD_COPY is latched on first use, in C++ and in Python independently, so a process
    runs exactly one mode for its whole life. The caller runs this script once per mode and
    compares the two reported curves; see test_mxfp8_wgrad_drift.py.

    torchrun --nproc_per_node=4 run_mxfp8_wgrad_drift.py --steps 200
"""
import argparse
import os
import sys
from contextlib import nullcontext

import torch
import torch.distributed as dist

import transformer_engine.pytorch as te
import transformer_engine.pytorch.cpp_extensions as tex
from transformer_engine.common.recipe import MXFP8BlockScaling, Format
from torch.utils.cpp_extension import IS_HIP_EXTENSION

# Same guard run_layer_with_overlap.py applies: without multicast support Userbuffers refuses to
# initialise unless told to fall back to IPC.
if IS_HIP_EXTENSION:
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    if not tex.device_supports_multicast():
        os.environ["UB_SKIPMC"] = "1"


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seq-length", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-heads", type=int, default=32)
    p.add_argument("--head-dim", type=int, default=48)
    p.add_argument("--in-features", type=int, default=6144)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fp8-format", type=str.lower, default="e4m3", choices=["e4m3", "hybrid"],
                   help="hybrid puts the grad output in E5M2, which the transpose barely loses on")
    p.add_argument(
        "--grad-scale-binades",
        type=int,
        default=6,
        help=(
            "Spread the per-token grad-output scales over this many binades. At 0 every MXFP8 "
            "block lands on the same scale, no element is ever rescaled, and the run proves "
            "nothing about the derivation."
        ),
    )
    p.add_argument("--report-every", type=int, default=0,
                   help="also print a per-step DRIFT line every N steps (0 = summary only)")
    return p.parse_args(argv)


def _global_rel_distance(test_params, ref_params):
    """|| W_test - W_ref ||_F / || W_ref ||_F, reduced over ranks.

    Squared norms are summed across ranks before the square root, so this is the Frobenius norm
    of the whole distributed weight, not a per-shard average.
    """
    num = torch.zeros((), dtype=torch.float64, device="cuda")
    den = torch.zeros((), dtype=torch.float64, device="cuda")
    for t, r in zip(test_params, ref_params):
        d = (t.detach().to(torch.float64) - r.detach().to(torch.float64))
        num += (d * d).sum()
        den += (r.detach().to(torch.float64) ** 2).sum()
    dist.all_reduce(num, op=dist.ReduceOp.SUM)
    dist.all_reduce(den, op=dist.ReduceOp.SUM)
    return (num.sqrt() / den.sqrt().clamp_min(1e-30)).item()


def _train(opts):
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", rank=rank, world_size=world)
    tp_group = dist.new_group(backend="nccl")

    hidden = opts.num_heads * opts.head_dim
    in_local = opts.in_features // world
    recipe = MXFP8BlockScaling(
        fp8_format=Format.HYBRID if opts.fp8_format == "hybrid" else Format.E4M3
    )

    te.module.base.initialize_ub(
        [opts.seq_length * opts.batch_size, hidden],
        world,
        quantization_modes=[te.module.base.UserBufferQuantizationMode.FP8],
        dtype=torch.bfloat16,
        bootstrap_backend="nccl",
    )

    # Row-parallel Linear: the only configuration whose backward needs dY twice, and so the only
    # one that exercises the column-scaled copy at all.
    def build(overlap):
        torch.manual_seed(opts.seed)          # identical initial weights in both arms
        return te.Linear(
            opts.in_features,
            hidden,
            params_dtype=torch.bfloat16,
            device="cuda",
            tp_group=tp_group,
            tp_size=world,
            sequence_parallel=True,
            parallel_mode="row",
            ub_overlap_ag=overlap,
            ub_overlap_rs=overlap,
            ub_name="proj",
            bias=False,
        )

    fp8_model = build(True)
    ref_model = build(False)
    for t, r in zip(fp8_model.parameters(), ref_model.parameters()):
        with torch.no_grad():
            r.copy_(t)
        torch.testing.assert_close(t, r, rtol=0.0, atol=0.0)   # must start bit-identical

    fp8_opt = torch.optim.SGD(fp8_model.parameters(), lr=opts.lr)
    ref_opt = torch.optim.SGD(ref_model.parameters(), lr=opts.lr)

    # Same data in both arms, and a fixed target so the loss is deterministic. Drawn from a
    # private generator so neither arm's internal RNG use can shift the other's inputs.
    gen = torch.Generator(device="cuda")
    gen.manual_seed(opts.seed + 1)
    shape = [opts.seq_length, opts.batch_size, in_local]
    out_shape = [opts.seq_length // world, opts.batch_size, hidden]
    target = torch.randn(out_shape, device="cuda", dtype=torch.bfloat16, generator=gen)

    # out.sum() would make dY identically 1.0 and every MXFP8 scale equal; the power-of-two
    # weighting spreads the scales and the normal factor makes the rescale actually round.
    half = opts.grad_scale_binades // 2
    if opts.grad_scale_binades > 0:
        exps = torch.randint(-half, opts.grad_scale_binades - half + 1,
                             (out_shape[0], 1, 1), device="cuda", generator=gen)
        loss_w = torch.exp2(exps.to(torch.float32)) * torch.randn(
            out_shape, device="cuda", dtype=torch.float32, generator=gen)
    else:
        loss_w = torch.ones(out_shape, device="cuda", dtype=torch.float32)

    def step(model, opt, x, use_fp8):
        opt.zero_grad(set_to_none=False)
        ctx = te.autocast(enabled=True, recipe=recipe) if use_fp8 else nullcontext()
        with ctx:
            y = model(x)
        loss = (((y.float() - target.float()) ** 2) * loss_w).mean()
        loss.backward()
        gn = torch.zeros((), dtype=torch.float64, device="cuda")
        for prm in model.parameters():
            if prm.grad is not None:
                gn += (prm.grad.detach().to(torch.float64) ** 2).sum()
        dist.all_reduce(gn, op=dist.ReduceOp.SUM)
        opt.step()
        return loss.item(), gn.sqrt().item()

    mode = os.getenv("NVTE_AG_WGRAD_COPY", "transpose")
    sub = os.getenv("NVTE_AG_WGRAD_TRANSPOSE", "inflight")
    label = mode if mode == "gather" else f"{mode}/{sub}"

    d_half = None
    d_final = None
    for n in range(1, opts.steps + 1):
        # requires_grad matters beyond dgrad itself: ub_fused_wgrad_ag is gated on
        # requires_dgrad, so without it the layer takes upstream's external-AG path instead of
        # the one under test -- and that path is a stub on P2P communicators.
        x = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=gen)
        l_fp8, g_fp8 = step(fp8_model, fp8_opt, x.detach().clone().requires_grad_(True), True)
        l_ref, _ = step(ref_model, ref_opt, x.detach().clone().requires_grad_(True), False)
        if n == opts.steps // 2:
            d_half = _global_rel_distance(list(fp8_model.parameters()),
                                          list(ref_model.parameters()))
        if n == opts.steps or (opts.report_every and n % opts.report_every == 0):
            d = _global_rel_distance(list(fp8_model.parameters()),
                                     list(ref_model.parameters()))
            if n == opts.steps:
                d_final = d
            if rank == 0 and opts.report_every:
                print(f"DRIFT step={n} d={d:.6e} gnorm_fp8={g_fp8:.10e} "
                      f"loss_fp8={l_fp8:.6f}", flush=True)

    if rank == 0:
        # growth is d_final/d_half: ~1 means the gap stopped opening, ~2 means it tracks sqrt(n)
        # or slower, a large value means the two trajectories are separating.
        growth = (d_final / d_half) if (d_half and d_half > 0) else float("nan")
        print(f"DRIFT_SUMMARY mode={label} steps={opts.steps} "
              f"binades={opts.grad_scale_binades} fp8_format={opts.fp8_format} "
              f"d_half={d_half:.6e} d_final={d_final:.6e} growth={growth:.4f}", flush=True)

    te.module.base.destroy_ub()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(_train(_parse_args()))
