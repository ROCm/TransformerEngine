# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
"""Does deriving wgrad's column-scaled dY drift further from bf16 than gathering it?

The rest of the MXFP8 overlap suite checks a single step: the derived region matches a reference
transpose byte for byte, and the gradients land inside tolerance. Neither says whether the error
a scheme introduces per step ACCUMULATES, which is the question asked of any low-precision
training recipe and the one arXiv:2511.02302 Sec 4.1 answers for its own by training 16B
parameters for 200B tokens.

The shape of the check here:

    for each mode, train one row-parallel Linear against a bf16 twin from identical weights on
    identical data, and record  d(n) = || W_fp8(n) - W_bf16(n) || / || W_bf16(n) ||

    pass if  d_inflight(N) is no worse than d_gather(N), and neither is still growing

`gather` is the bar rather than an absolute threshold: it is the scheme already in use, so
matching it is what "no additional error in training" means. bf16 is not the bar -- it is the
fixed point that makes the two modes' curves comparable, since two MXFP8 trajectories diverge
from each other chaotically whether or not either is worse.

NOT a convergence test. It runs a few hundred steps on one layer; it can show that errors are
not compounding, and it cannot show that a model converges. That still needs a real run.
"""
import os
import re
import subprocess
from pathlib import Path

import pytest
import torch
import transformer_engine.pytorch as te

from torch.utils.cpp_extension import IS_HIP_EXTENSION

TEST_ROOT = Path(__file__).parent
NUM_PROCS = min(4, torch.cuda.device_count())
STEPS = 200

fused_available = IS_HIP_EXTENSION and torch.cuda.device_count() >= 4
mxfp8_available, reason_for_no_mxfp8 = te.is_mxfp8_available(return_reason=True)

# The derived copy is exact on normal-range elements and only loses on underflow, so these are
# generous on purpose: the test is meant to catch a scheme that DRIFTS, not to police the few
# percent of run-to-run variation two stochastic trajectories show.
DRIFT_TOL = 1.15          # d_inflight(N) <= d_gather(N) * this
GROWTH_TOL = 2.5          # d(N) / d(N/2); sqrt-like growth passes, separation does not


def _run(mode, steps=STEPS, binades=6, fp8_format="e4m3", nprocs=NUM_PROCS):
    cmd = [
        "torchrun", f"--nproc_per_node={nprocs}",
        str(TEST_ROOT / "run_mxfp8_wgrad_drift.py"),
        f"--steps={steps}",
        f"--grad-scale-binades={binades}",
        f"--fp8-format={fp8_format}",
    ]
    env = os.environ.copy()
    env["PYTORCH_JIT"] = "0"
    env["NVTE_TORCH_COMPILE"] = "0"
    env["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"
    env["NVTE_ROCM_ENABLE_MXFP8"] = "1"
    # One mode per process: both the C++ and the Python side latch this on first use.
    if mode == "gather":
        env["NVTE_AG_WGRAD_COPY"] = "gather"
    else:
        env["NVTE_AG_WGRAD_COPY"] = "transpose"
        env["NVTE_AG_WGRAD_TRANSPOSE"] = mode
    result = subprocess.run(cmd, env=env, capture_output=True, check=False)
    out = result.stdout.decode()
    assert result.returncode == 0, f"{mode} run failed\n{out}\n{result.stderr.decode()}"
    m = re.search(r"DRIFT_SUMMARY .*d_half=(\S+) d_final=(\S+) growth=(\S+)", out)
    assert m is not None, f"{mode} run printed no summary\n{out}"
    return {"d_half": float(m.group(1)), "d_final": float(m.group(2)),
            "growth": float(m.group(3)), "stdout": out}


@pytest.mark.skipif(not fused_available, reason="needs the ROCm fused backend and >= 4 GPUs")
@pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8)
def test_wgrad_drift_inflight_vs_gather():
    """The derived copy must not pull the weights further from bf16 than the gathered one."""
    gather = _run("gather")
    inflight = _run("inflight")

    # Both must have actually moved away from bf16 -- a zero distance means the run degenerated
    # (no fp8, or no steps) and would pass the comparison vacuously.
    assert gather["d_final"] > 0.0, gather["stdout"]
    assert inflight["d_final"] > 0.0, inflight["stdout"]

    assert inflight["d_final"] <= gather["d_final"] * DRIFT_TOL, (
        f"deriving the column-scaled copy drifts further than gathering it: "
        f"inflight {inflight['d_final']:.3e} vs gather {gather['d_final']:.3e} "
        f"after {STEPS} steps"
    )


@pytest.mark.skipif(not fused_available, reason="needs the ROCm fused backend and >= 4 GPUs")
@pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8)
@pytest.mark.parametrize("mode", ("gather", "inflight"))
def test_wgrad_drift_does_not_compound(mode):
    """The gap to bf16 must stop opening, rather than growing with the step count.

    d(N)/d(N/2) near 1 means it has plateaued; sqrt-like growth is ~1.4. A value near 2 or above
    means each step is adding error the previous ones did not absorb, which is the signature that
    would matter over a real training run.
    """
    r = _run(mode)
    assert r["growth"] <= GROWTH_TOL, (
        f"{mode}: distance to bf16 still growing at step {STEPS} "
        f"(d_half={r['d_half']:.3e} -> d_final={r['d_final']:.3e}, "
        f"growth={r['growth']:.2f} > {GROWTH_TOL})"
    )


@pytest.mark.skipif(not fused_available, reason="needs the ROCm fused backend and >= 4 GPUs")
@pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8)
@pytest.mark.parametrize("binades", (0, 12))
def test_wgrad_drift_scale_spread(binades):
    """Both ends of the regime the derivation's loss depends on.

    At 0 binades every MXFP8 block shares a scale, nothing is ever rescaled, and the derived copy
    is a straight copy -- so inflight should track gather almost exactly. At 12 the row scales
    span far enough that E4M3 underflows ~2.4% of elements, which is the worst realistic case for
    the tile-max rule. A failure at 12 but not at 0 points at underflow rather than at the
    plumbing.
    """
    gather = _run("gather", binades=binades)
    inflight = _run("inflight", binades=binades)
    assert inflight["d_final"] <= gather["d_final"] * DRIFT_TOL, (
        f"binades={binades}: inflight {inflight['d_final']:.3e} vs "
        f"gather {gather['d_final']:.3e}"
    )


@pytest.mark.skipif(not fused_available, reason="needs the ROCm fused backend and >= 4 GPUs")
@pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8)
def test_wgrad_drift_hybrid_e5m2():
    """HYBRID puts the grad output in E5M2, where the transpose is lossless on every realistic
    case -- so this is the arm that should be indistinguishable from gather, and the first thing
    to check if the E4M3 arm ever regresses."""
    gather = _run("gather", fp8_format="hybrid")
    inflight = _run("inflight", fp8_format="hybrid")
    assert inflight["d_final"] <= gather["d_final"] * DRIFT_TOL, (
        f"E5M2: inflight {inflight['d_final']:.3e} vs gather {gather['d_final']:.3e}"
    )
