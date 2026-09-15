#!/usr/bin/env python
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Sweep launcher for the distributed comm/compute overlap benchmarks.

Thin driver that runs the existing overlap harnesses under ``torchrun`` across a
grid of shapes and overlap variants, having each run append summary statistics
(median/mean/stdev/min/max, and optionally per-sample rows) to a shared CSV via
``--emit-csv`` / ``--emit-samples``.  The harnesses do the real work and timing;
this file only orchestrates the sweep and aggregates the CSV.

New overlap types are added by registering another entry in ``VARIANTS``; no
harness changes needed once a harness supports ``--emit-csv``.

Example (2 GPUs):

    python benchmark_comm_overlap.py --nproc-per-node 2 --csv results/comm_overlap.csv
    python benchmark_comm_overlap.py --only bulk_ag_gemm --samples
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

# tests/pytorch/distributed, where the overlap harnesses + emit helper live.
_HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS_DIR = os.path.normpath(os.path.join(_HERE, "..", "..", "tests", "pytorch", "distributed"))

# Sweep shapes come from the single-GPU GEMM microbenchmarks' model list so they
# line up with the normal benchmarks. The overlap GEMM is MLP-style: for a hidden
# size H it uses K = 4*H (the FC1 intermediate), so only (M, H) are free here --
# per-sublayer QKV/AttnOut shapes would need an explicit intermediate-size arg on
# the harness.
try:
    from utils import MODEL_HIDDEN_SIZES, M_SIZE_LIST
except Exception:  # keep the launcher importable without torch/utils available
    MODEL_HIDDEN_SIZES = [("Llama3-8B", 4096), ("Llama3-70B", 8192), ("Llama3-405B", 16384)]
    M_SIZE_LIST = [1024, 2048, 4096, 8192]

HEAD_DIM = 128  # heads = H / HEAD_DIM; the model hidden sizes are all multiples of 128


def model_shapes(m_sizes):
    """Harness -n/-d/-s/-b args for each (model hidden size, M) combination."""
    shapes = []
    for name, hidden in MODEL_HIDDEN_SIZES:
        if hidden % HEAD_DIM:
            continue
        for m in m_sizes:
            shapes.append({
                "label": name,
                "seq_length": m,
                "batch_size": 1,
                "num_heads": hidden // HEAD_DIM,
                "head_dim": HEAD_DIM,
            })
    return shapes


def _gemm_bulk_ag_args(shape, cfg, csv, samples):
    """PR #713: bulk all-gather overlapped with the dgrad GEMM (bf16)."""
    args = [
        "--bulk-overlap", "--comm-type", "ag",
        "-s", str(shape["seq_length"]), "-b", str(shape["batch_size"]),
        "-n", str(shape["num_heads"]), "-d", str(shape["head_dim"]),
        "--warmup-iters", str(cfg.warmup_iters), "--timing-iters", str(cfg.timing_iters),
        "--variant", "bulk_ag_gemm", "--emit-csv", csv,
    ]
    if samples:
        args += ["--emit-samples", samples]
    return "run_gemm_with_overlap.py", args


def _layer_ln_mlp_args(shape, cfg, csv, samples):
    """LayerNormMLP with TP comm overlap (bf16), timed over the fwd/bwd step."""
    args = [
        "-l", "LayerNormMLP", "--benchmark", "--benchmark-iter", str(cfg.timing_iters),
        "-s", str(shape["seq_length"]), "-b", str(shape["batch_size"]),
        "-n", str(shape["num_heads"]), "-d", str(shape["head_dim"]),
        "--variant", "layernormmlp", "--emit-csv", csv,
    ]
    if samples:
        args += ["--emit-samples", samples]
    return "run_layer_with_overlap.py", args


# name -> builder(shape, cfg, csv, samples) -> (harness_script, argv)
VARIANTS = {
    "bulk_ag_gemm": _gemm_bulk_ag_args,
    "layernormmlp": _layer_ln_mlp_args,
}


def _default_nproc() -> int:
    try:
        import torch

        return max(1, min(2, torch.cuda.device_count()))
    except Exception:
        return 2


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--only", nargs="+", choices=sorted(VARIANTS), default=sorted(VARIANTS),
                   help="subset of overlap variants to run (default: all)")
    p.add_argument("--m-sizes", nargs="+", type=int, default=list(M_SIZE_LIST),
                   help="token counts (GEMM M) to sweep; default matches the GEMM microbenchmarks")
    p.add_argument("--nproc-per-node", type=int, default=_default_nproc(),
                   help="GPUs per node passed to torchrun (== TP size)")
    p.add_argument("--csv", default=os.path.join(os.getcwd(), "results", "comm_overlap.csv"),
                   help="shared summary CSV the harnesses append to")
    p.add_argument("--samples", nargs="?", const="", default=None, metavar="PATH",
                   help="also emit per-sample rows (default path: <csv stem>-samples.csv)")
    p.add_argument("--warmup-iters", type=int, default=10)
    p.add_argument("--timing-iters", type=int, default=50)
    p.add_argument("--dry-run", action="store_true", help="print the torchrun commands only")
    return p.parse_args(argv)


def main(argv=None) -> int:
    cfg = parse_args(argv)
    csv = os.path.abspath(cfg.csv)
    samples = None
    if cfg.samples is not None:
        samples = os.path.abspath(cfg.samples) if cfg.samples else (
            os.path.splitext(csv)[0] + "-samples.csv"
        )

    commands = []
    for name in cfg.only:
        build = VARIANTS[name]
        for shape in model_shapes(cfg.m_sizes):
            script, args = build(shape, cfg, csv, samples)
            commands.append(
                ["torchrun", f"--nproc-per-node={cfg.nproc_per_node}",
                 os.path.join(HARNESS_DIR, script)] + args
            )

    failures = 0
    for cmd in commands:
        print("+ " + " ".join(cmd), flush=True)
        if cfg.dry_run:
            continue
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            failures += 1
            print(f"  -> exited with {result.returncode}", file=sys.stderr, flush=True)

    if not cfg.dry_run and os.path.exists(csv):
        print(f"\nsummary CSV: {csv}")
        with open(csv) as f:
            sys.stdout.write(f.read())
    if samples and not cfg.dry_run and os.path.exists(samples):
        print(f"samples CSV: {samples}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
