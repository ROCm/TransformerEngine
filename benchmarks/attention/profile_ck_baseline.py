#!/usr/bin/env python3
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Profile varlen_attn/attn_fwd and attn_bwd (CK baseline) with rocprof.
# Same configs and CSV format as profile_smallseq_rocm.py (TE path) for comparison.
#
# Usage (from repo root):
#   python benchmarks/attention/profile_ck_baseline.py
#   python benchmarks/attention/profile_ck_baseline.py --varlen-dir ./varlen_attn
#
# Requires: rocprof (ROCm), attn_fwd and attn_bwd built in varlen_attn/.

import argparse
import csv
import os
import shutil
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

# Same kernel names as TE smallseq (varlen_attn/attn_fwd.cpp, attn_bwd.cpp use same names)
CK_FWD_KERNELS = [
    "compute_scores_kernel",
    "apply_mask_and_softmax_kernel",
    "compute_output_kernel",
]
CK_BWD_KERNELS = [
    "compute_grad_v_kernel",
    "compute_grad_attn_kernel",
    "softmax_backward_kernel",
    "compute_grad_qk_kernel",
]

# Same configs as profile_smallseq_rocm.py DEFAULT_CONFIGS (attn_fwd/attn_bwd support these)
# (config_id, b, s_q, s_kv, h_q, h_kv, d_qk, d_v, dtype)
DEFAULT_CONFIGS = [
    ("config_1", 4000, 1, 2, 16, 16, 128, 128, "bf16"),
    ("config_2", 4000, 1, 4, 16, 16, 128, 128, "bf16"),
    ("config_3", 4000, 1, 6, 16, 16, 128, 128, "bf16"),
    ("config_4", 4000, 1, 8, 16, 16, 128, 128, "bf16"),
    ("config_5", 4000, 1, 12, 16, 16, 128, 128, "bf16"),
    ("config_6", 4000, 1, 16, 16, 16, 128, 128, "bf16"),
    ("config_7", 4000, 1, 2, 16, 16, 128, 128, "fp16"),
    ("config_8", 4000, 1, 4, 16, 16, 128, 128, "fp16"),
    ("config_9", 4000, 1, 6, 16, 16, 128, 128, "fp16"),
    ("config_10", 4000, 1, 8, 16, 16, 128, 128, "fp16"),
    ("config_11", 4000, 1, 12, 16, 16, 128, 128, "fp16"),
    ("config_12", 4000, 1, 16, 16, 16, 128, 128, "fp16"),
]

TABLE_NAME = "ck_baseline_timings.csv"


def _match_kernel(name, kernel_substring):
    """Match rocprof kernel name (standalone binary has no namespace prefix)."""
    return kernel_substring in name


def _extract_timings_from_csv(csv_path):
    """Extract kernel timings from rocprof results.stats.csv (AverageNs per call)."""
    fwd_ns = {k: 0 for k in CK_FWD_KERNELS}
    bwd_ns = {k: 0 for k in CK_BWD_KERNELS}
    if not os.path.isfile(csv_path):
        return fwd_ns, bwd_ns
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("Name", "")
            try:
                ns = int(row.get("AverageNs", 0))
            except (ValueError, TypeError):
                continue
            for k in CK_FWD_KERNELS:
                if _match_kernel(name, k):
                    fwd_ns[k] += ns
                    break
            for k in CK_BWD_KERNELS:
                if _match_kernel(name, k):
                    bwd_ns[k] += ns
                    break
    return fwd_ns, bwd_ns


def _run_rocprof(binary_path, args, cwd, output_dir, config_id, mode_str):
    """Run rocprof --stats on binary with args; move results.* into output_dir. Return path to stats.csv."""
    os.makedirs(output_dir, exist_ok=True)
    cmd = [binary_path] + [str(a) for a in args]
    rocprof_args = ["rocprof", "--basenames", "off", "-o", "results.csv", "--stats"]
    full_cmd = rocprof_args + cmd
    print(f"profiling {config_id} ({mode_str}) ...", flush=True)
    try:
        subprocess.run(full_cmd, cwd=cwd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(f"rocprof failed: {e}", file=sys.stderr)
        raise
    except FileNotFoundError:
        print("rocprof not found. Install ROCm and ensure rocprof is on PATH.", file=sys.stderr)
        raise

    # rocprof writes results.csv and results.stats.csv to cwd
    stats_src = os.path.join(cwd, "results.stats.csv")
    if os.path.isfile(stats_src):
        dst = os.path.join(output_dir, "results.stats.csv")
        shutil.copy2(stats_src, dst)
        # clean cwd
        for f in os.listdir(cwd):
            if f.startswith("results."):
                try:
                    os.remove(os.path.join(cwd, f))
                except OSError:
                    pass
        return dst
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Profile CK baseline (varlen_attn attn_fwd/attn_bwd) with rocprof; same configs and CSV format as profile_smallseq_rocm.",
    )
    parser.add_argument(
        "--varlen-dir",
        default=os.path.join(REPO_ROOT, "varlen_attn"),
        help="Directory containing attn_fwd and attn_bwd binaries",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=os.path.join(SCRIPT_DIR, "profiler_outputs_ck_baseline"),
        help="Output directory for rocprof results and summary CSV",
    )
    parser.add_argument(
        "--configs",
        default="all",
        choices=("all", "bf16", "fp16"),
        help="Which configs to run: all (default), bf16 only, or fp16 only",
    )
    args = parser.parse_args()

    attn_fwd = os.path.join(args.varlen_dir, "attn_fwd")
    attn_bwd = os.path.join(args.varlen_dir, "attn_bwd")
    if not os.path.isfile(attn_fwd) or not os.path.isfile(attn_bwd):
        print(
            f"Binaries not found: {attn_fwd}, {attn_bwd}. Build them first (see README).",
            file=sys.stderr,
        )
        sys.exit(1)

    configs = list(DEFAULT_CONFIGS)
    if args.configs == "bf16":
        configs = [c for c in configs if c[8] == "bf16"]
    elif args.configs == "fp16":
        configs = [c for c in configs if c[8] == "fp16"]

    os.makedirs(args.output_dir, exist_ok=True)
    all_rows = []
    # Same columns as profile_smallseq_rocm (te_smallseq_timings.csv): no mode, no per-kernel
    fieldnames = [
        "config_id", "layout", "dtype", "batch_size", "seqlen_q", "seqlen_kv",
        "nheads", "dim", "gqa_ratio", "causal", "total_fwd_ms", "total_bwd_ms",
    ]

    for cfg in configs:
        config_id, b, s_q, s_kv, h_q, h_kv, d_qk, d_v, dtype_str = cfg
        argv = [b, s_q, s_kv, h_q, h_kv, d_qk, d_v, dtype_str]

        dir_fwd = os.path.join(args.output_dir, f"{config_id}_fwd")
        dir_bwd = os.path.join(args.output_dir, f"{config_id}_bwd")

        _run_rocprof(attn_fwd, argv, args.varlen_dir, dir_fwd, config_id, "fwd")
        _run_rocprof(attn_bwd, argv, args.varlen_dir, dir_bwd, config_id, "bwd")

        fwd_stats = os.path.join(dir_fwd, "results.stats.csv")
        bwd_stats = os.path.join(dir_bwd, "results.stats.csv")

        fwd_fwd_ns, _ = _extract_timings_from_csv(fwd_stats)
        _, bwd_bwd_ns = _extract_timings_from_csv(bwd_stats)
        total_fwd_ms = sum(fwd_fwd_ns[k] for k in CK_FWD_KERNELS) / 1e6
        total_bwd_ms = sum(bwd_bwd_ns[k] for k in CK_BWD_KERNELS) / 1e6

        gqa = (h_q / h_kv) if h_kv else 1.0
        row = {
            "config_id": config_id,
            "layout": "THD_THD_THD",
            "dtype": dtype_str,
            "batch_size": b,
            "seqlen_q": s_q,
            "seqlen_kv": s_kv,
            "nheads": h_q,
            "dim": d_qk,
            "gqa_ratio": gqa,
            "causal": "false",
            "total_fwd_ms": round(total_fwd_ms, 4),
            "total_bwd_ms": round(total_bwd_ms, 4),
        }
        all_rows.append(row)

    out_csv = os.path.join(args.output_dir, TABLE_NAME)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Table: {out_csv}")
    print(f"Output dir: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
