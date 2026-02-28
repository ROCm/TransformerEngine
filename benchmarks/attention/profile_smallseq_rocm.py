#!/usr/bin/env python3
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Usage (from repo root):
#   NVTE_FUSED_ATTN_CK_SMALLSEQ=1 python benchmarks/attention/profile_smallseq_rocm.py
#   --fwd  : Also run fwd-only profiling per config

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
# Kernels from fused_attn_smallseq.cpp
SMALLSEQ_FWD_KERNELS = [
    "compute_scores_kernel",
    "apply_mask_and_softmax_kernel",
    "compute_output_kernel",
]
SMALLSEQ_BWD_KERNELS = [
    "compute_grad_v_kernel",
    "compute_grad_attn_kernel",
    "softmax_backward_kernel",
    "compute_grad_qk_kernel",
]
TABLE_NAME = "te_smallseq_timings.csv"

NUM_ITERS = 40
# Default configs: list of (config_id, b, s_q, s_kv, h_q, h_kv, d_qk, d_v, dtype)
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


def _setup_paths():
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    tests_jax = os.path.join(REPO_ROOT, "tests", "jax")
    if tests_jax not in sys.path:
        sys.path.insert(0, tests_jax)


def _parse_config(config_str):
    parts = [p.strip() for p in config_str.split(",")]
    if len(parts) != 8:
        raise ValueError(f"Expected 8 values, got {len(parts)}: {config_str}")
    b, s_q, s_kv, h_q, h_kv, d_qk, d_v = (int(x) for x in parts[:7])
    dtype_str = parts[7].lower()
    return (b, s_q, s_kv, h_q, h_kv, d_qk, d_v), dtype_str


def _get_runner(b, s_q, s_kv, h_q, h_kv, d_qk, d_v, dtype_str):
    _setup_paths()
    import jax.numpy as jnp
    from test_fused_attn import AttnBiasType, AttnMaskType, QKVLayout, SeqDescFormat
    from benchmark_attention_jax import FusedAttnBenchRunner

    dtype = jnp.bfloat16 if dtype_str in ("bf16", "bfloat16") else jnp.float16
    return FusedAttnBenchRunner(
        batch_size=b,
        max_seqlen_q=s_q,
        max_seqlen_kv=s_kv,
        num_heads_q=h_q,
        num_heads_kv=h_kv,
        head_dim_qk=d_qk,
        head_dim_v=d_v,
        attn_bias_type=AttnBiasType.NO_BIAS,
        attn_mask_type=AttnMaskType.PADDING_MASK,
        dropout_prob=0.0,
        use_old_rng=True,
        dtype=dtype,
        is_training=True,
        qkv_layout=QKVLayout.THD_THD_THD,
        bias_shape=None,
        window_size=None,
        seq_desc_format=SeqDescFormat.Seqlens,
    )


def _profiler_iterations_from_env():
    config_str = os.environ.get("NVTE_PROFILE_CK_SMALLSEQ_CONFIG", "")
    num_iters = int(os.environ.get("NVTE_PROFILE_ITERS", str(NUM_ITERS)))
    bench_bwd = os.environ.get("NVTE_PROFILE_BWD", "1") == "1"
    shape, dtype_str = _parse_config(config_str)
    b, s_q, s_kv, h_q, h_kv, d_qk, d_v = shape
    runner = _get_runner(b, s_q, s_kv, h_q, h_kv, d_qk, d_v, dtype_str)
    with tempfile.TemporaryDirectory(prefix="te_smallseq_") as timings_dir:
        if bench_bwd:
            runner.bench_backward(0, num_iters, timings_dir)
        else:
            runner.bench_forward(0, num_iters, timings_dir)


def _match_kernel(name, kernel_substring):
    return "fused_attn_rocm::" in name and kernel_substring in name


def _extract_timings_from_csv(csv_path):
    """Extract smallseq kernel timings from AverageNs (per-call) column."""
    fwd_ns = {k: 0 for k in SMALLSEQ_FWD_KERNELS}
    bwd_ns = {k: 0 for k in SMALLSEQ_BWD_KERNELS}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("Name", "")
            try:
                ns = int(row.get("AverageNs", 0))
            except (ValueError, TypeError):
                continue
            for k in SMALLSEQ_FWD_KERNELS:
                if _match_kernel(name, k):
                    fwd_ns[k] += ns
                    break
            for k in SMALLSEQ_BWD_KERNELS:
                if _match_kernel(name, k):
                    bwd_ns[k] += ns
                    break
    return fwd_ns, bwd_ns


def _run_one_profile(config_id, b, s_q, s_kv, h_q, h_kv, d_qk, d_v, dtype_str, fwd_only, output_dir, num_iters, no_stats, no_hip_trace, verbose):
    """Run rocprof for num_iters (no warmup; run with more iters for better average)."""
    config_str = f"{b},{s_q},{s_kv},{h_q},{h_kv},{d_qk},{d_v},{dtype_str}"
    bench_bwd = not fwd_only
    env = os.environ.copy()
    env["NVTE_FUSED_ATTN_CK_SMALLSEQ"] = "1"
    xla = env.get("XLA_FLAGS", "")
    if "--xla_gpu_graph_level=0" not in xla:
        env["XLA_FLAGS"] = (xla + " --xla_gpu_graph_level=0").strip()
    env["NVTE_PROFILE_CK_SMALLSEQ_CONFIG"] = config_str
    env["NVTE_PROFILE_ITERS"] = str(num_iters)
    env["NVTE_PROFILE_BWD"] = "1" if bench_bwd else "0"

    os.makedirs(output_dir, exist_ok=True)
    cwd = REPO_ROOT
    before = set(os.listdir(cwd))
    tests_jax = os.path.join(REPO_ROOT, "tests", "jax")
    rocprof_args = ["rocprof", "--basenames", "off", "-o", "results.csv"]
    if not no_stats:
        rocprof_args.append("--stats")
    if not no_hip_trace:
        rocprof_args.append("--hip-trace")
    profiler_cmd = [
        sys.executable, "-c",
        f"import sys; sys.path.insert(0, {repr(REPO_ROOT)}); sys.path.insert(0, {repr(tests_jax)}); "
        f"sys.path.insert(0, {repr(SCRIPT_DIR)}); import profile_smallseq_rocm as m; m._profiler_iterations_from_env()",
    ]
    cmd = rocprof_args + profiler_cmd
    mode_str = "fwd" if fwd_only else "bwd"
    print(f"profiling {config_id} ({mode_str}) ...", flush=True)
    try:
        subprocess.run(cmd, cwd=cwd, env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(f"rocprof failed for {config_id} ({'fwd' if fwd_only else 'bwd'}): {e}", file=sys.stderr)
        raise
    except FileNotFoundError:
        print("rocprof not found. Install ROCm and ensure rocprof is on PATH.", file=sys.stderr)
        raise

    for f in set(os.listdir(cwd)) - before:
        src = os.path.join(cwd, f)
        if os.path.isfile(src) and (f == "results.csv" or f.startswith("results.")):
            shutil.move(src, os.path.join(output_dir, f))


def _build_table_rows(config_id, fwd_stats_path, bwd_stats_path, b, s_q, s_kv, h_q, h_kv, d_qk, d_v, dtype_str, include_fwd_row=True, causal=False):
    """Build one row per config with total_fwd_ms and total_bwd_ms from bwd run."""
    gqa = (h_q / h_kv) if h_kv else 1.0
    base = {
        "config_id": config_id,
        "layout": "THD_THD_THD",
        "dtype": dtype_str,
        "batch_size": b,
        "seqlen_q": s_q,
        "seqlen_kv": s_kv,
        "nheads": h_q,
        "dim": d_qk,
        "gqa_ratio": gqa,
        "causal": str(causal).lower(),
    }
    bwd_run_fwd = {k: 0.0 for k in SMALLSEQ_FWD_KERNELS}
    bwd_run_bwd = {k: 0.0 for k in SMALLSEQ_BWD_KERNELS}
    if bwd_stats_path and os.path.isfile(bwd_stats_path):
        fn, bn = _extract_timings_from_csv(bwd_stats_path)
        for k in SMALLSEQ_FWD_KERNELS:
            bwd_run_fwd[k] = fn[k] / 1e6
        for k in SMALLSEQ_BWD_KERNELS:
            bwd_run_bwd[k] = bn[k] / 1e6
    total_fwd_from_bwd_run = sum(bwd_run_fwd.values())
    total_bwd_from_bwd_run = sum(bwd_run_bwd.values())
    input_cols = ["config_id", "layout", "dtype", "batch_size", "seqlen_q", "seqlen_kv", "nheads", "dim", "gqa_ratio", "causal"]
    rows = [{**base, "total_fwd_ms": round(total_fwd_from_bwd_run, 4), "total_bwd_ms": round(total_bwd_from_bwd_run, 4)}]
    fieldnames = input_cols + ["total_fwd_ms", "total_bwd_ms"]
    return rows, fieldnames


def main():
    parser = argparse.ArgumentParser(description="Profile CK smallseq (bwd by default; --fwd to also run fwd).")
    parser.add_argument("--fwd", action="store_true", help="Also run fwd-only profiling per config")
    args = parser.parse_args()

    # Required for CK smallseq benchmarks
    os.environ["NVTE_FUSED_ATTN_CK_SMALLSEQ"] = "1"
    xla = os.environ.get("XLA_FLAGS", "")
    if "--xla_gpu_graph_level=0" not in xla:
        os.environ["XLA_FLAGS"] = (xla + " --xla_gpu_graph_level=0").strip()

    configs = list(DEFAULT_CONFIGS)
    output_dir = os.path.join(SCRIPT_DIR, "te_profiler_outputs_smallseq")
    os.makedirs(output_dir, exist_ok=True)
    all_rows = []
    fieldnames = None

    for cfg in configs:
        cid, b, s_q, s_kv, h_q, h_kv, d_qk, d_v, dtype_str = cfg
        dir_fwd = os.path.join(output_dir, f"{cid}_fwd")
        dir_bwd = os.path.join(output_dir, f"{cid}_bwd")

        if args.fwd:
            print(f"--- {cid}: fwd ---")
            _run_one_profile(cid, b, s_q, s_kv, h_q, h_kv, d_qk, d_v, dtype_str, fwd_only=True, output_dir=dir_fwd, num_iters=NUM_ITERS, no_stats=False, no_hip_trace=False, verbose=False)
        print(f"--- {cid}: bwd ---")
        _run_one_profile(cid, b, s_q, s_kv, h_q, h_kv, d_qk, d_v, dtype_str, fwd_only=False, output_dir=dir_bwd, num_iters=NUM_ITERS, no_stats=False, no_hip_trace=False, verbose=False)

        fwd_stats = os.path.join(dir_fwd, "results.stats.csv") if args.fwd else None
        bwd_stats = os.path.join(dir_bwd, "results.stats.csv")
        rows, fieldnames = _build_table_rows(cid, fwd_stats, bwd_stats, b, s_q, s_kv, h_q, h_kv, d_qk, d_v, dtype_str, include_fwd_row=args.fwd)
        all_rows.extend(rows)

    if fieldnames and all_rows:
        table_path = os.path.join(output_dir, TABLE_NAME)
        with open(table_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print("Table:", table_path)
    print("Output dir:", os.path.abspath(output_dir))


if __name__ == "__main__":
    main()
