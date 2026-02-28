#!/usr/bin/env python3
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Unified profiler for smallseq attention: run either TE (TransformerEngine) or CK
# baseline (varlen_attn) with rocprof. Same configs and CSV format for both.
#
# Usage (from repo root):
#   # Profile TE path (JAX + fused_attn_smallseq):
#   NVTE_FUSED_ATTN_CK_SMALLSEQ=1 python benchmarks/attention/profile_attention_rocm.py --te
#
#   # Profile CK baseline (standalone attn_fwd / attn_bwd); --varlen-dir required:
#   python benchmarks/attention/profile_attention_rocm.py --ck --varlen-dir ./varlen_attn
#
#   # Run both TE and CK, then write combined benchmark.csv:
#   NVTE_FUSED_ATTN_CK_SMALLSEQ=1 python benchmarks/attention/profile_attention_rocm.py --te --ck --varlen-dir ./varlen_attn
#
# Requires: rocprof (ROCm). For --te: JAX, TE. For --ck: --varlen-dir and attn_fwd/attn_bwd binaries.

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

# Shared: same kernel names in fused_attn_smallseq.cpp and varlen_attn/attn_fwd.cpp, attn_bwd.cpp
FWD_KERNELS = [
    "compute_scores_kernel",
    "apply_mask_and_softmax_kernel",
    "compute_output_kernel",
]
BWD_KERNELS = [
    "compute_grad_v_kernel",
    "compute_grad_attn_kernel",
    "softmax_backward_kernel",
    "compute_grad_qk_kernel",
]

# Shared configs: (config_id, b, s_q, s_kv, h_q, h_kv, d_qk, d_v, dtype)
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

NUM_ITERS_TE = 40
CSV_FIELDNAMES = [
    "config_id", "layout", "dtype", "batch_size", "seqlen_q", "seqlen_kv",
    "nheads", "dim", "gqa_ratio", "causal", "total_fwd_ms", "total_bwd_ms",
]
BENCHMARK_CSV_FIELDNAMES = [
    "config_id", "layout", "dtype", "batch_size", "seqlen_q", "seqlen_kv",
    "nheads", "dim", "gqa_ratio", "causal",
    "te_total_fwd_ms", "te_total_bwd_ms", "ck_total_fwd_ms", "ck_total_bwd_ms",
]
BENCHMARK_CSV_NAME = "benchmark.csv"


def _match_kernel(name, kernel_substring, backend):
    """Match rocprof kernel name. TE has namespace prefix; CK (standalone) does not."""
    if backend == "te":
        return "fused_attn_rocm::" in name and kernel_substring in name
    return kernel_substring in name


def _extract_timings_from_csv(csv_path, backend):
    """Extract kernel timings from rocprof results.stats.csv (AverageNs per call)."""
    fwd_ns = {k: 0 for k in FWD_KERNELS}
    bwd_ns = {k: 0 for k in BWD_KERNELS}
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
            for k in FWD_KERNELS:
                if _match_kernel(name, k, backend):
                    fwd_ns[k] += ns
                    break
            for k in BWD_KERNELS:
                if _match_kernel(name, k, backend):
                    bwd_ns[k] += ns
                    break
    return fwd_ns, bwd_ns


def _row_from_config_and_timings(config_id, b, s_q, s_kv, h_q, h_kv, d_qk, d_v, dtype_str,
                                 total_fwd_ms, total_bwd_ms, causal=False):
    """Build one CSV row (shared format)."""
    gqa = (h_q / h_kv) if h_kv else 1.0
    return {
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
        "total_fwd_ms": round(total_fwd_ms, 4),
        "total_bwd_ms": round(total_bwd_ms, 4),
    }


# ---------- TE path ----------

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
    """Entry point for TE subprocess: read config from env and run JAX benchmark."""
    config_str = os.environ.get("NVTE_PROFILE_CK_SMALLSEQ_CONFIG", "")
    num_iters = int(os.environ.get("NVTE_PROFILE_ITERS", str(NUM_ITERS_TE)))
    bench_bwd = os.environ.get("NVTE_PROFILE_BWD", "1") == "1"
    shape, dtype_str = _parse_config(config_str)
    b, s_q, s_kv, h_q, h_kv, d_qk, d_v = shape
    runner = _get_runner(b, s_q, s_kv, h_q, h_kv, d_qk, d_v, dtype_str)
    with tempfile.TemporaryDirectory(prefix="te_smallseq_") as timings_dir:
        if bench_bwd:
            runner.bench_backward(0, num_iters, timings_dir)
        else:
            runner.bench_forward(0, num_iters, timings_dir)


def _run_te_profile(config_id, b, s_q, s_kv, h_q, h_kv, d_qk, d_v, dtype_str, fwd_only,
                    output_dir, num_iters):
    """Run rocprof on TE path (JAX subprocess)."""
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
    rocprof_args = ["rocprof", "--basenames", "off", "-o", "results.csv", "--stats"]
    profiler_cmd = [
        sys.executable, "-c",
        f"import sys; sys.path.insert(0, {repr(REPO_ROOT)}); sys.path.insert(0, {repr(tests_jax)}); "
        f"sys.path.insert(0, {repr(SCRIPT_DIR)}); import profile_attention_rocm as m; m._profiler_iterations_from_env()",
    ]
    cmd = rocprof_args + profiler_cmd
    mode_str = "fwd" if fwd_only else "bwd"
    print(f"profiling {config_id} ({mode_str}) ...", flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    for f in set(os.listdir(cwd)) - before:
        src = os.path.join(cwd, f)
        if os.path.isfile(src) and (f == "results.csv" or f.startswith("results.")):
            shutil.move(src, os.path.join(output_dir, f))


# ---------- CK path ----------

def _run_ck_rocprof(binary_path, argv, cwd, output_dir, config_id, mode_str):
    """Run rocprof --stats on attn_fwd or attn_bwd binary; copy results to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    cmd = [binary_path] + [str(a) for a in argv]
    rocprof_args = ["rocprof", "--basenames", "off", "-o", "results.csv", "--stats"]
    full_cmd = rocprof_args + cmd
    print(f"profiling {config_id} ({mode_str}) ...", flush=True)
    subprocess.run(full_cmd, cwd=cwd, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    dst = os.path.join(output_dir, "results.stats.csv")
    shutil.copy2(os.path.join(cwd, "results.stats.csv"), dst)


# ---------- Main ----------

def _run_te(args, configs, output_dir, table_name):
    os.environ["NVTE_FUSED_ATTN_CK_SMALLSEQ"] = "1"
    xla = os.environ.get("XLA_FLAGS", "")
    if "--xla_gpu_graph_level=0" not in xla:
        os.environ["XLA_FLAGS"] = (xla + " --xla_gpu_graph_level=0").strip()

    all_rows = []
    for cfg in configs:
        cid, b, s_q, s_kv, h_q, h_kv, d_qk, d_v, dtype_str = cfg
        dir_fwd = os.path.join(output_dir, f"{cid}_fwd")
        dir_bwd = os.path.join(output_dir, f"{cid}_bwd")

        if args.fwd:
            _run_te_profile(cid, b, s_q, s_kv, h_q, h_kv, d_qk, d_v, dtype_str,
                            fwd_only=True, output_dir=dir_fwd, num_iters=args.num_iters)
        _run_te_profile(cid, b, s_q, s_kv, h_q, h_kv, d_qk, d_v, dtype_str,
                        fwd_only=False, output_dir=dir_bwd, num_iters=args.num_iters)

        bwd_stats = os.path.join(dir_bwd, "results.stats.csv")
        fn, bn = _extract_timings_from_csv(bwd_stats, "te")
        total_fwd_ms = sum(fn[k] for k in FWD_KERNELS) / 1e6
        total_bwd_ms = sum(bn[k] for k in BWD_KERNELS) / 1e6
        row = _row_from_config_and_timings(
            cid, b, s_q, s_kv, h_q, h_kv, d_qk, d_v, dtype_str,
            total_fwd_ms, total_bwd_ms,
        )
        all_rows.append(row)

    return all_rows


def _run_ck(args, configs, output_dir, table_name):
    varlen_dir = os.path.abspath(os.path.expanduser(args.varlen_dir))
    attn_fwd = os.path.join(varlen_dir, "attn_fwd")
    attn_bwd = os.path.join(varlen_dir, "attn_bwd")
    if not os.path.isfile(attn_fwd) or not os.path.isfile(attn_bwd):
        print(
            f"Binaries not found: {attn_fwd}, {attn_bwd}. Build them first (see README).",
            file=sys.stderr,
        )
        sys.exit(1)

    all_rows = []
    for cfg in configs:
        config_id, b, s_q, s_kv, h_q, h_kv, d_qk, d_v, dtype_str = cfg
        argv = [b, s_q, s_kv, h_q, h_kv, d_qk, d_v, dtype_str]
        dir_fwd = os.path.join(output_dir, f"{config_id}_fwd")
        dir_bwd = os.path.join(output_dir, f"{config_id}_bwd")

        _run_ck_rocprof(attn_fwd, argv, varlen_dir, dir_fwd, config_id, "fwd")
        _run_ck_rocprof(attn_bwd, argv, varlen_dir, dir_bwd, config_id, "bwd")

        fwd_stats = os.path.join(dir_fwd, "results.stats.csv")
        bwd_stats = os.path.join(dir_bwd, "results.stats.csv")
        fwd_fwd_ns, _ = _extract_timings_from_csv(fwd_stats, "ck")
        _, bwd_bwd_ns = _extract_timings_from_csv(bwd_stats, "ck")
        total_fwd_ms = sum(fwd_fwd_ns[k] for k in FWD_KERNELS) / 1e6
        total_bwd_ms = sum(bwd_bwd_ns[k] for k in BWD_KERNELS) / 1e6

        row = _row_from_config_and_timings(
            config_id, b, s_q, s_kv, h_q, h_kv, d_qk, d_v, dtype_str,
            total_fwd_ms, total_bwd_ms,
        )
        all_rows.append(row)

    return all_rows


def _merge_te_ck_csvs(te_csv_path, ck_csv_path, out_csv_path):
    """Read te_smallseq_timings.csv and ck_baseline_timings.csv; merge by config_id; write benchmark.csv."""
    te_by_id = {}
    if os.path.isfile(te_csv_path):
        with open(te_csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                te_by_id[row["config_id"]] = row
    ck_by_id = {}
    if os.path.isfile(ck_csv_path):
        with open(ck_csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ck_by_id[row["config_id"]] = row

    config_ids = [c[0] for c in DEFAULT_CONFIGS]
    rows = []
    for cid in config_ids:
        te_row = te_by_id.get(cid, {})
        ck_row = ck_by_id.get(cid, {})
        # Use config shape from either; prefer te then ck
        base = te_row or ck_row
        rows.append({
            "config_id": cid,
            "layout": base.get("layout", "THD_THD_THD"),
            "dtype": base.get("dtype", ""),
            "batch_size": base.get("batch_size", ""),
            "seqlen_q": base.get("seqlen_q", ""),
            "seqlen_kv": base.get("seqlen_kv", ""),
            "nheads": base.get("nheads", ""),
            "dim": base.get("dim", ""),
            "gqa_ratio": base.get("gqa_ratio", ""),
            "causal": base.get("causal", "false"),
            "te_total_fwd_ms": te_row.get("total_fwd_ms", ""),
            "te_total_bwd_ms": te_row.get("total_bwd_ms", ""),
            "ck_total_fwd_ms": ck_row.get("total_fwd_ms", ""),
            "ck_total_bwd_ms": ck_row.get("total_bwd_ms", ""),
        })

    os.makedirs(os.path.dirname(out_csv_path) or ".", exist_ok=True)
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=BENCHMARK_CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Benchmark: {out_csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Profile smallseq attention (TE and/or CK baseline) with rocprof. Use --te and/or --ck; both produces combined benchmark.csv.",
    )
    parser.add_argument("--te", action="store_true", help="Profile TransformerEngine (JAX) path")
    parser.add_argument("--ck", action="store_true", help="Profile CK baseline (varlen_attn attn_fwd/attn_bwd)")

    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Output directory (defaults: te_profiler_outputs_smallseq, profiler_outputs_ck_baseline; when both, base dir for combined benchmark.csv)",
    )

    # TE-specific
    parser.add_argument("--fwd", action="store_true", help="[--te only] Also run fwd-only profiling per config")
    parser.add_argument("--num-iters", type=int, default=NUM_ITERS_TE, help="[--te only] Iterations per profile")

    # CK-specific: required when --ck is used
    parser.add_argument(
        "--varlen-dir",
        default=None,
        help="[Required when --ck] Directory containing attn_fwd and attn_bwd binaries",
    )

    args = parser.parse_args()

    if not args.te and not args.ck:
        parser.error("At least one of --te or --ck is required")
    if args.ck and not args.varlen_dir:
        parser.error("--varlen-dir is required when --ck is used")

    configs = list(DEFAULT_CONFIGS)
    run_both = args.te and args.ck
    base_output = args.output_dir or SCRIPT_DIR

    te_csv_path = None
    ck_csv_path = None

    if args.te:
        output_dir_te = (
            os.path.join(base_output, "te_profiler_outputs_smallseq")
            if run_both
            else (args.output_dir or os.path.join(SCRIPT_DIR, "te_profiler_outputs_smallseq"))
        )
        os.makedirs(output_dir_te, exist_ok=True)
        all_rows_te = _run_te(args, configs, output_dir_te, "te_smallseq_timings.csv")
        if all_rows_te:
            te_csv_path = os.path.join(output_dir_te, "te_smallseq_timings.csv")
            with open(te_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
                writer.writeheader()
                writer.writerows(all_rows_te)
            print(f"Table: {te_csv_path}")
        print(f"TE output dir: {os.path.abspath(output_dir_te)}")

    if args.ck:
        output_dir_ck = (
            os.path.join(base_output, "profiler_outputs_ck_baseline")
            if run_both
            else (args.output_dir or os.path.join(SCRIPT_DIR, "profiler_outputs_ck_baseline"))
        )
        os.makedirs(output_dir_ck, exist_ok=True)
        all_rows_ck = _run_ck(args, configs, output_dir_ck, "ck_baseline_timings.csv")
        if all_rows_ck:
            ck_csv_path = os.path.join(output_dir_ck, "ck_baseline_timings.csv")
            with open(ck_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
                writer.writeheader()
                writer.writerows(all_rows_ck)
            print(f"Table: {ck_csv_path}")
        print(f"CK output dir: {os.path.abspath(output_dir_ck)}")

    if run_both and te_csv_path and ck_csv_path:
        benchmark_path = os.path.join(base_output, BENCHMARK_CSV_NAME)
        _merge_te_ck_csvs(te_csv_path, ck_csv_path, benchmark_path)


if __name__ == "__main__":
    main()
