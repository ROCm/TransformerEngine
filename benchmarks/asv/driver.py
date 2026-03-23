#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""ASV benchmark driver — runs bench classes in-process and saves ASV-compatible results.

Usage:
    python driver.py <suite> [method_filter] [-w W] [-n N] [--no-save]
    python bench_gemm.py [method_filter] [-w W] [-n N] [--no-save]
"""

import argparse
import hashlib
import importlib
import inspect
import itertools
import json
import math
import os
import platform
import subprocess
import sys
import textwrap
import time


# ---------------------------------------------------------------------------
# ASV result generation
# ---------------------------------------------------------------------------

def _get_benchmark_version(cls, method_name):
    """Compute the version hash the same way ASV does.

    ASV hashes a code string built from the time_* and setup methods.
    The string is class header + indented time method + indented setup,
    with no trailing newline.
    """
    time_src = textwrap.dedent(inspect.getsource(getattr(cls, method_name)))
    setup_src = textwrap.dedent(inspect.getsource(cls.setup))
    code = (
        f"class {cls.__name__}:\n"
        + textwrap.indent(time_src, "    ") + "\n"
        + textwrap.indent(setup_src, "    ")
    ).rstrip("\n")
    return hashlib.sha256(code.encode()).hexdigest()


def _format_param_value(v):
    """Format a parameter value the way ASV stores it in JSON."""
    if isinstance(v, str):
        return f"'{v}'"
    return repr(v)


def _get_machine_info():
    """Build the params/machine dict ASV expects."""
    machine = platform.node()
    info = {
        "arch": platform.machine(),
        "cpu": "",
        "machine": machine,
        "num_cpu": str(os.cpu_count()),
        "os": f"{platform.system()} {platform.release()}",
        "ram": "",
    }
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu"] = line.split(":", 1)[1].strip()
                    break
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    info["ram"] = line.split()[1]  # kB
                    break
    except OSError:
        pass
    return machine, info


def _get_commit_hash():
    """Get the current git HEAD hash."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _compute_stats(samples):
    """Compute statistics from a list of timing samples."""
    if not samples:
        return None, None, None, None, None
    s = sorted(samples)
    n = len(s)
    median = s[n // 2]
    mean = sum(s) / n
    q25 = s[max(0, n // 4)]
    q75 = s[min(n - 1, 3 * n // 4)]
    stdev = math.sqrt(sum((t - mean) ** 2 for t in s) / n)
    ci_lo = max(0, mean - 2.576 * stdev / math.sqrt(n))  # 99% CI
    ci_hi = mean + 2.576 * stdev / math.sqrt(n)
    return median, ci_lo, ci_hi, q25, q75


def _get_results_dir():
    """Read results_dir from asv.conf.json, resolved to an absolute path."""
    conf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "asv.conf.json")
    with open(conf_path) as f:
        conf = json.load(f)
    conf_dir = os.path.dirname(conf_path)
    return os.path.normpath(os.path.join(conf_dir, conf["results_dir"]))


def save_asv_results(all_results):
    """Write results to ASV's results directory."""
    commit_hash = _get_commit_hash()
    machine_name, machine_info = _get_machine_info()
    env_name = "existing-" + sys.executable.replace("/", "_").strip("_")
    results_dir = _get_results_dir()
    machine_dir = os.path.join(results_dir, machine_name)
    os.makedirs(machine_dir, exist_ok=True)

    # Write machine.json if missing
    machine_json = os.path.join(machine_dir, "machine.json")
    if not os.path.exists(machine_json):
        with open(machine_json, "w") as f:
            json.dump({**machine_info, "version": 1}, f, indent=4)

    # Load existing result file or start fresh
    filename = f"{commit_hash[:8]}-{env_name}.json"
    result_path = os.path.join(machine_dir, filename)
    if os.path.exists(result_path):
        with open(result_path) as f:
            data = json.load(f)
    else:
        data = {
            "commit_hash": commit_hash,
            "env_name": env_name,
            "date": int(time.time() * 1000),
            "params": {**machine_info, "python": sys.executable},
            "python": sys.executable,
            "requirements": {},
            "env_vars": {},
            "result_columns": [
                "result", "params", "version",
                "started_at", "duration",
                "stats_ci_99_a", "stats_ci_99_b",
                "stats_q_25", "stats_q_75",
                "stats_number", "stats_repeat",
                "samples",
            ],
            "results": {},
            "durations": {},
            "version": 2,
        }

    # Merge new results
    for bench_key, bench_data in all_results.items():
        data["results"][bench_key] = bench_data

    with open(result_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to {result_path}")


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_class(suite_name, cls, class_name, method_filter=None, warmup=3, iters=7):
    """Run all benchmarks in a class, returning ASV-formatted results."""
    methods = sorted(m for m in dir(cls) if m.startswith("time_"))
    if method_filter:
        methods = [m for m in methods if method_filter in m]
    if not methods:
        return {}

    params = getattr(cls, "params", [[]])
    param_names = getattr(cls, "param_names", [])
    combos = list(itertools.product(*params))

    print(f"\n{class_name}  ({len(combos)} combos x {len(methods)} methods, "
          f"{warmup} warmup, {iters} timed)")
    HDR = (f"  {'median':>10}  {'mean':>10}  {'stdev':>10}"
           f"  {'q25':>10}  {'q75':>10}  {'min':>10}  {'max':>10}"
           f"  {'method':<30}  params")
    print("-" * len(HDR))
    print(HDR)
    print("-" * len(HDR))

    # ASV stores params as lists of string representations
    asv_params = [[_format_param_value(v) for v in dim] for dim in params]

    all_results = {}

    for method_name in methods:
        bench_key = f"{suite_name}.{class_name}.{method_name}"
        version = _get_benchmark_version(cls, method_name)

        medians = []
        ci_los = []
        ci_his = []
        q25s = []
        q75s = []
        numbers = []
        repeats = []
        started_at = int(time.time() * 1000)
        t_start = time.perf_counter()

        for combo in combos:
            label = ", ".join(f"{n}={v}" for n, v in zip(param_names, combo))
            instance = cls()
            try:
                instance.setup(*combo)
            except Exception as e:
                print(f"  SKIP  {label}  setup failed: {e}")
                medians.append(None)
                ci_los.append(None)
                ci_his.append(None)
                q25s.append(None)
                q75s.append(None)
                numbers.append(None)
                repeats.append(None)
                continue

            method = getattr(instance, method_name)

            for _ in range(warmup):
                method(*combo)

            samples = []
            for _ in range(iters):
                t0 = time.perf_counter()
                result = method(*combo)
                wall = time.perf_counter() - t0
                samples.append(wall if result is None else result)

            median, ci_lo, ci_hi, q25, q75 = _compute_stats(samples)
            mean = sum(samples) / len(samples)
            stdev = math.sqrt(sum((t - mean) ** 2 for t in samples) / len(samples))
            s_min, s_max = min(samples), max(samples)

            medians.append(median)
            ci_los.append(ci_lo)
            ci_his.append(ci_hi)
            q25s.append(q25)
            q75s.append(q75)
            numbers.append(1)
            repeats.append(iters)

            print(f"  {median*1000:>8.3f}ms  {mean*1000:>8.3f}ms  "
                  f"{stdev*1000:>8.3f}ms  {q25*1000:>8.3f}ms  {q75*1000:>8.3f}ms  "
                  f"{s_min*1000:>8.3f}ms  {s_max*1000:>8.3f}ms  "
                  f"{method_name:<30}  {label}")

        duration = time.perf_counter() - t_start

        # ASV result row: [result, params, version, started_at, duration,
        #   ci_99_a, ci_99_b, q_25, q_75, number, repeat, samples]
        all_results[bench_key] = [
            medians,
            asv_params,
            version,
            started_at,
            round(duration, 2),
            ci_los,
            ci_his,
            q25s,
            q75s,
            numbers,
            repeats,
        ]

    return all_results


def run_as_main(caller_file=None):
    """Run benchmarks from a bench file or from the command line.

    When called with a file path (from a bench file's ``__main__`` block),
    the suite is derived from the filename.  When called without arguments
    (i.e. ``python driver.py bench_gemm``), the suite is taken from argv.

    Usage from a bench file::

        if __name__ == "__main__":
            from driver import run_as_main
            run_as_main(__file__)
    """
    parser = argparse.ArgumentParser(
        description="Run ASV benchmarks directly in-process (no subprocess overhead).")
    if caller_file is None:
        parser.add_argument("suite", help="Benchmark module name (e.g. bench_casting)")
    parser.add_argument("method_filter", nargs="?", default=None,
                        help="Only run time_* methods containing this string")
    parser.add_argument("-w", "--warmup", type=int, default=3,
                        help="Number of warmup iterations (default: 3)")
    parser.add_argument("-n", "--iters", type=int, default=7,
                        help="Number of timed iterations (default: 7)")
    parser.add_argument("--no-save", action="store_true",
                        help="Skip saving results to ASV format")
    args = parser.parse_args()

    if caller_file is not None:
        script_dir = os.path.dirname(os.path.abspath(caller_file))
        suite_name = os.path.splitext(os.path.basename(caller_file))[0]
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        suite_name = args.suite

    os.chdir(script_dir)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    mod = importlib.import_module(suite_name)

    all_results = {}
    for name in sorted(dir(mod)):
        obj = getattr(mod, name)
        if isinstance(obj, type) and name.startswith("Bench"):
            results = run_class(
                suite_name, obj, name, args.method_filter, args.warmup, args.iters)
            all_results.update(results)

    if all_results and not args.no_save:
        save_asv_results(all_results)


if __name__ == "__main__":
    run_as_main()
