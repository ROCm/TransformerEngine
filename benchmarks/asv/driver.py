#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""ASV benchmark driver — runs bench classes in-process and saves ASV-compatible results.

Usage:
    python driver.py <suite> [method_filter] [-w W] [-n N] [--no-save]
    python driver.py --all [-w W] [-n N] [--no-save]
    python bench_gemm.py [method_filter] [-w W] [-n N] [--no-save]
"""

import argparse
import glob
import hashlib
import importlib
import inspect
import itertools
import json
import os
import platform
import random
import re
import subprocess
import sys
import textwrap
import time
import numpy as np


# ---------------------------------------------------------------------------
# ASV result generation
# ---------------------------------------------------------------------------

def _get_benchmark_code_and_version(cls, method_name):
    """Build the code string and version hash the same way ASV does.

    ASV hashes a code string built from the time_* and setup methods.
    The string is class header + indented time method + indented setup,
    with no trailing newline.

    Returns (code, version_hash).
    """
    time_src = textwrap.dedent(inspect.getsource(getattr(cls, method_name)))
    setup_src = textwrap.dedent(inspect.getsource(cls.setup))
    code = (
        f"class {cls.__name__}:\n"
        + textwrap.indent(time_src, "    ") + "\n"
        + textwrap.indent(setup_src, "    ")
    ).rstrip("\n")
    return code, hashlib.sha256(code.encode()).hexdigest()


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
    """Return (median, mean, stdev, ci_lo, ci_hi, q25, q75) for *samples*.

    Quartiles use linear interpolation (numpy default) — more meaningful at
    small n than the index-floor approach. stdev is population stdev to
    match the prior wire format; CI is a normal-approximation 99% half-width.
    """
    s = np.asarray(samples, dtype=np.float64)
    mean = float(s.mean())
    stdev = float(s.std(ddof=0))
    median, q25, q75 = (float(x) for x in np.quantile(s, [0.5, 0.25, 0.75]))
    ci = 2.576 * stdev / np.sqrt(s.size)  # 99% normal-approx half-width
    return median, mean, stdev, max(0.0, mean - ci), mean + ci, q25, q75


def _get_results_dir():
    """Read results_dir from asv.conf.json, resolved to an absolute path."""
    conf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "asv.conf.json")
    with open(conf_path) as f:
        conf = json.load(f)
    conf_dir = os.path.dirname(conf_path)
    return os.path.normpath(os.path.join(conf_dir, conf["results_dir"]))


def save_asv_results(all_results, bench_meta, label=None):
    """Write results and benchmark index to ASV's results directory.

    *label*, when given, is folded into the result filename so multiple runs on
    the same commit (e.g. prototyping with a dirty working tree, where the HEAD
    hash is unchanged) land in distinct files that ``compare_results.py`` can
    compare instead of overwriting each other.
    """
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

    # Load existing result file or start fresh. A label is sanitized to keep the
    # filename safe (no path separators / whitespace) and inserted after the hash.
    if label:
        safe_label = re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("_")
        filename = f"{commit_hash[:8]}-{safe_label}-{env_name}.json"
    else:
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

    # Update benchmarks.json index so ASV dashboard stays in sync
    benchmarks_path = os.path.join(results_dir, "benchmarks.json")
    if os.path.exists(benchmarks_path):
        with open(benchmarks_path) as f:
            benchmarks_data = json.load(f)
    else:
        benchmarks_data = {"version": 2}

    benchmarks_data.update(bench_meta)

    with open(benchmarks_path, "w") as f:
        json.dump(benchmarks_data, f, indent=4)

    print(f"Updated {benchmarks_path}")


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

_ASV_META_DEFAULTS = {
    "min_run_count": 2, "number": 0, "repeat": 0, "rounds": 2,
    "sample_time": 0.01, "type": "time", "unit": "seconds", "warmup_time": -1,
}


def _make_scratch(mb):
    """Allocate a scratch buffer used to evict the GPU cache between samples.

    Sized by default to exceed the MI300 Infinity Cache (256 MB) and the L2
    (16 MB), so a single fill writes through every level of cache.
    """
    import torch  # noqa: deferred import — only needed when cold-cache is on
    n = max(1, (mb * 1024 * 1024) // 4)  # float32 = 4 bytes
    return torch.empty(n, dtype=torch.float32, device="cuda")


def _autotune_inner(instance, method_name, combo, target_s, max_inner=10000):
    """Pick an inner-loop count so one timed window lasts >= target_s.

    The bench class is expected to honor instance._inner inside its time_*
    method (loop the kernel that many times in one CUDA event window and
    divide).  This probe runs two single invocations: one to settle algorithm
    selection / cache state, and one to estimate the per-call cost.
    """
    method = getattr(instance, method_name)
    saved_inner = instance._inner
    instance._inner = 1
    try:
        method(*combo)               # discard: cold cache + autotuner warmup
        t_per = method(*combo)       # seconds per single invocation
    finally:
        instance._inner = saved_inner
    if t_per is None or t_per <= 0:
        return 1
    return max(1, min(max_inner, int(target_s / t_per) + 1))


def _free_gpu_cache():
    """Release cached GPU memory between interleave chunks.

    No-op when torch was never imported (e.g. CPU-only test harnesses), so the
    driver stays importable and runnable without torch present.
    """
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def run_class(
    suite_name, cls, class_name, method_filter=None,
    warmup=3, iters=7,
    inner="auto", target_window_ms=1.0,
    cold_cache=False, cache_flush_mb=256,
    interleave_group=8, rng=None, shuffle=True,
):
    """Run all benchmarks in a class, returning (results, metadata) dicts.

    Samples are collected in round-robin chunks of ``interleave_group``
    ``(method, combo)`` benchmarks: one sample is taken from each benchmark in
    the chunk per round, for ``iters`` rounds. This spreads every benchmark's
    samples across the same wall-clock window so time-correlated GPU noise
    (thermal ramp, DVFS throttle) becomes shared variance rather than a bias on
    whichever benchmark happened to own a contiguous block of time. See
    ``repro/transient_noise_sim.py``. ``interleave_group=1`` reproduces the
    original contiguous (sequential) behavior; larger groups interleave more
    benchmarks but keep that many GPU instances live at once.

    When ``shuffle`` is true the per-round visit order is randomly permuted
    (seeded by *rng*, a ``random.Random``; one is created with seed 0 if not
    given). Fixed round-robin still pins each benchmark to a constant phase
    within the round, so a monotonic ramp leaves a small constant per-benchmark
    offset and each benchmark always sees the same predecessor's cache/clock
    state. Permuting each round makes both uniform in expectation, turning that
    residual bias into variance. The per-round structure is kept (each benchmark
    still gets exactly ``iters`` evenly-spread samples) -- a balanced randomized
    design, not a global shuffle that could re-cluster a benchmark's samples.
    """
    methods = sorted(m for m in dir(cls) if m.startswith("time_"))
    if method_filter:
        methods = [m for m in methods if method_filter in m]
    if not methods:
        return {}, {}

    params = getattr(cls, "params", [[]])
    param_names = getattr(cls, "param_names", [])
    combos = list(itertools.product(*params))
    asv_params = [[_format_param_value(v) for v in dim] for dim in params]

    # Discover throughput columns from work_* companions
    # Each entry: (dict_key, column_header, unit_divisor)
    probe_keys = set()
    for m in methods:
        wfn = getattr(cls, "work_" + m[5:], None)
        if wfn:
            try:
                probe_keys.update(wfn(cls(), *combos[0]))
            except Exception:
                pass
    throughput_cols = []
    if "flops" in probe_keys:
        throughput_cols.append(("flops", "TFLOPS", 1e12))
    if "bytes" in probe_keys:
        throughput_cols.append(("bytes", "GB/s", 1e9))

    # Print table header
    target_window_s = target_window_ms / 1000.0
    group = max(1, int(interleave_group))
    if rng is None:
        rng = random.Random(0)
    inner_desc = (
        "cold-cache (inner=1)" if cold_cache
        else f"inner={inner}" if inner != "auto"
        else f"inner=auto (>={target_window_ms:g}ms window)"
    )
    if group == 1:
        sched_desc = "sequential"
    else:
        sched_desc = f"interleaved group={group}, " + ("shuffled" if shuffle else "fixed-order")
    print(f"\n{class_name}  ({len(combos)} combos x {len(methods)} methods, "
          f"{warmup} warmup, {iters} timed, {inner_desc}, {sched_desc})")
    extra_hdr = "".join(f"  {label:>10}" for _, label, _ in throughput_cols)
    HDR = (f"  {'median':>10}  {'mean':>10}  {'stdev':>10}"
           f"  {'q25':>10}  {'q75':>10}  {'min':>10}  {'max':>10}"
           + extra_hdr + f"  {'inner':>5}  {'method':<30}  params")
    print("-" * len(HDR))
    print(HDR)
    print("-" * len(HDR))

    all_results = {}
    all_meta = {}

    # Per-method result columns, indexed by combo position. Filling by index
    # decouples the wire format from the order samples are actually collected in,
    # so interleaved scheduling leaves the saved JSON identical to sequential.
    n_combos = len(combos)
    cols = {
        m: {k: [None] * n_combos for k in
            ("median", "ci_lo", "ci_hi", "q25", "q75", "number", "repeat", "samples")}
        for m in methods
    }
    versions = {}
    for method_name in methods:
        bench_key = f"{suite_name}.{class_name}.{method_name}"
        code, version = _get_benchmark_code_and_version(cls, method_name)
        versions[method_name] = version
        all_meta[bench_key] = {
            **_ASV_META_DEFAULTS,
            "code": code, "name": bench_key, "version": version,
            "param_names": list(param_names), "params": asv_params,
            "timeout": getattr(cls, "timeout", 300),
        }

    def _label(combo):
        return ", ".join(f"{nm}={v}" for nm, v in zip(param_names, combo))

    # Flatten to (method, combo) tasks, method-major so printed rows keep the
    # same grouping as before, then sample them in round-robin chunks.
    tasks = [(mi, ci) for mi in range(len(methods)) for ci in range(n_combos)]
    started_at = int(time.time() * 1000)
    t_start = time.perf_counter()

    for chunk_start in range(0, len(tasks), group):
        chunk = tasks[chunk_start:chunk_start + group]

        # Setup phase: prepare every benchmark in the chunk (allocate tensors,
        # pick _inner, warm up) and keep its instance live for round-robin timing.
        live = []  # (instance, method_obj, method_name, combo, combo_idx)
        for mi, ci in chunk:
            method_name = methods[mi]
            combo = combos[ci]
            instance = cls()
            try:
                instance.setup(*combo)
            except Exception as e:
                print(f"  SKIP  {_label(combo)}  setup failed: {e}")
                continue  # leaves None in this (method, combo) slot

            # Inner-loop and cache configuration. Cold-cache mode forces
            # inner=1 so only the first invocation in the window sees a
            # cold cache; otherwise the 2nd..Nth invocations would refill
            # it and we'd be back to a warm-cache measurement.
            if cold_cache:
                instance._scratch = _make_scratch(cache_flush_mb)
                instance._inner = 1
            elif inner == "auto":
                instance._inner = _autotune_inner(
                    instance, method_name, combo, target_window_s)
            else:
                instance._inner = max(1, int(inner))

            method = getattr(instance, method_name)
            for _ in range(warmup):
                method(*combo)
            live.append((instance, method, method_name, combo, ci))

        # Timed phase: one sample from each live benchmark per round, so a
        # transient spike lands on one sample of each rather than corrupting a
        # whole benchmark's contiguous block. The visit order is re-permuted
        # each round (when shuffle is on) so no benchmark is pinned to a fixed
        # phase / predecessor; chunk_samples stays keyed by the stable index i.
        chunk_samples = [[] for _ in live]
        order = list(range(len(live)))
        for _ in range(iters):
            if shuffle and len(order) > 1:
                rng.shuffle(order)
            for i in order:
                instance, method, method_name, combo, ci = live[i]
                t0 = time.perf_counter()
                result = method(*combo)
                wall = time.perf_counter() - t0
                chunk_samples[i].append(wall if result is None else result)

        # Finalize phase: stats, throughput, print, store into the combo slot.
        for i, (instance, method, method_name, combo, ci) in enumerate(live):
            samples = chunk_samples[i]
            median, mean, stdev, ci_lo, ci_hi, q25, q75 = _compute_stats(samples)
            s_min, s_max = min(samples), max(samples)

            c = cols[method_name]
            c["median"][ci] = median
            c["ci_lo"][ci] = ci_lo
            c["ci_hi"][ci] = ci_hi
            c["q25"][ci] = q25
            c["q75"][ci] = q75
            c["number"][ci] = instance._inner
            c["repeat"][ci] = iters
            # Keep the raw samples (seconds) for statistical comparison
            # (compare_results.py). Rounded to 1 ns to keep the JSON compact
            # without losing meaningful timing resolution.
            c["samples"][ci] = [round(x, 9) for x in samples]

            # Derive throughput from work_* companion
            work = {}
            wfn = getattr(instance, "work_" + method_name[5:], None)
            if wfn and median > 0:
                try:
                    work = wfn(*combo)
                except Exception:
                    pass
            extra_cols = ""
            for key, _, divisor in throughput_cols:
                if key in work and median > 0:
                    extra_cols += f"  {work[key] / median / divisor:>10.1f}"
                else:
                    extra_cols += f"  {'':>10}"

            print(f"  {median*1000:>8.3f}ms  {mean*1000:>8.3f}ms  "
                  f"{stdev*1000:>8.3f}ms  {q25*1000:>8.3f}ms  {q75*1000:>8.3f}ms  "
                  f"{s_min*1000:>8.3f}ms  {s_max*1000:>8.3f}ms"
                  f"{extra_cols}  "
                  f"{instance._inner:>5}  {method_name:<30}  {_label(combo)}")

        # Release this chunk's GPU instances before setting up the next chunk.
        live.clear()
        _free_gpu_cache()

    duration = time.perf_counter() - t_start
    for method_name in methods:
        bench_key = f"{suite_name}.{class_name}.{method_name}"
        c = cols[method_name]
        all_results[bench_key] = [
            c["median"], asv_params, versions[method_name], started_at,
            round(duration, 2),
            c["ci_lo"], c["ci_hi"], c["q25"], c["q75"], c["number"], c["repeat"],
            c["samples"],
        ]

    return all_results, all_meta


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
        parser.add_argument("suite", nargs="?", default=None,
                            help="Benchmark module name (e.g. bench_casting)")
        parser.add_argument("--all", action="store_true",
                            help="Run all bench_*.py suites in the directory")
    parser.add_argument("method_filter", nargs="?", default=None,
                        help="Only run time_* methods containing this string")
    parser.add_argument("-w", "--warmup", type=int, default=10,
                        help="Number of warmup iterations (default: 3)")
    parser.add_argument("-n", "--iters", type=int, default=20,
                        help="Number of timed iterations (default: 7)")
    parser.add_argument("--inner", default="auto",
                        help="Inner kernel invocations per timed window: "
                             "'auto' (tune to --target-window-ms) or an integer "
                             "(default: auto). Larger values amortize CUDA event "
                             "and kernel-launch overhead.")
    parser.add_argument("--target-window-ms", type=float, default=1.0,
                        help="Target duration of one timed window when "
                             "--inner=auto (default: 1.0 ms).")
    parser.add_argument("--cold-cache", action="store_true",
                        help="Flush the GPU cache (write a >LLC scratch buffer) "
                             "before each sample. Forces --inner=1 because "
                             "subsequent inner calls would refill the cache.")
    parser.add_argument("--cache-flush-mb", type=int, default=256,
                        help="Size in MB of the cache-flush buffer for "
                             "--cold-cache (default: 256, sized for the MI300 "
                             "Infinity Cache).")
    parser.add_argument("--interleave-group", type=int, default=8,
                        help="Number of (method, combo) benchmarks sampled "
                             "round-robin together so time-correlated GPU noise "
                             "(thermal ramp / DVFS throttle) is shared across "
                             "them instead of biasing whichever benchmark owns a "
                             "contiguous block of wall-clock time (default: 8). "
                             "Each benchmark in a group keeps a live GPU "
                             "instance, so lower this on out-of-memory. 1 = "
                             "sequential. See repro/transient_noise_sim.py.")
    parser.add_argument("--sequential", action="store_true",
                        help="Collect each benchmark's samples in one contiguous "
                             "block (equivalent to --interleave-group 1). Lowest "
                             "memory, but biased under thermal drift.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for the per-round shuffle of the interleave "
                             "order (default: 0), kept fixed so runs are "
                             "reproducible.")
    parser.add_argument("--no-shuffle", action="store_true",
                        help="Disable the per-round random permutation and use a "
                             "fixed round-robin order. Each benchmark then keeps "
                             "a constant within-round phase and predecessor, "
                             "leaving a small residual ordering bias.")
    parser.add_argument("--no-save", action="store_true",
                        help="Skip saving results to ASV format")
    parser.add_argument("--label", default=None,
                        help="Tag folded into the result filename "
                             "(<hash>-<label>-<env>.json). Use it to keep "
                             "multiple runs on the same commit (e.g. a dirty "
                             "working tree) in distinct files for comparison.")
    args = parser.parse_args()
    if args.inner != "auto":
        try:
            args.inner = max(1, int(args.inner))
        except ValueError:
            parser.error("--inner must be 'auto' or a positive integer")
    if args.sequential:
        args.interleave_group = 1
    args.interleave_group = max(1, args.interleave_group)

    if caller_file is not None:
        script_dir = os.path.dirname(os.path.abspath(caller_file))
        suite_names = [os.path.splitext(os.path.basename(caller_file))[0]]
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        run_all = getattr(args, "all", False)
        if run_all:
            suite_names = sorted(
                os.path.splitext(os.path.basename(f))[0]
                for f in glob.glob(os.path.join(script_dir, "bench_*.py"))
            )
        elif args.suite:
            suite_names = [args.suite]
        else:
            parser.error("provide a suite name or use --all")

    os.chdir(script_dir)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    # One RNG for the whole run so the interleave order is reproducible given
    # --seed. Shared across classes so the stream is deterministic end-to-end.
    rng = random.Random(args.seed)
    shuffle = not args.no_shuffle
    if args.interleave_group > 1 and shuffle:
        print(f"Interleave: group={args.interleave_group}, shuffled (seed={args.seed})")

    all_results = {}
    all_meta = {}
    for suite_name in suite_names:
        mod = importlib.import_module(suite_name)
        for name in sorted(dir(mod)):
            obj = getattr(mod, name)
            if isinstance(obj, type) and name.startswith("Bench"):
                results, meta = run_class(
                    suite_name, obj, name, args.method_filter,
                    warmup=args.warmup, iters=args.iters,
                    inner=args.inner, target_window_ms=args.target_window_ms,
                    cold_cache=args.cold_cache,
                    cache_flush_mb=args.cache_flush_mb,
                    interleave_group=args.interleave_group,
                    rng=rng, shuffle=shuffle,
                )
                all_results.update(results)
                all_meta.update(meta)

    if all_results and not args.no_save:
        save_asv_results(all_results, all_meta, label=args.label)


if __name__ == "__main__":
    run_as_main()
