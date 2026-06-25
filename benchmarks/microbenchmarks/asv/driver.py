#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""In-process microbenchmark driver.

Discovers ``Bench*`` classes in ``bench_*.py`` files, runs their ``time_*``
methods with robust GPU timing (inner-loop amortization, optional cold cache,
round-robin interleaving), prints a table with throughput, and saves the raw
per-call samples to JSON for ``compare_results.py``.

Usage:
    python driver.py <suite> [method_filter] [-w W] [-n N] [--no-save]
    python driver.py --all [-w W] [-n N]
    python bench_gemm.py [method_filter] [-w W] [-n N]      # bench file as main
"""

import argparse
import glob
import importlib
import itertools
import json
import os
import random
import re
import subprocess
import sys
import time

import numpy as np


# ---------------------------------------------------------------------------
# Benchmark base class
# ---------------------------------------------------------------------------

class BenchBase:
    """Base for benchmark classes: driver-controlled knobs + the timing helper.

    The driver sets timing knobs per (combo, method) before the timed phase:
      _min_run_time_s  -- blocked_autorange target window (auto mode, default 1ms)
      _use_autorange   -- True: blocked_autorange; False: timeit(_inner)
      _inner           -- fixed iteration count when _use_autorange is False
      _scratch         -- cache-flush buffer for --cold-cache mode

    Subclasses time their kernels through :meth:`_time`.
    """

    _inner = 1
    _scratch = None
    _min_run_time_s = 0.001  # 1 ms default, overridden by driver per instance
    _use_autorange = True

    def _time(self, fn):
        """Time *fn* via torch.utils.benchmark; return seconds/call.

        Uses a C++ measurement loop to avoid Python per-iteration overhead.

        * Normal mode (``_use_autorange=True``): ``blocked_autorange`` selects
          the inner iteration count automatically so each sample window lasts
          >= ``_min_run_time_s``.  Mirrors stats suite behaviour.
        * Fixed mode (``_use_autorange=False``): ``timeit(_inner)`` runs
          exactly ``_inner`` iterations.  Used with ``--inner N`` or
          ``--cold-cache`` (which forces ``_inner=1`` and flushes scratch).
        """
        import torch.utils.benchmark as benchmark  # deferred
        if self._scratch is not None:
            self._scratch.fill_(1.0)
        timer = getattr(self, "_timer", None)
        if timer is None:
            timer = self._timer = benchmark.Timer(stmt="fn()", globals={"fn": fn})
        if self._use_autorange:
            if self._inner == 1:
                # First call: use blocked_autorange to calibrate number_per_run
                # and collect the first sample in one shot.
                m = timer.blocked_autorange(min_run_time=self._min_run_time_s)
                self._inner = m.number_per_run  # cache for subsequent calls
            else:
                # Subsequent calls: skip re-calibration, reuse cached count.
                m = timer.timeit(self._inner)
            return m.mean
        return timer.timeit(self._inner).mean


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def _get_commit_hash():
    """Current git HEAD hash, or 'unknown' outside a checkout."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _results_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def save_results(all_results, label=None, results_dir=None):
    """Write raw per-call samples to ``<results_dir>/<hash>[-<label>].json``.

    *label*, when given, is folded into the filename so multiple runs on the same
    commit (e.g. a dirty working tree, where HEAD is unchanged) land in distinct
    files that ``compare_results.py`` can compare instead of overwriting.
    """
    commit = _get_commit_hash()
    results_dir = results_dir or _results_dir()
    os.makedirs(results_dir, exist_ok=True)

    suffix = ""
    if label:
        suffix = "-" + re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("_")
    path = os.path.join(results_dir, f"{commit[:8]}{suffix}.json")

    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
    else:
        data = {"commit_hash": commit, "date": int(time.time() * 1000), "results": {}}
    data["results"].update(all_results)

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {path}")


def _compute_stats(samples):
    """Return ``(median, mean, stdev, q25, q75)`` for *samples*.

    Quartiles use linear interpolation (numpy default), more meaningful at small
    n than index-floor; stdev is the population standard deviation.
    """
    s = np.asarray(samples, dtype=np.float64)
    median, q25, q75 = (float(x) for x in np.quantile(s, [0.5, 0.25, 0.75]))
    return median, float(s.mean()), float(s.std(ddof=0)), q25, q75


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _make_scratch(mb):
    """Allocate a scratch buffer used to evict the GPU cache between samples.

    Sized by default to exceed the MI300 Infinity Cache (256 MB) and the L2
    (16 MB), so a single fill writes through every level of cache.
    """
    import torch  # deferred: only needed when cold-cache is on
    n = max(1, (mb * 1024 * 1024) // 4)  # float32 = 4 bytes
    return torch.empty(n, dtype=torch.float32, device="cuda")


def _autotune_inner(instance, method_name, combo, target_s, max_inner=10000):
    """Pick an inner-loop count so one timed window lasts >= *target_s*.

    Runs two single invocations: one to settle algorithm selection / cache
    state, and one to estimate the per-call cost.
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
    """Release cached GPU memory between interleave chunks (no-op without torch)."""
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
    """Run all ``time_*`` methods in *cls*, returning a ``{bench_key: record}`` dict.

    Samples are collected in round-robin chunks of ``interleave_group``
    ``(method, combo)`` benchmarks: one sample from each per round, for *iters*
    rounds. This spreads every benchmark's samples across the same wall-clock
    window so time-correlated GPU noise (thermal ramp, DVFS throttle) becomes
    shared variance rather than a bias on whichever benchmark owned a contiguous
    block of time. ``interleave_group=1`` reproduces sequential behavior; larger
    groups interleave more but keep that many GPU instances live at once.

    When *shuffle* is true the per-round visit order is randomly permuted (seeded
    by *rng*), making each benchmark's within-round phase and predecessor uniform
    in expectation, turning residual ordering bias into variance. The per-round
    structure is kept (each benchmark still gets exactly *iters* evenly-spread
    samples) -- a balanced randomized design, not a global shuffle.
    """
    methods = sorted(m for m in dir(cls) if m.startswith("time_"))
    if method_filter:
        methods = [m for m in methods if method_filter in m]
    if not methods:
        return {}

    params = getattr(cls, "params", [[]])
    param_names = list(getattr(cls, "param_names", []))
    combos = list(itertools.product(*params))
    n_combos = len(combos)

    # Discover throughput columns from work_* companions.
    # Each entry: (dict_key, column_header, unit_divisor).
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

    target_window_s = target_window_ms / 1000.0
    group = max(1, int(interleave_group))
    if rng is None:
        rng = random.Random(0)
    inner_desc = (
        "cold-cache (inner=1)" if cold_cache
        else f"inner={inner}" if inner != "auto"
        else f"blocked_autorange (>={target_window_ms:g}ms)"
    )
    sched_desc = ("sequential" if group == 1
                  else f"interleaved group={group}, " + ("shuffled" if shuffle else "fixed-order"))
    print(f"\n{class_name}  ({len(combos)} combos x {len(methods)} methods, "
          f"{warmup} warmup, {iters} timed, {inner_desc}, {sched_desc})")
    extra_hdr = "".join(f"  {label:>10}" for _, label, _ in throughput_cols)
    HDR = (f"  {'median':>10}  {'mean':>10}  {'stdev':>10}"
           f"  {'q25':>10}  {'q75':>10}  {'min':>10}  {'max':>10}"
           + extra_hdr + f"  {'inner':>5}  {'method':<30}  params")
    print("-" * len(HDR))
    print(HDR)
    print("-" * len(HDR))

    def _label(combo):
        return ", ".join(f"{nm}={v}" for nm, v in zip(param_names, combo))

    # Samples per method, indexed by combo position. Filling by index decouples
    # the wire format from the order samples are actually collected in, so
    # interleaved scheduling leaves the saved JSON identical to sequential.
    samples_by_method = {m: [None] * n_combos for m in methods}

    # Flatten to (method, combo) tasks, method-major so printed rows keep their
    # grouping, then sample them in round-robin chunks.
    tasks = [(mi, ci) for mi in range(len(methods)) for ci in range(n_combos)]

    for chunk_start in range(0, len(tasks), group):
        chunk = tasks[chunk_start:chunk_start + group]

        # Setup phase: prepare every benchmark in the chunk (allocate tensors,
        # pick _inner, warm up) and keep its instance live for round-robin timing.
        live = []  # (instance, method_name, combo, combo_idx)
        for mi, ci in chunk:
            method_name = methods[mi]
            combo = combos[ci]
            instance = cls()
            try:
                instance.setup(*combo)
            except Exception as e:
                print(f"  SKIP  {_label(combo)}  setup failed: {e}")
                continue  # leaves None in this (method, combo) slot

            # Cold-cache mode forces inner=1 so only the first invocation in the
            # window sees a cold cache; otherwise the 2nd..Nth would refill it.
            if cold_cache:
                instance._scratch = _make_scratch(cache_flush_mb)
                instance._inner = 1
                instance._use_autorange = False
            elif inner == "auto":
                instance._min_run_time_s = target_window_s
                instance._use_autorange = True
            else:
                instance._inner = max(1, int(inner))
                instance._use_autorange = False

            method = getattr(instance, method_name)
            for _ in range(warmup):
                method(*combo)
            live.append((instance, method_name, combo, ci))

        # Timed phase: one sample from each live benchmark per round, so a
        # transient spike lands on one sample of each rather than corrupting a
        # whole benchmark's contiguous block. Visit order is re-permuted each
        # round (when shuffle is on); chunk_samples stays keyed by index i.
        chunk_samples = [[] for _ in live]
        order = list(range(len(live)))
        for _ in range(iters):
            if shuffle and len(order) > 1:
                rng.shuffle(order)
            for i in order:
                instance, method_name, combo, ci = live[i]
                method = getattr(instance, method_name)
                t0 = time.perf_counter()
                result = method(*combo)
                wall = time.perf_counter() - t0
                chunk_samples[i].append(wall if result is None else result)

        # Finalize: stats, throughput, print, store into the combo slot.
        for i, (instance, method_name, combo, ci) in enumerate(live):
            samples = chunk_samples[i]
            median, mean, stdev, q25, q75 = _compute_stats(samples)
            s_min, s_max = min(samples), max(samples)

            # Raw samples (seconds) for statistical comparison; rounded to 1 ns
            # to keep the JSON compact without losing timing resolution.
            samples_by_method[method_name][ci] = [round(x, 9) for x in samples]

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

        live.clear()
        _free_gpu_cache()

    combos_json = [list(c) for c in combos]
    return {
        f"{suite_name}.{class_name}.{m}": {
            "param_names": param_names,
            "combos": combos_json,
            "samples": samples_by_method[m],
        }
        for m in methods
    }


# ---------------------------------------------------------------------------
# Kernel profiling
# ---------------------------------------------------------------------------

_KERNEL_NAME_MAX_WIDTH = 80


def _shorten_kernel_name(name):
    """Shorten verbose C++/HIP kernel names for readable output.

    Strips a leading 'void ', removes template arguments (one level of nesting),
    collapses whitespace, and truncates to ``_KERNEL_NAME_MAX_WIDTH``.
    """
    s = name[5:] if name.startswith("void ") else name
    s = re.sub(r"<[^<>]*(?:<[^<>]*>[^<>]*)*>", "", s)
    s = " ".join(s.split())
    if len(s) > _KERNEL_NAME_MAX_WIDTH:
        s = s[:_KERNEL_NAME_MAX_WIDTH - 3] + "..."
    return s


def profile_class(suite_name, cls, class_name, method_filter=None, warmup=3, inner=1):
    """Per-kernel CUDA-time breakdown for each time_* method x parameter combo.

    Unlike :func:`run_class` (timing distributions), this runs each benchmark
    once under ``torch.profiler`` and reports the GPU kernels it launched, sorted
    by total device time. Returns ``{bench_key: {combo_label: [kernel_row, ...]}}``.
    """
    import torch
    from torch.profiler import profile, ProfilerActivity

    methods = sorted(m for m in dir(cls) if m.startswith("time_"))
    if method_filter:
        methods = [m for m in methods if method_filter in m]
    if not methods:
        return {}

    params = getattr(cls, "params", [[]])
    param_names = list(getattr(cls, "param_names", []))
    combos = list(itertools.product(*params))

    def _label(combo):
        return ", ".join(f"{nm}={v}" for nm, v in zip(param_names, combo))

    out = {}
    for method_name in methods:
        bench_key = f"{suite_name}.{class_name}.{method_name}"
        out[bench_key] = {}
        for combo in combos:
            instance = cls()
            try:
                instance.setup(*combo)
            except Exception as e:
                print(f"  SKIP  {_label(combo)}  setup failed: {e}")
                continue
            instance._inner = max(1, int(inner))
            method = getattr(instance, method_name)
            for _ in range(warmup):
                method(*combo)
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                method(*combo)
                torch.cuda.synchronize()

            events = [e for e in prof.key_averages() if e.self_device_time_total > 0]
            events.sort(key=lambda e: e.self_device_time_total, reverse=True)
            total = sum(e.self_device_time_total for e in events)

            w = _KERNEL_NAME_MAX_WIDTH
            hdr = (f"  {'kernel':<{w}}  {'total us':>11}  {'calls':>6}"
                   f"  {'avg us':>10}  {'%':>6}")
            print(f"\n{bench_key}  ({_label(combo)})")
            print(hdr)
            print("  " + "-" * (len(hdr) - 2))
            rows = []
            for e in events:
                avg = e.self_device_time_total / e.count if e.count else 0.0
                pct = 100.0 * e.self_device_time_total / total if total else 0.0
                print(f"  {_shorten_kernel_name(e.key):<{w}}  {e.self_device_time_total:>11.1f}"
                      f"  {e.count:>6}  {avg:>10.2f}  {pct:>5.1f}%")
                rows.append({
                    "kernel": e.key, "total_us": round(e.self_device_time_total, 1),
                    "calls": e.count, "avg_us": round(avg, 2), "pct": round(pct, 1),
                })
            print(f"  {'TOTAL':<{w}}  {total:>11.1f}")
            out[bench_key][_label(combo)] = rows
    return out


def save_kernel_profile(all_profiles, label=None, results_dir=None):
    """Write per-kernel profiles to ``<results_dir>/<hash>[-<label>]-kernelprofile.json``."""
    commit = _get_commit_hash()
    results_dir = results_dir or _results_dir()
    os.makedirs(results_dir, exist_ok=True)
    suffix = ""
    if label:
        suffix = "-" + re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("_")
    path = os.path.join(results_dir, f"{commit[:8]}{suffix}-kernelprofile.json")
    with open(path, "w") as f:
        json.dump(
            {"commit_hash": commit, "date": int(time.time() * 1000),
             "kernel_profile": all_profiles}, f, indent=2,
        )
    print(f"\nKernel profile saved to {path}")


def run_as_main(caller_file=None):
    """Run benchmarks from a bench file's ``__main__`` block or the command line.

    From a bench file::

        if __name__ == "__main__":
            from driver import run_as_main
            run_as_main(__file__)
    """
    parser = argparse.ArgumentParser(
        description="Run microbenchmarks in-process (no subprocess overhead).")
    if caller_file is None:
        parser.add_argument("suite", nargs="?", default=None,
                            help="Benchmark module name (e.g. bench_casting)")
        parser.add_argument("--all", action="store_true",
                            help="Run all bench_*.py suites in the directory")
    parser.add_argument("method_filter", nargs="?", default=None,
                        help="Only run time_* methods containing this string")
    parser.add_argument("-w", "--warmup", type=int, default=10,
                        help="Number of warmup iterations (default: 10)")
    parser.add_argument("-n", "--iters", type=int, default=20,
                        help="Number of timed iterations (default: 20)")
    parser.add_argument("--inner", default="auto",
                        help="Inner kernel invocations per timed window: 'auto' "
                             "(tune to --target-window-ms) or an integer "
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
                             "is shared across them instead of biasing whichever "
                             "benchmark owns a contiguous block of time "
                             "(default: 8). Each keeps a live GPU instance, so "
                             "lower this on out-of-memory. 1 = sequential.")
    parser.add_argument("--sequential", action="store_true",
                        help="Collect each benchmark's samples contiguously "
                             "(equivalent to --interleave-group 1). Lowest "
                             "memory, but biased under thermal drift.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for the per-round shuffle of the interleave "
                             "order (default: 0), kept fixed for reproducibility.")
    parser.add_argument("--no-shuffle", action="store_true",
                        help="Disable the per-round random permutation and use a "
                             "fixed round-robin order, leaving a small residual "
                             "ordering bias.")
    parser.add_argument("--kernel-profile", action="store_true",
                        help="Profile per-kernel CUDA time via torch.profiler "
                             "instead of measuring timing distributions. Runs each "
                             "benchmark once and prints a per-kernel breakdown "
                             "(saved to <hash>-kernelprofile.json unless --no-save).")
    parser.add_argument("--profile-inner", type=int, default=1,
                        help="Kernel invocations per profiled run in "
                             "--kernel-profile mode (default: 1).")
    parser.add_argument("--no-save", action="store_true",
                        help="Skip saving results to JSON.")
    parser.add_argument("--label", default=None,
                        help="Tag folded into the result filename "
                             "(<hash>-<label>.json). Use it to keep multiple runs "
                             "on the same commit in distinct files for comparison.")
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
        if getattr(args, "all", False):
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
    # --seed; shared across classes so the stream is deterministic end-to-end.
    rng = random.Random(args.seed)
    shuffle = not args.no_shuffle
    if not args.kernel_profile and args.interleave_group > 1 and shuffle:
        print(f"Interleave: group={args.interleave_group}, shuffled (seed={args.seed})")

    all_results = {}
    all_profiles = {}
    for suite_name in suite_names:
        mod = importlib.import_module(suite_name)
        for name in sorted(dir(mod)):
            obj = getattr(mod, name)
            # Any Bench* class that defines a time_* method (excludes BenchBase,
            # and is robust to the bench-file/driver __main__ double-import).
            if not (isinstance(obj, type) and name.startswith("Bench")
                    and any(m.startswith("time_") for m in dir(obj))):
                continue
            if args.kernel_profile:
                all_profiles.update(profile_class(
                    suite_name, obj, name, args.method_filter,
                    warmup=args.warmup, inner=args.profile_inner,
                ))
            else:
                all_results.update(run_class(
                    suite_name, obj, name, args.method_filter,
                    warmup=args.warmup, iters=args.iters,
                    inner=args.inner, target_window_ms=args.target_window_ms,
                    cold_cache=args.cold_cache, cache_flush_mb=args.cache_flush_mb,
                    interleave_group=args.interleave_group, rng=rng, shuffle=shuffle,
                ))

    if args.kernel_profile:
        if all_profiles and not args.no_save:
            save_kernel_profile(all_profiles, label=args.label)
    elif all_results and not args.no_save:
        save_results(all_results, label=args.label)


if __name__ == "__main__":
    run_as_main()
