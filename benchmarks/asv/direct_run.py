#!/usr/bin/env python3
"""Run ASV benchmark classes directly in-process, bypassing subprocess overhead.

Usage:
    python benchmarks/asv/direct_run.py [options] <suite> [method_filter]

Examples:
    python benchmarks/asv/direct_run.py bench_casting
    python benchmarks/asv/direct_run.py bench_gemm time_forward
    python benchmarks/asv/direct_run.py -w 5 -n 20 bench_casting
"""

import argparse
import importlib
import itertools
import math
import sys
import time


def run_class(cls, class_name, method_filter=None, warmup=3, iters=7):
    methods = sorted(m for m in dir(cls) if m.startswith("time_"))
    if method_filter:
        methods = [m for m in methods if method_filter in m]
    if not methods:
        return

    params = getattr(cls, "params", [[]])
    param_names = getattr(cls, "param_names", [])
    combos = list(itertools.product(*params))

    print(f"\n{class_name}  ({len(combos)} combos x {len(methods)} methods, "
          f"{warmup} warmup, {iters} timed)")
    print("-" * 90)
    print(f"  {'median':>10}  {'mean':>10}  {'stdev':>10}  {'method':<30}  params")
    print("-" * 90)

    for combo in combos:
        label = ", ".join(f"{n}={v}" for n, v in zip(param_names, combo))
        instance = cls()
        try:
            instance.setup(*combo)
        except Exception as e:
            print(f"  SKIP  {label}  setup failed: {e}")
            continue

        for method_name in methods:
            method = getattr(instance, method_name)

            for _ in range(warmup):
                method(*combo)

            times = []
            for _ in range(iters):
                t0 = time.perf_counter()
                method(*combo)
                times.append(time.perf_counter() - t0)

            times.sort()
            median = times[len(times) // 2]
            mean = sum(times) / len(times)
            stdev = math.sqrt(sum((t - mean) ** 2 for t in times) / len(times))
            print(f"  {median*1000:>8.3f}ms  {mean*1000:>8.3f}ms  "
                  f"{stdev*1000:>8.3f}ms  {method_name:<30}  {label}")


def main():
    parser = argparse.ArgumentParser(
        description="Run ASV benchmarks directly in-process (no subprocess overhead).")
    parser.add_argument("suite", help="Benchmark module name (e.g. bench_casting)")
    parser.add_argument("method_filter", nargs="?", default=None,
                        help="Only run time_* methods containing this string")
    parser.add_argument("-w", "--warmup", type=int, default=3,
                        help="Number of warmup iterations (default: 3)")
    parser.add_argument("-n", "--iters", type=int, default=7,
                        help="Number of timed iterations (default: 7)")
    args = parser.parse_args()

    mod = importlib.import_module(args.suite)

    for name in sorted(dir(mod)):
        obj = getattr(mod, name)
        if isinstance(obj, type) and name.startswith("Bench"):
            run_class(obj, name, args.method_filter, args.warmup, args.iters)


if __name__ == "__main__":
    import os

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, ".")
    main()
