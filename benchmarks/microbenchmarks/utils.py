#!/usr/bin/env python
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Shared utilities for microbenchmarks: model configs, timing, throughput, runner."""

import argparse
import importlib.util
import itertools
import math
from types import SimpleNamespace
import torch
import torch.utils.benchmark as benchmark

# ---------------------------------------------------------------------------
# Sequence / batch-token sizes
# ---------------------------------------------------------------------------
M_SIZE_LIST = [1024, 2048, 4096, 8192]

# Shared dtype sweep for TE activation benchmarks. Extend this list to add
# additional precisions such as torch.float16.
DTYPE_LIST = [torch.bfloat16]

DEFAULT_MIN_RUN_TIME_SECONDS = 0.2

# ---------------------------------------------------------------------------
# Model configurations
# ---------------------------------------------------------------------------
# (name, hidden, intermediate, num_q_heads, num_kv_heads, head_dim, tp)
#
# Sources:
# - Llama 3.1 8B   https://huggingface.co/meta-llama/Llama-3.1-8B/blob/main/config.json
# - Llama 3.1 70B  https://huggingface.co/meta-llama/Llama-3.1-70B/blob/main/config.json
# - Llama 3.1 405B https://huggingface.co/meta-llama/Llama-3.1-405B/blob/main/config.json
# - Qwen 2.5 7B  https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json
# - Qwen 2.5 72B https://huggingface.co/Qwen/Qwen2.5-72B-Instruct/blob/main/config.json

MODEL_CONFIGS = [
    ("Llama3.1-8B/TP1",   4096,  14336,  32,  8, 128,  1),
    ("Llama3.1-8B/TP8",   4096,  14336,  32,  8, 128,  8),
    ("Llama3.1-70B/TP8",  8192,  28672,  64,  8, 128,  8),
    ("Llama3.1-405B/TP8", 16384, 53248, 128,  8, 128,  8),
    ("Qwen2.5-7B/TP1",  3584, 18944,  28,  4, 128,  1),
    ("Qwen2.5-72B/TP8", 8192, 29568,  64,  8, 128,  8),
]

# Unique (model_name, hidden_size) pairs for element-wise benchmarks
MODEL_HIDDEN_SIZES = [
    ("Llama3-8B",   4096),
    ("Llama3-70B",  8192),
    ("Llama3-405B", 16384),
    ("Qwen2.5-7B",  3584),
    ("Qwen2.5-72B", 8192),
]


def gemm_shapes(configs=None):
    """Generate {case_name: (N, K)} dict from MODEL_CONFIGS.

    Each model contributes up to four GEMM shapes:
      QKV, AttnOut, GateUp (SwiGLU), Down.
    """
    shapes = {}
    for (name, hidden, intermediate, n_q, n_kv, hd, tp) in (configs or MODEL_CONFIGS):
        shapes[f"{name}-QKV"]     = ((n_q * hd + 2 * n_kv * hd) // tp, hidden)
        shapes[f"{name}-AttnOut"] = (hidden, (n_q * hd) // tp)
        shapes[f"{name}-GateUp"]  = ((2 * intermediate) // tp, hidden)
        shapes[f"{name}-Down"]    = (hidden, intermediate // tp)
    return shapes


def generate_gemm_test_cases(configs=None, m_sizes=None, dtypes=None):
    """Generate dense GEMM benchmark cases shared by BF16 and FP8 GEMM."""
    test_cases = []
    active_shapes = gemm_shapes(configs)
    for m_value in (m_sizes or M_SIZE_LIST):
        for case_name, (n_value, k_value) in active_shapes.items():
            for dtype in (dtypes or DTYPE_LIST):
                test_cases.append({
                    "Case": case_name,
                    "M": m_value,
                    "N": n_value,
                    "K": k_value,
                    "dtype": dtype,
                })
    return test_cases


# ---------------------------------------------------------------------------
# Low-precision recipe sweep (shared by the dense and grouped GEMM benchmarks)
# ---------------------------------------------------------------------------
# Transformer Engine imports are deferred into the helpers so importing
# utils.py stays free of a GPU / built TE (keeps the non-GEMM benchmarks and
# offline tooling importable).


def _check_mxfp4_support_with_aiter():
    """MXFP4 gate: device support plus the aiter FP4 GEMM backend.

    The MXFP4 GEMM path calls into aiter's a4w4 kernels, so a missing aiter
    package would crash at benchmark time even on supported hardware.
    """
    from transformer_engine.pytorch.quantization import check_mxfp4_support

    supported, reason = check_mxfp4_support()
    if not supported:
        return supported, reason
    if importlib.util.find_spec("aiter") is None:
        return False, "aiter is not installed (required for the MXFP4 GEMM backend)."
    return True, ""


def _precision_specs():
    """Ordered sweep of (name, recipe factory | None, support check | None).

    A ``None`` factory is the bf16 baseline (no autocast). The fp8 entry uses
    HYBRID delayed scaling and is shared by the dense and grouped GEMM
    benchmarks so their fp8 numbers stay comparable.
    """
    from transformer_engine.common.recipe import (
        DelayedScaling,
        Format,
        MXFP4BlockScaling,
        MXFP8BlockScaling,
        NVFP4BlockScaling,
    )
    from transformer_engine.pytorch.quantization import (
        check_fp8_support,
        check_mxfp8_support,
        check_nvfp4_support,
    )

    return (
        ("bf16", None, None),
        (
            "fp8",
            lambda: DelayedScaling(
                fp8_format=Format.HYBRID,
                amax_history_len=16,
                amax_compute_algo="max",
            ),
            check_fp8_support,
        ),
        ("mxfp8", MXFP8BlockScaling, check_mxfp8_support),
        ("mxfp4", MXFP4BlockScaling, _check_mxfp4_support_with_aiter),
        ("nvfp4", NVFP4BlockScaling, check_nvfp4_support),
    )


def build_recipes(names=None):
    """Build an ordered ``{name: recipe_or_None}`` sweep of supported precisions.

    ``bf16`` maps to ``None`` (no autocast). Each low-precision entry is
    included only when its support check passes on the current device;
    unsupported ones are dropped with a short notice. Pass *names* to restrict
    and order the sweep, e.g. ``("bf16", "fp8", "mxfp8", "nvfp4")`` for grouped
    GEMM, which has no MXFP4 grouped kernel.
    """
    specs = _precision_specs()
    if names is not None:
        by_name = {spec[0]: spec for spec in specs}
        specs = tuple(by_name[name] for name in names)
    recipes = {}
    for name, factory, support_check in specs:
        if support_check is not None:
            supported, reason = support_check()
            if not supported:
                print(f"Skipping {name} precision: {reason}")
                continue
        recipes[name] = factory() if factory is not None else None
    return recipes


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def time_func(fn, method="adaptive", min_run_time=DEFAULT_MIN_RUN_TIME_SECONDS):
    """Time *fn* and return ``(mean_ms, measurement)``.

    The ``Measurement`` object carries per-sample times accessible via
    ``measurement.times`` (total wall time per run) and
    ``measurement.number_per_run``.

    method: "adaptive" uses adaptive_autorange (good for compute-bound),
            "blocked"  uses blocked_autorange  (good for memory-bound).
    """
    timer = benchmark.Timer(stmt="fn()", globals={"fn": fn})
    if method == "blocked":
        m = timer.blocked_autorange(min_run_time=min_run_time)
    else:
        m = timer.adaptive_autorange(min_run_time=min_run_time)
    return m.mean * 1e3, m


# ---------------------------------------------------------------------------
# Rotating input buffers (on by default; disable via --no-rotating)
# ---------------------------------------------------------------------------
# Benchmark inputs are cycled through a ring of buffers so that back-to-back
# kernel launches read different input memory and don't benefit from artificial
# cache residency.  Populated by run_benchmarks() from the parsed CLI args.
_ROTATE_BUFFERS = True
_ROTATE_MB = 0  # rotation memory budget in MB; 0 => auto-size to exceed the LLC
# Ceiling on the rotation ring size. hipBLASLt-bench caps its rotating block
# count at the iteration count (max(cold_iters, iters)) so it never allocates a
# buffer it won't revisit. torch.utils.benchmark picks the iteration count
# adaptively, so there is no fixed value to cap against; we instead bound the
# ring at a fixed maximum (mirroring hipBLASLt's default cold_iters of 1000).
# With the auto budget this ceiling is never reached; it only guards a very
# large explicit --rotating budget on a small buffer, which would otherwise
# allocate a copy per few MB up to the whole budget.
_ROTATE_MAX_BUFFERS = 1000


def _last_level_cache_bytes():
    """Bytes of the last-level cache that buffer rotation must exceed.

    HIP reports ``L2_cache_size`` as the small per-XCD L2 (e.g. 4 MB on gfx950),
    but the real last-level cache is the much larger AMD Infinity Cache.

    Actual last-level/Infinity Cache sizes:
      - gfx942 / gfx950: 256 MB
      - gfx1250:         192 MB

    We use 256 MB for all devices: a slightly oversized ring is harmless (it
    only allocates a little more memory) and avoids per-arch probing.
    """
    return 256 * 1024 * 1024


def _rotation_count(bytes_per_buffer, cache_mult=2.0, min_buffers=2):
    """Number of buffers so the rotation ring spans the requested memory budget.

    With an explicit ``--rotating MB`` the budget is that many megabytes; when
    omitted it is *cache_mult* x the last-level cache (the ~256 MB AMD Infinity
    Cache), so a buffer is evicted before it is reused.  The ring is floored at
    *min_buffers* (so enabling rotation always rotates) and capped at
    ``_ROTATE_MAX_BUFFERS`` (the adaptive-timer analog of hipBLASLt-bench capping
    its block count at the iteration count, so a huge budget on a small buffer
    can't allocate an unbounded ring).
    """
    if bytes_per_buffer <= 0:
        return min_buffers
    if _ROTATE_MB and _ROTATE_MB > 0:
        budget = _ROTATE_MB * 1024 * 1024
    else:
        cache = _last_level_cache_bytes()
        if not cache:
            return min_buffers
        budget = cache_mult * cache
    count = math.ceil(budget / bytes_per_buffer)
    if _ROTATE_MAX_BUFFERS and _ROTATE_MAX_BUFFERS > 0:
        count = min(count, _ROTATE_MAX_BUFFERS)
    return max(min_buffers, count)


def _tensor_nbytes(t):
    """Byte size of a torch tensor, or 0 if it can't be determined."""
    numel = getattr(t, "numel", None)
    element_size = getattr(t, "element_size", None)
    if callable(numel) and callable(element_size):
        return int(numel()) * int(element_size())
    return 0


def rotating(build, *, bytes_per_buffer=None):
    """Return a zero-arg callable yielding an input buffer to time.

    Rotation is on by default: it builds a ring of ``build()`` buffers (spanning
    the ``--rotating MB`` budget, or ~2x the last-level cache when the size is
    omitted) and returns the next one on each call.  With ``--no-rotating`` it
    returns a single cached buffer from ``build()`` on every call, matching the
    original single-buffer behavior.

    ``build`` is a zero-arg callable returning one fresh buffer.
    ``bytes_per_buffer`` overrides the auto-sizing hint for buffers whose byte
    size can't be inferred (e.g. FP8 tensors).
    """
    first = build()
    if not _ROTATE_BUFFERS:
        return lambda: first
    nbytes = bytes_per_buffer if bytes_per_buffer is not None else _tensor_nbytes(first)
    count = _rotation_count(nbytes)
    buffers = [first] + [build() for _ in range(max(0, count - 1))]
    ring = itertools.cycle(buffers)
    return lambda: next(ring)


def make_input(shape, dtype, *, device="cuda", requires_grad=False):
    """Rotation-aware input: a zero-arg callable returning a ``randn`` tensor.

    Honors ``--rotating`` (see :func:`rotating`); on by default, so it returns
    the next tensor in the ring each call (``--no-rotating`` for a single one).
    """
    return rotating(
        lambda: torch.randn(
            *shape, dtype=dtype, device=device, requires_grad=requires_grad
        )
    )


# ---------------------------------------------------------------------------
# Throughput helpers
# ---------------------------------------------------------------------------

def compute_tflops(flops, ms):
    """TFLOPS from operation count and milliseconds."""
    return flops / (ms * 1e-3) / 1e12


def compute_gbps(nbytes, ms):
    """GB/s from byte count and milliseconds."""
    return nbytes / (ms * 1e-3) / 1e9


def make_metric_record(label, ms, unit, throughput, derived=False,
                       ms_precision=3, throughput_precision=2,
                       measurement=None, samples_only=False):
    """Create a structured metric record for stdout and CSV generation.

    Each record describes one benchmark line item such as "GEMM Forward".
    ``run_benchmarks`` formats these records for stdout and expands them into
    ``<label> Time (ms)`` and ``<label> <unit>`` CSV columns.

    If *measurement* is provided (a ``torch.utils.benchmark.Measurement``),
    the per-sample times are available for the ``--csv-samples`` output.
    Records with *samples_only=True* are excluded from stdout and the main
    CSV but their samples are still written to the samples CSV.
    """
    return {
        "label": label,
        "ms": ms,
        "unit": unit,
        "throughput": throughput,
        "derived": derived,
        "ms_precision": ms_precision,
        "throughput_precision": throughput_precision,
        "measurement": measurement,
        "samples_only": samples_only,
    }


def make_forward_backward_metric_records(label_prefix, unit,
                                         forward_ms, forward_throughput,
                                         backward_ms, backward_throughput,
                                         backward_derived=False,
                                         ms_precision=3,
                                         throughput_precision=2,
                                         fwd_measurement=None,
                                         bwd_measurement=None,
                                         fwd_bwd_measurement=None):
    """Create standard forward/backward metric records for a benchmark.

    When *backward_derived* is True and *fwd_bwd_measurement* is provided,
    an extra samples-only record for "Forward+Backward" is emitted so that
    the raw timing samples are preserved in the ``--csv-samples`` output.
    """
    records = [
        make_metric_record(
            f"{label_prefix} Forward",
            forward_ms,
            unit,
            forward_throughput,
            ms_precision=ms_precision,
            throughput_precision=throughput_precision,
            measurement=fwd_measurement,
        ),
        make_metric_record(
            f"{label_prefix} Backward",
            backward_ms,
            unit,
            backward_throughput,
            derived=backward_derived,
            ms_precision=ms_precision,
            throughput_precision=throughput_precision,
            measurement=bwd_measurement,
        ),
    ]
    if fwd_bwd_measurement is not None:
        records.append(make_metric_record(
            f"{label_prefix} Forward+Backward",
            forward_ms + backward_ms,
            unit,
            0,
            samples_only=True,
            measurement=fwd_bwd_measurement,
        ))
    return records


def direction_records(direction, label, unit, throughput,
                      fwd_func, fwd_bwd_func, fwd_work, bwd_work):
    """Metric records for a forward-only or a derived-backward timing.

    *direction* is ``"fwd"`` or ``"bwd"``. *throughput* is ``compute_tflops`` or
    ``compute_gbps`` and *fwd_work* / *bwd_work* the matching flops / bytes.
    Backward is ``(fwd+bwd) - fwd``; its per-sample distribution is each fwd+bwd
    sample shifted by the fwd mean (fwd and fwd+bwd are timed separately, so the
    spread is inherited from fwd+bwd).
    """
    if direction == "fwd":
        fwd_ms, fwd_measurement = time_func(fwd_func)
        return [make_metric_record(
            label, fwd_ms, unit, throughput(fwd_work, fwd_ms), measurement=fwd_measurement,
        )]
    fwd_bwd_func()  # warm the backward graph
    fwd_ms, fwd_measurement = time_func(fwd_func)
    fwd_bwd_ms, fwd_bwd_measurement = time_func(fwd_bwd_func)
    bwd_ms = fwd_bwd_ms - fwd_ms
    fwd_mean_s = fwd_measurement.mean
    bwd_measurement = SimpleNamespace(
        times=[t - fwd_mean_s for t in fwd_bwd_measurement.times]
    )
    return [make_metric_record(
        label, bwd_ms, unit, throughput(bwd_work, bwd_ms),
        derived=True, measurement=bwd_measurement,
    )]


def _metric_time_key(metric):
    return f"{metric['label']} Time (ms)"


def _metric_throughput_key(metric):
    return f"{metric['label']} {metric['unit']}"


def _format_metric_number(value, precision):
    return f"{value:.{precision}f}"


def _metric_row_from_records(metric_records):
    row = {}
    for metric in metric_records:
        if metric.get("samples_only"):
            continue
        row[_metric_time_key(metric)] = _format_metric_number(
            metric["ms"], metric.get("ms_precision", 3)
        )
        row[_metric_throughput_key(metric)] = _format_metric_number(
            metric["throughput"], metric.get("throughput_precision", 2)
        )
    return row


def _print_metric_records(metric_records):
    printable = [m for m in metric_records if not m.get("samples_only")]
    if not printable:
        return
    label_width = max(24, *(len(metric["label"]) for metric in printable))
    for metric in printable:
        ms_str = _format_metric_number(metric["ms"], metric.get("ms_precision", 3))
        throughput_str = _format_metric_number(
            metric["throughput"], metric.get("throughput_precision", 2)
        )
        derived_suffix = " (derived)" if metric.get("derived", False) else ""
        print(
            f"  {metric['label']:<{label_width}} {ms_str} ms | "
            f"{throughput_str} {metric['unit']}{derived_suffix}"
        )


def _default_csv_name(bench_fn):
    import inspect
    from pathlib import Path
    return Path(inspect.getfile(bench_fn)).with_suffix(".csv").name


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def make_parser(**kwargs):
    """Return an :class:`~argparse.ArgumentParser` with ``--csv``, ``--csv-samples``, and ``--kernel-profile`` flags.

    Any *kwargs* are forwarded to the ``ArgumentParser`` constructor, so
    callers can set ``description``, ``parents``, etc.
    """
    parser = argparse.ArgumentParser(**kwargs)
    parser.add_argument(
        "--csv", nargs="?", const=True, default=None, metavar="FILE",
        help="Write results to CSV. Optional filename; default derived from script name.",
    )
    parser.add_argument(
        "--csv-samples", nargs="?", const=True, default=None, metavar="FILE",
        help=(
            "Write per-sample timing data to a CSV for downstream analysis. "
            "Optional filename; default derived from script name."
        ),
    )
    parser.add_argument(
        "--kernel-profile", action="store_true", default=False,
        help=(
            "Profile GPU kernels using torch.profiler in addition to normal "
            "timing. Prints per-kernel CUDA times output. "
            "Use with --csv to write kernel-level data to CSV. "
            "--csv-samples is ignored in this mode."
        ),
    )
    rotating_group = parser.add_mutually_exclusive_group()
    rotating_group.add_argument(
        "--rotating", nargs="?", type=int, const=0, default=None, metavar="MB",
        help=(
            "Rotate benchmark inputs through a ring of buffers so back-to-back "
            "launches touch different memory (avoids artificial cache "
            "residency), like hipBLASLt-bench --rotating. Optionally pass the "
            "rotating memory budget in MB; omit it to auto-size the ring to "
            "exceed the last-level cache (the 256 MB Infinity Cache on "
            "gfx942/gfx950, not just L2). On by default; disable with "
            "--no-rotating."
        ),
    )
    rotating_group.add_argument(
        "--no-rotating", action="store_true", default=False,
        help=(
            "Disable input buffer rotation (see --rotating) and time a single "
            "cached input buffer."
        ),
    )
    return parser


_KERNEL_NAME_MAX_WIDTH = 80


def _shorten_kernel_name(name):
    """Shorten verbose C++/HIP kernel names for readable terminal output.

    Strips ``void `` prefix and template arguments (``<...>``) from
    fully-qualified kernel names while preserving the function name and
    any top-level namespace.
    """
    import re
    s = name
    # Strip leading "void "
    if s.startswith("void "):
        s = s[5:]
    # Remove balanced template args (handles one level of nesting)
    s = re.sub(r"<[^<>]*(?:<[^<>]*>[^<>]*)*>", "", s)
    # Collapse whitespace
    s = " ".join(s.split())
    if len(s) > _KERNEL_NAME_MAX_WIDTH:
        s = s[: _KERNEL_NAME_MAX_WIDTH - 3] + "..."
    return s


def run_benchmarks(test_cases, bench_fn, param_columns, default_csv=None,
                   args=None):
    """Iterate *test_cases*, call *bench_fn*, and optionally write a CSV.

    Parameters
    ----------
    test_cases : list[dict]
        Each dict has at least the keys in *param_columns* plus any extra
        keys the bench_fn needs (passed as **case).
    bench_fn : callable
        Called as ``bench_fn(**case)`` and must return a list of metric
        records created by ``make_metric_record``. Each record corresponds to
        one stdout line and expands to a time column plus a throughput column in
        the CSV output.
    param_columns : list[str]
        Column names to pull from each test case into the output row.
    default_csv : str or None
        Default CSV filename used when ``--csv`` is passed without a
        filename. If omitted, the CSV name is derived from the caller's
        file name. CSV output is only written when the caller passes
        ``--csv`` on the command line.
    args : argparse.Namespace or None
        Pre-parsed arguments.  When a benchmark script needs its own CLI
        flags it can call ``parser = make_parser()``, add custom
        arguments, run ``args = parser.parse_args()``, and then pass
        *args* here.  If *None*, a default parser with only
        ``--csv`` / ``--csv-samples`` is created and ``parse_args()``
        is called automatically.
    """
    if args is None:
        args = make_parser().parse_args()

    configure_rotating(getattr(args, "rotating", None), getattr(args, "no_rotating", False))

    if args.kernel_profile:
        from torch.profiler import profile, ProfilerActivity

    rows = []
    all_case_metrics = []
    all_kernel_rows = []
    resolved_metric_columns = None

    for case in test_cases:
        label = "  ".join(f"{k}={case[k]}" for k in param_columns)
        print(f"\n{'='*60}")
        print(f"Testing: {label}")
        print(f"{'='*60}")

        metric_records = bench_fn(**case)
        metric_row = _metric_row_from_records(metric_records)
        _print_metric_records(metric_records)
        current_metric_columns = list(metric_row.keys())

        if resolved_metric_columns is None:
            resolved_metric_columns = current_metric_columns
        elif current_metric_columns != resolved_metric_columns:
            raise ValueError(
                f"Inconsistent metric columns for case {case}: "
                f"expected {resolved_metric_columns}, got {current_metric_columns}"
            )

        case_params = {k: (str(case[k]) if isinstance(case[k], torch.dtype) else case[k])
                       for k in param_columns}
        row = dict(case_params)
        row.update(metric_row)
        rows.append(row)
        all_case_metrics.append((case_params, metric_records))

        if args.kernel_profile:
            with profile(
                activities=[ProfilerActivity.CUDA],
            ) as prof:
                bench_fn(**case)
                torch.cuda.synchronize()

            averages = prof.key_averages()
            gpu_events = [e for e in averages if e.self_device_time_total > 0]
            gpu_events.sort(key=lambda e: e.self_device_time_total, reverse=True)

            if gpu_events:
                total_cuda_us = sum(e.self_device_time_total for e in gpu_events)
                w = _KERNEL_NAME_MAX_WIDTH
                print(
                    f"\n  | {'Kernel':<{w}} | "
                    f"{'Total (us)':>11} | {'Calls':>6} | "
                    f"{'Avg (us)':>10} | {'%':>6} |"
                )
                print(
                    f"  | {'-'*w} | "
                    f"{'-'*11} | {'-'*6} | "
                    f"{'-'*10} | {'-'*6} |"
                )
                for e in gpu_events:
                    avg_us = e.self_device_time_total / e.count if e.count > 0 else 0
                    pct = (
                        100.0 * e.self_device_time_total / total_cuda_us
                        if total_cuda_us > 0
                        else 0
                    )
                    short = _shorten_kernel_name(e.key)
                    print(
                        f"  | {short:<{w}} | {e.self_device_time_total:>11.1f} | "
                        f"{e.count:>6} | {avg_us:>10.2f} | {pct:>5.1f}% |"
                    )
                print(
                    f"  | {'TOTAL':<{w}} | {total_cuda_us:>11.1f} | "
                    f"{'---':>6} | {'---':>10} | {'---':>6} |"
                )

            for e in gpu_events:
                kr = dict(case_params)
                kr["kernel_name"] = e.key
                kr["cuda_time_total_us"] = round(e.self_device_time_total, 1)
                kr["num_calls"] = e.count
                kr["cuda_time_avg_us"] = (
                    round(e.self_device_time_total / e.count, 2) if e.count > 0 else 0
                )
                all_kernel_rows.append(kr)

    if args.csv is not None:
        import pandas as pd
        out_csv = args.csv if isinstance(args.csv, str) else (
            default_csv or _default_csv_name(bench_fn)
        )
        columns = param_columns + (resolved_metric_columns or [])
        results = pd.DataFrame(rows, columns=columns)
        results.to_csv(out_csv, index=False)
        print(f"\nResults saved to {out_csv}")

    if args.kernel_profile and args.csv is not None and all_kernel_rows:
        import pandas as pd
        from pathlib import Path
        base = default_csv or _default_csv_name(bench_fn)
        out_csv_name = args.csv if isinstance(args.csv, str) else (
            Path(base).stem + "_kernel_profile.csv"
        )
        # Don't overwrite the main CSV if --csv was given a filename
        if isinstance(args.csv, str):
            out_csv_name = Path(args.csv).stem + "_kernel_profile.csv"
        kernel_columns = param_columns + [
            "kernel_name", "cuda_time_total_us", "num_calls", "cuda_time_avg_us",
        ]
        df = pd.DataFrame(all_kernel_rows, columns=kernel_columns)
        df.to_csv(out_csv_name, index=False)
        print(f"Kernel profile saved to {out_csv_name}")

    if args.csv_samples is not None:
        import pandas as pd
        from pathlib import Path
        base = default_csv or _default_csv_name(bench_fn)
        samples_csv = args.csv_samples if isinstance(args.csv_samples, str) else (
            Path(base).stem + "_samples.csv"
        )
        sample_rows = []
        for case_params, records in all_case_metrics:
            for metric in records:
                measurement = metric.get("measurement")
                if measurement is None:
                    continue
                lbl = metric["label"]
                for i, t in enumerate(measurement.times):
                    sr = dict(case_params)
                    sr["label"] = lbl
                    sr["sample_idx"] = i
                    # measurement.times is already per-iteration (raw block time
                    # divided by number_per_run); convert seconds -> ms only.
                    sr["time_ms"] = t * 1e3
                    sample_rows.append(sr)
        if sample_rows:
            df = pd.DataFrame(
                sample_rows,
                columns=param_columns + ["label", "sample_idx", "time_ms"],
            )
            df.to_csv(samples_csv, index=False)
            print(f"Samples saved to {samples_csv}")


# ---------------------------------------------------------------------------
# pytest-based execution support
# ---------------------------------------------------------------------------
# The microbenchmarks can also run under pytest; conftest.py is a thin shim over
# the framework-agnostic helpers below (no pytest import here, so importing
# utils.py never requires pytest). Results are collected per family (test module)
# and written with the same CSV / samples / kernel-profile schema run_benchmarks
# produces, so downstream tooling (e.g. the dashboard ingest) is unaffected.

def configure_rotating(rotating, no_rotating):
    """Set module-level input-rotation state from parsed options."""
    global _ROTATE_BUFFERS, _ROTATE_MB
    if rotating is not None and rotating < 0:
        raise ValueError("--rotating expects a non-negative size in MB")
    _ROTATE_BUFFERS = not no_rotating
    _ROTATE_MB = rotating or 0


def apply_backend_env(monkeypatch, env):
    """Force a kernel backend for one test by setting/unsetting env vars.

    A ``None`` value unsets the var, so forcing one backend cleanly clears the
    toggles that would select a competing one; pytest restores them afterwards.
    """
    for key, value in env.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)


class _FamilyResults:
    """Accumulated rows / samples / kernel rows for one benchmark family."""

    def __init__(self):
        self.param_columns = None
        self.metric_columns = None
        self.rows = []
        self.case_metrics = []
        self.kernel_rows = []


def _stringify_params(case_params):
    return {k: (str(v) if isinstance(v, torch.dtype) else v) for k, v in case_params.items()}


def record_bench(store, family, case_params, metric_records, kernel_rows=None, node_name=""):
    """Record one benchmark case into *store* (a dict keyed by *family*)."""
    fam = store.setdefault(family, _FamilyResults())
    metric_row = _metric_row_from_records(metric_records)
    metric_columns = list(metric_row.keys())
    if fam.param_columns is None:
        fam.param_columns = list(case_params.keys())
        fam.metric_columns = metric_columns
    elif metric_columns != fam.metric_columns:
        raise ValueError(
            f"Inconsistent metric columns for {family}: "
            f"expected {fam.metric_columns}, got {metric_columns}"
        )
    row = _stringify_params(case_params)
    row.update(metric_row)
    fam.rows.append(row)
    fam.case_metrics.append((_stringify_params(case_params), metric_records, node_name))
    if kernel_rows:
        fam.kernel_rows.extend(kernel_rows)


def print_case(case_params, metric_records):
    """Print a case header and its metric lines (reused stdout format)."""
    label = "  ".join(f"{k}={v}" for k, v in case_params.items())
    print(f"\n{'='*60}\nTesting: {label}\n{'='*60}")
    _print_metric_records(metric_records)


def collect_kernel_rows(bench_callable, case_params):
    """Re-run *bench_callable* under torch.profiler and return per-kernel rows."""
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        bench_callable()
        torch.cuda.synchronize()
    events = [e for e in prof.key_averages() if e.self_device_time_total > 0]
    events.sort(key=lambda e: e.self_device_time_total, reverse=True)
    params = _stringify_params(case_params)
    rows = []
    for e in events:
        kr = dict(params)
        kr["kernel_name"] = e.key
        kr["cuda_time_total_us"] = round(e.self_device_time_total, 1)
        kr["num_calls"] = e.count
        kr["cuda_time_avg_us"] = round(e.self_device_time_total / e.count, 2) if e.count else 0
        rows.append(kr)
    return rows


def write_bench_outputs(store, *, csv=None, csv_samples=None, kernel_profile=False):
    """Write per-family CSV / samples / kernel-profile outputs; return paths written."""
    import pandas as pd
    from pathlib import Path

    # When an explicit filename is given but several families run in one session
    # (e.g. `pytest .`), insert the family name so they don't overwrite each other.
    multi = sum(1 for fam in store.values() if fam.rows) > 1

    def _dest(explicit, family, default_name):
        if not isinstance(explicit, str):
            return default_name
        if not multi:
            return explicit
        p = Path(explicit)
        return str(p.with_name(f"{p.stem}-{family}{p.suffix}"))

    written = []
    for family, fam in store.items():
        if not fam.rows:
            continue
        if csv is not None:
            out = _dest(csv, family, f"{family}.csv")
            pd.DataFrame(fam.rows, columns=fam.param_columns + fam.metric_columns).to_csv(
                out, index=False
            )
            written.append(out)
            if kernel_profile and fam.kernel_rows:
                kout = f"{Path(out).stem}_kernel_profile.csv"
                cols = fam.param_columns + [
                    "kernel_name", "cuda_time_total_us", "num_calls", "cuda_time_avg_us",
                ]
                pd.DataFrame(fam.kernel_rows, columns=cols).to_csv(kout, index=False)
                written.append(kout)
        if csv_samples is not None:
            sout = _dest(csv_samples, family, f"{family}_samples.csv")
            sample_rows = []
            for case_params, records, _node in fam.case_metrics:
                for metric in records:
                    m = metric.get("measurement")
                    if m is None:
                        continue
                    for i, t in enumerate(m.times):
                        sr = dict(case_params)
                        sr["label"] = metric["label"]
                        sr["sample_idx"] = i
                        sr["time_ms"] = t * 1e3
                        sample_rows.append(sr)
            if sample_rows:
                pd.DataFrame(
                    sample_rows,
                    columns=fam.param_columns + ["label", "sample_idx", "time_ms"],
                ).to_csv(sout, index=False)
                written.append(sout)
    return written


def _times_ms(measurement):
    if measurement is None:
        return []
    return [float(t) * 1e3 for t in getattr(measurement, "times", [])]


def _result_rows(store):
    """Flatten *store* into (name, stats_ms, throughput, unit) rows for the summary."""
    import numpy as np

    rows = []
    for fam in store.values():
        for _case_params, records, node_name in fam.case_metrics:
            base = node_name
            if base.endswith("]") and "[" in base:
                base = base[base.index("[") + 1 : -1]  # keep the parametrize id
            visible = [m for m in records if not m.get("samples_only")]
            for m in visible:
                name = base if len(visible) == 1 else f"{base} {m['label']}"
                times = _times_ms(m.get("measurement"))
                if times:
                    a = np.asarray(times)
                    stats = {
                        "min": float(a.min()), "median": float(np.median(a)),
                        "max": float(a.max()), "std": float(a.std()),
                    }
                else:  # no per-sample distribution: show the single value as the median
                    stats = {"min": None, "median": m["ms"], "max": None, "std": None}
                rows.append((name, stats, m["throughput"], m["unit"]))
    return rows


def format_results_table(store):
    """Render the results as a Markdown table (times in ms) with a caption line."""
    rows = _result_rows(store)
    if not rows:
        return ""

    def cell(v):
        return "-" if v is None else f"{v:.4f}"

    headers = ["Name", "Min (ms)", "Median (ms)", "Max (ms)", "StdDev (ms)", "Throughput"]
    body = []
    for name, s, thr, unit in sorted(rows, key=lambda r: r[0]):
        body.append([
            name, cell(s["min"]), cell(s["median"]),
            cell(s["max"]), cell(s["std"]), f"{thr:.2f} {unit}",
        ])
    widths = [max(len(headers[i]), *(len(r[i]) for r in body)) for i in range(len(headers))]

    def row(cells):
        padded = [
            c.ljust(widths[i]) if i == 0 else c.rjust(widths[i]) for i, c in enumerate(cells)
        ]
        return "| " + " | ".join(padded) + " |"

    # Markdown delimiter row: left-align Name, right-align the numeric columns.
    align = [
        ":" + "-" * (widths[i] - 1) if i == 0 else "-" * (widths[i] - 1) + ":"
        for i in range(len(headers))
    ]
    caption = f"benchmark: {len(body)} tests"
    return "\n".join(
        [caption, "", row(headers), "| " + " | ".join(align) + " |", *(row(r) for r in body)]
    )
