#!/usr/bin/env python
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Shared utilities for microbenchmarks: model configs, timing, throughput, runner."""

import argparse
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

# Minimum number of raw timing samples (blocks) ``time_func`` ensures when a
# caller passes ``min_samples=None``. ``run_benchmarks`` sets this from the
# ``--min-samples`` CLI flag so every benchmark script inherits the knob without
# per-script edits. ``None`` leaves torch's autorange result untouched.
_ACTIVE_MIN_SAMPLES = None

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
# Timing helpers
# ---------------------------------------------------------------------------

class _RawSamples:
    """Minimal ``torch...benchmark.Measurement`` stand-in holding raw block times.

    Exposes the ``times`` (per-run seconds, one entry per recorded timing block),
    ``number_per_run`` and ``mean`` attributes that ``time_func`` callers and the
    samples-CSV writer rely on.
    """

    def __init__(self, times, number_per_run, mean):
        self.times = times
        self.number_per_run = number_per_run
        self.mean = mean


def time_func(fn, method="adaptive", min_run_time=DEFAULT_MIN_RUN_TIME_SECONDS,
              min_samples=None):
    """Time *fn* and return ``(mean_ms, measurement)``.

    The returned measurement exposes per-run sample times via
    ``measurement.times`` -- one entry per recorded timing block (each block is
    an average over ``measurement.number_per_run`` executions, as chosen by
    torch to amortize timer overhead).

    method: "adaptive" uses adaptive_autorange (good for compute-bound),
            "blocked"  uses blocked_autorange  (good for memory-bound).

    min_samples: ensure at least this many raw timing blocks are recorded, so the
        per-sample data is large enough for statistical comparison
        (compare_results.py --stats). torch's autorange usually records only a
        few blocks; any shortfall is topped up with additional equal-sized blocks
        rather than re-running and re-averaging the whole measurement. ``None``
        falls back to the module-level ``_ACTIVE_MIN_SAMPLES`` (set from
        ``--min-samples``); ``None`` there too leaves the autorange result as-is.
    """
    if min_samples is None:
        min_samples = _ACTIVE_MIN_SAMPLES

    timer = benchmark.Timer(stmt="fn()", globals={"fn": fn})
    if method == "blocked":
        m = timer.blocked_autorange(min_run_time=min_run_time)
    else:
        m = timer.adaptive_autorange(min_run_time=min_run_time)

    if min_samples is None or len(m.times) >= min_samples:
        return m.mean * 1e3, m

    # Top up with additional equal-sized blocks (each timeit() records one block
    # averaged over number_per_run runs) until enough raw samples are collected.
    times = list(m.times)  # per-run seconds
    number = m.number_per_run
    while len(times) < min_samples:
        times.append(timer.timeit(number).mean)
    mean_s = sum(times) / len(times)
    return mean_s * 1e3, _RawSamples(times=times, number_per_run=number, mean=mean_s)


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
    """Return an :class:`~argparse.ArgumentParser` with ``--csv`` and ``--csv-samples`` flags.

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
        "--min-samples", type=int, default=12, metavar="N",
        help=(
            "Ensure at least N raw timing samples (blocks) are recorded per "
            "metric for statistical comparison (compare_results.py --stats). "
            "torch's autorange records only a few; any shortfall is topped up "
            "with additional equal-sized blocks. Use a small value (e.g. 2) to "
            "effectively disable top-up. Default: 12."
        ),
    )
    return parser


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

    # Let time_func (called by bench_fns without an explicit min_samples arg)
    # inherit the CLI value without editing every benchmark script.
    global _ACTIVE_MIN_SAMPLES
    _ACTIVE_MIN_SAMPLES = getattr(args, "min_samples", None)

    rows = []
    all_case_metrics = []
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

    if args.csv is not None:
        import pandas as pd
        out_csv = args.csv if isinstance(args.csv, str) else (
            default_csv or _default_csv_name(bench_fn)
        )
        columns = param_columns + (resolved_metric_columns or [])
        results = pd.DataFrame(rows, columns=columns)
        results.to_csv(out_csv, index=False)
        print(f"\nResults saved to {out_csv}")

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
                unit = metric.get("unit")
                thr_mean = metric.get("throughput") or 0.0
                ms_mean = metric.get("ms") or 0.0
                # Throughput is a deterministic function of time for a given
                # config (throughput = C / time), so a per-sample throughput is
                # recovered from the aggregate as thr_mean * ms_mean / sample_ms.
                # samples_only records (e.g. Forward+Backward) carry no
                # throughput and are left blank.
                has_thr = (
                    not metric.get("samples_only") and thr_mean > 0 and ms_mean > 0
                )
                for i, t in enumerate(measurement.times):
                    # measurement.times entries are already per-run (seconds).
                    sample_ms = t * 1e3
                    sr = dict(case_params)
                    sr["label"] = lbl
                    sr["sample_idx"] = i
                    sr["time_ms"] = sample_ms
                    sr["throughput"] = (
                        thr_mean * ms_mean / sample_ms
                        if has_thr and sample_ms > 0
                        else ""
                    )
                    sr["unit"] = unit if has_thr else ""
                    sample_rows.append(sr)
        if sample_rows:
            df = pd.DataFrame(
                sample_rows,
                columns=param_columns
                + ["label", "sample_idx", "time_ms", "throughput", "unit"],
            )
            df.to_csv(samples_csv, index=False)
            print(f"Samples saved to {samples_csv}")
