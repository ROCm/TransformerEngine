#!/usr/bin/env python
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Shared data layer for microbenchmark visualization.

Loads the two CSV shapes the microbenchmark suite produces and normalizes
both into a single tidy ("long") DataFrame so the plotting code does not need
to know which file it came from.

Schemas
-------
* **samples** (from ``--csv-samples``): one row per timing sample.
  ``<params...>, label, sample_idx, time_ms[, throughput, unit]``.
* **aggregate** (from ``--csv``): one row per case with paired metric columns
  ``<label> Time (ms)`` and ``<label> <unit>`` (e.g. ``GEMM Forward TFLOPS``).

Normalized long schema (returned by every loader)
-------------------------------------------------
``<params...>, bench, pass, time_ms, throughput, unit, dtype_short``
plus ``sample_idx`` for the samples source. ``source`` is attached as a
``DataFrame.attrs`` entry (``"samples"`` or ``"aggregate"``) and ``params``
lists the detected parameter columns.
"""

import pandas as pd

# Per-sample / metric bookkeeping columns that are never benchmark parameters.
_SAMPLE_META_COLS = {"label", "sample_idx", "time_ms", "throughput", "unit"}

# Recognized throughput units in aggregate-format metric column names.
_KNOWN_UNITS = ("TFLOPS", "GB/s")

# Pass names ordered longest-first so "Forward+Backward" matches before
# "Forward" when stripping the suffix from a metric label.
_PASS_NAMES = ("Forward+Backward", "Forward", "Backward")

_TIME_SUFFIX = " Time (ms)"

_DTYPE_SHORT = {
    "torch.bfloat16": "bf16",
    "torch.float16": "fp16",
    "torch.float32": "fp32",
    "torch.float8_e4m3fn": "fp8e4m3",
    "torch.float8_e5m2": "fp8e5m2",
}


def shorten_dtype(value):
    """Map a verbose dtype string (``torch.bfloat16``) to a short tag (``bf16``)."""
    text = str(value)
    if text in _DTYPE_SHORT:
        return _DTYPE_SHORT[text]
    return text.replace("torch.", "")


def split_label(label):
    """Split a metric label into ``(bench, pass)``.

    ``"GEMM Forward"`` -> ``("GEMM", "Forward")``. If no known pass suffix is
    present the whole label is treated as the benchmark name with an empty pass.
    """
    text = str(label).strip()
    for pass_name in _PASS_NAMES:
        if text == pass_name:
            return "", pass_name
        if text.endswith(" " + pass_name):
            return text[: -len(pass_name)].strip(), pass_name
    return text, ""


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def detect_format(df):
    """Return ``"samples"`` or ``"aggregate"`` for a loaded DataFrame."""
    cols = set(df.columns)
    if {"sample_idx", "time_ms"} <= cols:
        return "samples"
    if any(c.endswith(_TIME_SUFFIX) for c in df.columns):
        return "aggregate"
    raise ValueError(
        "Unrecognized benchmark CSV: expected per-sample columns "
        "(sample_idx, time_ms) or aggregate '<label> Time (ms)' columns. "
        f"Got columns: {list(df.columns)}"
    )


def _samples_param_columns(df):
    return [c for c in df.columns if c not in _SAMPLE_META_COLS]


def _aggregate_metric_columns(df):
    """Return ``(param_cols, metrics)`` for an aggregate-format frame.

    ``metrics`` is a list of ``(label, time_col, throughput_col, unit)`` tuples.
    """
    time_cols = [c for c in df.columns if c.endswith(_TIME_SUFFIX)]
    metrics = []
    metric_cols = set()
    for time_col in time_cols:
        label = time_col[: -len(_TIME_SUFFIX)]
        throughput_col, unit = None, None
        for candidate_unit in _KNOWN_UNITS:
            candidate = f"{label} {candidate_unit}"
            if candidate in df.columns:
                throughput_col, unit = candidate, candidate_unit
                break
        metrics.append((label, time_col, throughput_col, unit))
        metric_cols.add(time_col)
        if throughput_col is not None:
            metric_cols.add(throughput_col)
    param_cols = [c for c in df.columns if c not in metric_cols]
    return param_cols, metrics


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _finalize(df, params, source):
    """Attach derived columns and metadata shared by both loaders."""
    if "dtype" in df.columns:
        df["dtype_short"] = df["dtype"].map(shorten_dtype)
    if "throughput" not in df.columns:
        df["throughput"] = pd.NA
    if "unit" not in df.columns:
        df["unit"] = pd.NA
    df.attrs["source"] = source
    df.attrs["params"] = list(params)
    return df


def load_samples(path):
    """Load a samples-format CSV into the normalized long schema."""
    df = pd.read_csv(path)
    params = _samples_param_columns(df)

    split = df["label"].map(split_label)
    df["bench"] = [b for b, _ in split]
    df["pass"] = [p for _, p in split]

    keep = params + ["bench", "pass", "sample_idx", "time_ms"]
    if "throughput" in df.columns:
        keep.append("throughput")
    if "unit" in df.columns:
        keep.append("unit")
    df = df[keep].copy()
    return _finalize(df, params, "samples")


def load_aggregate(path):
    """Load an aggregate-format CSV into the normalized long schema."""
    df = pd.read_csv(path)
    param_cols, metrics = _aggregate_metric_columns(df)

    rows = []
    for _, row in df.iterrows():
        base = {c: row[c] for c in param_cols}
        for label, time_col, throughput_col, unit in metrics:
            bench, pass_name = split_label(label)
            rec = dict(base)
            rec["bench"] = bench
            rec["pass"] = pass_name
            rec["time_ms"] = pd.to_numeric(row[time_col], errors="coerce")
            rec["throughput"] = (
                pd.to_numeric(row[throughput_col], errors="coerce")
                if throughput_col is not None else pd.NA
            )
            rec["unit"] = unit
            rows.append(rec)
    out = pd.DataFrame(rows)
    return _finalize(out, param_cols, "aggregate")


def load_any(path):
    """Dispatch to :func:`load_samples` or :func:`load_aggregate` by schema."""
    head = pd.read_csv(path, nrows=1)
    if detect_format(head) == "samples":
        return load_samples(path)
    return load_aggregate(path)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def group_columns(df, value="time_ms"):
    """Columns that uniquely identify a benchmark group (params + bench/pass).

    ``sample_idx`` and the numeric value/throughput/unit columns are excluded.
    """
    exclude = {"sample_idx", "time_ms", "throughput", "unit", value}
    return [c for c in df.columns if c not in exclude]


def trim_percentile(df, value="time_ms", upper=0.95, lower=0.0, group_cols=None):
    """Drop per-group outliers outside ``[lower, upper]`` quantiles of *value*.

    Mirrors the percentile-trim control in the JAX attention dashboard: timing
    tails (warmup, scheduler jitter) are removed per group so the visible
    distribution reflects steady-state behavior. A no-op for aggregate frames
    (no ``sample_idx``) or when ``lower<=0`` and ``upper>=1``.
    """
    if "sample_idx" not in df.columns:
        return df
    if lower <= 0.0 and upper >= 1.0:
        return df
    if group_cols is None:
        group_cols = group_columns(df, value)
    if not group_cols:
        lo = df[value].quantile(lower)
        hi = df[value].quantile(upper)
        return df[(df[value] >= lo) & (df[value] <= hi)]

    grouped = df.groupby(group_cols, dropna=False)[value]
    lo = grouped.transform(lambda s: s.quantile(lower))
    hi = grouped.transform(lambda s: s.quantile(upper))
    return df[(df[value] >= lo) & (df[value] <= hi)].copy()


def aggregate_stats(df, value="time_ms", group_cols=None):
    """Per-group summary statistics for *value*.

    Returns a frame with ``median, q25, q75, vmin, vmax, count`` plus the
    grouping columns. For aggregate-source frames (one row per group) the
    median equals the value and the band collapses to it.
    """
    if group_cols is None:
        group_cols = group_columns(df, value)
    work = df.dropna(subset=[value])
    if not group_cols:
        s = work[value]
        return pd.DataFrame([{
            "median": s.median(), "q25": s.quantile(0.25), "q75": s.quantile(0.75),
            "vmin": s.min(), "vmax": s.max(), "count": s.count(),
        }])
    out = work.groupby(group_cols, dropna=False)[value].agg(
        median="median",
        q25=lambda s: s.quantile(0.25),
        q75=lambda s: s.quantile(0.75),
        vmin="min",
        vmax="max",
        count="count",
    ).reset_index()
    return out


def default_value_column(df, pass_name=None):
    """Pick ``throughput`` when it has data, else ``time_ms``.

    If *pass_name* is given, only that pass is considered (the
    ``Forward+Backward`` samples-only record carries no throughput).
    """
    sub = df if pass_name is None else df[df["pass"] == pass_name]
    if "throughput" in sub.columns and sub["throughput"].notna().any():
        return "throughput"
    return "time_ms"


def value_label(df, value):
    """Human-readable axis label for a value column."""
    if value == "time_ms":
        return "Time (ms)"
    units = [u for u in df.get("unit", pd.Series(dtype=object)).dropna().unique()]
    if len(units) == 1:
        return f"Throughput ({units[0]})"
    return "Throughput"
