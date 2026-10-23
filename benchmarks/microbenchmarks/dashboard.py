#!/usr/bin/env python
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Interactive Panel dashboard for microbenchmark CSVs.

A live-exploration companion to ``visualize.py`` (which emits static,
self-contained HTML). It mirrors the JAX fused-attention dashboard
(``benchmarks/attention/panel_app.py``): pick the independent variable, the
hue/series, the metric, per-attribute filters, and a percentile trim, and the
Plotly figure re-renders. Figure builders are shared with ``visualize.py`` so
the static and interactive views never diverge.

Usage
-----
    panel serve dashboard.py --show --args benchmark_gemm_samples.csv

The CSV path may also be given via the ``BENCH_CSV`` environment variable. If
neither is provided, the first ``*_samples.csv`` in the working directory is
used.
"""

import glob
import os
import sys

import panel as pn

import viz_data as vd
import visualize as viz

pn.extension("plotly", design="material", sizing_mode="stretch_width")


def _resolve_csv_path():
    """Find the CSV to load from --args, $BENCH_CSV, or the cwd."""
    for arg in sys.argv[1:]:
        if arg.endswith(".csv"):
            return arg
    if os.environ.get("BENCH_CSV"):
        return os.environ["BENCH_CSV"]
    candidates = sorted(glob.glob("*_samples.csv")) or sorted(glob.glob("*.csv"))
    if not candidates:
        raise SystemExit(
            "No CSV given. Pass one via '--args FILE' or set BENCH_CSV."
        )
    return candidates[0]


CSV_PATH = _resolve_csv_path()
DF = vd.load_any(CSV_PATH)
IS_SAMPLES = DF.attrs["source"] == "samples"
PARAMS = list(DF.attrs["params"])
AXIS_COLS = PARAMS + ["pass"]

KINDS = (["distribution", "scaling", "bars"] if IS_SAMPLES else ["scaling", "bars"])

# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------
kind_w = pn.widgets.Select(name="Plot", options=KINDS, value=KINDS[0])
value_w = pn.widgets.Select(
    name="Metric", options=["auto", "time_ms", "throughput"], value="auto",
)
x_w = pn.widgets.Select(
    name="Independent variable (x)", options=AXIS_COLS,
    value="M" if "M" in AXIS_COLS else AXIS_COLS[0],
)
color_w = pn.widgets.Select(
    name="Hue / series", options=AXIS_COLS,
    value="Case" if "Case" in AXIS_COLS else AXIS_COLS[-1],
)
facet_w = pn.widgets.Select(
    name="Facet", options=["none"] + AXIS_COLS,
    value="pass" if "pass" in AXIS_COLS else "none",
)
trim_w = pn.widgets.FloatSlider(
    name="Percentile trim (upper)", start=0.5, end=1.0, step=0.01, value=0.95,
    disabled=not IS_SAMPLES,
)

# One filter per parameter column (+ pass). Empty selection means "all".
filter_ws = {
    col: pn.widgets.MultiChoice(
        name=col, options=sorted(DF[col].astype(str).unique(), key=str), value=[],
    )
    for col in AXIS_COLS
}


def _resolve_value(kind, value):
    if value != "auto":
        return value
    if kind == "distribution":
        return "time_ms"
    return vd.default_value_column(DF, "Forward")


def _apply_filters(df):
    for col, widget in filter_ws.items():
        if widget.value:
            df = df[df[col].astype(str).isin(widget.value)]
    return df


@pn.depends(
    kind=kind_w, value=value_w, x=x_w, color=color_w, facet=facet_w, trim=trim_w,
    **{f"f_{c}": w for c, w in filter_ws.items()},
)
def make_plot(kind, value, x, color, facet, trim, **_filters):
    df = _apply_filters(DF)
    if df.empty:
        return pn.pane.Alert("No rows match the current filters.", alert_type="warning")

    value = _resolve_value(kind, value)
    facet_arg = None if facet == "none" else facet
    try:
        if kind == "distribution":
            fig = viz.fig_distribution(df, x=x, value=value, color=color,
                                       facet=facet_arg, trim_upper=trim)
        elif kind == "scaling":
            fig = viz.fig_scaling(vd.trim_percentile(df, value=value, upper=trim),
                                  x=x, value=value, color=color, facet=facet_arg)
        else:
            fig = viz.fig_throughput_bars(vd.trim_percentile(df, value=value, upper=trim),
                                          x=x, value=value, color=color, facet=facet_arg)
    except Exception as exc:  # surface builder errors in the UI instead of 500s
        return pn.pane.Alert(f"Could not build plot: {exc}", alert_type="danger")
    return pn.pane.Plotly(fig, sizing_mode="stretch_width", height=640)


template = pn.template.BootstrapTemplate(
    title=f"Microbenchmark Explorer — {os.path.basename(CSV_PATH)}",
    sidebar=[
        pn.pane.Markdown(f"**Source:** `{CSV_PATH}`  \n**Format:** {DF.attrs['source']}"),
        kind_w, value_w, x_w, color_w, facet_w, trim_w,
        pn.pane.Markdown("### Filters (empty = all)"),
        *filter_ws.values(),
    ],
)
template.main.append(pn.Column(make_plot))
template.servable()
