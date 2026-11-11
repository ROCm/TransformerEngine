#!/usr/bin/env python
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Generate interactive Plotly visualizations from microbenchmark CSVs.

Consumes the CSVs produced by the benchmarks in this directory (both the
per-sample ``--csv-samples`` output and the aggregate ``--csv`` output) and
writes self-contained, interactive HTML. No server is required to view the
result, which suits remote / SSH workflows and PR attachments.

Plot kinds
----------
* ``distribution`` box + all raw sample points per group, with percentile
  trimming. The honest distribution view for the suite's small (~12) sample
  counts, where a violin/KDE would over-smooth.
* ``scaling`` median throughput (or time) vs token count ``M`` per case, with
  a shaded min--max band.
* ``bars`` grouped median-throughput bars per case with IQR error bars.
* ``comparison`` baseline-vs-candidate speedup bars, one per benchmark group
  (visual complement to ``compare_results.py``).
* ``report`` (default) all applicable kinds in one HTML file.

Examples
--------
    python visualize.py benchmark_gemm_samples.csv
    python visualize.py benchmark_gemm_samples.csv --kind scaling --value throughput
    python visualize.py bench_gemm_candidate.csv --baseline benchmark_gemm_samples.csv
"""

import argparse
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import viz_data as vd

_PALETTE = px.colors.qualitative.Plotly
_FACET_WRAP = 4


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _rgba(color, alpha):
    """Convert a Plotly ``#RRGGBB`` color to an ``rgba(...)`` string."""
    color = color.lstrip("#")
    r, g, b = (int(color[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _color_map(values):
    ordered = sorted((str(v) for v in values), key=str)
    return {v: _PALETTE[i % len(_PALETTE)] for i, v in enumerate(ordered)}


def _usable_facet(df, name, *, exclude):
    """Return *name* if it is a sensible facet column, else ``None``."""
    if not name or name in exclude or name not in df.columns:
        return None
    return name if df[name].nunique(dropna=True) > 1 else None


def _key_series(df, cols):
    """Join a few identifying columns into one ``a / b / c`` label per row."""
    present = [c for c in cols if c in df.columns]
    return df[present].astype(str).agg(" / ".join, axis=1)


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------

def fig_distribution(df, x="M", value="time_ms", color="pass", facet=None,
                     trim_upper=0.95, trim_lower=0.0, title=None):
    """Box + overlaid raw points per group (the JAX-dashboard-style view)."""
    work = df.dropna(subset=[value]).copy()
    work = vd.trim_percentile(work, value=value, upper=trim_upper, lower=trim_lower)
    facet = _usable_facet(work, facet, exclude={x, color, value})
    work[x] = work[x].astype(str)

    n_facets = work[facet].nunique() if facet else 1
    height = 480 if not facet else 300 * ((n_facets + _FACET_WRAP - 1) // _FACET_WRAP)

    fig = px.box(
        work, x=x, y=value, color=color, points="all",
        facet_col=facet, facet_col_wrap=_FACET_WRAP if facet else None,
        title=title or "Per-sample distribution",
        labels={value: vd.value_label(work, value)},
        category_orders={x: sorted(work[x].unique(), key=lambda v: float(v)
                                   if str(v).replace('.', '', 1).isdigit() else v)},
    )
    fig.update_traces(boxmean=True, jitter=0.4, marker=dict(size=4, opacity=0.6))
    fig.update_layout(height=height, boxmode="group", legend_title_text=color)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    return fig


def fig_scaling(df, x="M", value="throughput", color="Case", facet="pass",
                band="minmax", title=None):
    """Median *value* vs *x* per series, with a shaded spread band."""
    work = df.dropna(subset=[value]).copy()
    facet = _usable_facet(work, facet, exclude={x, color, value})
    facets = (sorted(work[facet].dropna().unique(), key=str) if facet else [None])
    cmap = _color_map(work[color].unique())
    upper_col, lower_col = ("vmax", "vmin") if band == "minmax" else ("q75", "q25")

    fig = make_subplots(
        rows=1, cols=len(facets), shared_yaxes=True,
        subplot_titles=[f"{facet}={f}" for f in facets] if facet else None,
        horizontal_spacing=0.04,
    )

    for col_idx, fval in enumerate(facets, start=1):
        sub = work if fval is None else work[work[facet] == fval]
        for cval in sorted(sub[color].dropna().unique(), key=str):
            series = sub[sub[color] == cval]
            stats = vd.aggregate_stats(series, value=value, group_cols=[x]).sort_values(x)
            if stats.empty:
                continue
            xs = stats[x].tolist()
            line_color = cmap[str(cval)]
            if band and len(xs) > 1:
                up, lo = stats[upper_col].tolist(), stats[lower_col].tolist()
                fig.add_trace(go.Scatter(
                    x=xs + xs[::-1], y=up + lo[::-1], fill="toself",
                    fillcolor=_rgba(line_color, 0.15), line=dict(width=0),
                    hoverinfo="skip", legendgroup=str(cval), showlegend=False,
                ), row=1, col=col_idx)
            fig.add_trace(go.Scatter(
                x=xs, y=stats["median"].tolist(), mode="lines+markers",
                name=str(cval), legendgroup=str(cval),
                line=dict(color=line_color), marker=dict(size=6),
                showlegend=(col_idx == 1),
                hovertemplate=f"{color}={cval}<br>{x}=%{{x}}<br>median=%{{y:.3g}}<extra></extra>",
            ), row=1, col=col_idx)
        fig.update_xaxes(title_text=x, row=1, col=col_idx)

    fig.update_yaxes(title_text=vd.value_label(work, value), row=1, col=1)
    band_desc = "min-max" if band == "minmax" else "IQR"
    fig.update_layout(
        title=title or f"Scaling vs {x} (line = median, band = {band_desc})",
        height=480, legend_title_text=color,
    )
    return fig


def fig_throughput_bars(df, x="Case", value="throughput", color="pass",
                        facet="M", title=None):
    """Grouped median bars per *x* with IQR error bars."""
    work = df.dropna(subset=[value]).copy()
    facet = _usable_facet(work, facet, exclude={x, color, value})
    group_cols = [c for c in dict.fromkeys([x, color, facet]) if c]
    stats = vd.aggregate_stats(work, value=value, group_cols=group_cols)
    stats["err_plus"] = stats["q75"] - stats["median"]
    stats["err_minus"] = stats["median"] - stats["q25"]

    horizontal = work[x].nunique() > 12
    order = list(dict.fromkeys(work[x].astype(str)))
    common = dict(
        color=color, barmode="group",
        facet_col=facet, facet_col_wrap=_FACET_WRAP if facet else None,
        title=title or f"Median {vd.value_label(work, value).lower()} by {x}",
    )
    if horizontal:
        stats[x] = stats[x].astype(str)
        fig = px.bar(
            stats, x="median", y=x, orientation="h",
            error_x="err_plus", error_x_minus="err_minus",
            category_orders={x: order[::-1]},
            labels={"median": vd.value_label(work, value)}, **common,
        )
        fig.update_layout(height=max(420, 26 * len(order)))
    else:
        stats[x] = stats[x].astype(str)
        fig = px.bar(
            stats, x=x, y="median",
            error_y="err_plus", error_y_minus="err_minus",
            category_orders={x: order},
            labels={"median": vd.value_label(work, value)}, **common,
        )
        fig.update_layout(height=480)
    fig.update_layout(legend_title_text=color)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    return fig


def fig_comparison(base_df, cand_df, value="throughput",
                   base_name="baseline", cand_name="candidate", title=None):
    """Baseline-vs-candidate speedup bars (one bar per benchmark group)."""
    keys = vd.group_columns(base_df, value)
    keys = [c for c in keys if c in cand_df.columns]

    bm = vd.aggregate_stats(base_df.dropna(subset=[value]), value, keys)
    cm = vd.aggregate_stats(cand_df.dropna(subset=[value]), value, keys)
    merged = bm[keys + ["median"]].merge(
        cm[keys + ["median"]], on=keys, suffixes=("_base", "_cand"),
    )
    merged = merged[(merged["median_base"] > 0) & (merged["median_cand"] > 0)]
    if merged.empty:
        raise ValueError("No overlapping benchmark groups between the two CSVs.")

    higher_better = value != "time_ms"
    merged["speedup"] = (merged["median_cand"] / merged["median_base"]
                         if higher_better else merged["median_base"] / merged["median_cand"])
    merged["key"] = _key_series(merged, ["Case", "M", "pass", "dtype_short"])
    merged = merged.sort_values("speedup")
    color_key = "pass" if "pass" in merged.columns else None
    cmap = _color_map(merged[color_key].unique()) if color_key else None

    fig = go.Figure()
    # Speedup bars (horizontal; one bar per group).
    if color_key:
        for cval in sorted(merged[color_key].dropna().unique(), key=str):
            s = merged[merged[color_key] == cval]
            fig.add_trace(go.Bar(
                x=s["speedup"], y=s["key"], orientation="h", name=str(cval),
                legendgroup=str(cval), marker_color=cmap[str(cval)],
                hovertemplate="%{y}<br>speedup=%{x:.3f}x<extra></extra>",
            ))
    else:
        fig.add_trace(go.Bar(
            x=merged["speedup"], y=merged["key"], orientation="h",
            showlegend=False,
            hovertemplate="%{y}<br>speedup=%{x:.3f}x<extra></extra>",
        ))
    fig.add_vline(x=1.0, line=dict(color="black", dash="dash"))
    fig.update_xaxes(title_text="speedup (x)")

    median_sp = float(merged["speedup"].median())
    fig.update_layout(
        title=title or (
            f"Speedup: {cand_name} vs {base_name} "
            f"(>1 = {cand_name} faster) — median {median_sp:.3f}x"
        ),
        height=max(480, 22 * len(merged)), legend_title_text=color_key or "",
    )
    return fig


# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------

_REPORT_CSS = """
body { font-family: system-ui, -apple-system, Arial, sans-serif; margin: 24px;
       color: #1a1a1a; }
h1 { margin-bottom: 4px; }
nav { margin: 12px 0 28px; padding-bottom: 8px; border-bottom: 1px solid #ddd; }
nav a { margin-right: 16px; text-decoration: none; color: #1565c0; }
section { margin-bottom: 48px; }
section h2 { border-left: 4px solid #1565c0; padding-left: 8px; }
.meta { color: #666; font-size: 0.9em; }
"""


def _slug(text):
    return "".join(c if c.isalnum() else "-" for c in text.lower()).strip("-")


def write_report(sections, path, title, subtitle=None, include_plotlyjs="inline"):
    """Write ``[(heading, fig), ...]`` to a single self-contained HTML file."""
    nav = " ".join(
        f'<a href="#{_slug(h)}">{h}</a>' for h, _ in sections
    )
    blocks = []
    for i, (heading, fig) in enumerate(sections):
        include = include_plotlyjs if i == 0 else False
        div = fig.to_html(full_html=False, include_plotlyjs=include,
                          default_width="100%")
        blocks.append(f'<section id="{_slug(heading)}"><h2>{heading}</h2>{div}</section>')
    sub = f'<p class="meta">{subtitle}</p>' if subtitle else ""
    html = (
        f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{title}</title><style>{_REPORT_CSS}</style></head>"
        f"<body><h1>{title}</h1>{sub}<nav>{nav}</nav>{''.join(blocks)}</body></html>"
    )
    Path(path).write_text(html, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Defaults + CLI
# ---------------------------------------------------------------------------

def _pick_x(df):
    return "M" if "M" in df.columns else (df.attrs["params"][0] if df.attrs["params"] else "Case")


def _resolve(df, kind, args):
    """Resolve x/color/facet/value defaults for *kind*, honoring CLI overrides."""
    x = args.x or ("Case" if kind == "bars" else _pick_x(df))
    if kind == "distribution":
        color = args.color or "pass"
        facet = args.facet if args.facet is not None else "Case"
        value = args.value if args.value != "auto" else "time_ms"
    elif kind == "scaling":
        color = args.color or "Case"
        facet = args.facet if args.facet is not None else "pass"
        value = args.value if args.value != "auto" else vd.default_value_column(df, "Forward")
    else:  # bars
        color = args.color or "pass"
        facet = args.facet if args.facet is not None else "M"
        value = args.value if args.value != "auto" else vd.default_value_column(df, "Forward")
    facet = None if facet in ("", "none") else facet
    # Drop a color that carries no information (e.g. the empty 'pass' of the
    # single-metric casting benchmark) unless the user asked for it explicitly.
    if (kind in ("distribution", "bars") and not args.color
            and color in df.columns and df[color].nunique(dropna=True) <= 1):
        color = None
    return x, color, facet, value


def _collapsed_columns(df, shown):
    """Param columns that vary *within* a (shown-axis) group and are silently
    pooled together. e.g. grouped GEMM medians span expert counts ``B`` when
    ``B`` is on no axis. Columns fully determined by a shown column (a model's
    fixed ``hidden_size`` under ``Case``) are not flagged.
    """
    shown = [c for c in shown if c and c in df.columns]
    if not shown:
        return []
    candidates = [c for c in (list(df.attrs.get("params", [])) + ["pass"])
                  if c in df.columns and c not in shown]
    collapsed = []
    for col in candidates:
        if df[col].nunique(dropna=True) <= 1:
            continue
        if df.groupby(shown, dropna=False)[col].nunique(dropna=False).max() > 1:
            collapsed.append(col)
    return collapsed


def _build_kind(df, kind, args):
    if args.pass_filter:
        df = df[df["pass"] == args.pass_filter]
    x, color, facet, value = _resolve(df, kind, args)
    collapsed = _collapsed_columns(df, [x, color, facet])
    if collapsed:
        print(f"  [note] {kind}: each group pools multiple {', '.join(collapsed)} "
              f"value(s); add --facet/--color/--pass to separate them.")
    if kind == "distribution":
        return fig_distribution(df, x=x, value=value, color=color, facet=facet,
                                trim_upper=args.trim_upper, trim_lower=args.trim_lower)
    if kind == "scaling":
        return fig_scaling(df, x=x, value=value, color=color, facet=facet)
    return fig_throughput_bars(df, x=x, value=value, color=color, facet=facet)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("csv", help="Benchmark CSV (samples or aggregate format)")
    parser.add_argument("--baseline", metavar="FILE",
                        help="Baseline CSV; switches to / adds the comparison view")
    parser.add_argument("--kind", default="report",
                        choices=["distribution", "scaling", "bars", "comparison", "report"],
                        help="Visualization to produce (default: report = all)")
    parser.add_argument("--value", default="auto", choices=["auto", "time_ms", "throughput"],
                        help="Metric to plot (default: auto)")
    parser.add_argument("--x", help="Override the x-axis column")
    parser.add_argument("--color", help="Override the color/series column")
    parser.add_argument("--facet", help="Override the facet column ('none' to disable)")
    parser.add_argument("--pass", dest="pass_filter", metavar="PASS",
                        help="Restrict to one pass, e.g. 'Forward'")
    parser.add_argument("--trim-upper", type=float, default=0.95,
                        help="Upper percentile for per-group outlier trimming (default 0.95)")
    parser.add_argument("--trim-lower", type=float, default=0.0,
                        help="Lower percentile for per-group outlier trimming (default 0.0)")
    parser.add_argument("--cdn", action="store_true",
                        help="Load plotly.js from CDN instead of inlining it (smaller file)")
    parser.add_argument("-o", "--output", help="Output HTML path")
    args = parser.parse_args()

    df = vd.load_any(args.csv)
    base_df = vd.load_any(args.baseline) if args.baseline else None
    include_js = "cdn" if args.cdn else "inline"
    stem = Path(args.csv).stem

    sections = []
    if args.kind == "comparison" or (args.baseline and args.kind == "report"):
        if base_df is None:
            parser.error("--baseline is required for the comparison view")
        value = args.value if args.value != "auto" else vd.default_value_column(df, "Forward")
        sections.append((
            "Comparison",
            fig_comparison(base_df, df, value=value,
                           base_name=Path(args.baseline).stem, cand_name=stem),
        ))

    if args.kind != "comparison":
        kinds = (["distribution", "scaling", "bars"] if args.kind == "report"
                 else [args.kind])
        for kind in kinds:
            if kind == "distribution" and df.attrs["source"] != "samples":
                continue  # distribution needs per-sample data
            try:
                sections.append((kind.capitalize(), _build_kind(df, kind, args)))
            except Exception as exc:  # keep the report alive if one view fails
                print(f"  [skip] {kind}: {exc}")

    if not sections:
        parser.error("Nothing to plot for the given CSV/kind combination.")

    out = args.output or f"{stem}.html" if args.kind == "report" else (
        args.output or f"{stem}_{args.kind}.html"
    )
    title = f"Microbenchmark: {stem}"
    subtitle = f"source: {Path(args.csv).name}"
    if args.baseline:
        subtitle += f" — baseline: {Path(args.baseline).name}"
    write_report(sections, out, title, subtitle=subtitle, include_plotlyjs=include_js)
    print(f"Wrote {out}  ({len(sections)} view(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
