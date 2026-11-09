#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

"""Plot the work queue design doc's figures from the CSVs collect_ci_data.py writes.

Writes figures/*.png and data/summary.json (the numbers the doc quotes).

Usage: python3 make_figures.py [--data DIR] [--figures DIR]
"""

import argparse
import json
import os
import sys

import matplotlib
import matplotlib.dates

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position
import numpy as np  # pylint: disable=wrong-import-position
import pandas as pd  # pylint: disable=wrong-import-position
from matplotlib.lines import Line2D  # pylint: disable=wrong-import-position
from matplotlib.patches import Patch  # pylint: disable=wrong-import-position

HERE = os.path.dirname(os.path.abspath(__file__))

# The merge of PR #696, and the runs the schedule figures show
MERGED_AT = pd.Timestamp("2026-11-09T06:11:32Z")
DEV_RUN = 42571551993  # first dev push after the merge (no learned weights yet)
WARM_RUN = 42544023501  # last validation run of the PR (learned weights)
SUITES = ["examples", "torch", "jax", "core"]  # the old runner's GPU order

COLORS = {"static": "#9a9a9a", "cold": "#e8961e", "warm": "#1f6fb2", "timeout": "#c0392b"}
SUITE_COLORS = {"torch": "#4c72b0", "jax": "#dd8452", "core": "#55a868", "examples": "#c44e52"}
STEP_COLORS = {
    "Docker pull": "#b0b0b0",
    "Install": "#d9d9d9",
    "Tests": "#1f6fb2",
    "Other": "#f0f0f0",
}

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)


def load(data):
    """The collected tables, with timestamps parsed."""
    jobs = pd.read_csv(f"{data}/jobs.csv")
    for col in ("run_created", "started", "completed"):
        jobs[col] = pd.to_datetime(jobs[col], utc=True)
    items = pd.read_csv(f"{data}/queue_items.csv")
    static = pd.read_csv(f"{data}/static_suites.csv")
    scope = pd.read_csv(f"{data}/pr_scope.csv").fillna("")
    return jobs, items, static, scope


def queue_runs(jobs, items):
    """One row per queue job: makespan, work, utilisation, and the same items laid
    out the old way (each suite end to end on its own GPU)."""
    rows = []
    for job_id, its in items.groupby("job_id"):
        gpus = its.gpus.iloc[0]
        makespan = (its.start + its.secs).max()
        work = its.secs.sum()
        per_suite = its.groupby("label").secs.sum()
        rows.append(
            dict(
                job_id=job_id,
                n_items=len(its),
                gpus=gpus,
                makespan=makespan,
                work=work,
                util=work / (gpus * makespan) if makespan else np.nan,
                lower_bound=max(work / gpus, its.secs.max()),
                weights="warm" if its.est.notna().mean() > 0.5 else "cold",
                static_makespan=per_suite.max(),
                static_util=work / (gpus * per_suite.max()),
            )
        )
    runs = pd.DataFrame(rows).merge(jobs, on="job_id")
    runs["speedup"] = runs.static_makespan / runs.makespan
    # A full queue, not a re-run narrowed to a few items
    runs["full"] = runs.n_items >= 0.5 * runs.groupby(["level", "arch"]).n_items.transform("max")
    return runs


def static_runs(jobs, static):
    """One row per old static-runner job that logged all four suites."""
    cols = [f"{s}_s" for s in SUITES]
    full = static.dropna(subset=cols).merge(jobs, on="job_id")
    full["makespan"] = full[cols].max(axis=1)
    # Runs whose suites all died in setup are not schedules
    full = full[full.makespan > 20 * 60].copy()
    full["work"] = full[cols].sum(axis=1)
    full["util"] = full.work / (4 * full.makespan)
    full["longest"] = full[cols].idxmax(axis=1).str[:-2]
    return full


def queue_schedule(items, jobs, run_id, arch):
    """The item rows of a run's full sGPU queue."""
    job_ids = jobs[(jobs.run_id == run_id) & (jobs.arch == arch) & (jobs.jtype == "sgpu")].job_id
    its = items[items.job_id.isin(job_ids)]
    full_job = its.groupby("job_id").size().idxmax()
    return its[its.job_id == full_job].copy()


def draw_schedule(ax, rows, makespan, title):
    """Gantt bars: rows of (lane, start_s, secs, suite)."""
    for lane, start, secs, suite in rows:
        ax.barh(
            lane,
            secs / 60,
            left=start / 60,
            height=0.72,
            color=SUITE_COLORS[suite],
            edgecolor="white",
            linewidth=0.4,
        )
    ax.axvline(makespan / 60, color="black", linestyle="--", linewidth=1)
    ax.set_title(title, loc="left")
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)


def fig_same_work(items, jobs, out):
    """The dev run's items: laid out one suite per GPU vs as the queue ran them."""
    its = queue_schedule(items, jobs, DEV_RUN, "mi35x").sort_values("start")
    work = its.secs.sum()
    sums = its.groupby("label").secs.sum()

    static_rows = []
    for gpu, suite in enumerate(SUITES):
        offset = 0
        for secs in its[its.label == suite].secs:
            static_rows.append((f"GPU{gpu}\n{suite}", offset, secs, suite))
            offset += secs
    queue_rows = [(f"GPU{g}", r.start, r.secs, r.label) for r in its.itertuples() for g in [r.gpu]]
    static_mk, queue_mk = sums.max(), (its.start + its.secs).max()

    fig, axes = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True)
    draw_schedule(
        axes[0],
        static_rows,
        static_mk,
        f"Before: one suite per GPU -- {static_mk / 60:.0f} min, {100 * work / (4 * static_mk):.0f}% GPU utilisation",
    )
    draw_schedule(
        axes[1],
        sorted(queue_rows),
        queue_mk,
        f"After: one global queue -- {queue_mk / 60:.0f} min, {100 * work / (4 * queue_mk):.0f}% GPU utilisation",
    )
    axes[1].set_xlabel("minutes from the start of the tests")
    handles = [Patch(color=SUITE_COLORS[s], label=s) for s in SUITES]
    axes[0].legend(handles=handles, loc="lower right", ncol=4, frameon=False)
    fig.suptitle(
        f"Same {len(its)} work items, same machine (mi35x, level 3, dev run {DEV_RUN})",
        x=0.01,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    fig.savefig(f"{out}/same_work_schedule.png")
    plt.close(fig)
    return dict(
        items=len(its), static_min=static_mk / 60, queue_min=queue_mk / 60, work_min=work / 60
    )


def fig_timeline(jobs, qruns, out):
    """sGPU test-step wall clock of every level 3 run, Aug -> Nov, per arch."""
    s = jobs[
        (jobs.jtype == "sgpu") & (jobs.level == 3) & jobs.test_s.notna() & (jobs.test_s > 20 * 60)
    ]
    static = s[s.era == "static"]
    q = qruns[(qruns.level == 3) & qruns.full]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    stats = {}
    for ax, arch in zip(axes, ["mi30x", "mi35x"]):
        st = static[static.arch == arch]
        timeout = st.test_s >= 239.5 * 60
        ax.scatter(
            st.started[~timeout], st.test_s[~timeout] / 60, s=10, color=COLORS["static"], alpha=0.6
        )
        ax.scatter(
            st.started[timeout], st.test_s[timeout] / 60, s=22, marker="x", color=COLORS["timeout"]
        )
        qa = q[q.arch == arch]
        for weights in ("cold", "warm"):
            part = qa[(qa.weights == weights) & (qa.branch != "dev")]
            ax.scatter(
                part.started,
                part.test_s / 60,
                s=22,
                color=COLORS[weights],
                edgecolor="white",
                linewidth=0.4,
            )
        dev = qa[qa.branch == "dev"]
        ax.scatter(
            dev.started,
            dev.test_s / 60,
            s=150,
            marker="*",
            color=COLORS["timeout"],
            edgecolor="black",
            linewidth=0.5,
            zorder=5,
        )
        ax.axhline(240, color=COLORS["timeout"], linestyle=":", linewidth=1)
        ax.text(st.started.min(), 243, "240 min step timeout", color=COLORS["timeout"], fontsize=8)
        ax.axvline(MERGED_AT, color="black", linewidth=0.8)
        ax.text(MERGED_AT, 8, " merged", fontsize=8, ha="right", rotation=90, va="bottom")
        warm_median = qa[qa.weights == "warm"].test_s.median() / 60
        ax.axhline(st.test_s.median() / 60, color=COLORS["static"], linestyle="--", linewidth=1)
        ax.axhline(warm_median, color=COLORS["warm"], linestyle="--", linewidth=1)
        ax.set_title(
            f"{arch}: median {st.test_s.median() / 60:.0f} min before, {warm_median:.0f} min with the queue",
            loc="left",
        )
        ax.set_ylim(0, 265)
        ax.set_xlim(st.started.min() - pd.Timedelta(days=3), MERGED_AT + pd.Timedelta(days=4))
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %d"))
        stats[arch] = dict(
            static_n=len(st),
            static_median_min=st.test_s.median() / 60,
            static_timeouts=int(timeout.sum()),
            queue_warm_n=int((qa.weights == "warm").sum()),
            queue_warm_median_min=qa[qa.weights == "warm"].test_s.median() / 60,
            dev_min=dev.test_s.iloc[0] / 60 if len(dev) else None,
        )
    axes[0].set_ylabel("sGPU test step (minutes)")
    handles = [
        Line2D(
            [], [], marker="o", linestyle="", color=COLORS["static"], label="old: one suite per GPU"
        ),
        Line2D(
            [],
            [],
            marker="x",
            linestyle="",
            color=COLORS["timeout"],
            label="old: killed at the step timeout",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color=COLORS["cold"],
            label="queue, no learned weights yet",
        ),
        Line2D(
            [], [], marker="o", linestyle="", color=COLORS["warm"], label="queue, learned weights"
        ),
        Line2D(
            [],
            [],
            marker="*",
            linestyle="",
            markersize=12,
            color=COLORS["timeout"],
            markeredgecolor="black",
            label="first dev run after merge",
        ),
    ]
    fig.legend(
        handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.02)
    )
    fig.suptitle(
        "sGPU test step at level 3, every run since Aug 27 (dashed: medians)",
        x=0.01,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    fig.savefig(f"{out}/sgpu_timeline_l3.png")
    plt.close(fig)
    return stats


def fig_cold_warm(items, jobs, out):
    """The dev run (no weights) next to a run with learned weights: the idle tail."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 4.6))
    stats = {}
    for ax, run_id, label in (
        (axes[0], DEV_RUN, f"No learned weights (dev run {DEV_RUN})"),
        (axes[1], WARM_RUN, f"Learned weights (PR validation run {WARM_RUN})"),
    ):
        its = queue_schedule(items, jobs, run_id, "mi35x")
        makespan = (its.start + its.secs).max()
        work = its.secs.sum()
        lower = max(work / 4, its.secs.max())
        rows = sorted((f"GPU{r.gpu}", r.start, r.secs, r.label) for r in its.itertuples())
        draw_schedule(
            ax,
            rows,
            makespan,
            f"{label}: {100 * work / (4 * makespan):.1f}% busy, {100 * (makespan / lower - 1):.1f}% over the lower bound",
        )
        for lane, end in its.assign(end=its.start + its.secs).groupby("gpu").end.max().items():
            ax.barh(
                f"GPU{lane}",
                (makespan - end) / 60,
                left=end / 60,
                height=0.72,
                color="none",
                edgecolor=COLORS["timeout"],
                hatch="////",
                linewidth=0,
            )
        ax.set_xlim(0, makespan / 60 * 1.02)
        # How close the weights the queue sorted by were to this run's durations
        ratio = (its.secs / its.est).dropna()
        stats[str(run_id)] = dict(
            util=work / (4 * makespan),
            over_lower_bound=makespan / lower - 1,
            makespan_min=makespan / 60,
            items_within_25pct_of_weight=(
                float(((ratio > 0.75) & (ratio < 1.25)).mean()) if len(ratio) else None
            ),
        )
    axes[1].set_xlabel("minutes from the start of the queue (each run on its own scale)")
    handles = [Patch(color=SUITE_COLORS[s], label=s) for s in SUITES] + [
        Patch(facecolor="none", edgecolor=COLORS["timeout"], hatch="////", label="GPU idle")
    ]
    fig.legend(
        handles=handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.0)
    )
    fig.suptitle(
        "Longest-first ordering removes the idle tail (mi35x, level 3, 111 items)",
        x=0.01,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(f"{out}/cold_vs_warm_schedule.png")
    plt.close(fig)
    return stats


def fig_utilisation(sruns, qruns, out):
    """GPU utilisation of every run, and the speedup on identical work."""
    q = qruns[qruns.full]
    groups = [
        ("old\nlevel 1", sruns[sruns.level == 1].util, COLORS["static"]),
        ("old\nlevel 3", sruns[sruns.level == 3].util, COLORS["static"]),
        ("queue\nno weights", q[q.weights == "cold"].util, COLORS["cold"]),
        ("queue\nlearned weights", q[q.weights == "warm"].util, COLORS["warm"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.15]})
    ax = axes[0]
    rng = np.random.default_rng(0)
    for i, (name, values, color) in enumerate(groups):
        ax.boxplot(
            values * 100,
            positions=[i],
            widths=0.5,
            showfliers=False,
            medianprops={"color": "black"},
        )
        ax.scatter(
            i + rng.uniform(-0.18, 0.18, len(values)),
            values * 100,
            s=9,
            color=color,
            alpha=0.7,
            zorder=3,
        )
        ax.text(
            i, 3, f"n={len(values)}\nmedian {100 * values.median():.0f}%", ha="center", fontsize=8
        )
    ax.set_xticks(range(len(groups)), [g[0] for g in groups])
    ax.set_ylim(0, 105)
    ax.set_ylabel("GPU utilisation (%)")
    ax.set_title("Busy share of the 4 GPUs, per run", loc="left")

    ax = axes[1]
    lim = q.static_makespan.max() / 60 * 1.05
    ymax = q.makespan.max() / 60 * 1.25
    speed_lines = {1: ("-", "same speed"), 2: (":", "2x faster"), 3: ("--", "3x faster")}
    for factor, (style, _) in speed_lines.items():
        ax.plot([0, lim], [0, lim / factor], color="black", linewidth=0.7, linestyle=style)
    for weights in ("cold", "warm"):
        part = q[q.weights == weights]
        for arch, marker in (("mi30x", "o"), ("mi35x", "s")):
            p = part[part.arch == arch]
            ax.scatter(
                p.static_makespan / 60,
                p.makespan / 60,
                s=26,
                marker=marker,
                color=COLORS[weights],
                edgecolor="white",
                linewidth=0.4,
            )
    ax.set_xlim(0, lim)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("same items, one suite per GPU (minutes)")
    ax.set_ylabel("queue (minutes)")
    ax.set_title("Each queue run vs its own items laid out the old way", loc="left")
    handles = [
        Line2D([], [], marker="o", linestyle="", color="grey", label="mi30x"),
        Line2D([], [], marker="s", linestyle="", color="grey", label="mi35x"),
        Line2D([], [], marker="o", linestyle="", color=COLORS["cold"], label="no weights"),
        Line2D([], [], marker="o", linestyle="", color=COLORS["warm"], label="learned weights"),
    ] + [
        Line2D([], [], color="black", linewidth=0.7, linestyle=style, label=label)
        for style, label in speed_lines.values()
    ]
    ax.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=False, ncol=4
    )
    fig.tight_layout()
    fig.savefig(f"{out}/utilisation_and_speedup.png")
    plt.close(fig)
    warm = q[(q.weights == "warm") & (q.level == 3)]
    return dict(
        static_util_median={
            f"l{int(l)}": float(sruns[sruns.level == l].util.median()) for l in (1, 3)
        },
        static_n={f"l{int(l)}": int((sruns.level == l).sum()) for l in (1, 3)},
        torch_longest=int((sruns.longest == "torch").sum()),
        jax_longest=int((sruns.longest == "jax").sum()),
        queue_cold_util=[
            float(q[q.weights == "cold"].util.min()),
            float(q[q.weights == "cold"].util.max()),
        ],
        queue_warm_util=[float(warm.util.min()), float(warm.util.max())],
        queue_warm_over_lower_bound_max=float((warm.makespan / warm.lower_bound - 1).max()),
        speedup_warm_l3=dict(
            n=len(warm),
            median=float(warm.speedup.median()),
            min=float(warm.speedup.min()),
            max=float(warm.speedup.max()),
        ),
        static_util_same_work_median=float(warm.static_util.median()),
    )


def rerun_kind(jobs):
    """Label each sGPU job that ran in a later attempt: old re-run, narrowed or re-run all."""
    scope = jobs[jobs.jtype == "scope"][["run_id", "attempt"]]
    rerun = jobs[
        (jobs.jtype == "sgpu") & jobs.era.notna() & (jobs.attempt > 1) & jobs.test_s.notna()
    ].copy()

    def kind(row):
        if row.era == "static":
            return "old"
        reran_scope = ((scope.run_id == row.run_id) & (scope.attempt == row.attempt)).any()
        return "queue-all" if reran_scope else "queue-narrowed"

    rerun["kind"] = rerun.apply(kind, axis=1)
    return rerun


def fig_job_time(jobs, qruns, out):
    """Where a sGPU job's time goes: full runs and "Re-run failed jobs", before and after."""
    s = jobs[
        (jobs.jtype == "sgpu")
        & (jobs.level == 3)
        & jobs.test_s.notna()
        & (jobs.test_s > 20 * 60)
        & (jobs.attempt == 1)
    ]
    warm_jobs = qruns[(qruns.weights == "warm") & qruns.full].job_id
    rerun = rerun_kind(jobs)
    # The narrowed re-runs of the merged implementation (failed items + failed tests)
    narrowed = rerun[
        (rerun.kind == "queue-narrowed") & (rerun.started >= pd.Timestamp("2026-11-01", tz="UTC"))
    ]
    cases = [
        ("Full run, before", s[s.era == "static"]),
        (
            "Full run, after\n(learned weights)",
            jobs[jobs.job_id.isin(warm_jobs) & (jobs.level == 3)],
        ),
        ("Re-run failed jobs,\nbefore", rerun[rerun.kind == "old"]),
        ("Re-run failed jobs,\nafter", narrowed),
    ]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    stats = {}
    for i, (name, frame) in enumerate(cases):
        pull = frame.pull_s.median() / 60
        install = frame.install_s.median() / 60
        tests = frame.test_s.median() / 60
        other = max(
            (
                frame.dur_s - frame.pull_s.fillna(0) - frame.install_s.fillna(0) - frame.test_s
            ).median()
            / 60,
            0,
        )
        left = 0
        for part, value in (
            ("Docker pull", pull),
            ("Install", install),
            ("Tests", tests),
            ("Other", other),
        ):
            ax.barh(i, value, left=left, color=STEP_COLORS[part], edgecolor="grey", linewidth=0.4)
            left += value
        ax.text(
            left + 2,
            i,
            f"{frame.dur_s.median() / 60:.0f} min (n={len(frame)})",
            va="center",
            fontsize=8.5,
        )
        stats[name.replace("\n", " ")] = dict(
            n=len(frame), job_min=frame.dur_s.median() / 60, test_min=tests, pull_min=pull
        )
    ax.set_yticks(range(len(cases)), [c[0] for c in cases])
    ax.invert_yaxis()
    ax.set_xlabel("median minutes per sGPU job (level 3); bars are per-step medians")
    ax.legend(
        handles=[Patch(facecolor=c, edgecolor="grey", label=p) for p, c in STEP_COLORS.items()],
        loc="lower right",
        frameon=False,
    )
    ax.set_title("A sGPU job, before and after", loc="left")
    ax.set_xlim(0, 215)
    fig.savefig(f"{out}/sgpu_job_time.png")
    plt.close(fig)
    return stats


def fig_scope(jobs, static, scope, out):
    """What the merged suite-selection rules would have saved on the PRs that ran before them."""
    before = scope[(scope.pr != 696)].merge(jobs[["run_id"]].drop_duplicates(), on="run_id")
    pre = jobs[(jobs.kind == "pr") & (jobs.started < MERGED_AT)]
    before = before[before.run_id.isin(pre.run_id)]
    sel = before.set_index("run_id")

    # sGPU: the measured duration of each suite the rules would not have selected
    st = static.merge(jobs[["job_id", "run_id"]], on="job_id")
    st = st[st.run_id.isin(sel.index)]
    sgpu_total = sgpu_saved = 0.0
    for row in st.itertuples():
        chosen = set(sel.loc[row.run_id, "sgpu"].split())
        for suite in SUITES:
            value = getattr(row, f"{suite}_s")
            if pd.notna(value):
                sgpu_total += value
                sgpu_saved += 0 if suite in chosen else value
    # mGPU: the old per-framework jobs, 8 GPUs each
    m = jobs[
        jobs.jtype.isin(["mgpu-torch", "mgpu-jax"])
        & jobs.run_id.isin(sel.index)
        & jobs.test_s.notna()
    ]
    need = m.apply(
        lambda r: ("torch-mgpu" if r.jtype == "mgpu-torch" else "jax-mgpu")
        in sel.loc[r.run_id, "mgpu"].split(),
        axis=1,
    )
    mgpu_total = (m.test_s * 8).sum()
    mgpu_saved = (m.test_s[~need] * 8).sum()

    def label(row):
        if row.everything:
            return "every suite"
        suites = [s for s in (row.sgpu + " " + row.mgpu).split() if s]
        frameworks = sorted({s.replace("-mgpu", "") for s in suites if s != "examples"})
        return " + ".join(frameworks) + " only" if frameworks else "examples only"

    counts = before.apply(label, axis=1).value_counts()
    prs = before.groupby("pr").everything.first()

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    colors = [COLORS["static"] if k == "every suite" else COLORS["warm"] for k in counts.index]
    ax.barh(counts.index, counts.values, color=colors)
    for i, v in enumerate(counts.values):
        share = 100 * v / counts.sum()
        ax.text(
            v + 4,
            i,
            f"{v} ({share:.0f}%)" if share >= 1 else f"{v} (<1%)",
            va="center",
            fontsize=8.5,
        )
    ax.invert_yaxis()
    ax.set_xlabel(f"PR CI runs before the merge ({counts.sum()} runs, {len(prs)} PRs)")
    ax.set_title("What each PR run would have selected", loc="left")
    ax.set_xlim(0, counts.max() * 1.25)

    ax = axes[1]
    names = ["sGPU tests", "mGPU tests"]
    totals = np.array([sgpu_total, mgpu_total]) / 3600
    saved = np.array([sgpu_saved, mgpu_saved]) / 3600
    ax.barh(names, saved, color=COLORS["warm"], label="not needed by the PR's changes")
    ax.barh(names, totals - saved, left=saved, color=COLORS["static"], label="needed")
    for i, (t, sv) in enumerate(zip(totals, saved)):
        ax.text(
            t + 40, i, f"{sv:.0f} of {t:.0f} GPU-h ({100 * sv / t:.0f}%)", va="center", fontsize=8.5
        )
    ax.invert_yaxis()
    ax.set_xlabel("GPU-hours of PR test time before the merge")
    ax.set_title("GPU time the rules would have skipped", loc="left")
    ax.set_xlim(0, totals.max() * 1.45)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    fig.tight_layout()
    fig.savefig(f"{out}/scope_selection.png")
    plt.close(fig)
    return dict(
        runs=int(counts.sum()),
        prs=int(len(prs)),
        runs_subset=int(counts.sum() - counts.get("every suite", 0)),
        prs_subset=int((~prs).sum()),
        sgpu_gpu_hours=[float(sgpu_saved / 3600), float(sgpu_total / 3600)],
        mgpu_gpu_hours=[float(mgpu_saved / 3600), float(mgpu_total / 3600)],
    )


def dev_end_to_end(jobs):
    """Hours from a dev push to its last GPU job (first attempt)."""
    dev = jobs[
        (jobs.kind == "branch")
        & (jobs.branch == "dev")
        & (jobs.event == "push")
        & (jobs.attempt == 1)
    ]
    dev = dev[dev.jtype.isin(["sgpu", "mgpu-torch", "mgpu-jax", "mgpu-queue"])]
    per_run = dev.groupby("run_id").agg(created=("run_created", "first"), last=("completed", "max"))
    per_run["hours"] = (per_run["last"] - per_run["created"]).dt.total_seconds() / 3600
    before = per_run[per_run.created < MERGED_AT].hours
    after = per_run[per_run.created >= MERGED_AT].hours
    return dict(
        before_n=len(before),
        before_median_h=float(before.median()),
        after=[float(h) for h in after],
    )


def main():
    """Make every figure and write summary.json."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=os.path.join(HERE, "..", "data"))
    parser.add_argument("--figures", default=os.path.join(HERE, "..", "figures"))
    args = parser.parse_args()
    os.makedirs(args.figures, exist_ok=True)

    jobs, items, static, scope = load(args.data)
    qruns = queue_runs(jobs, items)
    sruns = static_runs(jobs, static)
    summary = {
        "same_work": fig_same_work(items, jobs, args.figures),
        "timeline_l3": fig_timeline(jobs, qruns, args.figures),
        "cold_vs_warm": fig_cold_warm(items, jobs, args.figures),
        "utilisation": fig_utilisation(sruns, qruns, args.figures),
        "job_time": fig_job_time(jobs, qruns, args.figures),
        "scope": fig_scope(jobs, static, scope, args.figures),
        "dev_end_to_end": dev_end_to_end(jobs),
    }
    with open(f"{args.data}/summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=float)
    print(json.dumps(summary, indent=2, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
