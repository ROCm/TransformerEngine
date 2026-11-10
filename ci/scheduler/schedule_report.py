#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Render one queue run's scheduling report from its measured timings."""

import argparse
import os
import sys

from queue_files import KILLED_RCS, read_timings
from build_weights import item_key, read_weights

# Where the scheduler puts things under its log directory
TIMINGS_PATH = ("queue", "timings.tsv")
QUEUE_PATH = ("queue", "queue.tsv")
REPORT_PATH = ("report", "schedule.md")



def outcome(rc, incomplete):
    """How one item ended: pass, fail, error, incomplete or killed.

    The first three ran to the end, so their duration is a measurement; the last two
    were stopped, so the weight table ignores them. "incomplete" means pytest died
    mid-run (as in junit_report.py); "killed" means it was stopped from outside.
    """
    if rc in KILLED_RCS:
        # 124 is a `timeout` expiry, 137 a SIGKILL or an OOM-kill
        return "killed"
    if incomplete == 1:
        # rc alone cannot tell: a thread-method timeout also exits 1
        return "incomplete"
    if rc == 0:
        return "pass"
    return "fail" if rc == 1 else "error"


def read_queue_keys(path):
    """Every item the queue intended to run, as "<label>/<tag>" keys.

    queue.tsv columns are weight, label, cmd, tag, rest; an empty tag is an
    opaque whole-suite item, which the timings file records as "whole".
    """
    keys = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 4:
                keys.append(f"{fields[1]}/{fields[3] or 'whole'}")
    return keys

def render_md(rows):
    """Rows as a GitHub-flavoured Markdown table."""
    out = []
    for n, row in enumerate(rows):
        out.append("| " + " | ".join(c.replace("|", "\\|") for c in row) + " |")
        if n == 0:
            out.append("|" + "---|" * len(row))
    return out

def calculate_gpu_utilisation_table(frame, wall, gpu_ids):
    """Per-GPU utilisation.

    A multi-GPU item is charged in full to every GPU it held.
    """
    frame = frame.assign(gpu=frame["gpus"].str.split(",")).explode("gpu")
    per_gpu = frame.groupby("gpu")
    busy = per_gpu["secs"].sum().reindex(gpu_ids, fill_value=0).astype(int)
    count = per_gpu.size().reindex(gpu_ids, fill_value=0).astype(int)
    failed = (
        frame[frame["rc"] != 0].groupby("gpu").size().reindex(gpu_ids, fill_value=0).astype(int)
    )

    return [["GPU", "Items", "Busy (s)", "Idle (s)", "Util (%)", "Failed"]] + [
        [
            f"gpu{gpu}",
            f"{count[gpu]}",
            f"{busy[gpu]}",
            f"{wall - busy[gpu]}",
            f"{busy[gpu] * 100 / wall:.1f}",
            f"{failed[gpu]}",
        ]
        for gpu in gpu_ids
    ]


def calculate_schedule_table(frame, default_weight, weighted=True):
    """What ran where, in execution order, against its cached duration.

    ``weighted`` is false for a queue that keeps no weight table; it has no
    cached duration to compare against, so those two columns are left out.
    """
    # Group by each item's first GPU; the GPUs column lists every GPU it held
    order = frame.assign(_gpu=frame["gpus"].str.split(",").str[0].astype(int)).sort_values(
        ["_gpu", "start_off", "name"], kind="stable"
    )
    header = ["GPUs", "Start (s)", "Duration (s)", "Result"]
    if weighted:
        header += ["Cached duration (s)", "Run vs cached"]
    rows = [header + ["Test item (suite/file.backend.label)"]]
    for row in order.itertuples(index=False):
        if not weighted:
            rows.append(
                [
                    f"gpu{row.gpus}",
                    f"{row.start_off}",
                    f"{row.secs}",
                    outcome(row.rc, row.incomplete),
                    row.name,
                ]
            )
            continue
        known = row.est > 0 and row.est != default_weight
        # A large positive miss is an item that should have started earlier
        change = f"{(row.secs - row.est) * 100 / row.est:+.0f}%" if known else "n/a"
        # Shown for stopped items too: the GPU time they took is real
        rows.append(
            [
                f"gpu{row.gpus}",
                f"{row.start_off}",
                f"{row.secs}",
                outcome(row.rc, row.incomplete),
                "none" if row.est == default_weight else f"{row.est}",
                change,
                row.name,
            ]
        )
    return rows


def calculate_updated_weights_table(frame, weights, default_weight):
    """What the next run will schedule each item with, and what moved it there.

    "Run vs weight" is how far this run was from the weight used; the weight
    itself moves only part of the way there.
    """
    rows = []
    for row in frame.itertuples(index=False):
        key = item_key(row.label, "" if row.tag == "whole" else row.tag)
        new = weights.get(key)
        known = row.est > 0 and row.est != default_weight
        if not row.measured:
            # No percentage: a stopped run's duration says nothing about the weight
            change = f"weight kept -- {outcome(row.rc, row.incomplete)}"
        elif known:
            change = f"{(row.secs - row.est) * 100 / row.est:+.0f}%"
        else:
            change = "first run"
        rows.append(
            (
                # Items with no weight sort first, as in order_queue
                new if new is not None else float("inf"),
                [
                    f"{new:.0f}" if new is not None else "none",
                    "none" if not known else f"{row.est}",
                    f"{row.secs}",
                    change,
                    row.name,
                ],
            )
        )
    # Break ties by name so the order is stable from run to run
    rows.sort(key=lambda r: (-r[0], r[1][4]))
    header = [
        "Next weight (s)",
        "Weight used (s)",
        "This run (s)",
        "Run vs weight",
        "Test item (suite/file.backend.label)",
    ]
    return [header] + [row for _, row in rows]


def missing_items(frame, queue_keys):
    """Items the queue held that produced no timing record, i.e. a worker died."""
    seen = set(frame["name"])
    return [key for key in queue_keys if key not in seen]


# ---------------------------------------------------------------------------
# Report.


def render_report_md(
    frame,
    wall,
    gpu_ids,
    total_items,
    default_weight,
    missing,
    weights,
    total_wall,
    title,
    rerun=False,
    skipped=(),
):
    """The Markdown report the workflow appends to the job summary."""
    n_gpus = len(gpu_ids)
    ran = len(frame)
    failed = int((frame["rc"] != 0).sum())
    # Flag stopped items in the headline: their durations are only a floor
    stopped = int((~frame["measured"]).sum())
    note = f" ({stopped} stopped short, so not timed)" if stopped else ""
    util = int(frame["gpu_secs"].sum()) * 100 / (wall * n_gpus)
    mark = ":x:" if failed or ran != total_items else ":white_check_mark:"
    # Every item held the whole box (mGPU), so per-GPU figures would all match
    whole_box = bool((frame["width"] == n_gpus).all())

    out = [
        f"## {title}",
        "",
        f"{mark} **{ran} items** on {n_gpus} GPUs -- {failed} failed{note} "
        f"-- {wall}s in the queue"
        # No utilisation for a re-run: a few items on many GPUs would look like a fault
        + ("" if rerun or whole_box else f" at {util:.1f}% GPU utilisation")
        + (f", {total_wall}s end to end" if total_wall else ""),
        "",
    ]

    if skipped:
        # Name the suites left out on purpose (--suites), so a short report does
        # not look like a lost one
        names = ", ".join(f"`{name}`" for name in skipped)
        out += [f"Not selected, so not run: {names}.", ""]

    if ran != total_items:
        out.append(f"> :warning: **Only {ran} of {total_items} items produced a timing")
        out.append("> record.** The rest were never dispatched, which means a worker died:")
        out.append("")
        out.extend(f"> - `{key}`" for key in missing)
        out.append("")

    # Full runs only: a re-run's short queue would look badly packed
    if not rerun and not whole_box:
        rows = calculate_gpu_utilisation_table(frame, wall, gpu_ids)
        out += ["### Per-GPU utilisation", ""] + render_md(rows) + [""]

    # The big tables are collapsed, each with a legend for its result words. The
    # schedule stays on a re-run as the record of what it queued and where.
    tables = [
        (
            "Schedule -- what ran where, in execution order",
            calculate_schedule_table(frame, default_weight, weighted=weights is not None),
            # A re-run's few rows need no legend
            ""
            if rerun
            else "`pass` / `fail` / `error` ran to the end. `incomplete` hard-exited "
            "mid-test (a per-test timeout, a segfault or an OOM-kill -- the same "
            "condition the JUnit report calls incomplete); `killed` was stopped "
            "from outside. Only the first three are timings; the last two are "
            "just where the item stopped.",
        ),
    ]
    # Not on a re-run, which does not update the weights, nor for a queue that
    # keeps no weight table
    if not rerun and weights is not None:
        tables.append(
            (
                "Updated weights for next run",
                calculate_updated_weights_table(frame, weights, default_weight),
                "Run vs weight is this run's duration against the weight used; the "
                "weight moves part of the way toward it, so the next weight lands in "
                "between. A failing item still updates its weight -- it cost what it "
                "cost. Only the rows marked `weight kept` are held out, and those keep "
                "the weight they came in with, so their Next and Used columns match.",
            )
        )
    for summary, rows, legend in tables:
        out += [f"<details><summary>{summary}</summary>", ""] + render_md(rows)
        out += ([""] + [legend] if legend else []) + ["", "</details>", ""]
    return out


def main():
    """Parse arguments, render the report, and never fail the run over it."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("log_dir", help="the queue's log directory, e.g. ci-data/test-results/logs")
    parser.add_argument(
        "--title", default="Queue schedule", help='report heading, e.g. "mGPU queue schedule"'
    )
    # Required: no file in the log directory records these
    parser.add_argument(
        "--gpus", required=True, metavar="IDS", help='space-separated device ids, e.g. "0 1 2 4"'
    )
    parser.add_argument(
        "--wall", type=int, required=True, metavar="SECS", help="queue wall clock in seconds"
    )
    # Optional: without it the report leaves out the end-to-end time
    parser.add_argument(
        "--total-wall",
        type=int,
        default=0,
        metavar="SECS",
        help="whole-script wall clock, queue plus expansion and setup; the "
        "number to compare against a run that predates the queue",
    )
    parser.add_argument(
        "--default-weight",
        type=int,
        default=999999,
        help="the est value that means 'no weight was known'",
    )
    parser.add_argument(
        "--weights",
        metavar="TABLE",
        help="the weight table build_weights.py has just rewritten; the report "
        "reads it to show what the next run will schedule with. Left out for a "
        "queue that keeps no weight table",
    )
    parser.add_argument(
        "--skipped",
        default="",
        metavar="LABELS",
        help="space-separated suites the run left out on purpose (run_queue.sh --suites)",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="this run queued only the items that failed a previous attempt; "
        "say so, because it makes the utilisation figures incomparable",
    )
    args = parser.parse_args()

    timings = os.path.join(args.log_dir, *TIMINGS_PATH)
    queue = os.path.join(args.log_dir, *QUEUE_PATH)
    out = os.path.join(args.log_dir, *REPORT_PATH)
    if not os.path.exists(timings):
        print(f"no {timings}: nothing to report on", file=sys.stderr)
        return 0

    frame = read_timings(timings)

    gpu_ids = args.gpus.split() or ["0"]
    wall = max(args.wall, 1)

    # Without queue.tsv, take the items that ran as all there were
    queue_keys = read_queue_keys(queue) if os.path.exists(queue) else list(frame["name"])
    missing = missing_items(frame, queue_keys)

    weights = read_weights(args.weights) if args.weights else None

    report = render_report_md(
        frame,
        wall,
        gpu_ids,
        len(queue_keys),
        args.default_weight,
        missing,
        weights,
        args.total_wall,
        args.title,
        rerun=args.rerun,
        skipped=args.skipped.split(),
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        handle.write("\n".join(report) + "\n")

    print(f"Scheduling report: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
