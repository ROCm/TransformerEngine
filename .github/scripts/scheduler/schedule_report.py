#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Render one sGPU queue run's scheduling report from its measured timings.

"""

import argparse
import os
import sys

from queue_files import KILLED_RCS, read_timings
from build_weights import item_key, read_weights

# Where the scheduler puts things under its log directory
TIMINGS_PATH = ("queue", "timings.tsv")
QUEUE_PATH = ("queue", "queue.tsv")
REPORT_PATH = ("report", "schedule.md")

REPORT_TITLE = "sGPU queue schedule"


def outcome(rc):
    """An exit code as the word that says what to do about it.
    """
    if rc == 0:
        return "pass"
    if rc == 1:
        return "fail"
    return "killed" if rc in KILLED_RCS else "error"


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

def calculate_efficiency_table(frame, wall, n_gpus):
    """What the run cost, and how much of the GPU-time it bought was used."""
    work = int(frame["secs"].sum())
    if len(frame):
        biggest = frame.loc[frame["secs"].idxmax()]
        big, bigname = int(biggest["secs"]), biggest["name"]
    else:
        big, bigname = 0, ""
    # Once one item exceeds the per-GPU average there is no ordering that
    # finishes sooner than that item, so it becomes the thing to split.
    floor = " -- floor: splitting it would now pay" if big > work / n_gpus else ""
    return [
        ["Metric", "Value", "Meaning"],
        ["total work", f"{work}s", f"sum of all {len(frame)} item durations"],
        ["actual run time", f"{wall}s", "wall clock, first item start to last item finish"],
        [
            "utilisation",
            f"{work * 100 / (wall * n_gpus):.1f}%",
            f"share of {wall}s x {n_gpus} GPUs actually spent running tests",
        ],
        ["largest item", f"{big}s", bigname + floor],
    ]


def calculate_gpu_utilisation_table(frame, wall, gpu_ids):
    """Per-GPU utilisation.
    """
    per_gpu = frame.groupby("gpu")
    busy = per_gpu["secs"].sum().reindex(gpu_ids, fill_value=0).astype(int)
    count = per_gpu.size().reindex(gpu_ids, fill_value=0).astype(int)
    failed = (
        frame[frame["rc"] != 0].groupby("gpu").size().reindex(gpu_ids, fill_value=0).astype(int)
    )

    return [["GPU", "Items", "Busy", "Idle", "Util", "Failed"]] + [
        [
            f"gpu{gpu}",
            f"{count[gpu]}",
            f"{busy[gpu]}s",
            f"{wall - busy[gpu]}s",
            f"{busy[gpu] * 100 / wall:.1f}%",
            f"{failed[gpu]}",
        ]
        for gpu in gpu_ids
    ]


def calculate_schedule_table(frame, default_weight):
    """What ran where, in execution order, against what it was estimated to cost."""
    order = frame.assign(_gpu=frame["gpu"].astype(int)).sort_values(
        ["_gpu", "start_off", "name"], kind="stable"
    )
    rows = [["GPU", "Start", "Duration", "Result", "Estimate", "% change", "Test name"]]
    for row in order.itertuples(index=False):
        known = row.est > 0 and row.est != default_weight
        # The % change is the scheduler feedback loop made visible: a large
        # positive miss is an item that should have been dispatched earlier.
        change = f"{(row.secs - row.est) * 100 / row.est:+.0f}%" if known else "n/a"
        # "cut" marks a duration that is where the item was stopped rather than
        # what it costs, which is also why the next run's weight table ignores
        # that row.
        result = outcome(row.rc) + (" (cut)" if row.incomplete == 1 else "")
        rows.append(
            [
                f"gpu{row.gpu}",
                f"t+{row.start_off}s",
                f"{row.secs}s",
                result,
                "unknown" if row.est == default_weight else f"{row.est}s",
                change,
                row.name,
            ]
        )
    return rows


def calculate_updated_weights_table(frame, weights, default_weight):
    """
    What the next run will schedule each item with.
    """
    rows = []
    for row in frame.itertuples(index=False):
        key = item_key(row.label, "" if row.tag == "whole" else row.tag)
        new = weights.get(key)
        known = row.est > 0 and row.est != default_weight
        rows.append(
            (
                # An item the table does not name is one the next run has no
                # weight for, and order_queue sorts those first -- so it sorts
                # first here too, for the ordering claim above to hold.
                new if new is not None else float("inf"),
                [
                    f"{new:.0f}s" if new is not None else "unknown",
                    "unknown" if not known else f"{row.est}s",
                    f"{row.secs}s" + (" (cut)" if row.incomplete == 1 else ""),
                    f"{(row.secs - row.est) * 100 / row.est:+.0f}%" if known else "n/a",
                    row.name,
                ],
            )
        )
    # Name breaks ties so the order is stable run to run rather than dependent on
    # dispatch order.
    rows.sort(key=lambda r: (-r[0], r[1][4]))
    return [["Updated weight", "This run", "Current weight", "% change", "Test name"]] + [
        row for _, row in rows
    ]


def missing_items(frame, queue_keys):
    """Items the queue held that produced no timing record.

    Only possible if a worker died outright, and silence here would read as
    success.
    """
    seen = set(frame["name"])
    return [key for key in queue_keys if key not in seen]


# ---------------------------------------------------------------------------
# Report.


def render_report_md(frame, wall, gpu_ids, total_items, default_weight, missing, weights):
    """The Markdown report the workflow appends to the job summary."""
    n_gpus = len(gpu_ids)
    ran = len(frame)
    failed = int((frame["rc"] != 0).sum())
    util = int(frame["secs"].sum()) * 100 / (wall * n_gpus)
    mark = ":x:" if failed or ran != total_items else ":white_check_mark:"

    out = [
        f"## {REPORT_TITLE}",
        "",
        f"{mark} **{ran} items** on {n_gpus} GPUs -- {failed} failed "
        f"-- {wall}s wall clock at {util:.1f}% GPU utilisation",
        "",
    ]

    if ran != total_items:
        out.append(f"> :warning: **Only {ran} of {total_items} items produced a timing")
        out.append("> record.** The rest were never dispatched, which means a worker died:")
        out.append("")
        out.extend(f"> - `{key}`" for key in missing)
        out.append("")

    for heading, rows in (
        ("Efficiency", calculate_efficiency_table(frame, wall, n_gpus)),
        ("Per-GPU utilisation", calculate_gpu_utilisation_table(frame, wall, gpu_ids)),
    ):
        out += [f"### {heading}", ""] + render_md(rows) + [""]

    # The two big tables are collapsed: they are the detail you open once you
    # know from the sections above that something is worth looking at.
    for summary, rows in (
        (
            "Schedule -- what ran where, in execution order",
            calculate_schedule_table(frame, default_weight),
        ),
        (
            "Updated weights for next run",
            calculate_updated_weights_table(frame, weights, default_weight),
        ),
    ):
        out += [f"<details><summary>{summary}</summary>", ""] + render_md(rows)
        out += ["", "</details>", ""]
    return out


def main():
    """Parse arguments, render the report, and never fail the run over it."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("log_dir", help="the queue's log directory, e.g. test-results/logs")
    # Required, both of them: they are facts about the run that no file in the
    # log directory records, and guessing at either yields a plausible-looking
    # report with the wrong utilisation in it.
    parser.add_argument(
        "--gpus", required=True, metavar="IDS", help='space-separated device ids, e.g. "0 1 2 4"'
    )
    parser.add_argument(
        "--wall", type=int, required=True, metavar="SECS", help="run wall clock in seconds"
    )
    parser.add_argument(
        "--default-weight",
        type=int,
        default=999999,
        help="the est value that means 'no weight was known'",
    )
    parser.add_argument(
        "--weights",
        required=True,
        metavar="TABLE",
        help="the weight table build_weights.py has just rewritten; the report "
        "reads it to show what the next run will schedule with",
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

    # Without queue.tsv there is nothing to say an item is missing, so every item
    # that ran is taken to be every item there was.
    queue_keys = read_queue_keys(queue) if os.path.exists(queue) else list(frame["name"])
    missing = missing_items(frame, queue_keys)

    weights = read_weights(args.weights)

    report = render_report_md(
        frame, wall, gpu_ids, len(queue_keys), args.default_weight, missing, weights
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        handle.write("\n".join(report) + "\n")

    print(f"Scheduling report: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
