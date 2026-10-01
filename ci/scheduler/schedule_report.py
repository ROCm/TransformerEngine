#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Render one queue run's scheduling report from its measured timings.

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



def outcome(rc, incomplete):
    """How one item ended, as the word that says what to do about it.

    Five words, and the line between them is the one the weight table draws:

      pass, fail, error   the suite ran to the end. Its duration is what the
                          item costs -- failing tests cost what they cost -- so
                          the next run's weight learns from it.
      incomplete, killed  the suite was stopped. Its duration is only where it
                          stopped, so the weight table ignores the row and the
                          item keeps the weight it came in with.

    "incomplete" is the same word ``ci/junit_report.py`` uses for the same
    condition, and both read it from the same signal: the result sidecar a dying
    pytest leaves behind. Neither can name the cause -- a per-test timeout, a
    segfault and an OOM-kill are indistinguishable from the outside, which is
    why the word says what happened rather than why.
    """
    if rc in KILLED_RCS:
        # Killed from outside, which is a different story from the suite dying
        # on its own: 124 is a `timeout` expiry, 137 a SIGKILL or an OOM-kill.
        return "killed"
    if incomplete == 1:
        # rc cannot say this on its own: a --timeout-method=thread expiry exits
        # 1, exactly like an ordinary test failure.
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


def calculate_schedule_table(frame, default_weight):
    """What ran where, in execution order, against its cached duration."""
    # Grouped by the first GPU an item held; an item's GPUs are listed in full,
    # so a 4-GPU item shows which three others it kept busy.
    order = frame.assign(_gpu=frame["gpus"].str.split(",").str[0].astype(int)).sort_values(
        ["_gpu", "start_off", "name"], kind="stable"
    )
    rows = [
        [
            "GPUs",
            "Start (s)",
            "Duration (s)",
            "Result",
            "Cached duration (s)",
            "Run vs cached",
            "Test item (suite/file.backend.label)",
        ]
    ]
    for row in order.itertuples(index=False):
        known = row.est > 0 and row.est != default_weight
        # Run vs cached is the scheduler feedback loop made visible: a large
        # positive miss is an item that should have been dispatched earlier.
        change = f"{(row.secs - row.est) * 100 / row.est:+.0f}%" if known else "n/a"
        # Reported for a stopped item too, unlike in the weights table below:
        # there the number would describe a learning step that is not going to
        # happen, but here it describes the GPU-time the item really did take,
        # and an item that sat on a GPU for ten times its cached duration is a
        # scheduling miss whether or not it got as far as finishing.
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

    A weight moves toward a run's duration by an exponential average rather
    than taking it whole, so "Run vs weight" says how far this run was from the
    weight -- the pull on it -- not how far the weight moved.
    """
    rows = []
    for row in frame.itertuples(index=False):
        key = item_key(row.label, "" if row.tag == "whole" else row.tag)
        new = weights.get(key)
        known = row.est > 0 and row.est != default_weight
        if not row.measured:
            # Deliberately not a percentage: this run's duration is where the
            # item stopped, so the gap between it and the weight measures
            # nothing, and printing "+1044%" next to a weight that did not move
            # invites the reader to go looking for the bug that ate the update.
            change = f"weight kept -- {outcome(row.rc, row.incomplete)}"
        elif known:
            change = f"{(row.secs - row.est) * 100 / row.est:+.0f}%"
        else:
            change = "first run"
        rows.append(
            (
                # An item the table does not name is one the next run has no
                # weight for, and order_queue sorts those first -- so it sorts
                # first here too, for the ordering claim above to hold.
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
    # Name breaks ties so the order is stable run to run rather than dependent on
    # dispatch order.
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
    """Items the queue held that produced no timing record.

    Only possible if a worker died outright, and silence here would read as
    success.
    """
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
):
    """The Markdown report the workflow appends to the job summary."""
    n_gpus = len(gpu_ids)
    ran = len(frame)
    failed = int((frame["rc"] != 0).sum())
    # Called out up front because it changes what the rest of the report means:
    # a stopped item's duration is a floor, so both the utilisation above and
    # the weights below are being read off a run that did not finish.
    stopped = int((~frame["measured"]).sum())
    note = f" ({stopped} stopped short, so not timed)" if stopped else ""
    util = int(frame["gpu_secs"].sum()) * 100 / (wall * n_gpus)
    mark = ":x:" if failed or ran != total_items else ":white_check_mark:"

    out = [
        f"## {title}",
        "",
        f"{mark} **{ran} items** on {n_gpus} GPUs -- {failed} failed{note} "
        f"-- {wall}s in the queue"
        # Utilisation is left out of a re-run's headline rather than shown and
        # disclaimed: a handful of items on eight GPUs reads as single digits,
        # which says only that the queue was short and invites the wrong fix.
        + ("" if rerun else f" at {util:.1f}% GPU utilisation")
        + (f", {total_wall}s end to end" if total_wall else ""),
        "",
    ]

    if ran != total_items:
        out.append(f"> :warning: **Only {ran} of {total_items} items produced a timing")
        out.append("> record.** The rest were never dispatched, which means a worker died:")
        out.append("")
        out.extend(f"> - `{key}`" for key in missing)
        out.append("")

    # This reads a re-run's short queue as a badly packed one -- five idle GPUs
    # are what queueing three items looks like, not a scheduling fault -- so it
    # is a full run's section only.
    if not rerun:
        rows = calculate_gpu_utilisation_table(frame, wall, gpu_ids)
        out += ["### Per-GPU utilisation", ""] + render_md(rows) + [""]

    # The big tables are collapsed: they are the detail you open once you know
    # from the sections above that something is worth looking at. Each carries a
    # legend, because the words in its Result / Run vs weight column are the whole point
    # of the table and are not self-explanatory.
    #
    # The schedule stays on a re-run even though the sections above it go. It is
    # the only place an opaque whole-suite item -- `examples/whole`, `core/whole`
    # -- reports its verdict at all, since those produce no per-test JUnit XML
    # for the suite reports to pick up. Dropping it would hide their failures.
    tables = [
        (
            "Schedule -- what ran where, in execution order",
            calculate_schedule_table(frame, default_weight),
            # A re-run's schedule is a handful of rows someone is reading to find
            # out whether the thing that failed still fails. The vocabulary is
            # worth a paragraph against a full run's ninety rows, not against
            # three, so the legend is a full run's.
            ""
            if rerun
            else "`pass` / `fail` / `error` ran to the end. `incomplete` hard-exited "
            "mid-test (a per-test timeout, a segfault or an OOM-kill -- the same "
            "condition the JUnit report calls incomplete); `killed` was stopped "
            "from outside. Only the first three are timings; the last two are "
            "just where the item stopped.",
        ),
    ]
    # Titled "for next run", and on a re-run that is a lie: phase 6 skips the
    # update entirely, so every Next weight here is one nothing will ever write.
    if not rerun:
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
    parser.add_argument("log_dir", help="the queue's log directory, e.g. test-results/logs")
    parser.add_argument(
        "--title", default="Queue schedule", help='report heading, e.g. "mGPU queue schedule"'
    )
    # Required, both of them: they are facts about the run that no file in the
    # log directory records, and guessing at either yields a plausible-looking
    # report with the wrong utilisation in it.
    parser.add_argument(
        "--gpus", required=True, metavar="IDS", help='space-separated device ids, e.g. "0 1 2 4"'
    )
    parser.add_argument(
        "--wall", type=int, required=True, metavar="SECS", help="queue wall clock in seconds"
    )
    # Optional, unlike --wall: without them the report simply omits the
    # end-to-end line rather than inventing one, which keeps the tool usable
    # against a timings.tsv salvaged from an artifact.
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
        required=True,
        metavar="TABLE",
        help="the weight table build_weights.py has just rewritten; the report "
        "reads it to show what the next run will schedule with",
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

    # Without queue.tsv there is nothing to say an item is missing, so every item
    # that ran is taken to be every item there was.
    queue_keys = read_queue_keys(queue) if os.path.exists(queue) else list(frame["name"])
    missing = missing_items(frame, queue_keys)

    weights = read_weights(args.weights)

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
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        handle.write("\n".join(report) + "\n")

    print(f"Scheduling report: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
