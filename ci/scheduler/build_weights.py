#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Apply or update the queue scheduler's weight table.

  order   sort a queue by the cached weights: widest first, then longest first.
          Without a table the queue keeps its listed order.
  update  blend a finished run's timings into the cached table.
"""

import argparse
import sys

from queue_files import read_timings


def measurements(frame):
    """Split one run's timings into ``(measured, dispatched)``.

    ``measured`` maps key -> seconds for the rows that are full measurements (see
    ``queue_files.read_timings``); ``merge_weights`` learns from it. ``dispatched``
    is every key the run started, measured or not; ``prune_weights`` uses it to
    know the item still exists.
    """
    # An opaque suite item ("whole") is keyed by its bare label
    key = frame["label"].where(frame["tag"] == "whole", frame["name"])
    usable = frame["measured"]
    measured = frame["secs"][usable].groupby(key[usable]).sum()
    return measured.to_dict(), set(key)


def item_key(label, tag):
    """The table key for a work item: ``<label>/<tag>``, or the bare label for an opaque suite."""
    return f"{label}/{tag}" if tag else label


def read_items(path):
    """Read items.tsv: every work item that exists at this TEST_LEVEL."""
    try:
        handle = open(path, encoding="utf-8")
    except OSError as err:
        print(f"warning: cannot read {path} ({err}); nothing will be pruned", file=sys.stderr)
        return None, None
    keys, labels = set(), set()
    with handle:
        for line in handle:
            label, _, tag = line.rstrip("\n").partition("\t")
            if not label:
                continue
            labels.add(label)
            keys.add(item_key(label, tag))
    return keys, labels


def read_weights(path):
    """Read the cached weight table the scheduler uses to sort the queue."""
    weights = {}
    try:
        handle = open(path, encoding="utf-8")
    except OSError:
        return weights
    with handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, value = line.rpartition(" ")
            try:
                weights[key] = float(value)
            except ValueError:
                continue
    return weights


QUEUE_COLUMNS = 5  # label, cmd, tag, rest, gpus


def read_queue(path):
    """Read the unweighted queue the expansion phase wrote."""
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            fields = line.split("\t")
            fields += [""] * (QUEUE_COLUMNS - len(fields))
            rows.append(tuple(fields[:QUEUE_COLUMNS]))
    return rows


def order_queue(rows, weights, default_weight):
    """Weight every work item and sort the queue widest first, then longest first.

    Longest first leaves only short items for the end, so no GPU is still running a
    long one after the others go idle; unknown items get ``default_weight`` and so
    start first. Widest first because the scheduler dispatches strictly in order:
    whole-box items then run back to back instead of waiting for the box to drain.
    """
    weighted = [
        (
            int(weights.get(item_key(label, tag), default_weight)),
            label,
            cmd,
            tag,
            rest,
            int(gpus or 1),
        )
        for label, cmd, tag, rest, gpus in rows
    ]
    weighted.sort(key=lambda row: (-row[5], -row[0]))
    return weighted


def plan_lines(queue, weights_path, n_weights, gpu_ids, default_weight):
    """Render what the scheduler intends to do, before it does any of it."""

    def est(weight):
        return "unknown" if weight == default_weight else f"{weight}s"

    n_gpus = len(gpu_ids) or 1
    lines = [f"== Queue: {len(queue)} items over {n_gpus} GPUs ({' '.join(gpu_ids)}) =="]
    if not weights_path:
        lines.append("   weights: none kept for this queue -- items run in the order listed")
        lines.append("== Queue plan (dispatch order) ==")
    else:
        if n_weights:
            lines.append(f"   weights: {weights_path} ({n_weights} items)")
        else:
            lines.append(
                f"   weights: none yet -- this run is unordered and will write {weights_path}"
            )
        lines.append("== Queue plan (dispatch order; est = weight used to sort) ==")

    total = 0
    for position, (weight, label, _cmd, tag, _rest, gpus) in enumerate(queue, 1):
        # GPU-seconds: an item on 4 GPUs counts 4 times its duration
        total += weight * gpus
        estimate = f"est={est(weight):<8} " if weights_path else ""
        lines.append(
            f"  {position:3d}. {estimate}gpus={gpus:<2} {label:<9} {tag or '(whole suite)'}"
        )
    if weights_path:
        lines.append("")
        lines.append(
            f"  estimated work {total} GPU-s over {n_gpus} GPUs -> lower bound {total / n_gpus:.0f}s"
            f" (longest item {est(max(row[0] for row in queue))})"
        )
    return lines


def merge_weights(old, new, alpha_up=0.5, alpha_down=0.1):
    """Blend this run's measurements into the cached weights.

    A known key moves by an exponential average,
    ``(1 - alpha) * previous + alpha * measured``; a new key takes its measurement
    as is. A rise is trusted faster (``alpha_up``) than a fall (``alpha_down``):
    underestimating an item leaves it running after the other GPUs go idle, which
    costs more than starting it early, and one fast or cut-short run barely moves it.
    """
    merged = dict(old)
    for key, value in new.items():
        previous = merged.get(key)
        if previous is None:
            merged[key] = value
            continue
        alpha = alpha_up if value > previous else alpha_down
        merged[key] = (1.0 - alpha) * previous + alpha * value
    return merged


def prune_weights(weights, dispatched, live_keys, live_labels):
    """Drop the weights of work items that no longer exist.

    A key is kept if items.tsv lists it or this run dispatched it. Otherwise a
    ``<label>`` key (opaque suite) is pruned, and a ``<label>/<tag>`` key only if
    its label is listed: a missing label may just mean the suite's list-all
    pass failed.

    Returns ``(kept, pruned)``, with the reason for each pruned key.
    """
    kept, pruned = {}, []
    for key, value in weights.items():
        label = key.split("/", 1)[0]
        if key in dispatched or key in live_keys:
            kept[key] = value
        elif key == label:
            pruned.append((key, "its suite is no longer in the config"))
        elif label not in live_labels:
            kept[key] = value
        else:
            pruned.append((key, "its suite no longer lists it"))
    return kept, pruned


def run_order(args):
    """Apply the cached weight table to the queue and send it to the scheduler."""
    try:
        rows = read_queue(args.queue)
    except OSError as err:
        print(f"error: cannot read {args.queue} ({err})", file=sys.stderr)
        return 1
    if not rows:
        print("error: queue is empty: every suite expanded to nothing", file=sys.stderr)
        return 1

    weights = read_weights(args.weights) if args.weights else {}
    queue = order_queue(rows, weights, args.default_weight)
    try:
        with open(args.output, "w", encoding="utf-8") as handle:
            for row in queue:
                handle.write("\t".join(str(field) for field in row) + "\n")
    except OSError as err:
        print(f"error: cannot write {args.output} ({err})", file=sys.stderr)
        return 1

    for line in plan_lines(
        queue, args.weights, len(weights), args.gpus.split(), args.default_weight
    ):
        print(line)
    return 0


def run_update(args):
    """Blend a finished run's measurements into the table and update the cache."""
    measured, dispatched = measurements(read_timings(args.timings))
    if not measured:
        # Nothing usable (every row killed, cut short or malformed): keep the table
        print(f"error: no usable measurements in {args.timings}", file=sys.stderr)
        return 1

    previous = read_weights(args.output)
    weights = merge_weights(previous, measured)
    print(
        f"merged {len(measured)} measurements into {len(previous)} known weights "
        f"({len(set(weights) - set(previous))} new)",
        file=sys.stderr,
    )

    if args.items:
        live_keys, live_labels = read_items(args.items)
        if live_keys is not None:
            weights, pruned = prune_weights(weights, dispatched, live_keys, live_labels)
            # List each prune: a deleted test looks the same as a broken expansion
            for key, why in sorted(pruned):
                print(f"pruned {key}: {why}", file=sys.stderr)

    lines = [
        f"{key} {value:.1f}\n" for key, value in sorted(weights.items(), key=lambda kv: -kv[1])
    ]
    with open(args.output, "w", encoding="utf-8") as handle:
        handle.writelines(lines)
    print(f"wrote {len(lines)} weights to {args.output}", file=sys.stderr)
    return 0


def main():
    """Dispatch to ``order`` (apply the table) or ``update`` (rewrite it)."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    order = sub.add_parser(
        "order", help="apply the table: weight a queue and sort it longest-first"
    )
    order.add_argument("queue", metavar="QUEUE", help="the unweighted queue from expansion")
    order.add_argument(
        "-w",
        "--weights",
        metavar="TABLE",
        help="weight table to read; without one the queue keeps its listed order",
    )
    order.add_argument(
        "-o", "--output", required=True, metavar="FILE", help="where to write the sorted queue"
    )
    order.add_argument(
        "--default-weight",
        type=int,
        required=True,
        metavar="SECS",
        help="weight for an item the table has never seen; it sorts first",
    )
    order.add_argument(
        "--gpus", default="", metavar="IDS", help='space-separated device ids, e.g. "0 1 2 4"'
    )

    update = sub.add_parser("update", help="fold a finished run's timings into the table")
    update.add_argument("timings", metavar="TIMINGS", help="timings.tsv written by run_queue.sh")
    update.add_argument(
        "-o", "--output", required=True, metavar="TABLE", help="weight table to update in place"
    )
    update.add_argument(
        "--items",
        metavar="FILE",
        help="items.tsv listing every work item that exists at this level; "
        "without it, no weight is ever pruned",
    )

    args = parser.parse_args()
    return run_order(args) if args.command == "order" else run_update(args)


if __name__ == "__main__":
    sys.exit(main())
