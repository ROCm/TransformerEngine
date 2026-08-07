#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Order/Update the sGPU scheduler's weight table.

  order  : Read the cached weight table and apply it to sort the scheduler's queue by longest-processing-time first.
  update : Read a finished run's timings and blend them into the cached weight table.
"""

import argparse
import sys

from queue_files import KILLED_RCS, read_timings


def measurements(frame):
    """Split one run's timings into what it measured and what it merely ran.

    Returns ``(measured, dispatched)``, which answer different questions and go
    to different places:

      measured    key -> seconds, for the rows worth learning from.
                  Blended into the table by ``merge_weights``.
      dispatched  every key the run started, learned from or not. Used by
                  ``prune_weights``, which only needs to know the item exists.

    A row is kept out of ``measured`` when its duration records where the item
    was stopped rather than what it costs:

      rc in KILLED_RCS   124 from a ``timeout`` expiry, 137 from a SIGKILL or an
                         OOM-kill: killed from outside, so the duration is a
                         ceiling imposed on it.
      incomplete == 1    pytest never reached its end-of-session write, which rc
                         cannot say on its own -- a ``--timeout-method=thread``
                         expiry exits 1, indistinguishable from an ordinary test
                         failure. The scheduler raises the flag from the result
                         sidecar the dying process left behind.

    Both still count as dispatched: the item plainly exists, it just taught the
    table nothing this run.
    """
    # "whole" is the scheduler's placeholder for an opaque suite item, which the
    # weight table keys by bare label rather than by "<label>/<tag>".
    key = frame["label"].where(frame["tag"] == "whole", frame["name"])
    usable = ~frame["rc"].isin(KILLED_RCS) & (frame["incomplete"] != 1)
    measured = frame["secs"][usable].groupby(key[usable]).sum()
    return measured.to_dict(), set(key)


def item_key(label, tag):
    """The table key for one work item.

    An opaque whole-suite item has no tag and is keyed by bare label; everything
    else by ``<label>/<tag>``. Every reader and writer goes through here
    """
    return f"{label}/{tag}" if tag else label


def read_items(path):
    """Read items.tsv -- every work item that exists at this TEST_LEVEL"""
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


QUEUE_COLUMNS = 4  # label, cmd, tag, rest


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
    """Weight every work item and sort the queue longest-processing-time first.

    LPT is the standard makespan heuristic: dispatch the long items first and
    what is left to fill the tail is short, so no GPU is still starting a big
    item once the others have gone idle. An item the table has never seen takes
    ``default_weight``, which sorts it first -- see the scheduler for why that
    gamble is the cheap direction.
    """
    weighted = [
        (int(weights.get(item_key(label, tag), default_weight)), label, cmd, tag, rest)
        for label, cmd, tag, rest in rows
    ]
    weighted.sort(key=lambda row: -row[0])
    return weighted


def plan_lines(queue, weights_path, n_weights, gpu_ids, default_weight):
    """Render what the scheduler intends to do, before it does any of it."""

    def est(weight):
        return "unknown" if weight == default_weight else f"{weight}s"

    n_gpus = len(gpu_ids) or 1
    lines = [f"== Queue: {len(queue)} items over {n_gpus} GPUs ({' '.join(gpu_ids)}) =="]
    if n_weights:
        lines.append(f"   weights: {weights_path} ({n_weights} items)")
    else:
        lines.append(f"   weights: none yet -- this run is unordered and will write {weights_path}")

    lines.append("== Queue plan (dispatch order; est = weight used to sort) ==")
    total = 0
    for position, (weight, label, _cmd, tag, _rest) in enumerate(queue, 1):
        total += weight
        lines.append(f"  {position:3d}. est={est(weight):<8} {label:<9} {tag or '(whole suite)'}")
    lines.append("")
    lines.append(
        f"  estimated work {total}s over {n_gpus} GPUs -> lower bound {total / n_gpus:.0f}s"
        f" (largest item {est(queue[0][0])})"
    )
    return lines


def merge_weights(old, new, alpha_up=0.5, alpha_down=0.1):
    """Update cached table with new weights from current run.

    Every key already in the table moves by an exponential average,
    ``(1 - alpha) * previous + alpha * measured``; a key the table has never
    seen takes its measurement whole. The rate depends on which way it moved:

      alpha_up    0.5, when the item got slower -- half the gap closed in one
                  run.
      alpha_down  0.1, when it got faster -- a tenth of it.

    The two differ because the scheduler's two error directions do not cost the
    same. Over-estimating an item only starts it earlier than it needed to be
    started; under-estimating one leaves it running after the other GPUs have
    gone idle, and that is what sets the makespan. So a rise is trusted quickly
    and a fall slowly.

    The slow fall also makes the table shrug off a single short run -- a crash
    partway through, or a machine having a fast day -- without anyone needing to
    decide whether that run was "valid".
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

    A key survives if item.tsv lists it.
    """
    kept, pruned = {}, []
    for key, value in weights.items():
        label = key.split("/", 1)[0]
        if key in dispatched or label not in live_labels or key in live_keys:
            kept[key] = value
        else:
            pruned.append(key)
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

    weights = read_weights(args.weights)
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
        # Every row was killed, cut short or malformed. There is nothing to
        # learn and the existing table is still the best estimate available, so
        # leave it exactly as it is rather than rewriting it from nothing.
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
            # Named individually rather than counted: a prune is how a deleted
            # test leaves the table, and it is also exactly what a broken suite
            # expansion would look like, so the list is worth reading.
            for key in sorted(pruned):
                print(f"pruned {key}: its suite no longer lists it", file=sys.stderr)

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
        "-w", "--weights", required=True, metavar="TABLE", help="weight table to read"
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
    update.add_argument(
        "timings", metavar="TIMINGS", help="timings.tsv written by run_queue_sgpu.sh"
    )
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
