#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Fold one CI run's measured timings into the sGPU scheduler's weight table.

``run_queue_sgpu.sh`` orders its queue longest-first, which needs a per-item cost
estimate. That estimate comes from the run that just finished, so no separate
measurement pass is required: the scheduler writes ``timings.tsv`` as it
dispatches, and every run refines the table that the next one reads.

Inputs and outputs:

  TIMINGS     ``timings.tsv`` written by the scheduler. Per-item wall clock, so
              it includes process startup and HIP init -- which is what the
              makespan is actually made of, and what a test-time-only source
              such as JUnit XML would understate several-fold.
  --items     ``items.tsv``, also from the scheduler: every work item that
              exists at this TEST_LEVEL, whether or not this host ran it.
  -o TABLE    the weight table, read and rewritten in place. Lines are
              ``<label>/<tag> <seconds>``, plus ``<label> <seconds>`` for opaque
              whole-suite items.

Keep one table per (GPU arch, TEST_LEVEL). Skip patterns differ between gfx942
and gfx950, and the level changes both which items run and how long several of
them take (tests/jax reads NVTE_JAX_UNITTEST_LEVEL, test_grouped_gemm.py reads
TEST_LEVEL), so that is the coarsest keying under which a weight still means one
thing. Holding them apart is the caller's job -- the workflow puts arch and level
in the file name.

Each run is blended into the table rather than replacing it, so no single noisy
sample can move a weight far; see ``merge_weights`` for why the blend is
asymmetric. Measurements that were cut short are dropped instead of blended
(``read_timings``), and items that no longer exist are dropped from the table
(``prune_weights``).
"""

import argparse
import sys
from collections import defaultdict

# rc values that mean the item was killed rather than measured. 124 is a GNU
# timeout expiry (the job's own step timeout, for instance), 137 a SIGKILL or an
# OOM-kill. Either way the recorded duration is a ceiling imposed from outside,
# not the item's cost, so blending it in would teach the table a number the item
# never actually took. Drop them; the previous weight stands.
KILLED_RCS = frozenset(("124", "137"))


def read_timings(path):
    """Read the scheduler's timings.tsv.

    Columns are label, tag, gpu, secs, rc, start_off, end_off, est, incomplete;
    only the first five and the last are read here.

    Returns ``(measured, dispatched)``. ``measured`` maps key -> seconds and
    holds only the rows worth learning from; ``dispatched`` is every key the run
    started, which is a weaker but separate fact -- an item can have run and
    still have produced no usable number, and only the second set answers "does
    this item still exist" (see ``prune_weights``).

    Two kinds of row are excluded from ``measured``, for the same reason: the
    duration records where the item was stopped, not what it costs. An rc in
    ``KILLED_RCS`` says so outright. The ``incomplete`` flag covers what rc
    cannot -- a ``--timeout-method=thread`` expiry exits 1, indistinguishable
    from an ordinary test failure (which is a perfectly good measurement), and a
    segfault exits 139. The scheduler sets that flag from the result sidecar the
    dying process left behind, which is the only reliable witness.
    """
    measured = defaultdict(float)
    dispatched = set()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 4:
                continue
            label, tag, _gpu, secs = fields[0], fields[1], fields[2], fields[3]
            # "whole" is the scheduler's placeholder for an opaque suite item,
            # which the queue keys by bare label.
            key = label if tag == "whole" else f"{label}/{tag}"
            dispatched.add(key)
            if len(fields) > 4 and fields[4] in KILLED_RCS:
                continue
            if len(fields) > 8 and fields[8] == "1":
                continue
            try:
                value = float(secs)
            except ValueError:
                continue
            measured[key] += value
    return measured, dispatched


def read_items(path):
    """Read items.tsv -- every work item that exists at this TEST_LEVEL.

    The file is ``<label>\\t<tag>`` per line, with an empty tag for an opaque
    whole-suite item. Returns ``(keys, labels)``: the item keys, and the set of
    suite labels the file speaks for.

    Both are needed because pruning has to be per label. A suite can be absent
    from the file entirely -- it was filtered out of the run, or its expansion
    came back untrustworthy -- and that absence must not be read as "none of its
    items exist any more".

    Returns ``(None, None)`` when the file cannot be read, which disables
    pruning. That is the conservative direction: a weight kept for a test that
    is gone wastes one line, while a weight dropped for a live test makes the
    scheduler treat it as unknown and mis-order a whole run.
    """
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
            keys.add(f"{label}/{tag}" if tag else label)
    return keys, labels


def read_weights(path):
    """Read an existing ``<key> <seconds>`` table. Missing file -> empty table."""
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


def merge_weights(old, new, alpha_up=0.5, alpha_down=0.1):
    """Blend a fresh measurement into the running table.

    The blend is deliberately asymmetric, because the scheduler's two error
    directions are not equally expensive. Over-estimating an item only starts it
    earlier than it needed to be started; under-estimating one leaves it running
    after the other GPUs have gone idle, and that is what sets the makespan. So
    a rise is trusted quickly (half the gap in one run) while a fall is trusted
    slowly (a tenth), which also makes the table shrug off a single short run --
    a crash partway through, or a machine having a fast day -- without needing
    to decide whether that run was "valid".
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

    A key survives if any of these holds:

      * the run dispatched it -- then it plainly exists, whatever else says, and
        that holds even when it was killed and so taught the table nothing;
      * its suite is not covered by the item list -- nothing is known about it;
      * the item list contains it.

    What is left is a key whose suite was listed in full and which that listing
    did not contain: a test that was deleted, or renamed. Renames need no
    handling at this end -- the new name simply arrives as an unknown key, which
    the scheduler already sorts first.

    Returns ``(kept, pruned)``.
    """
    kept, pruned = {}, []
    for key, value in weights.items():
        label = key.split("/", 1)[0]
        if key in dispatched or label not in live_labels or key in live_keys:
            kept[key] = value
        else:
            pruned.append(key)
    return kept, pruned


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "timings", metavar="TIMINGS", help="timings.tsv written by run_queue_sgpu.sh"
    )
    parser.add_argument(
        "-o", "--output", required=True, metavar="TABLE", help="weight table to update in place"
    )
    parser.add_argument(
        "--items",
        metavar="FILE",
        help="items.tsv listing every work item that exists at this level; "
        "without it, no weight is ever pruned",
    )
    args = parser.parse_args()

    measured, dispatched = read_timings(args.timings)
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


if __name__ == "__main__":
    sys.exit(main())
