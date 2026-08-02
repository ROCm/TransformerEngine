#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Build a work-item weight table for the sGPU queue scheduler.

``run_queue_sgpu.sh`` orders its queue longest-first, which needs a per-item
cost estimate. Both sources below are produced by a normal CI run, so no extra
measurement pass is required:

  --from-timing <file>  read ``timings.tsv`` written by the scheduler itself.
                        Preferred for scheduling: it is per-item wall clock, so
                        it includes process startup and CUDA/HIP init, which is
                        exactly what the makespan is made of.
  --from-junit <dir>    sum the per-testcase ``time`` attributes of every
                        ``<label>/<tag>.xml`` the run wrote. Useful when only
                        the ``logs-sgpu-*`` artifacts are available, but it
                        counts test time only -- a short item can look ~4x
                        cheaper than it really is once startup is included.

Output is ``<label>/<tag> <seconds>`` per line, plus ``<label> <seconds>`` for
opaque whole-suite items. Weights are per GPU architecture -- skip patterns
differ between gfx942 and gfx950 -- so keep one table per arch.

With ``--merge`` the output file is read first and the new measurement is
blended into it, so every CI run refines the table instead of replacing it from
a single noisy sample. See ``merge_weights`` for why the blend is asymmetric.
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict


def from_junit(root_dir):
    """Sum testcase durations for every <label>/<tag>.xml under root_dir."""
    weights = defaultdict(float)
    for label in sorted(os.listdir(root_dir)):
        subdir = os.path.join(root_dir, label)
        if not os.path.isdir(subdir):
            continue
        for name in sorted(os.listdir(subdir)):
            if not name.endswith(".xml"):
                continue
            tag = name[: -len(".xml")]
            path = os.path.join(subdir, name)
            try:
                tree = ET.parse(path)
            except ET.ParseError as err:
                # A truncated report means the run was cut off mid-item. Skipping
                # it keeps the weight unknown, which sorts the item first next
                # time -- the safe direction for something that may be slow.
                print(f"warning: skipping unparsable {path}: {err}", file=sys.stderr)
                continue
            total = 0.0
            for case in tree.getroot().iter("testcase"):
                try:
                    total += float(case.get("time") or 0.0)
                except ValueError:
                    pass
            weights[f"{label}/{tag}"] += total
    return weights


# rc values that mean the item was killed rather than measured. 124 is a GNU
# timeout expiry (the job's own step timeout, for instance), 137 a SIGKILL or an
# OOM-kill. Either way the recorded duration is a ceiling imposed from outside,
# not the item's cost, so blending it in would teach the table a number the item
# never actually took. Drop them; the previous weight stands.
KILLED_RCS = frozenset(("124", "137"))


def from_timing(path):
    """Read the scheduler's timings.tsv: label, tag, gpu, seconds, rc."""
    weights = defaultdict(float)
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 4:
                continue
            label, tag, _gpu, secs = fields[0], fields[1], fields[2], fields[3]
            if len(fields) > 4 and fields[4] in KILLED_RCS:
                continue
            try:
                value = float(secs)
            except ValueError:
                continue
            # "whole" is the scheduler's placeholder for an opaque suite item,
            # which the queue keys by bare label.
            key = label if tag == "whole" else f"{label}/{tag}"
            weights[key] += value
    return weights


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


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--from-junit", metavar="DIR", help="directory of per-suite JUnit XML subdirectories"
    )
    source.add_argument(
        "--from-timing", metavar="FILE", help="timings.tsv written by run_queue_sgpu.sh"
    )
    parser.add_argument("-o", "--output", default="-", help="output file (default: stdout)")
    parser.add_argument(
        "--merge",
        action="store_true",
        help="blend into the existing --output table instead of replacing it",
    )
    args = parser.parse_args()

    if args.from_junit:
        weights = from_junit(args.from_junit)
    else:
        weights = from_timing(args.from_timing)

    if not weights:
        print("error: no weights produced from the given source", file=sys.stderr)
        return 1

    if args.merge:
        if args.output == "-":
            parser.error("--merge needs -o/--output (the table to merge into)")
        previous = read_weights(args.output)
        weights = merge_weights(previous, weights)
        print(
            f"merged {len(weights)} weights "
            f"({len(previous)} known, {len(set(weights) - set(previous))} new)",
            file=sys.stderr,
        )

    lines = [
        f"{key} {value:.1f}\n" for key, value in sorted(weights.items(), key=lambda kv: -kv[1])
    ]
    if args.output == "-":
        sys.stdout.writelines(lines)
    else:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.writelines(lines)
        print(f"wrote {len(lines)} weights to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
