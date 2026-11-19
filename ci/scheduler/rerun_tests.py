#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Work out which individual tests each failed work item should re-run.

Reads ``label<TAB>tag`` lines on stdin (the failed items) and writes a
``<label>.<tag>.txt`` of pytest nodeids for each item that can be narrowed. An
item without a file re-runs whole, which is the answer whenever the records are
missing or cannot be trusted.

Per item it reads three files written by ci/te_ci_result_sink.py:

  <tag>.xml.failed    nodeids that failed
  <tag>.xml.partial   the session's progress, including the collected nodeids;
                      present only if the process died mid-run
  <tag>.xml.narrowed  present if this run was itself a narrowed re-run

If the process died, the tests that never finished have no result, so they are
re-run along with the failures.
"""

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET

# Narrow only if at most this fraction of the item needs re-running; beyond it,
# running it whole costs little more. Half, so that a timeout part-way through a
# file, which leaves the rest of it unrun, can still be narrowed.
MAX_FRACTION = 1.0 / 2.0


def read_failed(path):
    """Read the kept list of failing nodeids. Missing file means none."""
    try:
        with open(path, encoding="utf-8") as handle:
            return {line.strip() for line in handle if line.strip()}
    except OSError:
        return set()


def read_partial(path):
    """Read a sidecar into ``(collected, done)``, or ``(None, None)`` if it cannot tell.

    ``done`` holds the nodeids that finished: a ``call`` report of any outcome, or a
    setup that failed or skipped (the same rule as junit_report.py).
    """
    collected, phases = None, {}
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue  # tolerate a half-written final line
                if rec.get("e") == "collected":
                    collected = [n for n in rec.get("nodeids", []) if n]
                elif rec.get("e") == "report":
                    phases.setdefault(rec.get("nodeid", ""), {})[rec.get("when", "")] = rec.get(
                        "outcome", ""
                    )
    except OSError:
        return None, None
    if collected is None:
        return None, None

    done = set()
    for nodeid, when_outcome in phases.items():
        if "call" in when_outcome or when_outcome.get("setup") in ("failed", "skipped"):
            done.add(nodeid)
    return collected, done


def count_collected(xml_path):
    """How many tests the item's JUnit XML says it ran, or ``None`` if unknown.

    The re-run is sized against this, so ``None`` means do not narrow.
    """
    try:
        root = ET.parse(xml_path).getroot()
    except (ET.ParseError, OSError):
        return None
    total = 0
    for suite in root.iter("testsuite"):
        try:
            total += int(suite.get("tests") or 0)
        except ValueError:
            pass
    return total or None


def rerun_set(junit_base):
    """The nodeids one item should re-run, or ``(None, why)`` to run it whole."""
    failed = read_failed(junit_base + ".failed")
    partial = junit_base + ".partial"

    if os.path.exists(partial):
        collected, done = read_partial(partial)
        if collected is None:
            return None, "it died before recording what it had collected"
        # Tests that never finished: the one running when the process died and
        # every one after it
        unrun = [nodeid for nodeid in collected if nodeid not in done]
        tests = failed | set(unrun)
        total = len(collected)
    else:
        tests = failed
        total = count_collected(junit_base)

    if not tests:
        # Nothing named a test: a collection or usage error, a crash before the
        # first test, or a suite that failed for a reason outside pytest.
        return None, "nothing in it identified a failing test"
    if os.path.exists(junit_base + ".narrowed"):
        # A narrowed run's test count is the previous re-run set, not the item's
        # size, so the fraction check does not apply. The set can only shrink.
        return sorted(tests), None
    if not total:
        return None, "its report did not say how many tests it has to size the re-run against"
    if len(tests) > total * MAX_FRACTION:
        return None, f"{len(tests)} of {total} tests is too much of it to be worth narrowing"
    return sorted(tests), None


def main():
    """Write a nodeid file for each failed item that can be narrowed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--junit-prefix", default="", help="JUNITXML_PREFIX the run was dispatched with"
    )
    parser.add_argument(
        "--junit-suffix", default="", help="JUNITXML_SUFFIX the run was dispatched with"
    )
    parser.add_argument(
        "-o", "--out", required=True, metavar="DIR", help="where to write the nodeid files"
    )
    args = parser.parse_args()

    if not args.junit_prefix:
        # Without JUnit XML nothing recorded nodeids, so every item re-runs whole
        return 0
    os.makedirs(args.out, exist_ok=True)

    narrowed = 0
    for line in sys.stdin:
        label, _, tag = line.rstrip("\n").partition("\t")
        if not label or not tag:
            # An opaque suite item (examples) has no tag and re-runs whole
            continue
        base = f"{args.junit_prefix}{label}/{tag}{args.junit_suffix}"
        tests, why = rerun_set(base)
        if tests is None:
            print(f"  {label}/{tag}: re-runs whole -- {why}", file=sys.stderr)
            continue
        with open(os.path.join(args.out, f"{label}.{tag}.txt"), "w", encoding="utf-8") as handle:
            handle.write("".join(nodeid + "\n" for nodeid in tests))
        print(f"  {label}/{tag}: re-runs {len(tests)} test(s)", file=sys.stderr)
        narrowed += 1
    print(f"narrowed {narrowed} item(s) to individual tests", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
