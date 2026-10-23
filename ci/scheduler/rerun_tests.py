#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Work out which individual tests a failed work item should re-run.

Reads ``label<TAB>tag`` lines on stdin -- the items that failed -- and writes one
``<label>.<tag>.txt`` of pytest nodeids per item that can be narrowed. An item
with no file runs whole on the next attempt, which is the answer whenever the
records are missing, unreadable, or too incomplete to trust: narrowing wrongly
would report a pass for tests nobody ran, and running whole only costs time.

Three files per item, all written by ci/te_ci_result_sink.py:

  <tag>.xml.failed   every nodeid that reported ``failed``. Kept on a clean end.
  <tag>.xml.partial  NDJSON of the session's progress, including the collected
                     nodeid list. Deleted on a clean end, so its presence means
                     the process died -- a per-test timeout, a segfault, an OOM
                     kill -- part-way through.
  <tag>.xml.narrowed present when this run was already restricted to a nodeid
                     list, which changes what its test count means.

The second is what makes a timeout safe to narrow. When a process dies mid-file
the tests after it never ran and have no result at all, so the re-run set is the
failures *plus* everything collected that never reached a terminal phase. That
covers both the test that was hung when the process died and every test queued
behind it.
"""

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET

# Above this, narrowing stops paying: the saving over just running the item
# shrinks while the chance that the failures share one broken fixture -- and so
# want the whole file re-run anyway -- grows. Half, rather than something
# tighter, is what makes a timeout worth narrowing at all. The tail a dead
# process leaves behind is the whole rest of the item, so a stricter gate needs
# the crash to land in the last handful of tests and sends very nearly every
# real one back to running the item whole.
MAX_FRACTION = 1.0 / 2.0


def read_failed(path):
    """Read the kept list of failing nodeids. Missing file means none."""
    try:
        with open(path, encoding="utf-8") as handle:
            return {line.strip() for line in handle if line.strip()}
    except OSError:
        return set()


def read_partial(path):
    """Read a sidecar into ``(collected, done)``, or ``(None, None)``.

    ``collected`` is every nodeid the session collected and ``done`` every one
    that reached a terminal phase. The phase rule matches
    junit_report.py::reconstruct_from_partial -- a ``call`` report of any
    outcome, or a setup that failed or skipped, ends a test; anything else means
    it was still in flight.

    ``(None, None)`` says the sidecar cannot answer the question -- unreadable,
    or written before the collected list was recorded -- and the caller must not
    narrow.
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
    """How many tests the item's JUnit XML says it ran, or ``None``.

    This is the only thing the re-run gets sized against, so ``None`` is not a
    reason to skip the check -- it is a reason to refuse. Narrowing to a set
    nothing was measured against is how a re-run quietly turns into the whole
    item wearing a nodeid list.
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
        # Everything collected that never finished: the test that was running
        # when the process died, and all the ones behind it that never started.
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
        # This run was itself a narrowed one, so its test count is the last
        # attempt's re-run set and not the size of the item. Asking what fraction
        # of it failed would answer a different question, and answer it wrongly:
        # a two-test re-run that fixes one test fails "half" of them and would be
        # sent back to running the whole file, undoing the narrowing every second
        # attempt. Nothing bounds the set here because nothing needs to -- a
        # narrowed run collects only its own nodeids, so the failures and the
        # unrun tail are both subsets of what it was given and the set can only
        # shrink from one attempt to the next.
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
        # No XML was requested, so nothing recorded nodeids. Every failed item
        # re-runs whole, exactly as it did before per-test re-running existed.
        return 0
    os.makedirs(args.out, exist_ok=True)

    narrowed = 0
    for line in sys.stdin:
        label, _, tag = line.rstrip("\n").partition("\t")
        if not label or not tag:
            # An opaque whole-suite item has no tag and no pytest nodeids to
            # narrow to -- core and examples always re-run whole.
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
