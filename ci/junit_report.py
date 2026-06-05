#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Summarize pytest/ctest JUnit XML results into a GitHub Actions report.

Reads every ``*.xml`` file in the given directory -- each produced by a single
pytest file invocation (see ``get_pytest_junitxml`` in ``_utils.sh``) or a
ctest run -- aggregates pass/fail/error/skip/timeout counts, writes a Markdown
digest to ``$GITHUB_STEP_SUMMARY`` (or stdout when run locally), and emits
``::error::`` workflow annotations for the failing tests.

Design notes:
  * Standard library only, so it runs on any runner without provisioning.
  * Purely informational -- it always exits 0 and never gates the job. The
    pass/fail gate stays with the existing ``FAIL_*`` markers / suite exit
    codes. This keeps the change strictly additive.
  * A run that is cut off mid-way (hang/crash/job-timeout) still produces a
    digest for every test file that finished. Files whose XML is missing or
    truncated are surfaced explicitly as "incomplete" rather than silently
    dropped, which is exactly the signal that is invisible today.
"""

import glob
import os
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict

# UI shows at most ~10 annotations of each level; cap to keep the log readable.
ANNOTATION_CAP = 20


def iter_testsuites(root):
    """Yield every <testsuite>; handles both <testsuites> and bare roots."""
    if root.tag == "testsuite":
        yield root
    else:
        yield from root.iter("testsuite")


def _system_out(testcase):
    el = testcase.find("system-out")
    return (el.text or "").strip() if el is not None else ""


def classify(testcase):
    """Return (status, first-line-message) for a <testcase>.

    status is one of: passed, failed, error, skipped.

    Handles both pytest XML (results marked with child <failure>/<error>/
    <skipped> tags) and ctest's ``--output-junit`` XML, which instead carries a
    ``status`` attribute (e.g. "fail"/"notrun") and may omit the child tag.
    """
    for tag, status in (("failure", "failed"), ("error", "error"), ("skipped", "skipped")):
        el = testcase.find(tag)
        if el is not None:
            msg = (el.get("message") or el.text or "").strip()
            return status, msg

    # No child marker: fall back to the ctest `status` attribute.
    status_attr = (testcase.get("status") or "").strip().lower()
    if status_attr in ("fail", "failed", "error"):
        return "failed", _system_out(testcase)
    if status_attr in ("notrun", "disabled", "skipped"):
        return "skipped", ""
    return "passed", ""


def is_timeout(testcase, message):
    """True if a failed/errored <testcase> failed due to a timeout.

    Scans the failure message plus the failure `type`, the testcase `status`
    attribute, and captured stdout. pytest-timeout puts "Timeout" in the
    message, but ctest records a timed-out test via those other places instead.
    """
    parts = [message, testcase.get("status", "")]
    for tag in ("failure", "error"):
        el = testcase.find(tag)
        if el is not None:
            parts.append(el.get("type", ""))
    parts.append(_system_out(testcase))
    blob = " ".join(p for p in parts if p).lower()
    return "timeout" in blob or "timed out" in blob


def emit(lines):
    """Append the report to the step summary if available, else stdout."""
    text = "\n".join(lines) + "\n"
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(text)
    else:
        sys.stdout.write(text)


def main():
    if len(sys.argv) < 2:
        print("usage: junit_report.py <results-dir> [--title TITLE]", file=sys.stderr)
        return 0
    results_dir = sys.argv[1]
    title = "Test Results"
    if "--title" in sys.argv:
        title = sys.argv[sys.argv.index("--title") + 1]

    xml_files = sorted(glob.glob(os.path.join(results_dir, "*.xml")))

    lines = []
    lines.append(f"## {title}\n")

    if not xml_files:
        lines.append(
            "> :warning: **No JUnit XML files were produced.** No test file "
            "completed far enough to write results -- the run likely crashed or "
            "hung before any suite finished. Inspect the uploaded `*.log` "
            "artifacts to see where it stopped.\n"
        )
        emit(lines)
        return 0

    totals = defaultdict(float)  # passed/failed/error/skipped/timeout/incomplete/time
    per_file = []                # (name, counts, time)
    failures = []                # (file, testid, label, message)

    for xf in xml_files:
        name = os.path.basename(xf)[: -len(".xml")]
        counts = defaultdict(int)
        suite_time = 0.0

        try:
            root = ET.parse(xf).getroot()
        except (ET.ParseError, OSError) as exc:
            # Truncated/unreadable XML => the pytest process was killed while
            # writing it (hard timeout, segfault, or job cancellation).
            counts["incomplete"] += 1
            totals["incomplete"] += 1
            per_file.append((name, counts, 0.0))
            failures.append((name, "(whole file)", "incomplete",
                             f"unparseable/truncated XML: {exc}"))
            continue

        for ts in iter_testsuites(root):
            try:
                suite_time += float(ts.get("time") or 0.0)
            except ValueError:
                pass
            for tc in ts.findall("testcase"):
                status, msg = classify(tc)
                counts[status] += 1
                totals[status] += 1
                if status in ("failed", "error"):
                    label = status
                    if is_timeout(tc, msg):
                        label = "timeout"
                        totals["timeout"] += 1
                    cls = tc.get("classname", "")
                    tcname = tc.get("name", "")
                    testid = f"{cls}::{tcname}" if cls else tcname
                    failures.append((name, testid, label,
                                     msg.splitlines()[0] if msg else ""))

        totals["time"] += suite_time
        per_file.append((name, counts, suite_time))

    n_pass = int(totals["passed"])
    n_fail = int(totals["failed"])
    n_err = int(totals["error"])
    n_skip = int(totals["skipped"])
    n_to = int(totals["timeout"])
    n_incomplete = int(totals["incomplete"])
    total_tests = n_pass + n_fail + n_err + n_skip

    ok = (n_fail + n_err + n_incomplete) == 0
    headline = ":white_check_mark:" if ok else ":x:"
    summary_line = (
        f"{headline} **{total_tests} tests** -- {n_pass} passed, {n_fail} failed, "
        f"{n_err} errored, {n_skip} skipped"
    )
    if n_to:
        summary_line += f" ({n_to} timed out)"
    if n_incomplete:
        summary_line += f"; **{n_incomplete} file(s) incomplete**"
    summary_line += f" -- {totals['time']:.0f}s across {len(xml_files)} files\n"
    lines.append(summary_line)

    # Per-file breakdown (collapsed to keep the summary scannable).
    lines.append("<details><summary>Per-file breakdown</summary>\n")
    lines.append("| Test file (backend.label) | Pass | Fail | Error | Skip | Time (s) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name, counts, t in per_file:
        bad = counts["failed"] + counts["error"] + counts["incomplete"]
        mark = "" if bad == 0 else " :warning:"
        lines.append(
            f"| {name}{mark} | {counts['passed']} | {counts['failed']} | "
            f"{counts['error']} | {counts['skipped']} | {t:.0f} |"
        )
    lines.append("\n</details>\n")

    # Failure / error / timeout detail (always expanded -- this is the payload).
    if failures:
        lines.append("### Failures / errors / timeouts\n")
        for fname, testid, label, msg in failures:
            entry = f"- **[{label}]** `{testid}` _(in {fname})_"
            if msg:
                entry += f" -- {msg}"
            lines.append(entry)
        lines.append("")

    emit(lines)

    # Inline workflow annotations.
    for i, (fname, testid, label, msg) in enumerate(failures):
        if i >= ANNOTATION_CAP:
            print(
                f"::warning::{len(failures) - ANNOTATION_CAP} more failures "
                "omitted from annotations; see the job summary for the full list."
            )
            break
        body = msg or label
        print(f"::error title={label}: {testid}::{body}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
