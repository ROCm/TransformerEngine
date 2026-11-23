# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Incremental pytest result sink for crash/hang-resilient CI reporting.

pytest writes its ``--junitxml`` report only once, at session end. A per-test
``--timeout-method=thread`` firing, a segfault, or an OOM-kill calls
``os._exit`` (or is SIGKILLed) and skips that finalization, so the JUnit XML for
the whole file is lost -- including tests that already passed. junit_report.py
then sees no XML for the file and the job summary shows green despite the crash.

This plugin streams each test's progress to a sidecar NDJSON file (one JSON
object per line, flushed immediately) named ``<junitxml>.partial``. If the
process dies mid-file the sidecar survives, and junit_report.py reconstructs the
results from it: every completed outcome is preserved and the test that was in
flight when the process died is surfaced as a timeout/crash.

On a clean session end the sidecar is deleted (the real JUnit XML is
authoritative), so a leftover ``.partial`` is itself the signal that the run did
not finish.

The plugin also lets a re-run of a failed CI job run only the failed tests. It
is the only place that sees pytest nodeids (the JUnit XML does not record them),
so it both records and applies the re-run list:

  * failing nodeids are appended to ``<junitxml>.failed``, which, unlike the
    sidecar, is kept after a clean session end for the scheduler to read;
  * ``TE_CI_ONLY_TESTS`` names a file of nodeids to run; the rest are
    deselected at collection, which works with any ``-k`` the suite passes.

Activated only when ``TE_RESULT_SINK`` is set (done by ci/_utils.sh::pytest_run
whenever JUnit XML output is requested); a no-op otherwise.
"""

import json
import os

import pytest

# Captured at import time; ci/_utils.sh sets it inline for the pytest process.
_SINK_PATH = os.environ.get("TE_RESULT_SINK") or None

# Failing nodeids of this run; ci/_utils.sh empties it before the run starts.
_FAILED_PATH = os.environ.get("TE_CI_FAILED_TESTS") or None

# Nodeids to run, one per line; set by the scheduler on a re-run.
_ONLY_PATH = os.environ.get("TE_CI_ONLY_TESTS") or None

# Written only when the run was actually narrowed, so the scheduler knows the
# test count in the JUnit XML is the re-run's and not the whole file's.
_NARROWED_PATH = os.environ.get("TE_CI_NARROWED_MARK") or None

_reported_failed = set()


def _append(record):
    """Append one NDJSON record and flush so it survives a hard process exit."""
    if not _SINK_PATH:
        return
    try:
        with open(_SINK_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
            fh.flush()  # push to the OS so os._exit()/SIGKILL still leaves it on disk
    except OSError:
        pass  # bookkeeping must never break the test run


def _record_failed(nodeid):
    """Append a failing nodeid once, even if it fails in both call and teardown."""
    if not _FAILED_PATH or not nodeid or nodeid in _reported_failed:
        return
    _reported_failed.add(nodeid)
    try:
        with open(_FAILED_PATH, "a", encoding="utf-8") as fh:
            fh.write(nodeid + "\n")
            fh.flush()  # survive a hard exit, as for the sidecar
    except OSError:
        pass  # bookkeeping must never break the test run


def _narrow(config, items):
    """Keep only the nodeids ``TE_CI_ONLY_TESTS`` names and deselect the rest."""
    try:
        with open(_ONLY_PATH, encoding="utf-8") as fh:
            wanted = {line.strip() for line in fh if line.strip()}
    except OSError as err:
        print(f"te_ci_result_sink: cannot read {_ONLY_PATH} ({err}); running everything")
        return

    keep = [item for item in items if item.nodeid in wanted]
    if not keep:
        # None of them was collected (tests renamed, or excluded by -k). Run
        # everything: an empty selection would run nothing and report a pass.
        print(
            f"te_ci_result_sink: none of the {len(wanted)} nodeids in "
            f"{os.path.basename(_ONLY_PATH)} were collected; running everything"
        )
        return

    deselected = [item for item in items if item.nodeid not in wanted]
    items[:] = keep
    if deselected:
        config.hook.pytest_deselected(items=deselected)
    if _NARROWED_PATH:
        try:
            with open(_NARROWED_PATH, "w", encoding="utf-8") as fh:
                fh.write(f"{len(keep)}\n")
        except OSError:
            pass  # bookkeeping must never break the test run
    print(f"te_ci_result_sink: re-running {len(keep)} previously failing test(s) of this item")


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    """Apply the re-run list, if any, then record the nodeids this session will run.

    If the process dies, the tests in this list that never started have no
    result and are re-run along with the failures. ``trylast`` so the list is
    taken after ``-k``/``-m`` and conftest deselection, and after narrowing;
    earlier, it would name the whole file instead of what actually runs.
    """
    if _ONLY_PATH:
        _narrow(config, items)
    _append({"e": "collected", "nodeids": [item.nodeid for item in items]})


def pytest_runtest_logstart(nodeid, location):
    _append({"e": "start", "nodeid": nodeid})


def pytest_runtest_logreport(report):
    record = {
        "e": "report",
        "nodeid": report.nodeid,
        "when": report.when,
        "outcome": report.outcome,
    }
    if report.outcome == "failed" and report.longrepr is not None:
        # First line of the failure repr -- enough to identify it in the digest.
        first = str(report.longrepr).strip().splitlines()
        if first:
            record["msg"] = first[0][:500]
    if report.outcome == "failed":
        _record_failed(report.nodeid)
    _append(record)


def pytest_sessionfinish(session, exitstatus):
    # Clean session end: the real JUnit XML is authoritative, drop the sidecar.
    if _SINK_PATH and os.path.exists(_SINK_PATH):
        try:
            os.remove(_SINK_PATH)
        except OSError:
            pass
