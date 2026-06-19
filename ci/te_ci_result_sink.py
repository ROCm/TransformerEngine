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

Activated only when ``TE_RESULT_SINK`` is set (done by ci/_utils.sh::pytest_run
whenever JUnit XML output is requested); a no-op otherwise.
"""

import json
import os

# Captured at import time; ci/_utils.sh sets it inline for the pytest process.
_SINK_PATH = os.environ.get("TE_RESULT_SINK") or None


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
    _append(record)


def pytest_sessionfinish(session, exitstatus):
    # Clean session end: the real JUnit XML is authoritative, drop the sidecar.
    if _SINK_PATH and os.path.exists(_SINK_PATH):
        try:
            os.remove(_SINK_PATH)
        except OSError:
            pass
