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

The plugin also carries the two halves of per-test re-running, because it is the
only place in CI that sees a pytest *nodeid*. The JUnit XML cannot stand in for
it: the default ``xunit2`` family drops the ``file``/``line`` attributes, and a
dotted ``classname`` alone cannot say where the module path ends and a class
begins.

  writing   every failing nodeid is appended to ``<junitxml>.failed``, which --
            unlike the sidecar -- survives a clean session end. That file is
            what the scheduler turns into the next attempt's work.
  reading   ``TE_CI_ONLY_TESTS`` names a file of nodeids to keep; everything
            else is deselected at collection. Narrowing here rather than by
            rewriting the pytest command line means it composes with whatever
            the suite script already passes -- ``-k`` expressions, env prefixes,
            several target paths -- and can never turn into a usage error.
            Applying it also touches ``<junitxml>.narrowed``, without which a
            re-run of two tests that fixes one would look like a whole item
            failing half its tests and lose the narrowing it had earned.

Activated only when ``TE_RESULT_SINK`` is set (done by ci/_utils.sh::pytest_run
whenever JUnit XML output is requested); a no-op otherwise.
"""

import json
import os

import pytest

# Captured at import time; ci/_utils.sh sets it inline for the pytest process.
_SINK_PATH = os.environ.get("TE_RESULT_SINK") or None

# Failing nodeids, kept after a clean end. ci/_utils.sh truncates it up front so
# what is here is always this run's.
_FAILED_PATH = os.environ.get("TE_CI_FAILED_TESTS") or None

# Nodeids this run is restricted to, one per line; set by the scheduler when it
# is re-running individual tests rather than whole items.
_ONLY_PATH = os.environ.get("TE_CI_ONLY_TESTS") or None

# Touched only if that narrowing is really applied, which is not the same as
# having been asked for -- a list that matches nothing falls back to running
# everything. It tells the next round how to read this run's test count.
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
    """Append one failing nodeid to the kept list, once per test.

    A single test can report ``failed`` in more than one phase (a call failure
    followed by a teardown error), and the list is consumed as a set of things
    to re-run, so the duplicate is dropped here rather than by the reader.
    """
    if not _FAILED_PATH or not nodeid or nodeid in _reported_failed:
        return
    _reported_failed.add(nodeid)
    try:
        with open(_FAILED_PATH, "a", encoding="utf-8") as fh:
            fh.write(nodeid + "\n")
            fh.flush()  # same reasoning as the sidecar: survive a hard exit
    except OSError:
        pass  # bookkeeping must never break the test run


def _narrow(config, items):
    """Restrict ``items`` in place to the nodeids ``TE_CI_ONLY_TESTS`` names."""
    try:
        with open(_ONLY_PATH, encoding="utf-8") as fh:
            wanted = {line.strip() for line in fh if line.strip()}
    except OSError as err:
        print(f"te_ci_result_sink: cannot read {_ONLY_PATH} ({err}); running everything")
        return

    keep = [item for item in items if item.nodeid in wanted]
    if not keep:
        # The nodeids no longer name anything here -- tests renamed, parameters
        # changed, or a -k in the suite script already excluded them. Running the
        # item whole is the safe reading; honouring an empty selection would run
        # no test at all and report success.
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
    """Narrow to the re-run set if there is one, then record what will run.

    The collected list is what makes a hard exit recoverable at test
    granularity. Everything the sidecar records after this point is a test that
    *started*; subtracting those from this list leaves the tests the dead
    process never reached, which have to be re-run alongside the failures -- they
    have no result, and treating "no result" as "passed" would let a crash
    halfway through a file report green.

    Recorded after narrowing, not before, so it names the tests this session
    will actually run. Recorded before, a crash part-way through a narrowed
    re-run would subtract the handful that finished from the whole item and hand
    the next attempt nearly every test in the file -- growing the re-run set
    instead of shrinking it, and undoing the narrowing that got it this far.
    Taking it from the narrowed list instead makes both halves of that
    subtraction subsets of what ran, so the set can only shrink.

    ``trylast`` for the same reason, one level up: pytest applies ``-k`` and
    ``-m`` from its own copy of this hook, and conftests deselect from theirs.
    Running before them records the whole file rather than the slice the item
    actually asked for -- for ci/pytorch.sh's ``-k "MXFP8BlockScaling and 126m
    and not grouped"`` that is 1615 nodeids standing in for the 78 it runs. The
    unrun tail a timeout leaves is then the entire file, which no size gate will
    ever pass, so every ``-k`` item silently loses per-test re-running. Going
    last makes the roster the real one and narrowing compose with the filters
    instead of fighting them.
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
