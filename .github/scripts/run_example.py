#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

"""Run one example and record its outcome as a one-test JUnit file.

Usage: run_example.py <name> <command> [args...]

The command's output passes through and the script exits with its exit code.
With JUNITXML_PREFIX/JUNITXML_SUFFIX set (ci/run_queue.sh sets them per suite),
the outcome is written to ``<prefix><name><suffix>``, which gives the examples
their own section in the job summary.
"""

import collections
import json
import os
import re
import shlex
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Output kept in a failed example's XML; the item log has all of it.
TAIL_LINES = 100

# Colour codes clutter the summary, and control characters XML 1.0 cannot hold
# would make the file unparseable
_ANSI_ESCAPE = re.compile("\x1b\\[[0-9;?]*[A-Za-z]")
_XML_ILLEGAL = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f]")


def clean(text):
    """``text`` without colour codes or the characters XML cannot carry."""
    return _XML_ILLEGAL.sub("", _ANSI_ESCAPE.sub("", text))


def run(cmd):
    """Run ``cmd``, echoing its output; return its exit code and last lines."""
    tail = collections.deque(maxlen=TAIL_LINES)
    try:
        # Unbuffered, so an example's prints come before its traceback
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=dict(os.environ, PYTHONUNBUFFERED="1"),
        )
    except OSError as exc:
        line = f"run_example.py: cannot run {cmd[0]}: {exc}"
        print(line, flush=True)
        return 127, [line]
    for raw in proc.stdout:
        sys.stdout.buffer.write(raw)
        sys.stdout.buffer.flush()
        tail.append(raw.decode("utf-8", "replace").rstrip())
    return proc.wait(), list(tail)


def failure_message(rc, tail):
    """How the command ended and its last output line, e.g. the exception."""
    status = f"killed by signal {-rc}" if rc < 0 else f"exit {rc}"
    last = next((line.strip() for line in reversed(tail) if line.strip()), "")
    return f"{status}: {last}" if last else status


def write_junit(path, cmd, rc, secs, tail):
    """Write the outcome as a one-test JUnit file junit_report.py can read."""
    root = ET.Element("testsuites")
    suite = ET.SubElement(
        root,
        "testsuite",
        name="examples",
        tests="1",
        failures="1" if rc else "0",
        errors="0",
        skipped="0",
        time=f"{secs:.3f}",
    )
    # Shown as <dir>::<command>, e.g. examples/jax/mnist::python ...
    case = ET.SubElement(
        suite,
        "testcase",
        classname=os.path.relpath(os.getcwd(), REPO_ROOT),
        name=clean(shlex.join(cmd)),
        time=f"{secs:.3f}",
    )
    if rc:
        failure = ET.SubElement(case, "failure", message=clean(failure_message(rc, tail)))
        failure.text = clean("\n".join(tail))
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def main():
    """Run the example, record it, and exit as it did."""
    if len(sys.argv) < 3:
        print("usage: run_example.py <name> <command> [args...]", file=sys.stderr)
        return 2
    name, cmd = sys.argv[1], sys.argv[2:]

    prefix = os.environ.get("JUNITXML_PREFIX", "")
    suffix = os.environ.get("JUNITXML_SUFFIX", "")
    xml_path = f"{prefix}{name}{suffix}" if prefix or suffix else None
    if xml_path:
        os.makedirs(os.path.dirname(xml_path) or ".", exist_ok=True)
        # A result-sink sidecar, removed once the XML is written. If this process
        # dies mid-example it is left behind, so the example is reported as not
        # finished rather than missing.
        with open(xml_path + ".partial", "w", encoding="utf-8") as handle:
            handle.write(json.dumps({"e": "start", "nodeid": name}) + "\n")

    start = time.monotonic()
    rc, tail = run(cmd)
    secs = time.monotonic() - start

    if xml_path:
        write_junit(xml_path, cmd, rc, secs, tail)
        os.remove(xml_path + ".partial")
    # A signal comes back negative; report it the way a shell would.
    return 128 - rc if rc < 0 else rc


if __name__ == "__main__":
    sys.exit(main())
