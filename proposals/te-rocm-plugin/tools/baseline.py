# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
"""Write or compare a test/perf baseline for the seam prototype (plan P0 / P3 / P7).

  write    junit XML dir + import-time + env -> baselines/<label>.json
           Records the PASS/FAIL/SKIP *set* per test id, not just counts, so a later comparison
           can name exactly which tests changed.
  compare  two baselines -> the delta the plan's EXIT-A / EXIT-B gates ask for:
           tests that flipped (pass->fail, fail->pass, pass->skip, ...), files present in one
           side only, and import-time change vs thresholds.yaml.

Usage:
  baseline.py write   --junit <dir> --label 2026-08-30-fork --import-us 2918869 [--note ...]
  baseline.py compare --old baselines/a.json --new baselines/b.json [--thresholds thresholds.yaml]
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASELINES = HERE.parent / "baselines"


def sh(*a):
    return subprocess.run(a, capture_output=True, text=True).stdout.strip()


def env_facts() -> dict:
    facts = {
        "host": platform.node(),
        "python": platform.python_version(),
        "git_head": sh("git", "rev-parse", "HEAD"),
        "git_branch": sh("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "upstream_pin": sh("git", "-C", "3rdparty/transformer_engine_nvidia", "rev-parse", "HEAD"),
        "hip": sh("hipconfig", "--version"),
        "gpu_arch": sh("bash", "-c", "rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9a-f]+'"),
    }
    try:
        import torch  # noqa
        facts["torch"] = torch.__version__
    except Exception:
        facts["torch"] = None
    return facts


def parse_junit(d: Path) -> dict[str, dict]:
    """{file_label: {test_id: 'pass'|'fail'|'error'|'skip'}}"""
    out: dict[str, dict] = {}
    for x in sorted(d.glob("*.xml")):
        tests = {}
        for tc in ET.parse(x).getroot().iter("testcase"):
            tid = f"{tc.get('classname')}::{tc.get('name')}"
            if tc.find("failure") is not None:
                st = "fail"
            elif tc.find("error") is not None:
                st = "error"
            elif tc.find("skipped") is not None:
                st = "skip"
            else:
                st = "pass"
            tests[tid] = st
        out[x.stem] = tests
    return out


def summarize(files: dict[str, dict]) -> dict:
    tot = {"pass": 0, "fail": 0, "error": 0, "skip": 0}
    per = {}
    for f, tests in files.items():
        c = {"pass": 0, "fail": 0, "error": 0, "skip": 0}
        for st in tests.values():
            c[st] += 1
        per[f] = c
        for k in tot:
            tot[k] += c[k]
    return {"total": tot, "per_file": per}


def cmd_write(a):
    files = parse_junit(Path(a.junit))
    doc = {
        "label": a.label,
        "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": a.note,
        "env": env_facts(),
        "import_time_us": a.import_us,
        "summary": summarize(files),
        "tests": files,
    }
    BASELINES.mkdir(exist_ok=True)
    out = BASELINES / f"{a.label}.json"
    out.write_text(json.dumps(doc, indent=1, sort_keys=True))
    s = doc["summary"]["total"]
    print(f"wrote {out.relative_to(HERE.parent.parent.parent)}: "
          f"{s['pass']} pass / {s['fail']} fail / {s['error']} error / {s['skip']} skip "
          f"across {len(files)} files; import {a.import_us/1e6:.2f}s")


def cmd_compare(a):
    old = json.loads(Path(a.old).read_text())
    new = json.loads(Path(a.new).read_text())
    flips: list[tuple[str, str, str, str]] = []
    only_old, only_new = [], []
    for f in sorted(set(old["tests"]) | set(new["tests"])):
        if f not in new["tests"]:
            only_old.append(f); continue
        if f not in old["tests"]:
            only_new.append(f); continue
        o, n = old["tests"][f], new["tests"][f]
        for tid in sorted(set(o) | set(n)):
            so, sn = o.get(tid, "absent"), n.get(tid, "absent")
            if so != sn:
                flips.append((f, tid, so, sn))
    print(f"old: {old['label']}   new: {new['label']}")
    print(f"files only in old: {only_old or '-'}   only in new: {only_new or '-'}")
    print(f"test outcome flips: {len(flips)}")
    bad = [x for x in flips if x[3] in ("fail", "error") or (x[2] == "pass" and x[3] != "pass")]
    for f, tid, so, sn in flips[:60]:
        mark = "  !!" if (f, tid, so, sn) in bad else "    "
        print(f"{mark} {f}: {tid}  {so} -> {sn}")
    if len(flips) > 60:
        print(f"     ... {len(flips)-60} more")
    io, inew = old.get("import_time_us"), new.get("import_time_us")
    verdict_lines = []
    if io and inew:
        rel = (inew - io) / io
        line = f"import time: {io/1e6:.2f}s -> {inew/1e6:.2f}s ({rel:+.1%})"
        if a.thresholds:
            import yaml
            t = yaml.safe_load(Path(a.thresholds).read_text())
            lim = t["stage1_performance_budget"]["import_time_increase"]["max_relative"]
            line += f"   budget +{lim:.0%}: {'OK' if rel <= lim else 'EXCEEDED'}"
            if rel > lim:
                verdict_lines.append("import-time budget exceeded")
        print(line)
    if bad:
        verdict_lines.append(f"{len(bad)} regressions (pass->non-pass or new fail/error)")
    print("VERDICT:", "IDENTICAL/WITHIN BUDGET" if not verdict_lines else "; ".join(verdict_lines))
    sys.exit(1 if verdict_lines else 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write"); w.add_argument("--junit", required=True); w.add_argument("--label", required=True)
    w.add_argument("--import-us", type=int, required=True); w.add_argument("--note", default="")
    c = sub.add_parser("compare"); c.add_argument("--old", required=True); c.add_argument("--new", required=True)
    c.add_argument("--thresholds")
    a = ap.parse_args()
    cmd_write(a) if a.cmd == "write" else cmd_compare(a)


if __name__ == "__main__":
    main()
