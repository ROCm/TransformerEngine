# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
"""M1/M2 accounting and manifest governance check (plan S3.1, proposal sec 6.3). CI-runnable.

  M1 (physical overlap debt)   in-place divergence still carried against the pin: the added+
                               removed lines of every entry whose divergence has NOT terminally
                               retired (upstream-merged / relocated / deleted).
  M2 (carried compatibility)   what the plugin model actively maintains instead: active build
                               patches, runtime overrides, overlay skips/xfails, unsupported
                               contract items. Creating a patch closes M1 and opens M2; only
                               true retirement reduces M2.

Exits non-zero if governance is broken: a patch on disk without a live manifest entry, a live
p5-needed entry without a patch, a '# tests: TBD' header, or a REGENERATE marker left anywhere.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import yaml

PROP = Path(__file__).resolve().parents[1]
MANIFEST = PROP / "divergence-manifest.yaml"
PATCHES = PROP / "patches"


def main() -> int:
    d = yaml.safe_load(MANIFEST.read_text())
    entries = {e["id"]: e for e in d["entries"]}
    active = {p.stem: p for p in PATCHES.glob("*.patch")}
    retired_files = {p.stem for p in (PATCHES / "retired").glob("*.patch")} if (PATCHES / "retired").exists() else set()
    problems = []

    # --- governance invariants -------------------------------------------------------------
    for pid, p in active.items():
        if pid not in entries:
            problems.append(f"patch {pid} has no manifest entry")
        elif str(entries[pid].get("p5_status", "")).startswith("retired"):
            problems.append(f"patch {pid} active but manifest says retired")
        if re.search(r"^# tests: TBD", p.read_text(), re.M):
            problems.append(f"patch {pid}: '# tests: TBD'")
    for eid, e in entries.items():
        st = str(e.get("p5_status", ""))
        if st.startswith("needed") and eid not in active:
            problems.append(f"{eid} is p5-needed but has no active patch")
        if st.startswith("retired") and eid not in retired_files:
            problems.append(f"{eid} is p5-retired but not in patches/retired/")
    text = MANIFEST.read_text()
    if re.search(r":\s*REGENERATE\s*$", text, re.M):     # value position only; prose 'REGENERATED' is fine
        problems.append("manifest still contains 'added_class: REGENERATE' values")
    pending = len(re.findall(r"PENDING_RECLASSIFICATION", text))

    # --- M1 / M2 ----------------------------------------------------------------------------
    def lines(e):
        return int(e.get("added_lines", 0) or 0) + int(e.get("removed_lines", 0) or 0)

    m1_entries = [e for eid, e in entries.items()
                  if "added_lines" in e and not str(e.get("p5_status", "")).startswith("retired")]
    m1 = sum(lines(e) for e in m1_entries)
    m1_retired = sum(lines(e) for eid, e in entries.items()
                     if "added_lines" in e and str(e.get("p5_status", "")).startswith("retired"))

    overrides = 0  # runtime-override tier: nothing proven yet (S3.2)
    unsupported = sum(1 for e in entries.values() if e.get("support_state") == "unsupported-on-rocm")
    m2 = {"active_patches": len(active), "runtime_overrides": overrides,
          "overlay_skips_xfails": "n/a until TST-001 overlay exists",
          "unsupported_contract_items": unsupported}

    counts = {}
    for e in entries.values():
        counts[str(e.get("p5_status", "untracked"))] = counts.get(str(e.get("p5_status", "untracked")), 0) + 1

    print(f"M1 physical overlap debt : {m1} lines across {len(m1_entries)} live entries "
          f"({m1_retired} lines retired so far)")
    print(f"M2 carried compatibility : {json.dumps(m2)}")
    print(f"p5 status                : {json.dumps(counts, sort_keys=True)}")
    print(f"queue                    : {len(active)} active, {len(retired_files)} retired")
    if pending:
        print(f"outstanding (Stage 0)    : {pending} feature-level m1 attributions PENDING_RECLASSIFICATION (not a failure; blocks Stage-0 exit claim)")
    if problems:
        print("\nGOVERNANCE PROBLEMS:")
        for p in problems:
            print("  -", p)
        return 1
    print("governance: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
