# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
"""Stage 2 historical backtest (plan Track B): replay the 2.15 -> 2.17 sync as a patch queue.

A counterfactual, hindsight-advantaged experiment - NOT equivalent to a live pin bump. It answers
one falsifiable question: had the fork's divergence at the 2.15 sync been expressed as the
governed patch queue, what fraction would have REAPPLIED across the real upstream delta, and what
would have TRIPPED loudly by ID?

Historical inputs (derived, then verified at runtime - never hand-entered release tips):
  PRE   bc3766e6d  fork dev immediately before the sync (first parent of the sync's
                   auto-merge basis 2f9490ab7 on release-sync-v2.15-260630)
  BASE  42b840051  what that sync ACTUALLY merged = second parent of 2f9490ab7. It is the tip of
                   upstream release_v2.15 - the pre-policy era merged release branches; the replay
                   uses recorded history, not the corrected two-track policy.
  TGT   2e559f062  what the NEXT sync merged (second parent of the 2.17 IFU merge; on
                   release_v2.17, not main - again, recorded history)

Subcommands
  gen     write backtest patches: one per Python CASE (the manifest's backtest_plan.cases,
          re-derived at the 2.15 file location as diff(BASE:file, PRE:file)) and one per C++ ARM
          file (the guard edits, same derivation). Output: backtest/patches/.
  check   apply each patch with `git apply --check` against the TGT tree -> reapplied | TRIPPED,
          per ID. Also a 3-way `git merge-file` (base=BASE, ours=TGT, theirs=PRE) per tripped
          patch: conflict-hunk count is the repair-size proxy, and a clean 3-way merge means the
          trip is trivially repairable.
  report  outcomes vs thresholds.yaml (stage2_backtest + cxx_arm_reporting_bands) -> B3 packet.

Limitations, stated: 'silently wrong behavior' cannot be fully tested without building and
testing the 2.15-era tree; repair effort here is conflict-hunk counts and this session's repair
wall-clock, not the historical engineer-hours (mid-July -> Aug 22, recorded separately).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
PROP = HERE.parent
ROOT = PROP.parent.parent
OUT = PROP / "backtest"
PATCHES = OUT / "patches"

PRE = "bc3766e6da7f6a478c817e0c9934405b9fdbe039"
SYNC_MERGE = "2f9490ab7"           # auto-merge basis for release-sync-v2.15-260630
BASE = "42b840051647eef89761a16dfdff87e82bb253ab"
TGT = "2e559f062497bef768dfbe9d7e45548fadeca80a"

# The manifest's 12 risk-role cases, re-derived at their 2.15-era file (verified: every path
# below exists at PRE, BASE and TGT). ABI-001 rides inside CM-002's file; ABI-002-FAENUM is the
# pybind_helper.h C++ file, listed in the C++ arm as well by design.
CASES = {
    "BOOT-001+CM-001": "transformer_engine/__init__.py",
    "CM-002+ABI-001": "transformer_engine/common/__init__.py",
    "CM-003": "transformer_engine/common/recipe/__init__.py",
    "PT-002-CAP": "transformer_engine/pytorch/module/base.py",
    "PT-010-SEL": "transformer_engine/pytorch/attention/dot_product_attention/backends.py",
    "PT-006-UB": "transformer_engine/pytorch/module/layernorm_mlp.py",
    "PT-028": "transformer_engine/pytorch/graph.py",
    "HDR-A": "transformer_engine/common/include/transformer_engine/transformer_engine.h",
    "ABI-002-FAENUM": "transformer_engine/common/util/pybind_helper.h",
    "PT-004-MXFP4": "transformer_engine/pytorch/quantization.py",
    "PT-008-NVFP4FIX": "transformer_engine/pytorch/module/linear.py",
    "PT-032": "transformer_engine/pytorch/__init__.py",
}

# C++ arm (manifest backtest_plan.cxx_arm selection rule):
#   (a) 3 highest guard-density files at PRE, (b) 3 highest BASE->TGT churn existing on both
#   sides, (c) 2 single-guard controls (picked at gen time), (d) pybind_helper.h (in CASES),
#   (e) one public header (in CASES as HDR-A).
CXX_ARM = [
    "transformer_engine/common/hadamard_transform/hadamard_transform.cu",
    "transformer_engine/common/transpose/quantize_transpose_vector_blockwise_fp4.cu",
    "transformer_engine/common/gemm/cublaslt_gemm.cu",
    "transformer_engine/common/gemm/cublaslt_grouped_gemm.cu",
    "transformer_engine/common/fused_router/fused_topk_with_score_function.cu",
    "transformer_engine/common/fused_router/fused_score_for_moe_aux_loss.cu",
]


def sh(*a, **kw):
    return subprocess.run(a, capture_output=True, text=True, cwd=str(ROOT), **kw)


def show(sha, rel) -> bytes | None:
    r = sh("git", "show", f"{sha}:{rel}")
    return r.stdout.encode() if r.returncode == 0 else None


def verify_inputs():
    p = sh("git", "log", "-1", "--format=%P", SYNC_MERGE).stdout.split()
    assert p and p[0].startswith(PRE[:9]) and p[1].startswith(BASE[:9]), \
        f"derivation broken: {SYNC_MERGE} parents {p}"
    assert sh("git", "merge-base", "--is-ancestor", TGT, "refs/nvidia/release_v2.17").returncode == 0
    print(f"inputs: PRE={PRE[:9]} BASE={BASE[:9]} (release_v2.15 tip, as history merged) TGT={TGT[:9]}")


def single_guard_controls() -> list[str]:
    r = sh("git", "grep", "-c", "-e", "__HIP_PLATFORM_AMD__", "--or", "-e", "USE_ROCM", PRE,
           "--", "transformer_engine/common/**/*.cu", "transformer_engine/common/**/*.cpp")
    ones = [l.split(":", 1)[1].rsplit(":", 1)[0] for l in r.stdout.splitlines() if l.endswith(":1")]
    picked = [f for f in ones if show(BASE, f) and show(TGT, f)][:2]
    return picked


def gen():
    verify_inputs()
    PATCHES.mkdir(parents=True, exist_ok=True)
    cxx = CXX_ARM + single_guard_controls()
    items = [("case", cid, rel) for cid, rel in CASES.items()] + \
            [("cxx", f"CXX-{Path(rel).stem}", rel) for rel in cxx]
    meta = {}
    for kind, pid, rel in items:
        b, p = show(BASE, rel), show(PRE, rel)
        if b is None or p is None:
            print(f"  {pid}: SKIP ({'no BASE' if b is None else 'no PRE'} file) {rel}"); continue
        with tempfile.NamedTemporaryFile(delete=False) as fb, tempfile.NamedTemporaryFile(delete=False) as fp:
            fb.write(b); fp.write(p)
        r = subprocess.run(["diff", "-u", "--label", f"a/{rel}", "--label", f"b/{rel}", fb.name, fp.name],
                           capture_output=True, text=True)
        Path(fb.name).unlink(); Path(fp.name).unlink()
        if r.returncode == 0:
            meta[pid] = {"kind": kind, "rel": rel, "status": "no-divergence"}; print(f"  {pid}: no divergence at 2.15"); continue
        hdr = f"# backtest-case: {pid}\n# kind: {kind}\n# base: {BASE}\n# derived: diff(BASE:{rel}, PRE:{rel})\n"
        (PATCHES / f"{pid}.patch").write_text(hdr + r.stdout)
        add = sum(1 for l in r.stdout.splitlines() if l.startswith("+") and not l.startswith("+++"))
        rem = sum(1 for l in r.stdout.splitlines() if l.startswith("-") and not l.startswith("---"))
        meta[pid] = {"kind": kind, "rel": rel, "status": "generated", "added": add, "removed": rem}
        print(f"  {pid}: +{add}/-{rem}  {rel}")
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))


def check():
    verify_inputs()
    meta = json.loads((OUT / "meta.json").read_text())
    tree = Path(tempfile.mkdtemp(prefix="backtest-tgt-"))
    for pid, m in meta.items():
        if m["status"] != "generated":
            continue
        t = show(TGT, m["rel"])
        if t is None:
            m["outcome"] = "TRIPPED(file-gone-at-target)"; continue
        dst = tree / m["rel"]; dst.parent.mkdir(parents=True, exist_ok=True); dst.write_bytes(t)
        r = subprocess.run(["git", "apply", "--check", "-p1", str(PATCHES / f"{pid}.patch")],
                           capture_output=True, text=True, cwd=tree)
        if r.returncode == 0:
            m["outcome"] = "reapplied"
        else:
            m["outcome"] = "TRIPPED"
            # 3-way repair proxy: base=BASE, ours=TGT, theirs=PRE
            with tempfile.NamedTemporaryFile(delete=False) as fb, tempfile.NamedTemporaryFile(delete=False) as fo, tempfile.NamedTemporaryFile(delete=False) as ft:
                fb.write(show(BASE, m["rel"])); fo.write(t); ft.write(show(PRE, m["rel"]))
            t0 = time.time()
            mr = subprocess.run(["git", "merge-file", "-p", fo.name, fb.name, ft.name], capture_output=True, text=True)
            m["merge3"] = {"clean": mr.returncode == 0,
                           "conflict_hunks": mr.stdout.count("<<<<<<<"),
                           "secs": round(time.time() - t0, 2)}
            for f in (fb.name, fo.name, ft.name): Path(f).unlink()
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    for pid, m in sorted(meta.items(), key=lambda kv: (kv[1]["kind"], kv[0])):
        extra = ""
        if m.get("merge3"):
            extra = f"  3-way: {'CLEAN' if m['merge3']['clean'] else str(m['merge3']['conflict_hunks']) + ' conflict hunks'}"
        print(f"  {m['kind']:4s} {pid:18s} {m.get('outcome', m['status']):28s} +{m.get('added','-')}/-{m.get('removed','-')}{extra}")


def report():
    meta = json.loads((OUT / "meta.json").read_text())
    th = yaml.safe_load((PROP / "thresholds.yaml").read_text())
    cases = {k: v for k, v in meta.items() if v["kind"] == "case" and v["status"] == "generated"}
    cxx = {k: v for k, v in meta.items() if v["kind"] == "cxx" and v["status"] == "generated"}
    silent = [k for k, v in meta.items() if v["status"] == "generated" and "outcome" not in v]
    n_re = sum(1 for v in cases.values() if v["outcome"] == "reapplied")
    n_tr = len(cases) - n_re
    cxx_tr = sum(1 for v in cxx.values() if v["outcome"] != "reapplied")
    # ABI-002-FAENUM is C++-shaped; count it into the cxx trip stats as well
    trip_rate = (cxx_tr / len(cxx)) if cxx else 0.0
    bands = th["cxx_arm_reporting_bands"]["bands"]
    band = ("mechanized" if trip_rate < bands["mechanized"]["trip_rate_below"]
            else "relabelled_ifu" if trip_rate > bands["relabelled_ifu"]["trip_rate_above"] else "discuss")
    clean3 = sum(1 for v in meta.values() if v.get("merge3", {}).get("clean"))
    rep = {
        "inputs": {"pre_sync_fork": PRE, "merged_base": BASE + " (release_v2.15 tip, as recorded history)", "target": TGT},
        "case_rule": {"requirement": th["stage2_backtest"]["case_outcome_rule"]["requirement"],
                      "silent": silent, "satisfied": not silent},
        "python_cases": {"total": len(cases), "reapplied": n_re, "tripped": n_tr,
                         "tripped_ids": sorted(k for k, v in cases.items() if v["outcome"] != "reapplied")},
        "cxx_arm": {"total": len(cxx), "tripped": cxx_tr, "trip_rate": round(trip_rate, 2), "band": band,
                    "bands": bands, "informational_at_gate_a": True},
        "three_way_repair": {"clean_merges_among_tripped": clean3,
                             "note": "a clean 3-way means the trip is mechanically repairable; conflict hunks are the repair-size proxy"},
        "historical_effort_reference": th["stage2_backtest"]["repair_effort"]["historical_reference"],
        "limitations": "silently-wrong behavior not fully testable without a 2.15-era build; repair effort proxied by conflict hunks",
    }
    (PROP / "baselines" / "2026-08-31-backtest-b3.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["gen", "check", "report"])
    a = ap.parse_args()
    {"gen": gen, "check": check, "report": report}[a.cmd]()


if __name__ == "__main__":
    main()
