# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
"""P5: shrink the patch queue by trying upstream unchanged first - empirically, per patch.

For each patch: rebuild the overlay WITHOUT it, then run a check with PYTHONPATH at that
overlay. A patch whose removal breaks nothing is a retirement candidate. Three phases:

  smoke    import + te.Linear / LayerNormMLP fwd+bwd in BF16 and FP8 delayed scaling (~10 s).
           Removal breaks smoke -> the patch is NEEDED (recorded, no further testing).
  tests    for smoke survivors: the representative-suite files mapped to the patch's target
           file. Green -> RETIRE-CANDIDATE.
  confirm  remove ALL candidates at once, build, run the full representative suite (catches
           interactions between patches). Done by hand via assemble_overlay + baseline.py.

Results accumulate in build/shrink/results.json so phases can be re-run incrementally.
JX-* patches are skipped: the prototype builds no JAX extension to test them against (S7).

Usage:  shrink_queue.py smoke [IDS...]   |   shrink_queue.py tests [IDS...]   |   shrink_queue.py report
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROP = HERE.parent
ROOT = PROP.parent.parent
PATCHES = PROP / "patches"
HOLD = ROOT / "build" / "shrink" / "holdout"
RESULTS = Path(os.environ.get("SHRINK_RESULTS", ROOT / "build" / "shrink" / "results.json"))  # per-worker override
ASSEMBLE = HERE / "assemble_overlay.py"
ENV = {**os.environ, "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES", "0"), "NVTE_FRAMEWORK": "pytorch"}

SMOKE = r"""
import torch, transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling, Format
x = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
lin = te.Linear(128, 256).cuda().bfloat16(); lin(x).sum().backward()
mlp = te.LayerNormMLP(128, 512).cuda().bfloat16(); mlp(x).sum().backward()
with te.fp8_autocast(enabled=True, fp8_recipe=DelayedScaling(fp8_format=Format.HYBRID)):
    y = lin(x); y.sum().backward()
print("SMOKE_OK")
"""

# target file -> representative-suite files (plan sec 3.9) that exercise it
TESTMAP = [
    ("optimizers/", ["tests/pytorch/test_fused_optimizer.py"]),
    ("attention/", ["tests/pytorch/attention/test_attention.py -k 'not cp and not CP'"]),
    ("cpp_extensions/fused_attn.py", ["tests/pytorch/attention/test_attention.py -k 'not cp and not CP'"]),
    ("triton/cross_entropy.py", ["tests/pytorch/test_parallel_cross_entropy.py"]),
    ("quantization.py", ["tests/pytorch/test_recipe.py", "tests/pytorch/test_custom_recipe.py",
                         "tests/pytorch/test_numerics.py -k 'Linear or LayerNormMLP or layernorm_mlp'"]),
    ("common/recipe/", ["tests/pytorch/test_recipe.py", "tests/pytorch/test_custom_recipe.py"]),
    ("custom_recipes/", ["tests/pytorch/test_custom_recipe.py", "tests/pytorch/test_recipe.py"]),
    ("tensor/", ["tests/pytorch/test_numerics.py -k 'Linear or LayerNormMLP or layernorm_mlp'",
                 "tests/pytorch/mxfp4", "tests/pytorch/test_fusible_ops.py"]),
    ("quantized_tensor.py", ["tests/pytorch/test_numerics.py -k 'Linear or LayerNormMLP or layernorm_mlp'", "tests/pytorch/mxfp4"]),
    ("module/base.py", ["tests/pytorch/test_numerics.py -k 'Linear or LayerNormMLP or layernorm_mlp'",
                        "tests/pytorch/test_checkpoint.py", "tests/pytorch/test_cuda_graphs.py"]),
    ("module/", ["tests/pytorch/test_numerics.py -k 'Linear or LayerNormMLP or layernorm_mlp'", "tests/pytorch/test_cuda_graphs.py"]),
    ("cpp_extensions/gemm.py", ["tests/pytorch/test_numerics.py -k 'Linear or LayerNormMLP or layernorm_mlp'", "tests/pytorch/mxfp4"]),
    ("ops/", ["tests/pytorch/test_fusible_ops.py"]),
    ("graph.py", ["tests/pytorch/test_cuda_graphs.py"]),
    ("jit.py", ["tests/pytorch/test_torch_compile.py", "tests/pytorch/test_fusible_ops.py"]),
    ("_extra_state.py", ["tests/pytorch/test_checkpoint.py", "tests/pytorch/test_recipe.py"]),
    ("", ["tests/pytorch/test_numerics.py -k 'Linear or LayerNormMLP or layernorm_mlp'", "tests/pytorch/test_recipe.py"]),  # default
]


def load_results():
    """Merge every results*.json in the shrink dir (parallel workers write their own file);
    the per-worker file named by SHRINK_RESULTS wins for ids it contains."""
    merged = {}
    for f in sorted(RESULTS.parent.glob("results*.json")):
        if f != RESULTS:
            merged.update(json.loads(f.read_text()))
    if RESULTS.exists():
        merged.update(json.loads(RESULTS.read_text()))
    return merged


def save_results(r):
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(r, indent=1, sort_keys=True))


def patch_target(pid: str) -> str:
    for line in (PATCHES / f"{pid}.patch").read_text().splitlines():
        if line.startswith("--- a/"):
            return line[6:]
    return ""


def tests_for(target: str) -> list[str]:
    rel = target.replace("transformer_engine/pytorch/", "").replace("transformer_engine/", "")
    for key, tests in TESTMAP:
        if key and key in rel:
            return tests
    return TESTMAP[-1][1]


def build_without(pid: str) -> Path:
    """Overlay with this one patch left out - via the assembler's --exclude, so parallel
    workers never disturb each other's patch set."""
    out = ROOT / "build" / "shrink" / f"overlay-without-{pid}"
    r = subprocess.run([sys.executable, str(ASSEMBLE), "--out", str(out), "--exclude", pid, "build"],
                       capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError("assemble failed: " + (r.stderr or r.stdout)[-400:])
    return out


def run_py(code: str, overlay: Path, timeout=300) -> tuple[bool, str]:
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd="/tmp",
                       env={**ENV, "PYTHONPATH": str(overlay)}, timeout=timeout)
    return ("SMOKE_OK" in r.stdout), (r.stderr.strip().splitlines() or ["?"])[-1][:200]


def known_failures() -> list[str]:
    """Tests that FAIL on the fork baseline (P0). A removal is not blamed for those - otherwise
    every patch mapped to test_custom_recipe.py reads NEEDED because of the pre-existing
    dpa_fp8 capability gap (verified: PT-024 was a false NEEDED for exactly this reason)."""
    base = PROP / "baselines" / "2026-08-30-fork.json"
    if not base.exists():
        return []
    d = json.loads(base.read_text()); out = []
    for f, tests in d["tests"].items():
        for tid, st in tests.items():
            if st in ("fail", "error") and "<binary-param" not in tid:
                cls, _, name = tid.partition("::")
                out.append(f"{cls.replace('.', '/')}.py::{name}")
    return out


def run_tests(cmds: list[str], overlay: Path) -> tuple[bool, str]:
    """Runs from /tmp with ABSOLUTE test paths. From the repo root the cwd lands first on
    sys.path and shadows PYTHONPATH, so the tests would import the fork tree instead of the
    overlay - which silently invalidated the first Phase B/C run (2026-08-30)."""
    summ = []
    desel = " ".join(f"--deselect '{ROOT / t}'" for t in known_failures())
    for c in cmds:
        parts = c.split(" ", 1); abs_c = str(ROOT / parts[0]) + (" " + parts[1] if len(parts) > 1 else "")
        env = {**ENV, "PYTHONPATH": str(overlay),
               "NVTE_TEST_CHECKPOINT_ARTIFACT_PATH": os.environ.get("NVTE_TEST_CHECKPOINT_ARTIFACT_PATH", "")}
        try:
            # 20-minute cap: PT-028's removal hung test_cuda_graphs for 20 HOURS (hipGraph capture
            # deadlock on upstream's graph.py). A hang IS a NEEDED verdict; never wait forever.
            r = subprocess.run(f"{sys.executable} -m pytest -q -p no:cacheprovider -x --rootdir {ROOT} {desel} {abs_c}", shell=True,
                               capture_output=True, text=True, cwd="/tmp", env=env, timeout=1200)
        except subprocess.TimeoutExpired:
            return False, f"HANG >20min in {parts[0]} (killed; treat as NEEDED - see CLAUDE.md GPU-hang triage)"
        te_seen = subprocess.run([sys.executable, "-c", "import transformer_engine as t; print(t.__file__)"],
                                 capture_output=True, text=True, cwd="/tmp", env=env).stdout
        if str(overlay) not in te_seen:
            return False, f"OVERLAY NOT IMPORTED (got {te_seen.strip()[:80]}) - environment bug, verdict void"
        last = (r.stdout.strip().splitlines() or ["?"])[-1][:120]
        summ.append(f"{c.split()[0].split('/')[-1]}: {last}")
        if r.returncode:
            return False, " | ".join(summ)
    return True, " | ".join(summ)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["smoke", "tests", "report", "record"]); ap.add_argument("ids", nargs="*")
    a = ap.parse_args()
    results = load_results()
    all_ids = sorted(p.stem for p in PATCHES.glob("*.patch"))
    ids = a.ids or [i for i in all_ids if not i.startswith("JX-")]

    if a.phase == "record":
        # Fold results into the manifest as p5_* fields, line-based so layout/comments survive.
        mf = PROP / "divergence-manifest.yaml"
        lines = mf.read_text().split("\n"); out = []; cur = None; n = 0
        for line in lines:
            out.append(line)
            m = line if line.startswith("  - id: ") else None
            if m:
                cur = line[len("  - id: "):].strip()
                r = results.get(cur)
                if r:
                    out.append(f"    p5_status: {r['status']}")
                    out.append(f"    p5_detail: \"{r['detail'][:160].replace(chr(34), chr(39))}\"")
                    n += 1
        mf.write_text("\n".join(out)); print(f"recorded p5_status on {n} entries")
        return

    if a.phase == "report":
        for pid in all_ids:
            r = results.get(pid, {"status": "JX-deferred (S7)" if pid.startswith("JX-") else "untested"})
            print(f"  {pid:8s} {r.get('status', '?'):22s} {r.get('detail', '')[:110]}")
        by = {}
        for pid in all_ids:
            by.setdefault(results.get(pid, {}).get("status", "JX-deferred" if pid.startswith("JX-") else "untested"), []).append(pid)
        print({k: len(v) for k, v in by.items()})
        return

    own = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    for pid in ids:
        if a.phase == "tests" and results.get(pid, {}).get("status") != "smoke-ok":
            continue
        t0 = time.time(); target = patch_target(pid)
        try:
            ov = build_without(pid)
            if a.phase == "smoke":
                ok, detail = run_py(SMOKE, ov)
                status = "smoke-ok" if ok else "NEEDED(smoke)"
            else:
                ok, detail = run_tests(tests_for(target), ov)
                status = "RETIRE-CANDIDATE" if ok else "NEEDED(tests)"
        except Exception as e:  # noqa
            ok, detail, status = False, str(e)[:200], "ERROR"
        results[pid] = {"status": status, "detail": detail, "target": target, "secs": round(time.time() - t0)}
        own[pid] = results[pid]
        save_results(own)          # persist only this worker's own verdicts
        print(f"{pid:8s} {status:18s} {results[pid]['secs']:4d}s  {target.replace('transformer_engine/','')}  {'' if ok else detail[:90]}", flush=True)
        shutil.rmtree(ROOT / "build" / "shrink" / f"overlay-without-{pid}", ignore_errors=True)


if __name__ == "__main__":
    main()
