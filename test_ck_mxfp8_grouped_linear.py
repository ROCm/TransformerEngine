#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# License for AMD contributions = MIT. See LICENSE for more information

"""Forward-pass check for the CK MXFP8 grouped GEMM behind te.GroupedLinear.

The CK grouped GEMM is opt-in. Without NVTE_USE_CK_GROUPED_GEMM=1 the MXFP8
grouped path falls through to hipBLASLt, so a test that merely runs GroupedLinear
proves nothing about CK. This script therefore launches two clean worker
processes -- Transformer Engine is imported only after the backend environment is
established -- and compares them:

    ck       : NVTE_USE_CK_GROUPED_GEMM=1
    baseline : no backend flag (hipBLASLt)

Selection notes (transformer_engine/common/gemm/rocm_gemm.cu):

    kittens_grouped_mxfp8_enabled() == use_hk || (use_cutlass && !use_ck)

so NVTE_USE_CK_GROUPED_GEMM=1 on its own both enables CK and keeps HipKittens
out of the way -- provided NVTE_USE_HIPKITTENS_GROUPED_GEMM is not also set.

Reaching the CK entry point is not the same as CK serving the GEMM: it declines
(returns false) on unsupported shapes or insufficient workspace and the caller
quietly uses hipBLASLt instead. Both workers run with
NVTE_CUTLASS_GROUPED_GEMM_WARN_FALLBACK=1 so that decision is visible, and the
ck worker is treated as failed if the fallback warning appears.

Usage:

    python test_ck_mxfp8_grouped_linear.py
    python test_ck_mxfp8_grouped_linear.py --num-gemms 8 --hidden 2048 --ffn 4096
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Emitted by cublaslt_gemm.cu when handled_by_ck is false.
FALLBACK_MARKER = "Fallback to cuBLAS grouped GEMM."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-gemms", type=int, default=4, help="Number of experts.")
    parser.add_argument("--hidden", type=int, default=1024, help="in_features (K).")
    parser.add_argument("--ffn", type=int, default=1024, help="out_features (N).")
    parser.add_argument(
        "--tokens-per-expert",
        type=int,
        default=256,
        help="Rows per group (M). Must keep every split MXFP8-aligned.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--atol",
        type=float,
        default=None,
        help="Absolute tolerance vs the hipBLASLt baseline. Default scales with K.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=0.05,
        help="Relative tolerance vs the hipBLASLt baseline.",
    )
    # Worker-only arguments.
    parser.add_argument("--worker-backend", choices=("ck", "baseline"), default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--output-pt", type=Path, default=None)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    # ScaleBlockSize is 32 and the gfx1250 scale preshuffle needs KScale % 4 == 0,
    # i.e. K % 128 == 0. gfx950 packs scales in pairs, needing only K % 64 == 0.
    # Require the stricter of the two so one shape exercises both backends.
    if args.hidden % 128 != 0:
        raise SystemExit(f"--hidden must be a multiple of 128, got {args.hidden}")
    if args.ffn % 128 != 0:
        raise SystemExit(f"--ffn must be a multiple of 128, got {args.ffn}")
    if args.tokens_per_expert % 32 != 0:
        raise SystemExit(
            f"--tokens-per-expert must be a multiple of 32, got {args.tokens_per_expert}"
        )
    if args.num_gemms < 1:
        raise SystemExit("--num-gemms must be >= 1")


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def run_worker(args: argparse.Namespace) -> None:
    # Backend environment is already set by the parent before this process
    # imports torch or Transformer Engine.
    import torch

    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import Format, MXFP8BlockScaling

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm device is not available")

    backend = args.worker_backend
    assert backend is not None
    if backend == "ck" and os.environ.get("NVTE_USE_CK_GROUPED_GEMM") != "1":
        raise RuntimeError("ck worker was launched without NVTE_USE_CK_GROUPED_GEMM=1")
    if backend == "baseline" and os.environ.get("NVTE_USE_CK_GROUPED_GEMM"):
        raise RuntimeError("baseline worker unexpectedly has NVTE_USE_CK_GROUPED_GEMM set")

    # Same seed in both workers so the two GroupedLinear instances hold identical
    # weights and see identical inputs; any output difference is the GEMM backend.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_grad_enabled(False)

    device = torch.device("cuda")
    dtype = torch.bfloat16

    layer = te.GroupedLinear(
        args.num_gemms,
        args.hidden,
        args.ffn,
        bias=False,
        params_dtype=dtype,
        device=device,
    ).eval()

    m_splits = [args.tokens_per_expert] * args.num_gemms
    inp = torch.randn(sum(m_splits), args.hidden, device=device, dtype=dtype)

    recipe = MXFP8BlockScaling(fp8_format=Format.E4M3)

    def fp8_context():
        autocast = getattr(te, "autocast", None)
        if autocast is not None:
            try:
                return autocast(enabled=True, recipe=recipe)
            except TypeError:
                pass
        fp8_autocast = getattr(te, "fp8_autocast", None)
        if fp8_autocast is None:
            raise RuntimeError(
                "Transformer Engine exposes neither te.autocast nor te.fp8_autocast"
            )
        return fp8_autocast(enabled=True, fp8_recipe=recipe)

    with fp8_context():
        out = layer(inp, m_splits)
    torch.cuda.synchronize()

    if args.output_pt is not None:
        torch.save(out.detach().float().cpu(), args.output_pt)

    summary = {
        "backend": backend,
        "num_gemms": args.num_gemms,
        "hidden": args.hidden,
        "ffn": args.ffn,
        "m_splits": m_splits,
        "out_shape": list(out.shape),
        "out_dtype": str(out.dtype),
        "has_nan": bool(torch.isnan(out).any().item()),
        "has_inf": bool(torch.isinf(out).any().item()),
        "abs_mean": float(out.detach().float().abs().mean().item()),
        "NVTE_ROCM_ENABLE_MXFP8": os.environ.get("NVTE_ROCM_ENABLE_MXFP8"),
        "NVTE_USE_CK_GROUPED_GEMM": os.environ.get("NVTE_USE_CK_GROUPED_GEMM"),
        "NVTE_USE_HIPKITTENS_GROUPED_GEMM": os.environ.get(
            "NVTE_USE_HIPKITTENS_GROUPED_GEMM"
        ),
    }
    if args.summary_json is not None:
        args.summary_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# Parent
# ---------------------------------------------------------------------------


def worker_command(args: argparse.Namespace, backend: str, summary_json: Path,
                   output_pt: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--num-gemms", str(args.num_gemms),
        "--hidden", str(args.hidden),
        "--ffn", str(args.ffn),
        "--tokens-per-expert", str(args.tokens_per_expert),
        "--seed", str(args.seed),
        "--worker-backend", backend,
        "--summary-json", str(summary_json),
        "--output-pt", str(output_pt),
    ]


def run_backend(args: argparse.Namespace, backend: str, workdir: Path) -> dict:
    summary_json = workdir / f"{backend}_summary.json"
    output_pt = workdir / f"{backend}_out.pt"

    env = os.environ.copy()
    # MXFP8 is off by default on ROCm; "1" enables it (quantization.py:177).
    # "2" would also make it the default recipe, which we do not need -- the
    # worker passes MXFP8BlockScaling explicitly.
    env.setdefault("NVTE_ROCM_ENABLE_MXFP8", "1")
    # Make the CK-declined-and-fell-back case observable in both workers.
    env["NVTE_CUTLASS_GROUPED_GEMM_WARN_FALLBACK"] = "1"
    # HipKittens takes precedence over CK when this is set; keep it out of the way.
    env.pop("NVTE_USE_HIPKITTENS_GROUPED_GEMM", None)
    env.pop("NVTE_USE_CUTLASS_GROUPED_GEMM", None)
    if backend == "ck":
        env["NVTE_USE_CK_GROUPED_GEMM"] = "1"
    else:
        env.pop("NVTE_USE_CK_GROUPED_GEMM", None)

    proc = subprocess.run(
        worker_command(args, backend, summary_json, output_pt),
        env=env,
        capture_output=True,
        text=True,
    )
    combined = proc.stdout + proc.stderr
    print(f"----- {backend} worker (exit {proc.returncode}) -----")
    print(combined.strip())

    result = {
        "backend": backend,
        "returncode": proc.returncode,
        "fell_back": FALLBACK_MARKER in combined,
        "output_pt": output_pt if output_pt.exists() else None,
        "summary": json.loads(summary_json.read_text()) if summary_json.exists() else None,
    }
    return result


def main() -> None:
    args = parse_args()

    if args.worker_backend is not None:
        run_worker(args)
        return

    validate_args(args)
    args.m_splits_resolved = [args.tokens_per_expert] * args.num_gemms

    import torch  # parent only needs this for the comparison

    failures: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        ck = run_backend(args, "ck", workdir)
        baseline = run_backend(args, "baseline", workdir)

        for res in (ck, baseline):
            if res["returncode"] != 0:
                failures.append(f"{res['backend']} worker exited {res['returncode']}")
            elif res["summary"] is None:
                failures.append(f"{res['backend']} worker produced no summary")
            else:
                if res["summary"]["has_nan"]:
                    failures.append(f"{res['backend']} output contains NaN")
                if res["summary"]["has_inf"]:
                    failures.append(f"{res['backend']} output contains Inf")

        # The point of the test: CK must actually have served the GEMM.
        if ck["fell_back"]:
            failures.append(
                "CK declined the grouped GEMM and fell back to hipBLASLt "
                f'(saw "{FALLBACK_MARKER}")'
            )

        if ck["output_pt"] and baseline["output_pt"]:
            ck_out = torch.load(ck["output_pt"])
            base_out = torch.load(baseline["output_pt"])
            if ck_out.shape != base_out.shape:
                failures.append(f"shape mismatch {ck_out.shape} vs {base_out.shape}")
            else:
                atol = args.atol
                if atol is None:
                    # MXFP8 quantization error grows with the reduction length.
                    atol = 0.05 * (args.hidden ** 0.5)
                diff = (ck_out - base_out).abs()
                max_abs = float(diff.max().item())
                denom = base_out.abs().clamp_min(1e-6)
                max_rel = float((diff / denom).max().item())
                close = torch.allclose(ck_out, base_out, atol=atol, rtol=args.rtol)
                print(
                    f"\nck vs baseline: max_abs={max_abs:.6g} max_rel={max_rel:.6g} "
                    f"(atol={atol:.6g} rtol={args.rtol:.6g})"
                )
                # Diagnostics for the "right values, wrong places" failure mode:
                # a permuted output keeps the multiset of magnitudes intact, so
                # abs_mean matches while the elementwise diff is large.
                if not close:
                    ck_sorted = ck_out.flatten().sort().values
                    base_sorted = base_out.flatten().sort().values
                    perm_max = float((ck_sorted - base_sorted).abs().max().item())
                    print(f"  sorted-values max_abs={perm_max:.6g} "
                          "(small => same values, wrong positions)")
                    if ck_out.shape[0] == ck_out.shape[1]:
                        t_max = float((ck_out - base_out.T).abs().max().item())
                        print(f"  vs baseline transposed: max_abs={t_max:.6g}")

                    # Per-group diff, and whether group g of ck matches some
                    # other group of baseline (i.e. the groups got reordered).
                    offs = [0]
                    for s in args.m_splits_resolved:
                        offs.append(offs[-1] + s)
                    print("  per-group max_abs (ck[g] vs base[g]):", end=" ")
                    for g in range(len(args.m_splits_resolved)):
                        blk_c = ck_out[offs[g]:offs[g + 1]]
                        blk_b = base_out[offs[g]:offs[g + 1]]
                        print(f"g{g}={float((blk_c - blk_b).abs().max().item()):.4g}", end=" ")
                    print()
                    for g in range(len(args.m_splits_resolved)):
                        blk_c = ck_out[offs[g]:offs[g + 1]]
                        matches = [
                            h for h in range(len(args.m_splits_resolved))
                            if args.m_splits_resolved[h] == args.m_splits_resolved[g]
                            and torch.equal(blk_c, base_out[offs[h]:offs[h + 1]])
                        ]
                        if matches != [g]:
                            print(f"  ck group {g} exactly matches baseline group(s) {matches}")

                    # Row-level: does every ck row appear somewhere in baseline?
                    row_perm_exact = torch.equal(
                        ck_out.sort(dim=0).values, base_out.sort(dim=0).values
                    )
                    col_perm_exact = torch.equal(
                        ck_out.sort(dim=1).values, base_out.sort(dim=1).values
                    )
                    print(f"  columnwise-sorted equal: {row_perm_exact}  "
                          f"rowwise-sorted equal: {col_perm_exact}")

                    # Tile-local rearrangement: TransposeC flips the layout each
                    # warp/block tile writes, which is a permutation that is
                    # neither a row nor a column nor a whole-matrix transpose.
                    H, W = ck_out.shape
                    for T in (16, 32, 64, 128):
                        if H % T or W % T:
                            continue
                        a = ck_out.reshape(H // T, T, W // T, T)
                        b = base_out.reshape(H // T, T, W // T, T)
                        intra = float((a - b.permute(0, 3, 2, 1)).abs().max().item())
                        line = f"  tile {T}x{T}: intra-tile-transpose max_abs={intra:.6g}"
                        if H == W:
                            grid = float((a - b.permute(2, 1, 0, 3)).abs().max().item())
                            line += f"  tile-grid-transpose max_abs={grid:.6g}"
                        print(line)

                    # Shift/shear: a wrong leading dimension moves data by a
                    # constant offset in the flat buffer, preserving the multiset
                    # while matching none of the structured permutations above.
                    flat_c = ck_out.flatten()
                    flat_b = base_out.flatten()
                    first = flat_c[0]
                    cand = (flat_b == first).nonzero().flatten().tolist()[:8]
                    print(f"  ck[0,0]={float(first):.6g} occurs in baseline at flat idx {cand}")
                    for k in cand:
                        if torch.equal(flat_c, torch.roll(flat_b, -k)):
                            print(f"  EXACT: ck == baseline rolled by {k} elements "
                                  f"({k // W} rows + {k % W} cols)")
                            break
                    # Row-level shift.
                    row0 = ck_out[0]
                    row_matches = [
                        r for r in range(H) if torch.equal(base_out[r], row0)
                    ][:8]
                    print(f"  ck row 0 matches baseline rows {row_matches}")
                if not close:
                    failures.append(
                        f"ck and baseline outputs differ beyond tolerance "
                        f"(max_abs={max_abs:.6g}, max_rel={max_rel:.6g})"
                    )

    print()
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        sys.exit(1)
    print("PASS: CK MXFP8 grouped GEMM served te.GroupedLinear forward "
          "and matched the hipBLASLt baseline.")


if __name__ == "__main__":
    main()
